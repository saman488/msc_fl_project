"""
Verification of d2_07_evaluate_final_test.py.

Behavioural checks on synthetic fixtures in temporary directories, plus a few
small AST checks. Never reads the real held-out arrays and never creates the real
final-test output directory.
"""

from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.dont_write_bytecode = True

EVALUATOR = Path("d2_07_evaluate_final_test.py")
FORBIDDEN_ARRAYS = {"X_test.npy", "y_test.npy"}
TOL = 1e-9


def guard_np_load():
    """Raise if anything under verification tries to open a real held-out array."""
    real_load = np.load

    def guarded(file, *args, **kwargs):
        name = Path(str(file)).name
        if name in FORBIDDEN_ARRAYS:
            raise RuntimeError(f"verification attempted to load a held-out array: {file}")
        return real_load(file, *args, **kwargs)

    np.load = guarded
    return real_load


def load_evaluator():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("d2_07_evaluator", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules["d2_07_evaluator"] = module
    spec.loader.exec_module(module)
    return module


def write_history(path, best_round, rounds=40):
    scores = [0.10 + 0.001 * r for r in range(1, rounds + 1)]
    scores[best_round - 1] = 0.9
    history = pd.DataFrame({"round": range(1, rounds + 1), "macro_f1": scores})
    history.to_csv(path, index=False)
    return history


def make_config(module, algorithm, seed, condition, best_round, models_dir, tag):
    """A synthetic config carrying exactly the fields the evaluator requires."""
    return {
        **module.COMMON_CONFIG,
        **module.ALGORITHM_CONFIG[algorithm],
        "partition_seed": seed,
        "condition": condition,
        "best_round": best_round,
        "aggregation_weights": [0.3, 0.25, 0.2, 0.15, 0.1],
        "partition_path": str(module.PART_ROOT / f"seed_{seed}" / condition),
        "initial_state_path": str(module.FEDAVG_INIT_PATH),
        "best_checkpoint_path": str(Path(models_dir) / f"best_{tag}.pt"),
        "final_checkpoint_path": str(Path(models_dir) / f"final_{tag}.pt"),
        "script_sha256": module.file_sha256(module.TRAINER_SCRIPTS[algorithm]),
        "best_checkpoint_reloaded_val_macro_f1": 0.9,
    }


def save_state(module, path):
    torch.save(module.MLPMultiClassClassifier().state_dict(), path)


def build_run(module, root, algorithm, prefix, seed, condition, best_round, config_edit=None):
    tag = f"{prefix}_seed{seed}_{condition}"
    history = write_history(root / f"history_{tag}.csv", best_round)
    config = make_config(module, algorithm, seed, condition, best_round, root, tag)
    if config_edit:
        config = config_edit(config)
    json.dump(config, open(root / f"config_{tag}.json", "w"))
    save_state(module, root / f"best_{tag}.pt")
    return {"method": algorithm, "partition_seed": seed, "condition": condition,
            "best_round": best_round, "best_val_macro_f1": float(history["macro_f1"].max())}


def build_matrix(module, root, algorithm="FedAvg", prefix="fedavg_k5", **overrides):
    """A full 12-run synthetic matrix, optionally corrupting one run."""
    rows = []
    for seed in module.PARTITION_SEEDS:
        for condition in module.CONDITIONS:
            best_round = 5 + (seed % 3) + module.CONDITIONS.index(condition)
            target = (seed, condition) == overrides.get("corrupt_run")
            row = build_run(module, root, algorithm, prefix, seed, condition, best_round,
                            overrides.get("config_edit") if target else None)
            if target and "summary_edit" in overrides:
                row.update(overrides["summary_edit"])
            rows.append(row)
    summary = root / "final_summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    return summary


def build_central(module, root, summary_edit=None):
    """Synthetic centralised history, summary and checkpoint."""
    best_epoch, best_value = 5, 0.6652865190820361
    scores = [0.10 + 0.001 * e for e in range(1, 21)]
    scores[best_epoch - 1] = best_value
    pd.DataFrame({"epoch": range(1, 21), "val_macro_f1": scores}).to_csv(
        root / "central_mlp_training_history.csv", index=False)

    script = module.TRAINER_SCRIPTS["Centralised"]
    summary = {"best_epoch": best_epoch, "best_val_macro_f1": best_value,
               "reloaded_val_macro_f1": best_value, "reload_macro_f1_delta": 0.0,
               "epochs_run": 20, "holdout_evaluation_split_used": False,
               "script_name": script.name, "script_sha256": module.file_sha256(script)}
    if summary_edit:
        summary.update(summary_edit)
    pd.DataFrame([summary]).to_csv(root / "validation_summary.csv", index=False)

    models = root / "models"
    models.mkdir(exist_ok=True)
    save_state(module, models / "central_mlp_best.pt")
    return models


def central_paths(module, root, models):
    """Point the evaluator's centralised constants at a temporary fixture."""
    return {"CENTRAL_RESULTS": root, "CENTRAL_MODELS": models}


def swap(module, values):
    previous = {k: getattr(module, k) for k in values}
    for k, v in values.items():
        setattr(module, k, v)
    return previous


def expect_failure(call, fragment, description):
    try:
        call()
    except RuntimeError as error:
        assert fragment in str(error), f"{description}: unexpected message: {error}"
    else:
        raise AssertionError(f"{description} was not rejected")


def check_01_no_test_access_in_selection(ctx):
    module, tree = ctx["module"], ctx["tree"]
    main_node = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "main")
    test_lines = [n.lineno for n in ast.walk(main_node)
                  if isinstance(n, ast.Constant) and n.value in FORBIDDEN_ARRAYS]
    return_lines = [n.lineno for n in ast.walk(main_node) if isinstance(n, ast.Return)]
    assert test_lines, "main() never loads the test arrays; this check would be vacuous"
    assert len(return_lines) >= 2, "main() lacks the two early returns for the safe modes"
    assert sorted(return_lines)[1] < min(test_lines), \
        "a safe-mode return does not precede every test-array load"

    # The selection and smoke paths must not name the arrays at all.
    body = ""
    for name in ("check_selection", "select_central", "select_federated",
                 "check_checkpoint", "check_config", "smoke_checkpoint", "smoke_all"):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        body += ast.unparse(node)
    for token in ("X_test", "y_test"):
        assert token not in body, f"selection or smoke logic references {token}"


def check_02_smoke_before_output(ctx):
    """Inside main(): smoke test, then the output guard, then any test-array load."""
    main_node = next(n for n in ast.walk(ctx["tree"])
                     if isinstance(n, ast.FunctionDef) and n.name == "main")

    smoke = [n.lineno for n in ast.walk(main_node) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "smoke_all"]
    guard = [n.lineno for n in ast.walk(main_node) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "exists"
             and getattr(n.func.value, "id", None) == "OUT_DIR"]
    loads = [n.lineno for n in ast.walk(main_node)
             if isinstance(n, ast.Constant) and n.value in FORBIDDEN_ARRAYS]

    assert smoke, "main() does not call smoke_all()"
    assert guard, "main() does not check whether OUT_DIR exists"
    assert loads, "main() never loads the test arrays; this check would be vacuous"
    assert max(smoke) < min(guard), "the smoke test does not precede the output-directory guard"
    assert max(guard) < min(loads), "the output-directory guard does not precede the test load"


def check_03_no_production_asserts(ctx):
    asserts = [n.lineno for n in ast.walk(ctx["tree"]) if isinstance(n, ast.Assert)]
    assert not asserts, f"the evaluator still uses assert at lines {asserts}"
    expect_failure(lambda: ctx["module"].require(False, "probe"), "probe",
                   "require(False, ...)")


def check_04_best_not_final(ctx):
    module, source = ctx["module"], ctx["source"]
    assert 'models_dir / f"best_{tag}.pt"' in source, "federated selection does not use best_*"
    assert '"central_mlp_best.pt"' in source, "centralised selection does not use best"
    assert 'require(not any("final_" in Path(p).name for p in paths)' in source, \
        "check_selection does not reject final-round checkpoints"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary = build_matrix(module, root)
        selected = module.select_federated("FedAvg", root, root, summary, "fedavg_k5")
        assert len(selected) == 12, f"expected 12 runs, found {len(selected)}"
        assert all(Path(m["checkpoint_path"]).name.startswith("best_") for m in selected), \
            "a non-best checkpoint was selected"
        assert all(m["val_macro_f1"] == 0.9 for m in selected), "selection is not the maximum"


def check_05_history_and_summary(ctx):
    module = ctx["module"]
    cases = [
        ({"summary_edit": {"best_round": 99}}, "best_round", "a best_round mismatch"),
        ({"summary_edit": {"best_val_macro_f1": 0.123}}, "best_val_macro_f1",
         "a macro-F1 mismatch"),
    ]
    for overrides, fragment, description in cases:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_matrix(module, root, corrupt_run=(42, "iid"), **overrides)
            expect_failure(
                lambda: module.select_federated("FedAvg", root, root, summary, "fedavg_k5"),
                fragment, description)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary = build_matrix(module, root)
        path = root / "history_fedavg_k5_seed42_iid.csv"
        pd.read_csv(path).head(10).to_csv(path, index=False)
        expect_failure(
            lambda: module.select_federated("FedAvg", root, root, summary, "fedavg_k5"),
            "rounds", "a truncated history")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary = build_matrix(module, root)
        path = root / "history_fedavg_k5_seed42_iid.csv"
        pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
        expect_failure(
            lambda: module.select_federated("FedAvg", root, root, summary, "fedavg_k5"),
            "order", "a disordered history")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary = build_matrix(module, root)
        pd.read_csv(summary).head(11).to_csv(summary, index=False)
        expect_failure(
            lambda: module.select_federated("FedAvg", root, root, summary, "fedavg_k5"),
            "12 rows", "an incomplete 12-run matrix")


def check_06_config_identity(ctx):
    module = ctx["module"]
    cases = [
        (lambda c: {**c, "condition": "alpha_1p0"}, "condition", "a condition mismatch"),
        (lambda c: {**c, "partition_seed": 99}, "partition_seed", "a seed mismatch"),
        (lambda c: {**c, "lr": 0.5}, "lr", "a wrong learning rate"),
        (lambda c: {**c, "local_epochs": 3}, "local_epochs", "wrong local epochs"),
        (lambda c: {k: v for k, v in c.items() if k != "batch_size"}, "batch_size",
         "a missing required field"),
        (lambda c: {**c, "aggregation_weights": [0.5, 0.5]}, "aggregation weights",
         "a wrong participation count"),
    ]
    for edit, fragment, description in cases:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_matrix(module, root, corrupt_run=(42, "iid"), config_edit=edit)
            expect_failure(
                lambda: module.select_federated("FedAvg", root, root, summary, "fedavg_k5"),
                fragment, description)


def check_07_path_and_script_provenance(ctx):
    module = ctx["module"]
    cases = [
        (lambda c: {**c, "best_checkpoint_path": "models/elsewhere/best.pt"},
         "best_checkpoint_path", "a wrong best-checkpoint path"),
        (lambda c: {**c, "final_checkpoint_path": "models/elsewhere/final.pt"},
         "final_checkpoint_path", "a wrong final-checkpoint path"),
        (lambda c: {**c, "partition_path": "data/elsewhere"}, "partition_path",
         "a wrong partition path"),
        (lambda c: {**c, "initial_state_path": "models/elsewhere/init.pt"},
         "initial_state_path", "a wrong initial-state path"),
        (lambda c: {**c, "script_sha256": "0" * 64}, "script_sha256",
         "a training-script SHA mismatch"),
        (lambda c: {**c, "best_checkpoint_reloaded_val_macro_f1": 0.2},
         "reloaded val macro-F1", "a reloaded-score disagreement"),
    ]
    for edit, fragment, description in cases:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_matrix(module, root, corrupt_run=(43, "iid"), config_edit=edit)
            expect_failure(
                lambda: module.select_federated("FedAvg", root, root, summary, "fedavg_k5"),
                fragment, description)


def check_08_fedprox_mu(ctx):
    module = ctx["module"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary = build_matrix(module, root, "FedProx", "fedprox_k5_mu1em05")
        selected = module.select_federated("FedProx", root, root, summary, "fedprox_k5_mu1em05")
        assert len(selected) == 12, "the correct FedProx matrix was rejected"
        assert all(m["mu"] == 1e-5 for m in selected), "mu was not recorded as 1e-5"

    for edit, description in ((lambda c: {**c, "mu": 1e-3}, "a wrong FedProx mu"),
                              (lambda c: {k: v for k, v in c.items() if k != "mu"},
                               "a missing FedProx mu")):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_matrix(module, root, "FedProx", "fedprox_k5_mu1em05",
                                   corrupt_run=(42, "iid"), config_edit=edit)
            expect_failure(lambda: module.select_federated("FedProx", root, root, summary,
                                                           "fedprox_k5_mu1em05"),
                           "mu", description)


def check_09_scaffold_settings(ctx):
    module = ctx["module"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary = build_matrix(module, root, "SCAFFOLD", "scaffold_k5")
        assert len(module.select_federated("SCAFFOLD", root, root, summary,
                                           "scaffold_k5")) == 12, \
            "the correct SCAFFOLD matrix was rejected"

    for key, wrong in (("control_variate_option", "Option I"),
                       ("gradient_correction", "g + c_i - c"),
                       ("server_control_aggregation", "uniform mean")):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_matrix(module, root, "SCAFFOLD", "scaffold_k5",
                                   corrupt_run=(42, "iid"),
                                   config_edit=lambda c, k=key, w=wrong: {**c, k: w})
            expect_failure(lambda: module.select_federated("SCAFFOLD", root, root, summary,
                                                           "scaffold_k5"),
                           key, f"a wrong SCAFFOLD {key}")


def check_10_central_selection(ctx):
    module = ctx["module"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        models = build_central(module, root)
        previous = swap(module, central_paths(module, root, models))
        try:
            selected = module.select_central()
            assert selected["selected_round_or_epoch"] == 5, "the wrong epoch was selected"
            assert abs(selected["val_macro_f1"] - 0.6652865190820361) < TOL, \
                "the wrong validation score was selected"
            assert Path(selected["checkpoint_path"]).name == "central_mlp_best.pt", \
                "a non-best centralised checkpoint was selected"
        finally:
            swap(module, previous)

    edits = [
        ({"best_epoch": 9}, "best_epoch", "a centralised epoch mismatch"),
        ({"script_sha256": "0" * 64}, "script_sha256", "a centralised script SHA mismatch"),
        ({"script_name": "something_else.py"}, "script_name", "a wrong centralised script name"),
        ({"holdout_evaluation_split_used": True}, "holdout", "a recorded held-out access"),
        ({"holdout_evaluation_split_used": "garbage"}, "holdout",
         "a malformed holdout_evaluation_split_used value"),
        ({"reloaded_val_macro_f1": 0.2}, "reloaded", "a centralised reload disagreement"),
    ]
    for edit, fragment, description in edits:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = build_central(module, root, summary_edit=edit)
            previous = swap(module, central_paths(module, root, models))
            try:
                expect_failure(module.select_central, fragment, description)
            finally:
                swap(module, previous)


def check_11_checkpoint_contract(ctx):
    module = ctx["module"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = root / "good.pt"
        save_state(module, good)
        assert module.check_checkpoint(good) == hashlib.sha256(good.read_bytes()).hexdigest(), \
            "check_checkpoint returned the wrong sha256"

        empty = root / "empty.pt"
        empty.write_bytes(b"")
        malformed = root / "malformed.pt"
        malformed.write_bytes(b"not a checkpoint")
        state = module.MLPMultiClassClassifier().state_dict()
        wrong_shape = root / "wrong_shape.pt"
        torch.save({k: (v[..., :1].clone() if v.ndim > 1 else v.clone())
                    for k, v in state.items()}, wrong_shape)
        missing_key = root / "missing_key.pt"
        partial = dict(state)
        partial.pop("network.6.bias")
        torch.save(partial, missing_key)
        non_finite = root / "non_finite.pt"
        broken = {k: v.clone() for k, v in state.items()}
        broken["network.0.bias"][0] = float("nan")
        torch.save(broken, non_finite)

        for path in (root / "absent.pt", empty, malformed, wrong_shape, missing_key, non_finite):
            try:
                module.check_checkpoint(path)
            except (RuntimeError, EOFError):
                pass
            else:
                raise AssertionError(f"{path.name} should have been rejected")


def check_12_smoke_inference(ctx):
    module = ctx["module"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = root / "good.pt"
        save_state(module, good)
        module.smoke_checkpoint(good)   # a valid checkpoint must pass

        # An extra key must fail strict loading.
        extra = root / "extra_key.pt"
        state = module.MLPMultiClassClassifier().state_dict()
        torch.save({**state, "network.9.weight": torch.zeros(1)}, extra)
        try:
            module.smoke_checkpoint(extra)
        except RuntimeError:
            pass
        else:
            raise AssertionError("strict load accepted an unexpected key")

        # Non-finite weights must surface as non-finite logits.
        broken_path = root / "broken.pt"
        broken = {k: v.clone() for k, v in state.items()}
        broken["network.0.weight"][0, 0] = float("inf")
        torch.save(broken, broken_path)
        expect_failure(lambda: module.smoke_checkpoint(broken_path), "non-finite",
                       "a checkpoint producing non-finite logits")


def check_13_metrics(ctx):
    module = ctx["module"]
    class_names = [f"c{i}" for i in range(7)]
    y_true = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 6], dtype=int)
    y_pred = np.array([0, 1, 1, 1, 2, 3, 4, 5, 6, 0], dtype=int)
    y_prob = np.full((10, 7), 0.05)
    y_prob[np.arange(10), y_pred] = 0.7

    result = module.compute_metrics(y_true, y_pred, y_prob, class_names)
    overall, per_class = result["overall"], result["per_class"]

    assert abs(overall["accuracy"] - accuracy_score(y_true, y_pred)) < TOL, "accuracy"
    assert abs(overall["macro_f1"] - f1_score(y_true, y_pred, labels=range(7),
                                              average="macro", zero_division=0)) < TOL, \
        "macro_f1 disagrees with an independent sklearn calculation"
    assert "macro_average_precision" in overall, "macro average precision is not reported"
    assert "macro_pr_auc" not in overall, "mean average precision is still labelled PR-AUC"
    assert overall["n_samples"] == 10, "n_samples is wrong"

    assert [r["support"] for r in per_class] == [2, 2, 1, 1, 1, 1, 2], "per-class support"
    assert [r["predicted_count"] for r in per_class] == [2, 3, 1, 1, 1, 1, 1], "predicted counts"
    assert all("average_precision" in r for r in per_class), "per-class average precision"
    assert abs(overall["worst_class_f1"] - min(r["f1"] for r in per_class)) < TOL, \
        "worst_class_f1 is not the minimum per-class F1"


def check_14_confusion_matrices(ctx):
    module = ctx["module"]
    class_names = [f"c{i}" for i in range(7)]
    y_true = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 6], dtype=int)
    y_pred = np.array([0, 1, 1, 1, 2, 3, 4, 5, 6, 0], dtype=int)
    y_prob = np.full((10, 7), 0.05)
    y_prob[np.arange(10), y_pred] = 0.7

    raw = module.compute_metrics(y_true, y_pred, y_prob, class_names)["confusion"]
    assert np.array_equal(raw, confusion_matrix(y_true, y_pred, labels=range(7))), \
        "the confusion matrix disagrees with an independent sklearn calculation"
    assert raw.sum() == 10, "the confusion matrix does not account for every sample"

    # Row-normalisation is recall per true class; a zero-support row stays zero.
    with_gap = np.array([[3, 1], [0, 0]], dtype=int)
    totals = with_gap.sum(axis=1, keepdims=True)
    normalised = np.divide(with_gap, totals, out=np.zeros(with_gap.shape, dtype=float),
                           where=totals > 0)
    assert abs(normalised[0, 0] - 0.75) < TOL, "row normalisation is wrong"
    assert normalised[1].sum() == 0.0, "a zero-support row did not stay zero"
    assert "np.divide(raw, totals" in ctx["source"], \
        "the evaluator does not row-normalise with a zero-support guard"


def check_15_evaluate_and_save(ctx):
    """evaluate_model() and save_results() end-to-end on synthetic arrays only."""
    module = ctx["module"]
    generator = np.random.default_rng(0)
    x_test = generator.normal(size=(64, 36)).astype(np.float32)
    y_test = np.tile(np.arange(7), 10)[:64].astype(np.int64)
    class_names = module.load_class_names()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint = root / "best_synthetic.pt"
        save_state(module, checkpoint)
        result = module.evaluate_model(checkpoint, x_test, y_test,
                                       torch.device("cpu"), class_names)
        assert result["overall"]["n_samples"] == 64, "evaluate_model saw the wrong row count"
        assert len(result["per_class"]) == 7, "evaluate_model produced the wrong class count"
        assert result["confusion"].sum() == 64, "the confusion matrix lost samples"

        selected = [{"algorithm": "FedAvg", "seed": 42, "condition": "iid", "mu": None,
                     "selected_round_or_epoch": 7, "val_macro_f1": 0.5,
                     "checkpoint_path": str(checkpoint),
                     "checkpoint_sha256": module.file_sha256(checkpoint),
                     "tag": "synthetic"}]
        out_dir = root / "out"
        module.save_results(selected, [result], class_names, torch.device("cpu"), out_dir)

        for name in ("selected_checkpoints.csv", "test_results_overall.csv",
                     "test_results_per_class.csv", "evaluation_config.json"):
            assert (out_dir / name).exists(), f"save_results did not write {name}"
        for name in ("confusion_raw_synthetic.csv", "confusion_normalised_synthetic.csv"):
            assert (out_dir / "confusion_matrices" / name).exists(), f"missing {name}"

        overall = pd.read_csv(out_dir / "test_results_overall.csv")
        for column in ("algorithm", "seed", "condition", "mu", "selected_round_or_epoch",
                       "val_macro_f1", "macro_f1", "macro_average_precision"):
            assert column in overall.columns, f"test_results_overall.csv lacks {column}"
        chosen = pd.read_csv(out_dir / "selected_checkpoints.csv")
        for column in ("algorithm", "seed", "condition", "mu", "checkpoint_path",
                       "checkpoint_sha256"):
            assert column in chosen.columns, f"selected_checkpoints.csv lacks {column}"


def check_16_no_real_output_or_test_access(ctx):
    module = ctx["module"]
    assert not module.OUT_DIR.exists(), \
        f"verification created the real final-test directory {module.OUT_DIR}"
    expect_failure(lambda: np.load(module.PROCESSED_DIR / "X_test.npy"),
                   "held-out", "the np.load guard")
    own_source = Path(__file__).read_text()
    assert "PROCESSED_DIR / \"X_test.npy\"" not in own_source.replace(
        'np.load(module.PROCESSED_DIR / "X_test.npy")', ""), \
        "the verifier loads a real held-out array outside the guard test"


CHECKS = [
    (1, "selection and smoke logic cannot reach the test arrays", check_01_no_test_access_in_selection),
    (2, "smoke and output guard precede any test-array load", check_02_smoke_before_output),
    (3, "production checks do not rely on assert", check_03_no_production_asserts),
    (4, "best-validation checkpoints are selected", check_04_best_not_final),
    (5, "history length, order and 12-run matrix", check_05_history_and_summary),
    (6, "config run identity and required fields", check_06_config_identity),
    (7, "recorded paths and training-script SHA", check_07_path_and_script_provenance),
    (8, "FedProx mu must be exactly 1e-5", check_08_fedprox_mu),
    (9, "SCAFFOLD control settings", check_09_scaffold_settings),
    (10, "centralised selection and provenance", check_10_central_selection),
    (11, "checkpoint contract", check_11_checkpoint_contract),
    (12, "strict loading and smoke inference", check_12_smoke_inference),
    (13, "metrics against independent sklearn calls", check_13_metrics),
    (14, "raw and row-normalised confusion matrices", check_14_confusion_matrices),
    (15, "evaluate_model and save_results on synthetic data", check_15_evaluate_and_save),
    (16, "no real output directory or test-array access", check_16_no_real_output_or_test_access),
]


def main() -> int:
    real_load = guard_np_load()
    source = EVALUATOR.read_text()
    ctx = {"module": load_evaluator(), "source": source,
           "tree": ast.parse(source, filename=str(EVALUATOR))}

    print(f"verifying {EVALUATOR}")
    print("synthetic fixtures only; no real test array is read\n", flush=True)

    passed, failed = 0, []
    for check_id, title, function in CHECKS:
        try:
            function(ctx)
        except Exception as error:  # report, do not abort the suite
            failed.append((check_id, title, error))
            print(f"[{check_id:2d}] FAIL  {title}\n      {type(error).__name__}: {error}",
                  flush=True)
            traceback.print_exc()
        else:
            passed += 1
            print(f"[{check_id:2d}] PASS  {title}", flush=True)

    np.load = real_load
    print(f"\npassed={passed} failed={len(failed)}")
    print(f"real final-test directory exists: {ctx['module'].OUT_DIR.exists()}")
    print(f"RESULT: {'PASS' if not failed else 'FAIL'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
