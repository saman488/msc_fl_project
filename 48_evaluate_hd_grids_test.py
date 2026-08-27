"""
Held-out test evaluation of the two Dataset-1 HD-matched FedAvg grids.

Adapted from 39_evaluate_final_test.py, which evaluated 37 models across four
algorithm families on the fixed-alpha partitions. This one evaluates 30: FedAvg
only, 15 runs on the partitions from 42_select_partitions_by_hd.py and 15 on those
from 45_select_partitions_by_hd_fedartml.py, at matched HD. Selection is re-derived
from the saved histories, summaries and configs; the test set chooses nothing.

    --verify-selection   selection checks only, no test-array access
    --smoke-checkpoints  the same checks, then a synthetic forward pass per model
    --evaluate-test      the same checks, then held-out evaluation

Both trainers were copied from 32_train_final_fedavg_37f.py without changing how
tags are built, so a run's tag is identical across the two grids -- ours and
FedArtML both produce fedavg37f_k5_seed1_hd_0p9. Grids are therefore separated by
directory, every output row carries a partitioner column, confusion-matrix
filenames are prefixed with the partitioner, and each row also carries a run_key of
the form partitioner/tag. Joining on partition_id alone would silently merge the two
grids, because they reuse the same partition_id strings for different partitions.

No centralised model is evaluated here: the centralised baseline was trained on the
whole training set and has no partition, so it belongs to neither grid. Compare
against results/centralised_corrected_37f if a centralised reference is needed.

Reads the run configs, histories and checkpoints of both grids, each grid's
seed_mapping.csv, and on --evaluate-test the 37-feature test arrays. Writes only
under results/hd_grids_test_37f, and refuses to start if that directory exists.
"""

from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch import nn

PROCESSED_DIR = Path("data/processed_37f")
LABEL_MAP = Path("configs/label_mapping.json")
OUT_DIR = Path("results/hd_grids_test_37f")

FEATURE_MANIFEST = PROCESSED_DIR / "feature_manifest.json"

INPUT_DIM = 37
NUM_CLASSES = 10
BATCH_SIZE = 4096
PARTITION_SEEDS = [1, 2, 3]
CONDITIONS = ["iid", "hd_0p25", "hd_0p5", "hd_0p75", "hd_0p9"]
EXPECTED_ROUNDS = 40
EXPECTED_MODELS = 30
EXPECTED_TEST_ROWS = 358542
RELOAD_F1_TOL = 1e-4

# The two grids differ only in which partitioner produced their partitions, so each
# needs its own partition root, output directories and trainer script. Everything
# downstream of selection is shared. Keeping them in one table means a grid cannot be
# half-configured: a wrong path fails on the first checkpoint rather than silently
# reading the other grid's files.
GRIDS = {
    "classwise_dirichlet": {
        "results_dir": Path("results/final_fedavg_k5_hd_selected"),
        "models_dir": Path("models/final_fedavg_k5_hd_selected"),
        "part_root": Path("data/fl_clients/hd_selected_partitions/k_5"),
        "trainer_script": Path("44_train_fedavg_hd_selected.py"),
        "seed_mapping": Path("data/fl_clients/hd_selected_partitions/k_5/seed_mapping.csv"),
    },
    "fedartml": {
        "results_dir": Path("results/final_fedavg_k5_hd_fedartml"),
        "models_dir": Path("models/final_fedavg_k5_hd_fedartml"),
        "part_root": Path("data/fl_clients/hd_selected_fedartml/k_5"),
        "trainer_script": Path("46_train_fedavg_hd_fedartml.py"),
        "seed_mapping": Path("data/fl_clients/hd_selected_fedartml/k_5/seed_mapping.csv"),
    },
}

# Both grids write tags with this prefix, because both trainers were copied from
# 32_train_final_fedavg_37f.py without changing the tag construction. The grid is
# distinguished by directory, not by tag, so tags collide across grids by design and
# every output column carries the partitioner name.
TAG_PREFIX = "fedavg37f_k5"

# Deterministic synthetic batch for --smoke-checkpoints; never real data.
SMOKE_ROWS = 8
SMOKE_SEED = 0
# Summary and history hold the same value; this allows only CSV round-tripping.
SELECTION_TOL = 1e-12

# These checkpoints are raw state_dicts with no recorded metadata, so the model
# contract is enforced on tensor geometry instead: a 37-wide input layer and a
# 10-way output layer, with exactly these six keys and nothing else.
EXPECTED_STATE_SHAPES = {
    "network.0.weight": (128, INPUT_DIM), "network.0.bias": (128,),
    "network.3.weight": (64, 128), "network.3.bias": (64,),
    "network.6.weight": (NUM_CLASSES, 64), "network.6.bias": (NUM_CLASSES,),
}

# Settings every Dataset-1 37-feature federated config records, checked on every run.
COMMON_CONFIG = {
    "representation": "37f",
    "input_dim": INPUT_DIM,
    "K": 5,
    "num_clients": 5,
    "training_seed": 42,
    "batch_size": 4096,
    "lr": 0.1,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "optimizer": "SGD",
    "local_epochs": 1,
    "max_rounds": EXPECTED_ROUNDS,
    "selection_metric": "val_macro_f1",
    "aggregation": "sample_weighted_fedavg",
    "processed_dir": str(PROCESSED_DIR),
}

# Both grids are FedAvg; the partitioner is not recorded inside the run config, so
# there is nothing algorithm-specific to separate them here.
ALGORITHM_CONFIG = {
    "FedAvg": {"method": "FedAvg"},
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class MLPMultiClassClassifier(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def file_sha256(path: Path) -> str:
    # Read in chunks so large arrays are not held in memory.
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def load_class_names() -> list[str]:
    name_to_id = json.load(open(LABEL_MAP))
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    require(sorted(id_to_name) == list(range(NUM_CLASSES)),
            f"label mapping must define ids 0..{NUM_CLASSES - 1}, found {sorted(id_to_name)}")
    return [id_to_name[c] for c in range(NUM_CLASSES)]


def check_checkpoint(checkpoint_path: Path) -> str:
    """Verify a checkpoint exists, is non-empty and matches the model geometry."""
    require(checkpoint_path.exists(), f"missing checkpoint: {checkpoint_path}")
    require(checkpoint_path.stat().st_size > 0, f"empty checkpoint: {checkpoint_path}")
    # torch.load raises pickle errors on a malformed file; report them as our own.
    try:
        state = torch.load(checkpoint_path, map_location="cpu")
    except Exception as error:
        raise RuntimeError(f"{checkpoint_path}: could not be loaded "
                           f"({type(error).__name__}: {error})") from error
    require(isinstance(state, dict) and state, f"{checkpoint_path}: not a state_dict")
    shapes = {k: tuple(v.shape) for k, v in state.items()}
    require(shapes == EXPECTED_STATE_SHAPES,
            f"{checkpoint_path}: state_dict does not match the Dataset-1 37-feature model.\n"
            f"  found:    {shapes}\n  expected: {EXPECTED_STATE_SHAPES}")
    for key, tensor in state.items():
        require(bool(torch.isfinite(tensor).all()),
                f"{checkpoint_path}: non-finite values in {key}")
    return file_sha256(checkpoint_path)


def check_config(config, algorithm, seed, condition, best_round, best_macro_f1,
                 grid, tag) -> None:
    """Confirm the saved config identifies this run, this grid and this algorithm.

    The grid is identified by its partition root, its checkpoint directory and the
    trainer that produced it, so a config from the other grid cannot pass.
    """
    models_dir = grid["models_dir"]
    part_root = grid["part_root"]
    expected = {**COMMON_CONFIG, **ALGORITHM_CONFIG[algorithm],
                "partition_seed": seed, "condition": condition, "best_round": best_round}
    for key, value in expected.items():
        require(key in config, f"{tag}: config is missing required field '{key}'")
        require(config[key] == value,
                f"{tag}: config {key} is {config[key]!r}, expected {value!r}")

    # Full participation: one aggregation weight per client, summing to 1.
    require("aggregation_weights" in config, f"{tag}: config is missing aggregation_weights")
    weights = config["aggregation_weights"]
    require(len(weights) == COMMON_CONFIG["K"],
            f"{tag}: {len(weights)} aggregation weights, expected {COMMON_CONFIG['K']}")
    require(abs(sum(weights) - 1.0) < 1e-9,
            f"{tag}: aggregation weights sum to {sum(weights)!r}, expected 1.0")

    expected_paths = {
        "partition_path": str(part_root / f"seed_{seed}" / condition),
        "initial_state_path": str(models_dir / "initial_global_model.pt"),
        "best_checkpoint_path": str(models_dir / f"best_{tag}.pt"),
        "final_checkpoint_path": str(models_dir / f"final_{tag}.pt"),
    }
    for key, value in expected_paths.items():
        require(key in config, f"{tag}: config is missing required field '{key}'")
        require(config[key] == value,
                f"{tag}: config {key} is {config[key]!r}, expected {value!r}")

    # Bind the stored result to the training script still present in the repository.
    require("script_sha256" in config, f"{tag}: config is missing script_sha256")
    script = grid["trainer_script"]
    require(script.exists(), f"{tag}: training script not found: {script}")
    require(config["script_sha256"] == file_sha256(script),
            f"{tag}: config script_sha256 {config['script_sha256']} does not match "
            f"the current {script}")

    require("best_checkpoint_reloaded_val_macro_f1" in config,
            f"{tag}: config is missing best_checkpoint_reloaded_val_macro_f1")
    reloaded = float(config["best_checkpoint_reloaded_val_macro_f1"])
    require(abs(reloaded - best_macro_f1) < RELOAD_F1_TOL,
            f"{tag}: reloaded val macro-F1 {reloaded!r} disagrees with the selected "
            f"{best_macro_f1!r}")


def select_federated(algorithm, partitioner, grid) -> list[dict]:
    """Re-derive each run's validation selection from its own history and config."""
    results_dir = grid["results_dir"]
    models_dir = grid["models_dir"]
    summary_path = results_dir / "final_summary.csv"
    expected_runs = len(PARTITION_SEEDS) * len(CONDITIONS)

    require(summary_path.exists(), f"missing summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    require(len(summary) == expected_runs,
            f"{summary_path}: expected {expected_runs} rows, found {len(summary)}")

    selected = []
    for seed in PARTITION_SEEDS:
        for condition in CONDITIONS:
            tag = f"{TAG_PREFIX}_seed{seed}_{condition}"
            rows = summary[(summary["partition_seed"] == seed)
                           & (summary["condition"] == condition)]
            require(len(rows) == 1, f"{summary_path}: {tag} appears {len(rows)} times")
            row = rows.iloc[0]

            history_path = results_dir / f"history_{tag}.csv"
            config_path = results_dir / f"config_{tag}.json"
            require(history_path.exists(), f"missing history: {history_path}")
            require(config_path.exists(), f"missing config: {config_path}")

            history = pd.read_csv(history_path)
            require(len(history) == EXPECTED_ROUNDS,
                    f"{history_path}: expected {EXPECTED_ROUNDS} rounds, found {len(history)}")
            require(history["round"].tolist() == list(range(1, EXPECTED_ROUNDS + 1)),
                    f"{history_path}: rounds are not 1..{EXPECTED_ROUNDS} in order")

            best_round = int(history.loc[history["macro_f1"].idxmax(), "round"])
            best_macro_f1 = float(history["macro_f1"].max())
            require(best_round == int(row["best_round"]),
                    f"{tag}: summary best_round {int(row['best_round'])} != "
                    f"history argmax {best_round}")
            require(abs(best_macro_f1 - float(row["best_val_macro_f1"])) <= SELECTION_TOL,
                    f"{tag}: summary best_val_macro_f1 {row['best_val_macro_f1']!r} != "
                    f"history maximum {best_macro_f1!r}")

            config = json.load(open(config_path))
            check_config(config, algorithm, seed, condition, best_round, best_macro_f1,
                         grid, tag)

            # The best-validation checkpoint, never final_*.
            checkpoint_path = models_dir / f"best_{tag}.pt"
            selected.append({
                "partitioner": partitioner,
                "method": algorithm,
                "partition_seed": seed,
                "condition": condition,
                "partition_id": f"k5_seed{seed}_{condition}",
                "selected_round_or_epoch": best_round,
                "val_macro_f1": best_macro_f1,
                "checkpoint_file": checkpoint_path.name,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": check_checkpoint(checkpoint_path),
                # Tags collide across grids, so carry a key that does not.
                "run_key": f"{partitioner}/{tag}",
                "tag": tag,
            })
    return selected


def check_selection() -> list[dict]:
    """All validation-based selection checks, both grids. Reads no test array."""
    selected = []
    for partitioner, grid in GRIDS.items():
        selected += select_federated("FedAvg", partitioner, grid)

    require(len(selected) == EXPECTED_MODELS,
            f"expected {EXPECTED_MODELS} selected models, found {len(selected)}")
    paths = [m["checkpoint_path"] for m in selected]
    require(len(set(paths)) == len(paths), "a checkpoint was selected more than once")
    require(not any("final_" in Path(p).name for p in paths),
            "a final-round checkpoint was selected instead of a best-validation checkpoint")

    # Tags are identical across the two grids, so a mistake in the directory
    # constants could quietly select the same 15 checkpoints twice. The path check
    # above catches that, and this confirms the intended split explicitly.
    for partitioner, grid in GRIDS.items():
        runs = [m for m in selected if m["partitioner"] == partitioner]
        require(len(runs) == len(PARTITION_SEEDS) * len(CONDITIONS),
                f"{partitioner}: {len(runs)} runs selected, expected "
                f"{len(PARTITION_SEEDS) * len(CONDITIONS)}")
        require(all(str(grid["models_dir"]) in m["checkpoint_path"] for m in runs),
                f"{partitioner}: a checkpoint came from outside {grid['models_dir']}")

    # Every run must match a row in its own grid's seed_mapping.csv, since the
    # analysis joins on partition_id within a grid; an unmatched id would silently
    # drop a run. The two grids use the same partition_id strings for different
    # partitions, so this is checked per grid rather than pooled.
    for partitioner, grid in GRIDS.items():
        mapping_path = grid["seed_mapping"]
        require(mapping_path.exists(), f"{partitioner}: missing {mapping_path}")
        mapping = pd.read_csv(mapping_path)
        known = {f"k5_{row.directory_label.replace('seed_', 'seed')}_{row.condition}"
                 for row in mapping.itertuples()}
        ids = {m["partition_id"] for m in selected if m["partitioner"] == partitioner}
        missing = sorted(ids - known)
        require(not missing,
                f"{partitioner}: partition_id values absent from {mapping_path}: {missing}")
    return selected


def check_test_arrays_against_manifest() -> None:
    """Confirm the test arrays match the branch manifest 38 wrote.

    Only called on the --evaluate-test path, where the arrays are read anyway.
    """
    require(FEATURE_MANIFEST.exists(), f"missing feature manifest: {FEATURE_MANIFEST}")
    manifest = json.load(open(FEATURE_MANIFEST))
    require(manifest["corrected_feature_count"] == INPUT_DIM,
            f"{FEATURE_MANIFEST}: corrected_feature_count is "
            f"{manifest['corrected_feature_count']}, expected {INPUT_DIM}")
    require(len(manifest["retained_original_indices"]) == INPUT_DIM,
            f"{FEATURE_MANIFEST}: retained_original_indices has "
            f"{len(manifest['retained_original_indices'])} entries, expected {INPUT_DIM}")
    require("test" in manifest["row_counts"],
            f"{FEATURE_MANIFEST}: row_counts does not record the test split")
    require(int(manifest["row_counts"]["test"]) == EXPECTED_TEST_ROWS,
            f"{FEATURE_MANIFEST}: row_counts test is {manifest['row_counts']['test']}, "
            f"expected {EXPECTED_TEST_ROWS}")


@torch.inference_mode()
def smoke_checkpoint(checkpoint_path) -> None:
    """Load a selected checkpoint strictly and run one deterministic synthetic batch."""
    model = MLPMultiClassClassifier()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    model.eval()

    generator = torch.Generator().manual_seed(SMOKE_SEED)
    batch = torch.randn(SMOKE_ROWS, INPUT_DIM, generator=generator, dtype=torch.float32)
    logits = model(batch)
    require(tuple(logits.shape) == (SMOKE_ROWS, NUM_CLASSES),
            f"{checkpoint_path}: logits shape {tuple(logits.shape)}, "
            f"expected {(SMOKE_ROWS, NUM_CLASSES)}")
    require(bool(torch.isfinite(logits).all()), f"{checkpoint_path}: non-finite logits")

    probs = torch.softmax(logits, dim=1)
    require(bool(torch.isfinite(probs).all()), f"{checkpoint_path}: non-finite probabilities")
    row_sums = probs.sum(dim=1)
    require(bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)),
            f"{checkpoint_path}: probability rows do not sum to 1")


def smoke_all(selected) -> None:
    for model in selected:
        smoke_checkpoint(Path(model["checkpoint_path"]))
        print(f"  SMOKE PASS  {model['tag']}", flush=True)


def compute_metrics(y_true, y_pred, y_prob, class_names) -> dict:
    """Test metrics, matching the definitions used during training."""
    labels = list(range(NUM_CLASSES))
    support = np.bincount(y_true, minlength=NUM_CLASSES)
    per_p, per_r, per_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    predicted_count = np.bincount(y_pred, minlength=NUM_CLASSES)

    per_ap = np.full(NUM_CLASSES, np.nan)
    for c in labels:
        if support[c] > 0:
            per_ap[c] = average_precision_score((y_true == c).astype(int), y_prob[:, c])
    supported = [c for c in labels if support[c] > 0]

    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels,
                                                 average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels,
                                           average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels,
                                      average="weighted", zero_division=0)),
        # Mean one-vs-rest average precision. Named accurately rather than "PR-AUC".
        "macro_average_precision": float(np.nanmean(per_ap)),
        "worst_class_f1": float(min(per_f[c] for c in supported)),
        "worst_class_recall": float(min(per_r[c] for c in supported)),
        "n_samples": int(y_true.shape[0]),
    }
    per_class = [{
        "class_id": c,
        "class_name": class_names[c],
        "precision": float(per_p[c]),
        "recall": float(per_r[c]),
        "f1": float(per_f[c]),
        "average_precision": float(per_ap[c]),
        "support": int(support[c]),
        "predicted_count": int(predicted_count[c]),
    } for c in labels]
    return {"overall": overall, "per_class": per_class,
            "confusion": confusion_matrix(y_true, y_pred, labels=labels)}


@torch.inference_mode()
def evaluate_model(checkpoint_path, x_test, y_test, device, class_names) -> dict:
    model = MLPMultiClassClassifier().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    preds, probs = [], []
    for start in range(0, x_test.shape[0], BATCH_SIZE):
        block = np.asarray(x_test[start:start + BATCH_SIZE], dtype=np.float32)
        # Checked per batch rather than in a separate pass over the whole memmap.
        require(bool(np.isfinite(block).all()),
                f"non-finite test features in rows {start}..{start + block.shape[0]}")
        batch = torch.from_numpy(block).to(device)
        logits = model(batch)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())

    y_pred = np.concatenate(preds).astype(int)
    y_prob = np.concatenate(probs)
    require(y_pred.shape[0] == y_test.shape[0],
            f"{checkpoint_path}: {y_pred.shape[0]} predictions for {y_test.shape[0]} test rows")
    return compute_metrics(y_test, y_pred, y_prob, class_names)


def save_results(selected, results, class_names, device, out_dir=OUT_DIR) -> None:
    out_dir = Path(out_dir)
    confusion_dir = out_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True)

    pd.DataFrame([{k: m[k] for k in
                   ("partitioner", "method", "partition_seed", "condition", "partition_id",
                    "selected_round_or_epoch", "val_macro_f1", "checkpoint_file",
                    "checkpoint_path", "checkpoint_sha256", "run_key")}
                  for m in selected]).to_csv(out_dir / "selected_checkpoints.csv", index=False)

    overall_rows, per_class_rows = [], []
    for model, result in zip(selected, results):
        identity = {"partitioner": model["partitioner"], "method": model["method"],
                    "partition_seed": model["partition_seed"],
                    "condition": model["condition"], "partition_id": model["partition_id"],
                    "tag": model["tag"], "run_key": model["run_key"],
                    "checkpoint_file": model["checkpoint_file"]}
        overall_rows.append({**identity,
                             "selected_round_or_epoch": model["selected_round_or_epoch"],
                             "val_macro_f1": model["val_macro_f1"],
                             **result["overall"]})
        for entry in result["per_class"]:
            per_class_rows.append({**identity, **entry})

        raw = result["confusion"]
        index = pd.Index(class_names, name="true")
        # The two grids produce identical tags, so the partitioner has to be in the
        # filename or the second grid silently overwrites the first grid's matrices.
        stem = f"{model['partitioner']}_{model['tag']}"
        pd.DataFrame(raw, index=index, columns=class_names).to_csv(
            confusion_dir / f"confusion_raw_{stem}.csv")
        # Row-normalised: recall per true class. Rows with no support stay 0.
        totals = raw.sum(axis=1, keepdims=True)
        normalised = np.divide(raw, totals, out=np.zeros(raw.shape, dtype=float),
                               where=totals > 0)
        pd.DataFrame(normalised, index=index, columns=class_names).to_csv(
            confusion_dir / f"confusion_normalised_{stem}.csv")

    pd.DataFrame(overall_rows).to_csv(out_dir / "test_results_overall.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out_dir / "test_results_per_class.csv", index=False)

    config = {
        "dataset": "NF-UNSW-NB15-v2",
        "representation": "37f",
        "evaluation": "final_held_out_test",
        "model_selection": "validation macro-F1 only; test never used for selection",
        "n_models": len(selected),
        "models_by_partitioner": pd.Series(
            [m["partitioner"] for m in selected]).value_counts().to_dict(),
        "grids": {name: {key: str(value) for key, value in grid.items()}
                  for name, grid in GRIDS.items()},
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
        "class_names": class_names,
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "test_rows": int(results[0]["overall"]["n_samples"]),
        "processed_dir": str(PROCESSED_DIR),
        "join_key": "partitioner + partition_id; partition_id alone is ambiguous "
                    "because both grids use the same partition_id strings",
        "x_test_sha256": file_sha256(PROCESSED_DIR / "X_test.npy"),
        "y_test_sha256": file_sha256(PROCESSED_DIR / "y_test.npy"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git_commit": git_commit_hash(),
    }
    with open(out_dir / "evaluation_config.json", "w") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Held-out test evaluation of the two Dataset-1 HD-matched "
                    "FedAvg grids.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-selection", action="store_true",
                      help="check every selection; never opens the test arrays")
    mode.add_argument("--smoke-checkpoints", action="store_true",
                      help="selection checks, then a synthetic forward pass per checkpoint")
    mode.add_argument("--evaluate-test", action="store_true",
                      help="selection and smoke checks, then evaluate on the held-out test set")
    args = parser.parse_args()

    class_names = load_class_names()
    selected = check_selection()
    print(f"selection verified: {len(selected)} models")
    for model in selected:
        label = (f"{model['partitioner']} seed{model['partition_seed']} "
                 f"{model['condition']}")
        print(f"  {label:<34} round={model['selected_round_or_epoch']:>3} "
              f"val_macro_f1={model['val_macro_f1']:.6f} "
              f"sha256={model['checkpoint_sha256'][:16]}")

    if args.verify_selection:
        print("\n--verify-selection complete. No test array was opened and no output written.")
        return

    print(f"\nsmoke-testing {len(selected)} checkpoints on a synthetic "
          f"({SMOKE_ROWS}, {INPUT_DIM}) batch", flush=True)
    smoke_all(selected)

    if args.smoke_checkpoints:
        print("\n--smoke-checkpoints complete. No test array was opened and no output written.")
        return

    require(not OUT_DIR.exists(),
            f"Refusing to run: {OUT_DIR} already exists. Move or inspect it manually.")
    device = get_device()

    # Test arrays are opened only after every check above has passed.
    check_test_arrays_against_manifest()
    x_test = np.load(PROCESSED_DIR / "X_test.npy", mmap_mode="r")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    require(x_test.ndim == 2 and x_test.shape[1] == INPUT_DIM,
            f"X_test has shape {x_test.shape}, expected (rows, {INPUT_DIM})")
    require(x_test.shape[0] == EXPECTED_TEST_ROWS,
            f"X_test has {x_test.shape[0]} rows, expected {EXPECTED_TEST_ROWS}")
    require(y_test.ndim == 1, f"y_test has shape {y_test.shape}, expected one dimension")
    require(np.issubdtype(y_test.dtype, np.integer),
            f"y_test dtype is {y_test.dtype}, expected an integer type")
    require(x_test.shape[0] == y_test.shape[0],
            f"X_test/y_test row mismatch: {x_test.shape[0]} vs {y_test.shape[0]}")
    require(sorted(int(v) for v in np.unique(y_test).tolist()) == list(range(NUM_CLASSES)),
            f"y_test does not contain exactly the {NUM_CLASSES} Dataset-1 classes")
    print(f"\nevaluating {len(selected)} models on {y_test.shape[0]:,} test rows "
          f"(device={device})", flush=True)

    results = []
    for model in selected:
        result = evaluate_model(model["checkpoint_path"], x_test, y_test, device, class_names)
        results.append(result)
        print(f"  {model['run_key']:<56} test_macro_f1={result['overall']['macro_f1']:.6f}",
              flush=True)

    OUT_DIR.mkdir(parents=True)
    save_results(selected, results, class_names, device)
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
