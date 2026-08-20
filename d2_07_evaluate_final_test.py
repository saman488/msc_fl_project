"""
Dataset-2 final held-out test evaluation.

Evaluates the 37 validation-selected models: 1 centralised, 12 FedAvg, 12 FedProx,
12 SCAFFOLD. Selection is re-derived from the saved histories, summaries and
configs; the test set is never used to choose anything.

    --verify-selection   selection checks only, no test-array access
    --evaluate-test      the same checks, then held-out evaluation
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

PROCESSED_DIR = Path("data/nf_cse_cic_ids2018_v2/processed")
LABEL_MAP = Path("configs/nf_cse_cic_ids2018_v2/label_mapping.json")
OUT_DIR = Path("results/nf_cse_cic_ids2018_v2/final_test")

CENTRAL_RESULTS = Path("results/nf_cse_cic_ids2018_v2/centralised")
CENTRAL_MODELS = Path("models/nf_cse_cic_ids2018_v2/centralised")
FEDAVG_RESULTS = Path("results/nf_cse_cic_ids2018_v2/final_fedavg_k5")
FEDAVG_MODELS = Path("models/nf_cse_cic_ids2018_v2/final_fedavg_k5")
FEDPROX_RESULTS = Path("results/nf_cse_cic_ids2018_v2/final_fedprox_k5")
FEDPROX_MODELS = Path("models/nf_cse_cic_ids2018_v2/final_fedprox_k5")
SCAFFOLD_RESULTS = Path("results/nf_cse_cic_ids2018_v2/final_scaffold_k5")
SCAFFOLD_MODELS = Path("models/nf_cse_cic_ids2018_v2/final_scaffold_k5")

INPUT_DIM = 36
NUM_CLASSES = 7
BATCH_SIZE = 4096
PARTITION_SEEDS = [42, 43, 44]
CONDITIONS = ["iid", "alpha_0p1", "alpha_0p5", "alpha_1p0"]
FEDPROX_MU_FRAGMENT = "1em05"
EXPECTED_ROUNDS = 40
EXPECTED_EPOCHS = 20
EXPECTED_MODELS = 37
RELOAD_F1_TOL = 1e-4

PART_ROOT = Path("data/nf_cse_cic_ids2018_v2/fl_clients/final_partitions/k_5")
FEDAVG_INIT_PATH = FEDAVG_MODELS / "initial_global_model.pt"
MANIFEST_PATH = Path("results/nf_cse_cic_ids2018_v2/preprocessing/artifact_manifest_sha256.json")

# Each stored config records the SHA-256 of the script that produced it.
TRAINER_SCRIPTS = {
    "Centralised": Path("d2_03_train_central_mlp.py"),
    "FedAvg": Path("d2_04_train_fedavg.py"),
    "FedProx": Path("d2_05_train_fedprox.py"),
    "SCAFFOLD": Path("d2_06_train_scaffold.py"),
}

# Deterministic synthetic batch for --smoke-checkpoints; never real data.
SMOKE_ROWS = 8
SMOKE_SEED = 0
# Summary and history hold the same value; this allows only CSV round-tripping.
SELECTION_TOL = 1e-12

EXPECTED_STATE_SHAPES = {
    "network.0.weight": (128, INPUT_DIM), "network.0.bias": (128,),
    "network.3.weight": (64, 128), "network.3.bias": (64,),
    "network.6.weight": (NUM_CLASSES, 64), "network.6.bias": (NUM_CLASSES,),
}

# Settings every Dataset-2 federated config records, checked on every run.
COMMON_CONFIG = {
    "dataset": "nf_cse_cic_ids2018_v2",
    "input_dim": INPUT_DIM,
    "num_classes": NUM_CLASSES,
    "K": 5,
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
}

# Algorithm-identifying settings, taken from the saved configs.
ALGORITHM_CONFIG = {
    "FedAvg": {"method": "FedAvg"},
    "FedProx": {
        "method": "FedProx",
        "mu": 1e-5,
        "production_mu": 1e-5,
        "mu_tuned_on_dataset2": False,
        "proximal_term": "0.5 * mu * sum_i ||w_i - w_i_server||^2",
    },
    "SCAFFOLD": {
        "method": "SCAFFOLD",
        "control_variate_option": "SCAFFOLD Option II",
        "gradient_correction": "g - c_i + c",
        "tau_source": "actual optimizer.step() count",
        "local_steps_source": "counted optimizer.step() calls",
        "model_aggregation": "sample weighted by actual client sizes",
        "server_control_aggregation":
            "sample weighted using the same n_k / sum_j n_j weights as the model",
    },
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
    """Verify a checkpoint exists, is non-empty and matches the model contract."""
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
            f"{checkpoint_path}: state_dict does not match the Dataset-2 model.\n"
            f"  found:    {shapes}\n  expected: {EXPECTED_STATE_SHAPES}")
    for key, tensor in state.items():
        require(bool(torch.isfinite(tensor).all()),
                f"{checkpoint_path}: non-finite values in {key}")
    return file_sha256(checkpoint_path)


def check_config(config, algorithm, seed, condition, best_round, best_macro_f1,
                 results_dir, models_dir, tag) -> None:
    """Confirm the saved config identifies this Dataset-2 run and its algorithm."""
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
        "partition_path": str(PART_ROOT / f"seed_{seed}" / condition),
        "initial_state_path": str(FEDAVG_INIT_PATH),
        "best_checkpoint_path": str(models_dir / f"best_{tag}.pt"),
        "final_checkpoint_path": str(models_dir / f"final_{tag}.pt"),
    }
    for key, value in expected_paths.items():
        require(key in config, f"{tag}: config is missing required field '{key}'")
        require(config[key] == value,
                f"{tag}: config {key} is {config[key]!r}, expected {value!r}")

    # Bind the stored result to the training script still present in the repository.
    require("script_sha256" in config, f"{tag}: config is missing script_sha256")
    script = TRAINER_SCRIPTS[algorithm]
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


def select_federated(algorithm, results_dir, models_dir, summary_path, tag_prefix) -> list[dict]:
    """Re-derive each run's validation selection from its own history and config."""
    require(summary_path.exists(), f"missing summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    require(len(summary) == len(PARTITION_SEEDS) * len(CONDITIONS),
            f"{summary_path}: expected 12 rows, found {len(summary)}")

    selected = []
    for seed in PARTITION_SEEDS:
        for condition in CONDITIONS:
            tag = f"{tag_prefix}_seed{seed}_{condition}"
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
                         results_dir, models_dir, tag)

            # The best-validation checkpoint, never final_*.
            checkpoint_path = models_dir / f"best_{tag}.pt"
            selected.append({
                "algorithm": algorithm,
                "seed": seed,
                "condition": condition,
                "mu": config["mu"] if algorithm == "FedProx" else None,
                "selected_round_or_epoch": best_round,
                "val_macro_f1": best_macro_f1,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": check_checkpoint(checkpoint_path),
                "tag": tag,
            })
    return selected


def select_central() -> dict:
    """Re-derive the centralised selection from its history and validation summary."""
    history_path = CENTRAL_RESULTS / "central_mlp_training_history.csv"
    summary_path = CENTRAL_RESULTS / "validation_summary.csv"
    require(history_path.exists(), f"missing history: {history_path}")
    require(summary_path.exists(), f"missing summary: {summary_path}")

    history = pd.read_csv(history_path)
    require(len(history) == EXPECTED_EPOCHS,
            f"{history_path}: expected {EXPECTED_EPOCHS} epochs, found {len(history)}")
    require(history["epoch"].tolist() == list(range(1, EXPECTED_EPOCHS + 1)),
            f"{history_path}: epochs are not 1..{EXPECTED_EPOCHS} in order")

    best_epoch = int(history.loc[history["val_macro_f1"].idxmax(), "epoch"])
    best_macro_f1 = float(history["val_macro_f1"].max())

    summary = pd.read_csv(summary_path)
    require(len(summary) == 1, f"{summary_path}: expected 1 row, found {len(summary)}")
    row = summary.iloc[0]
    require(best_epoch == int(row["best_epoch"]),
            f"centralised: summary best_epoch {int(row['best_epoch'])} != "
            f"history argmax {best_epoch}")
    require(abs(best_macro_f1 - float(row["best_val_macro_f1"])) <= SELECTION_TOL,
            f"centralised: summary best_val_macro_f1 {row['best_val_macro_f1']!r} != "
            f"history maximum {best_macro_f1!r}")

    for key in ("reloaded_val_macro_f1", "reload_macro_f1_delta",
                "holdout_evaluation_split_used", "script_name", "script_sha256", "epochs_run"):
        require(key in summary.columns, f"centralised: summary is missing '{key}'")
    require(int(row["epochs_run"]) == EXPECTED_EPOCHS,
            f"centralised: epochs_run is {row['epochs_run']!r}, expected {EXPECTED_EPOCHS}")

    reloaded = float(row["reloaded_val_macro_f1"])
    require(abs(reloaded - best_macro_f1) < RELOAD_F1_TOL,
            f"centralised: reloaded val macro-F1 {reloaded!r} disagrees with the selected "
            f"{best_macro_f1!r}")
    require(abs(float(row["reload_macro_f1_delta"]) - abs(reloaded - best_macro_f1))
            < RELOAD_F1_TOL, "centralised: reload_macro_f1_delta is inconsistent")

    # The CSV value may load as a bool or as the string "True"/"False"; accept only
    # those forms, so an unexpected value fails instead of being coerced.
    holdout = row["holdout_evaluation_split_used"]
    if isinstance(holdout, (bool, np.bool_)):
        holdout_used = bool(holdout)
    else:
        text = str(holdout).strip().lower()
        require(text in ("true", "false"),
                f"centralised: holdout_evaluation_split_used is {holdout!r}, "
                "expected a boolean or 'true'/'false'")
        holdout_used = text == "true"
    require(not holdout_used,
            "centralised: holdout_evaluation_split_used is not false")

    script = TRAINER_SCRIPTS["Centralised"]
    require(str(row["script_name"]) == script.name,
            f"centralised: script_name is {row['script_name']!r}, expected {script.name!r}")
    require(script.exists(), f"centralised: training script not found: {script}")
    require(str(row["script_sha256"]) == file_sha256(script),
            f"centralised: summary script_sha256 does not match the current {script}")

    checkpoint_path = CENTRAL_MODELS / "central_mlp_best.pt"
    return {
        "algorithm": "Centralised",
        "seed": None,
        "condition": None,
        "mu": None,
        "selected_round_or_epoch": best_epoch,
        "val_macro_f1": best_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": check_checkpoint(checkpoint_path),
        "tag": "central_mlp_best",
    }


def check_selection() -> list[dict]:
    """All validation-based selection checks. Reads no test array."""
    selected = [select_central()]
    selected += select_federated(
        "FedAvg", FEDAVG_RESULTS, FEDAVG_MODELS,
        FEDAVG_RESULTS / "final_summary.csv", "fedavg_k5")
    selected += select_federated(
        "FedProx", FEDPROX_RESULTS, FEDPROX_MODELS,
        FEDPROX_RESULTS / f"final_summary_mu{FEDPROX_MU_FRAGMENT}.csv",
        f"fedprox_k5_mu{FEDPROX_MU_FRAGMENT}")
    selected += select_federated(
        "SCAFFOLD", SCAFFOLD_RESULTS, SCAFFOLD_MODELS,
        SCAFFOLD_RESULTS / "final_summary.csv", "scaffold_k5")

    require(len(selected) == EXPECTED_MODELS,
            f"expected {EXPECTED_MODELS} selected models, found {len(selected)}")
    paths = [m["checkpoint_path"] for m in selected]
    require(len(set(paths)) == len(paths), "a checkpoint was selected more than once")
    require(not any("final_" in Path(p).name for p in paths),
            "a final-round checkpoint was selected instead of a best-validation checkpoint")
    return selected


def check_test_file_hashes() -> None:
    """Confirm the test arrays match the SHA-256 recorded by preprocessing.

    Only called on the --evaluate-test path, where the arrays are read anyway.
    """
    require(MANIFEST_PATH.exists(), f"missing preprocessing manifest: {MANIFEST_PATH}")
    manifest = json.load(open(MANIFEST_PATH))
    recorded = {entry["path"]: entry["sha256"] for entry in manifest["artifacts"]}
    for name in ("X_test.npy", "y_test.npy"):
        path = PROCESSED_DIR / name
        key = str(path)
        require(key in recorded, f"the preprocessing manifest does not record {key}")
        require(file_sha256(path) == recorded[key],
                f"{path} does not match the SHA-256 recorded by preprocessing")


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
                   ("algorithm", "seed", "condition", "mu", "selected_round_or_epoch",
                    "val_macro_f1", "checkpoint_path", "checkpoint_sha256")}
                  for m in selected]).to_csv(out_dir / "selected_checkpoints.csv", index=False)

    overall_rows, per_class_rows = [], []
    for model, result in zip(selected, results):
        identity = {"algorithm": model["algorithm"], "seed": model["seed"],
                    "condition": model["condition"], "mu": model["mu"], "tag": model["tag"]}
        overall_rows.append({**identity,
                             "selected_round_or_epoch": model["selected_round_or_epoch"],
                             "val_macro_f1": model["val_macro_f1"],
                             **result["overall"]})
        for entry in result["per_class"]:
            per_class_rows.append({**identity, **entry})

        raw = result["confusion"]
        index = pd.Index(class_names, name="true")
        pd.DataFrame(raw, index=index, columns=class_names).to_csv(
            confusion_dir / f"confusion_raw_{model['tag']}.csv")
        # Row-normalised: recall per true class. Rows with no support stay 0.
        totals = raw.sum(axis=1, keepdims=True)
        normalised = np.divide(raw, totals, out=np.zeros(raw.shape, dtype=float),
                               where=totals > 0)
        pd.DataFrame(normalised, index=index, columns=class_names).to_csv(
            confusion_dir / f"confusion_normalised_{model['tag']}.csv")

    pd.DataFrame(overall_rows).to_csv(out_dir / "test_results_overall.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out_dir / "test_results_per_class.csv", index=False)

    config = {
        "dataset": "nf_cse_cic_ids2018_v2",
        "evaluation": "final_held_out_test",
        "model_selection": "validation macro-F1 only; test never used for selection",
        "n_models": len(selected),
        "models_by_algorithm": pd.Series([m["algorithm"] for m in selected]).value_counts().to_dict(),
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
        "class_names": class_names,
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "test_rows": int(results[0]["overall"]["n_samples"]),
        "processed_dir": str(PROCESSED_DIR),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git_commit": git_commit_hash(),
    }
    with open(out_dir / "evaluation_config.json", "w") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset-2 final held-out test evaluation.")
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
        label = model["algorithm"] if model["seed"] is None else \
            f"{model['algorithm']} seed{model['seed']} {model['condition']}"
        print(f"  {label:<34} round/epoch={model['selected_round_or_epoch']:>3} "
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
    check_test_file_hashes()
    x_test = np.load(PROCESSED_DIR / "X_test.npy", mmap_mode="r")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    require(x_test.ndim == 2 and x_test.shape[1] == INPUT_DIM,
            f"X_test has shape {x_test.shape}, expected (rows, {INPUT_DIM})")
    require(y_test.ndim == 1, f"y_test has shape {y_test.shape}, expected one dimension")
    require(np.issubdtype(y_test.dtype, np.integer),
            f"y_test dtype is {y_test.dtype}, expected an integer type")
    require(x_test.shape[0] == y_test.shape[0],
            f"X_test/y_test row mismatch: {x_test.shape[0]} vs {y_test.shape[0]}")
    require(sorted(int(v) for v in np.unique(y_test).tolist()) == list(range(NUM_CLASSES)),
            "y_test does not contain exactly the 7 Dataset-2 classes")
    print(f"\nevaluating {len(selected)} models on {y_test.shape[0]:,} test rows "
          f"(device={device})", flush=True)

    results = []
    for model in selected:
        result = evaluate_model(model["checkpoint_path"], x_test, y_test, device, class_names)
        results.append(result)
        print(f"  {model['tag']:<40} test_macro_f1={result['overall']['macro_f1']:.6f}",
              flush=True)

    OUT_DIR.mkdir(parents=True)
    save_results(selected, results, class_names, device)
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
