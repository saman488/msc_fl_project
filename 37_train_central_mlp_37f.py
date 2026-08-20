"""
Corrected centralised MLP baseline for MULTI-CLASS intrusion detection on
NF-UNSW-NB15-v2 (Dataset 1), 37-FEATURE BRANCH.

This is the 37-feature counterpart of ``d1_30_train_central_mlp_corrected.py``.
It reads ``data/processed_37f`` and writes to the ``centralised_corrected_37f``
namespace, so that the centralised baseline is measured on the same feature set
as the federated 37-feature runs (scripts 32 / 34 / 35) rather than on the
41-feature arrays.

The existing 41-feature run is left completely untouched: ``data/processed``,
``models/centralised_corrected``, ``results/centralised_corrected`` and
``configs/centralised_corrected`` are neither read nor written by this file.

Two things differ from the 41-feature script.  The input fan-in follows
``INPUT_DIM``, and training runs to a 100-epoch cap with early stopping on
validation macro-F1 (PATIENCE=15, MIN_DELTA=0.0) instead of a fixed 20 epochs.
The 20-epoch run stopped with validation macro-F1 still rising and validation
loss still falling, so its result was a lower bound rather than a converged
ceiling; the two centralised runs are therefore NOT directly comparable to each
other on epochs, and only the 37-feature run should be compared against the
37-feature federated results.

Early stopping changes only when training halts.  Model selection is unchanged:
the reported checkpoint is still the epoch with the best validation macro-F1,
verified by reload.  The rest of the contract is preserved exactly: the model
(37 -> 128 -> 64 -> 10), the training contract (AdamW, lr=1e-3,
weight_decay=1e-5, batch_size=4096, seed 42) and the balanced class weighting
(weight[c] = N / (C * n_c)).

Corrections relative to the original script:

1.  Weighted-loss accounting.
    ``CrossEntropyLoss(weight=w, reduction="mean")`` returns a *weighted* batch
    mean, i.e. sum_i w_{y_i} l_i / sum_i w_{y_i}.  Aggregating it with
    ``loss.item() * batch_size`` is therefore wrong whenever the per-batch
    weight mass differs from batch to batch.  This script accumulates the
    exact dataset-level weighted loss:

        batch_weight       = criterion.weight[labels].sum()
        loss_numerator    += loss.detach() * batch_weight
        loss_denominator  += batch_weight
        weighted_loss      = loss_numerator / loss_denominator

    applied to both the training epoch loss and the validation loss.
    The backward pass itself is unchanged.

2.  Generic class handling.  No class ids are hard-coded; all 10 classes are
    handled through ``configs/label_mapping.json``.

3.  Full per-class reporting (precision / recall / F1 / support /
    predicted count) plus worst-class F1 and worst-class recall, recorded for
    every epoch.

4.  Checkpoint-selection verification: the best epoch is checked against the
    argmax of validation macro-F1 in the saved history, the best checkpoint is
    reloaded and re-evaluated, and the reloaded macro-F1 is asserted to match
    the stored best value.

5.  Strict evaluation isolation: this script has ZERO held-out-evaluation-split
    access.  It reads only the train and validation arrays, and a source-level
    safeguard refuses to run if the forbidden array names appear anywhere in
    this file.

6.  Isolated outputs.  Nothing produced by the original centralised run or by
    the 41-feature corrected run is read or overwritten; the script refuses to
    start if any of its own intended outputs already exist.

Federated-learning partitions, federated results and federated scripts are
neither read nor written by this file.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
import random
import subprocess
import sys

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
# Paths (isolated corrected-central namespace)
# --------------------------------------------------------------------------- #

PROCESSED_DIR = Path("data/processed_37f")
LABEL_MAPPING_PATH = Path("configs/label_mapping.json")

MODELS_DIR = Path("models/centralised_corrected_37f")
RESULTS_DIR = Path("results/centralised_corrected_37f")
CONFIGS_DIR = Path("configs/centralised_corrected_37f")

BEST_MODEL_PATH = MODELS_DIR / "central_mlp_best.pt"
FINAL_MODEL_PATH = MODELS_DIR / "central_mlp_final.pt"

HISTORY_PATH = RESULTS_DIR / "central_mlp_training_history.csv"
CLASS_WEIGHTS_PATH = RESULTS_DIR / "class_weights.npy"
VAL_REPORT_PATH = RESULTS_DIR / "validation_classification_report.csv"
VAL_CONFUSION_PATH = RESULTS_DIR / "validation_confusion_matrix.csv"
VAL_SUMMARY_PATH = RESULTS_DIR / "validation_summary.csv"

CONFIG_PATH = CONFIGS_DIR / "central_mlp_config.json"

INTENDED_OUTPUTS = (
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    HISTORY_PATH,
    CLASS_WEIGHTS_PATH,
    VAL_REPORT_PATH,
    VAL_CONFUSION_PATH,
    VAL_SUMMARY_PATH,
    CONFIG_PATH,
)


# --------------------------------------------------------------------------- #
# Contracts (unchanged from the original centralised baseline)
# --------------------------------------------------------------------------- #

RANDOM_STATE = 42
INPUT_DIM = 37
NUM_CLASSES = 10
BATCH_SIZE = 4096
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Early stopping on validation macro-F1.  PATIENCE is 15 because the epoch-to-
# epoch noise on the 41-feature curve (+/-0.008) is roughly four times the mean
# per-epoch gain over the second half of training (+0.002), and that curve
# contained a six-epoch non-improving trough (epochs 12-17) before reaching its
# eventual maximum at epoch 20.  A shorter patience would have halted inside
# that trough and discarded the best model.
PATIENCE = 15
MIN_DELTA = 0.0

# Tolerance for the best-checkpoint reload / re-evaluation consistency check.
RELOAD_TOLERANCE = 1e-6


# --------------------------------------------------------------------------- #
# Source-level evaluation-isolation safeguard
# --------------------------------------------------------------------------- #

def _forbidden_tokens() -> tuple[str, ...]:
    """Forbidden array names, assembled at runtime so the literals never appear
    in this file's own source text."""
    stem = "te" + "st"
    return ("X" + "_" + stem, "y" + "_" + stem)


def read_own_source() -> str:
    return Path(__file__).read_text(encoding="utf-8")


def assert_no_holdout_references(source: str) -> None:
    """Refuse to run if this script references the held-out evaluation arrays."""
    offending = [token for token in _forbidden_tokens() if token in source]
    if offending:
        raise RuntimeError(
            "Evaluation-isolation safeguard failed: this script references "
            f"forbidden arrays {offending}. The corrected centralised training "
            "run must have zero held-out-split access."
        )


def script_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Output safety
# --------------------------------------------------------------------------- #

def assert_outputs_absent() -> None:
    """Never overwrite: refuse to run if any intended output already exists."""
    existing = [str(path) for path in INTENDED_OUTPUTS if path.exists()]
    if existing:
        raise RuntimeError(
            "Refusing to run: the following corrected-central outputs already "
            f"exist and would be overwritten: {existing}. Remove or archive "
            "them manually before re-running."
        )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class NumpyDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"Feature and label row counts differ for {x_path.name}/"
                f"{y_path.name}: {self.x.shape[0]} vs {self.y.shape[0]}"
            )

        if self.x.ndim != 2 or self.x.shape[1] != INPUT_DIM:
            raise ValueError(
                f"Expected {INPUT_DIM} input features in {x_path.name}, "
                f"found shape {tuple(self.x.shape)}"
            )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(np.asarray(self.x[index], dtype=np.float32).copy())
        # CrossEntropyLoss expects integer class indices (long).
        label = torch.tensor(int(self.y[index]), dtype=torch.long)
        return features, label


def assert_label_array(y: np.ndarray, name: str) -> None:
    """Labels must be integer-typed and cover 0..NUM_CLASSES-1 exactly."""
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype, found {y.dtype}")

    if y.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, found shape {y.shape}")

    minimum = int(y.min())
    maximum = int(y.max())
    if minimum < 0 or maximum > NUM_CLASSES - 1:
        raise ValueError(
            f"{name} labels must lie in 0..{NUM_CLASSES - 1}, "
            f"found range [{minimum}, {maximum}]"
        )

    counts = np.bincount(y, minlength=NUM_CLASSES)
    missing = np.where(counts == 0)[0].tolist()
    if missing:
        raise ValueError(f"{name} is missing classes: {missing}")


def assert_feature_array(x: np.ndarray, name: str) -> None:
    if x.ndim != 2 or x.shape[1] != INPUT_DIM:
        raise ValueError(
            f"{name} must have width {INPUT_DIM}, found shape {tuple(x.shape)}"
        )


def load_class_names() -> list[str]:
    with open(LABEL_MAPPING_PATH) as file:
        name_to_id = json.load(file)

    id_to_name = {int(value): key for key, value in name_to_id.items()}
    if sorted(id_to_name) != list(range(NUM_CLASSES)):
        raise ValueError(
            f"Label mapping must define class ids 0..{NUM_CLASSES - 1}, "
            f"found {sorted(id_to_name)}"
        )

    return [id_to_name[class_id] for class_id in range(NUM_CLASSES)]


def build_loaders(seed: int) -> tuple[DataLoader, DataLoader]:
    """Train and validation loaders only. No held-out split is constructed."""
    train_dataset = NumpyDataset(
        PROCESSED_DIR / "X_train.npy",
        PROCESSED_DIR / "y_train.npy",
    )
    val_dataset = NumpyDataset(
        PROCESSED_DIR / "X_val.npy",
        PROCESSED_DIR / "y_val.npy",
    )

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class MLPMultiClassClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = NUM_CLASSES) -> None:
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


def assert_state_finite(state_dict: dict, context: str) -> None:
    for key, tensor in state_dict.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite values in {context} parameter '{key}'")


# --------------------------------------------------------------------------- #
# Reproducibility and environment
# --------------------------------------------------------------------------- #

def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_state() -> tuple[str, str]:
    """Return (commit hash, dirty status) for provenance; degrade gracefully."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ("unavailable", "unknown")

    return (commit, "dirty" if status else "clean")


def environment_provenance(device: torch.device, source: str) -> dict:
    commit, dirty = git_state()
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "git_commit": commit,
        "git_status": dirty,
        "script_name": Path(__file__).name,
        "script_sha256": script_sha256(source),
    }


# --------------------------------------------------------------------------- #
# Class weighting (unchanged formula)
# --------------------------------------------------------------------------- #

def compute_class_weights(
    y_train: np.ndarray, num_classes: int = NUM_CLASSES
) -> torch.Tensor:
    """Balanced class weights: weight[c] = N_total / (num_classes * count_c)."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)

    if (counts == 0).any():
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Training set is missing classes: {missing}")

    total = counts.sum()
    weights = total / (num_classes * counts)

    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError(f"Class weights must be finite and positive: {weights}")

    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Training and evaluation with exact weighted-loss accounting
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One epoch of training; returns the exact dataset-level weighted loss.

    ``criterion`` uses reduction="mean", which for a weighted CrossEntropyLoss
    is sum_i w_{y_i} l_i / sum_i w_{y_i}. Re-weighting each batch mean by that
    batch's weight mass and dividing by the total weight mass recovers the
    exact epoch-level weighted loss.
    """
    model.train()

    # float32 accumulators: the MPS backend does not support float64.
    loss_numerator = torch.zeros((), dtype=torch.float32, device=device)
    loss_denominator = torch.zeros((), dtype=torch.float32, device=device)

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_weight = criterion.weight[labels].sum().detach()
        loss_numerator += loss.detach() * batch_weight
        loss_denominator += batch_weight

    if float(loss_denominator) <= 0.0:
        raise ValueError("Empty training loader: total class-weight mass is zero")

    return float(loss_numerator / loss_denominator)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict:
    """Evaluate on a loader, returning the exact weighted loss plus metrics."""
    model.eval()

    # float32 accumulators: the MPS backend does not support float64.
    loss_numerator = torch.zeros((), dtype=torch.float32, device=device)
    loss_denominator = torch.zeros((), dtype=torch.float32, device=device)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        batch_weight = criterion.weight[labels].sum().detach()
        loss_numerator += loss.detach() * batch_weight
        loss_denominator += batch_weight

        predictions.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        targets.append(labels.detach().cpu().numpy())

    if float(loss_denominator) <= 0.0:
        raise ValueError("Empty evaluation loader: total class-weight mass is zero")

    weighted_loss = float(loss_numerator / loss_denominator)
    if not np.isfinite(weighted_loss):
        raise ValueError(f"Non-finite weighted evaluation loss: {weighted_loss}")

    y_pred = np.concatenate(predictions).astype(int)
    y_true = np.concatenate(targets).astype(int)

    return {
        "weighted_loss": weighted_loss,
        **compute_metrics(y_true, y_pred, class_names),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict:
    """Aggregate and per-class metrics for all NUM_CLASSES classes generically."""
    labels_range = list(range(NUM_CLASSES))

    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels_range,
            average=None,
            zero_division=0,
        )
    )
    predicted_counts = np.bincount(y_pred, minlength=NUM_CLASSES)

    worst_f1_index = int(np.argmin(per_class_f1))
    worst_recall_index = int(np.argmin(per_class_recall))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels_range,
                            average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels_range,
                         average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels_range,
                     average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels_range,
                     average="weighted", zero_division=0)
        ),
        "worst_class_f1": float(per_class_f1[worst_f1_index]),
        "worst_class_f1_id": worst_f1_index,
        "worst_class_f1_name": class_names[worst_f1_index],
        "worst_class_recall": float(per_class_recall[worst_recall_index]),
        "worst_class_recall_id": worst_recall_index,
        "worst_class_recall_name": class_names[worst_recall_index],
    }

    for class_id, class_name in enumerate(class_names):
        prefix = f"class{class_id}_{class_name}"
        metrics[f"precision_{prefix}"] = float(per_class_precision[class_id])
        metrics[f"recall_{prefix}"] = float(per_class_recall[class_id])
        metrics[f"f1_{prefix}"] = float(per_class_f1[class_id])
        metrics[f"support_{prefix}"] = int(per_class_support[class_id])
        metrics[f"predicted_count_{prefix}"] = int(predicted_counts[class_id])

    assert_metrics_finite(metrics)
    return metrics


def assert_metrics_finite(metrics: dict) -> None:
    """Every numeric metric must be finite; identifier strings are skipped."""
    for key, value in metrics.items():
        if isinstance(value, str):
            continue
        if not np.isfinite(value):
            raise ValueError(f"Non-finite metric '{key}': {value}")


def history_row(epoch: int, train_loss: float, val_metrics: dict) -> dict:
    """Flatten one epoch into a history record (excluding raw predictions)."""
    row = {
        "epoch": epoch,
        "weighted_train_loss": train_loss,
        "weighted_val_loss": val_metrics["weighted_loss"],
    }
    for key, value in val_metrics.items():
        if key in {"weighted_loss", "y_true", "y_pred"}:
            continue
        row[f"val_{key}"] = value
    return row


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def write_validation_reports(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> str:
    labels_range = list(range(NUM_CLASSES))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_range,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report_dict).transpose().to_csv(VAL_REPORT_PATH)

    confusion = confusion_matrix(y_true, y_pred, labels=labels_range)
    pd.DataFrame(confusion, index=class_names, columns=class_names).to_csv(
        VAL_CONFUSION_PATH
    )

    return classification_report(
        y_true,
        y_pred,
        labels=labels_range,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )


def verify_best_checkpoint(
    history_df: pd.DataFrame,
    best_epoch: int,
    best_val_macro_f1: float,
    device: torch.device,
    val_loader: DataLoader,
    criterion: nn.Module,
    class_names: list[str],
) -> dict:
    """Cross-check the recorded best epoch, then reload and re-evaluate it."""
    argmax_epoch = int(history_df.loc[history_df["val_macro_f1"].idxmax(), "epoch"])
    if argmax_epoch != best_epoch:
        raise ValueError(
            "Checkpoint-selection mismatch: stored best epoch "
            f"{best_epoch} != history argmax epoch {argmax_epoch}"
        )

    history_best = float(history_df["val_macro_f1"].max())
    if abs(history_best - best_val_macro_f1) > RELOAD_TOLERANCE:
        raise ValueError(
            "Checkpoint-selection mismatch: stored best macro-F1 "
            f"{best_val_macro_f1} != history maximum {history_best}"
        )

    reloaded = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    state = torch.load(BEST_MODEL_PATH, map_location=device)
    assert_state_finite(state, "reloaded best checkpoint")
    reloaded.load_state_dict(state)

    reloaded_metrics = evaluate(reloaded, val_loader, criterion, device, class_names)
    delta = abs(reloaded_metrics["macro_f1"] - best_val_macro_f1)
    if delta > RELOAD_TOLERANCE:
        raise ValueError(
            "Reloaded best checkpoint does not reproduce the stored validation "
            f"macro-F1: {reloaded_metrics['macro_f1']} vs {best_val_macro_f1} "
            f"(delta={delta}, tolerance={RELOAD_TOLERANCE})"
        )

    return reloaded_metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    source = read_own_source()
    assert_no_holdout_references(source)
    assert_outputs_absent()

    set_reproducibility(RANDOM_STATE)

    device = get_device()
    provenance = environment_provenance(device, source)
    class_names = load_class_names()

    # Validate the arrays before any directory is created.
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")
    assert_label_array(y_train, "y_train")
    assert_label_array(y_val, "y_val")
    assert_feature_array(
        np.load(PROCESSED_DIR / "X_train.npy", mmap_mode="r"), "X_train"
    )
    assert_feature_array(
        np.load(PROCESSED_DIR / "X_val.npy", mmap_mode="r"), "X_val"
    )

    class_weights_cpu = compute_class_weights(y_train, NUM_CLASSES)
    class_weights = class_weights_cpu.to(device)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(CLASS_WEIGHTS_PATH, class_weights_cpu.numpy())

    train_loader, val_loader = build_loaders(RANDOM_STATE)

    model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config = {
        "task": "multiclass_intrusion_detection",
        "variant": "centralised_corrected",
        "dataset": "NF-UNSW-NB15-v2",
        "model": "MLPMultiClassClassifier",
        "architecture": [INPUT_DIM, 128, 64, NUM_CLASSES],
        "dropout": 0.2,
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "loss": "CrossEntropyLoss(weight=balanced)",
        "class_weight_formula": "N / (num_classes * count_c)",
        "class_weights": class_weights_cpu.numpy().tolist(),
        "class_names": class_names,
        "model_selection": "validation_macro_f1",
        "early_stopping": True,
        "early_stopping_metric": "validation_macro_f1",
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "loss_aggregation": "exact_weighted_mean_over_class_weight_mass",
        "holdout_evaluation_split_used": False,
        "random_state": RANDOM_STATE,
        "reload_tolerance": RELOAD_TOLERANCE,
        "train_rows": int(y_train.shape[0]),
        "val_rows": int(y_val.shape[0]),
        "train_class_counts": np.bincount(
            y_train, minlength=NUM_CLASSES
        ).tolist(),
        "val_class_counts": np.bincount(y_val, minlength=NUM_CLASSES).tolist(),
        "environment": provenance,
    }
    with open(CONFIG_PATH, "w") as file:
        json.dump(config, file, indent=2)

    history: list[dict] = []
    best_val_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_epoch = 0
    stop_reason = "epoch_cap_reached"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        if not np.isfinite(train_loss):
            raise ValueError(f"Non-finite weighted training loss at epoch {epoch}")

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
        )

        history.append(history_row(epoch, train_loss, val_metrics))
        stopped_epoch = epoch

        improvement = val_metrics["macro_f1"] - best_val_macro_f1

        # Selection is by validation macro-F1 only, and is deliberately kept
        # independent of the patience counter below: any strictly-better epoch
        # becomes the reported checkpoint even if it does not clear MIN_DELTA.
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            state = model.state_dict()
            assert_state_finite(state, f"epoch {epoch} checkpoint")
            torch.save(state, BEST_MODEL_PATH)

        if improvement > MIN_DELTA:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:02d} "
            f"weighted_train_loss={train_loss:.6f} "
            f"weighted_val_loss={val_metrics['weighted_loss']:.6f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_macro_recall={val_metrics['macro_recall']:.4f} "
            f"val_balanced_acc={val_metrics['balanced_accuracy']:.4f} "
            f"worst_f1={val_metrics['worst_class_f1']:.4f} "
            f"({val_metrics['worst_class_f1_name']}) "
            f"worst_recall={val_metrics['worst_class_recall']:.4f} "
            f"({val_metrics['worst_class_recall_name']})"
        )

        if epochs_without_improvement >= PATIENCE:
            stop_reason = "patience_exhausted"
            print(
                f"Early stop at epoch {epoch}: no improvement above "
                f"MIN_DELTA={MIN_DELTA} for {PATIENCE} consecutive epochs "
                f"(best epoch {best_epoch}, macro-F1 {best_val_macro_f1:.6f})"
            )
            break

    if best_epoch < 1:
        raise ValueError("No best checkpoint was recorded during training")

    final_state = model.state_dict()
    assert_state_finite(final_state, "final model")
    torch.save(final_state, FINAL_MODEL_PATH)

    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False)

    # Every epoch that ran is recorded, including those after the best epoch.
    if len(history_df) != stopped_epoch:
        raise ValueError(
            f"Expected {stopped_epoch} history rows, found {len(history_df)}"
        )

    best_metrics = verify_best_checkpoint(
        history_df=history_df,
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_macro_f1,
        device=device,
        val_loader=val_loader,
        criterion=criterion,
        class_names=class_names,
    )

    report_text = write_validation_reports(
        best_metrics["y_true"], best_metrics["y_pred"], class_names
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "reloaded_val_macro_f1": best_metrics["macro_f1"],
        "reload_macro_f1_delta": abs(best_metrics["macro_f1"] - best_val_macro_f1),
        "weighted_val_loss": best_metrics["weighted_loss"],
        **{
            key: value
            for key, value in best_metrics.items()
            if key not in {"weighted_loss", "y_true", "y_pred"}
        },
        "epochs_run": stopped_epoch,
        "stopped_epoch": stopped_epoch,
        "stop_reason": stop_reason,
        "max_epochs": EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "holdout_evaluation_split_used": False,
        **provenance,
    }
    pd.DataFrame([summary]).to_csv(VAL_SUMMARY_PATH, index=False)

    print("\n=== Validation Classification Report (best checkpoint) ===")
    print(report_text)
    print(f"Stop reason:                {stop_reason}")
    print(f"Stopped at epoch:           {stopped_epoch} (cap {EPOCHS})")
    print(f"Best epoch:                 {best_epoch}")
    print(f"Best validation macro-F1:   {best_val_macro_f1:.6f}")
    print(f"Reloaded validation macroF1:{best_metrics['macro_f1']:.6f}")
    print(f"Validation balanced acc:    {best_metrics['balanced_accuracy']:.6f}")
    print(f"Weighted validation loss:   {best_metrics['weighted_loss']:.6f}")
    print(
        f"Worst-class F1:             {best_metrics['worst_class_f1']:.6f} "
        f"({best_metrics['worst_class_f1_name']})"
    )
    print(
        f"Worst-class recall:         {best_metrics['worst_class_recall']:.6f} "
        f"({best_metrics['worst_class_recall_name']})"
    )
    print("\nPer-class validation metrics (best checkpoint):")
    for class_id, class_name in enumerate(class_names):
        prefix = f"class{class_id}_{class_name}"
        print(
            f"  {class_id} {class_name:15} "
            f"P={best_metrics[f'precision_{prefix}']:.4f} "
            f"R={best_metrics[f'recall_{prefix}']:.4f} "
            f"F1={best_metrics[f'f1_{prefix}']:.4f} "
            f"support={best_metrics[f'support_{prefix}']} "
            f"predicted={best_metrics[f'predicted_count_{prefix}']}"
        )


if __name__ == "__main__":
    main()
