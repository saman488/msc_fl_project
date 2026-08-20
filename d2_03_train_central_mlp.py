"""
Centralised MLP baseline for MULTI-CLASS intrusion detection on
NF-CSE-CIC-IDS2018-v2 (Dataset 2).

Dataset-2 counterpart of d1_30_train_central_mlp_corrected.py. The audited
centralised logic is preserved: the model contract (36 -> 128 -> 64 -> 7), the
training contract (AdamW, lr=1e-3, weight_decay=1e-5, batch_size=4096, 20 epochs,
seed 42), the balanced class weighting (weight[c] = N / (C * n_c)), the exact
weighted-loss accounting, validation Macro-F1 checkpoint selection with no early
stopping, the best-checkpoint reload verification, the finite-state checks, the
full per-class reporting, the no-overwrite protection and the provenance record.

Differences from the Dataset-1 script are confined to the dataset contract, the
isolated Dataset-2 output roots, the device selection, a set of hard pre-training
assertions on the Dataset-2 training state, and the CUDA data path described
below. No hyperparameter, optimiser, schedule, precision mode or weighting change
is introduced.

CUDA-resident data path
-----------------------
On CUDA the train and validation arrays are placed on the device once at startup
and batches are gathered from those resident tensors, removing the row-by-row
mmap fetch that otherwise dominates the epoch. This is the engineering pattern
already proven in d2_04_train_fedavg.py; the training method is untouched.

The DataLoader keeps ownership of shuffling. The training loader is still built
with the same single generator seeded from RANDOM_STATE and the same
``batch_size``/``shuffle=True``/``num_workers=0`` arguments over a dataset of the
same length, so the permutation stream across all 20 epochs is unchanged; it
simply yields row positions instead of feature rows. The validation loader is
constructed without a generator, so each of its iterators draws one int64 base
seed from the global RNG - the resident validation loader makes the same draw, so
the global stream, and therefore every later epoch's dropout mask, is unchanged.

On MPS and CPU the original NumpyDataset/DataLoader path is used unchanged.

Properties carried over unchanged:

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

2.  Generic class handling.  No class ids are hard-coded in the metric or
    reporting code; the 7 classes are handled through
    ``configs/nf_cse_cic_ids2018_v2/label_mapping.json``.  The expected class
    order is asserted against that mapping before training starts.

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

6.  Isolated outputs.  Nothing produced by any Dataset-1 run or by any other
    Dataset-2 stage is read or overwritten; the script refuses to start if any
    of its own intended outputs already exist.

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
# Paths (isolated Dataset-2 centralised namespace)
# --------------------------------------------------------------------------- #

PROCESSED_DIR = Path("data/nf_cse_cic_ids2018_v2/processed")
LABEL_MAPPING_PATH = Path("configs/nf_cse_cic_ids2018_v2/label_mapping.json")

MODELS_DIR = Path("models/nf_cse_cic_ids2018_v2/centralised")
RESULTS_DIR = Path("results/nf_cse_cic_ids2018_v2/centralised")
CONFIGS_DIR = Path("configs/nf_cse_cic_ids2018_v2/centralised")

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
# Contracts
# --------------------------------------------------------------------------- #

RANDOM_STATE = 42
INPUT_DIM = 36
NUM_CLASSES = 7
BATCH_SIZE = 4096
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Tolerance for the best-checkpoint reload / re-evaluation consistency check.
RELOAD_TOLERANCE = 1e-6

# Expected Dataset-2 training state, asserted before any training begins.
EXPECTED_TRAIN_ROWS = 13_255_011
EXPECTED_VAL_ROWS = 2_821_063
EXPECTED_CLASS_ORDER = [
    "Benign",        # 0
    "Bot",           # 1
    "BruteForce",    # 2
    "DDoS",          # 3
    "DoS",           # 4
    "Infiltration",  # 5
    "Web Attacks",   # 6
]


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
            f"forbidden arrays {offending}. The Dataset-2 centralised training "
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
            "Refusing to run: the following Dataset-2 centralised outputs already "
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


def assert_row_count(array: np.ndarray, name: str, expected_rows: int) -> None:
    """Hard row-count assertion against the expected Dataset-2 training state."""
    found = int(array.shape[0])
    if found != expected_rows:
        raise ValueError(
            f"{name} has {found} rows, expected exactly {expected_rows} "
            "for the Dataset-2 contract"
        )


def assert_exact_class_ids(y: np.ndarray, name: str) -> None:
    """The observed label ids must be exactly {0, .., NUM_CLASSES-1}."""
    observed = sorted(int(value) for value in np.unique(y).tolist())
    expected = list(range(NUM_CLASSES))
    if observed != expected:
        raise ValueError(
            f"{name} label ids {observed} do not equal the expected {expected}"
        )


def assert_class_order(class_names: list[str]) -> None:
    """The label mapping must define the exact Dataset-2 class order."""
    if class_names != EXPECTED_CLASS_ORDER:
        raise ValueError(
            f"Label mapping class order {class_names} does not match the expected "
            f"Dataset-2 order {EXPECTED_CLASS_ORDER}"
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
# CUDA-resident data path (engineering only; the training method is unchanged)
# --------------------------------------------------------------------------- #

class PositionDataset(Dataset):
    """Row positions 0..n-1 only; carries no features.

    Lets the DataLoader keep doing the shuffling exactly as with NumpyDataset -
    same RandomSampler, same generator, same length, same batch boundaries -
    while the feature and label rows are gathered from device-resident tensors.
    """

    def __init__(self, n: int) -> None:
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        return index


class ResidentTrainBatches:
    """Turns batches of row positions into on-device (features, labels) batches.

    The wrapped DataLoader is untouched, so the shuffled order it produces is the
    order consumed here.
    """

    def __init__(self, position_loader, x_device, y_device, device) -> None:
        self.position_loader = position_loader
        self.x_device = x_device
        self.y_device = y_device
        self.device = device

    def __len__(self) -> int:
        return len(self.position_loader)

    def __iter__(self):
        for positions in self.position_loader:
            rows = positions.to(self.device, non_blocking=True)
            yield self.x_device[rows], self.y_device[rows]


class ResidentValLoader:
    """Contiguous validation batches over the device-resident validation tensors.

    Same sequential order and batch size as the shuffle=False validation
    DataLoader. That DataLoader is built without a generator, so each of its
    iterators draws one int64 base seed from the global RNG; the same draw is made
    here, so the global stream - and therefore every later epoch's dropout mask -
    is identical to the Dataset path.
    """

    def __init__(self, x_device, y_device, batch_size: int) -> None:
        self.x_device = x_device
        self.y_device = y_device
        self.batch_size = int(batch_size)
        self.total = int(y_device.shape[0])

    def __len__(self) -> int:
        return (self.total + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Reproduces the base-seed draw a generator-less DataLoader iterator makes.
        torch.empty((), dtype=torch.int64).random_()
        for start in range(0, self.total, self.batch_size):
            stop = min(start + self.batch_size, self.total)
            yield self.x_device[start:stop], self.y_device[start:stop]


def build_resident_loaders(seed: int, device: torch.device):
    """CUDA path: place the train/validation arrays on the device once.

    Returns the same (train_loader, val_loader) pair as build_loaders(); only the
    batch delivery differs. Consumes no global RNG, so the model initialisation
    that follows draws from exactly the same position in the stream.
    """
    x_train = torch.from_numpy(np.load(PROCESSED_DIR / "X_train.npy")).to(torch.float32).to(device)
    y_train = torch.from_numpy(np.load(PROCESSED_DIR / "y_train.npy")).to(torch.long).to(device)
    x_val = torch.from_numpy(np.load(PROCESSED_DIR / "X_val.npy")).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.load(PROCESSED_DIR / "y_val.npy")).to(torch.long).to(device)

    for name, x, y, expected_rows in (
        ("train", x_train, y_train, EXPECTED_TRAIN_ROWS),
        ("validation", x_val, y_val, EXPECTED_VAL_ROWS),
    ):
        if x.ndim != 2 or int(x.shape[1]) != INPUT_DIM:
            raise ValueError(
                f"resident {name} features must have width {INPUT_DIM}, "
                f"found shape {tuple(x.shape)}"
            )
        if int(x.shape[0]) != expected_rows or int(y.numel()) != expected_rows:
            raise ValueError(
                f"resident {name} arrays have {int(x.shape[0])}/{int(y.numel())} rows, "
                f"expected {expected_rows}"
            )
        if x.dtype != torch.float32 or y.dtype != torch.long:
            raise ValueError(
                f"resident {name} dtypes are {x.dtype}/{y.dtype}, expected float32/int64"
            )

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)

    position_loader = DataLoader(
        PositionDataset(int(x_train.shape[0])),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=shuffle_generator,
    )
    train_loader = ResidentTrainBatches(position_loader, x_train, y_train, device)
    val_loader = ResidentValLoader(x_val, y_val, BATCH_SIZE)

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

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """CUDA if available, else MPS if available, else CPU.

    The training contract is unchanged; only the accelerator selection differs,
    so this script runs on a CUDA host as well as on Apple silicon.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
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
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
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

    # Hard Dataset-2 contract assertions, all before any directory is created
    # and before any file is written.
    assert_class_order(class_names)

    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")
    x_train = np.load(PROCESSED_DIR / "X_train.npy", mmap_mode="r")
    x_val = np.load(PROCESSED_DIR / "X_val.npy", mmap_mode="r")

    # Exact row counts.
    assert_row_count(y_train, "y_train", EXPECTED_TRAIN_ROWS)
    assert_row_count(y_val, "y_val", EXPECTED_VAL_ROWS)
    assert_row_count(x_train, "X_train", EXPECTED_TRAIN_ROWS)
    assert_row_count(x_val, "X_val", EXPECTED_VAL_ROWS)

    # Width 36.
    assert_feature_array(x_train, "X_train")
    assert_feature_array(x_val, "X_val")

    # Integer labels in range, and every one of the 7 classes present in both
    # the training and the validation split.
    assert_label_array(y_train, "y_train")
    assert_label_array(y_val, "y_val")

    # Exact class ids, in both splits.
    assert_exact_class_ids(y_train, "y_train")
    assert_exact_class_ids(y_val, "y_val")

    train_class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    val_class_counts = np.bincount(y_val, minlength=NUM_CLASSES)
    if int(train_class_counts.sum()) != EXPECTED_TRAIN_ROWS:
        raise ValueError("Training class counts do not sum to the expected row count")
    if int(val_class_counts.sum()) != EXPECTED_VAL_ROWS:
        raise ValueError("Validation class counts do not sum to the expected row count")

    print(
        f"Dataset-2 contract OK: train={EXPECTED_TRAIN_ROWS:,} "
        f"val={EXPECTED_VAL_ROWS:,} input_dim={INPUT_DIM} classes={NUM_CLASSES} "
        f"device={device}"
    )
    for class_id, class_name in enumerate(class_names):
        print(
            f"  {class_id} {class_name:<14} "
            f"train={int(train_class_counts[class_id]):>12,} "
            f"val={int(val_class_counts[class_id]):>10,}"
        )

    class_weights_cpu = compute_class_weights(y_train, NUM_CLASSES)
    class_weights = class_weights_cpu.to(device)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(CLASS_WEIGHTS_PATH, class_weights_cpu.numpy())

    # CUDA: place the arrays on the device once and gather batches from them.
    # Anything else keeps the original mmap Dataset/DataLoader path.
    resident_path_active = device.type == "cuda"
    if resident_path_active:
        train_loader, val_loader = build_resident_loaders(RANDOM_STATE, device)
        data_path = "cuda_resident"
        print(
            f"Data path: CUDA-RESIDENT ACTIVE on {device} - train "
            f"({EXPECTED_TRAIN_ROWS:,} x {INPUT_DIM}) and validation "
            f"({EXPECTED_VAL_ROWS:,} x {INPUT_DIM}) tensors held on the device; "
            "batches gathered on device, DataLoader still owns shuffling",
            flush=True,
        )
    else:
        train_loader, val_loader = build_loaders(RANDOM_STATE)
        data_path = "numpy_mmap_dataloader"
        print(
            f"Data path: CUDA-resident NOT active (device={device}) - using the "
            "NumpyDataset/DataLoader mmap path",
            flush=True,
        )

    model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config = {
        "task": "multiclass_intrusion_detection",
        "variant": "centralised",
        "dataset": "NF-CSE-CIC-IDS2018-v2",
        "dataset_key": "nf_cse_cic_ids2018_v2",
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
        "expected_class_order": EXPECTED_CLASS_ORDER,
        "model_selection": "validation_macro_f1",
        "early_stopping": False,
        "loss_aggregation": "exact_weighted_mean_over_class_weight_mass",
        "holdout_evaluation_split_used": False,
        "random_state": RANDOM_STATE,
        "reload_tolerance": RELOAD_TOLERANCE,
        "device_selection": "cuda_if_available_else_mps_if_available_else_cpu",
        "data_path": data_path,
        "cuda_resident_data_path": resident_path_active,
        "data_path_note": (
            "engineering only: on CUDA the train/validation arrays are device-resident "
            "and batches are gathered on device; the DataLoader still owns shuffling "
            "with the same generator, batch size and dataset length, and the "
            "generator-less validation loader's global-RNG base-seed draw is "
            "reproduced, so the training method and RNG streams are unchanged"
        ),
        "expected_train_rows": EXPECTED_TRAIN_ROWS,
        "expected_val_rows": EXPECTED_VAL_ROWS,
        "train_rows": int(y_train.shape[0]),
        "val_rows": int(y_val.shape[0]),
        "train_class_counts": train_class_counts.tolist(),
        "val_class_counts": val_class_counts.tolist(),
        "processed_dir": str(PROCESSED_DIR),
        "label_mapping_path": str(LABEL_MAPPING_PATH),
        "environment": provenance,
    }
    with open(CONFIG_PATH, "w") as file:
        json.dump(config, file, indent=2)

    history: list[dict] = []
    best_val_macro_f1 = -1.0
    best_epoch = -1

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

        # Selection is by validation macro-F1 only; all 20 epochs always run.
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            state = model.state_dict()
            assert_state_finite(state, f"epoch {epoch} checkpoint")
            torch.save(state, BEST_MODEL_PATH)

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
            f"({val_metrics['worst_class_recall_name']})",
            flush=True,
        )

    if best_epoch < 1:
        raise ValueError("No best checkpoint was recorded during training")

    final_state = model.state_dict()
    assert_state_finite(final_state, "final model")
    torch.save(final_state, FINAL_MODEL_PATH)

    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False)

    if len(history_df) != EPOCHS:
        raise ValueError(
            f"Expected {EPOCHS} history rows, found {len(history_df)}"
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
        "epochs_run": EPOCHS,
        "holdout_evaluation_split_used": False,
        **provenance,
    }
    pd.DataFrame([summary]).to_csv(VAL_SUMMARY_PATH, index=False)

    print("\n=== Validation Classification Report (best checkpoint) ===")
    print(report_text)
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
