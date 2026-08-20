"""
Dataset-2 (NF-CSE-CIC-IDS2018-v2) K=5 SCAFFOLD final training matrix.

This is d2_04_train_fedavg.py with the audited SCAFFOLD control-variate logic from
35_train_final_scaffold_37f.py added, and nothing else changed. Everything the two
methods share comes from d2_04: the MLP (36 -> 128 -> 64 -> 7, dropout 0.2), the
Dataset-2 arrays and K=5 partitions, seeds {42, 43, 44} x conditions
{iid, alpha_0p1, alpha_0p5, alpha_1p0} = 12 runs, SGD at lr 0.1 (momentum 0, weight
decay 0), batch 4096, one local epoch, 40 rounds, full participation, the
per-round-per-client seed scheme, sample-weighted model aggregation, the
CUDA-resident data path, the validation metrics, Macro-F1 checkpoint selection and
the best-checkpoint reload check.

ALGORITHM (SCAFFOLD, Option II)
-------------------------------
Each independent (seed, condition) run owns one server control c and five client
controls c_i. They are keyed by the trainable named parameters, start at zero, live
on the training device, persist across the 40 rounds, and are rebuilt at zero for
every new run.

Round r, with x the global trainable parameters at the start of the round and c_old
one server-control snapshot shared by all five clients:

    per minibatch:  loss.backward()
                    grad <- grad - c_i_old + c_old
                    optimizer.step()
                    local_steps += 1

    after training: c_i_new   = c_i_old - c_old + (x - y_i) / (local_steps * LR)
                    delta_c_i = c_i_new - c_i_old

    model:   theta <- sum_i p_i * theta_i      with p_i = n_i / sum_j n_j
    control: c     <- c_old + sum_i p_i * delta_c_i

local_steps is the counted number of optimizer.step() calls, never inferred from
client size or batch size. Under full participation the update preserves
c == sum_i p_i * c_i; that identity is checked every round and never repaired.

WEIGHTED EXTENSION - SCIENTIFIC PROVENANCE
------------------------------------------
The original SCAFFOLD paper presents an equal-client objective and a UNIFORM
server-control aggregation in its displayed base equations. This project uses a
SAMPLE/EXAMPLE-WEIGHTED extension, because the global empirical-risk objective and
the model aggregation weight clients by n_i / sum_j n_j; uniform control
aggregation alongside a sample-weighted model would mean the server control no
longer represents the same weighted objective as the global model.

So: the server-control rule here is a project weighted extension aligned with the
sample-weighted empirical-risk objective. It is NOT the verbatim original displayed
equation, and the paper's convergence guarantees are NOT claimed to carry over
unchanged to it.

MANDATORY READ-ONLY INPUTS
--------------------------
    models/nf_cse_cic_ids2018_v2/final_fedavg_k5/initial_global_model.pt
    results/nf_cse_cic_ids2018_v2/final_fedavg_k5/class_weights.npy

SCAFFOLD must start from the same initialisation and class weights as FedAvg, so
both files are required, hashed, loaded read-only and checked before any output is
written: the initial state must be finite, have the exact 36 -> 128 -> 64 -> 7
shapes, and be bitwise equal to an independent reproduction of d2_04's
initialisation; the class weights must equal a fresh computation from y_train
exactly. Any mismatch aborts before training. This runner never writes its own
initial-model checkpoint, and both inputs are re-hashed afterwards.

Outputs go only under results/nf_cse_cic_ids2018_v2/final_scaffold_k5/ and
models/nf_cse_cic_ids2018_v2/final_scaffold_k5/. Reads the train and validation
arrays only; never the held-out arrays.

SCAFFOLD_EXECUTION_ENABLED is False: main() aborts before creating any directory or
writing any file. It is opened only after an independent source audit and a passing
RunPod gate.
"""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import platform
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROCESSED_DIR = Path("data/nf_cse_cic_ids2018_v2/processed")
LABEL_MAP = Path("configs/nf_cse_cic_ids2018_v2/label_mapping.json")
PART_ROOT = Path("data/nf_cse_cic_ids2018_v2/fl_clients/final_partitions/k_5")
RESULTS_DIR = Path("results/nf_cse_cic_ids2018_v2/final_scaffold_k5")
MODELS_DIR = Path("models/nf_cse_cic_ids2018_v2/final_scaffold_k5")
CLASS_WEIGHTS_PATH = RESULTS_DIR / "class_weights.npy"

# Mandatory read-only inputs from the completed Dataset-2 FedAvg K=5 run.
INIT_PATH = Path("models/nf_cse_cic_ids2018_v2/final_fedavg_k5/initial_global_model.pt")
FEDAVG_CLASS_WEIGHTS_PATH = Path("results/nf_cse_cic_ids2018_v2/final_fedavg_k5/class_weights.npy")

METHOD = "SCAFFOLD"
DATASET = "nf_cse_cic_ids2018_v2"
INPUT_DIM = 36
NUM_CLASSES = 7
NUM_CLIENTS = 5
LOCAL_EPOCHS = 1
MAX_ROUNDS = 40
TRAIN_SEED = 42
LR = 0.1
MOMENTUM = 0.0
WEIGHT_DECAY = 0.0
BATCH_SIZE = 4096
RELOAD_F1_TOL = 1e-4

# While False, main() aborts before creating any directory or writing any file.
SCAFFOLD_EXECUTION_ENABLED = True

# Tolerance for the identity c == sum_i p_i c_i, scaled by the control magnitude.
# Diagnostic only: the invariant is reported and asserted, never repaired.
CONTROL_INVARIANT_TOL = 1e-4

PARTITION_SEEDS = [42, 43, 44]
CONDITIONS = ["iid", "alpha_0p1", "alpha_0p5", "alpha_1p0"]

EXPECTED_TRAIN_ROWS = 13_255_011
EXPECTED_VAL_ROWS = 2_821_063
EXPECTED_CLASS_ORDER = [
    "Benign", "Bot", "BruteForce", "DDoS", "DoS", "Infiltration", "Web Attacks",
]

SERVER_CONTROL_AGGREGATION = "sample weighted using the same n_k / sum_j n_j weights as the model"
CONTROL_WEIGHTING_PROVENANCE = (
    "The original SCAFFOLD paper presents an equal-client objective and uniform "
    "server-control aggregation in its displayed base equations. This project uses an "
    "explicit sample/example-weighted extension, because the global empirical-risk "
    "objective and the model aggregation weight clients by n_i / sum_j n_j. It is a "
    "project weighted extension aligned with the sample-weighted empirical-risk "
    "objective, not the verbatim original displayed equation."
)
CONTROL_THEORY_CLAIM = (
    "none: the original convergence guarantees are stated for the paper's own "
    "weighting and are not claimed to carry over unchanged to this weighted extension"
)


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


class IndexedDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path, indices: np.ndarray) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int):
        row = int(self.indices[i])
        features = torch.from_numpy(np.asarray(self.x[row], dtype=np.float32).copy())
        label = torch.tensor(int(self.y[row]), dtype=torch.long)
        return features, label


class FullDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        features = torch.from_numpy(np.asarray(self.x[i], dtype=np.float32).copy())
        label = torch.tensor(int(self.y[i]), dtype=torch.long)
        return features, label


class LocalPositionDataset(Dataset):
    """Local positions 0..n-1 only; carries no features.

    Used on CUDA so the DataLoader keeps doing the shuffling exactly as before -
    same RandomSampler, same generator, same length n_k, same batch boundaries -
    while the feature and label rows are gathered from device-resident tensors.
    """

    def __init__(self, n: int) -> None:
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


class ResidentClientBatches:
    """Turns batches of local positions into on-device (features, labels) batches.

    positions -> client_indices_cuda[positions] -> global training rows ->
    x_train_cuda[rows], y_train_cuda[rows]. The wrapped DataLoader is untouched,
    so the shuffled order it produces is the order consumed here.
    """

    def __init__(self, position_loader, client_indices_cuda, x_train_cuda,
                 y_train_cuda, device) -> None:
        self.position_loader = position_loader
        self.client_indices_cuda = client_indices_cuda
        self.x_train_cuda = x_train_cuda
        self.y_train_cuda = y_train_cuda
        self.device = device

    def __iter__(self):
        for positions in self.position_loader:
            rows = self.client_indices_cuda[positions.to(self.device, non_blocking=True)]
            yield self.x_train_cuda[rows], self.y_train_cuda[rows]


class ResidentValLoader:
    """Contiguous validation batches over the device-resident validation tensors.

    Same sequential order and batch size as the shuffle=False validation
    DataLoader. Labels are handed back on the host, as the Dataset path does, so
    the metric inputs are unchanged.
    """

    def __init__(self, x_val_cuda, y_val_cuda, batch_size: int) -> None:
        self.x_val_cuda = x_val_cuda
        self.y_val_cuda = y_val_cuda
        self.batch_size = int(batch_size)
        self.total = int(y_val_cuda.shape[0])

    def __iter__(self):
        for start in range(0, self.total, self.batch_size):
            stop = min(start + self.batch_size, self.total)
            yield self.x_val_cuda[start:stop], self.y_val_cuda[start:stop].cpu()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def seed_accelerator(seed: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    # Make CUDA/MPS timings wall-accurate; no-op on CPU.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode()
        return bool(out.strip())
    except Exception:
        return None


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_class_names() -> list[str]:
    with open(LABEL_MAP) as f:
        name_to_id = json.load(f)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    assert sorted(id_to_name) == list(range(NUM_CLASSES)), (
        f"label mapping must define class ids 0..{NUM_CLASSES - 1}, found {sorted(id_to_name)}"
    )
    return [id_to_name[c] for c in range(len(id_to_name))]


def class_weights_full(y_train: np.ndarray) -> np.ndarray:
    # N / (num_classes * class_count), computed once from the full training labels.
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    return counts.sum() / (NUM_CLASSES * counts)


def assert_state_finite(state: dict, name: str) -> None:
    for k, v in state.items():
        if v.is_floating_point():
            assert torch.isfinite(v).all(), f"{name}: non-finite values in {k}"


def state_l2_distance(a: dict, b: dict) -> float:
    total = 0.0
    for k in a:
        if a[k].is_floating_point():
            total += torch.sum((a[k].to(torch.float32) - b[k].to(torch.float32)) ** 2).item()
    return float(total ** 0.5)


def aggregate_sample_weighted(states: list[dict], sizes: list[int]) -> dict:
    # global parameter = sum_k (n_k / sum_j n_j) * client_parameter_k
    assert len(states) == len(sizes) == NUM_CLIENTS, "aggregation: states/sizes count mismatch"
    assert all(n > 0 for n in sizes), "aggregation: a client size is not positive"
    total = float(sum(sizes))
    weights = [n / total for n in sizes]
    assert abs(sum(weights) - 1.0) < 1e-9, "aggregation weights do not sum to 1"
    agg = {}
    for key in states[0]:
        stacked = torch.stack([s[key].to(torch.float32) for s in states], dim=0)
        w = torch.tensor(weights, dtype=torch.float32).view(-1, *([1] * (stacked.dim() - 1)))
        agg[key] = (stacked * w).sum(dim=0)
    return agg


def states_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def expected_state_shapes() -> dict:
    probe = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES)
    return {k: tuple(v.shape) for k, v in probe.state_dict().items()}


# --------------------------- SCAFFOLD control variates ---------------------- #
def init_controls(model: nn.Module, device: torch.device) -> dict:
    """Zero control keyed by the trainable named parameters, on the training device.

    Detached, and allocated on `device` so no control tensor is moved host<->device
    during training. Called once per client and once for the server at the start of
    each independent run, so control state never carries across runs.
    """
    return {name: torch.zeros_like(p, device=device).detach()
            for name, p in model.named_parameters() if p.requires_grad}


def clone_controls(controls: dict) -> dict:
    """Detached copy, used to freeze c_old for the duration of a round."""
    return {name: t.detach().clone() for name, t in controls.items()}


def controls_l2_norm(controls: dict) -> float:
    total = sum(float(torch.sum(t.to(torch.float32) ** 2).item()) for t in controls.values())
    return float(total ** 0.5)


def controls_diff_l2_norm(a: dict, b: dict) -> float:
    """L2 norm of (a - b); used for the constant per-round gradient shift."""
    assert a.keys() == b.keys(), "control difference: key mismatch"
    total = sum(float(torch.sum((a[n].to(torch.float32) - b[n].to(torch.float32)) ** 2).item())
                for n in a)
    return float(total ** 0.5)


def option_ii_client_control(c_i_old: dict, c_old: dict, x_params: dict,
                             y_params: dict, local_steps: int, lr: float) -> tuple[dict, dict]:
    """SCAFFOLD Option II: c_i_new = c_i_old - c_old + (x - y_i) / (local_steps * lr).

    Returns (c_i_new, delta_c_i). delta_c_i is formed from the un-replaced c_i_old,
    so the caller must not overwrite the client's stored control before calling.
    """
    assert local_steps > 0, "Option II needs a positive optimizer-step count"
    scale = float(local_steps) * lr
    c_i_new = {name: c_i_old[name] - c_old[name] + (x_params[name] - y_params[name]) / scale
               for name in c_i_old}
    delta_c_i = {name: c_i_new[name] - c_i_old[name] for name in c_i_old}
    return c_i_new, delta_c_i


def weighted_controls(control_list: list[dict], weights: list[float]) -> dict:
    """sum_k weights[k] * control_k, using the same p_k as the model aggregation."""
    assert len(control_list) == len(weights) == NUM_CLIENTS, \
        "weighted_controls: controls/weights count mismatch"
    assert all(w > 0.0 for w in weights), "weighted_controls: a weight is not positive"
    assert abs(sum(weights) - 1.0) < 1e-9, "weighted_controls: weights do not sum to 1"
    keys = control_list[0].keys()
    assert all(c.keys() == keys for c in control_list), "weighted_controls: key mismatch"
    return {name: sum(control_list[k][name] * weights[k] for k in range(len(control_list)))
            for name in keys}


def max_controls_abs_diff(a: dict, b: dict) -> float:
    assert a.keys() == b.keys(), "control comparison: key mismatch"
    return max(float((a[name] - b[name]).abs().max().item()) for name in a)


def max_controls_abs(controls: dict) -> float:
    return max(float(t.abs().max().item()) for t in controls.values())


def assert_controls_finite(controls: dict, name: str) -> None:
    for key, t in controls.items():
        assert torch.isfinite(t).all(), f"{name}: non-finite control values in {key}"


def train_one_epoch(model, loader, criterion, optimizer, device, c_old, c_i_old) -> dict:
    # Identical to d2_04's local epoch apart from the two SCAFFOLD lines: the
    # gradient correction between backward() and step(), and the step counter.
    # The loss is unchanged - SCAFFOLD corrects the gradient, not the objective.
    model.train()
    loss_num = torch.zeros((), device=device)
    loss_den = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    finite = torch.ones((), dtype=torch.bool, device=device)
    n_samples, n_batches, local_steps = 0, 0, 0

    sync_device(device)
    train_start = time.perf_counter()
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"no gradient for trainable parameter {name}"
                    param.grad = param.grad - c_i_old[name] + c_old[name]
        optimizer.step()
        # Counted, never inferred from client size, batch size or local epochs.
        local_steps += 1
        # On-device accumulation only; no host sync inside the batch loop.
        batch_weight = criterion.weight[labels].sum()
        loss_num = loss_num + loss.detach() * batch_weight
        loss_den = loss_den + batch_weight
        correct = correct + (logits.detach().argmax(dim=1) == labels).sum()
        finite = finite & torch.isfinite(loss.detach())
        n_samples += labels.size(0)
        n_batches += 1
    sync_device(device)
    train_seconds = time.perf_counter() - train_start

    assert n_batches > 0, "no batches processed in local training"
    assert local_steps == n_batches, "optimizer-step count does not match the batch count"
    assert bool(finite.item()), "non-finite training loss"
    return {
        "weighted_train_loss": float((loss_num / loss_den).item()),
        "online_train_accuracy": float((correct / n_samples).item()),
        "n_samples": n_samples,
        "n_batches": n_batches,
        "local_steps": local_steps,
        "train_seconds": train_seconds,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    loss_num, loss_den = 0.0, 0.0
    preds, targets, probs = [], [], []
    for features, labels in loader:
        labels_dev = labels.to(device)
        logits = model(features.to(device))
        loss = criterion(logits, labels_dev)
        assert torch.isfinite(loss).item(), "non-finite validation batch loss"
        batch_weight = criterion.weight[labels_dev].sum()
        loss_num += loss.item() * batch_weight.item()
        loss_den += batch_weight.item()
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        targets.append(labels.numpy())
    y_prob = np.concatenate(probs)
    y_pred = np.concatenate(preds).astype(int)
    y_true = np.concatenate(targets).astype(int)
    labels_range = list(range(NUM_CLASSES))

    support = np.bincount(y_true, minlength=NUM_CLASSES)
    assert (support > 0).all(), f"validation set is missing one or more of the {NUM_CLASSES} classes"
    val_loss = loss_num / loss_den
    assert np.isfinite(val_loss), "non-finite validation loss"

    per_p = precision_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    per_r = recall_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    per_f = f1_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    predicted_count = np.bincount(y_pred, minlength=NUM_CLASSES)
    pr_auc = np.full(NUM_CLASSES, np.nan)
    for c in labels_range:
        if support[c] > 0:
            pr_auc[c] = average_precision_score((y_true == c).astype(int), y_prob[:, c])
    supported = [c for c in labels_range if support[c] > 0]
    return {
        "val_loss": val_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_pr_auc": float(np.nanmean(pr_auc)),
        "worst_class_f1": float(min(per_f[c] for c in supported)),
        "worst_class_recall": float(min(per_r[c] for c in supported)),
        "per_precision": per_p, "per_recall": per_r, "per_f1": per_f,
        "support": support, "predicted_count": predicted_count,
    }


def assert_no_test_reference() -> None:
    """Confirm the script never references a held-out-array filename."""
    tok_x = "X_" + "test"
    tok_y = "y_" + "test"
    source = Path(__file__).read_text()
    assert tok_x not in source and tok_y not in source, "Held-out-array reference found in script."


def load_partition(partition_seed: int, condition: str, y_train: np.ndarray,
                   global_class_counts: np.ndarray) -> dict:
    """Load and verify one final partition; returns datasets, sizes, and class counts."""
    part_dir = PART_ROOT / f"seed_{partition_seed}" / condition
    files = sorted(part_dir.glob("client_*_indices.npy"))
    assert len(files) == NUM_CLIENTS, f"{part_dir}: expected {NUM_CLIENTS} client files, got {len(files)}"
    client_indices = [np.load(part_dir / f"client_{k:02d}_indices.npy") for k in range(NUM_CLIENTS)]

    assert all(len(ci) > 0 for ci in client_indices), f"{part_dir}: a client is empty"
    all_assigned = np.concatenate(client_indices)
    total_records = len(y_train)
    assert len(all_assigned) == total_records, f"{part_dir}: indices do not cover all training records"
    assert len(np.unique(all_assigned)) == total_records, f"{part_dir}: duplicate training indices"
    assert np.array_equal(np.sort(all_assigned), np.arange(total_records)), f"{part_dir}: coverage is not 0..N-1"
    counts = np.stack([np.bincount(y_train[ci], minlength=NUM_CLASSES) for ci in client_indices])
    assert np.array_equal(counts.sum(axis=0), global_class_counts), f"{part_dir}: class totals differ from global"

    datasets = [IndexedDataset(PROCESSED_DIR / "X_train.npy", PROCESSED_DIR / "y_train.npy", ci)
                for ci in client_indices]
    sizes = [int(len(ci)) for ci in client_indices]
    return {"datasets": datasets, "client_indices": client_indices, "sizes": sizes,
            "client_class_counts": counts, "part_dir": str(part_dir)}


def run(partition_seed, condition, part_data, initial_state,
        weight_f32, val_loader, device, resident=None) -> dict:
    tag = f"scaffold_k{NUM_CLIENTS}_seed{partition_seed}_{condition}"
    datasets = part_data["datasets"]
    sizes = part_data["sizes"]
    total_size = float(sum(sizes))
    agg_weights = [n / total_size for n in sizes]

    # CUDA: this partition's five saved client index arrays, moved to the device
    # once. Same membership, same order, same sizes.
    client_indices_cuda = None
    if resident is not None:
        client_indices_cuda = [torch.from_numpy(ci.astype(np.int64)).to(device)
                               for ci in part_data["client_indices"]]
        for k, ci in enumerate(client_indices_cuda):
            assert int(ci.numel()) == sizes[k], f"{tag}: client {k} index count changed on device"

    set_all_seeds(TRAIN_SEED)
    global_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    global_model.load_state_dict(copy.deepcopy(initial_state))
    assert states_equal(global_model.state_dict(), initial_state), f"{tag}: initial state not loaded identically"
    local_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight_f32, dtype=torch.float32, device=device))

    # Controls for this run only: zero at the start, persisting across the rounds
    # below, discarded when this run returns.
    server_control = init_controls(global_model, device)
    client_controls = [init_controls(global_model, device) for _ in range(NUM_CLIENTS)]
    assert all(max_controls_abs(c) == 0.0 for c in [server_control] + client_controls), \
        f"{tag}: controls did not start at zero"

    best = {"round": -1, "macro_f1": -1.0}
    best_path = MODELS_DIR / f"best_{tag}.pt"
    final_path = MODELS_DIR / f"final_{tag}.pt"
    hist_path = RESULTS_DIR / f"history_{tag}.csv"
    history = []
    total_fl_seconds, total_validation_seconds, total_run_seconds = 0.0, 0.0, 0.0

    for rnd in range(1, MAX_ROUNDS + 1):
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        # x and c_old are fixed for the whole round: every client below starts from
        # these same two snapshots.
        x_params = {name: p.detach().clone()
                    for name, p in global_model.named_parameters() if p.requires_grad}
        c_old = clone_controls(server_control)

        client_states, client_stats, client_seconds, client_update_l2 = [], [], [], []
        client_new_controls, client_delta_controls = [], []
        client_local_steps, client_correction_norm, client_control_norm = [], [], []

        loop_wall_start = time.perf_counter()
        control_seconds = 0.0
        for client_id in range(NUM_CLIENTS):
            local_model.load_state_dict(global_state)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=LR,
                                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            # Isolate this round/client's stochastic stream (Dropout + DataLoader)
            # from RNG consumed by previously trained clients.
            local_seed = TRAIN_SEED + rnd * 100 + client_id
            torch.manual_seed(local_seed)
            seed_accelerator(local_seed)
            generator = torch.Generator()
            generator.manual_seed(local_seed)
            if resident is not None:
                position_loader = DataLoader(LocalPositionDataset(sizes[client_id]),
                                             batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=0, generator=generator)
                loader = ResidentClientBatches(position_loader, client_indices_cuda[client_id],
                                               resident["x_train"], resident["y_train"], device)
            else:
                loader = DataLoader(datasets[client_id], batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=0, generator=generator)

            # This client's persistent control from the previous round; replaced only
            # after delta_c_i has been computed from it.
            c_i_old = client_controls[client_id]
            # c_old and c_i_old are both fixed for this client's whole local update,
            # so the constant gradient shift is a property of the round, not a batch.
            correction_norm = controls_diff_l2_norm(c_old, c_i_old)

            epoch_seconds, local_steps = 0.0, 0
            for _ in range(LOCAL_EPOCHS):
                stats = train_one_epoch(local_model, loader, criterion, optimizer, device,
                                        c_old, c_i_old)
                epoch_seconds += stats["train_seconds"]
                local_steps += stats["local_steps"]

            sync_device(device)
            control_start = time.perf_counter()
            y_params = {name: p.detach().clone()
                        for name, p in local_model.named_parameters() if p.requires_grad}
            c_i_new, delta_c_i = option_ii_client_control(
                c_i_old, c_old, x_params, y_params, local_steps, LR)
            sync_device(device)
            control_seconds += time.perf_counter() - control_start

            assert_controls_finite(c_i_new, f"{tag} r{rnd} client{client_id} c_i_new")
            client_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
            assert_state_finite(client_state, f"{tag} r{rnd} client{client_id}")

            client_states.append(client_state)
            client_stats.append(stats)
            client_seconds.append(epoch_seconds)
            client_update_l2.append(state_l2_distance(client_state, global_state))
            client_new_controls.append(c_i_new)
            client_delta_controls.append(delta_c_i)
            client_local_steps.append(local_steps)
            client_correction_norm.append(correction_norm)
            client_control_norm.append(controls_l2_norm(c_i_new))

            # Persist for the next round. Done after delta_c_i was taken, and it does
            # not affect c_old, which stays frozen for the remaining clients.
            client_controls[client_id] = c_i_new
        loop_wall_seconds = time.perf_counter() - loop_wall_start

        round_train_seconds = float(sum(client_seconds))
        # Only the client-side control work happened inside the client loop; the
        # server-control update below is outside it, so orchestration must be
        # measured against this figure, not the client+server total.
        client_control_seconds = control_seconds

        # Server control, sample-weighted with the same p_k as the model. This is the
        # project weighted extension, not the paper's uniform rule.
        sync_device(device)
        control_start = time.perf_counter()
        weighted_delta_c = weighted_controls(client_delta_controls, agg_weights)
        c_new = {name: c_old[name] + weighted_delta_c[name] for name in c_old}
        sync_device(device)
        control_seconds += time.perf_counter() - control_start
        assert_controls_finite(c_new, f"{tag} r{rnd} server control")

        # Full-participation identity, verified and recorded, never repaired.
        weighted_c_i_new = weighted_controls(client_new_controls, agg_weights)
        control_invariant_max_abs_error = max_controls_abs_diff(c_new, weighted_c_i_new)
        control_scale = max(1.0, max_controls_abs(c_new), max_controls_abs(weighted_c_i_new))
        assert control_invariant_max_abs_error <= CONTROL_INVARIANT_TOL * control_scale, (
            f"{tag} r{rnd}: server control does not match the sample-weighted "
            f"client-control aggregate (max abs diff {control_invariant_max_abs_error})"
        )
        server_control = c_new

        # Sample-weighted model aggregation, identical to d2_04.
        sync_device(device)
        agg_start = time.perf_counter()
        agg_state = aggregate_sample_weighted(client_states, sizes)
        global_model.load_state_dict(agg_state)
        sync_device(device)
        aggregation_seconds = time.perf_counter() - agg_start
        assert_state_finite(agg_state, f"{tag} r{rnd} aggregated")

        # Control-state work is real algorithm cost, so it is inside fl_round_seconds.
        # The per-batch gradient correction is already inside round_train_seconds.
        fl_round_seconds = round_train_seconds + control_seconds + aggregation_seconds
        round_orchestration_seconds = (loop_wall_seconds - round_train_seconds
                                       - client_control_seconds)

        sync_device(device)
        val_start = time.perf_counter()
        val = evaluate(global_model, val_loader, criterion, device)
        sync_device(device)
        validation_seconds = time.perf_counter() - val_start
        round_total_seconds = fl_round_seconds + validation_seconds

        total_fl_seconds += fl_round_seconds
        total_validation_seconds += validation_seconds
        total_run_seconds += round_total_seconds

        record = {"round": rnd,
                  "server_control_norm": controls_l2_norm(server_control),
                  "control_invariant_max_abs_error": control_invariant_max_abs_error,
                  "val_loss": val["val_loss"], "accuracy": val["accuracy"],
                  "balanced_accuracy": val["balanced_accuracy"], "macro_precision": val["macro_precision"],
                  "macro_recall": val["macro_recall"], "macro_f1": val["macro_f1"],
                  "weighted_f1": val["weighted_f1"], "macro_pr_auc": val["macro_pr_auc"],
                  "worst_class_f1": val["worst_class_f1"], "worst_class_recall": val["worst_class_recall"]}
        for c in range(NUM_CLASSES):
            record[f"precision_c{c}"] = float(val["per_precision"][c])
            record[f"recall_c{c}"] = float(val["per_recall"][c])
            record[f"f1_c{c}"] = float(val["per_f1"][c])
            record[f"support_c{c}"] = int(val["support"][c])
            record[f"predicted_count_c{c}"] = int(val["predicted_count"][c])
        for client_id in range(NUM_CLIENTS):
            record[f"train_loss_client_{client_id}"] = client_stats[client_id]["weighted_train_loss"]
            record[f"train_accuracy_client_{client_id}"] = client_stats[client_id]["online_train_accuracy"]
            record[f"train_samples_client_{client_id}"] = client_stats[client_id]["n_samples"]
            record[f"train_batches_client_{client_id}"] = client_stats[client_id]["n_batches"]
            record[f"update_l2_client_{client_id}"] = client_update_l2[client_id]
            record[f"local_steps_client_{client_id}"] = client_local_steps[client_id]
            record[f"correction_norm_client_{client_id}"] = client_correction_norm[client_id]
            record[f"control_norm_client_{client_id}"] = client_control_norm[client_id]
            record[f"train_seconds_client_{client_id}"] = client_seconds[client_id]
        record["round_train_seconds"] = round_train_seconds
        record["client_control_seconds"] = client_control_seconds
        record["control_seconds"] = control_seconds
        record["aggregation_seconds"] = aggregation_seconds
        record["fl_round_seconds"] = fl_round_seconds
        record["validation_seconds"] = validation_seconds
        record["round_total_seconds"] = round_total_seconds
        record["round_orchestration_seconds"] = round_orchestration_seconds
        history.append(record)

        if val["macro_f1"] > best["macro_f1"]:
            best = {"round": rnd, "macro_f1": float(val["macro_f1"])}
            torch.save(global_model.state_dict(), best_path)

        print(f"[{tag}] round={rnd:02d} val_macro_f1={val['macro_f1']:.4f} "
              f"bal_acc={val['balanced_accuracy']:.4f} acc={val['accuracy']:.4f} "
              f"worst_f1={val['worst_class_f1']:.4f} "
              f"fl_s={fl_round_seconds:.1f} val_s={validation_seconds:.1f}", flush=True)

        pd.DataFrame(history).to_csv(hist_path, index=False)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(hist_path, index=False)
    torch.save(global_model.state_dict(), final_path)

    argmax_round = int(hist_df.loc[hist_df["macro_f1"].idxmax(), "round"])
    assert best["round"] == argmax_round, f"{tag}: best_round {best['round']} != argmax {argmax_round}"

    reloaded_state = torch.load(best_path, map_location=device)
    assert_state_finite(reloaded_state, f"{tag} best checkpoint")
    local_model.load_state_dict(reloaded_state)
    reloaded_macro_f1 = float(evaluate(local_model, val_loader, criterion, device)["macro_f1"])
    assert abs(reloaded_macro_f1 - best["macro_f1"]) < RELOAD_F1_TOL, (
        f"{tag}: reloaded macro-F1 {reloaded_macro_f1} != best {best['macro_f1']}"
    )

    config = {
        "method": METHOD,
        "dataset": DATASET,
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
        "K": NUM_CLIENTS,
        "partition_seed": partition_seed,
        "condition": condition,
        "training_seed": TRAIN_SEED,
        "client_sizes": sizes,
        "client_class_counts": part_data["client_class_counts"].tolist(),
        "aggregation": "sample_weighted_fedavg",
        "aggregation_weights": agg_weights,
        "control_variate_option": "SCAFFOLD Option II",
        "gradient_correction": "g - c_i + c",
        "control_update_rule": "c_i_new = c_i_old - c_old + (x - y_i) / (local_steps * lr)",
        "tau_source": "actual optimizer.step() count",
        "model_aggregation": "sample weighted by actual client sizes",
        "server_control_rule": "c_new = c_old + sum_k (n_k / sum_j n_j) * delta_c_i",
        "server_control_aggregation": SERVER_CONTROL_AGGREGATION,
        "control_weighting_provenance": CONTROL_WEIGHTING_PROVENANCE,
        "control_weighting_is_project_extension": True,
        "control_weighting_matches_original_displayed_equation": False,
        "original_theoretical_guarantees_claimed": CONTROL_THEORY_CLAIM,
        "client_controls_persistent_across_rounds": True,
        "controls_reset_between_independent_runs": True,
        "control_invariant": "c == sample-weighted aggregate of c_i",
        "control_invariant_tolerance": CONTROL_INVARIANT_TOL,
        "control_invariant_handling": "verified and recorded per round; never repaired",
        "control_scope": "trainable named parameters",
        "control_init": "zeros, reset per (partition_seed, condition) run",
        "control_device": str(device),
        "local_steps_source": "counted optimizer.step() calls",
        "batch_size": BATCH_SIZE,
        "local_epochs": LOCAL_EPOCHS,
        "optimizer": "SGD",
        "lr": LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "max_rounds": MAX_ROUNDS,
        "best_round": best["round"],
        "best_val_macro_f1": best["macro_f1"],
        "best_checkpoint_reloaded_val_macro_f1": reloaded_macro_f1,
        "class_weights": weight_f32.tolist(),
        "class_weights_source": str(FEDAVG_CLASS_WEIGHTS_PATH),
        "class_weights_sha256": file_sha256(FEDAVG_CLASS_WEIGHTS_PATH),
        "selection_metric": "val_macro_f1",
        "processed_dir": str(PROCESSED_DIR),
        "partition_root": str(PART_ROOT),
        "initial_state_path": str(INIT_PATH),
        "initial_state_sha256": file_sha256(INIT_PATH),
        "initial_state_verified_against_reproduced_d2_04_init": True,
        "scaffold_initial_state_written": False,
        "best_checkpoint_path": str(best_path),
        "final_checkpoint_path": str(final_path),
        "partition_path": part_data["part_dir"],
        "device": str(device),
        "resident_data_path": resident is not None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git_commit": git_commit_hash(),
        "git_dirty": git_dirty(),
        "script_sha256": script_sha256(),
    }
    with open(RESULTS_DIR / f"config_{tag}.json", "w") as f:
        json.dump(config, f, indent=2)

    return {"method": METHOD, "dataset": DATASET, "input_dim": INPUT_DIM, "K": NUM_CLIENTS,
            "partition_seed": partition_seed, "condition": condition,
            "training_seed": TRAIN_SEED, "batch_size": BATCH_SIZE,
            "lr": LR, "max_rounds": MAX_ROUNDS,
            "best_round": best["round"], "best_val_macro_f1": best["macro_f1"],
            "total_fl_seconds": total_fl_seconds,
            "total_validation_seconds": total_validation_seconds,
            "total_run_seconds": total_run_seconds}


def preflight_outputs() -> None:
    """Fail if any intended SCAFFOLD output already exists.

    No initial-model entry: this runner never creates one, because the
    authoritative initialisation is the read-only FedAvg checkpoint.
    """
    intended = [CLASS_WEIGHTS_PATH, RESULTS_DIR / "final_summary.csv"]
    for partition_seed in PARTITION_SEEDS:
        for condition in CONDITIONS:
            tag = f"scaffold_k{NUM_CLIENTS}_seed{partition_seed}_{condition}"
            intended += [RESULTS_DIR / f"history_{tag}.csv",
                         RESULTS_DIR / f"config_{tag}.json",
                         MODELS_DIR / f"best_{tag}.pt",
                         MODELS_DIR / f"final_{tag}.pt"]
    existing = [p for p in intended if p.exists()]
    if existing:
        listing = "\n  ".join(str(p) for p in existing)
        raise RuntimeError(f"Refusing to run; Dataset-2 SCAFFOLD outputs already present:\n  {listing}")


def assert_execution_enabled() -> None:
    """Refuse to start while the gate is closed, before anything is created."""
    if not SCAFFOLD_EXECUTION_ENABLED:
        raise SystemExit(
            "d2_06_train_scaffold.py: SCAFFOLD_EXECUTION_ENABLED is False, so this run "
            "refuses to start. Nothing has been created or written."
        )


def reproduce_d2_04_initial_state(device: torch.device) -> dict:
    """Reproduce d2_04's initialisation: seed, build on device, clone to CPU."""
    set_all_seeds(TRAIN_SEED)
    model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_and_verify_fedavg_initial_state(device: torch.device) -> tuple[dict, str]:
    """Load the mandatory FedAvg initial model and prove it is d2_04's initialisation.

    SCAFFOLD must start from the same weights as FedAvg for the comparison to be
    meaningful, so any doubt about this file aborts the run before training.
    """
    if not INIT_PATH.exists():
        raise RuntimeError(
            f"Refusing to run: the FedAvg initial global model is mandatory and was not "
            f"found at {INIT_PATH}."
        )
    sha_before = file_sha256(INIT_PATH)
    try:
        initial_state = torch.load(INIT_PATH, map_location="cpu")
    except Exception as error:
        raise RuntimeError(
            f"Refusing to run: the FedAvg initial global model at {INIT_PATH} could not "
            f"be read ({type(error).__name__}: {error})."
        ) from error

    assert isinstance(initial_state, dict) and initial_state, \
        f"{INIT_PATH} did not contain a non-empty state_dict"
    assert_state_finite(initial_state, "FedAvg initial_state")
    loaded_shapes = {k: tuple(v.shape) for k, v in initial_state.items()}
    assert loaded_shapes == expected_state_shapes(), (
        f"FedAvg initial state at {INIT_PATH} does not match a {INPUT_DIM}-input, "
        f"{NUM_CLASSES}-class model: {loaded_shapes}"
    )

    reproduced = reproduce_d2_04_initial_state(device)
    assert reproduced.keys() == initial_state.keys(), "initial-state key sets differ"
    for key in reproduced:
        a, b = reproduced[key].cpu(), initial_state[key].cpu()
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), (
            f"Refusing to run: the FedAvg initial global model does not match an "
            f"independent reproduction of the d2_04 initialisation at '{key}'. "
            f"checkpoint={INIT_PATH} sha256={sha_before}"
        )
    print(f"FedAvg initial state verified: {INIT_PATH} sha256={sha_before} "
          f"EXACT_STATE_EQUAL=True", flush=True)
    return initial_state, sha_before


def load_and_verify_fedavg_class_weights(y_train: np.ndarray) -> tuple[np.ndarray, str]:
    """Load the mandatory FedAvg class weights and require exact equality with a fresh
    computation, so SCAFFOLD trains against the same weighted objective as FedAvg."""
    if not FEDAVG_CLASS_WEIGHTS_PATH.exists():
        raise RuntimeError(
            f"Refusing to run: the FedAvg class-weight vector is mandatory and was not "
            f"found at {FEDAVG_CLASS_WEIGHTS_PATH}."
        )
    sha_before = file_sha256(FEDAVG_CLASS_WEIGHTS_PATH)
    fedavg_weights = np.load(FEDAVG_CLASS_WEIGHTS_PATH)

    weight_f32 = class_weights_full(y_train).astype(np.float32)
    assert np.isfinite(weight_f32).all() and (weight_f32 > 0).all(), \
        "class weights must be finite and strictly positive"
    assert fedavg_weights.dtype == np.float32, \
        f"FedAvg class weights dtype {fedavg_weights.dtype}, expected float32"
    assert fedavg_weights.shape == (NUM_CLASSES,), \
        f"FedAvg class weights shape {fedavg_weights.shape}, expected ({NUM_CLASSES},)"
    assert np.array_equal(fedavg_weights, weight_f32), (
        "FedAvg class weights differ from freshly computed full-y_train float32 weights"
    )
    print(f"FedAvg class weights verified: {FEDAVG_CLASS_WEIGHTS_PATH} sha256={sha_before} "
          f"EXACT_WEIGHTS_EQUAL=True", flush=True)
    return weight_f32, sha_before


def main() -> None:
    argparse.ArgumentParser(
        description="Dataset-2 K=5 SCAFFOLD production matrix (gated by "
                    "SCAFFOLD_EXECUTION_ENABLED)."
    ).parse_args()

    assert_no_test_reference()
    # Gate first: nothing below runs while it is closed, so a refused invocation
    # creates no directory and writes no file.
    assert_execution_enabled()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preflight_outputs()

    device = get_device()
    class_names = load_class_names()
    assert class_names == EXPECTED_CLASS_ORDER, (
        f"label mapping class order {class_names} does not match {EXPECTED_CLASS_ORDER}"
    )

    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")
    assert np.issubdtype(y_train.dtype, np.integer) and np.issubdtype(y_val.dtype, np.integer), \
        "labels are not integers"
    expected_ids = list(range(NUM_CLASSES))
    assert sorted(int(v) for v in np.unique(y_train).tolist()) == expected_ids, \
        "y_train label ids do not equal the expected Dataset-2 ids"
    assert sorted(int(v) for v in np.unique(y_val).tolist()) == expected_ids, \
        "y_val label ids do not equal the expected Dataset-2 ids"
    assert len(y_train) == EXPECTED_TRAIN_ROWS, f"y_train has {len(y_train)} rows"
    assert len(y_val) == EXPECTED_VAL_ROWS, f"y_val has {len(y_val)} rows"

    global_class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    assert (global_class_counts > 0).all(), "a class has zero training examples"
    assert (np.bincount(y_val, minlength=NUM_CLASSES) > 0).all(), \
        "a class has zero validation examples"

    x_train_peek = np.load(PROCESSED_DIR / "X_train.npy", mmap_mode="r")
    x_val_peek = np.load(PROCESSED_DIR / "X_val.npy", mmap_mode="r")
    assert x_train_peek.shape == (EXPECTED_TRAIN_ROWS, INPUT_DIM), f"X_train {x_train_peek.shape}"
    assert x_val_peek.shape == (EXPECTED_VAL_ROWS, INPUT_DIM), f"X_val {x_val_peek.shape}"

    # Both mandatory FedAvg inputs are verified before any production artefact is
    # written, so a failed prerequisite leaves no partial output behind.
    weight_f32, fedavg_weights_sha = load_and_verify_fedavg_class_weights(y_train)
    initial_state, fedavg_init_sha = load_and_verify_fedavg_initial_state(device)
    np.save(CLASS_WEIGHTS_PATH, weight_f32)

    # CUDA: the row-by-row CPU fetch is the bottleneck and the arrays fit in device
    # memory, so make them resident once, after every assertion above has passed.
    resident = None
    if device.type == "cuda":
        resident = {
            "x_train": torch.from_numpy(np.load(PROCESSED_DIR / "X_train.npy")).to(torch.float32).to(device),
            "y_train": torch.from_numpy(np.load(PROCESSED_DIR / "y_train.npy")).to(torch.long).to(device),
            "x_val": torch.from_numpy(np.load(PROCESSED_DIR / "X_val.npy")).to(torch.float32).to(device),
            "y_val": torch.from_numpy(np.load(PROCESSED_DIR / "y_val.npy")).to(torch.long).to(device),
        }
        assert tuple(resident["x_train"].shape) == (EXPECTED_TRAIN_ROWS, INPUT_DIM), "resident X_train shape"
        assert tuple(resident["x_val"].shape) == (EXPECTED_VAL_ROWS, INPUT_DIM), "resident X_val shape"
        val_loader = ResidentValLoader(resident["x_val"], resident["y_val"], 4096)
        print(f"resident on {device}: X_train{tuple(resident['x_train'].shape)} "
              f"X_val{tuple(resident['x_val'].shape)}", flush=True)
    else:
        val_loader = DataLoader(FullDataset(PROCESSED_DIR / "X_val.npy", PROCESSED_DIR / "y_val.npy"),
                                batch_size=4096, shuffle=False, num_workers=0)

    part_cache = {}
    for partition_seed in PARTITION_SEEDS:
        for condition in CONDITIONS:
            part_cache[(partition_seed, condition)] = load_partition(
                partition_seed, condition, y_train, global_class_counts)

    print(f"method={METHOD} dataset={DATASET} input_dim={INPUT_DIM} classes={NUM_CLASSES} "
          f"K={NUM_CLIENTS} lr={LR} batch_size={BATCH_SIZE} local_epochs={LOCAL_EPOCHS} "
          f"max_rounds={MAX_ROUNDS} seeds={PARTITION_SEEDS} conditions={CONDITIONS} "
          f"device={device}", flush=True)

    summary_rows = []
    for partition_seed in PARTITION_SEEDS:
        for condition in CONDITIONS:
            row = run(partition_seed, condition, part_cache[(partition_seed, condition)],
                      initial_state, weight_f32, val_loader, device, resident)
            summary_rows.append(row)
            print(f"[scaffold_k{NUM_CLIENTS}_seed{partition_seed}_{condition}] DONE "
                  f"best_round={row['best_round']} best_val_macro_f1={row['best_val_macro_f1']:.4f}\n",
                  flush=True)

    # The FedAvg inputs must be exactly as they were before this run.
    saved_init = torch.load(INIT_PATH, map_location="cpu")
    assert states_equal(saved_init, initial_state), "initial state changed during training"
    assert file_sha256(INIT_PATH) == fedavg_init_sha, "FedAvg initial state file was modified"
    assert file_sha256(FEDAVG_CLASS_WEIGHTS_PATH) == fedavg_weights_sha, \
        "FedAvg class-weight file was modified"

    assert len(summary_rows) == len(PARTITION_SEEDS) * len(CONDITIONS), "unexpected number of runs"
    summary_path = RESULTS_DIR / "final_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
