"""
Corrected 37-feature Dataset-1 K=5 SCAFFOLD final training runner.

This is the implemented Dataset-1 37-feature SCAFFOLD runner, not a skeleton or a
placeholder: the control-variate mechanism described below is present and active,
and every SCAFFOLD-specific quantity it produces is recorded per client and per
round. Execution is separately gated by SCAFFOLD_EXECUTION_ENABLED, which is
currently True. While that gate is False, main() refuses to start and no
directory, class-weight file, history, config, or checkpoint is written.

ALGORITHM
---------
SCAFFOLD with Option II control variates, full participation, K=5.

Each independent (partition_seed, condition) run owns one server control variate c
and five client control variates c_i. All are keyed by the trainable named model
parameters, start at zero, are detached, and persist across the 40 communication
rounds; they are rebuilt at zero for every new seed/condition run, so runs never
share control state.

Round r:
  x      = global parameters at the start of the round
  c_old  = server control snapshotted at the start of the round; every one of the
           five clients in this round uses this same c_old
  Each client k starts from x with its own persistent c_i_old and, for every
  minibatch, applies the corrected direction

      loss.backward()
      param.grad = param.grad - c_i_old[name] + c_old[name]     (all trainable params)
      optimizer.step()
      local_steps += 1

  local_steps is the counted number of optimizer.step() calls, never inferred from
  client size, batch size or local epochs. After local training, with y_i the final
  local parameters and LR the local learning rate:

      c_i_new   = c_i_old - c_old + (x - y_i) / (local_steps * LR)
      delta_c_i = c_i_new - c_i_old            (computed before c_i_old is replaced)

  Both aggregations use the same sample weights p_k = n_k / sum_j n_j:
      global MODEL   - sample-weighted by the actual client sizes n_k, exactly as in
                       32_train_final_fedavg_37f.py
      server CONTROL - c_new = c_old + sum_k p_k * delta_c_i

  Under full participation this preserves c == sum_k p_k * c_i; that identity is
  verified numerically after every server-control update and is never repaired by
  overwriting either side.

The task loss is the unchanged class-weighted CrossEntropyLoss. SCAFFOLD adds no
loss term of its own: the correction acts on the gradient, not on the objective.

Scope
-----
32_train_final_fedavg_37f.py is the authority for every behaviour shared across
methods, and this runner reproduces it exactly: the same MLP, the corrected
37-feature Dataset-1 representation (data/processed_37f), the final K=5 partitions
(data/fl_clients/final_partitions/k_5), partition seeds {42, 43, 44} x conditions
{iid, alpha_0p1, alpha_0p5, alpha_1p0} = 12 runs, class-weighted CrossEntropyLoss
from the complete y_train, plain SGD at learning rate 0.1 (momentum 0, weight
decay 0), batch size 4096, one local epoch, up to 40 rounds, full client
participation, training/init seed 42 with the per-round-per-client seed scheme,
identical DataLoader behaviour, the partition coverage assertions, the validation
metric set, validation Macro-F1 checkpoint selection, and the best-checkpoint
reload verification. Learning rate, batch size, and rounds are identical for every
condition; there is no per-alpha or additional tuning.

The global MODEL aggregation is sample-weighted by the actual client sizes n_k,
exactly as in script 32. The server control uses those same weights, so it
represents the same weighted client objective as the global model.

DIAGNOSTICS
-----------
Per client per round the history records local_steps_client_k together with four
control quantities, each with a single unambiguous meaning:

    control_norm_before_client_k = ||c_i_old||_2
    control_norm_after_client_k  = ||c_i_new||_2
    control_delta_norm_client_k  = ||delta_c_i||_2
    correction_norm_client_k     = ||c_old - c_i_old||_2

correction_norm is the size of the constant direction shift that client k applies
to every minibatch gradient this round. c_old and c_i_old are both fixed for the
whole of that client's local update, so it is computed once outside the minibatch
loop rather than per batch.

Per round the history records the server-side counterparts:

    server_control_norm_before      = ||c_old||_2
    server_control_norm_after       = ||c_new||_2
    server_control_delta_norm       = ||c_new - c_old||_2
    control_invariant_max_abs_error = max |c_new - sum_k p_k * c_i_new|

control_invariant_max_abs_error is the numerical residual of the full-participation
identity. It is asserted against a scaled tolerance and reported; it is never
repaired by overwriting either side.

TIMING
------
SCAFFOLD's control-state work is real algorithm cost and is timed explicitly, so
the FL runtime accounting is not just local minibatch training:

    round_train_seconds           sum over clients of the local minibatch training
                                  loop, including SCAFFOLD gradient correction
    round_client_control_seconds  sum over clients of the Option II client-control
                                  state update (y_i snapshot, c_i_new, delta_c_i)
    aggregation_seconds           sample-weighted MODEL aggregation and load
    server_control_update_seconds server-control state delta and update
    round_control_state_update_seconds = round_client_control_seconds
                                     + server_control_update_seconds
    fl_round_seconds              = round_train_seconds
                                     + round_client_control_seconds
                                     + aggregation_seconds
                                     + server_control_update_seconds

round_control_state_update_seconds covers only the control-STATE updates: the
Option II client-control state update plus the server-control state update. It
does NOT include the per-minibatch gradient correction g <- g - c_i + c, which is
executed inside the training loop and is therefore already counted in
round_train_seconds. The same holds for total_control_state_update_seconds.

fl_round_seconds deliberately excludes validation, CSV writing, checkpoint
writing, provenance hashing, and general Python orchestration and diagnostics.
round_orchestration_seconds is reported separately and is exactly the non-algorithm
overhead inside the client loop: per-client model loading, optimizer and DataLoader
construction, seeding, CPU state copies, control-norm diagnostics, and finiteness
assertions. Nothing about the algorithm was changed in order to time it.

34_train_final_fedprox_37f.py is used only as the engineering pattern for the
read-only input handling: loading the exact shared FedAvg initial model, verifying
the exact FedAvg class weights against a fresh computation, provenance hashing,
isolated output roots, and overwrite protection.

Read-only inputs, from the completed 37-feature FedAvg K=5 run:
    models/final_fedavg_k5_37f/initial_global_model.pt
    results/final_fedavg_k5_37f/class_weights.npy
Both are hashed; the initial state is additionally checked to have the exact
parameter shapes of a 37-input model, and is re-verified unmodified on disk and in
memory after training.

Outputs go only under results/final_scaffold_k5_37f/ and
models/final_scaffold_k5_37f/. No existing FedAvg or FedProx artefact is read for
anything other than the two inputs named above, and none is written.

Reads X_train/y_train (train) and X_val/y_val (validation) of the 37-feature branch
only; never reads the held-out arrays. Both are asserted to have exactly 37
columns. Checkpoint selection is validation Macro-F1; the selected round may occur
before round 40, which is only the fixed maximum budget. Timings are simulation
runtime, not communication latency.
"""

from pathlib import Path
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

PROCESSED_DIR = Path("data/processed_37f")
LABEL_MAP = Path("configs/label_mapping.json")
PART_ROOT = Path("data/fl_clients/final_partitions/k_5")

# Read-only inputs produced by the completed 37-feature FedAvg K=5 run (script 32).
FEDAVG_MODELS_DIR = Path("models/final_fedavg_k5_37f")
FEDAVG_RESULTS_DIR = Path("results/final_fedavg_k5_37f")
INIT_PATH = FEDAVG_MODELS_DIR / "initial_global_model.pt"
FEDAVG_CLASS_WEIGHTS_PATH = FEDAVG_RESULTS_DIR / "class_weights.npy"

# SCAFFOLD 37-feature outputs only.
RESULTS_DIR = Path("results/final_scaffold_k5_37f")
MODELS_DIR = Path("models/final_scaffold_k5_37f")
CLASS_WEIGHTS_PATH = RESULTS_DIR / "class_weights.npy"

METHOD = "SCAFFOLD"
REPRESENTATION = "37f"
INPUT_DIM = 37
NUM_CLASSES = 10
NUM_CLIENTS = 5
LOCAL_EPOCHS = 1
MAX_ROUNDS = 40
TRAIN_SEED = 42
LR = 0.1
MOMENTUM = 0.0
WEIGHT_DECAY = 0.0
BATCH_SIZE = 4096
RELOAD_F1_TOL = 1e-4

# Stage gate, retained: while False, main() refuses to run so that results produced
# without the control mechanism cannot be written into the scaffold roots and later
# mistaken for SCAFFOLD runs. The control variates are implemented, so this is True.
CONTROL_VARIATES_IMPLEMENTED = True
SCAFFOLD_STAGE = "control_variates_implemented"
# Separate execution gate. Implemented is not the same as cleared to run: while this
# is False, main() refuses to start before it creates any directory or writes any
# class-weight file, history, config, or checkpoint. Set it to True only as a
# deliberate decision to launch the SCAFFOLD training matrix.
SCAFFOLD_EXECUTION_ENABLED = True
# Tolerance for the full-participation identity c == sum_k p_k * c_i, scaled by the
# magnitude of the controls. Diagnostic verification only; never used to repair.
CONTROL_INVARIANT_TOL = 1e-4

PARTITION_SEEDS = [42, 43, 44]
CONDITIONS = ["iid", "alpha_0p1", "alpha_0p5", "alpha_1p0"]


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


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    # Make MPS timings wall-accurate; no-op on CPU.
    if device.type == "mps":
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
    return [id_to_name[c] for c in range(len(id_to_name))]


def class_weights_full(y_train: np.ndarray) -> np.ndarray:
    # N / (num_classes * class_count), computed once from the full training labels.
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    total = counts.sum()
    return total / (NUM_CLASSES * counts)


def assert_state_finite(state: dict, name: str) -> None:
    # Every floating tensor in the state_dict must be finite.
    for k, v in state.items():
        if v.is_floating_point():
            assert torch.isfinite(v).all(), f"{name}: non-finite values in {k}"


def state_l2_distance(a: dict, b: dict) -> float:
    # L2 distance over floating tensors shared by both state_dicts.
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
    """Parameter/buffer shapes of a freshly built 37-input model."""
    probe = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES)
    return {k: tuple(v.shape) for k, v in probe.state_dict().items()}


# --------------------------- SCAFFOLD control variates ---------------------- #
def init_controls(model: nn.Module, device: torch.device) -> dict:
    """Zero control variate keyed by the trainable named parameters of the model.

    Detached and independent of the model's autograd graph. Called once per client
    and once for the server at the start of each independent run, so control state
    never carries across seed/condition runs.
    """
    return {
        name: torch.zeros_like(p, device=device).detach()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def clone_controls(controls: dict) -> dict:
    """Detached copy, used to freeze c_old for the duration of a round."""
    return {name: t.detach().clone() for name, t in controls.items()}


def controls_l2_norm(controls: dict) -> float:
    """L2 norm over all control tensors, flattened together."""
    total = sum(float(torch.sum(t.to(torch.float32) ** 2).item()) for t in controls.values())
    return float(total ** 0.5)


def controls_diff_l2_norm(a: dict, b: dict) -> float:
    """L2 norm of (a - b) over all control tensors, flattened together."""
    assert a.keys() == b.keys(), "control difference: key mismatch"
    total = sum(float(torch.sum((a[name].to(torch.float32) - b[name].to(torch.float32)) ** 2).item())
                for name in a)
    return float(total ** 0.5)


def weighted_controls(control_list: list[dict], weights: list[float]) -> dict:
    """Weighted sum of client control tensors."""
    assert len(control_list) == len(weights) == NUM_CLIENTS, \
        "weighted_controls: controls/weights count mismatch"
    assert all(w > 0.0 for w in weights), "weighted_controls: a weight is not positive"
    assert abs(sum(weights) - 1.0) < 1e-9, "weighted_controls: weights do not sum to 1"
    keys = control_list[0].keys()
    assert all(c.keys() == keys for c in control_list), "weighted_controls: key mismatch"
    return {name: sum(control_list[k][name] * weights[k] for k in range(len(control_list)))
            for name in keys}


def max_controls_abs_diff(a: dict, b: dict) -> float:
    """Max elementwise |a - b| across all control tensors."""
    assert a.keys() == b.keys(), "control comparison: key mismatch"
    return max(float((a[name] - b[name]).abs().max().item()) for name in a)


def max_controls_abs(controls: dict) -> float:
    """Largest absolute control entry, used to scale the invariant tolerance."""
    return max(float(t.abs().max().item()) for t in controls.values())


def assert_controls_finite(controls: dict, name: str) -> None:
    for key, t in controls.items():
        assert torch.isfinite(t).all(), f"{name}: non-finite control values in {key}"


def train_one_epoch(model, loader, criterion, optimizer, device,
                    c_old, c_i_old) -> dict:
    # Backward loss is unchanged (weighted mean); SCAFFOLD adds no loss term. The
    # control correction is applied to the gradients after loss.backward() and
    # before optimizer.step(). Diagnostics accumulate as detached on-device tensors;
    # no host sync (.item()/.cpu()/bool) happens inside the batch loop. Timing
    # covers only the training batch loop, not the final conversion.
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
        # SCAFFOLD corrected direction: g <- g - c_i + c, applied to every trainable
        # parameter between backward() and step().
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                assert param.grad is not None, f"no gradient for trainable parameter {name}"
                param.grad = param.grad - c_i_old[name] + c_old[name]
        optimizer.step()
        # Counted, never inferred from client size, batch size or local epochs.
        local_steps += 1
        # On-device diagnostic accumulation only.
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
    # Single host transfer of the accumulated diagnostics, after the epoch.
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
    assert (support > 0).all(), "validation set is missing one or more of the 10 classes"
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
    """Load and verify one final partition; returns datasets, sizes, and class counts.

    Read-only: the partition files are never written, moved, or regenerated here.
    """
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
    return {"datasets": datasets, "sizes": sizes, "client_class_counts": counts,
            "part_dir": str(part_dir)}


def run(partition_seed, condition, part_data, initial_state,
        weight_f32, val_loader, device) -> dict:
    tag = f"scaffold37f_k{NUM_CLIENTS}_seed{partition_seed}_{condition}"
    datasets = part_data["datasets"]
    sizes = part_data["sizes"]
    total_size = float(sum(sizes))
    agg_weights = [n / total_size for n in sizes]

    # Start every run from an identical copy of the shared initial state.
    set_all_seeds(TRAIN_SEED)
    global_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    global_model.load_state_dict(copy.deepcopy(initial_state))
    assert states_equal(global_model.state_dict(), initial_state), f"{tag}: initial state not loaded identically"
    local_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight_f32, dtype=torch.float32, device=device))

    # Control variates for this run only: one server control and one per client, all
    # zero-initialised over the trainable named parameters. They persist across the
    # rounds below and are discarded when this run returns, so each independent
    # (partition_seed, condition) run starts from zero controls.
    server_control = init_controls(global_model, device)
    client_controls = [init_controls(global_model, device) for _ in range(NUM_CLIENTS)]
    assert len(client_controls) == NUM_CLIENTS, f"{tag}: expected {NUM_CLIENTS} client controls"
    for client_id, c_i in enumerate(client_controls):
        assert c_i.keys() == server_control.keys(), f"{tag}: client {client_id} control keys differ from server"
        assert all(float(t.abs().max().item()) == 0.0 for t in c_i.values()), \
            f"{tag}: client {client_id} control did not start at zero"
    assert all(float(t.abs().max().item()) == 0.0 for t in server_control.values()), \
        f"{tag}: server control did not start at zero"

    best = {"round": -1, "macro_f1": -1.0}
    best_path = MODELS_DIR / f"best_{tag}.pt"
    final_path = MODELS_DIR / f"final_{tag}.pt"
    hist_path = RESULTS_DIR / f"history_{tag}.csv"
    history = []
    total_fl_seconds, total_validation_seconds, total_run_seconds = 0.0, 0.0, 0.0
    total_control_state_update_seconds = 0.0

    for rnd in range(1, MAX_ROUNDS + 1):
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        # x: the global trainable parameters at the start of this round, and c_old:
        # the server control frozen for the round. Every client below uses these same
        # two snapshots.
        x_params = {name: p.detach().clone()
                    for name, p in global_model.named_parameters() if p.requires_grad}
        c_old = clone_controls(server_control)
        client_states, participated = [], []
        client_stats, client_seconds, client_update_l2 = [], [], []
        client_delta_controls, client_new_controls = [], []
        client_local_steps = []
        # Per-client SCAFFOLD control diagnostics and control-update timing.
        client_control_norm_before, client_control_norm_after = [], []
        client_control_delta_norm, client_correction_norm = [], []
        client_control_seconds = []

        loop_wall_start = time.perf_counter()
        for client_id in range(NUM_CLIENTS):
            local_model.load_state_dict(global_state)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=LR,
                                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            # Isolate this round/client's stochastic stream (Dropout + DataLoader)
            # from RNG consumed by previously trained clients.
            local_seed = TRAIN_SEED + rnd * 100 + client_id
            torch.manual_seed(local_seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(local_seed)
            generator = torch.Generator()
            generator.manual_seed(local_seed)
            loader = DataLoader(datasets[client_id], batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=0, generator=generator)

            # This client's persistent control from the previous round. It is read
            # here and only replaced after delta_c_i has been computed from it.
            c_i_old = client_controls[client_id]

            # c_old and c_i_old are both fixed for the whole of this client's local
            # update, so the constant gradient shift ||c_old - c_i_old||_2 is a
            # property of the round, not of a batch. Computed once here, outside the
            # minibatch loop, and never recomputed per batch.
            control_norm_before = controls_l2_norm(c_i_old)
            correction_norm = controls_diff_l2_norm(c_old, c_i_old)

            # train_seconds covers only the local minibatch training loop, including
            # SCAFFOLD gradient correction (per train_one_epoch); the control-state
            # update below is timed separately.
            epoch_seconds = 0.0
            local_steps = 0
            for _ in range(LOCAL_EPOCHS):
                stats = train_one_epoch(local_model, loader, criterion, optimizer, device,
                                        c_old, c_i_old)
                epoch_seconds += stats["train_seconds"]
                local_steps += stats["local_steps"]
            client_seconds.append(epoch_seconds)
            assert local_steps > 0, f"{tag} r{rnd} client{client_id}: local_steps must be positive"

            # SCAFFOLD Option II, using y_i (the final local parameters) and the
            # counted optimizer-step total:
            #   c_i_new = c_i_old - c_old + (x - y_i) / (local_steps * LR)
            # This is control-STATE work, distinct from the local minibatch training
            # loop above, so it is timed in its own right. The per-minibatch gradient
            # correction is not part of it; that already sits inside train_seconds.
            # The timed block contains only the computation; the finiteness checks and
            # norm diagnostics below are left outside it.
            sync_device(device)
            control_start = time.perf_counter()
            y_params = {name: p.detach().clone()
                        for name, p in local_model.named_parameters() if p.requires_grad}
            scale = float(local_steps) * LR
            c_i_new = {
                name: c_i_old[name] - c_old[name] + (x_params[name] - y_params[name]) / scale
                for name in c_i_old
            }
            # delta_c_i is computed from the un-replaced c_i_old.
            delta_c_i = {name: c_i_new[name] - c_i_old[name] for name in c_i_old}
            sync_device(device)
            control_update_seconds = time.perf_counter() - control_start
            client_control_seconds.append(control_update_seconds)

            assert_controls_finite(c_i_new, f"{tag} r{rnd} client{client_id} c_i_new")
            assert_controls_finite(delta_c_i, f"{tag} r{rnd} client{client_id} delta_c_i")

            client_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
            assert_state_finite(client_state, f"{tag} r{rnd} client{client_id}")
            client_states.append(client_state)
            client_stats.append(stats)
            client_update_l2.append(state_l2_distance(client_state, global_state))
            client_new_controls.append(c_i_new)
            client_delta_controls.append(delta_c_i)
            client_local_steps.append(local_steps)
            client_control_norm_before.append(control_norm_before)
            client_control_norm_after.append(controls_l2_norm(c_i_new))
            client_control_delta_norm.append(controls_l2_norm(delta_c_i))
            client_correction_norm.append(correction_norm)
            participated.append(client_id)

            # Persist this client's control for the next communication round. Done
            # after delta_c_i has been taken, and it does not affect c_old, which
            # stays frozen for the remaining clients in this round.
            client_controls[client_id] = c_i_new
        loop_wall_seconds = time.perf_counter() - loop_wall_start

        assert sorted(participated) == list(range(NUM_CLIENTS)), f"{tag} round {rnd}: participation {participated}"
        assert len(client_delta_controls) == NUM_CLIENTS, f"{tag} round {rnd}: missing client control deltas"

        # The local minibatch training loop (including SCAFFOLD gradient correction)
        # and the Option II client-control state update are both real algorithm cost
        # and are summed separately.
        round_train_seconds = float(sum(client_seconds))
        round_client_control_seconds = float(sum(client_control_seconds))
        # Everything else the client loop spent: model loading, optimizer and
        # DataLoader construction, seeding, CPU state copies, control-norm
        # diagnostics, and finiteness assertions. Not part of fl_round_seconds.
        round_orchestration_seconds = (loop_wall_seconds - round_train_seconds
                                       - round_client_control_seconds)

        server_control_norm_before = controls_l2_norm(c_old)

        # Server control update, sample-weighted over clients using the same
        # p_k = n_k / sum_j n_j as the model (full participation):
        #   c_new = c_old + sum_k p_k * delta_c_i
        # Timed as control-STATE work; the invariant check and norm diagnostics
        # below stay outside the timed block.
        sync_device(device)
        server_control_start = time.perf_counter()
        weighted_delta_c = weighted_controls(client_delta_controls, agg_weights)
        c_new = {name: c_old[name] + weighted_delta_c[name] for name in c_old}
        sync_device(device)
        server_control_update_seconds = time.perf_counter() - server_control_start

        assert_controls_finite(c_new, f"{tag} r{rnd} server control")

        # Formed explicitly from the two server-control states rather than reused from
        # weighted_delta_c, so the recorded delta is exactly c_new - c_old.
        server_control_delta = {name: c_new[name] - c_old[name] for name in c_old}
        server_control_delta_norm = controls_l2_norm(server_control_delta)
        server_control_norm_after = controls_l2_norm(c_new)

        # Full-participation identity: c must equal the sample-weighted aggregate of
        # the client controls. Verified, never repaired - neither side is overwritten
        # to force agreement, and the residual is recorded in the history.
        weighted_c_i_new = weighted_controls(client_new_controls, agg_weights)
        control_invariant_max_abs_error = max_controls_abs_diff(c_new, weighted_c_i_new)
        control_scale = max(1.0, max_controls_abs(c_new), max_controls_abs(weighted_c_i_new))
        assert control_invariant_max_abs_error <= CONTROL_INVARIANT_TOL * control_scale, (
            f"{tag} r{rnd}: server control does not match the sample-weighted "
            f"client-control aggregate (max abs diff {control_invariant_max_abs_error}, "
            f"scale {control_scale})"
        )
        server_control = c_new

        # Sample-weighted model aggregation over variable client sizes, identical to
        # script 32; time aggregation + load.
        sync_device(device)
        agg_start = time.perf_counter()
        agg_state = aggregate_sample_weighted(client_states, sizes)
        global_model.load_state_dict(agg_state)
        sync_device(device)
        aggregation_seconds = time.perf_counter() - agg_start
        assert_state_finite(agg_state, f"{tag} r{rnd} aggregated")

        # FL runtime accounting includes the actual control-STATE work on both sides,
        # not just local minibatch training and model aggregation. The per-minibatch
        # gradient correction is already inside round_train_seconds and is not counted
        # again here. Validation, CSV writing, checkpoint writing, provenance hashing,
        # and general orchestration and diagnostics are all excluded.
        round_control_state_update_seconds = round_client_control_seconds + server_control_update_seconds
        fl_round_seconds = (round_train_seconds
                            + round_client_control_seconds
                            + aggregation_seconds
                            + server_control_update_seconds)

        sync_device(device)
        val_start = time.perf_counter()
        val = evaluate(global_model, val_loader, criterion, device)
        sync_device(device)
        validation_seconds = time.perf_counter() - val_start
        round_total_seconds = fl_round_seconds + validation_seconds

        total_fl_seconds += fl_round_seconds
        total_control_state_update_seconds += round_control_state_update_seconds
        total_validation_seconds += validation_seconds
        total_run_seconds += round_total_seconds

        record = {"round": rnd,
                  "server_control_norm_before": server_control_norm_before,
                  "server_control_norm_after": server_control_norm_after,
                  "server_control_delta_norm": server_control_delta_norm,
                  "control_invariant_max_abs_error": control_invariant_max_abs_error,
                  "round_local_steps": int(sum(client_local_steps)),
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
            record[f"control_norm_before_client_{client_id}"] = client_control_norm_before[client_id]
            record[f"control_norm_after_client_{client_id}"] = client_control_norm_after[client_id]
            record[f"control_delta_norm_client_{client_id}"] = client_control_delta_norm[client_id]
            record[f"correction_norm_client_{client_id}"] = client_correction_norm[client_id]
            record[f"train_seconds_client_{client_id}"] = client_seconds[client_id]
            record[f"control_update_seconds_client_{client_id}"] = client_control_seconds[client_id]
        record["round_train_seconds"] = round_train_seconds
        record["round_client_control_seconds"] = round_client_control_seconds
        record["server_control_update_seconds"] = server_control_update_seconds
        record["round_control_state_update_seconds"] = round_control_state_update_seconds
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
              f"fl_s={fl_round_seconds:.1f} ctrl_s={round_control_state_update_seconds:.1f} "
              f"val_s={validation_seconds:.1f}", flush=True)

        # Persist the accumulated history after checkpoint logic, outside timing.
        pd.DataFrame(history).to_csv(hist_path, index=False)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(hist_path, index=False)

    # Save the final-round global model before reloading the best checkpoint.
    torch.save(global_model.state_dict(), final_path)

    # Stored best round must equal the argmax validation macro-F1 round.
    argmax_round = int(hist_df.loc[hist_df["macro_f1"].idxmax(), "round"])
    assert best["round"] == argmax_round, f"{tag}: best_round {best['round']} != argmax {argmax_round}"

    # Reload the saved best checkpoint from disk and confirm it reproduces best macro-F1.
    reloaded_state = torch.load(best_path, map_location=device)
    assert_state_finite(reloaded_state, f"{tag} best checkpoint")
    local_model.load_state_dict(reloaded_state)
    reloaded_macro_f1 = float(evaluate(local_model, val_loader, criterion, device)["macro_f1"])
    assert abs(reloaded_macro_f1 - best["macro_f1"]) < RELOAD_F1_TOL, (
        f"{tag}: reloaded macro-F1 {reloaded_macro_f1} != best {best['macro_f1']}"
    )

    config = {
        "method": METHOD,
        "scaffold_stage": SCAFFOLD_STAGE,
        "control_variates_implemented": CONTROL_VARIATES_IMPLEMENTED,
        "scaffold_execution_enabled": SCAFFOLD_EXECUTION_ENABLED,
        "representation": REPRESENTATION,
        "input_dim": INPUT_DIM,
        "K": NUM_CLIENTS,
        "partition_seed": partition_seed,
        "condition": condition,
        "training_seed": TRAIN_SEED,
        "num_clients": NUM_CLIENTS,
        "client_sizes": sizes,
        "client_class_counts": part_data["client_class_counts"].tolist(),
        "aggregation": "sample_weighted_fedavg",
        "aggregation_weights": agg_weights,
        # Explicit algorithm provenance, stated in the same terms as the audit.
        "gradient_correction": "g - c_i + c",
        "control_update": "SCAFFOLD Option II",
        "tau_source": "actual optimizer.step() count",
        "model_aggregation": "sample weighted by actual client sizes",
        "server_control_aggregation":
            "sample weighted using the same n_k / sum_j n_j weights as the model",
        "client_controls_persistent_across_rounds": True,
        "controls_reset_between_independent_runs": True,
        "control_invariant": "c == sample-weighted aggregate of c_i",
        "control_variate_option": "option_ii",
        "control_update_rule": "c_i_new = c_i_old - c_old + (x - y_i) / (local_steps * lr)",
        "control_correction": "grad <- grad - c_i_old + c_old, applied before optimizer.step()",
        "server_control_rule": "c_new = c_old + sum_k (n_k / sum_j n_j) * delta_c_i",
        "control_scope": "trainable named parameters",
        "control_init": "zeros, reset per (partition_seed, condition) run",
        "control_participation": "full",
        "control_invariant_tolerance": CONTROL_INVARIANT_TOL,
        "control_invariant_handling": "verified and recorded per round; never repaired",
        "local_steps_source": "counted optimizer.step() calls",
        # Meaning of every timing field written to the history CSV.
        "timing_definitions": {
            "train_seconds_client_k":
                "local minibatch training loop, including SCAFFOLD gradient correction, "
                "for client k",
            "control_update_seconds_client_k":
                "Option II client-control state update for client k: y_i snapshot, c_i_new, "
                "delta_c_i",
            "round_train_seconds": "sum over clients of train_seconds_client_k",
            "round_client_control_seconds": "sum over clients of control_update_seconds_client_k",
            "server_control_update_seconds":
                "server-control state delta and update: "
                "c_new = c_old + sum_k (n_k / sum_j n_j) * delta_c_i",
            "round_control_state_update_seconds":
                "round_client_control_seconds + server_control_update_seconds; control-STATE "
                "updates only. Excludes the per-minibatch gradient correction g <- g - c_i + c, "
                "which runs inside the training loop and is already counted in "
                "round_train_seconds",
            "aggregation_seconds": "sample-weighted MODEL aggregation and load into the global model",
            "fl_round_seconds":
                "round_train_seconds + round_client_control_seconds + aggregation_seconds "
                "+ server_control_update_seconds; excludes validation, CSV writing, checkpoint "
                "writing, provenance hashing, and general orchestration/diagnostics",
            "validation_seconds": "validation pass over the full validation set",
            "round_total_seconds": "fl_round_seconds + validation_seconds",
            "round_orchestration_seconds":
                "non-algorithm overhead inside the client loop only: model loading, optimizer "
                "and DataLoader construction, seeding, CPU state copies, control-norm "
                "diagnostics, finiteness assertions; excluded from fl_round_seconds",
            "total_control_state_update_seconds":
                "run-level sum of round_control_state_update_seconds; same scope, control-STATE "
                "updates only",
        },
        "timing_note": "simulation runtime, not communication latency; no algorithm "
                       "behaviour was changed in order to measure it",
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
        "best_checkpoint_path": str(best_path),
        "final_checkpoint_path": str(final_path),
        "partition_path": part_data["part_dir"],
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git_commit": git_commit_hash(),
        "git_dirty": git_dirty(),
        "script_sha256": script_sha256(),
    }
    with open(RESULTS_DIR / f"config_{tag}.json", "w") as f:
        json.dump(config, f, indent=2)

    return {"method": METHOD, "scaffold_stage": SCAFFOLD_STAGE,
            "representation": REPRESENTATION, "input_dim": INPUT_DIM,
            "K": NUM_CLIENTS, "partition_seed": partition_seed, "condition": condition,
            "training_seed": TRAIN_SEED, "batch_size": BATCH_SIZE,
            "lr": LR, "max_rounds": MAX_ROUNDS,
            "best_round": best["round"], "best_val_macro_f1": best["macro_f1"],
            "total_fl_seconds": total_fl_seconds,
            "total_control_state_update_seconds": total_control_state_update_seconds,
            "total_validation_seconds": total_validation_seconds,
            "total_run_seconds": total_run_seconds}


def preflight_outputs() -> None:
    """Fail if any intended SCAFFOLD 37-feature output already exists."""
    intended = [CLASS_WEIGHTS_PATH, RESULTS_DIR / "final_summary.csv"]
    for partition_seed in PARTITION_SEEDS:
        for condition in CONDITIONS:
            tag = f"scaffold37f_k{NUM_CLIENTS}_seed{partition_seed}_{condition}"
            intended += [RESULTS_DIR / f"history_{tag}.csv",
                         RESULTS_DIR / f"config_{tag}.json",
                         MODELS_DIR / f"best_{tag}.pt",
                         MODELS_DIR / f"final_{tag}.pt"]
    existing = [p for p in intended if p.exists()]
    if existing:
        listing = "\n  ".join(str(p) for p in existing)
        raise RuntimeError(f"Refusing to run; SCAFFOLD 37-feature outputs already present:\n  {listing}")


def assert_stage_ready() -> None:
    """Refuse to produce results until the control variates exist.

    Without them this runner is behaviourally FedAvg, and its output would sit in
    scaffold-labelled directories. The control-variate logic is now in place and
    CONTROL_VARIATES_IMPLEMENTED is True, so this gate is satisfied; it is retained
    as a standing guard. Clearance to actually run is a separate decision, held by
    assert_execution_enabled().
    """
    if not CONTROL_VARIATES_IMPLEMENTED:
        raise SystemExit(
            "35_train_final_scaffold_37f.py is at stage '" + SCAFFOLD_STAGE + "': the SCAFFOLD "
            "control variates are not implemented, so this run would only reproduce FedAvg "
            "under a SCAFFOLD label. Set CONTROL_VARIATES_IMPLEMENTED = True once the "
            "control-variate step is in place."
        )


def assert_execution_enabled() -> None:
    """Refuse to start while the SCAFFOLD execution gate is closed.

    Implemented is not the same as cleared to run. This is called before any output
    directory is created and before any class-weight file, history, config, or
    checkpoint is written, so a refused invocation leaves the filesystem untouched.
    """
    if not SCAFFOLD_EXECUTION_ENABLED:
        raise SystemExit(
            "35_train_final_scaffold_37f.py: SCAFFOLD_EXECUTION_ENABLED is False, so this "
            "run refuses to start. The control variates are implemented; execution is "
            "gated separately and deliberately. Nothing has been created or written. Set "
            "SCAFFOLD_EXECUTION_ENABLED = True only when the SCAFFOLD training matrix is "
            "intended to launch."
        )


def main() -> None:
    assert_no_test_reference()
    assert_stage_ready()
    # Execution gate first: nothing below this line runs while it is closed, so no
    # directory is created and no artefact is written by a refused invocation.
    assert_execution_enabled()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preflight_outputs()

    device = get_device()
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    # Label sanity: integer ids in range and every class present.
    assert np.issubdtype(y_train.dtype, np.integer), "y_train labels are not integers"
    assert y_train.min() >= 0 and y_train.max() <= NUM_CLASSES - 1, "y_train labels outside 0..NUM_CLASSES-1"
    global_class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    assert (global_class_counts > 0).all(), "a class has zero training examples"

    # Confirm the corrected 37-feature representation width for train and validation.
    x_train_peek = np.load(PROCESSED_DIR / "X_train.npy", mmap_mode="r")
    x_val_peek = np.load(PROCESSED_DIR / "X_val.npy", mmap_mode="r")
    assert x_train_peek.shape[1] == INPUT_DIM, f"X_train has {x_train_peek.shape[1]} columns, expected {INPUT_DIM}"
    assert x_val_peek.shape[1] == INPUT_DIM, f"X_val has {x_val_peek.shape[1]} columns, expected {INPUT_DIM}"

    # Class weights from the unchanged y_train; the exact float32 values are shared
    # between the loss and the saved file.
    weight_f32 = class_weights_full(y_train).astype(np.float32)
    assert np.isfinite(weight_f32).all(), "class weights contain non-finite values"
    assert (weight_f32 > 0).all(), "class weights must be strictly positive"

    # Verify the 37-feature FedAvg class-weight vector (read-only) against the fresh
    # computation.
    assert FEDAVG_CLASS_WEIGHTS_PATH.exists(), f"37f FedAvg class weights not found: {FEDAVG_CLASS_WEIGHTS_PATH}"
    fedavg_weights = np.load(FEDAVG_CLASS_WEIGHTS_PATH)
    assert fedavg_weights.dtype == np.float32, f"FedAvg class weights dtype {fedavg_weights.dtype}, expected float32"
    assert fedavg_weights.shape == (NUM_CLASSES,), f"FedAvg class weights shape {fedavg_weights.shape}"
    assert np.array_equal(fedavg_weights, weight_f32), (
        "37f FedAvg class weights differ from freshly computed full-y_train float32 weights"
    )
    # Record the verified vector under the SCAFFOLD results root (source untouched).
    np.save(CLASS_WEIGHTS_PATH, weight_f32)

    # Reuse the shared initial state produced by the 37-feature FedAvg run (read-only).
    assert INIT_PATH.exists(), f"Shared 37f initial state not found: {INIT_PATH}"
    init_sha_before = file_sha256(INIT_PATH)
    initial_state = torch.load(INIT_PATH, map_location="cpu")
    assert_state_finite(initial_state, "initial_state")
    # The initial state must have the exact shapes of a 37-input model; this rejects
    # a 41-feature checkpoint before any training begins.
    loaded_shapes = {k: tuple(v.shape) for k, v in initial_state.items()}
    assert loaded_shapes == expected_state_shapes(), (
        f"initial state at {INIT_PATH} does not match a {INPUT_DIM}-input model: {loaded_shapes}"
    )

    val_loader = DataLoader(FullDataset(PROCESSED_DIR / "X_val.npy", PROCESSED_DIR / "y_val.npy"),
                            batch_size=4096, shuffle=False, num_workers=0)

    # Load and verify each partition once (one per seed x condition).
    part_cache = {}
    for partition_seed in PARTITION_SEEDS:
        for condition in CONDITIONS:
            part_cache[(partition_seed, condition)] = load_partition(
                partition_seed, condition, y_train, global_class_counts)

    print(f"method={METHOD} stage={SCAFFOLD_STAGE} representation={REPRESENTATION} "
          f"input_dim={INPUT_DIM} K={NUM_CLIENTS} lr={LR} momentum={MOMENTUM} "
          f"weight_decay={WEIGHT_DECAY} batch_size={BATCH_SIZE} local_epochs={LOCAL_EPOCHS} "
          f"max_rounds={MAX_ROUNDS} seeds={PARTITION_SEEDS} conditions={CONDITIONS} "
          f"processed_dir={PROCESSED_DIR} device={device}", flush=True)

    summary_rows = []
    for partition_seed in PARTITION_SEEDS:
        for condition in CONDITIONS:
            row = run(partition_seed, condition, part_cache[(partition_seed, condition)],
                      initial_state, weight_f32, val_loader, device)
            summary_rows.append(row)
            print(f"[scaffold37f_k{NUM_CLIENTS}_seed{partition_seed}_{condition}] DONE "
                  f"best_round={row['best_round']} best_val_macro_f1={row['best_val_macro_f1']:.4f}\n", flush=True)

    # Confirm the shared initial state was neither mutated in memory nor on disk.
    saved_init = torch.load(INIT_PATH, map_location="cpu")
    assert states_equal(saved_init, initial_state), "initial state changed during training"
    assert file_sha256(INIT_PATH) == init_sha_before, "initial state file was modified during training"

    assert len(summary_rows) == len(PARTITION_SEEDS) * len(CONDITIONS), "unexpected number of runs"
    summary_path = RESULTS_DIR / "final_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
