"""
Dataset-2 (NF-CSE-CIC-IDS2018-v2) K=5 FedAvg training matrix, HD-selected partitions.

Runs on the partitions written by 50_d2_select_partitions_by_hd.py, which uses the
project's own pure class-wise Dirichlet partitioner and selects by measured Hellinger
distance. The FedAvg algorithm is preserved exactly: the same MLP, a K=5 partition
grid over directory labels {1, 2, 3} x conditions
{iid, hd_0p25, hd_0p5, hd_0p75, hd_0p9} = 15 runs,
plain SGD at learning rate 0.1 (momentum 0, weight decay 0), batch size 4096, one
local epoch, up to 40 rounds, full client participation, training/init seed 42 with
the per-round-per-client seed scheme, one shared deterministic initial state copied
into every run, a fresh local model and a fresh optimizer per client per round,
global class-weighted CrossEntropyLoss from the complete y_train, sample-weighted
aggregation over the actual client sizes n_k, the full validation metric set
including per-class PR-AUC, validation Macro-F1 checkpoint selection, and the
best-checkpoint reload verification. Learning rate, batch size, and rounds are
identical for every condition; there is no per-condition or additional tuning.

Differences from the Dataset-1 script are confined to the dataset contract
(36 features, 7 classes, Dataset-2 paths), the isolated Dataset-2 output roots,
CUDA-aware device selection, seeding and synchronisation, and a set of hard
pre-training assertions on the Dataset-2 training state.

On CUDA the train/validation arrays are made resident on the device once at
startup, after those assertions pass, and feature rows are gathered on device.
The DataLoader still owns shuffling: it yields local positions 0..n_k-1 with the
same sampler, generator and batch size as before, so the per-round-per-client
batch order is unchanged; only the row-by-row CPU feature fetch is removed. The
Dataset/DataLoader path is used unchanged on MPS and CPU.

Inputs (read-only):
    data/nf_cse_cic_ids2018_v2/processed/{X_train,y_train,X_val,y_val}.npy
    configs/nf_cse_cic_ids2018_v2/label_mapping.json
    data/nf_cse_cic_ids2018_v2/fl_clients/hd_selected_partitions/k_5/seed_*/<condition>/

Outputs go only under results/nf_cse_cic_ids2018_v2/final_fedavg_k5_hd_selected/
and models/nf_cse_cic_ids2018_v2/final_fedavg_k5_hd_selected/; the script refuses to
start if any intended output already exists.

Reads X_train/y_train (train) and X_val/y_val (validation) only; never reads the
held-out arrays. Checkpoint selection is validation Macro-F1; the selected round
may occur before round 40, which is only the fixed maximum budget. Timings are
simulation runtime, not communication latency.
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

PROCESSED_DIR = Path("data/nf_cse_cic_ids2018_v2/processed")
LABEL_MAP = Path("configs/nf_cse_cic_ids2018_v2/label_mapping.json")
PART_ROOT = Path("data/nf_cse_cic_ids2018_v2/fl_clients/hd_selected_partitions/k_5")
RESULTS_DIR = Path("results/nf_cse_cic_ids2018_v2/final_fedavg_k5_hd_selected")
MODELS_DIR = Path("models/nf_cse_cic_ids2018_v2/final_fedavg_k5_hd_selected")
INIT_PATH = MODELS_DIR / "initial_global_model.pt"
CLASS_WEIGHTS_PATH = RESULTS_DIR / "class_weights.npy"

METHOD = "FedAvg"
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

# Full grid: 3 labels x 5 conditions = 15 runs against the CLASS-WISE DIRICHLET set at
# data/nf_cse_cic_ids2018_v2/fl_clients/hd_selected_partitions/k_5, written by
# 50_d2_select_partitions_by_hd.py. Seeds here are the directory labels seed_1/2/3,
# not the generating seeds; that mapping is in the set's seed_mapping.csv.
PARTITION_SEEDS = [1, 2, 3]
CONDITIONS = ["iid", "hd_0p25", "hd_0p5", "hd_0p75", "hd_0p9"]

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
    """Per-round-per-client accelerator seeding, alongside torch.manual_seed."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    # CUDA if available, else MPS if available, else CPU.
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


def train_one_epoch(model, loader, criterion, optimizer, device) -> dict:
    # Backward loss is unchanged (weighted mean). Diagnostics accumulate as detached
    # on-device tensors; no host sync (.item()/.cpu()/bool) happens inside the batch
    # loop. Timing covers only the training batch loop, not the final conversion.
    model.train()
    loss_num = torch.zeros((), device=device)
    loss_den = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    finite = torch.ones((), dtype=torch.bool, device=device)
    n_samples, n_batches = 0, 0

    sync_device(device)
    train_start = time.perf_counter()
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
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
    # Single host transfer of the accumulated diagnostics, after the epoch.
    assert bool(finite.item()), "non-finite training loss"
    return {
        "weighted_train_loss": float((loss_num / loss_den).item()),
        "online_train_accuracy": float((correct / n_samples).item()),
        "n_samples": n_samples,
        "n_batches": n_batches,
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
    # client_indices are the verified arrays as saved; the CUDA path moves these
    # same arrays to the device without reordering or reselecting.
    return {"datasets": datasets, "client_indices": client_indices, "sizes": sizes,
            "client_class_counts": counts, "part_dir": str(part_dir)}


def run(partition_seed, condition, part_data, initial_state,
        weight_f32, val_loader, device, resident=None) -> dict:
    tag = f"fedavg_k{NUM_CLIENTS}_seed{partition_seed}_{condition}"
    datasets = part_data["datasets"]
    sizes = part_data["sizes"]
    total_size = float(sum(sizes))
    agg_weights = [n / total_size for n in sizes]

    # CUDA path: this partition's five saved client index arrays become resident
    # int64 tensors once per run. Same membership, same order, same sizes.
    client_indices_cuda = None
    if resident is not None:
        client_indices_cuda = [torch.from_numpy(ci.astype(np.int64)).to(device)
                               for ci in part_data["client_indices"]]
        assert len(client_indices_cuda) == NUM_CLIENTS, f"{tag}: expected {NUM_CLIENTS} client index tensors"
        for k, ci in enumerate(client_indices_cuda):
            assert int(ci.numel()) == sizes[k], f"{tag}: client {k} index count changed on device"

    # Start every run from an identical copy of the shared initial state.
    set_all_seeds(TRAIN_SEED)
    global_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    global_model.load_state_dict(copy.deepcopy(initial_state))
    assert states_equal(global_model.state_dict(), initial_state), f"{tag}: initial state not loaded identically"
    local_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight_f32, dtype=torch.float32, device=device))

    best = {"round": -1, "macro_f1": -1.0}
    best_path = MODELS_DIR / f"best_{tag}.pt"
    final_path = MODELS_DIR / f"final_{tag}.pt"
    hist_path = RESULTS_DIR / f"history_{tag}.csv"
    history = []
    total_fl_seconds, total_validation_seconds, total_run_seconds = 0.0, 0.0, 0.0

    for rnd in range(1, MAX_ROUNDS + 1):
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        client_states, participated = [], []
        client_stats, client_seconds, client_update_l2 = [], [], []

        loop_wall_start = time.perf_counter()
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
                # Same DataLoader arguments and same dataset length n_k, so the
                # sampler draws the same permutation from the same generator; it
                # just yields local positions instead of feature rows.
                position_loader = DataLoader(LocalPositionDataset(sizes[client_id]),
                                             batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=0, generator=generator)
                loader = ResidentClientBatches(position_loader, client_indices_cuda[client_id],
                                               resident["x_train"], resident["y_train"], device)
            else:
                loader = DataLoader(datasets[client_id], batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=0, generator=generator)

            # train_seconds covers only the training batch loop (per train_one_epoch).
            epoch_seconds = 0.0
            for _ in range(LOCAL_EPOCHS):
                stats = train_one_epoch(local_model, loader, criterion, optimizer, device)
                epoch_seconds += stats["train_seconds"]
            client_seconds.append(epoch_seconds)

            client_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
            assert_state_finite(client_state, f"{tag} r{rnd} client{client_id}")
            client_states.append(client_state)
            client_stats.append(stats)
            client_update_l2.append(state_l2_distance(client_state, global_state))
            participated.append(client_id)
        loop_wall_seconds = time.perf_counter() - loop_wall_start

        assert sorted(participated) == list(range(NUM_CLIENTS)), f"{tag} round {rnd}: participation {participated}"

        # FL training time excludes diagnostic/orchestration overhead.
        round_train_seconds = float(sum(client_seconds))
        round_orchestration_seconds = loop_wall_seconds - round_train_seconds

        # Sample-weighted FedAvg over variable client sizes; time aggregation + load.
        sync_device(device)
        agg_start = time.perf_counter()
        agg_state = aggregate_sample_weighted(client_states, sizes)
        global_model.load_state_dict(agg_state)
        sync_device(device)
        aggregation_seconds = time.perf_counter() - agg_start
        assert_state_finite(agg_state, f"{tag} r{rnd} aggregated")
        fl_round_seconds = round_train_seconds + aggregation_seconds

        sync_device(device)
        val_start = time.perf_counter()
        val = evaluate(global_model, val_loader, criterion, device)
        sync_device(device)
        validation_seconds = time.perf_counter() - val_start
        round_total_seconds = fl_round_seconds + validation_seconds

        total_fl_seconds += fl_round_seconds
        total_validation_seconds += validation_seconds
        total_run_seconds += round_total_seconds

        record = {"round": rnd, "val_loss": val["val_loss"], "accuracy": val["accuracy"],
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
            record[f"train_seconds_client_{client_id}"] = client_seconds[client_id]
        record["round_train_seconds"] = round_train_seconds
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
        "dataset": DATASET,
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
        "K": NUM_CLIENTS,
        "partition_seed": partition_seed,
        "condition": condition,
        "training_seed": TRAIN_SEED,
        "num_clients": NUM_CLIENTS,
        "client_sizes": sizes,
        "client_class_counts": part_data["client_class_counts"].tolist(),
        "aggregation": "sample_weighted_fedavg",
        "aggregation_weights": agg_weights,
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
        "selection_metric": "val_macro_f1",
        "processed_dir": str(PROCESSED_DIR),
        "partition_root": str(PART_ROOT),
        "initial_state_path": str(INIT_PATH),
        "best_checkpoint_path": str(best_path),
        "final_checkpoint_path": str(final_path),
        "partition_path": part_data["part_dir"],
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git_commit": git_commit_hash(),
        "git_dirty": git_dirty(),
        "script_sha256": script_sha256(),
    }
    with open(RESULTS_DIR / f"config_{tag}.json", "w") as f:
        json.dump(config, f, indent=2)

    return {"method": METHOD, "dataset": DATASET, "input_dim": INPUT_DIM,
            "K": NUM_CLIENTS,
            "partition_seed": partition_seed, "condition": condition,
            "training_seed": TRAIN_SEED, "batch_size": BATCH_SIZE,
            "lr": LR, "max_rounds": MAX_ROUNDS,
            "best_round": best["round"], "best_val_macro_f1": best["macro_f1"],
            "total_fl_seconds": total_fl_seconds,
            "total_validation_seconds": total_validation_seconds,
            "total_run_seconds": total_run_seconds}


def preflight_outputs() -> None:
    """Fail if any intended Dataset-2 FedAvg output already exists."""
    intended = [INIT_PATH, CLASS_WEIGHTS_PATH, RESULTS_DIR / "final_summary.csv"]
    for partition_seed in PARTITION_SEEDS:
        for condition in CONDITIONS:
            tag = f"fedavg_k{NUM_CLIENTS}_seed{partition_seed}_{condition}"
            intended += [RESULTS_DIR / f"history_{tag}.csv",
                         RESULTS_DIR / f"config_{tag}.json",
                         MODELS_DIR / f"best_{tag}.pt",
                         MODELS_DIR / f"final_{tag}.pt"]
    existing = [p for p in intended if p.exists()]
    if existing:
        listing = "\n  ".join(str(p) for p in existing)
        raise RuntimeError(f"Refusing to run; Dataset-2 FedAvg outputs already present:\n  {listing}")


def main() -> None:
    assert_no_test_reference()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preflight_outputs()

    device = get_device()
    class_names = load_class_names()

    # Exact Dataset-2 class order.
    assert class_names == EXPECTED_CLASS_ORDER, (
        f"label mapping class order {class_names} does not match the expected "
        f"Dataset-2 order {EXPECTED_CLASS_ORDER}"
    )
    assert len(class_names) == NUM_CLASSES, f"expected {NUM_CLASSES} classes, found {len(class_names)}"

    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")

    # Label sanity: integer ids in range and every class present.
    assert np.issubdtype(y_train.dtype, np.integer), "y_train labels are not integers"
    assert np.issubdtype(y_val.dtype, np.integer), "y_val labels are not integers"
    assert y_train.min() >= 0 and y_train.max() <= NUM_CLASSES - 1, "y_train labels outside 0..NUM_CLASSES-1"
    assert y_val.min() >= 0 and y_val.max() <= NUM_CLASSES - 1, "y_val labels outside 0..NUM_CLASSES-1"

    # Exact label ids, in both splits.
    expected_ids = list(range(NUM_CLASSES))
    assert sorted(int(v) for v in np.unique(y_train).tolist()) == expected_ids, \
        "y_train label ids do not equal the expected Dataset-2 ids"
    assert sorted(int(v) for v in np.unique(y_val).tolist()) == expected_ids, \
        "y_val label ids do not equal the expected Dataset-2 ids"

    global_class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    val_class_counts = np.bincount(y_val, minlength=NUM_CLASSES)
    assert (global_class_counts > 0).all(), "a class has zero training examples"
    assert (val_class_counts > 0).all(), "a class has zero validation examples"

    # Exact Dataset-2 row counts.
    assert len(y_train) == EXPECTED_TRAIN_ROWS, \
        f"y_train has {len(y_train)} rows, expected {EXPECTED_TRAIN_ROWS}"
    assert len(y_val) == EXPECTED_VAL_ROWS, \
        f"y_val has {len(y_val)} rows, expected {EXPECTED_VAL_ROWS}"

    # Confirm the Dataset-2 representation width and row counts for train and validation.
    x_train_peek = np.load(PROCESSED_DIR / "X_train.npy", mmap_mode="r")
    x_val_peek = np.load(PROCESSED_DIR / "X_val.npy", mmap_mode="r")
    assert x_train_peek.shape[1] == INPUT_DIM, f"X_train has {x_train_peek.shape[1]} columns, expected {INPUT_DIM}"
    assert x_val_peek.shape[1] == INPUT_DIM, f"X_val has {x_val_peek.shape[1]} columns, expected {INPUT_DIM}"
    assert x_train_peek.shape[0] == EXPECTED_TRAIN_ROWS, \
        f"X_train has {x_train_peek.shape[0]} rows, expected {EXPECTED_TRAIN_ROWS}"
    assert x_val_peek.shape[0] == EXPECTED_VAL_ROWS, \
        f"X_val has {x_val_peek.shape[0]} rows, expected {EXPECTED_VAL_ROWS}"

    # Class weights from the unchanged y_train; the exact float32 values are shared
    # between the loss and the saved file.
    weight_f32 = class_weights_full(y_train).astype(np.float32)
    assert np.isfinite(weight_f32).all(), "class weights contain non-finite values"
    assert (weight_f32 > 0).all(), "class weights must be strictly positive"
    np.save(CLASS_WEIGHTS_PATH, weight_f32)

    # Build and save one deterministic initial state; all runs start from a copy.
    set_all_seeds(TRAIN_SEED)
    init_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    initial_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
    assert_state_finite(initial_state, "initial_state")
    torch.save(initial_state, INIT_PATH)

    # CUDA: the row-by-row CPU feature fetch is the bottleneck, and the Dataset-2
    # train/validation arrays fit comfortably in device memory, so make them
    # resident once here - after every assertion above has passed.
    resident = None
    if device.type == "cuda":
        resident = {
            "x_train": torch.from_numpy(np.load(PROCESSED_DIR / "X_train.npy")).to(torch.float32).to(device),
            "y_train": torch.from_numpy(np.load(PROCESSED_DIR / "y_train.npy")).to(torch.long).to(device),
            "x_val": torch.from_numpy(np.load(PROCESSED_DIR / "X_val.npy")).to(torch.float32).to(device),
            "y_val": torch.from_numpy(np.load(PROCESSED_DIR / "y_val.npy")).to(torch.long).to(device),
        }
        assert tuple(resident["x_train"].shape) == (EXPECTED_TRAIN_ROWS, INPUT_DIM), \
            f"resident X_train shape {tuple(resident['x_train'].shape)}"
        assert tuple(resident["x_val"].shape) == (EXPECTED_VAL_ROWS, INPUT_DIM), \
            f"resident X_val shape {tuple(resident['x_val'].shape)}"
        assert int(resident["y_train"].numel()) == EXPECTED_TRAIN_ROWS, "resident y_train row count"
        assert int(resident["y_val"].numel()) == EXPECTED_VAL_ROWS, "resident y_val row count"
        assert resident["x_train"].dtype == torch.float32 and resident["x_val"].dtype == torch.float32, \
            "resident features are not float32"
        assert resident["y_train"].dtype == torch.long and resident["y_val"].dtype == torch.long, \
            "resident labels are not int64"
        val_loader = ResidentValLoader(resident["x_val"], resident["y_val"], 4096)
        print(f"resident on {device}: X_train{tuple(resident['x_train'].shape)} "
              f"X_val{tuple(resident['x_val'].shape)}", flush=True)
    else:
        val_loader = DataLoader(FullDataset(PROCESSED_DIR / "X_val.npy", PROCESSED_DIR / "y_val.npy"),
                                batch_size=4096, shuffle=False, num_workers=0)

    # Load and verify each partition once (one per seed x condition).
    part_cache = {}
    for partition_seed in PARTITION_SEEDS:
        for condition in CONDITIONS:
            part_cache[(partition_seed, condition)] = load_partition(
                partition_seed, condition, y_train, global_class_counts)

    print(f"method={METHOD} dataset={DATASET} input_dim={INPUT_DIM} classes={NUM_CLASSES} "
          f"K={NUM_CLIENTS} lr={LR} momentum={MOMENTUM} weight_decay={WEIGHT_DECAY} "
          f"batch_size={BATCH_SIZE} local_epochs={LOCAL_EPOCHS} max_rounds={MAX_ROUNDS} "
          f"seeds={PARTITION_SEEDS} conditions={CONDITIONS} train_rows={len(y_train)} "
          f"val_rows={len(y_val)} processed_dir={PROCESSED_DIR} device={device}", flush=True)

    summary_rows = []
    for partition_seed in PARTITION_SEEDS:
        for condition in CONDITIONS:
            row = run(partition_seed, condition, part_cache[(partition_seed, condition)],
                      initial_state, weight_f32, val_loader, device, resident)
            summary_rows.append(row)
            print(f"[fedavg_k{NUM_CLIENTS}_seed{partition_seed}_{condition}] DONE "
                  f"best_round={row['best_round']} best_val_macro_f1={row['best_val_macro_f1']:.4f}\n", flush=True)

    # Confirm the saved initial state was not mutated by any run.
    saved_init = torch.load(INIT_PATH, map_location="cpu")
    assert states_equal(saved_init, initial_state), "initial state changed during training"

    assert len(summary_rows) == len(PARTITION_SEEDS) * len(CONDITIONS), "unexpected number of runs"
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "final_summary.csv", index=False)
    print(f"Wrote summary: {RESULTS_DIR / 'final_summary.csv'}")


if __name__ == "__main__":
    main()
