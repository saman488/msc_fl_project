"""
FedAvg + plain SGD pilot on the final frozen K=5 partitions.

Fresh configuration-checking pilot that uses the final variable-size client
partitions and sample-weighted aggregation. Client sizes are variable and are NOT
assumed equal. Aggregation is sample-weighted FedAvg (weight n_k / sum_j n_j), not
simple averaging.

Two 40-round runs: seed 42 IID and seed 44 alpha_0p1, both with batch size 4096
and SGD learning rate 0.1. All runs
use one fixed training seed and start from one identical initial model state built
and saved here. Reads X_train/y_train (train) and X_val/y_val (validation) only;
never reads the test arrays. Timings are simulation runtime, not communication
latency.
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

PROCESSED_DIR = Path("data/processed")
LABEL_MAP = Path("configs/label_mapping.json")
PART_ROOT = Path("data/fl_clients/final_partitions/k_5")
RESULTS_DIR = Path("results/final_fedavg_sgd_pilot_lr0p1_bs4096_round40_check")
MODELS_DIR = Path("models/final_fedavg_sgd_pilot_lr0p1_bs4096_round40_check")
INIT_PATH = MODELS_DIR / "initial_global_model.pt"
CLASS_WEIGHTS_PATH = RESULTS_DIR / "class_weights.npy"

INPUT_DIM = 41
NUM_CLASSES = 10
NUM_CLIENTS = 5
LOCAL_EPOCHS = 1
MAX_ROUNDS = 40
TRAIN_SEED = 42
LR = 0.1
MOMENTUM = 0.0
WEIGHT_DECAY = 0.0
RELOAD_F1_TOL = 1e-4

BATCH_SIZES = [4096]
PILOT_CONDITIONS = [(42, "iid"), (44, "alpha_0p1")]


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
    """Confirm the script never references a test-array filename."""
    tok_x = "X_" + "test"
    tok_y = "y_" + "test"
    source = Path(__file__).read_text()
    assert tok_x not in source and tok_y not in source, "Test-array reference found in script."


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
    return {"datasets": datasets, "sizes": sizes, "client_class_counts": counts,
            "part_dir": str(part_dir)}


def run(partition_seed, condition, batch_size, part_data, initial_state,
        weight_f32, val_loader, device) -> dict:
    tag = f"seed{partition_seed}_{condition}_bs{batch_size}"
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
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(local_seed)
            generator = torch.Generator()
            generator.manual_seed(local_seed)
            loader = DataLoader(datasets[client_id], batch_size=batch_size,
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
        "partition_seed": partition_seed,
        "condition": condition,
        "training_seed": TRAIN_SEED,
        "num_clients": NUM_CLIENTS,
        "client_sizes": sizes,
        "client_class_counts": part_data["client_class_counts"].tolist(),
        "aggregation": "sample_weighted_fedavg",
        "aggregation_weights": agg_weights,
        "batch_size": batch_size,
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
        "initial_state_path": str(INIT_PATH),
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

    return {"partition_seed": partition_seed, "condition": condition,
            "batch_size": batch_size, "best_round": best["round"],
            "best_val_macro_f1": best["macro_f1"],
            "total_fl_seconds": total_fl_seconds,
            "total_validation_seconds": total_validation_seconds,
            "total_run_seconds": total_run_seconds}


def preflight_outputs() -> None:
    """Fail if any intended pilot output already exists."""
    intended = [INIT_PATH, CLASS_WEIGHTS_PATH, RESULTS_DIR / "pilot_summary.csv"]
    for partition_seed, condition in PILOT_CONDITIONS:
        for batch_size in BATCH_SIZES:
            tag = f"seed{partition_seed}_{condition}_bs{batch_size}"
            intended += [RESULTS_DIR / f"history_{tag}.csv",
                         RESULTS_DIR / f"config_{tag}.json",
                         MODELS_DIR / f"best_{tag}.pt",
                         MODELS_DIR / f"final_{tag}.pt"]
    existing = [p for p in intended if p.exists()]
    if existing:
        listing = "\n  ".join(str(p) for p in existing)
        raise RuntimeError(f"Refusing to run; pilot outputs already present:\n  {listing}")


def main() -> None:
    assert_no_test_reference()
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

    # Class weights from full y_train; the exact float32 values are shared between
    # the loss and the saved file.
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

    val_loader = DataLoader(FullDataset(PROCESSED_DIR / "X_val.npy", PROCESSED_DIR / "y_val.npy"),
                            batch_size=4096, shuffle=False, num_workers=0)

    # Load and verify each unique partition once.
    part_cache = {}
    for partition_seed, condition in PILOT_CONDITIONS:
        part_cache[(partition_seed, condition)] = load_partition(
            partition_seed, condition, y_train, global_class_counts)

    summary_rows = []
    for partition_seed, condition in PILOT_CONDITIONS:
        for batch_size in BATCH_SIZES:
            row = run(partition_seed, condition, batch_size, part_cache[(partition_seed, condition)],
                      initial_state, weight_f32, val_loader, device)
            summary_rows.append(row)
            print(f"[seed{partition_seed}_{condition}_bs{batch_size}] DONE "
                  f"best_round={row['best_round']} best_val_macro_f1={row['best_val_macro_f1']:.4f}\n", flush=True)

    # Confirm the saved initial state was not mutated by any run.
    saved_init = torch.load(INIT_PATH, map_location="cpu")
    assert states_equal(saved_init, initial_state), "initial state changed during the pilot"

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "pilot_summary.csv", index=False)
    print(f"Wrote summary: {RESULTS_DIR / 'pilot_summary.csv'}")


if __name__ == "__main__":
    main()
