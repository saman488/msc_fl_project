"""
FedProx with plain SGD — separate extension of the verified
13_select_fedavg_sgd_config.py.

Everything preserved from 13: model, controlled partitions, balanced loss,
five-client full participation, one local epoch, batch size 4096, shared initial
model state, DataLoader seed scheme, SGD optimiser arguments, simple-average
aggregation, validation evaluation, macro-F1 checkpoint rule, prediction /
probability saving, and test-set separation.

Added: the FedProx proximal term. A non-negative --mu argument controls its
strength. At the start of every client update a detached dictionary of the
server model's named trainable parameters is captured and held fixed for that
client's local epoch. Per minibatch:

    task_loss        = criterion(logits, targets)
    proximal_norm    = sum((local_param - global_reference[name]).pow(2).sum()
                           for name, local_param in model.named_parameters()
                           if local_param.requires_grad)
    proximal_penalty = 0.5 * mu * proximal_norm
    total_loss       = task_loss + proximal_penalty

total_loss is backpropagated. Server aggregation is unchanged (simple average).
Outputs go ONLY under results/fl_fedprox_sgd/ and models/fl_fedprox_sgd/.
"""

from pathlib import Path
import argparse
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROCESSED_DIR = Path("data/processed")
PART_ROOT = Path("data/fl_clients/controlled_partitions")
# Shared initial global model produced by 11 (read-only reuse; never overwritten).
AUDITED_INIT = Path("models/fl_noniid_controlled/initial_global_model.pt")
RESULTS_DIR = Path("results/fl_fedprox_sgd")
MODELS_DIR = Path("models/fl_fedprox_sgd")

INPUT_DIM = 41
NUM_CLASSES = 10
NUM_CLIENTS = 5
CLIENT_SIZE = 50_000
VAL_SIZE = 358_542
BATCH_SIZE = 4096
LOCAL_EPOCHS = 1
BASE_SEED = 42


# ----------------------------- reused from 13 (unchanged) ------------------- #
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


def load_class_names() -> list[str]:
    with open(Path("configs/label_mapping.json")) as f:
        name_to_id = json.load(f)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    return [id_to_name[c] for c in range(len(id_to_name))]


def aggregate_simple_average(states: list[dict]) -> dict:
    avg = {}
    for key in states[0]:
        avg[key] = torch.stack([s[key].to(torch.float32) for s in states], dim=0).mean(dim=0)
    return avg


# ----------------------------- FedProx additions ---------------------------- #
def proximal_norm(model: nn.Module, global_reference: dict) -> torch.Tensor:
    """Sum of squared L2 distances between local trainable params and the fixed
    detached server reference: sum_i ||w_i - w_i^server||^2."""
    return sum(
        (local_param - global_reference[name]).pow(2).sum()
        for name, local_param in model.named_parameters()
        if local_param.requires_grad
    )


def train_one_epoch(model, loader, criterion, optimizer, device, global_reference, mu):
    """One local epoch with the FedProx proximal term. Returns per-sample means of
    (task_loss, proximal_penalty, total_loss) plus (n_samples, n_batches, steps)."""
    model.train()
    task_sum, prox_sum, total_sum = 0.0, 0.0, 0.0
    n_samples, n_batches = 0, 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        task_loss = criterion(logits, labels)
        proximal_penalty = 0.5 * mu * proximal_norm(model, global_reference)
        total_loss = task_loss + proximal_penalty
        total_loss.backward()
        optimizer.step()
        bs = labels.size(0)
        task_sum += task_loss.item() * bs
        prox_sum += float(proximal_penalty) * bs
        total_sum += total_loss.item() * bs
        n_samples += bs
        n_batches += 1
    return (task_sum / n_samples, prox_sum / n_samples, total_sum / n_samples,
            n_samples, n_batches, n_batches)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    loss_sum, n = 0.0, 0
    preds, targets, probs = [], [], []
    for features, labels in loader:
        logits = model(features.to(device))
        loss = criterion(logits, labels.to(device))
        loss_sum += loss.item() * labels.size(0)
        n += labels.size(0)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        targets.append(labels.numpy())
    y_prob = np.concatenate(probs)
    y_pred = np.concatenate(preds).astype(int)
    y_true = np.concatenate(targets).astype(int)
    labels_range = list(range(NUM_CLASSES))

    per_p = precision_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    per_r = recall_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    per_f = f1_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    support = np.bincount(y_true, minlength=NUM_CLASSES)
    predicted_count = np.bincount(y_pred, minlength=NUM_CLASSES)
    pr_auc = np.full(NUM_CLASSES, np.nan)
    for c in labels_range:
        if support[c] > 0:
            pr_auc[c] = average_precision_score((y_true == c).astype(int), y_prob[:, c])
    supported = [c for c in labels_range if support[c] > 0]
    return {
        "val_loss": loss_sum / n,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_pr_auc": float(np.nanmean(pr_auc)),
        "worst_class_recall": float(min(per_r[c] for c in supported)),
        "worst_class_f1": float(min(per_f[c] for c in supported)),
        "per_precision": per_p, "per_recall": per_r, "per_f1": per_f,
        "support": support, "predicted_count": predicted_count, "pr_auc": pr_auc,
    }


def balanced_weights(dpool_counts: np.ndarray) -> np.ndarray:
    n_pool = int(dpool_counts.sum())
    return n_pool / (NUM_CLASSES * dpool_counts.astype(np.float64))


@torch.no_grad()
def predict_full(model, loader, device):
    model.eval()
    preds, targets, probs = [], [], []
    for features, labels in loader:
        logits = model(features.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        targets.append(labels.numpy())
    return (np.concatenate(targets).astype(int),
            np.concatenate(preds).astype(int),
            np.concatenate(probs).astype(np.float32))


def assert_no_test_reference() -> None:
    tok_x = "X_" + "test"
    tok_y = "y_" + "test"
    source = Path(__file__).read_text()
    assert tok_x not in source and tok_y not in source, "Test-array reference found in script."


def states_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def run(seed, partition, args, initial_state, val_loader, class_names, device):
    tag = args.tag
    mu = args.mu
    part_dir = PART_ROOT / f"seed_{seed}" / partition
    client_indices = [np.load(part_dir / f"client_{k:02d}_indices.npy") for k in range(NUM_CLIENTS)]

    # Assertions: five clients, each with 50,000 samples.
    assert len(client_indices) == NUM_CLIENTS, f"Expected {NUM_CLIENTS} clients, got {len(client_indices)}."
    for k, idx in enumerate(client_indices):
        assert len(idx) == CLIENT_SIZE, f"Client {k} has {len(idx)} samples, expected {CLIENT_SIZE}."

    client_datasets = [
        IndexedDataset(PROCESSED_DIR / "X_train.npy", PROCESSED_DIR / "y_train.npy", idx)
        for idx in client_indices
    ]

    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    dpool_counts = np.bincount(y_train[np.concatenate(client_indices)], minlength=NUM_CLASSES).astype(np.int64)
    weight_vec = balanced_weights(dpool_counts)

    set_all_seeds(BASE_SEED)
    global_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    global_model.load_state_dict(copy.deepcopy(initial_state))
    assert states_equal(global_model.state_dict(), initial_state), "Initial state not loaded correctly."
    local_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    set_all_seeds(BASE_SEED)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight_vec, dtype=torch.float32, device=device))

    best = {"macro_f1": -1.0}
    best_path = MODELS_DIR / f"{tag}_best_seed{seed}_{partition}.pt"
    history = []

    for rnd in range(1, args.rounds + 1):
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        client_states, participated = [], []
        round_task, round_prox, round_total = [], [], []

        for client_id in range(NUM_CLIENTS):
            local_model.load_state_dict(global_state)
            # Fixed detached reference to the server model's trainable parameters,
            # captured at the start of this client's update and held for its epoch.
            global_reference = {
                name: p.detach().clone()
                for name, p in local_model.named_parameters()
                if p.requires_grad
            }
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
            )
            generator = torch.Generator()
            generator.manual_seed(BASE_SEED + rnd * 100 + client_id)
            loader = DataLoader(client_datasets[client_id], batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=0, generator=generator)
            task_l, prox_l, total_l = 0.0, 0.0, 0.0
            for _ in range(LOCAL_EPOCHS):
                task_l, prox_l, total_l, _, _, _ = train_one_epoch(
                    local_model, loader, criterion, optimizer, device, global_reference, mu)
            client_states.append({k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()})
            participated.append(client_id)
            round_task.append(task_l)
            round_prox.append(prox_l)
            round_total.append(total_l)

        assert sorted(participated) == list(range(NUM_CLIENTS)), f"Round {rnd} participation: {participated}"
        assert len(client_states) == NUM_CLIENTS

        # Server aggregation unchanged (simple average).
        global_model.load_state_dict(aggregate_simple_average(client_states))
        val = evaluate(global_model, val_loader, criterion, device)

        record = {"round": rnd,
                  "mean_task_loss": float(np.mean(round_task)),
                  "mean_proximal_penalty": float(np.mean(round_prox)),
                  "mean_total_loss": float(np.mean(round_total)),
                  "val_loss": val["val_loss"], "accuracy": val["accuracy"],
                  "balanced_accuracy": val["balanced_accuracy"],
                  "macro_precision": val["macro_precision"], "macro_recall": val["macro_recall"],
                  "macro_f1": val["macro_f1"], "weighted_precision": val["weighted_precision"],
                  "weighted_recall": val["weighted_recall"], "weighted_f1": val["weighted_f1"],
                  "macro_pr_auc": val["macro_pr_auc"],
                  "worst_class_recall": val["worst_class_recall"], "worst_class_f1": val["worst_class_f1"]}
        for c in range(NUM_CLASSES):
            record[f"precision_c{c}"] = float(val["per_precision"][c])
            record[f"recall_c{c}"] = float(val["per_recall"][c])
            record[f"f1_c{c}"] = float(val["per_f1"][c])
            record[f"support_c{c}"] = int(val["support"][c])
            record[f"predicted_count_c{c}"] = int(val["predicted_count"][c])
            record[f"pr_auc_c{c}"] = float(val["pr_auc"][c])
        history.append(record)

        if val["macro_f1"] > best["macro_f1"]:
            best = {"round": rnd, **val}
            torch.save(global_model.state_dict(), best_path)

        print(f"[{tag}] mu={mu} seed={seed} {partition} round={rnd:02d} "
              f"task={record['mean_task_loss']:.4f} prox={record['mean_proximal_penalty']:.6f} "
              f"val_macro_f1={val['macro_f1']:.4f}", flush=True)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(RESULTS_DIR / f"{tag}_history_seed{seed}_{partition}.csv", index=False)

    argmax_round = int(hist_df.loc[hist_df["macro_f1"].idxmax(), "round"])
    assert best["round"] == argmax_round, f"best_round {best['round']} != argmax {argmax_round}"

    global_model.load_state_dict(torch.load(best_path, map_location=device))
    y_true, y_pred, y_prob = predict_full(global_model, val_loader, device)
    assert y_prob.shape == (VAL_SIZE, NUM_CLASSES), f"probs shape {y_prob.shape}"

    np.save(RESULTS_DIR / f"{tag}_val_true_seed{seed}_{partition}.npy", y_true)
    np.save(RESULTS_DIR / f"{tag}_val_pred_seed{seed}_{partition}.npy", y_pred)
    np.save(RESULTS_DIR / f"{tag}_val_probs_seed{seed}_{partition}.npy", y_prob)

    labels_range = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        RESULTS_DIR / f"{tag}_confusion_seed{seed}_{partition}.csv")
    rep = classification_report(y_true, y_pred, labels=labels_range, target_names=class_names,
                                zero_division=0, output_dict=True)
    pd.DataFrame(rep).transpose().to_csv(RESULTS_DIR / f"{tag}_report_seed{seed}_{partition}.csv")

    config = {
        "tag": tag, "algorithm": "fedprox", "mu": mu,
        "note": "smoke_test - not a tuning result" if tag == "smoke_test" else "fedprox run",
        "optimizer": "SGD", "learning_rate": args.lr, "momentum": args.momentum,
        "weight_decay": args.weight_decay, "rounds": args.rounds,
        "seed": seed, "partition": partition,
        "batch_size": BATCH_SIZE, "local_epochs": LOCAL_EPOCHS, "num_clients": NUM_CLIENTS,
        "aggregation": "simple_average", "selection_metric": "val_macro_f1",
        "base_seed": BASE_SEED, "initial_state_path": str(AUDITED_INIT),
        "dpool_counts": dpool_counts.tolist(), "class_weights": weight_vec.tolist(),
        "best_round": best["round"], "best_val_macro_f1": float(best["macro_f1"]),
    }
    with open(RESULTS_DIR / f"{tag}_config_seed{seed}_{partition}.json", "w") as f:
        json.dump(config, f, indent=2)

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="FedProx + plain SGD.")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--partitions", type=str, nargs="+", default=["iid"])
    parser.add_argument("--tag", type=str, default="run")
    args = parser.parse_args()

    assert args.rounds >= 1, "--rounds must be >= 1"
    assert args.mu >= 0.0, "--mu must be non-negative"
    assert_no_test_reference()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    class_names = load_class_names()

    assert AUDITED_INIT.exists(), f"Shared initial state not found: {AUDITED_INIT}"
    initial_state = torch.load(AUDITED_INIT, map_location="cpu")

    val_loader = DataLoader(
        FullDataset(PROCESSED_DIR / "X_val.npy", PROCESSED_DIR / "y_val.npy"),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"algorithm=FedProx optimizer=SGD mu={args.mu} lr={args.lr} momentum={args.momentum} "
          f"weight_decay={args.weight_decay} rounds={args.rounds} seeds={args.seeds} "
          f"partitions={args.partitions} tag={args.tag}", flush=True)

    for seed in args.seeds:
        for partition in args.partitions:
            best = run(seed, partition, args, initial_state, val_loader, class_names, device)
            print(f"[{args.tag}] DONE seed={seed} {partition} best_round={best['round']} "
                  f"best_val_macro_f1={best['macro_f1']:.4f}\n", flush=True)


if __name__ == "__main__":
    main()
