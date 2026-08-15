"""
Main controlled non-IID FedAvg pilot with exhaustive diagnostics.

Grid: seeds [42, 43, 44] x partitions [alpha_0.01, alpha_0.05, alpha_0.1,
alpha_0.5, alpha_1.0, alpha_5.0, iid] from data/fl_clients/controlled_partitions/.

Frozen config: MLPMultiClassClassifier, AdamW, lr=1e-3, weight_decay=1e-5,
batch_size=4096, E=1, 30 rounds, simple-average FedAvg (equal 50k clients),
`balanced` class weighting W_c = N_pool / (10 * count_c) from each seed's D_pool.

Reproducibility: one shared initial global model reused by every run; all seeds
reset per run; deterministic client DataLoader generators seeded with
base_seed + round*100 + client_id. Evaluation and selection use ONLY the global
validation set; the test set is never touched. Nothing overwrites prior outputs.
"""

from pathlib import Path
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
CONTROLLED_METRICS = Path("results/fl_partitions/controlled_metrics.csv")
RESULTS_DIR = Path("results/fl_noniid_controlled")
MODELS_DIR = Path("models/fl_noniid_controlled")

INPUT_DIM = 41
NUM_CLASSES = 10
NUM_CLIENTS = 5
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LOCAL_EPOCHS = 1
NUM_ROUNDS = 30
BASE_SEED = 42

SEEDS = [42, 43, 44]
PARTITIONS = ["alpha_0.01", "alpha_0.05", "alpha_0.1", "alpha_0.5", "alpha_1.0", "alpha_5.0", "iid"]
HIGH_RISK = {"Analysis": 1, "Backdoor": 2, "Exploits": 4, "Shellcode": 8, "Worms": 9}


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
    with open(Path("configs/label_mapping.json")) as file:
        name_to_id = json.load(file)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    return [id_to_name[c] for c in range(len(id_to_name))]


def aggregate_simple_average(states: list[dict]) -> dict:
    avg = {}
    for key in states[0]:
        avg[key] = torch.stack([s[key].to(torch.float32) for s in states], dim=0).mean(dim=0)
    return avg


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Return (mean_loss, n_samples, n_batches, optimiser_steps)."""
    model.train()
    loss_sum, n_samples, n_batches = 0.0, 0, 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
        bs = labels.size(0)
        loss_sum += loss.item() * bs
        n_samples += bs
        n_batches += 1
    return loss_sum / n_samples, n_samples, n_batches, n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, n = 0.0, 0
    preds, targets, probs = [], [], []
    for features, labels in loader:
        features = features.to(device)
        labels_d = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels_d)
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


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for features, labels in loader:
        preds.append(torch.argmax(model(features.to(device)), dim=1).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(targets).astype(int), np.concatenate(preds).astype(int)


def balanced_weights(dpool_counts: np.ndarray) -> np.ndarray:
    n_pool = int(dpool_counts.sum())
    return n_pool / (NUM_CLASSES * dpool_counts.astype(np.float64))


def run(seed, partition, weight_vec, initial_state, val_loader, class_names, device):
    part_dir = PART_ROOT / f"seed_{seed}" / partition
    client_indices = [np.load(part_dir / f"client_{k:02d}_indices.npy") for k in range(NUM_CLIENTS)]
    client_datasets = [
        IndexedDataset(PROCESSED_DIR / "X_train.npy", PROCESSED_DIR / "y_train.npy", idx)
        for idx in client_indices
    ]

    set_all_seeds(BASE_SEED)
    global_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    global_model.load_state_dict(copy.deepcopy(initial_state))
    local_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    set_all_seeds(BASE_SEED)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight_vec, dtype=torch.float32, device=device))

    best = {"macro_f1": -1.0}
    best_path = MODELS_DIR / f"best_seed{seed}_{partition}.pt"
    history, client_diag = [], []

    for rnd in range(1, NUM_ROUNDS + 1):
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        client_states, round_losses, total_steps = [], [], 0

        for client_id in range(NUM_CLIENTS):
            local_model.load_state_dict(global_state)
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            generator = torch.Generator()
            generator.manual_seed(BASE_SEED + rnd * 100 + client_id)
            loader = DataLoader(client_datasets[client_id], batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=0, generator=generator)
            local_loss, n_samples, n_batches, steps = train_one_epoch(
                local_model, loader, criterion, optimizer, device)
            client_states.append({k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()})
            round_losses.append(local_loss)
            total_steps += steps
            client_diag.append({
                "round": rnd, "client_id": client_id, "local_loss": local_loss,
                "n_samples": n_samples, "n_batches": n_batches,
                "optimiser_steps": steps, "local_epochs": LOCAL_EPOCHS,
            })

        global_model.load_state_dict(aggregate_simple_average(client_states))
        val = evaluate(global_model, val_loader, criterion, device)

        record = {
            "round": rnd,
            "mean_local_loss": float(np.mean(round_losses)),
            "total_optimiser_steps": total_steps,
            "val_loss": val["val_loss"], "accuracy": val["accuracy"],
            "balanced_accuracy": val["balanced_accuracy"],
            "macro_precision": val["macro_precision"], "macro_recall": val["macro_recall"],
            "macro_f1": val["macro_f1"], "weighted_precision": val["weighted_precision"],
            "weighted_recall": val["weighted_recall"], "weighted_f1": val["weighted_f1"],
            "macro_pr_auc": val["macro_pr_auc"],
            "worst_class_recall": val["worst_class_recall"], "worst_class_f1": val["worst_class_f1"],
        }
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

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(RESULTS_DIR / f"history_seed{seed}_{partition}.csv", index=False)
    pd.DataFrame(client_diag).to_csv(RESULTS_DIR / f"clientdiag_seed{seed}_{partition}.csv", index=False)

    # Best-checkpoint artefacts (confusion matrix + classification report).
    global_model.load_state_dict(torch.load(best_path, map_location=device))
    y_true, y_pred = predict(global_model, val_loader, device)
    labels_range = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        RESULTS_DIR / f"confusion_seed{seed}_{partition}.csv")
    rep = classification_report(y_true, y_pred, labels=labels_range, target_names=class_names,
                                zero_division=0, output_dict=True)
    pd.DataFrame(rep).transpose().to_csv(RESULTS_DIR / f"report_seed{seed}_{partition}.csv")

    # Stability metrics.
    macro_series = hist_df["macro_f1"].to_numpy()
    macro_final = float(macro_series[-1])
    var_last5 = float(np.var(macro_series[-5:]))
    return best, hist_df, macro_final, var_last5


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    class_names = load_class_names()
    y_train = np.load(PROCESSED_DIR / "y_train.npy")

    skew = pd.read_csv(CONTROLLED_METRICS).set_index(["seed", "partition_type"])

    # Shared initial global model across ALL runs.
    set_all_seeds(BASE_SEED)
    init_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    initial_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
    torch.save(initial_state, MODELS_DIR / "initial_global_model.pt")

    val_loader = DataLoader(
        FullDataset(PROCESSED_DIR / "X_val.npy", PROCESSED_DIR / "y_val.npy"),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Per-seed balanced weight vector from that seed's D_pool.
    weight_by_seed = {}
    for seed in SEEDS:
        d_pool = np.load(PART_ROOT / f"seed_{seed}" / "d_pool_indices.npy")
        counts = np.bincount(y_train[d_pool], minlength=NUM_CLASSES).astype(np.int64)
        weight_by_seed[seed] = balanced_weights(counts)
    with open(MODELS_DIR / "balanced_weight_vectors.json", "w") as f:
        json.dump({str(s): weight_by_seed[s].tolist() for s in SEEDS}, f, indent=2)

    summary_rows = []
    for seed in SEEDS:
        for partition in PARTITIONS:
            best, hist_df, macro_final, var_last5 = run(
                seed, partition, weight_by_seed[seed], initial_state, val_loader, class_names, device)

            key = (seed, partition)
            hd = float(skew.loc[key, "HD_skew"]); jsd = float(skew.loc[key, "JSD_skew"]); emd = float(skew.loc[key, "EMD_skew"])
            delta = float(best["macro_f1"]) - macro_final

            row = {
                "seed": seed, "partition": partition,
                "HD_skew": hd, "JSD_skew": jsd, "EMD_skew": emd,
                "best_macro_f1": float(best["macro_f1"]),
                "best_macro_recall": float(best["macro_recall"]),
                "best_round": int(best["round"]), "rounds_to_best": int(best["round"]),
                "macro_f1_final": macro_final,
                "delta_best_minus_final": delta,
                "variance_last_5_rounds": var_last5,
            }
            for name, cid in HIGH_RISK.items():
                row[f"{name}_precision"] = float(best["per_precision"][cid])
                row[f"{name}_recall"] = float(best["per_recall"][cid])
                row[f"{name}_f1"] = float(best["per_f1"][cid])
                row[f"{name}_pr_auc"] = float(best["pr_auc"][cid])
            summary_rows.append(row)
            print(f"seed={seed} {partition:10} best_mF1={best['macro_f1']:.4f} "
                  f"(round {best['round']:02d}) final={macro_final:.4f} "
                  f"Worms_recall={best['per_recall'][9]:.3f} HD={hd:.4f}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "final_summary.csv", index=False)

    print("\n=== FINAL SUMMARY (Seed | Partition | HD_skew | Best Macro-F1 | Delta Best-vs-Final | Worms Recall) ===")
    show = summary[["seed", "partition", "HD_skew", "best_macro_f1", "delta_best_minus_final", "Worms_recall"]].copy()
    for c in ["HD_skew", "best_macro_f1", "delta_best_minus_final", "Worms_recall"]:
        show[c] = show[c].round(4)
    print(show.to_string(index=False))
    print(f"\nSaved: {RESULTS_DIR / 'final_summary.csv'}")


if __name__ == "__main__":
    main()
