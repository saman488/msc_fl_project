"""
IID-only FedAvg pilot to freeze the loss weighting scheme for the multi-class
non-IID experiment.

Design / controls
-----------------
* Architecture + hyperparameters identical to the centralised baseline:
  MLPMultiClassClassifier (41->128->64->10), AdamW, lr=1e-3, weight_decay=1e-5,
  batch_size=4096, 1 local epoch, 30 communication rounds.
* Data: the controlled IID partition for seed_42 only
  (data/fl_clients/controlled_partitions/seed_42/iid/, 5 clients x 50,000).
* Class weights are computed ONCE from the seed-42 D_pool counts (aggregate of the
  5 clients) and applied identically to every client.
* Perfect cross-scheme reproducibility: one shared initial global model, seeds reset
  before each scheme, and a per-(round, client) torch.Generator seeded with
  base_seed + round*100 + client_id so batch order / client order / dropout match.
* Evaluation and model selection use the VALIDATION set only. The test set is never
  touched. Nothing here overwrites earlier artifacts.
"""

from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROCESSED_DIR = Path("data/processed")
IID_DIR = Path("data/fl_clients/controlled_partitions/seed_42/iid")
RESULTS_DIR = Path("results/fl_iid_pilot")
MODELS_DIR = Path("models/fl_iid_pilot")

INPUT_DIM = 41
NUM_CLASSES = 10
NUM_CLIENTS = 5
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LOCAL_EPOCHS = 1
NUM_ROUNDS = 30
BASE_SEED = 42

LOSS_SCHEMES = ["unweighted", "balanced", "sqrt_balanced"]
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
    """Client subset of the training arrays selected by an index file."""

    def __init__(self, x_path: Path, y_path: Path, indices: np.ndarray) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def build_weight_vectors(dpool_counts: np.ndarray) -> dict:
    """unweighted (None), balanced, sqrt_balanced from D_pool counts."""
    n_pool = int(dpool_counts.sum())
    balanced = n_pool / (NUM_CLASSES * dpool_counts.astype(np.float64))
    sqrt_balanced = np.sqrt(balanced)
    return {
        "unweighted": None,
        "balanced": balanced,
        "sqrt_balanced": sqrt_balanced,
    }


def aggregate_simple_average(states: list[dict]) -> dict:
    """Simple (unweighted) FedAvg — valid because clients are strictly equal size."""
    avg = {}
    for key in states[0]:
        stacked = torch.stack([s[key].to(torch.float32) for s in states], dim=0)
        avg[key] = stacked.mean(dim=0)
    return avg


def train_one_epoch(model, loader, criterion, optimizer, device) -> None:
    model.train()
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, targets = [], []
    for features, labels in loader:
        logits = model(features.to(device))
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        targets.append(labels.numpy())
    y_pred = np.concatenate(preds).astype(int)
    y_true = np.concatenate(targets).astype(int)

    labels_range = list(range(NUM_CLASSES))
    per_p = precision_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    per_r = recall_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    per_f = f1_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_precision": per_p,
        "per_recall": per_r,
        "per_f1": per_f,
    }


def run_scheme(scheme, weight_vec, initial_state, client_datasets, val_loader, device):
    """Run 30-round FedAvg for one loss scheme; return best-by-val-macro-F1 metrics."""
    set_all_seeds(BASE_SEED)

    global_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    global_model.load_state_dict(copy.deepcopy(initial_state))
    local_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)

    # Reset seeds AFTER model construction so the dropout RNG stream entering
    # training is identical across all three schemes.
    set_all_seeds(BASE_SEED)

    if weight_vec is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weight_vec, dtype=torch.float32, device=device)
        )

    best = {"macro_f1": -1.0}
    best_path = MODELS_DIR / f"best_{scheme}.pt"
    history = []

    for rnd in range(1, NUM_ROUNDS + 1):
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        client_states = []

        for client_id in range(NUM_CLIENTS):
            local_model.load_state_dict(global_state)
            optimizer = torch.optim.AdamW(
                local_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            generator = torch.Generator()
            generator.manual_seed(BASE_SEED + rnd * 100 + client_id)
            loader = DataLoader(
                client_datasets[client_id],
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                generator=generator,
            )
            for _ in range(LOCAL_EPOCHS):
                train_one_epoch(local_model, loader, criterion, optimizer, device)
            client_states.append(
                {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
            )

        global_model.load_state_dict(aggregate_simple_average(client_states))

        val = evaluate(global_model, val_loader, device)
        history.append({"round": rnd, "val_macro_f1": val["macro_f1"],
                        "val_macro_recall": val["macro_recall"]})

        if val["macro_f1"] > best["macro_f1"]:
            best = {"round": rnd, **val}
            torch.save(global_model.state_dict(), best_path)

        print(f"[{scheme}] round={rnd:02d} val_macro_f1={val['macro_f1']:.4f} "
              f"val_macro_recall={val['macro_recall']:.4f}", flush=True)

    pd.DataFrame(history).to_csv(RESULTS_DIR / f"history_{scheme}.csv", index=False)
    print(f"[{scheme}] BEST round={best['round']} macro_f1={best['macro_f1']:.4f}\n", flush=True)
    return best


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    class_names = load_class_names()

    # ---- client datasets (controlled seed-42 IID) ----
    client_indices = [
        np.load(IID_DIR / f"client_{k:02d}_indices.npy") for k in range(NUM_CLIENTS)
    ]
    client_datasets = [
        IndexedDataset(PROCESSED_DIR / "X_train.npy", PROCESSED_DIR / "y_train.npy", idx)
        for idx in client_indices
    ]

    # ---- D_pool counts = aggregate of the 5 controlled clients ----
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    pooled_indices = np.concatenate(client_indices)
    dpool_counts = np.bincount(y_train[pooled_indices], minlength=NUM_CLASSES).astype(np.int64)
    assert int(dpool_counts.sum()) == NUM_CLIENTS * 50_000

    weight_vectors = build_weight_vectors(dpool_counts)

    # Persist D_pool counts and weight vectors.
    np.save(MODELS_DIR / "dpool_counts.npy", dpool_counts)
    weights_serialisable = {
        "dpool_counts": dpool_counts.tolist(),
        "n_pool": int(dpool_counts.sum()),
        "unweighted": None,
        "balanced": weight_vectors["balanced"].tolist(),
        "sqrt_balanced": weight_vectors["sqrt_balanced"].tolist(),
    }
    with open(MODELS_DIR / "class_weight_vectors.json", "w") as file:
        json.dump(weights_serialisable, file, indent=2)
    pd.DataFrame({
        "class_id": range(NUM_CLASSES),
        "class_name": class_names,
        "dpool_count": dpool_counts,
        "balanced": weight_vectors["balanced"],
        "sqrt_balanced": weight_vectors["sqrt_balanced"],
    }).to_csv(RESULTS_DIR / "class_weight_vectors.csv", index=False)

    # ---- validation loader (evaluation/selection only) ----
    val_loader = DataLoader(
        FullDataset(PROCESSED_DIR / "X_val.npy", PROCESSED_DIR / "y_val.npy"),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # ---- single shared initial global model ----
    set_all_seeds(BASE_SEED)
    initial_model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    initial_state = {k: v.detach().cpu().clone() for k, v in initial_model.state_dict().items()}
    torch.save(initial_state, MODELS_DIR / "initial_global_model.pt")

    print("D_pool counts:", dpool_counts.tolist())
    print("balanced weights:", [round(w, 3) for w in weight_vectors["balanced"].tolist()])
    print("sqrt_balanced weights:", [round(w, 3) for w in weight_vectors["sqrt_balanced"].tolist()])
    print()

    # ---- run the three schemes ----
    rows = []
    for scheme in LOSS_SCHEMES:
        best = run_scheme(scheme, weight_vectors[scheme], initial_state,
                          client_datasets, val_loader, device)
        row = {
            "loss_scheme": scheme,
            "best_round": best["round"],
            "val_macro_f1": best["macro_f1"],
            "val_macro_recall": best["macro_recall"],
        }
        for name, cid in HIGH_RISK.items():
            row[f"{name}_precision"] = float(best["per_precision"][cid])
            row[f"{name}_recall"] = float(best["per_recall"][cid])
            row[f"{name}_f1"] = float(best["per_f1"][cid])
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "loss_comparison_val_metrics.csv", index=False)

    # ---- print comparison table ----
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("=== LOSS SCHEME COMPARISON (best validation checkpoint) ===")
    summary = comparison[["loss_scheme", "best_round", "val_macro_f1", "val_macro_recall"]].copy()
    for c in ["val_macro_f1", "val_macro_recall"]:
        summary[c] = summary[c].round(4)
    print(summary.to_string(index=False))

    print("\n=== HIGH-RISK per-class (Precision / Recall / F1) ===")
    for name in HIGH_RISK:
        sub = comparison[["loss_scheme", f"{name}_precision", f"{name}_recall", f"{name}_f1"]].copy()
        sub.columns = ["loss_scheme", "precision", "recall", "f1"]
        for c in ["precision", "recall", "f1"]:
            sub[c] = sub[c].round(4)
        print(f"\n[{name}]")
        print(sub.to_string(index=False))

    print(f"\nSaved: {RESULTS_DIR / 'loss_comparison_val_metrics.csv'}")


if __name__ == "__main__":
    main()
