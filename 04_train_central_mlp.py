"""
Centralised MLP baseline for binary intrusion detection on NF-UNSW-NB15-v2.

The script uses the processed arrays produced by 03_preprocess.py. Model
selection is based on validation F1-score, and the held-out test set is evaluated
once after loading the best validation checkpoint.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/centralised")
CONFIGS_DIR = Path("configs")

RANDOM_STATE = 42
INPUT_DIM = 41
BATCH_SIZE = 4096
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
THRESHOLD = 0.5


class NumpyDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"Feature and label row counts differ: {self.x.shape[0]} vs {self.y.shape[0]}"
            )

        if self.x.shape[1] != INPUT_DIM:
            raise ValueError(
                f"Expected {INPUT_DIM} input features, found {self.x.shape[1]}"
            )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(np.asarray(self.x[index], dtype=np.float32).copy())
        label = torch.tensor(self.y[index], dtype=torch.float32)
        return features, label


class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = NumpyDataset(
        PROCESSED_DIR / "X_train.npy",
        PROCESSED_DIR / "y_train.npy",
    )
    val_dataset = NumpyDataset(
        PROCESSED_DIR / "X_val.npy",
        PROCESSED_DIR / "y_val.npy",
    )
    test_dataset = NumpyDataset(
        PROCESSED_DIR / "X_test.npy",
        PROCESSED_DIR / "y_test.npy",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def compute_pos_weight() -> float:
    y_train = np.load(PROCESSED_DIR / "y_train.npy", mmap_mode="r")

    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())

    if positive == 0:
        raise ValueError("Training set contains no attack samples.")

    return negative / positive


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_rows = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_rows += batch_size

    return total_loss / total_rows


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()

    loss_sum = 0.0
    probabilities = []
    targets = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        y_prob = torch.sigmoid(logits).detach().cpu().numpy()
        y_true = labels.detach().cpu().numpy()

        loss_sum += loss.item() * len(y_true)
        probabilities.append(y_prob)
        targets.append(y_true)

    y_prob = np.concatenate(probabilities)
    y_true = np.concatenate(targets).astype(int)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    predicted_positive_rate = float(y_pred.mean())

    return {
    "loss": float(loss_sum / len(y_true)),
    "accuracy": accuracy_score(y_true, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

    "attack_precision": precision_score(y_true, y_pred, zero_division=0),
    "attack_recall": recall_score(y_true, y_pred, zero_division=0),
    "attack_f1": f1_score(y_true, y_pred, zero_division=0),

    "macro_precision": precision_score(
        y_true, y_pred, average="macro", zero_division=0
    ),
    "macro_recall": recall_score(
        y_true, y_pred, average="macro", zero_division=0
    ),
    "macro_f1": f1_score(
        y_true, y_pred, average="macro", zero_division=0
    ),

    "weighted_precision": precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "weighted_recall": recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "weighted_f1": f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),

    "roc_auc": roc_auc_score(y_true, y_prob),
    "pr_auc": average_precision_score(y_true, y_prob),
    "mcc": matthews_corrcoef(y_true, y_pred),

    "specificity": specificity,
    "false_positive_rate": false_positive_rate,
    "false_negative_rate": false_negative_rate,
    "predicted_positive_rate": predicted_positive_rate,

    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
}


def main() -> None:
    set_reproducibility(RANDOM_STATE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    train_loader, val_loader, test_loader = build_loaders()

    # Positive-class weighting offsets the 96/4 benign-attack imbalance.
    pos_weight_value = compute_pos_weight()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    model = MLPBinaryClassifier(INPUT_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config = {
        "task": "binary_intrusion_detection",
        "model": "MLPBinaryClassifier",
        "input_dim": INPUT_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "threshold": THRESHOLD,
        "loss": "BCEWithLogitsLoss",
        "pos_weight": pos_weight_value,
        "device": str(device),
        "random_state": RANDOM_STATE,
    }

    with open(CONFIGS_DIR / "central_mlp_config.json", "w") as file:
        json.dump(config, file, indent=2)

    history = []
    best_val_f1 = -1.0
    best_model_path = MODELS_DIR / "central_mlp_best.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,

        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        # Select by validation F1; the held-out test set is not used in training.
        if val_metrics["attack_f1"] > best_val_f1:
            best_val_f1 = val_metrics["attack_f1"]
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_attack_f1={val_metrics['attack_f1']:.4f} "
            f"val_attack_recall={val_metrics['attack_recall']:.4f} "
            f"val_attack_precision={val_metrics['attack_precision']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
        )

    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS_DIR / "central_mlp_training_history.csv", index=False)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    pd.DataFrame([test_metrics]).to_csv(
        RESULTS_DIR / "central_mlp_test_metrics.csv",
        index=False,
    )

    confusion_table = pd.DataFrame(
        [
            {
                "tn": test_metrics["tn"],
                "fp": test_metrics["fp"],
                "fn": test_metrics["fn"],
                "tp": test_metrics["tp"],
            }
        ]
    )
    confusion_table.to_csv(
        RESULTS_DIR / "central_mlp_test_confusion_matrix.csv",
        index=False,
    )

    print("\nBest validation F1:", round(best_val_f1, 6))
    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()