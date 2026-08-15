"""
Centralised MLP baseline for MULTI-CLASS intrusion detection on NF-UNSW-NB15-v2.

Architecture and hyperparameters are kept identical to the binary baseline
(41->128->64, AdamW, lr=1e-3, batch_size=4096); only the output layer (10 logits),
the loss (class-weighted CrossEntropyLoss), and the metrics (multi-class) change.
Model selection is based on validation macro-F1, and the held-out test set is
evaluated once after loading the best validation checkpoint.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
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
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/centralised")
CONFIGS_DIR = Path("configs")

RANDOM_STATE = 42
INPUT_DIM = 41
NUM_CLASSES = 10
BATCH_SIZE = 4096
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# High-risk / rare attack classes to track explicitly (class ids).
HIGH_RISK_CLASSES = {"Exploits": 4, "Shellcode": 8, "Worms": 9}


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
        # CrossEntropyLoss expects integer class indices (long).
        label = torch.tensor(self.y[index], dtype=torch.long)
        return features, label


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


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_class_names() -> list[str]:
    with open(CONFIGS_DIR / "label_mapping.json") as file:
        name_to_id = json.load(file)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    return [id_to_name[c] for c in range(len(id_to_name))]


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


def compute_class_weights(y_train: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Balanced class weights: weight[c] = N_total / (num_classes * count_c)."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)

    if (counts == 0).any():
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Training set is missing classes: {missing}")

    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


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
    predictions = []
    targets = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        y_pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        y_true = labels.detach().cpu().numpy()

        loss_sum += loss.item() * len(y_true)
        predictions.append(y_pred)
        targets.append(y_true)

    y_pred = np.concatenate(predictions).astype(int)
    y_true = np.concatenate(targets).astype(int)

    labels_range = list(range(NUM_CLASSES))
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels_range, average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        y_true, y_pred, labels=labels_range, average=None, zero_division=0
    )

    metrics = {
        "loss": float(loss_sum / len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    for class_id in labels_range:
        metrics[f"recall_class_{class_id}"] = float(per_class_recall[class_id])
        metrics[f"f1_class_{class_id}"] = float(per_class_f1[class_id])

    return metrics


def main() -> None:
    set_reproducibility(RANDOM_STATE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    class_names = load_class_names()
    train_loader, val_loader, test_loader = build_loaders()

    # Balanced class weighting offsets the extreme multi-class imbalance.
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    class_weights = compute_class_weights(y_train, NUM_CLASSES).to(device)

    model = MLPMultiClassClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config = {
        "task": "multiclass_intrusion_detection",
        "model": "MLPMultiClassClassifier",
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "CrossEntropyLoss",
        "class_weights": class_weights.detach().cpu().numpy().tolist(),
        "device": str(device),
        "random_state": RANDOM_STATE,
    }

    with open(CONFIGS_DIR / "central_mlp_config.json", "w") as file:
        json.dump(config, file, indent=2)

    history = []
    best_val_macro_f1 = -1.0
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

        # Select by validation macro-F1; the held-out test set is not used in training.
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_model_path)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_macro_recall={val_metrics['macro_recall']:.4f} "
            f"val_recall_Exploits={val_metrics['recall_class_4']:.4f} "
            f"val_recall_Shellcode={val_metrics['recall_class_8']:.4f} "
            f"val_recall_Worms={val_metrics['recall_class_9']:.4f} "
        )

    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS_DIR / "central_mlp_training_history.csv", index=False)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Final test evaluation on the best-validation checkpoint.
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features)
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(labels.numpy())
    y_pred = np.concatenate(predictions).astype(int)
    y_true = np.concatenate(targets).astype(int)

    test_metrics = evaluate(model, test_loader, criterion, device)
    pd.DataFrame([test_metrics]).to_csv(
        RESULTS_DIR / "central_mlp_test_metrics.csv", index=False
    )

    labels_range = list(range(NUM_CLASSES))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_range,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        RESULTS_DIR / "central_mlp_test_classification_report.csv"
    )

    confusion = confusion_matrix(y_true, y_pred, labels=labels_range)
    pd.DataFrame(confusion, index=class_names, columns=class_names).to_csv(
        RESULTS_DIR / "central_mlp_test_confusion_matrix.csv"
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels_range,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )

    print("\nBest validation macro-F1:", round(best_val_macro_f1, 6))
    print("\n=== Test Classification Report ===")
    print(report_text)
    print("Test macro-F1:    ", round(test_metrics["macro_f1"], 6))
    print("Test macro-recall:", round(test_metrics["macro_recall"], 6))
    print("\nHigh-risk class recall (test):")
    for name, class_id in HIGH_RISK_CLASSES.items():
        print(f"  {name:10} (class {class_id}): recall = {test_metrics[f'recall_class_{class_id}']:.4f}")


if __name__ == "__main__":
    main()
