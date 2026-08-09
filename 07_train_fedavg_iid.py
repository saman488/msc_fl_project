"""
FedAvg federated baseline on the IID client partition of NF-UNSW-NB15-v2.

The script reuses the exact model, optimizer settings, global pos_weight,
evaluation metrics, and processed-array paths from 04_train_central_mlp.py, and
consumes the client index files produced by 06_create_iid_clients.py.

Each communication round: every client is initialised from the current global
state with a fresh optimizer, trained for LOCAL_EPOCHS local epoch(s), and the
resulting client models are aggregated by client sample count (weighted FedAvg).
The aggregated global model is evaluated on the global validation split after
each round; the best-by-validation-attack-F1 checkpoint is reloaded for a single
final evaluation on the held-out test split.
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
CLIENT_DIR = Path("data/fl_clients/iid")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/fl_iid")
CONFIGS_DIR = Path("configs")

RANDOM_STATE = 42
INPUT_DIM = 41
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
THRESHOLD = 0.5

NUM_CLIENTS = 5
LOCAL_EPOCHS = 1
DEFAULT_ROUNDS = 30


# --- Reused verbatim from 04_train_central_mlp.py ----------------------------
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
# --- End reused block --------------------------------------------------------


class ClientDataset(Dataset):
    """Subset of the training arrays selected by a client's index file.

    Mirrors NumpyDataset's memory-mapped loading and tensor formatting, but maps
    each position through the client's stored training-row indices.
    """

    def __init__(self, x_path: Path, y_path: Path, indices: np.ndarray) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.indices = indices.astype(np.int64)

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"Feature and label row counts differ: {self.x.shape[0]} vs {self.y.shape[0]}"
            )

        if self.x.shape[1] != INPUT_DIM:
            raise ValueError(
                f"Expected {INPUT_DIM} input features, found {self.x.shape[1]}"
            )

        if self.indices.min() < 0 or self.indices.max() >= self.x.shape[0]:
            raise ValueError("Client indices fall outside the training range.")

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.indices[index])
        features = torch.from_numpy(np.asarray(self.x[row], dtype=np.float32).copy())
        label = torch.tensor(self.y[row], dtype=torch.float32)
        return features, label


def build_client_loaders() -> list[DataLoader]:
    """One shuffled training loader per client, driven by its index file."""
    loaders = []
    for client_id in range(NUM_CLIENTS):
        indices = np.load(CLIENT_DIR / f"client_{client_id:02d}_indices.npy")
        dataset = ClientDataset(
            PROCESSED_DIR / "X_train.npy",
            PROCESSED_DIR / "y_train.npy",
            indices,
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
            )
        )
    return loaders


def build_eval_loaders() -> tuple[DataLoader, DataLoader]:
    val_dataset = NumpyDataset(
        PROCESSED_DIR / "X_val.npy",
        PROCESSED_DIR / "y_val.npy",
    )
    test_dataset = NumpyDataset(
        PROCESSED_DIR / "X_test.npy",
        PROCESSED_DIR / "y_test.npy",
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
    return val_loader, test_loader


def build_val_loader() -> DataLoader:
    """Validation loader only, for pilot (validation-only) runs.

    The test dataset is deliberately not opened so the held-out test split is
    never touched during pilot experiments.
    """
    val_dataset = NumpyDataset(
        PROCESSED_DIR / "X_val.npy",
        PROCESSED_DIR / "y_val.npy",
    )
    return DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


def aggregate_states(
    client_states: list[dict], client_counts: list[int]
) -> dict:
    """Sample-count-weighted FedAvg over client state_dicts (all on CPU)."""
    total = float(sum(client_counts))
    aggregated = {}

    for key in client_states[0]:
        weighted_sum = None
        for state, count in zip(client_states, client_counts):
            contribution = state[key].to(torch.float32) * (count / total)
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        aggregated[key] = weighted_sum

    return aggregated


def run_round(
    global_model: nn.Module,
    client_loaders: list[DataLoader],
    pos_weight: torch.Tensor,
    device: torch.device,
) -> tuple[dict, list[float]]:
    """One communication round: local training on each client, then aggregate."""
    global_state = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }

    client_states = []
    client_counts = []
    client_losses = []

    for client_id, loader in enumerate(client_loaders):
        client_dataset = loader.dataset
        if not isinstance(client_dataset, ClientDataset):
            raise TypeError(
                f"Expected ClientDataset for client {client_id}, "
                f"found {type(client_dataset).__name__}."
            )

        # Fresh model initialised from the current global state.
        local_model = MLPBinaryClassifier(INPUT_DIM).to(device)
        local_model.load_state_dict(copy.deepcopy(global_state))

        # Fresh optimizer each round (no client-side optimizer state carried over).
        optimizer = torch.optim.AdamW(
            local_model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        local_loss = 0.0
        for _ in range(LOCAL_EPOCHS):
            local_loss = train_one_epoch(
                model=local_model,
                loader=loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

        client_states.append(
            {
                key: value.detach().cpu().clone()
                for key, value in local_model.state_dict().items()
            }
        )
        client_counts.append(len(client_dataset))
        client_losses.append(local_loss)

        print(
            f"  client={client_id:02d} "
            f"rows={len(client_dataset)} "
            f"local_train_loss={local_loss:.6f}"
        )

    aggregated = aggregate_states(client_states, client_counts)
    global_model.load_state_dict(aggregated)

    return aggregated, client_losses


def main() -> None:
    parser = argparse.ArgumentParser(description="FedAvg on IID NF-UNSW-NB15-v2.")
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help="Number of communication rounds.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Pilot mode: train and evaluate on validation only, never touching "
            "the held-out test split. Writes pilot-suffixed checkpoint, history, "
            "and config, and leaves the final test outputs untouched."
        ),
    )
    args = parser.parse_args()
    num_rounds = args.rounds
    validation_only = args.validation_only

    if num_rounds < 1:
        parser.error(f"--rounds must be >= 1, got {num_rounds}.")

    set_reproducibility(RANDOM_STATE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    client_loaders = build_client_loaders()
    if validation_only:
        val_loader = build_val_loader()
        test_loader = None
    else:
        val_loader, test_loader = build_eval_loaders()

    # Global positive-class weighting, identical to the centralised baseline.
    pos_weight_value = compute_pos_weight()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    global_model = MLPBinaryClassifier(INPUT_DIM).to(device)
    eval_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    config = {
        "task": "binary_intrusion_detection",
        "strategy": "fedavg",
        "partition": "iid_stratified",
        "model": "MLPBinaryClassifier",
        "input_dim": INPUT_DIM,
        "num_clients": NUM_CLIENTS,
        "num_rounds": num_rounds,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "threshold": THRESHOLD,
        "loss": "BCEWithLogitsLoss",
        "pos_weight": pos_weight_value,
        "aggregation": "sample_count_weighted",
        "device": str(device),
        "random_state": RANDOM_STATE,
    }
    config_filename = (
        "fedavg_iid_pilot_config.json" if validation_only else "fedavg_iid_config.json"
    )
    with open(CONFIGS_DIR / config_filename, "w") as file:
        json.dump(config, file, indent=2)

    history = []
    best_val_f1 = -1.0
    best_model_path = MODELS_DIR / (
        "fedavg_iid_pilot_best.pt" if validation_only else "fedavg_iid_best.pt"
    )
    best_checkpoint_saved = False

    for round_num in range(1, num_rounds + 1):
        print(f"round={round_num:02d}")
        _, client_losses = run_round(
            global_model=global_model,
            client_loaders=client_loaders,
            pos_weight=pos_weight,
            device=device,
        )

        val_metrics = evaluate(
            model=global_model,
            loader=val_loader,
            criterion=eval_criterion,
            device=device,
        )

        row = {
            "round": round_num,
            "mean_client_train_loss": float(np.mean(client_losses)),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        # Select the global model by validation attack-F1; test stays held out.
        if val_metrics["attack_f1"] > best_val_f1:
            best_val_f1 = val_metrics["attack_f1"]
            torch.save(global_model.state_dict(), best_model_path)
            best_checkpoint_saved = True

        print(
            f"round={round_num:02d} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_attack_f1={val_metrics['attack_f1']:.4f} "
            f"val_attack_recall={val_metrics['attack_recall']:.4f} "
            f"val_attack_precision={val_metrics['attack_precision']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
        )

    history_df = pd.DataFrame(history)
    history_filename = (
        "fedavg_iid_pilot_training_history.csv"
        if validation_only
        else "fedavg_iid_training_history.csv"
    )
    history_df.to_csv(RESULTS_DIR / history_filename, index=False)

    if validation_only:
        # Pilot mode stops here: the held-out test split is never evaluated and
        # no final test outputs are created or overwritten.
        print("\n[validation-only] Best validation F1:", round(best_val_f1, 6))
        return

    if not best_checkpoint_saved:
        raise RuntimeError(
            "No best checkpoint was saved during this run; refusing to load a "
            "possibly stale checkpoint for test evaluation."
        )
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Expected best checkpoint at {best_model_path}, but it is missing."
        )

    global_model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = evaluate(
        model=global_model,
        loader=test_loader,
        criterion=eval_criterion,
        device=device,
    )

    pd.DataFrame([test_metrics]).to_csv(
        RESULTS_DIR / "fedavg_iid_test_metrics.csv",
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
        RESULTS_DIR / "fedavg_iid_test_confusion_matrix.csv",
        index=False,
    )

    print("\nBest validation F1:", round(best_val_f1, 6))
    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
