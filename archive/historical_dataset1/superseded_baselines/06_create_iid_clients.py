"""
Create IID client partitions for the first federated-learning baseline.

The script partitions only the training split. Validation and test splits remain
global so that FL runs are evaluated against the same reference sets used by the
centralised baseline.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


PROCESSED_DIR = Path("data/processed")
CLIENT_DIR = Path("data/fl_clients/iid")
RESULTS_DIR = Path("results/fl_partitions")
CONFIGS_DIR = Path("configs")

NUM_CLIENTS = 5
RANDOM_STATE = 42


def summarise_client(client_id: int, indices: np.ndarray, y_train: np.ndarray) -> dict:
    labels = y_train[indices]

    benign_count = int((labels == 0).sum())
    attack_count = int((labels == 1).sum())
    rows = int(len(indices))

    return {
        "client_id": client_id,
        "rows": rows,
        "benign_count": benign_count,
        "attack_count": attack_count,
        "attack_ratio": attack_count / rows,
    }


def validate_partitions(partitions: list[np.ndarray], n_rows: int) -> None:
    all_indices = np.concatenate(partitions)

    if len(all_indices) != n_rows:
        raise ValueError(
            f"Partition row count mismatch: {len(all_indices)} indices for {n_rows} rows."
        )

    unique_indices = np.unique(all_indices)

    if len(unique_indices) != n_rows:
        raise ValueError(
            f"Partition overlap or missing rows detected: "
            f"{len(unique_indices)} unique indices for {n_rows} rows."
        )

    if unique_indices.min() != 0 or unique_indices.max() != n_rows - 1:
        raise ValueError("Partition indices do not cover the expected training range.")


def main() -> None:
    CLIENT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    y_train = np.load(PROCESSED_DIR / "y_train.npy")

    # Stratification keeps the benign/attack ratio close to the global training split.
    splitter = StratifiedKFold(
        n_splits=NUM_CLIENTS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    dummy_x = np.zeros(len(y_train))
    partitions = []
    summaries = []

    for client_id, (_, client_indices) in enumerate(splitter.split(dummy_x, y_train)):
        client_indices = client_indices.astype(np.int64)
        partitions.append(client_indices)

        # Store indices rather than duplicating the feature arrays.
        np.save(CLIENT_DIR / f"client_{client_id:02d}_indices.npy", client_indices)

        summaries.append(
            summarise_client(
                client_id=client_id,
                indices=client_indices,
                y_train=y_train,
            )
        )

    # Each training row must belong to exactly one client.
    validate_partitions(partitions, n_rows=len(y_train))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS_DIR / "iid_client_summary.csv", index=False)

    config = {
        "partition": "iid_stratified",
        "source_split": "train",
        "num_clients": NUM_CLIENTS,
        "random_state": RANDOM_STATE,
        "method": "StratifiedKFold",
        "coverage_check": "all training rows assigned exactly once",
    }

    with open(CONFIGS_DIR / "iid_client_partition_config.json", "w") as file:
        json.dump(config, file, indent=2)

    print(summary_df)
    print("Total rows:", int(summary_df["rows"].sum()))
    print("Total benign:", int(summary_df["benign_count"].sum()))
    print("Total attack:", int(summary_df["attack_count"].sum()))


if __name__ == "__main__":
    main()