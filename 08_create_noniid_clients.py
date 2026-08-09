"""
Create non-IID (Dirichlet label-skew) client partitions with FedArtML.

For each Dirichlet concentration alpha, the training split is partitioned across
five clients with label skew, then every client is randomly subsampled to an
identical sample count so that quantity skew is removed and only label skew
remains. Client indices (into X_train / y_train) are saved per alpha.

FedArtML returns the actual feature rows per client, not indices. Because some
feature rows are duplicated, indices are recovered by consuming each
(feature-bytes, label) occurrence exactly once, with a label-consistency check.
"""

from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from fedartml import SplitAsFederatedData


PROCESSED_DIR = Path("data/processed")
CLIENT_ROOT = Path("data/fl_clients/non_iid")
RESULTS_DIR = Path("results/fl_partitions")

NUM_CLIENTS = 5
RANDOM_STATE = 42
HARD_CAP = 5000
ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]


def build_master_index(x_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Map (feature-bytes, label) -> list of original row indices."""
    master = defaultdict(list)
    for i in range(x_train.shape[0]):
        master[(x_train[i].tobytes(), int(y_train[i]))].append(i)
    return master


def recover_client_indices(
    clients_dict: dict, master: dict, y_train: np.ndarray
) -> dict:
    """Recover original indices for each client, consuming each occurrence once."""
    cursor: dict = defaultdict(int)
    recovered = {}

    for name, samples in clients_dict.items():
        idxs = np.empty(len(samples), dtype=np.int64)
        for j, (feature, label) in enumerate(samples):
            key = (np.asarray(feature, dtype=np.float32).tobytes(), int(label))
            index_list = master.get(key)
            position = cursor[key]
            if index_list is None or position >= len(index_list):
                raise RuntimeError(f"Index exhaustion recovering client {name}.")
            index = index_list[position]
            cursor[key] = position + 1
            if int(y_train[index]) != int(label):
                raise RuntimeError(
                    f"Label mismatch at index {index}: {y_train[index]} vs {label}."
                )
            idxs[j] = index
        recovered[name] = idxs

    return recovered


def sorted_client_names(names) -> list[str]:
    """Order client_1, client_2, ... by their numeric suffix."""
    return sorted(names, key=lambda n: int(n.split("_")[-1]))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    n_rows = x_train.shape[0]
    print(f"Loaded X_train {x_train.shape}, y_train {y_train.shape}", flush=True)

    # Build once; reused across alphas via a per-alpha cursor reset inside recovery.
    master = build_master_index(x_train, y_train)
    print(f"Built master index over {len(master):,} (feature,label) groups", flush=True)

    summary_rows = []

    for alpha in ALPHAS:
        alpha_dir = CLIENT_ROOT / f"alpha_{alpha}"
        alpha_dir.mkdir(parents=True, exist_ok=True)

        # NOTE: image_list=X_train per the specified API call.
        my_plot = SplitAsFederatedData(random_state=RANDOM_STATE)
        clients_glob, list_distr_glob, _, _ = my_plot.create_clients(
            image_list=x_train,
            label_list=y_train,
            num_clients=NUM_CLIENTS,
            method="dirichlet",
            alpha=alpha,
        )

        # True disjoint partition (sums to N; no injected duplicate samples).
        clients = clients_glob["without_class_completion"]
        recovered = recover_client_indices(clients, master, y_train)

        # Integrity: every training row assigned at most once, partition covers N.
        all_idx = np.concatenate([recovered[name] for name in recovered])
        if len(all_idx) != n_rows or len(np.unique(all_idx)) != n_rows:
            raise RuntimeError(
                f"alpha={alpha}: partition is not a clean cover "
                f"({len(all_idx)} total, {len(np.unique(all_idx))} unique, N={n_rows})."
            )

        raw_sizes = {name: int(len(recovered[name])) for name in recovered}
        smallest = min(raw_sizes.values())

        # CRITICAL QUANTITY EQUALIZATION: subsample every client to the same n.
        target_n = min(smallest, HARD_CAP)
        rng = np.random.RandomState(RANDOM_STATE)

        for client_id, name in enumerate(sorted_client_names(recovered.keys())):
            client_idx = recovered[name]
            chosen = rng.choice(client_idx, size=target_n, replace=False)
            chosen.sort()
            np.save(alpha_dir / f"client_{client_id:02d}_indices.npy", chosen)

            summary_rows.append(
                {
                    "alpha": alpha,
                    "client_id": client_id,
                    "fedartml_client": name,
                    "raw_size": raw_sizes[name],
                    "equalized_n": int(target_n),
                }
            )

        print(
            f"alpha={alpha}: raw sizes {list(raw_sizes.values())} "
            f"-> smallest={smallest}, cap={HARD_CAP}, equalized_n={target_n}",
            flush=True,
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "non_iid_partition_summary.csv", index=False)

    config = {
        "partition": "non_iid_dirichlet",
        "source_split": "train",
        "library": "fedartml",
        "num_clients": NUM_CLIENTS,
        "alphas": ALPHAS,
        "random_state": RANDOM_STATE,
        "quantity_equalization": "subsample all clients to min(smallest_client, hard_cap)",
        "hard_cap": HARD_CAP,
        "class_completion": "without_class_completion",
    }
    with open(CLIENT_ROOT / "non_iid_partition_config.json", "w") as file:
        json.dump(config, file, indent=2)

    print("\nPartition summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
