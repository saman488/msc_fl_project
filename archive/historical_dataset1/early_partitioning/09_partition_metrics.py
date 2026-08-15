"""
Partition-skew metrics for the non-IID Dirichlet client splits.

For each alpha and each client, the client's 10-class label distribution is
compared against the global training distribution using three metrics:

  * Hellinger Distance (HD)      - euclidean distance of the sqrt-probability
                                   vectors, scaled by 1/sqrt(2) (standard HD).
  * Jensen-Shannon Distance(JSD) - scipy.spatial.distance.jensenshannon.
  * Wasserstein / EMD            - optimal transport (ot.emd2) under a custom
                                   10x10 attack-severity cost matrix.

The per-client metrics are averaged per alpha and written to
results/fl_partitions/non_iid_metric_summary.csv.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import ot
from scipy.spatial.distance import euclidean, jensenshannon


PROCESSED_DIR = Path("data/processed")
CLIENT_ROOT = Path("data/fl_clients/non_iid")
RESULTS_DIR = Path("results/fl_partitions")
CONFIGS_DIR = Path("configs")

NUM_CLIENTS = 5
ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

# Attack-severity risk score per category. Cost between two classes is the
# absolute difference of their risk scores, giving:
#   Benign->Benign = 0,  Benign->{Reconnaissance,Analysis} = 1,
#   Benign->{Exploits,Shellcode,Worms} = 5.
RISK_BY_NAME = {
    "Benign": 0,          # normal
    "Reconnaissance": 1,  # low-risk
    "Analysis": 1,        # low-risk
    "Generic": 2,         # medium
    "Fuzzers": 2,         # medium
    "DoS": 3,             # medium
    "Backdoor": 4,        # medium-high
    "Exploits": 5,        # high-risk / rare
    "Shellcode": 5,       # high-risk / rare
    "Worms": 5,           # high-risk / rare
}


def load_label_mapping() -> dict[int, str]:
    with open(CONFIGS_DIR / "label_mapping.json") as file:
        name_to_id = json.load(file)
    return {int(v): k for k, v in name_to_id.items()}


def build_cost_matrix(id_to_name: dict[int, str]) -> np.ndarray:
    """10x10 cost matrix M[i,j] = |risk(i) - risk(j)| over class ids."""
    num_classes = len(id_to_name)
    risk = np.array(
        [RISK_BY_NAME[id_to_name[c]] for c in range(num_classes)], dtype=np.float64
    )
    return np.abs(risk[:, None] - risk[None, :])


def label_histogram(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Normalised class-probability vector over 0..num_classes-1."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2.0))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    id_to_name = load_label_mapping()
    num_classes = len(id_to_name)

    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    global_dist = label_histogram(y_train, num_classes)

    cost_matrix = build_cost_matrix(id_to_name)

    print("Label mapping (id -> name):")
    for c in range(num_classes):
        print(f"  {c}: {id_to_name[c]:16} risk={RISK_BY_NAME[id_to_name[c]]}")
    print("\nGlobal train distribution:")
    print("  " + "  ".join(f"{id_to_name[c][:4]}={global_dist[c]:.4f}" for c in range(num_classes)))
    print("\n10x10 cost matrix (|risk_i - risk_j|):")
    print(cost_matrix.astype(int))

    per_client_rows = []
    summary_rows = []

    for alpha in ALPHAS:
        alpha_dir = CLIENT_ROOT / f"alpha_{alpha}"
        hd_vals, jsd_vals, emd_vals = [], [], []

        for client_id in range(NUM_CLIENTS):
            indices = np.load(alpha_dir / f"client_{client_id:02d}_indices.npy")
            client_dist = label_histogram(y_train[indices], num_classes)

            hd = hellinger_distance(client_dist, global_dist)
            jsd = float(jensenshannon(client_dist, global_dist))
            emd = float(ot.emd2(client_dist, global_dist, cost_matrix))

            hd_vals.append(hd)
            jsd_vals.append(jsd)
            emd_vals.append(emd)

            per_client_rows.append(
                {
                    "alpha": alpha,
                    "client_id": client_id,
                    "n": int(len(indices)),
                    "hellinger": hd,
                    "jsd": jsd,
                    "emd": emd,
                }
            )

        summary_rows.append(
            {
                "alpha": alpha,
                "hellinger_mean": float(np.mean(hd_vals)),
                "jsd_mean": float(np.mean(jsd_vals)),
                "emd_mean": float(np.mean(emd_vals)),
                "hellinger_std": float(np.std(hd_vals)),
                "jsd_std": float(np.std(jsd_vals)),
                "emd_std": float(np.std(emd_vals)),
            }
        )

    pd.DataFrame(per_client_rows).to_csv(
        RESULTS_DIR / "non_iid_metric_per_client.csv", index=False
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "non_iid_metric_summary.csv", index=False)

    print("\n=== non_iid_metric_summary.csv ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
