from pathlib import Path
import argparse

import numpy as np
from fedartml import SplitAsFederatedData
from fedartml.function_base import hellinger_distance


ROOT = Path(__file__).resolve().parents[1]
K = 5

DATASETS = {
    "d1": {
        "y_train": ROOT / "data" / "processed_37f" / "y_train.npy",
        "num_classes": 10,
    },
    "d2": {
        "y_train": ROOT / "data" / "nf_cse_cic_ids2018_v2" / "processed" / "y_train.npy",
        "num_classes": 7,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--partition-seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()

    if args.alpha <= 0:
        raise ValueError("alpha must be > 0")

    cfg = DATASETS[args.dataset]
    y_path = cfg["y_train"]
    expected_classes = cfg["num_classes"]

    if not y_path.exists():
        raise RuntimeError(f"Missing y_train: {y_path}")

    y_train = np.load(y_path)

    if y_train.ndim != 1:
        raise RuntimeError(
            f"Expected 1-D y_train, got {y_train.shape}"
        )

    classes = np.unique(y_train)
    expected = np.arange(expected_classes)

    if not np.array_equal(classes, expected):
        raise RuntimeError(
            f"Expected labels {expected.tolist()}, "
            f"got {classes.tolist()}"
        )

    pctg_distr, _, idx_distr, _ = (
        SplitAsFederatedData.dirichlet_method(
            labels=y_train,
            local_nodes=K,
            alpha=args.alpha,
            random_state=args.partition_seed,
        )
    )

    pctg_distr = np.asarray(
        pctg_distr, dtype=np.float64
    )

    fedartml_hd = float(
        hellinger_distance(pctg_distr)
    )

    client_indices = [
        np.asarray(indices, dtype=np.int64)
        for indices in idx_distr
    ]

    if len(client_indices) != K:
        raise RuntimeError(
            f"Expected {K} clients, "
            f"got {len(client_indices)}"
        )

    if any(len(indices) == 0 for indices in client_indices):
        raise RuntimeError("Empty client detected")

    all_indices = np.concatenate(client_indices)
    n = len(y_train)

    if len(all_indices) != n:
        raise RuntimeError(
            f"Coverage count failed: "
            f"{len(all_indices)} != {n}"
        )

    if len(np.unique(all_indices)) != n:
        raise RuntimeError("Duplicate indices detected")

    if not np.array_equal(
        np.sort(all_indices), np.arange(n)
    ):
        raise RuntimeError(
            "Indices do not cover exactly 0..N-1"
        )

    client_counts = np.stack([
        np.bincount(
            y_train[indices],
            minlength=expected_classes,
        )
        for indices in client_indices
    ])

    global_counts = np.bincount(
        y_train,
        minlength=expected_classes,
    )

    if not np.array_equal(
        client_counts.sum(axis=0), global_counts
    ):
        raise RuntimeError(
            "Client class totals do not reproduce y_train"
        )

    reconstructed_pctg = (
        client_counts
        / client_counts.sum(axis=1, keepdims=True)
    ).astype(np.float64)

    if not np.allclose(
        reconstructed_pctg,
        pctg_distr,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "FedArtML pctg_distr disagrees "
            "with returned client indices"
        )

    reconstructed_hd = float(
        hellinger_distance(reconstructed_pctg)
    )

    if not np.isclose(
        reconstructed_hd,
        fedartml_hd,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"Reconstructed HD {reconstructed_hd:.12f} "
            f"!= FedArtML HD {fedartml_hd:.12f}"
        )

    print("FedArtML calibration verification")
    print(f"dataset:        {args.dataset}")
    print(f"y_train:        {y_path}")
    print(f"records:        {n:,}")
    print(f"classes:        {expected_classes}")
    print(f"K:              {K}")
    print(f"partition_seed: {args.partition_seed}")
    print(f"alpha:          {args.alpha}")
    print(f"HD:             {fedartml_hd:.12f}")
    print(
        "client_sizes:   "
        f"{[len(x) for x in client_indices]}"
    )
    print("coverage:       PASS")
    print("class totals:   PASS")
    print("index/HD match: PASS")


if __name__ == "__main__":
    main()
