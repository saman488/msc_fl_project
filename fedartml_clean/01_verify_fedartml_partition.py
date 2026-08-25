from pathlib import Path
import argparse

import numpy as np
from fedartml import SplitAsFederatedData
from fedartml.function_base import hellinger_distance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
Y_TRAIN_PATH = PROJECT_ROOT / "data" / "processed_37f" / "y_train.npy"

K = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if args.alpha <= 0:
        raise ValueError("alpha must be > 0")

    # Existing preprocessed Dataset-1 labels. Read only.
    y_train = np.load(Y_TRAIN_PATH)

    if y_train.ndim != 1:
        raise RuntimeError(f"Expected 1-D y_train, got shape {y_train.shape}")

    classes = np.unique(y_train)
    expected_classes = np.arange(len(classes))

    if not np.array_equal(classes, expected_classes):
        raise RuntimeError(
            f"FedArtML class assumption not satisfied: "
            f"found {classes.tolist()}, expected {expected_classes.tolist()}"
        )

    # FedArtML partition generation.
    pctg_distr, _, idx_distr, _ = SplitAsFederatedData.dirichlet_method(
        labels=y_train,
        local_nodes=K,
        alpha=args.alpha,
        random_state=args.seed,
    )

    # FedArtML's own HD implementation.
    pctg_distr = np.asarray(pctg_distr, dtype=np.float64)
    fedartml_hd = float(hellinger_distance(pctg_distr))

    # Convert FedArtML indices only for verification.
    client_indices = [
        np.asarray(indices, dtype=np.int64)
        for indices in idx_distr
    ]

    if len(client_indices) != K:
        raise RuntimeError(
            f"FedArtML returned {len(client_indices)} clients, expected {K}"
        )

    if any(len(indices) == 0 for indices in client_indices):
        raise RuntimeError("FedArtML returned an empty client")

    all_indices = np.concatenate(client_indices)
    n = len(y_train)

    if len(all_indices) != n:
        raise RuntimeError(
            f"Coverage failure: assigned {len(all_indices):,} indices for {n:,} records"
        )

    if len(np.unique(all_indices)) != n:
        raise RuntimeError("Duplicate client indices detected")

    if not np.array_equal(np.sort(all_indices), np.arange(n)):
        raise RuntimeError("FedArtML indices do not cover exactly 0..N-1")

    client_counts = np.stack([
        np.bincount(y_train[indices], minlength=len(classes))
        for indices in client_indices
    ])

    global_counts = np.bincount(y_train, minlength=len(classes))

    if not np.array_equal(client_counts.sum(axis=0), global_counts):
        raise RuntimeError("Client class counts do not reproduce global class counts")

    print("FedArtML partition verification")
    print(f"y_train:      {n:,}")
    print(f"classes:      {classes.tolist()}")
    print(f"K:            {K}")
    print(f"alpha:        {args.alpha}")
    print(f"random_state: {args.seed}")
    print(f"HD:           {fedartml_hd:.12f}")
    print(f"client sizes: {[len(x) for x in client_indices]}")
    print(f"pctg shape:   {pctg_distr.shape}")
    print("coverage:     PASS")
    print("class totals: PASS")


if __name__ == "__main__":
    main()
