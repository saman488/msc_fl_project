from pathlib import Path
from itertools import combinations
import hashlib
import json

import numpy as np
from fedartml import SplitAsFederatedData
from fedartml.function_base import (
    hellinger_distance,
    jensen_shannon_distance,
    earth_movers_distance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_TRAIN_PATH = PROJECT_ROOT / "data" / "nf_cse_cic_ids2018_v2" / "processed" / "X_train.npy"
Y_TRAIN_PATH = PROJECT_ROOT / "data" / "nf_cse_cic_ids2018_v2" / "processed" / "y_train.npy"
OUT_DIR = (
    PROJECT_ROOT
    / "fedartml_clean"
    / "d2_partitions"
    / "k_5"
    / "seed_43"
    / "hd_0p5"
)

K = 5
NUM_CLASSES = 7
INPUT_DIM = 36
ALPHA = 0.5225
RANDOM_STATE = 43


def index_sha256(indices: np.ndarray) -> str:
    arr = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def main() -> None:
    if OUT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite existing partition: {OUT_DIR}")

    y_train = np.load(Y_TRAIN_PATH)
    x_train = np.load(X_TRAIN_PATH, mmap_mode="r")

    if y_train.ndim != 1:
        raise RuntimeError(f"Expected 1-D y_train, got {y_train.shape}")
    if x_train.ndim != 2 or x_train.shape != (len(y_train), INPUT_DIM):
        raise RuntimeError(
            f"Expected X_train shape ({len(y_train)}, {INPUT_DIM}), got {x_train.shape}"
        )

    classes = np.unique(y_train)
    if len(classes) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} classes, got {len(classes)}")
    expected = np.arange(NUM_CLASSES)
    if not np.array_equal(classes, expected):
        raise RuntimeError(
            f"Expected contiguous labels {expected.tolist()}, got {classes.tolist()}"
        )

    pctg_distr, _, idx_distr, _ = SplitAsFederatedData.dirichlet_method(
        labels=y_train,
        local_nodes=K,
        alpha=ALPHA,
        random_state=RANDOM_STATE,
    )

    pctg_distr = np.asarray(pctg_distr, dtype=np.float64)
    hd = float(hellinger_distance(pctg_distr))

    client_indices = [
        np.sort(np.asarray(indices, dtype=np.int64))
        for indices in idx_distr
    ]

    if len(client_indices) != K:
        raise RuntimeError(f"Expected {K} clients, got {len(client_indices)}")

    if any(len(indices) == 0 for indices in client_indices):
        raise RuntimeError("Empty client detected")

    all_indices = np.concatenate(client_indices)
    n = len(y_train)

    if len(all_indices) != n:
        raise RuntimeError("Training-record coverage count failed")

    if len(np.unique(all_indices)) != n:
        raise RuntimeError("Duplicate indices detected")

    if not np.array_equal(np.sort(all_indices), np.arange(n)):
        raise RuntimeError("Indices do not cover exactly 0..N-1")

    num_classes = len(classes)
    client_counts = np.stack([
        np.bincount(y_train[indices], minlength=num_classes)
        for indices in client_indices
    ])
    global_counts = np.bincount(y_train, minlength=num_classes)

    if not np.array_equal(client_counts.sum(axis=0), global_counts):
        raise RuntimeError("Client class totals do not reproduce y_train")

    # Verify that FedArtML's proportion table describes the exact
    # client indices that will later be used for training.
    reconstructed_pctg = (
        client_counts / client_counts.sum(axis=1, keepdims=True)
    ).astype(np.float64)

    if not np.allclose(reconstructed_pctg, pctg_distr, rtol=0.0, atol=1e-12):
        max_diff = float(np.max(np.abs(reconstructed_pctg - pctg_distr)))
        raise RuntimeError(
            f"FedArtML pctg_distr disagrees with idx_distr; max difference={max_diff}"
        )

    reconstructed_hd = float(hellinger_distance(reconstructed_pctg))
    if not np.isclose(reconstructed_hd, hd, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"HD from index-derived distributions {reconstructed_hd:.12f} "
            f"!= FedArtML HD {hd:.12f}"
        )

    js = float(jensen_shannon_distance(reconstructed_pctg))
    emd = float(earth_movers_distance(reconstructed_pctg))

    tv_pairs = np.array([
        0.5 * np.abs(reconstructed_pctg[i] - reconstructed_pctg[j]).sum()
        for i, j in combinations(range(K), 2)
    ], dtype=np.float64)

    target_hd = 0.50

    # Never write directly into the final partition directory.
    # A final directory exists only after disk reload verification succeeds.
    tmp_dir = OUT_DIR.with_name(OUT_DIR.name + ".tmp")
    if tmp_dir.exists():
        raise RuntimeError(f"Temporary partition directory already exists: {tmp_dir}")

    tmp_dir.mkdir(parents=True, exist_ok=False)

    for client_id, indices in enumerate(client_indices):
        np.save(tmp_dir / f"client_{client_id:02d}_indices.npy", indices)

    # Reload exactly what was written to disk.
    disk_indices = [
        np.load(tmp_dir / f"client_{client_id:02d}_indices.npy")
        for client_id in range(K)
    ]

    if any(arr.dtype != np.int64 for arr in disk_indices):
        raise RuntimeError("Saved client index arrays are not int64")

    disk_all = np.concatenate(disk_indices)

    if len(disk_all) != n:
        raise RuntimeError("Saved indices do not cover the expected number of records")

    if len(np.unique(disk_all)) != n:
        raise RuntimeError("Saved indices contain duplicates")

    if not np.array_equal(np.sort(disk_all), np.arange(n)):
        raise RuntimeError("Saved indices do not cover exactly 0..N-1")

    disk_counts = np.stack([
        np.bincount(y_train[indices], minlength=num_classes)
        for indices in disk_indices
    ])

    if not np.array_equal(disk_counts, client_counts):
        raise RuntimeError("Saved client class counts differ from verified in-memory counts")

    original_hashes = [index_sha256(indices) for indices in client_indices]
    disk_hashes = [index_sha256(indices) for indices in disk_indices]

    if disk_hashes != original_hashes:
        raise RuntimeError("Saved client index hashes differ from in-memory partition")

    manifest = {
        "partition_method": "fedartml.SplitAsFederatedData.dirichlet_method",
        "K": K,
        "alpha": ALPHA,
        "fedartml_random_state": RANDOM_STATE,
        "target_hd": target_hd,
        "fedartml_hellinger_distance": hd,
        "fedartml_jensen_shannon_distance": js,
        "tv_pairwise_mean": float(tv_pairs.mean()),
        "tv_pairwise_rms": float(np.sqrt(np.mean(tv_pairs ** 2))),
        "tv_pairwise_min": float(tv_pairs.min()),
        "tv_pairwise_max": float(tv_pairs.max()),
        "fedartml_emd_diagnostic": emd,
        "fedartml_emd_use": "diagnostic_only",
        "abs_error_vs_target_hd": abs(hd - target_hd),
        "total_records": n,
        "num_classes": num_classes,
        "client_sizes": [int(len(indices)) for indices in disk_indices],
        "global_class_counts": global_counts.tolist(),
        "client_class_counts": disk_counts.tolist(),
        "client_index_sha256": disk_hashes,
    }

    with open(tmp_dir / "partition_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    # Same filesystem: rename exposes the verified partition as the final directory.
    tmp_dir.rename(OUT_DIR)

    print(f"WROTE: {OUT_DIR}")
    print(f"Target HD:   {target_hd:.12f}")
    print(f"Achieved HD: {hd:.12f}")
    print(f"Abs error:   {abs(hd - target_hd):.12f}")
    print("client sizes:", manifest["client_sizes"])
    print("disk reload: PASS")
    print("coverage:    PASS")
    print("class totals: PASS")
    print("hashes:      PASS")


if __name__ == "__main__":
    main()
