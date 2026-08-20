"""
Generate the final Dataset-2 (NF-CSE-CIC-IDS2018 v2) K=5 training partitions from
the training labels only.

Dataset-2 counterpart of 25_create_final_partitions.py. The partition logic is
unchanged: same pure class-wise Dirichlet allocation, same deterministic RNG
semantics, same integrity checks and heterogeneity diagnostics. Only the dataset
paths and the expected training-state assertions differ.

Two partition types:
- non-IID: pure class-wise Dirichlet. Every real index of each class is used;
  the within-class shuffle is deterministic in (seed, K, class id); the Dirichlet
  draw is deterministic in (seed, K, alpha, class id); the full shuffled class
  array is split across clients by those proportions. No balancing mask, IPF,
  floors, oversampling, fixed capacities, or post-hoc equalisation.
- IID: same deterministic class shuffle, then the full class array is divided as
  evenly as possible across the 5 clients.

Quantity variation and absent classes are realised properties of the draw; they
are measured and recorded, never repaired.

Reads y_train.npy and the label mapping only; never reads X_train, validation, or
test. Client heterogeneity uses the normalized Hellinger distance on achieved
client distributions. Saves the actual client index arrays and per-condition
manifests plus a top-level summary.
"""

from pathlib import Path
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

PROCESSED_DIR = Path("data/nf_cse_cic_ids2018_v2/processed")
LABEL_MAP = Path("configs/nf_cse_cic_ids2018_v2/label_mapping.json")
OUT_ROOT = Path("data/nf_cse_cic_ids2018_v2/fl_clients/final_partitions/k_5")
SUMMARY_PATH = Path(
    "results/nf_cse_cic_ids2018_v2/fl_partitions/final_partitions/partition_summary.csv"
)

K = 5
PARTITION_SEEDS = [42, 43, 44]
NONIID_ALPHAS = [0.1, 0.5, 1.0]

# Expected Dataset-2 training state; asserted before any partitioning.
EXPECTED_TOTAL_RECORDS = 13_255_011
EXPECTED_CLASS_ORDER = [
    "Benign",        # 0
    "Bot",           # 1
    "BruteForce",    # 2
    "DDoS",          # 3
    "DoS",           # 4
    "Infiltration",  # 5
    "Web Attacks",   # 6
]


def load_class_order() -> list[str]:
    """Class names ordered by their integer id in the saved label mapping."""
    name_to_id = json.load(open(LABEL_MAP))
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    return [id_to_name[c] for c in range(len(id_to_name))]


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2.0))


def split_by_proportions(class_indices: np.ndarray, proportions: np.ndarray) -> list[np.ndarray]:
    """Split a full class index array into K disjoint parts covering it exactly."""
    cut = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
    return np.split(class_indices, cut)


def shuffle_rng(partition_seed: int, class_id: int) -> np.random.Generator:
    # Within-class shuffle depends only on seed, K, and class id.
    seq = np.random.SeedSequence([int(partition_seed), int(K), int(class_id)])
    return np.random.default_rng(seq)


def dirichlet_rng(partition_seed: int, dirichlet_alpha: float, class_id: int) -> np.random.Generator:
    # Dirichlet draw depends on seed, K, alpha, and class id.
    seq = np.random.SeedSequence([int(partition_seed), int(K),
                                  int(round(dirichlet_alpha * 1000)), int(class_id), 0])
    return np.random.default_rng(seq)


def partition_noniid(y_train: np.ndarray, num_classes: int,
                     dirichlet_alpha: float, partition_seed: int) -> list[np.ndarray]:
    """Pure class-wise Dirichlet split of every real class index."""
    client_parts = [[] for _ in range(K)]
    for c in range(num_classes):
        class_indices = np.where(y_train == c)[0]
        shuffle_rng(partition_seed, c).shuffle(class_indices)
        proportions = dirichlet_rng(partition_seed, dirichlet_alpha, c).dirichlet(np.repeat(dirichlet_alpha, K))
        for k, part in enumerate(split_by_proportions(class_indices, proportions)):
            client_parts[k].append(part)
    return [np.sort(np.concatenate(parts)) for parts in client_parts]


def partition_iid(y_train: np.ndarray, num_classes: int, partition_seed: int) -> list[np.ndarray]:
    """Even split of every real class index across clients (same class shuffle)."""
    client_parts = [[] for _ in range(K)]
    for c in range(num_classes):
        class_indices = np.where(y_train == c)[0]
        shuffle_rng(partition_seed, c).shuffle(class_indices)
        for k, part in enumerate(np.array_split(class_indices, K)):
            client_parts[k].append(part)
    return [np.sort(np.concatenate(parts)) for parts in client_parts]


def verify_partition(client_indices: list[np.ndarray], y_train: np.ndarray,
                     num_classes: int, global_class_counts: np.ndarray) -> np.ndarray:
    """Integrity checks; returns the K x num_classes client count matrix."""
    total_records = len(y_train)
    all_assigned = np.concatenate(client_indices)
    assert len(all_assigned) == total_records, "not all training indices assigned"
    assert len(np.unique(all_assigned)) == total_records, "duplicate indices across clients"
    assert np.array_equal(np.sort(all_assigned), np.arange(total_records)), "coverage is not exactly 0..N-1"
    assert all(len(ci) > 0 for ci in client_indices), "a client is empty"

    counts = np.stack([np.bincount(y_train[ci], minlength=num_classes) for ci in client_indices])
    assert np.array_equal(counts.sum(axis=0), global_class_counts), "class totals differ from global counts"
    return counts


def compute_metrics(counts: np.ndarray, global_class_counts: np.ndarray) -> dict:
    """HD and size diagnostics from achieved client distributions."""
    client_sizes = counts.sum(axis=1)
    proportions = counts / client_sizes[:, None]
    global_distribution = global_class_counts / global_class_counts.sum()

    pairwise_records = []
    for i, j in combinations(range(counts.shape[0]), 2):
        pairwise_records.append({
            "client_i": i, "client_j": j,
            "HD": hellinger_distance(proportions[i], proportions[j]),
        })
    hd_pairwise = np.array([r["HD"] for r in pairwise_records])
    hd_to_global = np.array([hellinger_distance(proportions[k], global_distribution)
                             for k in range(counts.shape[0])])
    absent_classes_per_client = (counts == 0).sum(axis=1)

    size_mean = float(client_sizes.mean())
    size_std = float(client_sizes.std())
    return {
        "pairwise_records": pairwise_records,
        "hd_pairwise_mean": float(hd_pairwise.mean()),
        "hd_pairwise_min": float(hd_pairwise.min()),
        "hd_pairwise_max": float(hd_pairwise.max()),
        "hd_pairwise_rms": float(np.sqrt(np.mean(hd_pairwise ** 2))),
        "hd_to_global_values": hd_to_global,
        "mean_client_to_global_hd": float(hd_to_global.mean()),
        "max_client_to_global_hd": float(hd_to_global.max()),
        "client_sizes": client_sizes,
        "size_min": int(client_sizes.min()),
        "size_max": int(client_sizes.max()),
        "size_mean": size_mean,
        "size_std": size_std,
        "size_cv": float(size_std / size_mean) if size_mean > 0 else 0.0,
        "size_max_min_ratio": float(client_sizes.max() / client_sizes.min()),
        "absent_classes_per_client": absent_classes_per_client,
        "mean_absent_classes_per_client": float(absent_classes_per_client.mean()),
    }


def condition_name(dirichlet_alpha) -> str:
    if dirichlet_alpha is None:
        return "iid"
    return f"alpha_{str(dirichlet_alpha).replace('.', 'p')}"


def index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest()


def main() -> None:
    class_order = load_class_order()
    num_classes = len(class_order)
    y_train = np.load(PROCESSED_DIR / "y_train.npy")  # labels only; no X/val/test read

    # Dataset-2 expected training state.
    assert class_order == EXPECTED_CLASS_ORDER, (
        f"class order {class_order} does not match expected Dataset-2 order {EXPECTED_CLASS_ORDER}"
    )
    assert num_classes == len(EXPECTED_CLASS_ORDER), (
        f"num_classes={num_classes}, expected {len(EXPECTED_CLASS_ORDER)}"
    )
    assert len(y_train) == EXPECTED_TOTAL_RECORDS, (
        f"y_train has {len(y_train)} rows, expected {EXPECTED_TOTAL_RECORDS}"
    )

    observed_label_ids = set(int(v) for v in np.unique(y_train).tolist())
    expected_label_ids = set(range(num_classes))
    assert observed_label_ids == expected_label_ids, (
        f"observed label IDs {sorted(observed_label_ids)} do not equal "
        f"expected {sorted(expected_label_ids)} (num_classes={num_classes})"
    )

    total_records = int(len(y_train))
    global_class_counts = np.bincount(y_train, minlength=num_classes)
    assert len(global_class_counts) == num_classes, "unexpected label id above num_classes-1"
    assert int(global_class_counts.sum()) == total_records, "class counts do not sum to N"

    print(f"Dataset-2 training state OK: N={total_records:,}, classes={num_classes}")
    for c, name in enumerate(class_order):
        print(f"  {c} {name:<14} {int(global_class_counts[c]):>12,}")

    # (condition_alpha, is_iid) per seed: non-IID alphas then one IID condition.
    conditions = [(a, False) for a in NONIID_ALPHAS] + [(None, True)]

    # Preflight: refuse to write if any intended condition directory is non-empty.
    intended_dirs = []
    for partition_seed in PARTITION_SEEDS:
        for dirichlet_alpha, _ in conditions:
            intended_dirs.append(OUT_ROOT / f"seed_{partition_seed}" / condition_name(dirichlet_alpha))
    non_empty = [d for d in intended_dirs if d.exists() and any(d.iterdir())]
    if non_empty:
        listing = "\n  ".join(str(d) for d in non_empty)
        raise RuntimeError(f"Refusing to write; these condition directories are non-empty:\n  {listing}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for partition_seed in PARTITION_SEEDS:
        for dirichlet_alpha, is_iid in conditions:
            cond = condition_name(dirichlet_alpha)
            out_dir = OUT_ROOT / f"seed_{partition_seed}" / cond
            out_dir.mkdir(parents=True, exist_ok=True)

            if is_iid:
                partition_method = "iid_even"
                client_indices = partition_iid(y_train, num_classes, partition_seed)
            else:
                partition_method = "pure_classwise_dirichlet"
                client_indices = partition_noniid(y_train, num_classes, dirichlet_alpha, partition_seed)

            counts = verify_partition(client_indices, y_train, num_classes, global_class_counts)
            metrics = compute_metrics(counts, global_class_counts)

            # Save the actual client index arrays.
            for k, ci in enumerate(client_indices):
                np.save(out_dir / f"client_{k:02d}_indices.npy", ci)
            client_hashes = [index_hash(ci) for ci in client_indices]

            partition_id = f"k{K}_seed{partition_seed}_{cond}"
            manifest = {
                "dataset": "nf_cse_cic_ids2018_v2",
                "partition_id": partition_id,
                "partition_method": partition_method,
                "K": K,
                "alpha": None if is_iid else dirichlet_alpha,
                "partition_seed": partition_seed,
                "total_records": total_records,
                "class_order": class_order,
                "global_class_counts": global_class_counts.tolist(),
                "client_sizes": metrics["client_sizes"].tolist(),
                "client_class_counts": counts.tolist(),
                "client_index_sha256": client_hashes,
                "pairwise_hellinger": metrics["pairwise_records"],
                "hd_pairwise_mean": metrics["hd_pairwise_mean"],
                "hd_pairwise_min": metrics["hd_pairwise_min"],
                "hd_pairwise_max": metrics["hd_pairwise_max"],
                "hd_pairwise_rms": metrics["hd_pairwise_rms"],
                "client_to_global_hd": metrics["hd_to_global_values"].tolist(),
                "mean_client_to_global_hd": metrics["mean_client_to_global_hd"],
                "max_client_to_global_hd": metrics["max_client_to_global_hd"],
                "size_min": metrics["size_min"],
                "size_max": metrics["size_max"],
                "size_mean": metrics["size_mean"],
                "size_std": metrics["size_std"],
                "size_cv": metrics["size_cv"],
                "size_max_min_ratio": metrics["size_max_min_ratio"],
                "absent_classes_per_client": metrics["absent_classes_per_client"].tolist(),
            }
            with open(out_dir / "partition_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            summary_rows.append({
                "dataset": "nf_cse_cic_ids2018_v2",
                "partition_id": partition_id,
                "partition_method": partition_method,
                "partition_seed": partition_seed,
                "alpha": "" if is_iid else dirichlet_alpha,
                "total_records": total_records,
                "hd_pairwise_mean": metrics["hd_pairwise_mean"],
                "hd_pairwise_min": metrics["hd_pairwise_min"],
                "hd_pairwise_max": metrics["hd_pairwise_max"],
                "hd_pairwise_rms": metrics["hd_pairwise_rms"],
                "mean_client_to_global_hd": metrics["mean_client_to_global_hd"],
                "max_client_to_global_hd": metrics["max_client_to_global_hd"],
                "size_min": metrics["size_min"],
                "size_max": metrics["size_max"],
                "size_mean": metrics["size_mean"],
                "size_std": metrics["size_std"],
                "size_cv": metrics["size_cv"],
                "size_max_min_ratio": metrics["size_max_min_ratio"],
                "mean_absent_classes_per_client": metrics["mean_absent_classes_per_client"],
            })
            print(f"[{partition_id}] OK  hd_pairwise_mean={metrics['hd_pairwise_mean']:.4f} "
                  f"size_cv={metrics['size_cv']:.4f}", flush=True)

    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)
    print(f"\nWrote summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
