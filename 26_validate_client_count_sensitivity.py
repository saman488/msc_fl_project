"""
Partition-only sensitivity analysis for federation size (num_clients).

Sweeps num_clients over the approved partition semantics, parameterised by K:
- non-IID: pure class-wise Dirichlet, every real class index used once, split of
  the full shuffled class array across clients. No balancing mask, IPF, floors,
  oversampling, fixed capacities, retries, or equalisation.
- IID: same deterministic class shuffle, then an even split across clients.

The RNG/allocation rules match the final partition generator, so the K=5
conditions reproduce the final client sets exactly. Reads y_train.npy and the
label mapping only; never reads X_train, validation, or test. Client
heterogeneity uses the normalized Hellinger distance on achieved client
distributions. Writes diagnostics only, not client index arrays.
"""

from pathlib import Path
from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

PROCESSED_DIR = Path("data/processed")
LABEL_MAP = Path("configs/label_mapping.json")
OUT_ROOT = Path("results/partition_validation/client_count_sensitivity")

NUM_CLIENTS_GRID = [5, 10, 20]
NONIID_ALPHAS = [0.1, 0.5, 1.0]
PARTITION_SEEDS = [42, 43, 44]


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


def shuffle_rng(partition_seed: int, num_clients: int, class_id: int) -> np.random.Generator:
    # Within-class shuffle depends only on seed, num_clients, and class id.
    seq = np.random.SeedSequence([int(partition_seed), int(num_clients), int(class_id)])
    return np.random.default_rng(seq)


def dirichlet_rng(partition_seed: int, num_clients: int, dirichlet_alpha: float, class_id: int) -> np.random.Generator:
    # Dirichlet draw depends on seed, num_clients, alpha, class id, and the fixed
    # attempt-0 component used by the final generator.
    seq = np.random.SeedSequence([int(partition_seed), int(num_clients),
                                  int(round(dirichlet_alpha * 1000)), int(class_id), 0])
    return np.random.default_rng(seq)


def partition_noniid(y_train: np.ndarray, num_classes: int, num_clients: int,
                     dirichlet_alpha: float, partition_seed: int) -> list[np.ndarray]:
    """Pure class-wise Dirichlet split of every real class index."""
    client_parts = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_indices = np.where(y_train == c)[0]
        shuffle_rng(partition_seed, num_clients, c).shuffle(class_indices)
        proportions = dirichlet_rng(partition_seed, num_clients, dirichlet_alpha, c).dirichlet(
            np.repeat(dirichlet_alpha, num_clients))
        for k, part in enumerate(split_by_proportions(class_indices, proportions)):
            client_parts[k].append(part)
    return [np.sort(np.concatenate(parts)) for parts in client_parts]


def partition_iid(y_train: np.ndarray, num_classes: int, num_clients: int,
                  partition_seed: int) -> list[np.ndarray]:
    """Even split of every real class index across clients (same class shuffle)."""
    client_parts = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_indices = np.where(y_train == c)[0]
        shuffle_rng(partition_seed, num_clients, c).shuffle(class_indices)
        for k, part in enumerate(np.array_split(class_indices, num_clients)):
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
        "proportions": proportions,
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


def condition_tag(num_clients: int, dirichlet_alpha, partition_seed: int) -> str:
    return f"k{num_clients}_{condition_name(dirichlet_alpha)}_seed{partition_seed}"


def save_condition(out_dir: Path, class_order: list[str], counts: np.ndarray,
                   metrics: dict, metadata: dict) -> None:
    cols = class_order
    idx = [f"client_{k}" for k in range(counts.shape[0])]

    pd.DataFrame(counts, index=idx, columns=cols).to_csv(out_dir / "client_class_counts.csv")
    pd.DataFrame(metrics["proportions"], index=idx, columns=cols).to_csv(out_dir / "client_class_proportions.csv")

    per_client = pd.DataFrame({
        "client_id": np.arange(counts.shape[0]),
        "client_size": metrics["client_sizes"],
        "absent_class_count": metrics["absent_classes_per_client"],
        "hd_to_global": metrics["hd_to_global_values"],
    })
    per_client.to_csv(out_dir / "per_client_diagnostics.csv", index=False)

    pd.DataFrame(metrics["pairwise_records"]).to_csv(out_dir / "pairwise_hd.csv", index=False)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    class_order = load_class_order()
    num_classes = len(class_order)
    y_train = np.load(PROCESSED_DIR / "y_train.npy")  # labels only; no X/val/test read

    observed_label_ids = set(int(v) for v in np.unique(y_train).tolist())
    expected_label_ids = set(range(num_classes))
    assert observed_label_ids == expected_label_ids, (
        f"observed label IDs {sorted(observed_label_ids)} do not equal "
        f"expected {sorted(expected_label_ids)} (num_classes={num_classes})"
    )

    total_records = int(len(y_train))
    global_class_counts = np.bincount(y_train, minlength=num_classes)

    # non-IID alphas then one IID condition per (K, seed).
    conditions = [(a, False) for a in NONIID_ALPHAS] + [(None, True)]

    summary_rows = []
    for num_clients in NUM_CLIENTS_GRID:
        for partition_seed in PARTITION_SEEDS:
            for dirichlet_alpha, is_iid in conditions:
                tag = condition_tag(num_clients, dirichlet_alpha, partition_seed)
                out_dir = OUT_ROOT / tag
                # Do not overwrite a pre-existing non-empty condition directory.
                if out_dir.exists() and any(out_dir.iterdir()):
                    raise RuntimeError(
                        f"Condition directory already exists and is non-empty: {out_dir}. "
                        "Remove it manually before re-running to avoid mixing stale outputs."
                    )
                out_dir.mkdir(parents=True, exist_ok=True)

                try:
                    if is_iid:
                        client_indices = partition_iid(y_train, num_classes, num_clients, partition_seed)
                    else:
                        client_indices = partition_noniid(y_train, num_classes, num_clients,
                                                          dirichlet_alpha, partition_seed)
                    counts = verify_partition(client_indices, y_train, num_classes, global_class_counts)
                    metrics = compute_metrics(counts, global_class_counts)
                except (RuntimeError, AssertionError) as exc:
                    print(f"[{tag}] FAILED: {exc}", flush=True)
                    summary_rows.append({
                        "num_clients": num_clients,
                        "alpha": "" if is_iid else dirichlet_alpha,
                        "partition_seed": partition_seed, "status": "FAILED",
                        "error": str(exc),
                    })
                    with open(out_dir / "metadata.json", "w") as f:
                        json.dump({"status": "FAILED", "error": str(exc),
                                   "num_clients": num_clients,
                                   "alpha": None if is_iid else dirichlet_alpha,
                                   "partition_seed": partition_seed}, f, indent=2)
                    continue

                metadata = {
                    "status": "OK",
                    "num_clients": num_clients,
                    "alpha": None if is_iid else dirichlet_alpha,
                    "partition_seed": partition_seed,
                    "num_classes": num_classes,
                    "class_order": class_order,
                    "total_records": total_records,
                    "global_class_counts": global_class_counts.tolist(),
                    "client_sizes": metrics["client_sizes"].tolist(),
                    "client_to_global_hd": metrics["hd_to_global_values"].tolist(),
                    "absent_classes_per_client": metrics["absent_classes_per_client"].tolist(),
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
                }
                save_condition(out_dir, class_order, counts, metrics, metadata)

                summary_rows.append({
                    "num_clients": num_clients,
                    "alpha": "" if is_iid else dirichlet_alpha,
                    "partition_seed": partition_seed, "status": "OK",
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
                print(f"[{tag}] OK  hd_pairwise_mean={metrics['hd_pairwise_mean']:.4f} "
                      f"size_cv={metrics['size_cv']:.4f}", flush=True)

    pd.DataFrame(summary_rows).to_csv(OUT_ROOT / "summary.csv", index=False)
    print(f"\nWrote summary: {OUT_ROOT / 'summary.csv'}")


if __name__ == "__main__":
    main()
