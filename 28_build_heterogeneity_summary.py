"""
Read-only heterogeneity summary across all existing partitions.

Grid: K in {5, 10, 20}, seed in {42, 43, 44}, condition in
{iid, alpha_0p1, alpha_0p5, alpha_1p0} -> 36 conditions.

Sources of achieved client class counts:
- K=5: the actual final client index arrays, with counts recomputed from
  data/processed/y_train.npy (source of truth).
- K=10 / K=20: the existing client_class_counts.csv under the client-count
  sensitivity outputs (not regenerated here).

Computes Hellinger distance, Jensen-Shannon divergence in bits, and categorical
EMD under a 0/1 ground metric (which equals total variation), plus quantity and
support diagnostics. Reads training labels and the label mapping only; never reads
validation or test data. Writes diagnostics only.
"""

from pathlib import Path
from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

PROCESSED_DIR = Path("data/processed")
LABEL_MAP = Path("configs/label_mapping.json")
FINAL_K5_ROOT = Path("data/fl_clients/final_partitions/k_5")
SENS_ROOT = Path("results/partition_validation/client_count_sensitivity")
SENS_SUMMARY = SENS_ROOT / "summary.csv"
FINAL_SUMMARY = Path("results/fl_partitions/final_partitions/partition_summary.csv")
OUT_ROOT = Path("results/partition_validation/heterogeneity_summary")

NUM_CLASSES = 10
K_LIST = [5, 10, 20]
SEEDS = [42, 43, 44]
CONDITIONS = ["iid", "alpha_0p1", "alpha_0p5", "alpha_1p0"]
COND_ALPHA = {"iid": None, "alpha_0p1": 0.1, "alpha_0p5": 0.5, "alpha_1p0": 1.0}
TOL = 1e-10
UNIT_TOL = 1e-9

# HD fields recomputed here and checked against the existing summaries.
HD_CHECK_FIELDS = ["hd_pairwise_mean", "hd_pairwise_min", "hd_pairwise_max",
                   "hd_pairwise_rms", "mean_client_to_global_hd", "max_client_to_global_hd"]


def load_class_order() -> list[str]:
    name_to_id = json.load(open(LABEL_MAP))
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    order = [id_to_name[c] for c in range(len(id_to_name))]
    assert len(order) == NUM_CLASSES, f"label mapping has {len(order)} classes, expected {NUM_CLASSES}"
    return order


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2.0))


def jsd_bits(p: np.ndarray, q: np.ndarray) -> float:
    # Jensen-Shannon divergence in bits (log base 2); zero-probability terms drop out.
    m = 0.5 * (p + q)

    def kl2(a: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / m[mask])))

    return 0.5 * kl2(p) + 0.5 * kl2(q)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q)))


def emd_discrete01(p: np.ndarray, q: np.ndarray) -> float:
    # EMD under the categorical 0/1 ground metric (same class 0, different class 1).
    return float(np.abs(p - q).sum()) / 2.0


def k5_counts(seed: int, condition: str, y_train: np.ndarray,
              global_class_counts: np.ndarray) -> np.ndarray:
    """Recompute K=5 client class counts from the actual final index arrays."""
    part_dir = FINAL_K5_ROOT / f"seed_{seed}" / condition
    files = sorted(part_dir.glob("client_*_indices.npy"))
    assert len(files) == 5, f"{part_dir}: expected 5 client files, got {len(files)}"
    client_indices = [np.load(part_dir / f"client_{k:02d}_indices.npy") for k in range(5)]
    assert all(len(ci) > 0 for ci in client_indices), f"{part_dir}: a client is empty"

    all_assigned = np.concatenate(client_indices)
    total_records = len(y_train)
    assert len(all_assigned) == total_records, f"{part_dir}: indices do not cover all training records"
    assert len(np.unique(all_assigned)) == total_records, f"{part_dir}: duplicate training indices"
    assert np.array_equal(np.sort(all_assigned), np.arange(total_records)), f"{part_dir}: coverage is not 0..N-1"
    counts = np.stack([np.bincount(y_train[ci], minlength=NUM_CLASSES) for ci in client_indices])
    assert np.array_equal(counts.sum(axis=0), global_class_counts), f"{part_dir}: class totals differ from global"
    return counts


def sensitivity_counts(K: int, seed: int, condition: str, class_order: list[str]) -> np.ndarray:
    """Read existing K=10/20 client class counts (not regenerated)."""
    path = SENS_ROOT / f"k{K}_{condition}_seed{seed}" / "client_class_counts.csv"
    df = pd.read_csv(path, index_col=0)
    assert list(df.columns) == class_order, f"{path}: class columns/order mismatch"
    assert df.shape[0] == K, f"{path}: expected {K} client rows, got {df.shape[0]}"
    values = df.to_numpy()
    assert np.isfinite(values).all(), f"{path}: non-finite count values"
    assert (values >= 0).all(), f"{path}: negative count values"
    assert np.array_equal(values, np.round(values)), f"{path}: non-integer count values"
    return values.astype(np.int64)


def condition_from_alpha(a) -> str:
    if a is None or a == "" or (isinstance(a, float) and np.isnan(a)):
        return "iid"
    return f"alpha_{str(float(a)).replace('.', 'p')}"


def load_summary_lookup(path: Path, k_column: str = None, fixed_k: int = None) -> dict:
    """Index an existing summary CSV by (K, partition_seed, condition).

    K comes from a real column (k_column) or a fixed value (fixed_k); it is never
    inferred from partition_seed.
    """
    assert (k_column is None) != (fixed_k is None), "provide exactly one of k_column/fixed_k"
    df = pd.read_csv(path)
    lookup = {}
    for _, row in df.iterrows():
        if "status" in df.columns and str(row["status"]) != "OK":
            continue
        k = int(row[k_column]) if k_column is not None else int(fixed_k)
        key = (k, int(row["partition_seed"]), condition_from_alpha(row.get("alpha")))
        lookup[key] = row
    return lookup


def compare_fields(recomputed: dict, source_row, fields: list[str], ctx: str) -> None:
    for f in fields:
        got = float(recomputed[f])
        ref = float(source_row[f])
        assert abs(got - ref) <= TOL, f"{ctx}: {f} mismatch recomputed={got} source={ref}"


def condition_metrics(counts: np.ndarray, global_class_counts: np.ndarray) -> dict:
    """All heterogeneity metrics for one condition from its achieved counts."""
    num_clients = counts.shape[0]
    sizes = counts.sum(axis=1)
    assert (sizes > 0).all(), "a client is empty"
    assert np.array_equal(counts.sum(axis=0), global_class_counts), "class totals differ from global"

    props = counts / sizes[:, None]
    global_dist = global_class_counts / global_class_counts.sum()
    absent = (counts == 0).sum(axis=1)

    pairwise = []
    for i, j in combinations(range(num_clients), 2):
        hd = hellinger(props[i], props[j])
        jsd = jsd_bits(props[i], props[j])
        emd = emd_discrete01(props[i], props[j])
        tv = total_variation(props[i], props[j])
        assert np.isfinite(jsd) and -UNIT_TOL <= jsd <= 1.0 + UNIT_TOL, f"pairwise JSD out of range: {jsd}"
        assert abs(emd - tv) <= 1e-12, f"emd_discrete01 != TV: {emd} vs {tv}"
        pairwise.append({"client_i": i, "client_j": j, "hd": hd, "jsd_bits": jsd,
                         "emd_discrete01": emd, "total_variation": tv})

    to_global = []
    for k in range(num_clients):
        hd = hellinger(props[k], global_dist)
        jsd = jsd_bits(props[k], global_dist)
        emd = emd_discrete01(props[k], global_dist)
        tv = total_variation(props[k], global_dist)
        assert np.isfinite(jsd) and -UNIT_TOL <= jsd <= 1.0 + UNIT_TOL, f"client->global JSD out of range: {jsd}"
        assert abs(emd - tv) <= 1e-12, f"emd_discrete01 != TV: {emd} vs {tv}"
        to_global.append({"client_id": k, "client_size": int(sizes[k]),
                          "absent_class_count": int(absent[k]),
                          "hd_to_global": hd, "jsd_bits_to_global": jsd,
                          "emd_discrete01_to_global": emd, "total_variation_to_global": tv})

    hd_pw = np.array([r["hd"] for r in pairwise])
    jsd_pw = np.array([r["jsd_bits"] for r in pairwise])
    emd_pw = np.array([r["emd_discrete01"] for r in pairwise])
    hd_g = np.array([r["hd_to_global"] for r in to_global])
    jsd_g = np.array([r["jsd_bits_to_global"] for r in to_global])
    emd_g = np.array([r["emd_discrete01_to_global"] for r in to_global])

    size_mean = float(sizes.mean())
    size_std = float(sizes.std())
    summary = {
        "num_clients": num_clients,
        "hd_pairwise_mean": float(hd_pw.mean()), "hd_pairwise_min": float(hd_pw.min()),
        "hd_pairwise_max": float(hd_pw.max()), "hd_pairwise_rms": float(np.sqrt(np.mean(hd_pw ** 2))),
        "mean_client_to_global_hd": float(hd_g.mean()), "max_client_to_global_hd": float(hd_g.max()),
        "jsd_bits_pairwise_mean": float(jsd_pw.mean()), "jsd_bits_pairwise_min": float(jsd_pw.min()),
        "jsd_bits_pairwise_max": float(jsd_pw.max()),
        "mean_client_to_global_jsd_bits": float(jsd_g.mean()), "max_client_to_global_jsd_bits": float(jsd_g.max()),
        "emd_discrete01_pairwise_mean": float(emd_pw.mean()), "emd_discrete01_pairwise_min": float(emd_pw.min()),
        "emd_discrete01_pairwise_max": float(emd_pw.max()),
        "mean_client_to_global_emd_discrete01": float(emd_g.mean()),
        "max_client_to_global_emd_discrete01": float(emd_g.max()),
        "size_min": int(sizes.min()), "size_max": int(sizes.max()),
        "size_mean": size_mean, "size_std": size_std,
        "size_cv": float(size_std / size_mean) if size_mean > 0 else 0.0,
        "size_max_min_ratio": float(sizes.max() / sizes.min()),
        "mean_absent_classes_per_client": float(absent.mean()),
        "max_absent_classes_per_client": int(absent.max()),
    }
    return {"summary": summary, "pairwise": pairwise, "to_global": to_global, "client_sizes": sizes.tolist()}


def main() -> None:
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()):
        raise RuntimeError(f"Output directory already exists and is non-empty: {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    class_order = load_class_order()
    y_train = np.load(PROCESSED_DIR / "y_train.npy")  # labels only; no val/test read
    assert np.issubdtype(y_train.dtype, np.integer), "y_train labels are not integers"
    assert y_train.min() >= 0, "y_train has negative labels"
    assert y_train.max() == NUM_CLASSES - 1, f"y_train max label != {NUM_CLASSES - 1}"
    assert set(np.unique(y_train).tolist()) == set(range(NUM_CLASSES)), "y_train labels are not exactly {0..9}"
    total_records = int(len(y_train))
    global_class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    assert (global_class_counts > 0).all(), "a class has zero training examples"

    sens_lookup = load_summary_lookup(SENS_SUMMARY, k_column="num_clients")
    final_lookup = load_summary_lookup(FINAL_SUMMARY, fixed_k=5)

    summary_rows, pairwise_rows, client_rows = [], [], []
    for K in K_LIST:
        for seed in SEEDS:
            for condition in CONDITIONS:
                alpha = COND_ALPHA[condition]
                if K == 5:
                    counts = k5_counts(seed, condition, y_train, global_class_counts)
                    source = "final_partition_indices"
                else:
                    counts = sensitivity_counts(K, seed, condition, class_order)
                    source = "client_count_sensitivity_csv"

                res = condition_metrics(counts, global_class_counts)
                m = res["summary"]
                ctx = f"K={K} seed={seed} {condition}"

                # Pair/client counts must match the expectation for this K.
                assert len(res["pairwise"]) == K * (K - 1) // 2, f"{ctx}: unexpected pair count"
                assert len(res["to_global"]) == K, f"{ctx}: unexpected client-to-global row count"

                # HD cross-checks against the existing summaries.
                assert (K, seed, condition) in sens_lookup, f"{ctx}: missing in sensitivity summary"
                compare_fields(m, sens_lookup[(K, seed, condition)], HD_CHECK_FIELDS, f"{ctx} vs sensitivity")
                if K == 5:
                    assert (5, seed, condition) in final_lookup, f"{ctx}: missing in final summary"
                    compare_fields(m, final_lookup[(5, seed, condition)], HD_CHECK_FIELDS, f"{ctx} vs final")

                summary_rows.append({"K": K, "seed": seed, "condition": condition, "alpha": alpha,
                                     "total_records": total_records, "source": source, **m})
                for r in res["pairwise"]:
                    pairwise_rows.append({"K": K, "seed": seed, "condition": condition, "alpha": alpha, **r})
                for r in res["to_global"]:
                    client_rows.append({"K": K, "seed": seed, "condition": condition, "alpha": alpha, **r})

    assert len(summary_rows) == 36, f"expected 36 summary rows, got {len(summary_rows)}"

    pd.DataFrame(summary_rows).to_csv(OUT_ROOT / "heterogeneity_summary.csv", index=False)
    pd.DataFrame(pairwise_rows)[["K", "seed", "condition", "alpha", "client_i", "client_j",
                                 "hd", "jsd_bits", "emd_discrete01", "total_variation"]].to_csv(
        OUT_ROOT / "pairwise_distances.csv", index=False)
    pd.DataFrame(client_rows)[["K", "seed", "condition", "alpha", "client_id", "client_size",
                               "absent_class_count", "hd_to_global", "jsd_bits_to_global",
                               "emd_discrete01_to_global", "total_variation_to_global"]].to_csv(
        OUT_ROOT / "client_to_global_distances.csv", index=False)

    metadata = {
        "grid": {"K": K_LIST, "seeds": SEEDS, "conditions": CONDITIONS},
        "class_order": class_order,
        "total_records": total_records,
        "sources": {
            "k5": "actual final client index arrays; counts recomputed from y_train.npy",
            "k10_k20": "existing client_count_sensitivity client_class_counts.csv (not regenerated)",
        },
        "formulas": {
            "hellinger": "HD(p,q) = ||sqrt(p) - sqrt(q)||_2 / sqrt(2); range [0,1]",
            "jsd_bits": "m=(p+q)/2; JSD = 0.5*KL2(p||m) + 0.5*KL2(q||m); log base 2 (bits); "
                        "zero-probability terms contribute zero; this is a DIVERGENCE, not a distance; "
                        "no square root is taken; range [0,1] bits",
            "emd_discrete01": "categorical EMD with a 0/1 ground metric (same class cost 0, different "
                              "class cost 1). Under this ground metric EMD equals total variation "
                              "TV(p,q) = 0.5*sum|p-q|. Verified emd_discrete01 == TV within 1e-12.",
        },
        "jsd_base": 2,
        "no_class_ordering": "No ordering or numeric distance between attack class IDs is assumed; "
                             "the EMD ground metric is purely categorical (0/1).",
        "verification": {
            "hd_fields_checked": HD_CHECK_FIELDS,
            "tolerance": TOL,
            "checked_against": [str(SENS_SUMMARY), str(FINAL_SUMMARY) + " (K=5 only)"],
        },
    }
    with open(OUT_ROOT / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {len(summary_rows)} summary rows, {len(pairwise_rows)} pairwise rows, "
          f"{len(client_rows)} client rows under {OUT_ROOT}")


if __name__ == "__main__":
    main()
