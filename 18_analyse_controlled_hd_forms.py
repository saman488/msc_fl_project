"""
Read-only analysis of the 21 controlled partitions: recompute five Hellinger-
distance forms per (seed, partition) from the saved client indices.

No training, no threshold analysis, no test-set access, no edits to existing
files. Writes exactly one new CSV.

Normalized Hellinger distance:
    H(P,Q) = (1/sqrt(2)) * sqrt(sum_c (sqrt(P(c)) - sqrt(Q(c)))^2)
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from fedartml.function_base import hellinger_distance as fedartml_hd

PROCESSED_DIR = Path("data/processed")
PART_ROOT = Path("data/fl_clients/controlled_partitions")
CONTROLLED_METRICS = Path("results/fl_partitions/controlled_metrics.csv")
OUT_CSV = Path("results/fl_partitions/controlled_hd_forms_verified.csv")

NUM_CLASSES = 10
NUM_CLIENTS = 5
CLIENT_SIZE = 50_000
POOL_SIZE = 250_000
SEEDS = [42, 43, 44]
PARTITIONS = ["alpha_0.01", "alpha_0.05", "alpha_0.1", "alpha_0.5", "alpha_1.0", "alpha_5.0", "iid"]


def normalise(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts.astype(np.float64) / total if total > 0 else counts.astype(np.float64)


def hd(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2.0))


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    y_train = np.load(PROCESSED_DIR / "y_train.npy")  # train labels only; test never read
    stored = pd.read_csv(CONTROLLED_METRICS).set_index(["seed", "partition_type"])

    rows = []
    integrity_ok = True
    for seed in SEEDS:
        for partition in PARTITIONS:
            pdir = PART_ROOT / f"seed_{seed}" / partition
            idx = [np.load(pdir / f"client_{k:02d}_indices.npy") for k in range(NUM_CLIENTS)]

            # Integrity checks.
            n_clients = len(idx)
            sizes = [int(len(i)) for i in idx]
            pooled = np.concatenate(idx)
            checks = (
                n_clients == NUM_CLIENTS
                and all(s == CLIENT_SIZE for s in sizes)
                and len(pooled) == POOL_SIZE
            )
            integrity_ok = integrity_ok and checks

            # Client + pool label distributions.
            client_dists = [normalise(np.bincount(y_train[i], minlength=NUM_CLASSES)) for i in idx]
            pool_dist = normalise(np.bincount(y_train[pooled], minlength=NUM_CLASSES))

            # Pairwise (10 unique pairs) and client-to-pool (5) HDs.
            pairwise = [hd(client_dists[i], client_dists[j]) for i, j in combinations(range(NUM_CLIENTS), 2)]
            to_pool = [hd(client_dists[k], pool_dist) for k in range(NUM_CLIENTS)]

            hd_pairwise_mean = float(np.mean(pairwise))
            hd_pairwise_rms = float(np.sqrt(np.mean(np.square(pairwise))))
            hd_pairwise_max = float(np.max(pairwise))
            hd_pool_mean = float(np.mean(to_pool))
            hd_pool_max = float(np.max(to_pool))

            # FedArtML 0.1.34 multi-client HD on the same 5 client distributions.
            fedartml_rms = float(fedartml_hd(np.array(client_dists)))

            stored_hd = float(stored.loc[(seed, partition), "HD_skew"])

            rows.append({
                "seed": seed, "partition": partition,
                "n_clients": n_clients, "client_sizes_equal_50000": all(s == CLIENT_SIZE for s in sizes),
                "pooled_records": int(len(pooled)),
                "hd_pairwise_mean": hd_pairwise_mean,
                "hd_pairwise_rms": hd_pairwise_rms,
                "hd_pairwise_max": hd_pairwise_max,
                "hd_client_to_pool_mean": hd_pool_mean,
                "hd_client_to_pool_max": hd_pool_max,
                "stored_HD_skew": stored_hd,
                "diff_pool_mean_vs_stored": abs(hd_pool_mean - stored_hd),
                "fedartml_rms": fedartml_rms,
                "diff_manual_rms_vs_fedartml": abs(hd_pairwise_rms - fedartml_rms),
            })

    df = pd.DataFrame(rows)
    df["partition"] = pd.Categorical(df["partition"], categories=PARTITIONS, ordered=True)
    df = df.sort_values(["seed", "partition"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    # ---- verification summary ----
    n_rows = len(df)
    max_pool_diff = float(df["diff_pool_mean_vs_stored"].max())
    max_rms_diff = float(df["diff_manual_rms_vs_fedartml"].max())

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    show = df[["seed", "partition", "n_clients", "pooled_records",
               "hd_pairwise_mean", "hd_pairwise_rms", "hd_pairwise_max",
               "hd_client_to_pool_mean", "hd_client_to_pool_max",
               "stored_HD_skew", "fedartml_rms"]].copy()
    for c in ["hd_pairwise_mean", "hd_pairwise_rms", "hd_pairwise_max",
              "hd_client_to_pool_mean", "hd_client_to_pool_max", "stored_HD_skew", "fedartml_rms"]:
        show[c] = show[c].round(6)
    print("=== 21-ROW CONTROLLED HD FORMS ===")
    print(show.to_string(index=False))

    print("\n=== VERIFICATION ===")
    print(f"  exactly 21 seed-partition rows:            {n_rows == 21}  ({n_rows})")
    print(f"  five clients per partition:                {(df['n_clients'] == 5).all()}")
    print(f"  50,000 records per client:                 {df['client_sizes_equal_50000'].all()}")
    print(f"  250,000 pooled records:                    {(df['pooled_records'] == POOL_SIZE).all()}")
    print(f"  max |recalc pool_mean - stored HD|:        {max_pool_diff:.2e}")
    print(f"  max |manual RMS - FedArtML 0.1.34 RMS|:    {max_rms_diff:.2e}")
    print(f"  overall integrity (clients/sizes/pool):    {integrity_ok}")
    print("  data accessed: data/processed/y_train.npy (train labels) + controlled client indices only")
    print("  NO training, NO threshold analysis, NO test-set (X_test/y_test) access.")
    print(f"\nWritten to {OUT_CSV}")


if __name__ == "__main__":
    main()
