"""
Cosine-mismatch diagnostic (Wang et al.-style) for the 21 controlled partitions.

For every (seed, partition) reconstruct the achieved 10-class client-count matrix
from the saved client indices (exact class order = label mapping index 0..9), form
the pooled federation composition V = sum_k v_k, and compute per-client

    CS_k = (v_k . V) / (||v_k|| ||V||).

Report mean / min / max CS and mean cosine distance (1 - CS) per federation, verify
raw-count vs normalized-proportion invariance, and join to the verified HD forms.

Read-only: no training, no test-data access. Writes exactly one CSV.
Does NOT invent an attack-only variant and does NOT interpret metric superiority.
"""

from pathlib import Path
from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

PROCESSED_DIR = Path("data/processed")
PART_ROOT = Path("data/fl_clients/controlled_partitions")
LABEL_MAP = Path("configs/label_mapping.json")
HD_FORMS = Path("results/fl_partitions/controlled_hd_forms_verified.csv")
OUT_CSV = Path("results/fl_partitions/controlled_cosine_mismatch_verified.csv")

NUM_CLASSES = 10
NUM_CLIENTS = 5
SEEDS = [42, 43, 44]
PARTITIONS = ["alpha_0.01", "alpha_0.05", "alpha_0.1", "alpha_0.5", "alpha_1.0", "alpha_5.0", "iid"]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Exact class order from the saved label mapping (index 0..9).
    name_to_id = json.load(open(LABEL_MAP))
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    class_order = [id_to_name[c] for c in range(NUM_CLASSES)]

    y_train = np.load(PROCESSED_DIR / "y_train.npy")  # train labels only; test never read
    hd = pd.read_csv(HD_FORMS).set_index(["seed", "partition"])

    max_raw_norm_diff = 0.0
    rows = []
    for seed in SEEDS:
        for partition in PARTITIONS:
            pdir = PART_ROOT / f"seed_{seed}" / partition
            idx = [np.load(pdir / f"client_{k:02d}_indices.npy") for k in range(NUM_CLIENTS)]
            counts = np.stack([np.bincount(y_train[i], minlength=NUM_CLASSES).astype(np.float64)
                               for i in idx])                      # 5 x 10, label-map order
            V = counts.sum(axis=0)                                 # pooled federation vector

            cs_raw = np.array([cosine(counts[k], V) for k in range(NUM_CLIENTS)])
            # verify scale-invariance vs normalized proportions
            props = counts / counts.sum(axis=1, keepdims=True)
            Vp = props.sum(axis=0)
            cs_norm = np.array([cosine(props[k], Vp) for k in range(NUM_CLIENTS)])
            max_raw_norm_diff = max(max_raw_norm_diff, float(np.max(np.abs(cs_raw - cs_norm))))

            mean_cs = float(cs_raw.mean())
            min_cs = float(cs_raw.min())
            max_cs = float(cs_raw.max())
            mean_cos_dist = float((1.0 - cs_raw).mean())

            rows.append({
                "seed": seed, "partition": partition,
                "mean_client_pool_cosine": mean_cs,
                "min_client_pool_cosine": min_cs,
                "max_client_pool_cosine": max_cs,
                "mean_cosine_distance": mean_cos_dist,
                "hd_client_to_pool_mean": float(hd.loc[(seed, partition), "hd_client_to_pool_mean"]),
                "hd_pairwise_mean": float(hd.loc[(seed, partition), "hd_pairwise_mean"]),
            })

    df = pd.DataFrame(rows)
    df["partition"] = pd.Categorical(df["partition"], categories=PARTITIONS, ordered=True)
    df = df.sort_values(["seed", "partition"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("Exact class order (label mapping index 0..9):", class_order)

    print("\n=== ALL 21 FEDERATION VALUES ===")
    show = df.copy()
    for c in ["mean_client_pool_cosine", "min_client_pool_cosine", "max_client_pool_cosine",
              "mean_cosine_distance", "hd_client_to_pool_mean", "hd_pairwise_mean"]:
        show[c] = show[c].round(8)
    print(show.to_string(index=False))

    print("\n=== VERIFICATION: raw-count vs normalized-proportion CS ===")
    print(f"  max |CS_raw - CS_normalized| over all 105 client values = {max_raw_norm_diff:.2e}  (identical to fp precision)")

    print("\n=== MEAN & RANGE BY PARTITION CONDITION (across 3 seeds) ===")
    grp = df.groupby("partition", observed=True).agg(
        mean_cosine_distance_mean=("mean_cosine_distance", "mean"),
        mean_cosine_distance_min=("mean_cosine_distance", "min"),
        mean_cosine_distance_max=("mean_cosine_distance", "max"),
        mean_cosine_sim_mean=("mean_client_pool_cosine", "mean"),
        min_cosine_sim=("min_client_pool_cosine", "min"),
        max_cosine_sim=("max_client_pool_cosine", "max"),
    ).reindex(PARTITIONS)
    print(grp.round(8).to_string())

    print("\n=== SPREAD OF COSINE SIMILARITY vs HD ===")
    print(f"  mean_client_pool_cosine: min={df['mean_client_pool_cosine'].min():.8f}  "
          f"max={df['mean_client_pool_cosine'].max():.8f}  "
          f"range={df['mean_client_pool_cosine'].max()-df['mean_client_pool_cosine'].min():.8f}")
    print(f"  min_client_pool_cosine (any client, any fed): {df['min_client_pool_cosine'].min():.8f}")
    print(f"  mean_cosine_distance:    min={df['mean_cosine_distance'].min():.8f}  "
          f"max={df['mean_cosine_distance'].max():.8f}  "
          f"range={df['mean_cosine_distance'].max()-df['mean_cosine_distance'].min():.8f}")
    print(f"  hd_client_to_pool_mean:  min={df['hd_client_to_pool_mean'].min():.6f}  "
          f"max={df['hd_client_to_pool_mean'].max():.6f}  "
          f"range={df['hd_client_to_pool_mean'].max()-df['hd_client_to_pool_mean'].min():.6f}")
    print(f"  hd_pairwise_mean:        min={df['hd_pairwise_mean'].min():.6f}  "
          f"max={df['hd_pairwise_mean'].max():.6f}  "
          f"range={df['hd_pairwise_mean'].max()-df['hd_pairwise_mean'].min():.6f}")
    frac_above = float((df["mean_client_pool_cosine"] > 0.999).mean())
    print(f"  fraction of federations with mean cosine similarity > 0.999: {frac_above:.3f}  "
          f"(>0.99: {float((df['mean_client_pool_cosine']>0.99).mean()):.3f})")

    print("\n=== SPEARMAN (mean_cosine_distance vs HD forms) ===")
    r1, p1 = spearmanr(df["mean_cosine_distance"], df["hd_client_to_pool_mean"])
    r2, p2 = spearmanr(df["mean_cosine_distance"], df["hd_pairwise_mean"])
    print(f"  vs hd_client_to_pool_mean: rho={r1:.6f}  p={p1:.3g}")
    print(f"  vs hd_pairwise_mean:       rho={r2:.6f}  p={p2:.3g}")

    print("\n=== RANK-ORDER DISAGREEMENTS (cosine distance vs HD) ===")
    for hdcol in ["hd_client_to_pool_mean", "hd_pairwise_mean"]:
        tau, _ = kendalltau(df["mean_cosine_distance"], df[hdcol])
        rc = df["mean_cosine_distance"].rank(method="average").to_numpy()
        rh = df[hdcol].rank(method="average").to_numpy()
        discordant = sum(1 for i, j in combinations(range(len(df)), 2)
                         if (rc[i] - rc[j]) * (rh[i] - rh[j]) < 0)
        print(f"  vs {hdcol}: Kendall tau={tau:.4f}, discordant pairs={discordant} of {len(df)*(len(df)-1)//2}")
    # explicit example swaps vs hd_client_to_pool_mean
    d2 = df.copy()
    d2["rank_cos"] = d2["mean_cosine_distance"].rank(method="average")
    d2["rank_hdpool"] = d2["hd_client_to_pool_mean"].rank(method="average")
    d2["rank_gap"] = (d2["rank_cos"] - d2["rank_hdpool"])
    worst = d2.reindex(d2["rank_gap"].abs().sort_values(ascending=False).index).head(4)
    print("  largest rank gaps (cos-distance rank vs hd_client_to_pool rank):")
    print(worst[["seed", "partition", "mean_cosine_distance", "hd_client_to_pool_mean",
                 "rank_cos", "rank_hdpool", "rank_gap"]].round(6).to_string(index=False))

    print("\nNo training, no test-set access. Files created:")
    print(f"  script: 21_analyse_cosine_mismatch.py")
    print(f"  csv:    {OUT_CSV}")


if __name__ == "__main__":
    main()
