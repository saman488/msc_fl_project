"""
Relate FedAvg validation macro-F1 to partition structure across both HD-matched grids.

Both grids hit the same four HD targets, one using the pure class-wise Dirichlet
partitioner in 25_create_final_partitions.py and one using FedArtML's
dirichlet_method. This script puts the 24 non-IID runs side by side with the
structural properties of the partition each was trained on, so the write-up can say
which property tracks performance and which does not.

IID runs are excluded: measured HD is ~0 and every structural predictor is at its
degenerate value, so including them would inflate every correlation without adding
information about heterogeneity.

Reads the run configs, the run histories and the partition manifests. Writes
results/analysis/run_level_predictors.csv and a correlation table beside it.
Refuses to run if the output directory exists. Trains nothing and reads no
checkpoint or test array.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

GRIDS = [
    ("fedartml",
     Path("results/final_fedavg_k5_hd_fedartml"),
     Path("data/fl_clients/hd_selected_fedartml/k_5")),
    ("ours",
     Path("results/final_fedavg_k5_hd_selected"),
     Path("data/fl_clients/hd_selected_partitions/k_5")),
]

OUT_DIR = Path("results/analysis")
TABLE_PATH = OUT_DIR / "run_level_predictors.csv"
CORRELATION_PATH = OUT_DIR / "run_level_correlations.csv"

LABELS = ["seed_1", "seed_2", "seed_3"]
CONDITIONS = ["hd_0p25", "hd_0p5", "hd_0p75", "hd_0p9"]
BENIGN_CLASS_ID = 0
NUM_CLIENTS = 5
NUM_CLASSES = 10

PREDICTORS = [
    "measured_hd",
    "attack_client_share",
    "absent_cells",
    "classes_absent_somewhere",
    "size_ratio",
    "size_cv",
]

# A predictor that takes one value across a subset cannot correlate with anything.
# Report that rather than a number, since scipy returns nan and a naive reader may
# mistake a blank cell for a weak result.
CONSTANT_TOLERANCE = 1e-12


def describe_run(partitioner: str, results_dir: Path, part_root: Path,
                 label: str, condition: str) -> dict:
    tag = f"fedavg37f_k{NUM_CLIENTS}_seed{label[len('seed_'):]}_{condition}"
    config = json.load(open(results_dir / f"config_{tag}.json"))
    history = pd.read_csv(results_dir / f"history_{tag}.csv")
    manifest = json.load(open(part_root / label / condition / "partition_manifest.json"))

    counts = np.array(manifest["client_class_counts"], dtype=np.int64)
    if counts.shape != (NUM_CLIENTS, NUM_CLASSES):
        raise RuntimeError(f"{part_root/label/condition}: counts shape {counts.shape}")
    sizes = counts.sum(axis=1)
    weights = sizes / sizes.sum()

    holds_attack = counts[:, BENIGN_CLASS_ID + 1:].sum(axis=1) > 0
    absent_cells = int((counts == 0).sum())
    classes_absent_somewhere = int(((counts == 0).any(axis=0)).sum())

    return {
        "partitioner": partitioner,
        "condition": condition,
        "label": label,
        "measured_hd": manifest["hd_pairwise_rms"],
        "best_val_macro_f1": float(config["best_val_macro_f1"]),
        "best_round": int(config["best_round"]),
        "distinct_macro_f1_values": int(history.macro_f1.nunique()),
        "attack_client_share": float(weights[holds_attack].sum()),
        "absent_cells": absent_cells,
        "classes_absent_somewhere": classes_absent_somewhere,
        "size_ratio": float(sizes.max() / sizes.min()),
        "size_cv": float(sizes.std() / sizes.mean()),
    }


def correlate(frame: pd.DataFrame, subset_name: str) -> list[dict]:
    rows = []
    outcome = frame["best_val_macro_f1"].to_numpy()
    for predictor in PREDICTORS:
        values = frame[predictor].to_numpy(dtype=float)
        if np.ptp(values) <= CONSTANT_TOLERANCE:
            rows.append({
                "subset": subset_name, "n": len(frame), "predictor": predictor,
                "spearman_rho": np.nan, "spearman_p": np.nan,
                "pearson_r": np.nan, "pearson_p": np.nan,
                "note": f"no variance (constant at {values[0]:g}); correlation undefined",
            })
            continue
        rho, p_spearman = spearmanr(values, outcome)
        r, p_pearson = pearsonr(values, outcome)
        rows.append({
            "subset": subset_name, "n": len(frame), "predictor": predictor,
            "spearman_rho": float(rho), "spearman_p": float(p_spearman),
            "pearson_r": float(r), "pearson_p": float(p_pearson),
            "note": "",
        })
    return rows


def print_correlations(rows: list[dict], title: str) -> None:
    print(f"--- {title} ---")
    header = (f"{'predictor':>26}{'n':>4}{'spearman':>11}{'p':>9}"
              f"{'pearson':>11}{'p':>9}")
    print(header)
    print("-" * len(header))
    ranked = sorted(rows, key=lambda r: (np.isnan(r["spearman_rho"]),
                                         -abs(r["spearman_rho"]) if not np.isnan(r["spearman_rho"]) else 0))
    for row in ranked:
        if row["note"]:
            print(f"{row['predictor']:>26}{row['n']:>4}   {row['note']}")
            continue
        print(f"{row['predictor']:>26}{row['n']:>4}{row['spearman_rho']:>11.4f}"
              f"{row['spearman_p']:>9.4f}{row['pearson_r']:>11.4f}{row['pearson_p']:>9.4f}")
    print()


def main() -> None:
    if OUT_DIR.exists():
        raise RuntimeError(f"Refusing to run; {OUT_DIR} already exists.")

    rows = []
    for partitioner, results_dir, part_root in GRIDS:
        if not results_dir.is_dir():
            raise RuntimeError(f"Missing {results_dir}")
        for condition in CONDITIONS:
            for label in LABELS:
                rows.append(describe_run(partitioner, results_dir, part_root, label, condition))

    table = pd.DataFrame(rows)
    if len(table) != 24:
        raise RuntimeError(f"expected 24 non-IID runs, built {len(table)}")

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    table.to_csv(TABLE_PATH, index=False)
    print(f"Wrote {TABLE_PATH} ({len(table)} rows)")
    print()

    display = table[["partitioner", "condition", "label", "measured_hd",
                     "best_val_macro_f1", "attack_client_share", "absent_cells",
                     "classes_absent_somewhere", "size_ratio", "size_cv"]]
    print(display.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    correlations = []
    correlations += correlate(table[table.partitioner == "fedartml"], "fedartml only")
    correlations += correlate(table[table.partitioner == "ours"], "ours only")
    correlations += correlate(table, "pooled")
    pd.DataFrame(correlations).to_csv(CORRELATION_PATH, index=False)
    print(f"Wrote {CORRELATION_PATH}")
    print()

    print("CORRELATION OF best_val_macro_f1 AGAINST EACH PREDICTOR")
    print("(ranked by absolute Spearman rho)")
    print()
    for subset in ("fedartml only", "ours only", "pooled"):
        print_correlations([r for r in correlations if r["subset"] == subset], subset)

    print("HD 0.75 vs HD 0.90, FedArtML: the rise HD does not explain")
    band = table[(table.partitioner == "fedartml")
                 & (table.condition.isin(["hd_0p75", "hd_0p9"]))]
    print(band[["condition", "label", "measured_hd", "best_val_macro_f1",
                "attack_client_share", "absent_cells"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    for condition in ("hd_0p75", "hd_0p9"):
        subset = band[band.condition == condition]
        print(f"  {condition}: mean macro-F1 {subset.best_val_macro_f1.mean():.4f}, "
              f"mean attack share {subset.attack_client_share.mean():.4f}, "
              f"mean HD {subset.measured_hd.mean():.4f}")


if __name__ == "__main__":
    main()
