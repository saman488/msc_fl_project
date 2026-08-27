"""
Alpha -> Hellinger Distance mapping sweep (measurement only).

WHAT THIS DOES
    Sweeps the Dirichlet concentration parameter alpha over a range of values,
    partitions the TRAINING LABELS across K clients for each (alpha, seed) pair,
    and records the resulting Hellinger Distance. The point is to establish, for
    THIS dataset, which alpha values are needed to reach given HD levels -- the
    calibration step that Jimenez-Gutierrez et al. performed on their own data.

WHAT THIS DOES NOT DO
    - No model is created, loaded, or trained.
    - No partition files are written to disk. Partitions are held in memory,
      measured, and discarded.
    - Nothing under data/ is written to.
    - Nothing existing is modified or overwritten.

WHAT IT READS
    data/processed/y_train.npy      (training labels only)
    configs/label_mapping.json      (class order)

WHAT IT WRITES
    results/alpha_hd_sweep/alpha_hd_sweep.csv        (one row per alpha x seed)
    results/alpha_hd_sweep/alpha_hd_summary.csv      (one row per alpha, seeds averaged)

    Both under a NEW directory. The script refuses to run if that directory
    already contains these files.

PROVENANCE
    The partitioning and Hellinger functions are imported from
    25_create_final_partitions.py rather than reimplemented, so the mapping is
    measured with exactly the same code that produced the study partitions.
    That file has an `if __name__ == "__main__"` guard, so importing it does
    not trigger a run.

USAGE
    python 40_alpha_hd_mapping_sweep.py

    Run from the project root (the folder containing 25_create_final_partitions.py).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SOURCE_SCRIPT = Path("25_create_final_partitions.py")
PROCESSED_DIR = Path("data/processed")
LABEL_MAP = Path("configs/label_mapping.json")

# NEW output directory. Nothing existing lives here.
OUT_DIR = Path("results/alpha_hd_sweep")
ROWS_PATH = OUT_DIR / "alpha_hd_sweep.csv"
SUMMARY_PATH = OUT_DIR / "alpha_hd_summary.csv"

K = 5
PARTITION_SEEDS = [42, 43, 44]

# Alpha grid. The three study values (0.1, 0.5, 1.0) are included so their
# recomputed HD can be checked against the values already recorded -- if they
# do not match, something is wrong and the sweep should not be trusted.
ALPHA_GRID = [
    0.02, 0.03, 0.05, 0.07,
    0.10,
    0.15, 0.20, 0.25, 0.30, 0.40,
    0.50,
    0.70,
    1.00,
    2.00, 5.00, 10.00,
]


# ----------------------------------------------------------------------------
# Import the study's own partitioning + HD functions
# ----------------------------------------------------------------------------

def load_source_module():
    """Import 25_create_final_partitions.py by path.

    The filename starts with a digit, so a normal `import` statement is not
    valid Python. importlib loads it by path instead. The module's main() is
    guarded by `if __name__ == "__main__"`, so this does not run it.
    """
    if not SOURCE_SCRIPT.exists():
        sys.exit(
            f"ERROR: cannot find {SOURCE_SCRIPT}.\n"
            f"Run this script from the project root."
        )
    spec = importlib.util.spec_from_file_location("partition_source", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# Hellinger aggregation
# ----------------------------------------------------------------------------

def client_proportions(client_indices, y_train, num_classes):
    """Per-client class proportions. Each row sums to 1."""
    props = []
    for idx in client_indices:
        counts = np.bincount(y_train[idx], minlength=num_classes).astype(float)
        total = counts.sum()
        props.append(counts / total if total > 0 else counts)
    return np.array(props)


def aggregate_hd(props, hellinger_fn):
    """Mean and RMS Hellinger distance over all client pairs.

    Reported separately because they are not interchangeable: FedArtML -- the
    tool the source study used -- returns the RMS, so RMS is the figure to
    compare against their published HD levels.
    """
    pairs = [
        hellinger_fn(props[i], props[j])
        for i, j in combinations(range(len(props)), 2)
    ]
    pairs = np.asarray(pairs, dtype=float)
    return float(pairs.mean()), float(np.sqrt((pairs ** 2).mean())), float(pairs.min()), float(pairs.max())


def hd_to_global(props, global_props, hellinger_fn):
    """Mean Hellinger distance from each client to the global class distribution."""
    return float(np.mean([hellinger_fn(p, global_props) for p in props]))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    # --- Refuse to clobber -------------------------------------------------
    for path in (ROWS_PATH, SUMMARY_PATH):
        if path.exists():
            sys.exit(
                f"ERROR: {path} already exists. Refusing to overwrite.\n"
                f"Move or delete it if you want to re-run the sweep."
            )

    src = load_source_module()

    # --- Load labels only --------------------------------------------------
    y_path = PROCESSED_DIR / "y_train.npy"
    if not y_path.exists():
        sys.exit(f"ERROR: cannot find {y_path}.")
    y_train = np.load(y_path)

    name_to_id = json.load(open(LABEL_MAP))
    num_classes = len(name_to_id)
    class_order = [None] * num_classes
    for name, cid in name_to_id.items():
        class_order[int(cid)] = name

    print(f"Loaded {len(y_train):,} training labels, {num_classes} classes.")
    print(f"K = {K}, seeds = {PARTITION_SEEDS}")
    print(f"Sweeping {len(ALPHA_GRID)} alpha values "
          f"({len(ALPHA_GRID) * len(PARTITION_SEEDS)} partitions total).\n")

    global_counts = np.bincount(y_train, minlength=num_classes).astype(float)
    global_props = global_counts / global_counts.sum()

    # --- Sweep -------------------------------------------------------------
    rows = []
    for alpha in ALPHA_GRID:
        for seed in PARTITION_SEEDS:
            client_indices = src.partition_noniid(y_train, num_classes, alpha, seed)
            props = client_proportions(client_indices, y_train, num_classes)

            hd_mean, hd_rms, hd_min, hd_max = aggregate_hd(props, src.hellinger_distance)
            hd_global = hd_to_global(props, global_props, src.hellinger_distance)

            sizes = np.array([len(idx) for idx in client_indices], dtype=float)
            absent = int(sum(int((props[i] == 0).sum()) for i in range(len(props))))

            rows.append({
                "alpha": alpha,
                "seed": seed,
                "K": K,
                "hd_pairwise_mean": hd_mean,
                "hd_pairwise_rms": hd_rms,
                "hd_pairwise_min": hd_min,
                "hd_pairwise_max": hd_max,
                "hd_to_global_mean": hd_global,
                "client_size_min": int(sizes.min()),
                "client_size_max": int(sizes.max()),
                "client_size_cv": float(sizes.std() / sizes.mean()),
                "size_max_min_ratio": float(sizes.max() / sizes.min()) if sizes.min() > 0 else float("inf"),
                "absent_class_slots": absent,
            })
            print(f"  alpha={alpha:<6} seed={seed}  "
                  f"HD_mean={hd_mean:.4f}  HD_rms={hd_rms:.4f}  "
                  f"absent_slots={absent}")

    df = pd.DataFrame(rows)

    # --- Per-alpha summary -------------------------------------------------
    summary = (
        df.groupby("alpha")
          .agg(
              hd_mean_avg=("hd_pairwise_mean", "mean"),
              hd_mean_sd=("hd_pairwise_mean", "std"),
              hd_rms_avg=("hd_pairwise_rms", "mean"),
              hd_rms_sd=("hd_pairwise_rms", "std"),
              hd_to_global_avg=("hd_to_global_mean", "mean"),
              size_ratio_avg=("size_max_min_ratio", "mean"),
              absent_slots_avg=("absent_class_slots", "mean"),
          )
          .reset_index()
          .sort_values("alpha")
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROWS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"\nWrote {ROWS_PATH}")
    print(f"Wrote {SUMMARY_PATH}\n")

    print("Alpha -> HD mapping (mean over 3 seeds):")
    print(summary[["alpha", "hd_mean_avg", "hd_rms_avg", "absent_slots_avg"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
