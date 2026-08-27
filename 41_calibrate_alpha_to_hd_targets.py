"""
Calibrate alpha to HD targets.

WHY
    Jimenez-Gutierrez et al. did not choose alpha values and then observe the
    resulting heterogeneity. They chose HD levels first -- {0, 0.25, 0.5, 0.75,
    0.9} -- and searched for the alpha values that produce them on their data,
    arriving at {1000, 6, 1, 0.3, 0.03}. Those alpha values are therefore a
    calibration result specific to their datasets and their client count, not a
    general recipe.

    This script performs the same calibration on this dataset at K = 5: for each
    of their HD targets, it searches for the alpha that reaches it here.

WHAT IT DOES
    For each HD target, a bisection search over alpha. At each candidate alpha
    the partition is generated with SEEDS_PER_ALPHA different seeds and the mean
    Hellinger distance recorded. The search narrows on the alpha whose mean HD
    is closest to the target.

    The chosen alpha is then re-measured with VERIFY_SEEDS seeds so that the
    spread -- not just the mean -- is reported. The spread matters: if it is
    wide, a single partition drawn at that alpha cannot be assumed to sit at the
    target, and alpha is not usable as a dial for setting heterogeneity.

WHAT IT DOES NOT DO
    - No model is created, loaded, or trained.
    - Nothing existing is read from or written to except the training labels.

WHAT IT READS
    data/processed/y_train.npy      (training labels only)
    configs/label_mapping.json      (class order)
    25_create_final_partitions.py   (imported for its partition and HD functions)

WHAT IT WRITES  (all under a NEW directory)
    results/alpha_calibration/calibration_summary.csv   one row per HD target
    results/alpha_calibration/search_trace.csv          every alpha tried
    results/alpha_calibration/verify_runs.csv           per-seed HD at each chosen alpha
    results/alpha_calibration/partitions/               saved partitions (if SAVE_PARTITIONS)

AGGREGATION
    HD is reported as the root mean square over client pairs, because that is
    what FedArtML returns and therefore what the source study's HD levels refer
    to. The pairwise mean is recorded alongside for reference; the two are not
    interchangeable.

USAGE
    env/bin/python 41_calibrate_alpha_to_hd_targets.py

    Run from the project root. Takes several minutes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
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

OUT_DIR = Path("results/alpha_calibration")
SUMMARY_PATH = OUT_DIR / "calibration_summary.csv"
TRACE_PATH = OUT_DIR / "search_trace.csv"
VERIFY_PATH = OUT_DIR / "verify_runs.csv"
PARTITION_DIR = OUT_DIR / "partitions"

K = 5

# The source study's HD levels. HD = 0 is the IID condition and is not searched
# for -- it is produced by even splitting, not by any value of alpha.
HD_TARGETS = [0.25, 0.50, 0.75, 0.90]

# Search bounds on alpha. Wide enough to cover the source study's own range
# (0.03 to 1000) with room either side.
ALPHA_LOW = 0.005
ALPHA_HIGH = 5000.0
BISECTION_STEPS = 14

# Seeds used to estimate mean HD at each candidate alpha during the search.
SEEDS_PER_ALPHA = 12
SEARCH_SEED_BASE = 1000

# Seeds used to characterise the chosen alpha once found. More than the search
# uses, because this is the number that gets reported.
VERIFY_SEEDS = 30
VERIFY_SEED_BASE = 2000

# Partitions saved at the study's own seeds, for possible later use.
SAVE_PARTITIONS = True
SAVE_SEEDS = [42, 43, 44]


# ----------------------------------------------------------------------------
# Import the study's own partitioning and HD functions
# ----------------------------------------------------------------------------

def load_source_module():
    """Import 25_create_final_partitions.py by path.

    The filename begins with a digit so a normal import statement is invalid.
    The module guards its main() with `if __name__ == "__main__"`, so importing
    it does not run it.
    """
    if not SOURCE_SCRIPT.exists():
        sys.exit(f"ERROR: cannot find {SOURCE_SCRIPT}. Run from the project root.")
    spec = importlib.util.spec_from_file_location("partition_source", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------

def client_proportions(client_indices, y_train, num_classes):
    """Per-client class proportions; each row sums to 1."""
    rows = []
    for idx in client_indices:
        counts = np.bincount(y_train[idx], minlength=num_classes).astype(float)
        total = counts.sum()
        rows.append(counts / total if total > 0 else counts)
    return np.array(rows)


def hd_mean_and_rms(props, hellinger_fn):
    """Mean and RMS Hellinger distance over all client pairs."""
    pairs = np.array([
        hellinger_fn(props[i], props[j])
        for i, j in combinations(range(len(props)), 2)
    ], dtype=float)
    return float(pairs.mean()), float(np.sqrt((pairs ** 2).mean()))


def measure_alpha(src, y_train, num_classes, alpha, seeds):
    """Mean and RMS HD at one alpha, across the given seeds."""
    means, rmss, absents = [], [], []
    for seed in seeds:
        client_indices = src.partition_noniid(y_train, num_classes, alpha, seed)
        props = client_proportions(client_indices, y_train, num_classes)
        m, r = hd_mean_and_rms(props, src.hellinger_distance)
        means.append(m)
        rmss.append(r)
        absents.append(int((props == 0).sum()))
    return np.array(means), np.array(rmss), np.array(absents)


# ----------------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------------

def bisect_for_target(src, y_train, num_classes, target, trace_rows):
    """Find the alpha whose mean RMS HD is closest to `target`.

    HD falls as alpha rises, so the search is a bisection on log(alpha):
    if the midpoint gives HD above the target, alpha must increase.
    """
    lo, hi = np.log10(ALPHA_LOW), np.log10(ALPHA_HIGH)
    search_seeds = [SEARCH_SEED_BASE + i for i in range(SEEDS_PER_ALPHA)]

    best = None
    for step in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        alpha = float(10 ** mid)
        _, rms, _ = measure_alpha(src, y_train, num_classes, alpha, search_seeds)
        achieved = float(rms.mean())
        gap = abs(achieved - target)

        trace_rows.append({
            "target_hd": target,
            "step": step + 1,
            "alpha": alpha,
            "hd_rms_mean": achieved,
            "hd_rms_sd": float(rms.std(ddof=1)),
            "gap": gap,
        })
        print(f"    step {step + 1:>2}  alpha={alpha:>10.4f}  "
              f"HD_rms={achieved:.4f}  gap={gap:.4f}")

        if best is None or gap < best[1]:
            best = (alpha, gap)

        if achieved > target:
            lo = mid          # too heterogeneous: raise alpha
        else:
            hi = mid          # too homogeneous: lower alpha

    return best[0]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    for path in (SUMMARY_PATH, TRACE_PATH, VERIFY_PATH):
        if path.exists():
            sys.exit(f"ERROR: {path} already exists. Refusing to overwrite.")

    src = load_source_module()

    y_path = PROCESSED_DIR / "y_train.npy"
    if not y_path.exists():
        sys.exit(f"ERROR: cannot find {y_path}.")
    y_train = np.load(y_path)

    name_to_id = json.load(open(LABEL_MAP))
    num_classes = len(name_to_id)
    class_order = [None] * num_classes
    for name, cid in name_to_id.items():
        class_order[int(cid)] = name

    print(f"Loaded {len(y_train):,} training labels, {num_classes} classes, K = {K}.")
    print(f"HD targets (from the source study): {HD_TARGETS}")
    print(f"Search: {BISECTION_STEPS} bisection steps x {SEEDS_PER_ALPHA} seeds.")
    print(f"Verification: {VERIFY_SEEDS} seeds at each chosen alpha.\n")

    trace_rows, verify_rows, summary_rows = [], [], []
    started = time.time()

    for target in HD_TARGETS:
        print(f"TARGET HD = {target}")
        alpha = bisect_for_target(src, y_train, num_classes, target, trace_rows)

        verify_seeds = [VERIFY_SEED_BASE + i for i in range(VERIFY_SEEDS)]
        means, rmss, absents = measure_alpha(
            src, y_train, num_classes, alpha, verify_seeds
        )

        for seed, m, r, a in zip(verify_seeds, means, rmss, absents):
            verify_rows.append({
                "target_hd": target, "alpha": alpha, "seed": seed,
                "hd_pairwise_mean": m, "hd_pairwise_rms": r,
                "absent_class_slots": a,
            })

        # How often does a single partition at this alpha land within 0.05 of
        # the target? This is the number that says whether alpha is usable as a
        # dial, as opposed to merely centring on the right value.
        within = float(np.mean(np.abs(rmss - target) <= 0.05))

        summary_rows.append({
            "target_hd": target,
            "alpha_found": alpha,
            "hd_rms_mean": float(rmss.mean()),
            "hd_rms_sd": float(rmss.std(ddof=1)),
            "hd_rms_min": float(rmss.min()),
            "hd_rms_max": float(rmss.max()),
            "hd_pairwise_mean_avg": float(means.mean()),
            "frac_within_0p05_of_target": within,
            "absent_slots_mean": float(absents.mean()),
            "verify_seeds": VERIFY_SEEDS,
        })

        print(f"  -> alpha = {alpha:.4f}")
        print(f"     HD_rms over {VERIFY_SEEDS} seeds: mean {rmss.mean():.4f}, "
              f"sd {rmss.std(ddof=1):.4f}, range {rmss.min():.4f}-{rmss.max():.4f}")
        print(f"     within 0.05 of target: {within:.0%} of partitions\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(trace_rows).to_csv(TRACE_PATH, index=False)
    pd.DataFrame(verify_rows).to_csv(VERIFY_PATH, index=False)

    # --- Optionally save partitions at the study's own seeds ----------------
    if SAVE_PARTITIONS:
        PARTITION_DIR.mkdir(parents=True, exist_ok=True)
        for row in summary_rows:
            target, alpha = row["target_hd"], row["alpha_found"]
            tag = f"hd_{str(target).replace('.', 'p')}"
            for seed in SAVE_SEEDS:
                client_indices = src.partition_noniid(y_train, num_classes, alpha, seed)
                props = client_proportions(client_indices, y_train, num_classes)
                _, rms = hd_mean_and_rms(props, src.hellinger_distance)
                d = PARTITION_DIR / tag / f"seed_{seed}"
                d.mkdir(parents=True, exist_ok=True)
                for cid, idx in enumerate(client_indices):
                    np.save(d / f"client_{cid}_indices.npy", idx)
                json.dump(
                    {
                        "target_hd": target, "alpha": alpha, "seed": seed, "K": K,
                        "achieved_hd_rms": rms,
                        "client_sizes": [int(len(i)) for i in client_indices],
                        "class_order": class_order,
                        "note": "Calibrated to an HD target, not to a chosen alpha. "
                                "Achieved HD is this partition's own value, which "
                                "may differ substantially from the target.",
                    },
                    open(d / "partition_meta.json", "w"), indent=2,
                )
        print(f"Saved partitions under {PARTITION_DIR}")

    print(f"\nWrote {SUMMARY_PATH}")
    print(f"Wrote {TRACE_PATH}")
    print(f"Wrote {VERIFY_PATH}")
    print(f"\nElapsed: {time.time() - started:.0f}s\n")

    print("CALIBRATION RESULT")
    print(summary[[
        "target_hd", "alpha_found", "hd_rms_mean", "hd_rms_sd",
        "hd_rms_min", "hd_rms_max", "frac_within_0p05_of_target",
    ]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("Source study, for comparison (their datasets, K = 30):")
    print("  HD 0.25 -> alpha 6      HD 0.50 -> alpha 1")
    print("  HD 0.75 -> alpha 0.3    HD 0.90 -> alpha 0.03")


if __name__ == "__main__":
    main()
