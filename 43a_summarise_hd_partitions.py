"""
Summarise the fifteen HD-selected partitions: selection provenance and full metrics.

Writes two files under results/hd_selection/:

selection_summary.csv
    Regenerated. The version written by 42_select_partitions_by_hd.py pointed at the
    pre-rename directories and had no IID rows, so it was stale and misleading. This
    version reflects the current layout and records how close each partition landed
    to its target HD, which is the one thing seed_mapping.csv does not carry.

heterogeneity_summary_hd_selected.csv
    Every metric 28_build_heterogeneity_summary.py computes -- Hellinger, Jensen
    Shannon in bits, total variation, EMD under the 0/1 ground metric, and the size
    diagnostics -- in that script's exact column order, so this file and
    results/partition_validation/heterogeneity_summary/heterogeneity_summary.csv can
    be concatenated and compared row for row. Script 28's own functions are imported
    and reused rather than reimplemented, so the two files cannot drift apart.

The 'seed' column holds the directory label (1, 2, 3), not the generating seed, so
its dtype matches script 28's output. The generating seed is in selection_summary.csv
and in each partition manifest. The 'source' column distinguishes the two files once
concatenated.

Reads the partitions, their manifests, y_train and the label mapping. Writes only the
two files above. Script 28 and its output are never modified.
"""

from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
LABEL_MAP = Path("configs/label_mapping.json")
PART_ROOT = Path("data/fl_clients/hd_selected_partitions/k_5")
HETEROGENEITY_SOURCE = Path("28_build_heterogeneity_summary.py")
REFERENCE_SUMMARY = Path(
    "results/partition_validation/heterogeneity_summary/heterogeneity_summary.csv")

OUT_DIR = Path("results/hd_selection")
SELECTION_PATH = OUT_DIR / "selection_summary.csv"
METRICS_PATH = OUT_DIR / "heterogeneity_summary_hd_selected.csv"

NUM_CLIENTS = 5
NUM_CLASSES = 10
DIRECTORY_LABELS = ["seed_1", "seed_2", "seed_3"]
CONDITIONS = ["iid", "hd_0p25", "hd_0p5", "hd_0p75", "hd_0p9"]

# IID is the HD = 0 level in the source study's framing, so it gets a target too.
TARGET_HD = {"iid": 0.0, "hd_0p25": 0.25, "hd_0p5": 0.50,
             "hd_0p75": 0.75, "hd_0p9": 0.90}

SOURCE_TAG = "hd_selected_partition_indices"


def load_heterogeneity_module():
    """Import 28_build_heterogeneity_summary.py by path for its metric functions.

    The filename starts with a digit so a normal import will not work. The module
    guards main() behind __main__, so importing it computes nothing and writes
    nothing.
    """
    if not HETEROGENEITY_SOURCE.exists():
        raise RuntimeError(f"Cannot find {HETEROGENEITY_SOURCE}; run from the project root.")
    spec = importlib.util.spec_from_file_location("heterogeneity_source", HETEROGENEITY_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["heterogeneity_source"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "condition_metrics"):
        raise RuntimeError(f"{HETEROGENEITY_SOURCE} does not define condition_metrics")
    if module.NUM_CLASSES != NUM_CLASSES:
        raise RuntimeError(f"{HETEROGENEITY_SOURCE} uses NUM_CLASSES={module.NUM_CLASSES}")
    return module


def reference_columns() -> list[str]:
    """The exact column order script 28 wrote, taken from its output file.

    Reading the real header rather than hardcoding a list means a mismatch shows up
    here instead of silently producing a file that will not concatenate.
    """
    if not REFERENCE_SUMMARY.exists():
        raise RuntimeError(f"Cannot find {REFERENCE_SUMMARY} to match its columns")
    return list(pd.read_csv(REFERENCE_SUMMARY, nrows=0).columns)


def load_client_counts(partition_dir: Path, y_train, global_class_counts):
    """Read one partition's index files and return its client class counts.

    The same integrity checks the training scripts make, repeated here because this
    summary is only meaningful if the partition it describes is intact.
    """
    files = sorted(partition_dir.glob("client_*_indices.npy"))
    if len(files) != NUM_CLIENTS:
        raise RuntimeError(f"{partition_dir}: {len(files)} client files, expected {NUM_CLIENTS}")
    client_indices = [np.load(partition_dir / f"client_{k:02d}_indices.npy")
                      for k in range(NUM_CLIENTS)]
    if any(len(indices) == 0 for indices in client_indices):
        raise RuntimeError(f"{partition_dir}: a client is empty")

    all_assigned = np.concatenate(client_indices)
    total_records = len(y_train)
    if len(all_assigned) != total_records:
        raise RuntimeError(f"{partition_dir}: indices do not cover all {total_records} records")
    if not np.array_equal(np.sort(all_assigned), np.arange(total_records)):
        raise RuntimeError(f"{partition_dir}: coverage is not exactly 0..N-1")

    counts = np.stack([np.bincount(y_train[indices], minlength=NUM_CLASSES)
                       for indices in client_indices])
    if not np.array_equal(counts.sum(axis=0), global_class_counts):
        raise RuntimeError(f"{partition_dir}: per-class totals differ from the global counts")
    return counts


def main() -> None:
    if not PART_ROOT.is_dir():
        raise RuntimeError(f"Missing {PART_ROOT}; run 42 and 43 first")

    heterogeneity = load_heterogeneity_module()
    columns = reference_columns()

    name_to_id = json.load(open(LABEL_MAP))
    class_order = [name for name, _ in sorted(name_to_id.items(), key=lambda kv: kv[1])]
    if len(class_order) != NUM_CLASSES:
        raise RuntimeError(f"label mapping has {len(class_order)} classes, expected {NUM_CLASSES}")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    total_records = len(y_train)
    global_class_counts = np.bincount(y_train, minlength=NUM_CLASSES)

    selection_rows = []
    metric_rows = []

    for label in DIRECTORY_LABELS:
        for condition in CONDITIONS:
            partition_dir = PART_ROOT / label / condition
            manifest_path = partition_dir / "partition_manifest.json"
            if not manifest_path.exists():
                raise RuntimeError(f"Missing manifest {manifest_path}")
            manifest = json.load(open(manifest_path))

            counts = load_client_counts(partition_dir, y_train, global_class_counts)
            metrics = heterogeneity.condition_metrics(counts, global_class_counts)
            summary = metrics["summary"]

            # Cross-check the freshly computed HD against what the manifest recorded.
            # A disagreement would mean the index files and the manifest have drifted.
            recorded_rms = manifest["hd_pairwise_rms"]
            if abs(summary["hd_pairwise_rms"] - recorded_rms) > 1e-9:
                raise RuntimeError(
                    f"{partition_dir}: recomputed RMS HD {summary['hd_pairwise_rms']!r} "
                    f"disagrees with the manifest value {recorded_rms!r}")

            label_number = int(label[len("seed_"):])
            target = TARGET_HD[condition]
            selection_rows.append({
                "directory_label": label,
                "condition": condition,
                "target_hd": target,
                "achieved_hd_rms": summary["hd_pairwise_rms"],
                "achieved_hd_pairwise_mean": summary["hd_pairwise_mean"],
                "abs_error_vs_target": abs(summary["hd_pairwise_rms"] - target),
                "generating_seed": manifest["partition_seed"],
                "alpha": manifest["alpha"],
                "partition_id": manifest["partition_id"],
                "size_min": summary["size_min"],
                "size_max": summary["size_max"],
                "size_max_min_ratio": summary["size_max_min_ratio"],
                "mean_absent_classes_per_client": summary["mean_absent_classes_per_client"],
                "path": str(partition_dir),
            })

            metric_rows.append({
                "K": NUM_CLIENTS,
                "seed": label_number,
                "condition": condition,
                "alpha": manifest["alpha"],
                "total_records": total_records,
                "source": SOURCE_TAG,
                **summary,
            })

    metrics_frame = pd.DataFrame(metric_rows)
    missing = [name for name in columns if name not in metrics_frame.columns]
    extra = [name for name in metrics_frame.columns if name not in columns]
    if missing or extra:
        raise RuntimeError(
            f"Column mismatch against {REFERENCE_SUMMARY}.\n"
            f"  missing: {missing}\n  unexpected: {extra}")
    metrics_frame = metrics_frame[columns]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).to_csv(SELECTION_PATH, index=False)
    metrics_frame.to_csv(METRICS_PATH, index=False)

    print(f"Wrote {SELECTION_PATH} ({len(selection_rows)} rows)")
    print(f"Wrote {METRICS_PATH} ({len(metrics_frame)} rows, {len(columns)} columns)")
    print(f"Column order matches {REFERENCE_SUMMARY}")
    print()

    print("Selection accuracy against target HD:")
    header = (f"{'label':>8}{'condition':>10}{'target':>8}{'achieved RMS':>14}"
              f"{'abs error':>11}{'alpha':>8}{'gen seed':>10}")
    print(header)
    print("-" * len(header))
    for row in selection_rows:
        alpha = "-" if row["alpha"] is None else f"{row['alpha']:g}"
        print(f"{row['directory_label']:>8}{row['condition']:>10}{row['target_hd']:>8.2f}"
              f"{row['achieved_hd_rms']:>14.6f}{row['abs_error_vs_target']:>11.6f}"
              f"{alpha:>8}{row['generating_seed']:>10}")


if __name__ == "__main__":
    main()
