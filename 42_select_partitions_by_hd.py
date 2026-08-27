"""
Select K=5 training partitions by measured Hellinger distance rather than by alpha.

Jimenez-Gutierrez et al. fix HD levels (0.25, 0.5, 0.75, 0.9) and report the alpha
that produced them. That only works if alpha maps reliably onto HD. On this dataset
it does not: at alpha=0.5 the three existing seeds gave RMS HD 0.311, 0.547 and
0.478, a spread wider than the gap between two of the target levels. So this script
inverts the procedure. It generates many candidate partitions across a range of
alphas and seeds, measures the HD each one actually achieved, and keeps the three
closest to each target.

HD here means the root mean square over the 10 client pairs, not the mean. That is
what FedArtML's hellinger_distance returns for a set of client distributions, so it
is the quantity the source study's levels refer to. The pairwise mean is computed
and recorded too, but selection is on RMS.

Reads:
    data/processed/y_train.npy      training labels only
    configs/label_mapping.json      class order
    25_create_final_partitions.py   imported for its partition and metric functions

Writes:
    data/fl_clients/hd_selected_partitions/k_5/seed_<S>/hd_<level>/
        client_00_indices.npy ... client_04_indices.npy
        partition_manifest.json
    results/hd_selection/selection_summary.csv
    results/hd_selection/all_candidates.csv

Never touches X_train, the validation split, the test split, or anything under
data/fl_clients/final_partitions. Refuses to run if its own outputs exist.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
PARTITION_SOURCE = Path("25_create_final_partitions.py")
OUT_ROOT = Path("data/fl_clients/hd_selected_partitions/k_5")
RESULTS_DIR = Path("results/hd_selection")
SELECTION_PATH = RESULTS_DIR / "selection_summary.csv"
CANDIDATES_PATH = RESULTS_DIR / "all_candidates.csv"

NUM_CLIENTS = 5
PARTITIONS_PER_TARGET = 3

# Directory names are fixed rather than derived, so that "0.5" becomes hd_0p5 and
# not hd_0p50. The training scripts match on these strings.
TARGETS = [
    (0.25, "hd_0p25"),
    (0.50, "hd_0p5"),
    (0.75, "hd_0p75"),
    (0.90, "hd_0p9"),
]

# Alphas whose mean RMS HD sat nearest each target in earlier sweeps, widened into a
# range because any single alpha lands over a broad spread of HD depending on seed.
# dirichlet_rng keys the draw on int(round(alpha * 1000)), so alphas must differ by
# at least 0.001 to be distinct draws; these all do.
ALPHA_SWEEPS = {
    0.25: [0.600, 0.680, 0.755, 0.830, 0.910, 1.000, 1.100],
    0.50: [0.240, 0.280, 0.321, 0.360, 0.400, 0.450, 0.500],
    0.75: [0.090, 0.110, 0.129, 0.150, 0.170, 0.190, 0.220],
    0.90: [0.015, 0.020, 0.028, 0.035, 0.045, 0.060, 0.080],
}

# 7 alphas x 30 seeds = 210 candidates per target. Seeds start well clear of
# 42/43/44 so these partitions can never be confused with the fixed-alpha set.
CANDIDATE_SEEDS = list(range(1000, 1030))


def load_partition_module():
    """Import 25_create_final_partitions.py by path.

    The filename starts with a digit so it is not a valid module name for a normal
    import. The module guards its main() behind __main__, so importing it defines
    the functions without generating anything.
    """
    if not PARTITION_SOURCE.exists():
        raise RuntimeError(f"Cannot find {PARTITION_SOURCE}; run from the project root.")
    spec = importlib.util.spec_from_file_location("partition_source", PARTITION_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["partition_source"] = module
    spec.loader.exec_module(module)
    for name in ("partition_noniid", "hellinger_distance", "compute_metrics",
                 "index_hash", "load_class_order", "K"):
        if not hasattr(module, name):
            raise RuntimeError(f"{PARTITION_SOURCE} does not define {name!r}")
    if module.K != NUM_CLIENTS:
        raise RuntimeError(f"{PARTITION_SOURCE} uses K={module.K}, expected {NUM_CLIENTS}")
    return module


def preflight() -> None:
    existing = [path for path in (OUT_ROOT, RESULTS_DIR) if path.exists()]
    if existing:
        listing = "\n  ".join(str(path) for path in existing)
        raise RuntimeError(f"Refusing to run; outputs already present:\n  {listing}")


def evaluate_candidate(partitions, y_train, num_classes, global_class_counts,
                       dirichlet_alpha, partition_seed):
    """Generate one candidate partition and measure it.

    Returns (record, client_indices, metrics, counts). client_indices is None when
    the candidate is rejected, in which case metrics and counts are None too.
    """
    client_indices = partitions.partition_noniid(
        y_train, num_classes, dirichlet_alpha, partition_seed)

    # Script 32 asserts at load time that no client is empty, so an empty client
    # makes the partition unusable for training however good its HD looks. Reject it
    # here rather than writing something that cannot be trained on. This also keeps
    # compute_metrics from dividing by a zero client size.
    empty_clients = [k for k, indices in enumerate(client_indices) if len(indices) == 0]
    if empty_clients:
        record = {
            "alpha": dirichlet_alpha,
            "seed": partition_seed,
            "rejected_empty_client": True,
            "empty_client_ids": ",".join(str(k) for k in empty_clients),
            "hd_pairwise_rms": np.nan,
            "hd_pairwise_mean": np.nan,
            "size_min": 0,
        }
        return record, None, None, None

    client_class_counts = np.stack([
        np.bincount(y_train[indices], minlength=num_classes) for indices in client_indices])
    metrics = partitions.compute_metrics(client_class_counts, global_class_counts)

    record = {
        "alpha": dirichlet_alpha,
        "seed": partition_seed,
        "rejected_empty_client": False,
        "empty_client_ids": "",
        "hd_pairwise_rms": metrics["hd_pairwise_rms"],
        "hd_pairwise_mean": metrics["hd_pairwise_mean"],
        "size_min": metrics["size_min"],
        "size_max": metrics["size_max"],
        "size_max_min_ratio": metrics["size_max_min_ratio"],
        "mean_absent_classes_per_client": metrics["mean_absent_classes_per_client"],
    }
    return record, client_indices, metrics, client_class_counts


def choose_closest(candidates, target_hd):
    """Pick the PARTITIONS_PER_TARGET closest candidates, each from a different seed.

    Taking the three nearest outright would often return the same seed at three
    neighbouring alphas, which would give three partitions sharing a within-class
    shuffle. Walking the sorted list and skipping seeds already used keeps the three
    genuinely independent.
    """
    ordered = sorted(candidates, key=lambda c: abs(c["hd_pairwise_rms"] - target_hd))
    chosen = []
    used_seeds = set()
    for candidate in ordered:
        if candidate["seed"] in used_seeds:
            continue
        chosen.append(candidate)
        used_seeds.add(candidate["seed"])
        if len(chosen) == PARTITIONS_PER_TARGET:
            break
    return chosen


def write_partition(partitions, out_dir, client_indices, client_class_counts, metrics,
                    target_hd, condition, dirichlet_alpha, partition_seed,
                    class_order, global_class_counts, total_records) -> dict:
    """Write the index arrays and manifest, mirroring what script 25 produces."""
    out_dir.mkdir(parents=True, exist_ok=False)
    for k, indices in enumerate(client_indices):
        np.save(out_dir / f"client_{k:02d}_indices.npy", indices)
    client_hashes = [partitions.index_hash(indices) for indices in client_indices]

    partition_id = f"k{NUM_CLIENTS}_seed{partition_seed}_{condition}"
    manifest = {
        "partition_id": partition_id,
        "partition_method": "pure_classwise_dirichlet",
        "selection_method": "measured_hd_rms_nearest_target",
        "K": NUM_CLIENTS,
        "target_hd": target_hd,
        "hd_condition": condition,
        "alpha": dirichlet_alpha,
        "partition_seed": partition_seed,
        "total_records": total_records,
        "class_order": class_order,
        "global_class_counts": global_class_counts.tolist(),
        "client_sizes": metrics["client_sizes"].tolist(),
        "client_class_counts": client_class_counts.tolist(),
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
    with open(out_dir / "partition_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def verify_written_partition(out_dir, y_train, num_classes, global_class_counts) -> None:
    """Reload from disk and repeat the checks script 32 makes at load time.

    Checking the in-memory arrays would not catch a bad write, so everything here is
    read back off disk.
    """
    files = sorted(out_dir.glob("client_*_indices.npy"))
    if len(files) != NUM_CLIENTS:
        raise RuntimeError(f"{out_dir}: expected {NUM_CLIENTS} client files, found {len(files)}")

    client_indices = [np.load(out_dir / f"client_{k:02d}_indices.npy") for k in range(NUM_CLIENTS)]
    if any(len(indices) == 0 for indices in client_indices):
        raise RuntimeError(f"{out_dir}: a client is empty")

    all_assigned = np.concatenate(client_indices)
    total_records = len(y_train)
    if len(all_assigned) != total_records:
        raise RuntimeError(f"{out_dir}: indices do not cover all {total_records} training records")
    if len(np.unique(all_assigned)) != total_records:
        raise RuntimeError(f"{out_dir}: duplicate training indices")
    if not np.array_equal(np.sort(all_assigned), np.arange(total_records)):
        raise RuntimeError(f"{out_dir}: coverage is not exactly 0..N-1")

    counts = np.stack([np.bincount(y_train[indices], minlength=num_classes)
                       for indices in client_indices])
    if not np.array_equal(counts.sum(axis=0), global_class_counts):
        raise RuntimeError(f"{out_dir}: per-class totals differ from the global counts")

    # The manifest records a hash per client; confirm the files still match it.
    manifest = json.load(open(out_dir / "partition_manifest.json"))
    for k, indices in enumerate(client_indices):
        digest = hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest()
        if digest != manifest["client_index_sha256"][k]:
            raise RuntimeError(f"{out_dir}: client {k} index file does not match its recorded hash")


def main() -> None:
    preflight()
    partitions = load_partition_module()

    class_order = partitions.load_class_order()
    num_classes = len(class_order)
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    total_records = len(y_train)
    global_class_counts = np.bincount(y_train, minlength=num_classes)
    if int(global_class_counts.sum()) != total_records:
        raise RuntimeError("class counts do not sum to the number of training records")
    print(f"y_train: {total_records:,} records, {num_classes} classes")
    print(f"candidates per target: {len(CANDIDATE_SEEDS)} seeds x 7 alphas")
    print()

    all_candidate_rows = []
    selected_rows = []
    report = []

    for target_hd, condition in TARGETS:
        accepted = []
        rejected = 0
        for dirichlet_alpha in ALPHA_SWEEPS[target_hd]:
            for partition_seed in CANDIDATE_SEEDS:
                record, client_indices, metrics, counts = evaluate_candidate(
                    partitions, y_train, num_classes, global_class_counts,
                    dirichlet_alpha, partition_seed)
                record["target_hd"] = target_hd
                record["hd_condition"] = condition
                all_candidate_rows.append(record)
                if client_indices is None:
                    rejected += 1
                    continue
                accepted.append({**record, "client_indices": client_indices,
                                 "metrics": metrics, "counts": counts})

        evaluated = len(ALPHA_SWEEPS[target_hd]) * len(CANDIDATE_SEEDS)
        print(f"target HD {target_hd:.2f} ({condition}): "
              f"{evaluated} candidates, {rejected} rejected for an empty client, "
              f"{len(accepted)} usable")

        if not accepted:
            print(f"  CANNOT MEET TARGET {target_hd:.2f}: every candidate had an empty "
                  f"client. No partitions written for {condition}.")
            report.append({"target_hd": target_hd, "condition": condition,
                           "rejected": rejected, "evaluated": evaluated,
                           "chosen": []})
            continue

        distinct_seeds = len({candidate["seed"] for candidate in accepted})
        if distinct_seeds < PARTITIONS_PER_TARGET:
            raise RuntimeError(
                f"target {target_hd}: only {distinct_seeds} distinct seeds survived "
                f"rejection, need {PARTITIONS_PER_TARGET}")

        chosen = choose_closest(accepted, target_hd)
        for candidate in chosen:
            out_dir = OUT_ROOT / f"seed_{candidate['seed']}" / condition
            manifest = write_partition(
                partitions, out_dir, candidate["client_indices"], candidate["counts"],
                candidate["metrics"], target_hd, condition, candidate["alpha"],
                candidate["seed"], class_order, global_class_counts, total_records)
            verify_written_partition(out_dir, y_train, num_classes, global_class_counts)
            print(f"  wrote {out_dir}  achieved RMS HD {manifest['hd_pairwise_rms']:.5f} "
                  f"(alpha {candidate['alpha']}, seed {candidate['seed']}) verified")

            selected_rows.append({
                "target_hd": target_hd,
                "hd_condition": condition,
                "partition_id": manifest["partition_id"],
                "alpha": candidate["alpha"],
                "partition_seed": candidate["seed"],
                "achieved_hd_rms": manifest["hd_pairwise_rms"],
                "achieved_hd_pairwise_mean": manifest["hd_pairwise_mean"],
                "abs_error_vs_target": abs(manifest["hd_pairwise_rms"] - target_hd),
                "size_min": manifest["size_min"],
                "size_max": manifest["size_max"],
                "size_max_min_ratio": manifest["size_max_min_ratio"],
                "mean_absent_classes_per_client": float(
                    np.mean(manifest["absent_classes_per_client"])),
                "path": str(out_dir),
            })

        report.append({"target_hd": target_hd, "condition": condition,
                       "rejected": rejected, "evaluated": evaluated, "chosen": chosen})
        print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(all_candidate_rows).to_csv(CANDIDATES_PATH, index=False)
    pd.DataFrame(selected_rows).to_csv(SELECTION_PATH, index=False)
    print(f"Wrote {CANDIDATES_PATH} ({len(all_candidate_rows)} candidates)")
    print(f"Wrote {SELECTION_PATH} ({len(selected_rows)} selected partitions)")
    print()

    print("SUMMARY")
    header = (f"{'target':>7}{'condition':>11}{'alphas chosen':>26}"
              f"{'achieved RMS HD':>34}{'seeds':>20}{'rejected':>10}")
    print(header)
    print("-" * len(header))
    for entry in report:
        if not entry["chosen"]:
            print(f"{entry['target_hd']:>7.2f}{entry['condition']:>11}"
                  f"{'NONE - target not met':>26}{'-':>34}{'-':>20}"
                  f"{entry['rejected']:>10}")
            continue
        alphas = ", ".join(f"{c['alpha']:g}" for c in entry["chosen"])
        achieved = ", ".join(f"{c['hd_pairwise_rms']:.4f}" for c in entry["chosen"])
        seeds = ", ".join(str(c["seed"]) for c in entry["chosen"])
        print(f"{entry['target_hd']:>7.2f}{entry['condition']:>11}{alphas:>26}"
              f"{achieved:>34}{seeds:>20}{entry['rejected']:>10}")


if __name__ == "__main__":
    main()
