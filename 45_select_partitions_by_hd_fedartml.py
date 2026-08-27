"""
Select K=5 partitions by measured HD using FedArtML's partitioner instead of ours.

42_select_partitions_by_hd.py runs this same selection procedure over
partition_noniid from 25_create_final_partitions.py, which is a pure class-wise
Dirichlet split with no constraints. FedArtML applies two that ours does not:

  - a per-client capacity cap, p * (len(idx_j) < N / local_nodes), which zeroes the
    Dirichlet weight of any client already holding N/K records, and
  - a rejection loop, while min_size < 10, which discards and redraws the whole
    partition if any client ends up below 10 records.

Those constraints change what a given alpha produces, so partitions generated here
come from the same code that produced the HD thresholds the source study reports.
Running both lets the write-up say whether a conclusion depends on the partitioner.

Selection is on FedArtML's own hellinger_distance, which returns the RMS over client
pairs. The pairwise mean is recorded alongside for reference but is not selected on.

Reads:
    data/processed/y_train.npy      training labels only
    configs/label_mapping.json      class order
    fedartml                        SplitAsFederatedData.dirichlet_method,
                                    function_base.hellinger_distance
    25_create_final_partitions.py   imported for compute_metrics and index_hash, so
                                    the manifests match the ones script 42 writes

Writes:
    data/fl_clients/hd_selected_fedartml/k_5/seed_<N>/<condition>/
        client_00_indices.npy ... client_04_indices.npy
        partition_manifest.json
    data/fl_clients/hd_selected_fedartml/k_5/seed_mapping.csv
    results/hd_selection_fedartml/selection_summary.csv
    results/hd_selection_fedartml/all_candidates.csv
    results/hd_selection_fedartml/comparison_vs_ours.csv

Never touches X_train, validation, test, data/fl_clients/hd_selected_partitions, or
data/fl_clients/final_partitions beyond reading the three IID partitions to copy.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys

import numpy as np
import pandas as pd

from fedartml import SplitAsFederatedData
from fedartml.function_base import hellinger_distance as fedartml_hellinger

PROCESSED_DIR = Path("data/processed")
LABEL_MAP = Path("configs/label_mapping.json")
PARTITION_SOURCE = Path("25_create_final_partitions.py")
FEDARTML_SOURCE = Path(
    "/Users/zobeiry/Documents/-MSc-dataScience-qmul/FedArtML/fedartml/fl_split_as_federated_data.py")

OUT_ROOT = Path("data/fl_clients/hd_selected_fedartml/k_5")
MAPPING_PATH = OUT_ROOT / "seed_mapping.csv"
RESULTS_DIR = Path("results/hd_selection_fedartml")
SELECTION_PATH = RESULTS_DIR / "selection_summary.csv"
CANDIDATES_PATH = RESULTS_DIR / "all_candidates.csv"
COMPARISON_PATH = RESULTS_DIR / "comparison_vs_ours.csv"

IID_SOURCE_ROOT = Path("data/fl_clients/final_partitions/k_5")
IID_SOURCE_SEEDS = [42, 43, 44]

# Written by 42_select_partitions_by_hd.py and 43a_summarise_hd_partitions.py.
OURS_SELECTION = Path("results/hd_selection/selection_summary.csv")
OURS_PART_ROOT = Path("data/fl_clients/hd_selected_partitions/k_5")

NUM_CLIENTS = 5
NUM_CLASSES = 10
PARTITIONS_PER_TARGET = 3
DIRECTORY_LABELS = ["seed_1", "seed_2", "seed_3"]

TARGETS = [
    (0.25, "hd_0p25"),
    (0.50, "hd_0p5"),
    (0.75, "hd_0p75"),
    (0.90, "hd_0p9"),
]

# Alphas measured on this dataset with FedArtML at K=5 over 10 seeds each:
#   1.0 -> 0.292   0.68 -> 0.316   0.6 -> 0.333   0.45 -> 0.396   0.36 -> 0.442
#   0.11 -> 0.756  0.09 -> 0.788   0.06 -> 0.855  0.045 -> 0.884  0.035 -> 0.907
# The HD 0.25 sweep reaches above alpha=1.0 because alpha=1.0 already averages 0.292,
# so the target sits on the far side of it and a downward-only sweep would miss.
ALPHA_SWEEPS = {
    0.25: [0.680, 0.800, 1.000, 1.300, 1.700, 2.200, 3.000],
    0.50: [0.280, 0.320, 0.360, 0.400, 0.450, 0.520, 0.600],
    0.75: [0.070, 0.085, 0.090, 0.110, 0.130, 0.160, 0.200],
    0.90: [0.020, 0.028, 0.035, 0.045, 0.055, 0.070, 0.090],
}

# 7 alphas x 30 seeds = 210 per target. Seeds start at 2000 so they cannot be
# confused with the 1000-block script 42 used or the 42/43/44 of the original runs.
CANDIDATE_SEEDS = list(range(2000, 2030))


def check_fedartml_source() -> str:
    """Confirm the imported fedartml is the copy at the path we were pointed at.

    pip installs a copy, so `import fedartml` could in principle resolve to a
    different version than the working tree. Comparing the two files means a
    divergence fails here rather than silently producing partitions from code the
    write-up does not describe.
    """
    import fedartml.fl_split_as_federated_data as installed_module
    installed_path = Path(installed_module.__file__)
    if not FEDARTML_SOURCE.exists():
        raise RuntimeError(f"Cannot find the FedArtML working tree at {FEDARTML_SOURCE}")
    installed_text = installed_path.read_text()
    source_text = FEDARTML_SOURCE.read_text()
    if installed_text != source_text:
        raise RuntimeError(
            f"The imported fedartml at {installed_path} differs from {FEDARTML_SOURCE}. "
            "Reinstall so the two match before generating partitions.")
    return hashlib.sha256(source_text.encode()).hexdigest()


def load_partition_module():
    """Import 25_create_final_partitions.py for compute_metrics and index_hash.

    Reusing them keeps these manifests structurally identical to the ones script 42
    writes, which is what makes the two partitioners comparable. The filename starts
    with a digit so it needs a path-based import, and the module guards main() behind
    __main__ so importing it generates nothing.
    """
    if not PARTITION_SOURCE.exists():
        raise RuntimeError(f"Cannot find {PARTITION_SOURCE}; run from the project root.")
    spec = importlib.util.spec_from_file_location("partition_source_fedartml", PARTITION_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["partition_source_fedartml"] = module
    spec.loader.exec_module(module)
    for name in ("compute_metrics", "index_hash", "load_class_order"):
        if not hasattr(module, name):
            raise RuntimeError(f"{PARTITION_SOURCE} does not define {name!r}")
    return module


def preflight() -> None:
    existing = [path for path in (OUT_ROOT, RESULTS_DIR) if path.exists()]
    if existing:
        listing = "\n  ".join(str(path) for path in existing)
        raise RuntimeError(f"Refusing to run; outputs already present:\n  {listing}")


def evaluate_candidate(partitions, y_train, global_class_counts, dirichlet_alpha,
                       partition_seed):
    """Generate one FedArtML candidate and measure it.

    Returns (record, client_indices, metrics, counts); the last three are None when
    the candidate is rejected.
    """
    pctg_distr, _, idx_distr, _ = SplitAsFederatedData.dirichlet_method(
        labels=y_train, local_nodes=NUM_CLIENTS, alpha=dirichlet_alpha,
        random_state=partition_seed)

    # FedArtML returns each client's indices in shuffled order; sort so the saved
    # arrays and their hashes match the convention script 25 and script 42 use.
    client_indices = [np.sort(np.asarray(indices, dtype=np.int64)) for indices in idx_distr]

    # FedArtML's own retry loop already refuses a partition with a client under 10
    # records, so an empty client should be impossible. Check anyway rather than
    # trusting it, because an empty client would trip the training script's
    # assertion much later and would make compute_metrics divide by zero.
    empty_clients = [k for k, indices in enumerate(client_indices) if len(indices) == 0]
    if empty_clients:
        record = {
            "alpha": dirichlet_alpha,
            "seed": partition_seed,
            "rejected_empty_client": True,
            "empty_client_ids": ",".join(str(k) for k in empty_clients),
            "fedartml_hd": np.nan,
            "hd_pairwise_mean": np.nan,
        }
        return record, None, None, None

    fedartml_hd = float(fedartml_hellinger(np.asarray(pctg_distr, dtype=np.float64)))

    client_class_counts = np.stack([
        np.bincount(y_train[indices], minlength=NUM_CLASSES) for indices in client_indices])
    metrics = partitions.compute_metrics(client_class_counts, global_class_counts)

    # FedArtML's HD is the RMS over client pairs, which is exactly what
    # compute_metrics reports as hd_pairwise_rms. They are computed from different
    # inputs -- FedArtML from its own proportion table, ours from the recomputed
    # counts -- so agreement confirms the two views of the partition match.
    if abs(fedartml_hd - metrics["hd_pairwise_rms"]) > 1e-9:
        raise RuntimeError(
            f"alpha={dirichlet_alpha} seed={partition_seed}: FedArtML HD {fedartml_hd!r} "
            f"disagrees with recomputed RMS {metrics['hd_pairwise_rms']!r}")

    absent_cells = int((client_class_counts == 0).sum())
    record = {
        "alpha": dirichlet_alpha,
        "seed": partition_seed,
        "rejected_empty_client": False,
        "empty_client_ids": "",
        "fedartml_hd": fedartml_hd,
        "hd_pairwise_mean": metrics["hd_pairwise_mean"],
        "hd_pairwise_rms": metrics["hd_pairwise_rms"],
        "absent_cells": absent_cells,
        "size_min": metrics["size_min"],
        "size_max": metrics["size_max"],
        "size_max_min_ratio": metrics["size_max_min_ratio"],
        "mean_absent_classes_per_client": metrics["mean_absent_classes_per_client"],
    }
    return record, client_indices, metrics, client_class_counts


def choose_closest(candidates, target_hd):
    """Three candidates nearest the target, each from a different seed.

    Without the distinct-seed rule the three nearest are often one seed at three
    neighbouring alphas, which would not be three independent draws.
    """
    ordered = sorted(candidates, key=lambda c: abs(c["fedartml_hd"] - target_hd))
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


def write_partition(partitions, out_dir, label, condition, target_hd, candidate,
                    class_order, global_class_counts, total_records,
                    fedartml_sha256) -> dict:
    """Write index arrays and manifest, matching the structure script 42 produces."""
    out_dir.mkdir(parents=True, exist_ok=False)
    client_indices = candidate["client_indices"]
    for k, indices in enumerate(client_indices):
        np.save(out_dir / f"client_{k:02d}_indices.npy", indices)
    client_hashes = [partitions.index_hash(indices) for indices in client_indices]

    metrics = candidate["metrics"]
    counts = candidate["counts"]
    manifest = {
        "partition_id": f"k{NUM_CLIENTS}_seed{candidate['seed']}_{condition}",
        "partition_method": "fedartml_dirichlet_method",
        "partitioner": "fedartml.SplitAsFederatedData.dirichlet_method",
        "partitioner_source_sha256": fedartml_sha256,
        "selection_method": "measured_fedartml_hd_nearest_target",
        "K": NUM_CLIENTS,
        "target_hd": target_hd,
        "hd_condition": condition,
        "directory_label": label,
        "alpha": candidate["alpha"],
        "partition_seed": candidate["seed"],
        "total_records": total_records,
        "class_order": class_order,
        "global_class_counts": global_class_counts.tolist(),
        "client_sizes": metrics["client_sizes"].tolist(),
        "client_class_counts": counts.tolist(),
        "client_index_sha256": client_hashes,
        "fedartml_hellinger_distance": candidate["fedartml_hd"],
        "absent_cells_total": int((counts == 0).sum()),
        "absent_cells_possible": NUM_CLIENTS * NUM_CLASSES,
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


def copy_iid_partitions() -> list[dict]:
    """Copy the three existing IID partitions in, unchanged, as the HD = 0 level."""
    manifests = []
    for label, generating_seed in zip(DIRECTORY_LABELS, IID_SOURCE_SEEDS):
        source_dir = IID_SOURCE_ROOT / f"seed_{generating_seed}" / "iid"
        target_dir = OUT_ROOT / label / "iid"
        if not source_dir.is_dir():
            raise RuntimeError(f"Missing IID source {source_dir}")
        if target_dir.exists():
            raise RuntimeError(f"Refusing to overwrite existing {target_dir}")
        shutil.copytree(source_dir, target_dir)

        manifest_path = target_dir / "partition_manifest.json"
        manifest = json.load(open(manifest_path))
        manifest["directory_label"] = label
        manifest["hd_condition"] = "iid"
        manifest["target_hd"] = 0.0
        # These are our own IID partitions, copied so the two partitioner families
        # share an identical HD = 0 baseline. Say so rather than leaving the reader
        # to assume FedArtML generated them.
        manifest["partitioner"] = "copied from final_partitions (iid_even); not FedArtML"
        counts = np.array(manifest["client_class_counts"], dtype=np.int64)
        manifest["absent_cells_total"] = int((counts == 0).sum())
        manifest["absent_cells_possible"] = NUM_CLIENTS * NUM_CLASSES
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)
        manifests.append(manifest)
        print(f"  iid      seed_{generating_seed} -> {label}  (copied from {source_dir})")
    return manifests


def verify_partition(label, condition, y_train, global_class_counts) -> None:
    """Reload from disk and repeat the checks the training scripts make."""
    partition_dir = OUT_ROOT / label / condition
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
    if len(np.unique(all_assigned)) != total_records:
        raise RuntimeError(f"{partition_dir}: duplicate training indices")
    if not np.array_equal(np.sort(all_assigned), np.arange(total_records)):
        raise RuntimeError(f"{partition_dir}: coverage is not exactly 0..N-1")

    counts = np.stack([np.bincount(y_train[indices], minlength=NUM_CLASSES)
                       for indices in client_indices])
    if not np.array_equal(counts.sum(axis=0), global_class_counts):
        raise RuntimeError(f"{partition_dir}: per-class totals differ from the global counts")

    manifest = json.load(open(partition_dir / "partition_manifest.json"))
    for k, indices in enumerate(client_indices):
        digest = hashlib.sha256(
            np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest()
        if digest != manifest["client_index_sha256"][k]:
            raise RuntimeError(f"{partition_dir}: client {k} does not match its recorded hash")
    if manifest.get("directory_label") != label:
        raise RuntimeError(f"{partition_dir}: directory_label is "
                           f"{manifest.get('directory_label')!r}, expected {label!r}")


def build_comparison(selection_rows) -> pd.DataFrame:
    """One row per HD target with both partitioners side by side.

    Rows are aggregated per target rather than paired partition to partition,
    because the directory labels are independent draws under each partitioner and
    pairing seed_1 with seed_1 would imply a correspondence that does not exist.
    """
    if not OURS_SELECTION.exists():
        raise RuntimeError(f"Cannot find {OURS_SELECTION}; run 43a first")
    ours = pd.read_csv(OURS_SELECTION)

    # Absent-cell counts are not in our selection summary, so read them from the
    # manifests the partitions carry.
    ours_absent = {}
    for _, row in ours.iterrows():
        manifest_path = OURS_PART_ROOT / row["directory_label"] / row["condition"] / \
            "partition_manifest.json"
        counts = np.array(json.load(open(manifest_path))["client_class_counts"], dtype=np.int64)
        ours_absent[(row["directory_label"], row["condition"])] = int((counts == 0).sum())

    theirs = pd.DataFrame(selection_rows)
    comparison = []
    for target_hd, condition in [(0.0, "iid")] + TARGETS:
        our_rows = ours[ours.condition == condition]
        their_rows = theirs[theirs.condition == condition]
        our_cells = [ours_absent[(r["directory_label"], condition)]
                     for _, r in our_rows.iterrows()]
        comparison.append({
            "target_hd": target_hd,
            "condition": condition,
            "ours_alphas": ", ".join("-" if pd.isna(a) else f"{a:g}" for a in our_rows.alpha),
            "ours_hd_values": ", ".join(f"{v:.4f}" for v in our_rows.achieved_hd_rms),
            "ours_hd_mean": float(our_rows.achieved_hd_rms.mean()),
            "ours_absent_cells": ", ".join(str(c) for c in our_cells),
            "ours_absent_cells_mean": float(np.mean(our_cells)),
            "ours_size_ratio_mean": float(our_rows.size_max_min_ratio.mean()),
            "fedartml_alphas": ", ".join(
                "-" if pd.isna(a) else f"{a:g}" for a in their_rows.alpha),
            "fedartml_hd_values": ", ".join(f"{v:.4f}" for v in their_rows.achieved_hd),
            "fedartml_hd_mean": float(their_rows.achieved_hd.mean()),
            "fedartml_absent_cells": ", ".join(str(c) for c in their_rows.absent_cells),
            "fedartml_absent_cells_mean": float(their_rows.absent_cells.mean()),
            "fedartml_size_ratio_mean": float(their_rows.size_max_min_ratio.mean()),
        })
    return pd.DataFrame(comparison)


def main() -> None:
    preflight()
    fedartml_sha256 = check_fedartml_source()
    partitions = load_partition_module()

    class_order = partitions.load_class_order()
    if len(class_order) != NUM_CLASSES:
        raise RuntimeError(f"label mapping has {len(class_order)} classes, expected {NUM_CLASSES}")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    total_records = len(y_train)
    global_class_counts = np.bincount(y_train, minlength=NUM_CLASSES)

    print(f"FedArtML source verified, sha256 {fedartml_sha256[:16]}")
    print(f"y_train: {total_records:,} records, {NUM_CLASSES} classes")
    print(f"candidates per target: {len(CANDIDATE_SEEDS)} seeds x 7 alphas")
    print()

    all_candidate_rows = []
    selection_rows = []
    report = []

    for target_hd, condition in TARGETS:
        accepted = []
        rejected = 0
        for dirichlet_alpha in ALPHA_SWEEPS[target_hd]:
            for partition_seed in CANDIDATE_SEEDS:
                record, client_indices, metrics, counts = evaluate_candidate(
                    partitions, y_train, global_class_counts, dirichlet_alpha, partition_seed)
                record["target_hd"] = target_hd
                record["hd_condition"] = condition
                all_candidate_rows.append(record)
                if client_indices is None:
                    rejected += 1
                    continue
                accepted.append({**record, "client_indices": client_indices,
                                 "metrics": metrics, "counts": counts})

        evaluated = len(ALPHA_SWEEPS[target_hd]) * len(CANDIDATE_SEEDS)
        print(f"target HD {target_hd:.2f} ({condition}): {evaluated} candidates, "
              f"{rejected} rejected for an empty client, {len(accepted)} usable")

        if not accepted:
            print(f"  CANNOT MEET TARGET {target_hd:.2f}: every candidate was rejected. "
                  f"No partitions written for {condition}.")
            report.append({"target_hd": target_hd, "condition": condition,
                           "rejected": rejected, "chosen": []})
            continue

        distinct_seeds = len({candidate["seed"] for candidate in accepted})
        if distinct_seeds < PARTITIONS_PER_TARGET:
            raise RuntimeError(
                f"target {target_hd}: only {distinct_seeds} distinct seeds survived, "
                f"need {PARTITIONS_PER_TARGET}")

        # Labels go in ascending generating-seed order, the same rule script 43 used,
        # so the mapping is reproducible rather than dependent on ranking order.
        chosen = sorted(choose_closest(accepted, target_hd), key=lambda c: c["seed"])
        for label, candidate in zip(DIRECTORY_LABELS, chosen):
            out_dir = OUT_ROOT / label / condition
            manifest = write_partition(
                partitions, out_dir, label, condition, target_hd, candidate,
                class_order, global_class_counts, total_records, fedartml_sha256)
            print(f"  {label}/{condition}: HD {manifest['fedartml_hellinger_distance']:.5f} "
                  f"(alpha {candidate['alpha']:g}, seed {candidate['seed']}) "
                  f"absent cells {manifest['absent_cells_total']}/50")
            selection_rows.append({
                "directory_label": label,
                "condition": condition,
                "target_hd": target_hd,
                "achieved_hd": manifest["fedartml_hellinger_distance"],
                "achieved_hd_pairwise_mean": manifest["hd_pairwise_mean"],
                "abs_error_vs_target": abs(manifest["fedartml_hellinger_distance"] - target_hd),
                "generating_seed": candidate["seed"],
                "alpha": candidate["alpha"],
                "partition_id": manifest["partition_id"],
                "absent_cells": manifest["absent_cells_total"],
                "size_min": manifest["size_min"],
                "size_max": manifest["size_max"],
                "size_max_min_ratio": manifest["size_max_min_ratio"],
                "path": str(out_dir),
            })

        report.append({"target_hd": target_hd, "condition": condition,
                       "rejected": rejected, "chosen": chosen})
        print()

    print("Copying IID partitions:")
    iid_manifests = copy_iid_partitions()
    for label, manifest in zip(DIRECTORY_LABELS, iid_manifests):
        selection_rows.append({
            "directory_label": label,
            "condition": "iid",
            "target_hd": 0.0,
            "achieved_hd": manifest["hd_pairwise_rms"],
            "achieved_hd_pairwise_mean": manifest["hd_pairwise_mean"],
            "abs_error_vs_target": abs(manifest["hd_pairwise_rms"]),
            "generating_seed": manifest["partition_seed"],
            "alpha": manifest["alpha"],
            "partition_id": manifest["partition_id"],
            "absent_cells": manifest["absent_cells_total"],
            "size_min": manifest["size_min"],
            "size_max": manifest["size_max"],
            "size_max_min_ratio": manifest["size_max_min_ratio"],
            "path": str(OUT_ROOT / label / "iid"),
        })
    print()

    print("Verifying all written partitions:")
    all_conditions = ["iid"] + [condition for _, condition in TARGETS]
    for label in DIRECTORY_LABELS:
        for condition in all_conditions:
            if not (OUT_ROOT / label / condition).is_dir():
                print(f"  SKIP {label}/{condition} (target not met)")
                continue
            verify_partition(label, condition, y_train, global_class_counts)
            print(f"  OK  {label}/{condition}")
    print()

    mapping = pd.DataFrame([{
        "directory_label": row["directory_label"],
        "condition": row["condition"],
        "generating_seed": row["generating_seed"],
        "alpha": row["alpha"],
        "achieved_hd": row["achieved_hd"],
        "achieved_hd_pairwise_mean": row["achieved_hd_pairwise_mean"],
        "absent_cells": row["absent_cells"],
    } for row in selection_rows])
    condition_order = {name: i for i, name in enumerate(all_conditions)}
    mapping = mapping.sort_values(
        by=["condition", "directory_label"],
        key=lambda col: col.map(condition_order) if col.name == "condition" else col)
    mapping.to_csv(MAPPING_PATH, index=False)
    print(f"Wrote {MAPPING_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(all_candidate_rows).to_csv(CANDIDATES_PATH, index=False)
    pd.DataFrame(selection_rows).to_csv(SELECTION_PATH, index=False)
    comparison = build_comparison(selection_rows)
    comparison.to_csv(COMPARISON_PATH, index=False)
    print(f"Wrote {CANDIDATES_PATH} ({len(all_candidate_rows)} candidates)")
    print(f"Wrote {SELECTION_PATH} ({len(selection_rows)} rows)")
    print(f"Wrote {COMPARISON_PATH} ({len(comparison)} rows)")
    print()

    print("SUMMARY (FedArtML partitioner)")
    header = (f"{'target':>7}{'condition':>10}{'alphas':>22}{'achieved HD':>26}"
              f"{'seeds':>18}{'absent cells':>16}{'size ratios':>28}")
    print(header)
    print("-" * len(header))
    for entry in report:
        if not entry["chosen"]:
            print(f"{entry['target_hd']:>7.2f}{entry['condition']:>10}"
                  f"{'TARGET NOT MET':>22}{'-':>26}{'-':>18}{'-':>16}{'-':>28}")
            continue
        alphas = ", ".join(f"{c['alpha']:g}" for c in entry["chosen"])
        achieved = ", ".join(f"{c['fedartml_hd']:.4f}" for c in entry["chosen"])
        seeds = ", ".join(str(c["seed"]) for c in entry["chosen"])
        cells = ", ".join(str(c["absent_cells"]) for c in entry["chosen"])
        ratios = ", ".join(f"{c['size_max_min_ratio']:.1f}" for c in entry["chosen"])
        print(f"{entry['target_hd']:>7.2f}{entry['condition']:>10}{alphas:>22}"
              f"{achieved:>26}{seeds:>18}{cells:>16}{ratios:>28}")
    print()

    print("COMPARISON, ours vs FedArtML")
    print(comparison[["target_hd", "condition", "ours_hd_mean", "fedartml_hd_mean",
                      "ours_absent_cells_mean", "fedartml_absent_cells_mean",
                      "ours_size_ratio_mean", "fedartml_size_ratio_mean"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
