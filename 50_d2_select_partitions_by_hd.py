"""
DATASET 2 (NF-CSE-CIC-IDS2018-v2): select K=5 partitions by measured Hellinger
distance, using the project's own pure class-wise Dirichlet partitioner.

This is the Dataset-2 counterpart of 42_select_partitions_by_hd.py. The procedure is
the same -- generate many candidates across alphas and seeds, measure the HD each one
actually achieves, keep the three nearest each target -- but the alpha ranges are
different and were re-measured on these labels rather than carried over.

Why they had to be re-measured. Dataset 2 has 7 classes and is 87.85 per cent Benign;
Dataset 1 has 10 classes and is 96.0 per cent Benign. Both the class count and the
dominant-class share feed into the achieved HD, so the same alpha lands somewhere
else. Measured on these labels at K=5, 10 seeds per alpha:

    alpha  1.00 -> HD 0.2752    alpha  0.30 -> HD 0.5478
    alpha  0.75 -> HD 0.3117    alpha  0.20 -> HD 0.6770
    alpha  0.50 -> HD 0.4351    alpha  0.10 -> HD 0.7738
    alpha  0.05 -> HD 0.8540    alpha  0.02 -> HD 0.9207

Nearest per target: 1.00 for HD 0.25, 0.30 for HD 0.50, 0.10 for HD 0.75, 0.02 for
HD 0.90. Dataset 1 needed roughly 0.755, 0.321, 0.129 and 0.028 for the same targets,
so the ranges here are deliberately not the Dataset-1 ones.

Empty clients appear below alpha 0.05 on this dataset, and HD 0.90 needs alpha near
0.02, so that target sits inside the rejection region. Candidates with an empty
client are rejected and counted; if a target cannot be met, nothing is written for it.

Zero-cell counts are out of 35 here (5 clients x 7 classes), not 50 as on Dataset 1.

HD means the root mean square over the 10 client pairs, which is what FedArtML's
hellinger_distance returns for a set of client distributions and therefore the
quantity the source study's levels refer to. The pairwise mean is recorded alongside
but is not selected on.

Reads:
    data/nf_cse_cic_ids2018_v2/processed/y_train.npy        training labels only
    configs/nf_cse_cic_ids2018_v2/label_mapping.json        class order
    d2_02_create_final_partitions.py                        partition and metric functions

Writes:
    data/nf_cse_cic_ids2018_v2/fl_clients/hd_selected_partitions/k_5/
        seed_<N>/<condition>/client_00_indices.npy ... client_04_indices.npy
        seed_<N>/<condition>/partition_manifest.json
        seed_mapping.csv
    results/nf_cse_cic_ids2018_v2/hd_selection/selection_summary.csv
    results/nf_cse_cic_ids2018_v2/hd_selection/all_candidates.csv

Never touches X_train, the validation split, the test split, or any Dataset-1 path.
Reads the three existing IID partitions only in order to copy them. Refuses to run if
its own outputs exist.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/nf_cse_cic_ids2018_v2/processed")
LABEL_MAP = Path("configs/nf_cse_cic_ids2018_v2/label_mapping.json")
PARTITION_SOURCE = Path("d2_02_create_final_partitions.py")

OUT_ROOT = Path("data/nf_cse_cic_ids2018_v2/fl_clients/hd_selected_partitions/k_5")
MAPPING_PATH = OUT_ROOT / "seed_mapping.csv"
RESULTS_DIR = Path("results/nf_cse_cic_ids2018_v2/hd_selection")
SELECTION_PATH = RESULTS_DIR / "selection_summary.csv"
CANDIDATES_PATH = RESULTS_DIR / "all_candidates.csv"

IID_SOURCE_ROOT = Path("data/nf_cse_cic_ids2018_v2/fl_clients/final_partitions/k_5")
IID_SOURCE_SEEDS = [42, 43, 44]

NUM_CLIENTS = 5
NUM_CLASSES = 7
PARTITIONS_PER_TARGET = 3
DIRECTORY_LABELS = ["seed_1", "seed_2", "seed_3"]

# Every written partition must sit within this of the target its directory name
# claims. Without it the search returns the nearest candidates however far away they
# are, so a target that is simply unreachable on this dataset would still produce
# three partitions in, say, hd_0p9/ that are nowhere near HD 0.90. On Dataset 1 both
# partitioners landed within 0.005, so 0.02 is generous rather than tight.
MAX_ABS_ERROR = 0.02

# Directory names are fixed rather than derived, so that "0.5" becomes hd_0p5 and
# not hd_0p50. The training scripts match on these strings.
TARGETS = [
    (0.25, "hd_0p25"),
    (0.50, "hd_0p5"),
    (0.75, "hd_0p75"),
    (0.90, "hd_0p9"),
]

# Centred on the Dataset-2 measurements in the docstring and widened, because a single
# alpha lands over a broad spread of HD depending on seed -- at alpha 0.10 the ten
# measured seeds spanned HD 0.464 to 0.909. dirichlet_rng keys the draw on
# int(round(alpha * 1000)), so alphas must differ by at least 0.001 to be distinct
# draws; these all do.
ALPHA_SWEEPS = {
    0.25: [0.700, 0.800, 0.900, 1.000, 1.150, 1.300, 1.500],
    0.50: [0.200, 0.240, 0.280, 0.300, 0.350, 0.400, 0.450],
    0.75: [0.060, 0.080, 0.100, 0.120, 0.150, 0.180, 0.220],
    0.90: [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050],
}

# 7 alphas x 30 seeds = 210 candidates per target. The 4000 block keeps these clear of
# Dataset 1's 1000 and 2000 blocks and of the 42/43/44 of the fixed-alpha runs.
CANDIDATE_SEEDS = list(range(4000, 4030))


def load_partition_module():
    """Import d2_02_create_final_partitions.py by path and check it is what we expect.

    The filename starts with a digit so a normal import will not work, and the module
    guards its main() behind __main__, so importing it defines the functions without
    generating any partitions.
    """
    if not PARTITION_SOURCE.exists():
        raise RuntimeError(f"Cannot find {PARTITION_SOURCE}; run from the project root.")
    spec = importlib.util.spec_from_file_location("d2_partition_source", PARTITION_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["d2_partition_source"] = module
    spec.loader.exec_module(module)
    for name in ("partition_noniid", "hellinger_distance", "compute_metrics",
                 "index_hash", "load_class_order", "K"):
        if not hasattr(module, name):
            raise RuntimeError(f"{PARTITION_SOURCE} does not define {name!r}")
    if module.K != NUM_CLIENTS:
        raise RuntimeError(f"{PARTITION_SOURCE} uses K={module.K}, expected {NUM_CLIENTS}")
    # load_class_order reads the label mapping named inside that module, so confirm it
    # is the Dataset-2 one rather than silently picking up Dataset 1's.
    if module.LABEL_MAP != LABEL_MAP:
        raise RuntimeError(
            f"{PARTITION_SOURCE} reads {module.LABEL_MAP}, expected {LABEL_MAP}")
    return module


def preflight() -> None:
    """Stop before doing anything if either output tree already exists."""
    existing = [path for path in (OUT_ROOT, RESULTS_DIR) if path.exists()]
    if existing:
        listing = "\n  ".join(str(path) for path in existing)
        raise RuntimeError(f"Refusing to run; outputs already present:\n  {listing}")


def evaluate_candidate(partitions, y_train, global_class_counts, dirichlet_alpha,
                       partition_seed):
    """Generate one candidate partition and measure it.

    Returns the record plus the partition itself, or a rejection record and three
    Nones when a client came out empty.
    """
    client_indices = partitions.partition_noniid(
        y_train, NUM_CLASSES, dirichlet_alpha, partition_seed)

    # The training scripts assert at load time that no client is empty, so an empty
    # client makes the partition unusable however good its HD looks. Rejecting here
    # also keeps compute_metrics from dividing by a zero client size.
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
        np.bincount(y_train[indices], minlength=NUM_CLASSES) for indices in client_indices])
    metrics = partitions.compute_metrics(client_class_counts, global_class_counts)

    record = {
        "alpha": dirichlet_alpha,
        "seed": partition_seed,
        "rejected_empty_client": False,
        "empty_client_ids": "",
        "hd_pairwise_rms": metrics["hd_pairwise_rms"],
        "hd_pairwise_mean": metrics["hd_pairwise_mean"],
        "zero_cells": int((client_class_counts == 0).sum()),
        "size_min": metrics["size_min"],
        "size_max": metrics["size_max"],
        "size_max_min_ratio": metrics["size_max_min_ratio"],
        "mean_absent_classes_per_client": metrics["mean_absent_classes_per_client"],
    }
    return record, client_indices, metrics, client_class_counts


def choose_closest(candidates, target_hd):
    """Pick the three candidates nearest the target, each from a different seed.

    Taking the three nearest outright would often return one seed at three
    neighbouring alphas, which share a within-class shuffle and are not three
    independent draws.
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


def write_partition(partitions, out_dir, label, condition, target_hd, candidate,
                    class_order, global_class_counts, total_records) -> dict:
    """Write one partition's index arrays and manifest."""
    out_dir.mkdir(parents=True, exist_ok=False)
    client_indices = candidate["client_indices"]
    for k, indices in enumerate(client_indices):
        np.save(out_dir / f"client_{k:02d}_indices.npy", indices)
    client_hashes = [partitions.index_hash(indices) for indices in client_indices]

    metrics = candidate["metrics"]
    counts = candidate["counts"]
    manifest = {
        "partition_id": f"k{NUM_CLIENTS}_seed{candidate['seed']}_{condition}",
        "dataset": "nf_cse_cic_ids2018_v2",
        "partition_method": "pure_classwise_dirichlet",
        "partitioner": "d2_02_create_final_partitions.partition_noniid",
        "selection_method": "measured_hd_rms_nearest_target",
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
    """Copy the three existing Dataset-2 IID partitions in as the HD = 0 level.

    Copied rather than regenerated so these results stay directly comparable to the
    existing fixed-alpha Dataset-2 runs, which used exactly these partitions.
    """
    manifests = []
    for label, generating_seed in zip(DIRECTORY_LABELS, IID_SOURCE_SEEDS):
        source_dir = IID_SOURCE_ROOT / f"seed_{generating_seed}" / "iid"
        target_dir = OUT_ROOT / label / "iid"
        if not source_dir.is_dir():
            raise RuntimeError(f"Missing IID source {source_dir}")
        if target_dir.exists():
            raise RuntimeError(f"Refusing to overwrite existing {target_dir}")
        shutil.copytree(source_dir, target_dir)

        # Confirm the copy is byte-identical before the manifest is touched, so a bad
        # copy is caught rather than masked by the edit that follows.
        for source_file in sorted(source_dir.iterdir()):
            target_file = target_dir / source_file.name
            if source_file.read_bytes() != target_file.read_bytes():
                raise RuntimeError(f"{target_file} does not match {source_file}")

        manifest_path = target_dir / "partition_manifest.json"
        manifest = json.load(open(manifest_path))
        manifest["directory_label"] = label
        manifest["hd_condition"] = "iid"
        manifest["target_hd"] = 0.0
        counts = np.array(manifest["client_class_counts"], dtype=np.int64)
        manifest["absent_cells_total"] = int((counts == 0).sum())
        manifest["absent_cells_possible"] = NUM_CLIENTS * NUM_CLASSES
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)
        manifests.append(manifest)
        print(f"  iid      seed_{generating_seed} -> {label}  "
              f"(copied byte-identical from {source_dir})")
    return manifests


def verify_partition(label, condition, y_train, global_class_counts) -> None:
    """Reload one written partition from disk and repeat the training scripts' checks.

    Everything is read back off disk, because checking the in-memory arrays would not
    catch a bad write.
    """
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


def main() -> None:
    preflight()
    partitions = load_partition_module()

    class_order = partitions.load_class_order()
    if len(class_order) != NUM_CLASSES:
        raise RuntimeError(
            f"label mapping has {len(class_order)} classes, expected {NUM_CLASSES}")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    total_records = len(y_train)
    global_class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    if int(global_class_counts.sum()) != total_records:
        raise RuntimeError("class counts do not sum to the number of training records")

    benign_share = global_class_counts[0] / global_class_counts.sum()
    print(f"Dataset 2 y_train: {total_records:,} records, {NUM_CLASSES} classes, "
          f"Benign {100 * benign_share:.2f}%")
    print(f"candidates per target: {len(CANDIDATE_SEEDS)} seeds x "
          f"{len(ALPHA_SWEEPS[TARGETS[0][0]])} alphas")
    print(f"zero-cell counts are out of {NUM_CLIENTS * NUM_CLASSES}")
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
            print(f"  CANNOT MEET TARGET {target_hd:.2f}: every candidate had an empty "
                  f"client. No partitions written for {condition}.")
            report.append({"target_hd": target_hd, "condition": condition,
                           "rejected": rejected, "chosen": []})
            continue

        distinct_seeds = len({candidate["seed"] for candidate in accepted})
        if distinct_seeds < PARTITIONS_PER_TARGET:
            raise RuntimeError(
                f"target {target_hd}: only {distinct_seeds} distinct seeds survived "
                f"rejection, need {PARTITIONS_PER_TARGET}")

        # Labels go in ascending generating-seed order so the mapping is reproducible
        # rather than dependent on how close each candidate happened to rank.
        chosen = sorted(choose_closest(accepted, target_hd), key=lambda c: c["seed"])

        # Every one of the three must be within tolerance, not just the best, because
        # each is written into a directory whose name claims this target.
        errors = [abs(c["hd_pairwise_rms"] - target_hd) for c in chosen]
        if max(errors) > MAX_ABS_ERROR:
            achieved = ", ".join(f"{c['hd_pairwise_rms']:.4f}" for c in chosen)
            print(f"  CANNOT MEET TARGET {target_hd:.2f}: nearest three achieved "
                  f"{achieved}, worst error {max(errors):.4f} exceeds the "
                  f"{MAX_ABS_ERROR} tolerance. No partitions written for {condition}.")
            report.append({"target_hd": target_hd, "condition": condition,
                           "rejected": rejected, "chosen": []})
            continue

        for label, candidate in zip(DIRECTORY_LABELS, chosen):
            out_dir = OUT_ROOT / label / condition
            manifest = write_partition(
                partitions, out_dir, label, condition, target_hd, candidate,
                class_order, global_class_counts, total_records)
            print(f"  {label}/{condition}: HD {manifest['hd_pairwise_rms']:.5f} "
                  f"(alpha {candidate['alpha']:g}, seed {candidate['seed']}) "
                  f"zero cells {manifest['absent_cells_total']}/{NUM_CLIENTS * NUM_CLASSES}")
            selection_rows.append({
                "directory_label": label,
                "condition": condition,
                "target_hd": target_hd,
                "achieved_hd_rms": manifest["hd_pairwise_rms"],
                "achieved_hd_pairwise_mean": manifest["hd_pairwise_mean"],
                "abs_error_vs_target": abs(manifest["hd_pairwise_rms"] - target_hd),
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
            "achieved_hd_rms": manifest["hd_pairwise_rms"],
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

    all_conditions = ["iid"] + [condition for _, condition in TARGETS]
    print("Verifying all written partitions:")
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
        "achieved_hd_rms": row["achieved_hd_rms"],
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
    print(f"Wrote {CANDIDATES_PATH} ({len(all_candidate_rows)} candidates)")
    print(f"Wrote {SELECTION_PATH} ({len(selection_rows)} rows)")
    print()

    print("SUMMARY (Dataset 2, own partitioner)")
    header = (f"{'target':>7}{'condition':>10}{'alphas':>22}{'achieved HD':>26}"
              f"{'seeds':>18}{'zero cells':>14}{'rejected':>10}")
    print(header)
    print("-" * len(header))
    for entry in report:
        if not entry["chosen"]:
            print(f"{entry['target_hd']:>7.2f}{entry['condition']:>10}"
                  f"{'TARGET NOT MET':>22}{'-':>26}{'-':>18}{'-':>14}"
                  f"{entry['rejected']:>10}")
            continue
        alphas = ", ".join(f"{c['alpha']:g}" for c in entry["chosen"])
        achieved = ", ".join(f"{c['hd_pairwise_rms']:.4f}" for c in entry["chosen"])
        seeds = ", ".join(str(c["seed"]) for c in entry["chosen"])
        cells = ", ".join(str(c["zero_cells"]) for c in entry["chosen"])
        print(f"{entry['target_hd']:>7.2f}{entry['condition']:>10}{alphas:>22}"
              f"{achieved:>26}{seeds:>18}{cells:>14}{entry['rejected']:>10}")


if __name__ == "__main__":
    main()
