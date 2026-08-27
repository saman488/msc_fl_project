"""
DATASET 2 (NF-CSE-CIC-IDS2018-v2): select K=5 partitions by measured Hellinger
distance, using FedArtML's partitioner instead of the project's own.

This is the Dataset-2 counterpart of 45_select_partitions_by_hd_fedartml.py, and the
FedArtML-partitioner sibling of 50_d2_select_partitions_by_hd.py. Running all four
lets the write-up separate what depends on the dataset from what depends on the
partitioner.

FedArtML applies two constraints the project's partitioner does not:

  - a per-client capacity cap, p * (len(idx_j) < N / local_nodes), which zeroes the
    Dirichlet weight of any client already holding N/K records, and
  - a rejection loop, while min_size < 10, which redraws the whole partition if any
    client falls below 10 records.

Alpha ranges are established by this script, not assumed. Two reasons they cannot be
carried over. Dataset 2 has 7 classes and is 87.85 per cent Benign against Dataset 1's
10 classes and 96.0 per cent, so the alpha-to-HD mapping already differs between
datasets; and the capacity cap shifts that mapping again relative to the project's own
partitioner. Measured on these labels with the project's partitioner, alpha 1.00 gave
HD 0.2752 and alpha 0.02 gave HD 0.9207 -- those figures are the Dataset-2 baseline
and are NOT what FedArtML will produce. So the script runs a calibration sweep with
FedArtML first, prints the mapping it finds, and derives each target's search range
from that measurement.

Note on the Benign class. It is class id 0, so it is allocated before any client can
exceed N/K and be capped. On Dataset 1 that produced a client holding only Benign
records in every partition. Whether the same happens at 87.85 per cent is an open
question this script's zero-cell and client-size output will answer.

Zero-cell counts are out of 35 here (5 clients x 7 classes), not 50 as on Dataset 1.

Empty clients should be impossible given FedArtML's own retry loop, but every
candidate is checked anyway and rejections are counted per target. If a target cannot
be met, nothing is written for it.

Expect a long run. FedArtML's partitioner builds Python lists over all 13,255,011
training rows, so a single candidate takes appreciably longer than on Dataset 1;
calibration plus selection is on the order of tens of minutes.

Reads:
    data/nf_cse_cic_ids2018_v2/processed/y_train.npy        training labels only
    configs/nf_cse_cic_ids2018_v2/label_mapping.json        class order
    fedartml                                                dirichlet_method,
                                                            hellinger_distance
    d2_02_create_final_partitions.py                        compute_metrics, index_hash,
                                                            so manifests match script 50's

Writes:
    data/nf_cse_cic_ids2018_v2/fl_clients/hd_selected_fedartml/k_5/
        seed_<N>/<condition>/client_00_indices.npy ... client_04_indices.npy
        seed_<N>/<condition>/partition_manifest.json
        seed_mapping.csv
    results/nf_cse_cic_ids2018_v2/hd_selection_fedartml/selection_summary.csv
    results/nf_cse_cic_ids2018_v2/hd_selection_fedartml/all_candidates.csv
    results/nf_cse_cic_ids2018_v2/hd_selection_fedartml/fedartml_alpha_calibration.csv

Never touches X_train, the validation split, the test split, any Dataset-1 path, or
the partitions written by script 50. Refuses to run if its own outputs exist.
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

PROCESSED_DIR = Path("data/nf_cse_cic_ids2018_v2/processed")
LABEL_MAP = Path("configs/nf_cse_cic_ids2018_v2/label_mapping.json")
PARTITION_SOURCE = Path("d2_02_create_final_partitions.py")
FEDARTML_SOURCE = Path(
    "/Users/zobeiry/Documents/-MSc-dataScience-qmul/FedArtML/fedartml/fl_split_as_federated_data.py")

OUT_ROOT = Path("data/nf_cse_cic_ids2018_v2/fl_clients/hd_selected_fedartml/k_5")
MAPPING_PATH = OUT_ROOT / "seed_mapping.csv"
RESULTS_DIR = Path("results/nf_cse_cic_ids2018_v2/hd_selection_fedartml")
SELECTION_PATH = RESULTS_DIR / "selection_summary.csv"
CANDIDATES_PATH = RESULTS_DIR / "all_candidates.csv"
CALIBRATION_PATH = RESULTS_DIR / "fedartml_alpha_calibration.csv"

IID_SOURCE_ROOT = Path("data/nf_cse_cic_ids2018_v2/fl_clients/final_partitions/k_5")
IID_SOURCE_SEEDS = [42, 43, 44]

NUM_CLIENTS = 5
NUM_CLASSES = 7
PARTITIONS_PER_TARGET = 3
DIRECTORY_LABELS = ["seed_1", "seed_2", "seed_3"]

# Every written partition must sit within this of the target its directory name
# claims. Without it the search returns the nearest candidates however far away they
# are, so a target the capacity cap puts out of reach would still produce three
# partitions in, say, hd_0p9/ that are nowhere near HD 0.90. On Dataset 1 both
# partitioners landed within 0.005, so 0.02 is generous rather than tight.
MAX_ABS_ERROR = 0.02

TARGETS = [
    (0.25, "hd_0p25"),
    (0.50, "hd_0p5"),
    (0.75, "hd_0p75"),
    (0.90, "hd_0p9"),
]

# Calibration grid, spanning three orders of magnitude so it brackets every target
# whatever the capacity cap does to the mapping. Kept coarse in seeds because its job
# is only to locate each target, not to measure precisely.
CALIBRATION_ALPHAS = [0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200,
                      0.300, 0.400, 0.500, 0.750, 1.000, 1.500, 2.500, 4.000]
CALIBRATION_SEEDS = list(range(5900, 5906))

# How many calibration alphas feed each target's search, chosen by nearest mean HD.
ALPHAS_PER_TARGET = 7

# 7 alphas x 30 seeds = 210 candidates per target. The 5000 block keeps these clear of
# script 50's 4000 block, Dataset 1's 1000 and 2000 blocks, and the 42/43/44 of the
# fixed-alpha runs.
CANDIDATE_SEEDS = list(range(5000, 5030))


def check_fedartml_source() -> str:
    """Confirm the imported fedartml is the copy in the working tree we were given.

    pip installs its own copy, so `import fedartml` could resolve to a different
    version. Comparing the files means a divergence fails here rather than silently
    producing partitions from code the write-up does not describe.
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
    """Import d2_02_create_final_partitions.py for compute_metrics and index_hash.

    Reusing them keeps these manifests structurally identical to script 50's, which is
    what makes the two partitioners comparable. The module guards main() behind
    __main__, so importing it generates nothing.
    """
    if not PARTITION_SOURCE.exists():
        raise RuntimeError(f"Cannot find {PARTITION_SOURCE}; run from the project root.")
    spec = importlib.util.spec_from_file_location("d2_partition_source_fedartml",
                                                  PARTITION_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["d2_partition_source_fedartml"] = module
    spec.loader.exec_module(module)
    for name in ("compute_metrics", "index_hash", "load_class_order"):
        if not hasattr(module, name):
            raise RuntimeError(f"{PARTITION_SOURCE} does not define {name!r}")
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
    """Generate one FedArtML candidate and measure it.

    Returns the record plus the partition itself, or a rejection record and three
    Nones when a client came out empty.
    """
    pctg_distr, _, idx_distr, _ = SplitAsFederatedData.dirichlet_method(
        labels=y_train, local_nodes=NUM_CLIENTS, alpha=dirichlet_alpha,
        random_state=partition_seed)

    # FedArtML returns each client's indices shuffled; sort so the saved arrays and
    # their hashes follow the same convention as script 50's.
    client_indices = [np.sort(np.asarray(indices, dtype=np.int64)) for indices in idx_distr]

    # FedArtML's retry loop already refuses a partition with a client under 10 records,
    # so an empty client should be impossible. Check anyway rather than trusting it,
    # because an empty client would trip the training scripts' assertion much later and
    # would make compute_metrics divide by zero.
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

    # FedArtML's HD is the RMS over client pairs, which is what compute_metrics reports
    # as hd_pairwise_rms. They come from different inputs -- FedArtML from its own
    # proportion table, ours from the recomputed counts -- so agreement confirms the two
    # views of the partition match.
    if abs(fedartml_hd - metrics["hd_pairwise_rms"]) > 1e-9:
        raise RuntimeError(
            f"alpha={dirichlet_alpha} seed={partition_seed}: FedArtML HD {fedartml_hd!r} "
            f"disagrees with recomputed RMS {metrics['hd_pairwise_rms']!r}")

    record = {
        "alpha": dirichlet_alpha,
        "seed": partition_seed,
        "rejected_empty_client": False,
        "empty_client_ids": "",
        "fedartml_hd": fedartml_hd,
        "hd_pairwise_mean": metrics["hd_pairwise_mean"],
        "hd_pairwise_rms": metrics["hd_pairwise_rms"],
        "zero_cells": int((client_class_counts == 0).sum()),
        "size_min": metrics["size_min"],
        "size_max": metrics["size_max"],
        "size_max_min_ratio": metrics["size_max_min_ratio"],
        "mean_absent_classes_per_client": metrics["mean_absent_classes_per_client"],
    }
    return record, client_indices, metrics, client_class_counts


def calibrate_alpha_to_hd(partitions, y_train, global_class_counts) -> pd.DataFrame:
    """Measure what HD each calibration alpha produces under FedArtML on this dataset.

    The capacity cap moves the mapping relative to the project's own partitioner, and
    Dataset 2's class balance moves it again relative to Dataset 1, so the search
    ranges are derived from this rather than assumed.
    """
    rows = []
    for dirichlet_alpha in CALIBRATION_ALPHAS:
        hd_values = []
        zero_cells = []
        rejected = 0
        for partition_seed in CALIBRATION_SEEDS:
            record, client_indices, _, counts = evaluate_candidate(
                partitions, y_train, global_class_counts, dirichlet_alpha, partition_seed)
            if client_indices is None:
                rejected += 1
                continue
            hd_values.append(record["fedartml_hd"])
            zero_cells.append(record["zero_cells"])
        if not hd_values:
            rows.append({"alpha": dirichlet_alpha, "n": 0, "hd_mean": np.nan,
                         "hd_min": np.nan, "hd_max": np.nan, "zero_cells_mean": np.nan,
                         "rejected": rejected})
            continue
        hd = np.array(hd_values)
        rows.append({
            "alpha": dirichlet_alpha, "n": len(hd), "hd_mean": float(hd.mean()),
            "hd_min": float(hd.min()), "hd_max": float(hd.max()),
            "zero_cells_mean": float(np.mean(zero_cells)), "rejected": rejected,
        })
    return pd.DataFrame(rows)


def sweeps_from_calibration(calibration: pd.DataFrame) -> tuple[dict, dict]:
    """Pick each target's search alphas, and note any target the calibration cannot reach.

    Reachability is judged against the widest HD any calibration seed produced, not
    against the per-alpha means, because a single seed strays a long way from its
    alpha's mean and could still land on the target.
    """
    usable = calibration.dropna(subset=["hd_mean"])
    if len(usable) < ALPHAS_PER_TARGET:
        raise RuntimeError(
            f"only {len(usable)} calibration alphas produced a usable partition, "
            f"need at least {ALPHAS_PER_TARGET}")

    reachable_low = float(usable.hd_min.min())
    reachable_high = float(usable.hd_max.max())

    sweeps = {}
    unreachable = {}
    for target_hd, _ in TARGETS:
        if not (reachable_low - MAX_ABS_ERROR <= target_hd <= reachable_high + MAX_ABS_ERROR):
            unreachable[target_hd] = (reachable_low, reachable_high)
            continue
        ranked = usable.assign(distance=(usable.hd_mean - target_hd).abs())
        ranked = ranked.sort_values("distance").head(ALPHAS_PER_TARGET)
        sweeps[target_hd] = sorted(float(a) for a in ranked.alpha)
    return sweeps, unreachable


def choose_closest(candidates, target_hd):
    """Pick the three candidates nearest the target, each from a different seed."""
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
    """Write one partition's index arrays and manifest, matching script 50's structure."""
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
    """Copy the three existing Dataset-2 IID partitions in as the HD = 0 level.

    These are the project's own IID partitions, not FedArtML's, so that both Dataset-2
    partitioner families share an identical HD = 0 baseline. The manifest says so
    rather than leaving a reader to assume FedArtML generated them.
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

        for source_file in sorted(source_dir.iterdir()):
            target_file = target_dir / source_file.name
            if source_file.read_bytes() != target_file.read_bytes():
                raise RuntimeError(f"{target_file} does not match {source_file}")

        manifest_path = target_dir / "partition_manifest.json"
        manifest = json.load(open(manifest_path))
        manifest["directory_label"] = label
        manifest["hd_condition"] = "iid"
        manifest["target_hd"] = 0.0
        manifest["partitioner"] = "copied from final_partitions (iid_even); not FedArtML"
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
    """Reload one written partition from disk and repeat the training scripts' checks."""
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
    fedartml_sha256 = check_fedartml_source()
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
    print(f"FedArtML source verified, sha256 {fedartml_sha256[:16]}")
    print(f"Dataset 2 y_train: {total_records:,} records, {NUM_CLASSES} classes, "
          f"Benign {100 * benign_share:.2f}%")
    print(f"zero-cell counts are out of {NUM_CLIENTS * NUM_CLASSES}")
    print()

    print(f"CALIBRATION: FedArtML alpha to HD on Dataset 2, "
          f"{len(CALIBRATION_ALPHAS)} alphas x {len(CALIBRATION_SEEDS)} seeds")
    calibration = calibrate_alpha_to_hd(partitions, y_train, global_class_counts)
    header = (f"{'alpha':>8}{'n':>4}{'HD mean':>10}{'HD min':>9}{'HD max':>9}"
              f"{'zero cells':>12}{'rejected':>10}")
    print(header)
    print("-" * len(header))
    for _, row in calibration.iterrows():
        if row["n"] == 0:
            print(f"{row['alpha']:>8.3f}{0:>4}   all candidates rejected")
            continue
        print(f"{row['alpha']:>8.3f}{int(row['n']):>4}{row['hd_mean']:>10.4f}"
              f"{row['hd_min']:>9.4f}{row['hd_max']:>9.4f}"
              f"{row['zero_cells_mean']:>12.1f}{int(row['rejected']):>10}")
    print()

    alpha_sweeps, unreachable = sweeps_from_calibration(calibration)
    print("Search alphas derived from the calibration, nearest calibrated HD per target:")
    for target_hd, condition in TARGETS:
        if target_hd in unreachable:
            low, high = unreachable[target_hd]
            print(f"  HD {target_hd:.2f} ({condition}): UNREACHABLE - calibration spanned "
                  f"HD {low:.4f} to {high:.4f}, target lies outside that by more than "
                  f"{MAX_ABS_ERROR}. Nothing will be written for this condition.")
            continue
        alphas = ", ".join(f"{a:g}" for a in alpha_sweeps[target_hd])
        print(f"  HD {target_hd:.2f} ({condition}): {alphas}")
    print()

    all_candidate_rows = []
    selection_rows = []
    report = []

    for target_hd, condition in TARGETS:
        if target_hd in unreachable:
            low, high = unreachable[target_hd]
            print(f"target HD {target_hd:.2f} ({condition}): SKIPPED, calibration "
                  f"reached only HD {low:.4f} to {high:.4f}")
            report.append({"target_hd": target_hd, "condition": condition,
                           "rejected": 0, "chosen": []})
            print()
            continue

        accepted = []
        rejected = 0
        for dirichlet_alpha in alpha_sweeps[target_hd]:
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

        evaluated = len(alpha_sweeps[target_hd]) * len(CANDIDATE_SEEDS)
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

        # Labels go in ascending generating-seed order so the mapping is reproducible.
        chosen = sorted(choose_closest(accepted, target_hd), key=lambda c: c["seed"])

        # Every one of the three must be within tolerance, not just the best, because
        # each is written into a directory whose name claims this target.
        errors = [abs(c["fedartml_hd"] - target_hd) for c in chosen]
        if max(errors) > MAX_ABS_ERROR:
            achieved = ", ".join(f"{c['fedartml_hd']:.4f}" for c in chosen)
            print(f"  CANNOT MEET TARGET {target_hd:.2f}: nearest three achieved "
                  f"{achieved}, worst error {max(errors):.4f} exceeds the "
                  f"{MAX_ABS_ERROR} tolerance. No partitions written for {condition}.")
            report.append({"target_hd": target_hd, "condition": condition,
                           "rejected": rejected, "chosen": []})
            print()
            continue

        for label, candidate in zip(DIRECTORY_LABELS, chosen):
            out_dir = OUT_ROOT / label / condition
            manifest = write_partition(
                partitions, out_dir, label, condition, target_hd, candidate,
                class_order, global_class_counts, total_records, fedartml_sha256)
            print(f"  {label}/{condition}: HD {manifest['fedartml_hellinger_distance']:.5f} "
                  f"(alpha {candidate['alpha']:g}, seed {candidate['seed']}) "
                  f"zero cells {manifest['absent_cells_total']}/{NUM_CLIENTS * NUM_CLASSES}")
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
    calibration.to_csv(CALIBRATION_PATH, index=False)
    pd.DataFrame(all_candidate_rows).to_csv(CANDIDATES_PATH, index=False)
    pd.DataFrame(selection_rows).to_csv(SELECTION_PATH, index=False)
    print(f"Wrote {CALIBRATION_PATH} ({len(calibration)} alphas)")
    print(f"Wrote {CANDIDATES_PATH} ({len(all_candidate_rows)} candidates)")
    print(f"Wrote {SELECTION_PATH} ({len(selection_rows)} rows)")
    print()

    print("SUMMARY (Dataset 2, FedArtML partitioner)")
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
        achieved = ", ".join(f"{c['fedartml_hd']:.4f}" for c in entry["chosen"])
        seeds = ", ".join(str(c["seed"]) for c in entry["chosen"])
        cells = ", ".join(str(c["zero_cells"]) for c in entry["chosen"])
        print(f"{entry['target_hd']:>7.2f}{entry['condition']:>10}{alphas:>22}"
              f"{achieved:>26}{seeds:>18}{cells:>14}{entry['rejected']:>10}")


if __name__ == "__main__":
    main()
