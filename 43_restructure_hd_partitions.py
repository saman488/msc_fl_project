"""
ONE-OFF MIGRATION, ALREADY APPLIED ON 2026-08-20. DO NOT RERUN.

This script restructured the partitions written by 42_select_partitions_by_hd.py and
added IID as a fifth condition. It is kept in the repository so the layout under
data/fl_clients/hd_selected_partitions/k_5/ is traceable to code rather than to an
undocumented manual edit, not because it needs running again. It refuses to start if
the restructured layout is already present, which it is.

Rerunning it after a fresh 42_ run would be the only legitimate use.

Two changes, both confined to data/fl_clients/hd_selected_partitions/k_5/:

1. Each HD level currently sits under its own generating seed, so the tree has
   seed_1009/hd_0p25 but no seed_1009/hd_0p5. Every level is relabelled onto the
   same three directory names seed_1, seed_2, seed_3, assigned in ascending order
   of the generating seed within each level so the mapping is reproducible rather
   than arbitrary. The generating seed stays recorded in each manifest and a new
   directory_label field records the label, so the mapping reads both ways.

2. The source study's HD = 0 level is IID. The three existing IID partitions are
   copied in from data/fl_clients/final_partitions/k_5/seed_{42,43,44}/iid, mapping
   42 -> seed_1, 43 -> seed_2, 44 -> seed_3. They are copied rather than
   regenerated so the new results stay comparable to the existing runs.

data/fl_clients/final_partitions is read only. Nothing outside the HD-selected root
is created or modified.
"""

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
LABEL_MAP = Path("configs/label_mapping.json")
NEW_ROOT = Path("data/fl_clients/hd_selected_partitions/k_5")
IID_SOURCE_ROOT = Path("data/fl_clients/final_partitions/k_5")
MAPPING_PATH = NEW_ROOT / "seed_mapping.csv"

NUM_CLIENTS = 5
HD_CONDITIONS = ["hd_0p25", "hd_0p5", "hd_0p75", "hd_0p9"]
ALL_CONDITIONS = ["iid"] + HD_CONDITIONS
DIRECTORY_LABELS = ["seed_1", "seed_2", "seed_3"]

# The existing IID partitions carry these generating seeds, in this order.
IID_SOURCE_SEEDS = [42, 43, 44]


def find_existing_hd_partitions():
    """Map each HD condition to its generating-seed directories, ascending.

    Reads the tree rather than the selection summary, so the move is driven by what
    is actually on disk.
    """
    found = {condition: [] for condition in HD_CONDITIONS}
    for seed_dir in sorted(NEW_ROOT.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        label = seed_dir.name[len("seed_"):]
        if not label.isdigit():
            raise RuntimeError(f"Unexpected directory {seed_dir}; expected seed_<number>")
        for condition_dir in sorted(seed_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            if condition_dir.name not in HD_CONDITIONS:
                raise RuntimeError(f"Unexpected condition directory {condition_dir}")
            found[condition_dir.name].append((int(label), condition_dir))

    for condition, entries in found.items():
        if len(entries) != len(DIRECTORY_LABELS):
            raise RuntimeError(
                f"{condition}: found {len(entries)} partitions, expected {len(DIRECTORY_LABELS)}")
        entries.sort(key=lambda pair: pair[0])
    return found


def relabel_hd_partitions(found) -> None:
    """Move each HD partition to its new label directory.

    Done per (condition, generating seed) rather than per seed directory, because a
    generating seed can serve two levels under different labels: seed 1015 is
    seed_2 for hd_0p25 but seed_1 for hd_0p5.
    """
    for condition in HD_CONDITIONS:
        for label, (generating_seed, source_dir) in zip(DIRECTORY_LABELS, found[condition]):
            target_dir = NEW_ROOT / label / condition
            if target_dir.exists():
                raise RuntimeError(f"Refusing to overwrite existing {target_dir}")
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_dir), str(target_dir))
            print(f"  {condition:8} seed_{generating_seed} -> {label}")

    # The old numeric seed directories are now empty shells; remove them so the tree
    # shows only the three labels.
    for seed_dir in sorted(NEW_ROOT.glob("seed_*")):
        if not seed_dir.is_dir() or seed_dir.name in DIRECTORY_LABELS:
            continue
        remaining = list(seed_dir.iterdir())
        if remaining:
            raise RuntimeError(f"{seed_dir} still contains {remaining}; not removing")
        seed_dir.rmdir()


def copy_iid_partitions() -> None:
    """Copy the three existing IID partitions into the new root, unchanged."""
    for label, generating_seed in zip(DIRECTORY_LABELS, IID_SOURCE_SEEDS):
        source_dir = IID_SOURCE_ROOT / f"seed_{generating_seed}" / "iid"
        target_dir = NEW_ROOT / label / "iid"
        if not source_dir.is_dir():
            raise RuntimeError(f"Missing IID source {source_dir}")
        if target_dir.exists():
            raise RuntimeError(f"Refusing to overwrite existing {target_dir}")
        # copytree rather than a file loop, so a source file we did not anticipate
        # is carried over instead of silently dropped.
        shutil.copytree(source_dir, target_dir)
        print(f"  iid      seed_{generating_seed} -> {label}  (copied from {source_dir})")


def add_directory_label(label: str, condition: str) -> dict:
    """Record the directory label inside the manifest and return the manifest."""
    manifest_path = NEW_ROOT / label / condition / "partition_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest {manifest_path}")
    manifest = json.load(open(manifest_path))
    manifest["directory_label"] = label
    manifest["hd_condition"] = condition
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def verify_partition(label: str, condition: str, y_train, num_classes,
                     global_class_counts) -> None:
    """Reload from disk and repeat the checks the training scripts make."""
    partition_dir = NEW_ROOT / label / condition
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

    counts = np.stack([np.bincount(y_train[indices], minlength=num_classes)
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
        raise RuntimeError(f"{partition_dir}: manifest directory_label is "
                           f"{manifest.get('directory_label')!r}, expected {label!r}")


def main() -> None:
    if not NEW_ROOT.is_dir():
        raise RuntimeError(f"Missing {NEW_ROOT}; run 42_select_partitions_by_hd.py first")

    # This migration has already been applied. Detect the finished layout and stop,
    # so an accidental rerun cannot move directories a second time or clobber the
    # IID copies. Either marker alone is enough to conclude it has run.
    already_applied = [path for path in
                       [MAPPING_PATH] + [NEW_ROOT / label for label in DIRECTORY_LABELS]
                       if path.exists()]
    if already_applied:
        listing = "\n  ".join(str(path) for path in already_applied)
        raise RuntimeError(
            "Refusing to run: the restructured layout is already present. This is a "
            "one-off migration that has already been applied.\n"
            f"Found:\n  {listing}")

    name_to_id = json.load(open(LABEL_MAP))
    class_order = [name for name, _ in sorted(name_to_id.items(), key=lambda kv: kv[1])]
    num_classes = len(class_order)
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    global_class_counts = np.bincount(y_train, minlength=num_classes)

    print("Relabelling HD partitions:")
    found = find_existing_hd_partitions()
    relabel_hd_partitions(found)
    print()

    print("Copying IID partitions:")
    copy_iid_partitions()
    print()

    mapping_rows = []
    for label in DIRECTORY_LABELS:
        for condition in ALL_CONDITIONS:
            manifest = add_directory_label(label, condition)
            mapping_rows.append({
                "directory_label": label,
                "condition": condition,
                "generating_seed": manifest["partition_seed"],
                "alpha": manifest["alpha"],
                "achieved_hd_rms": manifest["hd_pairwise_rms"],
                "achieved_hd_pairwise_mean": manifest["hd_pairwise_mean"],
            })

    mapping = pd.DataFrame(mapping_rows)
    # Sort by condition then label so the write-up table reads level by level.
    condition_order = {name: i for i, name in enumerate(ALL_CONDITIONS)}
    mapping = mapping.sort_values(
        by=["condition", "directory_label"],
        key=lambda col: col.map(condition_order) if col.name == "condition" else col)
    mapping.to_csv(MAPPING_PATH, index=False)
    print(f"Wrote {MAPPING_PATH}")
    print()

    print("Verifying all 15 partitions:")
    for label in DIRECTORY_LABELS:
        for condition in ALL_CONDITIONS:
            verify_partition(label, condition, y_train, num_classes, global_class_counts)
            print(f"  OK  {label}/{condition}")
    print()

    print("Final tree:")
    for label in DIRECTORY_LABELS:
        print(f"  {label}/")
        for condition_dir in sorted((NEW_ROOT / label).iterdir()):
            n_files = len(list(condition_dir.iterdir()))
            print(f"    {condition_dir.name:9} ({n_files} files)")
    print(f"  {MAPPING_PATH.name}")
    print()

    print("seed_mapping.csv:")
    print(mapping.to_string(index=False))


if __name__ == "__main__":
    main()
