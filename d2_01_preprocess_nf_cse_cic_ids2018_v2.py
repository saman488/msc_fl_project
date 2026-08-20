"""
Dataset-2 preprocessing: NF-CSE-CIC-IDS2018-v2 -> 7-class group-disjoint splits.

Streaming, checksum-verified preprocessing of the official UQ NetFlow v2 release.
Builds a 36-feature, 7-parent-class supervised dataset with a deterministic,
feature-hash group-disjoint train/validation/test split, a StandardScaler fitted on
training rows only, and full provenance/manifest artefacts. All heavy outputs are
written to staging paths and promoted to final names only after every validation
passes. Never loads the full 18.9M x 36 feature matrix into RAM.

Feature transform: raw float64 values -> StandardScaler.transform() -> clip the
standardised values to [-10, 10] -> cast to float32 -> save. The scaler is still
fitted on training rows only and is unchanged by the clip; the clip is applied
identically to train, validation and test, after standardisation only. There is no
raw-value clipping, no imputation, no winsorisation and no log transform. Every
clipped value is counted and published per feature and per split in
scaled_clipping_summary.csv / .json, and validate_outputs() asserts that every
saved feature lies within [-10, 10].

This file only implements Dataset-2 preprocessing. It does not train models, build
FL clients, compute Dirichlet partitions, calculate class weights, or touch any
existing Dataset-1 artefact.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from numpy.lib.format import open_memmap
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATASET_ROOT = Path("data/nf_cse_cic_ids2018_v2")
RAW_CSV = DATASET_ROOT / "raw/b3427ed8ad063a09_MOHANAD_A4706/data/NF-CSE-CIC-IDS2018-v2.csv"
FEATURE_DEF_CSV = DATASET_ROOT / "raw/b3427ed8ad063a09_MOHANAD_A4706/data/NetFlow_v2_Features.csv"

PROC_FINAL = DATASET_ROOT / "processed"
MODELS_FINAL = Path("models/nf_cse_cic_ids2018_v2")
CONFIGS_FINAL = Path("configs/nf_cse_cic_ids2018_v2")
RESULTS_FINAL = Path("results/nf_cse_cic_ids2018_v2/preprocessing")

STAGING_SUFFIX = ".staging"
PROC_STAGE = PROC_FINAL.with_name(PROC_FINAL.name + STAGING_SUFFIX)
MODELS_STAGE = MODELS_FINAL.with_name(MODELS_FINAL.name + STAGING_SUFFIX)
CONFIGS_STAGE = CONFIGS_FINAL.with_name(CONFIGS_FINAL.name + STAGING_SUFFIX)
RESULTS_STAGE = RESULTS_FINAL.with_name(RESULTS_FINAL.name + STAGING_SUFFIX)

# --------------------------------------------------------------------------- #
# Verified raw provenance
# --------------------------------------------------------------------------- #
EXPECTED_ROWS = 18_893_708
EXPECTED_CSV_SIZE_BYTES = 3_221_378_977
EXPECTED_CSV_SHA1 = "80ff7fd07e2bc31164197f07fe9d9606ee565f9a"
EXPECTED_FEATURE_DEFINITION_SHA1 = "6cf84d2d76c29321175d9429996d7cda2f3b15d6"

OFFICIAL_URL = "https://rdm.uq.edu.au/files/ce5161d0-ef9c-11ed-827d-e762de186848"
DOI = "10.48610/E9636B7"
PACKAGE_ZIP_SHA256 = "ad6307995d08ff8825d0193d5a1cfa61cf3dc4c6f223962ba3762a7775b5dc24"
DOWNLOAD_DATE = "2026-08-11"
SOURCE_ORGANIZATION = "University of Queensland"
LICENCE_TEXT = "Permitted Re-Use with Commercial Use Restriction"

# --------------------------------------------------------------------------- #
# Feature contract
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE",
    "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
]

EXCLUDED_COLUMNS = {
    "IPV4_SRC_ADDR": "Endpoint identifier; shortcut-learning risk.",
    "IPV4_DST_ADDR": "Endpoint identifier; shortcut-learning risk.",
    "L4_SRC_PORT": "Endpoint/service identifier used in source-event matching; shortcut-learning risk.",
    "L4_DST_PORT": "Endpoint/service identifier used in source-event matching; shortcut-learning risk.",
    "SRC_TO_DST_SECOND_BYTES": "Verified invalid extreme values in the official checksum-valid CSV; 250 values exceed float32 range.",
    "DST_TO_SRC_SECOND_BYTES": "Verified invalid extreme values in the official checksum-valid CSV; 5 values exceed float32 range.",
    "FTP_COMMAND_RET_CODE": "Exactly zero for all 18,893,708 rows; zero variance and no information.",
    "Label": "Binary target; never a model input.",
    "Attack": "Source of the multi-class target; never a model input.",
}

NUM_FEATURES = 36
NUM_CLASSES = 7

# Symmetric bound applied to STANDARDISED values only, after StandardScaler.transform()
# and before the float32 cast. Never applied to raw values.
SCALED_CLIP_LIMIT = 10.0

# --------------------------------------------------------------------------- #
# Target taxonomy
# --------------------------------------------------------------------------- #
RAW_TO_PARENT = {
    "Benign": "Benign",
    "Bot": "Bot",
    "FTP-BruteForce": "BruteForce",
    "SSH-Bruteforce": "BruteForce",
    "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-Hulk": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "Infilteration": "Infiltration",
    "Brute Force -Web": "Web Attacks",
    "Brute Force -XSS": "Web Attacks",
    "SQL Injection": "Web Attacks",
}

LABEL_MAPPING = {
    "Benign": 0, "Bot": 1, "BruteForce": 2, "DDoS": 3, "DoS": 4,
    "Infiltration": 5, "Web Attacks": 6,
}
ID_TO_PARENT = {v: k for k, v in LABEL_MAPPING.items()}
RAW_TO_ID = {raw: LABEL_MAPPING[parent] for raw, parent in RAW_TO_PARENT.items()}

EXPECTED_RAW_ATTACK_COUNTS = {
    "Benign": 16_635_567, "Bot": 143_097, "Brute Force -Web": 2_143,
    "Brute Force -XSS": 927, "DDOS attack-HOIC": 1_080_858, "DDOS attack-LOIC-UDP": 2_112,
    "DDoS attacks-LOIC-HTTP": 307_300, "DoS attacks-GoldenEye": 27_723,
    "DoS attacks-Hulk": 432_648, "DoS attacks-SlowHTTPTest": 14_116,
    "DoS attacks-Slowloris": 9_512, "FTP-BruteForce": 25_933, "Infilteration": 116_361,
    "SQL Injection": 432, "SSH-Bruteforce": 94_979,
}

EXPECTED_PARENT_COUNTS = {
    "Benign": 16_635_567, "Bot": 143_097, "BruteForce": 120_912, "DDoS": 1_390_270,
    "DoS": 483_999, "Infiltration": 116_361, "Web Attacks": 3_502,
}
EXPECTED_LABEL0 = 16_635_567
EXPECTED_LABEL1 = 2_258_141

# --------------------------------------------------------------------------- #
# Split configuration
# --------------------------------------------------------------------------- #
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_STATE = 42
PROPORTION_TOLERANCE = 0.005
SPLIT_NAMES = ["train", "validation", "test"]  # codes 0, 1, 2

CHUNKSIZE = 1_000_000

GROUP_HASH_METHOD = "pd.util.hash_pandas_object(chunk[FEATURE_COLUMNS], index=False) -> uint64"
SPLIT_ALGORITHM_DESCRIPTION = (
    "Feature-hash group-disjoint split. Groups are exact uint64 hashes of the 36 "
    "float64 feature values; every row with the same hash shares one split. A single "
    "np.random.default_rng(42).permutation over all groups provides a per-group tie "
    "key used as the secondary ordering key throughout. "
    "Per-class integer targets: val=round(0.15*total), test=round(0.15*total), "
    "train=total-val-test. Assignment order: (A) mixed-label groups by descending "
    "size, then the seed-42 tie key, then ascending group hash, each placed into the "
    "split minimising the normalised squared deviation over the full 3x7 split/class "
    "matrix, exact ties resolved train<validation<test; (B) pure repeated groups "
    "(size>1) per class by descending size, then the seed-42 tie key, then ascending "
    "hash, placed to minimise that class's normalised squared deviation, ties "
    "train<validation<test; (C) singleton groups per class ordered by the seed-42 tie "
    "key then ascending hash, filling positive per-split class deficits largest-first "
    "(ties train<validation<test), then remaining singletons to the split with the "
    "lowest assigned/target ratio (ties train<validation<test). Seed 42 governs the "
    "deterministic tie ordering; the final tie order is always ascending uint64 group hash."
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def sha1_of_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_of_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode()
        return bool(out.strip())
    except Exception:
        return None


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


# --------------------------------------------------------------------------- #
# Safety: preflight
# --------------------------------------------------------------------------- #
def preflight_no_outputs() -> None:
    """Refuse to run if any intended final or staging output already exists."""
    final_dirs = [PROC_FINAL, MODELS_FINAL, CONFIGS_FINAL, RESULTS_FINAL]
    stage_dirs = [PROC_STAGE, MODELS_STAGE, CONFIGS_STAGE, RESULTS_STAGE]
    offenders = [p for p in final_dirs + stage_dirs if p.exists()]
    if offenders:
        listing = "\n  ".join(str(p) for p in offenders)
        raise RuntimeError(
            "Refusing to run; final or staging outputs already exist. Remove or "
            f"inspect them manually first:\n  {listing}"
        )


def make_staging_dirs() -> None:
    for p in [PROC_STAGE, MODELS_STAGE, CONFIGS_STAGE, RESULTS_STAGE]:
        p.mkdir(parents=True, exist_ok=False)


# --------------------------------------------------------------------------- #
# Provenance verification
# --------------------------------------------------------------------------- #
def load_documented_features(path: Path) -> set[str]:
    """Extract the documented feature-name set from the NetFlow v2 features file."""
    df = pd.read_csv(path)
    for col in df.columns:
        values = {str(v).strip() for v in df[col].dropna().tolist()}
        if {"PROTOCOL", "L7_PROTO", "IN_BYTES"}.issubset(values):
            return values
    raise ValueError(f"Could not locate a feature-name column in {path}")


def verify_provenance() -> dict:
    """Verify raw CSV and feature-definition provenance; fail on any mismatch."""
    assert RAW_CSV.exists(), f"Raw CSV not found: {RAW_CSV}"
    assert FEATURE_DEF_CSV.exists(), f"Feature-definition file not found: {FEATURE_DEF_CSV}"

    csv_size = RAW_CSV.stat().st_size
    assert csv_size == EXPECTED_CSV_SIZE_BYTES, (
        f"CSV byte size {csv_size} != expected {EXPECTED_CSV_SIZE_BYTES}"
    )
    print("Provenance: hashing raw CSV (SHA-1)...", flush=True)
    csv_sha1 = sha1_of_file(RAW_CSV)
    assert csv_sha1 == EXPECTED_CSV_SHA1, f"CSV SHA-1 {csv_sha1} != expected {EXPECTED_CSV_SHA1}"
    fdef_sha1 = sha1_of_file(FEATURE_DEF_CSV)
    assert fdef_sha1 == EXPECTED_FEATURE_DEFINITION_SHA1, (
        f"Feature-definition SHA-1 {fdef_sha1} != expected {EXPECTED_FEATURE_DEFINITION_SHA1}"
    )

    header = pd.read_csv(RAW_CSV, nrows=0).columns.tolist()
    assert len(header) == 45, f"Raw CSV has {len(header)} columns, expected 45"
    assert len(set(header)) == len(header), "Duplicate column names in raw CSV header"
    assert header[-2] == "Label" and header[-1] == "Attack", (
        f"Last two columns are {header[-2:]}, expected ['Label', 'Attack']"
    )
    raw_features = [c for c in header if c not in ("Label", "Attack")]
    assert len(raw_features) == 43, f"Expected 43 NetFlow features, found {len(raw_features)}"

    documented = load_documented_features(FEATURE_DEF_CSV)
    assert documented == set(raw_features), (
        "Documented 43-feature set does not equal the raw 43-feature set.\n"
        f"  only in documented: {sorted(documented - set(raw_features))}\n"
        f"  only in raw:        {sorted(set(raw_features) - documented)}"
    )

    # Feature contract sanity against the raw header.
    assert len(FEATURE_COLUMNS) == NUM_FEATURES, "FEATURE_COLUMNS length must be 36"
    assert len(set(FEATURE_COLUMNS)) == NUM_FEATURES, "FEATURE_COLUMNS has duplicates"
    missing = [c for c in FEATURE_COLUMNS if c not in header]
    assert not missing, f"Feature columns missing from raw CSV: {missing}"
    missing_excl = [c for c in EXCLUDED_COLUMNS if c not in header]
    assert not missing_excl, f"Excluded columns missing from raw CSV: {missing_excl}"
    assert set(FEATURE_COLUMNS) | set(EXCLUDED_COLUMNS) == set(header), (
        "Feature + excluded columns do not partition the raw header exactly"
    )

    print("Provenance verified.", flush=True)
    return {"header": header, "raw_features": raw_features,
            "csv_sha1": csv_sha1, "feature_definition_sha1": fdef_sha1,
            "csv_size_bytes": csv_size}


# --------------------------------------------------------------------------- #
# First streaming pass: hashes, labels, target validation
# --------------------------------------------------------------------------- #
def first_pass() -> tuple[np.ndarray, np.ndarray]:
    """Stream the CSV once; return (feature_hash uint64, parent_label uint8)."""
    feature_hash = np.empty(EXPECTED_ROWS, dtype=np.uint64)
    parent_label = np.empty(EXPECTED_ROWS, dtype=np.uint8)
    raw_counter: Counter = Counter()

    dtypes = {c: np.float64 for c in FEATURE_COLUMNS}
    dtypes["Label"] = "string"
    dtypes["Attack"] = "string"
    usecols = FEATURE_COLUMNS + ["Label", "Attack"]

    offset = 0
    reader = pd.read_csv(RAW_CSV, usecols=usecols, dtype=dtypes, chunksize=CHUNKSIZE)
    for chunk_index, chunk in enumerate(reader):
        n = len(chunk)

        features = chunk[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        assert np.isfinite(features).all(), f"Chunk {chunk_index}: non-finite feature value"

        attack = chunk["Attack"]
        assert attack.notna().all(), f"Chunk {chunk_index}: empty Attack value"
        attack_str = attack.astype(str)
        assert (attack_str.str.len() > 0).all(), f"Chunk {chunk_index}: blank Attack value"
        unknown = set(attack_str.unique()) - set(RAW_TO_PARENT)
        assert not unknown, f"Chunk {chunk_index}: unknown Attack labels {sorted(unknown)}"

        label = chunk["Label"]
        assert label.notna().all(), f"Chunk {chunk_index}: empty Label value"
        label_int = pd.to_numeric(label, errors="raise").to_numpy()
        assert np.isin(label_int, (0, 1)).all(), f"Chunk {chunk_index}: non-binary Label value"

        is_benign = (attack_str == "Benign").to_numpy()
        assert np.array_equal(label_int == 0, is_benign), (
            f"Chunk {chunk_index}: Label==0 not equivalent to Attack=='Benign'"
        )
        assert np.array_equal(label_int == 1, ~is_benign), (
            f"Chunk {chunk_index}: Label==1 not equivalent to Attack!='Benign'"
        )

        pid = attack_str.map(RAW_TO_ID)
        assert pid.notna().all(), f"Chunk {chunk_index}: Attack failed to map to a parent id"
        parent_label[offset:offset + n] = pid.to_numpy(dtype=np.uint8)

        h = pd.util.hash_pandas_object(chunk[FEATURE_COLUMNS], index=False).to_numpy(dtype=np.uint64)
        feature_hash[offset:offset + n] = h

        raw_counter.update(Counter(attack_str.tolist()))
        offset += n
        print(f"Pass 1: chunk {chunk_index} rows={offset:,}", flush=True)

    assert offset == EXPECTED_ROWS, f"Read {offset} rows, expected {EXPECTED_ROWS}"

    # Raw attack counts must match exactly.
    assert dict(raw_counter) == EXPECTED_RAW_ATTACK_COUNTS, "Raw attack counts differ from expected"

    # Parent counts must match exactly.
    parent_counts = np.bincount(parent_label, minlength=NUM_CLASSES)
    for cid in range(NUM_CLASSES):
        name = ID_TO_PARENT[cid]
        assert int(parent_counts[cid]) == EXPECTED_PARENT_COUNTS[name], (
            f"Parent count for {name} = {parent_counts[cid]} != {EXPECTED_PARENT_COUNTS[name]}"
        )
    # Binary reconciliation.
    assert int(parent_counts[0]) == EXPECTED_LABEL0, "Benign count != expected Label 0 count"
    assert int(EXPECTED_ROWS - parent_counts[0]) == EXPECTED_LABEL1, "Non-benign count != expected Label 1 count"

    print("Pass 1 complete: schema, targets and counts verified.", flush=True)
    return feature_hash, parent_label


# --------------------------------------------------------------------------- #
# Grouping and target computation
# --------------------------------------------------------------------------- #
def build_groups(feature_hash: np.ndarray, parent_label: np.ndarray) -> dict:
    """Sort by feature hash and derive per-group size, hash, min/max label, order."""
    order = np.argsort(feature_hash, kind="stable")
    sorted_hash = feature_hash[order]
    sorted_parent = parent_label[order]

    change = np.empty(feature_hash.shape[0], dtype=bool)
    change[0] = True
    np.not_equal(sorted_hash[1:], sorted_hash[:-1], out=change[1:])
    group_starts = np.nonzero(change)[0].astype(np.int64)
    group_sizes = np.diff(np.append(group_starts, feature_hash.shape[0])).astype(np.int64)
    group_hash = sorted_hash[group_starts]
    group_min = np.minimum.reduceat(sorted_parent, group_starts)
    group_max = np.maximum.reduceat(sorted_parent, group_starts)

    return {"order": order, "sorted_parent": sorted_parent,
            "group_starts": group_starts, "group_sizes": group_sizes,
            "group_hash": group_hash, "group_min": group_min, "group_max": group_max}


def compute_targets(parent_label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return class_total[7] and target matrix T[3,7] (train, validation, test)."""
    class_total = np.bincount(parent_label, minlength=NUM_CLASSES).astype(np.int64)
    val_target = np.array([int(round(t * VALIDATION_FRACTION)) for t in class_total], dtype=np.int64)
    test_target = np.array([int(round(t * TEST_FRACTION)) for t in class_total], dtype=np.int64)
    train_target = class_total - val_target - test_target
    T = np.stack([train_target, val_target, test_target], axis=0)
    return class_total, T


# --------------------------------------------------------------------------- #
# Deterministic unit-allocation sequences (for singleton assignment)
# --------------------------------------------------------------------------- #
def greedy_deficit_sequence(deficits: np.ndarray) -> np.ndarray:
    """Sequence of split codes filling positive deficits largest-first (ties by index)."""
    bins_parts, rem_parts = [], []
    for s in range(3):
        d = int(deficits[s])
        if d > 0:
            rem_parts.append(np.arange(d, 0, -1, dtype=np.int64))
            bins_parts.append(np.full(d, s, dtype=np.int64))
    if not bins_parts:
        return np.empty(0, dtype=np.int64)
    rem = np.concatenate(rem_parts)
    bins = np.concatenate(bins_parts)
    order = np.lexsort((bins, -rem))  # primary: remaining descending; tie: split index ascending
    return bins[order]


def ratio_sequence(start: np.ndarray, denom: np.ndarray, count: int) -> np.ndarray:
    """Sequence of `count` split codes minimising assigned/target ratio (ties by index)."""
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    bins_parts, ratio_parts = [], []
    steps = np.arange(count, dtype=np.float64)
    for s in range(3):
        ratio_parts.append((start[s] + steps) / denom[s])
        bins_parts.append(np.full(count, s, dtype=np.int64))
    ratios = np.concatenate(ratio_parts)
    bins = np.concatenate(bins_parts)
    order = np.lexsort((bins, ratios))  # primary: ratio ascending; tie: split index ascending
    return bins[order][:count]


# --------------------------------------------------------------------------- #
# Group -> split assignment
# --------------------------------------------------------------------------- #
def assign_groups(groups: dict, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each group to one split; return (group_split uint8, assigned[3,7])."""
    group_sizes = groups["group_sizes"]
    group_hash = groups["group_hash"]
    group_min = groups["group_min"]
    group_max = groups["group_max"]
    group_starts = groups["group_starts"]
    sorted_parent = groups["sorted_parent"]

    n_groups = group_sizes.shape[0]
    group_split = np.full(n_groups, 255, dtype=np.uint8)
    assigned = np.zeros((3, NUM_CLASSES), dtype=np.int64)
    T_safe = np.maximum(T, 1)

    # One reproducible seed-42 per-group tie key/rank; used as the secondary
    # ordering key throughout, with the ascending uint64 group hash as the final
    # tie-break. This gives RANDOM_STATE an actual computational effect.
    tie_key = np.random.default_rng(RANDOM_STATE).permutation(n_groups)

    # (A) mixed-label groups: descending size, then seeded tie key, then ascending hash.
    mixed_gids = np.nonzero(group_min != group_max)[0]
    mixed_order = mixed_gids[np.lexsort((group_hash[mixed_gids], tie_key[mixed_gids], -group_sizes[mixed_gids]))]
    for gid in mixed_order:
        start = int(group_starts[gid])
        cvec = np.bincount(sorted_parent[start:start + int(group_sizes[gid])], minlength=NUM_CLASSES)
        best_split, best_score = 0, np.inf
        for split in range(3):
            candidate = assigned.copy()
            candidate[split] += cvec
            score = float((((candidate - T) / T_safe) ** 2).sum())
            if score < best_score - 1e-12:
                best_score, best_split = score, split
        assigned[best_split] += cvec
        group_split[gid] = best_split
    print(f"Assigned {len(mixed_order):,} mixed-label groups.", flush=True)

    pure_mask = group_min == group_max

    # (B) pure repeated groups (size > 1), per class.
    for c in range(NUM_CLASSES):
        sel = np.nonzero(pure_mask & (group_min == c) & (group_sizes > 1))[0]
        # descending size, then seeded tie key, then ascending hash.
        sel_order = sel[np.lexsort((group_hash[sel], tie_key[sel], -group_sizes[sel]))]
        tc = T[:, c].astype(np.float64)
        tc_safe = np.maximum(T[:, c], 1).astype(np.float64)
        for gid in sel_order:
            size = int(group_sizes[gid])
            best_split, best_score = 0, np.inf
            for split in range(3):
                counts = assigned[:, c].astype(np.float64).copy()
                counts[split] += size
                score = float((((counts - tc) / tc_safe) ** 2).sum())
                if score < best_score - 1e-12:
                    best_score, best_split = score, split
            assigned[best_split, c] += size
            group_split[gid] = best_split

    # (C) singleton groups, per class.
    for c in range(NUM_CLASSES):
        sel = np.nonzero((group_sizes == 1) & (group_min == c))[0]
        if sel.size == 0:
            continue
        sel_order = sel[np.lexsort((group_hash[sel], tie_key[sel]))]  # seeded tie key, then ascending hash
        n = sel_order.shape[0]

        deficits = np.maximum(T[:, c] - assigned[:, c], 0).astype(np.int64)
        seq1 = greedy_deficit_sequence(deficits)
        if seq1.shape[0] >= n:
            combined = seq1[:n]
        else:
            start_assigned = (assigned[:, c] + deficits).astype(np.float64)
            denom = np.maximum(T[:, c], 1).astype(np.float64)
            seq2 = ratio_sequence(start_assigned, denom, n - seq1.shape[0])
            combined = np.concatenate([seq1, seq2])

        group_split[sel_order] = combined.astype(np.uint8)
        assigned[:, c] += np.bincount(combined, minlength=3).astype(np.int64)

    assert (group_split != 255).all(), "Some group was left unassigned"
    print("All groups assigned to a split.", flush=True)
    return group_split, assigned


def map_row_split(groups: dict, group_split: np.ndarray, n_rows: int) -> np.ndarray:
    """Expand per-group split codes back to per-row (raw order) split codes."""
    sorted_split = np.repeat(group_split, groups["group_sizes"])
    row_split = np.empty(n_rows, dtype=np.uint8)
    row_split[groups["order"]] = sorted_split
    return row_split


# --------------------------------------------------------------------------- #
# Split validation
# --------------------------------------------------------------------------- #
def validate_split(row_split: np.ndarray, parent_label: np.ndarray,
                   class_total: np.ndarray, T: np.ndarray) -> dict:
    """Assert all group-disjoint split invariants; return achieved counts/proportions."""
    n = row_split.shape[0]
    assert np.isin(row_split, (0, 1, 2)).all(), "Invalid split code present"

    split_indices = [np.nonzero(row_split == s)[0] for s in range(3)]
    sizes = [int(idx.shape[0]) for idx in split_indices]
    assert sum(sizes) == n, "Split sizes do not sum to the row count"

    union = np.concatenate(split_indices)
    assert union.shape[0] == n, "A row was dropped or duplicated across splits"
    assert np.array_equal(np.sort(union), np.arange(n)), "Split union is not exactly 0..N-1"
    for idx in split_indices:
        assert np.all(np.diff(idx) > 0), "Raw indices within a split are not strictly increasing"

    achieved = np.zeros((3, NUM_CLASSES), dtype=np.int64)
    for s in range(3):
        achieved[s] = np.bincount(parent_label[split_indices[s]], minlength=NUM_CLASSES)
    assert np.array_equal(achieved.sum(axis=0), class_total), "Split class totals != global totals"
    assert (achieved > 0).all(), "A class is missing from one of the splits"

    proportions = np.array([s / n for s in sizes], dtype=np.float64)
    targets = np.array([TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION])
    deviations = np.abs(proportions - targets)
    assert (deviations <= PROPORTION_TOLERANCE).all(), (
        f"Overall proportions {proportions} exceed tolerance vs {targets}"
    )

    return {"split_indices": split_indices, "sizes": sizes, "achieved": achieved,
            "proportions": proportions, "deviations": deviations}


# --------------------------------------------------------------------------- #
# Second streaming pass: scaler fit on train rows only
# --------------------------------------------------------------------------- #
def fit_scaler(row_split: np.ndarray, n_train: int) -> StandardScaler:
    """Fit StandardScaler with partial_fit over training rows only (streaming)."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    seen = 0
    offset = 0
    reader = pd.read_csv(RAW_CSV, usecols=FEATURE_COLUMNS, dtype={c: np.float64 for c in FEATURE_COLUMNS},
                         chunksize=CHUNKSIZE)
    for chunk_index, chunk in enumerate(reader):
        n = len(chunk)
        mask = row_split[offset:offset + n] == 0
        count = int(mask.sum())
        if count:
            features = chunk[FEATURE_COLUMNS].to_numpy(dtype=np.float64)[mask]
            scaler.partial_fit(features)
            seen += count
        offset += n
        print(f"Pass 2 (scaler): chunk {chunk_index} train_rows_seen={seen:,}", flush=True)

    assert offset == EXPECTED_ROWS, "Scaler pass did not stream all rows"
    assert seen == n_train, f"Scaler saw {seen} train rows, expected {n_train}"
    assert scaler.n_features_in_ == NUM_FEATURES, "Scaler n_features_in_ != 36"
    assert np.isfinite(scaler.mean_).all(), "Scaler means are not finite"
    assert np.isfinite(scaler.var_).all(), "Scaler variances are not finite"
    assert np.isfinite(scaler.scale_).all() and (scaler.scale_ > 0).all(), "Scaler scales must be finite and > 0"
    print("Pass 2 complete: scaler fitted on training rows only.", flush=True)
    return scaler


# --------------------------------------------------------------------------- #
# Third streaming pass: write scaled arrays
# --------------------------------------------------------------------------- #
def write_arrays(row_split: np.ndarray, parent_label: np.ndarray,
                 scaler: StandardScaler, split_info: dict) -> dict:
    """Stream, scale and write X/y/raw-index staging arrays without holding X in RAM.

    Transform stage: raw float64 -> scaler.transform() -> clip to
    [-SCALED_CLIP_LIMIT, SCALED_CLIP_LIMIT] -> float32. Finiteness is asserted on
    the standardised values before the clip, so an infinite value still fails
    rather than being silently bounded. Returns the clipping accounting.
    """
    sizes = split_info["sizes"]
    split_indices = split_info["split_indices"]

    # Clipping accounting, per split x feature.
    clipped_values = np.zeros((3, NUM_FEATURES), dtype=np.int64)
    clipped_below = np.zeros((3, NUM_FEATURES), dtype=np.int64)
    clipped_above = np.zeros((3, NUM_FEATURES), dtype=np.int64)
    clipped_rows = np.zeros(3, dtype=np.int64)

    x_paths = [PROC_STAGE / f"X_{name}.npy" for name in ("train", "val", "test")]
    x_arrays = [open_memmap(p, mode="w+", dtype=np.float32, shape=(sizes[s], NUM_FEATURES))
                for s, p in enumerate(x_paths)]

    # y and raw indices are 1-D; derive directly from the split membership.
    for s, name in enumerate(("train", "val", "test")):
        idx = split_indices[s].astype(np.int64)
        np.save(PROC_STAGE / f"raw_indices_{name}.npy", idx)
        np.save(PROC_STAGE / f"y_{name}.npy", parent_label[idx].astype(np.int64))

    cursors = [0, 0, 0]
    offset = 0
    reader = pd.read_csv(RAW_CSV, usecols=FEATURE_COLUMNS, dtype={c: np.float64 for c in FEATURE_COLUMNS},
                         chunksize=CHUNKSIZE)
    for chunk_index, chunk in enumerate(reader):
        n = len(chunk)
        features = chunk[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        standardized = scaler.transform(features)
        assert np.isfinite(standardized).all(), f"Chunk {chunk_index}: non-finite standardized value"

        below = standardized < -SCALED_CLIP_LIMIT
        above = standardized > SCALED_CLIP_LIMIT
        clipped_mask = below | above
        transformed = np.clip(standardized, -SCALED_CLIP_LIMIT, SCALED_CLIP_LIMIT).astype(np.float32)

        codes = row_split[offset:offset + n]
        for s in range(3):
            mask = codes == s
            count = int(mask.sum())
            if count:
                x_arrays[s][cursors[s]:cursors[s] + count] = transformed[mask]
                cursors[s] += count
                split_clipped = clipped_mask[mask]
                clipped_values[s] += split_clipped.sum(axis=0).astype(np.int64)
                clipped_below[s] += below[mask].sum(axis=0).astype(np.int64)
                clipped_above[s] += above[mask].sum(axis=0).astype(np.int64)
                clipped_rows[s] += int(split_clipped.any(axis=1).sum())
        offset += n
        print(f"Pass 3 (write): chunk {chunk_index} rows={offset:,}", flush=True)

    assert offset == EXPECTED_ROWS, "Write pass did not stream all rows"
    # Verify all cursors first, then flush every memmap without mutating the list
    # during indexed iteration, then release the list.
    for s in range(3):
        assert cursors[s] == sizes[s], f"Wrote {cursors[s]} rows to split {s}, expected {sizes[s]}"
    for array in x_arrays:
        array.flush()
    del x_arrays
    total_clipped = int(clipped_values.sum())
    print(f"Pass 3 complete: scaled arrays written. Clipped {total_clipped:,} standardized "
          f"values to +/-{SCALED_CLIP_LIMIT:g} across "
          f"{int(clipped_rows.sum()):,} rows.", flush=True)
    return {"clipped_values": clipped_values, "clipped_below": clipped_below,
            "clipped_above": clipped_above, "clipped_rows": clipped_rows,
            "split_rows": np.asarray(sizes, dtype=np.int64)}


def validate_outputs() -> None:
    """Reopen every staging array read-only and re-verify shape/dtype/range/finiteness.

    Includes the post-standardisation clip bound: every saved feature value must
    satisfy -SCALED_CLIP_LIMIT <= x <= SCALED_CLIP_LIMIT.
    """
    all_indices = []
    for name in ("train", "val", "test"):
        x = np.load(PROC_STAGE / f"X_{name}.npy", mmap_mode="r")
        y = np.load(PROC_STAGE / f"y_{name}.npy", mmap_mode="r")
        idx = np.load(PROC_STAGE / f"raw_indices_{name}.npy", mmap_mode="r")
        assert x.dtype == np.float32 and x.shape[1] == NUM_FEATURES, f"{name}: bad X dtype/width"
        assert y.dtype == np.int64 and idx.dtype == np.int64, f"{name}: bad y/index dtype"
        assert x.shape[0] == y.shape[0] == idx.shape[0], f"{name}: row-count mismatch"
        assert np.all(np.diff(idx) > 0), f"{name}: raw indices not strictly increasing"
        for start in range(0, x.shape[0], CHUNKSIZE):
            block = np.asarray(x[start:start + CHUNKSIZE])
            assert np.isfinite(block).all(), f"{name}: non-finite value on reopen"
            assert block.min() >= -SCALED_CLIP_LIMIT and block.max() <= SCALED_CLIP_LIMIT, (
                f"{name}: feature value outside [-{SCALED_CLIP_LIMIT:g}, {SCALED_CLIP_LIMIT:g}] "
                f"on reopen (min {block.min()}, max {block.max()})"
            )
            yb = np.asarray(y[start:start + CHUNKSIZE])
            assert yb.min() >= 0 and yb.max() <= NUM_CLASSES - 1, f"{name}: label out of range"
        all_indices.append(np.asarray(idx))
    union = np.concatenate(all_indices)
    assert union.shape[0] == EXPECTED_ROWS, "Reopen: total rows != expected"
    assert np.array_equal(np.sort(union), np.arange(EXPECTED_ROWS)), "Reopen: indices not exactly 0..N-1"
    print("Output arrays re-validated on reopen.", flush=True)


# --------------------------------------------------------------------------- #
# Artefacts
# --------------------------------------------------------------------------- #
def write_result_tables(class_total: np.ndarray, T: np.ndarray, split_info: dict,
                        scaler: StandardScaler, raw_counter_expected: dict) -> None:
    """Write the CSV/JSON result tables into the results staging directory."""
    achieved = split_info["achieved"]
    sizes = split_info["sizes"]
    proportions = split_info["proportions"]

    pd.DataFrame({"position": range(NUM_FEATURES), "feature": FEATURE_COLUMNS}).to_csv(
        RESULTS_STAGE / "feature_columns.csv", index=False)

    pd.DataFrame(
        [{"excluded_column": k, "reason": v} for k, v in EXCLUDED_COLUMNS.items()]
    ).to_csv(RESULTS_STAGE / "excluded_columns.csv", index=False)

    pd.DataFrame(
        [{"raw_label": raw, "count": raw_counter_expected[raw], "parent_class": RAW_TO_PARENT[raw]}
         for raw in EXPECTED_RAW_ATTACK_COUNTS]
    ).to_csv(RESULTS_STAGE / "raw_attack_distribution.csv", index=False)

    pd.DataFrame(
        [{"class_id": cid, "parent_class": ID_TO_PARENT[cid]} for cid in range(NUM_CLASSES)]
    ).to_csv(RESULTS_STAGE / "label_mapping.csv", index=False)

    dist_rows = []
    for s, sname in enumerate(SPLIT_NAMES):
        for cid in range(NUM_CLASSES):
            count = int(achieved[s, cid])
            dist_rows.append({
                "split": sname, "class_id": cid, "class_name": ID_TO_PARENT[cid],
                "count": count,
                "fraction_within_split": count / sizes[s] if sizes[s] else 0.0,
                "fraction_of_global_class": count / int(class_total[cid]) if class_total[cid] else 0.0,
            })
    pd.DataFrame(dist_rows).to_csv(RESULTS_STAGE / "class_distribution.csv", index=False)

    target_fracs = [TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION]
    pd.DataFrame([
        {"split": SPLIT_NAMES[s], "rows": sizes[s], "achieved_fraction": float(proportions[s]),
         "target_fraction": target_fracs[s], "deviation": float(proportions[s] - target_fracs[s])}
        for s in range(3)
    ]).to_csv(RESULTS_STAGE / "split_summary.csv", index=False)

    pd.DataFrame({
        "position": range(NUM_FEATURES), "feature": FEATURE_COLUMNS,
        "train_mean": scaler.mean_, "train_variance": scaler.var_, "train_scale": scaler.scale_,
        "scaler_sample_count": int(scaler.n_samples_seen_),
    }).to_csv(RESULTS_STAGE / "scaler_summary.csv", index=False)


def write_clipping_summary(clipping: dict) -> None:
    """Publish the post-standardisation clipping accounting per feature and split."""
    clipped_values = clipping["clipped_values"]
    clipped_below = clipping["clipped_below"]
    clipped_above = clipping["clipped_above"]
    clipped_rows = clipping["clipped_rows"]
    split_rows = clipping["split_rows"]

    rows = []
    for s, sname in enumerate(SPLIT_NAMES):
        split_values = int(split_rows[s]) * NUM_FEATURES
        for position, feature in enumerate(FEATURE_COLUMNS):
            count = int(clipped_values[s, position])
            rows.append({
                "split": sname,
                "feature_position": position,
                "feature": feature,
                "clipped_values": count,
                "clipped_below_lower": int(clipped_below[s, position]),
                "clipped_above_upper": int(clipped_above[s, position]),
                "split_rows": int(split_rows[s]),
                "fraction_of_split_rows_clipped": (count / int(split_rows[s])) if split_rows[s] else 0.0,
                "fraction_of_split_values_clipped": (count / split_values) if split_values else 0.0,
                "rows_with_at_least_one_clipped_feature": int(clipped_rows[s]),
            })
    pd.DataFrame(rows).to_csv(RESULTS_STAGE / "scaled_clipping_summary.csv", index=False)

    total_rows = int(split_rows.sum())
    total_values = total_rows * NUM_FEATURES
    total_clipped = int(clipped_values.sum())
    summary = {
        "scaled_clip_limit": SCALED_CLIP_LIMIT,
        "clip_stage": "after StandardScaler.transform(), before the float32 cast",
        "applied_to_splits": SPLIT_NAMES,
        "raw_value_clipping": False,
        "per_split": {
            SPLIT_NAMES[s]: {
                "rows": int(split_rows[s]),
                "values": int(split_rows[s]) * NUM_FEATURES,
                "clipped_values": int(clipped_values[s].sum()),
                "clipped_below_lower": int(clipped_below[s].sum()),
                "clipped_above_upper": int(clipped_above[s].sum()),
                "rows_with_at_least_one_clipped_feature": int(clipped_rows[s]),
                "fraction_of_rows_with_any_clip": (int(clipped_rows[s]) / int(split_rows[s])) if split_rows[s] else 0.0,
                "fraction_of_values_clipped": (int(clipped_values[s].sum()) / (int(split_rows[s]) * NUM_FEATURES)) if split_rows[s] else 0.0,
                "clipped_values_by_feature": {
                    FEATURE_COLUMNS[p]: int(clipped_values[s, p]) for p in range(NUM_FEATURES)
                },
            } for s in range(3)
        },
        "total_rows": total_rows,
        "total_values": total_values,
        "total_clipped_values": total_clipped,
        "total_rows_with_at_least_one_clipped_feature": int(clipped_rows.sum()),
        "total_fraction_of_values_clipped": (total_clipped / total_values) if total_values else 0.0,
    }
    with open(RESULTS_STAGE / "scaled_clipping_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def write_group_split_summary(groups: dict, group_split: np.ndarray, T: np.ndarray,
                             split_info: dict) -> None:
    group_sizes = groups["group_sizes"]
    group_min = groups["group_min"]
    group_max = groups["group_max"]
    n_groups = int(group_sizes.shape[0])
    repeated_mask = group_sizes > 1
    mixed_mask = group_min != group_max

    # Explicit group-disjoint validation before publishing the summary.
    assert np.isin(group_split, (0, 1, 2)).all(), "group_split contains a code outside {0,1,2}"
    assert int(group_split.shape[0]) == n_groups, "group assignment count != number of unique groups"
    all_classes_present = bool((split_info["achieved"] > 0).all())
    assert all_classes_present, "not all classes are present in all splits"
    group_overlap_zero = True  # each group maps to exactly one split code (asserted above)

    summary = {
        "group_hash_method": GROUP_HASH_METHOD,
        "seed": RANDOM_STATE,
        "num_unique_hash_groups": n_groups,
        "num_repeated_groups": int(repeated_mask.sum()),
        "rows_in_repeated_groups": int(group_sizes[repeated_mask].sum()),
        "num_mixed_label_groups": int(mixed_mask.sum()),
        "rows_in_mixed_label_groups": int(group_sizes[mixed_mask].sum()),
        "largest_group_size": int(group_sizes.max()),
        "target_split_counts_by_class": {
            SPLIT_NAMES[s]: {ID_TO_PARENT[c]: int(T[s, c]) for c in range(NUM_CLASSES)}
            for s in range(3)
        },
        "achieved_split_counts_by_class": {
            SPLIT_NAMES[s]: {ID_TO_PARENT[c]: int(split_info["achieved"][s, c]) for c in range(NUM_CLASSES)}
            for s in range(3)
        },
        "achieved_overall_proportions": {
            SPLIT_NAMES[s]: float(split_info["proportions"][s]) for s in range(3)
        },
        "per_class_achieved_proportions": {
            ID_TO_PARENT[c]: {
                SPLIT_NAMES[s]: (int(split_info["achieved"][s, c]) / int(split_info["achieved"][:, c].sum()))
                for s in range(3)
            } for c in range(NUM_CLASSES)
        },
        "group_overlap_zero": group_overlap_zero,
        "all_classes_present_in_all_splits": all_classes_present,
        "zero_group_overlap_assertion": "each feature-hash group is assigned to exactly one split (enforced by construction and validated)",
        "all_classes_present_assertion": "all seven classes occur in all three splits (validated)",
        "split_assignment_algorithm": SPLIT_ALGORITHM_DESCRIPTION,
    }
    with open(RESULTS_STAGE / "group_split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def write_provenance_json(provenance: dict) -> None:
    payload = {
        "official_url": OFFICIAL_URL,
        "doi": DOI,
        "source_organization": SOURCE_ORGANIZATION,
        "creator": "Mr Mohanad Sarhan",
        "licence_text": LICENCE_TEXT,
        "package_filename": "b3427ed8ad063a09_MOHANAD_A4706.zip",
        "package_identifier": "b3427ed8ad063a09_MOHANAD_A4706",
        "download_date": DOWNLOAD_DATE,
        "package_zip_sha256": PACKAGE_ZIP_SHA256,
        "raw_csv_path": str(RAW_CSV),
        "raw_csv_size_bytes": provenance["csv_size_bytes"],
        "raw_csv_sha1": provenance["csv_sha1"],
        "feature_definition_path": str(FEATURE_DEF_CSV),
        "feature_definition_sha1": provenance["feature_definition_sha1"],
    }
    with open(RESULTS_STAGE / "source_provenance.json", "w") as f:
        json.dump(payload, f, indent=2)


def write_configs(class_total: np.ndarray, T: np.ndarray, split_info: dict,
                  provenance: dict) -> None:
    with open(CONFIGS_STAGE / "label_mapping.json", "w") as f:
        json.dump(LABEL_MAPPING, f, indent=2)
    with open(CONFIGS_STAGE / "raw_to_parent_mapping.json", "w") as f:
        json.dump(RAW_TO_PARENT, f, indent=2)

    config = {
        "task": "multiclass_intrusion_detection_7class",
        "parent_classes": [ID_TO_PARENT[c] for c in range(NUM_CLASSES)],
        "raw_to_parent_mapping": RAW_TO_PARENT,
        "label_mapping": LABEL_MAPPING,
        "feature_columns": FEATURE_COLUMNS,
        "num_features": NUM_FEATURES,
        "excluded_columns": EXCLUDED_COLUMNS,
        "one_hot_encoding": False,
        "numerical_representation": (
            "all 36 features treated as float32 numerical inputs; no encoding/embedding/"
            "bit-expansion/log transform/winsorisation/imputation. The only bounding is a "
            "symmetric clip of the STANDARDISED values to [-10, 10], applied after "
            "StandardScaler.transform() and before the float32 cast."
        ),
        "group_hash_method": GROUP_HASH_METHOD,
        "split_method": "feature_hash_group_disjoint",
        "split_algorithm": SPLIT_ALGORITHM_DESCRIPTION,
        "target_fractions": {"train": TRAIN_FRACTION, "validation": VALIDATION_FRACTION, "test": TEST_FRACTION},
        "achieved_fractions": {SPLIT_NAMES[s]: float(split_info["proportions"][s]) for s in range(3)},
        "seed": RANDOM_STATE,
        "deduplication": False,
        "relabelling": False,
        "all_rows_retained": True,
        "total_rows": EXPECTED_ROWS,
        "scaler": "StandardScaler",
        "scaler_with_mean": True,
        "scaler_with_std": True,
        "scaler_fit_split": "train_only",
        "transform_pipeline": [
            "raw float64 feature values",
            "StandardScaler.transform() (fitted on train rows only)",
            f"clip standardised values to [-{SCALED_CLIP_LIMIT:g}, {SCALED_CLIP_LIMIT:g}]",
            "cast to float32",
            "save",
        ],
        "post_standardisation_clipping": True,
        "scaled_clip_limit": SCALED_CLIP_LIMIT,
        "scaled_clip_range": [-SCALED_CLIP_LIMIT, SCALED_CLIP_LIMIT],
        "scaled_clip_applied_to_splits": SPLIT_NAMES,
        "scaled_clip_affects_scaler_fit": False,
        "clipping_accounting_artifacts": ["scaled_clipping_summary.csv", "scaled_clipping_summary.json"],
        "raw_value_clipping": False,
        "imputation": False,
        "winsorisation": False,
        "log_transform": False,
        "output_dtypes": {"X": "float32", "y": "int64", "raw_indices": "int64"},
        "raw_csv_path": str(RAW_CSV),
        "raw_csv_size_bytes": provenance["csv_size_bytes"],
        "raw_csv_sha1": provenance["csv_sha1"],
        "feature_definition_sha1": provenance["feature_definition_sha1"],
        "package_zip_sha256": PACKAGE_ZIP_SHA256,
        "doi": DOI,
        "official_url": OFFICIAL_URL,
        "licence_text": LICENCE_TEXT,
        "script_sha256": script_sha256(),
        "git_commit": git_commit_hash(),
        "git_dirty": git_dirty(),
        "software_versions": software_versions(),
    }
    with open(CONFIGS_STAGE / "preprocessing_config.json", "w") as f:
        json.dump(config, f, indent=2)


def write_scaler(scaler: StandardScaler) -> None:
    joblib.dump(scaler, MODELS_STAGE / "standard_scaler.joblib")


# --------------------------------------------------------------------------- #
# Promotion and manifest
# --------------------------------------------------------------------------- #
def promote_staging() -> None:
    """Rename each staging directory to its final name (finals must not yet exist)."""
    pairs = [(PROC_STAGE, PROC_FINAL), (MODELS_STAGE, MODELS_FINAL),
             (CONFIGS_STAGE, CONFIGS_FINAL), (RESULTS_STAGE, RESULTS_FINAL)]
    for stage, final in pairs:
        final.parent.mkdir(parents=True, exist_ok=True)
        assert not final.exists(), f"Final path unexpectedly exists: {final}"
        os.replace(stage, final)
    print("Promoted all staging directories to final names.", flush=True)


def build_manifest() -> None:
    """Hash every final artefact (except the manifest itself) and write the manifest."""
    manifest_path = RESULTS_FINAL / "artifact_manifest_sha256.json"
    entries = []
    for root in [PROC_FINAL, MODELS_FINAL, CONFIGS_FINAL, RESULTS_FINAL]:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.resolve() != manifest_path.resolve():
                entries.append({
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_of_file(path),
                })
    with open(manifest_path, "w") as f:
        json.dump({"artifacts": entries, "count": len(entries)}, f, indent=2)
    print(f"Wrote manifest with {len(entries)} artefacts: {manifest_path}", flush=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    preflight_no_outputs()
    provenance = verify_provenance()
    make_staging_dirs()

    feature_hash, parent_label = first_pass()

    class_total, T = compute_targets(parent_label)
    groups = build_groups(feature_hash, parent_label)
    group_split, assigned = assign_groups(groups, T)
    row_split = map_row_split(groups, group_split, feature_hash.shape[0])

    split_info = validate_split(row_split, parent_label, class_total, T)
    n_train = split_info["sizes"][0]

    # Free large sort-order buffers before the scaling/writing passes.
    del feature_hash
    del groups["order"], groups["sorted_parent"]

    scaler = fit_scaler(row_split, n_train)
    clipping = write_arrays(row_split, parent_label, scaler, split_info)
    validate_outputs()

    # Artefacts (staging), then atomic promotion, then manifest over finals. The
    # `groups` dict still holds the per-group size/min/max arrays needed here.
    write_scaler(scaler)
    write_configs(class_total, T, split_info, provenance)
    write_result_tables(class_total, T, split_info, scaler, EXPECTED_RAW_ATTACK_COUNTS)
    write_clipping_summary(clipping)
    write_group_split_summary(groups, group_split, T, split_info)
    write_provenance_json(provenance)

    promote_staging()
    build_manifest()

    print("\n=== Dataset-2 preprocessing complete ===", flush=True)
    for s, name in enumerate(SPLIT_NAMES):
        print(f"  {name:10s} rows={split_info['sizes'][s]:>12,}  "
              f"fraction={split_info['proportions'][s]:.4f}", flush=True)
    print(f"  features={NUM_FEATURES}  classes={NUM_CLASSES}  total_rows={EXPECTED_ROWS:,}", flush=True)


if __name__ == "__main__":
    main()
