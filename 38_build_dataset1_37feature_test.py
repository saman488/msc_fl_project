"""
Add the held-out test arrays to the Dataset-1 37-feature branch.

30_build_dataset1_37feature_branch.py built the branch from the train and
validation splits only and left the test split untouched on purpose, so that no
test data was read while the federated and centralised models were being
developed. Final evaluation now needs the test split in the same representation,
so this script applies the identical column slice to it.

The four removed columns are the same verified positions in the raw
NF-UNSW-NB15-v2 header order:
  0  L4_SRC_PORT
  1  L4_DST_PORT
  14 MIN_TTL
  15 MAX_TTL
DNS_TTL_ANSWER is retained. The result is 37 features.

No scaling and no new preprocessing: the arrays in data/processed were already
transformed in 03_preprocess.py with a StandardScaler fit on the training split
only. StandardScaler is per-column, so dropping columns leaves every retained
value bit-identical to what it already was; there is nothing to re-fit and no
train/test leakage is introduced by this script. Labels and row order are
preserved exactly.

Writes X_test.npy and y_test.npy under data/processed_37f, and adds the test
split's row count and source paths to the existing feature_manifest.json. The
manifest's feature fields are read and verified but never rewritten, and the
train/validation arrays are not touched.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

SRC_DIR = Path("data/processed")
OUT_DIR = Path("data/processed_37f")
FEATURE_ORDER_CSV = Path("results/preprocessing/feature_columns.csv")
MANIFEST_PATH = OUT_DIR / "feature_manifest.json"

ORIGINAL_FEATURE_COUNT = 41
CORRECTED_FEATURE_COUNT = 37
REMOVE_INDICES = [0, 1, 14, 15]
EXPECTED_REMOVED_NAMES = ["L4_SRC_PORT", "L4_DST_PORT", "MIN_TTL", "MAX_TTL"]


def main() -> None:
    assert OUT_DIR.resolve() != SRC_DIR.resolve(), "output must not be data/processed"

    feature_order = pd.read_csv(FEATURE_ORDER_CSV)["feature"].tolist()
    assert len(feature_order) == ORIGINAL_FEATURE_COUNT, (
        f"feature order has {len(feature_order)} names, expected {ORIGINAL_FEATURE_COUNT}"
    )

    removed_names = [feature_order[i] for i in REMOVE_INDICES]
    assert removed_names == EXPECTED_REMOVED_NAMES, (
        f"removed positions {REMOVE_INDICES} hold {removed_names}, expected {EXPECTED_REMOVED_NAMES}"
    )
    assert "DNS_TTL_ANSWER" not in removed_names, "DNS_TTL_ANSWER must be kept"

    remove_set = set(REMOVE_INDICES)
    keep_indices = [i for i in range(ORIGINAL_FEATURE_COUNT) if i not in remove_set]
    retained_names = [feature_order[i] for i in keep_indices]
    assert len(keep_indices) == CORRECTED_FEATURE_COUNT, (
        f"kept {len(keep_indices)} features, expected {CORRECTED_FEATURE_COUNT}"
    )
    assert "DNS_TTL_ANSWER" in retained_names, "DNS_TTL_ANSWER missing from retained features"

    # The train/val arrays were sliced with the indices recorded in the manifest.
    # Recomputing them here and disagreeing would mean the test split is being cut
    # differently from the split the models were trained on, so treat any mismatch
    # as fatal rather than trusting the freshly computed indices.
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    assert manifest["original_feature_count"] == ORIGINAL_FEATURE_COUNT, (
        f"manifest original_feature_count is {manifest['original_feature_count']}, "
        f"expected {ORIGINAL_FEATURE_COUNT}"
    )
    assert manifest["corrected_feature_count"] == CORRECTED_FEATURE_COUNT, (
        f"manifest corrected_feature_count is {manifest['corrected_feature_count']}, "
        f"expected {CORRECTED_FEATURE_COUNT}"
    )
    assert manifest["original_feature_order"] == feature_order, (
        "manifest original_feature_order disagrees with " f"{FEATURE_ORDER_CSV}"
    )
    assert manifest["retained_original_indices"] == keep_indices, (
        f"manifest retained_original_indices {manifest['retained_original_indices']} "
        f"disagrees with recomputed {keep_indices}"
    )
    assert manifest["retained_features"] == retained_names, (
        "manifest retained_features disagrees with recomputed retained names"
    )
    manifest_removed = [[d["original_index"], d["name"]] for d in manifest["removed_features"]]
    assert manifest_removed == [[i, n] for i, n in zip(REMOVE_INDICES, EXPECTED_REMOVED_NAMES)], (
        f"manifest removed_features {manifest_removed} disagrees with "
        f"{list(zip(REMOVE_INDICES, EXPECTED_REMOVED_NAMES))}"
    )

    intended = ["X_test.npy", "y_test.npy"]
    existing = [name for name in intended if (OUT_DIR / name).exists()]
    if existing:
        listing = "\n  ".join(str(OUT_DIR / name) for name in existing)
        raise RuntimeError(f"Refusing to write; test outputs already present:\n  {listing}")

    x_test_src = np.load(SRC_DIR / "X_test.npy", mmap_mode="r")
    y_test = np.load(SRC_DIR / "y_test.npy")

    assert x_test_src.dtype == np.float32, f"X_test dtype is {x_test_src.dtype}, expected float32"
    assert x_test_src.ndim == 2, f"X_test has {x_test_src.ndim} dimensions, expected 2"
    assert x_test_src.shape[1] == ORIGINAL_FEATURE_COUNT, (
        f"X_test has {x_test_src.shape[1]} columns, expected {ORIGINAL_FEATURE_COUNT}"
    )
    assert x_test_src.shape[0] == y_test.shape[0], "source X_test / y_test row count mismatch"

    x_test_37 = np.ascontiguousarray(x_test_src[:, keep_indices], dtype=np.float32)

    assert x_test_37.shape[1] == CORRECTED_FEATURE_COUNT, "X_test output column count wrong"
    assert x_test_37.shape[0] == x_test_src.shape[0], "X_test output row count changed"
    assert x_test_37.shape[0] == y_test.shape[0], "X_test / y_test row count mismatch"
    assert np.isfinite(x_test_37).all(), "X_test output contains non-finite values"

    # Fancy indexing returns rows in their original order, but check it explicitly:
    # the first and last retained columns must still match their source columns
    # value for value, which fails immediately if any row permutation crept in.
    assert np.array_equal(x_test_37[:, 0], x_test_src[:, keep_indices[0]]), (
        "first retained column does not match its source column; row order changed"
    )
    assert np.array_equal(x_test_37[:, -1], x_test_src[:, keep_indices[-1]]), (
        "last retained column does not match its source column; row order changed"
    )

    np.save(OUT_DIR / "X_test.npy", x_test_37)
    np.save(OUT_DIR / "y_test.npy", y_test)   # preserved exactly

    # Record the test split in the manifest 30 wrote, so the manifest describes
    # every file in the branch rather than only the train/val pair. The feature
    # fields are left exactly as 30 wrote them: they describe the column slice,
    # which is identical for all three splits and must not be rewritten here.
    manifest["row_counts"]["test"] = int(x_test_37.shape[0])
    manifest["source_paths"]["X_test"] = str(SRC_DIR / "X_test.npy")
    manifest["source_paths"]["y_test"] = str(SRC_DIR / "y_test.npy")
    manifest["test_split_note"] = (
        "X_test.npy and y_test.npy were added by "
        "38_build_dataset1_37feature_test.py after model development and "
        "selection were complete. 30_build_dataset1_37feature_branch.py "
        "deliberately excluded the test split. The same retained_original_indices "
        "were applied, verified against this manifest before slicing."
    )
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Removed 4 features: {removed_names}")
    print(f"Features: {ORIGINAL_FEATURE_COUNT} -> {CORRECTED_FEATURE_COUNT} (DNS_TTL_ANSWER retained)")
    print(f"X_test {x_test_37.shape} y_test {y_test.shape} dtype {x_test_37.dtype}/{y_test.dtype}")
    print(f"Wrote test arrays under {OUT_DIR}")


if __name__ == "__main__":
    main()
