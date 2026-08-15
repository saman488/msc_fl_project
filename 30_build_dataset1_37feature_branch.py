"""
Build an isolated corrected Dataset-1 37-feature branch from the existing
standardized arrays, without touching data/processed.

Takes the already-standardized 41-feature train/validation arrays and removes four
columns by their verified positions in the raw NF-UNSW-NB15-v2 header order:
  0  L4_SRC_PORT
  1  L4_DST_PORT
  14 MIN_TTL
  15 MAX_TTL
DNS_TTL_ANSWER is retained. The result is 37 features. Column slicing preserves
the existing standardized values for the retained features (StandardScaler was
originally fit on the training split only); no re-scaling or new preprocessing is
introduced. Labels and row order are preserved exactly. The test arrays are never
read or written.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

SRC_DIR = Path("data/processed")
OUT_DIR = Path("data/processed_37f")
FEATURE_ORDER_CSV = Path("results/preprocessing/feature_columns.csv")

ORIGINAL_FEATURE_COUNT = 41
CORRECTED_FEATURE_COUNT = 37
REMOVE_INDICES = [0, 1, 14, 15]
EXPECTED_REMOVED_NAMES = ["L4_SRC_PORT", "L4_DST_PORT", "MIN_TTL", "MAX_TTL"]


def load_feature_order() -> list[str]:
    """Verified 41-feature order (raw header minus excluded columns)."""
    order = pd.read_csv(FEATURE_ORDER_CSV)["feature"].tolist()
    assert len(order) == ORIGINAL_FEATURE_COUNT, f"feature order has {len(order)} names, expected {ORIGINAL_FEATURE_COUNT}"
    return order


def main() -> None:
    assert OUT_DIR.resolve() != SRC_DIR.resolve(), "output must not be data/processed"

    feature_order = load_feature_order()

    # Confirm the four positions to remove hold the expected names.
    removed_names = [feature_order[i] for i in REMOVE_INDICES]
    assert removed_names == EXPECTED_REMOVED_NAMES, (
        f"removed positions {REMOVE_INDICES} hold {removed_names}, expected {EXPECTED_REMOVED_NAMES}"
    )
    assert "DNS_TTL_ANSWER" not in removed_names, "DNS_TTL_ANSWER must be kept"

    remove_set = set(REMOVE_INDICES)
    keep_indices = [i for i in range(ORIGINAL_FEATURE_COUNT) if i not in remove_set]
    retained_names = [feature_order[i] for i in keep_indices]
    assert len(keep_indices) == CORRECTED_FEATURE_COUNT, f"kept {len(keep_indices)} features, expected {CORRECTED_FEATURE_COUNT}"
    assert "DNS_TTL_ANSWER" in retained_names, "DNS_TTL_ANSWER missing from retained features"

    # Refuse to overwrite an existing branch: fail if any intended output is present.
    intended = ["X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy", "feature_manifest.json"]
    existing = [name for name in intended if (OUT_DIR / name).exists()]
    if existing:
        listing = "\n  ".join(str(OUT_DIR / name) for name in existing)
        raise RuntimeError(f"Refusing to write; branch outputs already present:\n  {listing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Memory-mapped source X arrays; labels loaded normally (preserved exactly).
    x_train_src = np.load(SRC_DIR / "X_train.npy", mmap_mode="r")
    x_val_src = np.load(SRC_DIR / "X_val.npy", mmap_mode="r")
    y_train = np.load(SRC_DIR / "y_train.npy")
    y_val = np.load(SRC_DIR / "y_val.npy")

    assert x_train_src.dtype == np.float32, f"X_train dtype is {x_train_src.dtype}, expected float32"
    assert x_val_src.dtype == np.float32, f"X_val dtype is {x_val_src.dtype}, expected float32"
    assert x_train_src.shape[1] == ORIGINAL_FEATURE_COUNT, f"X_train has {x_train_src.shape[1]} columns, expected {ORIGINAL_FEATURE_COUNT}"
    assert x_val_src.shape[1] == ORIGINAL_FEATURE_COUNT, f"X_val has {x_val_src.shape[1]} columns, expected {ORIGINAL_FEATURE_COUNT}"

    # Column slice preserves row order and standardized values of retained features.
    x_train_37 = np.ascontiguousarray(x_train_src[:, keep_indices], dtype=np.float32)
    x_val_37 = np.ascontiguousarray(x_val_src[:, keep_indices], dtype=np.float32)

    assert x_train_37.shape[1] == CORRECTED_FEATURE_COUNT, "X_train output column count wrong"
    assert x_val_37.shape[1] == CORRECTED_FEATURE_COUNT, "X_val output column count wrong"
    assert x_train_37.shape[0] == y_train.shape[0], "X_train / y_train row count mismatch"
    assert x_val_37.shape[0] == y_val.shape[0], "X_val / y_val row count mismatch"
    assert np.isfinite(x_train_37).all(), "X_train output contains non-finite values"
    assert np.isfinite(x_val_37).all(), "X_val output contains non-finite values"

    np.save(OUT_DIR / "X_train.npy", x_train_37)
    np.save(OUT_DIR / "X_val.npy", x_val_37)
    np.save(OUT_DIR / "y_train.npy", y_train)   # preserved exactly
    np.save(OUT_DIR / "y_val.npy", y_val)       # preserved exactly

    manifest = {
        "original_feature_count": ORIGINAL_FEATURE_COUNT,
        "corrected_feature_count": CORRECTED_FEATURE_COUNT,
        "removed_features": [{"original_index": i, "name": feature_order[i]} for i in REMOVE_INDICES],
        "retained_features": retained_names,
        "retained_original_indices": keep_indices,
        "original_feature_order": feature_order,
        "source_paths": {
            "X_train": str(SRC_DIR / "X_train.npy"),
            "X_val": str(SRC_DIR / "X_val.npy"),
            "y_train": str(SRC_DIR / "y_train.npy"),
            "y_val": str(SRC_DIR / "y_val.npy"),
            "feature_order": str(FEATURE_ORDER_CSV),
        },
        "output_dir": str(OUT_DIR),
        "output_dtype": "float32",
        "row_counts": {"train": int(x_train_37.shape[0]), "val": int(x_val_37.shape[0])},
        "scaler_note": (
            "StandardScaler was originally fit on the training split only in "
            "03_preprocess.py. This branch does not re-scale: it slices out four "
            "columns, so the retained features keep their existing standardized "
            "(train-fit) values with row order and labels unchanged."
        ),
    }
    with open(OUT_DIR / "feature_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Removed 4 features: {removed_names}")
    print(f"Features: {ORIGINAL_FEATURE_COUNT} -> {CORRECTED_FEATURE_COUNT} (DNS_TTL_ANSWER retained)")
    print(f"X_train {x_train_37.shape} y_train {y_train.shape} | X_val {x_val_37.shape} y_val {y_val.shape}")
    print(f"Wrote branch under {OUT_DIR}")


if __name__ == "__main__":
    main()
