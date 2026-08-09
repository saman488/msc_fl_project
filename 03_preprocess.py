"""
Prepare NF-UNSW-NB15-v2 for MULTI-CLASS intrusion-detection experiments.

The target is the attack category (Benign + nine attack families) label-encoded
to integers 0-9, so specific rare attacks (e.g. Worms, Shellcode) can be tracked
through the federated experiments. The script creates a fixed train/validation/
test split, standardises numeric features using training-set statistics only, and
stores preprocessing artefacts needed to reproduce the experiments.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RAW_FILE = Path("data/raw/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/preprocessing")
CONFIGS_DIR = Path("configs")

# The multi-class target is derived from the Attack category column.
CATEGORY_COLUMN = "Attack"
BENIGN_CATEGORY = "Benign"

EXCLUDED_COLUMNS = [
    "Label",          # Binary target; superseded by the multi-class Attack label.
    "Attack",         # Source of the multi-class target; excluded from features.
    "IPV4_SRC_ADDR",  # Host identifier; excluded from the first baseline.
    "IPV4_DST_ADDR",  # Host identifier; excluded from the first baseline.
]

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15


def build_feature_columns(columns: list[str]) -> list[str]:
    """Return model-input columns after removing target and identifier fields."""
    missing_columns = [column for column in EXCLUDED_COLUMNS if column not in columns]
    if missing_columns:
        raise ValueError(f"Expected columns not found: {missing_columns}")

    return [column for column in columns if column not in EXCLUDED_COLUMNS]


def build_label_mapping(categories: list[str]) -> dict[str, int]:
    """Deterministic category -> integer mapping: Benign=0, attacks alphabetical 1..9."""
    if BENIGN_CATEGORY not in categories:
        raise ValueError(f"Benign category {BENIGN_CATEGORY!r} not present in data.")

    attack_categories = sorted(c for c in categories if c != BENIGN_CATEGORY)
    mapping = {BENIGN_CATEGORY: 0}
    for index, category in enumerate(attack_categories, start=1):
        mapping[category] = index
    return mapping


def validate_target(y: pd.Series, mapping: dict[str, int]) -> None:
    """Check that every encoded label is within the expected 0..K-1 range."""
    expected = set(mapping.values())
    observed = set(y.unique().tolist())
    if not observed.issubset(expected):
        raise ValueError(f"Unexpected encoded labels: {sorted(observed - expected)}")
    if observed != expected:
        raise ValueError(
            f"Missing classes after encoding: {sorted(expected - observed)}"
        )


def class_distribution(name: str, y: pd.Series, mapping: dict[str, int]) -> list[dict]:
    """Per-class counts for one dataset split (long form)."""
    inverse = {index: category for category, index in mapping.items()}
    counts = y.value_counts().to_dict()
    total = int(len(y))
    rows = []
    for class_id in sorted(mapping.values()):
        count = int(counts.get(class_id, 0))
        rows.append(
            {
                "split": name,
                "class_id": class_id,
                "class_name": inverse[class_id],
                "count": count,
                "ratio": count / total,
            }
        )
    return rows


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    all_columns = pd.read_csv(RAW_FILE, nrows=0).columns.tolist()
    feature_columns = build_feature_columns(all_columns)

    use_columns = feature_columns + [CATEGORY_COLUMN]
    dtype_map = {column: np.float32 for column in feature_columns}
    dtype_map[CATEGORY_COLUMN] = "string"

    df = pd.read_csv(RAW_FILE, usecols=use_columns, dtype=dtype_map)

    # Label-encode the attack category into a multi-class integer target.
    categories = df[CATEGORY_COLUMN].unique().tolist()
    label_mapping = build_label_mapping(categories)

    X = df[feature_columns]
    y = df[CATEGORY_COLUMN].map(label_mapping).astype(np.int64)

    validate_target(y, label_mapping)

    # Stratification preserves the per-class ratio (incl. rare attacks) in each split.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # The validation split is taken from the remaining train-validation subset.
    validation_fraction = VALIDATION_SIZE / (1.0 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    # Fit scaling parameters on training data only to avoid evaluation leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    np.save(PROCESSED_DIR / "X_train.npy", X_train_scaled)
    np.save(PROCESSED_DIR / "X_val.npy", X_val_scaled)
    np.save(PROCESSED_DIR / "X_test.npy", X_test_scaled)

    np.save(PROCESSED_DIR / "y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save(PROCESSED_DIR / "y_val.npy", y_val.to_numpy(dtype=np.int64))
    np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy(dtype=np.int64))

    joblib.dump(scaler, MODELS_DIR / "standard_scaler.joblib")

    # Per-class distribution table across splits (includes rare attacks).
    distribution_rows = []
    for split_name, split_y in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        distribution_rows.extend(class_distribution(split_name, split_y, label_mapping))
    pd.DataFrame(distribution_rows).to_csv(
        RESULTS_DIR / "class_distribution.csv", index=False
    )

    split_table = pd.DataFrame(
        [
            {"split": "train", "rows": int(len(y_train))},
            {"split": "validation", "rows": int(len(y_val))},
            {"split": "test", "rows": int(len(y_test))},
        ]
    )
    split_table.to_csv(RESULTS_DIR / "split_summary.csv", index=False)

    pd.Series(feature_columns, name="feature").to_csv(
        RESULTS_DIR / "feature_columns.csv",
        index=False,
    )

    scaler_table = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_train": scaler.mean_,
            "scale_train": scaler.scale_,
        }
    )
    scaler_table.to_csv(RESULTS_DIR / "scaler_summary.csv", index=False)

    # Persist the label mapping so downstream scripts encode/decode consistently.
    with open(CONFIGS_DIR / "label_mapping.json", "w") as file:
        json.dump(label_mapping, file, indent=2)
    pd.DataFrame(
        [{"class_id": v, "class_name": k} for k, v in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    ).to_csv(RESULTS_DIR / "label_mapping.csv", index=False)

    # Store preprocessing choices that affect reproducibility and interpretation.
    config = {
        "raw_file": str(RAW_FILE),
        "task": "multiclass_intrusion_detection",
        "target": "attack_category",
        "num_classes": len(label_mapping),
        "label_mapping": label_mapping,
        "excluded_columns": EXCLUDED_COLUMNS,
        "feature_count": len(feature_columns),
        "random_state": RANDOM_STATE,
        "train_size": 0.70,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "split_method": "stratified",
        "scaler": "StandardScaler",
        "scaler_fit_split": "train",
    }

    with open(CONFIGS_DIR / "preprocessing_config.json", "w") as file:
        json.dump(config, file, indent=2)

    print("Preprocessing complete (multi-class).")
    print("Label mapping:", label_mapping)
    print(split_table)
    print(f"Feature count: {len(feature_columns)}  |  classes: {len(label_mapping)}")


if __name__ == "__main__":
    main()
