"""
Prepare NF-UNSW-NB15-v2 for binary intrusion-detection experiments.

The script creates a fixed train/validation/test split, standardises numeric
features using training-set statistics only, and stores preprocessing artefacts
needed to reproduce the centralised and federated experiments.
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

TARGET_COLUMN = "Label"

EXCLUDED_COLUMNS = [
    "Label",          # Binary target; retained separately as y.
    "Attack",         # Target-derived class name; excluded to prevent leakage.
    "IPV4_SRC_ADDR",  # Host identifier; excluded from the first baseline.
    "IPV4_DST_ADDR",  # Host identifier; excluded from the first baseline.
]

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
EXPECTED_LABELS = {0, 1}


def build_feature_columns(columns: list[str]) -> list[str]:
    """Return model-input columns after removing target and identifier fields."""
    missing_columns = [column for column in EXCLUDED_COLUMNS if column not in columns]
    if missing_columns:
        raise ValueError(f"Expected columns not found: {missing_columns}")

    return [column for column in columns if column not in EXCLUDED_COLUMNS]


def validate_target(y: pd.Series) -> None:
    """Check that the binary target has the expected label encoding."""
    observed_labels = set(y.unique().tolist())
    if observed_labels != EXPECTED_LABELS:
        raise ValueError(
            f"Unexpected labels in {TARGET_COLUMN}: {sorted(observed_labels)}"
        )


def summarise_split(name: str, y: pd.Series) -> dict:
    """Summarise class balance for one dataset split."""
    benign_count = int((y == 0).sum())
    attack_count = int((y == 1).sum())
    row_count = int(len(y))

    return {
        "split": name,
        "rows": row_count,
        "benign_count": benign_count,
        "attack_count": attack_count,
        "attack_ratio": attack_count / row_count,
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    all_columns = pd.read_csv(RAW_FILE, nrows=0).columns.tolist()
    feature_columns = build_feature_columns(all_columns)

    use_columns = feature_columns + [TARGET_COLUMN]
    dtype_map = {column: np.float32 for column in feature_columns}
    dtype_map[TARGET_COLUMN] = np.int64

    df = pd.read_csv(RAW_FILE, usecols=use_columns, dtype=dtype_map)

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    validate_target(y)

    # Stratification preserves the global benign/attack ratio in each split.
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

    split_table = pd.DataFrame(
        [
            summarise_split("train", y_train),
            summarise_split("validation", y_val),
            summarise_split("test", y_test),
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

    # Store preprocessing choices that affect reproducibility and interpretation.
    config = {
        "raw_file": str(RAW_FILE),
        "target_column": TARGET_COLUMN,
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

    print("Preprocessing complete.")
    print(split_table)
    print(f"Feature count: {len(feature_columns)}")


if __name__ == "__main__":
    main()