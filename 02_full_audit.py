import numpy as np
import pandas as pd

FILENAME = "data/raw/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv"

DROP_COLUMNS = [
    "Label",
    "Attack",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
]

total_rows = 0
missing_counts = None
inf_counts = None

label_counts = pd.Series(dtype="int64")
attack_counts = pd.Series(dtype="int64")

chunksize = 200_000

for chunk in pd.read_csv(FILENAME, chunksize=chunksize):
    total_rows += len(chunk)

    # Missing values
    chunk_missing = chunk.isna().sum()
    if missing_counts is None:
        missing_counts = chunk_missing
    else:
        missing_counts = missing_counts.add(chunk_missing, fill_value=0)

    # Label distribution
    label_counts = label_counts.add(chunk["Label"].value_counts(), fill_value=0)

    # Attack-category distribution
    attack_counts = attack_counts.add(chunk["Attack"].value_counts(), fill_value=0)

    # Infinite values in numeric columns
    numeric_chunk = chunk.select_dtypes(include="number")
    chunk_inf = np.isinf(numeric_chunk).sum()

    if inf_counts is None:
        inf_counts = chunk_inf
    else:
        inf_counts = inf_counts.add(chunk_inf, fill_value=0)

    print(f"Processed {total_rows:,} rows", end="\r")

print("\n\n=== FULL DATASET AUDIT ===")
print(f"Total rows: {total_rows:,}")

print("\nLabel counts:")
print(label_counts.astype(int).sort_index())

print("\nAttack counts:")
print(attack_counts.astype(int).sort_values(ascending=False))

print("\nMissing values:")
missing_nonzero = missing_counts[missing_counts > 0]
if len(missing_nonzero) == 0:
    print("No missing values found.")
else:
    print(missing_nonzero.astype(int))

print("\nInfinite values in numeric columns:")
inf_nonzero = inf_counts[inf_counts > 0]
if len(inf_nonzero) == 0:
    print("No infinite values found.")
else:
    print(inf_nonzero.astype(int))

all_columns = pd.read_csv(FILENAME, nrows=0).columns.tolist()
feature_columns = [col for col in all_columns if col not in DROP_COLUMNS]

print("\nDropped columns:")
print(DROP_COLUMNS)

print("\nFinal feature columns:")
print(feature_columns)

print(f"\nFinal number of input features: {len(feature_columns)}")