"""
Attack detection and false alarm rates for the HD-matched grids, from saved test
confusion matrices.

Macro-F1 penalises confusion between attack classes, so a model that flags an
attack but names the wrong type scores badly on it. Intrusion detection also cares
about the coarser question of whether anything non-Benign was flagged at all, which
is what this computes. The two rates are reported together because a detection rate
on its own rises trivially for a model that over-predicts attacks.

Detection rate is over true non-Benign records predicted as any non-Benign class.
False alarm rate is over true Benign records predicted as any non-Benign class.
Per-class recall is included so the write-up can say which class fails first.

Reads results/hd_grids_test_37f/confusion_matrices/confusion_raw_*.csv. Writes
results/hd_grids_test_37f/attack_detection_rates.csv. Retrains nothing, re-evaluates
nothing, and reads no checkpoint or test array. Refuses to run if the output exists.
"""

from pathlib import Path
import re

import pandas as pd

CM_DIR = Path("results/hd_grids_test_37f/confusion_matrices")
OUT_PATH = Path("results/hd_grids_test_37f/attack_detection_rates.csv")
BENIGN = "Benign"
NAME_RE = re.compile(r"^confusion_raw_(.+?)_fedavg37f_k5_seed(\d+)_(.+)\.csv$")
EXPECTED_RUNS = 30


def parse_name(path):
    m = NAME_RE.match(path.name)
    if m is None:
        raise RuntimeError(f"unexpected filename: {path.name}")
    return m.group(1), int(m.group(2)), m.group(3)


def load_matrix(path):
    cm = pd.read_csv(path, index_col=0)
    if list(cm.index) != list(cm.columns):
        raise RuntimeError(f"row and column classes differ in {path.name}")
    if cm.index[0] != BENIGN:
        raise RuntimeError(f"{BENIGN} is not the first class in {path.name}")
    return cm


def rates(cm):
    attacks = [c for c in cm.index if c != BENIGN]
    attack_rows = cm.loc[attacks]

    n_attack = int(attack_rows.values.sum())
    detected = int(attack_rows[attacks].values.sum())

    n_benign = int(cm.loc[BENIGN].sum())
    false_alarms = int(cm.loc[BENIGN, attacks].sum())

    row = {
        "n_attack_records": n_attack,
        "n_attack_detected": detected,
        "attack_detection_rate": detected / n_attack,
        "n_benign_records": n_benign,
        "n_benign_flagged_as_attack": false_alarms,
        "false_alarm_rate": false_alarms / n_benign,
    }
    for c in attacks:
        support = int(cm.loc[c].sum())
        # recall is undefined with no test records for the class
        row[f"recall_{c}"] = cm.loc[c, c] / support if support else float("nan")
    return row


def main():
    if OUT_PATH.exists():
        raise RuntimeError(f"{OUT_PATH} exists; not overwriting")
    if not CM_DIR.exists():
        raise RuntimeError(f"cannot find {CM_DIR}; run from the project root")

    paths = sorted(CM_DIR.glob("confusion_raw_*.csv"))
    if len(paths) != EXPECTED_RUNS:
        raise RuntimeError(f"found {len(paths)} raw matrices, expected {EXPECTED_RUNS}")

    rows = []
    for path in paths:
        partitioner, seed, condition = parse_name(path)
        cm = load_matrix(path)
        rows.append({"partitioner": partitioner, "partition_seed": seed,
                     "condition": condition, **rates(cm)})

    df = pd.DataFrame(rows).sort_values(["partitioner", "condition", "partition_seed"])
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(df)} runs)")

    summary = (df.groupby(["partitioner", "condition"])
                 [["attack_detection_rate", "false_alarm_rate"]]
                 .agg(["mean", "std"]).round(4))
    print(summary.to_string())


if __name__ == "__main__":
    main()
