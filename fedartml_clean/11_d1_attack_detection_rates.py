from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TEST_ROOT = ROOT / "fedartml_clean" / "final_test_100r"
CM_DIR = TEST_ROOT / "confusion_matrices"
OVERALL_PATH = TEST_ROOT / "test_results_overall.csv"
OUT_PATH = TEST_ROOT / "attack_detection_rates.csv"

SEEDS = [42, 43]
CONDITIONS = ["iid", "hd_0p25", "hd_0p5", "hd_0p75", "hd_0p9"]
BENIGN = "Benign"


def load_matrix(path):
    cm = pd.read_csv(path, index_col=0)

    if list(cm.index) != list(cm.columns):
        raise RuntimeError(f"row/column classes differ: {path}")

    if BENIGN not in cm.index:
        raise RuntimeError(f"{BENIGN} missing from {path}")

    return cm


def rates(cm):
    attacks = [c for c in cm.index if c != BENIGN]

    attack_rows = cm.loc[attacks]

    n_attack = int(attack_rows.to_numpy().sum())
    detected = int(attack_rows[attacks].to_numpy().sum())

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

    for attack in attacks:
        support = int(cm.loc[attack].sum())
        row[f"recall_{attack}"] = (
            float(cm.loc[attack, attack] / support)
            if support > 0 else float("nan")
        )

    return row


def main():
    if OUT_PATH.exists():
        raise RuntimeError(f"refusing to overwrite: {OUT_PATH}")

    if not OVERALL_PATH.exists():
        raise RuntimeError(
            "final D1 test results do not exist yet; run clean final test first"
        )

    overall = pd.read_csv(OVERALL_PATH)

    if len(overall) != 10:
        raise RuntimeError(
            f"expected 10 final-test models, found {len(overall)}"
        )

    rows = []

    for seed in SEEDS:
        for condition in CONDITIONS:
            cm_path = (
                CM_DIR
                / f"confusion_raw_seed{seed}_{condition}.csv"
            )

            if not cm_path.exists():
                raise RuntimeError(f"missing: {cm_path}")

            match = overall[
                (overall["partition_seed"] == seed)
                & (overall["condition"] == condition)
            ]

            if len(match) != 1:
                raise RuntimeError(
                    f"expected one final-test row for seed={seed}, "
                    f"condition={condition}"
                )

            meta = match.iloc[0]

            rows.append({
                "partition_seed": seed,
                "training_seed": int(meta["training_seed"]),
                "condition": condition,
                "target_hd": meta["target_hd"],
                "achieved_hd": meta["achieved_hd"],
                "best_round": int(meta["best_round"]),
                "test_macro_f1": meta["macro_f1"],
                **rates(load_matrix(cm_path)),
            })

    df = pd.DataFrame(rows).sort_values(
        ["condition", "partition_seed"]
    )

    df.to_csv(OUT_PATH, index=False)

    print(f"WROTE: {OUT_PATH}")
    print(f"runs: {len(df)}")


if __name__ == "__main__":
    main()
