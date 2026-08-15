"""
Read-only join of verified controlled-partition HD measurements with the existing
FedAvg-SGD (lr=0.1) and FedProx-SGD (mu in {0.001,0.01,0.1,0.5,1.0}) VALIDATION
results, keyed by (seed, partition).

No training. No test-data access. No correlations, curves, or thresholds here.
Validation metrics are taken from the already-saved per-run histories (per round)
and best-checkpoint confusion matrices; nothing is recomputed from test data.
Writes exactly one new CSV.
"""

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

HD_FORMS = Path("results/fl_partitions/controlled_hd_forms_verified.csv")
OUT_CSV = Path("results/fl_partitions/controlled_hd_performance_join_verified.csv")

SEEDS = [42, 43, 44]
PARTITIONS = ["alpha_0.01", "alpha_0.05", "alpha_0.1", "alpha_0.5", "alpha_1.0", "alpha_5.0", "iid"]
CONFIG_MUS = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]   # 0.0 = FedAvg baseline
HD_COLS = ["hd_pairwise_mean", "hd_pairwise_rms", "hd_pairwise_max",
           "hd_client_to_pool_mean", "hd_client_to_pool_max"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    m16 = load_module("m16", "16_analyse_fedprox_mu_selection.py")

    # No-test-array assertion for this script (and 16 asserts for itself).
    tok_x, tok_y = "X_" + "test", "y_" + "test"
    assert tok_x not in Path(__file__).read_text() and tok_y not in Path(__file__).read_text()

    hd = pd.read_csv(HD_FORMS)[["seed", "partition"] + HD_COLS].set_index(["seed", "partition"])
    assert len(hd) == 21, f"expected 21 HD rows, got {len(hd)}"
    assert not hd[HD_COLS].isna().any().any(), "missing HD values in HD-forms table"

    source_files = set()
    rows = []
    for mu in CONFIG_MUS:
        d, tag = m16.dir_tag(mu)
        algo = "FedAvg-SGD" if mu == 0.0 else "FedProx-SGD"
        for seed in SEEDS:
            for part in PARTITIONS:
                hist, cfg, conf = m16.load_run(mu, seed, part)
                source_files.add(str(m16.path_of(d, tag, seed, part, "history", "csv")))
                source_files.add(str(m16.path_of(d, tag, seed, part, "confusion", "csv")))
                perf = m16.metrics_row(mu, seed, part, hist, conf)  # validation metrics only
                row = {
                    "algorithm": algo, "mu": mu, "learning_rate": 0.1,
                    "seed": seed, "partition": part,
                }
                for c in HD_COLS:
                    row[c] = float(hd.loc[(seed, part), c])
                # rename to the requested performance labels; keep the rest as-is
                row["best_val_macro_f1"] = perf["best_macro_f1"]
                row["final_val_macro_f1"] = perf["final_macro_f1"]
                row["val_macro_recall"] = perf["macro_recall"]
                row["val_macro_pr_auc"] = perf["macro_pr_auc"]
                row["best_round"] = perf["best_round"]
                row["num_predicted_classes"] = perf["num_predicted_classes"]
                for k in ["mean_macro_f1_41_50", "balanced_accuracy", "worst_class_f1",
                          "predicted_benign_proportion", "one_class_indicator",
                          "two_class_restricted_indicator", "mean_task_loss",
                          "mean_proximal_penalty", "mean_total_loss"]:
                    row[k] = perf[k]
                for cc in range(m16.NUM_CLASSES):
                    for m in ("precision", "recall", "f1", "pr_auc", "support", "predicted_count"):
                        row[f"{m}_c{cc}"] = perf[f"{m}_c{cc}"]
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # ---------- verification ----------
    counts = df.groupby(["algorithm", "mu"]).size()
    n_fedavg = int(df[df.algorithm == "FedAvg-SGD"].shape[0])
    per_mu = {mu: int(df[(df.algorithm == "FedProx-SGD") & (df.mu == mu)].shape[0]) for mu in CONFIG_MUS[1:]}
    dup = int(df.duplicated(subset=["mu", "seed", "partition"]).sum())
    missing_hd = int(df[HD_COLS].isna().sum().sum())
    # identical HD across the 6 configs per (seed, partition)
    hd_consistent = bool((df.groupby(["seed", "partition"])[HD_COLS].nunique() == 1).all().all())
    missing_counts = df.isna().sum()
    missing_nonzero = missing_counts[missing_counts > 0]

    print("=== SOURCE FILES USED ===")
    print(f"  HD-forms table: {HD_FORMS}")
    print(f"  per-run validation artefacts (history + confusion), {len(source_files)} files across:")
    print("    - results/fl_sgd_config_selection/ (tag sgd_lr_0p1)  [FedAvg-SGD lr=0.1]")
    print("    - results/fl_fedprox_sgd/ (tags fedprox_mu*_lr0p1_r50) [FedProx-SGD]")

    print("\n=== ROW COUNTS BY ALGORITHM AND MU ===")
    print(counts.to_string())
    print(f"  total rows: {len(df)}")

    print("\n=== JOINED COLUMN NAMES ===")
    print(f"  ({len(df.columns)} columns)")
    print("  " + ", ".join(df.columns[:26]))
    print("  ... per-class block: precision_c0..9, recall_c0..9, f1_c0..9, pr_auc_c0..9, support_c0..9, predicted_count_c0..9")

    print("\n=== MISSING-VALUE COUNTS ===")
    if len(missing_nonzero) == 0:
        print("  no missing values in any column")
    else:
        print(missing_nonzero.to_string())
        print("  (mean_task/proximal/total_loss are NaN for FedAvg-SGD by design — those columns exist only in FedProx histories)")

    print("\n=== VERIFICATION ===")
    print(f"  exactly 21 FedAvg rows:                    {n_fedavg == 21}  ({n_fedavg})")
    for mu, n in per_mu.items():
        print(f"  exactly 21 FedProx mu={mu} rows:            {n == 21}  ({n})")
    print(f"  exactly 126 total rows:                    {len(df) == 126}  ({len(df)})")
    print(f"  duplicate config-seed-partition keys:      {dup}")
    print(f"  missing HD values:                         {missing_hd}")
    print(f"  identical HD across all 6 configs/(s,p):   {hd_consistent}")
    print("  only VALIDATION results used (per-round histories + best-checkpoint confusion); no test data")

    print("\n=== COMPLETE 21-ROW FedAvg-SGD SUBSET ===")
    fed = df[df.algorithm == "FedAvg-SGD"][[
        "seed", "partition", "hd_pairwise_mean", "hd_pairwise_rms", "hd_pairwise_max",
        "hd_client_to_pool_mean", "hd_client_to_pool_max",
        "best_val_macro_f1", "final_val_macro_f1", "val_macro_recall",
        "val_macro_pr_auc", "best_round", "num_predicted_classes"]].copy()
    fed["partition"] = pd.Categorical(fed["partition"], categories=PARTITIONS, ordered=True)
    fed = fed.sort_values(["seed", "partition"]).reset_index(drop=True)
    for c in fed.columns:
        if fed[c].dtype == float:
            fed[c] = fed[c].round(6)
    print(fed.to_string(index=False))

    print("\nNO training, NO test-set access, NO correlations, NO threshold analysis performed.")
    print(f"Written to {OUT_CSV}")


if __name__ == "__main__":
    main()
