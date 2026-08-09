"""
Analysis & verification of the controlled non-IID FedAvg pilot (read-only).

Consumes ONLY existing artefacts (final_summary, per-run history / confusion /
report CSVs, controlled_metrics, and the controlled client index files). It does
not train, does not touch the test set, and does not modify any model / loss /
partition / checkpoint.

Verifies six integrity conditions, then writes six analysis tables and prints
four concise summaries. All skew-vs-performance associations are labelled
exploratory (pilot, n=21).
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr


RESULTS_DIR = Path("results/fl_noniid_controlled")
PART_METRICS = Path("results/fl_partitions/controlled_metrics.csv")
PART_ROOT = Path("data/fl_clients/controlled_partitions")
PROCESSED_DIR = Path("data/processed")
OUT_DIR = RESULTS_DIR / "analysis"

NUM_CLASSES = 10
NUM_CLIENTS = 5
VAL_SIZE = 358542
SEEDS = [42, 43, 44]
PARTITIONS = ["alpha_0.01", "alpha_0.05", "alpha_0.1", "alpha_0.5", "alpha_1.0", "alpha_5.0", "iid"]
HIGHLIGHTED = {"Analysis": 1, "Backdoor": 2, "Exploits": 4, "Shellcode": 8, "Worms": 9}
# A healthy model here already predicts ~95% Benign (val is ~96% Benign), so a
# high Benign fraction alone is NOT collapse. Collapse = the model degenerates to
# predicting essentially only Benign: >=99% Benign predictions OR <=2 distinct
# classes ever predicted.
BENIGN_FRACTION_COLLAPSE = 0.99   # 'almost Benign-only' threshold on predictions


def load_class_names() -> list[str]:
    with open(Path("configs/label_mapping.json")) as f:
        name_to_id = json.load(f)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    return [id_to_name[c] for c in range(len(id_to_name))]


def parse_alpha(partition: str) -> float:
    m = re.match(r"alpha_([0-9.]+)", partition)
    return float(m.group(1)) if m else np.nan


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2.0))


def normalise(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts.astype(np.float64) / total if total > 0 else counts.astype(np.float64)


# --------------------------------------------------------------------------- #
# Load artefacts
# --------------------------------------------------------------------------- #
def load_all():
    final = pd.read_csv(RESULTS_DIR / "final_summary.csv")
    part = pd.read_csv(PART_METRICS)
    histories, confusions = {}, {}
    for seed in SEEDS:
        for partition in PARTITIONS:
            key = (seed, partition)
            histories[key] = pd.read_csv(RESULTS_DIR / f"history_seed{seed}_{partition}.csv")
            confusions[key] = pd.read_csv(
                RESULTS_DIR / f"confusion_seed{seed}_{partition}.csv", index_col=0
            )
    return final, part, histories, confusions


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify(final, part, histories, confusions):
    print("===== VERIFICATION =====")
    checks = []

    runs = set(zip(final["seed"], final["partition"]))
    c1 = len(final) == 21 and len(runs) == 21
    checks.append(("1. Exactly 21 unique seed+partition runs", c1, f"{len(runs)} unique / {len(final)} rows"))

    c2 = all(len(histories[k]) == 30 and sorted(histories[k]["round"]) == list(range(1, 31))
             for k in histories)
    checks.append(("2. Every run has 30 rounds", c2, f"{len(histories)} history files"))

    merged = final.merge(part, left_on=["seed", "partition"],
                         right_on=["seed", "partition_type"], how="outer", indicator=True)
    c3 = (merged["_merge"] == "both").all() and len(merged) == 21
    checks.append(("3. Train<->partition merge is one-to-one", c3,
                   f"{(merged['_merge']=='both').sum()}/21 matched, {len(merged)} rows"))

    perf_cols = ["HD_skew", "JSD_skew", "EMD_skew", "best_macro_f1", "best_macro_recall",
                 "macro_f1_final", "variance_last_5_rounds"]
    rare_cols = [f"{n}_{m}" for n in HIGHLIGHTED for m in ("precision", "recall", "f1", "pr_auc")]
    c4 = not final[perf_cols + rare_cols].isna().any().any()
    n_missing = int(final[perf_cols + rare_cols].isna().sum().sum())
    checks.append(("4. No missing HD/JSD/EMD/performance values", c4, f"{n_missing} NaNs"))

    mismatches = []
    for k in histories:
        seed, partition = k
        stored = int(final[(final.seed == seed) & (final.partition == partition)]["best_round"].iloc[0])
        argmax_round = int(histories[k].loc[histories[k]["macro_f1"].idxmax(), "round"])
        if stored != argmax_round:
            mismatches.append((k, stored, argmax_round))
    c5 = len(mismatches) == 0
    checks.append(("5. Stored best_round == argmax(val macro_f1)", c5,
                   "all match" if c5 else f"{len(mismatches)} mismatches: {mismatches[:3]}"))

    bad_totals = [(k, int(confusions[k].values.sum())) for k in confusions
                  if int(confusions[k].values.sum()) != VAL_SIZE]
    c6 = len(bad_totals) == 0
    checks.append(("6. Confusion totals == validation size", c6,
                   f"all == {VAL_SIZE}" if c6 else f"bad: {bad_totals[:3]}"))

    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    if not all(ok for _, ok, _ in checks):
        raise SystemExit("Verification failed - aborting analysis.")
    print("All six checks passed.\n")


# --------------------------------------------------------------------------- #
# 1. run_level_master
# --------------------------------------------------------------------------- #
def build_run_master(final, part, histories, confusions, class_names):
    rows = []
    for _, fr in final.iterrows():
        seed, partition = int(fr["seed"]), fr["partition"]
        key = (seed, partition)
        hist = histories[key]
        best_round = int(fr["best_round"])
        b = hist[hist["round"] == best_round].iloc[0]

        conf = confusions[key].values
        pred_counts = conf.sum(axis=0)               # predicted per class (best checkpoint)
        total = pred_counts.sum()
        benign_prop = float(pred_counts[0] / total)
        num_pred_classes = int((pred_counts > 0).sum())

        pr = part[(part.seed == seed) & (part.partition_type == partition)].iloc[0]

        row = {
            "seed": seed, "partition": partition, "alpha": parse_alpha(partition),
            "HD_skew": fr["HD_skew"], "JSD_skew": fr["JSD_skew"], "EMD_skew": fr["EMD_skew"],
            "alloc_error_L1": float(pr["alloc_error_L1"]),
            "best_round": best_round, "rounds_to_best": int(fr["rounds_to_best"]),
            "best_macro_f1": float(fr["best_macro_f1"]),
            "macro_f1_final": float(fr["macro_f1_final"]),
            "delta_best_minus_final": float(fr["delta_best_minus_final"]),
            "variance_last_5_rounds": float(fr["variance_last_5_rounds"]),
            "best_macro_precision": float(b["macro_precision"]),
            "best_macro_recall": float(b["macro_recall"]),
            "best_balanced_accuracy": float(b["balanced_accuracy"]),
            "best_macro_pr_auc": float(b["macro_pr_auc"]),
            "best_worst_class_recall": float(b["worst_class_recall"]),
            "best_worst_class_f1": float(b["worst_class_f1"]),
            "predicted_benign_proportion": benign_prop,
            "num_predicted_classes": num_pred_classes,
        }
        for name in HIGHLIGHTED:
            for m in ("precision", "recall", "f1", "pr_auc"):
                row[f"{name}_{m}"] = float(fr[f"{name}_{m}"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df["partition"] = pd.Categorical(df["partition"], categories=PARTITIONS, ordered=True)
    df = df.sort_values(["partition", "seed"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "run_level_master.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 2. partition_summary
# --------------------------------------------------------------------------- #
def build_partition_summary(master):
    metrics = ["HD_skew", "JSD_skew", "EMD_skew", "best_macro_f1", "best_macro_recall",
               "best_macro_pr_auc", "best_worst_class_f1"]
    for name in HIGHLIGHTED:
        metrics += [f"{name}_precision", f"{name}_recall", f"{name}_f1", f"{name}_pr_auc"]

    means = {}
    rows = []
    for partition in PARTITIONS:
        sub = master[master["partition"] == partition]
        row = {"partition": partition}
        for metric in metrics:
            by_seed = {int(s): float(sub[sub.seed == s][metric].iloc[0]) for s in SEEDS}
            row[f"{metric}_s42"] = by_seed[42]
            row[f"{metric}_s43"] = by_seed[43]
            row[f"{metric}_s44"] = by_seed[44]
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=1))
        rows.append(row)
        means[partition] = {m: float(sub[m].mean()) for m in metrics}

    df = pd.DataFrame(rows)
    iid = means["iid"]
    for metric in metrics:
        df[f"{metric}_absdiff_iid"] = df[f"{metric}_mean"] - iid[metric]
        df[f"{metric}_pctdiff_iid"] = np.where(
            iid[metric] != 0, (df[f"{metric}_mean"] - iid[metric]) / iid[metric] * 100.0, np.nan
        )
    df.to_csv(OUT_DIR / "partition_summary.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 3. metric_associations
# --------------------------------------------------------------------------- #
def build_metric_associations(master):
    skew_metrics = ["HD_skew", "JSD_skew", "EMD_skew"]
    perf_metrics = ["best_macro_f1", "best_macro_recall", "best_macro_pr_auc",
                    "best_worst_class_f1", "Worms_f1", "Shellcode_f1"]
    rows = []
    for sk in skew_metrics:
        for pf in perf_metrics:
            rho_all, p_all = spearmanr(master[sk], master[pf])
            within = {}
            for s in SEEDS:
                sub = master[master.seed == s]
                rho_s, _ = spearmanr(sub[sk], sub[pf])
                within[s] = rho_s
            rows.append({
                "skew_metric": sk, "perf_metric": pf,
                "spearman_overall": rho_all, "p_overall": p_all,
                "spearman_seed42": within[42], "spearman_seed43": within[43],
                "spearman_seed44": within[44],
                "mean_within_seed": float(np.nanmean(list(within.values()))),
                "n_overall": len(master), "note": "exploratory (pilot, n=21)",
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "metric_associations.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 4. per_class_degradation
# --------------------------------------------------------------------------- #
def build_per_class_degradation(histories, final, class_names):
    per_metrics = ["precision", "recall", "f1", "pr_auc", "support", "predicted_count"]
    records = []
    for partition in PARTITIONS:
        for c in range(NUM_CLASSES):
            vals = {m: [] for m in per_metrics}
            for seed in SEEDS:
                key = (seed, partition)
                best_round = int(final[(final.seed == seed) & (final.partition == partition)]["best_round"].iloc[0])
                b = histories[key][histories[key]["round"] == best_round].iloc[0]
                for m in per_metrics:
                    vals[m].append(float(b[f"{m}_c{c}"]))
            rec = {"partition": partition, "class_id": c, "class_name": class_names[c]}
            for m in per_metrics:
                rec[m] = float(np.mean(vals[m]))
            records.append(rec)

    df = pd.DataFrame(records)
    iid = df[df["partition"] == "iid"].set_index("class_id")
    for m in ["precision", "recall", "f1", "pr_auc", "predicted_count"]:
        df[f"{m}_diff_iid"] = df.apply(lambda r: r[m] - iid.loc[r["class_id"], m], axis=1)
    df["partition"] = pd.Categorical(df["partition"], categories=PARTITIONS, ordered=True)
    df = df.sort_values(["class_id", "partition"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "per_class_degradation.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 5. convergence_and_collapse
# --------------------------------------------------------------------------- #
def build_convergence_collapse(master):
    df = master[[
        "seed", "partition", "alpha", "best_round", "best_macro_f1", "macro_f1_final",
        "delta_best_minus_final", "variance_last_5_rounds",
        "predicted_benign_proportion", "num_predicted_classes",
    ]].copy()
    df["collapsed_benign_only"] = (
        (df["predicted_benign_proportion"] >= BENIGN_FRACTION_COLLAPSE)
        | (df["num_predicted_classes"] <= 2)
    )
    df.to_csv(OUT_DIR / "convergence_and_collapse.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 6. partition_diagnostics (from controlled client index files)
# --------------------------------------------------------------------------- #
def build_partition_diagnostics(part, y_train, class_names):
    rows = []
    for seed in SEEDS:
        for partition in PARTITIONS:
            pdir = PART_ROOT / f"seed_{seed}" / partition
            client_idx = [np.load(pdir / f"client_{k:02d}_indices.npy") for k in range(NUM_CLIENTS)]
            counts = np.stack([np.bincount(y_train[idx], minlength=NUM_CLASSES) for idx in client_idx])
            pool_counts = counts.sum(axis=0)
            pool_dist = normalise(pool_counts)
            client_dists = [normalise(counts[k]) for k in range(NUM_CLIENTS)]

            to_pool = [hellinger(client_dists[k], pool_dist) for k in range(NUM_CLIENTS)]
            pairwise = [hellinger(client_dists[i], client_dists[j])
                        for i in range(NUM_CLIENTS) for j in range(i + 1, NUM_CLIENTS)]
            missing_cells = int((counts == 0).sum())

            pr = part[(part.seed == seed) & (part.partition_type == partition)].iloc[0]
            row = {
                "seed": seed, "partition": partition,
                "mean_client_pool_hd": float(np.mean(to_pool)),
                "median_client_pool_hd": float(np.median(to_pool)),
                "max_client_pool_hd": float(np.max(to_pool)),
                "mean_pairwise_hd": float(np.mean(pairwise)),
                "max_pairwise_hd": float(np.max(pairwise)),
                "missing_client_class_cells": missing_cells,
                "alloc_error_L1": float(pr["alloc_error_L1"]),
            }
            for name, cid in HIGHLIGHTED.items():
                row[f"clients_with_{name}"] = int((counts[:, cid] > 0).sum())
                for k in range(NUM_CLIENTS):
                    row[f"{name}_c{k}"] = int(counts[k, cid])
            rows.append(row)

    df = pd.DataFrame(rows)
    df["partition"] = pd.Categorical(df["partition"], categories=PARTITIONS, ordered=True)
    df = df.sort_values(["partition", "seed"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "partition_diagnostics.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# Console tables
# --------------------------------------------------------------------------- #
def print_tables(part_summary, assoc, per_class, conv, class_names):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)

    print("=== (A) PARTITION SUMMARY (mean over 3 seeds) ===")
    t = pd.DataFrame({
        "partition": part_summary["partition"],
        "HD": part_summary["HD_skew_mean"].round(4),
        "macroF1": part_summary["best_macro_f1_mean"].round(4),
        "macroF1_std": part_summary["best_macro_f1_std"].round(4),
        "macroF1_%dIID": part_summary["best_macro_f1_pctdiff_iid"].round(1),
        "macroRec": part_summary["best_macro_recall_mean"].round(4),
        "macroPRAUC": part_summary["best_macro_pr_auc_mean"].round(4),
        "worstF1": part_summary["best_worst_class_f1_mean"].round(4),
        "Worms_f1": part_summary["Worms_f1_mean"].round(4),
        "Shellcode_f1": part_summary["Shellcode_f1_mean"].round(4),
    })
    print(t.to_string(index=False))

    print("\n=== (B) SKEW vs PERFORMANCE — Spearman (exploratory, pilot n=21) ===")
    piv = assoc.pivot(index="perf_metric", columns="skew_metric", values="spearman_overall").round(3)
    piv = piv[["HD_skew", "JSD_skew", "EMD_skew"]]
    print(piv.to_string())
    print("  (overall rho across 21 runs; within-seed rho in metric_associations.csv)")

    print("\n=== (C) PER-CLASS mean F1 by partition (best checkpoint) ===")
    fpiv = per_class.pivot(index="class_name", columns="partition", values="f1")
    fpiv = fpiv.reindex([class_names[c] for c in range(NUM_CLASSES)])[PARTITIONS].round(3)
    print(fpiv.to_string())

    print("\n=== (D) CONVERGENCE / COLLAPSE ===")
    cc = conv.copy()
    cc = cc.sort_values(["partition", "seed"])
    show = pd.DataFrame({
        "seed": cc["seed"], "partition": cc["partition"],
        "best_rnd": cc["best_round"],
        "bestF1": cc["best_macro_f1"].round(4),
        "d_best_final": cc["delta_best_minus_final"].round(4),
        "var_last5": cc["variance_last_5_rounds"].round(5),
        "benign_frac": cc["predicted_benign_proportion"].round(4),
        "n_pred_cls": cc["num_predicted_classes"],
        "collapsed": cc["collapsed_benign_only"],
    })
    print(show.to_string(index=False))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names()
    y_train = np.load(PROCESSED_DIR / "y_train.npy")

    final, part, histories, confusions = load_all()
    verify(final, part, histories, confusions)

    master = build_run_master(final, part, histories, confusions, class_names)
    part_summary = build_partition_summary(master)
    assoc = build_metric_associations(master)
    per_class = build_per_class_degradation(histories, final, class_names)
    conv = build_convergence_collapse(master)
    diag = build_partition_diagnostics(part, y_train, class_names)

    print_tables(part_summary, assoc, per_class, conv, class_names)

    print("\nWritten to", OUT_DIR)
    for name in ["run_level_master", "partition_summary", "metric_associations",
                 "per_class_degradation", "convergence_and_collapse", "partition_diagnostics"]:
        print(f"  {name}.csv")


if __name__ == "__main__":
    main()
