"""
Analysis-only FedProx mu-screening study (validation-only).

Reads the 105 positive-mu FedProx runs (results/fl_fedprox_sgd/), the 21 matched
FedAvg-SGD lr=0.1 baseline runs (results/fl_sgd_config_selection/, treated as
mu=0), and the controlled partition skew metrics (HD/JSD/EMD). It does NOT train,
NOT touch the test set, and NOT modify any existing artefact. It does NOT select
a final mu. Spearman associations are labelled exploratory. Mean proximal penalty
is a training diagnostic only (it scales with mu; not a cross-mu drift measure).
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEDPROX_DIR = Path("results/fl_fedprox_sgd")
FEDAVG_DIR = Path("results/fl_sgd_config_selection")
FEDAVG_TAG = "sgd_lr_0p1"
PART_METRICS = Path("results/fl_partitions/controlled_metrics.csv")
OUT_DIR = FEDPROX_DIR / "analysis"
MANIFEST = Path("/tmp/manifest_before_fedprox_grid.txt")
INIT_PATH = "models/fl_noniid_controlled/initial_global_model.pt"

NUM_CLASSES = 10
VAL_SIZE = 358_542
EXPECTED_ROUNDS = 50
SEEDS = [42, 43, 44]
PARTITIONS = ["alpha_0.01", "alpha_0.05", "alpha_0.1", "alpha_0.5", "alpha_1.0", "alpha_5.0", "iid"]
POSITIVE_MUS = [0.001, 0.01, 0.1, 0.5, 1.0]
ALL_MUS = [0.0] + POSITIVE_MUS

CLASS_NAMES = ["Benign", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
               "Generic", "Reconnaissance", "Shellcode", "Worms"]
NAMED = {"Shellcode": 8, "Exploits": 4, "Reconnaissance": 7, "DoS": 3, "Worms": 9}


def mu_to_frag(mu):
    return {0.001: "0p001", 0.01: "0p01", 0.1: "0p1", 0.5: "0p5", 1.0: "1p0"}[mu]


def dir_tag(mu):
    if mu == 0.0:
        return FEDAVG_DIR, FEDAVG_TAG
    return FEDPROX_DIR, f"fedprox_mu{mu_to_frag(mu)}_lr0p1_r50"


def path_of(d, tag, seed, part, kind, ext):
    return d / f"{tag}_{kind}_seed{seed}_{part}.{ext}"


def load_run(mu, seed, part):
    d, tag = dir_tag(mu)
    hist = pd.read_csv(path_of(d, tag, seed, part, "history", "csv"))
    with open(path_of(d, tag, seed, part, "config", "json")) as f:
        cfg = json.load(f)
    conf = pd.read_csv(path_of(d, tag, seed, part, "confusion", "csv"), index_col=0).values
    return hist, cfg, conf


def metrics_row(mu, seed, part, hist, conf):
    hist = hist.sort_values("round").reset_index(drop=True)
    best_round = int(hist.loc[hist["macro_f1"].idxmax(), "round"])
    b = hist[hist["round"] == best_round].iloc[0]
    pred_counts = conf.sum(axis=0)
    total = pred_counts.sum()
    benign_prop = float(pred_counts[0] / total)
    n_pred = int((pred_counts > 0).sum())
    row = {
        "mu": mu, "seed": seed, "partition": part,
        "best_round": best_round,
        "best_macro_f1": float(b["macro_f1"]),
        "final_macro_f1": float(hist[hist["round"] == 50]["macro_f1"].iloc[0]),
        "mean_macro_f1_41_50": float(hist[(hist["round"] >= 41) & (hist["round"] <= 50)]["macro_f1"].mean()),
        "macro_recall": float(b["macro_recall"]),
        "macro_pr_auc": float(b["macro_pr_auc"]),
        "balanced_accuracy": float(b["balanced_accuracy"]),
        "worst_class_f1": float(b["worst_class_f1"]),
        "predicted_benign_proportion": benign_prop,
        "num_predicted_classes": n_pred,
        "one_class_indicator": int(n_pred == 1),
        "two_class_restricted_indicator": int(n_pred == 2),
    }
    # training-loss diagnostics (FedProx histories only)
    for col in ("mean_task_loss", "mean_proximal_penalty", "mean_total_loss"):
        row[col] = float(hist[col].mean()) if col in hist.columns else np.nan
    for c in range(NUM_CLASSES):
        row[f"precision_c{c}"] = float(b[f"precision_c{c}"])
        row[f"recall_c{c}"] = float(b[f"recall_c{c}"])
        row[f"f1_c{c}"] = float(b[f"f1_c{c}"])
        row[f"pr_auc_c{c}"] = float(b[f"pr_auc_c{c}"])
        row[f"support_c{c}"] = int(b[f"support_c{c}"])
        row[f"predicted_count_c{c}"] = int(b[f"predicted_count_c{c}"])
    return row


# --------------------------- verification --------------------------- #
def verify(runs, fedavg):
    print("===== VERIFICATION =====")
    checks = []
    pos_keys = [(mu, s, p) for mu in POSITIVE_MUS for s in SEEDS for p in PARTITIONS]
    have_pos = [k for k in pos_keys if k in runs]
    checks.append(("1. Exactly 105 positive-mu FedProx runs", len(have_pos) == 105, f"{len(have_pos)} found"))
    checks.append(("2. Exactly 21 matched FedAvg baseline runs", len(fedavg) == 21, f"{len(fedavg)} found"))

    c3 = all(len(runs[k][0]) == EXPECTED_ROUNDS for k in runs) and all(len(fedavg[k][0]) == EXPECTED_ROUNDS for k in fedavg)
    checks.append(("3. Every valid run has 50 rounds", c3, ""))

    # 4. one-to-one join by (seed,partition) for every mu
    join_ok = all(set((s, p) for s in SEEDS for p in PARTITIONS) ==
                  set((s, p) for (mu, s, p) in have_pos if mu == m) for m in POSITIVE_MUS) and \
              set(fedavg.keys()) == set((s, p) for s in SEEDS for p in PARTITIONS)
    checks.append(("4. seed x partition join is one-to-one", join_ok, ""))

    # 5. stored mu/lr/momentum/wd match
    cfg_bad = []
    for (mu, s, p), (_, cfg, _) in runs.items():
        if not (abs(cfg.get("mu", -1) - mu) < 1e-12 and cfg.get("learning_rate") == 0.1
                and cfg.get("momentum") == 0 and cfg.get("weight_decay") == 0):
            cfg_bad.append((mu, s, p))
    for (s, p), (_, cfg, _) in fedavg.items():
        if not (cfg.get("learning_rate") == 0.1 and cfg.get("momentum") == 0 and cfg.get("weight_decay") == 0):
            cfg_bad.append((0.0, s, p))
    checks.append(("5. Stored mu/lr/momentum/wd match config", len(cfg_bad) == 0, str(cfg_bad[:3])))

    # 6. same initial checksum (path) everywhere
    paths = {cfg.get("initial_state_path") for _, cfg, _ in list(runs.values()) + list(fedavg.values())}
    init_ck = hashlib.sha256(open(INIT_PATH, "rb").read()).hexdigest() if Path(INIT_PATH).exists() else "MISSING"
    checks.append(("6. Same initial-state path everywhere", paths == {INIT_PATH}, f"paths={paths}, sha256={init_ck}"))

    # 7. best_round == first argmax macro_f1
    br_bad = []
    for (mu, s, p), (hist, cfg, _) in runs.items():
        if int(cfg["best_round"]) != int(hist.loc[hist["macro_f1"].idxmax(), "round"]):
            br_bad.append((mu, s, p))
    for (s, p), (hist, cfg, _) in fedavg.items():
        if int(cfg["best_round"]) != int(hist.loc[hist["macro_f1"].idxmax(), "round"]):
            br_bad.append((0.0, s, p))
    checks.append(("7. best_round == first max val macro_f1", len(br_bad) == 0, str(br_bad[:3])))

    # 8. probs shape + row sums (all 126)
    shp_bad, rs_bad, max_dev = [], [], 0.0
    for (mu, s, p) in list(runs.keys()):
        d, tag = dir_tag(mu)
        pr = np.load(path_of(d, tag, s, p, "val_probs", "npy"))
        if pr.shape != (VAL_SIZE, NUM_CLASSES):
            shp_bad.append((mu, s, p, pr.shape))
        dev = float(np.max(np.abs(pr.sum(1) - 1.0))); max_dev = max(max_dev, dev)
        if not np.allclose(pr.sum(1), 1.0, atol=1e-3):
            rs_bad.append((mu, s, p, dev))
        del pr
    for (s, p) in fedavg:
        pr = np.load(path_of(FEDAVG_DIR, FEDAVG_TAG, s, p, "val_probs", "npy"))
        if pr.shape != (VAL_SIZE, NUM_CLASSES):
            shp_bad.append((0.0, s, p, pr.shape))
        dev = float(np.max(np.abs(pr.sum(1) - 1.0))); max_dev = max(max_dev, dev)
        del pr
    checks.append(("8. Probs shape (358542,10) & row sums~1", len(shp_bad) == 0 and len(rs_bad) == 0, f"max|rowsum-1|={max_dev:.1e}"))

    # 9. no NaN/inf in histories
    nan_bad = []
    for k, (hist, _, _) in list(runs.items()) + [((0.0, *k), v) for k, v in fedavg.items()]:
        num = hist.select_dtypes(include=[np.number]).to_numpy()
        if not np.isfinite(num).all():
            nan_bad.append(k)
    checks.append(("9. No NaN/inf in run histories", len(nan_bad) == 0, str(nan_bad[:3])))

    # 10. no test reference in 15 & 16
    tok_x, tok_y = "X_" + "test", "y_" + "test"
    c10 = all(tok_x not in Path(f).read_text() and tok_y not in Path(f).read_text()
              for f in ("15_train_fedprox_sgd.py", __file__))
    checks.append(("10. No test array referenced (15 & 16)", c10, ""))

    # 11. existing research artefacts unchanged (manifest)
    if MANIFEST.exists():
        mism = []
        for line in MANIFEST.read_text().splitlines():
            if not line.strip():
                continue
            digest, path = line.split(maxsplit=1)
            p = Path(path)
            cur = hashlib.sha256(open(p, "rb").read()).hexdigest() if p.exists() else "MISSING"
            if cur != digest:
                mism.append(path)
        checks.append(("11. Existing research artefacts unchanged", len(mism) == 0, f"{len(mism)} changed" if mism else "all match"))
    else:
        checks.append(("11. Existing research artefacts unchanged", None, "manifest not found"))

    for name, ok, detail in checks:
        flag = "PASS" if ok else ("WARN" if ok is None else "FAIL")
        print(f"  [{flag}] {name}  ({detail})")
    if any(ok is False for _, ok, _ in checks):
        raise SystemExit("Verification failed.")
    print("Verification complete.\n")
    return init_ck


# --------------------------- outputs --------------------------- #
def build_tables(runs, fedavg, skew):
    # run-level metric rows keyed by (mu,seed,part), including mu=0
    rows = {}
    for (mu, s, p), (hist, cfg, conf) in runs.items():
        rows[(mu, s, p)] = metrics_row(mu, s, p, hist, conf)
    for (s, p), (hist, cfg, conf) in fedavg.items():
        rows[(0.0, s, p)] = metrics_row(0.0, s, p, hist, conf)

    def add_skew(r):
        hd = float(skew.loc[(r["seed"], r["partition"]), "HD_skew"])
        js = float(skew.loc[(r["seed"], r["partition"]), "JSD_skew"])
        emd = float(skew.loc[(r["seed"], r["partition"]), "EMD_skew"])
        return {**r, "HD": hd, "JS": js, "EMD": emd}

    # ---- fedprox_mu_run_level.csv (positive mu only) ----
    rl = [add_skew(rows[(mu, s, p)]) for mu in POSITIVE_MUS for s in SEEDS for p in PARTITIONS]
    rl_df = pd.DataFrame(rl)
    front = ["mu", "seed", "partition", "HD", "JS", "EMD", "best_round", "best_macro_f1",
             "final_macro_f1", "mean_macro_f1_41_50", "macro_recall", "macro_pr_auc",
             "balanced_accuracy", "predicted_benign_proportion", "num_predicted_classes",
             "one_class_indicator", "two_class_restricted_indicator",
             "mean_task_loss", "mean_proximal_penalty", "mean_total_loss"]
    rl_df = rl_df[front + [c for c in rl_df.columns if c not in front]]
    rl_df.to_csv(OUT_DIR / "fedprox_mu_run_level.csv", index=False)

    # ---- fedprox_mu_vs_fedavg.csv ----
    diff_rows = []
    diff_metrics = ["best_macro_f1", "final_macro_f1", "macro_recall", "macro_pr_auc",
                    "balanced_accuracy", "predicted_benign_proportion",
                    "num_predicted_classes", "best_round"]
    for mu in POSITIVE_MUS:
        for s in SEEDS:
            for p in PARTITIONS:
                fp = rows[(mu, s, p)]; fa = rows[(0.0, s, p)]
                d = {"mu": mu, "seed": s, "partition": p}
                for m in diff_metrics:
                    d[f"d_{m}"] = fp[m] - fa[m]
                for c in range(NUM_CLASSES):
                    d[f"d_f1_c{c}"] = fp[f"f1_c{c}"] - fa[f"f1_c{c}"]
                    d[f"d_recall_c{c}"] = fp[f"recall_c{c}"] - fa[f"recall_c{c}"]
                diff_rows.append(d)
    vs_df = pd.DataFrame(diff_rows)
    vs_df.to_csv(OUT_DIR / "fedprox_mu_vs_fedavg.csv", index=False)

    # ---- fedprox_mu_summary.csv (mu=0 + positives) ----
    summ_metrics = ["best_macro_f1", "final_macro_f1", "mean_macro_f1_41_50", "macro_recall",
                    "macro_pr_auc", "balanced_accuracy", "predicted_benign_proportion",
                    "num_predicted_classes"]
    srows = []
    for mu in ALL_MUS:
        sub = pd.DataFrame([rows[(mu, s, p)] for s in SEEDS for p in PARTITIONS])
        row = {"mu": mu, "n_runs": len(sub)}
        for m in summ_metrics:
            row[f"{m}_mean"] = float(sub[m].mean()); row[f"{m}_std"] = float(sub[m].std(ddof=1))
        row["count_one_class"] = int(sub["one_class_indicator"].sum())
        row["count_two_class_restricted"] = int(sub["two_class_restricted_indicator"].sum())
        srows.append(row)
    summ_df = pd.DataFrame(srows)
    summ_df.to_csv(OUT_DIR / "fedprox_mu_summary.csv", index=False)

    # ---- fedprox_mu_partition_summary.csv ----
    prows = []
    part_metrics = {"best_macro_f1": "best_macro_f1", "macro_recall": "macro_recall",
                    "macro_pr_auc": "macro_pr_auc", "Shellcode_f1": "f1_c8",
                    "Exploits_f1": "f1_c4", "Reconnaissance_f1": "f1_c7", "DoS_f1": "f1_c3",
                    "Worms_f1": "f1_c9", "num_predicted_classes": "num_predicted_classes"}
    for mu in ALL_MUS:
        for p in PARTITIONS:
            sub = {s: rows[(mu, s, p)] for s in SEEDS}
            row = {"mu": mu, "partition": p}
            for label, col in part_metrics.items():
                vals = [sub[s][col] for s in SEEDS]
                row[f"{label}_s42"], row[f"{label}_s43"], row[f"{label}_s44"] = vals
                row[f"{label}_mean"] = float(np.mean(vals)); row[f"{label}_std"] = float(np.std(vals, ddof=1))
            prows.append(row)
    part_df = pd.DataFrame(prows)
    part_df.to_csv(OUT_DIR / "fedprox_mu_partition_summary.csv", index=False)

    # ---- fedprox_mu_associations.csv (exploratory) ----
    assoc_targets = {"macro_f1": "best_macro_f1", "macro_recall": "macro_recall",
                     "macro_pr_auc": "macro_pr_auc", "worst_class_f1": "worst_class_f1",
                     "Shellcode_f1": "f1_c8", "Exploits_f1": "f1_c4"}
    arows = []
    for mu in ALL_MUS:
        sub = pd.DataFrame([add_skew(rows[(mu, s, p)]) for s in SEEDS for p in PARTITIONS])
        for label, col in assoc_targets.items():
            rho_all, p_all = spearmanr(sub["HD"], sub[col])
            within = {}
            for s in SEEDS:
                ss = sub[sub.seed == s]
                within[s], _ = spearmanr(ss["HD"], ss[col])
            arows.append({"mu": mu, "target": label, "spearman_HD_overall": rho_all, "p_overall": p_all,
                          "spearman_seed42": within[42], "spearman_seed43": within[43],
                          "spearman_seed44": within[44], "mean_within_seed": float(np.nanmean(list(within.values()))),
                          "n_overall": len(sub), "note": "exploratory (pilot, n=21)"})
    assoc_df = pd.DataFrame(arows)
    assoc_df.to_csv(OUT_DIR / "fedprox_mu_associations.csv", index=False)

    return rl_df, vs_df, summ_df, part_df, assoc_df, rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    skew = pd.read_csv(PART_METRICS).set_index(["seed", "partition_type"])

    runs, fedavg, missing, invalid = {}, {}, [], []
    for mu in POSITIVE_MUS:
        for s in SEEDS:
            for p in PARTITIONS:
                d, tag = dir_tag(mu)
                if not path_of(d, tag, s, p, "history", "csv").exists():
                    missing.append((mu, s, p)); continue
                try:
                    hist, cfg, conf = load_run(mu, s, p)
                    if len(hist) != EXPECTED_ROUNDS:
                        invalid.append(((mu, s, p), f"{len(hist)} rounds"))
                    runs[(mu, s, p)] = (hist, cfg, conf)
                except Exception as e:
                    invalid.append(((mu, s, p), repr(e)))
    for s in SEEDS:
        for p in PARTITIONS:
            if not path_of(FEDAVG_DIR, FEDAVG_TAG, s, p, "history", "csv").exists():
                missing.append((0.0, s, p)); continue
            hist, cfg, conf = load_run(0.0, s, p)
            fedavg[(s, p)] = (hist, cfg, conf)

    print(f"Loaded {len(runs)} positive-mu runs, {len(fedavg)} FedAvg baselines; "
          f"missing={len(missing)}, invalid={len(invalid)}\n")
    if missing: print("MISSING:", missing)
    if invalid: print("INVALID:", invalid)

    init_ck = verify(runs, fedavg)
    rl_df, vs_df, summ_df, part_df, assoc_df, rows = build_tables(runs, fedavg, skew)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 60)

    print("=== (A) MU SUMMARY (mean over 21 runs; mu=0 is FedAvg baseline) ===")
    t = summ_df[["mu", "best_macro_f1_mean", "best_macro_f1_std", "final_macro_f1_mean",
                 "macro_recall_mean", "macro_pr_auc_mean", "balanced_accuracy_mean",
                 "predicted_benign_proportion_mean", "num_predicted_classes_mean",
                 "count_one_class", "count_two_class_restricted"]].copy()
    for c in t.columns:
        if c not in ("mu", "count_one_class", "count_two_class_restricted"):
            t[c] = t[c].round(4)
    print(t.to_string(index=False))

    print("\n=== (B) MU x PARTITION best macro-F1 (mean over 3 seeds) ===")
    piv = part_df.pivot(index="partition", columns="mu", values="best_macro_f1_mean").reindex(PARTITIONS)[ALL_MUS].round(4)
    print(piv.to_string())

    print("\n=== (C) MEAN matched FedProx-minus-FedAvg differences by mu ===")
    dsumm = vs_df.groupby("mu")[["d_best_macro_f1", "d_final_macro_f1", "d_macro_recall",
                                 "d_macro_pr_auc", "d_balanced_accuracy",
                                 "d_predicted_benign_proportion", "d_num_predicted_classes",
                                 "d_best_round"]].mean().round(4)
    print(dsumm.to_string())

    print("\n=== (D) ONE-CLASS and TWO-CLASS restricted-prediction counts by mu (of 21) ===")
    print(summ_df[["mu", "count_one_class", "count_two_class_restricted"]].to_string(index=False))

    print("\n=== INVALID / FAILED RUNS ===")
    print("missing:", missing if missing else "none")
    print("invalid:", invalid if invalid else "none")

    print(f"\nInitial-state sha256: {init_ck}")
    print("Written to", OUT_DIR)
    for n in ["fedprox_mu_run_level", "fedprox_mu_vs_fedavg", "fedprox_mu_summary",
              "fedprox_mu_partition_summary", "fedprox_mu_associations"]:
        print(f"  {n}.csv")


if __name__ == "__main__":
    main()
