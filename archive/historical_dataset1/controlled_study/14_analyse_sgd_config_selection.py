"""
Analysis-only verification & summary of the plain-SGD learning-rate config
selection (reads saved artefacts from results/fl_sgd_config_selection/).

Covers 3 learning rates x 3 seeds x 7 partitions = 63 runs. It does NOT train,
does NOT touch the test set, and does NOT modify any AdamW artefact. It does not
select a learning rate.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

SGD_DIR = Path("results/fl_sgd_config_selection")
OUT_DIR = SGD_DIR / "analysis"
PART_METRICS = Path("results/fl_partitions/controlled_metrics.csv")
AUDITED_BASELINE = Path("/tmp/audited_before_full.txt")  # recorded before the SGD campaign

NUM_CLASSES = 10
VAL_SIZE = 358_542
EXPECTED_ROUNDS = 50
SEEDS = [42, 43, 44]
PARTITIONS = ["alpha_0.01", "alpha_0.05", "alpha_0.1", "alpha_0.5", "alpha_1.0", "alpha_5.0", "iid"]
TAG_TO_LR = {"sgd_lr_0p001": 0.001, "sgd_lr_0p01": 0.01, "sgd_lr_0p1": 0.1}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def h(tag, seed, part, kind, ext):
    return SGD_DIR / f"{tag}_{kind}_seed{seed}_{part}.{ext}"


def load_run(tag, seed, part):
    hist = pd.read_csv(h(tag, seed, part, "history", "csv"))
    with open(h(tag, seed, part, "config", "json")) as f:
        cfg = json.load(f)
    conf = pd.read_csv(h(tag, seed, part, "confusion", "csv"), index_col=0).values
    return hist, cfg, conf


# ---------------- verification ---------------- #
def verify(runs, skew):
    print("===== VERIFICATION =====")
    checks = []
    keys = list(runs.keys())

    c1 = len(keys) == 63 and len(set(keys)) == 63
    checks.append(("1. Exactly 63 unique lr-seed-partition runs", c1, f"{len(set(keys))} runs"))

    c2 = all(len(runs[k][0]) == EXPECTED_ROUNDS and
             sorted(runs[k][0]["round"]) == list(range(1, EXPECTED_ROUNDS + 1)) for k in keys)
    checks.append(("2. Every run has exactly 50 rounds", c2, ""))

    init_paths = {runs[k][1].get("initial_state_path") for k in keys}
    same_init = len(init_paths) == 1
    init_ck = None
    if same_init:
        ip = Path(next(iter(init_paths)))
        init_ck = sha256(ip) if ip.exists() else "MISSING"
    checks.append(("3. Every run loaded the same initial-state path", same_init,
                   f"path={init_paths}, sha256={init_ck}"))

    c4 = all(runs[k][1].get("momentum") == 0 and runs[k][1].get("weight_decay") == 0 for k in keys)
    checks.append(("4. Every run momentum==0 and weight_decay==0", c4, ""))

    bad_best = []
    for k in keys:
        hist, cfg, _ = runs[k]
        argmax_round = int(hist.loc[hist["macro_f1"].idxmax(), "round"])
        if int(cfg.get("best_round")) != argmax_round:
            bad_best.append((k, cfg.get("best_round"), argmax_round))
    c5 = len(bad_best) == 0
    checks.append(("5. Stored best_round == argmax(val macro_f1)", c5, "" if c5 else str(bad_best[:3])))

    bad_shape, bad_rowsum, max_dev = [], [], 0.0
    for k in keys:
        tag, seed, part = k
        probs = np.load(h(tag, seed, part, "val_probs", "npy"))
        if probs.shape != (VAL_SIZE, NUM_CLASSES):
            bad_shape.append((k, probs.shape))
        rs = probs.sum(axis=1)
        dev = float(np.max(np.abs(rs - 1.0)))
        max_dev = max(max_dev, dev)
        if not np.allclose(rs, 1.0, atol=1e-3):
            bad_rowsum.append((k, dev))
        del probs
    checks.append(("6. Every probs array shape (358542, 10)", len(bad_shape) == 0, "" if not bad_shape else str(bad_shape[:3])))
    checks.append(("7. Every probs row sums ~1", len(bad_rowsum) == 0, f"max|rowsum-1|={max_dev:.2e}"))

    # 8. no test-array reference in the producing script or this one
    tok_x, tok_y = "X_" + "test", "y_" + "test"
    src13 = Path("13_select_fedavg_sgd_config.py").read_text()
    src14 = Path(__file__).read_text()
    c8 = tok_x not in src13 and tok_y not in src13 and tok_x not in src14 and tok_y not in src14
    checks.append(("8. No test array referenced (13 & 14 source)", c8, ""))

    # 9. audited AdamW artefacts byte-for-byte unchanged
    if AUDITED_BASELINE.exists():
        mism = []
        for line in AUDITED_BASELINE.read_text().splitlines():
            if not line.strip():
                continue
            digest, path = line.split(maxsplit=1)
            p = Path(path)
            cur = sha256(p) if p.exists() else "MISSING"
            if cur != digest:
                mism.append(path)
        c9 = len(mism) == 0
        checks.append(("9. Audited AdamW artefacts unchanged", c9, f"{len(mism)} changed" if mism else "all match baseline"))
    else:
        checks.append(("9. Audited AdamW artefacts unchanged", None, "baseline manifest not found"))

    # 10. no NaN / inf in any history numeric cell
    bad_nan = []
    for k in keys:
        hist = runs[k][0]
        num = hist.select_dtypes(include=[np.number]).to_numpy()
        if not np.isfinite(num).all():
            bad_nan.append(k)
    c10 = len(bad_nan) == 0
    checks.append(("10. No NaN/inf in run histories", c10, "" if c10 else str(bad_nan[:3])))

    for name, ok, detail in checks:
        flag = "PASS" if ok else ("WARN" if ok is None else "FAIL")
        print(f"  [{flag}] {name}  ({detail})")
    hard_fail = [c for c in checks if c[1] is False]
    if hard_fail:
        raise SystemExit(f"Verification failed: {[c[0] for c in hard_fail]}")
    print("Verification complete.\n")
    return init_ck


# ---------------- run-level table ---------------- #
def build_run_level(runs, skew):
    rows = []
    for (tag, seed, part), (hist, cfg, conf) in runs.items():
        hist = hist.sort_values("round").reset_index(drop=True)
        mf = hist["macro_f1"].to_numpy()
        rounds = hist["round"].to_numpy()

        m30 = hist[hist["round"] <= 30]
        best30_idx = int(m30["macro_f1"].idxmax())
        best30 = float(m30.loc[best30_idx, "macro_f1"]); best30_round = int(m30.loc[best30_idx, "round"])
        best50_idx = int(hist["macro_f1"].idxmax())
        best50 = float(hist.loc[best50_idx, "macro_f1"]); best50_round = int(hist.loc[best50_idx, "round"])

        f1_r30 = float(hist[hist["round"] == 30]["macro_f1"].iloc[0])
        f1_r50 = float(hist[hist["round"] == 50]["macro_f1"].iloc[0])
        mean_41_50 = float(hist[(hist["round"] >= 41) & (hist["round"] <= 50)]["macro_f1"].mean())

        chk = hist[hist["round"] == best50_round].iloc[0]
        chk_recall = float(chk["macro_recall"]); chk_prauc = float(chk["macro_pr_auc"])
        pred_counts = conf.sum(axis=0); total = pred_counts.sum()
        benign_prop = float(pred_counts[0] / total); n_pred = int((pred_counts > 0).sum())

        hd = float(skew.loc[(seed, part), "HD_skew"])
        rows.append({
            "learning_rate": TAG_TO_LR[tag], "seed": seed, "partition": part, "HD_skew": hd,
            "best_macro_f1_r1_30": best30, "best_round_1_30": best30_round,
            "best_macro_f1_r1_50": best50, "best_round_1_50": best50_round,
            "macro_f1_round30": f1_r30, "macro_f1_round50": f1_r50,
            "mean_macro_f1_r41_50": mean_41_50,
            "checkpoint_macro_recall": chk_recall, "checkpoint_macro_pr_auc": chk_prauc,
            "checkpoint_benign_proportion": benign_prop, "checkpoint_num_predicted_classes": n_pred,
        })
    df = pd.DataFrame(rows)
    df["partition"] = pd.Categorical(df["partition"], categories=PARTITIONS, ordered=True)
    df = df.sort_values(["learning_rate", "partition", "seed"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "sgd_config_run_level.csv", index=False)
    return df


def build_lr_summary(run_level):
    metrics = {
        "best_macro_f1_r1_30": "best_f1_30", "best_macro_f1_r1_50": "best_f1_50",
        "macro_f1_round30": "f1_round30", "macro_f1_round50": "f1_round50",
        "mean_macro_f1_r41_50": "mean_f1_41_50",
        "checkpoint_macro_recall": "macro_recall", "checkpoint_macro_pr_auc": "macro_pr_auc",
    }
    rows = []
    for lr in sorted(run_level["learning_rate"].unique()):
        sub = run_level[run_level["learning_rate"] == lr]
        row = {"learning_rate": lr, "n_runs": len(sub)}
        for col, short in metrics.items():
            row[f"{short}_mean"] = float(sub[col].mean())
            row[f"{short}_std"] = float(sub[col].std(ddof=1))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sgd_config_lr_summary.csv", index=False)
    return df


def build_partition_summary(run_level):
    metrics = {"best_macro_f1_r1_50": "best50_f1", "macro_f1_round50": "f1_r50",
               "checkpoint_macro_recall": "macro_recall", "checkpoint_macro_pr_auc": "macro_pr_auc"}
    rows = []
    for lr in sorted(run_level["learning_rate"].unique()):
        for part in PARTITIONS:
            sub = run_level[(run_level["learning_rate"] == lr) & (run_level["partition"] == part)]
            row = {"learning_rate": lr, "partition": part}
            for col, short in metrics.items():
                by_seed = {int(s): float(sub[sub.seed == s][col].iloc[0]) for s in SEEDS}
                row[f"{short}_s42"], row[f"{short}_s43"], row[f"{short}_s44"] = by_seed[42], by_seed[43], by_seed[44]
                row[f"{short}_mean"] = float(sub[col].mean())
                row[f"{short}_std"] = float(sub[col].std(ddof=1))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sgd_config_partition_summary.csv", index=False)
    return df


def build_round_budget(run_level):
    rows = []
    for lr in sorted(run_level["learning_rate"].unique()):
        sub = run_level[run_level["learning_rate"] == lr]
        improvement = sub["best_macro_f1_r1_50"] - sub["best_macro_f1_r1_30"]
        rows.append({
            "learning_rate": lr,
            "mean_improvement_30_to_50": float(improvement.mean()),
            "runs_best_round_gt_30": int((sub["best_round_1_50"] > 30).sum()),
            "runs_improve_ge_0.01_after_30": int((improvement >= 0.01).sum()),
            "n_runs": len(sub),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sgd_round_budget_comparison.csv", index=False)
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    skew = pd.read_csv(PART_METRICS).set_index(["seed", "partition_type"])

    runs, missing, invalid = {}, [], []
    for tag in TAG_TO_LR:
        for seed in SEEDS:
            for part in PARTITIONS:
                hp = h(tag, seed, part, "history", "csv")
                if not hp.exists():
                    missing.append((tag, seed, part)); continue
                try:
                    hist, cfg, conf = load_run(tag, seed, part)
                    if len(hist) != EXPECTED_ROUNDS:
                        invalid.append(((tag, seed, part), f"{len(hist)} rounds"))
                    runs[(tag, seed, part)] = (hist, cfg, conf)
                except Exception as e:
                    invalid.append(((tag, seed, part), repr(e)))

    print(f"Loaded {len(runs)} runs; missing={len(missing)}; invalid={len(invalid)}\n")
    if missing:
        print("MISSING runs:", missing)
    if invalid:
        print("INVALID runs:", invalid)

    init_ck = verify(runs, skew)

    run_level = build_run_level(runs, skew)
    lr_summary = build_lr_summary(run_level)
    part_summary = build_partition_summary(run_level)
    budget = build_round_budget(run_level)

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

    print("=== (A) LEARNING-RATE SUMMARY (mean/std over 21 seed-partition runs) ===")
    t = lr_summary.copy()
    for c in [col for col in t.columns if col not in ("learning_rate", "n_runs")]:
        t[c] = t[c].round(4)
    print(t.to_string(index=False))

    print("\n=== (B) LEARNING-RATE x PARTITION (best 50-round macro-F1 mean/std) ===")
    piv_m = part_summary.pivot(index="partition", columns="learning_rate", values="best50_f1_mean").reindex(PARTITIONS).round(4)
    piv_s = part_summary.pivot(index="partition", columns="learning_rate", values="best50_f1_std").reindex(PARTITIONS).round(4)
    print("mean:\n" + piv_m.to_string())
    print("std:\n" + piv_s.to_string())

    print("\n=== (C) ROUND-BUDGET COMPARISON (30 -> 50) ===")
    b = budget.copy()
    b["mean_improvement_30_to_50"] = b["mean_improvement_30_to_50"].round(5)
    print(b.to_string(index=False))

    print("\n=== FAILED / INVALID RUNS ===")
    print("missing:", missing if missing else "none")
    print("invalid:", invalid if invalid else "none")

    print(f"\nShared initial-state sha256: {init_ck}")
    print("Written to", OUT_DIR)


if __name__ == "__main__":
    main()
