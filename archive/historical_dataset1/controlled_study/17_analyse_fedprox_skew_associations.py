"""
Analysis-only: extend the FedProx mu-screening skew associations to ALL THREE
skew metrics (HD, JSD, EMD).

16_analyse_fedprox_mu_selection.py computed Spearman associations against HD only.
This companion reuses 16's per-run metric extraction (imported, not re-run) and
computes, for each mu level (FedAvg mu=0 baseline + the five positive mus), the
overall and within-seed Spearman association between each of HD / JSD / EMD and
six validation performance targets. Associations are exploratory (pilot, n=21).

Reads existing artefacts only. Does not train, does not touch the test set, and
does not modify or overwrite any existing file (writes one new CSV).
"""

from pathlib import Path
import importlib.util
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PART_METRICS = Path("results/fl_partitions/controlled_metrics.csv")
OUT_DIR = Path("results/fl_fedprox_sgd/analysis")
OUT_CSV = OUT_DIR / "fedprox_skew_metric_associations.csv"
MANIFEST = Path("/tmp/manifest_before_fedprox_grid.txt")

SKEW_METRICS = {"HD": "HD_skew", "JSD": "JSD_skew", "EMD": "EMD_skew"}
TARGETS = {
    "macro_f1": "best_macro_f1",
    "macro_recall": "macro_recall",
    "macro_pr_auc": "macro_pr_auc",
    "worst_class_f1": "worst_class_f1",
    "Shellcode_f1": "f1_c8",
    "Exploits_f1": "f1_c4",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m16 = load_module("m16", "16_analyse_fedprox_mu_selection.py")

    # No-test-array assertion for this script (16 already asserts for itself).
    tok_x, tok_y = "X_" + "test", "y_" + "test"
    src = Path(__file__).read_text()
    assert tok_x not in src and tok_y not in src, "Test-array reference found in 17."

    skew = pd.read_csv(PART_METRICS).set_index(["seed", "partition_type"])
    seeds, partitions, all_mus = m16.SEEDS, m16.PARTITIONS, m16.ALL_MUS

    # Build per-run metric rows (all mu incl. 0) reusing 16's extraction.
    rows = {}
    missing = []
    for mu in all_mus:
        for s in seeds:
            for p in partitions:
                d, tag = m16.dir_tag(mu)
                hp = d / f"{tag}_history_seed{s}_{p}.csv"
                if not hp.exists():
                    missing.append((mu, s, p)); continue
                hist, cfg, conf = m16.load_run(mu, s, p)
                r = m16.metrics_row(mu, s, p, hist, conf)
                r["HD_skew"] = float(skew.loc[(s, p), "HD_skew"])
                r["JSD_skew"] = float(skew.loc[(s, p), "JSD_skew"])
                r["EMD_skew"] = float(skew.loc[(s, p), "EMD_skew"])
                rows[(mu, s, p)] = r
    if missing:
        print("MISSING runs:", missing)
    assert len(rows) == len(all_mus) * len(seeds) * len(partitions), f"expected {len(all_mus)*21}, got {len(rows)}"

    # Compute associations: mu x skew_metric x target.
    out = []
    for mu in all_mus:
        sub = pd.DataFrame([rows[(mu, s, p)] for s in seeds for p in partitions])
        for sk_label, sk_col in SKEW_METRICS.items():
            for tg_label, tg_col in TARGETS.items():
                rho_all, p_all = spearmanr(sub[sk_col], sub[tg_col])
                within = {}
                for s in seeds:
                    ss = sub[sub.seed == s]
                    within[s], _ = spearmanr(ss[sk_col], ss[tg_col])
                out.append({
                    "mu": mu, "skew_metric": sk_label, "target": tg_label,
                    "spearman_overall": rho_all, "p_overall": p_all,
                    "spearman_seed42": within[42], "spearman_seed43": within[43],
                    "spearman_seed44": within[44],
                    "mean_within_seed": float(np.nanmean(list(within.values()))),
                    "n_overall": len(sub), "note": "exploratory (pilot, n=21)",
                })
    assoc = pd.DataFrame(out)
    assoc.to_csv(OUT_CSV, index=False)

    # Verify existing artefacts unchanged (this script only reads + writes a NEW file).
    unchanged = "manifest not found"
    if MANIFEST.exists():
        mism = []
        for line in MANIFEST.read_text().splitlines():
            if not line.strip():
                continue
            digest, path = line.split(maxsplit=1)
            pth = Path(path)
            cur = hashlib.sha256(open(pth, "rb").read()).hexdigest() if pth.exists() else "MISSING"
            if cur != digest:
                mism.append(path)
        unchanged = "all match" if not mism else f"{len(mism)} CHANGED: {mism[:3]}"

    # Console tables.
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
    print("=== OVERALL Spearman(skew, target) by mu — target = macro_f1 (exploratory, n=21) ===")
    piv = assoc[assoc.target == "macro_f1"].pivot(index="mu", columns="skew_metric", values="spearman_overall")[["HD", "JSD", "EMD"]].round(3)
    print(piv.to_string())

    print("\n=== OVERALL Spearman by skew_metric x target, averaged over the 6 mu levels ===")
    avg = assoc.groupby(["skew_metric", "target"])["spearman_overall"].mean().unstack("target")
    avg = avg[["macro_f1", "macro_recall", "macro_pr_auc", "worst_class_f1", "Shellcode_f1", "Exploits_f1"]].reindex(["HD", "JSD", "EMD"]).round(3)
    print(avg.to_string())

    print("\n=== mean within-seed Spearman (skew, macro_f1) by mu ===")
    piv2 = assoc[assoc.target == "macro_f1"].pivot(index="mu", columns="skew_metric", values="mean_within_seed")[["HD", "JSD", "EMD"]].round(3)
    print(piv2.to_string())

    print(f"\nRows written: {len(assoc)} (6 mu x 3 skew x 6 targets)")
    print("All associations labelled:", assoc['note'].unique().tolist())
    print("Existing research artefacts unchanged:", unchanged)
    print("Written to", OUT_CSV)


if __name__ == "__main__":
    main()
