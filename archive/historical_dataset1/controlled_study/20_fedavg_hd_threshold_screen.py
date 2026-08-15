"""
Exploratory FedAvg-SGD (mu=0, lr=0.1) piecewise-linear threshold screen.

x = hd_pairwise_mean, y = best_val_macro_f1, with seed as a fixed blocking effect
(seed intercept dummies). Continuous piecewise-linear regressions:
  M0 = no breakpoint, M1 = one breakpoint, M2 = two ordered breakpoints.
Candidate breakpoints are midpoints strictly between observed HD values.

Read-only from results/fl_partitions/controlled_hd_performance_join_verified.csv.
No training, no test-data access, no alpha used as x-axis. Writes one CSV + one PNG.
This is exploratory: it does NOT force or claim that thresholds exist.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JOIN = Path("results/fl_partitions/controlled_hd_performance_join_verified.csv")
OUT_CSV = Path("results/fl_partitions/fedavg_pairwise_hd_macro_f1_threshold_screen.csv")
OUT_PNG = Path("results/fl_partitions/fedavg_pairwise_hd_macro_f1_threshold_screen.png")
SEEDS = [42, 43, 44]


# --------------------------- model machinery --------------------------- #
def seed_dummies(seed):
    # reference seed = 42
    return np.column_stack([(seed == 43).astype(float), (seed == 44).astype(float)])


def design(x, seed, breaks):
    cols = [np.ones_like(x), *seed_dummies(seed).T, x]
    for t in breaks:
        cols.append(np.maximum(x - t, 0.0))
    return np.column_stack(cols)


def fit(x, y, seed, breaks):
    X = design(x, seed, breaks)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    return beta, rss


def candidate_breaks(x):
    u = np.unique(x)
    return (u[:-1] + u[1:]) / 2.0


def segment_counts(x, breaks):
    edges = [-np.inf, *breaks, np.inf]
    return [int(((x > edges[i]) & (x < edges[i + 1])).sum()) for i in range(len(edges) - 1)]


def slopes_from_beta(beta, n_breaks):
    # beta layout: [intercept, s43, s44, b_x, b_hinge1, ...]
    base = beta[3]
    hinges = beta[4:4 + n_breaks]
    seg = [base]
    acc = base
    for h in hinges:
        acc += h
        seg.append(acc)
    return seg


def metrics(n, rss, n_mean_params, n_breaks, tss):
    k = n_mean_params + n_breaks + 1          # + variance
    p_adj = n_mean_params + n_breaks           # mean params incl. breakpoints
    aic = n * np.log(rss / n) + 2 * k
    aicc = aic + (2 * k * (k + 1) / (n - k - 1)) if (n - k - 1) > 0 else np.nan
    bic = n * np.log(rss / n) + k * np.log(n)
    r2 = 1 - rss / tss
    adj_r2 = 1 - (rss / (n - p_adj)) / (tss / (n - 1)) if (n - p_adj) > 0 else np.nan
    return k, aic, aicc, bic, r2, adj_r2


def search_m1(x, y, seed, min_side):
    best = None
    for t in candidate_breaks(x):
        if min(segment_counts(x, [t])) < min_side:
            continue
        beta, rss = fit(x, y, seed, [t])
        if best is None or rss < best[1]:
            best = (t, rss, beta)
    return best  # (t1, rss, beta) or None


def search_m2(x, y, seed, min_seg):
    cands = candidate_breaks(x)
    best = None
    for t1, t2 in combinations(cands, 2):
        if t1 >= t2:
            continue
        if min(segment_counts(x, [t1, t2])) < min_seg:
            continue
        beta, rss = fit(x, y, seed, [t1, t2])
        if best is None or rss < best[2]:
            best = (t1, t2, rss, beta)
    return best  # (t1,t2,rss,beta) or None


def predict_avg_seed(beta, xg, breaks):
    # average seed effect: intercept + mean(0, b_s43, b_s44)
    seff = np.mean([0.0, beta[1], beta[2]])
    y = beta[0] + seff + beta[3] * xg
    for h, t in zip(beta[4:4 + len(breaks)], breaks):
        y = y + h * np.maximum(xg - t, 0.0)
    return y


def main():
    df = pd.read_csv(JOIN)
    fed = df[(df.algorithm == "FedAvg-SGD") & (df.mu == 0.0)].copy()
    assert len(fed) == 21, f"expected 21 FedAvg rows, got {len(fed)}"

    x = fed["hd_pairwise_mean"].to_numpy(float)
    y = fed["best_val_macro_f1"].to_numpy(float)
    seed = fed["seed"].to_numpy(int)
    n = len(x)
    tss = float(((y - y.mean()) ** 2).sum())

    print("=== 21-ROW FedAvg-SGD (mu=0) DATA sorted by hd_pairwise_mean ===")
    show = fed[["seed", "partition", "hd_pairwise_mean", "best_val_macro_f1"]].copy()
    show = show.sort_values("hd_pairwise_mean").reset_index(drop=True)
    show["hd_pairwise_mean"] = show["hd_pairwise_mean"].round(6)
    show["best_val_macro_f1"] = show["best_val_macro_f1"].round(6)
    print(show.to_string(index=False))

    print("\n=== SPEARMAN (hd_pairwise_mean vs best_val_macro_f1) ===")
    rho_all, p_all = spearmanr(x, y)
    print(f"  overall (n=21): rho={rho_all:.4f}  p={p_all:.4g}")
    for s in SEEDS:
        m = seed == s
        rs, ps = spearmanr(x[m], y[m])
        print(f"  seed {s} (n={m.sum()}): rho={rs:.4f}  p={ps:.4g}")

    # ---- full-data models ----
    beta0, rss0 = fit(x, y, seed, [])
    k0, aic0, aicc0, bic0, r2_0, adj0 = metrics(n, rss0, 4, 0, tss)
    slopes0 = slopes_from_beta(beta0, 0)

    m1 = search_m1(x, y, seed, min_side=5)
    m2 = search_m2(x, y, seed, min_seg=5)

    rows = []
    rows.append({"analysis": "full", "model": "M0_no_break", "rss": rss0, "k_params": k0,
                 "aic": aic0, "aicc": aicc0, "bic": bic0, "r2": r2_0, "adj_r2": adj0,
                 "break1": np.nan, "break2": np.nan,
                 "slope_seg1": slopes0[0], "slope_seg2": np.nan, "slope_seg3": np.nan})

    if m1:
        t1, rss1, b1 = m1
        k1, aic1, aicc1, bic1, r2_1, adj1 = metrics(n, rss1, 5, 1, tss)
        s1 = slopes_from_beta(b1, 1)
        rows.append({"analysis": "full", "model": "M1_one_break", "rss": rss1, "k_params": k1,
                     "aic": aic1, "aicc": aicc1, "bic": bic1, "r2": r2_1, "adj_r2": adj1,
                     "break1": t1, "break2": np.nan,
                     "slope_seg1": s1[0], "slope_seg2": s1[1], "slope_seg3": np.nan})
    if m2:
        t1b, t2b, rss2, b2 = m2
        k2, aic2, aicc2, bic2, r2_2, adj2 = metrics(n, rss2, 6, 2, tss)
        s2 = slopes_from_beta(b2, 2)
        rows.append({"analysis": "full", "model": "M2_two_breaks", "rss": rss2, "k_params": k2,
                     "aic": aic2, "aicc": aicc2, "bic": bic2, "r2": r2_2, "adj_r2": adj2,
                     "break1": t1b, "break2": t2b,
                     "slope_seg1": s2[0], "slope_seg2": s2[1], "slope_seg3": s2[2]})

    # ---- leave-one-seed-out sensitivity (>=4 per segment) ----
    loso = {}
    for excl in SEEDS:
        mask = seed != excl
        xs, ys, ss = x[mask], y[mask], seed[mask]
        r1 = search_m1(xs, ys, ss, min_side=4)
        r2 = search_m2(xs, ys, ss, min_seg=4)
        loso[excl] = (r1, r2)
        if r1:
            rows.append({"analysis": f"loso_excl_{excl}", "model": "M1_one_break",
                         "rss": r1[1], "k_params": np.nan, "aic": np.nan, "aicc": np.nan,
                         "bic": np.nan, "r2": np.nan, "adj_r2": np.nan,
                         "break1": r1[0], "break2": np.nan,
                         "slope_seg1": slopes_from_beta(r1[2], 1)[0],
                         "slope_seg2": slopes_from_beta(r1[2], 1)[1], "slope_seg3": np.nan})
        if r2:
            s2 = slopes_from_beta(r2[3], 2)
            rows.append({"analysis": f"loso_excl_{excl}", "model": "M2_two_breaks",
                         "rss": r2[2], "k_params": np.nan, "aic": np.nan, "aicc": np.nan,
                         "bic": np.nan, "r2": np.nan, "adj_r2": np.nan,
                         "break1": r2[0], "break2": r2[1],
                         "slope_seg1": s2[0], "slope_seg2": s2[1], "slope_seg3": s2[2]})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    # ---- console: model comparison ----
    print("\n=== MODEL COMPARISON (full 21-row data; seed as fixed blocking effect) ===")
    comp = out[out.analysis == "full"][["model", "rss", "k_params", "aicc", "bic", "adj_r2",
                                        "break1", "break2", "slope_seg1", "slope_seg2", "slope_seg3"]].copy()
    for c in ["rss", "aicc", "bic", "adj_r2"]:
        comp[c] = comp[c].round(4)
    for c in ["break1", "break2", "slope_seg1", "slope_seg2", "slope_seg3"]:
        comp[c] = comp[c].round(4)
    print(comp.to_string(index=False))

    aicc_by = {r["model"]: r["aicc"] for _, r in out[out.analysis == "full"].iterrows()}
    bic_by = {r["model"]: r["bic"] for _, r in out[out.analysis == "full"].iterrows()}
    print("\n  delta AICc vs M0:  "
          f"M1={aicc_by.get('M0_no_break')-aicc_by.get('M1_one_break',np.nan):+.3f}  "
          f"M2={aicc_by.get('M0_no_break')-aicc_by.get('M2_two_breaks',np.nan):+.3f}   "
          "(positive = better than M0)")
    print("  delta BIC vs M0:   "
          f"M1={bic_by.get('M0_no_break')-bic_by.get('M1_one_break',np.nan):+.3f}  "
          f"M2={bic_by.get('M0_no_break')-bic_by.get('M2_two_breaks',np.nan):+.3f}")

    # ---- console: LOSO breakpoint stability ----
    print("\n=== LEAVE-ONE-SEED-OUT BREAKPOINT SENSITIVITY (>=4 per segment) ===")
    m1_bps = []
    if m1:
        m1_bps.append(("full", m1[0]))
    for excl in SEEDS:
        r1 = loso[excl][0]
        r2 = loso[excl][1]
        b1 = f"{r1[0]:.4f}" if r1 else "none"
        b2 = f"({r2[0]:.4f},{r2[1]:.4f})" if r2 else "none"
        if r1:
            m1_bps.append((f"excl_{excl}", r1[0]))
        print(f"  exclude seed {excl}: M1 break1={b1}   M2 breaks={b2}")
    if m1:
        print(f"  full-data       : M1 break1={m1[0]:.4f}   "
              f"M2 breaks=({m2[0]:.4f},{m2[1]:.4f})" if m2 else f"  full: M1 break1={m1[0]:.4f}")
    if len(m1_bps) >= 2:
        vals = [v for _, v in m1_bps]
        rng = max(vals) - min(vals)
        xr = x.max() - x.min()
        print(f"\n  M1 breakpoint locations across full+LOSO: {[round(v,4) for v in vals]}")
        print(f"  spread (max-min) = {rng:.4f}  = {100*rng/xr:.1f}% of observed HD range ({xr:.4f})")

    # ---- verdict (cautious, evidence-based) ----
    best_aicc_model = min(aicc_by, key=lambda m: aicc_by[m])
    best_bic_model = min(bic_by, key=lambda m: bic_by[m])
    print("\n=== EVIDENCE SUMMARY (exploratory; not a claim that thresholds exist) ===")
    print(f"  lowest AICc: {best_aicc_model} ; lowest BIC: {best_bic_model}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {42: "#1f77b4", 43: "#ff7f0e", 44: "#2ca02c"}
    for s in SEEDS:
        m = seed == s
        ax.scatter(x[m], y[m], c=colors[s], label=f"seed {s}", s=55, edgecolor="k", linewidth=0.4, zorder=3)
    xg = np.linspace(x.min(), x.max(), 400)
    ax.plot(xg, predict_avg_seed(beta0, xg, []), "k--", lw=1.4, label="M0 linear", zorder=2)
    if m1:
        ax.plot(xg, predict_avg_seed(m1[2], xg, [m1[0]]), color="crimson", lw=1.8, label="M1 one-break", zorder=2)
        ax.axvline(m1[0], color="crimson", ls=":", lw=1.0, alpha=0.7)
    if m2:
        ax.plot(xg, predict_avg_seed(m2[3], xg, [m2[0], m2[1]]), color="purple", lw=1.4, ls="-.", label="M2 two-break", zorder=2)
        ax.axvline(m2[0], color="purple", ls=":", lw=0.8, alpha=0.5)
        ax.axvline(m2[1], color="purple", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("hd_pairwise_mean (client-to-client HD, arithmetic mean)")
    ax.set_ylabel("best validation macro-F1")
    ax.set_title("FedAvg-SGD (mu=0): exploratory piecewise-linear threshold screen\n(seed as fixed blocking effect; fit shown at average seed effect)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)

    print("\nOnly VALIDATION results were used (best_val_macro_f1 from saved histories); "
          "no training, no test-set access, no correlations beyond the requested Spearman, no thresholds forced.")
    print(f"Written: {OUT_CSV}")
    print(f"Written: {OUT_PNG}")


if __name__ == "__main__":
    main()
