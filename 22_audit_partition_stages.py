"""
Read-only mathematical audit of 08b_create_controlled_partitions.py for the 18
non-IID controlled partitions (3 seeds x 6 alphas).

Reconstructs, per (seed, alpha), the full transformation
  raw Dirichlet target T  ->  IPF-adjusted F  ->  integer allocation A(final)
using 08b's OWN functions (imported), and quantifies each stage transition plus
feasibility bounds. No training, no test-data access. Writes one CSV.
"""

from pathlib import Path
from itertools import combinations
import importlib.util
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROCESSED_DIR = Path("data/processed")
PART_ROOT = Path("data/fl_clients/controlled_partitions")
LABEL_MAP = Path("configs/label_mapping.json")
CONTROLLED_METRICS = Path("results/fl_partitions/controlled_metrics.csv")
OUT_CSV = Path("results/fl_partitions/controlled_partition_stage_diagnostics.csv")

NUM_CLASSES = 10
NUM_CLIENTS = 5
CLIENT_SIZE = 50_000
POOL_SIZE = 250_000
SEEDS = [42, 43, 44]
ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def row_normalise(M):
    s = M.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return M / s


def hd(p, q):
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2.0))


def pairwise_mean_hd(M):
    P = row_normalise(M)
    vals = [hd(P[i], P[j]) for i, j in combinations(range(NUM_CLIENTS), 2)]
    return float(np.mean(vals)), float(np.max(vals))


def mean_client_hd(M1, M2):
    P1, P2 = row_normalise(M1), row_normalise(M2)
    return float(np.mean([hd(P1[k], P2[k]) for k in range(NUM_CLIENTS)]))


def l1_normalized(M1, M2):
    # L1 between the two matrices normalized to joint proportions (sum to 1)
    A = M1 / M1.sum(); B = M2 / M2.sum()
    return float(np.abs(A - B).sum())


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    m08b = load_module("m08b", "08b_create_controlled_partitions.py")

    y_train = np.load(PROCESSED_DIR / "y_train.npy")  # train labels only; no test read
    name_to_id = json.load(open(LABEL_MAP)); id_to_name = {int(v): k for k, v in name_to_id.items()}
    quota = m08b.build_quota_vector(y_train, id_to_name)          # recomputed via 08b
    saved_quota = pd.read_csv(PART_ROOT / "quota_vector.csv")["quota"].to_numpy()
    assert np.array_equal(quota, saved_quota), "quota mismatch vs saved"
    cm = pd.read_csv(CONTROLLED_METRICS).set_index(["seed", "partition_type"])

    row_m = np.full(NUM_CLIENTS, CLIENT_SIZE, dtype=np.float64)
    col_m = quota.astype(np.float64)
    row_mi = np.full(NUM_CLIENTS, CLIENT_SIZE, dtype=np.int64)
    col_mi = quota.astype(np.int64)

    rows = []
    for seed in SEEDS:
        for alpha in ALPHAS:
            adir = PART_ROOT / f"seed_{seed}" / f"alpha_{alpha}"
            T = np.load(adir / "theoretical_matrix.npy")           # raw Dirichlet target
            A_saved = np.load(adir / "realized_matrix.npy")        # final integer (saved)

            # Reconstruct IPF stage F from the SAVED T (deterministic, no RNG).
            F = m08b.ipf(T, row_m, col_m)
            ipf_resid = float(max(np.abs(F.sum(1) - row_m).max(), np.abs(F.sum(0) - col_m).max()))
            ipf_conv = ipf_resid < 1e-12

            # Reconstruct integer stage and confirm it equals the saved realized matrix.
            A_recon = m08b.round_preserving_margins(F.copy(), row_mi, col_mi)
            A_recon, n_swaps = m08b.repair_by_swaps(A_recon, row_mi, col_mi)
            recon_ok = bool(np.array_equal(A_recon, A_saved))

            # Confirm saved realized == achieved counts from client indices.
            idx = [np.load(adir / f"client_{k:02d}_indices.npy") for k in range(NUM_CLIENTS)]
            counts = np.stack([np.bincount(y_train[i], minlength=NUM_CLASSES) for i in idx])
            counts_ok = bool(np.array_equal(counts, A_saved))

            A = A_saved.astype(np.float64)
            # margin exactness
            row_exact = bool((A.sum(1) == CLIENT_SIZE).all())
            col_exact = bool(np.array_equal(A.sum(0).astype(int), quota))

            # stage-transition magnitudes (raw counts)
            max_TF = float(np.abs(T - F).max())
            max_FA = float(np.abs(F - A).max())
            max_TA = float(np.abs(T - A).max())
            # L1 on normalized allocation proportions
            l1_TF = l1_normalized(T, F); l1_FA = l1_normalized(F, A); l1_TA = l1_normalized(T, A)
            # mean per-client HD between stages
            hd_TF = mean_client_hd(T, F); hd_FA = mean_client_hd(F, A); hd_TA = mean_client_hd(T, A)
            # pairwise-mean HD at each stage
            pm_T, px_T = pairwise_mean_hd(T)
            pm_F, px_F = pairwise_mean_hd(F)
            pm_A, px_A = pairwise_mean_hd(A)

            # observed min Benign (class 0) per client
            benign_min = int(A[:, 0].min()); benign_min_prop = benign_min / CLIENT_SIZE

            rows.append({
                "seed": seed, "alpha": alpha,
                "ipf_converged": ipf_conv, "ipf_max_margin_resid": ipf_resid,
                "recon_matches_saved_realized": recon_ok, "realized_matches_indices": counts_ok,
                "row_totals_exact_50000": row_exact, "col_totals_exact_quota": col_exact,
                "n_swaps": int(n_swaps),
                "max_abs_T_F": max_TF, "max_abs_F_A": max_FA, "max_abs_T_A": max_TA,
                "L1norm_T_F": l1_TF, "L1norm_F_A": l1_FA, "L1norm_T_A": l1_TA,
                "meanHD_client_T_F": hd_TF, "meanHD_client_F_A": hd_FA, "meanHD_client_T_A": hd_TA,
                "hd_pairwise_mean_T": pm_T, "hd_pairwise_mean_F": pm_F, "hd_pairwise_mean_A": pm_A,
                "hd_pairwise_max_A": px_A,
                "benign_min_client_count_A": benign_min, "benign_min_client_prop_A": benign_min_prop,
                "alloc_error_L1_TA": float(np.abs(T - A).sum()),
                "stored_alloc_error_L1": float(cm.loc[(seed, f"alpha_{alpha}"), "alloc_error_L1"]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # ---------------- feasibility derivation ----------------
    non_benign_total = int(POOL_SIZE - quota[0])
    min_benign_count = CLIENT_SIZE - non_benign_total        # one client absorbs all non-Benign
    min_benign_prop = min_benign_count / CLIENT_SIZE
    # extreme pair: A=[min_benign Benign, rest non-Benign], B=[all Benign]
    pA = np.zeros(NUM_CLASSES); pA[0] = min_benign_count / CLIENT_SIZE; pA[1] = non_benign_total / CLIENT_SIZE
    pB = np.zeros(NUM_CLASSES); pB[0] = 1.0
    hd_bound = hd(pA, pB)
    observed_max_pairwise = float(df["hd_pairwise_max_A"].max())

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)

    print("========== (1) EXACT CODE BEHAVIOR ==========")
    print("Dirichlet draw (dirichlet_allocation, 08b:214-223):")
    print("  dir_seed = 10000*seed + int(round(alpha*100))  -> RNG = np.random.RandomState(dir_seed)")
    print("  for c in range(10):  proportions = rng.dirichlet([alpha]*5)  (SYMMETRIC alpha, length-5 vector)")
    print("                       theoretical[:, c] = proportions * quota[c]")
    print("  => sampled vector is length NUM_CLIENTS=5 (client split of ONE class); drawn PER CLASS (10 draws).")
    print("     Symmetric concentration [alpha,alpha,alpha,alpha,alpha]. One RNG stream per (seed,alpha).")
    print("IPF (08b:133-143):  f = target + 1e-15 ; repeat up to iters=5000:")
    print("  row scale:  f *= (row_margin / f.sum(axis=1))[:,None]")
    print("  col scale:  f *= (col_margin / f.sum(axis=0))[None,:]")
    print("  converge when max|rowsum-row_margin|<tol AND max|colsum-col_margin|<tol,  tol=1e-12")
    print("  -> standard alternating-scaling IPF/RAS; +1e-15 gives full support (NO structural zeros preserved).")
    print("round_preserving_margins (08b:147-181): a=floor(f); rdef=row-Σ_c a; cdef=col-Σ_k a;")
    print("  assert rdef>=0,cdef>=0, sum(rdef)==sum(cdef); greedily +1 at highest-frac cell with rdef&cdef>0.")

    print("\n========== (2) MATHEMATICAL CONSEQUENCES ==========")
    print("  IPF fixed point = KL I-projection of (T+1e-15) onto {row=50000, col=quota} (Csiszar 1975),")
    print("  valid here because margins are consistent (both sum to 250000) and the seed is strictly positive;")
    print("  claim is conditional on convergence to tol (verified per-row below). Not asserted beyond achieved tol.")
    print("  Integer stage preserves margins EXACTLY: floor(f) has rowsum<=50000 & colsum<=quota (tol), so")
    print("  rdef,cdef >=0 and sum(rdef)=5*50000-Sigma=250000-Sigma=Sigma(quota)-Sigma=sum(cdef). Each +1 at")
    print("  (k,c) decrements rdef[k],cdef[c] by 1, preserving sum(rdef)=sum(cdef); while cdef.sum()>0 some")
    print("  cdef[c]>0 and (sum equal) some rdef[k]>0 => cell (k,c) always available => no stall => terminates")
    print("  with rdef=cdef=0 => rowsum=50000 exactly, colsum=quota exactly. repair_by_swaps is an unused guard.")

    print("\n========== (3) EMPIRICAL VALUES (18 non-IID federations) ==========")
    checks = df[["ipf_converged", "recon_matches_saved_realized", "realized_matches_indices",
                 "row_totals_exact_50000", "col_totals_exact_quota", "n_swaps"]]
    print("  integrity: ipf_converged all =", bool(df.ipf_converged.all()),
          "| recon==saved all =", bool(df.recon_matches_saved_realized.all()),
          "| saved==indices all =", bool(df.realized_matches_indices.all()),
          "| row=50000 all =", bool(df.row_totals_exact_50000.all()),
          "| col=quota all =", bool(df.col_totals_exact_quota.all()),
          "| n_swaps max =", int(df.n_swaps.max()))
    show = df[["seed", "alpha", "max_abs_T_F", "max_abs_F_A", "max_abs_T_A",
               "L1norm_T_F", "L1norm_F_A", "L1norm_T_A",
               "meanHD_client_T_F", "meanHD_client_F_A", "meanHD_client_T_A",
               "hd_pairwise_mean_T", "hd_pairwise_mean_F", "hd_pairwise_mean_A", "hd_pairwise_max_A",
               "benign_min_client_prop_A"]].copy()
    for c in show.columns:
        if show[c].dtype == float: show[c] = show[c].round(6)
    print(show.to_string(index=False))

    print("\n  HD COMPRESSION per stage (pairwise-mean HD): T (raw target) -> F (IPF) -> A (final integer)")
    comp = df.groupby("alpha").agg(pmT=("hd_pairwise_mean_T", "mean"),
                                   pmF=("hd_pairwise_mean_F", "mean"),
                                   pmA=("hd_pairwise_mean_A", "mean")).reindex(ALPHAS)
    comp["compress_T_to_F"] = comp["pmT"] - comp["pmF"]
    comp["compress_F_to_A"] = comp["pmF"] - comp["pmA"]
    comp["compress_T_to_A"] = comp["pmT"] - comp["pmA"]
    print(comp.round(6).to_string())

    print("\n  MONOTONICITY: lower alpha -> higher final hd_pairwise_mean_A?  (per seed, alphas ascending)")
    reversals = []
    for seed in SEEDS:
        sub = df[df.seed == seed].sort_values("alpha")
        vals = sub["hd_pairwise_mean_A"].to_numpy()
        al = sub["alpha"].to_numpy()
        # expected strictly decreasing as alpha increases
        for i in range(len(al) - 1):
            if vals[i] <= vals[i + 1]:  # reversal (not higher at lower alpha)
                reversals.append((seed, al[i], round(vals[i], 6), al[i + 1], round(vals[i + 1], 6)))
    rho, p = spearmanr(df["alpha"], df["hd_pairwise_mean_A"])
    print(f"  overall Spearman(alpha, hd_pairwise_mean_A) = {rho:.4f} (p={p:.3g}); expected negative.")
    if reversals:
        print(f"  {len(reversals)} adjacent rank reversal(s) (alpha_i has NOT-higher HD than alpha_{{i+1}}):")
        for r in reversals:
            print(f"    seed {r[0]}: alpha {r[1]} HD={r[2]}  vs alpha {r[3]} HD={r[4]}")
    else:
        print("  no adjacent reversals: strictly monotone decreasing within every seed.")

    print("\n========== FEASIBILITY BOUNDS (5 clients, 50000 each, Benign quota=%d) ==========" % quota[0])
    print(f"  non-Benign total = 250000 - {quota[0]} = {non_benign_total}")
    print(f"  minimum possible Benign per client = 50000 - {non_benign_total} = {min_benign_count}"
          f"  (prop = {min_benign_prop:.6f})  [one client absorbs ALL non-Benign]")
    print(f"  observed minimum Benign proportion across all clients/federations = {df['benign_min_client_prop_A'].min():.6f}")
    print(f"  theoretical UPPER BOUND on pairwise HD (extreme pair: min-Benign client vs all-Benign client) = {hd_bound:.6f}")
    print(f"  observed MAX pairwise HD (final A, any federation) = {observed_max_pairwise:.6f}"
          f"  -> {100*observed_max_pairwise/hd_bound:.1f}% of the bound")

    print("\n========== (4) UNRESOLVED ASSUMPTIONS ==========")
    print("  - IPF KL-projection is asserted only up to the achieved tolerance (1e-12) and strict positivity")
    print("    from the +1e-15 floor; exact I-projection would require exact convergence (not claimed).")
    print("  - 'Intended HD' at the raw-target stage T uses row-normalized T rows; T row totals are NOT 50000,")
    print("    so T is a target *composition*, not a feasible allocation — HD(T) is indicative, not achievable.")
    print("  - The pairwise-HD upper bound is a per-pair maximum under the margins; it is an upper bound on any")
    print("    single pairwise HD, not on the pairwise-MEAN. Other clients set to all-Benign to satisfy margins.")
    print("  - Rare-class floor (Worms=30) and integer rounding introduce sub-unit perturbations not modelled here.")

    print("\nFiles created:")
    print("  script: 22_audit_partition_stages.py")
    print(f"  csv:    {OUT_CSV}")
    print("No training, no test-set access.")


if __name__ == "__main__":
    main()
