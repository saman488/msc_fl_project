"""
Controlled non-IID partition design for a pilot study.

Goal: isolate the effect of *label skew* by holding the pooled class distribution
and the client sizes strictly constant across all alphas and seeds. Only the
per-client label composition is allowed to vary.

Design
------
* A single fixed 10-class integer quota vector Q (sum = N_pool = 250,000) mirrors
  the global training distribution, with a floor of 30 for the rare classes.
* For each seed, a pool D_pool is sampled from y_train to fill Q exactly (different
  records per seed where inventory allows; identical per-class counts).
* For each alpha, D_pool is split into 5 clients of exactly 50,000 rows each, with
  per-class column sums equal to Q, following constrained Dirichlet label skew.
* A perfectly stratified IID baseline is built per seed (no random shuffle-split).

The constrained allocation guarantees BOTH margins (rows = 50,000, cols = Q) by:
  1) drawing theoretical Dirichlet proportions pi_c and target T[k,c] = pi_c[k]*Q_c,
  2) IPF-scaling T so it respects the 50,000 row cap and the Q column totals,
  3) margin-preserving integer rounding (floor + deficit/remainder repair),
  4) a swap-based feasibility guard (evict a majority-class unit from a full client)
     that activates only if a residual infeasibility were ever detected.

EMD is treated as an exploratory metric only. No training is prepared or run.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import ot
from scipy.spatial.distance import euclidean, jensenshannon


PROCESSED_DIR = Path("data/processed")
CONFIGS_DIR = Path("configs")
OUT_ROOT = Path("data/fl_clients/controlled_partitions")
RESULTS_DIR = Path("results/fl_partitions")
METRICS_CSV = RESULTS_DIR / "controlled_metrics.csv"

NUM_CLASSES = 10
NUM_CLIENTS = 5
N_POOL = 250_000
CLIENT_SIZE = 50_000            # N_POOL / NUM_CLIENTS
RARE_FLOOR = 30
SEEDS = [42, 43, 44]
ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

# Rare classes (by name) that receive the minimum floor.
RARE_NAMES = ["Analysis", "Backdoor", "Shellcode", "Worms"]

# Attack-severity risk scores (same convention as 09_partition_metrics.py).
RISK_BY_NAME = {
    "Benign": 0,
    "Reconnaissance": 1, "Analysis": 1,
    "Generic": 2, "Fuzzers": 2,
    "DoS": 3,
    "Backdoor": 4,
    "Exploits": 5, "Shellcode": 5, "Worms": 5,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_label_mapping() -> dict[int, str]:
    with open(CONFIGS_DIR / "label_mapping.json") as file:
        name_to_id = json.load(file)
    return {int(v): k for k, v in name_to_id.items()}


def normalise(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts.astype(np.float64) / total if total > 0 else counts.astype(np.float64)


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2.0))


def build_cost_matrix(id_to_name: dict[int, str]) -> np.ndarray:
    risk = np.array(
        [RISK_BY_NAME[id_to_name[c]] for c in range(NUM_CLASSES)], dtype=np.float64
    )
    return np.abs(risk[:, None] - risk[None, :])


# --------------------------------------------------------------------------- #
# 1. Fixed quota vector
# --------------------------------------------------------------------------- #
def build_quota_vector(y_train: np.ndarray, id_to_name: dict[int, str]) -> np.ndarray:
    """Fixed 10-class integer quota summing to N_POOL, mirroring P_train with a
    floor of RARE_FLOOR on rare classes. Benign absorbs the rounding/floor slack."""
    inventory = np.bincount(y_train, minlength=NUM_CLASSES)
    p_train = normalise(inventory)

    rare_ids = {cid for cid, name in id_to_name.items() if name in RARE_NAMES}

    quota = np.round(p_train * N_POOL).astype(np.int64)
    for cid in rare_ids:
        quota[cid] = max(int(quota[cid]), RARE_FLOOR)

    # Benign (class 0) absorbs the difference so the total is exactly N_POOL.
    quota[0] += N_POOL - int(quota.sum())

    # Validity checks.
    assert quota.sum() == N_POOL, quota.sum()
    assert (quota >= 0).all()
    for cid in rare_ids:
        assert quota[cid] >= RARE_FLOOR, (cid, quota[cid])
    assert (quota <= inventory).all(), "Quota exceeds available inventory."
    return quota


# --------------------------------------------------------------------------- #
# 2. Seed-based pool generation
# --------------------------------------------------------------------------- #
def build_pool(y_train: np.ndarray, quota: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    """Sample exactly quota[c] indices of each class c (without replacement)."""
    rng = np.random.RandomState(seed)
    pool_by_class = {}
    for c in range(NUM_CLASSES):
        class_indices = np.where(y_train == c)[0]
        chosen = rng.choice(class_indices, size=int(quota[c]), replace=False)
        pool_by_class[c] = np.sort(chosen)
    return pool_by_class


# --------------------------------------------------------------------------- #
# 3. Constrained Dirichlet allocation (integer matrix with fixed margins)
# --------------------------------------------------------------------------- #
def ipf(target: np.ndarray, row_margin: np.ndarray, col_margin: np.ndarray,
        iters: int = 5000, tol: float = 1e-12) -> np.ndarray:
    """Iterative proportional fitting: scale `target` to hit both margins."""
    f = target.astype(np.float64) + 1e-15
    for _ in range(iters):
        f *= (row_margin / f.sum(axis=1))[:, None]
        f *= (col_margin / f.sum(axis=0))[None, :]
        if (np.abs(f.sum(axis=1) - row_margin).max() < tol
                and np.abs(f.sum(axis=0) - col_margin).max() < tol):
            break
    return f


def round_preserving_margins(f: np.ndarray, row_margin: np.ndarray,
                             col_margin: np.ndarray) -> np.ndarray:
    """Integer-round `f` so row sums == row_margin and col sums == col_margin.
    Implements the 'floor + distribute remainders under capacity' step."""
    a = np.floor(f).astype(np.int64)
    frac = f - a
    rdef = row_margin - a.sum(axis=1)   # per-client capacity still to fill
    cdef = col_margin - a.sum(axis=0)   # per-class quota still to place

    assert (rdef >= 0).all() and (cdef >= 0).all(), (rdef, cdef)
    assert rdef.sum() == cdef.sum()

    # Greedily add one unit at a time to the highest-fraction cell whose row and
    # column both still need capacity. Respects the 50,000 cap (rdef never < 0).
    while cdef.sum() > 0:
        candidates = [
            (frac[k, c], k, c)
            for k in range(NUM_CLIENTS)
            for c in range(NUM_CLASSES)
            if rdef[k] > 0 and cdef[c] > 0
        ]
        if not candidates:
            break
        candidates.sort(reverse=True)
        progressed = False
        for _, k, c in candidates:
            if rdef[k] > 0 and cdef[c] > 0:
                a[k, c] += 1
                rdef[k] -= 1
                cdef[c] -= 1
                progressed = True
        if not progressed:
            break

    return a


def repair_by_swaps(a: np.ndarray, row_margin: np.ndarray,
                    col_margin: np.ndarray) -> tuple[np.ndarray, int]:
    """Feasibility guard: if any row still exceeds/undershoots its margin (a full
    client blocking a rare class), evict a majority-class unit from an over-full
    client into an under-full one, keeping column sums intact. Returns (a, n_swaps)."""
    swaps = 0
    for _ in range(10 * NUM_CLIENTS):
        row_err = a.sum(axis=1) - row_margin
        if not row_err.any():
            break
        over = int(np.argmax(row_err))     # client above capacity
        under = int(np.argmin(row_err))    # client below capacity
        # Find a class present in `over` that `under` can receive (col sum fixed).
        moved = False
        for c in np.argsort(-a[over]):     # majority classes first
            if a[over, c] > 0:
                a[over, c] -= 1
                a[under, c] += 1
                swaps += 1
                moved = True
                break
        if not moved:
            break
    return a, swaps


def dirichlet_allocation(quota: np.ndarray, alpha: float, seed: int):
    """Return (theoretical T[5,10], realized integer A[5,10], n_swaps)."""
    dir_seed = 10_000 * seed + int(round(alpha * 100))
    rng = np.random.RandomState(dir_seed)

    theoretical = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=np.float64)
    for c in range(NUM_CLASSES):
        proportions = rng.dirichlet([alpha] * NUM_CLIENTS)
        theoretical[:, c] = proportions * quota[c]

    row_margin_int = np.full(NUM_CLIENTS, CLIENT_SIZE, dtype=np.int64)
    col_margin_int = quota.astype(np.int64)

    feasible = ipf(theoretical, row_margin_int.astype(np.float64), col_margin_int.astype(np.float64))
    realized = round_preserving_margins(feasible, row_margin_int, col_margin_int)
    realized, n_swaps = repair_by_swaps(realized, row_margin_int, col_margin_int)

    assert (realized.sum(axis=1) == CLIENT_SIZE).all(), realized.sum(axis=1)
    assert (realized.sum(axis=0) == quota).all(), realized.sum(axis=0)
    return theoretical, realized, n_swaps


# --------------------------------------------------------------------------- #
# 4. Stratified IID baseline (deterministic, no random shuffle-split)
# --------------------------------------------------------------------------- #
def stratified_iid_allocation(quota: np.ndarray) -> np.ndarray:
    """Even per-class split (integer division) + global round-robin remainders.
    Guarantees exactly CLIENT_SIZE rows per client (remainder total is a
    multiple of NUM_CLIENTS by construction)."""
    a = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=np.int64)
    pointer = 0
    for c in range(NUM_CLASSES):
        base = int(quota[c]) // NUM_CLIENTS
        rem = int(quota[c]) % NUM_CLIENTS
        a[:, c] = base
        for _ in range(rem):
            a[pointer, c] += 1
            pointer = (pointer + 1) % NUM_CLIENTS

    assert (a.sum(axis=0) == quota).all()
    assert (a.sum(axis=1) == CLIENT_SIZE).all(), a.sum(axis=1)
    return a


# --------------------------------------------------------------------------- #
# Assign concrete indices from a per-class count matrix
# --------------------------------------------------------------------------- #
def counts_to_client_indices(alloc: np.ndarray, pool_by_class: dict[int, np.ndarray]) -> list[np.ndarray]:
    """Slice each class's pooled indices to clients according to `alloc`."""
    client_indices = [[] for _ in range(NUM_CLIENTS)]
    for c in range(NUM_CLASSES):
        start = 0
        column = pool_by_class[c]
        assert int(alloc[:, c].sum()) == len(column)
        for k in range(NUM_CLIENTS):
            take = int(alloc[k, c])
            client_indices[k].append(column[start:start + take])
            start += take
    return [np.sort(np.concatenate(parts)).astype(np.int64) for parts in client_indices]


# --------------------------------------------------------------------------- #
# Metrics for one configuration
# --------------------------------------------------------------------------- #
def config_metrics(alloc: np.ndarray, fed_dist: np.ndarray, cost_matrix: np.ndarray):
    hd, jsd, emd = [], [], []
    for k in range(NUM_CLIENTS):
        p = normalise(alloc[k].astype(np.float64))
        hd.append(hellinger(p, fed_dist))
        jsd.append(float(jensenshannon(p, fed_dist)))
        emd.append(float(ot.emd2(p, fed_dist, cost_matrix)))
    return float(np.mean(hd)), float(np.mean(jsd)), float(np.mean(emd))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    id_to_name = load_label_mapping()
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    cost_matrix = build_cost_matrix(id_to_name)

    # 1. Fixed quota vector (identical across all seeds and alphas).
    quota = build_quota_vector(y_train, id_to_name)
    fed_dist = normalise(quota)                       # pooled distribution (held constant)
    p_train = normalise(np.bincount(y_train, minlength=NUM_CLASSES))
    hd_distortion = hellinger(fed_dist, p_train)      # pool-vs-global distortion

    print("Fixed quota vector Q (sum = {}):".format(int(quota.sum())))
    for c in range(NUM_CLASSES):
        print(f"  {c} {id_to_name[c]:14} quota={int(quota[c]):>7}  "
              f"pool_frac={fed_dist[c]:.5f}  train_frac={p_train[c]:.5f}")
    print(f"HD_distortion (pool vs global train) = {hd_distortion:.6f}\n")

    # Persist the quota vector.
    pd.DataFrame({"class_id": range(NUM_CLASSES),
                  "class_name": [id_to_name[c] for c in range(NUM_CLASSES)],
                  "quota": quota}).to_csv(OUT_ROOT / "quota_vector.csv", index=False)

    rare_ids = [cid for cid, name in id_to_name.items() if name in RARE_NAMES]
    metric_rows = []
    pools = {}

    for seed in SEEDS:
        pool_by_class = build_pool(y_train, quota, seed)
        pools[seed] = pool_by_class
        d_pool = np.sort(np.concatenate([pool_by_class[c] for c in range(NUM_CLASSES)])).astype(np.int64)

        # Integrity of the pool.
        assert len(d_pool) == N_POOL and len(np.unique(d_pool)) == N_POOL
        pool_counts = np.bincount(y_train[d_pool], minlength=NUM_CLASSES)
        assert (pool_counts == quota).all(), "Pool class counts != quota."

        seed_dir = OUT_ROOT / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(seed_dir / "d_pool_indices.npy", d_pool)

        # ---- constrained Dirichlet configs ----
        for alpha in ALPHAS:
            theoretical, realized, n_swaps = dirichlet_allocation(quota, alpha, seed)
            client_indices = counts_to_client_indices(realized, pool_by_class)

            # Partition integrity.
            allidx = np.concatenate(client_indices)
            assert len(allidx) == N_POOL and len(np.unique(allidx)) == N_POOL
            assert set(allidx.tolist()).issubset(set(d_pool.tolist()))
            for k in range(NUM_CLIENTS):
                assert len(client_indices[k]) == CLIENT_SIZE

            cfg_dir = seed_dir / f"alpha_{alpha}"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            for k in range(NUM_CLIENTS):
                np.save(cfg_dir / f"client_{k:02d}_indices.npy", client_indices[k])
            np.save(cfg_dir / "theoretical_matrix.npy", theoretical)
            np.save(cfg_dir / "realized_matrix.npy", realized)
            pd.DataFrame(theoretical, columns=[id_to_name[c] for c in range(NUM_CLASSES)]).to_csv(
                cfg_dir / "theoretical_matrix.csv", index_label="client")
            pd.DataFrame(realized, columns=[id_to_name[c] for c in range(NUM_CLASSES)]).to_csv(
                cfg_dir / "realized_matrix.csv", index_label="client")

            hd_s, jsd_s, emd_s = config_metrics(realized, fed_dist, cost_matrix)
            alloc_error = float(np.abs(theoretical - realized).sum())

            row = {
                "seed": seed, "partition_type": f"alpha_{alpha}",
                "HD_skew": hd_s, "JSD_skew": jsd_s, "EMD_skew": emd_s,
                "HD_distortion": hd_distortion, "alloc_error_L1": alloc_error,
                "n_swaps": n_swaps,
            }
            for cid in rare_ids:
                row[f"pooled_{id_to_name[cid]}"] = int(realized[:, cid].sum())
            row["pooled_rare_total"] = int(sum(realized[:, cid].sum() for cid in rare_ids))
            for k in range(NUM_CLIENTS):
                for c in range(NUM_CLASSES):
                    row[f"n_c{k}_cls{c}"] = int(realized[k, c])
            metric_rows.append(row)

        # ---- stratified IID baseline ----
        iid_alloc = stratified_iid_allocation(quota)
        iid_indices = counts_to_client_indices(iid_alloc, pool_by_class)
        allidx = np.concatenate(iid_indices)
        assert len(allidx) == N_POOL and len(np.unique(allidx)) == N_POOL

        iid_dir = seed_dir / "iid"
        iid_dir.mkdir(parents=True, exist_ok=True)
        for k in range(NUM_CLIENTS):
            np.save(iid_dir / f"client_{k:02d}_indices.npy", iid_indices[k])
        np.save(iid_dir / "realized_matrix.npy", iid_alloc)
        pd.DataFrame(iid_alloc, columns=[id_to_name[c] for c in range(NUM_CLASSES)]).to_csv(
            iid_dir / "realized_matrix.csv", index_label="client")

        hd_s, jsd_s, emd_s = config_metrics(iid_alloc, fed_dist, cost_matrix)
        row = {
            "seed": seed, "partition_type": "iid",
            "HD_skew": hd_s, "JSD_skew": jsd_s, "EMD_skew": emd_s,
            "HD_distortion": hd_distortion, "alloc_error_L1": 0.0, "n_swaps": 0,
        }
        for cid in rare_ids:
            row[f"pooled_{id_to_name[cid]}"] = int(iid_alloc[:, cid].sum())
        row["pooled_rare_total"] = int(sum(iid_alloc[:, cid].sum() for cid in rare_ids))
        for k in range(NUM_CLIENTS):
            for c in range(NUM_CLASSES):
                row[f"n_c{k}_cls{c}"] = int(iid_alloc[k, c])
        metric_rows.append(row)

    # Verify seeds selected different records where inventory allowed.
    diff_report = []
    for c in range(NUM_CLASSES):
        s42 = set(pools[42][c].tolist())
        s43 = set(pools[43][c].tolist())
        inventory_c = int((y_train == c).sum())
        differs = (s42 != s43) if inventory_c > int(quota[c]) else None
        diff_report.append((id_to_name[c], inventory_c, int(quota[c]), differs))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    # ---- summary ----
    print("Seed selection differs across seeds (where inventory > quota):")
    for name, inv, q, differs in diff_report:
        tag = "n/a (no slack)" if differs is None else ("DIFFERS" if differs else "IDENTICAL")
        print(f"  {name:14} inventory={inv:>9} quota={q:>7} -> {tag}")

    print("\n=== CONTROLLED PARTITION SUMMARY ===")
    show = metrics_df[["seed", "partition_type", "HD_skew", "JSD_skew", "EMD_skew",
                       "pooled_Analysis", "pooled_Backdoor", "pooled_Shellcode",
                       "pooled_Worms", "pooled_rare_total"]].copy()
    for col in ["HD_skew", "JSD_skew", "EMD_skew"]:
        show[col] = show[col].round(6)
    print(show.to_string(index=False))
    print(f"\nHD_distortion (constant, pool vs global) = {hd_distortion:.6f}")
    print(f"Metrics written to {METRICS_CSV}")


if __name__ == "__main__":
    main()
