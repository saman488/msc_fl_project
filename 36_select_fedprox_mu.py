"""
Record the Dataset-1 37-feature FedProx mu selection as a repository artefact.

The selection rule: six candidate mu values; score each by the equal-weight mean
of run-level best_val_macro_f1 across all 12 runs (4 conditions x 3 seeds);
highest mean wins; ties broken toward the smaller mu; validation data only, no
test data involved at any stage.

Reads the six existing final_summary_mu*.csv files and writes mu_selection.csv
beside them. Reads no checkpoint, no training data and no test data.
"""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results/final_fedprox_k5_37f")
OUT_PATH = RESULTS_DIR / "mu_selection.csv"

EXPECTED_MU_COUNT = 6
EXPECTED_RUNS = 12

# The value recorded as MU_SELECTED_SCORE_DATASET1 in d2_05_train_fedprox.py. It was
# written to 15 significant figures, so it is not bit-identical to the recomputed
# mean (they differ by about 1.5 ULP); compare within a tolerance rather than by ==.
RECORDED_D1_SCORE = 0.366411072723754
SCORE_TOL = 1e-12


def main() -> None:
    if OUT_PATH.exists():
        raise RuntimeError(f"Refusing to overwrite {OUT_PATH}; remove or archive it first.")

    files = sorted(RESULTS_DIR.glob("final_summary_mu*.csv"))
    assert len(files) == EXPECTED_MU_COUNT, \
        f"expected {EXPECTED_MU_COUNT} mu summaries, found {len(files)}"

    rows = []
    for path in files:
        summary = pd.read_csv(path)
        # One file per mu: a mixed file would silently average across candidates.
        assert summary["mu"].nunique() == 1, f"{path}: contains more than one mu value"
        assert len(summary) == EXPECTED_RUNS, \
            f"{path}: expected {EXPECTED_RUNS} runs, found {len(summary)}"

        scores = summary["best_val_macro_f1"]
        rows.append({
            "mu": float(summary["mu"].iloc[0]),
            "n_runs": int(len(summary)),
            "mean_val_macro_f1": float(scores.mean()),
            "std_val_macro_f1": float(scores.std(ddof=1)),
            "min_val_macro_f1": float(scores.min()),
            "max_val_macro_f1": float(scores.max()),
        })

    rows.sort(key=lambda r: r["mu"])
    assert len({r["mu"] for r in rows}) == EXPECTED_MU_COUNT, "duplicate mu across summaries"

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    # Highest mean wins; on an exact tie the smaller mu wins, so the weaker proximal
    # term is preferred when validation cannot separate two candidates.
    ranked = sorted(rows, key=lambda r: (-r["mean_val_macro_f1"], r["mu"]))
    winner, runner_up = ranked[0], ranked[1]
    margin = winner["mean_val_macro_f1"] - runner_up["mean_val_macro_f1"]

    assert abs(winner["mean_val_macro_f1"] - RECORDED_D1_SCORE) < SCORE_TOL, (
        f"selected mean {winner['mean_val_macro_f1']!r} does not reproduce the recorded "
        f"Dataset-1 score {RECORDED_D1_SCORE!r}"
    )

    print(f"{'mu':>10} {'n':>4} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'range':>10}")
    for row in rows:
        spread = row["max_val_macro_f1"] - row["min_val_macro_f1"]
        print(f"{row['mu']:>10.5g} {row['n_runs']:>4} {row['mean_val_macro_f1']:>10.6f} "
              f"{row['std_val_macro_f1']:>10.6f} {row['min_val_macro_f1']:>10.6f} "
              f"{row['max_val_macro_f1']:>10.6f} {spread:>10.6f}")

    print(f"\nselected mu: {winner['mu']:g}")
    print(f"selected mean val macro-F1: {winner['mean_val_macro_f1']!r}")
    print(f"recorded in d2_05_train_fedprox.py: {RECORDED_D1_SCORE!r} "
          f"(differs by {abs(winner['mean_val_macro_f1'] - RECORDED_D1_SCORE):.3g})")
    print(f"margin over next best (mu={runner_up['mu']:g}): {margin:.6f}")
    print(f"selected mu min-max range: "
          f"{winner['min_val_macro_f1']:.6f} to {winner['max_val_macro_f1']:.6f}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
