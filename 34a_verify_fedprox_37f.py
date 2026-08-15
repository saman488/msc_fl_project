"""
Verification of the 37-feature FedProx implementation in 34_train_final_fedprox_37f.py.

Three tests:
  (1) Gradient test  - the proximal gradient equals mu * (w - w_ref).
  (2) Reference test - the captured reference gives a zero proximal norm, a
                       positive one after a synthetic parameter change, and is
                       itself bitwise unchanged by that modification.
  (3) mu=0 equivalence - FedProx 37f (34) at mu=0 must reproduce FedAvg 37f (32)
                       under an identical initial state, partitions, class weights,
                       optimiser, seeds and DataLoader order.

Test 3 requires EXACT equivalence, not agreement within a tolerance. Every shared
non-timing history field must be elementwise identical (NaNs in matching positions
count as equal), and the best and final checkpoints must have identical keys with
torch.equal() holding for every tensor at its stored dtype - no float32 casting is
involved in the pass/fail decision. Maximum numerical differences are printed as
diagnostics only. The proximal penalty must be exactly 0.0, and both the selected
round and the selected validation macro-F1 must match exactly.

Test 3 is a short 3-round run on two partitions (seed 42 / iid and seed 44 /
alpha_0p1), not the 40-round experiment. It runs on CPU for bitwise determinism.
Modules 32 and 34 are imported unchanged; only their output-directory and
round-budget globals are redirected, so nothing is written outside the
37-feature verification directories. Both modules are additionally checked to
agree on the corrected 37-feature contract (input_dim 37, data/processed_37f),
and X_train/X_val are asserted to have 37 columns.

The research roots are hashed before and after and must be identical. The guarded
set covers the 37-feature roots, the retained 41-feature FedAvg/FedProx roots, and
the read-only inputs (the 37-feature processed arrays and the K=5 final
partitions), so this verifier cannot disturb any baseline or its inputs. Reads the
train and validation arrays only.
"""

from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# TOL applies only to the analytic gradient check in Test 1. The mu=0 equivalence
# in Test 3 is exact and uses no tolerance.
TOL = 1e-6
ROUNDS = 3
MU = 0.0
CASES = [(42, "iid"), (44, "alpha_0p1")]

INPUT_DIM = 37
PROCESSED_DIR = Path("data/processed_37f")
PART_ROOT = Path("data/fl_clients/final_partitions/k_5")
FEDAVG_SCRIPT = "32_train_final_fedavg_37f.py"
FEDPROX_SCRIPT = "34_train_final_fedprox_37f.py"

# Isolated 37-feature verification roots (distinct from the 41-feature ones).
VERIF_RESULTS = Path("results/fedprox_verification_37f")
VERIF_MODELS = Path("models/fedprox_verification_37f")

# Research roots that must not change while the verification runs: the 37-feature
# roots, the retained 41-feature baselines, and the read-only inputs (the corrected
# processed arrays and the K=5 final partitions). The before/after manifest hashes
# every file under each of these, so the input arrays are proven untouched too.
GUARDED_DIRS = [Path("results/final_fedavg_k5_37f"), Path("models/final_fedavg_k5_37f"),
                Path("results/final_fedprox_k5_37f"), Path("models/final_fedprox_k5_37f"),
                Path("results/final_fedavg_k5"), Path("models/final_fedavg_k5"),
                Path("results/final_fedprox_k5"), Path("models/final_fedprox_k5"),
                PROCESSED_DIR, PART_ROOT]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def manifest(dirs) -> dict:
    out = {}
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file():
                out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def redirect(module, results_dir: Path, models_dir: Path) -> None:
    """Point a training module at the verification area and shorten its budget."""
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    module.RESULTS_DIR = results_dir
    module.MODELS_DIR = models_dir
    module.MAX_ROUNDS = ROUNDS
    assert str(module.RESULTS_DIR).startswith(str(VERIF_RESULTS)), "results redirect failed"
    assert str(module.MODELS_DIR).startswith(str(VERIF_MODELS)), "models redirect failed"


def assert_37f_contract(m32, m34, avg_models_root: Path, avg_results_root: Path) -> None:
    """Both modules must target the corrected 37-feature branch.

    avg_models_root/avg_results_root are 32's declared research roots, captured
    before redirect() rewrites them.
    """
    print("=== TEST 0: 37-feature contract ===")
    assert m32.INPUT_DIM == INPUT_DIM, f"32 INPUT_DIM is {m32.INPUT_DIM}, expected {INPUT_DIM}"
    assert m34.INPUT_DIM == INPUT_DIM, f"34 INPUT_DIM is {m34.INPUT_DIM}, expected {INPUT_DIM}"
    assert Path(m32.PROCESSED_DIR) == PROCESSED_DIR, f"32 PROCESSED_DIR is {m32.PROCESSED_DIR}"
    assert Path(m34.PROCESSED_DIR) == PROCESSED_DIR, f"34 PROCESSED_DIR is {m34.PROCESSED_DIR}"
    assert Path(m34.INIT_PATH) == avg_models_root / "initial_global_model.pt", \
        f"34 does not reference the 37f FedAvg initial state: {m34.INIT_PATH}"
    assert Path(m34.FEDAVG_CLASS_WEIGHTS_PATH) == avg_results_root / "class_weights.npy", \
        f"34 does not reference the 37f FedAvg class weights: {m34.FEDAVG_CLASS_WEIGHTS_PATH}"

    x_train = np.load(PROCESSED_DIR / "X_train.npy", mmap_mode="r")
    x_val = np.load(PROCESSED_DIR / "X_val.npy", mmap_mode="r")
    assert x_train.shape[1] == INPUT_DIM, f"X_train has {x_train.shape[1]} columns, expected {INPUT_DIM}"
    assert x_val.shape[1] == INPUT_DIM, f"X_val has {x_val.shape[1]} columns, expected {INPUT_DIM}"
    print(f"  input_dim=32:{m32.INPUT_DIM} 34:{m34.INPUT_DIM}  "
          f"X_train cols={x_train.shape[1]}  X_val cols={x_val.shape[1]}  -> PASS")


def exact_array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Elementwise exact equality. NaNs occupying the same position count as equal,
    so a shared field that is legitimately NaN (e.g. an undefined per-class PR-AUC)
    does not fail the check, while a NaN opposite a number does."""
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=True))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Diagnostic only. Maximum |a - b| over positions where the difference is
    finite; returns 0.0 if no such position exists."""
    diff = np.abs(a - b)
    finite = np.isfinite(diff)
    return float(diff[finite].max()) if bool(finite.any()) else 0.0


def checkpoint_compare(path_a: Path, path_b: Path) -> tuple[bool, float]:
    """Compare two checkpoints. The pass/fail decision is exact: identical key sets
    and torch.equal() for every tensor, evaluated at the stored dtype with no
    casting (torch.equal also requires matching dtype and shape). The returned
    maximum absolute difference is a float64 diagnostic and is never the acceptance
    criterion."""
    a = torch.load(path_a, map_location="cpu")
    b = torch.load(path_b, map_location="cpu")
    if a.keys() != b.keys():
        print(f"  checkpoint keys differ: {path_a} vs {path_b}")
        return False, float("inf")
    exact = all(torch.equal(a[k], b[k]) for k in a)
    diff = max(float((a[k].to(torch.float64) - b[k].to(torch.float64)).abs().max()) for k in a)
    return exact, diff


def gradient_test(m34) -> float:
    print("\n=== TEST 1: proximal gradient ===")
    torch.manual_seed(0)
    mu = 0.7
    layer = nn.Linear(3, 2)
    # A fixed reference offset from the current parameters.
    reference = {name: p.detach().clone() + 0.37 for name, p in layer.named_parameters()}
    layer.zero_grad(set_to_none=True)
    penalty = 0.5 * mu * m34.proximal_norm(layer, reference)
    penalty.backward()
    max_diff = 0.0
    for name, p in layer.named_parameters():
        expected = mu * (p.detach() - reference[name])
        max_diff = max(max_diff, float((p.grad - expected).abs().max()))
    ok = max_diff <= TOL
    print(f"  max|grad - mu*(w - w_ref)| = {max_diff:.2e}  -> {'PASS' if ok else 'FAIL'}")
    assert ok, "Gradient test failed."
    return max_diff


def reference_test(m34) -> tuple[float, float]:
    print("\n=== TEST 2: fixed reference ===")
    torch.manual_seed(0)
    model = m34.MLPMultiClassClassifier(m34.INPUT_DIM, m34.NUM_CLASSES)
    reference = m34.capture_global_reference(model)
    # Independent snapshot taken immediately after capture, used to prove the
    # reference is not a view onto the live parameters.
    snapshot = {name: t.clone() for name, t in reference.items()}
    norm0 = float(m34.proximal_norm(model, reference))
    with torch.no_grad():
        first = next(p for p in model.parameters() if p.requires_grad)
        first.add_(1.0)
    norm1 = float(m34.proximal_norm(model, reference))
    # Every reference tensor must be bitwise identical to its snapshot after the
    # model was modified in place.
    reference_unchanged = reference.keys() == snapshot.keys() and all(
        torch.equal(reference[name], snapshot[name])
        for name in snapshot
    )
    ok = (norm0 == 0.0) and (norm1 > 0.0) and reference_unchanged
    print(f"  norm after capture = {norm0:.2e} (expect exactly 0)")
    print(f"  norm after change  = {norm1:.6f} (expect > 0)")
    print(f"  reference unchanged = {reference_unchanged} (expect True)")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    assert ok, "Reference test failed."
    return norm0, norm1


def equivalence_test(m32, m34, device) -> dict:
    print(f"\n=== TEST 3: mu=0 equivalence ({ROUNDS} rounds, CPU) ===")
    avg_res, avg_mod = VERIF_RESULTS / "fedavg37f", VERIF_MODELS / "fedavg37f"
    prox_res, prox_mod = VERIF_RESULTS / "fedprox37f", VERIF_MODELS / "fedprox37f"
    redirect(m32, avg_res, avg_mod)
    redirect(m34, prox_res, prox_mod)

    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    class_counts = np.bincount(y_train, minlength=m32.NUM_CLASSES)
    weights = m32.class_weights_full(y_train).astype(np.float32)
    assert np.array_equal(weights, m34.class_weights_full(y_train).astype(np.float32)), \
        "class weight formulas differ between 32 and 34"

    # The shared initial state stays the read-only artefact of the 37f FedAvg run.
    initial_state = torch.load(m34.INIT_PATH, map_location="cpu")
    val_loader = DataLoader(
        m32.FullDataset(PROCESSED_DIR / "X_val.npy", PROCESSED_DIR / "y_val.npy"),
        batch_size=m32.BATCH_SIZE, shuffle=False, num_workers=0)

    frag = m34.mu_fragment(MU)
    results = {}
    for seed, condition in CASES:
        part_avg = m32.load_partition(seed, condition, y_train, class_counts)
        part_prox = m34.load_partition(seed, condition, y_train, class_counts)
        assert part_avg["sizes"] == part_prox["sizes"], "client sizes differ between modules"
        assert np.array_equal(part_avg["client_class_counts"], part_prox["client_class_counts"]), \
            "client class counts differ between modules"

        m32.run(seed, condition, part_avg, initial_state, weights, val_loader, device)
        m34.run(seed, condition, MU, part_prox, initial_state, weights, val_loader, device)

        avg_tag = f"fedavg37f_k{m32.NUM_CLIENTS}_seed{seed}_{condition}"
        prox_tag = f"fedprox37f_k{m34.NUM_CLIENTS}_mu{frag}_seed{seed}_{condition}"

        h_avg = pd.read_csv(avg_res / f"history_{avg_tag}.csv")
        h_prox = pd.read_csv(prox_res / f"history_{prox_tag}.csv")
        assert len(h_avg) == ROUNDS and len(h_prox) == ROUNDS, "unexpected history length"

        # Shared fields, excluding wall-clock timings. Equality is exact; the
        # max-abs difference is retained only as a diagnostic.
        shared = [c for c in h_avg.columns if c in h_prox.columns and "seconds" not in c]
        hist_a = h_avg[shared].to_numpy(dtype=float)
        hist_b = h_prox[shared].to_numpy(dtype=float)
        history_exact = exact_array_equal(hist_a, hist_b)
        d_hist = max_abs_diff(hist_a, hist_b)

        loss_cols = [c for c in shared if c.startswith("train_loss_client_")]
        assert len(loss_cols) == m32.NUM_CLIENTS, f"expected {m32.NUM_CLIENTS} client loss columns"
        loss_a = h_avg[loss_cols].to_numpy(dtype=float)
        loss_b = h_prox[loss_cols].to_numpy(dtype=float)
        loss_exact = exact_array_equal(loss_a, loss_b)
        d_loss = max_abs_diff(loss_a, loss_b)

        prox_cols = ["round_proximal_penalty"] + [
            c for c in h_prox.columns if c.startswith("train_proximal_penalty_client_")]
        max_prox = float(np.max(np.abs(h_prox[prox_cols].to_numpy(dtype=float))))

        with open(avg_res / f"config_{avg_tag}.json") as f:
            cfg_avg = json.load(f)
        with open(prox_res / f"config_{prox_tag}.json") as f:
            cfg_prox = json.load(f)

        # Both configs must record the corrected 37-feature contract explicitly.
        assert cfg_avg["input_dim"] == INPUT_DIM and cfg_prox["input_dim"] == INPUT_DIM, \
            "a config does not record input_dim=37"
        assert Path(cfg_avg["processed_dir"]) == PROCESSED_DIR, "32 config processed_dir mismatch"
        assert Path(cfg_prox["processed_dir"]) == PROCESSED_DIR, "34 config processed_dir mismatch"

        # Exact selection agreement: same round and the identical macro-F1 value.
        round_exact = cfg_avg["best_round"] == cfg_prox["best_round"]
        score_exact = cfg_avg["best_val_macro_f1"] == cfg_prox["best_val_macro_f1"]

        # torch.equal() over both checkpoints decides pass/fail; the returned
        # differences are diagnostics.
        best_exact, d_best = checkpoint_compare(
            avg_mod / f"best_{avg_tag}.pt", prox_mod / f"best_{prox_tag}.pt")
        final_exact, d_final = checkpoint_compare(
            avg_mod / f"final_{avg_tag}.pt", prox_mod / f"final_{prox_tag}.pt")

        results[f"seed{seed}_{condition}"] = {
            # Exact pass/fail criteria.
            "history_exact": history_exact,
            "loss_exact": loss_exact,
            "best_checkpoint_exact": best_exact,
            "final_checkpoint_exact": final_exact,
            "best_round_exact": round_exact,
            "best_score_exact": score_exact,
            "max_proximal_penalty": max_prox,
            # Diagnostics only.
            "d_history": d_hist, "d_train_loss_client": d_loss,
            "d_best_ckpt": d_best, "d_final_ckpt": d_final,
            "best_round_avg": int(cfg_avg["best_round"]),
            "best_round_prox": int(cfg_prox["best_round"]),
            "best_score_avg": float(cfg_avg["best_val_macro_f1"]),
            "best_score_prox": float(cfg_prox["best_val_macro_f1"]),
            "n_shared_fields": len(shared),
        }
    return results


def main() -> None:
    # No held-out array may be referenced by either training script or this one.
    tok_x, tok_y = "X_" + "test", "y_" + "test"
    for path in (FEDAVG_SCRIPT, FEDPROX_SCRIPT, __file__):
        source = Path(path).read_text()
        assert tok_x not in source and tok_y not in source, f"Held-out reference found in {path}"

    VERIF_RESULTS.mkdir(parents=True, exist_ok=True)
    VERIF_MODELS.mkdir(parents=True, exist_ok=True)
    before = manifest(GUARDED_DIRS)

    device = torch.device("cpu")
    m32 = load_module("fedavg32_37f", FEDAVG_SCRIPT)
    m34 = load_module("fedprox34_37f", FEDPROX_SCRIPT)

    # Capture 32's declared research roots before redirect() overwrites them, so the
    # 37f-contract check can compare against the real 37-feature paths.
    avg_models_root = Path(m32.MODELS_DIR)
    avg_results_root = Path(m32.RESULTS_DIR)

    assert_37f_contract(m32, m34, avg_models_root, avg_results_root)
    grad_diff = gradient_test(m34)
    norm0, norm1 = reference_test(m34)
    eq = equivalence_test(m32, m34, device)

    after = manifest(GUARDED_DIRS)
    research_ok = before == after

    # Test 3 acceptance is exact; no tolerance is applied to any of these criteria.
    print("\n=== EQUIVALENCE RESULTS (exact equality required; no tolerance) ===")
    criteria = ("history_exact", "loss_exact", "best_checkpoint_exact",
                "final_checkpoint_exact", "best_round_exact", "best_score_exact")
    header = ("case", "hist", "loss", "best_ckpt", "final_ckpt", "round", "score", "prox==0")
    print("  " + "  ".join(f"{h:>12}" for h in header))
    all_ok = True
    for case, r in eq.items():
        prox_ok = r["max_proximal_penalty"] == 0.0
        case_ok = all(r[c] for c in criteria) and prox_ok
        all_ok = all_ok and case_ok
        print("  " + "  ".join(
            [f"{case:>12}"] + [f"{str(bool(r[c])):>12}" for c in criteria]
            + [f"{str(prox_ok):>12}"]))
        print(f"      -> {'PASS' if case_ok else 'FAIL'}   "
              f"diagnostics: d_hist={r['d_history']:.2e} d_loss={r['d_train_loss_client']:.2e} "
              f"d_best={r['d_best_ckpt']:.2e} d_final={r['d_final_ckpt']:.2e} "
              f"max_prox={r['max_proximal_penalty']:.2e}")
        print(f"         best_round avg={r['best_round_avg']} prox={r['best_round_prox']}   "
              f"best_val_macro_f1 avg={r['best_score_avg']!r} prox={r['best_score_prox']!r}")

    print(f"\nGradient test max diff: {grad_diff:.2e}")
    print(f"Reference norms: after_capture={norm0:.2e}, after_change={norm1:.6f}")
    print(f"Shared history fields compared: {next(iter(eq.values()))['n_shared_fields']}")
    print(f"Research artefacts unchanged: {'PASS' if research_ok else 'FAIL'}")
    print(f"OVERALL: {'PASS' if all_ok and research_ok else 'FAIL'}")
    if not (all_ok and research_ok):
        raise SystemExit("37-feature FedProx verification FAILED.")
    print("Verification artefacts under:", VERIF_RESULTS, "and", VERIF_MODELS)


if __name__ == "__main__":
    main()
