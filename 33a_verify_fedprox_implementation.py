"""
Verification of the FedProx implementation in 33_train_final_fedprox.py.

Three tests:
  (1) Gradient test  - the proximal gradient equals mu * (w - w_ref).
  (2) Reference test - the captured reference gives a zero proximal norm, and a
                       positive one after a synthetic parameter change.
  (3) mu=0 equivalence - FedProx (33) at mu=0 must reproduce FedAvg (29) under an
                       identical initial state, partitions, class weights,
                       optimiser, seeds and DataLoader order.

Test 3 is a short 3-round run on two partitions (seed 42 / iid and seed 44 /
alpha_0p1), not the 40-round experiment. It runs on CPU for bitwise determinism.
Modules 29 and 33 are imported unchanged; only their output-directory and
round-budget globals are redirected, so nothing is written outside the
verification directories. The research roots are hashed before and after and must
be identical. Reads the train and validation arrays only.
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

TOL = 1e-6
ROUNDS = 3
MU = 0.0
CASES = [(42, "iid"), (44, "alpha_0p1")]

PROCESSED_DIR = Path("data/processed")
FEDAVG_SCRIPT = "29_train_final_fedavg.py"
FEDPROX_SCRIPT = "33_train_final_fedprox.py"

VERIF_RESULTS = Path("results/fedprox_verification")
VERIF_MODELS = Path("models/fedprox_verification")

# Research roots that must not change while the verification runs.
GUARDED_DIRS = [Path("results/final_fedavg_k5"), Path("models/final_fedavg_k5"),
                Path("results/final_fedprox_k5"), Path("models/final_fedprox_k5")]


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


def checkpoint_diff(path_a: Path, path_b: Path) -> float:
    a = torch.load(path_a, map_location="cpu")
    b = torch.load(path_b, map_location="cpu")
    assert a.keys() == b.keys(), f"checkpoint keys differ: {path_a} vs {path_b}"
    return max(float((a[k].to(torch.float32) - b[k].to(torch.float32)).abs().max()) for k in a)


def gradient_test(m33) -> float:
    print("=== TEST 1: proximal gradient ===")
    torch.manual_seed(0)
    mu = 0.7
    layer = nn.Linear(3, 2)
    # A fixed reference offset from the current parameters.
    reference = {name: p.detach().clone() + 0.37 for name, p in layer.named_parameters()}
    layer.zero_grad(set_to_none=True)
    penalty = 0.5 * mu * m33.proximal_norm(layer, reference)
    penalty.backward()
    max_diff = 0.0
    for name, p in layer.named_parameters():
        expected = mu * (p.detach() - reference[name])
        max_diff = max(max_diff, float((p.grad - expected).abs().max()))
    ok = max_diff <= TOL
    print(f"  max|grad - mu*(w - w_ref)| = {max_diff:.2e}  -> {'PASS' if ok else 'FAIL'}")
    assert ok, "Gradient test failed."
    return max_diff


def reference_test(m33) -> tuple[float, float]:
    print("\n=== TEST 2: fixed reference ===")
    torch.manual_seed(0)
    model = m33.MLPMultiClassClassifier(m33.INPUT_DIM, m33.NUM_CLASSES)
    reference = m33.capture_global_reference(model)
    norm0 = float(m33.proximal_norm(model, reference))
    with torch.no_grad():
        first = next(p for p in model.parameters() if p.requires_grad)
        first.add_(1.0)
    norm1 = float(m33.proximal_norm(model, reference))
    ok = (norm0 <= 1e-12) and (norm1 > 0.0)
    print(f"  norm after capture = {norm0:.2e} (expect 0)")
    print(f"  norm after change  = {norm1:.6f} (expect > 0)")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    assert ok, "Reference test failed."
    return norm0, norm1


def equivalence_test(m29, m33, device) -> dict:
    print(f"\n=== TEST 3: mu=0 equivalence ({ROUNDS} rounds, CPU) ===")
    avg_res, avg_mod = VERIF_RESULTS / "fedavg", VERIF_MODELS / "fedavg"
    prox_res, prox_mod = VERIF_RESULTS / "fedprox", VERIF_MODELS / "fedprox"
    redirect(m29, avg_res, avg_mod)
    redirect(m33, prox_res, prox_mod)

    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    class_counts = np.bincount(y_train, minlength=m29.NUM_CLASSES)
    weights = m29.class_weights_full(y_train).astype(np.float32)
    assert np.array_equal(weights, m33.class_weights_full(y_train).astype(np.float32)), \
        "class weight formulas differ between 29 and 33"

    # The shared initial state stays the read-only artefact of the FedAvg run.
    initial_state = torch.load(m29.INIT_PATH, map_location="cpu")
    val_loader = DataLoader(
        m29.FullDataset(PROCESSED_DIR / "X_val.npy", PROCESSED_DIR / "y_val.npy"),
        batch_size=m29.BATCH_SIZE, shuffle=False, num_workers=0)

    frag = m33.mu_fragment(MU)
    results = {}
    for seed, condition in CASES:
        part_avg = m29.load_partition(seed, condition, y_train, class_counts)
        part_prox = m33.load_partition(seed, condition, y_train, class_counts)
        assert part_avg["sizes"] == part_prox["sizes"], "client sizes differ between modules"
        assert np.array_equal(part_avg["client_class_counts"], part_prox["client_class_counts"]), \
            "client class counts differ between modules"

        m29.run(seed, condition, part_avg, initial_state, weights, val_loader, device)
        m33.run(seed, condition, MU, part_prox, initial_state, weights, val_loader, device)

        avg_tag = f"fedavg_k{m29.NUM_CLIENTS}_seed{seed}_{condition}"
        prox_tag = f"fedprox_k{m33.NUM_CLIENTS}_mu{frag}_seed{seed}_{condition}"

        h_avg = pd.read_csv(avg_res / f"history_{avg_tag}.csv")
        h_prox = pd.read_csv(prox_res / f"history_{prox_tag}.csv")
        assert len(h_avg) == ROUNDS and len(h_prox) == ROUNDS, "unexpected history length"

        # Shared fields, excluding wall-clock timings.
        shared = [c for c in h_avg.columns if c in h_prox.columns and "seconds" not in c]
        d_hist = float(np.max(np.abs(
            h_avg[shared].to_numpy(dtype=float) - h_prox[shared].to_numpy(dtype=float))))

        loss_cols = [c for c in shared if c.startswith("train_loss_client_")]
        assert len(loss_cols) == m29.NUM_CLIENTS, f"expected {m29.NUM_CLIENTS} client loss columns"
        d_loss = float(np.max(np.abs(
            h_avg[loss_cols].to_numpy(dtype=float) - h_prox[loss_cols].to_numpy(dtype=float))))

        prox_cols = ["round_proximal_penalty"] + [
            c for c in h_prox.columns if c.startswith("train_proximal_penalty_client_")]
        max_prox = float(np.max(np.abs(h_prox[prox_cols].to_numpy(dtype=float))))

        with open(avg_res / f"config_{avg_tag}.json") as f:
            cfg_avg = json.load(f)
        with open(prox_res / f"config_{prox_tag}.json") as f:
            cfg_prox = json.load(f)

        d_best = checkpoint_diff(avg_mod / f"best_{avg_tag}.pt", prox_mod / f"best_{prox_tag}.pt")
        d_final = checkpoint_diff(avg_mod / f"final_{avg_tag}.pt", prox_mod / f"final_{prox_tag}.pt")

        results[f"seed{seed}_{condition}"] = {
            "d_history": d_hist, "d_train_loss_client": d_loss,
            "max_proximal_penalty": max_prox,
            "best_round_avg": int(cfg_avg["best_round"]),
            "best_round_prox": int(cfg_prox["best_round"]),
            "d_best_ckpt": d_best, "d_final_ckpt": d_final,
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
    m29 = load_module("fedavg29", FEDAVG_SCRIPT)
    m33 = load_module("fedprox33", FEDPROX_SCRIPT)

    grad_diff = gradient_test(m33)
    norm0, norm1 = reference_test(m33)
    eq = equivalence_test(m29, m33, device)

    after = manifest(GUARDED_DIRS)
    research_ok = before == after

    print(f"\n=== EQUIVALENCE RESULTS (max abs differences; tol = {TOL}) ===")
    header = ("case", "d_hist", "d_loss", "d_best", "d_final", "max_prox", "best_avg", "best_prox")
    print("  " + "  ".join(f"{h:>16}" for h in header))
    all_ok = True
    for case, r in eq.items():
        floats_ok = max(r["d_history"], r["d_train_loss_client"],
                        r["d_best_ckpt"], r["d_final_ckpt"]) <= TOL
        prox_ok = r["max_proximal_penalty"] == 0.0
        round_ok = r["best_round_avg"] == r["best_round_prox"]
        all_ok = all_ok and floats_ok and prox_ok and round_ok
        print("  " + "  ".join([
            f"{case:>16}", f"{r['d_history']:>16.2e}", f"{r['d_train_loss_client']:>16.2e}",
            f"{r['d_best_ckpt']:>16.2e}", f"{r['d_final_ckpt']:>16.2e}",
            f"{r['max_proximal_penalty']:>16.2e}",
            f"{r['best_round_avg']:>16}", f"{r['best_round_prox']:>16}"]))

    print(f"\nGradient test max diff: {grad_diff:.2e}")
    print(f"Reference norms: after_capture={norm0:.2e}, after_change={norm1:.6f}")
    print(f"Shared history fields compared: {next(iter(eq.values()))['n_shared_fields']}")
    print(f"Research artefacts unchanged: {'PASS' if research_ok else 'FAIL'}")
    print(f"OVERALL: {'PASS' if all_ok and research_ok else 'FAIL'}")
    if not (all_ok and research_ok):
        raise SystemExit("FedProx verification FAILED.")
    print("Verification artefacts under:", VERIF_RESULTS, "and", VERIF_MODELS)


if __name__ == "__main__":
    main()
