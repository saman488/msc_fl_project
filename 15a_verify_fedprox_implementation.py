"""
Verification of the FedProx implementation in 15_train_fedprox_sgd.py.

Three tests:
  (1) Gradient unit test  - proximal gradient equals mu * (local - global_reference).
  (2) Reference test      - proximal norm is 0 right after loading the server model,
                            and positive after a synthetic non-zero parameter update.
  (3) mu=0 equivalence    - FedAvg (13) and FedProx (15, mu=0) with identical settings
                            must produce matching outputs. Confusion matrices and
                            predicted labels must match EXACTLY; floating-point outputs
                            must differ by no more than 1e-6.

Verification artefacts are written under a dedicated verification directory,
separate from research results. The test set is never accessed. Run on CPU for
bitwise determinism. Modules 13 and 15 are imported unchanged; only their in-memory
output-directory globals are redirected so nothing touches research artefacts.
"""

from pathlib import Path
from types import SimpleNamespace
import importlib.util

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

TOL = 1e-6
PROCESSED_DIR = Path("data/processed")
AUDITED_INIT = Path("models/fl_noniid_controlled/initial_global_model.pt")
VERIF_RESULTS = Path("results/fl_fedprox_verification")
VERIF_MODELS = Path("models/fl_fedprox_verification")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gradient_unit_test(m15):
    print("=== TEST 1: gradient unit test (CPU, positive mu) ===")
    torch.manual_seed(0)
    mu = 0.7
    layer = nn.Linear(3, 2)
    # A fixed detached reference that differs from the current params.
    global_reference = {name: (p.detach().clone() + 0.37)
                        for name, p in layer.named_parameters() if p.requires_grad}
    layer.zero_grad(set_to_none=True)
    penalty = 0.5 * mu * m15.proximal_norm(layer, global_reference)
    penalty.backward()
    max_diff = 0.0
    for name, p in layer.named_parameters():
        expected = mu * (p.detach() - global_reference[name])
        max_diff = max(max_diff, float((p.grad - expected).abs().max()))
    ok = max_diff <= TOL
    print(f"  max|grad - mu*(local-global)| = {max_diff:.2e}  -> {'PASS' if ok else 'FAIL'}")
    assert ok, "Gradient unit test failed."
    return max_diff


def reference_test(m15):
    print("\n=== TEST 2: reference test ===")
    torch.manual_seed(0)
    model = m15.MLPMultiClassClassifier(m15.INPUT_DIM, m15.NUM_CLASSES)
    global_reference = {name: p.detach().clone()
                        for name, p in model.named_parameters() if p.requires_grad}
    norm0 = float(m15.proximal_norm(model, global_reference))
    with torch.no_grad():
        first = next(p for p in model.parameters() if p.requires_grad)
        first.add_(1.0)  # synthetic non-zero update
    norm1 = float(m15.proximal_norm(model, global_reference))
    ok = (norm0 <= 1e-12) and (norm1 > 0.0)
    print(f"  proximal_norm after load = {norm0:.2e} (expect 0)")
    print(f"  proximal_norm after synthetic update = {norm1:.6f} (expect > 0)")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    assert ok, "Reference test failed."
    return norm0, norm1


def build_args(mu=None):
    a = SimpleNamespace(lr=0.1, momentum=0.0, weight_decay=0.0, rounds=3, tag="verif")
    if mu is not None:
        a.mu = mu
    return a


def equivalence_test(m13, m15, device):
    print("\n=== TEST 3: mu=0 equivalence (FedAvg 13 vs FedProx 15, mu=0) ===")
    partitions = ["iid", "alpha_0.01"]

    # Redirect module output dirs to the verification area (research dirs untouched).
    avg_res, avg_mod = VERIF_RESULTS / "fedavg", VERIF_MODELS / "fedavg"
    prox_res, prox_mod = VERIF_RESULTS / "fedprox", VERIF_MODELS / "fedprox"
    for d in (avg_res, avg_mod, prox_res, prox_mod):
        d.mkdir(parents=True, exist_ok=True)
    m13.RESULTS_DIR, m13.MODELS_DIR = avg_res, avg_mod
    m15.RESULTS_DIR, m15.MODELS_DIR = prox_res, prox_mod

    # Capture per-(round, client) local task losses via lightweight wrappers.
    avg_task, prox_task = [], []
    orig13 = m13.train_one_epoch
    orig15 = m15.train_one_epoch

    def wrap13(*a, **k):
        r = orig13(*a, **k)
        avg_task.append(r[0])
        return r

    def wrap15(*a, **k):
        r = orig15(*a, **k)
        prox_task.append(r[0])  # r[0] is the task loss for FedProx
        return r

    m13.train_one_epoch = wrap13
    m15.train_one_epoch = wrap15

    class_names = m13.load_class_names()
    initial_state = torch.load(AUDITED_INIT, map_location="cpu")
    val_loader = DataLoader(
        m13.FullDataset(PROCESSED_DIR / "X_val.npy", PROCESSED_DIR / "y_val.npy"),
        batch_size=m13.BATCH_SIZE, shuffle=False, num_workers=0)

    results = {}
    for partition in partitions:
        avg_task.clear(); prox_task.clear()
        m13.run(42, partition, build_args(), initial_state, val_loader, class_names, device)
        m15.run(42, partition, build_args(mu=0.0), initial_state, val_loader, class_names, device)

        # ---- compare local task losses ----
        d_task = float(np.max(np.abs(np.array(avg_task) - np.array(prox_task))))

        # ---- histories (shared val columns) ----
        h_avg = pd.read_csv(avg_res / f"verif_history_seed42_{partition}.csv")
        h_prox = pd.read_csv(prox_res / f"verif_history_seed42_{partition}.csv")
        shared = [c for c in h_avg.columns if c in h_prox.columns]
        d_hist = float(np.max(np.abs(
            h_avg[shared].to_numpy(dtype=float) - h_prox[shared].to_numpy(dtype=float))))

        # ---- best round ----
        br_avg = int(h_avg.loc[h_avg["macro_f1"].idxmax(), "round"])
        br_prox = int(h_prox.loc[h_prox["macro_f1"].idxmax(), "round"])

        # ---- confusion matrices (exact) ----
        cm_avg = pd.read_csv(avg_res / f"verif_confusion_seed42_{partition}.csv", index_col=0).values
        cm_prox = pd.read_csv(prox_res / f"verif_confusion_seed42_{partition}.csv", index_col=0).values
        cm_exact = bool(np.array_equal(cm_avg, cm_prox))

        # ---- predicted labels (exact) ----
        pred_avg = np.load(avg_res / f"verif_val_pred_seed42_{partition}.npy")
        pred_prox = np.load(prox_res / f"verif_val_pred_seed42_{partition}.npy")
        pred_exact = bool(np.array_equal(pred_avg, pred_prox))

        # ---- probability arrays ----
        pr_avg = np.load(avg_res / f"verif_val_probs_seed42_{partition}.npy")
        pr_prox = np.load(prox_res / f"verif_val_probs_seed42_{partition}.npy")
        d_probs = float(np.max(np.abs(pr_avg - pr_prox)))

        # ---- checkpoint parameters ----
        sd_avg = torch.load(avg_mod / f"verif_best_seed42_{partition}.pt", map_location="cpu")
        sd_prox = torch.load(prox_mod / f"verif_best_seed42_{partition}.pt", map_location="cpu")
        d_ckpt = max(float((sd_avg[k] - sd_prox[k]).abs().max()) for k in sd_avg)

        results[partition] = {
            "d_local_task_loss": d_task, "d_history": d_hist,
            "best_round_avg": br_avg, "best_round_prox": br_prox,
            "confusion_exact": cm_exact, "pred_exact": pred_exact,
            "d_probs": d_probs, "d_checkpoint": d_ckpt,
        }

    m13.train_one_epoch = orig13
    m15.train_one_epoch = orig15
    return results


def main():
    VERIF_RESULTS.mkdir(parents=True, exist_ok=True)
    VERIF_MODELS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")  # bitwise determinism for equivalence

    m13 = load_module("fedavg13", "13_select_fedavg_sgd_config.py")
    m15 = load_module("fedprox15", "15_train_fedprox_sgd.py")

    # No-test-array assertion for both this script and 15.
    tok_x, tok_y = "X_" + "test", "y_" + "test"
    for src_path in ("15_train_fedprox_sgd.py", __file__):
        s = Path(src_path).read_text()
        assert tok_x not in s and tok_y not in s, f"Test-array reference found in {src_path}"

    g = gradient_unit_test(m15)
    r0, r1 = reference_test(m15)
    eq = equivalence_test(m13, m15, device)

    print("\n=== EQUIVALENCE RESULTS (max abs differences; tol = 1e-6) ===")
    header = ("partition", "d_task", "d_hist", "d_probs", "d_ckpt",
              "cm_exact", "pred_exact", "best_avg", "best_prox")
    print("  " + "  ".join(f"{h:>10}" for h in header))
    all_ok = True
    for part, r in eq.items():
        floats_ok = max(r["d_local_task_loss"], r["d_history"], r["d_probs"], r["d_checkpoint"]) <= TOL
        exact_ok = r["confusion_exact"] and r["pred_exact"] and (r["best_round_avg"] == r["best_round_prox"])
        all_ok = all_ok and floats_ok and exact_ok
        print("  " + "  ".join([
            f"{part:>10}", f"{r['d_local_task_loss']:>10.2e}", f"{r['d_history']:>10.2e}",
            f"{r['d_probs']:>10.2e}", f"{r['d_checkpoint']:>10.2e}",
            f"{str(r['confusion_exact']):>10}", f"{str(r['pred_exact']):>10}",
            f"{r['best_round_avg']:>10}", f"{r['best_round_prox']:>10}"]))

    print(f"\nGradient unit test max diff: {g:.2e}")
    print(f"Reference norms: after_load={r0:.2e}, after_update={r1:.6f}")
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'} (all float diffs <= {TOL}, exact matches hold)")
    if not all_ok:
        raise SystemExit("Equivalence verification FAILED.")
    print("Verification artefacts under:", VERIF_RESULTS, "and", VERIF_MODELS)


if __name__ == "__main__":
    main()
