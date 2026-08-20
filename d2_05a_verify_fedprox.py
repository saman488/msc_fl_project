"""
Verification of the Dataset-2 FedProx implementation in d2_05_train_fedprox.py.

Adapted from the verified 37-feature verifier 34a_verify_fedprox_37f.py. The
default suite is lightweight: every check runs on CPU against synthetic tensors,
temporary fixtures or the parsed source, so it can be run while a Dataset-2 FedAvg
production run is active. No Dataset-2 round, partition or production matrix is
executed by the default suite, and nothing outside VERIF_ROOT is written.

Default checks
--------------
 1. proximal gradient equals mu * (w - w_ref), computed independently
 2. proximal norm is exactly zero when the local parameters equal the reference
 3. proximal norm is > 0 after a controlled perturbation, reference bitwise intact
 4. mu=0 contributes an exact zero to the total loss
 5. mu=0 local update reproduces d2_04 FedAvg under matched init/seed/batches/optimizer
 6. the reference stays fixed and detached across a real local update
 7. aggregation is sample-size weighted and identical to d2_04's
 8. no held-out array is referenced by the trainer or by this verifier
 9. FedProx output roots cannot collide with the FedAvg or other protected roots
10. both scripts parse, import and satisfy the shared Dataset-2 contract
11. production mu is the frozen constant 1e-5 and is not selectable from the CLI
12. the FedAvg initial checkpoint is mandatory, and a missing, unreadable,
    wrong-shaped or non-reproducing checkpoint is fatal before training
13. no FedProx initial-global-model file is produced or preflighted
14. the FedAvg class-weight vector is mandatory and exactly checked
15. the production config records the frozen mu transfer policy and input provenance
16. both mandatory FedAvg inputs are verified before the first production write
17. the guarded imports write no bytecode outside VERIF_ROOT
18. the RunPod gate verifies the real saved class weights, not a recomputation
19. the integrity manifest hashes only the four train/validation arrays

Checks 12 and 14 exercise the real trainer functions against temporary fixtures
built under VERIF_ROOT, with the module's input paths monkeypatched for the
duration; the real FedAvg artefacts are never required, read or written by them.

RunPod integration/stability gate (opt-in)
------------------------------------------
    ./env/bin/python d2_05a_verify_fedprox.py --runpod-gate

runs a short REAL-path mu=0 equivalence gate on the real Dataset-2 partitions and
processed arrays, in the manner of 34a: d2_04.run() and d2_05.run(mu=0) are driven
directly for GATE_ROUNDS rounds on GATE_CASES partitions with their output roots
redirected into VERIF_ROOT, and their histories and checkpoints are compared
exactly. This is expensive and must not be run alongside a production job; it is
never a substitute for the synthetic check 5 and is not part of the default suite.
It writes only under VERIF_ROOT and hashes the research roots before and after to
prove nothing else changed.

Usage
-----
    ./env/bin/python d2_05a_verify_fedprox.py                # checks 1-19
    ./env/bin/python d2_05a_verify_fedprox.py --runpod-gate  # adds the real-path gate
    ./env/bin/python d2_05a_verify_fedprox.py --list         # list checks and exit
"""

from pathlib import Path
import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import shutil
import sys
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Importing d2_04/d2_05 through importlib would otherwise write .pyc files into the
# repository-root __pycache__/, i.e. OUTSIDE VERIF_ROOT. Disabled before any guarded
# import so the "nothing outside VERIF_ROOT is written" claim actually holds; check 17
# verifies this behaviourally rather than trusting the flag.
sys.dont_write_bytecode = True

FEDAVG_SCRIPT = Path("d2_04_train_fedavg.py")
FEDPROX_SCRIPT = Path("d2_05_train_fedprox.py")

# CPU only: a Dataset-2 FedAvg run may be occupying the accelerator.
DEVICE = torch.device("cpu")

# Everything this verifier writes lives here and nowhere else.
VERIF_ROOT = Path("results/nf_cse_cic_ids2018_v2/fedprox_verification")
FIXTURE_ROOT = VERIF_ROOT / "fixtures"

# Tolerance for the analytic gradient check. The mu=0 equivalence checks are exact.
TOL = 1e-6
# A deliberately wrong variant must miss by more than this.
WRONG_ANSWER_MARGIN = 1e-3

MU_TEST = 0.7
SYNTHETIC_SEED = 1234

# The frozen production policy this verifier enforces.
EXPECTED_PRODUCTION_MU = 1e-5
EXPECTED_D1_SCORE = 0.366411072723754
EXPECTED_D1_CANDIDATES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

# RunPod gate configuration (opt-in only).
GATE_ROUNDS = 2
GATE_CASES = [(42, "iid")]
GATE_RESULTS = VERIF_ROOT / "runpod_gate/results"
GATE_MODELS = VERIF_ROOT / "runpod_gate/models"
GUARDED_DIRS = [
    Path("results/nf_cse_cic_ids2018_v2/final_fedavg_k5"),
    Path("models/nf_cse_cic_ids2018_v2/final_fedavg_k5"),
    Path("results/nf_cse_cic_ids2018_v2/final_fedprox_k5"),
    Path("models/nf_cse_cic_ids2018_v2/final_fedprox_k5"),
    Path("data/nf_cse_cic_ids2018_v2/fl_clients"),
]

# The processed directory is deliberately NOT guarded recursively: rglob over it
# would enumerate and hash the held-out arrays. Integrity hashing covers only these
# four train/validation files, named explicitly. Check 19 proves the allowlist holds
# exactly these four names and that nothing reachable by the manifest is held out.
D2_PROCESSED = Path("data/nf_cse_cic_ids2018_v2/processed")
GUARDED_FILES = [
    D2_PROCESSED / "X_train.npy",
    D2_PROCESSED / "y_train.npy",
    D2_PROCESSED / "X_val.npy",
    D2_PROCESSED / "y_val.npy",
]


def load_guarded_module(path: Path, name: str):
    """Import a training script without running it, and without writing bytecode."""
    assert path.exists(), f"required script not found: {path}"
    # Defensive: re-asserted here so no future caller can import before the flag is set.
    sys.dont_write_bytecode = True
    source = path.read_text()
    assert 'if __name__ == "__main__":' in source, f"{path}: no __main__ guard"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_batch(n_rows: int, input_dim: int, num_classes: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n_rows, input_dim, generator=generator, dtype=torch.float32)
    labels = torch.randint(0, num_classes, (n_rows,), generator=generator)
    return features, labels


def class_weight_vector(num_classes: int) -> torch.Tensor:
    # Unequal, strictly positive, mirroring the weighted loss actually used.
    return torch.linspace(0.5, 2.0, num_classes, dtype=torch.float32)


def max_abs_state_diff(a: dict, b: dict) -> float:
    assert a.keys() == b.keys(), "state key mismatch"
    return max(float((a[k].detach().to(torch.float64) - b[k].detach().to(torch.float64))
                     .abs().max().item()) for k in a)


def tiny_state(reference_state: dict, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {k: torch.randn(v.shape, generator=generator, dtype=torch.float32)
            for k, v in reference_state.items()}


def function_source(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name}() not found")


def config_keys(tree: ast.AST) -> set[str]:
    """Keys of the config dict literal assigned inside run()."""
    run_node = function_source(tree, "run")
    for node in ast.walk(run_node):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "config" for t in node.targets):
            assert isinstance(node.value, ast.Dict), "config is not a dict literal"
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError("no `config = {...}` assignment found in run()")


def reset_fixture_dir(name: str) -> Path:
    """Recreate a fixture subdirectory, refusing to touch anything outside FIXTURE_ROOT.

    The only destructive filesystem operation in this verifier lives here, and it
    is guarded: the resolved target must be a strict descendant of FIXTURE_ROOT,
    so no argument can make it delete a research artefact.
    """
    target = (FIXTURE_ROOT / name).resolve()
    root = FIXTURE_ROOT.resolve()
    assert root in target.parents and target != root, \
        f"refusing to reset {target}: not a strict descendant of {root}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


class temporarily_set:
    """Context manager that swaps module attributes and always restores them."""

    def __init__(self, module, **attributes) -> None:
        self.module = module
        self.attributes = attributes
        self.previous = {}

    def __enter__(self):
        for name, value in self.attributes.items():
            self.previous[name] = getattr(self.module, name)
            setattr(self.module, name, value)
        return self.module

    def __exit__(self, *exc):
        for name, value in self.previous.items():
            setattr(self.module, name, value)
        return False


# ================================== CHECKS ================================== #
def check_01_proximal_gradient(ctx) -> None:
    """1. d(0.5*mu*sum||w - w_ref||^2)/dw == mu * (w - w_ref)."""
    m05 = ctx["m05"]
    torch.manual_seed(0)
    layer = nn.Linear(3, 2)
    # A fixed reference deliberately offset from the current parameters, so a
    # dropped or sign-flipped term cannot coincidentally give the right answer.
    reference = {name: p.detach().clone() + 0.37 for name, p in layer.named_parameters()}

    layer.zero_grad(set_to_none=True)
    penalty = 0.5 * MU_TEST * m05.proximal_norm(layer, reference)
    penalty.backward()

    max_diff = 0.0
    wrong_sign_diff = 0.0
    for name, p in layer.named_parameters():
        expected = MU_TEST * (p.detach() - reference[name])
        max_diff = max(max_diff, float((p.grad - expected).abs().max().item()))
        wrong_sign_diff = max(wrong_sign_diff,
                              float((p.grad - (-expected)).abs().max().item()))
    assert max_diff <= TOL, f"max|grad - mu*(w - w_ref)| = {max_diff:.3e} exceeds {TOL:.1e}"
    assert wrong_sign_diff > WRONG_ANSWER_MARGIN, \
        "the sign-flipped variant is indistinguishable; this check cannot discriminate"
    ctx["report"]["gradient_max_diff"] = max_diff


def check_02_zero_norm_at_reference(ctx) -> None:
    """2. The proximal norm is exactly zero when local params equal the reference."""
    m05 = ctx["m05"]
    torch.manual_seed(0)
    model = m05.MLPMultiClassClassifier(m05.INPUT_DIM, m05.NUM_CLASSES)
    reference = m05.capture_global_reference(model)
    norm = float(m05.proximal_norm(model, reference).detach())
    assert norm == 0.0, f"norm at the reference is {norm!r}, expected exactly 0.0"
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert set(reference.keys()) == set(trainable), "reference keys != trainable parameters"
    ctx["report"]["norm_at_reference"] = norm


def check_03_positive_norm_after_perturbation(ctx) -> None:
    """3. A controlled perturbation makes the norm positive; the reference is intact."""
    m05 = ctx["m05"]
    torch.manual_seed(0)
    model = m05.MLPMultiClassClassifier(m05.INPUT_DIM, m05.NUM_CLASSES)
    reference = m05.capture_global_reference(model)
    snapshot = {name: t.clone() for name, t in reference.items()}

    with torch.no_grad():
        first_name, first = next((n, p) for n, p in model.named_parameters() if p.requires_grad)
        first.add_(1.0)
    norm = float(m05.proximal_norm(model, reference).detach())

    # The perturbation is a known +1.0 on one tensor, so the norm is its numel.
    expected = float(dict(model.named_parameters())[first_name].numel())
    assert norm > 0.0, f"norm after perturbation is {norm!r}, expected > 0"
    assert abs(norm - expected) <= TOL * max(1.0, expected), \
        f"norm {norm} != independently computed {expected}"
    assert reference.keys() == snapshot.keys() and all(
        torch.equal(reference[n], snapshot[n]) for n in snapshot), \
        "the reference changed when the model was modified in place"
    ctx["report"]["norm_after_perturbation"] = norm


def check_04_mu_zero_contributes_exact_zero(ctx) -> None:
    """4. At mu=0 the proximal term is an exact zero, even far from the reference."""
    m05 = ctx["m05"]
    torch.manual_seed(0)
    model = m05.MLPMultiClassClassifier(m05.INPUT_DIM, m05.NUM_CLASSES)
    reference = m05.capture_global_reference(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(3.5)  # far from the reference: the norm itself is large

    norm = float(m05.proximal_norm(model, reference).detach())
    assert norm > WRONG_ANSWER_MARGIN, "the perturbed norm is ~0; this check is vacuous"
    penalty = 0.5 * 0.0 * m05.proximal_norm(model, reference)
    assert float(penalty.detach()) == 0.0, \
        f"mu=0 penalty is {float(penalty.detach())!r}, expected exactly 0.0"

    # And the trainer's own epoch-level guard requires exactly zero at mu=0.
    source = ctx["sq05"]
    assert 'ifmu==0.0:' in source and 'assertweighted_prox==0.0' in source, \
        "the mu=0 zero-penalty assertion is missing from the trainer"
    ctx["report"]["mu0_norm_when_far"] = norm


def _local_update(module, model_factory, epoch_fn, extra_args, features, labels,
                  initial_state, local_seed, criterion):
    """One local epoch from a shared initial state, with matched seeds and batches."""
    model = model_factory()
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.SGD(model.parameters(), lr=module.LR,
                                momentum=module.MOMENTUM, weight_decay=module.WEIGHT_DECAY)
    generator = torch.Generator()
    generator.manual_seed(local_seed)
    loader = DataLoader(TensorDataset(features, labels), batch_size=8,
                        shuffle=True, num_workers=0, generator=generator)
    torch.manual_seed(local_seed)
    stats = epoch_fn(model, loader, criterion, optimizer, DEVICE, *extra_args)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, stats


def check_05_mu_zero_matches_fedavg(ctx) -> None:
    """5. At mu=0 the local update equals d2_04's FedAvg local update exactly.

    Synthetic batches, so this runs anywhere. The real-path counterpart is the
    opt-in RunPod gate; this check is not a substitute for it and vice versa.
    """
    m04, m05 = ctx["m04"], ctx["m05"]
    features, labels = synthetic_batch(40, m05.INPUT_DIM, m05.NUM_CLASSES, SYNTHETIC_SEED)
    criterion = nn.CrossEntropyLoss(weight=class_weight_vector(m05.NUM_CLASSES).to(DEVICE))

    torch.manual_seed(m05.TRAIN_SEED)
    seed_model = m05.MLPMultiClassClassifier(m05.INPUT_DIM, m05.NUM_CLASSES).to(DEVICE)
    initial_state = {k: v.detach().cpu().clone() for k, v in seed_model.state_dict().items()}

    # The real round-1 client-0 seed formula.
    local_seed = m05.TRAIN_SEED + 1 * 100 + 0

    reference = m05.capture_global_reference(seed_model)
    prox_state, prox_stats = _local_update(
        m05, lambda: m05.MLPMultiClassClassifier(m05.INPUT_DIM, m05.NUM_CLASSES).to(DEVICE),
        m05.train_one_epoch, (reference, 0.0), features, labels, initial_state,
        local_seed, criterion)
    avg_state, avg_stats = _local_update(
        m04, lambda: m04.MLPMultiClassClassifier(m04.INPUT_DIM, m04.NUM_CLASSES).to(DEVICE),
        m04.train_one_epoch, (), features, labels, initial_state, local_seed, criterion)

    assert prox_stats["n_batches"] == avg_stats["n_batches"], "batch counts differ"
    assert prox_stats["n_samples"] == avg_stats["n_samples"], "sample counts differ"
    assert prox_stats["weighted_proximal_penalty"] == 0.0, \
        f"mu=0 proximal penalty is {prox_stats['weighted_proximal_penalty']!r}"
    assert prox_stats["weighted_task_loss"] == prox_stats["weighted_total_loss"], \
        "at mu=0 total loss must equal task loss exactly"

    # Non-vacuity: the epoch must have moved the weights.
    moved = max_abs_state_diff(prox_state, initial_state)
    assert moved > WRONG_ANSWER_MARGIN, f"the local epoch barely moved the weights ({moved:.3e})"

    diff = max_abs_state_diff(prox_state, avg_state)
    assert diff == 0.0, f"mu=0 local state differs from FedAvg by {diff:.3e} (expected exact)"

    loss_diff = abs(prox_stats["weighted_task_loss"] - avg_stats["weighted_train_loss"])
    assert loss_diff == 0.0, f"weighted task loss differs from FedAvg by {loss_diff:.3e}"
    ctx["report"]["mu0_state_diff"] = diff
    ctx["report"]["mu0_moved"] = moved


def check_06_reference_fixed_during_update(ctx) -> None:
    """6. The server reference does not move while local SGD runs."""
    m05 = ctx["m05"]
    features, labels = synthetic_batch(40, m05.INPUT_DIM, m05.NUM_CLASSES, SYNTHETIC_SEED + 1)
    criterion = nn.CrossEntropyLoss(weight=class_weight_vector(m05.NUM_CLASSES).to(DEVICE))

    torch.manual_seed(m05.TRAIN_SEED)
    model = m05.MLPMultiClassClassifier(m05.INPUT_DIM, m05.NUM_CLASSES).to(DEVICE)
    start_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    reference = m05.capture_global_reference(model)
    snapshot = {name: t.clone() for name, t in reference.items()}

    # Not a view onto the live parameters, and detached from autograd.
    live = dict(model.named_parameters())
    for name, tensor in reference.items():
        assert tensor.data_ptr() != live[name].data_ptr(), f"reference aliases parameter {name}"
        assert not tensor.requires_grad, f"reference tensor {name} still requires grad"

    optimizer = torch.optim.SGD(model.parameters(), lr=m05.LR,
                                momentum=m05.MOMENTUM, weight_decay=m05.WEIGHT_DECAY)
    generator = torch.Generator()
    generator.manual_seed(m05.TRAIN_SEED + 100)
    loader = DataLoader(TensorDataset(features, labels), batch_size=8,
                        shuffle=True, num_workers=0, generator=generator)
    torch.manual_seed(m05.TRAIN_SEED + 100)
    stats = m05.train_one_epoch(model, loader, criterion, optimizer, DEVICE, reference, MU_TEST)

    end_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    moved = max_abs_state_diff(end_state, start_state)
    assert moved > WRONG_ANSWER_MARGIN, f"the model did not move ({moved:.3e}); check is vacuous"
    assert all(torch.equal(reference[n], snapshot[n]) for n in snapshot), \
        "the reference moved during the local update"
    assert stats["weighted_proximal_penalty"] > 0.0, \
        f"mu>0 penalty is {stats['weighted_proximal_penalty']!r}, expected > 0"

    # The reference is captured from the CURRENT round's server state, inside the
    # client loop and after the global state is loaded - not from a round-0 model.
    run_node = function_source(ctx["tree05"], "run")
    round_loop = next(n for n in ast.walk(run_node)
                      if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                      and n.target.id == "rnd")
    client_loop = next(n for n in ast.walk(round_loop)
                       if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                       and n.target.id == "client_id")
    captures = [n for n in ast.walk(client_loop) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "global_reference" for t in n.targets)]
    assert len(captures) == 1, f"expected one global_reference capture, found {len(captures)}"
    call = captures[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
        and call.func.id == "capture_global_reference", "reference is not captured by the helper"
    assert [a.id for a in call.args if isinstance(a, ast.Name)] == ["local_model"], \
        "the reference is not captured from the freshly loaded local model"
    load_lines = [n.lineno for n in ast.walk(client_loop) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == "load_state_dict"]
    assert load_lines and min(load_lines) < captures[0].lineno, \
        "the reference is captured before the current global state is loaded"
    ctx["report"]["reference_moved"] = False
    ctx["report"]["mu_positive_penalty"] = stats["weighted_proximal_penalty"]


def check_07_aggregation_matches_fedavg(ctx) -> None:
    """7. Aggregation is sample-size weighted and identical to d2_04's."""
    m04, m05 = ctx["m04"], ctx["m05"]
    reference_state = m05.MLPMultiClassClassifier(m05.INPUT_DIM, m05.NUM_CLASSES).state_dict()
    states = [tiny_state(reference_state, seed=2000 + k) for k in range(m05.NUM_CLIENTS)]
    sizes = [100, 5000, 20, 900, 77]
    total = float(sum(sizes))

    independent = {key: sum(states[k][key] * (sizes[k] / total) for k in range(len(states)))
                   for key in reference_state}
    uniform = {key: sum(s[key] for s in states) / float(len(states)) for key in reference_state}
    assert max_abs_state_diff(independent, uniform) > WRONG_ANSWER_MARGIN, \
        "synthetic sizes make weighted and uniform aggregation indistinguishable"

    agg05 = m05.aggregate_sample_weighted(states, sizes)
    agg04 = m04.aggregate_sample_weighted(states, sizes)
    assert max_abs_state_diff(agg05, independent) <= TOL, \
        "FedProx aggregation != independent sample-weighted sum"
    assert max_abs_state_diff(agg05, agg04) == 0.0, "FedProx and FedAvg aggregation differ"
    assert max_abs_state_diff(agg05, uniform) > WRONG_ANSWER_MARGIN, \
        "FedProx aggregation is uniform, not size-weighted"

    assert "agg_state=aggregate_sample_weighted(client_states,sizes)" in ctx["sq05"], \
        "the FedProx round does not aggregate with the real client sizes"
    ctx["report"]["aggregation_matches"] = True


def check_08_no_test_reference(ctx) -> None:
    """8. Neither the trainer nor this verifier references a held-out array."""
    token_x = "X_" + "te" + "st"
    token_y = "y_" + "te" + "st"
    token_idx = "raw_indices_" + "te" + "st"
    for label, source in (("trainer", ctx["src05"]), ("verifier", ctx["src05a"])):
        for token in (token_x, token_y, token_idx):
            assert token not in source, f"{label} references {token}"

    import re
    loaded = set(re.findall(r'PROCESSED_DIR\s*/\s*"([^"]+)"', ctx["src05"]))
    assert loaded <= {"X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"}, \
        f"trainer loads unexpected arrays: {sorted(loaded)}"

    ctx["m05"].assert_no_test_reference()
    main_node = function_source(ctx["tree05"], "main")
    calls = [n for n in ast.walk(main_node) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "assert_no_test_reference"]
    assert calls, "main() does not call assert_no_test_reference()"
    ctx["report"]["arrays_read"] = sorted(loaded)


def check_09_output_roots_isolated(ctx) -> None:
    """9. FedProx output roots cannot collide with FedAvg or any protected root."""
    m04, m05 = ctx["m04"], ctx["m05"]
    prox_results = Path(m05.RESULTS_DIR).resolve()
    prox_models = Path(m05.MODELS_DIR).resolve()
    avg_results = Path(m04.RESULTS_DIR).resolve()
    avg_models = Path(m04.MODELS_DIR).resolve()

    for prox, avg, label in ((prox_results, avg_results, "results"),
                             (prox_models, avg_models, "models")):
        assert prox != avg, f"FedProx {label} root equals the FedAvg root"
        assert avg not in prox.parents, f"FedProx {label} root is inside the FedAvg root"
        assert prox not in avg.parents, f"FedAvg {label} root is inside the FedProx root"

    protected = [
        Path("results/nf_cse_cic_ids2018_v2/final_fedavg_k5"),
        Path("models/nf_cse_cic_ids2018_v2/final_fedavg_k5"),
        Path("results/nf_cse_cic_ids2018_v2/diagnostics"),
        Path("results/nf_cse_cic_ids2018_v2/preprocessing"),
        Path("data/nf_cse_cic_ids2018_v2/processed"),
        Path("data/nf_cse_cic_ids2018_v2/fl_clients"),
        Path("results/final_scaffold_k5_37f"),
        Path("models/final_scaffold_k5_37f"),
    ]
    for root in protected:
        resolved = root.resolve()
        for prox in (prox_results, prox_models):
            assert prox != resolved and resolved not in prox.parents, \
                f"FedProx output root {prox} collides with protected {root}"

    # The FedAvg inputs are read, never written.
    for forbidden in ("torch.save(initial_state,INIT_PATH", "np.save(FEDAVG",
                      "torch.save(global_model.state_dict(),INIT_PATH",
                      "np.save(FEDAVG_CLASS_WEIGHTS_PATH"):
        assert forbidden not in ctx["sq05"], f"trainer writes into a FedAvg path: {forbidden}"

    preflight = function_source(ctx["tree05"], "preflight_outputs")
    assert any(isinstance(n, ast.Raise) for n in ast.walk(preflight)), \
        "preflight_outputs() does not raise on collision"
    assert list(inspect.signature(m05.preflight_outputs).parameters) == ["mu"], \
        "preflight_outputs() is not mu-scoped"
    ctx["report"]["prox_roots"] = [str(m05.RESULTS_DIR), str(m05.MODELS_DIR)]


def check_10_static_contract(ctx) -> None:
    """10. Both scripts parse and import, and share the Dataset-2 contract."""
    m04, m05 = ctx["m04"], ctx["m05"]

    expected = {
        "DATASET": "nf_cse_cic_ids2018_v2",
        "INPUT_DIM": 36,
        "NUM_CLASSES": 7,
        "NUM_CLIENTS": 5,
        "LOCAL_EPOCHS": 1,
        "MAX_ROUNDS": 40,
        "TRAIN_SEED": 42,
        "LR": 0.1,
        "MOMENTUM": 0.0,
        "WEIGHT_DECAY": 0.0,
        "BATCH_SIZE": 4096,
        "RELOAD_F1_TOL": 1e-4,
        "PARTITION_SEEDS": [42, 43, 44],
        "CONDITIONS": ["iid", "alpha_0p1", "alpha_0p5", "alpha_1p0"],
        "EXPECTED_TRAIN_ROWS": 13_255_011,
        "EXPECTED_VAL_ROWS": 2_821_063,
        "PROCESSED_DIR": Path("data/nf_cse_cic_ids2018_v2/processed"),
        "LABEL_MAP": Path("configs/nf_cse_cic_ids2018_v2/label_mapping.json"),
        "PART_ROOT": Path("data/nf_cse_cic_ids2018_v2/fl_clients/final_partitions/k_5"),
    }
    for key, value in expected.items():
        got05, got04 = getattr(m05, key), getattr(m04, key)
        assert got05 == value, f"d2_05.{key} = {got05!r}, expected {value!r}"
        assert got04 == value, f"d2_04.{key} = {got04!r}, expected {value!r}"
    assert m05.METHOD == "FedProx" and m04.METHOD == "FedAvg", "method labels are wrong"

    shapes05 = {k: tuple(v.shape) for k, v in
                m05.MLPMultiClassClassifier(36, 7).state_dict().items()}
    shapes04 = {k: tuple(v.shape) for k, v in
                m04.MLPMultiClassClassifier(36, 7).state_dict().items()}
    expected_shapes = {
        "network.0.weight": (128, 36), "network.0.bias": (128,),
        "network.3.weight": (64, 128), "network.3.bias": (64,),
        "network.6.weight": (7, 64), "network.6.bias": (7,),
    }
    assert shapes05 == expected_shapes, f"FedProx architecture {shapes05}"
    assert shapes04 == expected_shapes, f"FedAvg architecture {shapes04}"
    assert str(m05.MLPMultiClassClassifier(36, 7)) == str(m04.MLPMultiClassClassifier(36, 7)), \
        "module structure differs between FedProx and FedAvg"

    for name in ("LocalPositionDataset", "ResidentClientBatches", "ResidentValLoader"):
        assert hasattr(m05, name), f"FedProx is missing {name}"
        assert list(inspect.signature(getattr(m05, name).__init__).parameters) == \
            list(inspect.signature(getattr(m04, name).__init__).parameters), \
            f"{name} signature differs from d2_04"

    for fragment in (
        "torch.optim.SGD(local_model.parameters(),lr=LR,",
        "local_seed=TRAIN_SEED+rnd*100+client_id",
        'ifval["macro_f1"]>best["macro_f1"]:',
        '"selection_metric":"val_macro_f1"',
        "reloaded_macro_f1=float(evaluate(local_model,val_loader,criterion,device)",
    ):
        assert fragment in ctx["sq05"], f"FedProx is missing shared behaviour: {fragment}"
        assert fragment in ctx["sq04"], f"FedAvg is missing shared behaviour: {fragment}"

    y = np.array([0, 0, 1, 2, 3, 4, 5, 6, 6, 6], dtype=np.int64)
    assert np.array_equal(m05.class_weights_full(y), m04.class_weights_full(y)), \
        "class weight formulas differ between FedProx and FedAvg"
    ctx["report"]["contract_ok"] = True


def check_11_production_mu_frozen(ctx) -> None:
    """11. Production mu is the frozen constant 1e-5 and is not CLI-selectable."""
    m05 = ctx["m05"]

    assert hasattr(m05, "PRODUCTION_MU"), "PRODUCTION_MU is not defined"
    assert m05.PRODUCTION_MU == EXPECTED_PRODUCTION_MU, \
        f"PRODUCTION_MU is {m05.PRODUCTION_MU!r}, expected {EXPECTED_PRODUCTION_MU!r}"
    assert isinstance(m05.PRODUCTION_MU, float), "PRODUCTION_MU is not a float"
    assert m05.MU_TUNED_ON_DATASET2 is False, "MU_TUNED_ON_DATASET2 must be False"
    assert m05.MU_SELECTED_SCORE_DATASET1 == EXPECTED_D1_SCORE, \
        f"D1 selected score is {m05.MU_SELECTED_SCORE_DATASET1!r}, expected {EXPECTED_D1_SCORE!r}"
    assert list(m05.MU_CANDIDATES_DATASET1) == EXPECTED_D1_CANDIDATES, \
        f"D1 candidate grid is {m05.MU_CANDIDATES_DATASET1!r}"
    assert "optimal" not in m05.MU_OPTIMALITY_CLAIM_DATASET2.split("none:")[0], \
        "the optimality-claim field does not begin by disclaiming optimality"

    # The CLI cannot choose mu: no --mu argument anywhere in main().
    main_node = function_source(ctx["tree05"], "main")
    for node in ast.walk(main_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "add_argument":
            literals = [a.value for a in node.args if isinstance(a, ast.Constant)]
            assert not any(isinstance(v, str) and "mu" in v.lower() for v in literals), \
                f"main() still defines a mu command-line argument: {literals}"
    # No "--mu" string CONSTANT anywhere in the module. Scanned over the AST rather
    # than the raw text, because only a string literal can become a CLI option; a
    # comment or docstring explaining that the option was removed cannot.
    mu_option_literals = [
        node.value for node in ast.walk(ctx["tree05"])
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and "--mu" in node.value
    ]
    assert not mu_option_literals, \
        f"the trainer still contains a --mu option literal: {mu_option_literals}"

    # main() binds mu from the constant.
    assert "mu=float(PRODUCTION_MU)" in ctx["sq05"], \
        "main() does not bind mu from PRODUCTION_MU"

    # Lower-level functions still take mu, so mu=0 verification remains possible.
    assert "mu" in inspect.signature(m05.train_one_epoch).parameters, \
        "train_one_epoch() no longer accepts mu"
    assert "mu" in inspect.signature(m05.run).parameters, "run() no longer accepts mu"
    assert m05.mu_fragment(EXPECTED_PRODUCTION_MU) == "1em05", \
        f"mu fragment for 1e-5 is {m05.mu_fragment(EXPECTED_PRODUCTION_MU)!r}, expected '1em05'"
    ctx["report"]["production_mu"] = m05.PRODUCTION_MU
    ctx["report"]["mu_fragment"] = m05.mu_fragment(EXPECTED_PRODUCTION_MU)


def check_12_fedavg_initial_checkpoint_mandatory(ctx) -> None:
    """12. Missing, unreadable, wrong-shaped or non-reproducing init is fatal.

    Exercises the real trainer function against temporary fixtures; the real
    FedAvg artefact is neither required nor touched.
    """
    m05 = ctx["m05"]
    fixtures = reset_fixture_dir("init")

    # (a) missing file is fatal
    with temporarily_set(m05, INIT_PATH=fixtures / "absent.pt"):
        try:
            m05.load_and_verify_fedavg_initial_state(DEVICE)
        except RuntimeError as error:
            assert "mandatory" in str(error).lower(), f"unexpected message: {error}"
        else:
            raise AssertionError("a missing FedAvg initial checkpoint was not fatal")

    # (b) unreadable file is fatal
    corrupt = fixtures / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with temporarily_set(m05, INIT_PATH=corrupt):
        try:
            m05.load_and_verify_fedavg_initial_state(DEVICE)
        except RuntimeError as error:
            assert "could not be read" in str(error), f"unexpected message: {error}"
        else:
            raise AssertionError("an unreadable FedAvg initial checkpoint was not fatal")

    # (c) a correctly reproduced state passes, and is returned
    good = fixtures / "good.pt"
    reproduced = m05.reproduce_d2_04_initial_state(DEVICE)
    torch.save(reproduced, good)
    with temporarily_set(m05, INIT_PATH=good):
        loaded, sha = m05.load_and_verify_fedavg_initial_state(DEVICE)
    assert max_abs_state_diff(loaded, reproduced) == 0.0, "the verified state was altered"
    assert sha == hashlib.sha256(good.read_bytes()).hexdigest(), "returned sha256 is wrong"

    # (d) a perturbed state is fatal, even by one ulp-scale value
    perturbed_state = {k: v.clone() for k, v in reproduced.items()}
    key = next(iter(perturbed_state))
    flat = perturbed_state[key].reshape(-1)
    flat[0] = flat[0] + 1e-3
    perturbed = fixtures / "perturbed.pt"
    torch.save(perturbed_state, perturbed)
    with temporarily_set(m05, INIT_PATH=perturbed):
        try:
            m05.load_and_verify_fedavg_initial_state(DEVICE)
        except RuntimeError as error:
            assert "independent reproduction" in str(error), f"unexpected message: {error}"
        else:
            raise AssertionError("a non-reproducing FedAvg initial checkpoint was not fatal")

    # (e) a wrong-shaped state is fatal
    wrong = fixtures / "wrong_shape.pt"
    torch.save({k: v[..., :1].clone() if v.ndim > 1 else v.clone()
                for k, v in reproduced.items()}, wrong)
    with temporarily_set(m05, INIT_PATH=wrong):
        try:
            m05.load_and_verify_fedavg_initial_state(DEVICE)
        except (AssertionError, RuntimeError):
            pass
        else:
            raise AssertionError("a wrong-shaped FedAvg initial checkpoint was not fatal")

    # (f) the verification happens before any training in main()
    main_node = function_source(ctx["tree05"], "main")
    verify_lines = [n.lineno for n in ast.walk(main_node) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == "load_and_verify_fedavg_initial_state"]
    run_lines = [n.lineno for n in ast.walk(main_node) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "run"]
    assert verify_lines, "main() does not call load_and_verify_fedavg_initial_state()"
    assert run_lines, "main() does not call run()"
    assert max(verify_lines) < min(run_lines), "initial-state verification runs after training"

    # (g) the trainer re-checks the FedAvg input hashes after training
    assert "file_sha256(INIT_PATH)==fedavg_init_sha" in ctx["sq05"], \
        "the trainer does not re-verify the FedAvg initial-state hash after training"
    ctx["report"]["init_checkpoint_mandatory"] = True


def check_13_no_fedprox_initial_model(ctx) -> None:
    """13. No FedProx initial-global-model file is produced or preflighted."""
    m05 = ctx["m05"]
    assert "initial_global_model_mu" not in ctx["src05"], \
        "the trainer still references a FedProx initial-global-model filename"

    preflight = function_source(ctx["tree05"], "preflight_outputs")
    literals = [n.value for n in ast.walk(preflight) if isinstance(n, ast.Constant)
                and isinstance(n.value, str)]
    assert not any("initial_global_model" in v for v in literals), \
        f"preflight still accounts for a FedProx initial model: {literals}"

    # The only torch.save targets in run() are the best/final checkpoints.
    run_node = function_source(ctx["tree05"], "run")
    saved = []
    for node in ast.walk(run_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "save" and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "torch":
            target = node.args[1]
            saved.append(target.id if isinstance(target, ast.Name) else ast.dump(target))
    assert set(saved) <= {"best_path", "final_path"}, \
        f"run() saves unexpected checkpoints: {saved}"

    main_node = function_source(ctx["tree05"], "main")
    main_saves = [n for n in ast.walk(main_node) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == "save"
                  and isinstance(n.func.value, ast.Name) and n.func.value.id == "torch"]
    assert not main_saves, "main() still writes a checkpoint of its own"

    frag = m05.mu_fragment(m05.PRODUCTION_MU)
    intended_names = [str(p) for p in
                      [Path(m05.RESULTS_DIR) / f"class_weights_mu{frag}.npy",
                       Path(m05.RESULTS_DIR) / f"final_summary_mu{frag}.csv"]]
    ctx["report"]["no_fedprox_init_model"] = True
    ctx["report"]["preflight_examples"] = intended_names


def check_14_fedavg_class_weights_mandatory(ctx) -> None:
    """14. The FedAvg class-weight vector is mandatory and exactly checked."""
    m05 = ctx["m05"]
    fixtures = reset_fixture_dir("weights")

    # A synthetic y_train covering all 7 classes with unequal counts.
    y = np.concatenate([np.full(n, c, dtype=np.int64)
                        for c, n in enumerate([50, 7, 11, 23, 3, 17, 5])])
    expected = m05.class_weights_full(y).astype(np.float32)

    # (a) missing file is fatal
    with temporarily_set(m05, FEDAVG_CLASS_WEIGHTS_PATH=fixtures / "absent.npy"):
        try:
            m05.load_and_verify_fedavg_class_weights(y)
        except RuntimeError as error:
            assert "mandatory" in str(error).lower(), f"unexpected message: {error}"
        else:
            raise AssertionError("missing FedAvg class weights were not fatal")

    # (b) the exact vector passes and is returned
    good = fixtures / "good.npy"
    np.save(good, expected)
    with temporarily_set(m05, FEDAVG_CLASS_WEIGHTS_PATH=good):
        weights, sha = m05.load_and_verify_fedavg_class_weights(y)
    assert np.array_equal(weights, expected), "the verified weights were altered"
    assert sha == hashlib.sha256(good.read_bytes()).hexdigest(), "returned sha256 is wrong"

    # (c) a perturbed vector is fatal
    bad = fixtures / "bad.npy"
    perturbed = expected.copy()
    perturbed[0] = np.float32(perturbed[0] * 1.001)
    np.save(bad, perturbed)
    with temporarily_set(m05, FEDAVG_CLASS_WEIGHTS_PATH=bad):
        try:
            m05.load_and_verify_fedavg_class_weights(y)
        except AssertionError:
            pass
        else:
            raise AssertionError("mismatched FedAvg class weights were not fatal")

    # (d) a wrong dtype is fatal
    wrong_dtype = fixtures / "float64.npy"
    np.save(wrong_dtype, expected.astype(np.float64))
    with temporarily_set(m05, FEDAVG_CLASS_WEIGHTS_PATH=wrong_dtype):
        try:
            m05.load_and_verify_fedavg_class_weights(y)
        except AssertionError:
            pass
        else:
            raise AssertionError("wrong-dtype FedAvg class weights were not fatal")

    assert "file_sha256(FEDAVG_CLASS_WEIGHTS_PATH)==fedavg_weights_sha" in ctx["sq05"], \
        "the trainer does not re-verify the FedAvg class-weight hash after training"
    ctx["report"]["class_weights_mandatory"] = True


def check_15_config_provenance(ctx) -> None:
    """15. The production config records the frozen mu policy and input provenance."""
    keys = config_keys(ctx["tree05"])
    required = {
        "mu", "production_mu", "mu_is_production_constant", "mu_selectable_from_cli",
        "mu_policy", "mu_source", "mu_selection_dataset", "mu_selection_rule",
        "mu_candidates_dataset1", "mu_selected_val_score_dataset1",
        "mu_tuned_on_dataset2", "mu_optimality_claim_dataset2",
        "mu_selection_used_test_data",
        "initial_state_path", "initial_state_sha256", "initial_state_source",
        "initial_state_verified_against_reproduced_d2_04_init",
        "fedprox_initial_state_written",
        "class_weights_source", "class_weights_sha256",
        "class_weights_verified_against_fresh_computation",
        "proximal_term", "proximal_reference",
        "script_sha256", "git_commit", "git_dirty", "python_version", "torch_version",
    }
    missing = sorted(required - keys)
    assert not missing, f"the production config omits: {missing}"

    m05 = ctx["m05"]
    assert m05.MU_SELECTION_DATASET == "dataset1_37f", \
        f"mu selection dataset is {m05.MU_SELECTION_DATASET!r}"
    for phrase in ("validation only", "no test data", "equal-weight mean",
                   "smaller mu", "12-run grid"):
        assert phrase in m05.MU_SELECTION_RULE, \
            f"the recorded selection rule omits '{phrase}'"
    ctx["report"]["config_keys"] = len(keys)


def check_16_prerequisites_before_first_write(ctx) -> None:
    """16. Both mandatory FedAvg inputs are verified before any production write."""
    main_node = function_source(ctx["tree05"], "main")

    def call_lines(func_name: str) -> list[int]:
        return [n.lineno for n in ast.walk(main_node) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == func_name]

    weights_lines = call_lines("load_and_verify_fedavg_class_weights")
    init_lines = call_lines("load_and_verify_fedavg_initial_state")
    run_lines = call_lines("run")
    assert weights_lines, "main() does not verify the FedAvg class weights"
    assert init_lines, "main() does not verify the FedAvg initial state"
    assert run_lines, "main() does not call run()"
    last_verification = max(max(weights_lines), max(init_lines))

    # Every artefact-writing call in main(): np.save / torch.save / *.to_csv / open(w).
    write_lines = []
    for node in ast.walk(main_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            owner = getattr(node.func.value, "id", None)
            if (attr == "save" and owner in {"np", "torch"}) or attr == "to_csv":
                write_lines.append((node.lineno, f"{owner}.{attr}" if owner else attr))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "open":
            mode = [a.value for a in node.args[1:] if isinstance(a, ast.Constant)]
            if any(isinstance(m, str) and ("w" in m or "a" in m) for m in mode):
                write_lines.append((node.lineno, "open(w)"))
    assert write_lines, "no production write found in main(); the check would be vacuous"

    early = [(line, what) for line, what in write_lines if line < last_verification]
    assert not early, (
        "production artefacts are written before both FedAvg prerequisites are "
        f"verified (last verification at line {last_verification}): {early}"
    )
    assert last_verification < min(run_lines), "verification runs after training starts"

    # Specifically: the class-weight artefact is written after the initial-state check.
    class_weight_saves = [n.lineno for n in ast.walk(main_node)
                          if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                          and n.func.attr == "save"
                          and getattr(n.func.value, "id", None) == "np"]
    assert class_weight_saves, "main() never saves the class-weight artefact"
    assert min(class_weight_saves) > max(init_lines), (
        f"np.save(class_weights) at line {min(class_weight_saves)} precedes the "
        f"initial-state verification at line {max(init_lines)}"
    )
    ctx["report"]["first_write_line"] = min(line for line, _ in write_lines)
    ctx["report"]["last_prereq_verification_line"] = last_verification


def check_17_no_bytecode_outside_verif_root(ctx) -> None:
    """17. Guarded imports write no bytecode outside VERIF_ROOT."""
    assert sys.dont_write_bytecode is True, \
        "sys.dont_write_bytecode is not set; guarded imports would write .pyc files"

    def pycache_snapshot() -> set[str]:
        seen = set()
        verif = VERIF_ROOT.resolve()
        for cache in Path(".").rglob("__pycache__"):
            resolved = cache.resolve()
            if resolved == verif or verif in resolved.parents:
                continue
            for f in cache.glob("*.pyc"):
                seen.add(str(f))
        return seen

    # Behavioural proof: re-import both scripts and confirm no new .pyc appears
    # anywhere outside VERIF_ROOT.
    before = pycache_snapshot()
    load_guarded_module(FEDAVG_SCRIPT, "d2_04_bytecode_probe")
    load_guarded_module(FEDPROX_SCRIPT, "d2_05_bytecode_probe")
    after = pycache_snapshot()
    created = sorted(after - before)
    assert not created, f"guarded imports created bytecode outside VERIF_ROOT: {created}"

    ctx["report"]["bytecode_written_outside_verif_root"] = len(created)
    # Stale .pyc from earlier runs, if any, are reported but not deleted.
    stale = sorted(p for p in after
                   if Path(p).name.startswith(("d2_04_train_fedavg", "d2_05_train_fedprox")))
    ctx["report"]["stale_trainer_pyc_present"] = stale


def check_18_gate_uses_verified_class_weights(ctx) -> None:
    """18. The RunPod gate verifies the real saved class weights, not just a recomputation."""
    gate = function_source(ctx["tree05a"], "runpod_gate")

    verified = [n for n in ast.walk(gate) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "load_and_verify_fedavg_class_weights"]
    assert verified, \
        "runpod_gate() does not call load_and_verify_fedavg_class_weights()"
    assert any(getattr(n.func.value, "id", None) == "m05" for n in verified), \
        "runpod_gate() does not verify the class weights through the trainer module"

    # It must also confirm equality against the d2_04 recomputation.
    recomputed = [n for n in ast.walk(gate) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "class_weights_full"
                  and getattr(n.func.value, "id", None) == "m04"]
    assert recomputed, "runpod_gate() does not recompute the d2_04 class weights for comparison"
    equality_asserts = [n for n in ast.walk(gate) if isinstance(n, ast.Assert)
                        and isinstance(n.test, ast.Call)
                        and isinstance(n.test.func, ast.Attribute)
                        and n.test.func.attr == "array_equal"]
    assert equality_asserts, "runpod_gate() does not assert exact class-weight equality"

    # The gate must not train on a merely recomputed vector: the name passed to
    # run() must be the one returned by the verification call.
    assigns = [n for n in ast.walk(gate) if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == "load_and_verify_fedavg_class_weights"]
    assert len(assigns) == 1, "expected one verified class-weight assignment in the gate"
    target = assigns[0].targets[0]
    names = [e.id for e in target.elts] if isinstance(target, ast.Tuple) else [target.id]
    assert names[0] == "weights", \
        f"the verified vector is bound to {names[0]!r}, but the gate trains with 'weights'"
    run_calls = [n for n in ast.walk(gate) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "run"]
    assert run_calls, "runpod_gate() does not call run()"
    for call in run_calls:
        passed = [a.id for a in call.args if isinstance(a, ast.Name)]
        assert "weights" in passed, \
            f"a gate run() call does not receive the verified weights: {passed}"
    ctx["report"]["gate_uses_verified_weights"] = True


def check_19_manifest_allowlist_excludes_holdout(ctx) -> None:
    """19. The integrity manifest covers only the four train/validation arrays."""
    expected_names = {"X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"}
    names = [f.name for f in GUARDED_FILES]
    assert len(names) == len(set(names)) == 4, f"the allowlist is not four unique files: {names}"
    assert set(names) == expected_names, \
        f"the allowlist is {sorted(names)}, expected {sorted(expected_names)}"
    assert not any("test" in n.lower() for n in names), \
        f"the allowlist contains a held-out filename: {names}"
    assert all(f.parent == D2_PROCESSED for f in GUARDED_FILES), \
        "an allowlisted file is not in the Dataset-2 processed directory"

    # The processed directory must not be guarded recursively, directly or via an
    # ancestor: either would make rglob enumerate the held-out arrays.
    processed = D2_PROCESSED.resolve()
    for d in GUARDED_DIRS:
        resolved = d.resolve()
        assert resolved != processed, \
            f"{d} is guarded recursively; rglob would enumerate the held-out arrays"
        assert resolved not in processed.parents, \
            f"{d} is an ancestor of {D2_PROCESSED}; rglob would reach the held-out arrays"

    # Behavioural: enumerate exactly what manifest() would hash, without hashing it.
    targets = manifest_targets(GUARDED_DIRS, GUARDED_FILES)
    offenders = [str(p) for p in targets if "test" in p.name.lower()]
    assert not offenders, f"the manifest would hash held-out arrays: {offenders}"

    from_processed = sorted(p.name for p in targets if p.parent.resolve() == processed)
    assert set(from_processed) <= expected_names, \
        f"the manifest reaches unexpected processed files: {from_processed}"

    # manifest() must hash exactly manifest_targets(), so the proof above binds it.
    manifest_node = function_source(ctx["tree05a"], "manifest")
    calls = [n for n in ast.walk(manifest_node) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "manifest_targets"]
    assert calls, "manifest() does not derive its file set from manifest_targets()"
    reads = [n for n in ast.walk(manifest_node) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "rglob"]
    assert not reads, "manifest() enumerates directories directly instead of via the allowlist"

    ctx["report"]["manifest_allowlist"] = sorted(names)
    ctx["report"]["manifest_processed_files"] = from_processed
    ctx["report"]["manifest_target_count"] = len(targets)


CHECKS = [
    (1, "proximal gradient = mu * (w - w_ref)", check_01_proximal_gradient),
    (2, "proximal norm is exactly 0 at the reference", check_02_zero_norm_at_reference),
    (3, "proximal norm > 0 after perturbation; reference intact", check_03_positive_norm_after_perturbation),
    (4, "mu=0 contributes an exact zero", check_04_mu_zero_contributes_exact_zero),
    (5, "mu=0 local update matches d2_04 FedAvg (synthetic)", check_05_mu_zero_matches_fedavg),
    (6, "server reference fixed, detached, from the current round", check_06_reference_fixed_during_update),
    (7, "aggregation is sample-size weighted, matches d2_04", check_07_aggregation_matches_fedavg),
    (8, "no held-out array references", check_08_no_test_reference),
    (9, "output roots isolated from FedAvg", check_09_output_roots_isolated),
    (10, "static contract shared with d2_04", check_10_static_contract),
    (11, "production mu frozen at 1e-5, not CLI-selectable", check_11_production_mu_frozen),
    (12, "FedAvg initial checkpoint mandatory and verified", check_12_fedavg_initial_checkpoint_mandatory),
    (13, "no FedProx initial-global-model file", check_13_no_fedprox_initial_model),
    (14, "FedAvg class weights mandatory and exact", check_14_fedavg_class_weights_mandatory),
    (15, "config records the frozen mu transfer policy", check_15_config_provenance),
    (16, "prerequisites verified before the first production write", check_16_prerequisites_before_first_write),
    (17, "guarded imports write no bytecode outside VERIF_ROOT", check_17_no_bytecode_outside_verif_root),
    (18, "gate uses the verified saved class weights", check_18_gate_uses_verified_class_weights),
    (19, "manifest allowlist excludes every held-out array", check_19_manifest_allowlist_excludes_holdout),
]


# ======================= RunPod integration/stability gate ================== #
def manifest_targets(dirs, files) -> list[Path]:
    """Exact set of files manifest() will hash.

    Enumerated separately so the allowlist can be proven correct without hashing
    anything: the guarded directories recursively, plus the explicitly named
    train/validation files. No held-out array is reachable from either source.
    """
    targets = []
    for d in dirs:
        if not d.exists():
            continue
        targets.extend(sorted(p for p in d.rglob("*") if p.is_file()))
    targets.extend(f for f in files if f.is_file())
    return targets


def manifest(dirs, files) -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in manifest_targets(dirs, files)}


def exact_array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=True))


def checkpoint_compare(path_a: Path, path_b: Path) -> tuple[bool, float]:
    a = torch.load(path_a, map_location="cpu")
    b = torch.load(path_b, map_location="cpu")
    if a.keys() != b.keys():
        return False, float("inf")
    exact = all(torch.equal(a[k], b[k]) for k in a)
    diff = max(float((a[k].to(torch.float64) - b[k].to(torch.float64)).abs().max()) for k in a)
    return exact, diff


def runpod_gate(ctx) -> int:
    """REAL-path mu=0 equivalence gate on the real Dataset-2 partitions.

    Expensive; opt-in only. Never a substitute for check 5. Writes only under
    VERIF_ROOT and proves the research roots are unchanged.
    """
    m04, m05 = ctx["m04"], ctx["m05"]
    print(f"\n=== RunPod gate: real-path mu=0 equivalence "
          f"({GATE_ROUNDS} rounds, cases {GATE_CASES}) ===", flush=True)

    before = manifest(GUARDED_DIRS, GUARDED_FILES)
    device = m05.get_device()
    print(f"gate device: {device}", flush=True)

    avg_res, avg_mod = GATE_RESULTS / "fedavg", GATE_MODELS / "fedavg"
    prox_res, prox_mod = GATE_RESULTS / "fedprox", GATE_MODELS / "fedprox"
    for d in (avg_res, avg_mod, prox_res, prox_mod):
        d.mkdir(parents=True, exist_ok=True)

    y_train = np.load(m05.PROCESSED_DIR / "y_train.npy")
    class_counts = np.bincount(y_train, minlength=m05.NUM_CLASSES)

    # The gate must exercise the REAL saved FedAvg class-weight vector through the
    # trainer's own mandatory verification, not merely recompute one. The returned
    # vector is then independently confirmed to equal the d2_04 recomputation, so
    # both the stored artefact and the two formulas are covered.
    weights, weights_sha = m05.load_and_verify_fedavg_class_weights(y_train)
    d2_04_weights = m04.class_weights_full(y_train).astype(np.float32)
    assert np.array_equal(weights, d2_04_weights), \
        "the verified saved FedAvg class weights differ from the d2_04 recomputation"
    print(f"gate class weights: verified saved vector sha256={weights_sha}; "
          "equals the d2_04 recomputation exactly", flush=True)

    # The gate uses the authoritative FedAvg initial state, verified by the trainer.
    initial_state, _ = m05.load_and_verify_fedavg_initial_state(device)

    resident = None
    if device.type == "cuda":
        resident = {
            "x_train": torch.from_numpy(np.load(m05.PROCESSED_DIR / "X_train.npy")).to(torch.float32).to(device),
            "y_train": torch.from_numpy(np.load(m05.PROCESSED_DIR / "y_train.npy")).to(torch.long).to(device),
            "x_val": torch.from_numpy(np.load(m05.PROCESSED_DIR / "X_val.npy")).to(torch.float32).to(device),
            "y_val": torch.from_numpy(np.load(m05.PROCESSED_DIR / "y_val.npy")).to(torch.long).to(device),
        }
        val_loader = m05.ResidentValLoader(resident["x_val"], resident["y_val"], 4096)
    else:
        val_loader = DataLoader(
            m05.FullDataset(m05.PROCESSED_DIR / "X_val.npy", m05.PROCESSED_DIR / "y_val.npy"),
            batch_size=4096, shuffle=False, num_workers=0)

    failures = []
    for name, module, results_dir, models_dir in (
            ("fedavg", m04, avg_res, avg_mod), ("fedprox", m05, prox_res, prox_mod)):
        module.RESULTS_DIR = results_dir
        module.MODELS_DIR = models_dir
        module.MAX_ROUNDS = GATE_ROUNDS
        assert str(module.RESULTS_DIR).startswith(str(GATE_RESULTS)), f"{name} results redirect failed"
        assert str(module.MODELS_DIR).startswith(str(GATE_MODELS)), f"{name} models redirect failed"

    frag = m05.mu_fragment(0.0)
    for seed, condition in GATE_CASES:
        part_avg = m04.load_partition(seed, condition, y_train, class_counts)
        part_prox = m05.load_partition(seed, condition, y_train, class_counts)
        assert part_avg["sizes"] == part_prox["sizes"], "client sizes differ between modules"

        m04.run(seed, condition, part_avg, initial_state, weights, val_loader, device, resident)
        m05.run(seed, condition, 0.0, part_prox, initial_state, weights, val_loader, device, resident)

        avg_tag = f"fedavg_k{m04.NUM_CLIENTS}_seed{seed}_{condition}"
        prox_tag = f"fedprox_k{m05.NUM_CLIENTS}_mu{frag}_seed{seed}_{condition}"

        h_avg = pd.read_csv(avg_res / f"history_{avg_tag}.csv")
        h_prox = pd.read_csv(prox_res / f"history_{prox_tag}.csv")
        shared = [c for c in h_avg.columns if c in h_prox.columns and "seconds" not in c]
        hist_exact = exact_array_equal(h_avg[shared].to_numpy(dtype=float),
                                       h_prox[shared].to_numpy(dtype=float))
        prox_cols = ["round_proximal_penalty"] + [
            c for c in h_prox.columns if c.startswith("train_proximal_penalty_client_")]
        max_prox = float(np.max(np.abs(h_prox[prox_cols].to_numpy(dtype=float))))

        best_exact, d_best = checkpoint_compare(avg_mod / f"best_{avg_tag}.pt",
                                                prox_mod / f"best_{prox_tag}.pt")
        final_exact, d_final = checkpoint_compare(avg_mod / f"final_{avg_tag}.pt",
                                                  prox_mod / f"final_{prox_tag}.pt")
        case_ok = hist_exact and best_exact and final_exact and max_prox == 0.0
        if not case_ok:
            failures.append((seed, condition))
        print(f"  seed{seed}/{condition}: history_exact={hist_exact} best_exact={best_exact} "
              f"final_exact={final_exact} max_prox={max_prox:.2e} "
              f"d_best={d_best:.2e} d_final={d_final:.2e} "
              f"n_shared={len(shared)} -> {'PASS' if case_ok else 'FAIL'}", flush=True)

    after = manifest(GUARDED_DIRS, GUARDED_FILES)
    research_ok = before == after
    print(f"  research artefacts unchanged: {'PASS' if research_ok else 'FAIL'}", flush=True)
    ok = not failures and research_ok
    print(f"RUNPOD GATE: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Dataset-2 FedProx trainer.")
    parser.add_argument("--runpod-gate", action="store_true",
                        help="additionally run the expensive real-path mu=0 equivalence gate")
    parser.add_argument("--list", action="store_true", help="list checks and exit")
    args = parser.parse_args()

    if args.list:
        for check_id, title, _ in CHECKS:
            print(f"{check_id:2d}. {title}")
        print(" G. RunPod real-path mu=0 equivalence gate  [opt-in: --runpod-gate]")
        return 0

    assert DEVICE.type == "cpu", "the default suite must run on CPU"
    torch.set_num_threads(1)  # stay out of the way of the active FedAvg run
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)

    src04 = FEDAVG_SCRIPT.read_text()
    src05 = FEDPROX_SCRIPT.read_text()
    src05a = Path(__file__).read_text()
    context = {
        "m04": load_guarded_module(FEDAVG_SCRIPT, "d2_04_fedavg"),
        "m05": load_guarded_module(FEDPROX_SCRIPT, "d2_05_fedprox"),
        "src04": src04, "src05": src05, "src05a": src05a,
        "sq04": "".join(src04.split()), "sq05": "".join(src05.split()),
        "tree04": ast.parse(src04, filename=str(FEDAVG_SCRIPT)),
        "tree05": ast.parse(src05, filename=str(FEDPROX_SCRIPT)),
        "tree05a": ast.parse(src05a, filename=str(Path(__file__).name)),
        "report": {},
    }

    print(f"verifier: device={DEVICE} torch={torch.__version__}")
    print(f"trainer={FEDPROX_SCRIPT}  reference={FEDAVG_SCRIPT}")
    print(f"fixtures under {FIXTURE_ROOT}")
    print("default suite executes no Dataset-2 round, partition or production matrix\n", flush=True)

    passed, failed = 0, []
    for check_id, title, function in CHECKS:
        try:
            function(context)
        except Exception as error:  # noqa: BLE001 - the verifier reports, it does not raise
            failed.append((check_id, title, error))
            print(f"[{check_id:2d}] FAIL  {title}\n      {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
        else:
            passed += 1
            print(f"[{check_id:2d}] PASS  {title}", flush=True)

    print(f"\npassed={passed} failed={len(failed)}")
    for key, value in context["report"].items():
        print(f"  {key}: {value}")
    for check_id, title, error in failed:
        print(f"  FAILED {check_id:2d}: {title} -> {type(error).__name__}: {error}")
    print(f"DEFAULT SUITE: {'PASS' if not failed else 'FAIL'}")

    gate_status = 0
    if args.runpod_gate:
        if failed:
            print("\nRefusing to run the RunPod gate: the default suite failed.")
            return 1
        gate_status = runpod_gate(context)

    return 1 if (failed or gate_status) else 0


if __name__ == "__main__":
    raise SystemExit(main())
