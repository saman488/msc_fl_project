"""
Independent verifier for 35_train_final_scaffold_37f.py (Dataset-1, 37-feature, K=5).

This script tests the SCAFFOLD runner; it does not reuse it as its own oracle.
Wherever a numerical claim is made, the expected value is recomputed here from the
algorithm definition with explicit operands and explicit signs, and only then
compared against what script 35 produces. Where a claim is structural rather than
numerical - "this client control is written back", "this snapshot is taken once
per round outside the client loop", "controls are rebuilt per run" - it is checked
against the parsed abstract syntax tree of script 35, so the check binds to the
real code path rather than to a re-implementation of it.

Script 35's own expressions are exercised by extracting them from its source with
`ast` and evaluating them against synthetic inputs. That keeps the tested code the
code that actually runs, while the expected values remain independently derived.

Scope and safety
----------------
Everything here runs on CPU with tiny synthetic tensors. The verifier never selects
MPS, never touches data/processed_37f, the partitions, the validation set, or the
held-out arrays, and never writes any file. Importing scripts 32 and 35 executes
only their module-level definitions; both guard their entry point behind
`if __name__ == "__main__"`, so importing them starts no training.

Three checks (3, 4, 5) take a small number of real optimizer steps on synthetic
data, because gradient correction and step counting cannot be verified without
stepping. They are marked as training checks and can be skipped with --no-train
while a heavy job is occupying the machine.

Check 15, first-round FedAvg equivalence, is PREPARED but not part of the default
run: it is only executed when --run-fedavg-equivalence is passed explicitly.

Usage
-----
    python3 35a_verify_scaffold_37f.py                       # checks 1-14
    python3 35a_verify_scaffold_37f.py --no-train            # skips 3, 4, 5
    python3 35a_verify_scaffold_37f.py --run-fedavg-equivalence   # adds check 15
    python3 35a_verify_scaffold_37f.py --list                # list checks only
"""

from pathlib import Path
import argparse
import ast
import copy
import importlib.util
import inspect
import re
import sys
import traceback

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCAFFOLD_PATH = Path("35_train_final_scaffold_37f.py")
FEDAVG_PATH = Path("32_train_final_fedavg_37f.py")

# CPU only. The verifier must never claim an accelerator that a running experiment
# is using, and every check here is small enough that CPU is the correct choice.
DEVICE = torch.device("cpu")

# Tolerances. The arithmetic under test is a handful of elementwise float32 ops, so
# agreement should be near-exact; EXACT_TOL is deliberately tight.
EXACT_TOL = 1e-6
# A wrong-sign or wrong-weighting variant must miss by far more than this.
WRONG_ANSWER_MARGIN = 1e-3


# ----------------------------- module loading ------------------------------- #
def load_module(path: Path, name: str):
    """Import a numerically-named script as a module without running its main()."""
    assert path.exists(), f"required script not found: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_entry_point_guarded(source: str, label: str) -> None:
    """A module we import must not start work at import time."""
    assert 'if __name__ == "__main__":' in source, f"{label}: no __main__ guard; import would run it"


# ------------------------- source / AST extraction -------------------------- #
def squeeze(source: str) -> str:
    """All whitespace removed, so source-contract checks survive reformatting."""
    return re.sub(r"\s+", "", source)


def find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name}() not found in script 35")


def name_assigns(scope: ast.AST, target_name: str) -> list[ast.Assign]:
    """Every `target_name = ...` assignment anywhere inside scope."""
    out = []
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == target_name:
                    out.append(node)
    return out


def sole_name_assign(scope: ast.AST, target_name: str) -> ast.Assign:
    hits = name_assigns(scope, target_name)
    assert len(hits) == 1, (
        f"expected exactly one assignment to `{target_name}`, found {len(hits)}; "
        "the verifier must not guess which one is under test"
    )
    return hits[0]


def eval_scaffold_expression(tree: ast.AST, func_name: str, target_name: str, namespace: dict):
    """Evaluate script 35's own right-hand side for `target_name` on our inputs.

    The expression object comes from script 35's parsed source, so this exercises
    the shipped code rather than a copy of it; the namespace and the expected value
    are constructed here.
    """
    assign = sole_name_assign(find_function(tree, func_name), target_name)
    expr = ast.Expression(body=assign.value)
    ast.fix_missing_locations(expr)
    return eval(compile(expr, f"<35:{func_name}:{target_name}>", "eval"), namespace)


def for_loop_over(scope: ast.AST, target_name: str) -> ast.For:
    for node in ast.walk(scope):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) \
                and node.target.id == target_name:
            return node
    raise AssertionError(f"no `for {target_name} in ...` loop found")


def calls_to(scope: ast.AST, func_name: str) -> list[ast.Call]:
    return [n for n in ast.walk(scope)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == func_name]


# ------------------------------ tensor helpers ------------------------------ #
def patterned_controls(reference: dict, seed: int, scale: float, shift: float) -> dict:
    """Deterministic, non-zero, sign-mixed synthetic control tensors.

    Deliberately not constant and not proportional to any other control set used in
    the same check, so a swapped sign or a swapped operand cannot coincidentally
    reproduce the correct answer.
    """
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for name, tensor in reference.items():
        values = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
        out[name] = (values * scale + shift).to(DEVICE)
    return out


def max_abs_diff(a: dict, b: dict) -> float:
    """Independent max |a - b| over a dict of tensors; does not use script 35."""
    assert a.keys() == b.keys(), "max_abs_diff: key mismatch"
    return max(float((a[k] - b[k]).abs().max().item()) for k in a)


def assert_dicts_close(actual: dict, expected: dict, tol: float, label: str) -> None:
    diff = max_abs_diff(actual, expected)
    assert diff <= tol, f"{label}: max abs difference {diff:.3e} exceeds {tol:.3e}"


def assert_dicts_differ(a: dict, b: dict, margin: float, label: str) -> None:
    """Negative control: a deliberately wrong variant must not pass as correct."""
    diff = max_abs_diff(a, b)
    assert diff > margin, (
        f"{label}: the wrong variant differs by only {diff:.3e}, so this check "
        "could not detect the error it is meant to catch"
    )


def tiny_state(reference_state: dict, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {k: torch.randn(v.shape, generator=generator, dtype=torch.float32)
            for k, v in reference_state.items()}


def synthetic_batch(n_rows: int, input_dim: int, num_classes: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n_rows, input_dim, generator=generator, dtype=torch.float32)
    labels = torch.randint(0, num_classes, (n_rows,), generator=generator)
    return features, labels


def class_weight_vector(num_classes: int) -> torch.Tensor:
    # Unequal, strictly positive, mirroring the weighted loss actually used.
    return torch.linspace(0.5, 2.0, num_classes, dtype=torch.float32)


# ================================== CHECKS ================================== #
def check_01_shared_contract(ctx) -> None:
    """1. Shared contract with script 32."""
    m35, m32 = ctx["m35"], ctx["m32"]

    # Expected values are stated here independently, so a value that is wrong in
    # both scripts still fails rather than agreeing with itself.
    expected = {
        "PROCESSED_DIR": Path("data/processed_37f"),
        "PART_ROOT": Path("data/fl_clients/final_partitions/k_5"),
        "INPUT_DIM": 37,
        "NUM_CLASSES": 10,
        "NUM_CLIENTS": 5,
        "LOCAL_EPOCHS": 1,
        "MAX_ROUNDS": 40,
        "TRAIN_SEED": 42,
        "BATCH_SIZE": 4096,
        "LR": 0.1,
        "MOMENTUM": 0.0,
        "WEIGHT_DECAY": 0.0,
        "RELOAD_F1_TOL": 1e-4,
        "PARTITION_SEEDS": [42, 43, 44],
        "CONDITIONS": ["iid", "alpha_0p1", "alpha_0p5", "alpha_1p0"],
    }
    for key, value in expected.items():
        got35, got32 = getattr(m35, key), getattr(m32, key)
        assert got35 == value, f"35.{key} = {got35!r}, expected {value!r}"
        assert got32 == value, f"32.{key} = {got32!r}, expected {value!r}"

    assert len(m35.PARTITION_SEEDS) * len(m35.CONDITIONS) == 12, "expected a 3 x 4 = 12 run matrix"

    # Architecture: shapes stated independently, then cross-checked against 32.
    net35 = m35.MLPMultiClassClassifier(37, 10)
    net32 = m32.MLPMultiClassClassifier(37, 10)
    shapes35 = {k: tuple(v.shape) for k, v in net35.state_dict().items()}
    shapes32 = {k: tuple(v.shape) for k, v in net32.state_dict().items()}
    expected_shapes = {
        "network.0.weight": (128, 37), "network.0.bias": (128,),
        "network.3.weight": (64, 128), "network.3.bias": (64,),
        "network.6.weight": (10, 64), "network.6.bias": (10,),
    }
    assert shapes35 == expected_shapes, f"35 architecture shapes {shapes35}"
    assert shapes32 == expected_shapes, f"32 architecture shapes {shapes32}"
    assert str(net35) == str(net32), "module structure differs between 35 and 32"
    assert squeeze(str(net35)).count("Dropout(p=0.2") == 2, "expected two Dropout(p=0.2) layers"

    # Exact read-only FedAvg input paths, stated literally and cross-checked.
    assert m35.INIT_PATH == Path("models/final_fedavg_k5_37f/initial_global_model.pt")
    assert m35.FEDAVG_CLASS_WEIGHTS_PATH == Path("results/final_fedavg_k5_37f/class_weights.npy")
    assert m35.INIT_PATH == m32.MODELS_DIR / "initial_global_model.pt", "initial-state path differs from 32's"
    assert m35.FEDAVG_CLASS_WEIGHTS_PATH == m32.RESULTS_DIR / "class_weights.npy", \
        "class-weight path differs from 32's"

    # Outputs must not collide with the FedAvg roots.
    assert m35.RESULTS_DIR != m32.RESULTS_DIR, "35 would write into 32's results root"
    assert m35.MODELS_DIR != m32.MODELS_DIR, "35 would write into 32's models root"
    assert m35.RESULTS_DIR == Path("results/final_scaffold_k5_37f")
    assert m35.MODELS_DIR == Path("models/final_scaffold_k5_37f")

    # Behavioural contract at source level, whitespace-insensitive.
    for label, squeezed in (("35", ctx["sq35"]), ("32", ctx["sq32"])):
        assert "torch.optim.SGD(local_model.parameters(),lr=LR,momentum=MOMENTUM," \
               "weight_decay=WEIGHT_DECAY)" in squeezed, f"{label}: optimizer construction differs"
        assert "forrndinrange(1,MAX_ROUNDS+1):" in squeezed, f"{label}: round loop differs"
        assert "for_inrange(LOCAL_EPOCHS):" in squeezed, f"{label}: local-epoch loop differs"
        assert "local_seed=TRAIN_SEED+rnd*100+client_id" in squeezed, f"{label}: seed scheme differs"
        assert "DataLoader(datasets[client_id],batch_size=BATCH_SIZE,shuffle=True," \
               "num_workers=0,generator=generator)" in squeezed, f"{label}: DataLoader differs"
        assert 'ifval["macro_f1"]>best["macro_f1"]:' in squeezed, f"{label}: selection is not macro-F1"
        assert '"selection_metric":"val_macro_f1"' in squeezed, f"{label}: selection metric not recorded"
        assert "abs(reloaded_macro_f1-best[" in squeezed, f"{label}: no best-checkpoint reload check"

    # The runner is cleared to train, so the execution gate must be open.
    assert m35.SCAFFOLD_EXECUTION_ENABLED is True, "SCAFFOLD_EXECUTION_ENABLED is not True"
    assert m35.CONTROL_VARIATES_IMPLEMENTED is True, "CONTROL_VARIATES_IMPLEMENTED is not True"


def check_02_control_initialisation(ctx) -> None:
    """2. Control initialisation: keys, shapes, dtypes, zeros, no aliasing."""
    m35 = ctx["m35"]
    model = m35.MLPMultiClassClassifier(37, 10).to(DEVICE)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert len(trainable) == 6, f"expected 6 trainable parameter tensors, got {len(trainable)}"

    server = m35.init_controls(model, DEVICE)
    clients = [m35.init_controls(model, DEVICE) for _ in range(m35.NUM_CLIENTS)]
    assert len(clients) == 5, "expected five client controls"

    for label, control in [("server", server)] + [(f"client{i}", c) for i, c in enumerate(clients)]:
        assert set(control.keys()) == set(trainable.keys()), \
            f"{label}: control keys do not match the trainable named parameters"
        for name, param in trainable.items():
            tensor = control[name]
            assert tensor.shape == param.shape, f"{label}.{name}: shape {tuple(tensor.shape)}"
            assert tensor.dtype == param.dtype, f"{label}.{name}: dtype {tensor.dtype}"
            assert tensor.device.type == param.device.type, f"{label}.{name}: device mismatch"
            assert not tensor.requires_grad, f"{label}.{name}: control is attached to autograd"
            # Zero verified independently of any helper in script 35.
            assert int(torch.count_nonzero(tensor).item()) == 0, f"{label}.{name}: not zero-initialised"

    # Distinct dict objects.
    all_dicts = [server] + clients
    assert len({id(d) for d in all_dicts}) == 6, "control dicts are not six distinct objects"

    # No tensor aliasing anywhere across the six control sets.
    seen = {}
    for label, control in [("server", server)] + [(f"client{i}", c) for i, c in enumerate(clients)]:
        for name, tensor in control.items():
            pointer = tensor.data_ptr()
            assert pointer not in seen, (
                f"{label}.{name} shares storage with {seen[pointer]}: controls are aliased"
            )
            seen[pointer] = f"{label}.{name}"

    # Behavioural independence: an in-place write to one control must not leak.
    probe_name = next(iter(trainable))
    clients[0][probe_name].add_(7.5)
    assert float(clients[0][probe_name].abs().max().item()) > 0.0, "probe write did not take effect"
    for label, control in [("server", server)] + [(f"client{i}", c) for i, c in enumerate(clients[1:], 1)]:
        assert int(torch.count_nonzero(control[probe_name]).item()) == 0, \
            f"{label} changed when client0 was mutated: controls are not independent"


def _one_epoch_with_reference(ctx, c_old: dict, c_i_old: dict, n_rows: int, batch_size: int):
    """Run script 35's training loop once and capture everything needed to redo it.

    Returns (start_state, params_after, raw_grads, stats, step_count). raw_grads are
    computed here by an independent forward/backward from the same starting weights
    under the same RNG state, so the dropout masks match and the comparison is exact.
    """
    m35 = ctx["m35"]
    features, labels = synthetic_batch(n_rows, m35.INPUT_DIM, m35.NUM_CLASSES, seed=1234)
    # A dedicated, fixed-seed generator. Without it, creating the DataLoader iterator
    # draws its base seed from the global RNG at iteration time - that is, inside
    # train_one_epoch, after the RNG state is captured and before the first forward -
    # which would desynchronise the Dropout stream that checks 3 and 4 reproduce.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(4242)
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size,
                        shuffle=False, num_workers=0, generator=loader_generator)
    criterion = nn.CrossEntropyLoss(weight=class_weight_vector(m35.NUM_CLASSES).to(DEVICE))

    torch.manual_seed(7)
    model = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).to(DEVICE)
    # Built before the RNG state is captured, because constructing a model draws
    # from the global generator.
    reference = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).to(DEVICE)

    start_state = copy.deepcopy(model.state_dict())
    reference.load_state_dict(copy.deepcopy(start_state))

    optimizer = torch.optim.SGD(model.parameters(), lr=m35.LR,
                                momentum=m35.MOMENTUM, weight_decay=m35.WEIGHT_DECAY)
    step_count = {"n": 0}
    inner_step = optimizer.step

    def counting_step(*args, **kwargs):
        step_count["n"] += 1
        return inner_step(*args, **kwargs)

    optimizer.step = counting_step

    rng_state = torch.get_rng_state()
    stats = m35.train_one_epoch(model, loader, criterion, optimizer, DEVICE, c_old, c_i_old)
    params_after = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    # Independent recomputation of the raw (uncorrected) gradients of the same
    # single pass, from the same weights and the same dropout stream.
    torch.set_rng_state(rng_state)
    reference.train()
    reference.zero_grad(set_to_none=True)
    loss = criterion(reference(features.to(DEVICE)), labels.to(DEVICE))
    loss.backward()
    raw_grads = {n: p.grad.detach().clone() for n, p in reference.named_parameters() if p.requires_grad}

    return start_state, params_after, raw_grads, stats, step_count["n"]


def check_03_gradient_correction(ctx) -> None:
    """3. Gradient correction equals raw_grad - c_i + c (non-zero synthetic controls)."""
    m35 = ctx["m35"]
    model_probe = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).to(DEVICE)
    trainable = {n: p.detach() for n, p in model_probe.named_parameters() if p.requires_grad}

    # Deliberately non-zero, mutually unrelated, sign-mixed. c_i is not c, not -c,
    # and neither is a scalar multiple of the other.
    c_i_old = patterned_controls(trainable, seed=101, scale=0.50, shift=+0.30)
    c_old = patterned_controls(trainable, seed=202, scale=0.70, shift=-0.40)
    for name in trainable:
        assert float(c_i_old[name].abs().min().item()) >= 0.0
        assert max_abs_diff({name: c_i_old[name]}, {name: c_old[name]}) > WRONG_ANSWER_MARGIN, \
            "synthetic c_i and c are too close to distinguish a sign error"

    start_state, params_after, raw_grads, stats, _ = _one_epoch_with_reference(
        ctx, c_old, c_i_old, n_rows=8, batch_size=8)
    assert stats["local_steps"] == 1, "this check needs exactly one optimizer step"

    # Plain SGD, momentum 0, weight decay 0: param_after = param_before - LR * grad,
    # so the gradient script 35 actually stepped with is recoverable exactly.
    implied_correction = {
        name: (start_state[name].to(DEVICE) - params_after[name]) / m35.LR
        for name in params_after
    }
    # The independent expectation, written out with explicit signs.
    expected_correction = {
        name: raw_grads[name] - c_i_old[name] + c_old[name]
        for name in params_after
    }
    assert_dicts_close(implied_correction, expected_correction, EXACT_TOL,
                       "corrected gradient != raw_grad - c_i + c")

    # Negative controls: every plausible sign or operand error must be detectable.
    wrong_variants = {
        "raw + c_i - c (signs swapped)":
            {n: raw_grads[n] + c_i_old[n] - c_old[n] for n in params_after},
        "raw - c_i - c":
            {n: raw_grads[n] - c_i_old[n] - c_old[n] for n in params_after},
        "raw + c_i + c":
            {n: raw_grads[n] + c_i_old[n] + c_old[n] for n in params_after},
        "raw only (correction dropped)":
            {n: raw_grads[n].clone() for n in params_after},
    }
    for label, variant in wrong_variants.items():
        assert_dicts_differ(implied_correction, variant, WRONG_ANSWER_MARGIN,
                            f"negative control '{label}'")


def check_04_zero_control_equivalence(ctx) -> None:
    """4. With c = 0 and c_i = 0 the corrected gradient is the raw gradient."""
    m35 = ctx["m35"]
    model_probe = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).to(DEVICE)
    zeros_a = m35.init_controls(model_probe, DEVICE)
    zeros_b = m35.init_controls(model_probe, DEVICE)
    for control in (zeros_a, zeros_b):
        for tensor in control.values():
            assert int(torch.count_nonzero(tensor).item()) == 0, "zero-control setup is not zero"

    start_state, params_after, raw_grads, stats, _ = _one_epoch_with_reference(
        ctx, zeros_a, zeros_b, n_rows=8, batch_size=8)
    assert stats["local_steps"] == 1, "this check needs exactly one optimizer step"

    implied_correction = {
        name: (start_state[name].to(DEVICE) - params_after[name]) / m35.LR
        for name in params_after
    }
    assert_dicts_close(implied_correction, raw_grads, EXACT_TOL,
                       "zero controls did not leave the gradient unchanged")

    # The gradients must be genuinely non-zero, or this check proves nothing.
    largest = max(float(g.abs().max().item()) for g in raw_grads.values())
    assert largest > WRONG_ANSWER_MARGIN, f"raw gradients are ~0 ({largest:.3e}); check is vacuous"


def check_05_actual_step_counting(ctx) -> None:
    """5. Reported local_steps equals the real number of optimizer.step() calls."""
    m35 = ctx["m35"]
    model_probe = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).to(DEVICE)
    trainable = {n: p.detach() for n, p in model_probe.named_parameters() if p.requires_grad}
    c_i_old = patterned_controls(trainable, seed=303, scale=0.20, shift=+0.10)
    c_old = patterned_controls(trainable, seed=404, scale=0.30, shift=-0.20)

    n_rows, batch_size = 7, 2
    # Independent expectation: ceil(7 / 2) = 4, computed without consulting stats.
    expected_steps = -(-n_rows // batch_size)
    assert expected_steps == 4 and expected_steps > 1, "this check needs more than one batch"

    _, _, _, stats, observed_steps = _one_epoch_with_reference(
        ctx, c_old, c_i_old, n_rows=n_rows, batch_size=batch_size)

    assert observed_steps == expected_steps, \
        f"optimizer.step() was called {observed_steps} times, expected {expected_steps}"
    assert stats["local_steps"] == observed_steps, (
        f"reported local_steps {stats['local_steps']} != real step count {observed_steps}"
    )
    assert stats["n_batches"] == expected_steps, "batch count disagrees with the step count"
    assert stats["n_samples"] == n_rows, f"sample count {stats['n_samples']} != {n_rows}"
    # tau must not be a stand-in for any other quantity that happens to be around.
    assert stats["local_steps"] not in (n_rows, batch_size, 1), \
        "local_steps coincides with a size/batch value; the check cannot discriminate"

    # Source-level: the counter is incremented next to the step, not derived.
    assert "local_steps+=1" in ctx["sq35"], "local_steps is not incremented per step"
    assert "assertlocal_steps==n_batches" in ctx["sq35"], "step/batch consistency assertion missing"
    for forbidden in ("local_steps=len(", "local_steps=n_samples//", "local_steps=int(len("):
        assert forbidden not in ctx["sq35"], f"local_steps appears to be inferred: {forbidden}"


def check_06_option_ii_identity(ctx) -> None:
    """6. c_i_new = c_i_old - c_old + (x - y_i) / (tau_i * LR), and delta = c_i_new - c_i_old."""
    m35, tree = ctx["m35"], ctx["tree35"]
    model_probe = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).to(DEVICE)
    trainable = {n: p.detach() for n, p in model_probe.named_parameters() if p.requires_grad}

    c_i_old = patterned_controls(trainable, seed=11, scale=0.60, shift=+0.25)
    c_old = patterned_controls(trainable, seed=22, scale=0.45, shift=-0.35)
    x_params = patterned_controls(trainable, seed=33, scale=1.10, shift=+0.05)
    y_params = patterned_controls(trainable, seed=44, scale=0.90, shift=-0.15)
    tau = 13  # not 1, so a missing tau divisor changes the answer

    # Script 35's own `scale` expression, evaluated on our operands.
    scale = eval_scaffold_expression(tree, "run", "scale",
                                     {"local_steps": tau, "LR": m35.LR})
    assert abs(scale - (float(tau) * 0.1)) < 1e-12, f"scale = {scale}, expected tau * LR = {tau * 0.1}"

    # Script 35's own Option II expression, evaluated on our operands.
    c_i_new = eval_scaffold_expression(tree, "run", "c_i_new", {
        "c_i_old": c_i_old, "c_old": c_old,
        "x_params": x_params, "y_params": y_params, "scale": scale,
    })
    delta_c_i = eval_scaffold_expression(tree, "run", "delta_c_i", {
        "c_i_new": c_i_new, "c_i_old": c_i_old,
    })

    # Independent expectation, written out term by term.
    expected_c_i_new = {
        name: c_i_old[name] - c_old[name]
        + (x_params[name] - y_params[name]) / (float(tau) * m35.LR)
        for name in c_i_old
    }
    expected_delta = {name: expected_c_i_new[name] - c_i_old[name] for name in c_i_old}

    assert set(c_i_new.keys()) == set(c_i_old.keys()), "c_i_new keys changed"
    assert_dicts_close(c_i_new, expected_c_i_new, EXACT_TOL, "Option II c_i_new mismatch")
    assert_dicts_close(delta_c_i, expected_delta, EXACT_TOL, "delta_c_i mismatch")

    # Negative controls for the terms that are easy to get wrong.
    assert_dicts_differ(c_i_new,
                        {n: c_i_old[n] + c_old[n] + (x_params[n] - y_params[n]) / scale for n in c_i_old},
                        WRONG_ANSWER_MARGIN, "negative control 'c_i_old + c_old'")
    assert_dicts_differ(c_i_new,
                        {n: c_i_old[n] - c_old[n] + (y_params[n] - x_params[n]) / scale for n in c_i_old},
                        WRONG_ANSWER_MARGIN, "negative control '(y - x)' instead of '(x - y)'")
    assert_dicts_differ(c_i_new,
                        {n: c_i_old[n] - c_old[n] + (x_params[n] - y_params[n]) / m35.LR for n in c_i_old},
                        WRONG_ANSWER_MARGIN, "negative control 'tau omitted'")
    assert_dicts_differ(delta_c_i,
                        {n: c_i_new[n] - c_old[n] for n in c_i_old},
                        WRONG_ANSWER_MARGIN, "negative control 'delta against c_old'")

    # delta must be taken from the un-replaced c_i_old: the write-back happens later.
    client_loop = for_loop_over(for_loop_over(find_function(tree, "run"), "rnd"), "client_id")
    delta_line = sole_name_assign(client_loop, "delta_c_i").lineno
    writeback = [n for n in ast.walk(client_loop)
                 if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                         and t.value.id == "client_controls" for t in n.targets)]
    assert len(writeback) == 1, f"expected one client_controls write-back, found {len(writeback)}"
    assert writeback[0].lineno > delta_line, \
        "client_controls is overwritten before delta_c_i is computed from c_i_old"


def check_07_control_persistence(ctx) -> None:
    """7. A client's updated control is what that client reads in the next round."""
    m35, tree = ctx["m35"], ctx["tree35"]
    run_node = find_function(tree, "run")
    round_loop = for_loop_over(run_node, "rnd")
    client_loop = for_loop_over(round_loop, "client_id")

    # Structural: read from the persistent list, write back into the same slot.
    read_assign = sole_name_assign(client_loop, "c_i_old")
    assert isinstance(read_assign.value, ast.Subscript) \
        and isinstance(read_assign.value.value, ast.Name) \
        and read_assign.value.value.id == "client_controls", \
        "c_i_old is not read from the persistent client_controls list"

    writeback = [n for n in ast.walk(client_loop)
                 if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                         and t.value.id == "client_controls" for t in n.targets)]
    assert len(writeback) == 1, "expected exactly one write-back into client_controls"
    assert isinstance(writeback[0].value, ast.Name) and writeback[0].value.id == "c_i_new", \
        "client_controls is not updated with c_i_new"

    # Nothing may reset the controls inside the round loop.
    assert not calls_to(round_loop, "init_controls"), \
        "init_controls() is called inside the round loop: controls would not persist"

    # Numerical: two rounds driven through script 35's own extracted expressions.
    model_probe = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).to(DEVICE)
    trainable = {n: p.detach() for n, p in model_probe.named_parameters() if p.requires_grad}
    client_controls = [m35.init_controls(model_probe, DEVICE) for _ in range(m35.NUM_CLIENTS)]

    round1_new = []
    for client_id in range(m35.NUM_CLIENTS):
        c_i_old = client_controls[client_id]
        c_i_new = eval_scaffold_expression(tree, "run", "c_i_new", {
            "c_i_old": c_i_old,
            "c_old": m35.init_controls(model_probe, DEVICE),
            "x_params": patterned_controls(trainable, seed=500 + client_id, scale=1.0, shift=0.2),
            "y_params": patterned_controls(trainable, seed=600 + client_id, scale=1.0, shift=-0.2),
            "scale": 5.0,
        })
        client_controls[client_id] = c_i_new  # same write-back the AST check pinned
        round1_new.append(c_i_new)

    for client_id in range(m35.NUM_CLIENTS):
        c_i_old_round2 = client_controls[client_id]
        assert c_i_old_round2 is round1_new[client_id], \
            f"client {client_id}: round 2 did not read round 1's updated control object"
        assert max_abs_diff(c_i_old_round2, round1_new[client_id]) == 0.0
        nonzero = max(float(t.abs().max().item()) for t in c_i_old_round2.values())
        assert nonzero > WRONG_ANSWER_MARGIN, \
            f"client {client_id}: control is still ~0, so persistence is untestable here"


def check_08_independent_run_reset(ctx) -> None:
    """8. Each independent run rebuilds fresh zero controls."""
    m35, tree = ctx["m35"], ctx["tree35"]
    run_node = find_function(tree, "run")
    round_loop = for_loop_over(run_node, "rnd")

    # Both control sets are created in run()'s own body, before the round loop, so a
    # new (partition_seed, condition) run cannot inherit control state.
    top_level_server = [n for n in run_node.body
                        if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "server_control" for t in n.targets)]
    assert len(top_level_server) == 1, "server_control is not initialised once in run()'s body"
    assert calls_to(top_level_server[0], "init_controls"), "server_control is not built by init_controls()"
    assert top_level_server[0].lineno < round_loop.lineno, "server_control is initialised after the round loop"

    top_level_clients = [n for n in run_node.body
                         if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "client_controls" for t in n.targets)]
    assert len(top_level_clients) == 1, "client_controls is not initialised once in run()'s body"
    assert len(calls_to(top_level_clients[0], "init_controls")) == 1, \
        "client_controls is not built by init_controls()"
    assert top_level_clients[0].lineno < round_loop.lineno, "client_controls is initialised after the round loop"

    # No control state at module scope, which would be shared between runs.
    module_assigns = [n for n in ctx["tree35"].body if isinstance(n, ast.Assign)]
    for node in module_assigns:
        for target in node.targets:
            if isinstance(target, ast.Name):
                assert target.id not in ("server_control", "client_controls"), \
                    f"module-level control state `{target.id}` would leak between runs"

    # main() calls run() once per (partition_seed, condition), i.e. 12 fresh resets.
    assert "row=run(partition_seed,condition,part_cache[(partition_seed,condition)]," in ctx["sq35"], \
        "main() does not call run() per (partition_seed, condition)"

    # Behavioural: a second construction is a fresh, all-zero, non-aliasing set.
    model_probe = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).to(DEVICE)
    first = m35.init_controls(model_probe, DEVICE)
    probe_name = next(iter(first))
    first[probe_name].add_(3.0)
    second = m35.init_controls(model_probe, DEVICE)
    assert int(torch.count_nonzero(second[probe_name]).item()) == 0, \
        "a newly built control set is contaminated by the previous run"
    assert second[probe_name].data_ptr() != first[probe_name].data_ptr(), "new run aliases old control storage"


def check_09_same_server_control_within_round(ctx) -> None:
    """9. All five clients in a round receive the same pre-round c_old."""
    m35, tree = ctx["m35"], ctx["tree35"]
    run_node = find_function(tree, "run")
    round_loop = for_loop_over(run_node, "rnd")
    client_loop = for_loop_over(round_loop, "client_id")

    # c_old is snapshotted once per round, in the round body, outside the client loop.
    c_old_assigns = name_assigns(round_loop, "c_old")
    assert len(c_old_assigns) == 1, f"c_old is assigned {len(c_old_assigns)} times inside a round"
    assert c_old_assigns[0] in round_loop.body, "c_old is not snapshotted directly in the round body"
    assert c_old_assigns[0].lineno < client_loop.lineno, "c_old is snapshotted after the client loop starts"
    assert not name_assigns(client_loop, "c_old"), "c_old is reassigned inside the client loop"
    assert calls_to(c_old_assigns[0], "clone_controls"), "c_old is not a clone_controls() snapshot"

    # Every client's training call receives that same c_old binding.
    train_calls = calls_to(client_loop, "train_one_epoch")
    assert len(train_calls) == 1, f"expected one train_one_epoch() call site, found {len(train_calls)}"
    positional = train_calls[0].args
    assert len(positional) == 7, f"train_one_epoch() called with {len(positional)} positional args"
    assert isinstance(positional[5], ast.Name) and positional[5].id == "c_old", \
        "the 6th argument to train_one_epoch() is not c_old"
    assert isinstance(positional[6], ast.Name) and positional[6].id == "c_i_old", \
        "the 7th argument to train_one_epoch() is not c_i_old"

    # Behavioural: the snapshot is immune to later control writes during the round.
    model_probe = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).to(DEVICE)
    trainable = {n: p.detach() for n, p in model_probe.named_parameters() if p.requires_grad}
    server_control = patterned_controls(trainable, seed=77, scale=0.4, shift=0.1)
    c_old = m35.clone_controls(server_control)
    baseline = {n: t.clone() for n, t in c_old.items()}

    client_controls = [m35.init_controls(model_probe, DEVICE) for _ in range(m35.NUM_CLIENTS)]
    seen_by_client = []
    for client_id in range(m35.NUM_CLIENTS):
        seen_by_client.append(c_old)
        client_controls[client_id] = patterned_controls(trainable, seed=800 + client_id,
                                                        scale=1.0, shift=0.5)
        for name in server_control:
            server_control[name].add_(1.0)  # a later in-place server write must not leak
    assert all(seen is seen_by_client[0] for seen in seen_by_client), \
        "clients did not all see the same c_old object"
    assert max_abs_diff(c_old, baseline) == 0.0, "c_old changed during the round"


def check_10_server_control_update(ctx) -> None:
    """10. c_new = c_old + sum_k p_k * delta_c_i, with p_k = n_k / sum_j n_j."""
    m35, tree = ctx["m35"], ctx["tree35"]
    model_probe = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).to(DEVICE)
    trainable = {n: p.detach() for n, p in model_probe.named_parameters() if p.requires_grad}

    c_old = patterned_controls(trainable, seed=91, scale=0.55, shift=-0.20)
    deltas = [patterned_controls(trainable, seed=1000 + k, scale=0.8 + 0.2 * k, shift=0.1 * k - 0.2)
              for k in range(m35.NUM_CLIENTS)]
    # Deliberately unequal, and far from equal, so the size-weighted and uniform
    # aggregates cannot coincide.
    sizes = [100, 5000, 20, 900, 77]
    total = float(sum(sizes))
    weights = [n / total for n in sizes]

    # Two independent aggregations of the same deltas.
    weighted_agg = {n: sum(deltas[k][n] * weights[k] for k in range(len(deltas)))
                    for n in c_old}
    uniform_mean = {n: sum(d[n] for d in deltas) / float(len(deltas)) for n in c_old}
    assert_dicts_differ(weighted_agg, uniform_mean, WRONG_ANSWER_MARGIN,
                        "synthetic deltas/sizes make weighted and uniform aggregates indistinguishable")

    weighted_delta_c = m35.weighted_controls(deltas, weights)
    assert_dicts_close(weighted_delta_c, weighted_agg, EXACT_TOL,
                       "weighted_controls is not the sample-weighted aggregate")
    assert_dicts_differ(weighted_delta_c, uniform_mean, WRONG_ANSWER_MARGIN,
                        "weighted_controls matches the uniform mean")

    # The aggregator takes the weights explicitly.
    params = list(inspect.signature(m35.weighted_controls).parameters)
    assert params == ["control_list", "weights"], f"weighted_controls takes {params}"

    # Script 35's own server update expression, on our operands.
    c_new = eval_scaffold_expression(tree, "run", "c_new",
                                     {"c_old": c_old, "weighted_delta_c": weighted_delta_c})
    expected_c_new = {n: c_old[n] + weighted_agg[n] for n in c_old}
    assert_dicts_close(c_new, expected_c_new, EXACT_TOL, "server control update mismatch")
    assert_dicts_differ(c_new, {n: c_old[n] + uniform_mean[n] for n in c_old},
                        WRONG_ANSWER_MARGIN, "negative control 'uniform server update'")

    # Production must aggregate the control deltas with the run-level model weights.
    assign = sole_name_assign(find_function(tree, "run"), "weighted_delta_c")
    call = assign.value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
        and call.func.id == "weighted_controls", \
        "weighted_delta_c is not produced by weighted_controls()"
    call_args = [a.id for a in call.args if isinstance(a, ast.Name)]
    assert call_args == ["client_delta_controls", "agg_weights"], \
        f"weighted_controls called with {call_args}"
    assert "agg_weights=[n/total_sizeforninsizes]" in ctx["sq35"], \
        "agg_weights are not derived from the client sizes"
    assert "total_size=float(sum(sizes))" in ctx["sq35"], \
        "total_size is not the sum of the client sizes"

    # The recorded delta is c_new - c_old, formed from the two states.
    server_control_delta = eval_scaffold_expression(tree, "run", "server_control_delta",
                                                    {"c_new": c_new, "c_old": c_old})
    assert_dicts_close(server_control_delta, {n: c_new[n] - c_old[n] for n in c_old},
                       EXACT_TOL, "server_control_delta != c_new - c_old")


def check_11_control_invariant(ctx) -> None:
    """11. After the update, c_new equals the sample-weighted aggregate of the c_i_new."""
    m35, tree = ctx["m35"], ctx["tree35"]
    model_probe = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).to(DEVICE)
    trainable = {n: p.detach() for n, p in model_probe.named_parameters() if p.requires_grad}
    k = m35.NUM_CLIENTS

    # Deliberately unequal positive client weights.
    sizes = [100, 5000, 20, 900, 77]
    total = float(sum(sizes))
    weights = [n / total for n in sizes]
    assert len(set(weights)) == k and all(w > 0 for w in weights), "client weights must be unequal and positive"

    # Enter the round in a state the weighted invariant already holds in:
    # sum_i p_i * c_i_old = c_old. Solve for the largest-weight client so the
    # construction does not amplify float error.
    c_old = patterned_controls(trainable, seed=1201, scale=0.30, shift=0.15)
    solved = int(max(range(k), key=lambda i: weights[i]))
    others = [i for i in range(k) if i != solved]
    partials = {i: patterned_controls(trainable, seed=1300 + i, scale=0.5, shift=0.1 * i - 0.2)
                for i in others}
    remainder = {n: (c_old[n] - sum(partials[i][n] * weights[i] for i in others)) / weights[solved]
                 for n in c_old}
    c_i_old_all = [partials[i] if i != solved else remainder for i in range(k)]
    weighted_start = {n: sum(c_i_old_all[i][n] * weights[i] for i in range(k)) for n in c_old}
    assert_dicts_close(weighted_start, c_old, EXACT_TOL,
                       "synthetic start state violates the weighted invariant")

    # Distinct per-client local outcomes and distinct step counts.
    c_i_new_all, delta_all = [], []
    for i in range(k):
        scale = eval_scaffold_expression(tree, "run", "scale",
                                         {"local_steps": 3 + 2 * i, "LR": m35.LR})
        c_i_new = eval_scaffold_expression(tree, "run", "c_i_new", {
            "c_i_old": c_i_old_all[i], "c_old": c_old,
            "x_params": patterned_controls(trainable, seed=1400 + i, scale=1.0, shift=0.3),
            "y_params": patterned_controls(trainable, seed=1500 + i, scale=1.0, shift=-0.3),
            "scale": scale,
        })
        c_i_new_all.append(c_i_new)
        delta_all.append(eval_scaffold_expression(tree, "run", "delta_c_i",
                                                  {"c_i_new": c_i_new, "c_i_old": c_i_old_all[i]}))

    weighted_delta_c = m35.weighted_controls(delta_all, weights)
    c_new = eval_scaffold_expression(tree, "run", "c_new",
                                     {"c_old": c_old, "weighted_delta_c": weighted_delta_c})

    # Independent sample-weighted aggregate of the new client controls.
    weighted_c_i_new = {n: sum(c_i_new_all[i][n] * weights[i] for i in range(k)) for n in c_old}

    # Bind to the aggregation production actually uses for the invariant, so a later
    # switch to uniform averaging or to the wrong weights fails here.
    invariant_assign = sole_name_assign(find_function(tree, "run"), "weighted_c_i_new")
    invariant_call = invariant_assign.value
    assert isinstance(invariant_call, ast.Call) and isinstance(invariant_call.func, ast.Name) \
        and invariant_call.func.id == "weighted_controls", \
        "weighted_c_i_new is not produced by weighted_controls()"
    invariant_args = [a.id for a in invariant_call.args if isinstance(a, ast.Name)]
    assert invariant_args == ["client_new_controls", "agg_weights"], \
        f"the invariant aggregate is built from {invariant_args}"
    production_weighted_c_i_new = eval_scaffold_expression(tree, "run", "weighted_c_i_new", {
        "weighted_controls": m35.weighted_controls,
        "client_new_controls": c_i_new_all,
        "agg_weights": weights,
    })
    assert_dicts_close(production_weighted_c_i_new, weighted_c_i_new, EXACT_TOL,
                       "production invariant aggregate != independent sum_i p_i c_i_new")

    residual = max_abs_diff(c_new, weighted_c_i_new)
    magnitude = max(1.0,
                    max(float(t.abs().max().item()) for t in c_new.values()),
                    max(float(t.abs().max().item()) for t in weighted_c_i_new.values()))
    # The production tolerance script 35 itself asserts against.
    assert residual <= m35.CONTROL_INVARIANT_TOL * magnitude, (
        f"invariant violated: max |c_new - sum_i p_i c_i_new| = {residual:.3e}, "
        f"tolerance {m35.CONTROL_INVARIANT_TOL * magnitude:.3e}"
    )
    # Independently, the identity is exact up to float32 rounding, so the residual
    # must also clear the verifier's own far tighter bound. This catches a drift that
    # the production tolerance would still absorb.
    assert residual <= EXACT_TOL * magnitude, (
        f"invariant residual {residual:.3e} exceeds the verifier's exact bound "
        f"{EXACT_TOL * magnitude:.3e}; the identity is not holding to float32 precision"
    )
    # A uniform aggregate must not satisfy the invariant here.
    uniform_c_i_new = {n: sum(c[n] for c in c_i_new_all) / float(k) for n in c_old}
    assert_dicts_differ(c_new, uniform_c_i_new, WRONG_ANSWER_MARGIN,
                        "negative control 'uniform client-control aggregate'")
    # Script 35's own residual measure must agree with the independent one.
    assert abs(m35.max_controls_abs_diff(c_new, weighted_c_i_new) - residual) <= EXACT_TOL, \
        "max_controls_abs_diff disagrees with the independent residual"

    # The invariant is asserted, and is not repaired by overwriting either side.
    assert "assertcontrol_invariant_max_abs_error<=CONTROL_INVARIANT_TOL*control_scale" in ctx["sq35"], \
        "the invariant assertion is missing"
    for forbidden in ("c_new=weighted_c_i_new", "weighted_c_i_new=c_new",
                      "server_control=weighted_c_i_new", "server_control=weighted_controls("):
        assert forbidden not in ctx["sq35"], f"the invariant appears to be repaired: {forbidden}"


def check_12_model_aggregation(ctx) -> None:
    """12. Model aggregation in 35 matches 32's sample-weighted rule under unequal sizes."""
    m35, m32 = ctx["m35"], ctx["m32"]
    reference_state = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).state_dict()
    states = [tiny_state(reference_state, seed=2000 + k) for k in range(m35.NUM_CLIENTS)]
    sizes = [100, 5000, 20, 900, 77]
    total = float(sum(sizes))

    # Independent expectation: sum_k (n_k / N) * theta_k.
    expected = {key: sum(states[k][key] * (sizes[k] / total) for k in range(len(states)))
                for key in reference_state}
    uniform = {key: sum(s[key] for s in states) / float(len(states)) for key in reference_state}
    assert_dicts_differ(expected, uniform, WRONG_ANSWER_MARGIN,
                        "synthetic sizes make sample-weighted and uniform aggregation identical")

    agg35 = m35.aggregate_sample_weighted(states, sizes)
    agg32 = m32.aggregate_sample_weighted(states, sizes)
    assert_dicts_close(agg35, expected, EXACT_TOL, "35 aggregation != independent sample-weighted sum")
    assert_dicts_close(agg32, expected, EXACT_TOL, "32 aggregation != independent sample-weighted sum")
    assert_dicts_close(agg35, agg32, 0.0, "35 and 32 aggregation differ")
    assert_dicts_differ(agg35, uniform, WRONG_ANSWER_MARGIN, "35 aggregation is uniform, not size-weighted")

    # Order sensitivity: permuting states without permuting sizes must change the result.
    permuted = [states[i] for i in (1, 0, 2, 3, 4)]
    assert_dicts_differ(m35.aggregate_sample_weighted(permuted, sizes), agg35,
                        WRONG_ANSWER_MARGIN, "aggregation ignores the state/size pairing")

    # The model aggregation call site uses the real client sizes.
    assert "agg_state=aggregate_sample_weighted(client_states,sizes)" in ctx["sq35"], \
        "35 does not aggregate the model with the real client sizes"


def check_13_state_isolation(ctx) -> None:
    """13. Clones and snapshots do not alias their mutable sources."""
    m35, tree = ctx["m35"], ctx["tree35"]
    model = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).to(DEVICE)
    trainable = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}

    # clone_controls
    source = patterned_controls(trainable, seed=3001, scale=0.5, shift=0.2)
    clone = m35.clone_controls(source)
    baseline = {n: t.clone() for n, t in clone.items()}
    for name in source:
        assert clone[name].data_ptr() != source[name].data_ptr(), f"clone_controls aliases {name}"
        assert not clone[name].requires_grad, f"clone_controls kept autograd on {name}"
        source[name].add_(5.0)
    assert max_abs_diff(clone, baseline) == 0.0, "clone_controls result changed with its source"

    # x_params / y_params snapshots, using script 35's own expressions.
    for target, module_name in (("x_params", "global_model"), ("y_params", "local_model")):
        snapshot = eval_scaffold_expression(tree, "run", target, {module_name: model})
        snapshot_baseline = {n: t.clone() for n, t in snapshot.items()}
        with torch.no_grad():
            for param in model.parameters():
                param.add_(2.0)
        assert max_abs_diff(snapshot, snapshot_baseline) == 0.0, \
            f"{target} snapshot aliases the live model parameters"
        for name, tensor in snapshot.items():
            assert not tensor.requires_grad, f"{target}[{name}] is still attached to autograd"

    # global_state snapshot
    global_state = eval_scaffold_expression(tree, "run", "global_state", {"global_model": model})
    gs_baseline = {k: v.clone() for k, v in global_state.items()}
    with torch.no_grad():
        for param in model.parameters():
            param.add_(3.0)
    assert max_abs_diff(global_state, gs_baseline) == 0.0, "global_state snapshot aliases the model"
    for key, tensor in global_state.items():
        assert tensor.device.type == "cpu", f"global_state[{key}] is not on CPU"

    # init_controls must not alias the parameters it is keyed by.
    controls = m35.init_controls(model, DEVICE)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert controls[name].data_ptr() != param.data_ptr(), f"init_controls aliases parameter {name}"


def check_14_test_isolation(ctx) -> None:
    """14. Script 35 never references the held-out arrays."""
    m35, source = ctx["m35"], ctx["src35"]
    # Tokens assembled at runtime so this verifier's own source stays clean.
    token_x = "X_" + "te" + "st"
    token_y = "y_" + "te" + "st"

    # Independent scan of script 35's source.
    for token in (token_x, token_y):
        assert token not in source, f"script 35 references {token}"
    assert not re.search(r"[\"'][^\"']*_te" + r"st\.npy[\"']", source), \
        "script 35 contains a held-out .npy path literal"

    # Only the intended arrays are loaded.
    loaded = set(re.findall(r'PROCESSED_DIR\s*/\s*"([^"]+)"', source))
    assert loaded <= {"X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"}, \
        f"script 35 loads unexpected arrays from PROCESSED_DIR: {sorted(loaded)}"

    # Script 35's own guard must also pass, and must actually be wired into main().
    m35.assert_no_test_reference()
    assert "assert_no_test_reference()" in ctx["sq35"], "the guard is never called"
    main_node = find_function(ctx["tree35"], "main")
    guard_calls = calls_to(main_node, "assert_no_test_reference")
    assert guard_calls, "main() does not call assert_no_test_reference()"

    # The execution gate must precede every directory creation and every write.
    main_body_lines = [(n.lineno, n) for n in main_node.body]
    gate = calls_to(main_node, "assert_execution_enabled")
    assert gate, "main() does not call assert_execution_enabled()"
    gate_line = gate[0].lineno
    first_mkdir = min((n.lineno for n in ast.walk(main_node)
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr == "mkdir"), default=None)
    assert first_mkdir is not None, "main() creates no directory; the ordering check is stale"
    assert gate_line < first_mkdir, "the execution gate runs after a directory is created"
    first_save = min((n.lineno for n in ast.walk(main_node)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr in ("save", "to_csv")), default=None)
    if first_save is not None:
        assert gate_line < first_save, "the execution gate runs after a file is written"
    assert main_body_lines, "main() has an empty body"


def check_15_first_round_fedavg_equivalence(ctx) -> None:
    """15. PREPARED: with zero controls, round 1 reproduces FedAvg across five clients.

    A full synthetic K=5 first round: five clients with unequal sample counts and
    distinct data, each run through both methods from one common initial state under
    the real round-1 seed scheme, compared per client and then after sample-weighted
    aggregation of the states the two methods actually returned.

    Not part of the default run. Executed only with --run-fedavg-equivalence, and
    even then it is CPU-only on synthetic data: it must not be started while a heavy
    experiment holds the machine.
    """
    m35, m32 = ctx["m35"], ctx["m32"]
    criterion = nn.CrossEntropyLoss(weight=class_weight_vector(m35.NUM_CLASSES).to(DEVICE))
    batch_size = 4
    round_number = 1  # this check covers the first communication round only

    # Five distinct synthetic clients with unequal sample counts, each from its own
    # deterministic data seed, so no two clients see the same data or the same
    # number of batches.
    sizes = [5, 9, 3, 11, 7]
    assert len(sizes) == m35.NUM_CLIENTS, "this check models a full K=5 round"
    assert len(set(sizes)) == len(sizes), "client sizes must be unequal"
    client_data = [synthetic_batch(n, m35.INPUT_DIM, m35.NUM_CLASSES, seed=9000 + client_id)
                   for client_id, n in enumerate(sizes)]

    # One common initial state for both methods and every client.
    torch.manual_seed(m35.TRAIN_SEED)
    seed_model = m35.MLPMultiClassClassifier(m35.INPUT_DIM, m35.NUM_CLASSES).to(DEVICE)
    initial_state = copy.deepcopy(seed_model.state_dict())
    initial_state_cpu = {k: v.detach().cpu().clone() for k, v in initial_state.items()}
    # Zero server control, shared by every client in the round, exactly as at round 1.
    zero_server_control = m35.init_controls(seed_model, DEVICE)

    def local_update(module, epoch_fn, extra_args, client_id, features, labels):
        """One client's round-1 local epoch, started from the common initial state."""
        model = module.MLPMultiClassClassifier(module.INPUT_DIM, module.NUM_CLASSES).to(DEVICE)
        model.load_state_dict(copy.deepcopy(initial_state))
        optimizer = torch.optim.SGD(model.parameters(), lr=module.LR,
                                    momentum=module.MOMENTUM, weight_decay=module.WEIGHT_DECAY)
        # The real per-round-per-client seed scheme at rnd = 1.
        local_seed = module.TRAIN_SEED + round_number * 100 + client_id
        generator = torch.Generator()
        generator.manual_seed(local_seed)
        loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size,
                            shuffle=True, num_workers=0, generator=generator)
        torch.manual_seed(local_seed)
        stats = epoch_fn(model, loader, criterion, optimizer, DEVICE, *extra_args)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return state, stats, local_seed

    scaffold_states, fedavg_states = [], []
    for client_id, (features, labels) in enumerate(client_data):
        # A fresh zero client control for this client, as at the start of round 1.
        zero_client_control = m35.init_controls(seed_model, DEVICE)
        for tensor in zero_client_control.values():
            assert int(torch.count_nonzero(tensor).item()) == 0, "client control is not zero"

        scaffold_state, scaffold_stats, scaffold_seed = local_update(
            m35, m35.train_one_epoch, (zero_server_control, zero_client_control),
            client_id, features, labels)
        fedavg_state, fedavg_stats, fedavg_seed = local_update(
            m32, m32.train_one_epoch, (), client_id, features, labels)

        assert scaffold_seed == fedavg_seed == m35.TRAIN_SEED + 100 + client_id, \
            f"client {client_id}: local seed is not TRAIN_SEED + 100 + client_id"
        assert scaffold_stats["n_samples"] == fedavg_stats["n_samples"] == sizes[client_id], \
            f"client {client_id}: sample counts disagree"
        assert scaffold_stats["n_batches"] == fedavg_stats["n_batches"], (
            f"client {client_id}: batch counts differ "
            f"({scaffold_stats['n_batches']} vs {fedavg_stats['n_batches']})"
        )

        # Non-vacuity: the local epoch must actually have moved the weights, or the
        # equality below would hold for the wrong reason.
        moved = max_abs_diff(scaffold_state, initial_state_cpu)
        assert moved > WRONG_ANSWER_MARGIN, \
            f"client {client_id}: local update barely moved the weights ({moved:.3e})"

        diff = max_abs_diff(scaffold_state, fedavg_state)
        assert diff <= EXACT_TOL, (
            f"client {client_id}: round-1 local state differs from FedAvg by {diff:.3e} "
            "with zero controls"
        )
        scaffold_states.append(scaffold_state)
        fedavg_states.append(fedavg_state)

    assert len(scaffold_states) == len(fedavg_states) == m35.NUM_CLIENTS, \
        "expected five local states from each method"

    # Aggregate the states each method actually returned, weighted by the real
    # synthetic client sizes, and require the two global models to agree.
    agg_scaffold = m35.aggregate_sample_weighted(scaffold_states, sizes)
    agg_fedavg = m32.aggregate_sample_weighted(fedavg_states, sizes)
    agg_diff = max_abs_diff(agg_scaffold, agg_fedavg)
    assert agg_diff <= EXACT_TOL, f"aggregated round-1 global models differ by {agg_diff:.3e}"


# ================================== RUNNER ================================== #
# (id, title, function, performs_training, default_on)
CHECKS = [
    (1, "shared contract with script 32", check_01_shared_contract, False, True),
    (2, "control initialisation and independence", check_02_control_initialisation, False, True),
    (3, "gradient correction g - c_i + c", check_03_gradient_correction, True, True),
    (4, "zero-control gradient equivalence", check_04_zero_control_equivalence, True, True),
    (5, "actual optimizer.step() counting", check_05_actual_step_counting, True, True),
    (6, "Option II client-control identity", check_06_option_ii_identity, False, True),
    (7, "client-control persistence across rounds", check_07_control_persistence, False, True),
    (8, "fresh controls per independent run", check_08_independent_run_reset, False, True),
    (9, "one shared c_old per round", check_09_same_server_control_within_round, False, True),
    (10, "server-control update is sample-weighted", check_10_server_control_update, False, True),
    (11, "control invariant c = sum_i p_i c_i", check_11_control_invariant, False, True),
    (12, "model aggregation matches script 32", check_12_model_aggregation, False, True),
    (13, "clone/snapshot state isolation", check_13_state_isolation, False, True),
    (14, "held-out test isolation and gate ordering", check_14_test_isolation, False, True),
    (15, "first-round FedAvg equivalence (PREPARED)",
     check_15_first_round_fedavg_equivalence, True, False),
]


def build_context() -> dict:
    src35 = SCAFFOLD_PATH.read_text()
    src32 = FEDAVG_PATH.read_text()
    assert_entry_point_guarded(src35, "script 35")
    assert_entry_point_guarded(src32, "script 32")
    return {
        "m35": load_module(SCAFFOLD_PATH, "scaffold37f"),
        "m32": load_module(FEDAVG_PATH, "fedavg37f"),
        "src35": src35,
        "src32": src32,
        "sq35": squeeze(src35),
        "sq32": squeeze(src32),
        "tree35": ast.parse(src35, filename=str(SCAFFOLD_PATH)),
        "tree32": ast.parse(src32, filename=str(FEDAVG_PATH)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent verifier for the 37f SCAFFOLD runner.")
    parser.add_argument("--no-train", action="store_true",
                        help="skip the checks that take real optimizer steps (3, 4, 5)")
    parser.add_argument("--run-fedavg-equivalence", action="store_true",
                        help="additionally run check 15, which trains on synthetic CPU data")
    parser.add_argument("--list", action="store_true", help="list the checks and exit")
    args = parser.parse_args()

    if args.list:
        for check_id, title, _, trains, default_on in CHECKS:
            flags = []
            if trains:
                flags.append("training")
            if not default_on:
                flags.append("opt-in")
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            print(f"{check_id:2d}. {title}{suffix}")
        return 0

    assert DEVICE.type == "cpu", "the verifier must run on CPU"
    torch.set_num_threads(1)  # stay out of the way of any running experiment

    context = build_context()
    print(f"verifier: device={DEVICE} torch={torch.__version__} "
          f"scaffold={SCAFFOLD_PATH} fedavg={FEDAVG_PATH}", flush=True)
    print(f"gate: SCAFFOLD_EXECUTION_ENABLED={context['m35'].SCAFFOLD_EXECUTION_ENABLED}\n", flush=True)

    passed, failed, skipped = 0, [], []
    for check_id, title, function, trains, default_on in CHECKS:
        if not default_on and not args.run_fedavg_equivalence:
            skipped.append((check_id, title, "opt-in; pass --run-fedavg-equivalence"))
            print(f"[{check_id:2d}] SKIP  {title}  (opt-in)", flush=True)
            continue
        if trains and args.no_train:
            skipped.append((check_id, title, "--no-train"))
            print(f"[{check_id:2d}] SKIP  {title}  (--no-train)", flush=True)
            continue
        try:
            function(context)
        except Exception as error:  # noqa: BLE001 - the verifier reports, it does not raise
            failed.append((check_id, title, error))
            print(f"[{check_id:2d}] FAIL  {title}\n      {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
        else:
            passed += 1
            print(f"[{check_id:2d}] PASS  {title}", flush=True)

    print(f"\npassed={passed} failed={len(failed)} skipped={len(skipped)}")
    for check_id, title, reason in skipped:
        print(f"  skipped {check_id:2d}: {title} ({reason})")
    for check_id, title, error in failed:
        print(f"  FAILED  {check_id:2d}: {title} -> {type(error).__name__}: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
