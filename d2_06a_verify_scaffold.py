"""
Verification of the Dataset-2 SCAFFOLD trainer, d2_06_train_scaffold.py.

Every check runs on CPU against synthetic tensors, small fixtures or the source
text, so the suite can be run while a production job occupies the GPU. It needs no
Dataset-2 partitions, no processed arrays and no held-out data, executes no
Dataset-2 round, and writes nothing outside VERIF_ROOT.

Expected values are computed here from the algorithm definition and compared
against what d2_06 produces; the trainer's own functions are called directly, so
the checks exercise the shipped code rather than a copy of it. Several checks use
a deliberately wrong variant as a negative control, so a check that could not
detect the error it is meant to catch fails rather than passing quietly.

    ./env/bin/python d2_06a_verify_scaffold.py                # the default suite
    ./env/bin/python d2_06a_verify_scaffold.py --runpod-gate  # adds the real-path gates
    ./env/bin/python d2_06a_verify_scaffold.py --list         # list checks and exit

The opt-in RunPod gate requires CUDA and the real saved FedAvg inputs. It runs a
real first-round SCAFFOLD-vs-FedAvg equivalence check and a real two-round SCAFFOLD
integration check into separate roots under VERIF_ROOT, aborting on output
collision rather than deleting anything, and hashes a research-integrity manifest
before and after. For the processed arrays the manifest hashes only the four
explicit train/validation files; it never enumerates a held-out array.
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

# Guarded imports would otherwise write .pyc files into the repository-root
# __pycache__/, outside VERIF_ROOT. Check 19 proves this actually holds.
sys.dont_write_bytecode = True

FEDAVG_SCRIPT = Path("d2_04_train_fedavg.py")
SCAFFOLD_SCRIPT = Path("d2_06_train_scaffold.py")

DEVICE = torch.device("cpu")   # a production job may be using the accelerator
VERIF_ROOT = Path("results/nf_cse_cic_ids2018_v2/scaffold_verification")
FIXTURE_ROOT = VERIF_ROOT / "fixtures"

TOL = 1e-6                 # agreement bound for exact arithmetic
WRONG_ANSWER_MARGIN = 1e-3  # a wrong variant must miss by more than this

GATE_SEED, GATE_CONDITION = 42, "iid"
GATE_EQUIV_ROOT = VERIF_ROOT / "runpod_gate_equivalence"
GATE_INTEGRATION_ROOT = VERIF_ROOT / "runpod_gate_integration"

GUARDED_DIRS = [
    Path("results/nf_cse_cic_ids2018_v2/final_fedavg_k5"),
    Path("models/nf_cse_cic_ids2018_v2/final_fedavg_k5"),
    Path("results/nf_cse_cic_ids2018_v2/final_scaffold_k5"),
    Path("models/nf_cse_cic_ids2018_v2/final_scaffold_k5"),
    Path("data/nf_cse_cic_ids2018_v2/fl_clients"),
]
# The processed directory is not guarded recursively: rglob over it would
# enumerate the held-out arrays. Only these four files are ever hashed.
D2_PROCESSED = Path("data/nf_cse_cic_ids2018_v2/processed")
GUARDED_FILES = [D2_PROCESSED / name for name in
                 ("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy")]


def load_module(path: Path, name: str):
    """Import a training script without running it, and without writing bytecode."""
    assert path.exists(), f"required script not found: {path}"
    sys.dont_write_bytecode = True
    assert 'if __name__ == "__main__":' in path.read_text(), f"{path}: no __main__ guard"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_batch(n_rows, input_dim, num_classes, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n_rows, input_dim, generator=g, dtype=torch.float32),
            torch.randint(0, num_classes, (n_rows,), generator=g))


def class_weight_vector(num_classes):
    return torch.linspace(0.5, 2.0, num_classes, dtype=torch.float32)


def patterned_controls(reference, seed, scale, shift):
    """Non-zero, sign-mixed synthetic controls, so a sign error cannot pass."""
    g = torch.Generator().manual_seed(seed)
    return {name: (torch.randn(t.shape, generator=g, dtype=torch.float32) * scale + shift).to(DEVICE)
            for name, t in reference.items()}


def random_state(reference_state, seed):
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randn(v.shape, generator=g, dtype=torch.float32)
            for k, v in reference_state.items()}


def max_diff(a, b):
    assert a.keys() == b.keys(), "key mismatch"
    return max(float((a[k].detach().to(torch.float64) - b[k].detach().to(torch.float64))
                     .abs().max().item()) for k in a)


def assert_close(actual, expected, tol, label):
    d = max_diff(actual, expected)
    assert d <= tol, f"{label}: max abs difference {d:.3e} exceeds {tol:.3e}"


def assert_differs(a, b, label):
    d = max_diff(a, b)
    assert d > WRONG_ANSWER_MARGIN, \
        f"{label}: differs by only {d:.3e}, so this check could not detect the error"


def fixture_dir(name):
    """Recreate a fixture subdirectory, refusing to touch anything outside FIXTURE_ROOT."""
    target = (FIXTURE_ROOT / name).resolve()
    root = FIXTURE_ROOT.resolve()
    assert root in target.parents, f"refusing to reset {target}: outside {root}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def trainable_of(model):
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


# ================================== CHECKS ================================== #
def check_01_shared_contract(ctx):
    """Everything shared with FedAvg is unchanged."""
    m04, m06 = ctx["m04"], ctx["m06"]
    expected = {
        "DATASET": "nf_cse_cic_ids2018_v2", "INPUT_DIM": 36, "NUM_CLASSES": 7,
        "NUM_CLIENTS": 5, "LOCAL_EPOCHS": 1, "MAX_ROUNDS": 40, "TRAIN_SEED": 42,
        "LR": 0.1, "MOMENTUM": 0.0, "WEIGHT_DECAY": 0.0, "BATCH_SIZE": 4096,
        "RELOAD_F1_TOL": 1e-4, "PARTITION_SEEDS": [42, 43, 44],
        "CONDITIONS": ["iid", "alpha_0p1", "alpha_0p5", "alpha_1p0"],
        "EXPECTED_TRAIN_ROWS": 13_255_011, "EXPECTED_VAL_ROWS": 2_821_063,
        "PROCESSED_DIR": Path("data/nf_cse_cic_ids2018_v2/processed"),
        "LABEL_MAP": Path("configs/nf_cse_cic_ids2018_v2/label_mapping.json"),
        "PART_ROOT": Path("data/nf_cse_cic_ids2018_v2/fl_clients/final_partitions/k_5"),
    }
    for key, value in expected.items():
        assert getattr(m06, key) == value, f"d2_06.{key} = {getattr(m06, key)!r}, expected {value!r}"
        assert getattr(m04, key) == value, f"d2_04.{key} = {getattr(m04, key)!r}, expected {value!r}"
    assert m06.METHOD == "SCAFFOLD" and m04.METHOD == "FedAvg"
    assert len(m06.PARTITION_SEEDS) * len(m06.CONDITIONS) == 12, "expected a 12-run matrix"

    shapes = {k: tuple(v.shape) for k, v in m06.MLPMultiClassClassifier(36, 7).state_dict().items()}
    assert shapes == {"network.0.weight": (128, 36), "network.0.bias": (128,),
                      "network.3.weight": (64, 128), "network.3.bias": (64,),
                      "network.6.weight": (7, 64), "network.6.bias": (7,)}, f"architecture {shapes}"
    assert str(m06.MLPMultiClassClassifier(36, 7)) == str(m04.MLPMultiClassClassifier(36, 7)), \
        "module structure differs from d2_04"

    for fragment in ("torch.optim.SGD(local_model.parameters(),lr=LR,",
                     "local_seed=TRAIN_SEED+rnd*100+client_id",
                     'ifval["macro_f1"]>best["macro_f1"]:',
                     '"selection_metric":"val_macro_f1"'):
        assert fragment in ctx["sq06"] and fragment in ctx["sq04"], \
            f"shared behaviour differs: {fragment}"

    y = np.array([0, 0, 1, 2, 3, 4, 5, 6, 6, 6], dtype=np.int64)
    assert np.array_equal(m06.class_weights_full(y), m04.class_weights_full(y)), \
        "class weight formulas differ from d2_04"


def check_02_control_initialisation(ctx):
    """Controls start at zero, detached, on-device, and mutually independent."""
    m06 = ctx["m06"]
    model = m06.MLPMultiClassClassifier(36, 7).to(DEVICE)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    server = m06.init_controls(model, DEVICE)
    clients = [m06.init_controls(model, DEVICE) for _ in range(m06.NUM_CLIENTS)]

    for label, control in [("server", server)] + [(f"c{i}", c) for i, c in enumerate(clients)]:
        assert set(control) == set(trainable), f"{label}: keys != trainable named parameters"
        for name, param in trainable.items():
            t = control[name]
            assert t.shape == param.shape and t.dtype == param.dtype, f"{label}.{name}: shape/dtype"
            assert t.device.type == param.device.type, f"{label}.{name}: wrong device"
            assert not t.requires_grad, f"{label}.{name}: attached to autograd"
            assert int(torch.count_nonzero(t).item()) == 0, f"{label}.{name}: not zero"

    # Distinct storage: mutating one control must not change any other.
    probe = next(iter(trainable))
    clients[0][probe].add_(7.5)
    for label, control in [("server", server)] + [(f"c{i}", c) for i, c in enumerate(clients[1:], 1)]:
        assert int(torch.count_nonzero(control[probe]).item()) == 0, \
            f"{label} changed when client 0 was mutated: controls share storage"


def local_epoch(ctx, module, controls, n_rows, batch_size):
    """One local epoch, plus an independent recomputation of the raw gradients.

    The reference model replays the same forward/backward from the same weights
    under the same RNG state, so the dropout masks match and the comparison of the
    corrected gradient against raw_grad - c_i + c is exact.
    """
    m06 = ctx["m06"]
    features, labels = synthetic_batch(n_rows, 36, 7, seed=1234)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(4242)
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size,
                        shuffle=False, num_workers=0, generator=loader_generator)
    criterion = nn.CrossEntropyLoss(weight=class_weight_vector(7).to(DEVICE))

    torch.manual_seed(7)
    model = module.MLPMultiClassClassifier(36, 7).to(DEVICE)
    reference = module.MLPMultiClassClassifier(36, 7).to(DEVICE)
    start = copy.deepcopy(model.state_dict())
    reference.load_state_dict(copy.deepcopy(start))

    optimizer = torch.optim.SGD(model.parameters(), lr=m06.LR,
                                momentum=m06.MOMENTUM, weight_decay=m06.WEIGHT_DECAY)
    steps = {"n": 0}
    inner = optimizer.step

    def counting_step(*a, **kw):
        steps["n"] += 1
        return inner(*a, **kw)

    optimizer.step = counting_step

    rng = torch.get_rng_state()
    stats = module.train_one_epoch(model, loader, criterion, optimizer, DEVICE, *controls)
    after = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    torch.set_rng_state(rng)
    reference.train()
    reference.zero_grad(set_to_none=True)
    criterion(reference(features.to(DEVICE)), labels.to(DEVICE)).backward()
    raw = {n: p.grad.detach().clone() for n, p in reference.named_parameters() if p.requires_grad}
    return start, after, raw, stats, steps["n"]


def check_03_gradient_correction(ctx):
    """The stepped gradient is raw_grad - c_i + c, and no sign variant passes."""
    m06 = ctx["m06"]
    trainable = trainable_of(m06.MLPMultiClassClassifier(36, 7).to(DEVICE))
    c_i_old = patterned_controls(trainable, seed=101, scale=0.50, shift=+0.30)
    c_old = patterned_controls(trainable, seed=202, scale=0.70, shift=-0.40)

    start, after, raw, stats, _ = local_epoch(ctx, m06, (c_old, c_i_old), 8, 8)
    assert stats["local_steps"] == 1, "this check needs exactly one optimizer step"

    # Plain SGD with momentum 0 and weight decay 0, so the gradient that was
    # actually stepped with is recoverable as (theta_before - theta_after) / lr.
    stepped = {n: (start[n].to(DEVICE) - after[n]) / m06.LR for n in after}
    assert_close(stepped, {n: raw[n] - c_i_old[n] + c_old[n] for n in after}, TOL,
                 "corrected gradient != raw_grad - c_i + c")
    for label, wrong in {
        "signs swapped": {n: raw[n] + c_i_old[n] - c_old[n] for n in after},
        "both subtracted": {n: raw[n] - c_i_old[n] - c_old[n] for n in after},
        "correction dropped": {n: raw[n].clone() for n in after},
    }.items():
        assert_differs(stepped, wrong, f"negative control '{label}'")


def check_04_zero_controls_match_fedavg(ctx):
    """With zero controls the local update is bitwise d2_04's."""
    m04, m06 = ctx["m04"], ctx["m06"]
    probe = m06.MLPMultiClassClassifier(36, 7).to(DEVICE)
    zeros = (m06.init_controls(probe, DEVICE), m06.init_controls(probe, DEVICE))

    start_s, after_s, _, stats_s, _ = local_epoch(ctx, m06, zeros, 24, 8)
    start_a, after_a, _, stats_a, _ = local_epoch(ctx, m04, (), 24, 8)

    assert max_diff(start_s, start_a) == 0.0, "the two runs did not start identically"
    assert stats_s["n_batches"] == stats_a["n_batches"], "batch counts differ"
    moved = max_diff(after_s, {k: v.to(DEVICE) for k, v in start_s.items() if k in after_s})
    assert moved > WRONG_ANSWER_MARGIN, f"the epoch barely moved the weights ({moved:.3e})"
    assert max_diff(after_s, after_a) == 0.0, \
        "zero-control SCAFFOLD does not reproduce the FedAvg local update exactly"
    assert stats_s["weighted_train_loss"] == stats_a["weighted_train_loss"], \
        "weighted training loss differs from FedAvg at zero controls"


def check_05_local_steps_counted(ctx):
    """local_steps is the real optimizer.step() count, not an inferred number."""
    m06 = ctx["m06"]
    trainable = trainable_of(m06.MLPMultiClassClassifier(36, 7).to(DEVICE))
    controls = (patterned_controls(trainable, 404, 0.3, -0.2),
                patterned_controls(trainable, 303, 0.2, +0.1))
    n_rows, batch_size = 7, 2
    expected = -(-n_rows // batch_size)   # ceil(7/2) = 4, and deliberately not 1

    _, _, _, stats, observed = local_epoch(ctx, m06, controls, n_rows, batch_size)
    assert observed == expected, f"optimizer.step() called {observed} times, expected {expected}"
    assert stats["local_steps"] == observed, \
        f"reported local_steps {stats['local_steps']} != real count {observed}"
    assert stats["local_steps"] not in (n_rows, batch_size, 1), \
        "local_steps coincides with a size value; this check cannot discriminate"
    assert "local_steps+=1" in ctx["sq06"], "local_steps is not incremented per step"


def check_06_option_ii_identity(ctx):
    """c_i_new = c_i_old - c_old + (x - y_i) / (tau * lr), and delta from the old control."""
    m06 = ctx["m06"]
    trainable = trainable_of(m06.MLPMultiClassClassifier(36, 7).to(DEVICE))
    c_i_old = patterned_controls(trainable, 11, 0.60, +0.25)
    c_old = patterned_controls(trainable, 22, 0.45, -0.35)
    x = patterned_controls(trainable, 33, 1.10, +0.05)
    y = patterned_controls(trainable, 44, 0.90, -0.15)
    tau, lr = 13, m06.LR   # tau != 1, so a missing divisor changes the answer

    c_i_new, delta = m06.option_ii_client_control(c_i_old, c_old, x, y, tau, lr)
    expected_new = {n: c_i_old[n] - c_old[n] + (x[n] - y[n]) / (tau * lr) for n in c_i_old}
    assert_close(c_i_new, expected_new, TOL, "Option II c_i_new mismatch")
    assert_close(delta, {n: expected_new[n] - c_i_old[n] for n in c_i_old}, TOL,
                 "delta_c_i != c_i_new - c_i_old")

    for label, wrong in {
        "+c_old instead of -c_old": {n: c_i_old[n] + c_old[n] + (x[n] - y[n]) / (tau * lr) for n in c_i_old},
        "(y - x) instead of (x - y)": {n: c_i_old[n] - c_old[n] + (y[n] - x[n]) / (tau * lr) for n in c_i_old},
        "tau omitted": {n: c_i_old[n] - c_old[n] + (x[n] - y[n]) / lr for n in c_i_old},
    }.items():
        assert_differs(c_i_new, wrong, f"negative control '{label}'")

    # The stored control must be replaced only after delta has been taken from it.
    assert ctx["sq06"].index("c_i_new,delta_c_i=option_ii_client_control(") \
        < ctx["sq06"].index("client_controls[client_id]=c_i_new"), \
        "client_controls is overwritten before delta_c_i is computed"


def check_07_controls_persist_and_reset(ctx):
    """Controls carry across rounds within a run and start at zero in a new run."""
    m06 = ctx["m06"]
    probe = m06.MLPMultiClassClassifier(36, 7).to(DEVICE)
    trainable = trainable_of(probe)
    zero = m06.init_controls(probe, DEVICE)

    # Two rounds driven through the trainer's own Option II function.
    controls = [m06.init_controls(probe, DEVICE) for _ in range(m06.NUM_CLIENTS)]
    for client_id in range(m06.NUM_CLIENTS):
        c_i_new, _ = m06.option_ii_client_control(
            controls[client_id], zero,
            patterned_controls(trainable, 500 + client_id, 1.0, 0.2),
            patterned_controls(trainable, 600 + client_id, 1.0, -0.2), 5, m06.LR)
        controls[client_id] = c_i_new
    for client_id in range(m06.NUM_CLIENTS):
        assert m06.controls_l2_norm(controls[client_id]) > WRONG_ANSWER_MARGIN, \
            f"client {client_id}: control did not move in round 1"

    # A new run rebuilds zero controls, unaffected by the previous run's tensors.
    fresh = m06.init_controls(probe, DEVICE)
    assert m06.controls_l2_norm(fresh) == 0.0, "a new run's controls are not zero"

    # Structural: read from the persistent list, write back once, never re-init
    # inside the round loop.
    assert ctx["sq06"].count("c_i_old=client_controls[client_id]") == 1, \
        "c_i_old is not read exactly once from the persistent list"
    assert ctx["sq06"].count("client_controls[client_id]=c_i_new") == 1, \
        "client_controls is not written back exactly once"
    round_loop = ctx["sq06"].index("forrndinrange(1,MAX_ROUNDS+1):")
    assert "init_controls" not in ctx["sq06"][round_loop:], \
        "init_controls() is called inside the round loop: controls would not persist"


def check_08_single_c_old_per_round(ctx):
    """All five clients use one frozen server-control snapshot per round."""
    m06 = ctx["m06"]
    trainable = trainable_of(m06.MLPMultiClassClassifier(36, 7).to(DEVICE))
    server_control = patterned_controls(trainable, 77, 0.4, 0.1)
    c_old = m06.clone_controls(server_control)
    baseline = {n: t.clone() for n, t in c_old.items()}

    # A later in-place write to the live server control must not reach the snapshot.
    for name in server_control:
        server_control[name].add_(1.0)
    assert max_diff(c_old, baseline) == 0.0, "c_old moved when the server control changed"

    assert ctx["sq06"].count("c_old=clone_controls(server_control)") == 1, \
        "c_old is not snapshotted exactly once per round"
    assert ctx["sq06"].index("c_old=clone_controls(server_control)") \
        < ctx["sq06"].index("forclient_idinrange(NUM_CLIENTS):"), \
        "c_old is snapshotted after the client loop starts"


def check_09_server_control_sample_weighted(ctx):
    """c_new = c_old + sum_i p_i delta_i, with p_i the model's sample weights."""
    m06 = ctx["m06"]
    trainable = trainable_of(m06.MLPMultiClassClassifier(36, 7).to(DEVICE))
    c_old = patterned_controls(trainable, 91, 0.55, -0.20)
    deltas = [patterned_controls(trainable, 1000 + k, 0.8 + 0.2 * k, 0.1 * k - 0.2)
              for k in range(m06.NUM_CLIENTS)]
    sizes = [100, 5000, 20, 900, 77]        # deliberately far from equal
    weights = [n / float(sum(sizes)) for n in sizes]

    weighted = {n: sum(deltas[k][n] * weights[k] for k in range(len(deltas))) for n in c_old}
    uniform = {n: sum(d[n] for d in deltas) / float(len(deltas)) for n in c_old}
    assert_differs(weighted, uniform, "synthetic sizes make weighted and uniform identical")

    produced = m06.weighted_controls(deltas, weights)
    assert_close(produced, weighted, TOL, "weighted_controls is not the sample-weighted sum")
    assert_differs(produced, uniform, "weighted_controls matches the uniform mean")
    assert list(inspect.signature(m06.weighted_controls).parameters) == ["control_list", "weights"], \
        "weighted_controls does not take explicit weights"
    assert "weighted_controls(client_delta_controls,agg_weights)" in ctx["sq06"], \
        "the server update does not use the model's aggregation weights"
    assert "agg_weights=[n/total_sizeforninsizes]" in ctx["sq06"], \
        "agg_weights are not n_k / sum_j n_j"


def check_10_control_invariant(ctx):
    """After the update, c == sum_i p_i c_i holds, and a uniform aggregate does not."""
    m06 = ctx["m06"]
    trainable = trainable_of(m06.MLPMultiClassClassifier(36, 7).to(DEVICE))
    k = m06.NUM_CLIENTS
    sizes = [100, 5000, 20, 900, 77]
    weights = [n / float(sum(sizes)) for n in sizes]

    # Start from a state where the invariant already holds, solving for the
    # largest-weight client so the construction does not amplify float error.
    c_old = patterned_controls(trainable, 1201, 0.30, 0.15)
    solved = max(range(k), key=lambda i: weights[i])
    others = [i for i in range(k) if i != solved]
    parts = {i: patterned_controls(trainable, 1300 + i, 0.5, 0.1 * i - 0.2) for i in others}
    parts[solved] = {n: (c_old[n] - sum(parts[i][n] * weights[i] for i in others)) / weights[solved]
                     for n in c_old}
    c_i_old = [parts[i] for i in range(k)]
    assert_close({n: sum(c_i_old[i][n] * weights[i] for i in range(k)) for n in c_old},
                 c_old, TOL, "the synthetic start state violates the invariant")

    c_i_new, deltas = [], []
    for i in range(k):
        new, delta = m06.option_ii_client_control(
            c_i_old[i], c_old,
            patterned_controls(trainable, 1400 + i, 1.0, 0.3),
            patterned_controls(trainable, 1500 + i, 1.0, -0.3), 3 + 2 * i, m06.LR)
        c_i_new.append(new)
        deltas.append(delta)

    weighted_delta = m06.weighted_controls(deltas, weights)
    c_new = {n: c_old[n] + weighted_delta[n] for n in c_old}
    expected = {n: sum(c_i_new[i][n] * weights[i] for i in range(k)) for n in c_old}

    residual = max_diff(c_new, expected)
    magnitude = max(1.0, max(float(t.abs().max().item()) for t in c_new.values()))
    assert residual <= m06.CONTROL_INVARIANT_TOL * magnitude, \
        f"invariant violated: residual {residual:.3e}"
    assert_differs(c_new, {n: sum(c[n] for c in c_i_new) / float(k) for n in c_old},
                   "negative control 'uniform client-control aggregate'")

    assert "assertcontrol_invariant_max_abs_error<=CONTROL_INVARIANT_TOL*control_scale" in ctx["sq06"], \
        "the invariant is not asserted each round"
    for repair in ("c_new=weighted_c_i_new", "server_control=weighted_c_i_new"):
        assert repair not in ctx["sq06"], f"the invariant appears to be repaired: {repair}"
    ctx["report"]["invariant_residual"] = residual


def check_11_model_aggregation(ctx):
    """Model aggregation is d2_04's sample-weighted rule, unchanged."""
    m04, m06 = ctx["m04"], ctx["m06"]
    reference = m06.MLPMultiClassClassifier(36, 7).state_dict()
    states = [random_state(reference, 2000 + k) for k in range(m06.NUM_CLIENTS)]
    sizes = [100, 5000, 20, 900, 77]
    total = float(sum(sizes))

    expected = {key: sum(states[k][key] * (sizes[k] / total) for k in range(len(states)))
                for key in reference}
    uniform = {key: sum(s[key] for s in states) / float(len(states)) for key in reference}
    assert_differs(expected, uniform, "synthetic sizes make weighted and uniform identical")

    produced = m06.aggregate_sample_weighted(states, sizes)
    assert_close(produced, expected, TOL, "aggregation != sample-weighted sum")
    assert max_diff(produced, m04.aggregate_sample_weighted(states, sizes)) == 0.0, \
        "SCAFFOLD and FedAvg aggregation differ"
    assert "agg_state=aggregate_sample_weighted(client_states,sizes)" in ctx["sq06"], \
        "the round does not aggregate with the real client sizes"


def check_12_no_aliasing(ctx):
    """Control clones and parameter snapshots are independent of their sources."""
    m06 = ctx["m06"]
    model = m06.MLPMultiClassClassifier(36, 7).to(DEVICE)
    trainable = trainable_of(model)

    source = patterned_controls(trainable, 3001, 0.5, 0.2)
    clone = m06.clone_controls(source)
    baseline = {n: t.clone() for n, t in clone.items()}
    for name in source:
        assert clone[name].data_ptr() != source[name].data_ptr(), f"clone_controls aliases {name}"
        assert not clone[name].requires_grad, f"clone_controls kept autograd on {name}"
        source[name].add_(5.0)
    assert max_diff(clone, baseline) == 0.0, "the clone changed with its source"

    # x_params / y_params are taken with the same detach().clone() idiom.
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    snap_baseline = {n: t.clone() for n, t in snapshot.items()}
    with torch.no_grad():
        for param in model.parameters():
            param.add_(2.0)
    assert max_diff(snapshot, snap_baseline) == 0.0, "a parameter snapshot aliases the live model"
    assert ctx["sq06"].count("p.detach().clone()") >= 2, \
        "x_params/y_params are not taken as detached clones"

    controls = m06.init_controls(model, DEVICE)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert controls[name].data_ptr() != param.data_ptr(), f"init_controls aliases {name}"


def check_13_holdout_isolation(ctx):
    """No held-out array is referenced, loaded, or reachable by the manifest."""
    tokens = ("X_" + "te" + "st", "y_" + "te" + "st", "raw_indices_" + "te" + "st")
    for label, source in (("trainer", ctx["src06"]), ("verifier", ctx["src06a"])):
        for token in tokens:
            assert token not in source, f"{label} references {token}"

    import re
    loaded = set(re.findall(r'PROCESSED_DIR\s*/\s*"([^"]+)"', ctx["src06"]))
    assert loaded <= {"X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"}, \
        f"trainer loads unexpected arrays: {sorted(loaded)}"
    ctx["m06"].assert_no_test_reference()
    assert "assert_no_test_reference()" in ctx["sq06"], "the guard is never called"

    # The integrity manifest must not reach a held-out array either.
    names = [f.name for f in GUARDED_FILES]
    assert set(names) == {"X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"}, \
        f"the manifest allowlist is {sorted(names)}"
    processed = D2_PROCESSED.resolve()
    for d in GUARDED_DIRS:
        assert d.resolve() != processed and d.resolve() not in processed.parents, \
            f"{d} would make the manifest enumerate the held-out arrays"
    offenders = [str(p) for p in manifest_targets() if "test" in p.name.lower()]
    assert not offenders, f"the manifest would hash held-out arrays: {offenders}"
    ctx["report"]["arrays_read"] = sorted(loaded)


def check_14_outputs_and_execution_gate(ctx):
    """Isolated output roots, no initial-model artefact, and a closed gate that
    aborts before anything is created."""
    m04, m06 = ctx["m04"], ctx["m06"]
    assert m06.RESULTS_DIR == Path("results/nf_cse_cic_ids2018_v2/final_scaffold_k5")
    assert m06.MODELS_DIR == Path("models/nf_cse_cic_ids2018_v2/final_scaffold_k5")
    for protected in (Path(m04.RESULTS_DIR), Path(m04.MODELS_DIR),
                      Path("results/nf_cse_cic_ids2018_v2/final_fedprox_k5"),
                      Path("data/nf_cse_cic_ids2018_v2/processed"),
                      Path("data/nf_cse_cic_ids2018_v2/fl_clients")):
        for out in (m06.RESULTS_DIR.resolve(), m06.MODELS_DIR.resolve()):
            assert out != protected.resolve() and protected.resolve() not in out.parents, \
                f"SCAFFOLD output root {out} collides with {protected}"
    for forbidden in ("torch.save(initial_state,INIT_PATH", "np.save(FEDAVG"):
        assert forbidden not in ctx["sq06"], f"the trainer writes into a FedAvg path: {forbidden}"

    # No SCAFFOLD initial-model checkpoint anywhere.
    assert "initial_global_model" not in ctx["sq06"].replace(
        "models/nf_cse_cic_ids2018_v2/final_fedavg_k5/initial_global_model.pt", ""), \
        "the trainer references a SCAFFOLD initial-model filename"

    # The gate is closed, and running main() creates nothing.
    assert m06.SCAFFOLD_EXECUTION_ENABLED is False, \
        "SCAFFOLD_EXECUTION_ENABLED must remain False until the audit and gate pass"
    before = [p for p in (m06.RESULTS_DIR, m06.MODELS_DIR) if p.exists()]
    saved_argv = sys.argv
    sys.argv = ["d2_06_train_scaffold.py"]
    try:
        m06.main()
    except SystemExit:
        pass
    else:
        raise AssertionError("main() did not abort while the execution gate is closed")
    finally:
        sys.argv = saved_argv
    after = [p for p in (m06.RESULTS_DIR, m06.MODELS_DIR) if p.exists()]
    assert before == after, f"a gated run created output roots: {set(after) - set(before)}"

    assert "raise RuntimeError" in ctx["src06"].split("def preflight_outputs")[1].split("def ")[0], \
        "preflight_outputs() does not raise on collision"
    ctx["report"]["execution_gate_closed"] = True


def check_15_mandatory_fedavg_inputs(ctx):
    """Both FedAvg inputs are mandatory, exactly verified, and checked before any write."""
    m06 = ctx["m06"]
    fixtures = fixture_dir("inputs")

    # --- initial state ---
    good_init = fixtures / "init_good.pt"
    reproduced = m06.reproduce_d2_04_initial_state(DEVICE)
    torch.save(reproduced, good_init)
    perturbed_state = {k: v.clone() for k, v in reproduced.items()}
    perturbed_state[next(iter(perturbed_state))].reshape(-1)[0] += 1e-3
    bad_init = fixtures / "init_perturbed.pt"
    torch.save(perturbed_state, bad_init)
    corrupt = fixtures / "init_corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")

    saved_init_path = m06.INIT_PATH
    try:
        for path, must_fail in ((fixtures / "absent.pt", True), (corrupt, True),
                                (bad_init, True), (good_init, False)):
            m06.INIT_PATH = path
            try:
                state, sha = m06.load_and_verify_fedavg_initial_state(DEVICE)
            except (RuntimeError, AssertionError):
                assert must_fail, f"{path.name} should have been accepted"
            else:
                assert not must_fail, f"{path.name} should have been rejected"
                assert max_diff(state, reproduced) == 0.0, "the verified state was altered"
                assert sha == hashlib.sha256(path.read_bytes()).hexdigest(), "wrong sha256"
    finally:
        m06.INIT_PATH = saved_init_path

    # --- class weights ---
    y = np.concatenate([np.full(n, c, dtype=np.int64)
                        for c, n in enumerate([50, 7, 11, 23, 3, 17, 5])])
    expected = m06.class_weights_full(y).astype(np.float32)
    good_w = fixtures / "weights_good.npy"
    np.save(good_w, expected)
    bad_w = fixtures / "weights_bad.npy"
    wrong = expected.copy()
    wrong[0] = np.float32(wrong[0] * 1.001)
    np.save(bad_w, wrong)
    f64_w = fixtures / "weights_float64.npy"
    np.save(f64_w, expected.astype(np.float64))

    saved_weights_path = m06.FEDAVG_CLASS_WEIGHTS_PATH
    try:
        for path, must_fail in ((fixtures / "absent.npy", True), (bad_w, True),
                                (f64_w, True), (good_w, False)):
            m06.FEDAVG_CLASS_WEIGHTS_PATH = path
            try:
                weights, sha = m06.load_and_verify_fedavg_class_weights(y)
            except (RuntimeError, AssertionError):
                assert must_fail, f"{path.name} should have been accepted"
            else:
                assert not must_fail, f"{path.name} should have been rejected"
                assert np.array_equal(weights, expected), "the verified weights were altered"
                assert sha == hashlib.sha256(path.read_bytes()).hexdigest(), "wrong sha256"
    finally:
        m06.FEDAVG_CLASS_WEIGHTS_PATH = saved_weights_path

    # Both verifications precede the first production write, and training.
    source = ctx["sq06"]
    weights_at = source.index("weight_f32,fedavg_weights_sha=load_and_verify_fedavg_class_weights(")
    init_at = source.index("initial_state,fedavg_init_sha=load_and_verify_fedavg_initial_state(")
    save_at = source.index("np.save(CLASS_WEIGHTS_PATH,weight_f32)")
    run_at = source.index("row=run(partition_seed,condition,")
    assert weights_at < save_at and init_at < save_at, \
        "the class-weight artefact is written before both prerequisites are verified"
    assert max(weights_at, init_at) < run_at, "verification happens after training starts"
    # And both source files are re-hashed afterwards.
    assert "file_sha256(INIT_PATH)==fedavg_init_sha" in source, \
        "the FedAvg initial state is not re-hashed after training"
    assert "file_sha256(FEDAVG_CLASS_WEIGHTS_PATH)==fedavg_weights_sha" in source, \
        "the FedAvg class weights are not re-hashed after training"


def check_16_round1_k5_equals_fedavg(ctx):
    """A full synthetic K=5 first round: SCAFFOLD equals FedAvg exactly.

    At round 1 every control is zero, so the correction vanishes and SCAFFOLD must
    reduce to FedAvg client-by-client and after aggregation.
    """
    m04, m06 = ctx["m04"], ctx["m06"]
    criterion = nn.CrossEntropyLoss(weight=class_weight_vector(7).to(DEVICE))
    sizes = [5, 9, 3, 11, 7]           # unequal, so aggregation weights are non-uniform
    client_data = [synthetic_batch(n, 36, 7, seed=9000 + k) for k, n in enumerate(sizes)]

    torch.manual_seed(m06.TRAIN_SEED)
    seed_model = m06.MLPMultiClassClassifier(36, 7).to(DEVICE)
    initial_state = copy.deepcopy(seed_model.state_dict())
    initial_cpu = {k: v.detach().cpu().clone() for k, v in initial_state.items()}
    zero_server = m06.init_controls(seed_model, DEVICE)

    def one_client(module, extra, client_id, features, labels):
        model = module.MLPMultiClassClassifier(36, 7).to(DEVICE)
        model.load_state_dict(copy.deepcopy(initial_state))
        optimizer = torch.optim.SGD(model.parameters(), lr=module.LR,
                                    momentum=module.MOMENTUM, weight_decay=module.WEIGHT_DECAY)
        local_seed = module.TRAIN_SEED + 1 * 100 + client_id   # the real round-1 formula
        generator = torch.Generator()
        generator.manual_seed(local_seed)
        loader = DataLoader(TensorDataset(features, labels), batch_size=4,
                            shuffle=True, num_workers=0, generator=generator)
        torch.manual_seed(local_seed)
        stats = module.train_one_epoch(model, loader, criterion, optimizer, DEVICE, *extra)
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, stats

    scaffold_states, fedavg_states = [], []
    for client_id, (features, labels) in enumerate(client_data):
        zero_client = m06.init_controls(seed_model, DEVICE)
        assert m06.controls_diff_l2_norm(zero_server, zero_client) == 0.0, \
            "the round-1 correction norm is not exactly zero"
        s_state, s_stats = one_client(m06, (zero_server, zero_client), client_id, features, labels)
        a_state, a_stats = one_client(m04, (), client_id, features, labels)

        assert s_stats["n_samples"] == a_stats["n_samples"] == sizes[client_id], \
            f"client {client_id}: sample counts disagree"
        assert s_stats["n_batches"] == a_stats["n_batches"], f"client {client_id}: batch counts differ"
        assert max_diff(s_state, initial_cpu) > WRONG_ANSWER_MARGIN, \
            f"client {client_id}: weights barely moved"
        assert max_diff(s_state, a_state) == 0.0, \
            f"client {client_id}: round-1 local state differs from FedAvg"
        scaffold_states.append(s_state)
        fedavg_states.append(a_state)

    agg_diff = max_diff(m06.aggregate_sample_weighted(scaffold_states, sizes),
                        m04.aggregate_sample_weighted(fedavg_states, sizes))
    assert agg_diff == 0.0, f"aggregated round-1 global models differ by {agg_diff:.3e}"
    ctx["report"]["round1_k5_diff"] = agg_diff


def check_17_cuda_resident_path(ctx):
    """The CUDA-resident batching path is d2_04's, and controls stay on device."""
    m04, m06 = ctx["m04"], ctx["m06"]
    for name in ("LocalPositionDataset", "ResidentClientBatches", "ResidentValLoader"):
        assert hasattr(m06, name), f"missing {name}"
        assert list(inspect.signature(getattr(m06, name).__init__).parameters) == \
            list(inspect.signature(getattr(m04, name).__init__).parameters), \
            f"{name} signature differs from d2_04"
    for fragment in ("position_loader=DataLoader(LocalPositionDataset(sizes[client_id]),",
                     "shuffle=True,num_workers=0,generator=generator)",
                     'val_loader=ResidentValLoader(resident["x_val"],resident["y_val"],4096)',
                     'ifdevice.type=="cuda":'):
        assert fragment in ctx["sq06"] and fragment in ctx["sq04"], \
            f"resident path differs from d2_04: {fragment}"
    # Controls are allocated on the training device and never moved per batch.
    assert "torch.zeros_like(p,device=device)" in ctx["sq06"], \
        "init_controls does not allocate on the training device"
    for moved in ("c_i_old.cpu()", "c_old.cpu()", "c_i_new.cpu()"):
        assert moved not in ctx["sq06"], f"a control is moved off device: {moved}"


def check_18_weighted_extension_provenance(ctx):
    """The config states the weighted extension and claims no theory carry-over."""
    m06 = ctx["m06"]
    for key in ("control_variate_option", "gradient_correction", "control_update_rule",
                "tau_source", "model_aggregation", "server_control_rule",
                "server_control_aggregation", "control_weighting_provenance",
                "control_weighting_is_project_extension",
                "control_weighting_matches_original_displayed_equation",
                "original_theoretical_guarantees_claimed", "control_invariant",
                "initial_state_sha256", "class_weights_sha256", "script_sha256"):
        assert f'"{key}"' in ctx["src06"], f"the config omits {key}"

    assert m06.SERVER_CONTROL_AGGREGATION.startswith("sample weighted"), \
        f"server-control aggregation described as {m06.SERVER_CONTROL_AGGREGATION!r}"
    for phrase in ("uniform", "sample/example-weighted extension",
                   "not the verbatim original displayed equation"):
        assert phrase in m06.CONTROL_WEIGHTING_PROVENANCE, f"provenance omits '{phrase}'"
    assert m06.CONTROL_THEORY_CLAIM.startswith("none:") \
        and "not claimed to carry over" in m06.CONTROL_THEORY_CLAIM, \
        f"the theory claim does not disclaim carry-over: {m06.CONTROL_THEORY_CLAIM!r}"
    assert '"control_weighting_matches_original_displayed_equation":False' in ctx["sq06"], \
        "the config does not record that this is not the original displayed equation"


def check_19_timing_accounting(ctx):
    """Orchestration is measured against client-side control work only.

    The server-control update runs after the client loop closes, so it is not part
    of loop_wall_seconds. Subtracting the client+server total would under-count
    orchestration and can make it negative, while fl_round_seconds must still
    include both halves of the control work.
    """
    source = ctx["sq06"]
    assert "client_control_seconds=control_seconds" in source, \
        "the client-side control time is not captured before the server update"
    assert ("round_orchestration_seconds=(loop_wall_seconds-round_train_seconds"
            "-client_control_seconds)") in source, \
        "orchestration is not measured against the client-side control time alone"
    assert "fl_round_seconds=round_train_seconds+control_seconds+aggregation_seconds" in source, \
        "fl_round_seconds does not include the full client+server control work"

    # The capture must sit between the client loop and the server-control update.
    capture_at = source.index("client_control_seconds=control_seconds")
    loop_end_at = source.index("loop_wall_seconds=time.perf_counter()-loop_wall_start")
    server_at = source.index("weighted_delta_c=weighted_controls(client_delta_controls,agg_weights)")
    assert loop_end_at < capture_at < server_at, \
        "the client-control time is not captured after the client loop and before the server update"

    # Both figures are recorded, so the split is auditable in the history CSV.
    for field in ('record["client_control_seconds"]', 'record["control_seconds"]'):
        assert field in source, f"the history does not record {field}"


def check_20_no_bytecode_outside_verif_root(ctx):
    """Importing the trainers writes no .pyc outside VERIF_ROOT."""
    assert sys.dont_write_bytecode is True, "sys.dont_write_bytecode is not set"

    def snapshot():
        verif = VERIF_ROOT.resolve()
        return {str(f) for cache in Path(".").rglob("__pycache__")
                if verif not in cache.resolve().parents and cache.resolve() != verif
                for f in cache.glob("*.pyc")}

    before = snapshot()
    load_module(FEDAVG_SCRIPT, "d2_04_probe")
    load_module(SCAFFOLD_SCRIPT, "d2_06_probe")
    created = sorted(snapshot() - before)
    assert not created, f"guarded imports created bytecode outside VERIF_ROOT: {created}"


CHECKS = [
    (1, "shared Dataset-2 contract with d2_04", check_01_shared_contract),
    (2, "controls start zero, detached, independent", check_02_control_initialisation),
    (3, "gradient correction g - c_i + c", check_03_gradient_correction),
    (4, "zero controls reproduce the FedAvg local update", check_04_zero_controls_match_fedavg),
    (5, "local_steps is the real optimizer.step() count", check_05_local_steps_counted),
    (6, "Option II client-control identity", check_06_option_ii_identity),
    (7, "controls persist across rounds and reset per run", check_07_controls_persist_and_reset),
    (8, "one frozen c_old per round", check_08_single_c_old_per_round),
    (9, "server control uses sample weights", check_09_server_control_sample_weighted),
    (10, "invariant c = sum_i p_i c_i", check_10_control_invariant),
    (11, "model aggregation matches d2_04", check_11_model_aggregation),
    (12, "clones and snapshots do not alias", check_12_no_aliasing),
    (13, "held-out isolation, including the manifest", check_13_holdout_isolation),
    (14, "output roots and closed execution gate", check_14_outputs_and_execution_gate),
    (15, "mandatory FedAvg inputs verified before any write", check_15_mandatory_fedavg_inputs),
    (16, "round-1 K=5 SCAFFOLD == FedAvg", check_16_round1_k5_equals_fedavg),
    (17, "CUDA-resident path matches d2_04", check_17_cuda_resident_path),
    (18, "weighted-extension provenance recorded", check_18_weighted_extension_provenance),
    (19, "round timing accounting is consistent", check_19_timing_accounting),
    (20, "no bytecode written outside VERIF_ROOT", check_20_no_bytecode_outside_verif_root),
]


# ======================= RunPod integration/stability gate ================== #
def manifest_targets():
    """Files the integrity manifest will hash: guarded dirs plus four named arrays."""
    targets = []
    for d in GUARDED_DIRS:
        if d.exists():
            targets.extend(sorted(p for p in d.rglob("*") if p.is_file()))
    targets.extend(f for f in GUARDED_FILES if f.is_file())
    return targets


def manifest():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in manifest_targets()}


def checkpoint_equal(path_a, path_b):
    a = torch.load(path_a, map_location="cpu")
    b = torch.load(path_b, map_location="cpu")
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def gate_roots(root):
    """Create isolated results/models roots, aborting on collision rather than deleting."""
    results_dir, models_dir = root / "results", root / "models"
    for d in (results_dir, models_dir):
        if d.exists() and any(d.iterdir()):
            raise RuntimeError(
                f"Refusing to run the gate: {d} exists and is non-empty. Move or inspect "
                "it manually; this verifier never deletes gate output."
            )
        d.mkdir(parents=True, exist_ok=True)
    return results_dir, models_dir


def gate_inputs(m04, m06, device):
    """Verified real inputs plus device-resident arrays, shared by both gates."""
    y_train = np.load(m06.PROCESSED_DIR / "y_train.npy")
    counts = np.bincount(y_train, minlength=m06.NUM_CLASSES)
    weights, weights_sha = m06.load_and_verify_fedavg_class_weights(y_train)
    assert np.array_equal(weights, m04.class_weights_full(y_train).astype(np.float32)), \
        "the verified saved class weights differ from the d2_04 recomputation"
    initial_state, init_sha = m06.load_and_verify_fedavg_initial_state(device)
    print(f"gate inputs verified: init={init_sha} weights={weights_sha}", flush=True)

    resident = {
        "x_train": torch.from_numpy(np.load(m06.PROCESSED_DIR / "X_train.npy")).to(torch.float32).to(device),
        "y_train": torch.from_numpy(np.load(m06.PROCESSED_DIR / "y_train.npy")).to(torch.long).to(device),
        "x_val": torch.from_numpy(np.load(m06.PROCESSED_DIR / "X_val.npy")).to(torch.float32).to(device),
        "y_val": torch.from_numpy(np.load(m06.PROCESSED_DIR / "y_val.npy")).to(torch.long).to(device),
    }
    val_loader = m06.ResidentValLoader(resident["x_val"], resident["y_val"], 4096)
    return y_train, counts, weights, initial_state, resident, val_loader


def gate_equivalence(m04, m06, device):
    """Real first round, seed 42 / iid: SCAFFOLD must equal FedAvg exactly."""
    print(f"\n=== GATE: real first-round equivalence (seed {GATE_SEED} / {GATE_CONDITION}) ===",
          flush=True)
    results_dir, models_dir = gate_roots(GATE_EQUIV_ROOT)
    avg_res, avg_mod = results_dir / "fedavg", models_dir / "fedavg"
    sca_res, sca_mod = results_dir / "scaffold", models_dir / "scaffold"
    for d in (avg_res, avg_mod, sca_res, sca_mod):
        d.mkdir(parents=True, exist_ok=True)

    y_train, counts, weights, initial_state, resident, val_loader = gate_inputs(m04, m06, device)
    for module, res, mod in ((m04, avg_res, avg_mod), (m06, sca_res, sca_mod)):
        module.RESULTS_DIR, module.MODELS_DIR, module.MAX_ROUNDS = res, mod, 1
        assert str(module.RESULTS_DIR).startswith(str(GATE_EQUIV_ROOT)), "redirect failed"

    part_avg = m04.load_partition(GATE_SEED, GATE_CONDITION, y_train, counts)
    part_sca = m06.load_partition(GATE_SEED, GATE_CONDITION, y_train, counts)
    assert part_avg["sizes"] == part_sca["sizes"], "client sizes differ between modules"
    m04.run(GATE_SEED, GATE_CONDITION, part_avg, initial_state, weights, val_loader, device, resident)
    m06.run(GATE_SEED, GATE_CONDITION, part_sca, initial_state, weights, val_loader, device, resident)

    avg_tag = f"fedavg_k5_seed{GATE_SEED}_{GATE_CONDITION}"
    sca_tag = f"scaffold_k5_seed{GATE_SEED}_{GATE_CONDITION}"
    h_avg = pd.read_csv(avg_res / f"history_{avg_tag}.csv")
    h_sca = pd.read_csv(sca_res / f"history_{sca_tag}.csv")
    shared = [c for c in h_avg.columns if c in h_sca.columns and "seconds" not in c]
    history_exact = bool(np.array_equal(h_avg[shared].to_numpy(dtype=float),
                                        h_sca[shared].to_numpy(dtype=float), equal_nan=True))
    corr_cols = [c for c in h_sca.columns if c.startswith("correction_norm_client_")]
    max_corr = float(np.max(np.abs(h_sca[corr_cols].to_numpy(dtype=float))))
    best_ok = checkpoint_equal(avg_mod / f"best_{avg_tag}.pt", sca_mod / f"best_{sca_tag}.pt")
    final_ok = checkpoint_equal(avg_mod / f"final_{avg_tag}.pt", sca_mod / f"final_{sca_tag}.pt")

    ok = history_exact and best_ok and final_ok and max_corr == 0.0
    print(f"  history_exact={history_exact} best_exact={best_ok} final_exact={final_ok} "
          f"round1_max_correction={max_corr:.2e} shared_fields={len(shared)} "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def gate_integration(m04, m06, device):
    """Real two rounds: controls must stay finite, start at zero, then engage."""
    print(f"\n=== GATE: real two-round SCAFFOLD integration "
          f"(seed {GATE_SEED} / {GATE_CONDITION}) ===", flush=True)
    results_dir, models_dir = gate_roots(GATE_INTEGRATION_ROOT)
    y_train, counts, weights, initial_state, resident, val_loader = gate_inputs(m04, m06, device)

    m06.RESULTS_DIR, m06.MODELS_DIR, m06.MAX_ROUNDS = results_dir, models_dir, 2
    assert str(m06.RESULTS_DIR).startswith(str(GATE_INTEGRATION_ROOT)), "redirect failed"
    part = m06.load_partition(GATE_SEED, GATE_CONDITION, y_train, counts)
    m06.run(GATE_SEED, GATE_CONDITION, part, initial_state, weights, val_loader, device, resident)

    hist = pd.read_csv(results_dir / f"history_scaffold_k5_seed{GATE_SEED}_{GATE_CONDITION}.csv")
    assert len(hist) == 2, f"expected 2 rounds, found {len(hist)}"
    finite = bool(np.isfinite(hist.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all())
    steps_cols = [c for c in hist.columns if c.startswith("local_steps_client_")]
    steps_ok = bool((hist[steps_cols].to_numpy(dtype=float) > 0).all())
    corr_cols = [c for c in hist.columns if c.startswith("correction_norm_client_")]
    r1 = float(np.max(np.abs(hist.loc[0, corr_cols].to_numpy(dtype=float))))
    r2 = float(np.max(np.abs(hist.loc[1, corr_cols].to_numpy(dtype=float))))
    invariant = float(np.max(np.abs(hist["control_invariant_max_abs_error"].to_numpy(dtype=float))))
    norm_cols = [c for c in hist.columns if c.startswith("control_norm_client_")]
    evolved = bool((hist.loc[1, norm_cols].to_numpy(dtype=float) > 0).any())
    server_moved = float(hist.loc[1, "server_control_norm"]) > 0.0

    ok = (finite and steps_ok and r1 == 0.0 and r2 > 0.0
          and invariant <= m06.CONTROL_INVARIANT_TOL and evolved and server_moved)
    print(f"  finite={finite} local_steps>0={steps_ok} r1_correction={r1:.2e} "
          f"r2_correction={r2:.6g} invariant={invariant:.2e} controls_evolved={evolved} "
          f"server_control_moved={server_moved} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def runpod_gate(ctx):
    m04, m06 = ctx["m04"], ctx["m06"]
    device = m06.get_device()
    if device.type != "cuda":
        print(f"\nRefusing to run the gate: CUDA required, device is {device}.")
        return 1
    if not m06.INIT_PATH.exists() or not m06.FEDAVG_CLASS_WEIGHTS_PATH.exists():
        print("\nRefusing to run the gate: the real saved FedAvg inputs are required.")
        return 1

    before = manifest()
    equivalence_ok = gate_equivalence(m04, m06, device)
    integration_ok = gate_integration(m04, m06, device)
    research_ok = manifest() == before
    print(f"\n  research artefacts unchanged: {'PASS' if research_ok else 'FAIL'}", flush=True)
    ok = equivalence_ok and integration_ok and research_ok
    print(f"RUNPOD GATE: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description="Verify the Dataset-2 SCAFFOLD trainer.")
    parser.add_argument("--runpod-gate", action="store_true",
                        help="also run the real-path gates (requires CUDA)")
    parser.add_argument("--list", action="store_true", help="list checks and exit")
    args = parser.parse_args()

    if args.list:
        for check_id, title, _ in CHECKS:
            print(f"{check_id:2d}. {title}")
        print(" G. real-path equivalence + integration gate  [opt-in: --runpod-gate]")
        return 0

    assert DEVICE.type == "cpu", "the default suite must run on CPU"
    torch.set_num_threads(1)
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)

    src04 = FEDAVG_SCRIPT.read_text()
    src06 = SCAFFOLD_SCRIPT.read_text()
    src06a = Path(__file__).read_text()
    for path, source in ((FEDAVG_SCRIPT, src04), (SCAFFOLD_SCRIPT, src06)):
        ast.parse(source, filename=str(path))   # both scripts must parse

    ctx = {
        "m04": load_module(FEDAVG_SCRIPT, "d2_04_fedavg"),
        "m06": load_module(SCAFFOLD_SCRIPT, "d2_06_scaffold"),
        "src06": src06, "src06a": src06a,
        "sq04": "".join(src04.split()), "sq06": "".join(src06.split()),
        "report": {},
    }

    print(f"verifier: device={DEVICE} torch={torch.__version__}")
    print(f"trainer={SCAFFOLD_SCRIPT}  reference={FEDAVG_SCRIPT}")
    print(f"execution gate: SCAFFOLD_EXECUTION_ENABLED={ctx['m06'].SCAFFOLD_EXECUTION_ENABLED}")
    print("no Dataset-2 round, partition or production matrix is executed\n", flush=True)

    passed, failed = 0, []
    for check_id, title, function in CHECKS:
        try:
            function(ctx)
        except Exception as error:  # noqa: BLE001 - the verifier reports, it does not raise
            failed.append((check_id, title, error))
            print(f"[{check_id:2d}] FAIL  {title}\n      {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
        else:
            passed += 1
            print(f"[{check_id:2d}] PASS  {title}", flush=True)

    print(f"\npassed={passed} failed={len(failed)}")
    for key, value in ctx["report"].items():
        print(f"  {key}: {value}")
    print(f"DEFAULT SUITE: {'PASS' if not failed else 'FAIL'}")

    if args.runpod_gate:
        if failed:
            print("\nRefusing to run the gate: the default suite failed.")
            return 1
        return runpod_gate(ctx)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
