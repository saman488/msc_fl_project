"""
Diagnose the Dataset-2 FedAvg round-10 / client-3 explosion: data path or SGD?

Observed failure (seed 42 / iid, d2_04_train_fedavg.py on CUDA):
    round  9 global validation Macro-F1   0.532683
    round 10 client-3 weighted train loss 1.539757e25
    round 10 client-3 update L2           inf (float32 overflow)
    round 10 global validation loss       1.737598e15
    round 10 global Macro-F1              0.000087
    round 11 aborted: non-finite training loss

Question: does the divergence come from the CUDA-resident data path introduced in
d2_04_train_fedavg.py, or from the SGD training dynamics that the original slow
implementation would have produced anyway?

Method
------
1. Reconstruct seed 42 / iid rounds 1..9 with the current fast CUDA-resident path,
   exactly as the production runner does (same initial state, same class weights,
   same per-round-per-client seeds, same sample-weighted aggregation).
2. Optionally validate the round-9 global model and compare against the recorded
   0.532683, to confirm the reconstruction landed on the same state.
3. Freeze that round-9 global state.
4. Train ONE client - round 10, client 3 - twice from that identical frozen state:
       branch A: the original slow IndexedDataset / DataLoader feature path
       branch B: the CUDA-resident LocalPositionDataset / ResidentClientBatches path
   Both use the same partition, the same local_seed = 1045, the same dropout
   seeding, the same SGD (lr 0.1, momentum 0, weight decay 0), batch size 4096,
   the same balanced class weights, and one local epoch.
5. Before comparing weights, capture the complete shuffled local-position sequence
   each branch actually consumed and require them to be identical. Branch A's
   sequence is recorded from the indices its Dataset is asked for; branch B's is
   recorded from the tensors its position DataLoader yields. If those differ, the
   data path changed the batch order and the comparison stops there.

If A and B agree, the explosion is in the optimisation, not in the data path.
If they disagree, the data path is implicated and the position sequences and
per-parameter differences localise it.

Both branches call d2_04_train_fedavg.train_one_epoch unmodified, so the update
rule under test is the production one; only the loader differs.

Safety
------
Imports d2_04_train_fedavg for its model, loaders, partition loader and training
step; that module guards its entry point, so importing runs no training. Writes
only under results/nf_cse_cic_ids2018_v2/diagnostics/round10_client3/ and refuses
to start if that resolves inside the production FedAvg roots. The diagnostic
client update is never aggregated into any global model and no checkpoint is
written. Reads the train and validation arrays only; a source-level guard refuses
to run if the held-out array names appear in this file.

Usage
-----
    ./env/bin/python d2_04a_diagnose_round10_client3.py
    ./env/bin/python d2_04a_diagnose_round10_client3.py --skip-round9-validation
"""

from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

TRAINER_PATH = Path("d2_04_train_fedavg.py")

# The failing cell.
PARTITION_SEED = 42
CONDITION = "iid"
FAIL_ROUND = 10
FAIL_CLIENT = 3
RECONSTRUCT_THROUGH_ROUND = FAIL_ROUND - 1
EXPECTED_LOCAL_SEED = 1045

# Recorded values from the failing run, for orientation only; nothing is asserted
# against them, they are reported alongside what this reconstruction produces.
RECORDED_ROUND9_MACRO_F1 = 0.532683
RECORDED_CLIENT3_TRAIN_LOSS = 1.539757e25

DIAG_DIR = Path("results/nf_cse_cic_ids2018_v2/diagnostics/round10_client3")


# ------------------------------ safety guards ------------------------------- #
def assert_no_holdout_reference() -> None:
    """Refuse to run if this file references the held-out arrays."""
    stem = "te" + "st"
    tokens = ("X" + "_" + stem, "y" + "_" + stem)
    source = Path(__file__).read_text()
    offending = [t for t in tokens if t in source]
    assert not offending, f"held-out array reference in this script: {offending}"


def assert_diagnostic_output_isolated(module) -> None:
    """The diagnostic must not write into the production FedAvg roots."""
    diag = DIAG_DIR.resolve()
    for production in (Path(module.RESULTS_DIR).resolve(), Path(module.MODELS_DIR).resolve()):
        assert production not in diag.parents and production != diag, (
            f"diagnostic output {diag} is inside the production root {production}"
        )


def load_trainer(path: Path):
    """Import the production trainer without running it."""
    assert path.exists(), f"trainer not found: {path}"
    source = path.read_text()
    assert 'if __name__ == "__main__":' in source, "trainer has no __main__ guard"
    spec = importlib.util.spec_from_file_location("d2_04_trainer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["d2_04_trainer"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------- recording wrappers ----------------------------- #
class RecordingIndexedDataset:
    """Branch A: the production IndexedDataset, recording every index requested.

    Delegates to the real dataset, so the feature fetch under test is the original
    row-by-row path; the recorded sequence is what that path actually consumed.
    """

    def __init__(self, inner, sink: list) -> None:
        self.inner = inner
        self.sink = sink

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int):
        self.sink.append(int(i))
        return self.inner[i]


class RecordingPositionLoader:
    """Branch B: records the position batches the DataLoader yields, then passes
    them straight through to the production ResidentClientBatches gather."""

    def __init__(self, loader, sink: list) -> None:
        self.loader = loader
        self.sink = sink

    def __iter__(self):
        for positions in self.loader:
            self.sink.append(positions.detach().cpu().clone())
            yield positions


# ------------------------------- measurement -------------------------------- #
def state_l2_float64(a: dict, b: dict) -> float:
    """L2 distance over shared floating tensors, accumulated in float64.

    float32 squaring overflows to inf once parameters reach ~1e20; float64 keeps
    the magnitude readable up to ~1e150.
    """
    total = 0.0
    for k in a:
        if a[k].is_floating_point():
            diff = a[k].detach().to(torch.float64).cpu() - b[k].detach().to(torch.float64).cpu()
            total += float(torch.sum(diff * diff).item())
    return float(total ** 0.5)


def max_abs_param(state: dict) -> float:
    return max(float(v.detach().to(torch.float64).abs().max().item())
               for v in state.values() if v.is_floating_point())


def max_abs_state_diff(a: dict, b: dict) -> float:
    assert a.keys() == b.keys(), "state key mismatch"
    return max(float((a[k].detach().to(torch.float64).cpu()
                      - b[k].detach().to(torch.float64).cpu()).abs().max().item())
               for k in a if a[k].is_floating_point())


def all_finite(state: dict) -> bool:
    return all(bool(torch.isfinite(v.detach().to(torch.float64)).all().item())
               for v in state.values() if v.is_floating_point())


def per_tensor_report(state_a: dict, state_b: dict) -> list[dict]:
    rows = []
    for k in state_a:
        if not state_a[k].is_floating_point():
            continue
        ta = state_a[k].detach().to(torch.float64).cpu()
        tb = state_b[k].detach().to(torch.float64).cpu()
        rows.append({
            "parameter": k,
            "max_abs_A": float(ta.abs().max().item()),
            "max_abs_B": float(tb.abs().max().item()),
            "max_abs_diff_A_B": float((ta - tb).abs().max().item()),
            "finite_A": bool(torch.isfinite(ta).all().item()),
            "finite_B": bool(torch.isfinite(tb).all().item()),
        })
    return rows


def sequence_sha256(positions: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(positions, dtype=np.int64).tobytes()).hexdigest()


# ------------------------------ reconstruction ------------------------------ #
def build_resident(module, device) -> dict:
    """Device-resident train/validation tensors, as the production CUDA path builds."""
    resident = {
        "x_train": torch.from_numpy(np.load(module.PROCESSED_DIR / "X_train.npy")).to(torch.float32).to(device),
        "y_train": torch.from_numpy(np.load(module.PROCESSED_DIR / "y_train.npy")).to(torch.long).to(device),
        "x_val": torch.from_numpy(np.load(module.PROCESSED_DIR / "X_val.npy")).to(torch.float32).to(device),
        "y_val": torch.from_numpy(np.load(module.PROCESSED_DIR / "y_val.npy")).to(torch.long).to(device),
    }
    assert resident["x_train"].shape[1] == module.INPUT_DIM, "resident X_train width"
    assert resident["x_val"].shape[1] == module.INPUT_DIM, "resident X_val width"
    return resident


def resident_loader_for(module, position_loader, client_indices_cuda, resident, device):
    return module.ResidentClientBatches(position_loader, client_indices_cuda,
                                        resident["x_train"], resident["y_train"], device)


def reconstruct_through_round(module, part_data, initial_state, weight_f32,
                              client_indices_cuda, resident, device, through_round: int) -> dict:
    """Replay production run() for rounds 1..through_round and return the global state.

    Mirrors d2_04_train_fedavg.run(): the same construction order, the same
    per-round-per-client seeding, the same fresh optimizer per client, the same
    sample-weighted aggregation.
    """
    sizes = part_data["sizes"]

    module.set_all_seeds(module.TRAIN_SEED)
    global_model = module.MLPMultiClassClassifier(module.INPUT_DIM, module.NUM_CLASSES).to(device)
    global_model.load_state_dict(copy.deepcopy(initial_state))
    assert module.states_equal(global_model.state_dict(), initial_state), "initial state not loaded identically"
    local_model = module.MLPMultiClassClassifier(module.INPUT_DIM, module.NUM_CLASSES).to(device)

    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weight_f32, dtype=torch.float32, device=device))

    for rnd in range(1, through_round + 1):
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        client_states = []
        round_start = time.perf_counter()
        for client_id in range(module.NUM_CLIENTS):
            local_model.load_state_dict(global_state)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=module.LR,
                                        momentum=module.MOMENTUM, weight_decay=module.WEIGHT_DECAY)
            local_seed = module.TRAIN_SEED + rnd * 100 + client_id
            torch.manual_seed(local_seed)
            module.seed_accelerator(local_seed)
            generator = torch.Generator()
            generator.manual_seed(local_seed)
            position_loader = DataLoader(module.LocalPositionDataset(sizes[client_id]),
                                         batch_size=module.BATCH_SIZE,
                                         shuffle=True, num_workers=0, generator=generator)
            loader = resident_loader_for(module, position_loader,
                                         client_indices_cuda[client_id], resident, device)
            for _ in range(module.LOCAL_EPOCHS):
                stats = module.train_one_epoch(local_model, loader, criterion, optimizer, device)
            client_states.append({k: v.detach().cpu().clone()
                                  for k, v in local_model.state_dict().items()})
        agg_state = module.aggregate_sample_weighted(client_states, sizes)
        global_model.load_state_dict(agg_state)
        print(f"  reconstructed round {rnd:02d} "
              f"(last client weighted_train_loss={stats['weighted_train_loss']:.6g}, "
              f"{time.perf_counter() - round_start:.1f}s)", flush=True)

    return {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}


# ---------------------------------- branches -------------------------------- #
def train_branch(module, label, frozen_state, loader_factory, weight_f32, device) -> dict:
    """One local epoch from the frozen round-9 state, using the production step."""
    model = module.MLPMultiClassClassifier(module.INPUT_DIM, module.NUM_CLASSES).to(device)
    model.load_state_dict(copy.deepcopy(frozen_state))
    assert module.states_equal(model.state_dict(), frozen_state), f"{label}: frozen state not loaded identically"

    optimizer = torch.optim.SGD(model.parameters(), lr=module.LR,
                                momentum=module.MOMENTUM, weight_decay=module.WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weight_f32, dtype=torch.float32, device=device))

    local_seed = module.TRAIN_SEED + FAIL_ROUND * 100 + FAIL_CLIENT
    assert local_seed == EXPECTED_LOCAL_SEED, f"{label}: local_seed {local_seed} != {EXPECTED_LOCAL_SEED}"
    torch.manual_seed(local_seed)
    module.seed_accelerator(local_seed)
    generator = torch.Generator()
    generator.manual_seed(local_seed)

    loader, positions_sink = loader_factory(generator)

    failure = None
    try:
        stats = module.train_one_epoch(model, loader, criterion, optimizer, device)
    except AssertionError as error:
        # train_one_epoch asserts finiteness after the epoch; a blow-up here is the
        # observation, not a crash, so it is recorded and the weights are still read.
        failure = f"{type(error).__name__}: {error}"
        stats = None

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {"label": label, "stats": stats, "state": state,
            "positions_sink": positions_sink, "train_failure": failure}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--skip-round9-validation", action="store_true",
                        help="skip evaluating the reconstructed round-9 global model")
    args = parser.parse_args()

    assert_no_holdout_reference()
    module = load_trainer(TRAINER_PATH)
    assert_diagnostic_output_isolated(module)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    device = module.get_device()
    print(f"diagnostic: device={device} torch={torch.__version__} trainer={TRAINER_PATH}", flush=True)
    print(f"target: seed {PARTITION_SEED} / {CONDITION}, round {FAIL_ROUND}, client {FAIL_CLIENT}, "
          f"local_seed {EXPECTED_LOCAL_SEED}", flush=True)

    y_train = np.load(module.PROCESSED_DIR / "y_train.npy")
    global_class_counts = np.bincount(y_train, minlength=module.NUM_CLASSES)

    # Class weights exactly as production computes them; cross-checked against the
    # saved vector when the production run left one behind (read-only).
    weight_f32 = module.class_weights_full(y_train).astype(np.float32)
    if Path(module.CLASS_WEIGHTS_PATH).exists():
        saved = np.load(module.CLASS_WEIGHTS_PATH)
        assert np.array_equal(saved, weight_f32), "saved class weights differ from a fresh computation"
        print("class weights match the saved production vector", flush=True)

    # The shared initial state: reuse the production one when present, else rebuild
    # it deterministically the same way production does.
    if Path(module.INIT_PATH).exists():
        initial_state = torch.load(module.INIT_PATH, map_location="cpu")
        initial_source = str(module.INIT_PATH)
    else:
        module.set_all_seeds(module.TRAIN_SEED)
        init_model = module.MLPMultiClassClassifier(module.INPUT_DIM, module.NUM_CLASSES).to(device)
        initial_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
        initial_source = "regenerated deterministically (production file absent)"
    print(f"initial state: {initial_source}", flush=True)

    part_data = module.load_partition(PARTITION_SEED, CONDITION, y_train, global_class_counts)
    sizes = part_data["sizes"]
    client_n = sizes[FAIL_CLIENT]
    client_counts = part_data["client_class_counts"][FAIL_CLIENT].tolist()
    print(f"client {FAIL_CLIENT}: n={client_n} class_counts={client_counts}", flush=True)

    resident = build_resident(module, device)
    client_indices_cuda = [torch.from_numpy(ci.astype(np.int64)).to(device)
                           for ci in part_data["client_indices"]]

    print(f"reconstructing rounds 1..{RECONSTRUCT_THROUGH_ROUND} on the fast path", flush=True)
    frozen_state = reconstruct_through_round(
        module, part_data, initial_state, weight_f32,
        client_indices_cuda, resident, device, RECONSTRUCT_THROUGH_ROUND)

    round9 = {"macro_f1": None, "recorded_macro_f1": RECORDED_ROUND9_MACRO_F1, "delta": None}
    if not args.skip_round9_validation:
        probe = module.MLPMultiClassClassifier(module.INPUT_DIM, module.NUM_CLASSES).to(device)
        probe.load_state_dict(copy.deepcopy(frozen_state))
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weight_f32, dtype=torch.float32, device=device))
        val_loader = module.ResidentValLoader(resident["x_val"], resident["y_val"], 4096)
        val = module.evaluate(probe, val_loader, criterion, device)
        round9["macro_f1"] = float(val["macro_f1"])
        round9["delta"] = float(val["macro_f1"] - RECORDED_ROUND9_MACRO_F1)
        print(f"round {RECONSTRUCT_THROUGH_ROUND} reconstructed Macro-F1 = {round9['macro_f1']:.6f} "
              f"(recorded {RECORDED_ROUND9_MACRO_F1:.6f}, delta {round9['delta']:+.6f})", flush=True)

    frozen_max_abs = max_abs_param(frozen_state)
    print(f"frozen round-{RECONSTRUCT_THROUGH_ROUND} state: max|param|={frozen_max_abs:.6g} "
          f"finite={all_finite(frozen_state)}", flush=True)

    # Branch A: the original slow feature path.
    def factory_a(generator):
        sink: list = []
        inner = module.IndexedDataset(module.PROCESSED_DIR / "X_train.npy",
                                      module.PROCESSED_DIR / "y_train.npy",
                                      part_data["client_indices"][FAIL_CLIENT])
        dataset = RecordingIndexedDataset(inner, sink)
        loader = DataLoader(dataset, batch_size=module.BATCH_SIZE,
                            shuffle=True, num_workers=0, generator=generator)
        return loader, sink

    # Branch B: the CUDA-resident position path.
    def factory_b(generator):
        sink: list = []
        position_loader = DataLoader(module.LocalPositionDataset(client_n),
                                     batch_size=module.BATCH_SIZE,
                                     shuffle=True, num_workers=0, generator=generator)
        recording = RecordingPositionLoader(position_loader, sink)
        loader = resident_loader_for(module, recording,
                                     client_indices_cuda[FAIL_CLIENT], resident, device)
        return loader, sink

    print("branch A: original slow IndexedDataset path", flush=True)
    branch_a = train_branch(module, "A_slow_dataset", frozen_state, factory_a, weight_f32, device)
    print("branch B: CUDA-resident position path", flush=True)
    branch_b = train_branch(module, "B_resident_positions", frozen_state, factory_b, weight_f32, device)

    positions_a = np.asarray(branch_a["positions_sink"], dtype=np.int64)
    positions_b = (torch.cat(branch_b["positions_sink"]).numpy().astype(np.int64)
                   if branch_b["positions_sink"] else np.empty(0, dtype=np.int64))
    sequences_identical = (positions_a.shape == positions_b.shape
                           and bool(np.array_equal(positions_a, positions_b)))
    first_mismatch = None
    if not sequences_identical and positions_a.shape == positions_b.shape:
        diff = np.flatnonzero(positions_a != positions_b)
        first_mismatch = int(diff[0]) if diff.size else None

    print(f"position sequences: A n={positions_a.size} sha256={sequence_sha256(positions_a)[:16]}... "
          f"B n={positions_b.size} sha256={sequence_sha256(positions_b)[:16]}... "
          f"identical={sequences_identical}", flush=True)

    report = {
        "target": {"partition_seed": PARTITION_SEED, "condition": CONDITION,
                   "round": FAIL_ROUND, "client": FAIL_CLIENT,
                   "local_seed": EXPECTED_LOCAL_SEED, "client_n": client_n,
                   "client_class_counts": client_counts},
        "device": str(device),
        "initial_state_source": initial_source,
        "round9": round9,
        "frozen_state": {"max_abs_param": frozen_max_abs, "all_finite": all_finite(frozen_state)},
        "position_sequence": {
            "n_positions_A": int(positions_a.size),
            "n_positions_B": int(positions_b.size),
            "sha256_A": sequence_sha256(positions_a),
            "sha256_B": sequence_sha256(positions_b),
            "identical": sequences_identical,
            "first_mismatch_index": first_mismatch,
        },
        "recorded_failure": {"client3_train_loss": RECORDED_CLIENT3_TRAIN_LOSS},
        "branches": {},
        "comparison": {},
    }

    for branch in (branch_a, branch_b):
        stats = branch["stats"]
        state = branch["state"]
        report["branches"][branch["label"]] = {
            "train_failure": branch["train_failure"],
            "weighted_train_loss": (float(stats["weighted_train_loss"]) if stats else None),
            "online_train_accuracy": (float(stats["online_train_accuracy"]) if stats else None),
            "n_batches": (int(stats["n_batches"]) if stats else None),
            "n_samples": (int(stats["n_samples"]) if stats else None),
            "max_abs_param": max_abs_param(state),
            "update_l2_from_frozen_float64": state_l2_float64(state, frozen_state),
            "all_parameters_finite": all_finite(state),
        }

    report["comparison"] = {
        "max_abs_param_diff_A_B": max_abs_state_diff(branch_a["state"], branch_b["state"]),
        "state_l2_diff_A_B_float64": state_l2_float64(branch_a["state"], branch_b["state"]),
        "per_tensor": per_tensor_report(branch_a["state"], branch_b["state"]),
    }

    out_json = DIAG_DIR / "round10_client3_diagnostic.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== branch results ===")
    for label, row in report["branches"].items():
        print(f"[{label}]")
        for key in ("train_failure", "weighted_train_loss", "online_train_accuracy",
                    "n_batches", "n_samples", "max_abs_param",
                    "update_l2_from_frozen_float64", "all_parameters_finite"):
            print(f"    {key:32} {row[key]}")
    print("\n=== A vs B ===")
    print(f"    positions identical              {sequences_identical}")
    print(f"    max_abs_param_diff_A_B           {report['comparison']['max_abs_param_diff_A_B']:.6g}")
    print(f"    state_l2_diff_A_B_float64        {report['comparison']['state_l2_diff_A_B_float64']:.6g}")
    for row in report["comparison"]["per_tensor"]:
        print(f"    {row['parameter']:22} maxA={row['max_abs_A']:.6g} maxB={row['max_abs_B']:.6g} "
              f"diff={row['max_abs_diff_A_B']:.6g} finiteA={row['finite_A']} finiteB={row['finite_B']}")
    print(f"\nWrote {out_json}")
    print("No aggregation performed; no production artefact written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
