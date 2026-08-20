"""
Batch-level trace of Dataset-2 FedAvg round 10 / client 3 (seed 42 / iid).

d2_04a established whether the slow and resident data paths agree. This script
answers the next question: at which minibatch does the local model leave the
normal numerical regime, and what is in that batch.

Method
------
Rebuild the round-9 global state with the verified reconstruction from
d2_04a_diagnose_round10_client3.py (same initial state, same class weights, same
per-round-per-client seeds, same sample-weighted aggregation), freeze it, then run
round 10 / client 3 on the current CUDA-resident path with local_seed 1045 - the
same partition, dropout seeding, SGD (lr 0.1, momentum 0, weight decay 0), batch
size 4096, balanced class weights and one local epoch as production.

The local epoch is stepped here rather than through train_one_epoch, because
per-batch instrumentation is the point. The sequence of operations is identical -
zero_grad, forward, weighted CE, backward, step - and nothing about the training
configuration is altered. The added measurements are reads only: they consume no
RNG and change no tensor, so the trajectory is the production one.

Per batch this records: batch number, class counts, max |input feature|, the
weighted CE loss before the update, max |logit|, total gradient L2 in float64
before optimizer.step(), max gradient magnitude, parameter L2 and max |parameter|
after the step, and finiteness of loss, logits, gradients and parameters.

Stopping
--------
The epoch runs to completion. The loop stops early only if the loss, the logits,
the gradients or the parameters become non-finite, because past that point the
recorded numbers carry no information. There is no magnitude-based early stop:
nothing is judged during the epoch, so no threshold can end the trace before the
data is in.

Analysis
--------
Detection happens afterwards, over the complete recorded sequence. For each of the
weighted CE loss, the gradient L2 and max |parameter|, the script reports the
first batch whose value exceeds ten times the median of all preceding finite
batches, together with that median, the ratio, and the five batches before the
transition.

For the earliest of those three transitions it then reports full forensics: the
global training row indices, class ids and counts, per-feature min/max, the
largest absolute feature (with feature index, position in batch and global row),
the logits, the gradients, and the class weight of every label present. Features
for that batch are re-gathered from the resident tensors using the recorded row
indices - the same values the batch was trained on, since only the model changes
during the epoch. Logits and gradients depend on the model state at that batch, so
they are retained per batch as the epoch runs (order 130 MB of host memory for a
client of this size).

Safety
------
Imports d2_04_train_fedavg and d2_04a_diagnose_round10_client3 for their model,
loaders, partition loader and reconstruction; both guard their entry points, so
importing runs nothing. Writes only under
results/nf_cse_cic_ids2018_v2/diagnostics/round10_client3_batch_trace/ and refuses
to start if that resolves inside the production FedAvg roots. No aggregation, no
checkpoint, no production artefact touched. Reads the train and validation arrays
only; a source-level guard refuses to run if the held-out array names appear here.

Usage
-----
    ./env/bin/python d2_04b_trace_round10_client3_batches.py
    ./env/bin/python d2_04b_trace_round10_client3_batches.py --skip-round9-validation
"""

from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

TRAINER_PATH = Path("d2_04_train_fedavg.py")
DIAGNOSTIC_A_PATH = Path("d2_04a_diagnose_round10_client3.py")

PARTITION_SEED = 42
CONDITION = "iid"
FAIL_ROUND = 10
FAIL_CLIENT = 3
RECONSTRUCT_THROUGH_ROUND = FAIL_ROUND - 1
EXPECTED_LOCAL_SEED = 1045

DIAG_DIR = Path("results/nf_cse_cic_ids2018_v2/diagnostics/round10_client3_batch_trace")

# Post-hoc analysis only. Nothing here is consulted during the epoch.
TRANSITION_FACTOR = 10.0
CONTEXT_BATCHES = 5

# The three tracked quantities, in the order they are reported.
TRACKED = (
    ("weighted_ce_loss", "weighted CE loss"),
    ("grad_l2_float64", "gradient L2"),
    ("max_abs_param_after", "max |parameter|"),
)


def assert_no_holdout_reference() -> None:
    """Refuse to run if this file references the held-out arrays."""
    stem = "te" + "st"
    tokens = ("X" + "_" + stem, "y" + "_" + stem)
    source = Path(__file__).read_text()
    offending = [t for t in tokens if t in source]
    assert not offending, f"held-out array reference in this script: {offending}"


def assert_diagnostic_output_isolated(module) -> None:
    diag = DIAG_DIR.resolve()
    for production in (Path(module.RESULTS_DIR).resolve(), Path(module.MODELS_DIR).resolve()):
        assert production not in diag.parents and production != diag, (
            f"diagnostic output {diag} is inside the production root {production}"
        )


def load_guarded_module(path: Path, name: str):
    """Import a script without running it."""
    assert path.exists(), f"required script not found: {path}"
    source = path.read_text()
    assert 'if __name__ == "__main__":' in source, f"{path}: no __main__ guard"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------- measurement -------------------------------- #
def grad_l2_float64(model) -> tuple[float, float, bool]:
    """(total gradient L2 in float64, max |gradient|, all gradients finite)."""
    total = 0.0
    max_abs = 0.0
    finite = True
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        assert param.grad is not None, f"no gradient for trainable parameter {name}"
        g = param.grad.detach().to(torch.float64)
        total += float(torch.sum(g * g).item())
        max_abs = max(max_abs, float(g.abs().max().item()))
        finite = finite and bool(torch.isfinite(g).all().item())
    return float(total ** 0.5), max_abs, finite


def param_l2_float64(model) -> tuple[float, float, bool]:
    """(parameter L2 in float64, max |parameter|, all parameters finite)."""
    total = 0.0
    max_abs = 0.0
    finite = True
    for param in model.parameters():
        p = param.detach().to(torch.float64)
        total += float(torch.sum(p * p).item())
        max_abs = max(max_abs, float(p.abs().max().item()))
        finite = finite and bool(torch.isfinite(p).all().item())
    return float(total ** 0.5), max_abs, finite


def parameter_layout(model) -> list[tuple[str, int]]:
    return [(name, int(p.numel())) for name, p in model.named_parameters() if p.requires_grad]


def gradient_vector(model) -> torch.Tensor:
    """Flat host copy of the current gradients, in named-parameter order."""
    return torch.cat([p.grad.detach().reshape(-1).to(torch.float32).cpu()
                      for p in model.parameters() if p.requires_grad])


def nonfinite_reasons(record: dict) -> list[str]:
    """Only reason the loop may stop early: the numbers stopped meaning anything."""
    reasons = []
    if not record["loss_finite"]:
        reasons.append("non-finite loss")
    if not record["logits_finite"]:
        reasons.append("non-finite logits")
    if not record["grad_finite"]:
        reasons.append("non-finite gradient")
    if not record["param_finite"]:
        reasons.append("non-finite parameter")
    return reasons


# -------------------------------- analysis ---------------------------------- #
def first_ratio_transition(values: list[float], factor: float) -> dict | None:
    """First index whose value exceeds factor x the median of all preceding finite values.

    Indices are 0-based into the record list; batch numbers are 1-based.
    """
    preceding: list[float] = []
    for index, value in enumerate(values):
        if preceding and np.isfinite(value):
            median = float(np.median(preceding))
            if median > 0.0 and value > factor * median:
                return {"index": index, "batch": index + 1, "value": float(value),
                        "preceding_median": median, "ratio": float(value / median),
                        "preceding_finite_batches": len(preceding)}
        if np.isfinite(value):
            preceding.append(float(value))
    return None


def context_slice(records: list[dict], index: int, count: int) -> list[dict]:
    """The `count` batches before `index`, plus the batch at `index`."""
    return records[max(0, index - count):index + 1]


# ------------------------------- forensics ---------------------------------- #
def batch_forensics(features: torch.Tensor, labels: torch.Tensor, rows: np.ndarray,
                    logits: torch.Tensor, gradients: torch.Tensor,
                    layout: list[tuple[str, int]], weight_f32: np.ndarray,
                    class_names: list[str], num_classes: int) -> dict:
    """Everything about the transition batch's inputs, outputs and gradients."""
    features_cpu = features.detach().to(torch.float64).cpu()
    labels_cpu = labels.detach().cpu().numpy().astype(np.int64)
    logits_cpu = logits.detach().to(torch.float64).cpu()

    counts = np.bincount(labels_cpu, minlength=num_classes)
    present = [int(c) for c in range(num_classes) if counts[c] > 0]

    feature_min = features_cpu.min(dim=0).values.numpy()
    feature_max = features_cpu.max(dim=0).values.numpy()

    abs_features = features_cpu.abs()
    flat_argmax = int(torch.argmax(abs_features).item())
    n_features = int(features_cpu.shape[1])
    arg_row = flat_argmax // n_features
    arg_col = flat_argmax % n_features

    grads64 = gradients.to(torch.float64)
    per_parameter = []
    offset = 0
    for name, numel in layout:
        chunk = grads64[offset:offset + numel]
        per_parameter.append({
            "parameter": name,
            "numel": int(numel),
            "grad_l2": float(torch.sqrt(torch.sum(chunk * chunk)).item()),
            "grad_max_abs": float(chunk.abs().max().item()),
            "grad_mean": float(chunk.mean().item()),
            "grad_finite": bool(torch.isfinite(chunk).all().item()),
        })
        offset += numel
    assert offset == grads64.numel(), "gradient layout does not cover the gradient vector"

    return {
        "batch_rows": int(features_cpu.shape[0]),
        "n_features": n_features,
        "global_row_indices_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
        "global_row_indices_head": rows[:50].tolist(),
        "global_row_indices_tail": rows[-50:].tolist(),
        "class_ids_present": present,
        "class_counts": counts.tolist(),
        "class_names_present": [class_names[c] for c in present],
        "class_weights_present": {class_names[c]: float(weight_f32[c]) for c in present},
        "per_feature_min": [float(v) for v in feature_min],
        "per_feature_max": [float(v) for v in feature_max],
        "largest_abs_feature": {
            "feature_index": int(arg_col),
            "value": float(features_cpu[arg_row, arg_col].item()),
            "position_in_batch": int(arg_row),
            "global_row_index": int(rows[arg_row]),
            "label": int(labels_cpu[arg_row]),
            "label_name": class_names[int(labels_cpu[arg_row])],
        },
        "max_abs_feature_overall": float(abs_features.max().item()),
        "features_finite": bool(torch.isfinite(features_cpu).all().item()),
        "logits": {
            "shape": list(logits_cpu.shape),
            "max_abs": float(logits_cpu.abs().max().item()),
            "min": float(logits_cpu.min().item()),
            "max": float(logits_cpu.max().item()),
            "mean": float(logits_cpu.mean().item()),
            "per_class_min": [float(v) for v in logits_cpu.min(dim=0).values.numpy()],
            "per_class_max": [float(v) for v in logits_cpu.max(dim=0).values.numpy()],
            "per_class_mean": [float(v) for v in logits_cpu.mean(dim=0).numpy()],
            "finite": bool(torch.isfinite(logits_cpu).all().item()),
        },
        "gradients": {
            "numel": int(grads64.numel()),
            "l2": float(torch.sqrt(torch.sum(grads64 * grads64)).item()),
            "max_abs": float(grads64.abs().max().item()),
            "finite": bool(torch.isfinite(grads64).all().item()),
            "per_parameter": per_parameter,
        },
    }


def print_records(records: list[dict], title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'batch':>6} {'loss':>14} {'max|logit|':>13} {'gradL2':>14} "
          f"{'max|grad|':>13} {'paramL2':>14} {'max|param|':>13} {'max|x|':>11} {'finite':>7}")
    for r in records:
        finite = "yes" if (r["loss_finite"] and r["logits_finite"]
                           and r["grad_finite"] and r["param_finite"]) else "NO"
        print(f"{r['batch']:>6} {r['weighted_ce_loss']:>14.6g} {r['max_abs_logit']:>13.6g} "
              f"{r['grad_l2_float64']:>14.6g} {r['max_abs_grad']:>13.6g} "
              f"{r['param_l2_float64']:>14.6g} {r['max_abs_param_after']:>13.6g} "
              f"{r['max_abs_feature']:>11.6g} {finite:>7}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch trace of round 10 / client 3.")
    parser.add_argument("--skip-round9-validation", action="store_true",
                        help="skip evaluating the reconstructed round-9 global model")
    args = parser.parse_args()

    assert_no_holdout_reference()
    diag_a = load_guarded_module(DIAGNOSTIC_A_PATH, "d2_04a_diagnostic")
    module = diag_a.load_trainer(TRAINER_PATH)
    assert_diagnostic_output_isolated(module)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    device = module.get_device()
    print(f"trace: device={device} torch={torch.__version__} trainer={TRAINER_PATH}", flush=True)
    print(f"target: seed {PARTITION_SEED} / {CONDITION}, round {FAIL_ROUND}, "
          f"client {FAIL_CLIENT}, local_seed {EXPECTED_LOCAL_SEED}", flush=True)

    class_names = module.load_class_names()
    y_train = np.load(module.PROCESSED_DIR / "y_train.npy")
    global_class_counts = np.bincount(y_train, minlength=module.NUM_CLASSES)

    weight_f32 = module.class_weights_full(y_train).astype(np.float32)
    if Path(module.CLASS_WEIGHTS_PATH).exists():
        saved = np.load(module.CLASS_WEIGHTS_PATH)
        assert np.array_equal(saved, weight_f32), "saved class weights differ from a fresh computation"

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
    print(f"client {FAIL_CLIENT}: n={client_n} "
          f"class_counts={part_data['client_class_counts'][FAIL_CLIENT].tolist()}", flush=True)

    resident = diag_a.build_resident(module, device)
    client_indices_cuda = [torch.from_numpy(ci.astype(np.int64)).to(device)
                           for ci in part_data["client_indices"]]

    print(f"reconstructing rounds 1..{RECONSTRUCT_THROUGH_ROUND} on the resident path", flush=True)
    frozen_state = diag_a.reconstruct_through_round(
        module, part_data, initial_state, weight_f32,
        client_indices_cuda, resident, device, RECONSTRUCT_THROUGH_ROUND)

    round9_macro_f1 = None
    if not args.skip_round9_validation:
        probe = module.MLPMultiClassClassifier(module.INPUT_DIM, module.NUM_CLASSES).to(device)
        probe.load_state_dict(copy.deepcopy(frozen_state))
        probe_criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weight_f32, dtype=torch.float32, device=device))
        val_loader = module.ResidentValLoader(resident["x_val"], resident["y_val"], 4096)
        round9_macro_f1 = float(module.evaluate(probe, val_loader, probe_criterion, device)["macro_f1"])
        print(f"round {RECONSTRUCT_THROUGH_ROUND} reconstructed Macro-F1 = {round9_macro_f1:.6f} "
              f"(recorded {diag_a.RECORDED_ROUND9_MACRO_F1:.6f})", flush=True)

    # Round 10 / client 3, exactly as production sets it up.
    model = module.MLPMultiClassClassifier(module.INPUT_DIM, module.NUM_CLASSES).to(device)
    model.load_state_dict(copy.deepcopy(frozen_state))
    assert module.states_equal(model.state_dict(), frozen_state), "frozen state not loaded identically"
    frozen_l2, frozen_max_abs, frozen_finite = param_l2_float64(model)
    print(f"frozen round-{RECONSTRUCT_THROUGH_ROUND} state: L2={frozen_l2:.6g} "
          f"max|param|={frozen_max_abs:.6g} finite={frozen_finite}", flush=True)

    layout = parameter_layout(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=module.LR,
                                momentum=module.MOMENTUM, weight_decay=module.WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weight_f32, dtype=torch.float32, device=device))

    local_seed = module.TRAIN_SEED + FAIL_ROUND * 100 + FAIL_CLIENT
    assert local_seed == EXPECTED_LOCAL_SEED, f"local_seed {local_seed} != {EXPECTED_LOCAL_SEED}"
    torch.manual_seed(local_seed)
    module.seed_accelerator(local_seed)
    generator = torch.Generator()
    generator.manual_seed(local_seed)

    position_sink: list = []
    position_loader = DataLoader(module.LocalPositionDataset(client_n),
                                 batch_size=module.BATCH_SIZE,
                                 shuffle=True, num_workers=0, generator=generator)
    recording = diag_a.RecordingPositionLoader(position_loader, position_sink)
    loader = module.ResidentClientBatches(recording, client_indices_cuda[FAIL_CLIENT],
                                          resident["x_train"], resident["y_train"], device)

    records: list[dict] = []
    kept_rows: list[np.ndarray] = []
    kept_logits: list[torch.Tensor] = []
    kept_gradients: list[torch.Tensor] = []
    stopped_nonfinite = None

    model.train()
    for batch_index, (features, labels) in enumerate(loader, start=1):
        positions = position_sink[-1].to(device)
        rows = client_indices_cuda[FAIL_CLIENT][positions]

        max_abs_feature = float(features.detach().to(torch.float64).abs().max().item())
        class_counts = np.bincount(labels.detach().cpu().numpy().astype(np.int64),
                                   minlength=module.NUM_CLASSES)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss_value = float(loss.detach().to(torch.float64).item())
        logits_detached = logits.detach()
        max_abs_logit = float(logits_detached.to(torch.float64).abs().max().item())
        logits_finite = bool(torch.isfinite(logits_detached).all().item())
        loss.backward()
        grad_l2, max_abs_grad, grad_finite = grad_l2_float64(model)
        grad_vec = gradient_vector(model)
        optimizer.step()
        param_l2, max_abs_param, param_finite = param_l2_float64(model)

        record = {
            "batch": batch_index,
            "n_samples": int(labels.shape[0]),
            "max_abs_feature": max_abs_feature,
            "weighted_ce_loss": loss_value,
            "max_abs_logit": max_abs_logit,
            "grad_l2_float64": grad_l2,
            "max_abs_grad": max_abs_grad,
            "param_l2_float64": param_l2,
            "max_abs_param_after": max_abs_param,
            "loss_finite": bool(np.isfinite(loss_value)),
            "logits_finite": logits_finite,
            "grad_finite": grad_finite,
            "param_finite": param_finite,
        }
        for c in range(module.NUM_CLASSES):
            record[f"class_count_c{c}"] = int(class_counts[c])
        records.append(record)

        # Retained so the transition batch can be reported in full afterwards.
        kept_rows.append(rows.detach().cpu().numpy().astype(np.int64))
        kept_logits.append(logits_detached.to(torch.float32).cpu())
        kept_gradients.append(grad_vec)

        reasons = nonfinite_reasons(record)
        if reasons:
            stopped_nonfinite = {"batch": batch_index, "reasons": reasons}
            print(f"\nstopping at batch {batch_index}: " + "; ".join(reasons), flush=True)
            break

    print(f"traced {len(records)} batches"
          + ("" if stopped_nonfinite else " (full local epoch, all values finite)"), flush=True)

    trace_df = pd.DataFrame(records)
    trace_path = DIAG_DIR / "batch_trace.csv"
    trace_df.to_csv(trace_path, index=False)

    # --- post-hoc transition detection over the complete recorded sequence --- #
    transitions = {}
    for key, label in TRACKED:
        found = first_ratio_transition([r[key] for r in records], TRANSITION_FACTOR)
        transitions[key] = found
        if found is None:
            print(f"\n{label}: no batch exceeded {TRANSITION_FACTOR:g}x the preceding median", flush=True)
        else:
            print(f"\n{label}: first >{TRANSITION_FACTOR:g}x jump at batch {found['batch']} "
                  f"value={found['value']:.6g} preceding_median={found['preceding_median']:.6g} "
                  f"ratio={found['ratio']:.6g} (over {found['preceding_finite_batches']} preceding "
                  f"finite batches)", flush=True)
            print_records(context_slice(records, found["index"], CONTEXT_BATCHES),
                          f"{label}: batches around the transition (transition batch is last)")

    detected = [(key, t) for key, t in transitions.items() if t is not None]
    earliest_key, earliest = (min(detected, key=lambda kt: kt[1]["index"])
                              if detected else (None, None))

    forensics = None
    if earliest is not None:
        index = earliest["index"]
        rows_np = kept_rows[index]
        # Features are a deterministic function of the row indices, so they are
        # re-gathered rather than retained; only the model changed during the epoch.
        rows_dev = torch.from_numpy(rows_np).to(device)
        features_again = resident["x_train"][rows_dev]
        labels_again = resident["y_train"][rows_dev]
        forensics = batch_forensics(features_again, labels_again, rows_np,
                                    kept_logits[index], kept_gradients[index],
                                    layout, weight_f32, class_names, module.NUM_CLASSES)

        np.save(DIAG_DIR / "transition_batch_row_indices.npy", rows_np)
        np.save(DIAG_DIR / "transition_batch_logits.npy", kept_logits[index].numpy())
        np.save(DIAG_DIR / "transition_batch_gradients.npy", kept_gradients[index].numpy())
        if index > 0:
            np.save(DIAG_DIR / "previous_batch_row_indices.npy", kept_rows[index - 1])

        label = dict(TRACKED)[earliest_key]
        print(f"\n=== earliest transition: batch {earliest['batch']} (first detected by {label}) ===")
        print(f"    rows in batch                 {forensics['batch_rows']}")
        print(f"    class ids present             {forensics['class_ids_present']}")
        print(f"    class counts                  {forensics['class_counts']}")
        print(f"    class weights present         {forensics['class_weights_present']}")
        print(f"    max |feature| in batch        {forensics['max_abs_feature_overall']:.6g}")
        largest = forensics["largest_abs_feature"]
        print(f"    largest |feature|             feature_index={largest['feature_index']} "
              f"value={largest['value']:.6g} position_in_batch={largest['position_in_batch']} "
              f"global_row={largest['global_row_index']} label={largest['label']} "
              f"({largest['label_name']})")
        print(f"    features all finite           {forensics['features_finite']}")
        lg = forensics["logits"]
        print(f"    logits                        shape={lg['shape']} max|logit|={lg['max_abs']:.6g} "
              f"min={lg['min']:.6g} max={lg['max']:.6g} mean={lg['mean']:.6g} finite={lg['finite']}")
        gr = forensics["gradients"]
        print(f"    gradients                     numel={gr['numel']} L2={gr['l2']:.6g} "
              f"max|grad|={gr['max_abs']:.6g} finite={gr['finite']}")
        for row in gr["per_parameter"]:
            print(f"        {row['parameter']:22} L2={row['grad_l2']:.6g} "
                  f"max|g|={row['grad_max_abs']:.6g} mean={row['grad_mean']:.6g} "
                  f"finite={row['grad_finite']}")
        print(f"    global row indices            n={forensics['batch_rows']} "
              f"sha256={forensics['global_row_indices_sha256'][:16]}... "
              f"(full array in transition_batch_row_indices.npy)")
        print("    per-feature min/max:")
        for i, (lo, hi) in enumerate(zip(forensics["per_feature_min"], forensics["per_feature_max"])):
            print(f"        f{i:02d} min={lo:>14.6g} max={hi:>14.6g}")
    else:
        print_records(records[-CONTEXT_BATCHES:], f"final {CONTEXT_BATCHES} batches")
        print(f"\nNo transition detected: no tracked quantity exceeded "
              f"{TRANSITION_FACTOR:g}x its preceding median over {len(records)} batches.")

    report = {
        "target": {"partition_seed": PARTITION_SEED, "condition": CONDITION,
                   "round": FAIL_ROUND, "client": FAIL_CLIENT,
                   "local_seed": EXPECTED_LOCAL_SEED, "client_n": client_n},
        "device": str(device),
        "initial_state_source": initial_source,
        "round9_reconstructed_macro_f1": round9_macro_f1,
        "round9_recorded_macro_f1": diag_a.RECORDED_ROUND9_MACRO_F1,
        "frozen_state": {"param_l2_float64": frozen_l2, "max_abs_param": frozen_max_abs,
                         "all_finite": frozen_finite},
        "stop_rule": "non-finite loss, logits, gradients or parameters only",
        "stopped_nonfinite": stopped_nonfinite,
        "batches_traced": len(records),
        "expected_batches": int(-(-client_n // module.BATCH_SIZE)),
        "transition_rule": {"factor": TRANSITION_FACTOR,
                            "baseline": "median of all preceding finite batches"},
        "transitions": {key: transitions[key] for key, _ in TRACKED},
        "earliest_transition": ({"detected_by": earliest_key, **earliest}
                                if earliest is not None else None),
        "earliest_transition_forensics": forensics,
        "trace_csv": str(trace_path),
    }
    report_path = DIAG_DIR / "batch_trace_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote {trace_path}")
    print(f"Wrote {report_path}")
    print("No aggregation performed; no production artefact written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
