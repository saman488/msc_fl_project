from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
TRAINER_SOURCE = ROOT / "44_train_fedavg_hd_selected.py"

EXPECTED_TRAINER_SHA256 = (
    "e27bb99fdc4168a6b5ceeafbe03e28634c56f3dc569b40c1692cca990c32f2c9"
)

VALID_CONDITIONS = ["iid", "hd_0p25", "hd_0p5", "hd_0p75", "hd_0p9"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trainer(train_seed: int, partition_root: Path):
    actual = file_sha256(TRAINER_SOURCE)
    if actual != EXPECTED_TRAINER_SHA256:
        raise RuntimeError(
            f"Trainer source changed: {actual} != {EXPECTED_TRAINER_SHA256}"
        )

    spec = importlib.util.spec_from_file_location(
        "verified_d1_fedavg_trainer", TRAINER_SOURCE
    )
    trainer = importlib.util.module_from_spec(spec)
    sys.modules["verified_d1_fedavg_trainer"] = trainer
    spec.loader.exec_module(trainer)

    trainer.PROCESSED_DIR = ROOT / "data" / "processed_37f"
    trainer.LABEL_MAP = ROOT / "configs" / "label_mapping.json"
    trainer.PART_ROOT = partition_root
    trainer.TRAIN_SEED = train_seed

    trainer.assert_no_test_reference()

    return trainer, actual


def run_one(args):
    trainer, trainer_hash = load_trainer(
        args.train_seed, args.partition_root
    )

    condition = args.single
    seed = args.partition_seed
    tag = f"fedavg37f_k{trainer.NUM_CLIENTS}_seed{seed}_{condition}"

    src_results = args.source_results_root / condition
    src_models = args.source_models_root / condition

    if args.source_format == "primary":
        src_config = src_results / f"config_{tag}.json"
        src_history = src_results / f"history_{tag}.csv"
        src_best = src_models / f"best_{tag}.pt"
        src_final = src_models / f"final_{tag}.pt"
    elif args.source_format == "extended":
        src_config = src_results / f"config_extended_{tag}.json"
        src_history = src_results / f"history_extended_{tag}.csv"
        src_best = src_models / f"best_extended_{tag}.pt"
        src_final = src_models / f"final_extended_{tag}.pt"
    else:
        raise RuntimeError(f"Unknown source format: {args.source_format}")

    # Class weights are invariant across continuation stages and remain
    # anchored to the original verified primary run.
    src_class_weights = (
        args.primary_results_root / condition / "class_weights.npy"
    )

    required = [
        src_config,
        src_history,
        src_class_weights,
        src_best,
        src_final,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "Missing source run files:\n  "
            + "\n  ".join(str(p) for p in missing)
        )

    config = json.loads(src_config.read_text())
    history_df = pd.read_csv(src_history)

    if len(history_df) == 0:
        raise RuntimeError(f"{condition}: empty history")

    last_round = int(history_df.iloc[-1]["round"])

    if list(history_df["round"].astype(int)) != list(
        range(1, last_round + 1)
    ):
        raise RuntimeError(
            f"{condition}: source history is not consecutive rounds 1..{last_round}"
        )

    if last_round != int(config["max_rounds"]):
        raise RuntimeError(
            f"{condition}: history ends at {last_round}, "
            f"config max_rounds={config['max_rounds']}"
        )

    if args.end_round <= last_round:
        raise RuntimeError(
            f"end_round={args.end_round} must be > source round {last_round}"
        )

    expected_settings = {
        "partition_seed": seed,
        "training_seed": args.train_seed,
        "local_epochs": trainer.LOCAL_EPOCHS,
        "batch_size": trainer.BATCH_SIZE,
        "lr": trainer.LR,
        "momentum": trainer.MOMENTUM,
        "weight_decay": trainer.WEIGHT_DECAY,
        "selection_metric": "val_macro_f1",
    }

    for key, expected in expected_settings.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"{condition}: config {key}={config.get(key)!r} "
                f"!= expected {expected!r}"
            )

    best_idx = history_df["macro_f1"].idxmax()
    best = {
        "round": int(history_df.loc[best_idx, "round"]),
        "macro_f1": float(history_df.loc[best_idx, "macro_f1"]),
    }

    if best["round"] != int(config["best_round"]):
        raise RuntimeError(
            f"{condition}: history best round {best['round']} "
            f"!= config best round {config['best_round']}"
        )

    if not np.isclose(
        best["macro_f1"],
        float(config["best_val_macro_f1"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{condition}: history/config best macro-F1 mismatch"
        )

    out_results = args.extension_root / "results" / condition
    out_models = args.extension_root / "models" / condition

    if out_results.exists() or out_models.exists():
        raise RuntimeError(
            f"Refusing existing extension output for {condition}"
        )

    out_results.mkdir(parents=True, exist_ok=False)
    out_models.mkdir(parents=True, exist_ok=False)

    out_history = out_results / f"history_extended_{tag}.csv"
    out_config = out_results / f"config_extended_{tag}.json"
    out_best = out_models / f"best_extended_{tag}.pt"
    out_final = out_models / f"final_extended_{tag}.pt"

    # Preserve the existing best checkpoint as best-so-far at round 40.
    shutil.copy2(src_best, out_best)

    device = trainer.get_device()

    y_train = np.load(trainer.PROCESSED_DIR / "y_train.npy")
    assert np.issubdtype(y_train.dtype, np.integer)
    assert y_train.min() >= 0
    assert y_train.max() <= trainer.NUM_CLASSES - 1

    global_class_counts = np.bincount(
        y_train, minlength=trainer.NUM_CLASSES
    )
    assert (global_class_counts > 0).all()

    x_train_peek = np.load(
        trainer.PROCESSED_DIR / "X_train.npy", mmap_mode="r"
    )
    x_val_peek = np.load(
        trainer.PROCESSED_DIR / "X_val.npy", mmap_mode="r"
    )

    assert x_train_peek.shape[1] == trainer.INPUT_DIM
    assert x_val_peek.shape[1] == trainer.INPUT_DIM

    # Recompute exactly as the original trainer and verify against saved file.
    weight_f32 = trainer.class_weights_full(y_train).astype(np.float32)
    saved_weight_f32 = np.load(src_class_weights)

    if not np.array_equal(weight_f32, saved_weight_f32):
        raise RuntimeError(
            f"{condition}: recomputed class weights differ from source run"
        )

    val_loader = DataLoader(
        trainer.FullDataset(
            trainer.PROCESSED_DIR / "X_val.npy",
            trainer.PROCESSED_DIR / "y_val.npy",
        ),
        batch_size=4096,
        shuffle=False,
        num_workers=0,
    )

    part_data = trainer.load_partition(
        seed,
        condition,
        y_train,
        global_class_counts,
    )

    datasets = part_data["datasets"]
    sizes = part_data["sizes"]
    total_size = float(sum(sizes))
    agg_weights = [n / total_size for n in sizes]

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            weight_f32,
            dtype=torch.float32,
            device=device,
        )
    )

    # Resume from FINAL round-40 global model, never from best checkpoint.
    final_state = torch.load(src_final, map_location="cpu")
    trainer.assert_state_finite(
        final_state, f"{tag} source final round {last_round}"
    )

    trainer.set_all_seeds(args.train_seed)

    global_model = trainer.MLPMultiClassClassifier(
        trainer.INPUT_DIM, trainer.NUM_CLASSES
    ).to(device)
    global_model.load_state_dict(copy.deepcopy(final_state))

    if not trainer.states_equal(
        global_model.state_dict(), final_state
    ):
        raise RuntimeError(
            f"{condition}: source final state not loaded identically"
        )

    local_model = trainer.MLPMultiClassClassifier(
        trainer.INPUT_DIM, trainer.NUM_CLASSES
    ).to(device)

    # Verify copied best checkpoint still reproduces source best validation F1.
    source_best_state = torch.load(out_best, map_location=device)
    trainer.assert_state_finite(
        source_best_state, f"{tag} source best checkpoint"
    )
    local_model.load_state_dict(source_best_state)

    source_best_reloaded_f1 = float(
        trainer.evaluate(
            local_model, val_loader, criterion, device
        )["macro_f1"]
    )

    if abs(source_best_reloaded_f1 - best["macro_f1"]) >= trainer.RELOAD_F1_TOL:
        raise RuntimeError(
            f"{condition}: source best checkpoint reload F1 "
            f"{source_best_reloaded_f1} != {best['macro_f1']}"
        )

    history = history_df.to_dict("records")

    print(
        f"EXTEND {tag}: rounds {last_round + 1}-{args.end_round}",
        flush=True,
    )

    for rnd in range(last_round + 1, args.end_round + 1):
        global_state = {
            k: v.detach().cpu().clone()
            for k, v in global_model.state_dict().items()
        }

        client_states = []
        participated = []
        client_stats = []
        client_seconds = []
        client_update_l2 = []

        loop_wall_start = time.perf_counter()

        for client_id in range(trainer.NUM_CLIENTS):
            local_model.load_state_dict(global_state)

            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=trainer.LR,
                momentum=trainer.MOMENTUM,
                weight_decay=trainer.WEIGHT_DECAY,
            )

            # EXACT original round/client RNG logic, now with rnd = 41..60.
            local_seed = (
                trainer.TRAIN_SEED + rnd * 100 + client_id
            )

            torch.manual_seed(local_seed)

            if torch.backends.mps.is_available():
                torch.mps.manual_seed(local_seed)

            generator = torch.Generator()
            generator.manual_seed(local_seed)

            loader = DataLoader(
                datasets[client_id],
                batch_size=trainer.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                generator=generator,
            )

            epoch_seconds = 0.0

            for _ in range(trainer.LOCAL_EPOCHS):
                stats = trainer.train_one_epoch(
                    local_model,
                    loader,
                    criterion,
                    optimizer,
                    device,
                )
                epoch_seconds += stats["train_seconds"]

            client_seconds.append(epoch_seconds)

            client_state = {
                k: v.detach().cpu().clone()
                for k, v in local_model.state_dict().items()
            }

            trainer.assert_state_finite(
                client_state,
                f"{tag} r{rnd} client{client_id}",
            )

            client_states.append(client_state)
            client_stats.append(stats)
            client_update_l2.append(
                trainer.state_l2_distance(
                    client_state, global_state
                )
            )
            participated.append(client_id)

        loop_wall_seconds = (
            time.perf_counter() - loop_wall_start
        )

        assert sorted(participated) == list(
            range(trainer.NUM_CLIENTS)
        )

        round_train_seconds = float(sum(client_seconds))
        round_orchestration_seconds = (
            loop_wall_seconds - round_train_seconds
        )

        trainer.sync_device(device)
        agg_start = time.perf_counter()

        # EXACT existing sample-weighted FedAvg implementation.
        agg_state = trainer.aggregate_sample_weighted(
            client_states, sizes
        )
        global_model.load_state_dict(agg_state)

        trainer.sync_device(device)
        aggregation_seconds = (
            time.perf_counter() - agg_start
        )

        trainer.assert_state_finite(
            agg_state, f"{tag} r{rnd} aggregated"
        )

        fl_round_seconds = (
            round_train_seconds + aggregation_seconds
        )

        trainer.sync_device(device)
        val_start = time.perf_counter()

        val = trainer.evaluate(
            global_model,
            val_loader,
            criterion,
            device,
        )

        trainer.sync_device(device)
        validation_seconds = (
            time.perf_counter() - val_start
        )

        round_total_seconds = (
            fl_round_seconds + validation_seconds
        )

        record = {
            "round": rnd,
            "val_loss": val["val_loss"],
            "accuracy": val["accuracy"],
            "balanced_accuracy": val["balanced_accuracy"],
            "macro_precision": val["macro_precision"],
            "macro_recall": val["macro_recall"],
            "macro_f1": val["macro_f1"],
            "weighted_f1": val["weighted_f1"],
            "macro_pr_auc": val["macro_pr_auc"],
            "worst_class_f1": val["worst_class_f1"],
            "worst_class_recall": val["worst_class_recall"],
        }

        for c in range(trainer.NUM_CLASSES):
            record[f"precision_c{c}"] = float(
                val["per_precision"][c]
            )
            record[f"recall_c{c}"] = float(
                val["per_recall"][c]
            )
            record[f"f1_c{c}"] = float(
                val["per_f1"][c]
            )
            record[f"support_c{c}"] = int(
                val["support"][c]
            )
            record[f"predicted_count_c{c}"] = int(
                val["predicted_count"][c]
            )

        for client_id in range(trainer.NUM_CLIENTS):
            record[f"train_loss_client_{client_id}"] = (
                client_stats[client_id]["weighted_train_loss"]
            )
            record[f"train_accuracy_client_{client_id}"] = (
                client_stats[client_id]["online_train_accuracy"]
            )
            record[f"train_samples_client_{client_id}"] = (
                client_stats[client_id]["n_samples"]
            )
            record[f"train_batches_client_{client_id}"] = (
                client_stats[client_id]["n_batches"]
            )
            record[f"update_l2_client_{client_id}"] = (
                client_update_l2[client_id]
            )
            record[f"train_seconds_client_{client_id}"] = (
                client_seconds[client_id]
            )

        record["round_train_seconds"] = round_train_seconds
        record["aggregation_seconds"] = aggregation_seconds
        record["fl_round_seconds"] = fl_round_seconds
        record["validation_seconds"] = validation_seconds
        record["round_total_seconds"] = round_total_seconds
        record["round_orchestration_seconds"] = (
            round_orchestration_seconds
        )

        history.append(record)

        if val["macro_f1"] > best["macro_f1"]:
            best = {
                "round": rnd,
                "macro_f1": float(val["macro_f1"]),
            }
            torch.save(
                global_model.state_dict(), out_best
            )

        print(
            f"[{tag}] round={rnd:02d} "
            f"val_macro_f1={val['macro_f1']:.4f} "
            f"bal_acc={val['balanced_accuracy']:.4f} "
            f"acc={val['accuracy']:.4f} "
            f"worst_f1={val['worst_class_f1']:.4f} "
            f"fl_s={fl_round_seconds:.1f} "
            f"val_s={validation_seconds:.1f}",
            flush=True,
        )

        pd.DataFrame(history).to_csv(
            out_history, index=False
        )

    extended_df = pd.DataFrame(history)
    extended_df.to_csv(out_history, index=False)

    # Save exact final global model after end_round.
    torch.save(
        global_model.state_dict(), out_final
    )

    argmax_round = int(
        extended_df.loc[
            extended_df["macro_f1"].idxmax(),
            "round",
        ]
    )

    if best["round"] != argmax_round:
        raise RuntimeError(
            f"{condition}: best round {best['round']} "
            f"!= history argmax {argmax_round}"
        )

    best_state = torch.load(
        out_best, map_location=device
    )
    trainer.assert_state_finite(
        best_state, f"{tag} extended best"
    )

    local_model.load_state_dict(best_state)

    best_reloaded_f1 = float(
        trainer.evaluate(
            local_model,
            val_loader,
            criterion,
            device,
        )["macro_f1"]
    )

    if abs(
        best_reloaded_f1 - best["macro_f1"]
    ) >= trainer.RELOAD_F1_TOL:
        raise RuntimeError(
            f"{condition}: extended best reload F1 "
            f"{best_reloaded_f1} != {best['macro_f1']}"
        )

    extension_rows = extended_df[
        extended_df["round"] > last_round
    ]

    out_cfg = dict(config)
    out_cfg.update({
        "max_rounds": args.end_round,
        "source_max_rounds": last_round,
        "resume_from_round": last_round,
        "resume_checkpoint_path": str(src_final),
        "resume_checkpoint_sha256": file_sha256(src_final),
        "source_best_checkpoint_path": str(src_best),
        "source_best_round": int(config["best_round"]),
        "source_best_val_macro_f1":
            float(config["best_val_macro_f1"]),
        "best_round": best["round"],
        "best_val_macro_f1": best["macro_f1"],
        "best_checkpoint_reloaded_val_macro_f1":
            best_reloaded_f1,
        "best_checkpoint_path": str(out_best),
        "final_checkpoint_path": str(out_final),
        "history_path": str(out_history),
        "continuation_round_start": last_round + 1,
        "continuation_round_end": args.end_round,
        "continuation_round_count":
            args.end_round - last_round,
        "continuation_type":
            "resume_from_final_global_model",
        "trainer_sha256": trainer_hash,
        "extension_script_sha256":
            file_sha256(Path(__file__).resolve()),
    })

    out_config.write_text(
        json.dumps(out_cfg, indent=2)
    )

    print(
        f"DONE EXTENDED {condition}: "
        f"best_round={best['round']} "
        f"best_val_macro_f1={best['macro_f1']:.6f} "
        f"extension_rounds={len(extension_rows)}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--partition-seed", type=int, default=42
    )
    p.add_argument(
        "--train-seed", type=int, default=42
    )
    p.add_argument(
        "--end-round", type=int, default=60
    )
    p.add_argument(
        "--single", choices=VALID_CONDITIONS
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        choices=VALID_CONDITIONS,
        default=VALID_CONDITIONS,
    )

    p.add_argument(
        "--partition-root",
        type=Path,
        default=ROOT
        / "fedartml_clean"
        / "partitions"
        / "k_5",
    )

    p.add_argument(
        "--source-format",
        choices=["primary", "extended"],
        default="primary",
    )

    p.add_argument(
        "--source-results-root",
        type=Path,
        default=ROOT / "fedartml_clean" / "results",
    )

    p.add_argument(
        "--source-models-root",
        type=Path,
        default=ROOT / "fedartml_clean" / "models",
    )

    p.add_argument(
        "--primary-results-root",
        type=Path,
        default=ROOT / "fedartml_clean" / "results",
    )

    p.add_argument(
        "--extension-root",
        type=Path,
        default=None,
    )

    args = p.parse_args()

    if args.extension_root is None:
        args.extension_root = (
            ROOT
            / "fedartml_clean"
            / "convergence_extension"
            / f"seed_{args.partition_seed}"
        )

    if args.single:
        run_one(args)
        return

    for condition in args.conditions:
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--partition-seed",
            str(args.partition_seed),
            "--train-seed",
            str(args.train_seed),
            "--end-round",
            str(args.end_round),
            "--partition-root",
            str(args.partition_root),
            "--source-format",
            args.source_format,
            "--source-results-root",
            str(args.source_results_root),
            "--source-models-root",
            str(args.source_models_root),
            "--primary-results-root",
            str(args.primary_results_root),
            "--extension-root",
            str(args.extension_root),
            "--single",
            condition,
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        returncode = proc.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)

    print("D1 CONVERGENCE EXTENSION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
