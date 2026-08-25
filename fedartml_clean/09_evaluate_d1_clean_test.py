from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed_37f"
FEATURE_MANIFEST = PROCESSED_DIR / "feature_manifest.json"
LABEL_MAP = ROOT / "configs" / "label_mapping.json"
PART_ROOT = ROOT / "fedartml_clean" / "partitions" / "k_5"
FINAL_ROOT = ROOT / "fedartml_clean" / "convergence_extension_100"
OUT_DIR = ROOT / "fedartml_clean" / "final_test_100r"
TRAINER = ROOT / "44_train_fedavg_hd_selected.py"

CONDITIONS = ["iid", "hd_0p25", "hd_0p5", "hd_0p75", "hd_0p9"]

TRAIN_SEED = 42
ROUNDS = 100
INPUT_DIM = 37
NUM_CLASSES = 10
EXPECTED_TEST_ROWS = 358542
BATCH_SIZE = 4096

VALUE_TOL = 1e-10
RELOAD_TOL = 1e-6


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_trainer():
    spec = importlib.util.spec_from_file_location("d1_clean_eval_trainer", TRAINER)
    trainer = importlib.util.module_from_spec(spec)
    sys.modules["d1_clean_eval_trainer"] = trainer
    spec.loader.exec_module(trainer)
    return trainer


def load_class_names():
    mapping = json.loads(LABEL_MAP.read_text())
    id_to_name = {int(v): name for name, v in mapping.items()}

    require(
        sorted(id_to_name) == list(range(NUM_CLASSES)),
        "label mapping is not exactly classes 0..9",
    )

    return [id_to_name[c] for c in range(NUM_CLASSES)]


def select_models(seeds):
    selected = []
    trainer_hash = file_sha256(TRAINER)

    for seed in seeds:
        for condition in CONDITIONS:
            tag = f"fedavg37f_k5_seed{seed}_{condition}"

            result_dir = (
                FINAL_ROOT / f"seed_{seed}" / "results" / condition
            )
            model_dir = (
                FINAL_ROOT / f"seed_{seed}" / "models" / condition
            )
            part_dir = PART_ROOT / f"seed_{seed}" / condition

            history_path = result_dir / f"history_extended_{tag}.csv"
            config_path = result_dir / f"config_extended_{tag}.json"
            checkpoint_path = model_dir / f"best_extended_{tag}.pt"
            manifest_path = part_dir / "partition_manifest.json"

            for path in (
                history_path,
                config_path,
                checkpoint_path,
                manifest_path,
            ):
                require(path.exists(), f"missing required file: {path}")

            # Explicitly forbid final checkpoint selection.
            require(
                checkpoint_path.name.startswith("best_extended_"),
                f"{tag}: evaluator is not using a best checkpoint",
            )

            history = pd.read_csv(history_path)
            config = json.loads(config_path.read_text())
            manifest = json.loads(manifest_path.read_text())

            require(
                len(history) == ROUNDS,
                f"{tag}: expected {ROUNDS} history rows, found {len(history)}",
            )
            require(
                history["round"].astype(int).tolist()
                == list(range(1, ROUNDS + 1)),
                f"{tag}: history is not exactly rounds 1..{ROUNDS}",
            )

            best_i = history["macro_f1"].idxmax()
            best_round = int(history.loc[best_i, "round"])
            best_f1 = float(history.loc[best_i, "macro_f1"])

            # Bind to the intended experiment.
            require(
                config.get("partition_seed") == seed,
                f"{tag}: wrong partition seed",
            )
            require(
                config.get("training_seed") == TRAIN_SEED,
                f"{tag}: training seed is not {TRAIN_SEED}",
            )
            require(
                config.get("condition") == condition,
                f"{tag}: wrong condition",
            )
            require(
                config.get("max_rounds") == ROUNDS,
                f"{tag}: max_rounds is not {ROUNDS}",
            )
            require(
                config.get("selection_metric") == "val_macro_f1",
                f"{tag}: wrong model-selection metric",
            )
            require(
                int(config.get("best_round")) == best_round,
                f"{tag}: config/history best-round mismatch",
            )
            require(
                abs(float(config["best_val_macro_f1"]) - best_f1)
                <= VALUE_TOL,
                f"{tag}: config/history best-F1 mismatch",
            )

            # Bind config to the exact CLEAN partition.
            require(
                "partition_path" in config,
                f"{tag}: config missing partition_path",
            )
            require(
                Path(config["partition_path"]).resolve()
                == part_dir.resolve(),
                f"{tag}: config points to a different partition",
            )

            # Bind config to the exact best checkpoint evaluated here.
            require(
                "best_checkpoint_path" in config,
                f"{tag}: config missing best_checkpoint_path",
            )
            require(
                Path(config["best_checkpoint_path"]).resolve()
                == checkpoint_path.resolve(),
                f"{tag}: config points to a different best checkpoint",
            )

            require(
                "best_checkpoint_reloaded_val_macro_f1" in config,
                f"{tag}: config missing reloaded best validation score",
            )
            require(
                abs(
                    float(config["best_checkpoint_reloaded_val_macro_f1"])
                    - best_f1
                )
                <= RELOAD_TOL,
                f"{tag}: reloaded best checkpoint validation F1 mismatch",
            )

            # Bind result to the unchanged verified trainer.
            recorded_trainer_hash = config.get(
                "trainer_sha256",
                config.get("script_sha256"),
            )
            require(
                recorded_trainer_hash == trainer_hash,
                f"{tag}: trainer hash mismatch",
            )

            # Bind result to clean partition manifest.
            manifest_seed = manifest.get(
                "partition_seed",
                manifest.get("fedartml_random_state"),
            )
            require(
                manifest_seed is not None and int(manifest_seed) == seed,
                f"{tag}: partition manifest seed mismatch",
            )

            achieved_hd = manifest.get("fedartml_hellinger_distance")
            require(
                achieved_hd is not None,
                f"{tag}: clean partition manifest missing achieved HD",
            )

            selected.append({
                "partition_seed": seed,
                "training_seed": TRAIN_SEED,
                "condition": condition,
                "target_hd": manifest.get("target_hd"),
                "achieved_hd": float(achieved_hd),
                "alpha": manifest.get("alpha"),
                "best_round": best_round,
                "val_macro_f1": best_f1,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "partition_manifest": str(manifest_path),
                "partition_manifest_sha256": file_sha256(manifest_path),
            })

    return selected


def compute_metrics(y_true, y_pred, y_prob, class_names):
    labels = list(range(NUM_CLASSES))
    support = np.bincount(y_true, minlength=NUM_CLASSES)

    per_p, per_r, per_f, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    predicted_count = np.bincount(y_pred, minlength=NUM_CLASSES)

    per_ap = np.full(NUM_CLASSES, np.nan)

    for c in labels:
        if support[c] > 0:
            per_ap[c] = average_precision_score(
                (y_true == c).astype(int),
                y_prob[:, c],
            )

    supported = [c for c in labels if support[c] > 0]

    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_average_precision": float(np.nanmean(per_ap)),
        "worst_class_f1": float(min(per_f[c] for c in supported)),
        "worst_class_recall": float(min(per_r[c] for c in supported)),
        "n_samples": int(len(y_true)),
    }

    per_class = [{
        "class_id": c,
        "class_name": class_names[c],
        "precision": float(per_p[c]),
        "recall": float(per_r[c]),
        "f1": float(per_f[c]),
        "average_precision": float(per_ap[c]),
        "support": int(support[c]),
        "predicted_count": int(predicted_count[c]),
    } for c in labels]

    return {
        "overall": overall,
        "per_class": per_class,
        "confusion": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ),
    }


@torch.inference_mode()
def evaluate_model(
    trainer,
    checkpoint_path,
    x_test,
    y_test,
    device,
    class_names,
):
    model = trainer.MLPMultiClassClassifier(
        INPUT_DIM,
        NUM_CLASSES,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds = []
    probs = []

    for start in range(0, len(y_test), BATCH_SIZE):
        block = np.asarray(
            x_test[start:start + BATCH_SIZE],
            dtype=np.float32,
        )

        require(
            np.isfinite(block).all(),
            f"non-finite test features near row {start}",
        )

        logits = model(
            torch.from_numpy(block).to(device)
        )

        probs.append(
            torch.softmax(logits, dim=1).cpu().numpy()
        )
        preds.append(
            torch.argmax(logits, dim=1).cpu().numpy()
        )

    y_pred = np.concatenate(preds).astype(int)
    y_prob = np.concatenate(probs)

    require(
        len(y_pred) == len(y_test),
        "prediction/test row mismatch",
    )

    return compute_metrics(
        y_test,
        y_pred,
        y_prob,
        class_names,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43],
    )

    parser.add_argument(
        "--execute-test",
        action="store_true",
        help="Load the untouched test set and run final evaluation.",
    )

    args = parser.parse_args()

    require(
        set(args.seeds).issubset({42, 43}),
        "only partition seeds 42 and 43 are allowed",
    )

    # MODEL SELECTION/PREFLIGHT OCCURS BEFORE ANY TEST ACCESS.
    selected = select_models(args.seeds)

    expected_models = len(args.seeds) * len(CONDITIONS)

    require(
        len(selected) == expected_models,
        f"expected {expected_models} locked models, found {len(selected)}",
    )

    print(
        f"PREFLIGHT PASS: {len(selected)} locked models "
        f"for partition seed(s) {args.seeds}"
    )
    print("TEST DATA NOT ACCESSED DURING PREFLIGHT")

    if not args.execute_test:
        return

    # Prevent accidental early testing on seed 42 alone.
    require(
        sorted(args.seeds) == [42, 43],
        "FINAL TEST REFUSED: both partition seeds 42 and 43 must be locked",
    )
    require(
        len(selected) == 10,
        "FINAL TEST REFUSED: expected exactly 10 locked models",
    )

    require(
        not OUT_DIR.exists(),
        f"refusing to overwrite existing final-test output: {OUT_DIR}",
    )

    # Verify corrected 37-feature test lineage.
    feature_manifest = json.loads(FEATURE_MANIFEST.read_text())

    require(
        int(feature_manifest["corrected_feature_count"]) == INPUT_DIM,
        "feature manifest is not the corrected 37-feature branch",
    )
    require(
        int(feature_manifest["row_counts"]["test"]) == EXPECTED_TEST_ROWS,
        "feature manifest test-row count mismatch",
    )

    # TEST ACCESS BEGINS HERE AND ONLY HERE.
    x_test = np.load(
        PROCESSED_DIR / "X_test.npy",
        mmap_mode="r",
    )
    y_test = np.load(
        PROCESSED_DIR / "y_test.npy"
    )

    require(
        x_test.shape == (EXPECTED_TEST_ROWS, INPUT_DIM),
        f"unexpected X_test shape: {x_test.shape}",
    )
    require(
        x_test.dtype == np.float32,
        f"unexpected X_test dtype: {x_test.dtype}",
    )
    require(
        y_test.shape == (EXPECTED_TEST_ROWS,),
        f"unexpected y_test shape: {y_test.shape}",
    )
    require(
        np.issubdtype(y_test.dtype, np.integer),
        "y_test labels are not integers",
    )
    require(
        np.array_equal(
            np.unique(y_test),
            np.arange(NUM_CLASSES),
        ),
        "y_test does not contain exactly classes 0..9",
    )

    trainer = load_trainer()
    device = trainer.get_device()
    class_names = load_class_names()

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    confusion_dir = OUT_DIR / "confusion_matrices"
    confusion_dir.mkdir()

    overall_rows = []
    per_class_rows = []

    for model_info in selected:
        result = evaluate_model(
            trainer,
            Path(model_info["checkpoint_path"]),
            x_test,
            y_test,
            device,
            class_names,
        )

        identity = {
            k: model_info[k]
            for k in (
                "partition_seed",
                "training_seed",
                "condition",
                "target_hd",
                "achieved_hd",
                "alpha",
                "best_round",
                "val_macro_f1",
                "checkpoint_sha256",
            )
        }

        overall_rows.append({
            **identity,
            **result["overall"],
        })

        for row in result["per_class"]:
            per_class_rows.append({
                **identity,
                **row,
            })

        stem = (
            f"seed{model_info['partition_seed']}_"
            f"{model_info['condition']}"
        )

        raw = result["confusion"]

        pd.DataFrame(
            raw,
            index=pd.Index(class_names, name="true"),
            columns=class_names,
        ).to_csv(
            confusion_dir / f"confusion_raw_{stem}.csv"
        )

        totals = raw.sum(axis=1, keepdims=True)

        normalised = np.divide(
            raw,
            totals,
            out=np.zeros(raw.shape, dtype=float),
            where=totals > 0,
        )

        pd.DataFrame(
            normalised,
            index=pd.Index(class_names, name="true"),
            columns=class_names,
        ).to_csv(
            confusion_dir
            / f"confusion_normalised_{stem}.csv"
        )

    pd.DataFrame(selected).to_csv(
        OUT_DIR / "selected_checkpoints.csv",
        index=False,
    )

    pd.DataFrame(overall_rows).to_csv(
        OUT_DIR / "test_results_overall.csv",
        index=False,
    )

    pd.DataFrame(per_class_rows).to_csv(
        OUT_DIR / "test_results_per_class.csv",
        index=False,
    )

    metadata = {
        "dataset": "D1",
        "representation": "37f",
        "evaluation": "final_held_out_test",
        "partition_seeds": [42, 43],
        "training_seed": TRAIN_SEED,
        "communication_round_budget": ROUNDS,
        "model_selection":
            "best validation macro-F1 within fixed 100-round budget",
        "n_models": 10,
        "processed_dir": str(PROCESSED_DIR),
        "x_test_sha256": file_sha256(
            PROCESSED_DIR / "X_test.npy"
        ),
        "y_test_sha256": file_sha256(
            PROCESSED_DIR / "y_test.npy"
        ),
        "test_rows": EXPECTED_TEST_ROWS,
        "test_access_rule":
            "test loaded only after all 10 model selections passed preflight",
    }

    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print(f"WROTE FINAL D1 TEST RESULTS: {OUT_DIR}")


if __name__ == "__main__":
    main()
