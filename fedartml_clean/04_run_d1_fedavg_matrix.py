from pathlib import Path
import argparse
import hashlib
import importlib.util
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "44_train_fedavg_hd_selected.py"

EXPECTED_SHA256 = "e27bb99fdc4168a6b5ceeafbe03e28634c56f3dc569b40c1692cca990c32f2c9"

CONDITIONS = ["iid", "hd_0p25", "hd_0p5", "hd_0p75", "hd_0p9"]


def outputs(results, models, seed, condition):
    tag = f"fedavg37f_k5_seed{seed}_{condition}"
    return [
        results / "class_weights.npy",
        results / f"config_{tag}.json",
        results / "final_summary.csv",
        results / f"history_{tag}.csv",
        models / "initial_global_model.pt",
        models / f"best_{tag}.pt",
        models / f"final_{tag}.pt",
    ]


def run_one(args):
    actual = hashlib.sha256(TRAINER.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Trainer hash mismatch: {actual}")

    part = args.partition_root / f"seed_{args.partition_seed}" / args.single
    required = [part / f"client_{k:02d}_indices.npy" for k in range(5)]
    required.append(part / "partition_manifest.json")

    missing = [p for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing partition files:\n" + "\n".join(map(str, missing)))

    results = args.results_root / args.single
    models = args.models_root / args.single
    expected = outputs(results, models, args.partition_seed, args.single)
    present = [p for p in expected if p.exists()]

    if len(present) == len(expected):
        print(f"SKIP COMPLETE: {args.single}", flush=True)
        return

    if present:
        raise RuntimeError(
            f"Partial existing outputs for {args.single}; refusing to overwrite:\n"
            + "\n".join(map(str, present))
        )

    spec = importlib.util.spec_from_file_location("d1_clean_trainer", TRAINER)
    trainer = importlib.util.module_from_spec(spec)
    sys.modules["d1_clean_trainer"] = trainer
    spec.loader.exec_module(trainer)

    trainer.PROCESSED_DIR = ROOT / "data" / "processed_37f"
    trainer.LABEL_MAP = ROOT / "configs" / "label_mapping.json"
    trainer.PART_ROOT = args.partition_root
    trainer.RESULTS_DIR = results
    trainer.MODELS_DIR = models
    trainer.INIT_PATH = models / "initial_global_model.pt"
    trainer.CLASS_WEIGHTS_PATH = results / "class_weights.npy"

    trainer.PARTITION_SEEDS = [args.partition_seed]
    trainer.CONDITIONS = [args.single]
    trainer.TRAIN_SEED = args.train_seed

    print(f"START {args.single} partition_seed={args.partition_seed} train_seed={args.train_seed}", flush=True)
    trainer.main()

    missing = [p for p in expected if not p.exists()]
    if missing:
        raise RuntimeError("Training returned but outputs missing:\n" + "\n".join(map(str, missing)))

    print(f"DONE: {args.single}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partition-seed", type=int, default=42)
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--single", choices=CONDITIONS)
    p.add_argument("--partition-root", type=Path,
                   default=ROOT / "fedartml_clean" / "partitions" / "k_5")
    p.add_argument("--results-root", type=Path,
                   default=ROOT / "fedartml_clean" / "results")
    p.add_argument("--models-root", type=Path,
                   default=ROOT / "fedartml_clean" / "models")
    args = p.parse_args()

    # Preserve existing seed-42 locations.
    # Any later partition seed gets isolated outputs automatically.
    if args.partition_seed != 42:
        default_results = ROOT / "fedartml_clean" / "results"
        default_models = ROOT / "fedartml_clean" / "models"

        if args.results_root == default_results:
            args.results_root = default_results / f"seed_{args.partition_seed}"
        if args.models_root == default_models:
            args.models_root = default_models / f"seed_{args.partition_seed}"

    if args.single:
        run_one(args)
        return

    for condition in CONDITIONS:
        subprocess.run([
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--partition-seed", str(args.partition_seed),
            "--train-seed", str(args.train_seed),
            "--partition-root", str(args.partition_root),
            "--results-root", str(args.results_root),
            "--models-root", str(args.models_root),
            "--single", condition,
        ], check=True)

    print("D1 MATRIX COMPLETE", flush=True)


if __name__ == "__main__":
    main()
