from pathlib import Path
import hashlib
import importlib.util
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER_SOURCE = PROJECT_ROOT / "44_train_fedavg_hd_selected.py"

EXPECTED_TRAINER_SHA256 = (
    "e27bb99fdc4168a6b5ceeafbe03e28634c56f3dc569b40c1692cca990c32f2c9"
)

def main() -> None:
    actual_hash = hashlib.sha256(TRAINER_SOURCE.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_TRAINER_SHA256:
        raise RuntimeError(
            f"Trainer source changed: {actual_hash} != {EXPECTED_TRAINER_SHA256}"
        )

    spec = importlib.util.spec_from_file_location(
        "clean_fedavg_trainer", TRAINER_SOURCE
    )
    trainer = importlib.util.module_from_spec(spec)
    sys.modules["clean_fedavg_trainer"] = trainer
    spec.loader.exec_module(trainer)

    trainer.PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_37f"
    trainer.LABEL_MAP = PROJECT_ROOT / "configs" / "label_mapping.json"

    trainer.PART_ROOT = PROJECT_ROOT / "fedartml_clean" / "partitions" / "k_5"
    trainer.RESULTS_DIR = PROJECT_ROOT / "fedartml_clean" / "results" / "hd_0p25"
    trainer.MODELS_DIR = PROJECT_ROOT / "fedartml_clean" / "models" / "hd_0p25"

    trainer.INIT_PATH = trainer.MODELS_DIR / "initial_global_model.pt"
    trainer.CLASS_WEIGHTS_PATH = trainer.RESULTS_DIR / "class_weights.npy"

    trainer.PARTITION_SEEDS = [42]
    trainer.CONDITIONS = ["hd_0p25"]

    print("CLEAN FEDAVG RUN")
    print(f"trainer_sha256: {actual_hash}")
    print(f"partition_root: {trainer.PART_ROOT}")
    print(f"results_dir:    {trainer.RESULTS_DIR}")
    print(f"models_dir:     {trainer.MODELS_DIR}")
    print()

    trainer.main()

if __name__ == "__main__":
    main()
