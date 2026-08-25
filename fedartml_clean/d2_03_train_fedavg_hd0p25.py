from pathlib import Path
import hashlib
import importlib.util
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER_SOURCE = PROJECT_ROOT / "d2_04_train_fedavg.py"

EXPECTED_TRAINER_SHA256 = (
    "7c9175da360432088c0e4ad33930eeb125af4a7624b88a83e05fa8fc22cab61e"
)


def main() -> None:
    actual_hash = hashlib.sha256(TRAINER_SOURCE.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_TRAINER_SHA256:
        raise RuntimeError(
            f"D2 trainer source changed: {actual_hash} != {EXPECTED_TRAINER_SHA256}"
        )

    spec = importlib.util.spec_from_file_location(
        "clean_d2_fedavg_trainer",
        TRAINER_SOURCE,
    )
    trainer = importlib.util.module_from_spec(spec)
    sys.modules["clean_d2_fedavg_trainer"] = trainer
    spec.loader.exec_module(trainer)

    trainer.PROCESSED_DIR = (
        PROJECT_ROOT / "data" / "nf_cse_cic_ids2018_v2" / "processed"
    )
    trainer.LABEL_MAP = (
        PROJECT_ROOT / "configs" / "nf_cse_cic_ids2018_v2" / "label_mapping.json"
    )

    trainer.PART_ROOT = (
        PROJECT_ROOT / "fedartml_clean" / "d2_partitions" / "k_5"
    )
    trainer.RESULTS_DIR = (
        PROJECT_ROOT / "fedartml_clean" / "d2_results" / "hd_0p25"
    )
    trainer.MODELS_DIR = (
        PROJECT_ROOT / "fedartml_clean" / "d2_models" / "hd_0p25"
    )

    trainer.INIT_PATH = trainer.MODELS_DIR / "initial_global_model.pt"
    trainer.CLASS_WEIGHTS_PATH = trainer.RESULTS_DIR / "class_weights.npy"

    trainer.PARTITION_SEEDS = [42]
    trainer.CONDITIONS = ["hd_0p25"]

    print("CLEAN D2 FEDAVG RUN")
    print(f"trainer_sha256: {actual_hash}")
    print(f"partition_root: {trainer.PART_ROOT}")
    print(f"results_dir:    {trainer.RESULTS_DIR}")
    print(f"models_dir:     {trainer.MODELS_DIR}")
    print()

    trainer.main()


if __name__ == "__main__":
    main()
