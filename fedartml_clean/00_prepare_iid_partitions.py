from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]

SETS = [
    {
        "name": "Dataset 1",
        "src": ROOT / "data/fl_clients/final_partitions/k_5",
        "dst": ROOT / "fedartml_clean/partitions/k_5",
        "patch_d1": True,
    },
    {
        "name": "Dataset 2",
        "src": ROOT / "data/nf_cse_cic_ids2018_v2/fl_clients/final_partitions/k_5",
        "dst": ROOT / "fedartml_clean/d2_partitions/k_5",
        "patch_d1": False,
    },
]

for dataset in SETS:
    for seed in (42, 43):
        src = dataset["src"] / f"seed_{seed}" / "iid"
        dst = dataset["dst"] / f"seed_{seed}" / "iid"

        if not src.exists():
            raise FileNotFoundError(f"Missing baseline IID partition: {src}")

        if dst.exists():
            print(f"EXISTS: {dst}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

        manifest_path = dst / "partition_manifest.json"
        manifest = json.loads(manifest_path.read_text())

        if dataset["patch_d1"]:
            hd = manifest.get("hd_pairwise_rms")
            if hd is None:
                raise RuntimeError(f"Missing hd_pairwise_rms in {manifest_path}")
            manifest["source"] = str(src.relative_to(ROOT))
            manifest["target_hd"] = 0.0
            manifest["fedartml_hellinger_distance"] = hd
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        print(f"PREPARED: {dataset['name']} seed {seed} -> {dst}")
