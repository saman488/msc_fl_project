from pathlib import Path
import importlib.util
import json
import argparse

import numpy as np
import pandas as pd

from fedartml.function_base import (
    hellinger_distance as fedartml_hellinger,
    jensen_shannon_distance as fedartml_js,
)


ROOT = Path(__file__).resolve().parents[1]

Y_TRAIN = ROOT / "data" / "processed_37f" / "y_train.npy"
PART_ROOT_BASE = ROOT / "fedartml_clean" / "partitions" / "k_5"
OUT_ROOT_BASE = ROOT / "fedartml_clean" / "diagnostics"

CONDITIONS = ["iid", "hd_0p25", "hd_0p5", "hd_0p75", "hd_0p9"]
K = 5

# Reuse the existing project's heterogeneity metric implementation.
METRICS_SOURCE = ROOT / "28_build_heterogeneity_summary.py"
spec = importlib.util.spec_from_file_location(
    "existing_heterogeneity_metrics", METRICS_SOURCE
)
existing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(existing)


def cosine_similarity_to_global(
    client_dist: np.ndarray,
    global_dist: np.ndarray,
) -> float:
    denom = float(
        np.linalg.norm(client_dist) * np.linalg.norm(global_dist)
    )
    if denom == 0.0:
        raise RuntimeError("Cosine similarity undefined for zero vector")
    return float(np.dot(client_dist, global_dist) / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-seed", type=int, required=True)
    args = parser.parse_args()

    partition_seed = args.partition_seed
    if partition_seed not in (42, 43):
        raise ValueError("partition seed must be 42 or 43")

    partition_root = PART_ROOT_BASE / f"seed_{partition_seed}"
    out_root = OUT_ROOT_BASE / f"seed_{partition_seed}"

    y_train = np.load(Y_TRAIN)

    if y_train.ndim != 1:
        raise RuntimeError(f"Expected 1-D y_train, got {y_train.shape}")

    classes = np.unique(y_train)
    expected = np.arange(len(classes))
    if not np.array_equal(classes, expected):
        raise RuntimeError(
            f"Unexpected labels: {classes.tolist()}"
        )

    num_classes = len(classes)
    global_counts = np.bincount(y_train, minlength=num_classes)
    global_dist = global_counts / global_counts.sum()

    summary_rows = []
    pairwise_rows = []
    client_global_rows = []

    for condition in CONDITIONS:
        part_dir = partition_root / condition
        manifest_path = part_dir / "partition_manifest.json"

        if not manifest_path.exists():
            raise RuntimeError(
                f"Missing manifest for {condition}: {manifest_path}"
            )

        manifest = json.loads(manifest_path.read_text())

        files = [
            part_dir / f"client_{k:02d}_indices.npy"
            for k in range(K)
        ]

        missing = [p for p in files if not p.exists()]
        if missing:
            raise RuntimeError(
                f"{condition}: missing client files:\n  "
                + "\n  ".join(str(p) for p in missing)
            )

        client_indices = [np.load(p) for p in files]

        if any(len(idx) == 0 for idx in client_indices):
            raise RuntimeError(f"{condition}: empty client")

        all_indices = np.concatenate(client_indices)

        if len(all_indices) != len(y_train):
            raise RuntimeError(f"{condition}: coverage count failed")

        if len(np.unique(all_indices)) != len(y_train):
            raise RuntimeError(f"{condition}: duplicate indices")

        if not np.array_equal(
            np.sort(all_indices), np.arange(len(y_train))
        ):
            raise RuntimeError(
                f"{condition}: indices do not cover exactly 0..N-1"
            )

        counts = np.stack([
            np.bincount(y_train[idx], minlength=num_classes)
            for idx in client_indices
        ])

        if not np.array_equal(
            counts.sum(axis=0), global_counts
        ):
            raise RuntimeError(
                f"{condition}: client class totals mismatch"
            )

        # Existing project code:
        # pairwise HD/JSD/TV and client->global HD/JSD/TV.
        result = existing.condition_metrics(
            counts, global_counts
        )

        sizes = counts.sum(axis=1)
        props = counts / sizes[:, None]

        # Recompute FedArtML metrics from the exact saved partition.
        fedartml_hd = float(fedartml_hellinger(props))
        fedartml_js_value = float(fedartml_js(props))

        recorded_hd = manifest.get("fedartml_hellinger_distance")
        if recorded_hd is not None and not np.isclose(
            fedartml_hd, float(recorded_hd), rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(
                f"{condition}: recomputed FedArtML HD {fedartml_hd:.12f} "
                f"!= manifest {float(recorded_hd):.12f}"
            )

        # New metric ONLY: client -> global cosine similarity.
        cosine_values = []

        for row, client_dist in zip(
            result["to_global"], props
        ):
            cosine = cosine_similarity_to_global(
                client_dist, global_dist
            )
            row["cosine_similarity_to_global"] = cosine
            cosine_values.append(cosine)

        # Give TV explicit summary names as well.
        pairwise_tv = np.asarray([
            row["total_variation"]
            for row in result["pairwise"]
        ])

        global_tv = np.asarray([
            row["total_variation_to_global"]
            for row in result["to_global"]
        ])

        summary = {
            "partition_seed": partition_seed,
            "condition": condition,
            "target_hd": manifest.get("target_hd"),
            "alpha": manifest.get("alpha"),
            "fedartml_hellinger_distance": fedartml_hd,
            "fedartml_jensen_shannon_distance": fedartml_js_value,
            **result["summary"],
            "tv_pairwise_mean": float(pairwise_tv.mean()),
            "tv_pairwise_rms":
                float(np.sqrt(np.mean(pairwise_tv ** 2))),
            "tv_pairwise_min": float(pairwise_tv.min()),
            "tv_pairwise_max": float(pairwise_tv.max()),
            "mean_client_to_global_tv": float(global_tv.mean()),
            "max_client_to_global_tv": float(global_tv.max()),
            "mean_client_to_global_cosine_similarity":
                float(np.mean(cosine_values)),
            "min_client_to_global_cosine_similarity":
                float(np.min(cosine_values)),
            "max_client_to_global_cosine_similarity":
                float(np.max(cosine_values)),
        }

        summary_rows.append(summary)

        for row in result["pairwise"]:
            pairwise_rows.append({
                "partition_seed": partition_seed,
                "condition": condition,
                **row,
            })

        for row in result["to_global"]:
            client_global_rows.append({
                "partition_seed": partition_seed,
                "condition": condition,
                **row,
            })

    out_root.mkdir(parents=True, exist_ok=False)

    pd.DataFrame(summary_rows).to_csv(
        out_root / "partition_summary.csv",
        index=False,
    )

    pd.DataFrame(pairwise_rows).to_csv(
        out_root / "pairwise_distances.csv",
        index=False,
    )

    pd.DataFrame(client_global_rows).to_csv(
        out_root / "client_to_global_distances.csv",
        index=False,
    )

    print(f"WROTE: {out_root}")
    print("conditions:", CONDITIONS)


if __name__ == "__main__":
    main()
