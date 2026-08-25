from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = {
    "iid": "IID",
    "hd_0p25": "HD≈0.25",
    "hd_0p5": "HD≈0.50",
    "hd_0p75": "HD≈0.75",
    "hd_0p9": "HD≈0.90",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partition-seed", type=int, choices=[42, 43], required=True)
    args = p.parse_args()

    result_root = (
        ROOT / "fedartml_clean" / "convergence_extension_100"
        / f"seed_{args.partition_seed}" / "results"
    )

    out_dir = ROOT / "fedartml_clean" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition, label in CONDITIONS.items():
        files = list((result_root / condition).glob("history_extended_*.csv"))

        if len(files) != 1:
            raise RuntimeError(
                f"{condition}: expected one 100-round history, found {len(files)}"
            )

        df = pd.read_csv(files[0])

        if df["round"].astype(int).tolist() != list(range(1, 101)):
            raise RuntimeError(f"{condition}: history is not exactly rounds 1..100")

        ax.plot(df["round"], df["macro_f1"], label=label)

    for rnd in (40, 60, 80):
        ax.axvline(rnd, linestyle="--", linewidth=0.8)

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title(
        f"D1 FedAvg convergence — partition seed {args.partition_seed}"
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out = out_dir / f"d1_seed{args.partition_seed}_fedavg_convergence.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)

    print("WROTE:", out)


if __name__ == "__main__":
    main()
