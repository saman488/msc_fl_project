## Experiment 1 — Initial Federated Learning Experiment

This is the **fixed-Dirichlet experiment with FedAvg, FedProx and SCAFFOLD** on both datasets.

### Dataset 1 — NF-UNSW-NB15-v2

The Dataset 1 implementation is the root-level sequence from `02_full_audit.py` through `39_evaluate_final_test.py`.

The client partitions are created by `25_create_final_partitions.py` using fixed Dirichlet α values and fixed partition seeds. The later files train and evaluate FedAvg, FedProx and SCAFFOLD using these partitions.

### Dataset 2 — NF-CSE-CIC-IDS2018-v2

The Dataset 2 counterpart is the root-level sequence from `d2_01_preprocess_nf_cse_cic_ids2018_v2.py` through `d2_07a_verify_final_test.py`.

These files carry out the corresponding Dataset 2 preprocessing, partitioning, centralised training, FedAvg, FedProx, SCAFFOLD and final test evaluation.


- **`02_full_audit.py` — Raw-data audit:** checks the NF-UNSW-NB15-v2 CSV for row and class counts, missing/infinite values, and feature columns. It does not create processed experiment data.
- **`03_preprocess.py` — Preprocessing:** creates the multiclass D1 data (Benign + 9 attack classes), uses a stratified 70/15/15 train/validation/test split, fits `StandardScaler` on the training set only, and saves the processed arrays under `data/processed/`.

### `25_create_final_partitions.py` — Experiment 1 partitioning

**Main settings**
- `K=5` clients
- α = `0.1`, `0.5`, `1.0`
- partition seeds `42`, `43`, `44`
- output: `data/fl_clients/final_partitions/k_5/`

**How partitions are created**
- `partition_iid()` creates the IID client split.
- `partition_noniid()` creates the class-wise Dirichlet splits.
- `dirichlet_rng()` controls the seeded Dirichlet draws.

**Connection to the dissertation experiment**

These are the Experiment 1 partitions. The dissertation experiment under `fedartml_clean/` uses FedArtML for its non-IID partitions, while `fedartml_clean/00_prepare_iid_partitions.py` copies the IID partitions created here into the `fedartml_clean/` structure.
