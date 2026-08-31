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
- **`30_build_dataset1_37feature_branch.py`:** continues Dataset 1 preprocessing by creating the corrected 37-feature train/validation branch from the already-standardised 41-feature arrays. It removes `L4_SRC_PORT`, `L4_DST_PORT`, `MIN_TTL`, and `MAX_TTL`, preserves the remaining scaled values and labels, and saves them under `data/processed_37f/`.

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


- **`26_validate_client_count_sensitivity.py`:** tests how changing the number of clients (`K=5, 10, 20`) changes the partition heterogeneity while keeping α = `0.1`, `0.5`, `1.0` and partition seeds `42`, `43`, `44`. It calculates Hellinger Distance, client sizes, and the number of classes missing from each client. **FedAvg, FedProx and SCAFFOLD are not run in this file; it is used only to examine how client count changes the partition structure.**
- **`27_final_fedavg_sgd_pilot.py`:** checks the selected FedAvg training configuration on two representative Experiment 1 partitions before the full training matrix is run: seed `42` IID and seed `44` with `alpha=0.1`. It uses sample-weighted FedAvg, SGD (`lr=0.1`), batch size `4096`, one local epoch and 40 rounds, with the same initial model state and training seed for both runs.
- **`28_build_heterogeneity_summary.py`:** uses the Python function `condition_metrics()` to calculate heterogeneity for the Experiment 1 **IID and non-IID client partitions**. `condition_metrics()` converts each client’s class counts into class proportions, then calculates **pairwise Hellinger Distance, pairwise Jensen–Shannon divergence, pairwise total variation / 0-1 EMD, HD-RMS, client-to-global Hellinger Distance, client-to-global JSD, client-to-global total variation / 0-1 EMD, client-size statistics, and missing-class counts**. The results are written under `results/partition_validation/heterogeneity_summary/`.
- **`29_train_final_fedavg.py`:** runs the full Dataset 1 FedAvg training matrix for Experiment 1 using the `K=5` partitions from `25_create_final_partitions.py`. It trains for 40 rounds with SGD (`lr=0.1`), batch size `4096`, one local epoch, full client participation and sample-weighted FedAvg aggregation, and saves the training histories and best/final model checkpoints.
- **`31_fedavg_37f_transfer_check.py`:** checks that the FedAvg configuration selected earlier still works on the corrected 37-feature Dataset 1 data created by `30_build_dataset1_37feature_branch.py`. It uses the same `K=5` client partitions and FedAvg settings, runs up to 40 rounds, and records the validation/checkpoint results before the full 37-feature FedAvg experiment is run.
- **`32_train_final_fedavg_37f.py`:** runs the full Dataset 1 FedAvg training experiment on the corrected 37-feature data. It uses the `K=5` client partitions from `25_create_final_partitions.py`, trains FedAvg for 40 rounds with SGD, and saves the training histories plus the best and final model checkpoints.

### Remaining Dataset 1 files

- **`34_train_final_fedprox_37f.py`:** trains FedProx on the corrected 37-feature Dataset 1 data using the Experiment 1 client partitions.
- **`35_train_final_scaffold_37f.py`:** trains SCAFFOLD on the corrected 37-feature Dataset 1 data using the same Experiment 1 client partitions.
- **`36_select_fedprox_mu.py`:** evaluates the FedProx proximal parameter `μ` and records the selected value used for the FedProx runs.
- **`37_train_central_mlp_37f.py`:** trains the centralised MLP on the corrected 37-feature Dataset 1 data to provide a non-federated reference.
- **`38_build_dataset1_37feature_test.py`:** creates the corrected 37-feature Dataset 1 test data using the same feature removal applied to the train and validation data.
- **`39_evaluate_final_test.py`:** evaluates the selected Experiment 1 models on the held-out Dataset 1 test set and writes the final test results.

### Dataset 2 files

- **`d2_01_preprocess_nf_cse_cic_ids2018_v2.py`:** prepares NF-CSE-CIC-IDS2018-v2 and creates the processed train, validation and test data.
- **`d2_02_create_final_partitions.py`:** creates the Dataset 2 IID and fixed-Dirichlet non-IID client partitions used in Experiment 1.
- **`d2_03_train_central_mlp.py`:** trains the centralised Dataset 2 MLP reference model.
- **`d2_04_train_fedavg.py`:** trains FedAvg on the Dataset 2 Experiment 1 client partitions.
- **`d2_04a_diagnose_round10_client3.py` / `d2_04b_trace_round10_client3_batches.py`:** investigate Dataset 2 FedAvg training behaviour at the specified round and client.
- **`d2_05_train_fedprox.py`:** trains FedProx on the Dataset 2 Experiment 1 client partitions.
- **`d2_05a_verify_fedprox.py`:** verifies the Dataset 2 FedProx implementation and outputs.
- **`d2_06_train_scaffold.py`:** trains SCAFFOLD on the Dataset 2 Experiment 1 client partitions.
- **`d2_06a_verify_scaffold.py`:** verifies the Dataset 2 SCAFFOLD implementation and outputs.
- **`d2_07_evaluate_final_test.py`:** evaluates the selected Dataset 2 Experiment 1 models on the held-out test set.
- **`d2_07a_verify_final_test.py`:** verifies the Dataset 2 final-test evaluation procedure and outputs.

