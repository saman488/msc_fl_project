# MSc Federated Learning Project

Supporting material for an MSc dissertation project in Federated Learning and Machine Learning.

## Repository Overview

This repository contains the source code, experiment configurations, analysis scripts, and selected results used in the dissertation.

Raw datasets are not stored in the repository.

The project uses:

- **NF-UNSW-NB15-v2**
- **NF-CSE-CIC-IDS2018-v2**

## Data Sources

Raw datasets are not redistributed in this repository.

- **NF-UNSW-NB15-v2** — University of Queensland — https://doi.org/10.48610/FFBB0C1
- **NF-CSE-CIC-IDS2018-v2** — University of Queensland — https://doi.org/10.48610/E9636B7

Both datasets are part of the University of Queensland Machine Learning-Based NIDS dataset collection.

## Experiment 1 — Baseline Federated Learning Study

The main federated learning study evaluates **FedAvg, FedProx, and SCAFFOLD** under controlled client-data heterogeneity.

For each dataset, training data is partitioned across five federated clients. The study includes an IID condition and fixed Dirichlet non-IID conditions, with multiple partition seeds used to avoid basing the comparison on a single client allocation.

The experimental workflow contains separate stages for:

1. dataset preprocessing;
2. federated client partition construction;
3. centralised reference-model training;
4. FedAvg training;
5. FedProx training;
6. SCAFFOLD training;
7. implementation verification;
8. held-out test evaluation.

The held-out test data is kept separate from training and model selection and is used for final evaluation.

The experiment examines how the three federated optimisation methods behave as client data heterogeneity changes under a controlled experimental setup.

## Experiment 2 — Label Skew Study

This study examines FedAvg under different levels of client label skew on both dissertation datasets.

The IID condition reuses the baseline IID partitions. Non-IID partitions are generated using FedArtML, with client heterogeneity measured using Hellinger distance. The study uses multiple partition seeds and separate held-out test evaluation.

## Environment

The development environment used for the dissertation was:

- Python 3.14.2
- NumPy 2.4.6
- pandas 3.0.3
- PyTorch 2.12.0
- scikit-learn 1.9.0
- Matplotlib 3.11.1
- FedArtML 0.1.34

Install the recorded dependencies with `python3 -m pip install -r requirements.txt`.

FedArtML is publicly available from PyPI; this project uses version `0.1.34`.

## Reproducing the Label Skew Study

The Label Skew Study builds on the preprocessing and IID partitioning used in the baseline study.

### Dataset 1

Prepare the dataset and baseline IID partitions:

`python3 03_preprocess.py`

`python3 30_build_dataset1_37feature_branch.py`

`python3 25_create_final_partitions.py`

Copy the baseline IID partitions into the Label Skew study layout:

`python3 fedartml_clean/00_prepare_iid_partitions.py`

Generate the non-IID partitions:

`python3 fedartml_clean/02_write_fedartml_partition.py`

`python3 fedartml_clean/02_write_fedartml_partition_hd0p5.py`

`python3 fedartml_clean/02_write_fedartml_partition_hd0p75.py`

`python3 fedartml_clean/02_write_fedartml_partition_hd0p9.py`

`python3 fedartml_clean/02_write_fedartml_partition_seed43_hd0p25.py`

`python3 fedartml_clean/02_write_fedartml_partition_seed43_hd0p5.py`

`python3 fedartml_clean/02_write_fedartml_partition_seed43_hd0p75.py`

`python3 fedartml_clean/02_write_fedartml_partition_seed43_hd0p9.py`

Run the initial FedAvg matrices:

`python3 fedartml_clean/04_run_d1_fedavg_matrix.py --partition-seed 42`

`python3 fedartml_clean/04_run_d1_fedavg_matrix.py --partition-seed 43`

Continue the runs to 100 communication rounds:

`python3 fedartml_clean/06_extend_d1_fedavg_rounds.py --partition-seed 42 --end-round 100 --extension-root fedartml_clean/convergence_extension_100/seed_42`

`python3 fedartml_clean/06_extend_d1_fedavg_rounds.py --partition-seed 43 --end-round 100 --source-results-root fedartml_clean/results/seed_43 --source-models-root fedartml_clean/models/seed_43 --primary-results-root fedartml_clean/results/seed_43 --extension-root fedartml_clean/convergence_extension_100/seed_43`

Run the held-out test evaluation:

`python3 fedartml_clean/09_evaluate_d1_clean_test.py --seeds 42 43 --execute-test`

### Dataset 2

Prepare the dataset and baseline IID partitions:

`python3 d2_01_preprocess_nf_cse_cic_ids2018_v2.py`

`python3 d2_02_create_final_partitions.py`

Prepare the IID study directories:

`python3 fedartml_clean/00_prepare_iid_partitions.py`

Generate the non-IID partitions:

`python3 fedartml_clean/d2_02_write_fedartml_partition.py`

`python3 fedartml_clean/d2_02_write_fedartml_partition_hd0p75.py`

`python3 fedartml_clean/d2_02_write_fedartml_partition_hd0p90.py`

`python3 fedartml_clean/d2_02_write_fedartml_partition_seed43_hd0p25.py`

`python3 fedartml_clean/d2_02_write_fedartml_partition_seed43_hd0p5.py`

`python3 fedartml_clean/d2_02_write_fedartml_partition_seed43_hd0p75.py`

`python3 fedartml_clean/d2_02_write_fedartml_partition_seed43_hd0p90.py`

Run the initial FedAvg matrices:

`python3 fedartml_clean/d2_03a_run_fedavg_matrix.py --partition-seed 42`

`python3 fedartml_clean/d2_03a_run_fedavg_matrix.py --partition-seed 43`

Continue the runs to 100 communication rounds:

`python3 fedartml_clean/d2_04_extend_fedavg_rounds.py --partition-seed 42 --end-round 100 --extension-root fedartml_clean/d2_convergence_extension_100/seed_42`

`python3 fedartml_clean/d2_04_extend_fedavg_rounds.py --partition-seed 43 --end-round 100 --source-results-root fedartml_clean/d2_results/seed_43 --source-models-root fedartml_clean/d2_models/seed_43 --primary-results-root fedartml_clean/d2_results/seed_43 --extension-root fedartml_clean/d2_convergence_extension_100/seed_43`

Run the held-out test evaluation:

`python3 fedartml_clean/d2_09_evaluate_clean_test.py --seeds 42 43 --execute-test`

### Included Final Outputs

The repository includes the final 100-round histories and checkpoints for both datasets, together with held-out test results, per-class results, confusion matrices, checkpoint-selection records, and attack-detection summaries.

## Author

Saman
