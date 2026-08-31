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

## Experiment 1 — Baseline Federated Learning Study

**Detailed Experiment 1 pipeline and file guide:** [`EXPERIMENT_1_README.md`](EXPERIMENT_1_README.md)

Earlier baseline experiments evaluated **FedAvg, FedProx, and SCAFFOLD** under controlled client-data heterogeneity.

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

## Experiment 2 — Label-Skew Study Reported in the Dissertation

The code for the experiment reported in the dissertation is organised under `fedartml_clean/`.

For both datasets, the experimental setting consists of `K=5` clients and partition seeds `42` and `43`.

### Dataset 1 — NF-UNSW-NB15-v2

Follow the Dataset 1 experiment through these files:

1. **IID partition preparation:** `fedartml_clean/00_prepare_iid_partitions.py`
2. **FedArtML non-IID partition generation:** `fedartml_clean/02_write_fedartml_partition*.py`
3. **Partition heterogeneity checks:** `fedartml_clean/05_d1_partition_diagnostics.py`
4. **FedAvg experiment runs:** `fedartml_clean/04_run_d1_fedavg_matrix.py`
5. **Final 100-round runs:** `fedartml_clean/06_extend_d1_fedavg_rounds.py`
6. **Held-out test evaluation:** `fedartml_clean/09_evaluate_d1_clean_test.py`

The Dataset 1 files under `fedartml_clean/` use the root-level FedAvg training implementation in `44_train_fedavg_hd_selected.py`. The connection between `fedartml_clean/05_d1_partition_diagnostics.py` and the root-level `28_build_heterogeneity_summary.py` is explained in the Dataset 1 diagnostics section below.

### Dataset 2 — NF-CSE-CIC-IDS2018-v2

Follow the Dataset 2 experiment through these files:

1. **IID partition preparation:** `fedartml_clean/00_prepare_iid_partitions.py`
2. **FedArtML non-IID partition generation:** `fedartml_clean/d2_02_write_fedartml_partition*.py`
3. **FedAvg experiment runs:** `fedartml_clean/d2_03a_run_fedavg_matrix.py`
4. **Final 100-round runs:** `fedartml_clean/d2_04_extend_fedavg_rounds.py`
5. **Held-out test evaluation:** `fedartml_clean/d2_09_evaluate_clean_test.py`

The Dataset 2 files under `fedartml_clean/` use the root-level FedAvg training implementation in `d2_04_train_fedavg.py`.

## Inspecting the Label Skew Study Structure

The following optional command displays the organisation of the Label Skew Study code and final outputs:

```bash
tree -L 2 fedartml_clean -I "*.log|*.pid|__pycache__|convergence_extension|convergence_extension_80|d2_calibration_logs|models|results|d2_models|d2_results|partitions|d2_partitions|paper_analysis|paper_data|paper_figures"
```

It shows the Dataset 1 and Dataset 2 partition-generation, FedAvg training and evaluation scripts, together with the final 100-round checkpoint and held-out test-output directories, while hiding logs, PID files, caches and intermediate directories.

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

For each run, the checkpoint with the highest validation Macro-F1 is selected before test access. The evaluation script then loads the selected checkpoint, runs inference on the held-out test set, calculates the final metrics, and writes the evaluation outputs.

`python3 fedartml_clean/09_evaluate_d1_clean_test.py --seeds 42 43 --execute-test`

### D1 heterogeneity diagnostics

**Step 1 — Diagnostic script**

The D1 partition diagnostics are carried out by:

`fedartml_clean/05_d1_partition_diagnostics.py`

This script examines the D1 FedArtML partitions after they have been created.

**Step 2 — Files used by the diagnostic script**

The D1 FedArtML partitions are stored under:

`fedartml_clean/partitions/k_5/`

The corresponding D1 training labels are loaded from:

`data/processed_37f/y_train.npy`

**Step 3 — Heterogeneity calculations**

`fedartml_clean/05_d1_partition_diagnostics.py` calls the Python function `condition_metrics()` defined inside the root-level file:

`28_build_heterogeneity_summary.py`

The `condition_metrics()` function takes the class counts for the five clients and the overall D1 class totals, converts the counts into class proportions, and calculates the heterogeneity statistics.

**Step 4 — Statistics calculated by `condition_metrics()`**

The `condition_metrics()` function calculates:

- pairwise Hellinger Distance
- pairwise HD-RMS
- Jensen–Shannon divergence
- total variation
- client-to-global distances
- client-size statistics
- missing-class counts

**Step 5 — FedArtML calculations in the diagnostic script**

`fedartml_clean/05_d1_partition_diagnostics.py` also directly imports `hellinger_distance()` and `jensen_shannon_distance()` from:

`fedartml.function_base`

These FedArtML functions calculate the Hellinger and Jensen–Shannon values from the same D1 partition class proportions. The Hellinger value is also checked against the value stored with the partition.

**Step 6 — Connection to the earlier root-level file**

Only the `condition_metrics()` function from `28_build_heterogeneity_summary.py` is called by `fedartml_clean/05_d1_partition_diagnostics.py`.

Other parts of `28_build_heterogeneity_summary.py`, including `k5_counts()`, `sensitivity_counts()` and its `main()` workflow, remain part of the earlier root-level experiment code and are separate from this D1 diagnostics step.

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

For each run, the checkpoint with the highest validation Macro-F1 is selected before test access. The evaluation script then loads the selected checkpoint, runs inference on the held-out test set, calculates the final metrics, and writes the evaluation outputs.

`python3 fedartml_clean/d2_09_evaluate_clean_test.py --seeds 42 43 --execute-test`

### Included Final Outputs

The repository includes the final 100-round histories and checkpoints for both datasets, together with held-out test results, per-class results, confusion matrices, checkpoint-selection records, and attack-detection summaries.

- **D1 100-round training outputs and checkpoints:** `fedartml_clean/convergence_extension_100/`
- **D1 held-out test outputs:** `fedartml_clean/final_test_100r/`
- **D2 100-round training outputs and checkpoints:** `fedartml_clean/d2_convergence_extension_100/`
- **D2 held-out test outputs:** `fedartml_clean/d2_final_test_100r/`

## Author

Saman
