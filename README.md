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

## Main Entry Points — Label Skew Study

### Dataset 1

Train a FedAvg matrix for a selected partition seed:

`python3 fedartml_clean/04_run_d1_fedavg_matrix.py --partition-seed <seed>`

Run held-out test evaluation for selected seeds:

`python3 fedartml_clean/09_evaluate_d1_clean_test.py --seeds <seed1> <seed2> --execute-test`

### Dataset 2

Train a FedAvg matrix for partition seed 42 or 43:

`python3 fedartml_clean/d2_03a_run_fedavg_matrix.py --partition-seed <42|43>`

Run held-out test evaluation for selected seeds:

`python3 fedartml_clean/d2_09_evaluate_clean_test.py --seeds <seed1> <seed2> --execute-test`

Use `--help` on each script for the complete set of supported options.

## Author

Saman
