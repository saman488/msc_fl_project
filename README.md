# MSc Federated Learning Project

Supporting material for an MSc dissertation project in Federated Learning and Machine Learning.

## Repository Contents

This repository contains the source code, experiment configurations, analysis scripts, and selected results used in the dissertation.

The repository also contains historical development material retained for research traceability. The final runnable workflow and relevant files will be documented clearly in this README.

## Data

Raw datasets are not stored in this repository.

Dataset sources, access instructions, and the exact preprocessing workflow used in the dissertation will be documented here before submission.

## Reproducibility

The final environment requirements and commands required to reproduce the dissertation experiments will be documented here after verification against the current codebase.

## Repository Structure

The repository includes:

- experiment and preprocessing scripts;
- configuration files;
- analysis and evaluation scripts;
- selected experimental results;
- historical development material under `archive/`.

Generated datasets, local environments, caches, and large intermediate partition files are excluded from version control.

## Important

Only workflows explicitly documented in this README should be treated as the final dissertation workflow. Historical and development files are retained for traceability and should not be assumed to represent the final methodology.


## Experiment 1 — Baseline Federated Learning Study

The main federated learning study evaluates **FedAvg, FedProx, and SCAFFOLD** under controlled client-data heterogeneity.

The experiments use both project datasets:

- **NF-UNSW-NB15-v2**
- **NF-CSE-CIC-IDS2018-v2**

For each dataset, training data is partitioned across five federated clients. The study includes an IID condition and fixed Dirichlet non-IID conditions, with multiple partition seeds used to avoid basing the comparison on a single client allocation.

The repository contains separate stages for:

1. dataset preprocessing;
2. federated client partition construction;
3. centralised reference-model training;
4. FedAvg training;
5. FedProx training;
6. SCAFFOLD training;
7. implementation verification;
8. held-out test evaluation.

The same held-out test data is kept separate from training and model selection and is used for final evaluation.

The purpose of this experiment is to examine how the three federated optimisation methods behave as client data heterogeneity changes, while keeping the main experimental setup controlled.

## Author

Saman
