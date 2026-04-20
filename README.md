# mFLIP: Metabolic Flux and Pathway Level Integration Platform

**mFLIP** is a machine learning and deep learning framework that predicts metabolic reaction flux ranges and pathway-level activity from metabolomic abundance data. It integrates constraint-based metabolic network models (Recon3D) with a suite of supervised learning algorithms, enabling researchers to infer biochemical flux distributions from measured metabolite concentrations.

This repository accompanies the publication:

> mFLIP: Metabolic Flux Interval Prediction, Baris Can, Sadi Celik, and Ali Cakmak. **BMC Bioinformatics**, 2026. DOI: 10.5281/zenodo.19581885

---

## Overview

Given a set of measured metabolite abundances for a biological sample, mFLIP predicts the minimum and maximum flux values for each metabolic reaction in the Recon3D human metabolic network, then aggregates them to pathway-level activity scores. Models are trained on datasets derived from MetaboLights studies and evaluated on independent cancer metabolomics cohorts (Aycan et al.).

**Key capabilities:**

- Multi-output regression for simultaneous prediction of reaction flux min/max values
- Pathway-level aggregation of reaction flux predictions
- 10-fold cross-validation with shared scalers across splits
- Support for Random Forest, XGBoost, MLP, SVM (scikit-learn), and deep learning models (FCNN, VAE, GNN, CNN, Transformer)
- Evaluation on held-out and external (unseen) datasets with RMSE, MAE, R², and Q-error metrics
- HPC-ready SLURM job scripts

---

## Repository Structure

```
mFLIP/
├── deep_metabolitics/          # Core Python package
│   ├── config.py               # Directory path configuration
│   ├── defined.py              # Pre-defined reaction lists
│   ├── vae.py                  # Variational autoencoder base
│   ├── data/                   # Dataset classes
│   │   ├── metabolight_dataset.py          # BaseDataset, PathwayFluxMinMaxDataset, ReactionMinMaxDataset
│   │   ├── metabolight_dataset_impute.py   # Imputation-enabled dataset variants
│   │   ├── fold_dataset.py                 # K-fold cross-validation split creation
│   │   ├── oneoutdataset.py                # Leave-one-out evaluation datasets
│   │   ├── properties.py                   # Data source queries (MetaboLights, Aycan)
│   │   └── fluxminmax_graph_dataset.py     # Graph dataset for GNN models
│   ├── networks/               # Neural network architectures
│   │   ├── metabolite_fcnn.py              # Fully Connected NN with residual blocks
│   │   ├── metabolite_vae.py               # Variational Autoencoder
│   │   ├── metabolite_vae_with_fcnn.py     # VAE + FCNN combined
│   │   ├── gnn.py                          # Graph Convolutional Network
│   │   ├── fluxminmax_gat.py               # Graph Attention Network
│   │   ├── transformers.py                 # Transformer encoder
│   │   ├── metabolite_cnn.py               # Convolutional Neural Network
│   │   ├── multiout_regressor_net_v2.py    # Multi-output regression network
│   │   └── multi_model_combined_fluxminmax*.py  # Ensemble architectures
│   ├── preprocessing/          # Feature engineering and scaling
│   │   ├── auto_scaler.py
│   │   ├── metabolitics_pipeline*.py
│   │   └── regressor_pipeline.py
│   ├── utils/                  # Training, evaluation, and helper utilities
│   │   ├── trainer_pm.py       # train_sklearn / predict_sklearn for scikit-learn models
│   │   ├── trainer_fcnn.py     # train / evaluate for PyTorch models
│   │   ├── trainer_graph.py    # Training for GNN models
│   │   ├── performance_metrics.py          # RMSE, MAE, R², Q-error evaluation
│   │   ├── performance_metrics_cls.py      # Classification-style metrics
│   │   ├── performance_metrics_unseen.py   # Metrics for external/unseen datasets
│   │   ├── metrics.py          # Core metric functions
│   │   ├── early_stopping.py
│   │   ├── logger.py
│   │   └── utils.py            # Device detection, database connection, COBRA loading
│   └── MetDIT/                 # Integrated MetDIT framework
│       ├── NetOmics/           # Network-based omics analysis
│       └── TransOmics/         # Transformer-based omics processing
├── scripts/                    # Experiment scripts (150+)
│   ├── pm_wm_pathwayfluxminmax_ml_pandas.py      # Main running example (Random Forest)
│   ├── pm_wm_pathwayfluxminmax_ml_pandas.slurm   # SLURM job script for HPC
│   ├── pm_wm_pathwayfluxminmax_fcnn_pandas.py    # FCNN deep learning variant
│   ├── pm_wm_pathwayfluxminmax_gnn_pandas.py     # GNN variant
│   ├── wm_create_fold_datasets*.py               # Fold dataset creation
│   └── ...
├── data/                       # Working datasets (populated by developer — see below)
├── outputs/                    # Model predictions and performance results
├── models/                     # Saved model checkpoints
├── logs/                       # Training logs and visualizations
├── oneleaveout_results/        # Leave-one-out cross-validation results
├── docker-compose.yml          # PostgreSQL database service
├── setup.py
└── requirements.txt
```

---

## Requirements

- Python 3.9+
- CUDA-capable GPU (optional, required for deep learning scripts)

Core dependencies (see `requirements.txt` for the full pinned list):

| Library | Version | Purpose |
|---|---|---|
| torch | 2.x | Deep learning |
| torch-geometric | 2.6.1 | Graph neural networks |
| lightning | 2.5.0 | PyTorch training loop |
| scikit-learn | 1.5.2 | ML models and pipelines |
| xgboost | 2.1.3 | Gradient boosting |
| pandas | 2.2.3 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| cobra | 0.29.1 | Constraint-based metabolic modelling |
| python-libsbml | 5.20.4 | SBML/Recon3D parsing |
| optuna | 4.1.0 | Hyperparameter optimisation |
| transformers | 4.47.1 | HuggingFace transformer models |
| wandb | 0.18.7 | Experiment tracking |
| psycopg | 3.2.3 | PostgreSQL connectivity |

---

## Installation

```bash
git clone https://github.com/<org>/mFLIP.git
cd mFLIP

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the deep_metabolitics package in editable mode
pip install -e .
```

---

## Data Setup

The `data/` directory is **not** included in this repository. You must populate it before running any experiments.

### Required directory layout

```
data/
├── network_models/
│   ├── Recon3D.json                        # Recon3D human metabolic network (COBRApy JSON)
│   ├── synonym-mapping.json                # Metabolite synonym mapping
│   └── metabolights_human_metadata.json    # MetaboLights study metadata
├── pathwayfluxminmax_10_folds/             # Pre-computed 10-fold split datasets
│   ├── train_0.pkl ... train_9.pkl
│   └── test_0.pkl  ... test_9.pkl
├── aycan/                                  # Cancer metabolomics datasets (Aycan et al.)
│   ├── after_name_mapp_fold_change/
│   │   ├── metabData_breast.*
│   │   ├── metabData_ccRCC3.*
│   │   ├── metabData_ccRCC4.*
│   │   ├── metabData_coad.*
│   │   ├── metabData_pdac.*
│   │   └── metabData_prostat.*
│   └── pathways_diff_scores/
├── work_workbench_metabolights_multiplied_by_factors/done/
│   └── <MetaboLights study files>
└── raw/
    └── csv/
        └── metabolites/
```

### PostgreSQL database (optional)

Several dataset classes can alternatively query a PostgreSQL database that stores metabolomic measurements from MetaboLights studies. Start the bundled service with:

```bash
docker-compose up -d
```

Default connection parameters: `host=localhost`, `port=5432`, `user=baris`, `password=123456`, `db=itu-bio`. These can be overridden by editing `deep_metabolitics/utils/utils.py`.

---

## Quick Start

The canonical example trains a Random Forest regressor on fold 9 of the 10-fold cross-validation splits and evaluates it on both the held-out validation fold and the external Aycan cancer cohorts.

```bash
cd scripts
python pm_wm_pathwayfluxminmax_ml_pandas.py
```

What the script does:

1. Loads `train_9` from `data/pathwayfluxminmax_10_folds/` and fits standard scalers on metabolite features and flux targets.
2. Loads `test_9` (validation) and all six Aycan cancer datasets (breast, ccRCC3, ccRCC4, COAD, PDAC, prostate) using the same scalers.
3. Trains a `MultiOutputRegressor(RandomForestRegressor(n_jobs=35))` for simultaneous min/max flux prediction across all reactions.
4. Saves the trained model to `outputs/`.
5. Computes and saves RMSE, MAE, R², and Q-error metrics for training, validation, and each external test set.

To switch to a different base estimator, edit `single_model_class_list` at the top of the script:

```python
single_model_class_list = [
    # RandomForestRegressor,
    XGBRegressor,
    # MLPRegressor,
]
```

---

## Running on HPC (SLURM)

A ready-to-use SLURM submission script is provided for the Truba HPC cluster:

```bash
cd scripts
sbatch pm_wm_pathwayfluxminmax_ml_pandas.slurm
```

The script requests:

| Resource | Value |
|---|---|
| Partition | barbun |
| Nodes | 1 |
| CPUs | 40 |
| RAM | 256 GB |
| Wall time | 3 days |

Adapt the `#SBATCH` directives and `module load` lines to match your cluster's environment before submitting.

---

## Deep Learning Variants

Other scripts in `scripts/` provide deep learning alternatives to the Random Forest baseline:

| Script | Architecture |
|---|---|
| `pm_wm_pathwayfluxminmax_fcnn_pandas.py` | Fully Connected NN (residual blocks, BatchNorm, LeakyReLU) |
| `pm_wm_pathwayfluxminmax_vae_pandas.py` | Variational Autoencoder |
| `pm_wm_pathwayfluxminmax_gnn_pandas.py` | Graph Convolutional / Attention Network |
| `pm_wm_pathwayfluxminmax_cnn_pandas.py` | Convolutional Neural Network |
| `pm_wm_pathwayfluxminmax_transformer_pandas.py` | Transformer encoder |

All deep learning scripts follow the same data layout and write results to `outputs/`.

---

## Evaluation Metrics

Results are written as JSON/CSV files to `outputs/`. The following metrics are reported:

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of determination
- **MAPE** — Mean Absolute Percentage Error
- **Q-error** — Quotient error (asymmetric measure suited to flux ratios)

Metrics are reported separately for min-flux targets, max-flux targets, and combined, as well as per-pathway breakdowns.

---

## Reproducibility

A fixed random seed (`seed = 10`) is set at the start of every script for NumPy, Python `random`, and PyTorch. Scikit-learn models receive this seed via their `random_state` parameter where applicable.

---

## Citation

If you use mFLIP in your research, please cite:

```bibtex
@article{mFLIP2025,
  title   = {mFLIP: Metabolic Flux Interval Prediction},
  author  = {Baris Can, Sadi Celik, and Ali Cakmak},
  journal = {BMC Bioinformatics},
  year    = {2026},
  doi     = {10.5281/zenodo.19581885}
}
```

---

## License

[License — e.g., MIT, Apache 2.0]
