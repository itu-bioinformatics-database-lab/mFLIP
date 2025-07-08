# %%
import os

import pandas as pd

from deep_metabolitics.config import outputs_dir
from deep_metabolitics.utils.experiments import (
    make_overall_metrics_for_all_metrics,
    plot_scatter_for_all_pathways,
)
from deep_metabolitics.utils.utils import load_pickle, save_pickle

experience_name = "ml_aycan_union_quantile_std"


metrics_folder = outputs_dir / experience_name
algorithm_name_mapping = {
    "fcnn_base": "FCNN Base",
    "fcnn_deep": "FCNN Deep",
    "fcnn_robust": "FCNN Robust",
    "vae": "VAE",
    "vae_with_fcnn": "VAE with FCNN",
}

dataset_mapping = {
    "all_test_metrics": "All",
    "metabData_breast": "Breast",
    "metabData_ccRCC3": "ccRCC3",
    "metabData_ccRCC4": "ccRCC4",
    "metabData_coad": "Coad",
    "metabData_pdac": "Pdac",
    "metabData_prostat": "Prostat",
    "metabData_ccRCC3_ccRCC4": "ccRCC3_ccRCC4",
}


def get_metrics(fname):
    return load_pickle(metrics_folder / fname)


remove_name = "metrics_map_10folds_aycan_union_quantile_std_expres_tez_"
metric_fnames = {
    fname: fname for fname in os.listdir(metrics_folder) if fname.endswith(".pickle")
}


# %%
overall_df = make_overall_metrics_for_all_metrics(
    metric_fnames=metric_fnames,
    dataset_mapping=dataset_mapping,
    experience_name=experience_name,
    metrics_folder=metrics_folder,
)

# %%


# %%
pd.set_option("display.max_columns", None)
overall_df


# %%
overall_df["R2"].mean()

# %%
overall_75_df = make_overall_metrics_for_all_metrics(
    metric_fnames=metric_fnames,
    dataset_mapping=dataset_mapping,
    experience_name=experience_name,
    metrics_folder=metrics_folder,
    stat_for_pathway_metrics_q=75,
)


# %%
list(metric_fnames.keys())

# %%
selected_algorithm = "FCNN Base"

metric_fname = list(metric_fnames.keys())[0]
test_metrics = get_metrics(metric_fname)
ds_name = "metabData_breast"
for ds_name in dataset_mapping.keys():
    print(ds_name)
    plot_scatter_for_all_pathways(test_metrics, ds_name, metrics_folder)

# %%
for ds_name in dataset_mapping.keys():
    print(ds_name)
    plot_scatter_for_all_pathways(test_metrics, ds_name, metrics_folder)

# %%
