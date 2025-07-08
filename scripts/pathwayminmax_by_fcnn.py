import os
import warnings

from sklearn.neural_network import MLPRegressor

# Settings the warnings to be ignored
warnings.filterwarnings("ignore")

import random
import joblib
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

# from deep_metabolitics.data.fold_dataset_aycan import get_fold_dataset_aycan
# from deep_metabolitics.utils.logger import create_logger
# from deep_metabolitics.utils.trainer_fcnn import (
#     evaluate,
#     train,
#     train_sklearn,
#     warmup_training,
# )
from deep_metabolitics.utils.utils import load_pickle, save_pickle
from deep_metabolitics.config import all_generated_datasets_dir, aycan_full_data_dir
from deep_metabolitics.data.properties import get_all_ds_ids
from deep_metabolitics.data.metabolight_dataset import PathwayMinMaxDataset
from deep_metabolitics.networks.metabolite_fcnn import MetaboliteFCNN
from deep_metabolitics.utils.trainer_fcnn import evaluate, train, warmup_training

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBRFRegressor

# from deep_metabolitics.networks.metabolite_fcnn import MetaboliteFCNN
# from deep_metabolitics.networks.metabolite_vae import MetaboliteVAE
# from deep_metabolitics.networks.metabolite_vae_with_fcnn import MetaboliteVAEWithFCNN

# # %%
# from deep_metabolitics.networks.multiout_regressor_net_v2 import MultioutRegressorNETV2
# metabolite_scaler_method = "quantile"
# target_scaler_method = "std"
experiment_name = os.path.basename(__file__).replace(".py", "")
print(f"{experiment_name = }")


metabolite_scaler_method = "quantile"
target_scaler_method = "std"


metabolite_coverage = "aycan_union"
# outputs_dir = f"{experiment_name}_pathwayminmax_{metabolite_scaler_method}_{target_scaler_method}"
outputs_dir = experiment_name



aycan_source_list = [
        "metabData_breast",
        "metabData_ccRCC3",
        "metabData_ccRCC4",
        "metabData_coad",
        "metabData_pdac",
        "metabData_prostat",
    ]


generated_ds_ids = get_all_ds_ids(folder_path=all_generated_datasets_dir)
uniform_dataset = PathwayMinMaxDataset(
    dataset_ids=generated_ds_ids,
    scaler_method=target_scaler_method,
    metabolite_scaler_method=metabolite_scaler_method,
    datasource="all_generated_datasets",
    metabolite_coverage=metabolite_coverage,
    pathway_features=False,
)

n_features = uniform_dataset.n_metabolights
out_features = uniform_dataset.n_labels


aycans_dataset = PathwayMinMaxDataset(
        dataset_ids=aycan_source_list,
        scaler_method=target_scaler_method,
        metabolite_scaler_method=metabolite_scaler_method,
        datasource="aycan",
        metabolite_coverage=metabolite_coverage,
        pathway_features=False,
        scaler=uniform_dataset.scaler,
        metabolite_scaler=uniform_dataset.metabolite_scaler,
    )


print(f"{uniform_dataset.metabolomics_df.isna().sum().sum()}")
print(f"{uniform_dataset.metabolomics_df.shape}")
print(f"{uniform_dataset.label_df.isna().sum().sum()}")
print(f"{uniform_dataset.label_df.shape}")


batch_size = 32
scaler = uniform_dataset.scaler
train_loader = DataLoader(uniform_dataset, batch_size=batch_size, shuffle=True)
fold = 0
metrics_map = {fold: {}}
model = MetaboliteFCNN(
    input_dim=n_features,
    output_dim=out_features,
    hidden_dims=[2048, 128],
    num_residual_blocks=0,
)
model, optimizer, train_metrics, val_metrics = train(
    epochs=200,
    dataloader=train_loader,
    train_dataset=uniform_dataset,
    validation_dataset=aycans_dataset,
    model=model,
    fold=fold,
    pathway_names=list(uniform_dataset.label_df.columns),
    print_every=2,
)
metrics_map[fold]["all_train_metrics"] = train_metrics
metrics_map[fold]["all_val_metrics"] = val_metrics
test_metrics = evaluate(
    model,
    aycans_dataset,
    pathway_names=list(uniform_dataset.label_df.columns),
    scaler=scaler,
)
metrics_map[fold]["all_test_metrics"] = test_metrics

for source in aycan_source_list:
    metrics_map[fold][source] = {}
    test_dataset = PathwayMinMaxDataset(
        dataset_ids=source,
        scaler_method=target_scaler_method,
        metabolite_scaler_method=metabolite_scaler_method,
        datasource="aycan",
        metabolite_coverage=metabolite_coverage,
        pathway_features=False,
        scaler=uniform_dataset.scaler,
        metabolite_scaler=uniform_dataset.metabolite_scaler,
    )
    test_metrics = evaluate(
        model=model,
        dataset=test_dataset,
        pathway_names=list(test_dataset.label_df.columns),
        scaler=scaler,
    )
    metrics_map[fold][source]["test_metrics"] = test_metrics

    # store_train_dataset = aycan_map[source][fold]["train_dataset"]
    # test_metrics = evaluate(
    #     model=model,
    #     dataset=store_train_dataset,
    #     pathway_names=list(store_train_dataset.label_df.columns),
    #     scaler=scaler,
    # )
    # metrics_map[fold][source]["train_metrics"] = test_metrics


fpath = save_pickle(
    metrics_map,
    f"{outputs_dir}/metrics_map_10folds_{metabolite_coverage}_{metabolite_scaler_method}_{target_scaler_method}.pickle",
)

