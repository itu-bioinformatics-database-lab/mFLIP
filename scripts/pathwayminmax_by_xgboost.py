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

metabolite_scaler_method = None
target_scaler_method = None


metabolite_coverage = "aycan_union"
# outputs_dir = f"pathwayminmax_{metabolite_scaler_method}_{target_scaler_method}"

experiment_name = os.path.basename(__file__).replace(".py", "")
print(f"{experiment_name = }")

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
    )


print(f"{uniform_dataset.metabolomics_df.isna().sum().sum()}")
print(f"{uniform_dataset.metabolomics_df.shape}")
print(f"{uniform_dataset.label_df.isna().sum().sum()}")
print(f"{uniform_dataset.label_df.shape}")
model_class = XGBRegressor
model = MultiOutputRegressor(model_class())

model = model.fit(uniform_dataset.metabolomics_df, uniform_dataset.label_df)

joblib.dump(model, f"old_models/{experiment_name}_{metabolite_scaler_method}_{target_scaler_method}.pkl")

predicted_minmax = model.predict(aycans_dataset.metabolomics_df)

import pandas as pd
p_df = pd.DataFrame(predicted_minmax, columns=aycans_dataset.label_df.columns)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

selected_columns = p_df.columns
print(f"{r2_score(aycans_dataset.label_df[selected_columns], p_df[selected_columns]) = }")
print(f"{mean_absolute_error(aycans_dataset.label_df[selected_columns], p_df[selected_columns]) = }")
print(f"{mean_squared_error(aycans_dataset.label_df[selected_columns], p_df[selected_columns]) = }")

# model, train_metrics, validation_metrics = train_sklearn(
#             train_dataset=uniform_dataset,
#             validation_dataset=aycans_dataset,
#             model=model,
#             pathway_names=list(uniform_dataset.label_df.columns),
#         )

# metrics_map = {}
# metrics_map["all_train_metrics"] = train_metrics
# metrics_map["all_val_metrics"] = validation_metrics
# test_metrics = evaluate(
#     model,
#     test_all_dataset,
#     pathway_names=list(train_all_dataset.label_df.columns),
#     scaler=scaler,
#     device=None,
# )
# metrics_map["all_test_metrics"] = test_metrics

# fpath = save_pickle(
#         metrics_map,
#         f"{outputs_dir}/metrics_map_{metabolite_coverage}_{metabolite_scaler_method}_{target_scaler_method}_{experiment_name}.pickle",
#     )
# print(fpath)