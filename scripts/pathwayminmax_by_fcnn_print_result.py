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


demo_fcnn = load_pickle(
    "../outputs/pathwayminmax_by_fcnn/metrics_map_10folds_aycan_union_quantile_std.pickle"
)

# for fold in range(10):
p_r2_list = []
for pname, vals in demo_fcnn[0]["metabData_pdac"]["pathway_metrics"].items():
    p_r2_list.append(r2_score(vals["actual"], vals["predicted"]))
print(f"{np.median(p_r2_list) = }")

fold = 0
print(demo_fcnn[fold]["all_test_metrics"]["mae"], demo_fcnn[fold]["all_test_metrics"]["rmse"], demo_fcnn[
    fold
]["all_test_metrics"]["r2"])