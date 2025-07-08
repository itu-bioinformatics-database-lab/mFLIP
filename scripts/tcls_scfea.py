import warnings

from deep_metabolitics.data.metabolight_dataset import ReactionMinMaxDataset, PathwayFluxMinMaxDataset

warnings.filterwarnings("ignore")

import os
import random
import joblib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from deep_metabolitics.config import outputs_dir

from deep_metabolitics.data.fold_dataset import get_fold_reactionminmaxdataset
from deep_metabolitics.data.properties import get_aycan_dataset_ids
from deep_metabolitics.utils.logger import create_logger
from deep_metabolitics.utils.trainer_fcnn import evaluate, train, warmup_training
from deep_metabolitics.utils.utils import load_pickle, save_pickle
from deep_metabolitics.data.properties import get_workbench_metabolights_dataset_ids
seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from deep_metabolitics.networks.metabolite_fcnn import MetaboliteFCNN
from deep_metabolitics.networks.metabolite_vae import MetaboliteVAE
from deep_metabolitics.networks.metabolite_vae_with_fcnn import MetaboliteVAEWithFCNN
from deep_metabolitics.networks.multiout_regressor_net_v2 import MultioutRegressorNETV2
from deep_metabolitics.utils.performance_metrics_cls import PerformanceMetrics
from deep_metabolitics.utils.performance_metrics_unseen import PerformanceMetricsUnseen
from deep_metabolitics.utils.trainer_pm import predict_own_dnn, train_own_dnn
from deep_metabolitics.config import data_dir
experiment_name = "pm_wm_pathwayfluxminmax_scfea"
print(f"{experiment_name = }")

metabolite_scaler_method = None
target_scaler_method = None
# metabolite_coverage = "fully"
metabolite_coverage = None
pathway_features = False
k_folds = 10
batch_size = 32


datasource = "pathwayfluxminmax_10_folds"

# experiment_name = f"{experiment_name}_{metabolite_scaler_method}_{target_scaler_method}_{metabolite_coverage}_{k_folds}_{batch_size}"

# ids_list = get_workbench_metabolights_dataset_ids()
# print(f"{len(ids_list) = }")

test_source_list = get_aycan_dataset_ids()
input_dir = data_dir / datasource

fold_list = [0]
# fold_list = list(range(k_folds))

for fold in fold_list:
    print(f"Fold: {fold}")
    train_df = pd.read_parquet(
        input_dir
        / f"metabolomics_train_{fold}.parquet.gzip"
    )
    train_df = train_df.reset_index(drop=True)
    train_label_df = pd.read_parquet(
        input_dir
        / f"label_train_{fold}.parquet.gzip"
    )
    train_label_df = train_label_df.reset_index(drop=True)

    train_df = train_df.loc[~(train_df > 10).any(axis=1)]
    train_df = train_df.loc[~(train_df < -10).any(axis=1)]
    # 2. Satırlarda eksik veri oranı %90'dan küçük olanları tut
    filter_null_columns_by = 0.9
    print(f"{train_df.shape = }")
    train_df = train_df.loc[train_df.isna().mean(axis=1) < filter_null_columns_by]
    train_label_df = train_label_df.loc[train_df.index]
    print(f"{train_df.shape = }")
    print(f"{train_label_df.shape = }")
    
    train_all_dataset = PathwayFluxMinMaxDataset(
        dataset_ids=[f"train_{fold}"],
        scaler_method=target_scaler_method,
        metabolite_scaler_method=metabolite_scaler_method,
        datasource=datasource,
        metabolite_coverage=metabolite_coverage,
        pathway_features=pathway_features,
        # eval_mode=False,
        run_init=True,
    )
    
    
    
    experiment_fold = f"{experiment_name}_fold_{fold}"

    scaler = None


    # model_file_path = outputs_dir / f"{experiment_fold}.joblib"
    # if model_file_path.exists():
    #     model = joblib.load(model_file_path)
    #     model = model.to(device="cuda")
    # else:
    #     print(f"Model not found: {model_file_path}")
    #     continue
    
    for ds_name in test_source_list:
        print(f"{ds_name = }")
        
        test_all_dataset = PathwayFluxMinMaxDataset(
            dataset_ids=[ds_name],
            scaler_method=target_scaler_method,
            metabolite_scaler_method=metabolite_scaler_method,
            datasource="aycan",
            metabolite_coverage=train_all_dataset.metabolites_feature_columns,
            pathway_features=pathway_features,
            scaler = train_all_dataset.scaler,
            metabolite_scaler = train_all_dataset.metabolite_scaler,
            # eval_mode=False,
            run_init=True,
        )
        test_label_df = test_all_dataset.label_df
        print(f"{test_label_df.shape = }")

        test_prediction_min_df = pd.read_csv(f"/arf/scratch/bacan/yl_tez/scFEA/output/metabolomics_{ds_name}_min.csv", index_col=0).fillna(0)
        test_prediction_min_df.columns = [f"{col}_min" for col in test_prediction_min_df.columns]
        test_prediction_max_df = pd.read_csv(f"/arf/scratch/bacan/yl_tez/scFEA/output/metabolomics_{ds_name}_max.csv", index_col=0).fillna(0)
        test_prediction_max_df.columns = [f"{col}_max" for col in test_prediction_max_df.columns]
        test_prediction_df = pd.concat([test_prediction_min_df, test_prediction_max_df], axis=1)

        true_test = test_label_df[train_label_df.columns].values
        pred_test = test_prediction_df[train_label_df.columns].values
 
        performance_metrics = PerformanceMetrics(
            target_names=list(train_label_df.columns),
            experience_name=experiment_fold,
            ds_name=ds_name,
            scaler=scaler,
        )

        performance_metrics.test_metric(y_true=true_test, y_pred=pred_test, factors_df=test_all_dataset.factors_df)
        
        performance_metrics_unseen = PerformanceMetricsUnseen(
            target_names=list(train_all_dataset.label_df.columns),
            experience_name=experiment_fold,
            ds_name=ds_name,
            scaler=scaler,
        )
        performance_metrics_unseen.test_metric(y_true=true_test, y_pred=pred_test)
        performance_metrics_unseen.complete()
