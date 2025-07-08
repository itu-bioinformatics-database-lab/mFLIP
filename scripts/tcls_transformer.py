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


from deep_metabolitics.utils.performance_metrics_cls import PerformanceMetrics
from deep_metabolitics.utils.performance_metrics_unseen import PerformanceMetricsUnseen

from deep_metabolitics.utils.trainer_pm import predict_pytorch_tabular, train_pytorch_tabular
from deep_metabolitics.config import data_dir

from pytorch_tabular import TabularModel

# experiment_name = 
experiment_name_list = [
    "pm_wm_pathwayfluxminmax_FTTransformerConfig_pandas",
    "pm_wm_pathwayfluxminmax_TabNetModelConfig_pandas",
    "pm_wm_pathwayfluxminmax_TabTransformerConfig_pandas",
    "pm_wm_pathwayfluxminmax_NodeConfig_pandas"
    
]

metabolite_scaler_method = None
target_scaler_method = None
# metabolite_coverage = "fully"
metabolite_coverage = None
pathway_features = False
k_folds = 10
batch_size = 32
max_epochs = 100
datasource = "pathwayfluxminmax_10_folds"

# experiment_name = f"{experiment_name}_{metabolite_scaler_method}_{target_scaler_method}_{metabolite_coverage}_{k_folds}_{batch_size}"

# ids_list = get_workbench_metabolights_dataset_ids()
# print(f"{len(ids_list) = }")

test_source_list = get_aycan_dataset_ids()
input_dir = data_dir / datasource

# fold_list = [0]
fold_list = list(range(k_folds))
for experiment_name in experiment_name_list:
    print(f"{experiment_name = }")
    for fold in fold_list:
        print(f"Fold: {fold}")
        train_df = pd.read_parquet(
            input_dir
            / f"metabolomics_train_{fold}.parquet.gzip"
        )
        train_df = train_df.reset_index(drop=True)

        label_df = pd.read_parquet(
            input_dir
            / f"label_train_{fold}.parquet.gzip"
        )
        label_df = label_df.reset_index(drop=True)
        target_columns = list(label_df.columns)
        
        train_df_j = train_df.join(label_df)
        
        experiment_fold = f"{experiment_name}_fold_{fold}"

        model_file_path = outputs_dir / f"{experiment_fold}_tabular"
        if model_file_path.exists():
            model = TabularModel.load_model(model_file_path)
        else:
            print(f"Model not found: {model_file_path}")
            continue

        scaler = None
        metabolite_scaler = None

        
        for ds_name in test_source_list:
            print(f"{ds_name = }")
            test_all_dataset = PathwayFluxMinMaxDataset(
                dataset_ids=[ds_name],
                scaler_method=target_scaler_method,
                metabolite_scaler_method=metabolite_scaler_method,
                datasource="aycan",
                metabolite_coverage=list(train_df.columns),
                pathway_features=pathway_features,
                scaler = scaler,
                metabolite_scaler = metabolite_scaler,
                # eval_mode=False,
                run_init=True,
            )
            test_df = test_all_dataset.metabolomics_df.join(test_all_dataset.label_df)
            test_df = test_df[train_df_j.columns]
            test_df = test_df.fillna(0)
            

            pred_test, true_test, _ = predict_pytorch_tabular(
            model=model, X=test_df, y=test_df[target_columns]
        )
            print(f"{true_test.shape = }", f"{pred_test.shape = }")
            print(f"{test_all_dataset.factors_df.shape = }")

            performance_metrics = PerformanceMetrics(
                target_names=list(label_df.columns),
                experience_name=experiment_fold,
                ds_name=ds_name,
                scaler=scaler,
            )

            performance_metrics.test_metric(y_true=true_test, y_pred=pred_test, factors_df=test_all_dataset.factors_df)
            
            performance_metrics_unseen = PerformanceMetricsUnseen(
                target_names=list(test_all_dataset.label_df.columns),
                experience_name=experiment_fold,
                ds_name=ds_name,
                scaler=scaler,
            )
            performance_metrics_unseen.test_metric(y_true=true_test, y_pred=pred_test)
            performance_metrics_unseen.complete()
