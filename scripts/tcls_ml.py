import warnings

from deep_metabolitics.data.metabolight_dataset import ReactionMinMaxDataset, PathwayFluxMinMaxDataset

warnings.filterwarnings("ignore")

import os
import random
import joblib
import gc

import numpy as np
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
from deep_metabolitics.utils.trainer_pm import predict_sklearn


from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor, XGBRFRegressor

single_model_class_list = [
    RandomForestRegressor,
    # XGBRegressor,
    # XGBRFRegressor,
    # SVR,
    # MLPRegressor,
]



experiment_name = "pm_wm_pathwayfluxminmax_ml_pandas"
print(f"{experiment_name = }")

metabolite_scaler_method = "std"
target_scaler_method = "std"
# metabolite_coverage = "fully"
metabolite_coverage = None
pathway_features = True
k_folds = 10
batch_size = 32

datasource = "pathwayfluxminmax_10_folds"

experiment_name = f"{experiment_name}_{metabolite_scaler_method}_{target_scaler_method}_{metabolite_coverage}_{k_folds}_{batch_size}"

# ids_list = get_workbench_metabolights_dataset_ids()
# print(f"{len(ids_list) = }")

test_source_list = get_aycan_dataset_ids()

# fold_list = [0]
# fold_list = list(range(k_folds))
# fold_list = [2, 3, 4, 5, 6, 7, 8, 9]
fold_list = [1]
for fold in fold_list:
    print(f"Fold: {fold}")
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
    
    
    # experiment_fold = f"{experiment_name}_fold_{fold}"

    scaler = train_all_dataset.scaler
    
    for model_class in single_model_class_list:
        experiment_fold = f"{experiment_name}_{model_class.__name__}_fold_{fold}"
        print(f"{experiment_fold = }")

        try:
            model_file_path = outputs_dir / f"{experiment_fold}.joblib"
            if model_file_path.exists():
                model = joblib.load(model_file_path, mmap_mode='r+')
                # model = joblib.load(model_file_path)
                # model = model.to(device="cuda")
            else:
                print(f"Model not found: {model_file_path}")
                continue
            
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

                pred_test, true_test, _ = predict_sklearn(
                    model=model, dataset=test_all_dataset
                )
                print(f"{true_test.shape = }", f"{pred_test.shape = }")
                print(f"{test_all_dataset.factors_df.shape = }")

                performance_metrics = PerformanceMetrics(
                    target_names=list(train_all_dataset.label_df.columns),
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
                
                del performance_metrics
                del performance_metrics_unseen
                del test_all_dataset
                del pred_test
                del true_test
                gc.collect()
            del model
            gc.collect()
        except Exception as e:
            print(f"{e = }")
            print(f"{fold = }")
    del train_all_dataset
    gc.collect()
