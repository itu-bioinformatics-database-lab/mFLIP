import warnings

from deep_metabolitics.data.metabolight_dataset_cnn import ReactionMinMaxDataset, PathwayFluxMinMaxDataset
from regex import T

warnings.filterwarnings("ignore")

import os
import random
import joblib

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

from deep_metabolitics.networks.metabolite_cnn import MetaboliteCNN
from deep_metabolitics.utils.performance_metrics import PerformanceMetrics
from deep_metabolitics.utils.trainer_pm import predict_own_dnn, train_own_dnn


experiment_name = os.path.basename(__file__).replace(".py", "")
print(f"{experiment_name = }")

metabolite_scaler_method = "std"
target_scaler_method = "std"
# metabolite_coverage = "fully"
metabolite_coverage = None
pathway_features = True
k_folds = 10
batch_size = 32
resnet_version = 18
resnet_pretrained = True
epochs = 30

datasource = "pathwayfluxminmax_10_folds"

experiment_name = f"kolyoz_{experiment_name}_{metabolite_scaler_method}_{target_scaler_method}_{metabolite_coverage}_{batch_size}_{resnet_version}_{resnet_pretrained}_{epochs}"

# ids_list = get_workbench_metabolights_dataset_ids()
# print(f"{len(ids_list) = }")

test_source_list = get_aycan_dataset_ids()

fold_list = [6]
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
    validation_all_dataset = PathwayFluxMinMaxDataset(
        dataset_ids=[f"test_{fold}"],
        scaler_method=target_scaler_method,
        metabolite_scaler_method=metabolite_scaler_method,
        datasource=datasource,
        metabolite_coverage=train_all_dataset.metabolites_feature_columns,
        pathway_features=pathway_features,
        scaler = train_all_dataset.scaler,
        metabolite_scaler = train_all_dataset.metabolite_scaler,
        # eval_mode=False,
        run_init=True,
    )
    test_all_dataset = PathwayFluxMinMaxDataset(
        dataset_ids=test_source_list,
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
    
    experiment_fold = f"{experiment_name}_fold_{fold}"

    scaler = train_all_dataset.scaler

    n_features = train_all_dataset.n_metabolights
    out_features = train_all_dataset.n_labels

    model_file_path = outputs_dir / f"{experiment_fold}.joblib"
    if not model_file_path.exists():

        model = MetaboliteCNN(
            out_features=out_features,
            resnet_pretrained=resnet_pretrained,
            resnet_version = resnet_version
        )
        model, train_elapsed_time = train_own_dnn(
            train_dataset=train_all_dataset,
            model=model,
            device="cuda",
            batch_size=batch_size,
            learning_rate=0.00001,
            weight_decay=0.01,
            epochs=epochs,
        )
        print(f"{train_elapsed_time = }")
        joblib.dump(model, outputs_dir / f"{experiment_fold}.joblib")
    else:
        model = joblib.load(model_file_path)
        model = model.to(device="cuda")

    pred_train, true_train, _ = predict_own_dnn(
        model=model, dataset=train_all_dataset
    )
    pred_validation, true_validation, validation_elapsed_time = predict_own_dnn(
        model=model, dataset=validation_all_dataset
    )
    pred_test, true_test, test_elapsed_time = predict_own_dnn(
        model=model, dataset=test_all_dataset
    )

    performance_metrics = PerformanceMetrics(
        target_names=list(train_all_dataset.label_df.columns),
        experience_name=experiment_fold,
        train_time=train_elapsed_time,
        test_time=test_elapsed_time,
        validation_time=validation_elapsed_time,
        scaler=scaler,
    )
    performance_metrics.train_metric(y_true=true_train, y_pred=pred_train)
    performance_metrics.validation_metric(
        y_true=true_validation, y_pred=pred_validation
    )
    performance_metrics.test_metric(y_true=true_test, y_pred=pred_test)
    performance_metrics.complete()  # TODO foldlari tek dosyada tutsak guzel olur
