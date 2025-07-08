import warnings

from deep_metabolitics.data.metabolight_dataset import ReactionMinMaxDataset, PathwayFluxMinMaxDataset

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

from deep_metabolitics.networks.metabolite_fcnn import MetaboliteFCNN
from deep_metabolitics.networks.metabolite_vae import MetaboliteVAE
from deep_metabolitics.networks.metabolite_vae_with_fcnn import MetaboliteVAEWithFCNN
from deep_metabolitics.networks.multiout_regressor_net_v2 import MultioutRegressorNETV2
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
batch_size = 256
epochs = 100

datasource = "pathwayfluxminmax_10_folds"

experiment_name = f"{experiment_name}_{metabolite_scaler_method}_{target_scaler_method}_{metabolite_coverage}_{k_folds}_{batch_size}"

# ids_list = get_workbench_metabolights_dataset_ids()
# print(f"{len(ids_list) = }")

test_source_list = get_aycan_dataset_ids()

for fold in list(range(k_folds)):
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

        model = MetaboliteFCNN(
            input_dim=n_features,
            output_dim=out_features,
            hidden_dims=[2048, 1024, 512, 256],
            num_residual_blocks=2,
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
