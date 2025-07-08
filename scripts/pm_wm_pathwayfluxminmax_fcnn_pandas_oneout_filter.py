import warnings

from deep_metabolitics.data.metabolight_dataset import ReactionMinMaxDataset, PathwayFluxMinMaxDataset
from deep_metabolitics.data.oneoutdataset_filter import OneoutDataset
warnings.filterwarnings("ignore")

import os
import random
import joblib

import torch

from deep_metabolitics.config import outputs_dir, data_dir

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
target_scaler_method = None
# metabolite_coverage = "fully"
metabolite_coverage = None
pathway_features = True
k_folds = 10
batch_size = 1024
epochs=100


datasource = "pathwayfluxminmax_10_folds"

experiment_name = f"{experiment_name}_{metabolite_scaler_method}_{target_scaler_method}_{metabolite_coverage}_{k_folds}_{batch_size}"

# ids_list = get_workbench_metabolights_dataset_ids()
# print(f"{len(ids_list) = }")

test_source_list = get_aycan_dataset_ids()

for fold in list(range(k_folds)):
    train_path = data_dir / datasource / f"train_oneout_{fold}.parquet.gzip"
    validation_path = data_dir / datasource / f"test_oneout_{fold}.parquet.gzip"
    test_path = data_dir / datasource / f"test_oneout_cancer.parquet.gzip"

    train_all_dataset = OneoutDataset(dataframe_path=train_path, scaler=metabolite_scaler_method)
    validation_all_dataset = OneoutDataset(dataframe_path=validation_path, scaler=train_all_dataset.scaler, one_hot_encoder=train_all_dataset.one_hot_encoder, features=train_all_dataset.features)
    test_all_dataset = OneoutDataset(dataframe_path=train_path, scaler=train_all_dataset.scaler, one_hot_encoder=train_all_dataset.one_hot_encoder, features=train_all_dataset.features)


    experiment_fold = f"{experiment_name}_fold_{fold}"

    # scaler = train_all_dataset.scaler
    scaler = None

    n_features = train_all_dataset.n_features
    out_features = train_all_dataset.n_labels

    model_file_path = outputs_dir / f"{experiment_fold}.joblib"
    if not model_file_path.exists():

        model = MetaboliteFCNN(
            input_dim=n_features,
            output_dim=out_features,
            hidden_dims=[2048, 128],
            num_residual_blocks=0,
        )
        model, train_elapsed_time = train_own_dnn(
            train_dataset=train_all_dataset,
            model=model,
            device="cuda",
            batch_size=batch_size,
            learning_rate=0.0001,
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
    train_all_dataset.set_predicted(predicted=pred_train)
    pred_validation, true_validation, validation_elapsed_time = predict_own_dnn(
        model=model, dataset=validation_all_dataset
    )
    validation_all_dataset.set_predicted(predicted=pred_validation)
    pred_test, true_test, test_elapsed_time = predict_own_dnn(
        model=model, dataset=test_all_dataset
    )
    test_all_dataset.set_predicted(predicted=pred_test)

    train_real_target = train_all_dataset.get_real_target()

    performance_metrics = PerformanceMetrics(
        target_names=list(train_real_target.columns),
        experience_name=experiment_fold,
        train_time=train_elapsed_time,
        test_time=test_elapsed_time,
        validation_time=validation_elapsed_time,
        scaler=scaler,
    )
    performance_metrics.train_metric(y_true=train_real_target.values, y_pred=train_all_dataset.get_predicted_target().values)
    performance_metrics.validation_metric(
        y_true=validation_all_dataset.get_real_target().values, y_pred=validation_all_dataset.get_predicted_target().values
    )
    performance_metrics.test_metric(y_true=test_all_dataset.get_real_target().values, y_pred=test_all_dataset.get_predicted_target().values)
    performance_metrics.complete()  # TODO foldlari tek dosyada tutsak guzel olur
