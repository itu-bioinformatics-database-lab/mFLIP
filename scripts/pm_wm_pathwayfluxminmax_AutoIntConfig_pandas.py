import warnings

from deep_metabolitics.data.metabolight_dataset import ReactionMinMaxDataset, PathwayFluxMinMaxDataset

warnings.filterwarnings("ignore")

import os
import random
import joblib
import gc
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
from pytorch_tabular import TabularModel
from pytorch_tabular.models import AutoIntConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)
from deep_metabolitics.utils.performance_metrics import PerformanceMetrics
from deep_metabolitics.utils.trainer_pm import predict_pytorch_tabular, train_pytorch_tabular
from deep_metabolitics.config import data_dir

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


torch.set_float32_matmul_precision('medium')  # Alternatif: 'medium'

experiment_name = os.path.basename(__file__).replace(".py", "")
print(f"{experiment_name = }")

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

for fold in list(range(k_folds)):
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

    scaler = None
    metabolite_scaler = None

    test_all_dataset = PathwayFluxMinMaxDataset(
        dataset_ids=test_source_list,
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


    n_features = train_df.shape[1]
    out_features = label_df.shape[1]


    num_col_names = list(train_df.columns)

    target_columns = list(label_df.columns)



    train_df = train_df.join(label_df)

    validation_df = pd.read_parquet(
        input_dir
        / f"metabolomics_test_{fold}.parquet.gzip"
    )
    label_df = pd.read_parquet(
        input_dir
        / f"label_test_{fold}.parquet.gzip"
    )
    validation_df = validation_df.reset_index(drop=True)
    label_df = label_df.reset_index(drop=True)
    validation_df = validation_df.join(label_df)
    validation_df = validation_df[train_df.columns]

    test_df = test_all_dataset.metabolomics_df.join(test_all_dataset.label_df)
    test_df = test_df[train_df.columns]

    train_df = train_df.fillna(0)
    validation_df = validation_df.fillna(0)
    test_df = test_df.fillna(0)

    
    experiment_fold = f"{experiment_name}_fold_{fold}"



    data_config = DataConfig(
        target=target_columns,
        continuous_cols=num_col_names,
        categorical_cols=[],
        num_workers=10,
        normalize_continuous_features=False,
        continuous_feature_transform=None,
        # continuous_feature_transform="quantile_normal",
        # continuous_feature_transform=continuous_feature_transform,
        # normalize_continuous_features=normalize_continuous_features,
    )
    # The allowable inputs are: ['quantile_normal', 'yeo-johnson', 'quantile_uniform', 'box-cox']

    trainer_config = TrainerConfig(
        # auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
        batch_size=batch_size,
        max_epochs=max_epochs,
        # early_stopping=1,
        # early_stopping_mode="min",
        # load_best=True,
        # checkpoints="valid_loss",
        # checkpoints_path=experiment_name,
        # checkpoints_mode="min",
        gradient_clip_val=1,
        profiler="advanced"
    )


    optimizer_config = OptimizerConfig(optimizer="lion_pytorch.Lion")

    model_config_params = {
        "task": "regression",
        "attn_embed_dim": 128,
        "num_heads": 8,
        "num_attn_blocks": 8,
    }

    target_range = True
    if target_range:
        _target_range = []
        for target in data_config.target:
            _target_range.append(
                (
                    float(-1000),
                    float(1000),
                )
            )
        model_config_params["target_range"] = _target_range

    model_config = AutoIntConfig(**model_config_params)

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        verbose=True,
        suppress_lightning_logger=True
    )


    model_file_path = outputs_dir / f"{experiment_fold}_tabular"
    if not model_file_path.exists():
        model, train_elapsed_time = train_pytorch_tabular(train_dataset=train_df, model=model, validation_df=validation_df)

        model.save_model(model_file_path)
    else:
        config = AutoIntConfig.load(model_file_path / "config.yaml")
        model = TabularModel(config=config)
        model.load_model(model_file_path)

    pred_train, true_train, _ = predict_pytorch_tabular(
        model=model, X=train_df, y=train_df[target_columns]
    )
    pred_validation, true_validation, validation_elapsed_time = predict_pytorch_tabular(
        model=model, X=validation_df, y=validation_df[target_columns]
    )
    pred_test, true_test, test_elapsed_time = predict_pytorch_tabular(
        model=model, X=test_df, y=test_df[target_columns]
    )

    performance_metrics = PerformanceMetrics(
        target_names=list(label_df.columns),
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
