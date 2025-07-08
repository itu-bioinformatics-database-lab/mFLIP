import warnings
import os

# Settings the warnings to be ignored
warnings.filterwarnings("ignore")

import random
import time
import torch
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import train_test_split

from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)
from pytorch_tabular.utils import make_mixed_dataset, print_metrics


from deep_metabolitics.config import all_generated_datasets_dir, aycan_full_data_dir

# from deep_metabolitics.data.properties import get_dataset_ids

from deep_metabolitics.data.metabolight_dataset import (
    PathwayMinMaxDataset,
)
from deep_metabolitics.data.properties import get_all_ds_ids

from pytorch_tabular import TabularModel
from pytorch_tabular.models import AutoIntConfig, TabNetModelConfig, FTTransformerConfig, DANetConfig
from pytorch_tabular.models.stacking import StackingModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)
from lion_pytorch import Lion
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.set_float32_matmul_precision('medium')  # Alternatif: 'medium'
# 'high' → En iyi performans, daha düşük hassasiyet.
# 'medium' → Performans ve hassasiyet arasında dengeli seçim.
# 'highest' (Varsayılan) → En yüksek hassasiyet, ancak daha yavaş.
# 🚀 Önerim: "high" modunu kullanarak hız kazanabilirsin.


experiment_name = os.path.basename(__file__).replace(".py", "")
print(f"{experiment_name = }")

metabolite_coverage = "aycan_union"

metabolite_scaler_method = None
target_scaler_method = None

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
    impute_metabolite=True,
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
    impute_metabolite=True,
)


num_col_names = list(uniform_dataset.metabolomics_df.columns)

target_columns = list(uniform_dataset.label_df.columns)


train_df = uniform_dataset.metabolomics_df.join(uniform_dataset.label_df)
test_df = aycans_dataset.metabolomics_df.join(aycans_dataset.label_df)


train, val = train_test_split(train_df, random_state=42, test_size=0.1)
print(train.shape, val.shape, test_df.shape)



data_config = DataConfig(
    target=target_columns,
    continuous_cols=num_col_names,
    categorical_cols=[],
    num_workers=10,
    normalize_continuous_features=True,
    continuous_feature_transform="quantile_uniform",
    # continuous_feature_transform=continuous_feature_transform,
    # normalize_continuous_features=normalize_continuous_features,
)
# The allowable inputs are: ['quantile_normal', 'yeo-johnson', 'quantile_uniform', 'box-cox']

trainer_config = TrainerConfig(
    # auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=32,
    max_epochs=100,
    # early_stopping=1,
    early_stopping_mode="min",
    early_stopping_patience=3,
    early_stopping="valid_loss",
    # load_best=True,
    # checkpoints="valid_loss",
    # checkpoints_path=experiment_name,
    # checkpoints_mode="min",
    gradient_clip_val=1,
    profiler="advanced"
)

optimizer_config = OptimizerConfig(optimizer="RAdam")
# optimizer_config = OptimizerConfig(optimizer="lion_pytorch.Lion")


model_config_params = {
    "task": "regression",
    "attn_embed_dim": 128,
    "num_heads": 8,
    "num_attn_blocks": 8,
    "learning_rate": 1e-3,

}

target_range = True
if target_range:
    _target_range = []
    for target in data_config.target:
        _target_range.append(
            (
                float(0),
                float(1000),
            )
        )
    model_config_params["target_range"] = _target_range


model_config_1 = FTTransformerConfig(
    task="regression",
    input_embed_dim=32,
    num_attn_blocks=2,
    num_heads=4,
    learning_rate=1e-3,
    target_range=_target_range.copy()
)
model_config_2 = TabNetModelConfig(
    task="regression",
    n_d=8,
    n_a=8,
    n_steps=3,
    learning_rate=1e-3,
    target_range=_target_range.copy()
)


model_config_3 = DANetConfig(
    task="regression",
    n_layers=32,
    abstlay_dim_1=128,
    k=2,
    dropout_rate=0.3,
    learning_rate=1e-3,
    target_range=_target_range.copy()
)
model_config_4 = AutoIntConfig(**model_config_params)

stacking_config = StackingModelConfig(
    task="regression",
    model_configs=[
        model_config_1,
        model_config_2,
        model_config_3,
        model_config_4
    ],
    head="LinearHead",
    head_config={
        "layers": "64",
        "activation": "ReLU",
        "dropout": 0.3
    },
    learning_rate=1e-3
)



tabular_model = TabularModel(
    data_config=data_config,
    model_config=stacking_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    verbose=True,
    suppress_lightning_logger=True
)
tabular_model.fit(train=train, validation=val)
result = tabular_model.evaluate(test_df)
pred_df = tabular_model.predict(test_df)
tabular_model.save_model(f"examples/{experiment_name}")
loaded_model = TabularModel.load_model(f"examples/{experiment_name}")


from pytorch_tabular.utils import print_metrics

print_metrics(
    [
        (mean_squared_error, "MSE", {}),
        (mean_absolute_error, "MAE", {}),
        (r2_score, "R2", {}),
        (mean_absolute_percentage_error, "MAPE", {}),
    ],
    test_df[target_columns],
    pred_df,
    tag="Holdout",
)
pred_df_copy = pred_df.copy()
pred_df_copy.columns = [c.replace("_prediction", "") for c in pred_df_copy.columns]

print(f"{mean_absolute_error(test_df[target_columns], pred_df_copy[target_columns]) = }")
print(f"{mean_squared_error(test_df[target_columns], pred_df_copy[target_columns]) = }")
print(f"{r2_score(test_df[target_columns], pred_df_copy[target_columns]) = }")
print(f"{mean_absolute_percentage_error(test_df[target_columns], pred_df_copy[target_columns]) = }")
