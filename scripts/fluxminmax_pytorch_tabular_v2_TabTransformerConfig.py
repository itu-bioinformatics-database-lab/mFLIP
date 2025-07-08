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
    ReactionMinMaxDataset,
)
from deep_metabolitics.data.properties import get_all_ds_ids

from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig
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

torch.set_float32_matmul_precision('high')  # Alternatif: 'medium'
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
uniform_dataset = ReactionMinMaxDataset(
    dataset_ids=generated_ds_ids,
    scaler_method=target_scaler_method,
    metabolite_scaler_method=metabolite_scaler_method,
    datasource="all_generated_datasets",
    metabolite_coverage=metabolite_coverage,
    pathway_features=False,
)

n_features = uniform_dataset.n_metabolights
out_features = uniform_dataset.n_labels

aycans_dataset = ReactionMinMaxDataset(
    dataset_ids=aycan_source_list,
    scaler_method=target_scaler_method,
    metabolite_scaler_method=metabolite_scaler_method,
    datasource="aycan",
    metabolite_coverage=metabolite_coverage,
    pathway_features=False,
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
    auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=100,
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
    # "input_embed_dim": 512,
    # "num_attn_blocks": 8,
    # "num_heads": 8,
    # "use_batch_norm": True,
    # "dropout": 0.3,
}
# target_range = True
# if target_range:
#     _target_range = []
#     for target in data_config.target:
#         _target_range.append(
#             (
#                 float(train[target].min()),
#                 float(train[target].max()),
#             )
#         )
#     model_config_params["target_range"] = _target_range

model_config = TabTransformerConfig(**model_config_params)


# model_config = CategoryEmbeddingModelConfig(
#     task="regression",
#     layers="1024-512-512",  # Number of nodes in each layer
#     activation="LeakyReLU",  # Activation between each layers
#     learning_rate=1e-3,
# )

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
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