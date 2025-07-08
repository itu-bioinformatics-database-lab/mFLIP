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



seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


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

target = list(uniform_dataset.label_df.columns)


train_df = uniform_dataset.metabolomics_df.join(uniform_dataset.label_df)
test_df = aycans_dataset.metabolomics_df.join(aycans_dataset.label_df)


train, val = train_test_split(train_df, random_state=42, test_size=0.2)
print(train.shape, val.shape, test_df.shape)


data_config = DataConfig(
    target=target,  # target should always be a list.
    continuous_cols=num_col_names,
    categorical_cols=[],
)
trainer_config = TrainerConfig(
    auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=100,
)
optimizer_config = OptimizerConfig()

model_config = CategoryEmbeddingModelConfig(
    task="regression",
    layers="1024-512-512",  # Number of nodes in each layer
    activation="LeakyReLU",  # Activation between each layers
    learning_rate=1e-3,
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
tabular_model.fit(train=train_df, validation=test_df)
result = tabular_model.evaluate(test_df)
pred_df = tabular_model.predict(test_df)
tabular_model.save_model("examples/basic2")
loaded_model = TabularModel.load_model("examples/basic2")

tabular_model.summary()

print_metrics(
    [
        (mean_squared_error, "MSE", {}),
        (mean_absolute_error, "MAE", {}),
        (r2_score, "R2", {}),
    ],
    test_df[target],
    pred_df,
    tag="Holdout",
)