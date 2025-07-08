import warnings

# Settings the warnings to be ignored
warnings.filterwarnings("ignore")

import random
import time
import torch

from sklearn.model_selection import KFold, StratifiedKFold
import os
import pandas as pd
from deep_metabolitics.config import aycan_full_data_dir

# from deep_metabolitics.data.properties import get_dataset_ids

from deep_metabolitics.utils.logger import create_logger

from deep_metabolitics.data.metabolight_dataset import (
    ReactionMinMaxDataset,
)
from deep_metabolitics.networks.metabolite_fcnn import MetaboliteFCNN
from torch.utils.data import ConcatDataset, DataLoader
from deep_metabolitics.utils.trainer_fcnn_dev import (
    evaluate,
    train,
    warmup_training
)
seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

aycan_source_list = [
    "metabData_breast",
    "metabData_ccRCC3",
    "metabData_ccRCC4",
    "metabData_coad",
    "metabData_pdac",
    "metabData_prostat",
]

experiment_name = "fluxminmax_base_uhem_continous_uniform"

metabolite_coverage = "aycan_union"
metabolite_scaler_method = None
target_scaler_method = None

VALIDATION_RATE = 0.1
uniform_dataset = ReactionMinMaxDataset(
    dataset_ids=list(range(10)),
    scaler_method=target_scaler_method,
    metabolite_scaler_method=metabolite_scaler_method,
    datasource="uniform_generated",
    metabolite_coverage=metabolite_coverage,
    pathway_features=False,
)
n_features = uniform_dataset.n_metabolights
out_features = uniform_dataset.n_labels
train_dataset, validation_dataset = torch.utils.data.random_split(
        uniform_dataset, [1 - VALIDATION_RATE, VALIDATION_RATE]
    )

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(
    validation_dataset, batch_size=len(validation_dataset), shuffle=False
)




model = MetaboliteFCNN(
    input_dim=n_features,
    output_dim=out_features,
    hidden_dims=[2048, 128],
    num_residual_blocks=0,
)
model = model.to(device="cuda")
# # model = warmup_training(model, train_loader, num_warmup_steps=1000)
model, optimizer, train_metrics, val_metrics = train(
    # epochs=25_000,
    epochs=10,
    # epochs=10,
    dataloader=train_loader,
    # train_dataset=train_dataset_splitted,
    # validation_dataset=validation_dataset_splitted,
    train_dataset=None,
    validation_dataset=validation_loader,
    model=model,
    logger=None,
    fold=None,
    pathway_names=list(uniform_dataset.label_df.columns),
    print_every=1,
    # learning_rate=0.0001,
    # early_stopping_patience=25,
    # early_stopping_min_delta=0.001,
)

from deep_metabolitics.utils.utils import save_network, save_pickle

save_network(model=model, fname=f"{experiment_name}.model")

save_pickle(
    data=train_metrics_list, fname=f"{experiment_name}/train_metrics_list.pickle"
)

save_pickle(
    data=validation_metrics_list,
    fname=f"{experiment_name}/validation_metrics_list.pickle",
)


for aycan_source in aycan_source_list:
    metabolomics_df, label_df, dataset_ids_df, factors_df = (
        ReactionMinMaxDataset.load_data_aycan_csv(dataset_ids=[aycan_source])
    )
    eval_dataset = ReactionMinMaxDataset(
        metabolomics_df=metabolomics_df,
        label_df=label_df,
        factors_df=factors_df,
        dataset_ids_df=dataset_ids_df,
        scaler_method=target_scaler_method,
        metabolite_scaler_method=metabolite_scaler_method,
        datasource="aycan",
        metabolite_coverage=metabolite_coverage,
        pathway_features=False,
    )
    test_metrics = evaluate(
        model=model,
        dataset=eval_dataset,
    )
    save_pickle(
        data=test_metrics,
        fname=f"{experiment_name}/{aycan_source}_test_metrics.pickle",
    )
