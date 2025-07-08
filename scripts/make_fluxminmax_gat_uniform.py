import warnings

# Settings the warnings to be ignored
warnings.filterwarnings("ignore")

import os
import random
import time

import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import ConcatDataset

from deep_metabolitics.config import aycan_full_data_dir
from deep_metabolitics.data.metabolight_dataset import ReactionMinMaxDataset
from deep_metabolitics.networks.fluxminmax_gat import GNNModel
from deep_metabolitics.utils.logger import create_logger
from deep_metabolitics.utils.trainer_fluxminmax_gnn import evaluate, train
from torch_geometric.loader import DataLoader

from deep_metabolitics.data.fluxminmax_graph_dataset import GraphDataset

metabolite_scaler_method = None
target_scaler_method = None

# from deep_metabolitics.data.properties import get_dataset_ids
experiment_name = "make_fluxminmax_gat_uniform_3layers"

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


metabolite_coverage = "aycan_union"

VALIDATION_RATE = 0.1
uniform_dataset = ReactionMinMaxDataset(
    dataset_ids=list(range(10)),
    scaler_method=target_scaler_method,
    metabolite_scaler_method=metabolite_scaler_method,
    datasource="uniform_generated",
    metabolite_coverage=metabolite_coverage,
    pathway_features=False,
)
train_dataset, validation_dataset = torch.utils.data.random_split(
        uniform_dataset, [1 - VALIDATION_RATE, VALIDATION_RATE]
    )


batch_size = 1
train_dataset = GraphDataset(
    dataset=train_dataset,
    fluxminmax_names=list(uniform_dataset.label_df.columns),
    metabolite_names=list(uniform_dataset.metabolomics_df.columns),
    device=uniform_dataset.device,
    div_flux=1000,
    div_metabolities=10
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validation_dataset = GraphDataset(
    dataset=validation_dataset,
    fluxminmax_names=list(uniform_dataset.label_df.columns),
    metabolite_names=list(uniform_dataset.metabolomics_df.columns),
    device=uniform_dataset.device,
    div_flux=1000,
    div_metabolities=10
)




num_reactions = len(train_dataset.reaction_id_index_map)
num_metabolities = len(train_dataset.metabolites)

# Model tanımla
model = GNNModel(
    reaction_dim=2,
    metabolite_dim=2,
    hidden_dim=32,
    num_reactions=num_reactions,
    num_metabolites=num_metabolities,
)

model, optimizer, train_metrics_list, validation_metrics_list = train(
    epochs=200,
    dataloader=train_loader,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    model=model,
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
    test_dataset = GraphDataset(
        dataset=eval_dataset,
        fluxminmax_names=list(eval_dataset.label_df.columns),
        metabolite_names=list(eval_dataset.metabolomics_df.columns),
        device=eval_dataset.device,
        div_flux=1000,
        div_metabolities=10
    )
    test_metrics = evaluate(
        model=model,
        dataset=eval_dataset,
    )
    save_pickle(
        data=test_metrics,
        fname=f"{experiment_name}/{aycan_source}_test_metrics.pickle",
    )
