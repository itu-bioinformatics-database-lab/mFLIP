import warnings

# Settings the warnings to be ignored
warnings.filterwarnings("ignore")

import os
import random
import time

import pandas as pd
import torch
import torch.multiprocessing as mp
from sklearn.model_selection import KFold, StratifiedKFold

from deep_metabolitics.config import all_generated_datasets_dir, aycan_full_data_dir

# %%
from deep_metabolitics.data.metabolight_dataset_impute import ReactionMinMaxDataset
from deep_metabolitics.data.properties import get_all_ds_ids
from deep_metabolitics.defined import reactions
from deep_metabolitics.utils.logger import create_logger
from lion_pytorch import Lion
# from deep_metabolitics.data.properties import get_dataset_ids




torch.backends.cudnn.benchmark = True  # Donanım için en iyi algoritmayı seçer
torch.backends.cudnn.enabled = True


import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_metabolitics.config import models_dir

# from deep_metabolitics.networks.metabolite_fcnn import MetaboliteFCNN
from deep_metabolitics.utils.utils import load_network, save_network
from deep_metabolitics.utils.trainer_fcnn_dev import train

class MetaboliteTransformer(nn.Module):
    def __init__(self, num_metabolites=5835, hidden_dim=1024, num_heads=16, num_layers=6, stoichiometry_matrix=None):
        super(MetaboliteTransformer, self).__init__()

        self.embedding = nn.Linear(num_metabolites, hidden_dim)  # Metabolite Embedding
        self.positional_encoding = nn.Parameter(torch.randn(1, num_metabolites, hidden_dim))  # Positional Encoding

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4),
            num_layers=num_layers
        )

        # Stoikiometri matrisini attention'a entegre etmek için çarpım yapacağız
        self.stoichiometry_matrix = stoichiometry_matrix

        # Min & Max flux prediction heads
        self.min_flux_head = nn.Linear(hidden_dim, 10600)  # 10600 reaksiyon için min değer
        self.max_flux_head = nn.Linear(hidden_dim, 10600)  # 10600 reaksiyon için max değer

    def forward(self, x):
        x = self.embedding(x)  # 5835 metabolite'yi hidden_dim boyutuna dönüştür
        x += self.positional_encoding  # Positional Encoding ekle

        # Stoikiometri matrisini attention'a dahil et
        if self.stoichiometry_matrix is not None:
            x = torch.matmul(self.stoichiometry_matrix, x)  # Weighted Attention için çarpım

        x = self.transformer_encoder(x)

        min_flux = self.min_flux_head(x)  # Min Flux Tahmini
        max_flux = self.max_flux_head(x)  # Max Flux Tahmini

        return min_flux, max_flux
    
    
    def loss_function(self, predictions, targets):
        """
        Kombine loss fonksiyonu
        """
        pred_min_flux, pred_max_flux = predictions
        target_min_flux = predictions[:, :10600]
        target_max_flux = predictions[:, -10600: ]
        
        # Input validation
        # if torch.isnan(predictions).any() or torch.isnan(targets).any():
        #     print("Warning: NaN values detected!")
        #     predictions = torch.nan_to_num(predictions, nan=0.0)
        #     targets = torch.nan_to_num(targets, nan=0.0)

        # L1 Loss (MAE)
        l1_loss = F.l1_loss(pred_min_flux, target_min_flux, reduction="mean") + F.l1_loss(pred_max_flux, target_max_flux, reduction="mean")
        # MSE Loss
        mse_loss = F.mse_loss(pred_min_flux, target_min_flux, reduction="mean") + F.mse_loss(pred_max_flux, target_max_flux, reduction="mean")

        # Huber Loss
        huber_loss = F.smooth_l1_loss(pred_min_flux, target_min_flux, reduction="mean", beta=0.1) + F.smooth_l1_loss(pred_max_flux, target_max_flux, reduction="mean", beta=0.1)

        # Total loss - weight'leri ayarla
        total_loss = (
            mse_loss  # MSE ağırlığı azaltıldı
            + l1_loss  # L1 loss eklendi
            + huber_loss  # Huber loss ağırlığı azaltıldı
        )
        # total_loss = correlation_loss

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "huber_loss": huber_loss
        }


if __name__ == "__main__":
    # mp.set_start_method("spawn")
    try:
        mp.set_start_method("spawn", force=True)  # force=True ile mevcut yöntemi zorla değiştirir
    except RuntimeError:
        pass  # Eğer zaten ayarlanmışsa, hata yok sayılır
    
    
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # %%
    experiment_name = "fluxminmax_fullbatch_transformers"

    # %%
    aycan_source_list = [
        "metabData_breast",
        "metabData_ccRCC3",
        "metabData_ccRCC4",
        "metabData_coad",
        "metabData_pdac",
        "metabData_prostat",
    ]

    # %%
    metabolite_coverage = "aycan_union"

    metabolite_scaler_method = "std"
    target_scaler_method = "std"

    # %%
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

    # %%
    aycans_dataset = ReactionMinMaxDataset(
        dataset_ids=aycan_source_list,
        scaler_method=target_scaler_method,
        metabolite_scaler_method=metabolite_scaler_method,
        datasource="aycan",
        metabolite_coverage=metabolite_coverage,
        pathway_features=False,
        scaler=uniform_dataset.scaler,
        metabolite_scaler=uniform_dataset.metabolite_scaler,
    )
    
    batch_size = 1024
    dataloader = DataLoader(
            uniform_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True
        )

    model = MetaboliteTransformer(num_metabolites=n_features, hidden_dim=1024, num_heads=16, num_layers=6, stoichiometry_matrix=None)
    model = torch.compile(model)
    
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)


    train(
    epochs=100,
    dataloader=None,
    train_dataset=uniform_dataset,
    validation_dataset=aycans_dataset,
    model=model,
    print_every=1)
    # %%
    # Eğitim sürecini başlat
    # pipeline.train_all_models(dataset=uniform_dataset, num_epochs=100_000, batch_size=1024, num_workers=32, prefetch_factor=2, validation_dataset=aycans_dataset, eval_every=100)
    
    # pipeline.evaluate_all_models(dataset=aycans_dataset, batch_size=len(aycans_dataset))
