import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_metabolitics.utils.trainer_fcnn import warmup_training

from deep_metabolitics.networks.metabolite_fcnn import MetaboliteFCNN
from deep_metabolitics.utils.utils import load_pathway_metabolites_map



class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class MetaboliteVAEWithMean(nn.Module):
    def __init__(
        self,
        metabolite_dim,  # metabolit boyutu
        pathway_dim,  # pathway boyutu
        metabolite_names,
        pathway_metabolites_columns,
        # pathway_names,
        hidden_dims=[2048, 1024, 512],
        latent_dim=256,
        dropout_rate=0.2,
        num_residual_blocks=2,
    ):
        super().__init__()
        # self.metabolite_dim = metabolite_dim
        self.pathway_dim = pathway_dim
        self.latent_dim = latent_dim
        # self.pathway_metabolites_map = load_pathway_metabolites_map(is_unique=True)
        self.selected_idxs = [metabolite_names.index(f) for f in pathway_metabolites_columns]
        self.metabolite_dim = len(self.selected_idxs)

        total_input_dim = metabolite_dim #+ pathway_dim

        # Encoder
        self.encoder_layers = nn.ModuleList()

        # İlk encoder katmanı
        self.encoder_input = nn.Sequential(
            nn.BatchNorm1d(total_input_dim),
            nn.Linear(total_input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )

        # Encoder hidden layers
        for i in range(len(hidden_dims) - 1):
            # Boyut değiştiren layer
            dimension_change = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
            )
            self.encoder_layers.append(dimension_change)

            # Residual blocks
            residual_blocks = nn.ModuleList(
                [
                    ResidualBlock(hidden_dims[i + 1], dropout_rate)
                    for _ in range(num_residual_blocks)
                ]
            )
            self.encoder_layers.append(residual_blocks)

        # VAE latent layers
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        decoder_dims = hidden_dims[::-1]  # Reverse hidden dims for decoder

        # İlk decoder katmanı
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, decoder_dims[0]),
            nn.BatchNorm1d(decoder_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )

        # Decoder hidden layers
        for i in range(len(decoder_dims) - 1):
            # Boyut değiştiren layer
            dimension_change = nn.Sequential(
                nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                nn.BatchNorm1d(decoder_dims[i + 1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
            )
            self.decoder_layers.append(dimension_change)

            # Residual blocks
            residual_blocks = nn.ModuleList(
                [
                    ResidualBlock(decoder_dims[i + 1], dropout_rate)
                    for _ in range(num_residual_blocks)
                ]
            )
            self.decoder_layers.append(residual_blocks)

        # Output layers
        self.metabolite_output = nn.Linear(decoder_dims[-1], self.metabolite_dim)
        self.pathway_output = nn.Linear(decoder_dims[-1], pathway_dim)

    def encode(self, x):
        # Encoder forward pass
        x = self.encoder_input(x)

        for i in range(0, len(self.encoder_layers), 2):
            x = self.encoder_layers[i](x)
            residual_blocks = self.encoder_layers[i + 1]
            for res_block in residual_blocks:
                x = res_block(x)

        # VAE latent space
        mu = self.mu(x)
        log_var = self.log_var(x)
        log_var = torch.clamp(log_var, min=-100, max=100)  # veya daha dar bir aralık
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Decoder forward pass
        x = self.decoder_input(z)

        for i in range(0, len(self.decoder_layers), 2):
            x = self.decoder_layers[i](x)
            residual_blocks = self.decoder_layers[i + 1]
            for res_block in residual_blocks:
                x = res_block(x)

        # Separate outputs for metabolites and pathways
        metabolites = self.metabolite_output(x)
        pathways = self.pathway_output(x)
        return metabolites, pathways

    def forward(self, x):
        # self.fcnn_model.eval()
        # with torch.no_grad():
        # pathway_features = self.fcnn_model(x)
        # pathway_features = pathway_features["pathways_pred"]

        # data = torch.cat([x, pathway_features], dim=1)
        data = x

        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        metabolites_pred, pathways_pred = self.decode(z)
        return {
            "metabolites_true": x[:, self.selected_idxs],
            "metabolites_pred": metabolites_pred,
            "pathways_pred": pathways_pred,
            "mu": mu,
            "log_var": log_var,
        }

    def loss_function(
        self,
        predictions,
        pathway_targets,
    ):
        metabolites_true = predictions["metabolites_true"]
        metabolites_pred = predictions["metabolites_pred"]
        pathways_pred = predictions["pathways_pred"]
        mu = predictions["mu"]
        log_var = predictions["log_var"]

        # Reconstruction loss for metabolites (MSE)
        metabolites_mse_loss = F.mse_loss(
            metabolites_pred, metabolites_true, reduction="mean"
        )
        pathways_mse_loss = F.mse_loss(pathways_pred, pathway_targets, reduction="mean")
        mse_loss = metabolites_mse_loss + pathways_mse_loss

        # Reconstruction loss for metabolites (MSE)
        metabolites_l1_loss = F.l1_loss(
            metabolites_pred, metabolites_true, reduction="mean"
        )
        pathways_l1_loss = F.l1_loss(pathways_pred, pathway_targets, reduction="mean")
        l1_loss = metabolites_l1_loss + pathways_l1_loss

        # Reconstruction loss for metabolites (MSE)
        metabolites_huber_loss = F.smooth_l1_loss(
            metabolites_pred, metabolites_true, reduction="mean", beta=0.1
        )
        pathways_huber_loss = F.smooth_l1_loss(
            pathways_pred, pathway_targets, reduction="mean", beta=0.1
        )
        huber_loss = metabolites_huber_loss + pathways_huber_loss

        # KL Divergence
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        # print(pathway_targets)
        # Total loss
        # ce_loss = F.binary_cross_entropy_with_logits(pathways_pred, pathway_targets)
        total_loss = mse_loss + 0.1 * kld_loss
        # + ce_loss
        # + huber_loss + 0.1 * kld_loss

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "huber_loss": huber_loss,
            "kld_loss": kld_loss,
        }
