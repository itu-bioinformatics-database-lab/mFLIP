import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_metabolitics.config import network_models_data_dir

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
        return x + self.block(x)  # Residual bağlantı: x + F(x)


class MetaboliteFCNN(nn.Module):
    def __init__(
        self,
        metabolities,
        label_names,
        input_dim=5835,
        output_dim=98,
        hidden_dims=[2048, 1024, 512, 256],
        dropout_rate=0.2,
        num_residual_blocks=2,  # Her hidden layer için residual block sayısı
    ):
        super().__init__()
        cm_file = "cmMat_recon_pathway.csv"
        self.cmMat = pd.read_csv(
            network_models_data_dir / cm_file,
            sep=',',
            header=0,
            index_col=0)
        self.n_comps = self.cmMat.shape[0]
        self.metabolities = metabolities
        self.label_names = label_names
        self.min_pathways = [f"{pathway_name}_min" for pathway_name in self.cmMat.columns]
        self.max_pathways = [f"{pathway_name}_max" for pathway_name in self.cmMat.columns]
        self.min_pathway_indices = [self.label_names.index(col) for col in self.min_pathways]
        self.max_pathway_indices = [self.label_names.index(col) for col in self.max_pathways]
        self.cmMat = torch.FloatTensor(self.cmMat.values).to("cuda")

        # Input normalization
        print(f"{input_dim = }")
        self.input_norm = nn.BatchNorm1d(input_dim)

        # İlk katman (boyut değiştiren)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )

        # Hidden layers with residual blocks
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Boyut değiştiren layer
            dimension_change = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
            )
            self.layers.append(dimension_change)

            # Aynı boyutta residual blocks
            residual_blocks = nn.ModuleList(
                [
                    ResidualBlock(hidden_dims[i + 1], dropout_rate)
                    for _ in range(num_residual_blocks)
                ]
            )
            self.layers.append(residual_blocks)

        # Output layer
        self.output = nn.Linear(hidden_dims[-1], output_dim)

        # self.to(device='cuda')

    def forward(self, x):
        # print(f"{x.shape = }")
        # Input normalization
        x = self.input_norm(x)

        # İlk katman
        x = self.input_layer(x)

        # Hidden layers with residual blocks
        for i in range(0, len(self.layers), 2):
            # Boyut değiştiren layer
            x = self.layers[i](x)

            # Residual blocks
            residual_blocks = self.layers[i + 1]
            for res_block in residual_blocks:
                x = res_block(x)

        # Output layer
        x = self.output(x)
        return {"pathways_pred": x}
    
    def updateC(self, m): # stoichiometric matrix
        
        c = torch.zeros((m.shape[0], self.n_comps))
        for i in range(c.shape[1]):
            tmp = m * self.cmMat[i,:]
            c[:,i] = torch.sum(tmp, dim=1)
        
        return c
    
    def min_flux_loss(self, predictions):
        lamb1 = 0.1
        lamb5 = 0.1

        m = predictions[:, self.min_pathway_indices]
        c = self.updateC(m=m)
        # balance constrain
        total1 = torch.pow(c, 2)
        total1 = torch.sum(total1, dim = 1) 

        # flux min loss
        total5 = torch.pow(m, 2)
        total5 = torch.sum(total5, dim = 1)
        # loss
        loss1 = torch.sum(lamb1 * total1)

        loss5 = torch.sum(lamb5 * total5)
        loss5 = loss5
        loss = loss1 + loss5
        print(f"loss1: {loss1}, loss5: {loss5}")
        return loss

    def max_flux_loss(self, predictions):
        lamb1 = 0.1
        lamb5 = 0.1

        m = predictions[:, self.max_pathway_indices]
        c = self.updateC(m=m)
        # balance constrain
        total1 = torch.pow(c, 2)
        total1 = torch.sum(total1, dim = 1) 

        # flux max loss
        total5 = torch.pow(m, 2)
        total5 = torch.sum(total5, dim = 1)
        # loss
        loss1 = torch.sum(lamb1 * total1)

        loss5 = torch.sum(lamb5 * total5)
        loss5 = 1 / (loss5 + 1e-8)
        loss = loss1 + loss5
        print(f"loss1: {loss1}, loss5: {loss5}")
        return loss

    def flux_loss(self, predictions):
        min_loss = self.min_flux_loss(predictions=predictions)
        max_loss = self.max_flux_loss(predictions=predictions)
        loss = min_loss + max_loss
        return loss


    def loss_function(self, predictions, targets):
        """
        Kombine loss fonksiyonu
        """
        predictions = predictions["pathways_pred"]
        # Input validation
        if torch.isnan(predictions).any() or torch.isnan(targets).any():
            print("Warning: NaN values detected!")
            predictions = torch.nan_to_num(predictions, nan=0.0)
            targets = torch.nan_to_num(targets, nan=0.0)

        flux_loss = self.flux_loss(predictions=predictions)
        # L1 Loss (MAE)
        l1_loss = F.l1_loss(predictions, targets, reduction="mean")

        # MSE Loss
        mse_loss = F.mse_loss(predictions, targets, reduction="mean")

        # Total loss - weight'leri ayarla
        total_loss = (
            (0.5 * mse_loss)  # MSE ağırlığı azaltıldı
            + (0.5 * l1_loss)  # L1 loss eklendi
            + flux_loss
        )
        # total_loss = correlation_loss

        return {
            "loss": total_loss,
        }
