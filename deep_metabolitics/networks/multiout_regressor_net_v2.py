import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_metabolitics.networks.base_network import BaseNetwork
from deep_metabolitics.utils.utils import get_device


class MultioutRegressorNETV2(BaseNetwork):
    def __init__(
        self,
        n_features: int,  # of metabolights
        out_features: int,  # of pathways
        n_start_layers: int,
        dropout_rate: float,
        loss_method: str = "MSE",
    ):
        """
        Parameters:
            n_features (int): Initial feature size of the specific study, columns
        """
        super(MultioutRegressorNETV2, self).__init__(loss_method=loss_method)

        self.n_features = n_features
        self.out_features = out_features
        self.n_start_layers = n_start_layers
        self.dropout_rate = dropout_rate

        start_layers = []
        input_size = self.n_features
        for i in range(self.n_start_layers):
            # output_size = int(input_size / 2 + i)
            # output_size = int(self.n_features / 2 + i)
            output_size = int(self.n_features / (2 + i))

            start_layers.append(torch.nn.Linear(input_size, output_size))
            # if i % 2 == 0:
            start_layers.append(torch.nn.BatchNorm1d(output_size))
            start_layers.append(torch.nn.LeakyReLU(0.2))
            # if i % 2 == 1:
            start_layers.append(torch.nn.Dropout(p=self.dropout_rate))

            input_size = output_size
        output_size = int(self.out_features / 2)
        end_layers = [
            torch.nn.Linear(input_size, output_size),
            torch.nn.BatchNorm1d(output_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=self.dropout_rate),
            torch.nn.Linear(output_size, self.out_features),
        ]

        layers = start_layers + end_layers
        self.model = torch.nn.Sequential(*layers)

        self.device = self.get_device()
        self.to(device=self.device)

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)

        x = self.model(x)

        return {"pathways_pred": x}

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

        # L1 Loss (MAE)
        l1_loss = F.l1_loss(predictions, targets, reduction="mean")

        # MSE Loss
        mse_loss = F.mse_loss(predictions, targets, reduction="mean")

        # Huber Loss
        huber_loss = F.smooth_l1_loss(predictions, targets, reduction="mean", beta=0.1)

        # Correlation loss
        eps = 1e-8
        pred_centered = predictions - predictions.mean(dim=0, keepdim=True)
        target_centered = targets - targets.mean(dim=0, keepdim=True)

        pred_std = torch.sqrt(torch.sum(pred_centered**2, dim=0, keepdim=True) + eps)
        target_std = torch.sqrt(
            torch.sum(target_centered**2, dim=0, keepdim=True) + eps
        )

        correlation = torch.sum(pred_centered * target_centered, dim=0) / (
            pred_std * target_std + eps
        )
        correlation_loss = 1 - correlation.mean()

        # kl_loss = self.kl_loss(predictions, targets)
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets)

        # Total loss - weight'leri ayarla
        total_loss = (
            mse_loss  # MSE ağırlığı azaltıldı
            + l1_loss  # L1 loss eklendi
            + ce_loss  # CE loss eklendi
            # + huber_loss  # Huber loss ağırlığı azaltıldı
            # + 1 * kl_loss  # KL loss eklendi
            # + 0.1 * correlation_loss  # Correlation loss ağırlığı azaltıldı
        )
        # total_loss = correlation_loss

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "huber_loss": huber_loss,
            "correlation_loss": correlation_loss,
        }
