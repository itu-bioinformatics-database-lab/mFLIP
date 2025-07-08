import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            # nn.BatchNorm1d(dim),
            nn.Tanhshrink(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            # nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return x + self.block(x)  # Residual bağlantı: x + F(x)


class MetaboliteImpFCNN(nn.Module):
    def __init__(
        self,
        input_dim=5835,
        output_dim=98,
        hidden_dims=[2048, 1024, 512, 256],
        dropout_rate=0.2,
        num_residual_blocks=2,  # Her hidden layer için residual block sayısı
    ):
        super().__init__()

        # Input normalization
        print(f"{input_dim = }")
        # self.input_norm = nn.BatchNorm1d(input_dim)

        # İlk katman (boyut değiştiren)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            # nn.BatchNorm1d(hidden_dims[0]),
            nn.Tanhshrink(),
            nn.Dropout(dropout_rate),
        )

        # Hidden layers with residual blocks
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Boyut değiştiren layer
            dimension_change = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                # nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.Tanhshrink(),
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
        # self.output = nn.Linear(hidden_dims[-1], output_dim)
        self.output = nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim),
                nn.Tanhshrink(),
            )

        # self.to(device='cuda')

    def forward(self, x):
        # print(f"{x.shape = }")
        # Input normalization
        # x = self.input_norm(x)

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
        # ce_loss = F.binary_cross_entropy_with_logits(predictions, targets)

        # Total loss - weight'leri ayarla
        total_loss = (
            mse_loss  # MSE ağırlığı azaltıldı
            + l1_loss  # L1 loss eklendi
            # + ce_loss  # CE loss eklendi
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

    @staticmethod
    def kl_loss(predictions, targets):
        # Normalize predictions ve targets (probability distribution haline getirmek için)
        pred_dist = torch.softmax(predictions, dim=1)
        target_dist = torch.softmax(targets, dim=1)

        # KL divergence
        kl = torch.sum(target_dist * torch.log(target_dist / pred_dist), dim=1)
        return kl.mean()
