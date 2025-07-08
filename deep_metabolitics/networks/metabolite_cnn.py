import torch

import torch.nn.functional as F

class MetaboliteCNN(torch.nn.Module):

    def __init__(
        self,
        out_features: int,  # of pathways
        resnet_pretrained: bool = True,
        resnet_version: int = 18
    ):
        """
        Parameters:
            n_features (int): Initial feature size of the specific study, columns
        """
        super().__init__()

        # self.n_features = n_features
        self.out_features = out_features
        # self.n_start_layers = n_start_layers
        # self.dropout_rate = dropout_rate
        self.resnet_pretrained = resnet_pretrained
        self.resnet_version = resnet_version

        self.first_layer = torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)

        if isinstance(resnet_version, int):
            self.resnet_model = torch.hub.load(
                "pytorch/vision:v0.10.0",
                f"resnet{self.resnet_version}",
                pretrained=self.resnet_pretrained,
            )
            self.resnet_model.fc = torch.nn.Linear(
                self.resnet_model.fc.in_features, out_features
            )
        else:
            self.resnet_model = torch.hub.load(
                "pytorch/vision:v0.10.0", "vgg16", pretrained=True
            )
            self.resnet_model.classifier[6] = torch.nn.Linear(
                in_features=4096, out_features=out_features
            )

        self.model = torch.nn.Sequential(self.first_layer, self.resnet_model)

    def forward(self, x: torch.Tensor):
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
        # huber_loss = F.smooth_l1_loss(predictions, targets, reduction="mean", beta=0.1)

        # Correlation loss
        # eps = 1e-8
        # pred_centered = predictions - predictions.mean(dim=0, keepdim=True)
        # target_centered = targets - targets.mean(dim=0, keepdim=True)

        # pred_std = torch.sqrt(torch.sum(pred_centered**2, dim=0, keepdim=True) + eps)
        # target_std = torch.sqrt(
        #     torch.sum(target_centered**2, dim=0, keepdim=True) + eps
        # )

        # correlation = torch.sum(pred_centered * target_centered, dim=0) / (
        #     pred_std * target_std + eps
        # )
        # correlation_loss = 1 - correlation.mean()

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
            # "mse_loss": mse_loss,
            # "l1_loss": l1_loss,
            # "huber_loss": huber_loss,
            # "correlation_loss": correlation_loss,
        }