import torch

from deep_metabolitics.utils.metrics import R2
from deep_metabolitics.utils.utils import get_device


class BaseNetwork(torch.nn.Module):

    def __init__(self, loss_method):
        super(BaseNetwork, self).__init__()
        self.loss_method = loss_method

        self.device = self.get_device()
        self.to(device=self.device)

    @staticmethod
    def get_device():
        device = get_device()
        print(f"PyTorch: Training model on device {device}.")
        return device

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)

        x = self.model(x)

        return x

    def loss_function(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ):
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        if self.loss_method == "mse":
            loss = torch.nn.MSELoss(reduction="mean")(y_true, y_pred)
        # elif self.loss_method == "RMSE":
        #     loss = torch.nn.RMSELoss(y_true, y_pred)  # RMSE
        elif self.loss_method == "mae":
            loss = torch.nn.L1Loss(reduction="mean")(y_true, y_pred)  # MAE
        elif self.loss_method == "rmse":
            loss = torch.nn.MSELoss(reduction="mean")(y_true, y_pred)
            loss = torch.sqrt(loss)  # RMSE
        elif self.loss_method == "R2Loss":
            loss = -R2(y_true, y_pred)  # R2Loss
        else:
            raise Exception("INVALID LOSS METHOD")

        return loss
