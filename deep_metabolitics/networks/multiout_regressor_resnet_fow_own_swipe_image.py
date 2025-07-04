import torch

from deep_metabolitics.networks.base_network import BaseNetwork
from deep_metabolitics.utils.utils import get_device


class MultioutRegressorResNet(BaseNetwork):

    def __init__(
        self,
        out_features: int,  # of pathways
        resnet_pretrained: bool = True,
        resnet_version: int = 18,
        loss_method: str = "MSE",
    ):
        """
        Parameters:
            n_features (int): Initial feature size of the specific study, columns
        """
        super(MultioutRegressorResNet, self).__init__(loss_method=loss_method)

        # self.n_features = n_features
        self.out_features = out_features
        # self.n_start_layers = n_start_layers
        # self.dropout_rate = dropout_rate
        self.resnet_pretrained = resnet_pretrained
        self.resnet_version = resnet_version
        # 5835
        self.first_layer = torch.nn.Conv2d(1, 3, kernel_size=37, stride=26, padding=0)

        self.resnet_model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            f"resnet{self.resnet_version}",
            pretrained=self.resnet_pretrained,
        )
        self.resnet_model.fc = torch.nn.Linear(self.resnet_model.fc.in_features, out_features)

        self.model = torch.nn.Sequential(self.first_layer, self.resnet_model)

        self.device = self.get_device()
        self.to(device=self.device)
