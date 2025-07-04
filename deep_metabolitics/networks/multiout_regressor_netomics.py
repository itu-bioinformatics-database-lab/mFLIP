import torch

from deep_metabolitics.MetDIT.NetOmics.models.resnet import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
)
from deep_metabolitics.networks.base_network import BaseNetwork

RESNET_MAP = {
    18: ResNet18,
    34: ResNet34,
    50: ResNet50,
    101: ResNet101,
    152: ResNet152,
}


class MultioutRegressorNetOmics(BaseNetwork):

    def __init__(
        self,
        out_features: int,  # of pathways
        resnet_version: int = 18,
        loss_method: str = "mse",
    ):
        """
        Parameters:
            n_features (int): Initial feature size of the specific study, columns
        """
        super(MultioutRegressorNetOmics, self).__init__(loss_method=loss_method)

        # self.n_features = n_features
        self.out_features = out_features
        # self.n_start_layers = n_start_layers
        # self.dropout_rate = dropout_rate
        self.resnet_version = resnet_version
        # N = 32
        # S = 1
        # P = 1
        # K = N + 2 * P - (N - 1) * S
        # K # 3
        self.first_layer = torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.resnet_model = RESNET_MAP[self.resnet_version](num_classes=out_features)
        self.model = torch.nn.Sequential(self.first_layer, self.resnet_model)
        # from deep_metabolitics.MetDIT.NetOmics.models.resnet import ResNet18

        # Modeli yeniden tanımlayın
        # model = ResNet18()

        # fpath = "../deep_metabolitics/MetDIT/NetOmics/pretrain/pretrain-r18.pth"
        # # Ağırlıkları yükleyin
        # model.load_state_dict(torch.load(fpath), strict=False)
        # model.linear = torch.nn.Linear(model.linear.in_features, self.out_features)
        # self.model = model

        self.device = self.get_device()
        self.to(device=self.device)
