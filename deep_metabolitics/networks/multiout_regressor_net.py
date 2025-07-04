import torch

from deep_metabolitics.networks.base_network import BaseNetwork
from deep_metabolitics.utils.utils import get_device


class MultioutRegressorNET(BaseNetwork):
    def __init__(
        self,
        n_features: int,  # of metabolights
        out_features: int,  # of pathways
        loss_method: str = "mse",
    ):
        """


        Parameters:
            n_features (int): Initial feature size of the specific study, columns
        """
        super(MultioutRegressorNET, self).__init__(loss_method=loss_method)

        self.n_features = n_features
        self.out_features = out_features
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_features, int(self.n_features / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear((int(self.n_features / 2)), int(self.n_features / 4)),
            torch.nn.ReLU(),
            torch.nn.Linear((int(self.n_features / 4)), int(self.out_features / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear((int(self.out_features / 2)), self.out_features),
        )

        self.device = self.get_device()
        self.to(device=self.device)
