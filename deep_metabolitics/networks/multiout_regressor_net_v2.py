import torch

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
