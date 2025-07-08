import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from deep_metabolitics.networks.base_network import BaseNetwork
from deep_metabolitics.utils.utils import get_device


class GNNModel(BaseNetwork):
    # def __init__(self, input_dim, hidden_dim, reaction_output_dim):
    def __init__(
        self,
        reaction_dim,
        metabolite_dim,
        hidden_dim,
        num_reactions,
        num_metabolites,
        loss_method="mse",
        dropout_rate=0.2,
    ):
        self.device = get_device()
        self.reaction_dim = reaction_dim
        self.metabolite_dim = metabolite_dim
        self.num_reactions = num_reactions
        self.num_metabolites = num_metabolites
        # self.pathway_reaction_list = pathway_reaction_list

        input_dim = min(self.reaction_dim, self.metabolite_dim)

        super(GNNModel, self).__init__(loss_method=loss_method)
        self.gat1 = GATConv(reaction_dim, hidden_dim, heads=4, concat=True)
        self.seq_1 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )

        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.seq_2 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )

        self.gat3 = GATConv(hidden_dim * 4, reaction_dim, heads=1, concat=True)

        self.to(device=self.device)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = self.gat1(x, edge_index, edge_attr=edge_weight)
        x = self.seq_1(x)
        x = self.gat2(x, edge_index, edge_attr=edge_weight)
        x = self.seq_2(x)
        x = self.gat3(x, edge_index, edge_attr=edge_weight)

        x = x[: self.num_reactions]

        return x

    def loss_function(self, predictions, targets):
        mse_loss = F.mse_loss(predictions, targets, reduction="mean").sqrt()
        total_loss = mse_loss

        loss_dict = {}
        loss_dict["loss"] = total_loss
        loss_dict["mse_loss"] = mse_loss

        return loss_dict
