import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from deep_metabolitics.utils.utils import get_device
from deep_metabolitics.networks.base_network import BaseNetwork

class GNNModel(BaseNetwork):
    # def __init__(self, input_dim, hidden_dim, reaction_output_dim):
    def __init__(
        self,
        reaction_dim,
        metabolite_dim,
        hidden_dim,
        out_dim,
        reaction_output_dim,
        num_reactions,
        num_metabolites,
        pathway_reaction_list,
        loss_method="mse",
    ):
        self.device = get_device()
        self.reaction_dim = reaction_dim
        self.metabolite_dim = metabolite_dim
        self.num_reactions = num_reactions
        self.num_metabolites = num_metabolites
        self.pathway_reaction_list = pathway_reaction_list
        self.out_dim = out_dim

        input_dim = min(self.reaction_dim, self.metabolite_dim)

        super(GNNModel, self).__init__(loss_method=loss_method)
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_reaction = nn.Linear(hidden_dim, reaction_output_dim)  # Reaction skorları
        self.fc_out = nn.Linear(reaction_output_dim, 2)
        # self.fc_metabolite = nn.Linear(hidden_dim, 1)  # Metabolite skorları
        # self.fc_pathway_list = []
        # for pathway_index in range(self.out_dim):
        #     fc_pathway = nn.Linear(reaction_output_dim, 1, device=self.device)  # Pathway skorları
        #     self.fc_pathway_list.append(fc_pathway)

        self.to(device=self.device)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # GCN Katmanları
        x = self.gcn1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.gcn2(x, edge_index, edge_weight)
        x = F.relu(x)

        # Reaction düğümlerine göre çıkış

        # x = torch.cat(
        #     [
        #         F.relu(self.reaction_proj(x[: self.num_reactions])),
        #         F.relu(self.metabolite_proj(x[self.num_reactions :])),
        #     ],
        #     dim=0,
        # )
        # reaction_scores = self.fc_reaction(x[: self.num_reactions])
        reaction_scores = self.fc_reaction(x[self.num_reactions+self.num_metabolites:])
        reaction_scores = F.relu(reaction_scores)
        reaction_scores = self.fc_out(reaction_scores)
        reaction_scores = reaction_scores.flatten(start_dim=1)
        reaction_scores = reaction_scores.flatten().unsqueeze(0)
        # mean_tensors = []
        # for pathway_index, selected_rows in enumerate(self.pathway_reaction_list):
        #     selected_tensor = reaction_scores[selected_rows]  # Satırları seç
        #     selected_tensor = self.fc_pathway_list[pathway_index](selected_tensor)
        #     mean_tensor = torch.mean(selected_tensor, dim=0)  # Ortalama al
        #     mean_tensors.append(mean_tensor)
        # pathway_scores = torch.stack(mean_tensors)
        # reaction_scores = self.fc_reaction(x[: self.num_reactions])

        # Reaction skorlarından pathway hesaplama
        # pathway_scores = self.fc_pathway(reaction_scores)

        return {"pathways_pred":reaction_scores}

    def loss_function(self, predictions, targets):
        """
        Kombine loss fonksiyonu
        """
        predictions = predictions["pathways_pred"]
        # print(f"predictions.shape = {predictions.shape}")
        # print(f"targets.shape = {targets.shape}")
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
            "loss": total_loss
        }
