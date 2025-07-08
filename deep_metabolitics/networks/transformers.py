import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaboliteTransformer(nn.Module):
    def __init__(self, num_metabolites=5835, hidden_dim=1024, num_heads=16, num_layers=6, stoichiometry_matrix=None):
        super(MetaboliteTransformer, self).__init__()

        self.embedding = nn.Linear(num_metabolites, hidden_dim)  # Metabolite Embedding
        self.positional_encoding = nn.Parameter(torch.randn(1, num_metabolites, hidden_dim))  # Positional Encoding

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4),
            num_layers=num_layers
        )

        # Stoikiometri matrisini attention'a entegre etmek için çarpım yapacağız
        self.stoichiometry_matrix = stoichiometry_matrix

        # Min & Max flux prediction heads
        self.min_flux_head = nn.Linear(hidden_dim, 10600)  # 10600 reaksiyon için min değer
        self.max_flux_head = nn.Linear(hidden_dim, 10600)  # 10600 reaksiyon için max değer

    def forward(self, x):
        x = self.embedding(x)  # 5835 metabolite'yi hidden_dim boyutuna dönüştür
        x += self.positional_encoding  # Positional Encoding ekle

        # Stoikiometri matrisini attention'a dahil et
        if self.stoichiometry_matrix is not None:
            x = torch.matmul(self.stoichiometry_matrix, x)  # Weighted Attention için çarpım

        x = self.transformer_encoder(x)

        min_flux = self.min_flux_head(x)  # Min Flux Tahmini
        max_flux = self.max_flux_head(x)  # Max Flux Tahmini

        return min_flux, max_flux
