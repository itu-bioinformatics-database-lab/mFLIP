import torch
import torch.nn as nn
import torch.nn.functional as F


class MetabolitePathwayVAE(nn.Module):
    def __init__(
        self,
        metabolite_dim=5835,
        pathway_dim=98,
        hidden_dims=[8192, 4096, 2048, 1024],
        dropout_rate=0.1,
        beta=0.01,
    ):
        super().__init__()

        self.metabolite_dim = metabolite_dim
        self.pathway_dim = pathway_dim
        self.beta = beta

        # Encoder
        encoder_layers = []

        # First encoder layer
        encoder_layers.extend(
            [
                nn.Linear(metabolite_dim, hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
            ]
        )

        # Hidden encoder layers with skip connections
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            block = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims[i + 1], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
            )
            self.encoder_blocks.append(block)

        self.encoder = nn.Sequential(*encoder_layers)

        # Pathway space projectors
        self.fc_mu = nn.Linear(hidden_dims[-1], pathway_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], pathway_dim)

        # Pathway predictor - Direct branch
        pathway_predictor_layers = []
        current_dim = hidden_dims[-1]
        predictor_dims = [1024, 512, 256, pathway_dim]

        for dim in predictor_dims[:-1]:
            pathway_predictor_layers.extend(
                [
                    nn.Linear(current_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout_rate),
                ]
            )
            current_dim = dim

        pathway_predictor_layers.append(nn.Linear(current_dim, pathway_dim))
        self.pathway_predictor = nn.Sequential(*pathway_predictor_layers)

        # Decoder
        decoder_dims = hidden_dims[::-1]
        decoder_layers = []

        decoder_layers.extend(
            [
                nn.Linear(pathway_dim, decoder_dims[0]),
                nn.BatchNorm1d(decoder_dims[0]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
            ]
        )

        for i in range(len(decoder_dims) - 1):
            decoder_layers.extend(
                [
                    nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                    nn.BatchNorm1d(decoder_dims[i + 1]),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout_rate),
                ]
            )

        decoder_layers.append(nn.Linear(decoder_dims[-1], metabolite_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        x = self.encoder(x)

        # Apply encoder blocks with skip connections
        features = x
        for block in self.encoder_blocks:
            block_out = block(features)
            features = (
                block_out + features if block_out.shape == features.shape else block_out
            )

        # Get pathway predictions directly from features
        pathways = self.pathway_predictor(features)

        # Get VAE parameters
        mu = self.fc_mu(features)
        log_var = self.fc_var(features)

        return mu, log_var, pathways

    def reparameterize(self, mu, log_var):
        """Sample from pathway distribution"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode pathways back to metabolites"""
        return self.decoder(z)

    def forward(self, metabolites):
        # Encode metabolites to pathway space
        mu, log_var, pathways = self.encode(metabolites)

        # Sample from pathway distribution
        z = self.reparameterize(mu, log_var)  # TODO burada x i kullanmayi da dene

        # Decode pathways back to metabolites
        metabolites_recon = self.decode(z)

        return metabolites_recon, pathways, mu, log_var

    def loss_function(
        self, metabolites_recon, metabolites, pathways, pathway_targets, mu, log_var
    ):
        """
        Calculate VAE loss with both reconstruction and pathway prediction components

        Args:
            metabolites_recon: Reconstructed metabolite values
            metabolites: Original metabolite values
            pathways: Sampled pathway values from latent space
            pathway_targets: True pathway values
        """
        # Metabolite reconstruction loss
        recon_loss = F.mse_loss(metabolites_recon, metabolites, reduction="mean")

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # Pathway prediction loss - MSE ve Huber loss kombinasyonu
        mse_loss = F.mse_loss(pathways, pathway_targets, reduction="mean")
        huber_loss = F.smooth_l1_loss(pathways, pathway_targets, reduction="mean")
        pathway_loss = 0.5 * mse_loss + 0.5 * huber_loss

        # Weight the components
        recon_weight = 1  # Metabolit rekonstrüksiyonu daha az önemli
        kl_weight = 0.01  # KL divergence minimal
        pathway_weight = 1  # Pathway tahmini çok daha önemli

        # Total loss
        total_loss = (
            recon_weight * recon_loss
            + kl_weight * kl_loss
            + pathway_weight * pathway_loss
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "pathway_loss": pathway_loss,
        }

    def predict_pathways(self, metabolites):
        """Direct pathway prediction from metabolites"""
        _, _, pathways = self.encode(metabolites)
        return pathways  # Use mean of pathway distribution as prediction
