import torch
from torch import nn

# Utils
from deep_metabolitics.utils.metrics import RMSELoss
from deep_metabolitics.utils.utils import get_device


class EncoderVAE(torch.nn.Module):
    def __init__(self, in_features: int):
        """
        Encoder module of the Metabolite Variational Autoencoder

        Parameters:
            - in_features (int): Initial feature size of the specific study, columns
        """
        super(EncoderVAE, self).__init__()

        self.in_features = in_features

        self.ENC_HIDDEN_1 = int(in_features / 2)
        self.ENC_HIDDEN_2 = int(self.ENC_HIDDEN_1 / 2)
        self.ENC_HIDDEN_3 = int(self.ENC_HIDDEN_2 / 2)

        # Encoder module
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.ENC_HIDDEN_1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=self.ENC_HIDDEN_1, out_features=self.ENC_HIDDEN_2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=self.ENC_HIDDEN_2, out_features=self.ENC_HIDDEN_3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DecoderVAE(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Decoder module of the Metabolite Variational Autoencoder

        Parameters:
            - in_features (int): Input size, equals to the latent dimension size
            - out_features (int): Initial feature size of the specific study, columns
        """
        super(DecoderVAE, self).__init__()

        self.out_features = out_features

        self.DEC_HIDDEN_3 = int(out_features / 2)
        self.DEC_HIDDEN_2 = int(self.DEC_HIDDEN_3 / 2)
        self.DEC_HIDDEN_1 = int(self.DEC_HIDDEN_2 / 2)

        self.in_features = in_features

        # Decoder module
        self.decoder = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=self.DEC_HIDDEN_1),
            # TODO: nn.BatchNorm1d(num_features=self.DEC_HIDDEN_1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=self.DEC_HIDDEN_1, out_features=self.DEC_HIDDEN_2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=self.DEC_HIDDEN_2, out_features=self.DEC_HIDDEN_3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=self.DEC_HIDDEN_3, out_features=self.out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class VAE(torch.nn.Module):
    def __init__(
        self,
        n_features: int,
        out_features: int,
        n_latent: int,
        kld_weight: float = 1e-2,
    ):
        """
        Fully connected Metabolite Variational Autoencoder

        Parameters:
            n_features (int): Initial feature size of the specific study, columns
            n_latent (int): Latent dimension to sample mu and sigma
        """
        super(VAE, self).__init__()

        self.n_features = n_features
        self.out_features = out_features
        self.n_latent = n_latent
        self.kld_weight = kld_weight

        # Encoder block
        self.encoder = EncoderVAE(in_features=self.n_features)

        # Fully connected layers for logvar and mu
        self.latent_mu = nn.Linear(
            in_features=int(self.n_features / 8), out_features=self.n_latent
        )
        self.latent_sigma = nn.Linear(
            in_features=int(self.n_features / 8), out_features=self.n_latent
        )

        # Decoder block
        self.decoder = DecoderVAE(
            in_features=self.n_latent, out_features=self.out_features
        )

        self.device = self.__get_device()
        self.to(device=self.device)

    @staticmethod
    def __get_device():
        device = get_device()
        print(f"PyTorch: Training model on device {device}.")
        return device

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        mu = None
        logvar = None

        # Encoder block
        x = self.encoder(x)
        # Update mu and logvar
        mu = self.latent_mu(x)
        logvar = self.latent_sigma(x)
        # Reparameterize
        x = self.reparameterize(mu=mu, logvar=logvar)

        # Decoder block
        x = self.decoder(x)

        return x, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        Parameters:
        - mu (torch.Tensor): Mean of the latent Gaussian [B x D]
        - logvar (torch.Tensor): Logarithm of the variance of the latent Gaussian [B x D]

        Returns:
        - out (torch.Tensor): Sampled tensor using reparameterization trick [B x D]
        """
        out = None

        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        out = torch.add(torch.multiply(eps, std), mu)

        return out

    def reconstruction_loss_function(
        self, recon_x: torch.Tensor, x: torch.Tensor, mask=None
    ) -> torch.Tensor:
        if mask is not None:
            recon_x = recon_x * mask
            x = x * mask
        # Reconstruction loss
        reconstruction_loss = nn.MSELoss(reduction="mean")(recon_x, x) * self.n_features
        # reconstruction_loss = nn.MSELoss(reduction="mean")(recon_x, x)  # MSE
        # reconstruction_loss = nn.L1Loss(reduction="mean")(recon_x, x)  # MAE
        # reconstruction_loss = RMSELoss(recon_x, x)  # RMSE
        return reconstruction_loss

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        mask=None,
    ):
        recon_x = recon_x.to(self.device)
        x = x.to(self.device)
        mu = mu.to(self.device)
        logvar = logvar.to(self.device)

        loss = None

        # Reconstruction loss
        reconstruction_loss = self.reconstruction_loss_function(recon_x, x, mask)

        # KL divergence loss
        # kl_loss = torch.mean(
        #     -0.5 * torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar), dim=1),
        #     dim=0,
        # )
        kl_loss = -0.5 * torch.sum(
            1 + logvar - torch.square(mu) - torch.exp(logvar), dim=-1
        )

        # Total loss = Reconstruction loss + KL loss
        # loss = reconstruction_loss + self.kld_weight * kl_loss
        loss = torch.mean(reconstruction_loss + self.kld_weight * kl_loss)

        return loss
