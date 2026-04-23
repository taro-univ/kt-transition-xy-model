"""
vae.py

Convolutional Variational Autoencoder (VAE) for 2D XY spin configurations.

Architecture:
    Input:  (B, C_in, L, L)
    Encoder:
        ConvBlock stack (stride=2 downsampling)
        → flatten
        → fully connected layers producing:
            - mu      (latent mean)
            - logvar  (latent log-variance)
    Reparameterization:
        z = mu + eps * sigma
    Decoder:
        Linear projection
        → reshape
        → DeconvBlock stack (stride=2 upsampling)
        → reconstructed output

Designed for:
    - Learning latent representations of XY spin configurations
    - Studying KT transition structure in latent space
    - Comparison with contrastive or supervised latent models

Loss (typical training):
    recon_loss = MSE(x_recon, x)
    kl_loss    = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
    total_loss = recon_weight * recon_loss + beta * kl_loss
"""

from typing import List, Dict, Any

import torch
import torch.nn as nn

from models.nn_utils import get_activation, ConvBlock, DeconvBlock


# ============================================================
# Convolutional VAE
# ============================================================

class ConvVAE(nn.Module):
    """
    Convolutional VAE for XY spin configurations.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    lattice_size : int
        Spatial lattice size L (assumes square input L×L).
    encoder_channels : List[int]
        Output channels for each encoder ConvBlock.
    decoder_channels : List[int]
        Output channels for each decoder DeconvBlock.
    latent_dim : int
        Dimensionality of latent variable z.
    use_batchnorm : bool
        Whether to use BatchNorm.
    activation_name : str
        Activation function name.
    """

    def __init__(
        self,
        in_channels: int,
        lattice_size: int,
        encoder_channels: List[int],
        decoder_channels: List[int],
        latent_dim: int,
        use_batchnorm: bool = True,
        activation_name: str = "relu",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.lattice_size = lattice_size
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.latent_dim = latent_dim
        self.use_batchnorm = use_batchnorm

        act = get_activation(activation_name)

        # -------------------------
        # Encoder
        # -------------------------
        enc_layers: List[nn.Module] = []
        prev_ch = in_channels
        h = lattice_size
        w = lattice_size

        for ch in encoder_channels:
            enc_layers.append(
                ConvBlock(
                    in_ch=prev_ch,
                    out_ch=ch,
                    use_batchnorm=use_batchnorm,
                    activation=act,
                )
            )
            prev_ch = ch
            h = h // 2
            w = w // 2

        self.encoder = nn.Sequential(*enc_layers)

        if h <= 0 or w <= 0:
            raise ValueError(
                f"Too many encoder downsampling layers. Final feature map size = ({h}, {w})"
            )

        self.enc_out_h = h
        self.enc_out_w = w
        self.enc_out_channels = prev_ch
        enc_out_dim = prev_ch * h * w

        # Latent mean and log-variance
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # Decoder projection
        self.fc_dec = nn.Linear(latent_dim, enc_out_dim)

        # -------------------------
        # Decoder
        # -------------------------
        dec_layers: List[nn.Module] = []
        prev_ch = self.enc_out_channels

        for ch in decoder_channels:
            dec_layers.append(
                DeconvBlock(
                    in_ch=prev_ch,
                    out_ch=ch,
                    use_batchnorm=use_batchnorm,
                    activation=act,
                    is_last=False,
                )
            )
            prev_ch = ch

        # Final layer restoring original channel count
        dec_layers.append(
            DeconvBlock(
                in_ch=prev_ch,
                out_ch=in_channels,
                use_batchnorm=False,
                activation=None,
                is_last=True,
            )
        )

        self.decoder = nn.Sequential(*dec_layers)

    # -------------------------
    # VAE operations
    # -------------------------

    def encode(self, x: torch.Tensor):
        """
        Encode input into latent parameters (mu, logvar).
        """
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick:
            z = mu + eps * sigma
            where sigma = exp(0.5 * logvar)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent variable z into reconstructed image.
        """
        h_flat = self.fc_dec(z)
        h = h_flat.view(
            z.size(0),
            self.enc_out_channels,
            self.enc_out_h,
            self.enc_out_w,
        )
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x: torch.Tensor):
        """
        Forward pass during training.

        Returns
        -------
        x_recon : reconstructed input
        mu      : latent mean
        logvar  : latent log-variance
        z       : sampled latent variable
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


# ============================================================
# Builder from config
# ============================================================

def build_vae_from_config(
    cfg_model: Dict[str, Any],
    cfg_data: Dict[str, Any],
) -> ConvVAE:
    """
    Construct ConvVAE from YAML config dictionaries.

    Expected structure:
        cfg_model = config["model"]
        cfg_data  = config["data"]
    """
    in_channels = int(cfg_data.get("in_channels", 1))
    lattice_size = int(cfg_data.get("lattice_size", 32))

    encoder_channels: List[int] = list(cfg_model["encoder_channels"])
    decoder_channels: List[int] = list(cfg_model["decoder_channels"])
    latent_dim = int(cfg_model["latent_dim"])

    use_batchnorm = bool(cfg_model.get("use_batchnorm", True))
    activation_name = str(cfg_model.get("activation", "relu"))

    model = ConvVAE(
        in_channels=in_channels,
        lattice_size=lattice_size,
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        latent_dim=latent_dim,
        use_batchnorm=use_batchnorm,
        activation_name=activation_name,
    )

    return model


"""
Usage example (training script):

import yaml
from models.vae import build_vae_from_config

with open("config/vae.yaml") as f:
    cfg = yaml.safe_load(f)

model = build_vae_from_config(cfg["model"], cfg["data"])
model.to(device)

# Training step:
x_recon, mu, logvar, z = model(x)

recon_loss = F.mse_loss(x_recon, x, reduction="mean")
kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

total_loss = recon_weight * recon_loss + beta * kl_loss
"""
