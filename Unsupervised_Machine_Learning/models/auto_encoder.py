"""
autoencoder.py

Convolutional Autoencoder for 2D XY spin configurations (phi arrays).

Architecture:
    Input:  (B, C_in, L, L)
    Encoder:
        ConvBlock stack (stride=2 downsampling)
        → flatten
        → Linear projection to latent vector z (latent_dim)
    Decoder:
        Linear projection back to flattened feature map
        → reshape
        → DeconvBlock stack (stride=2 upsampling)
        → reconstructed output

Designed for:
    - Learning compact latent representations of XY configurations
    - Baseline comparisons vs VAE and contrastive/helicity-aware models
    - Downstream analysis: latent vs temperature, clustering, correlations
"""

from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Small utilities
# ============================================================

def get_activation(name: str) -> nn.Module:
    """
    Return a PyTorch activation module from a config string.

    Expected values (case-insensitive):
        - "relu"
        - "leaky_relu"
        - "elu"
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif name == "elu":
        return nn.ELU(inplace=True)
    else:
        raise ValueError(f"Unknown activation: {name}")


class ConvBlock(nn.Module):
    """
    Basic encoder block:
        Conv2d → (BatchNorm2d) → Activation

    Uses kernel_size=4, stride=2, padding=1, which halves spatial resolution.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            use_batchnorm: bool = True,
            activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        if activation is not None:
            layers.append(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """
    Basic decoder block:
        ConvTranspose2d → (BatchNorm2d) → Activation

    Uses kernel_size=4, stride=2, padding=1, which doubles spatial resolution.

    Notes
    -----
    The final layer typically omits BatchNorm/Activation to keep output scaling free.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_batchnorm: bool = True,
        activation: Optional[nn.Module] = None,
        is_last: bool = False,
    ) -> None:
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        ]

        if not is_last:
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            if activation is not None:
                layers.append(activation)

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================
# Convolutional Autoencoder
# ============================================================

class ConvAutoEncoder(nn.Module):
    """
    Convolutional Autoencoder for XY-model phi arrays.

    - Input  : (B, C_in, L, L)
    - Encoder: ConvBlock stack reduces spatial size to L / 2^n
    - Latent : Linear maps flattened feature map -> latent_dim
    - Decoder: Linear maps latent_dim -> flattened feature map
              then DeconvBlock stack restores original resolution
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
            # With Conv(4,2,1), spatial size halves
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

        # Flatten -> latent vector
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)

        # Latent -> flattened feature map for decoder
        self.fc_dec = nn.Linear(latent_dim, enc_out_dim)

        # -------------------------
        # Decoder
        # -------------------------
        dec_layers: List[nn.Module] = []
        prev_ch = self.enc_out_channels

        # Convert channels according to decoder_channels
        for i, ch in enumerate(decoder_channels):
            # Another final layer will map to in_channels, so is_last=False here
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
                use_batchnorm=False,  # no BN/activation for output layer
                activation=None,
                is_last=True,
            )
        )

        self.decoder = nn.Sequential(*dec_layers)

    # -------------------------
    # Interface
    # -------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input x to latent vector z.
        """
        h = self.encoder(x)  # (B, C_enc, H, W)
        h_flat = h.view(h.size(0), -1)
        z = self.fc_mu(h_flat)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector z into reconstructed output.
        """
        h_flat = self.fc_dec(z)  # (B, C_enc * H * W)
        h = h_flat.view(
            z.size(0),
            self.enc_out_channels,
            self.enc_out_h,
            self.enc_out_w,
        )
        x_recon = self.decoder(h)  # (B, in_channels, L, L)
        return x_recon

    def forward(self, x: torch.Tensor):
        """
        Forward pass for training.

        Returns
        -------
        x_recon : reconstructed input
        z       : latent vector
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


# ============================================================
# Builder from config
# ============================================================

def build_autoencoder_from_config(
    cfg_model: Dict[str, Any],
    cfg_data: Dict[str, Any],
) -> ConvAutoEncoder:
    """
    Construct ConvAutoEncoder from YAML config dictionaries.

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

    model = ConvAutoEncoder(
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

In train/train_autoencoder.py

import yaml
from models.autoencoder import build_autoencoder_from_config

with open("config/autoencoder.yaml") as f:
    cfg = yaml.safe_load(f)

model = build_autoencoder_from_config(cfg["model"], cfg["data"])
model.to(device)
"""
