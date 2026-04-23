"""
contrastive_encoder.py
--------------------------------------------------

Convolutional encoder + projector used for SimCLR-style contrastive learning.

Design philosophy
-----------------
- Encoder produces latent representation h (before projection).
- Projector maps h -> z for contrastive loss.
- Helicity head (if used) must attach to h, not z.

Reproducibility notes
---------------------
- No randomness is used inside the encoder itself.
- Determinism depends on:
    * torch backend settings
    * CuDNN deterministic flags
    * seed control in training script
"""

import torch
import torch.nn as nn
from typing import List, Optional

from models.nn_utils import get_activation, ConvBlock


# ============================================================
# Standard SimCLR Encoder
# ============================================================

class SimCLREncoder(nn.Module):
    """
    CNN Encoder + MLP Projector.

    Input:
        x : (B, C, L, L)

    Output:
        z : (B, proj_out)

    Internal representation:
        h : (B, enc_out_dim)

    Notes
    -----
    - encoder_channels defines number of ConvBlocks.
    - Each block halves spatial resolution.
    - lattice_size must be divisible by 2^num_blocks.
    """

    def __init__(
        self,
        in_channels: int,
        lattice_size: int,
        encoder_channels: List[int],
        proj_hidden: int,
        proj_out: int,
        use_batchnorm: bool,
        activation_name: str,
    ):
        super().__init__()

        act = get_activation(activation_name)

        # ---------------- Encoder ----------------
        enc_layers = []
        prev_ch = in_channels
        h = lattice_size
        w = lattice_size

        for ch in encoder_channels:
            enc_layers.append(ConvBlock(prev_ch, ch, use_batchnorm=use_batchnorm, activation=act, kernel_size=3))
            prev_ch = ch
            h //= 2
            w //= 2

        if h <= 0 or w <= 0:
            raise ValueError("Too many downsampling layers for given lattice_size.")

        self.encoder = nn.Sequential(*enc_layers)
        self.enc_out_dim = prev_ch * h * w

        # ---------------- Projector ----------------
        self.projector = nn.Sequential(
            nn.Linear(self.enc_out_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            act,
            nn.Linear(proj_hidden, proj_out),
        )

    def encode(self, x):
        """
        Return latent representation h (before projection).
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return h

    def project(self, h):
        """
        Return projected representation z.
        """
        return self.projector(h)

    def forward(self, x):
        h = self.encode(x)
        z = self.project(h)
        return z


# ============================================================
# Multi-scale SimCLR Encoder
# ============================================================

class MultiScaleSimCLREncoder(nn.Module):
    """
    Multi-scale convolutional encoder.

    Concept:
        - Apply multiple convolution stems with different kernel sizes in parallel.
        - Concatenate features along channel dimension.
        - Apply downsampling ConvBlocks.
        - Apply projector.

    encoder_channels:
        encoder_channels[0] = total channels after concatenation.
        Must be divisible by number of kernel_sizes.
    """

    def __init__(
        self,
        in_channels: int,
        lattice_size: int,
        encoder_channels: List[int],
        proj_hidden: int,
        proj_out: int,
        use_batchnorm: bool,
        activation_name: str,
        kernel_sizes: Optional[List[int]] = None,
    ):
        super().__init__()

        act = get_activation(activation_name)

        if kernel_sizes is None:
            kernel_sizes = [3, 7, 15]

        self.kernel_sizes = kernel_sizes
        num_scales = len(kernel_sizes)

        if len(encoder_channels) < 1:
            raise ValueError("encoder_channels must contain at least one element.")

        stem_out_total = encoder_channels[0]

        if stem_out_total % num_scales != 0:
            raise ValueError(
                "encoder_channels[0] must be divisible by number of scales."
            )

        base_channels = stem_out_total // num_scales

        # ---------------- Multi-scale stem ----------------
        stems = []
        for k in kernel_sizes:
            layers = [
                nn.Conv2d(
                    in_channels,
                    base_channels,
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                )
            ]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(base_channels))
            layers.append(act)
            stems.append(nn.Sequential(*layers))

        self.stems = nn.ModuleList(stems)

        # ---------------- Downsampling blocks ----------------
        enc_layers = []
        prev_ch = stem_out_total
        h = lattice_size
        w = lattice_size

        for ch in encoder_channels[1:]:
            enc_layers.append(ConvBlock(prev_ch, ch, use_batchnorm=use_batchnorm, activation=act, kernel_size=3))
            prev_ch = ch
            h //= 2
            w //= 2

        if h <= 0 or w <= 0:
            raise ValueError("Too many downsampling layers for given lattice_size.")

        self.encoder_down = nn.Sequential(*enc_layers)
        self.enc_out_dim = prev_ch * h * w

        # ---------------- Projector ----------------
        self.projector = nn.Sequential(
            nn.Linear(self.enc_out_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            act,
            nn.Linear(proj_hidden, proj_out),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = [stem(x) for stem in self.stems]
        h = torch.cat(feats, dim=1)
        h = self.encoder_down(h)
        h = h.view(h.size(0), -1)
        return h

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return self.projector(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        z = self.project(h)
        return z
