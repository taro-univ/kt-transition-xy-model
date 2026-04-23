"""
nn_utils.py

Shared neural network building blocks for XY model representation learning.

Provides
--------
get_activation : factory returning a PyTorch activation module from a config string
ConvBlock      : Conv2d(stride=2) → (BN) → Act   — halves spatial resolution
DeconvBlock    : ConvTranspose2d(stride=2) → (BN) → Act — doubles spatial resolution

Notes
-----
- ConvBlock uses padding=1 for both kernel_size=4 (AE/VAE) and kernel_size=3 (contrastive).
  With stride=2 this halves any even spatial dimension in both cases.
- DeconvBlock fixes kernel_size=4, padding=1 (AE/VAE decoder convention).
- The helicity head has its own private _get_activation because it is intentionally
  kept self-contained; that copy is not consolidated here.

Used by
-------
models/auto_encoder.py
models/vae.py
models/contrastive_encoder.py
"""

from typing import Optional

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """
    Return a PyTorch activation module from a config string.

    Parameters
    ----------
    name : str
        Case-insensitive name. Supported: "relu", "leaky_relu", "elu".
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unknown activation: {name!r}. Choose from relu / leaky_relu / elu.")


class ConvBlock(nn.Module):
    """
    Strided convolutional block:
        Conv2d(kernel_size, stride=2, padding=1) → (BatchNorm2d) → Activation

    Halves the spatial resolution for any even input size.

    Parameters
    ----------
    kernel_size : int
        Kernel size for the convolution.
        Use 4 for AE / VAE encoders (default).
        Use 3 for the contrastive encoder.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_batchnorm: bool = True,
        activation: Optional[nn.Module] = None,
        kernel_size: int = 4,
    ) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=1)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        if activation is not None:
            layers.append(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """
    Transposed convolutional block:
        ConvTranspose2d(kernel_size=4, stride=2, padding=1) → (BatchNorm2d) → Activation

    Doubles the spatial resolution.

    Parameters
    ----------
    is_last : bool
        If True, BatchNorm and Activation are omitted (output layer convention).
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
        layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)]
        if not is_last:
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            if activation is not None:
                layers.append(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
