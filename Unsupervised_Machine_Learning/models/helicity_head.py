"""
helicity_head.py
--------------------------------------------------

Helicity modulus regression head (Υ auxiliary supervision).

Design philosophy
-----------------
- This head does NOT aim for high-precision regression.
- It injects a weak physical bias into the encoder latent space.
- The head must attach to encoder output h (NOT projector output z).

Reproducibility
---------------
- Fully deterministic given fixed torch seed.
- No internal randomness.
"""

from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class HelicityHeadConfig:
    """
    Configuration for helicity auxiliary head.

    Recommended settings:
        hidden_ratio ~ 0.25
        dropout = 0.0
        use_layernorm = False
        out_dim = 1
    """
    hidden_ratio: float = 0.25
    min_hidden: int = 16
    activation: str = "relu"
    dropout: float = 0.0
    use_layernorm: bool = False
    out_dim: int = 1


def _get_activation(name: str) -> nn.Module:
    name = str(name).lower().strip()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class HelicityModulusHead(nn.Module):
    """
    Predict helicity modulus Υ from latent representation h.

    Input:
        h : (B, D)

    Output:
        y : (B,)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        *,
        hidden_ratio: float = 0.25,
        min_hidden: int = 16,
        activation: str = "relu",
        dropout: float = 0.0,
        use_layernorm: bool = False,
        out_dim: int = 1,
    ):
        super().__init__()

        if in_dim <= 0:
            raise ValueError("in_dim must be positive.")

        if hidden_dim is None:
            hidden_dim = max(min_hidden, int(round(in_dim * hidden_ratio)))

        if out_dim != 1:
            raise ValueError("out_dim must be 1 for scalar helicity.")

        act = _get_activation(activation)

        layers = [nn.Linear(in_dim, hidden_dim)]

        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))

        layers.append(act)

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.net(h)
        return y.squeeze(-1)


def build_helicity_head_from_encoder(
    encoder: Any,
    *,
    hidden_dim: Optional[int] = None,
    hidden_ratio: float = 0.25,
    min_hidden: int = 16,
    activation: str = "relu",
    dropout: float = 0.0,
    use_layernorm: bool = False,
) -> HelicityModulusHead:
    """
    Utility function that builds helicity head
    assuming encoder has attribute `enc_out_dim`.
    """

    if not hasattr(encoder, "enc_out_dim"):
        raise AttributeError(
            "Encoder must expose attribute 'enc_out_dim'."
        )

    in_dim = int(getattr(encoder, "enc_out_dim"))

    return HelicityModulusHead(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        hidden_ratio=hidden_ratio,
        min_hidden=min_hidden,
        activation=activation,
        dropout=dropout,
        use_layernorm=use_layernorm,
        out_dim=1,
    )
