"""
unsupervised_xy_dataset.py
--------------------------------------------------

Dataset classes for XY model spin configurations.

Supports:
- Autoencoder / VAE mode (single view)
- Contrastive learning mode (two augmented views)
- Optional helicity modulus targets

All augmentations preserve XY model symmetries.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any

from dataset.helicity_modulus_targets import (
    HelicityTargetProvider,
    HelicityTargetConfig,
)


# ==========================================================
# Utility: cos-sin encoding
# ==========================================================

def cos_sin_encoding(phi: np.ndarray) -> np.ndarray:
    """
    Convert angle field (1, L, L) -> (2, L, L) using (cos φ, sin φ).
    """
    if phi.shape[0] == 1:
        theta = phi[0]
        return np.stack([np.cos(theta), np.sin(theta)], axis=0).astype(np.float32)
    elif phi.shape[0] == 2:
        return phi.astype(np.float32)
    else:
        raise ValueError("Unexpected channel dimension.")


# ==========================================================
# Base Dataset
# ==========================================================

class XYSpinBaseDataset(Dataset):
    """
    Base dataset for XY model configurations.
    """

    def __init__(self, npz_path: str, spin_key="phi", temp_key="T"):

        data = np.load(npz_path)

        self.phi = self._normalize_shape(data[spin_key])
        self.T = data[temp_key] if temp_key in data else None

    def _normalize_shape(self, phi: np.ndarray) -> np.ndarray:
        """
        Normalize shape to (N, C, L, L).
        """
        if phi.ndim == 3:
            phi = phi[:, None]
        elif phi.ndim == 4 and phi.shape[-1] in (1, 2):
            phi = np.moveaxis(phi, -1, 1)

        return phi.astype(np.float32)

    def __len__(self):
        return len(self.phi)

    def _get_raw(self, idx: int):
        return self.phi[idx], (None if self.T is None else self.T[idx])


# ==========================================================
# Contrastive Dataset
# ==========================================================

class XYSpinContrastiveDataset(XYSpinBaseDataset):

    def __init__(
        self,
        npz_path: str,
        augment_config: Optional[Dict[str, Any]] = None,
        use_cos_sin_encoding: bool = False,
        helicity_provider: Optional[HelicityTargetProvider] = None,
    ):
        super().__init__(npz_path)
        self.augment_config = augment_config or {}
        self.use_cos_sin_encoding = use_cos_sin_encoding
        self.helicity_provider = helicity_provider

    # ------------------------------------------------------
    # Augmentations
    # ------------------------------------------------------

    def _apply_augmentations(self, phi):

        aug = phi.copy()

        if self.augment_config.get("global_rotation", False):
            theta = np.random.uniform(0, 2 * np.pi)
            aug = np.mod(aug + theta, 2 * np.pi)

        if self.augment_config.get("spatial_flip", False):
            if np.random.rand() < 0.5:
                aug = np.flip(aug, axis=-1)
            if np.random.rand() < 0.5:
                aug = np.flip(aug, axis=-2)

        if self.augment_config.get("spatial_rotate90", False):
            k = np.random.randint(0, 4)
            aug = np.rot90(aug, k, axes=(-2, -1))

        return aug

    # ------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------

    def __getitem__(self, idx):

        phi, T = self._get_raw(idx)

        view1 = self._apply_augmentations(phi)
        view2 = self._apply_augmentations(phi)

        if self.use_cos_sin_encoding:
            view1 = cos_sin_encoding(view1)
            view2 = cos_sin_encoding(view2)

        out = {
            "view1": torch.tensor(view1, dtype=torch.float32),
            "view2": torch.tensor(view2, dtype=torch.float32),
        }

        if T is not None:
            out["T"] = torch.tensor(T, dtype=torch.float32)

        if self.helicity_provider is not None:
            h = self.helicity_provider.get(idx, float(T) if T is not None else None)
            out["helicity"] = torch.tensor(h, dtype=torch.float32)

        return out
