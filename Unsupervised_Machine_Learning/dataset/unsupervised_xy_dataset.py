import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any

# Add near your existing imports
from dataset.helicity_modulus_targets import HelicityTargetConfig, HelicityTargetProvider


def cos_sin_encoding(phi: np.ndarray) -> np.ndarray:
    """
    Convert angle field into (cos φ, sin φ) channels.

    Parameters
    ----------
    phi : np.ndarray of shape (C, L, L)
        - If C=1: assumes raw angles φ (radians) and converts to 2-channel (cos φ, sin φ).
        - If C=2: assumes already (cos, sin) and returns as-is.

    Returns
    -------
    np.ndarray of shape (2, L, L), dtype float32
    """
    if phi.shape[0] == 1:
        theta = phi[0]  # (L, L)
        cos_ = np.cos(theta)
        sin_ = np.sin(theta)
        out = np.stack([cos_, sin_], axis=0)  # (2, L, L)
        return out.astype(np.float32)
    elif phi.shape[0] == 2:
        return phi.astype(np.float32)
    else:
        raise ValueError(f"cos_sin_encoding: unexpected channel count C={phi.shape[0]}")


# ============================================================
# Base: load φ(T) samples from .npz
# ============================================================

class XYSpinBaseDataset(Dataset):
    """
    Base Dataset for XY model spin configurations.

    This dataset loads:
      - spin configurations φ from npz[spin_key]
      - temperatures T (optional) from npz[temp_key]

    Supported input shapes for φ inside the .npz:
      - (N, L, L)       -> normalized to (N, 1, L, L)
      - (N, 1, L, L)    -> kept as-is
      - (N, L, L, 1)    -> normalized to (N, 1, L, L) (channels-last support)
      - (N, 2, L, L) or (N, L, L, 2) also supported (for cos/sin pre-encoded inputs)

    Reproducibility note
    --------------------
    This class itself is deterministic. Randomness only appears in the contrastive
    dataset augmentations (see XYSpinContrastiveDataset). Seed control should be done
    by the training script (numpy/python/torch seeds, and DataLoader worker seeding).
    """

    def __init__(
        self,
        npz_path: str,
        spin_key: str = "phi",
        temp_key: str = "T",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        data = np.load(npz_path)

        if spin_key not in data:
            raise KeyError(f"spin_key '{spin_key}' not found in the npz file")
        self._phi = data[spin_key]  # numpy array

        if temp_key in data:
            self._T = data[temp_key]
        else:
            # Allow datasets without temperature
            self._T = None

        # Normalize to (N, C, L, L)
        self._phi = self._normalize_shape(self._phi)

        self.dtype = dtype

    @staticmethod
    def _normalize_shape(phi: np.ndarray) -> np.ndarray:
        """
        Normalize φ array into (N, C, L, L).

        Accepts:
          - (N, L, L)
          - (N, C, L, L)
          - (N, L, L, C)
        where C in {1, 2}.
        """
        if phi.ndim == 3:
            # (N, L, L) -> (N, 1, L, L)
            phi = phi[:, None, :, :]
        elif phi.ndim == 4:
            # Could be (N, C, L, L) or (N, L, L, C)
            if phi.shape[1] in (1, 2):
                # Treat as channels-first (N, C, L, L)
                pass
            elif phi.shape[-1] in (1, 2):
                # (N, L, L, C) -> (N, C, L, L)
                phi = np.moveaxis(phi, -1, 1)
            else:
                raise ValueError(f"Unexpected shape for phi: {phi.shape}")
        else:
            raise ValueError(f"Unexpected ndim for phi: {phi.ndim}")

        return np.float32(phi)

    def __len__(self) -> int:
        return self._phi.shape[0]

    def _get_item_np(self, idx: int) -> Dict[str, Any]:
        """
        Return raw numpy sample.

        Returns
        -------
        dict:
          - "phi": np.ndarray (C, L, L)
          - "T": float or None
        """
        phi = self._phi[idx]  # (C, L, L)
        T = self._T[idx] if self._T is not None else None
        return {"phi": phi, "T": T}


# ============================================================
# Autoencoder / VAE: single-view dataset
# ============================================================

class XYSpinDataset(XYSpinBaseDataset):
    """
    Dataset for Autoencoder / VAE training.

    __getitem__ returns:
      - phi_tensor                      (if temperature is not available)
      - (phi_tensor, T_tensor)          (if temperature exists)

    Notes
    -----
    If you want cos/sin encoding for AE/VAE, pass `transform=cos_sin_encoding`.
    """

    def __init__(
        self,
        npz_path: str,
        spin_key: str = "phi",
        temp_key: str = "T",
        dtype: torch.dtype = torch.float32,
        transform: Optional[Any] = None,
    ) -> None:
        super().__init__(npz_path, spin_key=spin_key, temp_key=temp_key, dtype=dtype)
        self.transform = transform

    def __getitem__(self, idx: int):
        sample = self._get_item_np(idx)
        phi = sample["phi"]  # (C, L, L) numpy
        T = sample["T"]      # scalar or None

        if self.transform is not None:
            phi = self.transform(phi)

        phi_tensor = torch.as_tensor(phi, dtype=self.dtype)

        if T is None:
            return phi_tensor
        else:
            T_tensor = torch.as_tensor(T, dtype=torch.float32)
            return phi_tensor, T_tensor


# ============================================================
# Contrastive (SimCLR-style): two-view dataset
# ============================================================

class XYSpinContrastiveDataset(XYSpinBaseDataset):
    """
    Dataset for SimCLR-style contrastive learning.

    __getitem__ returns one of:
      - (view1, view2)                           if T is missing or return_T=False and no helicity
      - (view1, view2, helicity)                 if return_T=False and helicity is enabled
      - (view1, view2, T)                        if return_T=True and no helicity
      - (view1, view2, helicity, T)              if return_T=True and helicity is enabled

    Augmentations are applied on φ (angle field) and then optionally converted
    to (cos φ, sin φ).

    Reproducibility note
    --------------------
    Augmentations use numpy random calls. To reproduce *exact* views, you must control:
      - numpy random seed
      - DataLoader worker seeds (worker_init_fn) if num_workers > 0
      - shuffle order (DataLoader shuffle + generator seed)

    This dataset intentionally does not set seeds internally (to avoid hidden global state).
    """

    def __init__(
        self,
        npz_path: str,
        spin_key: str = "phi",
        temp_key: str = "T",
        dtype: torch.dtype = torch.float32,
        augment_config: Optional[Dict[str, Any]] = None,
        use_cos_sin_encoding: bool = False,
        helicity_provider: Optional[HelicityTargetProvider] = None,
        return_T: bool = True,
    ) -> None:
        super().__init__(
            npz_path=npz_path,
            spin_key=spin_key,
            temp_key=temp_key,
            dtype=dtype,
        )

        self.augment_config = augment_config or {}
        self.dtype = dtype
        self.use_cos_sin_encoding = use_cos_sin_encoding
        self.helicity_provider = helicity_provider
        self.return_T = return_T

    # ----------------- augmentation helpers -----------------

    def _global_rotation(self, phi: np.ndarray) -> np.ndarray:
        """
        Global U(1) rotation: φ -> (φ + θ) mod 2π.

        phi: (C, L, L)
        """
        if not self.augment_config.get("global_rotation", False):
            return phi

        theta = np.float32(np.random.uniform(0.0, 2.0 * np.pi))
        phi_rot = phi + theta
        phi_rot = np.mod(phi_rot, 2.0 * np.pi)
        return phi_rot

    def _spatial_flip(self, phi: np.ndarray) -> np.ndarray:
        """
        Random spatial flips along x/y axes.
        """
        if not self.augment_config.get("spatial_flip", False):
            return phi

        if np.random.rand() < 0.5:
            phi = np.flip(phi, axis=-1)  # left-right
        if np.random.rand() < 0.5:
            phi = np.flip(phi, axis=-2)  # up-down
        return phi

    def _spatial_rotate90(self, phi: np.ndarray) -> np.ndarray:
        """
        Random rotation by k*90 degrees (k in {0,1,2,3}).
        """
        if not self.augment_config.get("spatial_rotate90", False):
            return phi

        k = np.random.randint(0, 4)
        if k > 0:
            phi = np.rot90(phi, k=k, axes=(-2, -1))
        return phi

    def _gaussian_noise(self, phi: np.ndarray) -> np.ndarray:
        """
        Add small Gaussian noise to φ, then wrap to [0, 2π).

        Note: This is not a symmetry operation; use with care and document it in experiments.
        """
        noise_cfg = self.augment_config.get("gaussian_noise", {})
        if not noise_cfg or not noise_cfg.get("enable", False):
            return phi

        sigma = float(noise_cfg.get("sigma", 0.03))
        noise = np.float32(np.random.normal(loc=0.0, scale=sigma, size=phi.shape))
        phi_noisy = phi + noise
        phi_noisy = np.mod(phi_noisy, 2.0 * np.pi)
        return phi_noisy

    def _apply_augmentations(self, phi: np.ndarray) -> np.ndarray:
        """
        Apply a sequence of augmentations specified by augment_config.
        """
        aug = phi.copy()
        aug = self._global_rotation(aug)
        aug = self._spatial_flip(aug)
        aug = self._spatial_rotate90(aug)
        aug = self._gaussian_noise(aug)
        return aug

    # ----------------- Dataset interface --------------------

    def __getitem__(self, idx: int):
        sample = self._get_item_np(idx)
        phi = sample["phi"]  # (C, L, L)
        T = sample["T"]

        # Create two independent views
        view1 = self._apply_augmentations(phi)
        view2 = self._apply_augmentations(phi)

        if self.use_cos_sin_encoding:
            view1 = cos_sin_encoding(view1)
            view2 = cos_sin_encoding(view2)

        view1 = np.ascontiguousarray(view1)
        view2 = np.ascontiguousarray(view2)

        view1_tensor = torch.as_tensor(view1, dtype=self.dtype)
        view2_tensor = torch.as_tensor(view2, dtype=self.dtype)

        helicity_tensor = None
        if self.helicity_provider is not None:
            # Pass temperature when available (needed for by_temperature mode)
            h = self.helicity_provider.get(idx, float(T) if T is not None else None)
            helicity_tensor = torch.as_tensor(h, dtype=torch.float32)

        if (T is None) or (not self.return_T):
            if helicity_tensor is None:
                return view1_tensor, view2_tensor
            else:
                return view1_tensor, view2_tensor, helicity_tensor
        else:
            T_tensor = torch.as_tensor(T, dtype=torch.float32)
            if helicity_tensor is None:
                return view1_tensor, view2_tensor, T_tensor
            else:
                return view1_tensor, view2_tensor, helicity_tensor, T_tensor


# ============================================================
# Factory: create Dataset from config dict
# ============================================================

def create_xy_dataset_from_config(
    cfg_data: Dict[str, Any],
    mode: str = "ae",
    augment_config: Optional[Dict[str, Any]] = None,
) -> Dataset:
    """
    Create a dataset instance from a parsed config dict.

    Parameters
    ----------
    cfg_data:
        Usually cfg["data"] from YAML.
    mode:
        - "ae" or "vae" for single-view dataset
        - "contrastive" for two-view dataset
    augment_config:
        Usually cfg.get("augment", {}) for contrastive mode.

    Returns
    -------
    torch.utils.data.Dataset
    """
    npz_path = cfg_data["npz_path"]
    spin_key = cfg_data.get("spin_key", "phi")
    temp_key = cfg_data.get("temp_key", "T")

    use_cos_sin = bool(cfg_data.get("use_cos_sin_encoding", False))

    helicity_cfg = cfg_data.get("helicity", {}) or {}
    helicity_enable = bool(helicity_cfg.get("enable", False))

    helicity_provider = None
    if helicity_enable:
        targets_path = helicity_cfg.get("targets_path", npz_path)

        hcfg = HelicityTargetConfig(
            targets_path=str(targets_path),
            mode=str(helicity_cfg.get("mode", "by_index")),
            target_key=str(helicity_cfg.get("target_key", "Y")),
            temp_key=str(helicity_cfg.get("temp_key", temp_key)),
            tol=float(helicity_cfg.get("tol", 1e-6)),
            quantize_decimals=int(helicity_cfg.get("quantize_decimals", 6)),
            reduce=str(helicity_cfg.get("reduce", "mean")),
        )
        helicity_provider = HelicityTargetProvider(hcfg)

    if mode in ("ae", "vae"):
        # For AE/VAE: apply transform at sample-level
        transform = cos_sin_encoding if use_cos_sin else None
        return XYSpinDataset(
            npz_path=npz_path,
            spin_key=spin_key,
            temp_key=temp_key,
            transform=transform,
        )

    if mode == "contrastive":
        # For contrastive: augment on φ then optionally convert to cos/sin
        return XYSpinContrastiveDataset(
            npz_path=npz_path,
            spin_key=spin_key,
            temp_key=temp_key,
            augment_config=augment_config,
            use_cos_sin_encoding=use_cos_sin,
            helicity_provider=helicity_provider,
            return_T=True,
        )

    raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# Physics utilities (used in analysis / diagnostics)
# ============================================================

def wrap_angle(dtheta: np.ndarray) -> np.ndarray:
    """Map angle differences to (-pi, pi]."""
    return (dtheta + np.pi) % (2 * np.pi) - np.pi


def vortex_density_from_phi(phi: np.ndarray) -> float:
    """
    Compute vortex density from an angle field.

    Parameters
    ----------
    phi : np.ndarray of shape (L, L)
        Angle field in radians.

    Returns
    -------
    float
        Vortex density defined as mean(|q|) over plaquettes,
        where q is the integer topological charge per plaquette.
    """
    # Forward differences with periodic boundary conditions
    dphi_x = wrap_angle(np.roll(phi, -1, axis=0) - phi)  # (i+1,j) - (i,j)
    dphi_y = wrap_angle(np.roll(phi, -1, axis=1) - phi)  # (i,j+1) - (i,j)

    # Plaquette circulation:
    # (i,j)->(i+1,j): dphi_x
    # (i+1,j)->(i+1,j+1): dphi_y at (i+1,j)
    # (i+1,j+1)->(i,j+1): -dphi_x at (i,j+1)
    # (i,j+1)->(i,j): -dphi_y
    dphi_y_x = np.roll(dphi_y, -1, axis=0)  # dphi_y at (i+1,j)
    dphi_x_y = np.roll(dphi_x, -1, axis=1)  # dphi_x at (i,j+1)

    circulation = wrap_angle(dphi_x + dphi_y_x - dphi_x_y - dphi_y)

    # Topological charge per plaquette: q = round(circ / 2π)
    q = np.rint(circulation / (2 * np.pi)).astype(np.int32)

    rho_v = np.mean(np.abs(q))
    return float(rho_v)


"""
Usage examples
--------------
# Autoencoder / VAE
from dataset.unsupervised_xy_dataset import create_xy_dataset_from_config
import yaml

cfg = yaml.safe_load(open("config/autoencoder.yaml"))
train_ds = create_xy_dataset_from_config(cfg["data"], mode="ae")

# Contrastive
cfg = yaml.safe_load(open("config/contrastive.yaml"))
train_ds = create_xy_dataset_from_config(
    cfg["data"],
    mode="contrastive",
    augment_config=cfg.get("augment", {})
)
"""
