"""
helicity_modulus_targets.py
--------------------------------------------------

Helicity modulus (Υ) target provider for contrastive learning.

Design goals
------------
- Decouple physical observable (Υ) from dataset indexing.
- Support multiple file formats (.npz, .npy, .csv, .json).
- Provide stable temperature matching via quantization + tolerance.
- Explicit and safe fallback behavior.

Modes
-----
1) by_index
   Direct lookup: Y[idx]
   Recommended when NPZ contains per-sample targets.

2) by_temperature
   Construct Υ(T) lookup table and return Y(T).
   Useful when targets are aggregated by temperature.

All outputs are returned as Python float.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Sequence, Union
import numpy as np
import warnings

ArrayLike = Union[int, Sequence[int], np.ndarray]


@dataclass(frozen=True)
class HelicityTargetConfig:
    targets_path: str
    mode: str = "by_index"

    target_key: str = "Y"
    temp_key: str = "T"

    tol: float = 1e-6
    quantize_decimals: int = 6
    reduce: str = "mean"


class HelicityTargetProvider:
    """
    Provider for helicity modulus targets.

    Public API
    ----------
    get(idx, T=None) -> float
    get_many(indices, T=None) -> np.ndarray
    """

    def __init__(self, cfg: HelicityTargetConfig):
        self.cfg = cfg
        self.mode = cfg.mode.lower()

        if self.mode not in ("by_index", "by_temperature"):
            raise ValueError("mode must be 'by_index' or 'by_temperature'")

        self._targets = None
        self._T_per_sample = None
        self._table_T = None
        self._table_Y = None
        self._map_quant = None

        self._load()

    # ==========================================================
    # Public API
    # ==========================================================

    def get(self, idx: int, T: Optional[float] = None) -> float:
        if self.mode == "by_index":
            return float(self._targets[idx])

        t_val = T if T is not None else self._T_per_sample[idx]
        return float(self._lookup_temperature(float(t_val)))

    def get_many(
        self,
        indices: ArrayLike,
        T: Optional[Union[Sequence[float], np.ndarray]] = None
    ) -> np.ndarray:

        idx_arr = np.asarray(indices, dtype=np.int64).reshape(-1)

        if self.mode == "by_index":
            return self._targets[idx_arr].astype(np.float32)

        if T is not None:
            T_arr = np.asarray(T, dtype=np.float64)
        else:
            if self._T_per_sample is None:
                raise ValueError("Temperature required in by_temperature mode.")
            T_arr = self._T_per_sample[idx_arr]

        out = np.empty(len(idx_arr), dtype=np.float32)
        for i, t_val in enumerate(T_arr):
            out[i] = self._lookup_temperature(float(t_val))

        return out

    # ==========================================================
    # Loading
    # ==========================================================

    def _load(self):
        path = self.cfg.targets_path

        if path.endswith(".npz"):
            self._load_npz(path)
        elif path.endswith(".npy"):
            self._targets = np.load(path).astype(np.float32)
        else:
            raise ValueError("Only .npz and .npy supported in minimal version.")

        if self.mode == "by_temperature":
            self._build_table()

    def _load_npz(self, path: str):
        data = np.load(path)

        if self.cfg.target_key not in data:
            raise KeyError(f"{self.cfg.target_key} not found in npz file.")

        self._targets = data[self.cfg.target_key].astype(np.float32)

        if self.cfg.temp_key in data:
            self._T_per_sample = data[self.cfg.temp_key].astype(np.float64)

    # ==========================================================
    # Temperature lookup
    # ==========================================================

    def _build_table(self):
        if self._T_per_sample is None:
            raise ValueError("Temperature array required for by_temperature mode.")

        T = self._T_per_sample
        Y = self._targets

        uniq_T = np.unique(T)
        table_Y = []

        for t0 in uniq_T:
            mask = np.isclose(T, t0, atol=self.cfg.tol)
            if self.cfg.reduce == "median":
                table_Y.append(np.median(Y[mask]))
            else:
                table_Y.append(np.mean(Y[mask]))

        self._table_T = uniq_T
        self._table_Y = np.array(table_Y)

        q = self.cfg.quantize_decimals
        self._map_quant = {
            round(float(t), q): float(y)
            for t, y in zip(self._table_T, self._table_Y)
        }

    def _lookup_temperature(self, T: float) -> float:
        qT = round(T, self.cfg.quantize_decimals)

        if qT in self._map_quant:
            return self._map_quant[qT]

        diffs = np.abs(self._table_T - T)
        j = int(np.argmin(diffs))

        if diffs[j] > self.cfg.tol:
            warnings.warn(
                f"Temperature {T} not matched within tolerance. "
                f"Using nearest T={self._table_T[j]}",
                RuntimeWarning,
            )

        return float(self._table_Y[j])


prov = HelicityTargetProvider.from_npz("ml_xy_L32_1000samples.npz", mode="by_index", target_key="Y")
y = prov.get(raw_idx)

"""
