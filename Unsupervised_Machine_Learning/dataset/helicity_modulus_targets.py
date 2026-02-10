# dataset/helicity_modulus_targets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union
import warnings

import numpy as np

ArrayLike = Union[int, Sequence[int], np.ndarray]


@dataclass(frozen=True)
class HelicityTargetConfig:
    """
    Configuration for providing helicity modulus targets (Υ).

    Modes
    -----
    - "by_index":
        Return targets[idx]. This is the safest option when the target array (N,)
        is aligned with the spin samples (phi) in the same order.

    - "by_temperature":
        Return Υ(T) keyed by temperature. Floating-point matching is tricky, so this
        mode uses quantization (rounding) and tolerance-based nearest matching.

    Supported file formats
    ----------------------
    - .npz:
        Loaded via numpy.load. Target array is read from `target_key`.
        Optionally, per-sample temperatures can be read from `temp_key`.

        Also supports (optional) precomputed table keys:
          - "temps_unique" and "Y_by_temp" (both 1D, same length)

    - .npy:
        1D array of targets (N,)

    - .csv:
        Either:
          - 1 column: Y only (treated as by_index targets)
          - 2 columns: (T, Y) table (treated as by_temperature table)

    - .json:
        Either:
          - {"T": [...], "Y": [...]} or {"T": [...], "<target_key>": [...]}
          - {"0.9": 0.123, ...} mapping temperature->Y
          - [Y0, Y1, ...] as by_index targets

    Table construction (by_temperature)
    ----------------------------------
    - If a (T_table, Y_table) is explicitly provided (csv/json/npz table), use it.
    - Else, if per-sample (T, Y) exist in an npz, aggregate Y by unique T to build a table.
    """

    targets_path: str
    mode: str = "by_index"

    # Keys for .npz
    target_key: str = "Y"
    temp_key: str = "T"

    # Matching for by_temperature
    tol: float = 1e-6            # accept |T - T_k| <= tol as a match
    quantize_decimals: int = 6   # use round(T, quantize_decimals) for dict keying

    # Aggregation method when building Υ(T) from per-sample data
    reduce: str = "mean"         # "mean" or "median"


class HelicityTargetProvider:
    """
    Provider to retrieve helicity modulus (Υ) targets by dataset index or by temperature.

    - by_index:
        Load a 1D target array (N,) and return get(idx).

    - by_temperature:
        Build a (T, Y) lookup table and return get(idx, T) or get(idx) if per-sample
        temperature is available from the source.

    All returned values are Python float. Batch retrieval returns float32 numpy array.
    """

    def __init__(self, cfg: HelicityTargetConfig):
        self.cfg = cfg
        mode = cfg.mode.lower().strip()
        if mode not in ("by_index", "by_temperature"):
            raise ValueError(f"Unknown mode: {cfg.mode}. Use 'by_index' or 'by_temperature'.")
        self.mode = mode

        self._targets: Optional[np.ndarray] = None       # (N,)
        self._T_per_sample: Optional[np.ndarray] = None  # (N,)

        self._table_T: Optional[np.ndarray] = None       # (K,)
        self._table_Y: Optional[np.ndarray] = None       # (K,)
        self._map_quant: Optional[Dict[float, float]] = None

        self._load()

    # -------------------------
    # public API
    # -------------------------

    def get(self, idx: int, T: Optional[float] = None) -> float:
        """
        Get a single target.

        Parameters
        ----------
        idx:
            Raw dataset index. If you use torch.utils.data.Subset, the caller should map
            subset index back to the raw index before calling this function.
        T:
            Temperature value (needed in by_temperature mode if per-sample temperature is
            not available in the source file).

        Returns
        -------
        float:
            Helicity modulus Υ as a Python float.
        """
        if self.mode == "by_index":
            if self._targets is None:
                raise RuntimeError("Targets are not loaded.")
            if idx < 0 or idx >= self._targets.shape[0]:
                raise IndexError(f"idx out of range: {idx} (N={self._targets.shape[0]})")
            return float(self._targets[idx])

        # by_temperature
        t_val = self._resolve_temperature(idx, T)
        return float(self._lookup_by_temperature(t_val))

    def get_many(
        self,
        indices: ArrayLike,
        T: Optional[Union[Sequence[float], np.ndarray]] = None
    ) -> np.ndarray:
        """
        Get multiple targets at once.

        Returns
        -------
        np.ndarray:
            (B,) float32 array.

        Notes (by_temperature)
        ----------------------
        - If `T` is provided, it must have the same length as `indices`.
        - If `T` is not provided, per-sample temperatures must be available in the source.
        """
        idx_arr = np.asarray(indices, dtype=np.int64)
        if idx_arr.ndim != 1:
            idx_arr = idx_arr.reshape(-1)

        if self.mode == "by_index":
            if self._targets is None:
                raise RuntimeError("Targets are not loaded.")
            if idx_arr.min(initial=0) < 0 or idx_arr.max(initial=0) >= self._targets.shape[0]:
                raise IndexError("Some indices are out of range.")
            return self._targets[idx_arr].astype(np.float32)

        # by_temperature
        if T is not None:
            t_arr = np.asarray(T, dtype=np.float64).reshape(-1)
            if t_arr.shape[0] != idx_arr.shape[0]:
                raise ValueError(f"T length mismatch: len(T)={t_arr.shape[0]} vs len(indices)={idx_arr.shape[0]}")
        else:
            if self._T_per_sample is None:
                raise ValueError(
                    "T is required for by_temperature unless per-sample T is available in the source file."
                )
            t_arr = self._T_per_sample[idx_arr].astype(np.float64)

        out = np.empty((idx_arr.shape[0],), dtype=np.float32)
        for i, t_val in enumerate(t_arr):
            out[i] = float(self._lookup_by_temperature(float(t_val)))
        return out

    # -------------------------
    # loading
    # -------------------------

    def _load(self) -> None:
        path = self.cfg.targets_path
        if path.endswith(".npz"):
            self._load_from_npz(path)
        elif path.endswith(".npy"):
            self._load_from_npy(path)
        elif path.endswith(".csv"):
            self._load_from_csv(path)
        elif path.endswith(".json"):
            self._load_from_json(path)
        else:
            raise ValueError(f"Unsupported file type: {path} (expected .npz/.npy/.csv/.json)")

        # Build temperature lookup table if needed
        if self.mode == "by_temperature":
            self._build_temperature_table()

    def _load_from_npz(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)

        # 1) Try the configured target key first
        if self.cfg.target_key in data:
            y = data[self.cfg.target_key]
        else:
            # 2) Try a few common alternative names (minimal, pragmatic support)
            for alt in ("Y", "helicity", "helicity_modulus", "upsilon", "Upsilon"):
                if alt in data:
                    y = data[alt]
                    break
            else:
                raise KeyError(f"target_key '{self.cfg.target_key}' not found in npz: {list(data.files)}")

        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"Target array must be 1D (N,), got shape={y.shape}")

        self._targets = y.astype(np.float32)

        # Keep per-sample temperatures if available and aligned
        if self.cfg.temp_key in data:
            t = np.asarray(data[self.cfg.temp_key])
            if t.ndim != 1 or t.shape[0] != y.shape[0]:
                # If shape doesn't match, silently disable per-sample T usage
                self._T_per_sample = None
            else:
                self._T_per_sample = t.astype(np.float64)
        else:
            self._T_per_sample = None

        # Optional precomputed (T_table, Y_table)
        if "temps_unique" in data and "Y_by_temp" in data:
            Tu = np.asarray(data["temps_unique"], dtype=np.float64)
            Yu = np.asarray(data["Y_by_temp"], dtype=np.float64)
            if Tu.ndim == 1 and Yu.ndim == 1 and Tu.shape[0] == Yu.shape[0]:
                self._table_T = Tu
                self._table_Y = Yu

    def _load_from_npy(self, path: str) -> None:
        y = np.load(path, allow_pickle=False)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"Target array must be 1D (N,), got shape={y.shape}")
        self._targets = y.astype(np.float32)
        self._T_per_sample = None

    def _load_from_csv(self, path: str) -> None:
        """
        Lightweight CSV loader without extra dependencies.
        Supports comma-separated files. (Space-separated may also work depending on numpy parsing.)
        """
        arr = np.genfromtxt(path, delimiter=",", dtype=np.float64)

        if arr.ndim == 1:
            # 1 column: treat as by_index targets Y
            self._targets = arr.astype(np.float32)
            self._T_per_sample = None
            return

        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"CSV must have 1 col (Y) or 2 cols (T,Y). got shape={arr.shape}")

        # 2 columns: (T, Y) treated as a temperature table
        T = arr[:, 0].astype(np.float64)
        Y = arr[:, 1].astype(np.float64)

        self._targets = None
        self._T_per_sample = None
        self._table_T = T
        self._table_Y = Y

    def _load_from_json(self, path: str) -> None:
        import json

        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # Case 1: {"T": [...], "Y": [...]} or {"T": [...], "<target_key>": [...]}
        if isinstance(obj, dict) and "T" in obj and ("Y" in obj or self.cfg.target_key in obj):
            T = np.asarray(obj["T"], dtype=np.float64)
            key = "Y" if "Y" in obj else self.cfg.target_key
            Y = np.asarray(obj[key], dtype=np.float64)
            if T.ndim != 1 or Y.ndim != 1 or T.shape[0] != Y.shape[0]:
                raise ValueError("JSON arrays must be 1D and have the same length for T and Y.")
            self._targets = None
            self._T_per_sample = None
            self._table_T = T
            self._table_Y = Y
            return

        # Case 2: {"0.900000": 0.123, ...} mapping temperature->Y
        if isinstance(obj, dict):
            keys = list(obj.keys())
            try:
                T = np.asarray([float(k) for k in keys], dtype=np.float64)
            except Exception as e:
                raise ValueError("JSON dict keys must be convertible to float temperatures.") from e
            Y = np.asarray([float(obj[k]) for k in keys], dtype=np.float64)
            self._targets = None
            self._T_per_sample = None
            self._table_T = T
            self._table_Y = Y
            return

        # Case 3: [Y0, Y1, ...] as by_index targets
        if isinstance(obj, list):
            Y = np.asarray(obj, dtype=np.float64)
            if Y.ndim != 1:
                raise ValueError("JSON list must be 1D for by_index targets.")
            self._targets = Y.astype(np.float32)
            self._T_per_sample = None
            return

        raise ValueError("Unsupported JSON structure for helicity targets.")

    # -------------------------
    # by_temperature helpers
    # -------------------------

    def _build_temperature_table(self) -> None:
        """
        Priority:
          1) Use existing (T_table, Y_table) if already loaded (csv/json/npz-table).
          2) Otherwise, if per-sample (T, Y) exist, aggregate by unique T to build a table.
        """
        if self._table_T is not None and self._table_Y is not None:
            self._finalize_table(self._table_T, self._table_Y)
            return

        if self._T_per_sample is None or self._targets is None:
            raise ValueError(
                "by_temperature mode requires either (T_table, Y_table) or per-sample (T, Y) in the source."
            )

        T = self._T_per_sample.astype(np.float64)
        Y = self._targets.astype(np.float64)

        uniq = np.unique(T)
        out_T = []
        out_Y = []
        for t0 in uniq:
            mask = np.isclose(T, t0, atol=self.cfg.tol, rtol=0.0)
            yvals = Y[mask]
            if yvals.size == 0:
                continue
            if self.cfg.reduce == "median":
                y_agg = float(np.median(yvals))
            else:
                y_agg = float(np.mean(yvals))
            out_T.append(float(t0))
            out_Y.append(y_agg)

        self._finalize_table(np.asarray(out_T, dtype=np.float64), np.asarray(out_Y, dtype=np.float64))

    def _finalize_table(self, T: np.ndarray, Y: np.ndarray) -> None:
        if T.ndim != 1 or Y.ndim != 1 or T.shape[0] != Y.shape[0]:
            raise ValueError(f"Temperature table must be 1D and aligned. got T={T.shape}, Y={Y.shape}")

        # Sort table for stable lookup / reproducibility
        order = np.argsort(T)
        T = T[order]
        Y = Y[order]

        self._table_T = T.astype(np.float64)
        self._table_Y = Y.astype(np.float64)

        # Build quantized map for fast + stable matching
        qdec = int(self.cfg.quantize_decimals)
        self._map_quant = {round(float(t), qdec): float(y) for t, y in zip(self._table_T, self._table_Y)}

    def _resolve_temperature(self, idx: int, T: Optional[float]) -> float:
        if T is not None:
            return float(T)
        if self._T_per_sample is None:
            raise ValueError("Temperature T is required for by_temperature mode (no per-sample T available).")
        if idx < 0 or idx >= self._T_per_sample.shape[0]:
            raise IndexError(f"idx out of range: {idx} (N={self._T_per_sample.shape[0]})")
        return float(self._T_per_sample[idx])

    def _lookup_by_temperature(self, T: float) -> float:
        if self._map_quant is None or self._table_T is None or self._table_Y is None:
            raise RuntimeError("Temperature table is not initialized.")

        # 1) Quantized dictionary lookup (fast, robust)
        qT = round(float(T), int(self.cfg.quantize_decimals))
        if qT in self._map_quant:
            return float(self._map_quant[qT])

        # 2) Tolerance-based nearest matching
        diffs = np.abs(self._table_T - float(T))
        j = int(np.argmin(diffs))
        if float(diffs[j]) <= float(self.cfg.tol):
            return float(self._table_Y[j])

        # 3) Fallback: nearest neighbor with warning
        warnings.warn(
            f"Temperature {T} not matched within tol={self.cfg.tol}. "
            f"Using nearest T={float(self._table_T[j]):.6f}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float(self._table_Y[j])

    # -------------------------
    # convenience constructors
    # -------------------------

    @classmethod
    def from_npz(
        cls,
        npz_path: str,
        mode: str = "by_index",
        target_key: str = "Y",
        temp_key: str = "T",
        tol: float = 1e-6,
        quantize_decimals: int = 6,
        reduce: str = "mean",
    ) -> "HelicityTargetProvider":
        cfg = HelicityTargetConfig(
            targets_path=npz_path,
            mode=mode,
            target_key=target_key,
            temp_key=temp_key,
            tol=tol,
            quantize_decimals=quantize_decimals,
            reduce=reduce,
        )
        return cls(cfg)


# -------------------------
# minimal self-test (optional)
# -------------------------
if __name__ == "__main__":
    # Example:
    #   /mnt/data/ml_xy_L32_1000samples.npz has keys "Y" and "T".
    provider = HelicityTargetProvider.from_npz(
        npz_path="/mnt/data/ml_xy_L32_1000samples.npz",
        mode="by_index",
        target_key="Y",
        temp_key="T",
    )
    print("Y[0] =", provider.get(0))
    print("Y[1..5] =", provider.get_many([1, 2, 3, 4, 5]))

"""
Usage example
-------------
from dataset.helicity_modulus_targets import HelicityTargetProvider

prov = HelicityTargetProvider.from_npz(
    "ml_xy_L32_1000samples.npz",
    mode="by_index",
    target_key="Y",
)
y = prov.get(raw_idx)
"""
