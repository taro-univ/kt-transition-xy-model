"""
analysis.py

Post-processing utilities for KT transition analysis in the 2D XY model.

This module operates on the output format produced by `simulate_temperature_sweep`
(e.g., from MCsim/loop.py). It focuses on:

- Estimating Tc from the crossing of helicity modulus Υ(T) with 2T/pi.
- Extracting effective exponents / length scales from correlation functions:
    - Low-T:  G(r) ~ r^{-eta}
    - High-T: G(r) ~ A exp(-r/xi)
- Checking KT-form scaling on the high-temperature side:
    ln xi ~ a + b / sqrt(T - Tc)
- Summarizing size dependence of crossing estimates.

Design goals
------------
- Keep dependencies minimal (numpy only).
- Be robust to missing/insufficient data (return NaN when fitting is impossible).
- Provide clear, typed, reusable functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

PI: float = float(np.pi)

__all__ = [
    "estimate_Tc_from_Y",
    "fit_powerlaw_eta",
    "fit_exponential_xi",
    "extract_eta_xi",
    "kt_fit_logxi",
    "collect_Y_vs_T_by_size",
    "crossing_Tc_by_size",
]


# -----------------------------------------------------------------------------
# 1) Tc estimation from universal jump line: Υ(T) crossing with 2T/pi
# -----------------------------------------------------------------------------
def estimate_Tc_from_Y(results: List[dict]) -> Tuple[float, dict]:
    """
    Estimate Tc from the crossing of Υ(T) with the universal jump line 2T/pi.

    Parameters
    ----------
    results : list[dict]
        Output list for a single system size L.
        Each element is expected to include:
          - res["T"]
          - res["time_series"]["Y"] (list-like)

    Returns
    -------
    Tc_est : float
        Estimated Tc from the first detected crossing (linear interpolation).
        Returns NaN if no crossing is found.
    meta : dict
        Minimal diagnostic info (T, Y, line, crossing index).
    """
    if len(results) < 2:
        return float("nan"), {"reason": "need at least 2 temperature points"}

    T = np.array([res["T"] for res in results], dtype=np.float64)
    Y = np.array([np.mean(res["time_series"]["Y"]) for res in results], dtype=np.float64)
    line = 2.0 * T / PI

    diff = Y - line
    s = diff[:-1] * diff[1:]  # sign change => crossing
    idx = np.where(s <= 0)[0]
    if len(idx) == 0:
        return float("nan"), {"T": T.tolist(), "Y": Y.tolist(), "line": line.tolist(), "reason": "no crossing"}

    k = int(idx[0])
    t1, t2 = T[k], T[k + 1]
    y1, y2 = diff[k], diff[k + 1]

    Tc = t1 + (t2 - t1) * (0.0 - y1) / (y2 - y1 + 1e-12)
    meta = {"T": T.tolist(), "Y": Y.tolist(), "line": line.tolist(), "k": k}
    return float(Tc), meta


# -----------------------------------------------------------------------------
# 2) Extract eta(T) / xi(T) from correlation function fits
# -----------------------------------------------------------------------------
def _fit_window(r: np.ndarray, G: np.ndarray, rmin: int, frac: float) -> np.ndarray:
    """Common mask for fitting windows."""
    if r.size == 0:
        return np.zeros(0, dtype=bool)
    rmax = int(frac * float(r.max()))
    mask = (r >= rmin) & (r <= rmax) & np.isfinite(G) & (G > 0)
    return mask


def fit_powerlaw_eta(r: np.ndarray, G: np.ndarray, rmin: int = 2, frac: float = 0.6) -> float:
    """
    Low-T effective exponent fit: G(r) ~ r^{-eta}.

    Performs linear regression in log-log space:
        log G = a log r + b  =>  eta = -a

    Returns NaN if insufficient valid points are available.
    """
    r = np.asarray(r, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)

    mask = _fit_window(r, G, rmin=rmin, frac=frac)
    if int(mask.sum()) < 3:
        return float("nan")

    x = np.log(r[mask])
    y = np.log(G[mask])
    a, b = np.polyfit(x, y, 1)  # y = a x + b
    return float(-a)


def fit_exponential_xi(r: np.ndarray, G: np.ndarray, rmin: int = 2, frac: float = 0.6) -> float:
    """
    High-T correlation length fit: G(r) ~ A * exp(-r/xi).

    Performs linear regression in semi-log space:
        log G = a r + b  =>  xi = -1/a  (if a < 0)

    Returns NaN if insufficient valid points are available or slope is non-negative.
    """
    r = np.asarray(r, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)

    mask = _fit_window(r, G, rmin=rmin, frac=frac)
    if int(mask.sum()) < 3:
        return float("nan")

    x = r[mask]
    y = np.log(G[mask])
    a, b = np.polyfit(x, y, 1)  # y = a x + b

    if a >= 0:
        return float("nan")
    return float(-1.0 / a)


def extract_eta_xi(results: List[dict], mode: str = "auto") -> Dict[float, Dict[str, float]]:
    """
    Fit correlation function at each temperature and return eta/xi estimates.

    Parameters
    ----------
    results : list[dict]
        Each element should include:
          - res["T"]
          - res["correlation"]["r"]
          - res["correlation"]["G"]
    mode : str
        "auto": compute both; downstream can decide usage
        "eta" : compute eta only (xi set to NaN)
        "xi"  : compute xi only (eta set to NaN)

    Returns
    -------
    dict
        {T: {"eta": eta, "xi": xi}}
    """
    if mode not in {"auto", "eta", "xi"}:
        raise ValueError("mode must be one of {'auto','eta','xi'}")

    out: Dict[float, Dict[str, float]] = {}
    for res in results:
        T = float(res["T"])
        r = np.asarray(res["correlation"]["r"], dtype=np.int64)
        G = np.asarray(res["correlation"]["G"], dtype=np.float64)

        eta = fit_powerlaw_eta(r, G)
        xi = fit_exponential_xi(r, G)

        if mode == "eta":
            xi = float("nan")
        elif mode == "xi":
            eta = float("nan")

        out[T] = {"eta": float(eta), "xi": float(xi)}
    return out


# -----------------------------------------------------------------------------
# 3) High-T KT form: ln xi ~ a + b / sqrt(T - Tc)
# -----------------------------------------------------------------------------
def kt_fit_logxi(results: List[dict], Tc_guess: float, Tmin: Optional[float] = None) -> Tuple[float, float]:
    """
    Fit the KT form on the high-temperature side using xi(T):

        ln xi ≈ a + b / sqrt(T - Tc_guess)

    Parameters
    ----------
    results : list[dict]
        Same format as in extract_eta_xi.
    Tc_guess : float
        External estimate of Tc (e.g., from estimate_Tc_from_Y).
    Tmin : float | None
        Optional lower bound for temperature (still must satisfy T > Tc_guess).

    Returns
    -------
    (b, a) : tuple[float, float]
        Linear regression parameters. Returns (NaN, NaN) if insufficient data.
    """
    Tc_guess = float(Tc_guess)
    if not np.isfinite(Tc_guess):
        return float("nan"), float("nan")

    eta_xi = extract_eta_xi(results, mode="xi")

    Xs: List[float] = []
    Ys: List[float] = []

    for res in results:
        T = float(res["T"])
        if T <= Tc_guess:
            continue
        if Tmin is not None and T < float(Tmin):
            continue

        xi = float(eta_xi[T]["xi"])
        if not np.isfinite(xi) or xi <= 0:
            continue

        Xs.append(1.0 / np.sqrt(T - Tc_guess))
        Ys.append(float(np.log(xi)))

    if len(Xs) < 3:
        return float("nan"), float("nan")

    X = np.asarray(Xs, dtype=np.float64)
    Y = np.asarray(Ys, dtype=np.float64)
    b, a = np.polyfit(X, Y, 1)  # Y = a + b X
    return float(b), float(a)


# -----------------------------------------------------------------------------
# 4) Size dependence helpers
# -----------------------------------------------------------------------------
def collect_Y_vs_T_by_size(results_by_L: Dict[int, List[dict]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Collect mean Υ(T) curves for each system size.

    Parameters
    ----------
    results_by_L : dict[int, list[dict]]
        {L: results_list}

    Returns
    -------
    dict[int, (T, Y)]
        {L: (T_array, Y_mean_array)}
    """
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for L, reslist in results_by_L.items():
        T = np.array([r["T"] for r in reslist], dtype=np.float64)
        Y = np.array([np.mean(r["time_series"]["Y"]) for r in reslist], dtype=np.float64)
        out[int(L)] = (T, Y)
    return out


def crossing_Tc_by_size(results_by_L: Dict[int, List[dict]]) -> Dict[int, float]:
    """
    Apply estimate_Tc_from_Y for each L and return Tc(L).

    Returns
    -------
    dict[int, float]
        {L: Tc_est(L)} (NaN if crossing is not found).
    """
    TcL: Dict[int, float] = {}
    for L, reslist in results_by_L.items():
        Tc, _ = estimate_Tc_from_Y(reslist)
        TcL[int(L)] = float(Tc)
    return TcL
