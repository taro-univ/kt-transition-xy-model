"""
measure.py

Observable measurements for the 2D XY model (KT transition).

Implemented observables
-----------------------
- Helicity modulus Υ (lattice expression)
- Direction-averaged spin correlation function G(r) accumulated into CorrBins
- Vortex detection via plaquette winding numbers (w = ±1)
- Vortex density n_v
- Opposite-charge nearest-pair distance histogram (optional diagnostic)
- Energy density and heat capacity accumulator

Design notes
------------
- This module is independent of the main simulation loop.
- It depends only on utilities (angles.py) and energy helpers (update.total_energy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple
import numpy as np

from MCsim.angles import nearest_diffs, angle_diff, winding_number
from MCsim.update import total_energy


class _StateLike(Protocol):
    """Minimal protocol for a simulation state used in measurements."""
    L: int
    phi: np.ndarray
    rng: np.random.Generator


class _CorrBufsLike(Protocol):
    """Minimal protocol for correlation buffers (e.g., RunBuffers)."""
    corr_bins: object
    corr_sum: np.ndarray
    corr_cnt: np.ndarray


# -----------------------------------------------------------------------------
# 1) Helicity modulus (lattice expression)
# -----------------------------------------------------------------------------
def helicity_modulus(phi: np.ndarray, beta: float, J: float = 1.0) -> float:
    """
    Helicity modulus Υ, averaged over x and y directions.

    Using lattice expression (with wrapped nearest-neighbor differences):
      Υx = <cos(dφx)> - β * <(Σ sin(dφx))^2> / L^2
      Υy = <cos(dφy)> - β * <(Σ sin(dφy))^2> / L^2
      Υ  = (Υx + Υy) / 2

    Parameters
    ----------
    phi : np.ndarray
        Angle field of shape (L, L).
    beta : float
        Inverse temperature, beta = 1/T (assuming k_B = 1).
    J : float
        Coupling constant.

    Returns
    -------
    float
        Helicity modulus.
    """
    L = int(phi.shape[0])
    dpx, dpy = nearest_diffs(phi)

    K1x = float(np.cos(dpx).mean())
    K2x = float(beta * (np.sin(dpx).sum() ** 2) / (L * L))

    K1y = float(np.cos(dpy).mean())
    K2y = float(beta * (np.sin(dpy).sum() ** 2) / (L * L))

    Yx = K1x - K2x
    Yy = K1y - K2y
    return 0.5 * (Yx + Yy)


# -----------------------------------------------------------------------------
# 2) Correlation function G(r) accumulation using CorrBins
# -----------------------------------------------------------------------------
def accumulate_correlation(bufs: _CorrBufsLike, state: _StateLike, n_refs: int = 16) -> None:
    """
    Accumulate direction-averaged correlation:
        G(r) = <cos(phi(0) - phi(r))>

    Uses CorrBins.ring_vectors (precomputed displacement vectors grouped by integer r).
    For each of n_refs random reference sites, we traverse all ring vectors and add:

        cos( phi[i0+dy, j0+dx] - phi[i0, j0] )

    Parameters
    ----------
    bufs : object
        Buffer object holding corr_bins, corr_sum, corr_cnt.
    state : object
        State object holding L, phi, rng.
    n_refs : int
        Number of random reference sites per call (duplicates allowed).
    """
    L = int(state.L)
    phi = state.phi
    rng = state.rng
    bins = bufs.corr_bins

    refs_i = rng.integers(0, L, size=n_refs)
    refs_j = rng.integers(0, L, size=n_refs)

    # CorrBins is expected to provide: ring_vectors: list[list[(dx,dy)]]
    ring_vectors = getattr(bins, "ring_vectors")

    for i0, j0 in zip(refs_i, refs_j):
        phi0 = phi[i0, j0]
        for r, vecs in enumerate(ring_vectors):
            if not vecs:
                continue
            s = 0.0
            c = 0
            for (dx, dy) in vecs:
                ii = (i0 + dy) % L  # y -> axis=0
                jj = (j0 + dx) % L  # x -> axis=1
                s += float(np.cos(angle_diff(phi[ii, jj], phi0)))
                c += 1
            bufs.corr_sum[r] += s
            bufs.corr_cnt[r] += c


def finalize_correlation(bufs: _CorrBufsLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finalize accumulated correlation and return arrays (r, G(r)).

    Returns
    -------
    r : np.ndarray
        Integer radii, shape (r_max+1,).
    G : np.ndarray
        Averaged correlation values.
    """
    r = np.arange(bufs.corr_sum.size, dtype=int)
    with np.errstate(invalid="ignore", divide="ignore"):
        G = bufs.corr_sum / np.maximum(bufs.corr_cnt, 1)
    return r, G


def reset_correlation(bufs: _CorrBufsLike) -> None:
    """Reset correlation accumulators in-place."""
    bufs.corr_sum[:] = 0.0
    bufs.corr_cnt[:] = 0


# -----------------------------------------------------------------------------
# 3) Vortex detection and statistics
# -----------------------------------------------------------------------------
def detect_vortices(phi: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Scan all plaquettes and detect vortices with winding number w = ±1.

    Returns
    -------
    positions : np.ndarray
        shape (N, 3) int array [i, j, w], where (i, j) is the plaquette corner.
        Only w in {-1, +1} are kept.
    nv : float
        Vortex density N / L^2.
    """
    L = int(phi.shape[0])
    pos = []
    for i in range(L):
        for j in range(L):
            w = winding_number(phi, i, j)
            if w == 1 or w == -1:
                pos.append((i, j, w))

    if pos:
        positions = np.array(pos, dtype=np.int32)
    else:
        positions = np.zeros((0, 3), dtype=np.int32)

    nv = float(positions.shape[0] / (L * L))
    return positions, nv


def pair_distance_histogram(vortices: np.ndarray, L: int, r_max: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each + vortex, compute the nearest - vortex distance (PBC minimum image),
    round it to an integer radius, and build a histogram.

    Parameters
    ----------
    vortices : np.ndarray
        shape (N, 3) array [i, j, w] with w in {-1, +1}.
    L : int
        Lattice size.
    r_max : int | None
        Max radius. Defaults to L//2.

    Returns
    -------
    r : np.ndarray
        Radii 0..r_max (int).
    hist : np.ndarray
        Counts for each radius bin.
    """
    if r_max is None:
        r_max = L // 2
    r = np.arange(r_max + 1, dtype=int)
    hist = np.zeros_like(r, dtype=np.int64)

    if vortices.size == 0:
        return r, hist

    pos_p = vortices[vortices[:, 2] == 1][:, :2]
    pos_m = vortices[vortices[:, 2] == -1][:, :2]
    if pos_p.size == 0 or pos_m.size == 0:
        return r, hist

    for (i, j) in pos_p:
        di = np.abs(pos_m[:, 0] - i)
        dj = np.abs(pos_m[:, 1] - j)
        di = np.minimum(di, L - di)
        dj = np.minimum(dj, L - dj)
        dist = np.rint(np.sqrt(di * di + dj * dj)).astype(int)
        rmin = int(dist.min())
        if rmin <= r_max:
            hist[rmin] += 1

    return r, hist


# -----------------------------------------------------------------------------
# 4) Energy and heat capacity
# -----------------------------------------------------------------------------
def energy_density(phi: np.ndarray, J: float = 1.0) -> float:
    """Energy per site E/N (N = L^2), using total_energy which counts each bond once."""
    E = total_energy(phi, J=J)
    return float(E / (phi.shape[0] * phi.shape[1]))


@dataclass(slots=True)
class HeatCapacityAccumulator:
    """
    Accumulator for heat capacity:
        C = beta^2 * ( <E^2> - <E>^2 ) / N

    Notes
    -----
    - E must be the total energy with each bond counted once.
    - N = L^2.
    """

    L: int
    sumE: float = 0.0
    sumE2: float = 0.0
    cnt: int = 0

    @property
    def N(self) -> int:
        return int(self.L * self.L)

    def add(self, E: float) -> None:
        """Add one sample of total energy E."""
        self.sumE += float(E)
        self.sumE2 += float(E) * float(E)
        self.cnt += 1

    def value(self, beta: float) -> float:
        """Return heat capacity estimate from accumulated samples."""
        if self.cnt == 0:
            return float("nan")
        meanE = self.sumE / self.cnt
        meanE2 = self.sumE2 / self.cnt
        return float((beta * beta) * (meanE2 - meanE * meanE) / self.N)
