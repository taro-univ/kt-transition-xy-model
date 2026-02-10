from __future__ import annotations

"""
xy_model.py

Data structures for Monte Carlo simulations of the 2D XY model (KT transition).

Design philosophy
-----------------
- XYState holds the minimal simulation state: lattice size, angle field, RNG,
  and optional cached nearest-neighbor differences.
- RunBuffers bundles per-run (typically per-temperature) accumulators such as
  correlation ring bins, time-series logs, and optional snapshots.
- CorrBins precomputes ring vectors for fast, direction-averaged correlation
  measurements under periodic boundary conditions (minimum-image convention).

Angles are treated in radians and wrapped to the interval (-pi, pi].
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import numpy as np

from MCsim.angles import wrap_angle


@dataclass(slots=True)
class CorrBins:
    """
    Precomputed ring bins for direction-averaged correlation functions.

    We bin lattice displacement vectors (dx, dy) by integer radius r:
        r = round(sqrt(dx^2 + dy^2))

    Only vectors whose Euclidean radius is exactly an integer (within tolerance)
    are included. This matches the common practice of reporting G(r) on integer r.

    Assumptions
    ----------
    - Periodic boundary conditions (PBC).
    - Minimum-image convention: dx, dy ∈ [-L/2, L/2].

    Attributes
    ----------
    L : int
        Linear system size.
    r_max : int
        Maximum radius (L//2).
    ring_vectors : list[list[tuple[int,int]]]
        ring_vectors[r] stores displacement vectors belonging to radius bin r.
    """

    L: int
    r_max: int = field(init=False)
    ring_vectors: List[List[Tuple[int, int]]] = field(init=False)

    def __post_init__(self) -> None:
        self.r_max = self.L // 2
        self.ring_vectors = [[] for _ in range(self.r_max + 1)]

        tol = 1e-9
        for dx in range(-self.r_max, self.r_max + 1):
            for dy in range(-self.r_max, self.r_max + 1):
                r_float = math.hypot(dx, dy)
                r = int(round(r_float))
                if r <= self.r_max and abs(r - r_float) < tol:
                    self.ring_vectors[r].append((dx, dy))

    def new_accumulators(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Create fresh accumulators for correlation measurement.

        Returns
        -------
        sums : np.ndarray
            Sum of correlation values for each radius bin r.
        counts : np.ndarray
            Number of samples accumulated in each bin r.
        """
        sums = np.zeros(self.r_max + 1, dtype=np.float64)
        counts = np.zeros(self.r_max + 1, dtype=np.int64)
        return sums, counts


@dataclass(slots=True)
class TimeSeries:
    """
    Time-series logs collected during measurement sweeps.

    The lists are append-only and are easy to serialize to JSON/NPZ.
    """

    E: List[float] = field(default_factory=list)       # e.g., energy density
    Y: List[float] = field(default_factory=list)       # helicity modulus
    nv: List[float] = field(default_factory=list)      # vortex density
    accept: List[float] = field(default_factory=list)  # acceptance rate (optional)


@dataclass(slots=True)
class SnapshotStore:
    """
    Optional snapshots for debugging/visualization.

    For large runs, reduce the snapshot frequency to keep memory usage low.
    """

    angles: List[np.ndarray] = field(default_factory=list)    # phi snapshots (L,L)
    vortices: List[np.ndarray] = field(default_factory=list)  # arrays of (x, y, charge)


@dataclass(slots=True)
class XYState:
    """
    Minimal simulation state of the 2D XY model.

    Attributes
    ----------
    L : int
        Linear system size.
    phi : np.ndarray
        Angle field of shape (L, L), dtype float32, values in (-pi, pi].
    rng : np.random.Generator
        Random number generator.
    dphi_x, dphi_y : Optional[np.ndarray]
        Optional cached nearest-neighbor angle differences (for speed).
    """

    L: int
    phi: np.ndarray
    rng: np.random.Generator
    dphi_x: Optional[np.ndarray] = None
    dphi_y: Optional[np.ndarray] = None

    @staticmethod
    def create(
        L: int,
        seed: Optional[int] = None,
        init: str = "random",
        angle: float = 0.0,
    ) -> "XYState":
        """
        Factory constructor for XYState.

        Parameters
        ----------
        L : int
            Lattice size (L x L).
        seed : Optional[int]
            RNG seed.
        init : str
            "random" or "uniform".
        angle : float
            Used only if init="uniform".

        Notes
        -----
        Vortex-pair initialization is provided in `initial.py`.
        """
        rng = np.random.default_rng(seed)

        if init == "random":
            phi = rng.uniform(-PI, PI, size=(L, L)).astype(np.float32)
            phi = wrap_angle(phi).astype(np.float32)
        elif init == "uniform":
            a = float(wrap_angle(angle))
            phi = np.full((L, L), fill_value=a, dtype=np.float32)
        else:
            raise ValueError("init must be 'random' or 'uniform'")

        return XYState(L=L, phi=phi, rng=rng)


@dataclass(slots=True)
class RunBuffers:
    """
    Bundle of per-run buffers/accumulators.

    Typically, one RunBuffers is created per temperature and reused during
    the measurement phase.

    Attributes
    ----------
    corr_bins : CorrBins
        Precomputed ring bins for correlation measurement.
    corr_sum, corr_cnt : np.ndarray
        Accumulators for the direction-averaged correlation function.
    tseries : TimeSeries
        Time-series logs.
    snaps : SnapshotStore
        Optional snapshots.
    """

    corr_bins: CorrBins
    corr_sum: np.ndarray
    corr_cnt: np.ndarray
    tseries: TimeSeries = field(default_factory=TimeSeries)
    snaps: SnapshotStore = field(default_factory=SnapshotStore)

    @staticmethod
    def create(L: int) -> "RunBuffers":
        """Create a fresh RunBuffers instance for a given lattice size."""
        bins = CorrBins(L)
        s, c = bins.new_accumulators()
        return RunBuffers(corr_bins=bins, corr_sum=s, corr_cnt=c)

