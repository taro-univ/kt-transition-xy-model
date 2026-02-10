"""
initial.py

Initialization utilities for the 2D XY model.

Implemented initializations
---------------------------
- Random initialization
- Uniform initialization
- Physically motivated vortex / antivortex field on a torus (PBC),
  including vortex-pair initialization (net topological charge = 0)

Notes
-----
- This module depends only on XYState (for writing into state.phi) and
  low-level utilities such as wrap_angle.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

from MCsim.xy_model import XYState
from MCsim.angles import wrap_angle
from MCsim.measure import detect_vortices, energy_density


def _min_image(dx: np.ndarray, L: int) -> np.ndarray:
    """
    Minimum-image convention for PBC on a torus:
        dx -> ((dx + L/2) % L) - L/2
    Works for numpy arrays.
    """
    return (dx + 0.5 * L) % L - 0.5 * L


def vortex_field(
    L: int,
    vortices: List[Tuple[float, float, int]],
    const_phase: float = 0.0,
) -> np.ndarray:
    """
    Construct a phase field φ(x, y) on an LxL torus with multiple vortices.

    Parameters
    ----------
    L : int
        Lattice size.
    vortices : list[(x0, y0, n)]
        Vortex centers (continuous coordinates, 0<=x,y<L) and winding n (±1, ±2, ...).
    const_phase : float
        Global phase offset.

    Returns
    -------
    np.ndarray
        Angle field of shape (L, L), dtype float32, wrapped to (-pi, pi].

    Implementation
    --------------
    φ(r) = Σ n * atan2(dy, dx), where dx, dy follow the minimum-image convention.
    """
    x = np.arange(L, dtype=np.float64)
    y = np.arange(L, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="xy")

    phi = np.zeros((L, L), dtype=np.float64)
    for (x0, y0, n) in vortices:
        dx = _min_image(X - x0, L)
        dy = _min_image(Y - y0, L)
        phi += int(n) * np.arctan2(dy, dx)

    phi += float(const_phase)
    return wrap_angle(phi).astype(np.float32)


def initialize_with_vortex_pair(
    state: XYState,
    sep: Optional[float] = None,
    center: Optional[Tuple[float, float]] = None,
    strength: int = 1,
    const_phase: float = 0.0,
) -> None:
    """
    Initialize the angle field with a vortex(+n) and antivortex(-n) pair (net charge 0).

    Parameters
    ----------
    state : XYState
        Simulation state to modify in-place.
    sep : float | None
        Pair separation in lattice units. Default is ~L/3 (at least 2).
    center : (float, float) | None
        Center position of the pair (continuous coords). Default is lattice center.
    strength : int
        Vortex winding number n (recommend ±1).
    const_phase : float
        Global phase offset.
    """
    L = int(state.L)
    if sep is None:
        sep = float(max(2, L / 3))
    if center is None:
        center = (0.5 * L, 0.5 * L)

    cx, cy = center
    # Place pair along x-direction at ±sep/2 around the center (continuous coordinates)
    x1, y1 = (cx - 0.5 * sep) % L, cy % L
    x2, y2 = (cx + 0.5 * sep) % L, cy % L

    phi = vortex_field(
        L,
        [(x1, y1, +int(strength)), (x2, y2, -int(strength))],
        const_phase=const_phase,
    )
    state.phi[:] = phi


def initialize_uniform(state: XYState, angle: float = 0.0) -> None:
    """Initialize all spins to the same direction (uniform state)."""
    state.phi[:] = wrap_angle(float(angle)).astype(np.float32)


def initialize_random(state: XYState) -> None:
    """Initialize spins with i.i.d. uniform angles in (-pi, pi]."""
    state.phi[:] = state.rng.uniform(-np.pi, np.pi, size=(state.L, state.L)).astype(np.float32)


# ---------------- Optional sanity test ----------------
def test_vortex_initialization(L: int = 64, seed: int = 0, sep: Optional[float] = None) -> None:
    """
    Quick sanity check:
    1) initialize vortex pair and confirm ±1 winding numbers are detected
    2) compare energy density with a uniform configuration (vortex pair should be higher)
    """
    st = XYState.create(L=L, seed=seed, init="uniform", angle=0.0)
    initialize_with_vortex_pair(st, sep=sep)
    vort, nv = detect_vortices(st.phi)
    charges = np.unique(vort[:, 2]) if vort.size else []
    print(f"detected vortices: {vort.shape[0]} (density {nv:.4f}), charges={charges}")

    e_vpair = energy_density(st.phi)
    initialize_uniform(st, 0.0)
    e_uniform = energy_density(st.phi)
    print(f"energy density (vortex pair) = {e_vpair:.4f}  vs  (uniform) = {e_uniform:.4f}")
