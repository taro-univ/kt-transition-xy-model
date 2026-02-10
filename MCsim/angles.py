"""
angles.py

Angle utilities and periodic-boundary-condition (PBC) helpers for the 2D XY model.

Design goals
------------
- Pure utilities: this module does NOT depend on XYState or simulation logic.
- Provide a consistent angle convention: angles/differences are wrapped to (-pi, pi].
- Provide PBC indexing helpers and vortex-related primitives (plaquette winding).

Notes
-----
- All angles are in radians.
- The wrap convention is (-pi, pi] (i.e., -pi is mapped to +pi).
"""

from __future__ import annotations

import numpy as np

PI: float = float(np.pi)
TWO_PI: float = float(2.0 * np.pi)

__all__ = [
    "PI",
    "TWO_PI",
    "wrap_angle",
    "angle_diff",
    "pbc",
    "right",
    "left",
    "nearest_diffs",
    "plaquette_circulation",
    "winding_number",
]


def wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    """
    Wrap angle(s) to the interval (-pi, pi].

    Parameters
    ----------
    x : np.ndarray | float
        Input angle(s) in radians.

    Returns
    -------
    np.ndarray | float
        Wrapped angle(s) in (-pi, pi]. Vectorized for numpy arrays.

    Notes
    -----
    The endpoint -pi is mapped to +pi to enforce the half-open convention.
    """
    y = (np.asarray(x) + PI) % TWO_PI - PI
    if isinstance(y, np.ndarray):
        y[y <= -PI] += TWO_PI
        return y
    return float(y + TWO_PI) if y <= -PI else float(y)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute wrapped angle difference (a - b) in (-pi, pi].

    This is the fundamental operation for phase differences in the XY model.
    """
    return wrap_angle(np.asarray(a) - np.asarray(b))  # type: ignore[return-value]


# ---------------- PBC (periodic boundary) indices ----------------
def pbc(i: int, L: int) -> int:
    """Fold index i into 0..L-1 under periodic boundary conditions."""
    return i % L


def right(i: int, L: int) -> int:
    """i+1 (mod L)."""
    return (i + 1) % L


def left(i: int, L: int) -> int:
    """i-1 (mod L)."""
    return (i - 1) % L


# -------------- Nearest-neighbor differences (used in measurements) --------------
def nearest_diffs(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute nearest-neighbor wrapped differences on an LxL lattice.

    Parameters
    ----------
    phi : np.ndarray
        Angle field with shape (L, L).

    Returns
    -------
    dphi_x : np.ndarray
        Wrapped differences to the right neighbor: phi[i, j+1] - phi[i, j] in (-pi, pi].
    dphi_y : np.ndarray
        Wrapped differences to the up neighbor:    phi[i+1, j] - phi[i, j] in (-pi, pi].
    """
    dphi_x = angle_diff(np.roll(phi, -1, axis=1), phi)
    dphi_y = angle_diff(np.roll(phi, -1, axis=0), phi)
    return dphi_x, dphi_y


# -------------- Plaquette circulation (vortex primitive) --------------
def plaquette_circulation(phi: np.ndarray, i: int, j: int) -> float:
    """
    Sum of wrapped phase differences around the 1x1 plaquette.

    The plaquette is defined with (i, j) as the lower-left corner and traversed
    counterclockwise. Each edge difference is wrapped to (-pi, pi] via angle_diff,
    so we do NOT wrap the final sum here.

    Returns
    -------
    float
        Circulation sum (approximately 2*pi * integer for isolated vortices).
    """
    L = int(phi.shape[0])

    i1, j1 = i, j
    i2, j2 = i, right(j, L)            # (i, j+1)
    i3, j3 = right(i, L), right(j, L)  # (i+1, j+1)
    i4, j4 = right(i, L), j            # (i+1, j)

    s1 = angle_diff(phi[i2, j2], phi[i1, j1])  # (i,j)   -> (i,j+1)
    s2 = angle_diff(phi[i3, j3], phi[i2, j2])  # (i,j+1) -> (i+1,j+1)
    s3 = angle_diff(phi[i4, j4], phi[i3, j3])  # (i+1,j+1)->(i+1,j)
    s4 = angle_diff(phi[i1, j1], phi[i4, j4])  # (i+1,j) -> (i,j)

    return float(s1 + s2 + s3 + s4)


def winding_number(phi: np.ndarray, i: int, j: int) -> int:
    """
    Integer winding number of plaquette (i, j):

        w = round( circulation / (2*pi) )

    In KT analyses, w is typically in {-1, 0, +1} for most configurations.
    """
    circ = plaquette_circulation(phi, i, j)
    return int(np.rint(circ / TWO_PI))
