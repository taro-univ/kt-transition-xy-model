"""
physics_utils.py

Standalone physics utility functions for the ML module.

These are intentionally independent of MCsim so that the
Unsupervised_Machine_Learning package has no hard dependency
on the simulation layer.

Provides
--------
wrap_angle            : wrap angle differences to (-pi, pi]
vortex_density_from_phi : compute vortex density from an angle field
"""

import numpy as np


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
    dphi_x = wrap_angle(np.roll(phi, -1, axis=0) - phi)
    dphi_y = wrap_angle(np.roll(phi, -1, axis=1) - phi)

    dphi_y_x = np.roll(dphi_y, -1, axis=0)
    dphi_x_y = np.roll(dphi_x, -1, axis=1)

    circulation = wrap_angle(dphi_x + dphi_y_x - dphi_x_y - dphi_y)
    q = np.rint(circulation / (2 * np.pi)).astype(np.int32)

    return float(np.mean(np.abs(q)))
