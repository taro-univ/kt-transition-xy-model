"""
update.py

Monte Carlo update kernels for the 2D XY model.

Implemented updates
-------------------
- Metropolis single-spin updates (one sweep visits all sites once in random order)
- Over-relaxation (microcanonical reflection update)
- Optional Wolff cluster update for O(2) (embedded Ising construction)

This module is intentionally stateless: it updates the given `state` in-place.
The `state` object is expected to have attributes:
- L : int
- phi : np.ndarray of shape (L, L)
- rng : np.random.Generator
"""

from __future__ import annotations

from typing import Protocol
import numpy as np

from MCsim.angles import wrap_angle, angle_diff, right, left
from MCsim.config import Config  # kept for type-level coherence (optional use)


class _StateLike(Protocol):
    L: int
    phi: np.ndarray
    rng: np.random.Generator


# --------------------------
# Energy helpers (local / total)
# --------------------------
def local_energy_site(phi: np.ndarray, i: int, j: int, J: float = 1.0) -> float:
    """
    Local energy contribution around site (i, j) using its 4 nearest-neighbor bonds.

    Note: This is mainly for diagnostics; for Metropolis we use ΔE from a single-site change.
    """
    L = phi.shape[0]
    e = 0.0
    e -= J * np.cos(angle_diff(phi[i, right(j, L)], phi[i, j]))
    e -= J * np.cos(angle_diff(phi[i, j], phi[i, left(j, L)]))
    e -= J * np.cos(angle_diff(phi[right(i, L), j], phi[i, j]))
    e -= J * np.cos(angle_diff(phi[i, j], phi[left(i, L), j]))
    return float(e)


def delta_energy_single_flip(phi: np.ndarray, i: int, j: int, new_phi_ij: float, J: float = 1.0) -> float:
    """
    Energy difference ΔE when phi[i, j] is replaced by new_phi_ij (neighbors fixed).
    """
    old = float(phi[i, j])
    if new_phi_ij == old:
        return 0.0

    L = phi.shape[0]
    dE = 0.0

    jR = right(j, L)
    dE -= J * (np.cos(angle_diff(phi[i, jR], new_phi_ij)) - np.cos(angle_diff(phi[i, jR], old)))

    jL = left(j, L)
    dE -= J * (np.cos(angle_diff(new_phi_ij, phi[i, jL])) - np.cos(angle_diff(old, phi[i, jL])))

    iU = right(i, L)
    dE -= J * (np.cos(angle_diff(phi[iU, j], new_phi_ij)) - np.cos(angle_diff(phi[iU, j], old)))

    iD = left(i, L)
    dE -= J * (np.cos(angle_diff(new_phi_ij, phi[iD, j])) - np.cos(angle_diff(old, phi[iD, j])))

    return float(dE)


def total_energy(phi: np.ndarray, J: float = 1.0) -> float:
    """
    Total energy counting each bond once (right and up bonds only).
    """
    e = -J * (
        np.cos(angle_diff(np.roll(phi, -1, axis=1), phi)).sum()
        + np.cos(angle_diff(np.roll(phi, -1, axis=0), phi)).sum()
    )
    return float(e)


# ------------------------------------
# Metropolis sweep (one visit per site)
# ------------------------------------
def metropolis_sweep(state: _StateLike, beta: float, delta_phi: float, J: float = 1.0) -> float:
    """
    One Metropolis sweep: propose L*L single-spin updates in random site order.

    Returns
    -------
    float
        Acceptance rate in [0, 1].
    """
    L = state.L
    phi = state.phi
    rng = state.rng

    idx = rng.permutation(L * L)
    acc = 0

    for k in idx:
        i = int(k // L)
        j = int(k % L)
        trial = wrap_angle(phi[i, j] + rng.uniform(-delta_phi, delta_phi))
        dE = delta_energy_single_flip(phi, i, j, float(trial), J=J)

        if dE <= 0.0 or rng.random() < np.exp(-beta * dE):
            phi[i, j] = float(trial)
            acc += 1

    return acc / (L * L)


def autotune_delta_phi(
    delta_phi: float,
    accept_rate: float,
    tgt_min: float = 0.40,
    tgt_max: float = 0.60,
) -> float:
    """
    Heuristically tune Metropolis proposal width to keep acceptance in a target range.
    """
    if accept_rate < tgt_min:
        delta_phi *= 0.9
    elif accept_rate > tgt_max:
        delta_phi *= 1.1

    return float(np.clip(delta_phi, 1e-3, np.pi))


# ------------------------------------
# Over-relaxation sweep (energy-conserving)
# ------------------------------------
def overrelaxation_sweep(state: _StateLike, J: float = 1.0) -> None:
    """
    Over-relaxation update (microcanonical reflection) at each site.

    For each site, compute the local field vector:
        S = Σ_neighbors (cos φ_k, sin φ_k)
    and reflect the spin angle about arg(S):
        φ' = 2 * arg(S) - φ
    """
    L = state.L
    phi = state.phi

    for i in range(L):
        for j in range(L):
            nb = [
                phi[i, right(j, L)],
                phi[i, left(j, L)],
                phi[right(i, L), j],
                phi[left(i, L), j],
            ]
            cx = np.cos(nb).sum()
            sx = np.sin(nb).sum()
            if cx == 0.0 and sx == 0.0:
                continue
            alpha = np.arctan2(sx, cx)
            phi[i, j] = float(wrap_angle(2.0 * alpha - phi[i, j]))


# ------------------------------------
# Wolff cluster update (optional)
# ------------------------------------
def wolff_cluster_step(state: _StateLike, beta: float, J: float = 1.0) -> int:
    """
    O(2) Wolff cluster (embedded Ising):

    1) choose random axis r with angle alpha
    2) define s_parallel = cos(phi - alpha) = s · r
    3) connect same-sign neighbors with probability:
         p = 1 - exp(-2 * beta * J * |s_par(i) * s_par(j)|)
    4) reflect spins in the cluster about axis r:
         phi -> 2*alpha - phi

    Returns
    -------
    int
        Cluster size.
    """
    L = state.L
    phi = state.phi
    rng = state.rng

    alpha = float(rng.uniform(-np.pi, np.pi))

    s_par = np.cos(phi - alpha)  # projection onto r
    sigma = np.sign(s_par)
    sigma[sigma == 0.0] = 1.0  # rare tie -> treat as +1 for stability

    i0 = int(rng.integers(L))
    j0 = int(rng.integers(L))
    sig0 = sigma[i0, j0]

    in_cluster = np.zeros((L, L), dtype=np.bool_)
    stack = [(i0, j0)]
    in_cluster[i0, j0] = True

    size = 0
    while stack:
        i, j = stack.pop()
        size += 1

        for (ni, nj) in (
            (i, right(j, L)),
            (i, left(j, L)),
            (right(i, L), j),
            (left(i, L), j),
        ):
            if in_cluster[ni, nj]:
                continue
            if sigma[ni, nj] != sig0:
                continue

            p = 1.0 - np.exp(-2.0 * beta * J * abs(s_par[i, j] * s_par[ni, nj]))
            if rng.random() < p:
                in_cluster[ni, nj] = True
                stack.append((ni, nj))

    phi[in_cluster] = wrap_angle(2.0 * alpha - phi[in_cluster])
    return int(size)
