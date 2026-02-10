"""
check.py

Consistency checks between KT theory and Monte Carlo results
for the 2D XY model.

This module is NOT for quantitative analysis.
Its purpose is to provide quick, human-readable sanity checks
confirming that the simulation reproduces well-known KT features.

Typical use:
    - run simulations
    - compute observables via analysis.py
    - visually confirm theoretical consistency using this module
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

PI = float(np.pi)


# ---------------------------------------------------------------------
# 1) Low-T spin-wave consistency: eta(T) ≈ T / (2πJ)
# ---------------------------------------------------------------------
def check_lowT_spinwave(results, J: float = 1.0) -> None:
    """
    Compare measured eta(T) with the spin-wave prediction:
        eta(T) ≈ T / (2πJ)

    Intended as a qualitative low-temperature consistency check.
    """
    T = np.array([r["T"] for r in results])
    eta = np.array([r.get("eta", np.nan) for r in results])
    eta_sw = T / (2.0 * PI * J)

    plt.figure(figsize=(6, 4))
    plt.plot(T, eta, "o-", label="measured η(T)")
    plt.plot(T, eta_sw, "--", label="spin-wave: T/(2πJ)")
    plt.xlabel("T")
    plt.ylabel("η(T)")
    plt.title("Low-temperature spin-wave consistency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ---------------------------------------------------------------------
# 2) Universal jump at Tc: Υ(Tc⁻) = 2Tc/π and η(Tc) ≈ 1/4
# ---------------------------------------------------------------------
def check_universal_jump(results, Tc_est: float | None = None) -> None:
    """
    Visual check of the KT universal jump:
        Υ(Tc⁻) = 2Tc / π
        η(Tc)  ≈ 1/4
    """
    T = np.array([r["T"] for r in results])
    Y = np.array([np.mean(r["time_series"]["Y"]) for r in results])
    eta = np.array([r.get("eta", np.nan) for r in results])

    plt.figure(figsize=(10, 4))

    # (a) Helicity modulus
    plt.subplot(1, 2, 1)
    plt.plot(T, Y, "o-", label="Υ(T)")
    plt.plot(T, 2.0 * T / PI, "--", label="2T/π")
    if Tc_est is not None:
        plt.axvline(Tc_est, ls=":", c="k", label=f"Tc ≈ {Tc_est:.3f}")
    plt.xlabel("T")
    plt.ylabel("Υ")
    plt.title("Universal jump check")
    plt.legend()

    # (b) eta at Tc
    plt.subplot(1, 2, 2)
    plt.plot(T, eta, "o-", label="η(T)")
    plt.axhline(0.25, ls="--", c="r", label="η = 1/4")
    if Tc_est is not None:
        plt.axvline(Tc_est, ls=":", c="k")
    plt.xlabel("T")
    plt.ylabel("η(T)")
    plt.title("Critical exponent at Tc")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 3) Vortex unbinding: qualitative density increase
# ---------------------------------------------------------------------
def check_vortex_unbinding(results) -> None:
    """
    Plot vortex density n_v(T) to visually confirm vortex unbinding
    across the KT transition.
    """
    T = np.array([r["T"] for r in results])
    nv = np.array([np.mean(r["time_series"]["nv"]) for r in results])

    plt.figure(figsize=(6, 4))
    plt.plot(T, nv, "o-", color="purple")
    plt.xlabel("T")
    plt.ylabel("vortex density n_v")
    plt.title("Vortex unbinding across KT transition")
    plt.grid(alpha=0.3)
    plt.show()
