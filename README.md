# kt-transition-xy-model

Reproducible Monte Carlo study of the Kosterlitz–Thouless (KT) transition in the 2D XY model,
with extensions toward data-driven phase identification.

---

## Overview
This repository provides a reproducible numerical study of the Kosterlitz–Thouless (KT)
transition in the two-dimensional XY model.

The KT transition is a topological phase transition that occurs without spontaneous
symmetry breaking or a local order parameter. Instead, it is driven by the binding and
unbinding of vortex–antivortex pairs.

This project combines:
- Monte Carlo simulations based on statistical physics
- Measurement of physically meaningful observables
- (Optional extension) Representation learning for phase-structure analysis

The implementation and analysis are based on my undergraduate thesis
(Department of Physics, Waseda University, 2026).

---

## Physical Background
In the 2D XY model, each lattice site carries a planar spin represented by an angle
\(\phi_i \in [0, 2\pi)\), with the Hamiltonian

\[
H = -J \sum_{\langle i, j \rangle} \cos(\phi_i - \phi_j).
\]

Due to the continuous \(U(1)\) symmetry, long-range order is forbidden at finite
temperature (Mermin–Wagner theorem). Nevertheless, the system exhibits a KT transition
characterized by:

- A low-temperature quasi-long-range ordered phase
- A high-temperature disordered phase
- A transition driven by vortex–antivortex unbinding

Key indicators of the transition include vortex density, helicity modulus, and the
behavior of spin–spin correlation functions.

---

## Features
- Monte Carlo simulation of the 2D XY model
- Metropolis single-spin updates
- Over-relaxation updates to reduce autocorrelation
- Adaptive proposal width for efficient sampling
- Measurement of:
  - Energy density
  - Helicity modulus (spin stiffness)
  - Vortex density
  - Spin–spin correlation functions
- Clear reproduction of KT-transition signatures

---
## Physical Background

In the 2D XY model, each lattice site carries a planar spin represented by
an angle `phi_i ∈ [0, 2π)`.

The Hamiltonian is given by:

    H = -J Σ_<i,j> cos(phi_i - phi_j)

Due to the continuous U(1) symmetry, long-range order is forbidden at finite
temperature (Mermin–Wagner theorem). Nevertheless, the system exhibits a
Kosterlitz–Thouless (KT) transition characterized by:

- A low-temperature quasi-long-range ordered phase
- A high-temperature disordered phase
- A transition driven by vortex–antivortex unbinding

Key indicators of the transition include vortex density, helicity modulus,
and the behavior of spin–spin correlation functions.

---
## Project Structure
```
kt-transition-xy-model/
└─ MCsim/ # Monte Carlo simulation core
├─ xy_model.py # XY model state and Hamiltonian container
├─ angles.py # Angle wrapping, PBC helpers, plaquette winding
├─ config.py # Simulation configuration and JSON I/O
├─ initial.py # Initial conditions (random, uniform, vortex pairs)
├─ update.py # MC update kernels (Metropolis / over-relaxation / Wolff)
├─ measure.py # Observable measurements (Υ, vortices, correlations)
├─ loop.py # Main simulation loop (thermalize, sample, save)
├─ analysis.py # Post-analysis (Tc, η, ξ, KT scaling)
└─ check.py # Theory-level consistency checks
```


Each module has a **single, well-defined responsibility**, and dependencies
are kept strictly one-directional.

---

## Typical Workflow

1. **Configure** the simulation using `config.py`
2. **Run** Monte Carlo sweeps via `loop.py`
3. **Measure** observables during sampling
4. **Analyze** results using `analysis.py`
5. **Verify** theoretical consistency with `check.py`

---

## Analysis and Theory Checks

- Tc estimation from the helicity modulus crossing with `2T / pi`
- Low-temperature behavior consistent with spin-wave theory
- Universal jump: `Upsilon(Tc-) = 2Tc / pi`
- Critical exponent at Tc: `eta(Tc) ≈ 1/4`
- Vortex unbinding observed through vortex density

Quantitative analysis is separated from qualitative theory checks by design.


---

## Design Philosophy

- Explicit separation of *model*, *algorithm*, *measurement*, and *analysis*
- Minimal hidden state; all randomness is explicit and reproducible
- Emphasis on **physical interpretability**, not black-box numerics
- Code written to be readable by both physicists and non-specialists

---

## Notes

This repository was originally developed as part of an undergraduate thesis
project and has been refactored for clarity, reproducibility, and public release.
