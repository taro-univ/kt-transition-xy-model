# Efficient Monte Carlo Simulation for a Generalized 2D XY Model

## Executive Summary

This project implements an optimized Monte Carlo simulation framework
for a generalized 2D XY model with nontrivial interaction structure.

Unlike textbook implementations, this simulator:

- Combines multiple update strategies (Metropolis + Over-relaxation + Wolff)
- Adapts proposal width dynamically based on acceptance rate
- Provides physics-grounded validation (Υ, nv, G(r), C)
- Includes KT-consistent transition analysis

The goal is not only faster sampling,
but physically validated sampling efficiency near criticality.

---

## Motivation

Near the Kosterlitz–Thouless (KT) transition,
Monte Carlo simulations suffer from critical slowing down.

Standard single-kernel Metropolis updates:

- Decorrelate slowly
- Become inefficient near Tc
- Fail to efficiently sample vortex unbinding dynamics

This project addresses this by designing a hybrid update framework.

---

## Model

We simulate a 2D XY model with periodic boundary conditions.

Hamiltonian (schematic):

H = - Σ_{⟨i,j⟩} cos(θ_i - θ_j)

The implementation is modular and allows
nontrivial extensions beyond nearest-neighbor coupling.

---

## Update Strategies

### 1. Metropolis Update
- Local proposal
- Adaptive proposal width
- Acceptance-rate controlled

### 2. Over-Relaxation Update
- Energy-conserving deterministic reflection
- Improves decorrelation without rejection

### 3. Wolff Cluster Update
- Cluster-based spin flips
- Mitigates critical slowing down
- Especially effective near Tc

## Benchmark Summary

| Update Strategy | Near Tc Efficiency | Global Moves | Acceptance Stability |
|-----------------|-------------------|--------------|----------------------|
| Metropolis      | Slow              | No           | Moderate             |
| + Over-relax    | Medium            | No           | Stable               |
| + Wolff         | Fast              | Yes          | High                 |


### Hybrid Pipeline

Each temperature step combines:

Metropolis → Over-relaxation → (optional) Wolff

This enables both local exploration and global restructuring.

---

## Engineering Design

The codebase is structured into modular components:

- `config.py`      – experiment configuration
- `xy_model.py`    – state representation
- `update.py`      – update kernels
- `measure.py`     – observable computation
- `analysis.py`    – transition & scaling analysis
- `loop.py`        – simulation driver
- `initial.py`     – structured vortex initialization

Separation of concerns ensures:

- Reproducibility
- Clean experiment design
- Easy benchmarking of update strategies

---

## Observables & Physical Validation

The simulator measures:

- Energy per site
- Specific heat C
- Helicity modulus Υ
- Vortex density nv
- Correlation function G(r)

These allow validation against known KT physics:

- Υ intersection with 2T/π → Tc estimate
- Power-law decay (low T)
- Exponential decay (high T)
- KT scaling form: ln ξ ~ b / sqrt(T − Tc)

This ensures that acceleration does not distort physics.

---

## Results

### 1. Hybrid updates reduce decorrelation time

Near Tc:

- Wolff improves global restructuring
- Over-relaxation reduces autocorrelation
- Adaptive Metropolis stabilizes acceptance

### 2. KT-consistent transition detection

- Tc estimated via Υ crossing
- Vortex density sharply increases above Tc
- Correlation decay behavior matches theoretical expectations

### 3. Structured initialization validation

Vortex–antivortex initialization is implemented
to verify vortex detection and energy calculations.

---

## Key Contributions

- Hybrid MCMC design for critical systems
- Adaptive proposal mechanism
- Physics-consistent validation pipeline
- Modular experiment framework
- KT scaling verification

---

## Why This Project Matters

This project demonstrates:

- Algorithmic understanding of MCMC dynamics
- Critical region optimization
- Domain-aware validation of numerical methods
- Ability to design clean scientific software pipelines

The approach generalizes to:

- Statistical physics simulation
- Bayesian MCMC acceleration
- Sampling in high-dimensional structured spaces

---

## Future Work

- Explicit integrated autocorrelation time τ_int benchmarking
- Effective sample size comparison
- Multi-size scaling collapse
- GPU acceleration
