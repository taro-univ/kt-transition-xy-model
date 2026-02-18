# Physics-Aware Representation Learning  

Quick review order:
1. MC simulation validation
2. Representation learning notebook

### Kosterlitz–Thouless Transition as a Structured ML Benchmark

This repository investigates whether **domain-aware objectives**
improve representation learning in systems without simple phase boundaries.

We use the 2D XY model (KT transition) as a benchmark,
where phase structure is topological and does not admit binary classification.

---

## 🚀 Project Overview

This project consists of two tightly integrated components:

### 1️⃣ Efficient Monte Carlo Engine (`MCsim/`)

A modular, physics-validated simulation framework:

- Hybrid updates (Metropolis + Over-relaxation + Wolff)
- Adaptive proposal width
- KT-consistent observables (Υ, nv, G(r), C)
- Scaling & transition analysis pipeline

See:
👉 `MCsim/Efficient_Monte_Carlo_Simulation.md`

---

### 2️⃣ Physics-Aware Representation Learning (`Unsupervised_Machine_Learning/`)

We compare:

- Autoencoder (AE)
- Variational AE (VAE)
- Contrastive learning
- **Helicity-aware Contrastive (proposed)**

Evaluation includes:

- UMAP latent geometry
- Temperature-dependent clustering
- Correlation with physical observables
- Latent slope sensitivity near transition

See:
👉 `Unsupervised_Machine_Learning/01_representation_learning.md`

---

## 🎯 Core Research Question

Rather than asking:

> "Can ML classify phases?"

We ask:

> "Can latent space encode intrinsic physical structure
> without imposing artificial binary boundaries?"

The KT transition is ideal for this,
as it features:

- No spontaneous symmetry breaking
- Topological vortex unbinding
- Continuous structural reorganization

---

## 🔬 Key Findings

- Helicity-aware objective reveals a clear three-regime structure
- Latent axes align with physical observables (Υ, nv)
- Latent slope peaks near Tc (transition sensitivity)
- Hybrid MCMC improves sampling efficiency near criticality

The proposed objective improves
**structural fidelity, interpretability, and transition encoding**.

---

## 🏗 Engineering Design

The project emphasizes clean separation of concerns:

### Simulation Layer
- `update.py`
- `measure.py`
- `analysis.py`
- `loop.py`

### ML Layer
- `dataset/`
- `models/`
- `train/`
- `analysis/`

Heavy computations are separated from notebooks
to maintain lightweight and reproducible visualization.

---

## 📦 Dataset Export for ML

Simulation outputs are exported as structured `.npz` files:

- Feature matrix `X`
- Temperature labels `y`
- Explicit feature names
- Physically validated observables

This enables:

- scikit-learn compatibility
- PyTorch training
- Reproducible ML experiments

The simulator acts as a **structured feature generator**.

---

## 🧠 Why This Matters

This project demonstrates:

- Domain knowledge improves latent geometry
- Objective design shapes representation quality
- Transition detection can be framed as sensitivity analysis
- Clean experimental architecture enhances reproducibility

The methodology generalizes to:

- Scientific machine learning
- Structured latent modeling
- Physics-informed AI systems
- Any system with non-binary regime transitions

---

## 📈 Skills Demonstrated

- Monte Carlo algorithm design
- Hybrid MCMC optimization
- Scaling & transition analysis
- Representation learning
- Latent geometry evaluation
- Modular scientific software engineering
- Reproducible ML pipelines

---

## 🔮 Future Extensions

- Integrated autocorrelation benchmarking
- Multi-size scaling collapse
- GPU acceleration
- Physics-informed regularization design

---

## Author

Physics × Machine Learning  
Designed as a research-grade benchmark for structured representation learning.


---

# 📓 Notebook Guide

This repository contains two main notebooks.
Below is a quick guide for reviewers.

---

## 🧪 1. `MCsim/Efficient_Monte_Carlo_Simulation.ipynb`

### What it demonstrates

This notebook introduces the 2D XY model and the KT transition,
and demonstrates a physics-validated Monte Carlo engine.

### Contents

- Brief explanation of the 2D XY Hamiltonian
- Hybrid update strategy (Metropolis + Over-relaxation + optional Wolff)
- Measurement of:
  - Helicity modulus (Υ)
  - Vortex density (n_v)
  - Correlation function G(r)
- Power-law vs exponential decay behavior
- KT-consistent transition validation

### Lightweight structure

Heavy temperature sweeps are separated from the notebook.

Instead:
- Reproducible code blocks are provided (commented)
- Pre-generated figures are loaded for fast inspection

This keeps the notebook GitHub-friendly while preserving reproducibility.

### What to look at

- Vortex visualization
- Correlation decay behavior
- Consistency with KT theory

---

## 🤖 2. `Unsupervised_Machine_Learning/01_representation_learning.ipynb`

### What it investigates

Whether physics-aware objectives improve
latent structural encoding in unsupervised learning.

### Flow of analysis

1. Train latent representations (AE, VAE, Contrastive, Helicity-aware)
2. Visualize latent geometry via UMAP
3. Perform temperature-dependent clustering
4. Evaluate correlation between latent axes and physical observables
5. Quantify transition sensitivity via latent slope analysis

### Design philosophy

- Heavy computations are separated into standalone scripts
- The notebook focuses on comparison and interpretation
- Reduced-size latent files are used for GitHub compatibility

### What to look at

- Clear three-regime separation in helicity-aware model
- Alignment between latent axes and Υ / vortex density
- Peak slope behavior near estimated Tc

---

## 🔗 Relationship Between the Two

The workflow is:

Simulation → Structured physical observables → `.npz` export → Latent training → Structural evaluation

The Monte Carlo engine generates physically validated features.
The ML module evaluates how well latent space encodes that structure.

This separation ensures:

- Reproducibility
- Clean engineering structure
- Clear linkage between physics and ML
