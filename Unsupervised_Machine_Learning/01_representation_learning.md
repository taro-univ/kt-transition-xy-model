# Representation Learning for the 2D XY Model

## Executive Summary

This project investigates whether physics-aware representation learning
can better capture phase structure in the 2D XY model.

Key findings:

- Helicity-aware contrastive learning reveals a clear three-regime structure.
- Cluster dominance shifts consistently across temperature.
- Latent sensitivity peaks near the estimated critical temperature.

These results suggest that incorporating domain knowledge
improves structural fidelity in unsupervised learning.
Unlike symmetry-breaking transitions, the KT transition does not admit a simple binary phase structure, making it a challenging benchmark for representation learning.

The project emphasizes objective design, latent geometry analysis,
and reproducible evaluation pipelines.

## Why This Matters

This experiment demonstrates how incorporating domain-specific signals
into representation learning can improve structural interpretability.

The approach is generalizable to:
- Scientific machine learning
- Structured data representation
- Physics-informed AI systems

This study reframes phase detection as a representation learning problem rather than a classification problem.


## 1. Problem Setting

### Background

The 2D XY model exhibits the Kosterlitz–Thouless (KT) transition — a topological phase transition that does not involve spontaneous symmetry breaking or a simple two-phase structure.

Instead, the system shows:

- A low-temperature quasi-ordered phase  
- A high-temperature disordered phase  
- A nontrivial intermediate regime driven by vortex unbinding  

This makes phase identification nontrivial using standard binary classification.

---

### Motivation

Many machine learning studies frame phase detection as a two-class problem (ordered vs disordered).  
However, such approaches may:

- Impose an artificial two-phase interpretation  
- Overlook intermediate structures  
- Collapse transitional regimes in latent space  

Our objective is different.

Rather than classifying phases, we aim to:

> Learn latent representations that faithfully encode the intrinsic structural organization of the system across temperature.

---

### Research Question

Can representation learning:

1. Separate low- and high-temperature regimes?
2. Reveal a continuous intermediate structure?
3. Align latent dimensions with physically meaningful observables?

---

### Approach

We compare four unsupervised methods:

- Autoencoder (AE)  
- Variational Autoencoder (VAE)  
- Contrastive learning  
- **Helicity-Contrastive learning (proposed)**  

We evaluate representations through:

- UMAP geometry  
- Temperature-dependent clustering  
- Correlation with physical observables  

The central question is whether helicity-aware training improves structural fidelity in latent space.

## 2. UMAP Comparison
To qualitatively compare the geometric structure of latent spaces across models,
we project each representation to 2D using UMAP.

For fair comparison:
- Same UMAP hyperparameters are used for all models.
- A fixed random seed ensures reproducibility.
- At most 3000 samples are used for visualization consistency.

Color indicates temperature \(T\).

This visualization allows us to inspect:
- Whether low- and high-temperature regimes separate clearly
- Whether an intermediate regime emerges
- Whether the proposed helicity-aware model exhibits improved structural organization
## UMAP Comparison

![UMAP comparison](results/notebook_data/umap_compare_4models.png)

<details>
<summary>Show code: UMAP visualization (AE / VAE / Contrastive / Helicity-Contrastive)</summary>

```python
# === UMAP visualization cell (AE / VAE / Contrastive / Helicity-Contrastive) ===
# Assumes you run from: notebooks/ (so ../results/notebook_data exists)

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- Config ---
DATA_DIR = Path("results/notebook_data")

files = {
    "AE (baseline)": DATA_DIR / "ae_latent.npz",
    "VAE (baseline)": DATA_DIR / "vae_latent.npz",
    "Contrastive (baseline)": DATA_DIR / "contrastive_latent_small.npz",
    "Helicity-Contrastive (proposed)": DATA_DIR / "helicity_contrastive_latent_small.npz",
}

SEED = 42
N_NEIGHBORS = 30
MIN_DIST = 0.10
N_SAMPLES_MAX = 3000
POINT_SIZE = 6
ALPHA = 0.9

# --- Import UMAP (install message if missing) ---
try:
    import umap
except ImportError as e:
    raise ImportError(
        "UMAP is not installed. Run one of the following in your environment:\n"
        "  pip install umap-learn\n"
        "or (conda)\n"
        "  conda install -c conda-forge umap-learn"
    ) from e


def load_latent(npz_path: Path, n_max: int, seed: int):
    d = np.load(npz_path)
    if "z" not in d or "T" not in d:
        raise KeyError(f"{npz_path.name} must contain 'z' and 'T'. keys={list(d.files)}")

    z = np.asarray(d["z"])
    T = np.asarray(d["T"]).reshape(-1)

    if z.shape[0] != T.shape[0]:
        raise ValueError(f"Length mismatch in {npz_path.name}: z={z.shape}, T={T.shape}")

    # Subsample if needed (deterministic)
    N = z.shape[0]
    if N > n_max:
        rs = np.random.RandomState(seed)
        idx = rs.choice(N, size=n_max, replace=False)
        z = z[idx]
        T = T[idx]

    return z, T


def umap_2d(z, seed, n_neighbors, min_dist):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(z)


# --- Load all data first (so failures show early) ---
loaded = {}
for name, fp in files.items():
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")
    z, T = load_latent(fp, n_max=N_SAMPLES_MAX, seed=SEED)
    loaded[name] = (z, T)
    print(f"[OK] {name:28s} | z={z.shape} dtype={z.dtype} | T={T.shape} dtype={T.dtype}")

# --- Compute UMAP for each model ---
embeddings = {}
for name, (z, T) in loaded.items():
    print(f"[INFO] UMAP fitting: {name}")
    xy = umap_2d(z, seed=SEED, n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST)
    embeddings[name] = (xy, T)

# --- Plot (2x2) ---
fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

names = list(files.keys())
vmin = min(np.min(embeddings[n][1]) for n in names)
vmax = max(np.max(embeddings[n][1]) for n in names)

for ax, name in zip(axes.ravel(), names):
    xy, T = embeddings[name]
    sc = ax.scatter(
        xy[:, 0], xy[:, 1],
        c=T,
        s=POINT_SIZE,
        alpha=ALPHA,
        vmin=vmin, vmax=vmax,
    )
    ax.set_title(f"{name}\nUMAP (n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST})")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if "proposed" in name:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)

# --- Shared colorbar (Temperature) ---
fig.subplots_adjust(right=0.86)
cbar = fig.colorbar(sc, ax=axes, location="right", shrink=0.8)
cbar.set_label("Temperature T")

out_path = DATA_DIR / "umap_compare_4models.png"
plt.savefig(out_path, dpi=300)
print(f"[INFO] Saved figure: {out_path}")

plt.show()
```
</details>


## 3. Cluster Structure vs Temperature
Based on the UMAP visualization, we perform **K-means clustering (k = 3)** in the original latent space.

The choice of \(k = 3\) is motivated by:

- The physical expectation of three regimes in the 2D XY model  
  (low-temperature, intermediate/critical, and high-temperature phases)
- The three-structure separation observed in the helicity-aware embedding

Clustering is applied directly in latent space (not in UMAP space) to avoid projection artifacts.

For each temperature \(T\), we compute the cluster probability  
\[
P_T(k)
\]
which represents the fraction of samples assigned to cluster \(k\).

Clear dominance shifts in \(P_T(k)\) across temperature indicate
the emergence of distinct structural regimes.

![Cluster probability vs T](results/notebook_data/4model_Cluster_prob_vs_T.png)
<details>
<summary>Show code: Cluster probability vs T visualization (AE / VAE / Contrastive / Helicity-Contrastive)</summary>
    
```python
import subprocess
from pathlib import Path
import sys

DATA_DIR = Path("results/notebook_data")
SCRIPT = Path("analysis/cluster_vs_T.py")

latent = DATA_DIR / "helicity_contrastive_latent_small.npz"
out_dir = DATA_DIR  

models = {
    "AE": "ae_latent.npz",
    "VAE": "vae_latent.npz",
    "Contrastive": "contrastive_latent_small.npz",
    "Helicity-Contrastive": "helicity_contrastive_latent_small.npz",
}

for name, filename in models.items():
    latent_path = DATA_DIR / filename
    
    cmd = [
        sys.executable,  
        str(SCRIPT),
        "--latent", str(latent_path),
        "--out_dir", str(DATA_DIR),
        "--n_clusters", "3",
        "--seed", "42",
    ]
    
    print(f"\n[Running cluster_vs_T for {name}]")
    subprocess.run(cmd, check=True)

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
axes = axes.ravel()

for ax, (name, filename) in zip(axes, models.items()):
    
    d = np.load(DATA_DIR / filename)
    T = d["T"]
    P = d["cluster_probs"]
    
    K = P.shape[1]
    
    for k in range(K):
        ax.plot(T, P[:, k], marker="o", linewidth=1.8, label=f"Cluster {k}")
    
    ax.set_title(name)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("P_T(cluster=k)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    
    if name == "Helicity-Contrastive":
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
    
    ax.legend(fontsize=8)

plt.show()

```
</details>

## 4. Latent Representation vs Temperature

To maintain a lightweight and reproducible notebook structure,
temperature-dependent latent statistics are computed using a standalone script:

`analysis/latent_vs_T.py`

This design separates:

- Core analysis logic (script)
- Visualization and comparison (notebook)

The notebook executes the script for each model and then loads
the generated figures for side-by-side comparison.

### Implementation Design

- Each model’s latent file is processed independently.
- Temperature-binned latent means are computed.
- Results are saved as PNG figures and optional CSV statistics.
- The notebook verifies successful generation before visualization.

This modular structure ensures:
- Reproducibility
- Clear experiment separation
- Clean pipeline design

![latent vs T](results/notebook_data/latent_vs_T_4models.png)
<details>
<summary>Show code: Cluster probability vs T visualization (AE / VAE / Contrastive / Helicity-Contrastive)</summary>
    
```python
import sys
import subprocess
from pathlib import Path

DATA_DIR = Path("results/notebook_data")
SCRIPT = Path("analysis/latent_vs_T.py")

latents = {
    "AE": DATA_DIR / "ae_latent.npz",
    "VAE": DATA_DIR / "vae_latent.npz",
    "Contrastive": DATA_DIR / "contrastive_latent_small.npz",
    "Helicity-Contrastive (Proposed)": DATA_DIR / "helicity_contrastive_latent_small.npz",
}

OUT_DIR = DATA_DIR
MAX_DIMS = 64

if not SCRIPT.exists():
    raise FileNotFoundError(f"Missing script: {SCRIPT}")

for name, fp in latents.items():
    if not fp.exists():
        raise FileNotFoundError(f"Missing latent for {name}: {fp}")

print("[OK] Inputs found. Running latent_vs_T.py ...")

for name, latent_path in latents.items():
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--latent", str(latent_path),
        "--out_dir", str(OUT_DIR),
        "--max_dims", str(MAX_DIMS),
    ]
    print(f"\n[RUN] {name}: {' '.join(cmd)}")
    res = subprocess.run(cmd, text=True, capture_output=True)

    if res.returncode != 0:
        print("\n[STDOUT]\n", res.stdout)
        print("\n[STDERR]\n", res.stderr)
        raise RuntimeError(f"latent_vs_T.py failed for {name}")

    
    if res.stdout:
        print("\n".join(res.stdout.splitlines()[-3:]))

print("\n[OK] Done.")

pngs = {
    "AE": OUT_DIR / "ae_latent_mean_vs_T.png",
    "VAE": OUT_DIR / "vae_latent_mean_vs_T.png",
    "Contrastive": OUT_DIR / "contrastive_latent_small_mean_vs_T.png",
    "Helicity-Contrastive (Proposed)": OUT_DIR / "helicity_contrastive_latent_small_mean_vs_T.png",
}

missing = [str(p) for p in pngs.values() if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Some expected plots were not generated:\n" + "\n".join(missing) +
        "\n\n(If your script uses different filenames, update the 'pngs' dict accordingly.)"
    )

# --- show 2x2 comparison ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
axes = axes.ravel()

for ax, (title, fp) in zip(axes, pngs.items()):
    ax.imshow(Image.open(fp))
    ax.set_title(title)
    ax.axis("off")

plt.show()

```
</details>


### ⚠ Dataset Size and Correlation Analysis

For GitHub compatibility, all notebook demonstrations use reduced latent files  
(`*_small.npz`, float32, ~3000 samples).

However, correlation analysis is sensitive to sample size.

Therefore:

- Spearman correlation coefficients were computed locally using the full dataset.
- Only the resulting ranked CSV files are included in this repository.
- The notebook loads these precomputed CSV files for visualization.

This approach balances:

- Repository size constraints
- Statistical reliability
- Reproducibility of visual results




## 5. Correlation with Physical Observables

## 6. Transition Sensitivity (Slope Analysis)

## 7. Comparison with Estimated Tc

## 8. Cluster-Conditioned Correlation Function

## 9. Conclusion

