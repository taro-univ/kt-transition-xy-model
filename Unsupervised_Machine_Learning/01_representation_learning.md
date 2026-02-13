# Representation Learning for the 2D XY Model

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


## 4. Cluster Structure vs Temperature

## 5. Latent vs Temperature

## 6. Correlation with Physical Observables

## 7. Transition Sensitivity (Slope Analysis)

## 8. Comparison with Estimated Tc

## 9. Cluster-Conditioned Correlation Function

## 10. Conclusion

