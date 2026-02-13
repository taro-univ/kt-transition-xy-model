"""
kmeans_tsne.py

Visualize latent representations with t-SNE and compare:
  (1) Unsupervised clusters from K-means in the ORIGINAL latent space (high-D z)
  (2) A simple 3-phase labeling defined by temperature thresholds (T-based rule)

This script:
  - Loads an NPZ file containing latent vectors z and temperatures T.
  - Optionally subsamples for faster t-SNE (deterministic with --seed).
  - Runs K-means on z (high-dimensional), then visualizes the labels on a 2D t-SNE embedding.
  - Creates a second plot using 3-phase labels derived from (t1, t2) thresholds on T.

Intended use:
  - Qualitative inspection of phase separation in latent space
  - Sanity-checking whether unsupervised clustering aligns with temperature-defined regimes
  - Producing presentation-friendly scatter plots

Outputs:
  - *_kmeansZ_tsne_*.png  : K-means labels (computed on high-D z) shown on t-SNE plane
  - *_tsne_3phase_*.png   : Temperature-threshold labels shown on t-SNE plane
  - Optional NPZ with 2D coordinates and labels if --save_npz is set
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def label_from_temperature(T: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """
    Assign 3-phase labels using temperature thresholds:

        0: low-T phase         (T < t1)
        1: mid / KT-near phase (t1 <= T < t2)
        2: high-T phase        (T >= t2)

    Parameters
    ----------
    T : np.ndarray
        Temperature array of shape (N,).
    t1 : float
        Threshold between low and mid phase.
    t2 : float
        Threshold between mid and high phase.

    Returns
    -------
    np.ndarray
        Integer labels in {0, 1, 2}, shape (N,).
    """
    labels = np.zeros_like(T, dtype=int)
    labels[(T >= t1) & (T < t2)] = 1
    labels[(T >= t2)] = 2
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent",
        type=str,
        required=True,
        help="Path to latent NPZ (must include z; requires T for 3-phase plot).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: same directory as latent file).",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity parameter.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Maximum number of samples for t-SNE (subsample if larger).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (subsampling, t-SNE, and K-means).",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=3,
        help="Number of K-means clusters (run on ORIGINAL high-D latent z).",
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=0.900,
        help="3-phase threshold t1 (low/mid boundary).",
    )
    parser.add_argument(
        "--t2",
        type=float,
        default=1.150,
        help="3-phase threshold t2 (mid/high boundary).",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=6.0,
        help="Scatter point size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.90,
        help="Scatter alpha (transparency).",
    )
    parser.add_argument(
        "--no_axes",
        action="store_true",
        help="Remove axis ticks/frames (presentation-friendly).",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Also save t-SNE coordinates and labels as NPZ.",
    )
    args = parser.parse_args()

    latent_path = Path(args.latent)
    if not latent_path.exists():
        raise FileNotFoundError(f"Latent file not found: {latent_path}")

    out_dir = Path(args.out_dir) if args.out_dir else latent_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading: {latent_path}")
    data = np.load(latent_path)

    # --- Load z ---
    if "z" in data:
        z = data["z"]
    else:
        first_key = list(data.files)[0]
        z = data[first_key]
        print(f"[WARN] Key 'z' not found. Using first array instead: '{first_key}'")

    # --- Load T (required to preserve the original script behavior) ---
    if "T" not in data:
        raise KeyError("This script requires temperature array 'T' in the NPZ for the 3-phase plot.")
    T = data["T"]

    if z.shape[0] != T.shape[0]:
        raise ValueError(f"Length mismatch: z has N={z.shape[0]} but T has N={T.shape[0]}")

    N = z.shape[0]
    print(f"[INFO] z shape: {z.shape}, N={N}")

    # --- Subsampling (deterministic) ---
    rs = np.random.RandomState(args.seed)
    if N > args.n_samples:
        idx = rs.choice(N, size=args.n_samples, replace=False)
        z_sub = z[idx]
        T_sub = T[idx]
        print(f"[INFO] Subsampled: {args.n_samples}/{N}")
    else:
        idx = np.arange(N)
        z_sub = z
        T_sub = T

    # --- K-means on ORIGINAL latent space (high-D z) ---
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError("scikit-learn is required. Install: pip install scikit-learn") from e

    print(f"[INFO] Running K-means on ORIGINAL z (k={args.n_clusters}) ...")
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        n_init=20,
    )
    kmeans_labels = kmeans.fit_predict(z_sub)

    # --- t-SNE for visualization (computed from z_sub) ---
    try:
        from sklearn.manifold import TSNE
    except ImportError as e:
        raise ImportError("scikit-learn is required. Install: pip install scikit-learn") from e

    print("[INFO] Running t-SNE ...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
    )
    z_2d = tsne.fit_transform(z_sub)

    # --- 3-phase labels by temperature thresholds ---
    phase_labels = label_from_temperature(T_sub, t1=args.t1, t2=args.t2)

    stem = latent_path.stem
    common = f"p{args.perplexity:g}_n{len(z_sub)}_seed{args.seed}"

    # ========== (1) K-means(z) labels visualized on t-SNE plane ==========
    fig, ax = plt.subplots(figsize=(6, 5))

    # Fixed discrete colors for clusters (assumes up to 3 clusters by default)
    cluster_cmap = ListedColormap(["blue", "orange", "green"])

    ax.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=kmeans_labels,
        cmap=cluster_cmap,
        s=args.point_size,
        alpha=args.alpha,
    )
    ax.set_title(f"K-means (on z) + t-SNE (k={args.n_clusters})")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    if args.no_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    out1 = out_dir / f"{stem}_kmeansZ_tsne_k{kmeans.n_clusters}_{common}.png"
    fig.savefig(out1, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved: {out1}")

    # ========== (2) Temperature-threshold 3-phase labels on t-SNE plane ==========
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=phase_labels,
        cmap="tab10",
        s=args.point_size,
        alpha=args.alpha,
    )
    ax.set_title(f"t-SNE + 3-phase (t1={args.t1:g}, t2={args.t2:g})")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    if args.no_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    out2 = out_dir / f"{stem}_tsne_3phase_t1{args.t1:g}_t2{args.t2:g}_{common}.png"
    fig.savefig(out2, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved: {out2}")

    # --- Optional: save computed coordinates + labels for downstream use ---
    if args.save_npz:
        out_npz = out_dir / f"{stem}_kmeansZ_tsne2d_labels_{common}.npz"
        np.savez(
            out_npz,
            idx=idx,  # indices into the original dataset (useful if subsampled)
            z_2d=z_2d,
            T=T_sub,
            kmeans_labels=kmeans_labels,
            phase_labels=phase_labels,
            t1=args.t1,
            t2=args.t2,
            perplexity=args.perplexity,
            n_clusters=args.n_clusters,
            seed=args.seed,
        )
        print(f"[INFO] Saved: {out_npz}")


if __name__ == "__main__":
    main()
