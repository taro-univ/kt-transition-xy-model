"""
plot_selected_latent_vs_T.py

Plot temperature-conditional mean (and optional ±1σ) for a user-selected
subset of latent dimensions.

This script is a lightweight "focused" alternative to latent_vs_T.py:
instead of plotting the first few latent dimensions, you explicitly specify
which indices to visualize (e.g., those ranked highly by correlation analysis).

Workflow example:
  1) Run corr_ranking_latent_vs_observables.py to identify promising latent dims
  2) Plot those dims vs temperature using this script:
       python plot_selected_latent_vs_T.py --latent ... --dims "581,583,591" --with_std

Inputs
------
NPZ file containing:
  - z : (N, D) latent vectors
  - T : (N,)   temperatures per sample

Outputs
-------
  - *_selected_stats_vs_T.npz  (T_unique, mean_z, std_z, idx)
  - *_selected_mean_vs_T.png   (plot of mean vs T for specified dims)
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def parse_dims(s: str) -> list[int]:
    """
    Parse a comma-separated list of integer indices.

    Examples
    --------
    "581,583,591" -> [581, 583, 591]
    " 1, 2, 10 "  -> [1, 2, 10]

    Notes
    -----
    Spaces are allowed. Raises ValueError for invalid tokens.
    """
    s = s.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.replace(" ", "").split(",") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            raise ValueError(f"Cannot parse dim index: '{p}' in '{s}'")
    return out


def stats_vs_T(T: np.ndarray, z_cols: np.ndarray):
    """
    Compute temperature-conditional statistics for selected latent dimensions.

    Parameters
    ----------
    T : np.ndarray, shape (N,)
        Temperature per sample.
    z_cols : np.ndarray, shape (N, K)
        Selected latent dimensions.

    Returns
    -------
    Tu : np.ndarray, shape (nT,)
        Sorted unique temperatures.
    mean : np.ndarray, shape (nT, K)
        Mean of selected dims at each temperature.
    std : np.ndarray, shape (nT, K)
        Standard deviation of selected dims at each temperature.
    """
    T = np.asarray(T).reshape(-1)
    z_cols = np.asarray(z_cols)

    Tu = np.sort(np.unique(T))
    nT = Tu.size
    K = z_cols.shape[1]

    mean = np.zeros((nT, K), dtype=np.float64)
    std = np.zeros((nT, K), dtype=np.float64)

    for i, t in enumerate(Tu):
        m = (T == t)
        zz = z_cols[m]
        mean[i] = zz.mean(axis=0)
        std[i] = zz.std(axis=0)

    return Tu, mean, std


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--latent",
        type=str,
        required=True,
        help="NPZ containing 'z' (N,D) and 'T' (N,).",
    )

    ap.add_argument(
        "--dims",
        type=str,
        required=True,
        help="Comma-separated latent indices, e.g. '581,583,591'.",
    )

    ap.add_argument(
        "--label_prefix",
        type=str,
        default="z",
        help="Legend prefix (e.g., 'z' or 'h').",
    )

    ap.add_argument(
        "--one_based",
        action="store_true",
        help="If set, the legend uses 1-based indexing (e.g., h[1] instead of z[0]).",
    )

    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: same as latent file).",
    )

    ap.add_argument(
        "--title",
        type=str,
        default="Selected latent mean vs T",
        help="Plot title.",
    )

    ap.add_argument(
        "--ylabel",
        type=str,
        default="⟨z_i⟩_T",
        help="Y-axis label (can be LaTeX-like text).",
    )

    ap.add_argument(
        "--with_std",
        action="store_true",
        help="If set, add ±1σ shading around the mean curves.",
    )

    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figure.",
    )

    args = ap.parse_args()

    latent_path = Path(args.latent)
    if not latent_path.exists():
        raise FileNotFoundError(f"Latent file not found: {latent_path}")

    dims = parse_dims(args.dims)
    if len(dims) == 0:
        raise ValueError("No dims provided. Example: --dims '581,583,591'")

    data = np.load(latent_path)
    if "z" not in data or "T" not in data:
        raise KeyError("NPZ must contain both 'z' and 'T'.")

    z = np.asarray(data["z"])
    T = np.asarray(data["T"]).reshape(-1)

    if z.ndim != 2:
        raise ValueError(f"z must be 2D (N,D). got {z.shape}")

    N, D = z.shape
    if T.shape[0] != N:
        raise ValueError(f"T length mismatch: T={T.shape}, z={z.shape}")

    # Bounds check for selected indices
    for d in dims:
        if d < 0 or d >= D:
            raise ValueError(f"Index out of range: {d} (D={D})")

    z_sel = z[:, dims]  # shape (N, K)
    Tu, mean, std = stats_vs_T(T, z_sel)

    # Output directory
    out_dir = Path(args.out_dir) if args.out_dir else latent_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save statistics for reproducibility
    stats_path = out_dir / f"{latent_path.stem}_selected_stats_vs_T.npz"
    np.savez_compressed(
        stats_path,
        T=Tu,
        mean_z=mean,
        std_z=std,
        idx=np.array(dims, dtype=int),
    )
    print(f"[INFO] Saved stats: {stats_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    K = len(dims)

    for j in range(K):
        idx = dims[j]
        shown = (j + 1) if args.one_based else idx
        label = f"{args.label_prefix}[{shown}]"

        ax.plot(
            Tu,
            mean[:, j],
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=label,
        )

        if args.with_std:
            ax.fill_between(
                Tu,
                mean[:, j] - std[:, j],
                mean[:, j] + std[:, j],
                alpha=0.15,
            )

    ax.set_xlabel("Temperature T")
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=10, frameon=True)

    plt.tight_layout()

    out_fig = out_dir / f"{latent_path.stem}_selected_mean_vs_T.png"
    plt.savefig(out_fig, dpi=args.dpi)
    plt.close(fig)

    print(f"[INFO] Saved figure: {out_fig}")


if __name__ == "__main__":
    main()
