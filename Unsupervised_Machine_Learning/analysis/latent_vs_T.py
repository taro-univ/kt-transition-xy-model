"""
latent_vs_T.py

Compute and plot temperature-conditional statistics of latent variables.

This script:
  1. Loads latent vectors z (N, D) and temperatures T (N,) from an NPZ file
     produced by latent_extraction.py.
  2. Groups samples by unique temperature values.
  3. Computes per-temperature mean and standard deviation of each latent dimension:
         mean_z[T, d] = < z_d >_T
         std_z[T, d]  = std(z_d)_T
  4. Saves the statistics to an NPZ file and plots mean vs temperature
     for the first few latent dimensions.

Typical use:
  - Inspect which latent dimensions behave like order-parameter proxies
  - Detect temperature regions where latent means shift (e.g., near KT regime)
  - Create quick diagnostic plots for reports / slides

Outputs:
  - *_stats_vs_T.npz   (T, mean_z, std_z)
  - *_mean_vs_T.png    (plot of mean_z[:, d] vs T for d < max_dims)
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path


# Add project root (two levels above) to sys.path
# This helps resolve imports when running as a standalone script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent",
        type=str,
        required=True,
        help="Path to NPZ created by latent_extraction.py (must contain 'z' and 'T').",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: same directory as latent file).",
    )
    parser.add_argument(
        "--max_dims",
        type=int,
        default=4,
        help="Maximum number of latent dimensions to plot (plots dims 0..max_dims-1).",
    )
    args = parser.parse_args()

    latent_path = Path(args.latent)
    if not latent_path.exists():
        raise FileNotFoundError(f"Latent file not found: {latent_path}")

    data = np.load(latent_path)
    if "z" not in data or "T" not in data:
        raise KeyError("NPZ file must contain both 'z' and 'T' arrays.")

    z = data["z"]  # shape (N, D)
    T = data["T"]  # shape (N,)

    N, D = z.shape
    print(f"[INFO] z shape = {z.shape}, T shape = {T.shape}")

    uniq_T = np.unique(T)
    uniq_T_sorted = np.sort(uniq_T)
    nT = len(uniq_T_sorted)

    mean_z = np.zeros((nT, D), dtype=np.float64)
    std_z = np.zeros((nT, D), dtype=np.float64)

    for i, t_val in enumerate(uniq_T_sorted):
        mask = (T == t_val)
        z_t = z[mask]
        mean_z[i] = z_t.mean(axis=0)
        std_z[i] = z_t.std(axis=0)
        print(f"[INFO] T={t_val}: n={z_t.shape[0]}")

    # Output directory
    if args.out_dir is None:
        out_dir = latent_path.parent
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save statistics
    stats_path = out_dir / f"{latent_path.stem}_stats_vs_T.npz"
    np.savez_compressed(
        stats_path,
        T=uniq_T_sorted,
        mean_z=mean_z,
        std_z=std_z,
    )
    print(f"[INFO] Saved stats to {stats_path}")

    # Plot mean latent values vs temperature
    max_dims = min(D, args.max_dims)
    if max_dims <= 0:
        raise ValueError("--max_dims must be >= 1")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for d in range(max_dims):
        # Keep the original style (h[1], h[2], ...) for the first four dimensions
        label = f"h[{d+1}]" if d < 4 else None

        ax.plot(
            uniq_T_sorted,
            mean_z[:, d],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=label,
        )

    ax.legend(
        loc="best",
        fontsize=10,
        frameon=True,
        handlelength=2.5
    )

    ax.set_xlabel("Temperature T")
    ax.set_ylabel(r"$\langle z_i \rangle_T$")
    ax.set_title("Latent Mean vs Temperature")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best", fontsize=9, frameon=True)

    plt.tight_layout()

    out_fig = out_dir / f"{latent_path.stem}_mean_vs_T.png"
    plt.savefig(out_fig, dpi=300)
    print(f"[INFO] Saved plot to {out_fig}")

    plt.close(fig)


if __name__ == "__main__":
    main()


"""
Example usage:

python Unsupervised_Machine_Learning/analysis/latent_vs_T.py \
    --latent Unsupervised_Machine_Learning/results/latent/autoencoder/autoencoder_latent.npz
"""
