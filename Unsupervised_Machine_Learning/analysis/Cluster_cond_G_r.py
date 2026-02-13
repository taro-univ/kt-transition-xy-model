"""
cluster_cond_G_r.py

Compute and visualize cluster-conditional two-point correlation functions G(r)
for the 2D XY model.

This script:
    1. Loads cluster labels (from latent-space clustering results).
    2. Loads Monte Carlo configurations (phi angles).
    3. Computes the spin-spin correlation function:
           G(r) = < s(0) · s(r) >
       using FFT-based convolution under periodic boundary conditions.
    4. Performs radial averaging.
    5. Plots cluster-conditional mean G(r) at selected temperatures.

Designed for:
    - KT transition analysis
    - Power-law vs exponential decay comparison
    - Cluster-phase structural validation
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# Predefined cluster colors for consistent visualization
cluster_colors = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
}


def get_key(d, candidates):
    """
    Utility function to retrieve a valid key from an npz file.

    Parameters
    ----------
    d : numpy.lib.npyio.NpzFile
        Loaded npz object.
    candidates : list[str]
        List of possible key names.

    Returns
    -------
    str
        First matching key found.

    Raises
    ------
    KeyError
        If none of the candidate keys are present.
    """
    for k in candidates:
        if k in d.files:
            return k
    raise KeyError(f"None of keys found: {candidates}. Available: {d.files}")


def autocorr2_fft(sx, sy):
    """
    Compute spin-spin autocorrelation using FFT.

    Parameters
    ----------
    sx, sy : ndarray of shape (L, L)
        Spin components:
            s_x = cos(phi)
            s_y = sin(phi)

    Returns
    -------
    corr : ndarray of shape (L, L)
        Correlation function:
            corr(dx, dy) = < s(i) · s(i + r) >
        with periodic boundary conditions.

    Notes
    -----
    Computed using:
        F^{-1}[ F(s) * conj(F(s)) ] / L^2
    exploiting convolution theorem.
    """
    L = sx.shape[0]

    fx = np.fft.fft2(sx)
    fy = np.fft.fft2(sy)

    cx = np.fft.ifft2(fx * np.conj(fx)).real / (L * L)
    cy = np.fft.ifft2(fy * np.conj(fy)).real / (L * L)

    return cx + cy


def radial_average_from_corr(corr, max_r=None):
    """
    Perform radial averaging of 2D correlation function.

    Parameters
    ----------
    corr : ndarray (L, L)
        Correlation in displacement space.
    max_r : int or None
        Maximum radius to compute. Default: L//2.

    Returns
    -------
    r_vals : ndarray
        Radii (1 ... max_r).
    G : ndarray
        Radially averaged correlation G(r).

    Notes
    -----
    Bins correlation values according to rounded radius:
        r = sqrt(dx^2 + dy^2)
    """
    L = corr.shape[0]

    if max_r is None:
        max_r = L // 2

    dx = np.arange(L)
    dx = np.where(dx <= L // 2, dx, dx - L)

    dy = np.arange(L)
    dy = np.where(dy <= L // 2, dy, dy - L)

    DX, DY = np.meshgrid(dx, dy, indexing="ij")
    R = np.sqrt(DX * DX + DY * DY)

    r_bin = np.rint(R).astype(int)

    mask = (r_bin >= 1) & (r_bin <= max_r)

    r_vals = np.arange(1, max_r + 1)
    G = np.full_like(r_vals, np.nan, dtype=float)

    for i, r in enumerate(r_vals):
        m = mask & (r_bin == r)
        if np.any(m):
            G[i] = corr[m].mean()

    return r_vals, G


def compute_G_r_from_phi(phi, max_r=None):
    """
    Compute radial correlation G(r) from XY spin angles.

    Parameters
    ----------
    phi : ndarray (L, L)
        Spin angle configuration.
    max_r : int or None
        Maximum radius.

    Returns
    -------
    r : ndarray
        Radii.
    G : ndarray
        Radially averaged correlation.
    """
    sx = np.cos(phi)
    sy = np.sin(phi)

    corr = autocorr2_fft(sx, sy)
    r, G = radial_average_from_corr(corr, max_r=max_r)

    return r, G


def main():
    """
    Main CLI execution.

    Example
    -------
    python cluster_cond_G_r.py \
        --cluster_stats results/cluster_vs_T.npz \
        --mc_npz data/mc_samples.npz \
        --temps 0.80 1.00 1.20 \
        --show_std
    """
    ap = argparse.ArgumentParser()

    ap.add_argument("--cluster_stats", type=str, required=True,
                    help="NPZ output from cluster_vs_T.py (contains labels + temperatures).")

    ap.add_argument("--mc_npz", type=str, required=True,
                    help="Monte Carlo NPZ containing phi configurations and temperatures.")

    ap.add_argument("--temps", type=float, nargs="+",
                    default=[0.80, 1.00, 1.20],
                    help="Temperatures to visualize.")

    ap.add_argument("--tol", type=float, default=1e-8,
                    help="Tolerance for floating-point temperature matching.")

    ap.add_argument("--max_r", type=int, default=None,
                    help="Maximum radius (default L//2).")

    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output directory for figure.")

    ap.add_argument("--show_std", action="store_true",
                    help="If set, plot mean ± std shading.")

    args = ap.parse_args()

    cs = np.load(args.cluster_stats)
    mc = np.load(args.mc_npz)

    labels = cs[get_key(cs, ["labels"])]
    T_samples = cs[get_key(cs, ["T_samples", "T", "temps", "T_latent"])]

    phi = mc[get_key(mc, ["phi"])]
    T_mc = mc[get_key(mc, ["T", "temps", "T_samples"])]

    if len(labels) != len(T_samples) or len(T_mc) != len(labels) or len(phi) != len(labels):
        raise ValueError("Sample alignment mismatch between cluster and MC data.")

    K = int(labels.max()) + 1
    L = phi.shape[1]
    max_r = args.max_r if args.max_r is not None else L // 2

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.cluster_stats).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    temps = args.temps
    fig, axes = plt.subplots(len(temps), 2,
                             figsize=(12, 3.6 * len(temps)),
                             squeeze=False)

    for row, T0 in enumerate(temps):

        maskT = np.abs(T_samples - T0) <= args.tol
        idxT = np.where(maskT)[0]

        if idxT.size == 0:
            maskT = np.isclose(np.round(T_samples, 2), np.round(T0, 2))
            idxT = np.where(maskT)[0]

        if idxT.size == 0:
            raise ValueError(f"No samples found for T={T0}")

        for col in range(2):
            ax = axes[row, col]
            ax.grid(True)
            ax.set_xlabel("r")
            ax.set_ylabel("G(r)")
            ax.set_title(f"G(r) at T={T0:.2f} "
                         f"({'log-log' if col==0 else 'semi-log'})")

        for k in range(K):
            idx = idxT[labels[idxT] == k]
            if idx.size == 0:
                continue

            G_list = []
            r_ref = None

            for i in idx:
                r, G = compute_G_r_from_phi(phi[i], max_r=max_r)
                if r_ref is None:
                    r_ref = r
                G_list.append(G)

            G_arr = np.vstack(G_list)

            G_mean = np.nanmean(G_arr, axis=0)
            G_std = np.nanstd(G_arr, axis=0)

            mask_pos = (G_mean > 0) & np.isfinite(G_mean)

            color = cluster_colors.get(k, "black")

            ax1 = axes[row, 0]
            ax1.plot(r_ref[mask_pos],
                     G_mean[mask_pos],
                     marker="o",
                     color=color,
                     label=f"cluster {k} (n={idx.size})")

            ax1.set_xscale("log")
            ax1.set_yscale("log")

            if args.show_std:
                lo = np.clip(G_mean - G_std, 1e-12, None)
                hi = G_mean + G_std
                m = (hi > 0) & np.isfinite(hi) & np.isfinite(lo)

                ax1.fill_between(r_ref[m], lo[m], hi[m],
                                 color=color, alpha=0.2)

            ax2 = axes[row, 1]
            ax2.plot(r_ref[mask_pos],
                     G_mean[mask_pos],
                     marker="o",
                     label=f"cluster {k} (n={idx.size})")

            ax2.set_yscale("log")

            if args.show_std:
                ax2.fill_between(r_ref[m], lo[m], hi[m], alpha=0.2)

        axes[row, 0].legend()
        axes[row, 1].legend()

    plt.tight_layout()

    out_path = out_dir / "cluster_cond_G_r_T080_T100_T120.png"
    plt.savefig(out_path, dpi=300)

    print("[OK] saved:", out_path)


if __name__ == "__main__":
    main()
