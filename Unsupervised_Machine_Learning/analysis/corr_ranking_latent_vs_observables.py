"""
corr_ranking_latent_vs_observables.py

Rank latent dimensions by correlation with physical observables.

This script computes correlation scores between each latent dimension z[:, d]
and one or more physical observables stored in a Monte Carlo (MC) NPZ file
(e.g., vortex density nv, helicity modulus Y).

Key features:
  - Supports Pearson and Spearman correlations (no SciPy dependency).
  - Flexible alignment:
      * sample: assumes latent NPZ and MC NPZ share the same sample ordering.
      * temperature: aggregates both latent variables and observables by temperature,
                    then correlates across temperature points.
      * auto: uses sample alignment when possible, otherwise falls back to temperature.
  - Exports:
      * full ranking CSV
      * top-k ranking CSV
      * top-k bar plot PNG
  - Writes a small metadata text file for reproducibility.

Outputs (per method & target):
  - corr_{method}_{target}_full.csv
  - corr_{method}_{target}_topK.csv
  - corr_{method}_{target}_topK.png
  - corr_run_meta.txt
"""

from pathlib import Path, PurePosixPath

import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt


# Add project root to sys.path (same pattern as other analysis scripts)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------
# Correlation utilities
# -------------------------

def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson correlation coefficient.

    Returns
    -------
    float
        Pearson correlation in [-1, 1]. Returns 0.0 if input is invalid,
        lengths mismatch, or either variance is zero.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return 0.0

    x = x - x.mean()
    y = y - y.mean()

    vx = np.dot(x, x)
    vy = np.dot(y, y)

    if vx <= 0.0 or vy <= 0.0:
        return 0.0

    return float(np.dot(x, y) / np.sqrt(vx * vy))


def _rankdata(a: np.ndarray) -> np.ndarray:
    """
    Rank transform with average ranks for ties (SciPy-free).

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Ranks (1..N) with average ranks assigned to ties.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    n = a.size

    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    # Assign average ranks for ties
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1

    return ranks


def _safe_spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman correlation coefficient via rank transform.

    Returns
    -------
    float
        Spearman correlation in [-1, 1]. Returns 0.0 if variance is zero.
    """
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _safe_pearsonr(rx, ry)


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    """
    Dispatch correlation method.

    Parameters
    ----------
    method : str
        "pearson" or "spearman".
    """
    method = method.lower().strip()

    if method == "pearson":
        return _safe_pearsonr(x, y)

    if method == "spearman":
        return _safe_spearmanr(x, y)

    raise ValueError(f"Unknown method: {method} (use pearson/spearman)")


# -------------------------
# Alignment helpers
# -------------------------

def _almost_equal_T(a: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check whether two temperature arrays match elementwise within tolerance.
    Used to decide if 'sample' alignment is valid in --aggregate auto mode.
    """
    if a.shape != b.shape:
        return False
    return bool(np.all(np.abs(a - b) <= tol))


def _aggregate_by_temperature(T: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate X by temperature groups (mean over samples at identical T).

    Parameters
    ----------
    T : np.ndarray, shape (N,)
        Temperatures per sample.
    X : np.ndarray, shape (N,) or (N, D)
        Values to aggregate.

    Returns
    -------
    (Tu, X_mean) : (np.ndarray, np.ndarray)
        Tu: sorted unique temperatures
        X_mean: mean values per temperature (shape (nT,) or (nT, D))
    """
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)

    if X.shape[0] != T.shape[0]:
        raise ValueError(f"Length mismatch for aggregation: T={T.shape}, X={X.shape}")

    Tu = np.unique(T)
    Tu = np.sort(Tu)

    if X.ndim == 1:
        out = np.zeros((Tu.size,), dtype=np.float64)
        for i, t in enumerate(Tu):
            mask = (T == t)
            out[i] = X[mask].mean()
        return Tu, out

    if X.ndim == 2:
        D = X.shape[1]
        out = np.zeros((Tu.size, D), dtype=np.float64)
        for i, t in enumerate(Tu):
            mask = (T == t)
            out[i] = X[mask].mean(axis=0)
        return Tu, out

    raise ValueError(f"X must be 1D or 2D, got shape={X.shape}")


# -------------------------
# Main
# -------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--latent_npz",
        type=str,
        required=True,
        help="NPZ containing 'z' (N,D) and 'T' (N,) "
             "(e.g., helicity_contrastive_latent.npz).",
    )

    p.add_argument(
        "--mc_npz",
        type=str,
        required=True,
        help="MC NPZ containing physical observables "
             "(e.g., ml_xy_L32_1000samples.npz).",
    )

    p.add_argument(
        "--targets",
        type=str,
        default="nv,Y",
        help="Comma-separated keys in mc_npz to correlate with (default: nv,Y).",
    )

    p.add_argument(
        "--method",
        type=str,
        default="pearson,spearman",
        help="Comma-separated list: pearson,spearman (default: both).",
    )

    p.add_argument(
        "--aggregate",
        type=str,
        default="auto",
        choices=["auto", "sample", "temperature"],
        help="Alignment mode before correlation: "
             "'sample' assumes identical sample order across NPZ files; "
             "'temperature' aggregates by T then correlates across temperatures; "
             "'auto' uses 'sample' if possible otherwise 'temperature'.",
    )

    p.add_argument("--topk", type=int, default=10, help="Top-k latent dims to show in bar plots.")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory (default: latent_dir/corr_ranking).")
    p.add_argument("--abs_sort", action="store_true", help="Sort by absolute correlation magnitude.")
    p.add_argument("--T_tol", type=float, default=1e-8, help="Tolerance for sample-wise temperature matching.")

    args = p.parse_args()

    latent_path = Path(args.latent_npz)
    mc_path = Path(args.mc_npz)

    if not latent_path.exists():
        raise FileNotFoundError(f"latent_npz not found: {latent_path}")

    if not mc_path.exists():
        raise FileNotFoundError(f"mc_npz not found: {mc_path}")

    ld = np.load(latent_path)
    if "z" not in ld or "T" not in ld:
        raise KeyError("latent_npz must contain both 'z' and 'T'")

    z = np.asarray(ld["z"], dtype=np.float64)
    Tz = np.asarray(ld["T"], dtype=np.float64).reshape(-1)

    if z.ndim != 2:
        raise ValueError(f"z must be 2D (N,D), got {z.shape}")

    N, D = z.shape
    if Tz.shape[0] != N:
        raise ValueError(f"latent T length mismatch: z={z.shape}, T={Tz.shape}")

    md = np.load(mc_path)
    Tm = np.asarray(md["T"], dtype=np.float64).reshape(-1) if "T" in md else None

    target_keys = [s.strip() for s in args.targets.split(",") if s.strip()]
    methods = [s.strip() for s in args.method.split(",") if s.strip()]

    # Output directory
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = latent_path.parent / "corr_ranking"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide alignment mode
    agg = args.aggregate
    if agg == "auto":
        # Sample alignment is valid if MC provides T, lengths match, and sequences match within tolerance.
        if Tm is not None and Tm.shape[0] == N and _almost_equal_T(Tz, Tm, tol=float(args.T_tol)):
            agg = "sample"
        else:
            agg = "temperature"

    # Prepare X (latent) and y (targets) according to alignment
    if agg == "sample":
        # Require each target to be sample-aligned with N
        for k in target_keys:
            if k not in md:
                raise KeyError(f"target '{k}' not found in mc_npz keys={list(md.files)}")
            if np.asarray(md[k]).reshape(-1).shape[0] != N:
                raise ValueError(
                    f"sample alignment requested but target '{k}' length != N. "
                    f"len={np.asarray(md[k]).reshape(-1).shape[0]} vs N={N}. "
                    "Use --aggregate temperature."
                )

        X = z
        T_used = Tz
        y_dict = {k: np.asarray(md[k], dtype=np.float64).reshape(-1) for k in target_keys}

    else:
        # Aggregate latent by latent temperatures
        Tu_z, z_mean = _aggregate_by_temperature(Tz, z)

        y_dict = {}
        for k in target_keys:
            if k not in md:
                raise KeyError(f"target '{k}' not found in mc_npz keys={list(md.files)}")

            if Tm is None:
                # Rare fallback: aggregate using latent T
                Tu_y, y_mean = _aggregate_by_temperature(Tz, np.asarray(md[k], dtype=np.float64).reshape(-1))
            else:
                y_arr = np.asarray(md[k], dtype=np.float64).reshape(-1)
                if y_arr.shape[0] != Tm.shape[0]:
                    raise ValueError(f"mc target '{k}' length mismatch with mc T: y={y_arr.shape}, Tm={Tm.shape}")
                Tu_y, y_mean = _aggregate_by_temperature(Tm, y_arr)

            # Intersect temperature grids and align
            Tu_common = np.intersect1d(Tu_z, Tu_y)
            if Tu_common.size < 2:
                raise ValueError(f"Too few common temperatures between latent and target '{k}'.")

            idx_z = np.searchsorted(Tu_z, Tu_common)
            idx_y = np.searchsorted(Tu_y, Tu_common)

            y_dict[k] = y_mean[idx_y]

            # X and T_used are shared across targets, so set them once
            if k == target_keys[0]:
                X = z_mean[idx_z]
                T_used = Tu_common

    # Compute correlations and export results
    for method in methods:
        for target in target_keys:
            y = y_dict[target]

            corr_vals = np.zeros((D,), dtype=np.float64)
            for d in range(D):
                corr_vals[d] = _corr(X[:, d], y, method=method)

            # Sort dimensions by correlation (optionally by abs)
            if args.abs_sort:
                order = np.argsort(-np.abs(corr_vals))
            else:
                order = np.argsort(-corr_vals)

            # Full ranking CSV
            full_csv = out_dir / f"corr_{method}_{target}_full.csv"
            with open(full_csv, "w", encoding="utf-8") as f:
                f.write("rank,dim,corr,abs_corr\n")
                for r, d in enumerate(order, start=1):
                    c = float(corr_vals[d])
                    f.write(f"{r},{d},{c:.8f},{abs(c):.8f}\n")

            # Top-k CSV
            K = min(int(args.topk), D)
            top_csv = out_dir / f"corr_{method}_{target}_top{K}.csv"
            with open(top_csv, "w", encoding="utf-8") as f:
                f.write("rank,dim,corr,abs_corr\n")
                for r in range(K):
                    d = int(order[r])
                    c = float(corr_vals[d])
                    f.write(f"{r+1},{d},{c:.8f},{abs(c):.8f}\n")

            # Top-k bar plot
            top_dims = [int(order[i]) for i in range(K)]
            top_corr = np.array([corr_vals[d] for d in top_dims], dtype=np.float64)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(np.arange(K), top_corr)
            ax.set_xticks(np.arange(K))
            ax.set_xticklabels([f"z[{d}]" for d in top_dims], rotation=45, ha="right")
            ax.set_ylabel(f"{method} corr")
            ax.set_title(f"Top-{K} {method} corr: latent vs {target}  (agg={agg})")
            ax.grid(True, axis="y", alpha=0.3)

            plt.tight_layout()

            out_fig = out_dir / f"corr_{method}_{target}_top{K}.png"
            plt.savefig(out_fig, dpi=300)
            plt.close(fig)

            print(f"[INFO] Saved {method} correlation ranking for target='{target}'")
            print(f"       full: {full_csv.name}")
            print(f"       top : {top_csv.name}")
            print(f"       fig : {out_fig.name}")

    # Save metadata for reproducibility
    meta_path = out_dir / "corr_run_meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"latent_npz: {latent_path}\n")
        f.write(f"mc_npz: {mc_path}\n")
        f.write(f"targets: {target_keys}\n")
        f.write(f"methods: {methods}\n")
        f.write(f"aggregate: {agg}\n")
        f.write(f"abs_sort: {args.abs_sort}\n")
        f.write(f"T_used_len: {len(T_used)}\n")
        f.write(f"T_used_minmax: {float(np.min(T_used))} .. {float(np.max(T_used))}\n")
        f.write(f"z_shape_used: {X.shape}\n")

    print(f"[INFO] Wrote meta: {meta_path}")


if __name__ == "__main__":
    main()
