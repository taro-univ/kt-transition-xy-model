"""
loop.py

High-level simulation loop for the 2D XY model (KT transition).

Main entry points
-----------------
- run_at_temperature(cfg, T, ...)
- simulate_temperature_sweep(cfg, temps, ...)

This module orchestrates:
initialization -> thermalization -> measurement -> saving lightweight results.

Notes
-----
- Full snapshots are kept in memory (RunBuffers.snaps) and are NOT written to JSON by default.
- For reproducibility, the config is always written to `config.json` in the run directory.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict, Any
import json
import time
import numpy as np

from MCsim.config import Config
from MCsim.xy_model import XYState, RunBuffers
from MCsim.update import (
    total_energy,
    metropolis_sweep,
    overrelaxation_sweep,
    wolff_cluster_step,
    autotune_delta_phi,
)
from MCsim.measure import (
    energy_density,
    helicity_modulus,
    accumulate_correlation,
    finalize_correlation,
    reset_correlation,
    detect_vortices,
    HeatCapacityAccumulator,
)


def measure_all(
    state: XYState,
    bufs: RunBuffers,
    beta: float,
    J: float = 1.0,
    save_snap: bool = False,
) -> None:
    """
    Measure core observables, append to time-series, accumulate correlations,
    and optionally store snapshots.
    """
    Y = helicity_modulus(state.phi, beta=beta, J=J)
    bufs.tseries.Y.append(float(Y))

    en = energy_density(state.phi, J=J)
    bufs.tseries.E.append(float(en))

    vort, nv = detect_vortices(state.phi)
    bufs.tseries.nv.append(float(nv))

    if save_snap:
        bufs.snaps.angles.append(state.phi.copy())
        bufs.snaps.vortices.append(vort.copy())

    accumulate_correlation(bufs, state, n_refs=16)


def run_at_temperature(
    cfg: Config,
    T: float,
    state: Optional[XYState] = None,
    continue_from_prev: bool = True,
    use_wolff: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one temperature point: thermalize -> measure.

    Returns
    -------
    dict
        Includes time-series, correlation, heat capacity estimate, and the final state
        (returned as 'state' for chaining to the next temperature; not JSON-serializable).
    """
    beta = 1.0 / float(T)
    J = float(cfg.J)
    L = int(cfg.L)

    # Initialize / reset
    if state is None:
        state = XYState.create(L=L, seed=cfg.seed, init="random")
    elif not continue_from_prev:
        state.phi[:] = state.rng.uniform(-np.pi, np.pi, size=(L, L)).astype(np.float32)

    bufs = RunBuffers.create(L=L)
    hc = HeatCapacityAccumulator(L=L)

    use_over = bool(cfg.use_overrelax)
    use_wolff = bool(cfg.use_wolff) if use_wolff is None else bool(use_wolff)

    # Proposal width (Metropolis)
    dphi = float(cfg.delta_phi_init)
    tgt_min, tgt_max = float(cfg.target_accept_min), float(cfg.target_accept_max)

    # ---- Thermalization ----
    acc_rate = 0.0
    therm_start = time.time()
    for _ in range(int(cfg.sweeps_therm)):
        acc_rate = float(metropolis_sweep(state, beta=beta, delta_phi=dphi, J=J))
        if use_over:
            overrelaxation_sweep(state, J=J)
        if use_wolff:
            wolff_cluster_step(state, beta=beta, J=J)
        dphi = float(autotune_delta_phi(dphi, acc_rate, tgt_min, tgt_max))
    therm_time = float(time.time() - therm_start)

    # ---- Measurement ----
    reset_correlation(bufs)
    meas_start = time.time()
    accepts: List[float] = []

    # snapshot frequency: every N "measurement events" (not every sweep)
    snap_every = int(cfg.save_snapshots_every)
    if snap_every <= 0:
        snap_every = 0

    meas_count = 0
    for s in range(int(cfg.sweeps_meas)):
        acc_rate = float(metropolis_sweep(state, beta=beta, delta_phi=dphi, J=J))
        accepts.append(acc_rate)

        if use_over:
            overrelaxation_sweep(state, J=J)
        if use_wolff:
            wolff_cluster_step(state, beta=beta, J=J)

        if (s % int(cfg.sample_interval)) == 0:
            save_snap = (snap_every > 0) and (meas_count % snap_every == 0)
            measure_all(state, bufs, beta=beta, J=J, save_snap=save_snap)
            hc.add(total_energy(state.phi, J=J))
            meas_count += 1

    meas_time = float(time.time() - meas_start)

    # Finalize correlation
    r, G = finalize_correlation(bufs)

    result: Dict[str, Any] = {
        "T": float(T),
        "beta": float(beta),
        "L": int(L),
        "dphi_last": float(dphi),
        "therm_time_sec": therm_time,
        "meas_time_sec": meas_time,
        "time_series": {
            "E": bufs.tseries.E,
            "Y": bufs.tseries.Y,
            "nv": bufs.tseries.nv,
            "accept": accepts,
        },
        "correlation": {
            "r": r.tolist(),
            "G": G.tolist(),
        },
        "C": float(hc.value(beta)),
        "snapshots": bufs.snaps,  # kept in-memory (not saved to JSON by default)
        "state": state,           # returned for chaining (not JSON-serializable)
    }

    if verbose:
        y_mean = float(np.mean(bufs.tseries.Y)) if bufs.tseries.Y else float("nan")
        nv_mean = float(np.mean(bufs.tseries.nv)) if bufs.tseries.nv else float("nan")
        acc_mean = float(np.mean(accepts)) if accepts else float("nan")
        print(
            f"[T={T:.3f}] therm {therm_time:.1f}s, meas {meas_time:.1f}s, "
            f"Y~{y_mean:.3f}, nv~{nv_mean:.3f}, dphi={dphi:.3f}, acc~{acc_mean:.2f}"
        )

    return result


def simulate_temperature_sweep(
    cfg: Config,
    temps: Optional[Iterable[float]] = None,
    resume_state: Optional[XYState] = None,
    save_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], XYState, Path]:
    """
    Run a temperature sweep. By default, the final state at T_k is used as the initial
    state for T_{k+1} (simulated annealing style).

    Returns
    -------
    results : list[dict]
        Per-temperature results (includes non-serializable 'state' key).
    last_state : XYState
        Final state after the sweep.
    save_dir : Path
        Directory where lightweight JSON files are written.
    """
    temps_list = cfg.temps if temps is None else list(temps)
    state = resume_state

    # Output directory
    if save_dir is None:
        save_dir_path = Path(cfg.materialize_out_dir())
    else:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with (save_dir_path / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    results: List[Dict[str, Any]] = []
    for T in temps_list:
        res = run_at_temperature(cfg, float(T), state=state, continue_from_prev=True, verbose=verbose)
        state = res["state"]
        results.append(res)

        # Lightweight JSON (serializable subset)
        out = {
            "T": res["T"],
            "beta": res["beta"],
            "L": res["L"],
            "C": res["C"],
            "time_series": res["time_series"],
            "correlation": res["correlation"],
        }
        with (save_dir_path / f"result_T{float(T):.3f}.json").open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    assert state is not None
    return results, state, save_dir_path
