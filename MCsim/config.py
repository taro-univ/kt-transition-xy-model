"""
config.py

Configuration for Monte Carlo simulations of the 2D XY model.

Design goals
------------
- Single dataclass for experiment parameters (size, temperature grid, sweeps, RNG seed).
- JSON serialization for reproducibility.
- Helper for creating a unique output directory per run.

Notes
-----
- Paths are relative by default to keep this repository portable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, List, Optional
import json
import math
import random
import time


def frange(start: float, stop: float, step: float) -> List[float]:
    """
    Floating-point range generator (inclusive of stop within tolerance).

    Parameters
    ----------
    start, stop, step : float
        Range definition. step must be positive.

    Returns
    -------
    list[float]
        Values rounded to avoid floating-point representation noise.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(math.floor((stop - start) / step + 1e-12)) + 1
    return [round(start + i * step, 10) for i in range(max(n, 0))]


@dataclass(slots=True)
class Config:
    """Configuration container for a simulation run."""

    # --- physics / lattice ---
    L: int = 64
    J: float = 1.0

    # --- temperature schedule ---
    temps: List[float] = field(default_factory=lambda: frange(0.2, 1.2, 0.05))

    # --- Monte Carlo schedule ---
    sweeps_therm: int = 5_000
    sweeps_meas: int = 20_000
    sample_interval: int = 20

    # --- RNG ---
    seed: Optional[int] = None

    # --- update method flags ---
    use_metropolis: bool = True
    use_overrelax: bool = True
    use_wolff: bool = False  # optional

    # --- Metropolis proposal width (auto-tuning starts from this) ---
    delta_phi_init: float = math.pi / 3  # ~60 degrees

    # --- output control (portable defaults) ---
    out_dir: str = "results"
    save_snapshots_every: int = 500  # <=0 to disable
    save_pair_stats: bool = True

    # --- visualization ---
    spin_quiver_subsample: int = 4
    hsv_image: bool = True

    # --- acceptance-rate tuning targets (internal) ---
    target_accept_min: float = 0.40
    target_accept_max: float = 0.60

    def validate(self) -> None:
        """Validate configuration values."""
        assert isinstance(self.L, int) and self.L > 0
        assert self.J > 0
        assert len(self.temps) > 0
        assert self.sweeps_therm >= 0 and self.sweeps_meas > 0
        assert self.sample_interval > 0
        assert self.delta_phi_init > 0
        assert self.spin_quiver_subsample >= 1
        assert 0.0 <= self.target_accept_min < self.target_accept_max <= 1.0

    def update(self, **kwargs: Any) -> "Config":
        """
        In-Python override utility.

        Example
        -------
        cfg = Config().update(L=128, temps=[0.7, 0.8, 0.9])
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown config key: {k}")
            setattr(self, k, v)
        self.validate()
        return self

    # --- JSON I/O ---
    def to_json(self, path: str | Path) -> None:
        """Write config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(path: str | Path) -> "Config":
        """Load config from a JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = Config(**data)
        cfg.validate()
        return cfg

    # --- run directory helper ---
    def materialize_out_dir(self, prefix: str = "run") -> Path:
        """
        Create and return a unique run directory under out_dir.

        Example
        -------
        results/run-20260210-103001-rid000123/
        """
        ts = time.strftime("%Y%m%d-%H%M%S")
        rid = random.randint(0, 1_000_000)
        base = Path(self.out_dir)
        p = base / f"{prefix}-{ts}-rid{rid:06d}"
        p.mkdir(parents=True, exist_ok=True)
        return p
