"""Moving load paths and their projection on the beam modes.

A *load path* is one load component that enters the beam at ``x_entry_m`` at
time ``t_entry_s``, travels at constant speed to ``x_exit_m`` and leaves after
``duration_s``.  Its magnitude may be constant (a point force traversing the
span), half-sine or triangular (an ice-floe contact pulse).

The modal force of a path is the Galerkin projection of the moving load on the
mode shapes,

    Q_n(t) = P(t) * phi_n(x(t)) = P(t) * sin(n pi x(t) / L),

which is the only coupling term between the load model and the structural
model.  Keeping it in one place means every load generator (single point
force, floe stream) feeds the same modal solver and the same finite-difference
reference solver -- so both solvers see *identical* loads and their agreement
is meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Union

import numpy as np

from .beam import HullGirderBeam

__all__ = [
    "LoadPath",
    "moving_point_load",
    "modal_force_matrix",
    "total_load_series",
]

_PULSE_SHAPES = ("constant", "half_sine", "triangular")


@dataclass(frozen=True)
class LoadPath:
    """One load component travelling along the beam.

    Parameters
    ----------
    magnitude_n
        Peak force of the path (N).  For ``shape='constant'`` this is the
        force itself.
    x_entry_m, x_exit_m
        Position at entry and at exit (m, measured from the left support).
    t_entry_s
        Time at which the load enters (s).
    duration_s
        Contact/residence time on the span (s).
    shape
        ``constant`` | ``half_sine`` | ``triangular`` -- the time envelope.
    """

    magnitude_n: float
    x_entry_m: float
    x_exit_m: float
    t_entry_s: float
    duration_s: float
    shape: str = "constant"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.magnitude_n) and self.magnitude_n > 0.0):
            raise ValueError("magnitude_n must be positive and finite")
        if not (math.isfinite(self.duration_s) and self.duration_s > 0.0):
            raise ValueError("duration_s must be positive and finite")
        if not math.isfinite(self.t_entry_s):
            raise ValueError("t_entry_s must be finite")
        if self.shape not in _PULSE_SHAPES:
            raise ValueError(f"shape must be one of {_PULSE_SHAPES}, got {self.shape!r}")

    @property
    def speed_m_s(self) -> float:
        """Travel speed of the load along the beam (m/s, signed)."""
        return (self.x_exit_m - self.x_entry_m) / self.duration_s

    def envelope(self, tau: np.ndarray) -> np.ndarray:
        """Time envelope normalised to the peak (1.0) over ``tau in [0, 1]``."""
        if self.shape == "constant":
            return np.ones_like(tau)
        if self.shape == "half_sine":
            return np.sin(math.pi * np.clip(tau, 0.0, 1.0))
        # triangular (rise then fall)
        return np.where(tau <= 0.5, 2.0 * tau, 2.0 * (1.0 - tau))

    def position_series(self, time_s: np.ndarray) -> np.ndarray:
        """Position of the load at each time (held at the entry/exit ends)."""
        t = np.asarray(time_s, dtype=float)
        frac = (t - self.t_entry_s) / self.duration_s
        frac = np.clip(frac, 0.0, 1.0)
        return self.x_entry_m + frac * (self.x_exit_m - self.x_entry_m)

    def magnitude_series(self, time_s: np.ndarray) -> np.ndarray:
        """Force magnitude at each time (zero outside the residence window)."""
        t = np.asarray(time_s, dtype=float)
        frac = (t - self.t_entry_s) / self.duration_s
        inside = (frac >= 0.0) & (frac <= 1.0)
        out = np.zeros_like(t)
        if np.any(inside):
            out[inside] = self.magnitude_n * self.envelope(np.clip(frac[inside], 0.0, 1.0))
        return out

    def is_active(self, t_s: float) -> bool:
        frac = (t_s - self.t_entry_s) / self.duration_s
        return 0.0 <= frac <= 1.0


def moving_point_load(
    beam: HullGirderBeam,
    magnitude_n: float,
    speed_m_s: float,
    x_entry_m: float = 0.0,
    t_entry_s: float = 0.0,
    direction: int = 1,
) -> LoadPath:
    """A constant point force traversing the span at ``speed_m_s``.

    ``direction=+1`` travels from ``x_entry_m`` towards the right support,
    ``direction=-1`` towards the left support.  The residence time is the time
    needed to travel the remaining span.
    """
    if speed_m_s <= 0.0:
        raise ValueError("speed_m_s must be positive")
    x_entry = float(x_entry_m)
    remaining = (beam.length_m - x_entry) if direction > 0 else x_entry
    if remaining <= 0.0:
        raise ValueError("load entry point lies outside the span")
    return LoadPath(
        magnitude_n=float(magnitude_n),
        x_entry_m=x_entry,
        x_exit_m=beam.length_m if direction > 0 else 0.0,
        t_entry_s=float(t_entry_s),
        duration_s=remaining / float(speed_m_s),
        shape="constant",
    )


def modal_force_matrix(
    beam: HullGirderBeam,
    paths: Sequence[LoadPath],
    time_s: np.ndarray,
    n_modes: int = 10,
) -> np.ndarray:
    """Generalised modal force ``Q[n_time, n_modes]`` of a set of load paths."""
    t = np.asarray(time_s, dtype=float)
    n_modes = int(n_modes)
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    Q = np.zeros((t.size, n_modes), dtype=float)
    mode_numbers = np.arange(1, n_modes + 1, dtype=float)
    for path in paths:
        mag = path.magnitude_series(t)
        active = mag != 0.0
        if not np.any(active):
            continue
        x = path.position_series(t)[active]
        phase = mode_numbers[None, :] * math.pi * x[:, None] / beam.length_m
        Q[active, :] += mag[active, None] * np.sin(phase)
    return Q


def total_load_series(
    paths: Sequence[LoadPath],
    time_s: np.ndarray,
) -> np.ndarray:
    """Sum of all path magnitudes at each time (N) -- the ice-load time history."""
    t = np.asarray(time_s, dtype=float)
    total = np.zeros_like(t)
    for path in paths:
        total += path.magnitude_series(t)
    return total