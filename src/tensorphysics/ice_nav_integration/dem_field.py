"""DEM ice field stage of the ice-navigation integration loop.

Scope (and the honest boundary of it)
-------------------------------------
TensorDEM owns the Lagrangian contact dynamics.  This module owns the *handoff
representation* of the floe field to the continuum solvers, i.e. the Eulerian
volume-fraction field ``alpha_i(x, y, t)`` plus the floe statistics that the
porous-drag closure needs.  It does **not** re-implement a DEM kernel:

* :func:`floe_statistics` -- analytic statistics of a power-law (truncated
  Pareto) floe-size distribution, which is the standard brash-ice idealisation
  (r in [1, 8] m, exponent 2.5).  From it we derive floe area density, mean
  radius (the porous-drag length scale) and the bow encounter rate.
* :func:`coverage_field` -- deterministic reconstruction of a 2-D coverage
  field with the exact nominal mean coverage ``C_i`` and a finite correlation
  length.  The construction (filtered white noise, clip to the packing cap,
  then rescale to restore the mean to machine precision) is *mass exact*: the
  reported ice volume is conserved by construction, not by luck.
* :func:`advect_upwind` -- first-order conservative upwind advection of the
  Eulerian field in the ship-fixed frame with a prescribed inflow column.  The
  discrete flux form guarantees the ice budget identity
  ``V^{n+1} - V^n = dt (inflow - outflow)`` to machine precision; the audit is
  returned with the field so the gate can certify it every step.

Units: SI throughout (m, s, kg, m^3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .contracts import canonical_hash

RHO_ICE = 917.0


@dataclass(frozen=True)
class ChannelGrid:
    """Structured Cartesian grid in the ship-fixed (bow-origin) frame."""

    nx: int
    ny: int
    dx: float
    dy: float
    x0: float = 0.0
    y0: float = 0.0
    frame: str = "ship_fixed_bow_origin"

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("grid must have positive extent")
        if not (self.dx > 0.0 and self.dy > 0.0):
            raise ValueError("grid spacing must be positive")

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    @property
    def domain_area(self) -> float:
        return float(self.nx * self.ny) * self.cell_area

    @property
    def length(self) -> float:
        return float(self.nx) * self.dx

    @property
    def width(self) -> float:
        return float(self.ny) * self.dy

    def cell_centers_x(self) -> np.ndarray:
        return self.x0 + (np.arange(self.nx) + 0.5) * self.dx

    def cell_centers_y(self) -> np.ndarray:
        return self.y0 + (np.arange(self.ny) + 0.5) * self.dy

    def descriptor(self) -> Dict[str, Any]:
        """JSON payload handed to every downstream module."""
        return {
            "kind": "cartesian_2d",
            "frame": self.frame,
            "nx": int(self.nx),
            "ny": int(self.ny),
            "dx_m": float(self.dx),
            "dy_m": float(self.dy),
            "x0_m": float(self.x0),
            "y0_m": float(self.y0),
            "cell_area_m2": float(self.cell_area),
            "domain_area_m2": float(self.domain_area),
        }

    def hash(self) -> str:
        return canonical_hash(self.descriptor())[:16]


# ---------------------------------------------------------------------------
# floe statistics (power-law / truncated Pareto size distribution)
# ---------------------------------------------------------------------------


def truncated_pareto_moments(r_min: float, r_max: float, alpha_par: float) -> Dict[str, float]:
    """First two moments of ``p(r) ~ r^-alpha_par`` on ``[r_min, r_max]``."""
    if not (0.0 < r_min < r_max):
        raise ValueError("require 0 < r_min < r_max")
    if abs(alpha_par - 1.0) < 1e-9 or abs(alpha_par - 3.0) < 1e-9:
        raise ValueError("alpha_par must stay away from the singular exponents 1 and 3")
    # p(r) ~ r^-alpha on [r_min, r_max]; normalisation includes the (1-alpha) factor
    norm = (r_max ** (1.0 - alpha_par) - r_min ** (1.0 - alpha_par)) / (1.0 - alpha_par)
    e1 = (r_max ** (2.0 - alpha_par) - r_min ** (2.0 - alpha_par)) / (2.0 - alpha_par) / norm
    e2 = (r_max ** (3.0 - alpha_par) - r_min ** (3.0 - alpha_par)) / (3.0 - alpha_par) / norm
    return {"r_mean": e1, "r2_mean": e2}


def floe_statistics(
    *,
    coverage: float,
    thickness_m: float,
    rho_ice: float = RHO_ICE,
    r_min: float = 1.0,
    r_max: float = 8.0,
    alpha_par: float = 2.5,
    lane_width_m: float,
    speed_m_s: float,
) -> Dict[str, float]:
    """Floe area density, drag length scale and bow encounter rate.

    The brash-ice field is idealised as a population of circular floes of equal
    thickness; the coverage fixes the area density ``n_A = c / (pi E[r^2])``.
    """
    if not (0.0 <= coverage <= 1.0):
        raise ValueError("coverage must be within [0, 1]")
    mom = truncated_pareto_moments(r_min, r_max, alpha_par)
    mean_area = math.pi * mom["r2_mean"]
    n_area = coverage / mean_area if mean_area > 0.0 else 0.0
    encounter_rate = n_area * lane_width_m * max(speed_m_s, 0.0)
    floe_volume = mean_area * thickness_m
    return {
        "r_mean_m": mom["r_mean"],
        "r_rms_m": math.sqrt(mom["r2_mean"]),
        "mean_floe_area_m2": mean_area,
        "mean_floe_volume_m3": floe_volume,
        "mean_floe_mass_kg": floe_volume * rho_ice,
        "n_floes_per_m2": n_area,
        "n_floes_per_m3": n_area / thickness_m,
        "encounter_rate_1_per_s": encounter_rate,
        "fracture_area_rate_m2_per_s": encounter_rate * mean_area,
        "drag_length_scale_m": mom["r_mean"],
    }


# ---------------------------------------------------------------------------
# coverage field reconstruction
# ---------------------------------------------------------------------------


def _smooth_field(nx: int, ny: int, corr: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian-filtered white noise (separable box blur repeated 3x)."""
    noise = rng.standard_normal((nx, ny))
    k = max(1, int(round(corr)))
    out = noise
    for _ in range(3):
        padded = np.pad(out, ((k, k), (k, k)), mode="reflect")
        c = np.cumsum(padded, axis=0)
        out = (c[2 * k:, :] - c[:-2 * k, :]) / (2 * k)
        c = np.cumsum(out, axis=1)
        out = (c[:, 2 * k:] - c[:, :-2 * k]) / (2 * k)
    return out


def coverage_field(
    grid: ChannelGrid,
    *,
    coverage: float,
    seed: int = 20261108,
    correlation_cells: float = 2.5,
    packing_cap: float = 0.90,
    modulation: float = 0.35,
) -> np.ndarray:
    """Deterministic coverage field with *exactly* the requested mean.

    ``alpha = clip(mean + modulation * std * xi, 0, packing_cap)`` followed by a
    uniform rescale so that the area mean is the nominal coverage to machine
    precision.  The clip/rescale pair is what makes the ice volume a conserved
    quantity rather than a statistical accident.
    """
    if not (0.0 <= coverage <= packing_cap):
        raise ValueError(
            "coverage %.4f outside [0, %.2f] (above the packing cap the Eulerian "
            "representation is not meaningful)" % (coverage, packing_cap)
        )
    if coverage == 0.0:
        return np.zeros((grid.nx, grid.ny))
    rng = np.random.default_rng(int(seed))
    xi = _smooth_field(grid.nx, grid.ny, correlation_cells, rng)
    xi = xi - xi.mean()
    std = float(xi.std())
    if std <= 0.0:  # pragma: no cover - degenerate RNG state
        return np.full((grid.nx, grid.ny), coverage)
    raw = coverage + modulation * coverage * xi / std
    clipped = np.clip(raw, 0.0, packing_cap)
    mean = float(clipped.mean())
    if mean <= 0.0:  # pragma: no cover - defensive
        return np.full((grid.nx, grid.ny), coverage)
    field = clipped * (coverage / mean)
    # one more clip/rescale pass; the correction is O(1e-12) because the scaled
    # field already sits inside the box
    field = np.clip(field, 0.0, packing_cap)
    field *= coverage / float(field.mean())
    return field


def field_ice_volume(alpha: np.ndarray, grid: ChannelGrid, thickness_m: float) -> float:
    return float(np.sum(alpha)) * grid.cell_area * thickness_m


# ---------------------------------------------------------------------------
# conservative advection
# ---------------------------------------------------------------------------


def advect_upwind(
    alpha: np.ndarray,
    grid: ChannelGrid,
    *,
    speed_m_s: float,
    dt_s: float,
    thickness_m: float,
    inflow_column: np.ndarray,
) -> Dict[str, Any]:
    """One conservative upwind step in the ship-fixed frame.

    Ice enters at the bow (``x = 0``) with the prescribed inflow column and
    leaves astern (``x = L``).  The flux form makes the discrete ice budget an
    identity, which is verified here and returned for the gate audit.
    """
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    courant = speed_m_s * dt_s / grid.dx
    if courant > 1.0:
        raise ValueError("upwind advection requires Courant <= 1, got %.3f" % courant)

    volume_before = field_ice_volume(alpha, grid, thickness_m)
    new = alpha.copy()
    upstream = np.vstack([inflow_column[None, :], alpha[:-1, :]])
    new = alpha - courant * (alpha - upstream)

    face_area = grid.dy * thickness_m
    inflow = float(np.sum(inflow_column)) * speed_m_s * face_area
    outflow = float(np.sum(alpha[-1, :])) * speed_m_s * face_area
    volume_after = field_ice_volume(new, grid, thickness_m)
    balance = volume_after - volume_before - dt_s * (inflow - outflow)
    scale = max(abs(volume_before), abs(dt_s * inflow), 1.0)
    return {
        "alpha": new,
        "courant": courant,
        "volume_before_m3": volume_before,
        "volume_after_m3": volume_after,
        "inflow_m3_per_s": inflow,
        "outflow_m3_per_s": outflow,
        "budget_residual_m3": balance,
        "budget_residual_relative": abs(balance) / scale,
    }


@dataclass
class IceFieldState:
    """Eulerian state carried by the loop."""

    grid: ChannelGrid
    thickness_m: float
    rho_ice: float
    nominal_coverage: float
    alpha: np.ndarray
    inflow_column: np.ndarray
    stats: Dict[str, float] = field(default_factory=dict)
    audit: List[Dict[str, float]] = field(default_factory=list)

    def cross_mean(self) -> np.ndarray:
        return self.alpha.mean(axis=1)

    def volume_m3(self) -> float:
        return field_ice_volume(self.alpha, self.grid, self.thickness_m)

    def step(self, *, speed_m_s: float, dt_s: float) -> Dict[str, Any]:
        out = advect_upwind(
            self.alpha,
            self.grid,
            speed_m_s=speed_m_s,
            dt_s=dt_s,
            thickness_m=self.thickness_m,
            inflow_column=self.inflow_column,
        )
        self.alpha = out["alpha"]
        self.audit.append(
            {
                "courant": out["courant"],
                "budget_residual_relative": out["budget_residual_relative"],
                "volume_m3": out["volume_after_m3"],
            }
        )
        return out


def build_ice_field_state(
    grid: ChannelGrid,
    *,
    coverage: float,
    thickness_m: float,
    lane_width_m: float,
    speed_m_s: float,
    seed: int = 20261108,
    rho_ice: float = RHO_ICE,
    r_min: float = 1.0,
    r_max: float = 8.0,
    alpha_par: float = 2.5,
    correlation_cells: float = 2.5,
) -> IceFieldState:
    alpha = coverage_field(
        grid, coverage=coverage, seed=seed, correlation_cells=correlation_cells
    )
    # frozen inflow column: the ice entering at the bow keeps the spatial
    # texture of the seeded field, so the inflow is stationary but not uniform
    inflow = alpha[0, :].copy()
    inflow *= coverage / float(inflow.mean()) if inflow.mean() > 0 else 0.0
    stats = floe_statistics(
        coverage=coverage,
        thickness_m=thickness_m,
        rho_ice=rho_ice,
        r_min=r_min,
        r_max=r_max,
        alpha_par=alpha_par,
        lane_width_m=lane_width_m,
        speed_m_s=speed_m_s,
    )
    return IceFieldState(
        grid=grid,
        thickness_m=thickness_m,
        rho_ice=rho_ice,
        nominal_coverage=coverage,
        alpha=alpha,
        inflow_column=inflow,
        stats=stats,
    )