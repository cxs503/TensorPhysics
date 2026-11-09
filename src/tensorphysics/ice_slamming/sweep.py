"""Parameter sweep: P_max(V_ice, theta, H_s) envelope + exponent fit.

The sweep covers the task's box:

    V_ice in [0.5, 3] m/s,  theta in [15, 60] deg,  H_s in [1, 4] m.

The velocity exponent ``alpha`` is the least-squares slope of
``log(P_max)`` vs ``log(V_ice)`` at fixed ``(theta, H_s)`` (and of the
envelope ``max_{theta, H_s} P_max``).  The acceptance gate is
``alpha in [1.5, 2.5]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .model import SlammingCase, SlammingResult, run_impact
from .properties import default_floe, default_structure, default_water

__all__ = ["SweepResult", "fit_velocity_exponent", "run_sweep"]


@dataclass
class SweepResult:
    """Aggregated sweep output (JSON-serialisable via .to_dict())."""

    v_grid: list
    theta_grid: list
    h_grid: list
    p_max: dict          # (v, theta, h) -> Pa
    envelope: dict       # v -> max over (theta, h)
    alpha_by_case: dict  # (theta, h) -> alpha
    alpha_envelope: float
    p_max_table: list    # rows for the report

    def to_dict(self) -> dict:
        return {
            "v_grid": self.v_grid,
            "theta_grid": self.theta_grid,
            "h_grid": self.h_grid,
            "p_max": {f"{v}|{th}|{h}": p for (v, th, h), p in self.p_max.items()},
            "envelope": self.envelope,
            "alpha_by_case": {f"{th}|{h}": a for (th, h), a in self.alpha_by_case.items()},
            "alpha_envelope": self.alpha_envelope,
            "p_max_table": self.p_max_table,
        }


def fit_velocity_exponent(v: list, p: list) -> float:
    """Least-squares slope of log(P) vs log(V)."""
    v = np.asarray(v, dtype=float)
    p = np.asarray(p, dtype=float)
    m = (v > 0) & (p > 0)
    if m.sum() < 2:
        return float("nan")
    lv = np.log(v[m])
    lp = np.log(p[m])
    a, b = np.polyfit(lv, lp, 1)
    return float(a)


def run_sweep(
    v_grid: list | None = None,
    theta_grid: list | None = None,
    h_grid: list | None = None,
    floe=None,
    water=None,
    structure=None,
    n_plate: int = 16,
    dt: float = 1.0e-5,
    t_end: float = 0.05,
    water_area_factor: float = 3.0,
) -> SweepResult:
    """Run the (V_ice, theta, H_s) sweep and fit the velocity exponent."""
    v_grid = v_grid or [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    theta_grid = theta_grid or [15.0, 30.0, 45.0, 60.0]
    h_grid = h_grid or [1.0, 2.0, 3.0, 4.0]
    floe = floe or default_floe()
    water = water or default_water()
    structure = structure or default_structure()

    p_max: dict = {}
    for th in theta_grid:
        for h in h_grid:
            for v in v_grid:
                case = SlammingCase(
                    v_ice=v, theta_deg=th, h_s=h, floe=floe, water=water,
                    structure=structure, n_plate=n_plate, dt=dt, t_end=t_end,
                    water_area_factor=water_area_factor,
                )
                res = run_impact(case)
                p_max[(v, th, h)] = res.p_max

    envelope: dict = {}
    for v in v_grid:
        vals = [p for (vv, th, hh), p in p_max.items() if vv == v]
        envelope[v] = max(vals) if vals else 0.0

    alpha_by_case: dict = {}
    for th in theta_grid:
        for h in h_grid:
            vv = v_grid
            pp = [p_max[(v, th, h)] for v in vv]
            alpha_by_case[(th, h)] = fit_velocity_exponent(vv, pp)

    alpha_env = fit_velocity_exponent(list(envelope.keys()), list(envelope.values()))

    table = []
    for v in v_grid:
        for th in theta_grid:
            for h in h_grid:
                table.append({
                    "v_ice": v, "theta_deg": th, "h_s": h,
                    "p_max_pa": p_max[(v, th, h)],
                })

    return SweepResult(
        v_grid=v_grid, theta_grid=theta_grid, h_grid=h_grid,
        p_max=p_max, envelope=envelope, alpha_by_case=alpha_by_case,
        alpha_envelope=alpha_env, p_max_table=table,
    )
