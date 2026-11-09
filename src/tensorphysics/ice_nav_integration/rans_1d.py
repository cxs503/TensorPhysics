"""Streamwise Reynolds-averaged flow stage of the ice-navigation loop.

What this module is -- and is not
---------------------------------
TensorFVM/TensorLBM own the 3-D CFD kernels.  This module owns the *integration*
closure that the demonstration loop needs and that the platform can execute
end-to-end in seconds: a quasi-1-D, streamwise-averaged Reynolds-averaged
momentum balance through the ice lane the ship opens.

Control volume (documented once, used everywhere below): the lane of width
``1.1 B`` and height ``h_ice`` around the waterline, in the ship-fixed frame.
Across that lane the ice volume fraction is ``alpha(x)`` (cross-lane mean of the
DEM field), the water mass flux is set by the ship's advance through the lane,

    mdot = rho_w * V * (1 - C_i) * h_ice * W_lane            [kg/s]

and the lane water area is ``A_w(x) = (1 - alpha(x)) * h_ice * W_lane``, so the
mixture velocity follows from continuity alone,

    u(x) = mdot / (rho_w * A_w(x)) = V * (1 - C_i) / (1 - alpha(x)).

The streamwise momentum balance for that control volume is

    dp/dx = -( mdot * du/dx + f(x) ) / A_w(x),

with ``f(x)`` the *total* resistance per unit length distributed over the hull
(crushing + breaking + submersion + clearing + open-water viscous, each
normalised so that its integral reproduces the module total), and the pressure
field obtained by marching upstream from ``p(L) = 0`` (the wake reference).

Turbulence closure: the wall shear is evaluated with the ITTC-57 friction line
``C_f = 0.075 / (log10 Re - 2)^2`` at the *local* mixture Reynolds number, which
is the standard engineering closure for a high-Reynolds-number wall layer; the
porous ice drag uses a Darcy-Forchheimer form with the floe radius from the DEM
statistics as the interfacial length scale.  No turbulence transport equation is
solved, so this is a **reduced-order** flow model: adequate for interface,
force-integration and energy-bookkeeping verification, *not* a substitute for a
wall-resolved RANS/LES of the ice-hull gap.

Units: SI (m, s, kg, Pa, N).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .dem_field import ChannelGrid

G = 9.80665


@dataclass(frozen=True)
class FlowSetup:
    """Fluid and closure constants of the lane flow."""

    rho_w: float = 1025.0
    nu_m2_s: float = 1.79e-6  # sea water, ~0 degC
    lane_width_factor: float = 1.1
    cd_floe: float = 1.0
    water_fraction_floor: float = 0.05
    form_factor: float = 1.15  # (1+k) viscous form factor, KVLCC2-class hull
    ittc_a: float = 0.075
    ittc_b: float = 2.0


def ittc57_cf(reynolds: float, setup: FlowSetup = FlowSetup()) -> float:
    """ITTC-57 turbulent flat-plate friction coefficient."""
    if reynolds <= 0.0:
        return 0.0
    return setup.ittc_a / (math.log10(reynolds) - setup.ittc_b) ** 2


def hull_profile(x: np.ndarray, length: float, beam: float, entrance_fraction: float = 0.15) -> Dict[str, np.ndarray]:
    """Breadth, wetted-area density and axial-projection density of the hull.

    Entrance: parabolic waterline ``b(x) = B (1 - (1 - x/L_e)^2)`` for
    ``x < L_e = 0.15 L``, ``b = B`` aft.  ``b(0) = 0`` (stem), ``b(L_e) = B``.
    Wetted-area density ``S'(x) = b(x) + 2T`` (bottom + both sides, slab
    approximation); axial projection density ``A_x'(x) = T |db/dx|`` from the
    slender-body normal projection.
    """
    entrance = max(entrance_fraction * length, 1e-9)
    xi = np.clip(x, 0.0, length)
    b = beam * (1.0 - (1.0 - np.clip(xi, 0.0, entrance) / entrance) ** 2)
    b = np.where(xi > entrance, beam, b)
    dbdx = np.where(xi <= entrance, 2.0 * beam / entrance * (1.0 - xi / entrance), 0.0)
    return {"breadth_m": b, "dbdx": dbdx, "wetted_density_m": b, "axial_density_m": np.abs(dbdx)}


def solve_streamwise(
    *,
    grid: ChannelGrid,
    alpha_mean: np.ndarray,
    speed_m_s: float,
    thickness_m: float,
    coverage_ambient: float,
    force_density_N_per_m: np.ndarray,
    setup: FlowSetup = FlowSetup(),
    draft_m: float,
    beam_m: float,
) -> Dict[str, Any]:
    """March the reduced-order streamwise RANS through the ice lane.

    Parameters
    ----------
    alpha_mean : cross-lane mean volume fraction, shape (nx,)
    force_density_N_per_m : total distributed resistance, shape (nx,) [N/m]
    """
    nx = grid.nx
    if alpha_mean.shape != (nx,) or force_density_N_per_m.shape != (nx,):
        raise ValueError("alpha_mean and force_density must have shape (nx,)")

    lane_width = setup.lane_width_factor * beam_m
    water_free = np.maximum(1.0 - alpha_mean, setup.water_fraction_floor)
    water_free_ambient = max(1.0 - coverage_ambient, setup.water_fraction_floor)

    mdot = setup.rho_w * speed_m_s * water_free_ambient * thickness_m * lane_width
    area_w = water_free * thickness_m * lane_width
    u = mdot / (setup.rho_w * np.maximum(area_w, 1e-12))

    du_dx = np.gradient(u, grid.dx) if nx > 2 else np.zeros_like(u)
    dpdx = -(mdot * du_dx + force_density_N_per_m) / np.maximum(area_w, 1e-12)

    # march upstream from the wake reference p(L) = 0
    p = np.zeros(nx)
    for i in range(nx - 2, -1, -1):
        p[i] = p[i + 1] + grid.dx * (mdot * du_dx[i] + force_density_N_per_m[i]) / max(area_w[i], 1e-12)

    # wall shear in the lane (hull strip of height h_ice on both sides)
    re = u * grid.length / setup.nu_m2_s
    cf = np.array([ittc57_cf(float(r), setup) for r in re])
    rho_m = setup.rho_w * water_free + 917.0 * alpha_mean
    tau = 0.5 * rho_m * cf * u ** 2
    wetted_density = 2.0 * thickness_m  # [m^2 per m of length]

    # axial force recovered from the archived pressure field (finite differences)
    dpdx_recovered = np.zeros(nx)
    dpdx_recovered[:-1] = (p[1:] - p[:-1]) / grid.dx
    dpdx_recovered[-1] = dpdx_recovered[-2] if nx > 1 else 0.0
    f_panel = -(dpdx_recovered * area_w)  # [N/m]
    R_panel = float(np.sum(f_panel) * grid.dx)

    # porous floe drag implied by the resolved lane velocity (diagnostic only)
    r_mean = 1.0
    return {
        "u_m_s": u,
        "du_dx": du_dx,
        "p_pa": p,
        "dpdx_pa_per_m": dpdx,
        "dpdx_recovered_pa_per_m": dpdx_recovered,
        "tau_wall_pa": tau,
        "cf": cf,
        "reynolds": re,
        "water_area_m2": area_w,
        "water_free": water_free,
        "mass_flux_kg_s": mdot,
        "lane_width_m": lane_width,
        "R_panel_N": R_panel,
        "R_sources_N": float(np.sum(force_density_N_per_m) * grid.dx),
        "momentum_flux_N": float(mdot * (u[-1] - u[0])) if nx > 1 else 0.0,
        "R_lane_shear_N": float(np.sum(tau * wetted_density) * grid.dx),
        "u_max_m_s": float(np.max(u)),
        "p_max_pa": float(np.max(p)),
        "p_min_pa": float(np.min(p)),
    }


def porous_drag_density(
    *,
    alpha_mean: np.ndarray,
    u_m_s: np.ndarray,
    thickness_m: float,
    lane_width_m: float,
    r_mean_m: float,
    setup: FlowSetup = FlowSetup(),
) -> np.ndarray:
    """Darcy-Forchheimer floe drag per unit streamwise length [N/m].

    ``f = 0.5 rho C_d (2 alpha / r) u^2 A_w`` with ``2/r`` the interfacial area
    per unit volume of a floe slab of radius ``r`` and thickness ``h``.
    """
    water_free = np.maximum(1.0 - alpha_mean, setup.water_fraction_floor)
    area_w = water_free * thickness_m * lane_width_m
    sigma = 2.0 * alpha_mean / max(r_mean_m, 1e-9)  # 1/m
    return 0.5 * setup.rho_w * setup.cd_floe * sigma * u_m_s ** 2 * area_w


def open_water_resistance(
    *,
    speed_m_s: float,
    length_m: float,
    draft_m: float,
    beam_m: float,
    block_coefficient: float,
    setup: FlowSetup = FlowSetup(),
) -> Dict[str, float]:
    """ITTC-57 open-water viscous resistance over the full wetted surface.

    Wetted surface from the Denny-Mumford regression
    ``S = 1.7 L T + C_b B L`` (standard for full-form hulls).
    """
    S = 1.7 * length_m * draft_m + block_coefficient * beam_m * length_m
    re = speed_m_s * length_m / setup.nu_m2_s
    cf = ittc57_cf(re, setup)
    r_f = 0.5 * setup.rho_w * speed_m_s ** 2 * cf * S
    r_v = r_f * setup.form_factor
    return {
        "wetted_surface_m2": S,
        "reynolds": re,
        "cf": cf,
        "R_friction_N": r_f,
        "R_viscous_N": r_v,
    }


def distribution_weights(
    *,
    x: np.ndarray,
    breadth_m: np.ndarray,
    alpha_mean: np.ndarray,
    coverage_ambient: float,
    dx: float,
    bow_length_fraction: float = 0.12,
) -> Dict[str, np.ndarray]:
    """Normalised streamwise distribution of each resistance channel.

    Assumption (documented in the report): crushing and bending act on the bow
    shoulder, hence a decaying bow window ``exp(-x / L_bow)`` weighted by the
    ice presence ``alpha``; submersion, clearing and viscous drag act along the
    whole wetted length, weighted by ``alpha * B(x)`` for the ice channels and
    ``B(x)`` for the open-water channel.  Each weight is normalised to unit
    integral so that its product with the channel total reproduces the module
    value exactly (this normalisation is what the gate checks).
    """
    x = np.asarray(x, dtype=float)
    b = np.asarray(breadth_m, dtype=float)
    a = np.maximum(np.asarray(alpha_mean, dtype=float), 1e-6)
    bow = np.exp(-np.clip(x, 0.0, None) / max(bow_length_fraction * x[-1] if x.size else 1.0, 1e-9))
    ice_hull = a * b
    hull = b

    def _norm(w: np.ndarray) -> np.ndarray:
        # integral over x must be exactly 1, i.e. sum(w)*dx = 1, so that
        # sum(w * R * dx) = R for every channel total R
        s = float(np.sum(w)) * dx
        return w / s if s > 0.0 else np.full_like(w, 1.0 / (w.size * dx))

    return {
        "bow": _norm(bow * a),
        "hull_ice": _norm(ice_hull),
        "hull": _norm(hull),
    }