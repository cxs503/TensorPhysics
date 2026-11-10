"""DEM -> FEM load transfer for the ice-structure coupling node.

What is transferred
-------------------
The DEM (see :mod:`.dem_ice`) delivers the ice resistance time history
``F_x(t)`` as a **line force per metre of waterline** [N/m].  The hull panel
receives it over an IACS ice patch of height ``h_p = 0.6 h`` (see
:mod:`tensorphysics.hull_icebelt_fem.iacs_pressure`), so the patch pressure
history is

.. math::   p(t) = F_x(t) / h_p    [Pa]

That is the boundary condition handed to the elasto-plastic plate
(:mod:`.fem_plate`).

Time synchronisation
--------------------
The DEM time step is far smaller than the structural one.  The transfer is
**impulse conserving**: for every structural interval ``[t_k, t_{k+1}]`` the
DEM history is integrated exactly (piecewise-linear sub-cycling),

.. math::  \\bar p_k = \\frac{1}{\\Delta t}\\int_{t_k}^{t_{k+1}} p(t)\\,dt

so ``sum_k \\bar p_k dt = \\int p dt`` to round-off.  Two error measures are
reported:

``sync_impulse_error``
    ``|sum_k \\bar p_k dt - int p dt| / int p dt`` -- the synchronisation error
    of the interface (< 1 % required).
``zero_order_hold_peak_error``
    the peak loss of a naive (piecewise-constant, nearest-sample) transfer,
    i.e. what the sub-cycling buys.

Energy accounting
-----------------
``run_coupled_case`` returns the work the ice does on the panel
(``W_ice``, the integral of ``p dv``-type work through the plate load
history), the panel's external work, plastic dissipation and internal work
closure, so the interface energy residual can be reported.

Nothing here duplicates solver kernels: the DEM comes from :mod:`.dem_ice`,
the plate from :mod:`.fem_plate`, the IACS patch geometry from
:mod:`tensorphysics.hull_icebelt_fem`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .dem_ice import IceSliceConfig, IceSliceResult, simulate_ice_slice
from .fem_plate import PlateConfig, PlateResponse, solve_plate_history
from .lindqvist_ref import LindqvistReference, reference_for

__all__ = [
    "PatchLoad",
    "patch_load_from_spectrum",
    "transfer_impulse_consistent",
    "synchronisation_errors",
    "CoupledResult",
    "run_coupled_case",
    "iacs_patch_height",
]


def iacs_patch_height(h: float) -> float:
    """IACS ice-patch height ``0.6 h_I`` (m)."""
    return 0.6 * float(h)


@dataclass
class PatchLoad:
    """Patch pressure history handed to the FEM."""

    time: np.ndarray
    pressure: np.ndarray
    patch_height: float
    patch: Tuple[float, float, float, float]
    source_peak_line_force: float

    @property
    def peak_pressure(self) -> float:
        return float(np.max(self.pressure)) if self.pressure.size else 0.0

    @property
    def impulse(self) -> float:
        return float(np.trapezoid(self.pressure, self.time)) if self.time.size > 1 else 0.0


def patch_load_from_spectrum(
    dem: IceSliceResult,
    patch_height: Optional[float] = None,
    patch: Tuple[float, float, float, float] = (0.0, 0.6, 0.0, 2.4),
) -> PatchLoad:
    """Convert the DEM line-force history into an equivalent patch pressure."""
    h_p = iacs_patch_height(dem.config.h) if patch_height is None else float(patch_height)
    if h_p <= 0:
        raise ValueError("patch height must be positive")
    p = np.maximum(dem.force_x, 0.0) / h_p
    return PatchLoad(
        time=np.array(dem.time, dtype=float),
        pressure=p,
        patch_height=h_p,
        patch=patch,
        source_peak_line_force=dem.peak_force,
    )


def transfer_impulse_consistent(
    time_src: np.ndarray,
    value_src: np.ndarray,
    time_dst: np.ndarray,
) -> np.ndarray:
    """Impulse-conserving (piecewise-linear exact) resampling onto ``time_dst``."""
    t_src = np.asarray(time_src, dtype=float)
    v_src = np.asarray(value_src, dtype=float)
    t_dst = np.asarray(time_dst, dtype=float)
    if t_src.size < 2:
        raise ValueError("source history needs at least two samples")
    out = np.zeros(t_dst.size)
    for k in range(t_dst.size - 1):
        a, b = t_dst[k], t_dst[k + 1]
        if b <= a:
            out[k] = 0.0
            continue
        # integrate the piecewise-linear source over [a, b] exactly
        nodes = np.unique(np.concatenate(([a, b], t_src[(t_src > a) & (t_src < b)])))
        vals = np.interp(nodes, t_src, v_src)
        out[k] = float(np.trapezoid(vals, nodes) / (b - a))
    out[-1] = out[-2] if t_dst.size > 1 else 0.0
    return out


def synchronisation_errors(
    load: PatchLoad,
    n_structural_steps: int,
) -> Dict[str, float]:
    """Impulse error of the sub-cycled transfer and the naive ZOH peak loss."""
    t_dst = np.linspace(load.time[0], load.time[-1], n_structural_steps + 1)
    p_dst = transfer_impulse_consistent(load.time, load.pressure, t_dst)
    impulse_src = float(np.trapezoid(load.pressure, load.time))
    impulse_dst = float(np.sum(p_dst[:-1] * np.diff(t_dst)))
    hold = np.interp(t_dst, load.time, load.pressure)
    peak = float(np.max(load.pressure)) or 1.0
    return {
        "n_structural_steps": int(n_structural_steps),
        "dem_time_step_s": float(load.time[1] - load.time[0]),
        "structural_time_step_s": float(t_dst[1] - t_dst[0]),
        "subcycling_factor": float((t_dst[1] - t_dst[0]) / (load.time[1] - load.time[0])),
        "impulse_source_Pa_s": impulse_src,
        "impulse_transferred_Pa_s": impulse_dst,
        "sync_impulse_error": abs(impulse_dst - impulse_src) / max(impulse_src, 1e-30),
        "zero_order_hold_peak_error": abs(float(np.max(hold)) - peak) / peak,
        "_time_dst": t_dst,
        "_pressure_dst": p_dst,
    }


@dataclass
class CoupledResult:
    dem: IceSliceResult
    reference: LindqvistReference
    load: PatchLoad
    fem: PlateResponse
    sync: Dict[str, float]
    energy: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        sync = {k: v for k, v in self.sync.items() if not k.startswith("_")}
        return {
            "condition": {"h_m": self.dem.config.h, "speed_ms": self.dem.config.speed},
            "dem": self.dem.as_dict(),
            "lindqvist": self.reference.as_dict(),
            "patch_load": {
                "patch_height_m": self.load.patch_height,
                "patch_rectangle_m": list(self.load.patch),
                "peak_pressure_Pa": self.load.peak_pressure,
                "impulse_Pa_s": self.load.impulse,
            },
            "fem": self.fem.as_dict(),
            "synchronisation": sync,
            "energy": self.energy,
        }


def run_coupled_case(
    h: float,
    speed: float,
    dem_config: Optional[IceSliceConfig] = None,
    fem_config: Optional[PlateConfig] = None,
    patch: Tuple[float, float, float, float] = (0.0, 0.6, 0.0, 2.4),
    n_structural_steps: int = 240,
    dem_result: Optional[IceSliceResult] = None,
) -> CoupledResult:
    """Run one DEM -> FEM coupled ice-structure case.

    ``dem_config`` (or a pre-computed ``dem_result``) provides the ice-load
    spectrum; ``fem_config`` is the elasto-plastic plate.  The patch pressure
    history is sub-cycled onto ``n_structural_steps`` structural increments and
    integrated quasi-statically.
    """
    dem = dem_result if dem_result is not None else simulate_ice_slice(
        dem_config or IceSliceConfig(h=h, speed=speed))
    load = patch_load_from_spectrum(dem, patch=patch)
    sync = synchronisation_errors(load, n_structural_steps)
    fem_cfg = fem_config or PlateConfig(a=patch[1] - patch[0], b=patch[3] - patch[2],
                                       nx=6, ny=24)
    t_dst = sync["_time_dst"]
    p_dst = sync["_pressure_dst"]
    fem = solve_plate_history(fem_cfg, p_dst, patch=patch)

    # work done by the ice on the panel and the interface energy ledger
    w_ext = fem.energy["external_work_J"]
    u_e = fem.energy["elastic_energy_J"]
    w_p = fem.energy["plastic_dissipation_J"]
    energy = {
        "W_external_J": w_ext,
        "U_elastic_J": u_e,
        "W_plastic_J": w_p,
        "interface_energy_residual": fem.energy["work_energy_closure"],
        "layer_quadrature_residual": fem.energy["layer_quadrature_residual"],
        "note": (
            "interface residual compares the work delivered through the "
            "transferred load history with the panel's stored elastic energy "
            "plus plastic dissipation (layer quadrature)"
        ),
    }
    return CoupledResult(
        dem=dem,
        reference=reference_for(h, speed),
        load=load,
        fem=fem,
        sync=sync,
        energy=energy,
    )