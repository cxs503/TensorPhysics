"""Sensitivity matrix and time-step convergence study for superstructure icing.

Two machine-checkable studies are provided:

``case_matrix``
    the 27-case ``LWC x MVD x T_air`` matrix (``U = 15 m/s``) required by the
    campaign node: for every case the impingement characteristics and the
    Messinger ice growth on a reference cylinder section are reported
    (``E``, ``beta_max``, stagnation and area-averaged ice thickness, freezing
    fraction, rime/glaze regime, icing intensity).

``time_convergence``
    three time-step levels (``dt = 10, 5, 2.5 s``) of the transient solver; the
    relative change of the final ice thickness between levels is the converged
    quantity (acceptance: < 10 %).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import air
from .droplets import ImpingementResult, impingement
from .flowfield import CylinderFlow
from .messinger import heat_transfer_coefficient, ice_growth_1d, ice_growth_steady

LWC_GRID_G_M3 = (0.5, 1.0, 2.0)  # g/m3
MVD_GRID_UM = (20.0, 50.0, 100.0)  # micrometre (MVD)
T_GRID_C = (-5.0, -15.0, -25.0)  # air temperature
U_DEFAULT = 15.0


@dataclass
class MatrixCase:
    lwc_g_m3: float
    mvd_um: float
    T_air: float
    U: float
    E_total: float
    beta_max: float
    s_limit: float
    h_stag_mm: float
    h_area_mm: float
    freezing_fraction: float
    T_surface: float
    regime: str
    load_kg_m2: float

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["regime"] = self.regime
        return d


def _regime_name(code: int) -> str:
    return "glaze" if int(code) == 1 else "rime"


def area_average(s: np.ndarray, h: np.ndarray) -> float:
    ds = np.gradient(s) if s.size > 1 else np.array([1.0])
    return float(np.sum(h * ds) / max(np.sum(ds), 1e-30))


def run_case(
    T_air: float,
    lwc_g_m3: float,
    mvd_um: float,
    U: float = U_DEFAULT,
    diameter: float = 0.2,
    t_end: float = 600.0,
    n_traj: int = 81,
    n_cell: int = 41,
    impingement_cache: dict | None = None,
) -> MatrixCase:
    """One case of the LWC x MVD x T matrix (cylinder section, steady Messinger)."""
    cache = impingement_cache if impingement_cache is not None else {}
    key = (round(mvd_um, 6), round(U, 6), round(diameter, 6))
    if key not in cache:
        body = CylinderFlow(R=diameter / 2.0, U=U)
        cache[key] = impingement(body, mvd_um * 1e-6, U=U, t_air=T_air, n_traj=n_traj)
    res: ImpingementResult = cache[key]
    s = np.linspace(0.0, res.s_limit if res.s_limit > 0 else 1e-3, n_cell)
    beta = res.beta_at(s)
    lwc = lwc_g_m3 * 1e-3
    h_c = heat_transfer_coefficient(U, diameter, T_air, kind="cylinder")
    out = ice_growth_steady(s, beta, U, T_air, lwc, t_end, h_c, L_char=diameter)
    h = out["h"]
    h_area = area_average(s, h)
    return MatrixCase(
        lwc_g_m3=lwc_g_m3,
        mvd_um=mvd_um,
        T_air=T_air,
        U=U,
        E_total=res.E_total,
        beta_max=float(np.max(res.beta)),
        s_limit=res.s_limit,
        h_stag_mm=float(h[0] * 1e3),
        h_area_mm=h_area * 1e3,
        freezing_fraction=float(out["freezing_fraction"][0]),
        T_surface=float(out["T_surface"][0]),
        regime=_regime_name(out["regime"][0]),
        load_kg_m2=air.RHO_ICE_GLAZE * h_area,
    )


def case_matrix(U: float = U_DEFAULT, **kwargs) -> list[MatrixCase]:
    """The 3x3x3 sensitivity matrix (27 cases)."""
    cache: dict = {}
    out = []
    for Ta in T_GRID_C:
        for lwc in LWC_GRID_G_M3:
            for mvd in MVD_GRID_UM:
                out.append(
                    run_case(Ta, lwc, mvd, U=U, impingement_cache=cache, **kwargs)
                )
    return out


def time_convergence(
    T_air: float = -15.0,
    lwc_g_m3: float = 1.5,
    mvd_um: float = 50.0,
    U: float = U_DEFAULT,
    diameter: float = 0.2,
    t_end: float = 600.0,
    dts=(10.0, 5.0, 2.5),
    n_traj: int = 81,
    n_cell: int = 41,
) -> dict:
    """Time-step convergence of the transient Messinger solver."""
    body = CylinderFlow(R=diameter / 2.0, U=U)
    res = impingement(body, mvd_um * 1e-6, U=U, t_air=T_air, n_traj=n_traj)
    s = np.linspace(0.0, res.s_limit, n_cell)
    beta = res.beta_at(s)
    lwc = lwc_g_m3 * 1e-3
    h_c = heat_transfer_coefficient(U, diameter, T_air, kind="cylinder")
    h_by_dt = {}
    for dt in dts:
        g = ice_growth_1d(s, beta, U, T_air, lwc, t_end, dt=dt, h_c=h_c, L_char=diameter, kind="cylinder")
        h_by_dt[dt] = g.h_final
    ref = h_by_dt[min(dts)]
    rel = {f"dt_{dt:g}": float(100.0 * np.max(np.abs(h_by_dt[dt] - ref)) / max(np.max(ref), 1e-12)) for dt in dts}
    # steady (time-step independent) solution as the limit
    st = ice_growth_steady(s, beta, U, T_air, lwc, t_end, h_c, L_char=diameter)
    rel_steady = float(100.0 * np.max(np.abs(ref - st["h"])) / max(np.max(st["h"]), 1e-12))
    return {
        "dt_values_s": list(dts),
        "rel_change_pct": rel,
        "max_rel_change_pct": float(max(rel.values())),
        "steady_limit_deviation_pct": rel_steady,
        "h_stag_mm": {f"dt_{dt:g}": float(h_by_dt[dt][0] * 1e3) for dt in dts},
        "h_max_mm": {f"dt_{dt:g}": float(np.max(h_by_dt[dt]) * 1e3) for dt in dts},
        "within_10pct": bool(max(rel.values()) < 10.0),
    }