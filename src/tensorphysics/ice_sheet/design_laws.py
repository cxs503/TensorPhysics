"""Design laws of a floating ice sheet: peak bending stress and first-crack load.

Contents
--------
1. :func:`foundation_patch_test` -- the **constant-pressure patch test** of the
   BFS plate element: for a uniform deflection field ``w = w0`` the foundation
   reaction of *every* element must be exactly ``q_e = k w0`` (Winkler) and the
   Pasternak shear layer must contribute exactly nothing (``grad w = 0``).  This
   is the element-level consistency check that a mixed foundation model has to
   pass before any field quantity is trusted.
2. :func:`unit_moment_field` -- peak bending moment per unit applied load of a
   circular contact patch, from the closed-form Pasternak/Hetényi field.  Owing
   to linearity the peak moment is proportional to the load, so one evaluation
   gives the whole load range: ``M_max(P) = P * m_hat``.
3. :func:`peak_bending_stress` / :func:`first_crack_load` -- the two design
   quantities ``sigma_max = 6 M_max / h^2`` and the load at which
   ``sigma_max = sigma_f(T, eps_dot)``.
4. :func:`dependence_study` -- the thickness / temperature / loading-rate
   dependence laws of ``sigma_max`` (and of the first-crack load), with
   log-log slopes and a sign report, so that the *signs* of the design trends
   are machine-checkable instead of being asserted in prose.

Physics
-------
The ice sheet is a Kirchhoff plate on a Pasternak foundation,
``D lap^4 w - G_p lap^2 w + k w = q``, with ``k = rho_w g`` and
``G_p = beta G_ice h``.  The closed-form patch field is
:func:`tensorphysics.ice_sheet.pasternak_foundation.pasternak_patch_field`.
``M > 0`` is sagging (bottom tension) in that module; the *first* ring crack of
level ice is driven by the interior **hogging** peak, so the peak magnitude of
``M_r`` and ``M_t`` is used.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import pasternak_foundation as pf
from .ice_material import IceSheetMaterial

__all__ = [
    "foundation_patch_test",
    "unit_moment_field",
    "peak_bending_stress",
    "first_crack_load",
    "dependence_study",
    "loglog_slope",
    "sign_report",
]


# ---------------------------------------------------------------------------
# 1. constant-pressure patch test
# ---------------------------------------------------------------------------
def foundation_patch_test(
    h: float = 1.0,
    E: float = 5.0e9,
    nu: float = 0.3,
    k_w: float = 1025.0 * 9.81,
    g_p: float = 0.0,
    nx: int = 6,
    ny: int = 6,
    length: float = 60.0,
    w0: float = 1.0e-3,
    foundation_model: Optional[str] = None,
) -> Dict[str, object]:
    """Element-wise foundation reaction for a uniform deflection ``w = w0``.

    Builds the BFS element matrices of
    :class:`tensorphysics.ice_sheet.kirchhoff_plate_fem.BFSPlateSolver` and
    applies the *constant-displacement* element vector ``[w0, 0, 0, 0] x 4``
    (BFS nodal DOFs are ``w, theta_x, theta_y, theta_xy``).  For each element the
    integrated foundation reaction is

    ``q_e = 1/A_e * sum(K_f^e @ u_e)``

    which must equal ``k w0`` (the Pasternak shear block ``g_p int grad N grad N``
    annihilates the constant field), and the *equivalent nodal pressure* must be
    uniform.  Reported errors are relative to ``k w0``; a consistent element
    gives ``<= 1e-12`` (round-off only).

    Returns a dict with the max/median relative errors over all elements, the
    separate shear-layer contribution, and ``passed`` for the 1e-10 criterion.
    """
    from .kirchhoff_plate_fem import BFSPlateSolver, KirchhoffWinklerPlate

    if foundation_model is None:
        foundation_model = "pasternak" if g_p > 0.0 else "winkler"
    plate = KirchhoffWinklerPlate(h=h, E=E, nu=nu, k_w=k_w, lx=length, ly=length)
    solver = BFSPlateSolver(
        plate, nx, ny, foundation_model=foundation_model, g_p=float(g_p)
    )
    _, Kf = solver.element_matrices()
    Kg = solver.element_shear_matrix()
    hx_e, hy_e = solver._elem_sizes()
    area = hx_e * hy_e

    u_e = np.zeros((Kf.shape[0], 16))
    u_e[:, [0, 4, 8, 12]] = float(w0)  # w-DOFs of the four corners

    f_total = np.einsum("eij,ej->ei", Kf, u_e)  # nodal foundation forces
    f_shear = np.einsum("eij,ej->ei", Kg, u_e)
    q_total = f_total.sum(axis=1) / area
    q_winkler_only = (f_total - f_shear).sum(axis=1) / area

    target = k_w * float(w0)
    rel_total = np.abs(q_total - target) / target
    rel_shear = np.abs(f_shear.sum(axis=1)) / (target * area)

    return {
        "foundation_model": foundation_model,
        "g_p_N_per_m": float(g_p),
        "k_w_Pa_per_m": float(k_w),
        "w0_m": float(w0),
        "n_elements": int(Kf.shape[0]),
        "target_reaction_Pa": float(target),
        "max_reaction_Pa": float(q_total.max()),
        "min_reaction_Pa": float(q_total.min()),
        "max_rel_error": float(rel_total.max()),
        "median_rel_error": float(np.median(rel_total)),
        "winkler_only_max_rel_error": float(
            np.abs(q_winkler_only - target).max() / target
        ),
        "shear_term_max_rel_contribution": float(rel_shear.max()),
        "total_reaction_N": float(q_total.sum() * 1.0),
        "threshold": 1.0e-10,
        "passed": bool(
            rel_total.max() < 1.0e-10 and rel_shear.max() < 1.0e-10
        ),
    }


# ---------------------------------------------------------------------------
# 2. unit-load moment field (closed form)
# ---------------------------------------------------------------------------
def unit_moment_field(
    material: IceSheetMaterial,
    radius: float,
    n_field: int = 801,
    n_source: int = 24,
    r_max_factor: float = 1.0,
) -> Dict[str, float]:
    """Peak bending moment **per unit applied load** of a disc contact patch.

    ``M_max(P) / P = m_hat`` is load independent (linear plate), so this single
    closed-form evaluation serves the whole load range.  The radial resolution
    matters: the moment peak of a disc load sits *inside* the patch, so the
    sample radius must resolve the contact radius (``r_max_factor = 1`` with
    ``n_field = 801`` gives ``dr ~ R_c/8``; the discretisation error against a
    ``2401 x 48`` reference is ``2e-4``, see the campaign report).  The returned peak is
    ``max(|M_r|, |M_t|)`` over the radial grid; ``r_at_peak`` locates it and
    ``M_r_center``/``M_t_center`` are the centre values (the hogging peak of a
    disc load sits at the centre or just outside the patch).

    All quantities scale with the state from :meth:`IceSheetMaterial.derived`.
    """
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius must be positive and finite, got {radius!r}")
    state = material.derived()
    D = state["D_Nm"]
    k_w = state["k_w_Pa_per_m"]
    g_p = state["G_p_N_per_m"]
    l = state["char_length_m"]
    field = pf.pasternak_patch_field(
        1.0,
        float(radius),
        D,
        k_w,
        g_p,
        nu=material.nu,
        r_max=float(r_max_factor) * l,
        n_field=int(n_field),
        n_source=int(n_source),
    )
    Mr = np.asarray(field["M_r"])
    Mt = np.asarray(field["M_t"])
    r = np.asarray(field["r"])
    mag = np.maximum(np.abs(Mr), np.abs(Mt))
    i = int(np.argmax(mag))
    return {
        "m_hat_Nm_per_m_per_N": float(mag[i]),
        "r_at_peak_m": float(r[i]),
        "M_r_center_Nm_per_m": float(Mr[0]),
        "M_t_center_Nm_per_m": float(Mt[0]),
        "D_Nm": float(D),
        "k_w_Pa_per_m": float(k_w),
        "G_p_N_per_m": float(g_p),
        "char_length_m": float(l),
        "contact_radius_m": float(radius),
        "radius_over_l": float(radius / l),
    }


def peak_bending_stress(
    material: IceSheetMaterial,
    load_N: float,
    radius: float,
    **kwargs,
) -> Dict[str, float]:
    """Peak bending stress ``sigma_max = 6 P m_hat / h^2`` [Pa].

    For a disc contact patch of ``radius`` on the ice sheet.  Additional keyword
    arguments are forwarded to :func:`unit_moment_field`.
    """
    if not np.isfinite(load_N) or load_N <= 0.0:
        raise ValueError(f"load_N must be positive and finite, got {load_N!r}")
    mf = unit_moment_field(material, radius, **kwargs)
    h = float(material.h)
    M_max = float(load_N) * mf["m_hat_Nm_per_m_per_N"]
    return {
        "load_N": float(load_N),
        "sigma_max_Pa": 6.0 * M_max / h**2,
        "M_max_Nm_per_m": M_max,
        "m_hat_Nm_per_m_per_N": mf["m_hat_Nm_per_m_per_N"],
        "r_at_peak_m": mf["r_at_peak_m"],
        "h_m": h,
        "contact_radius_m": mf["contact_radius_m"],
        "char_length_m": mf["char_length_m"],
    }


def first_crack_load(
    material: IceSheetMaterial,
    radius: float,
    **kwargs,
) -> Dict[str, float]:
    """First-crack (ring crack) load ``P_c = M_f / m_hat`` [N].

    ``M_f = sigma_f(T, eps_dot) h^2 / 6`` is the flexural moment capacity of the
    section; the plate being linear, the load that brings the peak moment to
    ``M_f`` is ``P_c = M_f / m_hat`` -- no bisection needed.
    """
    mf = unit_moment_field(material, radius, **kwargs)
    state = material.derived()
    M_f = state["moment_capacity_Nm_per_m"]
    return {
        "first_crack_load_N": M_f / mf["m_hat_Nm_per_m_per_N"],
        "moment_capacity_Nm_per_m": M_f,
        "sigma_f_Pa": state["sigma_f_Pa"],
        "m_hat_Nm_per_m_per_N": mf["m_hat_Nm_per_m_per_N"],
        "h_m": float(material.h),
        "temp_C": float(material.temp_C),
        "indentation_rate_m_s": float(material.indentation_rate_m_s),
        "contact_radius_m": mf["contact_radius_m"],
        "char_length_m": mf["char_length_m"],
    }


# ---------------------------------------------------------------------------
# 3. dependence laws
# ---------------------------------------------------------------------------
def loglog_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Least-squares slope of ``ln y`` against ``ln x`` (all values > 0)."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size < 2 or xa.size != ya.size:
        raise ValueError("need >= 2 paired samples")
    if np.any(xa <= 0.0) or np.any(ya <= 0.0):
        raise ValueError("loglog_slope requires strictly positive samples")
    return float(np.polyfit(np.log(xa), np.log(ya), 1)[0])


def _monotone_sign(x: Sequence[float], y: Sequence[float]) -> int:
    """+1 strictly increasing, -1 strictly decreasing, 0 mixed/constant."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    order = np.argsort(xa)
    d = np.diff(ya[order])
    if np.all(d > 1e-15 * max(np.max(np.abs(ya)), 1.0)):
        return 1
    if np.all(d < -1e-15 * max(np.max(np.abs(ya)), 1.0)):
        return -1
    return 0


def sign_report(rows: Iterable[Dict[str, float]], x_key: str, y_key: str) -> Dict[str, float]:
    """Monotonicity + log-log slope of ``y_key`` against ``x_key`` over rows."""
    rows = list(rows)
    x = [float(r[x_key]) for r in rows]
    y = [float(r[y_key]) for r in rows]
    out = {"x_key": x_key, "y_key": y_key, "sign": int(_monotone_sign(x, y))}
    if len(x) < 2:
        out["loglog_slope"] = float("nan")
        out["n_samples"] = len(x)
        return out
    out["n_samples"] = len(x)
    if min(y) > 0.0:
        out["loglog_slope"] = loglog_slope(x, y)
    # central-difference relative sensitivity at the geometric mid sample
    order = np.argsort(np.asarray(x))
    xs = np.asarray(x)[order]
    ys = np.asarray(y)[order]
    if xs.size >= 3:
        mid = xs.size // 2
        dlnx = math.log(xs[mid + 1]) - math.log(xs[mid - 1])
        dlny = math.log(ys[mid + 1]) - math.log(ys[mid - 1]) if min(ys) > 0 else float("nan")
        out["dln_y_dln_x_mid"] = float(dlny / dlnx)
        out["x_mid"] = float(xs[mid])
        out["y_mid"] = float(ys[mid])
    return out


def dependence_study(
    h_values: Sequence[float] = (0.4, 0.7, 1.0, 1.5, 2.0),
    abs_temp_values: Sequence[float] = (2.0, 5.0, 10.0, 20.0, 30.0),
    rate_values: Sequence[float] = (0.005, 0.02, 0.05, 0.2, 0.5),
    load_N: float = 5.0e5,
    contact_radius: float = 0.5,
    material_kwargs: Optional[Dict[str, float]] = None,
    n_field: int = 801,
    n_source: int = 24,
) -> Dict[str, object]:
    """Thickness / temperature / loading-rate dependence of the design quantities.

    One-factor-at-a-time sweeps around a reference state
    (``h = 1 m``, ``|T| = 10 K``, ``v = 0.05 m/s``) plus the ``(h, |T|)`` grid at
    the reference loading rate -- all at a fixed nominal load ``load_N``.
    For every sweep the log-log slope and the monotonicity sign are reported.

    The three quantities separated here matter:

    * ``sigma_max_Pa`` -- peak bending stress **induced by a fixed load**; this
      is what the campaign criterion names, and it depends on the state only
      through the plate stiffness (``E``), not through ``sigma_f``;
    * ``P_c_N`` -- first-crack load (the strength enters here);
    * ``utilization`` -- ``sigma_max / sigma_f``, i.e. how close the fixed load
      is to cracking the sheet.

    Returns ``{"rows": [...], "signs": {...}, "reference": {...},
    "grid": [...]}`` -- a JSON-serialisable dict.
    """
    mk = dict(material_kwargs or {})
    ref = IceSheetMaterial(h=1.0, temp_C=-10.0, indentation_rate_m_s=0.05, **mk)
    field_kw = {"n_field": int(n_field), "n_source": int(n_source)}

    def evaluate(material: IceSheetMaterial) -> Dict[str, float]:
        fail = first_crack_load(material, contact_radius, **field_kw)
        state = material.derived()
        stress = 6.0 * load_N * fail["m_hat_Nm_per_m_per_N"] / material.h**2
        return {
            "h_m": float(material.h),
            "temp_C": float(material.temp_C),
            "abs_temp_C": float(material.abs_temp_C),
            "indentation_rate_m_s": float(material.indentation_rate_m_s),
            "eps_dot_1_s": state["eps_dot_1_s"],
            "E_Pa": state["E_Pa"],
            "char_length_m": state["char_length_m"],
            "sigma_f_Pa": state["sigma_f_Pa"],
            "m_hat_Nm_per_m_per_N": fail["m_hat_Nm_per_m_per_N"],
            "sigma_max_Pa": float(stress),
            "P_c_N": float(fail["first_crack_load_N"]),
            "utilization": float(stress / state["sigma_f_Pa"]),
        }

    sweeps: Dict[str, List[Dict[str, float]]] = {}
    sweeps["thickness"] = [
        evaluate(ref.with_state(h=float(h))) for h in h_values
    ]
    sweeps["temperature"] = [
        evaluate(ref.with_state(temp_C=-float(a))) for a in abs_temp_values
    ]
    sweeps["loading_rate"] = [
        evaluate(ref.with_state(indentation_rate_m_s=float(v))) for v in rate_values
    ]

    signs = {
        "d_sigma_max_d_h": sign_report(sweeps["thickness"], "h_m", "sigma_max_Pa"),
        "d_sigma_max_d_absT": sign_report(
            sweeps["temperature"], "abs_temp_C", "sigma_max_Pa"
        ),
        "d_sigma_max_d_rate": sign_report(
            sweeps["loading_rate"], "indentation_rate_m_s", "sigma_max_Pa"
        ),
        "d_utilization_d_absT": sign_report(
            sweeps["temperature"], "abs_temp_C", "utilization"
        ),
        "d_utilization_d_h": sign_report(sweeps["thickness"], "h_m", "utilization"),
        "d_Pc_d_h": sign_report(sweeps["thickness"], "h_m", "P_c_N"),
        "d_Pc_d_absT": sign_report(sweeps["temperature"], "abs_temp_C", "P_c_N"),
        "d_Pc_d_rate": sign_report(sweeps["loading_rate"], "indentation_rate_m_s", "P_c_N"),
        "d_sigma_max_d_E": sign_report(sweeps["temperature"], "E_Pa", "sigma_max_Pa"),
    }

    grid: List[Dict[str, float]] = []
    for h in h_values:
        for a in abs_temp_values:
            grid.append(
                evaluate(
                    IceSheetMaterial(
                        h=float(h),
                        temp_C=-float(a),
                        indentation_rate_m_s=ref.indentation_rate_m_s,
                        **mk,
                    )
                )
            )

    return {
        "reference": evaluate(ref),
        "sweeps": sweeps,
        "signs": signs,
        "grid": grid,
        "load_N": float(load_N),
        "contact_radius_m": float(contact_radius),
        "material_kwargs": mk,
    }