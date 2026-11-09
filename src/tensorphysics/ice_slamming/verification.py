"""Machine-checkable acceptance-criteria harness (P33-W2-N01).

Each criterion is a *comparison* against an independent reference or a
physical identity, never a self-assessment.  The harness returns plain
JSON-serialisable data so every number quoted in the report is machine-read
from the artefact.

Criteria
--------
(a) negative control: ``V_ice = 0`` -> ``P_max = 0`` (physical consistency);
(b) velocity exponent: ``P_max ~ V_ice**alpha`` with ``alpha in [1.5, 2.5]``
    (literature-consistency gate, Wagner ``rho V^2`` scaling family);
(c) energy closure: ``KE_0 = E_kin + E_strain + E_fluid + E_fracture`` with
    relative error < 5 %;
(d) mesh convergence: ``P_max`` moves by < 3 % across three refinement levels
    (N, 2N, 4N);
(e) test suite: executed by the caller (pytest);
(f) reproducibility: two identical runs give bit-identical ``P_max``.
"""

from __future__ import annotations

import math

import numpy as np

from .model import SlammingCase, run_impact
from .properties import default_floe, default_structure, default_water
from .sweep import fit_velocity_exponent, run_sweep

__all__ = ["run_acceptance", "REFERENCE_CASE"]

# reference case for the single-case gates (b, c, d, f)
REFERENCE_CASE = dict(v_ice=2.0, theta_deg=45.0, h_s=2.5)


def _case(**over) -> SlammingCase:
    kw = dict(
        floe=default_floe(), water=default_water(),
        structure=default_structure(), n_plate=16, dt=1.0e-5, t_end=0.05,
    )
    kw.update(REFERENCE_CASE)
    kw.update(over)
    return SlammingCase(**kw)


def run_acceptance(
    v_grid: list | None = None,
    theta_grid: list | None = None,
    h_grid: list | None = None,
    n_base: int = 8,
) -> dict:
    """Run all acceptance gates and return a JSON-serialisable dict."""
    out: dict = {"criteria": {}}

    # (a) negative control ---------------------------------------------
    res0 = run_impact(_case(v_ice=0.0))
    out["criteria"]["a_negative_control"] = {
        "p_max_pa": res0.p_max,
        "threshold": 0.0,
        "pass": bool(res0.p_max == 0.0),
    }

    # (b) velocity exponent --------------------------------------------
    sweep = run_sweep(v_grid=v_grid, theta_grid=theta_grid, h_grid=h_grid)
    alphas = list(sweep.alpha_by_case.values())
    a_min, a_max = min(alphas), max(alphas)
    out["criteria"]["b_velocity_exponent"] = {
        "alpha_envelope": sweep.alpha_envelope,
        "alpha_min": a_min,
        "alpha_max": a_max,
        "alpha_by_case": sweep.alpha_by_case,
        "band": [1.5, 2.5],
        "pass": bool(
            1.5 <= sweep.alpha_envelope <= 2.5
            and all(1.5 <= a <= 2.5 for a in alphas if math.isfinite(a))
        ),
    }

    # (c) energy closure -------------------------------------------------
    res = run_impact(_case())
    out["criteria"]["c_energy_closure"] = {
        "ke0_j": res.ke0,
        "e_kin_end_j": float(res.e_kin[-1]),
        "e_strain_end_j": float(res.e_strain[-1]),
        "e_fluid_end_j": float(res.e_damp[-1] + res.e_water[-1]),
        "e_fracture_j": res.e_fracture,
        "closure_rel": res.energy_closure,
        "threshold": 0.05,
        "pass": bool(res.energy_closure < 0.05),
    }

    # (d) mesh convergence ----------------------------------------------
    pm = {}
    for n in (n_base, 2 * n_base, 4 * n_base):
        r = run_impact(_case(n_plate=n))
        pm[n] = r.p_max
    levels = [pm[n_base], pm[2 * n_base], pm[4 * n_base]]
    spread = (max(levels) - min(levels)) / max(levels)
    out["criteria"]["d_mesh_convergence"] = {
        "p_max_by_n": {str(k): v for k, v in pm.items()},
        "spread_rel": spread,
        "threshold": 0.03,
        "pass": bool(spread < 0.03),
    }

    # (f) reproducibility ------------------------------------------------
    r1 = run_impact(_case())
    r2 = run_impact(_case())
    out["criteria"]["f_reproducibility"] = {
        "p_max_run1_pa": r1.p_max,
        "p_max_run2_pa": r2.p_max,
        "identical": bool(r1.p_max == r2.p_max),
        "pass": bool(r1.p_max == r2.p_max),
    }

    out["sweep"] = sweep.to_dict()
    out["reference_case"] = REFERENCE_CASE
    out["all_pass"] = all(c["pass"] for c in out["criteria"].values())
    return out
