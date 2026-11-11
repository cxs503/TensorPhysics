"""Acceptance driver for the moving-load / FNV node (campaign p360w1-n05).

The driver runs the four machine-checkable criteria of the mission on one
documented design case and returns a JSON-serialisable payload in which every
number carries its own evidence:

``A1``  static midspan deflection of a simply supported beam under a central
        point load, modal superposition vs the closed form ``P L^3 / (48 EI)``
        (target < 3 %).
``A2``  constant force traversing a simply supported beam at the classical
        critical speed ``V_cr = omega_1 L / pi``: Newmark modal integration
        (10 modes) vs the closed-form series (300 modes), including the dynamic
        amplification factor (target < 10 %).
``A3``  FNV speed scan ``V/V_cr in {0.5, 0.8, 1.0, 1.2}`` for the broken-ice
        contact train on the free-free hull girder:
        ``FNV < 0.5`` below ``0.8 V_cr`` and ``FNV > 1.0`` at ``1.0 V_cr``.
``A4``  energy ledger of the moving-load runs: input work = strain + kinetic +
        damping dissipation (target residual < 5 %).

Two auxiliary studies explain the physics and are reported alongside (they are
*not* acceptance gates): the sensitivity of the resonance to the floe-spacing
regularity, and the difference between a sweeping contact and a contact pinned
at the stem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..ice_vibration.modal import HullParameters
from .beam import (
    FREE_FREE,
    SIMPLY_SUPPORTED,
    GirderBeam,
    beam_fem_frequencies,
    classic_critical_speed,
    static_midspan_deflection_analytic,
    static_midspan_deflection_modal,
)
from .fnv import FNV_HABITABLE_MAX, FNV_TOLERABLE_MAX, FnvResult, evaluate_speed
from .ice_field import IceFloeField, build_contact_history, dominant_frequency_hz
from .moving_load import (
    analytic_point_load_response,
    run_contact_history,
    run_moving_point_load,
)

__all__ = [
    "IceMovingLoadCase",
    "build_reference_case",
    "geometry_audit",
    "natural_frequency_audit",
    "static_verification",
    "classic_moving_load_verification",
    "fnv_speed_scan",
    "energy_verification",
    "floe_regularity_study",
    "sweep_vs_stationary_study",
    "evaluate_acceptance",
]

# -- acceptance thresholds (mission metadata) -------------------------------
A1_STATIC_TOLERANCE = 0.03
A2_MOVING_TOLERANCE = 0.10
A4_ENERGY_TOLERANCE = 0.05
MIN_TESTS = 12


@dataclass(frozen=True)
class IceMovingLoadCase:
    """Everything needed to reproduce the node: geometry, beam, ice field, numerics."""

    hull: HullParameters
    beam: GirderBeam
    field: IceFloeField
    zeta: float = 0.005
    n_modes: int = 10
    dt_s: float = 5.0e-4
    duration_s: float = 45.0
    steady_start_s: float = 15.0
    speed_ratios: Tuple[float, ...] = (0.5, 0.8, 1.0, 1.2)
    monitor_x_m: float | None = None
    name: str = "p360w1-n05-design-case"

    # -- derived -----------------------------------------------------------
    @property
    def monitor_x(self) -> float:
        return 0.5 * self.beam.length_m if self.monitor_x_m is None else float(self.monitor_x_m)

    @property
    def first_natural_frequency_hz(self) -> float:
        return float(self.beam.free_free_frequencies_hz(1)[0])

    @property
    def critical_speed_m_s(self) -> float:
        """Operational moving-load resonance speed ``V_cr = s_mean f_1`` [m/s]."""
        return self.field.critical_speed_m_s(self.first_natural_frequency_hz)

    @property
    def classic_critical_speed_m_s(self) -> float:
        """Classical single-force critical speed ``omega_1 L / pi`` of a simply supported beam."""
        return classic_critical_speed(self.beam, 1, SIMPLY_SUPPORTED)

    def static_midspan_load_n(self) -> float:
        """Reference load for the static / amplification normalisations: the floe peak force."""
        return self.field.peak_force_n

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "IceMovingLoadCase",
            "name": self.name,
            "hull": self.hull.verify(),
            "beam": self.beam.verify(),
            "ice_field": {
                "ice_thickness_m": self.field.ice_thickness_m,
                "crushing_pressure_pa": self.field.crushing_pressure_pa,
                "contact_width_m": self.field.contact_width_m,
                "contact_area_m2": self.field.nominal_contact_area_m2,
                "mean_floe_length_m": self.field.mean_floe_length_m,
                "floe_length_cv": self.field.floe_length_cv,
                "strength_cv": self.field.strength_cv,
                "sawtooth_cycles": self.field.sawtooth_cycles,
                "residual_fraction": self.field.residual_fraction,
                "seed": self.field.seed,
                "peak_force_n": self.field.peak_force_n,
            },
            "numerics": {
                "n_modes": self.n_modes,
                "zeta": self.zeta,
                "dt_s": self.dt_s,
                "duration_s": self.duration_s,
                "steady_start_s": self.steady_start_s,
                "monitor_x_m": self.monitor_x,
                "speed_ratios": list(self.speed_ratios),
            },
            "first_natural_frequency_hz": self.first_natural_frequency_hz,
            "critical_speed_m_s": self.critical_speed_m_s,
            "classic_critical_speed_m_s": self.classic_critical_speed_m_s,
        }


def build_reference_case() -> IceMovingLoadCase:
    """Design case of the node: a 150 m ice-class hull girder in 1 m broken ice.

    Parameter provenance (see the node report for the reference list):

    * ``L = 150 m``, ``B = 22 m``, ``T = 7.2 m``, ``Cb = 0.75`` -> displacement
      ``18.27e6 kg`` which the module checks against ``L B T Cb``;
    * ``EI = 7.0e12 N m^2`` (midship ``I = 34 m^4``, ``E = 2.06e11 Pa``);
    * ``rho A = displacement / L = 121 770 kg/m`` (dry hull girder);
    * ice: ``h = 1.0 m``, effective crushing pressure ``2.0 MPa``, contact width
      ``3.5 m`` -> peak contact force ``7 MN`` (local patch load of an
      ice-class 1A ship; global design loads are larger, individual floe
      contacts smaller);
    * floe length ``s = 4.13 m`` with ``cv = 5 %``: this puts the operational
      resonance ``V_cr = s f_1 = 4.96 m/s`` at the mission speed ``V = 5 m/s``;
    * damping ``zeta = 0.5 %`` critical in all modes (lower bound of the
      0.5 - 2 % hull-girder vertical-vibration range);
    * ``n_modes = 10``, ``dt = 0.5 ms``, ``45 s`` record, steady window
      ``t > 15 s`` (the resonance build-up time ``1/(zeta omega_1)`` is 26 s, so
      the window is long enough to show a built-up, not a transient, response).
    """
    hull = HullParameters(
        length_m=150.0,
        breadth_m=22.0,
        draught_m=7.2,
        block_coefficient=0.75,
        displacement_kg=150.0 * 22.0 * 7.2 * 0.75 * 1025.0,
        name="design-150m-ice-class-1A",
    )
    beam = GirderBeam(
        length_m=hull.length_m,
        bending_stiffness_nm2=7.0e12,
        mass_per_length_kg_m=hull.mass_per_length_kg_m,
        name="design-150m-hull-girder",
    )
    field = IceFloeField(
        ice_thickness_m=1.0,
        crushing_pressure_pa=2.0e6,
        contact_width_m=3.5,
        mean_floe_length_m=4.13,
        floe_length_cv=0.05,
        strength_cv=0.20,
        sawtooth_cycles=1,
        residual_fraction=0.40,
        seed=20261110,
    )
    return IceMovingLoadCase(hull=hull, beam=beam, field=field)


# ---------------------------------------------------------------------------
# A0: geometry audit
# ---------------------------------------------------------------------------


def geometry_audit(case: IceMovingLoadCase) -> Dict[str, Any]:
    """L/B/T/Cb consistency plus the Euler-Bernoulli applicability record."""
    hull = case.hull.verify()
    beam = case.beam.verify()
    return {
        "criterion": "geometry",
        "block_coefficient_stated": hull["block_coefficient_stated"],
        "block_coefficient_from_displacement": hull["block_coefficient_from_displacement"],
        "block_coefficient_relative_error": hull["relative_error"],
        "consistent": bool(hull["consistent"]),
        "euler_bernoulli_applicable": bool(beam["euler_bernoulli_applicable"]),
        "first_wavelength_slenderness": beam["first_wavelength_slenderness"],
        "passed": bool(hull["consistent"] and beam["euler_bernoulli_applicable"]),
    }


# ---------------------------------------------------------------------------
# Natural frequencies
# ---------------------------------------------------------------------------


def natural_frequency_audit(case: IceMovingLoadCase, n_elem: int = 60) -> Dict[str, Any]:
    """First three free-free frequencies: analytic vs Hermite beam FE."""
    analytic = case.beam.free_free_frequencies_hz(3)
    fem = beam_fem_frequencies(case.beam, n_elem=n_elem, support=FREE_FREE, n_modes=3)
    rel = np.abs(fem - analytic) / analytic
    return {
        "criterion": "natural_frequencies",
        "analytic_hz": analytic.tolist(),
        "fem_hz": fem.tolist(),
        "relative_error": rel.tolist(),
        "ratio_f2_over_f1": float(analytic[1] / analytic[0]),
        "ratio_f3_over_f1": float(analytic[2] / analytic[0]),
        "n_elem": n_elem,
        "tolerance": 0.01,
        "passed": bool(np.max(rel) < 0.01),
    }


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------


def static_verification(case: IceMovingLoadCase) -> Dict[str, Any]:
    """A1: modal-superposition static midspan deflection vs ``P L^3 / (48 EI)``."""
    load = case.static_midspan_load_n()
    analytic = static_midspan_deflection_analytic(load, case.beam)
    modal = static_midspan_deflection_modal(load, case.beam, n_modes=case.n_modes, support=SIMPLY_SUPPORTED)
    converged = static_midspan_deflection_modal(load, case.beam, n_modes=500, support=SIMPLY_SUPPORTED)
    err = abs(modal - analytic) / analytic
    return {
        "criterion": "A1_static_midspan",
        "load_n": load,
        "analytic_m": analytic,
        "modal_10modes_m": modal,
        "modal_500modes_m": converged,
        "relative_error": float(err),
        "tolerance": A1_STATIC_TOLERANCE,
        "passed": bool(err < A1_STATIC_TOLERANCE),
    }


# ---------------------------------------------------------------------------
# A2
# ---------------------------------------------------------------------------


def classic_moving_load_verification(
    case: IceMovingLoadCase,
    ratios: Sequence[float] = (0.5, 0.8, 1.0, 1.2),
    n_analytic_modes: int = 300,
    samples_per_transit: int = 4000,
) -> Dict[str, Any]:
    """A2: traversing constant force at the classical critical speed.

    The numeric run uses the mission-specified 10 mode truncation; the withheld
    reference is the closed-form modal series truncated at 300 modes (the
    truncation error of the reference is ~1e-6 relative).
    """
    v_cr = case.classic_critical_speed_m_s
    load = case.static_midspan_load_n()
    static_u = static_midspan_deflection_analytic(load, case.beam)
    points: List[Dict[str, Any]] = []
    for ratio in ratios:
        v = ratio * v_cr
        resp = run_moving_point_load(
            case.beam,
            load,
            v,
            n_modes=case.n_modes,
            zeta=0.0,
            support=SIMPLY_SUPPORTED,
            monitor_x_m=case.monitor_x,
            samples_per_transit=samples_per_transit,
        )
        ref = analytic_point_load_response(
            case.beam, load, v, resp.time_s, n_modes=n_analytic_modes, monitor_x_m=case.monitor_x
        )["displacement_m"]
        peak_num = resp.peak_displacement_m
        peak_ref = float(np.max(np.abs(ref)))
        points.append(
            {
                "speed_ratio": float(ratio),
                "speed_m_s": float(v),
                "numeric_peak_m": peak_num,
                "analytic_peak_m": peak_ref,
                "relative_error": float(abs(peak_num - peak_ref) / peak_ref),
                "dynamic_amplification": float(peak_num / static_u),
                "n_modes_numeric": int(resp.modal_q.shape[0]),
            }
        )
    at_cr = next(p for p in points if abs(p["speed_ratio"] - 1.0) < 1e-9)
    worst = max(p["relative_error"] for p in points)
    return {
        "criterion": "A2_classic_moving_load",
        "critical_speed_m_s": v_cr,
        "n_modes_numeric": case.n_modes,
        "n_modes_analytic": n_analytic_modes,
        "static_midspan_m": static_u,
        "points": points,
        "error_at_critical_speed": at_cr["relative_error"],
        "dynamic_amplification_at_critical_speed": at_cr["dynamic_amplification"],
        "worst_relative_error": float(worst),
        "tolerance": A2_MOVING_TOLERANCE,
        "passed": bool(worst < A2_MOVING_TOLERANCE),
    }


# ---------------------------------------------------------------------------
# A3 / A4
# ---------------------------------------------------------------------------


def fnv_speed_scan(
    case: IceMovingLoadCase,
    ratios: Sequence[float] | None = None,
    sweep: bool = True,
    durations: float | None = None,
    dt_s: float | None = None,
    field: IceFloeField | None = None,
    keep_responses: bool = True,
) -> Tuple[List[FnvResult], Dict[float, Any]]:
    """FNV over the ``V/V_cr`` scan; returns the results and the raw responses."""
    ratios = tuple(case.speed_ratios if ratios is None else ratios)
    fld = field or case.field
    v_cr = fld.critical_speed_m_s(case.first_natural_frequency_hz)
    duration = case.duration_s if durations is None else float(durations)
    dt = case.dt_s if dt_s is None else float(dt_s)
    static_u = static_midspan_deflection_modal(
        fld.peak_force_n, case.beam, n_modes=case.n_modes, support=FREE_FREE
    )
    results: List[FnvResult] = []
    responses: Dict[float, Any] = {}
    for ratio in ratios:
        v = ratio * v_cr
        history = build_contact_history(fld, v, duration, dt, case.beam.length_m, sweep=sweep)
        resp = run_contact_history(
            case.beam, history, n_modes=case.n_modes, zeta=case.zeta, support=FREE_FREE,
            monitor_x_m=case.monitor_x,
        )
        result = evaluate_speed(
            resp,
            speed_m_s=v,
            speed_ratio=float(ratio),
            f_ice_hz=fld.repetition_frequency_hz(v),
            steady_window_s=(case.steady_start_s, duration),
            static_displacement_m=static_u,
        )
        results.append(result)
        if keep_responses:
            responses[float(ratio)] = (resp, history)
    return results, responses


def energy_verification(responses: Dict[float, Any]) -> Dict[str, Any]:
    """A4: energy ledger residual of every moving-load run."""
    rows = []
    worst = 0.0
    for ratio in sorted(responses):
        resp, _ = responses[ratio]
        residual = resp.energy_residual_relative
        worst = max(worst, residual)
        rows.append(
            {
                "speed_ratio": float(ratio),
                "work_input_j": float(resp.work_input_j[-1]),
                "damping_dissipated_j": float(resp.damping_dissipated_j[-1]),
                "strain_energy_peak_j": float(np.max(resp.strain_energy_j)),
                "kinetic_energy_peak_j": float(np.max(resp.kinetic_energy_j)),
                "relative_residual": float(residual),
                "relative_residual_percent": float(100.0 * residual),
            }
        )
    return {
        "criterion": "A4_energy_balance",
        "rows": rows,
        "worst_relative_residual": float(worst),
        "tolerance": A4_ENERGY_TOLERANCE,
        "passed": bool(worst < A4_ENERGY_TOLERANCE),
    }


# ---------------------------------------------------------------------------
# Auxiliary physics studies (not acceptance gates)
# ---------------------------------------------------------------------------


def floe_regularity_study(
    case: IceMovingLoadCase, cvs: Sequence[float] = (0.05, 0.15, 0.35)
) -> Dict[str, Any]:
    """How floe-spacing irregularity destroys the coherent resonance.

    A resonant build-up needs the pulse train to stay phase-coherent for
    ``1/(zeta omega_1) ~ 26 s``, i.e. over ~30 contacts; a floe-length
    coefficient of variation of 15 % already randomises the arrival times by
    more than the resonance half-period.
    """
    rows = []
    v = case.critical_speed_m_s
    for cv in cvs:
        fld = IceFloeField(
            ice_thickness_m=case.field.ice_thickness_m,
            crushing_pressure_pa=case.field.crushing_pressure_pa,
            contact_width_m=case.field.contact_width_m,
            mean_floe_length_m=case.field.mean_floe_length_m,
            floe_length_cv=cv,
            strength_cv=case.field.strength_cv,
            sawtooth_cycles=case.field.sawtooth_cycles,
            residual_fraction=case.field.residual_fraction,
            seed=case.field.seed,
        )
        results, responses = fnv_speed_scan(
            case, ratios=(1.0,), field=fld, keep_responses=True
        )
        res = results[0]
        resp, history = responses[1.0]
        rows.append(
            {
                "floe_length_cv": float(cv),
                "speed_ratio": 1.0,
                "fnv": res.fnv,
                "a_band_m_s2": res.a_band_m_s2,
                "f_dominant_hz": res.f_dominant_hz,
                "force_dominant_hz": dominant_frequency_hz(history.time_s, history.force_n),
                "peak_displacement_m": res.peak_displacement_m,
                "energy_residual_relative": res.energy_residual_relative,
            }
        )
    return {"study": "floe_regularity", "speed_m_s": v, "rows": rows}


def sweep_vs_stationary_study(case: IceMovingLoadCase) -> Dict[str, Any]:
    """Moving contact vs contact pinned at the stem, both at ``V_cr``.

    Isolates the *moving-load* part of the response from the *periodic
    excitation* part, and adds a single constant force crossing at the same
    operational speed to show that the pure moving-load effect at 5 m/s is
    negligible (quasi-static) at hull-girder scale.
    """
    fld = case.field
    v = case.critical_speed_m_s
    static_u = static_midspan_deflection_modal(
        fld.peak_force_n, case.beam, n_modes=case.n_modes, support=FREE_FREE
    )
    out: Dict[str, Any] = {"study": "sweep_vs_stationary", "speed_m_s": v}
    for label, sweep in (("sweeping_contact", True), ("stationary_contact", False)):
        history = build_contact_history(fld, v, case.duration_s, case.dt_s, case.beam.length_m, sweep=sweep)
        resp = run_contact_history(
            case.beam, history, n_modes=case.n_modes, zeta=case.zeta, support=FREE_FREE,
            monitor_x_m=case.monitor_x,
        )
        res = evaluate_speed(
            resp,
            speed_m_s=v,
            speed_ratio=1.0,
            f_ice_hz=fld.repetition_frequency_hz(v),
            steady_window_s=(case.steady_start_s, case.duration_s),
            static_displacement_m=static_u,
        )
        out[label] = {
            "fnv": res.fnv,
            "a_band_m_s2": res.a_band_m_s2,
            "f_dominant_hz": res.f_dominant_hz,
            "peak_displacement_m": res.peak_displacement_m,
        }
    # Is the *translation* itself resonant at ship speed?  Answer with the
    # closed-form single-force solution on the simply supported reference beam
    # at the same speed, normalised by the static deflection: the traverse is
    # quasi-static, so the resonance must come from the contact *repetition*.
    t_transit = case.beam.length_m / v
    t = np.linspace(0.0, t_transit, 4000)
    analytic = analytic_point_load_response(
        case.beam, fld.peak_force_n, v, t, n_modes=300, monitor_x_m=case.monitor_x
    )["displacement_m"]
    static_ss = static_midspan_deflection_analytic(fld.peak_force_n, case.beam)
    out["single_force_transit"] = {
        "speed_m_s": v,
        "speed_ratio_to_classic_critical_speed": v / case.classic_critical_speed_m_s,
        "analytic_peak_m": float(np.max(np.abs(analytic))),
        "static_midspan_m": static_ss,
        "dynamic_amplification": float(np.max(np.abs(analytic)) / static_ss),
        "note": (
            "closed-form single-force traverse (simply supported reference beam) at the "
            "operational speed: the translation alone is quasi-static, so resonance comes "
            "from the floe-contact repetition f_ice = V/s"
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Overall driver
# ---------------------------------------------------------------------------


def evaluate_acceptance(
    case: IceMovingLoadCase | None = None,
    verbose: bool = False,
    collect: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run every criterion and return the report payload.

    ``collect`` (optional) receives the raw time histories of the FNV scan so
    that a plotting script can reuse them instead of re-integrating.
    """
    case = case or build_reference_case()

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    geometry = geometry_audit(case)
    log("geometry done")
    frequencies = natural_frequency_audit(case)
    log("frequencies done")
    a1 = static_verification(case)
    log(f"A1 done: err={a1['relative_error']*100:.3f}%")
    a2 = classic_moving_load_verification(case)
    log(f"A2 done: worst err={a2['worst_relative_error']*100:.3f}%")
    results, responses = fnv_speed_scan(case)
    if collect is not None:
        collect["scan"] = responses
        collect["case"] = case
    log("A3 scan done: FNV=" + ", ".join(f"{r.speed_ratio:g}:{r.fnv:.3f}" for r in results))
    a4 = energy_verification(responses)
    log(f"A4 done: worst residual={a4['worst_relative_residual']*100:.3f}%")
    regularity = floe_regularity_study(case)
    log("regularity study done")
    sweep = sweep_vs_stationary_study(case)
    log("sweep study done")

    by_ratio = {round(r.speed_ratio, 6): r for r in results}
    low = [r for r in results if r.speed_ratio < 0.8]
    at_cr = by_ratio.get(1.0)
    a3_checks = {
        "low_speed_all_habitable": bool(all(r.fnv < FNV_HABITABLE_MAX for r in low)),
        "low_speed_fnv": {f"{r.speed_ratio:g}": r.fnv for r in low},
        "resonance_fnv": None if at_cr is None else at_cr.fnv,
        "resonance_above_tolerable_max": None if at_cr is None else bool(at_cr.fnv > FNV_TOLERABLE_MAX),
    }
    a3 = {
        "criterion": "A3_fnv_speed_scan",
        "speed_ratios": list(case.speed_ratios),
        "critical_speed_m_s": case.critical_speed_m_s,
        "ice_frequency_at_critical_speed_hz": case.field.repetition_frequency_hz(case.critical_speed_m_s),
        "first_natural_frequency_hz": case.first_natural_frequency_hz,
        "tiers": {"habitable_max": FNV_HABITABLE_MAX, "tolerable_max": FNV_TOLERABLE_MAX},
        "results": [r.to_json_dict() for r in results],
        "checks": a3_checks,
        "passed": bool(
            a3_checks["low_speed_all_habitable"] and a3_checks["resonance_above_tolerable_max"]
        ),
    }

    criteria = {
        "geometry": geometry["passed"],
        "A1": a1["passed"],
        "A2": a2["passed"],
        "A3": a3["passed"],
        "A4": a4["passed"],
    }
    payload = {
        "schema": "tp.ice.moving_load_fnv/v1",
        "node": "p360w1-n05-moving-load-fnv",
        "campaign": "p360w1",
        "case": case.to_json_dict(),
        "geometry": geometry,
        "natural_frequencies": frequencies,
        "A1": a1,
        "A2": a2,
        "A3": a3,
        "A4": a4,
        "studies": {"floe_regularity": regularity, "sweep_vs_stationary": sweep},
        "criteria": criteria,
        "all_passed": bool(all(criteria.values())),
        "honest_notes": [
            "one-way coupled ice force (no velocity feedback): self-excited lock-in is out of scope",
            "Euler-Bernoulli uniform beam: no shear deformation, rotary inertia or added-mass distribution",
            "FNV uses the narrowband acceleration at the dominant frequency; the broad-band peak is reported as fnv_raw",
            "the ice field is a design case; FNV magnitudes scale linearly with the peak contact force",
        ],
    }
    return payload