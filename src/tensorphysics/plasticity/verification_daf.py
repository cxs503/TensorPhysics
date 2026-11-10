"""Acceptance-criteria harness for the Cowper-Symonds / ice-impact DAF module.

Every criterion is a *comparison*, never a self-assessment: the model under test
and an independent reference (closed form, exact modal solution, or a second
numerical method) are evaluated separately and the difference is reported as a
relative error against the threshold of the task acceptance criteria.

Criteria
--------
(a) Cowper-Symonds uniaxial tension at 3 total strain rates
    (0.001, 0.1, 10 s^-1) vs the closed-form steady-state flow stress
    (perfectly plastic: analytic CS formula; hardening: steady-state
    ``eps_p_dot = eps_dot/(1+H/E)``), tolerance 2 %.
(b) DAF vs literature/reference, 2 thicknesses x 3 pulse durations = 6 cases,
    tolerance 15 %; plus two literature anchors (undamped half-sine peak DAF
    ~ 1.77; mild-steel dynamic/static flow-stress ratio ~ 2 at 100 s^-1).
(c) Rate-independent limit -- exact at ``eps_p_dot == 0``; the literal
    ``eps_p_dot = 1e-4 s^-1`` variant is *reported as FAIL* (see
    ``c_rate_independent_literal``), because classical CS with D = 40.4 s^-1,
    p = 5 amplifies by 7.6 % at that rate; the rate that would be needed for a
    1 % threshold is returned with the criterion.
(d) Quasi-static limit -- ``tau = 10 s`` pulse gives ``|DAF - 1| <= 2 %``.
(e) pytest suite (executed by the caller / benchmark runner).
(f) generic fixture names ``metric_alpha`` / ``metric_beta`` (no domain
    vocabulary in test fixtures) -- asserted by the test module itself.

The harness returns plain JSON-serialisable data so that every number quoted in
the report is machine-read from the artefact.
"""

from __future__ import annotations

import math

import numpy as np

from .cowper_symonds import (
    MILD_STEEL_D,
    MILD_STEEL_P,
    CowperSymonds,
    uniaxial_tension_constant_rate,
)
from .impact_daf import (
    BeamHingeFEM,
    ImpactCase,
    beam_modal_elastic_daf,
    sdof_half_sine_daf,
    sdof_half_sine_daf_rk4,
)

__all__ = [
    "UNIAXIAL_RATES",
    "DAF_THICKNESSES",
    "DAF_DURATIONS",
    "LITERATURE_DAF_MAX_HALF_SINE",
    "LITERATURE_FLOW_STRESS_RATIO_AT_100",
    "verify_cowper_symonds_uniaxial",
    "verify_shock_spectrum",
    "verify_daf_six_cases",
    "verify_rate_independent_limit",
    "verify_daf_plastic_cases",
    "verify_quasi_static_limit",
    "verify_elastic_linearity",
    "daf_surface",
    "run_all_criteria",
]

#: strain rates required by criterion (a) [s^-1]
UNIAXIAL_RATES = (1.0e-3, 0.1, 10.0)
#: plate thicknesses for criterion (b) [m]
DAF_THICKNESSES = (0.008, 0.016)
#: pulse durations for criterion (b) [s]
DAF_DURATIONS = (0.010, 0.030, 0.100)
#: textbook peak DAF of an undamped SDOF under a half-sine pulse (dimensionless)
LITERATURE_DAF_MAX_HALF_SINE = 1.77
#: mild-steel dynamic/static flow-stress ratio at 100 s^-1 (Jones 1989;
#: Campbell & Cooper 1966) -- used as a literature anchor with 15 % tolerance
LITERATURE_FLOW_STRESS_RATIO_AT_100 = 2.0

#: "rate-insensitive" stand-in for D -> infinity: A - 1 < 1e-8 at 1e3 s^-1
RATE_FREE_D = 1.0e45

TOL_UNIAXIAL = 0.02
TOL_DAF = 0.15
TOL_RATE_INDEPENDENT = 0.01
TOL_QUASI_STATIC = 0.02


def _rel_err(model: float, reference: float) -> float:
    return abs(model - reference) / abs(reference) if reference != 0.0 else float("inf")


def _criterion(name, description, model, reference, error, tolerance, note=""):
    return {
        "name": name,
        "description": description,
        "model": float(model),
        "reference": float(reference),
        "relative_error": float(error),
        "tolerance": float(tolerance),
        "passed": bool(error <= tolerance * (1.0 + 1e-9) + 1e-15),
        "note": note,
    }


# ---------------------------------------------------------------------------
# (a) Cowper-Symonds uniaxial tension
# ---------------------------------------------------------------------------
def verify_cowper_symonds_uniaxial(rates=UNIAXIAL_RATES, n_steps: int = 1200) -> dict:
    """CS uniaxial tension at constant total strain rate vs the closed form."""
    cs = CowperSymonds()
    rows_h0, rows_h, criteria = [], [], []
    for rate in rates:
        resp = uniaxial_tension_constant_rate(rate, strain_max=0.02, n_steps=n_steps, hardening=0.0)
        analytic = resp.steady_state_analytic(cs)
        err = _rel_err(resp.final_stress, analytic)
        rows_h0.append(
            {
                "strain_rate": rate,
                "sigma_dynamic": resp.final_stress,
                "sigma_analytic": analytic,
                "relative_error": err,
                "plastic_strain_rate": resp.final_plastic_strain_rate,
                "amplification": resp.final_stress / resp.sigma_0,
            }
        )
        criteria.append(
            _criterion(
                f"a_uniaxial_H0_rate_{rate:g}",
                "Cowper-Symonds uniaxial tension, perfectly plastic, vs analytic sigma_0*A(rate)",
                resp.final_stress,
                analytic,
                err,
                TOL_UNIAXIAL,
            )
        )

    hardening = 2.0e9
    for rate in rates:
        resp = uniaxial_tension_constant_rate(
            rate, strain_max=0.02, n_steps=n_steps, hardening=hardening
        )
        analytic = resp.steady_state_analytic(cs)
        err = _rel_err(resp.final_stress, analytic)
        rows_h.append(
            {
                "strain_rate": rate,
                "sigma_dynamic": resp.final_stress,
                "sigma_analytic": analytic,
                "relative_error": err,
                "plastic_strain_rate": resp.final_plastic_strain_rate,
            }
        )
        criteria.append(
            _criterion(
                f"a_uniaxial_H2GPa_rate_{rate:g}",
                "Cowper-Symonds uniaxial tension, H = 2 GPa, vs steady-state closed form "
                "(eps_p_dot = eps_dot/(1+H/E))",
                resp.final_stress,
                analytic,
                err,
                TOL_UNIAXIAL,
            )
        )

    # analytic inverse round trip
    amp = np.array([1.05, 1.5, 2.2, 3.0])
    round_trip = float(np.max(np.abs(cs.amplification(cs.rate_for_amplification(amp)) - amp)))
    criteria.append(
        _criterion(
            "a_inverse_round_trip",
            "analytic inverse rate = D (A-1)^p reproduces A within machine precision",
            round_trip,
            0.0,
            round_trip,
            1e-12,
        )
    )

    # literature anchor: dynamic/static flow-stress ratio of mild steel at 100 s^-1
    amp100 = float(cs.amplification(100.0))
    criteria.append(
        _criterion(
            "b_literature_flow_stress_ratio_100",
            "CS(D=40.4,p=5) flow-stress ratio at 100 s^-1 vs mild-steel literature value ~2",
            amp100,
            LITERATURE_FLOW_STRESS_RATIO_AT_100,
            _rel_err(amp100, LITERATURE_FLOW_STRESS_RATIO_AT_100),
            TOL_DAF,
            "literature anchor: Jones (1989) ch. 8 / Campbell & Cooper (1966)",
        )
    )

    return {
        "D": MILD_STEEL_D,
        "p": MILD_STEEL_P,
        "perfectly_plastic": rows_h0,
        "hardening_2GPa": rows_h,
        "criteria": criteria,
    }


# ---------------------------------------------------------------------------
# (b) shock spectrum and the six DAF cases
# ---------------------------------------------------------------------------
def verify_shock_spectrum(ratios=(0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 5.0)) -> dict:
    """Exact closed-form half-sine spectrum vs an independent RK4 integration."""
    rows, criteria = [], []
    for ratio in ratios:
        exact = sdof_half_sine_daf(ratio)
        numeric = sdof_half_sine_daf_rk4(ratio, n_steps=24000)
        err = _rel_err(numeric, exact)
        rows.append({"tau_over_T": ratio, "closed_form": exact, "rk4": numeric, "relative_error": err})
        criteria.append(
            _criterion(
                f"b_shock_spectrum_tau_T_{ratio:g}",
                "closed-form half-sine shock spectrum vs independent RK4 integration",
                numeric,
                exact,
                err,
                0.02,
            )
        )
    grid = np.linspace(0.02, 3.0, 600)
    daf = np.array([sdof_half_sine_daf(g) for g in grid])
    i = int(np.argmax(daf))
    peak = {"tau_over_T": float(grid[i]), "daf_max": float(daf[i])}
    criteria.append(
        _criterion(
            "b_shock_spectrum_peak_vs_literature",
            "peak undamped half-sine DAF vs the textbook value 1.77",
            peak["daf_max"],
            LITERATURE_DAF_MAX_HALF_SINE,
            _rel_err(peak["daf_max"], LITERATURE_DAF_MAX_HALF_SINE),
            TOL_DAF,
            "literature anchor: Harris & Crede, Shock and Vibration Handbook",
        )
    )
    return {"rows": rows, "peak": peak, "criteria": criteria}


def verify_daf_six_cases(
    thicknesses=DAF_THICKNESSES,
    durations=DAF_DURATIONS,
    load_ratio: float = 0.5,
    length: float = 1.0,
    width: float = 1.0,
    dt_ratio: float | None = 1.0 / 300.0,
) -> dict:
    """2 thicknesses x 3 pulse durations DAF vs the exact modal reference.

    The load ratio is relative to the *first-yield* pressure of the clamped strip
    (``12 M_y/L^2``).  ``load_ratio = 0.5`` keeps the six reference cases
    elastic, so the comparison isolates the structural (inertia) amplification,
    which the exact modal superposition predicts without any time marching.
    """
    cases, criteria = [], []
    for thickness in thicknesses:
        base = ImpactCase(pressure_amplitude=1.0, pulse_duration=0.01, thickness=thickness,
                          length=length, width=width)
        p_y = base.yield_pressure
        for duration in durations:
            case = ImpactCase(
                pressure_amplitude=load_ratio * p_y,
                pulse_duration=duration,
                thickness=thickness,
                length=length,
                width=width,
            )
            fem = BeamHingeFEM(case)
            t1 = _first_period(case)
            dt = None if dt_ratio is None else dt_ratio * t1
            result = fem.daf(dt=dt)
            ref = beam_modal_elastic_daf(
                length, thickness, load_ratio * p_y, duration,
                youngs_modulus=case.youngs_modulus, density=case.density,
                width=width, poisson=case.poisson, support="clamped",
            )
            err = _rel_err(result["daf_global"], ref["daf_global"])
            row = {
                "thickness": thickness,
                "pulse_duration": duration,
                "pressure_amplitude": load_ratio * p_y,
                "sigma_static_global": result["sigma_static_global"],
                "sigma_dynamic_max": float(np.max(list(result["sigma_dynamic_max_sections"].values()))),
                "daf_fem": result["daf_global"],
                "daf_modal_reference": ref["daf_global"],
                "daf_modal_sections": {"end": ref["daf_end"], "mid": ref["daf_mid"]},
                "relative_error": err,
                "tau_over_T1": duration / ref["first_period"],
                "first_period": ref["first_period"],
                "max_plastic_strain_rate": result["max_plastic_strain_rate"],
                "plastic": result["max_plastic_strain_rate"] > 0.0,
            }
            cases.append(row)
            criteria.append(
                _criterion(
                    f"b_daf_h{thickness*1e3:.0f}mm_tau{duration*1e3:.0f}ms",
                    "plate-strip FE DAF vs exact modal superposition reference",
                    result["daf_global"],
                    ref["daf_global"],
                    err,
                    TOL_DAF,
                )
            )
    max_err = max(c["relative_error"] for c in cases)
    return {
        "load_ratio": load_ratio,
        "cases": cases,
        "max_relative_error": max_err,
        "criteria": criteria,
        "all_passed": all(c["passed"] for c in criteria),
    }


def verify_daf_plastic_cases(
    thicknesses=DAF_THICKNESSES,
    durations=DAF_DURATIONS,
    load_ratio: float = 0.8,
    length: float = 1.0,
    width: float = 1.0,
    dt_ratio: float = 1.0 / 300.0,
) -> dict:
    """Companion set at a load above first yield: inertia + strain-rate DAF.

    Above first yield the stress at the plastic hinge saturates at the
    Cowper-Symonds capacity ``sigma_0 A(eps_p_dot)``, so the DAF of that section
    measures the *material* rate factor, while elastic sections keep the
    structural amplification.  The same cases are re-run with ``D = 1e12 s^-1``
    (rate-insensitive) to separate the two contributions:

    ``DAF_rate = DAF_CS / DAF_rate_independent``.
    """
    rate_free = CowperSymonds(D=RATE_FREE_D, p=MILD_STEEL_P)
    cases, criteria = [], []
    for thickness in thicknesses:
        base = ImpactCase(pressure_amplitude=1.0, pulse_duration=0.01, thickness=thickness,
                          length=length, width=width)
        p_y = base.yield_pressure
        for duration in durations:
            kw = dict(
                pressure_amplitude=load_ratio * p_y,
                pulse_duration=duration,
                thickness=thickness,
                length=length,
                width=width,
            )
            case = ImpactCase(**kw)
            fem = BeamHingeFEM(case)
            t1 = _first_period(case)
            result = fem.daf(dt=dt_ratio * t1)
            case_ri = ImpactCase(cs=rate_free, **kw)
            result_ri = BeamHingeFEM(case_ri).daf(dt=dt_ratio * t1)
            ref = beam_modal_elastic_daf(
                length, thickness, load_ratio * p_y, duration,
                youngs_modulus=case.youngs_modulus, density=case.density,
                width=width, poisson=case.poisson, support="clamped",
            )
            cases.append(
                {
                    "thickness": thickness,
                    "pulse_duration": duration,
                    "pressure_amplitude": load_ratio * p_y,
                    "load_ratio_to_first_yield": load_ratio,
                    "collapse_ratio": case.collapse_pressure / case.pressure_amplitude,
                    "daf_cs": result["daf_global"],
                    "daf_rate_independent": result_ri["daf_global"],
                    "daf_rate_factor": result["daf_global"] / result_ri["daf_global"],
                    "daf_elastic_modal": ref["daf_global"],
                    "max_plastic_strain_rate": result["max_plastic_strain_rate"],
                    "final_plastic_strain": result["final_plastic_strain"],
                    "sigma_dynamic_max_sections": result["sigma_dynamic_max_sections"],
                    "sigma_static_global": result["sigma_static_global"],
                    "tau_over_T1": duration / ref["first_period"],
                }
            )
    # ---- two independent consistency checks of the rate-dependent plasticity ----
    cs = CowperSymonds()
    sigma_0 = ImpactCase(1.0, 0.01).yield_stress
    min_delta, worst_ratio = float("inf"), 0.0
    for row in cases:
        eps_max = max(row["final_plastic_strain"].values())
        rate_max = row["max_plastic_strain_rate"]
        capacity = (sigma_0 + 2.0e9 * eps_max) * float(cs.amplification(rate_max))
        sigma_peak = max(
            row["sigma_dynamic_max_sections"].values()
            if "sigma_dynamic_max_sections" in row
            else [row["daf_cs"]]
        )
        row["sigma_peak"] = sigma_peak
        row["capacity_bound"] = capacity
        row["capacity_ratio"] = sigma_peak / capacity
        min_delta = min(min_delta, row["daf_cs"] - row["daf_rate_independent"])
        worst_ratio = max(worst_ratio, row["capacity_ratio"])

    criteria.append(
        _criterion(
            "b_plastic_rate_effect_monotone",
            "strain-rate sensitivity can only raise the dynamic capacity: "
            "DAF(CS) >= DAF(rate-independent) in every case",
            0.0,
            -min_delta,
            abs(min(0.0, min_delta)),
            1e-12,
        )
    )
    criteria.append(
        _criterion(
            "b_plastic_cs_capacity_bound",
            "peak section stress stays below the CS capacity (sigma_0 + H eps_p) A(eps_p_dot) "
            "of the same run (constitutive self-consistency)",
            worst_ratio,
            1.0,
            worst_ratio - 1.0 if worst_ratio > 1.0 else 0.0,
            1e-6,
        )
    )
    return {"load_ratio": load_ratio, "cases": cases, "criteria": criteria}


# ---------------------------------------------------------------------------
# (c) rate-independent limit
# ---------------------------------------------------------------------------
def verify_rate_independent_limit(rates=(0.0, 1.0e-6, 1.0e-4), target_tol=TOL_RATE_INDEPENDENT) -> dict:
    """Rate-independent limit of the Cowper-Symonds law.

    ``eps_p_dot = 0`` gives ``A == 1`` *exactly* (0 % error).  The literal
    criterion "at 1e-4 s^-1 the amplification is below 1 %" cannot hold for
    D = 40.4 s^-1, p = 5: the CS law is a power law with no lower threshold, so
    the criterion is reported as a FAIL together with the rate that *would* be
    needed.
    """
    cs = CowperSymonds()
    sigma_0 = 235.0e6
    rows, criteria = [], []
    for rate in rates:
        amp = float(cs.amplification(rate))
        rows.append({"strain_rate": rate, "amplification": amp, "relative_deviation": amp - 1.0})
    exact = rows[0]
    criteria.append(
        _criterion(
            "c_exact_rate_zero",
            "A(0) == 1 exactly (rate-independent limit is structural, not a tolerance)",
            exact["amplification"],
            1.0,
            abs(exact["amplification"] - 1.0),
            TOL_RATE_INDEPENDENT,
        )
    )
    literal = rows[-1]
    rate_needed = float(cs.rate_for_target_amplification(1.0 + target_tol))
    criteria.append(
        _criterion(
            "c_rate_independent_literal_1e-4",
            "literal criterion: |sigma_d-sigma_0|/sigma_0 <= 1 % at 1e-4 s^-1 (classical CS, "
            "D=40.4, p=5) -- NOT ACHIEVABLE, reported as FAIL",
            literal["amplification"],
            1.0,
            abs(literal["amplification"] - 1.0),
            target_tol,
            f"CS is a threshold-free power law; a 1 % threshold requires eps_p_dot <= "
            f"{rate_needed:.3g} s^-1 (reported as c_rate_independent_at_needed_rate)",
        )
    )
    needed_amp = float(cs.amplification(rate_needed))
    criteria.append(
        _criterion(
            "c_rate_independent_at_needed_rate",
            "at the rate where CS reaches exactly 1 % amplification, the deviation is <= 1 %",
            needed_amp,
            1.0,
            abs(needed_amp - 1.0),
            target_tol,
        )
    )
    # D -> infinity limit of the model itself
    rate_free = CowperSymonds(D=RATE_FREE_D, p=MILD_STEEL_P)
    max_dev = max(abs(float(rate_free.amplification(r)) - 1.0) for r in (1.0e-3, 0.1, 10.0, 100.0))
    criteria.append(
        _criterion(
            "c_rate_independent_D_infinity",
            "D -> infinity (rate-insensitive material) leaves A == 1 at all rates",
            1.0 + max_dev,
            1.0,
            max_dev,
            TOL_RATE_INDEPENDENT,
        )
    )
    # uniaxial stress at zero rate equals the quasi-static yield stress
    resp = uniaxial_tension_constant_rate(1.0e-12, strain_max=0.02, n_steps=400, hardening=0.0)
    err = _rel_err(resp.final_stress, sigma_0)
    criteria.append(
        _criterion(
            "c_uniaxial_zero_rate",
            "uniaxial stress at eps_dot = 1e-12 s^-1 equals sigma_0",
            resp.final_stress,
            sigma_0,
            err,
            TOL_RATE_INDEPENDENT,
        )
    )
    return {"rows": rows, "rate_for_one_percent": rate_needed, "criteria": criteria}


# ---------------------------------------------------------------------------
# (d) quasi-static limit
# ---------------------------------------------------------------------------
def verify_quasi_static_limit(
    thicknesses=DAF_THICKNESSES, duration: float = 10.0, load_ratio: float = 0.5
) -> dict:
    """Very long pulse (10 s) must give ``DAF -> 1`` within 2 %."""
    criteria, rows = [], []
    for thickness in thicknesses:
        base = ImpactCase(pressure_amplitude=1.0, pulse_duration=0.01, thickness=thickness)
        case = ImpactCase(
            pressure_amplitude=load_ratio * base.yield_pressure,
            pulse_duration=duration,
            thickness=thickness,
        )
        t1 = _first_period(case)
        result = BeamHingeFEM(case).daf(dt=t1 / 40.0)
        rows.append(
            {
                "thickness": thickness,
                "pulse_duration": duration,
                "tau_over_T1": duration / t1,
                "daf": result["daf_global"],
                "deviation_from_one": abs(result["daf_global"] - 1.0),
                "sdof_reference": sdof_half_sine_daf(duration / t1),
            }
        )
        criteria.append(
            _criterion(
                f"d_quasi_static_h{thickness*1e3:.0f}mm",
                "pulse duration 10 s: |DAF - 1| <= 2 %",
                result["daf_global"],
                1.0,
                abs(result["daf_global"] - 1.0),
                TOL_QUASI_STATIC,
            )
        )
    return {"rows": rows, "criteria": criteria}


# ---------------------------------------------------------------------------
# negative control: linearity in the elastic range
# ---------------------------------------------------------------------------
def verify_elastic_linearity(
    ratios=(0.1, 0.25), duration: float = 0.010, thickness: float = 0.008
) -> dict:
    """Elastic DAF must not depend on the load amplitude (no rate effect below yield)."""
    base = ImpactCase(pressure_amplitude=1.0, pulse_duration=duration, thickness=thickness)
    p_y = base.yield_pressure
    rows = []
    for ratio in ratios:
        case = ImpactCase(
            pressure_amplitude=ratio * p_y, pulse_duration=duration, thickness=thickness
        )
        result = BeamHingeFEM(case).daf()
        ref = beam_modal_elastic_daf(1.0, thickness, ratio * p_y, duration)
        rows.append(
            {
                "load_ratio": ratio,
                "daf_fem": result["daf_global"],
                "daf_modal": ref["daf_global"],
                "max_plastic_strain_rate": result["max_plastic_strain_rate"],
            }
        )
    spread = max(r["daf_fem"] for r in rows) - min(r["daf_fem"] for r in rows)
    mean = float(np.mean([r["daf_fem"] for r in rows]))
    criteria = [
        _criterion(
            "d_elastic_amplitude_invariance",
            "elastic DAF is amplitude independent (spread of two amplitudes <= 1 %)",
            spread,
            0.0,
            spread / mean,
            0.01,
        ),
        _criterion(
            "c_no_plasticity_below_yield",
            "below first yield the plastic strain rate is exactly zero",
            max(r["max_plastic_strain_rate"] for r in rows),
            0.0,
            max(r["max_plastic_strain_rate"] for r in rows),
            1e-12,
        ),
    ]
    return {"rows": rows, "criteria": criteria}


# ---------------------------------------------------------------------------
# DAF(tau, P) surface
# ---------------------------------------------------------------------------
def daf_surface(
    thickness: float = 0.008,
    durations=(0.005, 0.010, 0.020, 0.030, 0.050, 0.100),
    load_ratios=(0.3, 0.5, 0.7, 0.8, 0.9),
    length: float = 1.0,
    width: float = 1.0,
    dt_ratio: float = 1.0 / 300.0,
) -> dict:
    """Parametric scan ``DAF(tau, P)`` for one plate thickness (JSON-ready)."""
    base = ImpactCase(pressure_amplitude=1.0, pulse_duration=0.01, thickness=thickness)
    p_y = base.yield_pressure
    daf = np.zeros((len(load_ratios), len(durations)))
    rate = np.zeros_like(daf)
    daf_ri = np.zeros_like(daf)
    rate_free = CowperSymonds(D=RATE_FREE_D, p=MILD_STEEL_P)
    for i, ratio in enumerate(load_ratios):
        for j, duration in enumerate(durations):
            case = ImpactCase(
                pressure_amplitude=ratio * p_y,
                pulse_duration=duration,
                thickness=thickness,
                length=length,
                width=width,
            )
            t1 = _first_period(case)
            res = BeamHingeFEM(case).daf(dt=dt_ratio * t1)
            case_ri = ImpactCase(
                cs=rate_free,
                pressure_amplitude=ratio * p_y,
                pulse_duration=duration,
                thickness=thickness,
                length=length,
                width=width,
            )
            res_ri = BeamHingeFEM(case_ri).daf(dt=dt_ratio * t1)
            daf[i, j] = res["daf_global"]
            daf_ri[i, j] = res_ri["daf_global"]
            rate[i, j] = res["max_plastic_strain_rate"]
    return {
        "thickness": thickness,
        "yield_pressure": p_y,
        "collapse_pressure": float(16.0 * ImpactCase(1.0, 0.01, thickness=thickness).yield_moment
                                   / (length**2 * width)),
        "durations": list(durations),
        "load_ratios": list(load_ratios),
        "pressure_amplitudes": [float(r * p_y) for r in load_ratios],
        "daf": daf.tolist(),
        "daf_rate_independent": daf_ri.tolist(),
        "daf_rate_factor": (daf / daf_ri).tolist(),
        "max_plastic_strain_rate": rate.tolist(),
        "first_period": float(_first_period(base)),
    }


def _first_period(case: ImpactCase) -> float:
    from .impact_daf import beam_first_mode_period

    return beam_first_mode_period(
        case.length,
        case.youngs_modulus,
        case.thickness,
        case.density,
        case.width,
        case.poisson,
        case.support,
    )


# ---------------------------------------------------------------------------
# full harness
# ---------------------------------------------------------------------------
def run_all_criteria(include_surface: bool = True, surface_scale: str = "full") -> dict:
    """Run every acceptance criterion and return a JSON-ready report."""
    report = {
        "module": "tensorphysics.plasticity.cowper_symonds + impact_daf",
        "cowper_symonds": {"D": MILD_STEEL_D, "p": MILD_STEEL_P},
        "thresholds": {
            "uniaxial": TOL_UNIAXIAL,
            "daf_vs_reference": TOL_DAF,
            "rate_independent": TOL_RATE_INDEPENDENT,
            "quasi_static": TOL_QUASI_STATIC,
        },
    }
    sections = {
        "a_cowper_symonds": verify_cowper_symonds_uniaxial(),
        "b_shock_spectrum": verify_shock_spectrum(),
        "b_daf_six_cases": verify_daf_six_cases(),
        "b_daf_plastic_cases": verify_daf_plastic_cases(),
        "c_rate_independent": verify_rate_independent_limit(),
        "d_quasi_static": verify_quasi_static_limit(),
        "d_elastic_linearity": verify_elastic_linearity(),
    }
    report.update(sections)

    criteria = []
    for key, section in sections.items():
        for crit in section.get("criteria", []):
            criteria.append({**crit, "group": key})
    report["criteria"] = criteria
    report["passed"] = [c["name"] for c in criteria if c["passed"]]
    report["failed"] = [c["name"] for c in criteria if not c["passed"]]
    report["all_passed"] = len(report["failed"]) == 0

    if include_surface:
        if surface_scale == "full":
            report["surface"] = daf_surface()
        else:
            report["surface"] = daf_surface(
                durations=(0.010, 0.030, 0.100), load_ratios=(0.5, 0.8)
            )
    return report


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(run_all_criteria(include_surface=False), indent=2, default=float))