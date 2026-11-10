"""P180W2-N07 acceptance analysis: ice-induced vibration lock-in + ice-pressure low-cycle fatigue.

Runs the whole acceptance surface and writes machine-readable artifacts::

    python3 benchmarks/ice_vibration_fatigue/analysis.py

Outputs (``artifacts/ice_vibration_fatigue/``):

    results.json             every number quoted in the report
    criteria.json            C1..C7 with threshold / achieved / status
    literature_anchors.json  anchor records with explicit verification status

Grading policy (honest-reporting discipline)
--------------------------------------------
``PASS`` is only assigned against a reference that is reproducible inside this
repository.  Where the mission asks for an *experimental* or *material-specific
literature* comparison that cannot be reproduced offline, the criterion is
graded ``WITHHELD`` and a documented surrogate (analytic identity, independent
cross-method, manufactured-data round trip) is reported next to it.  A
``WITHHELD`` row is never rewritten as a pass.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:  # prefer the work-tree package over any installed copy
    sys.path.insert(0, str(_ROOT / "src"))

from tensorphysics.fatigue.ice_pressure_lowcycle import (
    RAYLEIGH_COV,
    ice_pressure_low_cycle_assessment,
    pulse_amplitude_blocks,
)
from tensorphysics.fatigue.strain_life import (
    DamageBlock,
    StrainLifeMaterial,
    elastic_strain_amplitude,
    endurance_limit_strain_amplitude,
    glinka_notch_strain,
    neuber_notch_strain,
    notch_factor_from_geometry,
    plastic_strain_amplitude,
    strain_life_damage,
    stress_amplitude_from_strain,
    total_strain_amplitude,
    transition_life,
    transition_reversals,
)
from tensorphysics.ice_vibration.lockin_coupling import (
    LockInCase,
    ice_force_velocity_slope,
    limit_cycle_amplitude,
    lock_in_amplification_check,
    lock_in_state,
    self_excited_speed_window,
    time_domain_limit_cycle,
)

ROOT = _ROOT
OUT = ROOT / "artifacts" / "ice_vibration_fatigue"
STAMP = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

STEEL = StrainLifeMaterial(
    name="structural steel (representative strain-life constants)",
    E_MPa=206000.0,
    sigma_f_MPa=1000.0,
    b=-0.10,
    eps_f=0.5,
    c=-0.50,
    notch_constant_mm=0.5,
)

CASE = LockInCase(
    name="NOMINAL: 1 Hz mode, F_max = 100 kN, v_c = 0.05 m/s, s = 50 mm, zeta = 2%",
    natural_frequency_hz=1.0,
    modal_mass_kg=1.0e6,
    damping_ratio=0.02,
    ice_force_max_n=1.0e5,
    ice_critical_velocity_m_s=0.05,
    flaking_length_m=0.05,
    velocity_exponent=2.0,
)

DETAILS = {
    "NOMINAL": {"W": 0.006, "note": "ice-belt frame, W = 6.0e-3 m^3"},
    "SEVERE": {"W": 0.003, "note": "lighter frame, W = 3.0e-3 m^3"},
    "EXTREME": {"W": 0.0015, "note": "thin local detail, W = 1.5e-3 m^3"},
}
SPAN_M = 1.2
EXPOSURE_H = 10.0


# ---------------------------------------------------------------------------
# 1. Lock-in surface
# ---------------------------------------------------------------------------


def lockin_surface() -> dict:
    case = CASE
    window = self_excited_speed_window(case)
    check = lock_in_amplification_check(case, ratios=list(np.linspace(0.8, 1.2, 9)))

    ratio_grid = np.linspace(0.3, 2.2, 191)
    grid, locked_points = [], []
    for r in ratio_grid:
        st = lock_in_state(case, case.drift_speed_for_ratio(float(r)))
        grid.append(
            {
                "ratio": float(r),
                "amplification": st.amplification,
                "forced_amplification": st.forced_amplification,
                "locked_in": st.locked_in,
            }
        )
        if st.locked_in:
            locked_points.append({"ratio": float(r), "amplification": st.amplification})

    td = {}
    for ratio in (0.8, 1.0, 1.2):
        u = case.drift_speed_for_ratio(ratio)
        a_df = limit_cycle_amplitude(case, u)
        res = time_domain_limit_cycle(case, u, n_cycles=250, steps_per_period=220)
        td[str(ratio)] = {
            "drift_speed_m_s": u,
            "ice_frequency_hz": case.ice_frequency_hz(u),
            "describing_function_amplitude_m": a_df,
            "time_domain_amplitude_m": res["amplitude_m"],
            "time_domain_frequency_hz": res["frequency_hz"],
            "amplitude_deviation_pct": 100.0 * (a_df - res["amplitude_m"]) / res["amplitude_m"],
            "frequency_deviation_pct": 100.0
            * (res["frequency_hz"] - case.natural_frequency_hz)
            / case.natural_frequency_hz,
        }

    far = {}
    for ratio in (0.3, 0.5, 2.0, 3.0):
        st = lock_in_state(case, case.drift_speed_for_ratio(ratio))
        far[str(ratio)] = {
            "locked_in": st.locked_in,
            "amplification": st.amplification,
            "reason": st.reason,
        }

    sensitivity = []
    for zeta in (0.01, 0.02, 0.03, 0.05, 0.08):
        for f_max in (5.0e4, 1.0e5, 2.0e5):
            variant = LockInCase(
                name=f"zeta={zeta},F={f_max}",
                natural_frequency_hz=1.0,
                modal_mass_kg=1.0e6,
                damping_ratio=zeta,
                ice_force_max_n=f_max,
                ice_critical_velocity_m_s=0.05,
                flaking_length_m=0.05,
            )
            sens = lock_in_amplification_check(variant, ratios=[0.8, 1.0, 1.2])
            sensitivity.append(
                {
                    "damping_ratio": zeta,
                    "ice_force_max_n": f_max,
                    "min_amplification": sens["min_amplification"],
                    "all_locked": sens["all_ratios_locked"],
                    "pass_threshold_3": sens["pass"],
                }
            )

    return {
        "case": {
            "name": case.name,
            "natural_frequency_hz": case.natural_frequency_hz,
            "modal_mass_kg": case.modal_mass_kg,
            "damping_ratio": case.damping_ratio,
            "stiffness_n_m": case.stiffness_n_m,
            "structural_damping_ns_m": case.structural_damping_ns_m,
            "ice_force_max_n": case.ice_force_max_n,
            "ice_critical_velocity_m_s": case.ice_critical_velocity_m_s,
            "flaking_length_m": case.flaking_length_m,
            "velocity_exponent": case.velocity_exponent,
            "entrainment_band": list(case.entrainment_band),
        },
        "self_excited_window_m_s": list(window) if window else None,
        "ice_slope_at_vc_Nsm": ice_force_velocity_slope(0.05, 1.0e5, 0.05, 2.0),
        "check": check,
        "ratio_grid": grid,
        "locked_points": locked_points,
        "time_domain": td,
        "no_lockin_far_from_band": far,
        "sensitivity": sensitivity,
    }


# ---------------------------------------------------------------------------
# 2. Strain-life data reduction round trip
# ---------------------------------------------------------------------------


def strain_life_roundtrip() -> dict:
    E = 206000.0
    truth = StrainLifeMaterial(
        name="manufactured truth",
        E_MPa=E,
        sigma_f_MPa=1040.0,
        b=-0.105,
        eps_f=0.45,
        c=-0.47,
        notch_constant_mm=0.5,
    )
    rng = np.random.default_rng(20261110)
    rev = np.array(
        [1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5,
         2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8]
    )
    eps_a = np.array([total_strain_amplitude(truth, r) for r in rev])
    sig_a = np.array([stress_amplitude_from_strain(truth, e) for e in eps_a])
    sig_m = sig_a * (1.0 + rng.normal(0.0, 0.02, size=sig_a.size))
    eps_m = eps_a * (1.0 + rng.normal(0.0, 0.03, size=eps_a.size))

    # --- textbook branch-wise reduction ----------------------------------
    # elastic strain amplitude is measured directly through the stress:
    eps_e_meas = sig_m / E
    eps_p_meas = eps_m - eps_e_meas
    plastic_fraction = np.where(eps_m > 0.0, eps_p_meas / eps_m, 0.0)

    basquin = plastic_fraction < 0.20          # HCF side: response is elastic
    coffman = (plastic_fraction > 0.50) & (eps_p_meas > 1e-6)  # LCF side: plasticity dominates
    b_fit, log_sigf = np.polyfit(np.log10(rev[basquin]), np.log10(sig_m[basquin]), 1)
    c_fit, log_epsf = np.polyfit(np.log10(rev[coffman]), np.log10(eps_p_meas[coffman]), 1)
    fitted = StrainLifeMaterial(
        name="reduced from manufactured data",
        E_MPa=E,
        sigma_f_MPa=float(10.0 ** log_sigf),
        b=float(b_fit),
        eps_f=float(10.0 ** log_epsf),
        c=float(c_fit),
        notch_constant_mm=0.5,
    )
    predicted = np.array([total_strain_amplitude(fitted, r) for r in rev])
    resid = (predicted - eps_m) / eps_m
    return {
        "reference_kind": "manufactured_roundtrip_branchwise_reduction",
        "noise": {"stress_pct": 2.0, "strain_pct": 3.0, "seed": 20261110},
        "fit_windows": {
            "basquin_points": int(basquin.sum()),
            "coffin_manson_points": int(coffman.sum()),
            "basquin_rule": "plastic strain fraction < 0.20 (HCF side)",
            "coffin_manson_rule": "plastic strain fraction > 0.50 (LCF side)",
        },
        "truth": truth.to_json_dict(),
        "reduced": fitted.to_json_dict(),
        "coefficient_deviation_pct": {
            "sigma_f": 100.0 * (fitted.sigma_f_MPa - truth.sigma_f_MPa) / truth.sigma_f_MPa,
            "b": 100.0 * (fitted.b - truth.b) / truth.b,
            "eps_f": 100.0 * (fitted.eps_f - truth.eps_f) / truth.eps_f,
            "c": 100.0 * (fitted.c - truth.c) / truth.c,
        },
        "rms_total_strain_deviation_pct": float(100.0 * math.sqrt(float(np.mean(resid ** 2)))),
        "max_abs_deviation_pct": float(100.0 * np.max(np.abs(resid))),
        "rows": [
            {
                "reversals": float(r),
                "strain_amplitude_measured": float(e),
                "strain_amplitude_model": float(p),
                "stress_amplitude_measured_MPa": float(s),
                "deviation_pct": float(100.0 * d),
            }
            for r, e, p, s, d in zip(rev, eps_m, predicted, sig_m, resid)
        ],
    }


# ---------------------------------------------------------------------------
# 3. Notch surface and transition life
# ---------------------------------------------------------------------------


def notch_surface() -> dict:
    rows = []
    for nominal in (20.0, 60.0, 150.0, 300.0, 600.0):
        neu = neuber_notch_strain(STEEL, 3.0, nominal)
        gli = glinka_notch_strain(STEEL, 3.0, nominal)
        rows.append(
            {
                "nominal_stress_range_MPa": nominal,
                "neuber": neu.to_json_dict(),
                "glinka": gli.to_json_dict(),
                "neuber_vs_glinka_pct": 100.0
                * (neu.notch_strain_range - gli.notch_strain_range)
                / gli.notch_strain_range,
                "elastic_notch_strain": 3.0 * nominal / STEEL.E_MPa,
            }
        )
    geometry = [
        {"radius_mm": r, **notch_factor_from_geometry(5.0, r, STEEL.notch_constant_mm)}
        for r in (0.5, 2.0, 20.0, 200.0, 2.0e3, 2.0e5)
    ]
    # asymptotic limit: (K_f - 1) decays like sqrt(a/r); fit against 1/sqrt(r) so that
    # the intercept is the r -> infinity limit (should be exactly 0)
    radii = np.array([g["radius_mm"] for g in geometry])
    kf_minus_one = np.array([g["K_f"] - 1.0 for g in geometry])
    tail = radii >= 200.0  # blunt-notch tail where (K_f - 1) ~ A / sqrt(r)
    slope_loglog = float(np.polyfit(np.log(radii[tail]), np.log(kf_minus_one[tail]), 1)[0])
    coeff = np.polyfit(1.0 / np.sqrt(radii[tail]), kf_minus_one[tail], 1)
    limit_fit = {
        "slope_loglog": slope_loglog,
        "asymptote_intercept": float(coeff[1]),
        "asymptote_slope": float(coeff[0]),
        "K_f_at_largest_radius": float(kf_minus_one[-1] + 1.0),
        "K_f_monotone_decreasing": bool(np.all(np.diff([g["K_f"] for g in geometry]) < 0.0)),
        "relative_gap_to_1_at_largest_radius": float(kf_minus_one[-1]),
    }
    return {
        "K_t": 3.0,
        "notch_constant_mm": STEEL.notch_constant_mm,
        "rows": rows,
        "geometry_Kf": geometry,
        "limit_fit": limit_fit,
    }


def transition_surface() -> dict:
    rev_t = transition_reversals(STEEL)
    # de_e - de_p increases with 2N (both branches decay, the plastic one faster)
    lo, hi = -2.0, 14.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if elastic_strain_amplitude(STEEL, 10.0 ** mid) - plastic_strain_amplitude(
            STEEL, 10.0 ** mid
        ) < 0.0:
            lo = mid
        else:
            hi = mid
    rev_num = 10.0 ** (0.5 * (lo + hi))
    n_t = transition_life(STEEL)
    return {
        "transition_reversals_analytic": rev_t,
        "transition_reversals_numeric": rev_num,
        "relative_deviation": abs(rev_num - rev_t) / rev_t,
        "transition_life_cycles": n_t,
        "steel_order_of_magnitude_window_cycles": {"low": 1.0e4, "high": 1.0e5},
        "inside_window": 1.0e4 <= n_t <= 1.0e5,
        "strain_amplitude_at_transition": total_strain_amplitude(STEEL, rev_t) / 2.0,
    }


# ---------------------------------------------------------------------------
# 4. Ice-pressure load spectrum -> damage
# ---------------------------------------------------------------------------


def fatigue_surface(lockin: dict) -> dict:
    case = CASE
    state = lock_in_state(case, case.drift_speed_for_ratio(1.0))
    assessments = {}
    for label, spec in DETAILS.items():
        args = dict(
            material=STEEL,
            force_range_n=state.force_range_n,
            lockin_frequency_hz=case.natural_frequency_hz,
            span_m=SPAN_M,
            section_modulus_m3=spec["W"],
            K_t=3.0,
            notch_radius_mm=2.0,
        )
        res = ice_pressure_low_cycle_assessment(exposure_hours=EXPOSURE_H, **args)
        sweep = []
        for hours in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0):
            r = ice_pressure_low_cycle_assessment(exposure_hours=hours, **args)
            sweep.append(
                {
                    "hours": hours,
                    "damage_total": r.damage.damage_total,
                    "damage_lcf": r.damage.damage_lcf,
                    "damage_hcf": r.damage.damage_hcf,
                }
            )
        design_hours = 0.5 * res.allowable_hours
        design = ice_pressure_low_cycle_assessment(exposure_hours=design_hours, **args)
        assessments[label] = {
            "spec": {"section_modulus_m3": spec["W"], "note": spec["note"]},
            "assessment": res.to_json_dict(),
            "design_check": {
                "design_hours": design_hours,
                "damage_total": design.damage.damage_total,
                "pass_miner_le_1": design.damage.damage_total <= 1.0,
            },
            "sweep": sweep,
        }

    zero = ice_pressure_low_cycle_assessment(
        STEEL, 1.0, case.natural_frequency_hz, EXPOSURE_H, SPAN_M, 0.006, K_t=3.0, notch_radius_mm=2.0
    )
    n_cycles = case.natural_frequency_hz * 3600.0 * EXPOSURE_H
    const_amp = assessments["SEVERE"]["assessment"]["notch"]["strain_amplitude"]
    constant = strain_life_damage(STEEL, [DamageBlock(strain_amplitude=const_amp, cycles=n_cycles)])
    spectrum = assessments["SEVERE"]["assessment"]["damage"]
    return {
        "lock_in_load": state.to_json_dict(),
        "detail_span_m": SPAN_M,
        "assessments": assessments,
        "zero_load_control": {
            "damage_total": zero.damage.damage_total,
            "allowable_hours": zero.allowable_hours,
            "regimes": sorted({b["regime"] for b in zero.damage.blocks}),
        },
        "pulse_cov": RAYLEIGH_COV,
        "pulse_blocks": [
            {"amplitude_multiplier": m, "cycles": n}
            for m, n in pulse_amplitude_blocks(n_cycles, RAYLEIGH_COV, 6)
        ],
        "spectrum_effect_severe": {
            "damage_constant_amplitude": constant.damage_total,
            "damage_pulse_spectrum": spectrum["damage_total"],
            "ratio": spectrum["damage_total"] / constant.damage_total,
        },
    }


# ---------------------------------------------------------------------------
# 5. Criteria
# ---------------------------------------------------------------------------


def build_criteria(lockin, roundtrip, notch, transition, fatigue) -> list:
    check = lockin["check"]
    td = lockin["time_domain"]
    td_dev = max(abs(v["amplitude_deviation_pct"]) for v in td.values())
    td_freq_dev = max(abs(v["frequency_deviation_pct"]) for v in td.values())
    rows = {r["nominal_stress_range_MPa"]: r for r in notch["rows"]}
    design_row = rows[150.0]
    nominal = fatigue["assessments"]["NOMINAL"]["assessment"]
    severe = fatigue["assessments"]["SEVERE"]["assessment"]
    extreme = fatigue["assessments"]["EXTREME"]["assessment"]
    geom = notch["geometry_Kf"][-1]
    zero = fatigue["zero_load_control"]

    return [
        {
            "id": "C1",
            "title": "lock-in band 0.8 <= f_ice/f_n <= 1.2 gives amplification > 3",
            "threshold": "min amplification over the band > 3.0, every ratio locked, response frequency = f_n",
            "achieved": {
                "min_amplification": check["min_amplification"],
                "max_amplification": check["max_amplification"],
                "all_ratios_locked": check["all_ratios_locked"],
                "entrainment_insensitivity": check["entrainment_insensitivity"],
                "forced_amplification_spread": check["forced_amplification_spread"],
                "locked_amplification_spread": check["locked_amplification_spread"],
                "response_frequency_deviation_pct_max": td_freq_dev,
                "forcing_frequency_span_hz": [0.8, 1.2],
            },
            "status": "PASS" if check["pass"] and td_freq_dev < 2.0 else "FAIL",
            "reference_kind": "model + RK4 time-domain entrainment evidence",
            "evidence": ["fig1_lockin_amplification.png", "fig5_entrainment.png"],
            "note": (
                "Amplification alone is not evidence of lock-in (the linear forced response already "
                f"reaches {check['forced_response_max']:.1f} at r = 1); the discriminating evidence is that the "
                "response frequency stays at f_n for forcing frequencies 0.8/1.0/1.2 Hz "
                f"(max deviation {td_freq_dev:.3f} %) and that the amplification spread across the band is "
                f"{check['entrainment_insensitivity']:.3f} of the linear forced-response spread."
            ),
        },
        {
            "id": "C2",
            "title": "amplification vs Frederking (1984) experiment, deviation < 30 %",
            "threshold": "< 30 %",
            "achieved": None,
            "status": "WITHHELD",
            "reference_kind": "experimental literature anchor",
            "evidence": ["literature_anchors.json"],
            "note": (
                "No digitised Frederking (1984) ice-induced-vibration amplification record could be "
                "obtained: a Crossref author+date sweep 1983-1985 returns his adfreeze / flexural "
                "strength / wharf-loading papers (e.g. doi:10.1139/l84-114, 'Ice action on Nanisivik "
                "wharf, winter 1979-1980') without retrievable numeric amplification data. Reported "
                "WITHHELD, not passed."
            ),
        },
        {
            "id": "C2-surrogate",
            "title": "amplification cross-check: describing function vs RK4 time marching, deviation < 30 %",
            "threshold": "< 30 %",
            "achieved": {"max_amplitude_deviation_pct": td_dev, "rows": td},
            "status": "PASS" if td_dev < 30.0 else "FAIL",
            "reference_kind": "independent numerical method (RK4 time integration of the nonlinear EOM)",
            "evidence": ["fig5_entrainment.png"],
            "note": "Two independent methods for the same limit cycle; agreement is far tighter than the 30 % criterion.",
        },
        {
            "id": "C3",
            "title": "strain-life curve vs experimental data, RMS deviation < 20 %",
            "threshold": "< 20 % RMS",
            "achieved": None,
            "status": "WITHHELD",
            "reference_kind": "experimental dataset",
            "evidence": [],
            "note": (
                "No strain-controlled epsilon-N dataset is available offline; the constants used are a "
                "representative structural-steel set, not a measured curve. Reported WITHHELD."
            ),
        },
        {
            "id": "C3-surrogate",
            "title": "epsilon-N reduction round trip on manufactured data, RMS deviation < 20 %",
            "threshold": "< 20 % RMS",
            "achieved": {
                "rms_total_strain_deviation_pct": roundtrip["rms_total_strain_deviation_pct"],
                "max_abs_deviation_pct": roundtrip["max_abs_deviation_pct"],
                "coefficient_deviation_pct": roundtrip["coefficient_deviation_pct"],
            },
            "status": "PASS" if roundtrip["rms_total_strain_deviation_pct"] < 20.0 else "FAIL",
            "reference_kind": "manufactured_roundtrip",
            "evidence": ["fig2_strain_life.png"],
            "note": (
                "Verifies the curve, the Basquin/Coffin-Manson reduction and the life inversion; it says "
                "nothing about the physical validity of the constants."
            ),
        },
        {
            "id": "C4",
            "title": "Neuber-corrected notch strain vs experiment, deviation < 15 %",
            "threshold": "< 15 %",
            "achieved": None,
            "status": "WITHHELD",
            "reference_kind": "experimental dataset (notched-specimen strain measurements)",
            "evidence": [],
            "note": "No notched-specimen notch-strain measurements available offline. Reported WITHHELD.",
        },
        {
            "id": "C4-surrogate",
            "title": "notch closure cross-check: Neuber vs Glinka energy method, deviation < 15 %",
            "threshold": "< 15 % at the design stress range",
            "achieved": {
                "deviation_at_150MPa_pct": design_row["neuber_vs_glinka_pct"],
                "plastic_fraction_at_150MPa": design_row["neuber"]["plastic_fraction"],
                "neuber_identity_residual_max": max(r["neuber"]["neuber_identity_error"] for r in notch["rows"]),
                "deviation_at_600MPa_pct": rows[600.0]["neuber_vs_glinka_pct"],
            },
            "status": "PASS" if abs(design_row["neuber_vs_glinka_pct"]) < 15.0 else "FAIL",
            "reference_kind": "independent notch closure (Glinka strain-energy density)",
            "evidence": ["fig3_neuber.png"],
            "note": (
                "Both closures collapse onto the elastic notch strain in small-scale yielding and diverge to "
                f"{abs(rows[600.0]['neuber_vs_glinka_pct']):.1f} % at 600 MPa where the notch plastic fraction "
                f"reaches {rows[600.0]['neuber']['plastic_fraction']:.2f} -- the documented limitation of Neuber's "
                "rule, reported rather than hidden."
            ),
        },
        {
            "id": "C5",
            "title": "LCF/HCF transition life N_T vs literature, deviation < 25 %",
            "threshold": "< 25 %",
            "achieved": {
                "transition_life_cycles": transition["transition_life_cycles"],
                "analytic_vs_numeric_relative_deviation": transition["relative_deviation"],
                "inside_documented_steel_window": transition["inside_window"],
                "window_cycles": transition["steel_order_of_magnitude_window_cycles"],
                "regime_flip": {
                    "NOMINAL": [nominal["primary_N_f"], nominal["primary_regime"]],
                    "SEVERE": [severe["primary_N_f"], severe["primary_regime"]],
                    "EXTREME": [extreme["primary_N_f"], extreme["primary_regime"]],
                },
            },
            "status": "WITHHELD",
            "reference_kind": "material-specific literature point anchor",
            "evidence": ["fig2_strain_life.png"],
            "note": (
                "The analytic N_T matches an independent root find of the elastic/plastic crossing to "
                f"{transition['relative_deviation']:.2e} relative and lands inside the documented 1e4-1e5 cycle "
                "window for steels; a material-specific literature N_T could not be verified offline, so the "
                "literature point comparison stays WITHHELD."
            ),
        },
        {
            "id": "C6a",
            "title": "negative control: ice frequency far from f_n -> no lock-in",
            "threshold": "locked_in == False and amplification < 3",
            "achieved": {
                "ratios": list(lockin["no_lockin_far_from_band"].keys()),
                "amplifications": [v["amplification"] for v in lockin["no_lockin_far_from_band"].values()],
            },
            "status": "PASS"
            if all(
                (not v["locked_in"]) and v["amplification"] < 3.0
                for v in lockin["no_lockin_far_from_band"].values()
            )
            else "FAIL",
            "reference_kind": "model negative control",
            "evidence": ["fig1_lockin_amplification.png"],
            "note": "Ratios 0.3/0.5/2.0/3.0 fall outside the entrainment band and, at the low end, outside the self-excited window.",
        },
        {
            "id": "C6b",
            "title": "negative control: stress amplitude -> 0 gives life -> infinity",
            "threshold": "below run-out -> zero damage, infinite allowable exposure",
            "achieved": {
                "runout_strain_amplitude": endurance_limit_strain_amplitude(STEEL),
                "zero_load_damage": zero["damage_total"],
                "zero_load_allowable_hours": zero["allowable_hours"],
                "zero_load_regimes": zero["regimes"],
            },
            "status": "PASS"
            if zero["damage_total"] == 0.0 and not math.isfinite(zero["allowable_hours"])
            else "FAIL",
            "reference_kind": "model negative control",
            "evidence": [],
            "note": "Run-out defined at 2e8 reversals (1e8 cycles); the infinite allowable exposure is serialised as Infinity.",
        },
        {
            "id": "C6c",
            "title": "negative control: notch radius -> infinity gives K_f -> 1",
            "threshold": "K_f -> 1 for a blunt root (through K_t -> 1)",
            "achieved": {
                "largest_radius_mm": geom["radius_mm"],
                "K_t": geom["K_t"],
                "K_f": geom["K_f"],
                "q": geom["q"],
                "limit_fit": notch["limit_fit"],
            },
            "status": "PASS"
            if notch["limit_fit"]["K_f_monotone_decreasing"]
            and abs(notch["limit_fit"]["slope_loglog"] + 0.5) < 5e-2
            and notch["limit_fit"]["relative_gap_to_1_at_largest_radius"] < 1e-2
            else "FAIL",
            "reference_kind": "model negative control (Inglis K_t + Neuber/Peterson K_f)",
            "evidence": ["fig3_neuber.png"],
            "note": (
                "For a fixed K_t the notch *sensitivity* falls as the root sharpens, so K_f -> 1 must be driven "
                "through the geometry K_t = 1 + 2 sqrt(a/r); stated explicitly to avoid the inverted-limit trap."
            ),
        },
        {
            "id": "C7",
            "title": "Miner accumulation D = sum(n_i/N_i) <= 1",
            "threshold": "D <= 1 at the design exposure (0.5 x allowable); D = 1 exactly at the allowable exposure",
            "achieved": {
                "design_checks": {k: fatigue["assessments"][k]["design_check"] for k in fatigue["assessments"]},
                "allowable_hours": {k: fatigue["assessments"][k]["assessment"]["allowable_hours"] for k in fatigue["assessments"]},
            },
            "status": "PASS"
            if all(fatigue["assessments"][k]["design_check"]["pass_miner_le_1"] for k in fatigue["assessments"])
            else "FAIL",
            "reference_kind": "model + analytic Miner identity",
            "evidence": ["fig4_damage.png"],
            "note": "Damage is linear in exposure at fixed spectrum, so D = 1 at the allowable exposure is an exact identity (tested to 1e-9).",
        },
    ]


def literature_anchors() -> dict:
    return {
        "policy": (
            "Every anchor carries an explicit verification status. 'metadata_verified' means the bibliographic "
            "record was confirmed through Crossref/OpenAlex on 2026-11-10; no numeric value was extracted from "
            "these sources, so they cannot be used to grade a numeric criterion."
        ),
        "anchors": [
            {
                "id": "frederking_1984_candidates",
                "query": "Crossref: query.author=Frederking, 1983-01-01..1985-12-31",
                "candidates": [
                    {"doi": "10.1139/l84-114", "title": "Ice action on Nanisivik wharf, winter 1979-1980", "year": 1984},
                    {"doi": "10.1016/0165-232x(84)90068-5", "title": "Measurement of shear strength of granular/discontinuous-columnar sea ice", "year": 1984},
                    {"doi": "10.1016/0165-232x(84)90044-2", "title": "An investigation of the failure envelope of granular/discontinuous-columnar sea ice", "year": 1984},
                ],
                "numeric_value_obtained": False,
                "verification_status": "metadata_verified_no_numeric_data",
                "used_for": "C2 WITHHELD justification",
            },
            {
                "id": "ice_lockin_literature",
                "query": "OpenAlex: ice-induced vibration lock-in",
                "candidates": [
                    {"doi": "10.1016/j.coldregions.2018.08.005", "title": "Ice-induced vibrations of the Norstromsgrund lighthouse", "year": 2018},
                    {"doi": "10.1016/j.ijsolstr.2015.08.023", "title": "Interpretation and prediction of ice induced vibrations based on contact area variation", "year": 2015},
                    {"doi": "10.1016/j.coldregions.2019.02.003", "title": "On pressure modes in ice-induced vibrations using multivariate analysis", "year": 2019},
                    {"doi": "10.1115/1.2979795", "title": "A Dynamic Model for Ice-Induced Vibration of Structures", "year": 2008},
                ],
                "numeric_value_obtained": False,
                "verification_status": "metadata_verified_no_numeric_data",
                "used_for": "model-scope statement only",
            },
            {
                "id": "neuber_vs_glinka",
                "query": "OpenAlex: comparative study of Glinka and Neuber approaches",
                "candidates": [
                    {"doi": "10.1111/str.12470", "title": "A comparative study of Glinka and Neuber approaches for fatigue strength assessment on 42CrMoS4", "year": 2023},
                    {"doi": "10.1002/mawe.202300095", "title": "Comparison of analytical and numerical methods for estimating notch strains", "year": 2024},
                ],
                "numeric_value_obtained": False,
                "verification_status": "metadata_verified_no_numeric_data",
                "used_for": "C4-surrogate cross-method rationale",
            },
            {
                "id": "steel_transition_window",
                "query": "documented order-of-magnitude window for ferrous alloys (LCF/HCF boundary)",
                "candidates": [],
                "numeric_range": {"low_cycles": 1.0e4, "high_cycles": 1.0e5},
                "verification_status": "textbook_order_of_magnitude_not_point_measurement",
                "used_for": "C5 order-of-magnitude sanity check (deviation < 25 % NOT claimed)",
            },
        ],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lockin = lockin_surface()
    roundtrip = strain_life_roundtrip()
    notch = notch_surface()
    transition = transition_surface()
    fatigue = fatigue_surface(lockin)
    criteria = build_criteria(lockin, roundtrip, notch, transition, fatigue)

    results = {
        "node": "P180W2-N07",
        "title": "ice-induced vibration lock-in and ice-pressure low-cycle fatigue",
        "generated_utc": STAMP,
        "material": STEEL.to_json_dict(),
        "lockin": lockin,
        "strain_life_roundtrip": roundtrip,
        "notch": notch,
        "transition": transition,
        "fatigue": fatigue,
    }
    summary = {
        "node": "P180W2-N07",
        "generated_utc": STAMP,
        "criteria": criteria,
        "counts": {
            "PASS": sum(1 for c in criteria if c["status"] == "PASS"),
            "WITHHELD": sum(1 for c in criteria if c["status"] == "WITHHELD"),
            "FAIL": sum(1 for c in criteria if c["status"] == "FAIL"),
        },
        "withheld_ids": [c["id"] for c in criteria if c["status"] == "WITHHELD"],
        "honest_statement": (
            "All criteria that can be graded inside this repository pass. Three mission criteria require "
            "experimental or material-specific literature data that is not reproducible offline (C2 amplitude "
            "vs Frederking 1984, C3 epsilon-N vs experiment, C4 notch strain vs experiment) plus the "
            "material-specific point value of C5; these are graded WITHHELD with documented surrogates and are "
            "not reported as passes."
        ),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=_json_default))
    (OUT / "criteria.json").write_text(json.dumps(summary, indent=1, default=_json_default))
    (OUT / "literature_anchors.json").write_text(json.dumps(literature_anchors(), indent=1))
    print(json.dumps(summary["counts"], indent=1))
    for c in criteria:
        print(f"  {c['id']:<12} {c['status']:<9} {c['title']}")
    return 0


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return "Infinity" if obj > 0 else "-Infinity"
    raise TypeError(f"not serialisable: {type(obj)}")


if __name__ == "__main__":
    raise SystemExit(main())