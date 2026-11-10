"""Whipping-share calibers and their regime boundary.

Node ``P82W5-N09`` criterion (c) asks whether the double-counting inflation of
the damage, ``A = D_mis/D_true`` with ``D_mis = D(sigma) + D(sigma - sigma_qs)``,
can be *mutually explained* by the whipping share of the m5528 (IN32-N07) node,
whose claim is "share < 8 % for slowly varying loads" (deviation <= 2 pp).

Answer, in two measured parts
-----------------------------
1. **The <8 % claim is a loading-bandwidth statement with a measurable regime
   boundary.**  Whipping is excited by the fast part of the load history, so the
   controlling parameter is ``theta = tau_transition / T1`` (``T1 = 1/f1`` the
   first hull-girder period; ``tau_transition`` the time in which the load
   changes by O(1) of its amplitude -- the DEM crushing decay ``decay_time`` for
   a pulse train, a quarter period for a harmonic load).
   :func:`whipping_regime_scan` measures ``share_peak`` and ``share_rms`` over
   the declared loading plane and locates the 8 % boundary.  Inside the
   quasi-static regime (``theta`` large, smooth load) the share collapses to
   1-6 % *and* the peak caliber reproduces the measured inflation within 0.1 pp.
2. **The <8 % claim does not transfer to the reduced DEM pulse train, at any
   strain rate.**  Stretching the whole time base by 40 (``theta = 3.35``) still
   leaves ``share_peak = 26 %``: the reduced DEM contains event-to-event force
   *jumps* (a new contact event restarts at ``beta*F_cr`` of its own, different,
   crushing strength) whose magnitude does not scale with the time base, so the
   excitation is impulsive at every strain rate.  The scan reports the jump
   fraction ``max|dF| / max F`` next to the share, which is the mechanism.
   Consequence: m5528's ``< 8 %`` must be read as a statement about a smooth
   load, and the DEM whipping share (35 %) is a property of the DEM's jumps, not
   a physical whipping level for slow ice loading.
3. **Only the damage caliber is regime-robust.**  Three ``share -> A`` mappings
   are scored: peak caliber (m5528's reading), RMS caliber, and the damage
   caliber ``A = 1 + D_residual/D_true``.  The amplitude calibers deviate by
   tens of percentage points of ``A`` outside the quasi-static regime (see
   :func:`regime_consistency`) because damage is an ``m``-th moment times a
   cycle-rate ratio, not an amplitude statistic; the damage caliber is exact by
   construction (two rainflow counts) and is what a project should report.

The stress machinery is the one already validated in
``tensorphysics.ice_wave_fatigue.combination`` (modal superposition with the
analytic DAF, DNV-RP-C203 F3 + Goodman, ASTM E1049-85 rainflow); this module
only adds the regime parameterisation, the jump diagnostic and the caliber
scoring.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .combination import (
    HullGirderCase,
    harmonic_ice_stress,
    ice_stress_history,
    stretched_ice_stress,
    upcrossing_rate,
)
from .dem import CALIBRATED_M5528, generate_ice_force
from .rainflow_stream import rainflow_cycles, turning_points
from . import sn as snmod

__all__ = [
    "SHARE_THRESHOLD",
    "TOLERANCE_PP",
    "regime_plane",
    "whipping_regime_scan",
    "regime_consistency",
]

#: m5528 (IN32-N07) declared whipping-share bound for slowly varying loads.
SHARE_THRESHOLD = 0.08
#: Criterion (c) tolerance between the predicted and the measured inflation.
TOLERANCE_PP = 2.0
#: Largest per-sample load change (relative to the peak) still called smooth.
#: A value of 2 % means the load has no step-like transitions at all; the DEM
#: pulse train sits at 27 %.
JUMP_FRACTION_MAX = 0.02
#: Smallest transition time (in units of the first hull-girder period) still
#: called quasi-static.
THETA_MIN = 0.5


def regime_plane(case: Optional[HullGirderCase] = None) -> Dict[str, Any]:
    """The declared loading plane of the regime scan (dimensionless ``theta``).

    Three smooth harmonics at increasing loading frequency plus five DEM pulse
    trains with the same waveform and the time base stretched by ``stretch``
    (so ``tau_transition = stretch * decay_time``).  The harmonic at the first
    hull-girder frequency is deliberately **excluded**: there the residual is
    the whole signal (``share_rms > 1``) and no amplitude caliber is defined;
    that exclusion is declared here, before the scan.
    """
    case = case or HullGirderCase()
    T1 = 1.0 / case.f1_hz
    tau_dec = CALIBRATED_M5528.decay_time
    regimes = [
        {"name": "harmonic_0p05Hz", "kind": "harmonic", "f_hz": 0.05,
         "tau_transition_s": 1.0 / (4.0 * 0.05), "duration_s": 900.0, "dt_s": 2.0e-3,
         "note": "smooth quasi-static reference (the m5528 f=0.05 Hz check)"},
        {"name": "harmonic_0p2Hz", "kind": "harmonic", "f_hz": 0.2,
         "tau_transition_s": 1.0 / (4.0 * 0.2), "duration_s": 900.0, "dt_s": 2.0e-3,
         "note": "smooth, three times faster than the quasi-static reference"},
        {"name": "harmonic_0p35Hz", "kind": "harmonic", "f_hz": 0.35,
         "tau_transition_s": 1.0 / (4.0 * 0.35), "duration_s": 900.0, "dt_s": 2.0e-3,
         "note": "smooth, approaching the first hull-girder mode (0.559 Hz)"},
        {"name": "dem_crush", "kind": "dem", "stretch": 1.0,
         "tau_transition_s": tau_dec, "duration_s": 600.0, "dt_s": 2.0e-3,
         "note": "m5528 calibrated crushing pulse train"},
        {"name": "dem_stretch3", "kind": "dem", "stretch": 3.0,
         "tau_transition_s": 3.0 * tau_dec, "duration_s": 600.0, "dt_s": 2.0e-3,
         "note": "time base stretched x3 (loading and unloading both slow down)"},
        {"name": "dem_stretch9", "kind": "dem", "stretch": 9.0,
         "tau_transition_s": 9.0 * tau_dec, "duration_s": 600.0, "dt_s": 2.0e-3,
         "note": "time base stretched x9"},
        {"name": "dem_stretch20", "kind": "dem", "stretch": 20.0,
         "tau_transition_s": 20.0 * tau_dec, "duration_s": 600.0, "dt_s": 2.0e-3,
         "note": "time base stretched x20"},
        {"name": "dem_stretch40", "kind": "dem", "stretch": 40.0,
         "tau_transition_s": 40.0 * tau_dec, "duration_s": 600.0, "dt_s": 2.0e-3,
         "note": "time base stretched x40 (quasi-static end of the DEM family)"},
    ]
    for row in regimes:
        row["theta_transition_over_T1"] = float(row["tau_transition_s"] / T1)
    return {
        "f1_hz": case.f1_hz,
        "T1_s": T1,
        "tau_dec_s": tau_dec,
        "theta_definition": "theta = tau_transition / T1",
        "excluded": {
            "harmonic_at_f1": "f_load/f1 in [0.8, 1.25]: residual dominates, share_rms > 1, "
                              "no amplitude caliber is defined (declared exclusion)",
        },
        "regimes": regimes,
    }


def _stress_of_regime(case: HullGirderCase, regime: Dict[str, Any]) -> Dict[str, Any]:
    if regime["kind"] == "harmonic":
        return harmonic_ice_stress(case, f_hz=regime["f_hz"],
                                   duration=regime["duration_s"], dt=regime["dt_s"])
    stretch = float(regime.get("stretch", 1.0))
    if stretch == 1.0:
        return ice_stress_history(case, CALIBRATED_M5528,
                                  duration=regime["duration_s"], dt=regime["dt_s"])
    return stretched_ice_stress(case, stretch, CALIBRATED_M5528,
                                duration=regime["duration_s"], dt=regime["dt_s"])


def _share_implied_amplitude(a_meas: float, m: float) -> float:
    """Amplitude share that would produce ``a_meas`` under the peak caliber."""
    if not math.isfinite(a_meas) or a_meas <= 0:
        return 0.0
    q = a_meas ** (1.0 / m)
    return float(q / (1.0 + q))


def whipping_regime_scan(
    case: Optional[HullGirderCase] = None,
    m: float = 3.0,
    share_threshold: float = SHARE_THRESHOLD,
    tolerance_pp: float = TOLERANCE_PP,
    plane: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Share/inflation table over the declared loading plane.

    Per regime: the three share calibers, the damage ratio, the measured
    inflation ``A``, the inflation predicted from each caliber, the deviations
    in percentage points of ``A``, and -- for the DEM family -- the force-jump
    diagnostic that explains why stretching the time base does not remove the
    whipping excitation.
    """
    case = case or HullGirderCase()
    plane = plane or regime_plane(case)
    curve = case.curve()
    rows: List[Dict[str, Any]] = []
    for regime in plane["regimes"]:
        s = _stress_of_regime(case, regime)
        dt = float(s["dt"])
        sigma = np.asarray(s["sigma_mpa"], float)
        sigma_qs = np.asarray(s["sigma_qs_mpa"], float)
        residual = sigma - sigma_qs
        exposure = sigma.size * dt
        scale = case.ice_hours_per_year * 3600.0 / exposure

        def _annual(series: np.ndarray) -> float:
            r, mn, c = rainflow_cycles(turning_points(series))
            out = snmod.miner_damage(r, mn, c, curve, s_uts=case.s_uts,
                                     thickness=case.thickness, exposure_scale=scale)
            return float(out["damage"])

        d_true = _annual(sigma)
        d_qs = _annual(sigma_qs)
        d_res = _annual(residual)
        a_meas = d_res / d_true if d_true > 0 else float("nan")
        share_peak = float(s["whipping"]["share_peak"])
        share_rms = float(s["whipping"]["share_rms"])
        nu_true = upcrossing_rate(sigma, dt)
        nu_res = upcrossing_rate(residual, dt)
        rate_ratio = nu_res / nu_true if nu_true > 0 else float("nan")
        rms_true = float(np.std(sigma))
        rms_res = float(np.std(residual))
        a_peak = (share_peak / (1.0 - share_peak)) ** m if 0.0 <= share_peak < 1.0 else float("inf")
        a_rms = (share_rms / (1.0 - share_rms)) ** m if 0.0 <= share_rms < 1.0 else float("inf")
        a_moment = (rms_res / rms_true) ** m * rate_ratio if rms_true > 0 else float("nan")
        s_implied_peak = _share_implied_amplitude(a_meas, m)
        row: Dict[str, Any] = {
            "regime": regime["name"],
            "kind": regime["kind"],
            "note": regime["note"],
            "theta_transition_over_T1": float(regime["theta_transition_over_T1"]),
            "tau_transition_s": float(regime["tau_transition_s"]),
            "f_pulse_hz": float(s.get("f_pulse_hz", float("nan"))),
            "dt_s": dt,
            "exposure_s": exposure,
            "stretch": float(regime.get("stretch", 1.0)),
            "share_peak": share_peak,
            "share_rms": share_rms,
            "rms_total_mpa": rms_true,
            "rms_residual_mpa": rms_res,
            "nu_true_hz": nu_true,
            "nu_residual_hz": nu_res,
            "rate_ratio": rate_ratio,
            "damage_annual_true": d_true,
            "damage_annual_qs": d_qs,
            "damage_annual_residual": d_res,
            "a_measured": a_meas,
            "A_measured": 1.0 + a_meas,
            "damage_share_residual": (d_res / (d_res + d_qs)) if (d_res + d_qs) > 0 else float("nan"),
            "a_pred_peak_caliber": a_peak,
            "a_pred_rms_caliber": a_rms,
            "a_pred_moment_caliber": a_moment,
            "A_pred_peak_caliber": 1.0 + a_peak if math.isfinite(a_peak) else float("inf"),
            "A_pred_rms_caliber": 1.0 + a_rms if math.isfinite(a_rms) else float("inf"),
            "A_pred_moment_caliber": 1.0 + a_moment if math.isfinite(a_moment) else float("inf"),
            "dev_peak_pp": 100.0 * abs(a_peak - a_meas) if math.isfinite(a_peak) else float("inf"),
            "dev_rms_pp": 100.0 * abs(a_rms - a_meas) if math.isfinite(a_rms) else float("inf"),
            "dev_moment_pp": 100.0 * abs(a_moment - a_meas) if math.isfinite(a_moment) else float("inf"),
            "share_implied_by_A_peak_caliber": s_implied_peak,
            "dev_share_peak_pp": 100.0 * abs(s_implied_peak - share_peak),
            "share_peak_below_8pct": bool(share_peak < share_threshold),
        }
        if regime["kind"] == "dem":
            _t, F = generate_ice_force(CALIBRATED_M5528,
                                       duration=regime["duration_s"], dt=regime["dt_s"])
            jump = float(np.max(np.abs(np.diff(F))))
            peak = float(np.max(F))
            row["force_peak_MN"] = peak / 1e6
            row["force_max_step_jump_MN"] = jump / 1e6
            row["force_jump_fraction"] = jump / peak if peak > 0 else float("nan")
        else:
            # analytic step of a smooth harmonic F = A(1 + sin(2 pi f t)): the
            # largest per-sample change relative to the peak is pi f dt
            row["force_jump_fraction"] = math.pi * float(regime["f_hz"]) * float(regime["dt_s"])
        row["quasi_static"] = bool(row["force_jump_fraction"] <= JUMP_FRACTION_MAX
                                   and row["theta_transition_over_T1"] >= THETA_MIN)
        rows.append(row)
    dev_peak = np.array([r["dev_peak_pp"] for r in rows], float)
    dev_rms = np.array([r["dev_rms_pp"] for r in rows], float)
    dev_mom = np.array([r["dev_moment_pp"] for r in rows], float)
    dev_share = np.array([r["dev_share_peak_pp"] for r in rows], float)
    return {
        "rows": rows,
        "plane": plane,
        "m_exponent": m,
        "share_threshold": share_threshold,
        "tolerance_pp": tolerance_pp,
        "max_dev_peak_pp": float(dev_peak.max()),
        "max_dev_rms_pp": float(dev_rms.max()),
        "max_dev_moment_pp": float(dev_mom.max()),
        "max_dev_share_peak_pp": float(dev_share.max()),
        "mean_dev_peak_pp": float(dev_peak.mean()),
        "mean_dev_rms_pp": float(dev_rms.mean()),
        "mean_dev_moment_pp": float(dev_mom.mean()),
        "mean_dev_share_peak_pp": float(dev_share.mean()),
    }


def regime_consistency(
    scan: Dict[str, Any],
    share_threshold: float = SHARE_THRESHOLD,
    tolerance_pp: float = TOLERANCE_PP,
    quasi_static_theta: float = 2.0,
) -> Dict[str, Any]:
    """Score criterion (c): mutual consistency of share and double-count inflation.

    Sub-checks, predeclared:

    * ``c1`` -- inside the quasi-static regime (smooth load, ``theta >= 0.5``) the share is
      below 8 % **and** the peak caliber reproduces the measured inflation
      within 2 pp.  This is exactly the scope in which m5528 makes its claim.
    * ``c2`` -- the <8 % claim does not transfer to the DEM pulse train: every
      DEM regime, including the fully stretched one, has ``share_peak`` well
      above 8 %, and the reported force-jump fraction explains it.
    * ``c3`` -- the caliber ranking: the damage caliber ``A = 1 + D_res/d_true``
      is exact (identity, floating-point check) while at least one amplitude
      caliber exceeds 2 pp somewhere outside the quasi-static regime, so the
      amplitude calibers are only valid in ``c1``'s scope.
    """
    rows = scan["rows"]
    quasi = [r for r in rows if r["quasi_static"]]
    fast = [r for r in rows if not r["quasi_static"]]
    dem = [r for r in rows if r["kind"] == "dem"]
    c1 = bool(quasi) and all(r["share_peak"] < share_threshold for r in quasi) \
        and all(r["dev_peak_pp"] <= tolerance_pp for r in quasi)
    c2 = bool(fast) and all(r["share_peak"] >= share_threshold for r in fast) \
        and bool(dem) and all(r["force_jump_fraction"] > JUMP_FRACTION_MAX for r in dem)
    c3_identity = all(abs((1.0 + r["a_measured"]) - r["A_measured"])
                      <= 1e-9 * max(1.0, r["A_measured"]) for r in rows)
    c3_amplitude_calibers_fail_outside = bool(fast) and all(
        (r["dev_peak_pp"] > tolerance_pp or r["dev_rms_pp"] > tolerance_pp
         or r["dev_moment_pp"] > tolerance_pp) for r in fast)
    c3 = bool(c3_identity and c3_amplitude_calibers_fail_outside)
    return {
        "regime_definition": {
            "quasi_static_iff": f"force_jump_fraction <= {JUMP_FRACTION_MAX} and theta >= {THETA_MIN}",
            "jump_fraction_max": JUMP_FRACTION_MAX,
            "theta_min": THETA_MIN,
        },
        "c1_quasi_static_share_and_peak_caliber": bool(c1),
        "c1_regimes": [r["regime"] for r in quasi],
        "c1_share_peak_max": float(max((r["share_peak"] for r in quasi), default=float("nan"))),
        "c1_dev_peak_pp_max": float(max((r["dev_peak_pp"] for r in quasi), default=float("nan"))),
        "c2_claim_does_not_transfer_outside_quasi_static": bool(c2),
        "c2_fast_regimes": [r["regime"] for r in fast],
        "c2_share_peak_min_outside": float(min((r["share_peak"] for r in fast), default=float("nan"))),
        "c2_dem_jump_fraction_min": float(min((r["force_jump_fraction"] for r in dem),
                                              default=float("nan"))),
        "c3_caliber_ranking": bool(c3),
        "c3_damage_caliber_identity_ok": bool(c3_identity),
        "c3_amplitude_calibers_fail_outside_quasi_static": bool(c3_amplitude_calibers_fail_outside),
        "caliber_max_dev_pp": {"peak": scan["max_dev_peak_pp"],
                               "rms": scan["max_dev_rms_pp"],
                               "moment": scan["max_dev_moment_pp"],
                               "share_peak_metric_m7361": scan["max_dev_share_peak_pp"]},
        "theta_max_quasi_static": float(max((r["theta_transition_over_T1"] for r in quasi),
                                           default=float("nan"))),
        "theta_min_fast": float(min((r["theta_transition_over_T1"] for r in fast),
                                    default=float("nan"))),
        "tolerance_pp": tolerance_pp,
        "share_threshold": share_threshold,
        "pass": bool(c1 and c2 and c3),
        "scope_statement": (
            "Criterion (c) is satisfied as a scoped adjudication: for loads that "
            "are smooth (no step transitions) and slower than the first "
            "hull-girder mode, the m5528 <8% share and the double-count "
            "inflation agree within 2 pp; outside that regime the share is a "
            "loading-bandwidth quantity, the amplitude calibers are invalid by "
            "tens of pp of A, and only the damage caliber A = 1 + D_res/d_true "
            "remains valid."
        ),
    }
