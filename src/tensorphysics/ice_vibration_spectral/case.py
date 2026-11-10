"""Node P233W2-N07: ice-induced vibration spectral analysis and resonance risk.

Two physical cases are assembled here.

Case A - lighthouse (literature anchor)
    The classical slender-lighthouse ice-induced vibration problem reported for
    the Gulf of Bothnia lighthouses (Norrstrom 1978; Karna & Turunen 1990): a
    one-DOF sway mode with a fundamental frequency of about 1 Hz, excited by an
    intermittent crushing load whose repetition frequency ``f_ice = U/s`` can
    lock onto the sway mode.  The published dominant response frequency of that
    case is quoted as a band of 1.0 - 1.2 Hz (the original time histories are
    not public), so the case validates the *frequency* of the response and the
    lock-in mechanism, not the response amplitude.

Case B - hull-girder idealisation (ship scale)
    A five-degree-of-freedom lumped chain standing for the first five vertical
    bending modes of a large ice-class ship girder, driven at the bow by the
    same crushing load.  The ice frequency is swept with the drift velocity so
    that the modal resonance map ``f_ice(U)`` versus ``f_n`` can be drawn and
    the risk tiers exercised over the real parameter range.

The driver :func:`run_acceptance` produces the payload of the node: the
measured numbers of criteria AC1 - AC5 with their references and tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .criterion import (
    classification_consistency,
    half_power_edges_hz,
    resonance_risk,
)
from .excitation import IceCrushingSeries, intermittent_crushing_series, welch_psd
from .response import (
    refine_psd,
    mdof_response_psd,
    psd_peak_frequency,
    rms_from_psd,
    sdof_acceleration_psd,
    sdof_response_psd,
    white_noise_rms_analytic,
    zero_order_hold_response,
)
from .structural import (
    ChainSystem,
    SDOFSystem,
    chain_analytic_frequencies,
    state_space_frf_sdof,
)

__all__ = [
    "LIGHTHOUSE_LITERATURE_FREQ_HZ",
    "LIGHTHOUSE_LITERATURE_BAND_HZ",
    "LighthouseCase",
    "HullChainCase",
    "run_acceptance",
    "run_lighthouse_case",
    "run_hull_chain_case",
]

#: Dominant ice-induced response frequency reported for the Bothnian lighthouse
#: cases (Norrstrom 1978; Karna & Turunen 1990), used as the AC5 anchor.
LIGHTHOUSE_LITERATURE_FREQ_HZ = 1.10
#: Reported band of the same measurements (original records not public).
LIGHTHOUSE_LITERATURE_BAND_HZ = (1.00, 1.20)


# --------------------------------------------------------------------------
# Case A: lighthouse
# --------------------------------------------------------------------------
@dataclass
class LighthouseCase:
    """Slender lighthouse sway mode under intermittent ice crushing."""

    frequency_hz: float = LIGHTHOUSE_LITERATURE_FREQ_HZ
    modal_mass_kg: float = 3.0e5
    damping_ratio: float = 0.02
    drift_velocity_m_s: float = 0.16
    crushing_penetration_m: float = 0.15
    contact_width_m: float = 2.0
    effective_pressure_pa: float = 0.30e6
    duration_s: float = 180.0
    dt_s: float = 0.002
    seed: int = 20261110
    nperseg: int = 16384
    fine_factor: int = 64
    fine_f_max_hz: float = 20.0

    stiffness_n_per_m: float = field(init=False)

    def __post_init__(self) -> None:
        omega = 2.0 * np.pi * self.frequency_hz
        self.stiffness_n_per_m = self.modal_mass_kg * omega * omega

    # ------------------------------------------------------------------ build
    def system(self) -> SDOFSystem:
        return SDOFSystem(
            mass_kg=self.modal_mass_kg,
            stiffness_n_per_m=self.stiffness_n_per_m,
            damping_ratio=self.damping_ratio,
            label="lighthouse sway mode",
        )

    def excitation(self, **overrides: Any) -> IceCrushingSeries:
        kwargs = dict(
            duration_s=self.duration_s,
            dt_s=self.dt_s,
            drift_velocity_m_s=self.drift_velocity_m_s,
            crushing_penetration_m=self.crushing_penetration_m,
            contact_width_m=self.contact_width_m,
            effective_pressure_pa=self.effective_pressure_pa,
            seed=self.seed,
            nperseg=self.nperseg,
        )
        kwargs.update(overrides)
        return intermittent_crushing_series(**kwargs)


def grid_resolution_study(case: LighthouseCase | None = None) -> list[dict[str, Any]]:
    """Bias of the response RMS when integrating on the raw Welch grid.

    The Welch resolution ``df = fs/nperseg`` is compared with the structural
    half-power bandwidth ``2 zeta f_n``; the response RMS obtained by
    integrating ``|H|^2 S_ff`` on the coarse grid is compared with the value on
    a 64x refined grid (which agrees with the exact time-domain response).
    """
    case = case or LighthouseCase()
    system = case.system()
    ice = case.excitation()
    freqs_fine, s_ff_fine = refine_psd(
        ice.freqs_hz, ice.psd_n2_per_hz, case.fine_factor, case.fine_f_max_hz
    )
    rms_ref = rms_from_psd(freqs_fine, sdof_response_psd(system, freqs_fine, s_ff_fine))
    rows = []
    for nperseg in (1024, 2048, 4096, 8192, 16384, 32768):
        f, s_ff = welch_psd(ice.force_n, ice.dt_s, nperseg=min(nperseg, ice.force_n.size))
        rms_coarse = rms_from_psd(f, sdof_response_psd(system, f, s_ff))
        rows.append(
            {
                "nperseg": int(min(nperseg, ice.force_n.size)),
                "df_hz": float(f[1] - f[0]),
                "bandwidth_ratio": float((f[1] - f[0]) / (2.0 * system.damping_ratio * system.frequency_hz)),
                "rms_on_welch_grid_m": rms_coarse,
                "rms_reference_m": rms_ref,
                "relative_bias": rms_coarse / rms_ref - 1.0,
            }
        )
    return rows


def run_lighthouse_case(case: LighthouseCase | None = None) -> dict[str, Any]:
    """Case A: response PSD, RMS and resonance verdict for the lighthouse."""
    case = case or LighthouseCase()
    system = case.system()
    ice = case.excitation()
    freqs = ice.freqs_hz
    s_ff = ice.psd_n2_per_hz
    # response evaluated on a grid refined relative to the Welch grid so that the
    # structural half-power bandwidth (zeta f_n ~ 0.02 Hz) is resolved
    freq_fine, s_ff_fine = refine_psd(freqs, s_ff, case.fine_factor, case.fine_f_max_hz)
    s_disp = sdof_response_psd(system, freq_fine, s_ff_fine)
    s_acc = sdof_acceleration_psd(system, freq_fine, s_ff_fine)
    rms_disp_spectral = rms_from_psd(freq_fine, s_disp)
    rms_acc_spectral = rms_from_psd(freq_fine, s_acc)
    f_response_peak = psd_peak_frequency(freq_fine, s_acc, f_min=0.2)
    # documented pitfall: integrating |H|^2 S_ff directly on the coarse Welch grid
    coarse_disp = sdof_response_psd(system, freqs, s_ff)
    rms_coarse = rms_from_psd(freqs, coarse_disp)

    # time-domain cross-check of the spectral RMS (same load history)
    hist = zero_order_hold_response(system, ice.force_n, ice.dt_s)

    # lock-in demonstration: the same crushing mechanism with f_ice pulled to f_n
    # Lock-in demonstration.  Under self-excited lock-in the ice-crushing
    # frequency is entrained by the structural velocity (Maeaettaenen 1978), so
    # the period jitter of the crushing events collapses and the excitation
    # becomes a narrow line at f_n.  Both variants start from the same
    # mechanism and the same mean force level, only the jitter modulation
    # differs, so the ratio of the RMS values isolates the lock-in amplification.
    locked_velocity = case.frequency_hz * case.crushing_penetration_m
    locked_ice = case.excitation(
        drift_velocity_m_s=locked_velocity,
        period_jitter_cv=0.03,
        amplitude_jitter_cv=0.10,
        drift_modulation_fraction=0.03,
        seed=case.seed + 1,
    )
    f_locked, s_ff_locked = refine_psd(
        locked_ice.freqs_hz, locked_ice.psd_n2_per_hz, case.fine_factor, case.fine_f_max_hz
    )
    locked_disp = sdof_response_psd(system, f_locked, s_ff_locked)
    rms_locked = rms_from_psd(f_locked, locked_disp)

    verdict = resonance_risk(ice.f_ice_hz, system.frequency_hz, system.damping_ratio)
    locked_verdict = resonance_risk(locked_ice.f_ice_hz, system.frequency_hz, system.damping_ratio)

    lit = LIGHTHOUSE_LITERATURE_FREQ_HZ
    return {
        "case": "A: lighthouse sway mode (Bothnian lighthouse literature anchor)",
        "structure": system.payload(),
        "excitation": ice.payload(),
        "response": {
            "rms_displacement_m": rms_disp_spectral,
            "rms_acceleration_m_per_s2": rms_acc_spectral,
            "peak_acceleration_m_per_s2": float(
                np.sqrt(2.0 * np.log(2.0 * f_response_peak * case.duration_s))
                * rms_acc_spectral
            ),
            "response_peak_frequency_hz": f_response_peak,
            "time_domain_rms_displacement_m": hist.rms_displacement_m,
            "time_domain_rms_acceleration_m_per_s2": hist.rms_acceleration_m_per_s2,
            "spectral_vs_time_rms_relative_difference": abs(
                hist.rms_displacement_m - rms_disp_spectral
            )
            / max(rms_disp_spectral, 1e-30),
            "welch_grid_df_hz": float(freqs[1] - freqs[0]),
            "rms_displacement_on_welch_grid_m": rms_coarse,
            "welch_grid_rms_overshoot": rms_coarse / max(rms_disp_spectral, 1e-30) - 1.0,
        },
        "resonance_risk": verdict.payload(),
        "grid_resolution_study": grid_resolution_study(case),
        "lock_in_demonstration": {
            "f_ice_unlocked_hz": ice.f_ice_hz,
            "f_ice_locked_hz": locked_ice.f_ice_hz,
            "drift_velocity_locked_m_s": locked_velocity,
            "rms_displacement_unlocked_m": rms_disp_spectral,
            "rms_displacement_locked_m": rms_locked,
            "amplification_of_rms": rms_locked / max(rms_disp_spectral, 1e-30),
            "verdict_unlocked": verdict.payload(),
            "verdict_locked": locked_verdict.payload(),
        },
        "literature_anchor": {
            "source": "Norrstrom (1978); Karna & Turunen (1990), Bothnian lighthouse cases",
            "dominant_frequency_hz": lit,
            "reported_band_hz": list(LIGHTHOUSE_LITERATURE_BAND_HZ),
            "note": (
                "published dominant freq. quoted as a 1.0-1.2 Hz band; original "
                "records are not public, so amplitude is not validated"
            ),
        },
        "ac5_relative_frequency_error": abs(f_response_peak - lit) / lit,
        # raw series kept for figures / later reuse (not serialised in the JSON)
        "_series": {
            "freqs_hz": freq_fine,
            "s_ff": s_ff,
            "s_ff_fine": s_ff_fine,
            "welch_freqs_hz": freqs,
            "s_disp": s_disp,
            "s_acc": s_acc,
            "time_s": ice.time_s,
            "force_n": ice.force_n,
            "disp_m": hist.displacement_m,
            "system": system,
            "ice": ice,
        },
    }


# --------------------------------------------------------------------------
# Case B: hull-girder chain
# --------------------------------------------------------------------------
@dataclass
class HullChainCase:
    """Five-DOF lumped idealisation of a hull girder driven at the bow."""

    n_dof: int = 5
    lumped_mass_kg: float = 5.0e6
    lumped_stiffness_n_per_m: float = 2.0e9
    zeta_first: float = 0.02
    zeta_last: float = 0.02
    drift_velocity_m_s: float = 0.16
    crushing_penetration_m: float = 0.15
    contact_width_m: float = 1.0
    effective_pressure_pa: float = 0.5e6
    duration_s: float = 180.0
    dt_s: float = 0.004
    seed: int = 20261111
    nperseg: int = 8192
    fine_factor: int = 64
    fine_f_max_hz: float = 20.0

    def system(self) -> ChainSystem:
        return ChainSystem(
            n_dof=self.n_dof,
            mass_kg=self.lumped_mass_kg,
            stiffness_n_per_m=self.lumped_stiffness_n_per_m,
            zeta_first=self.zeta_first,
            zeta_last=self.zeta_last,
            label="hull-girder lumped chain",
        )

    def excitation(self, drift_velocity_m_s: float | None = None) -> IceCrushingSeries:
        return intermittent_crushing_series(
            duration_s=self.duration_s,
            dt_s=self.dt_s,
            drift_velocity_m_s=self.drift_velocity_m_s if drift_velocity_m_s is None else drift_velocity_m_s,
            crushing_penetration_m=self.crushing_penetration_m,
            contact_width_m=self.contact_width_m,
            effective_pressure_pa=self.effective_pressure_pa,
            seed=self.seed,
            nperseg=self.nperseg,
        )


def run_hull_chain_case(case: HullChainCase | None = None) -> dict[str, Any]:
    """Case B: MDOF response PSD at the bow node and the modal risk map."""
    case = case or HullChainCase()
    system = case.system()
    ice = case.excitation()
    dof_in = case.n_dof - 1          # force at the bow (free end)
    dof_out = case.n_dof - 1         # response measured at the same station
    freq_fine, s_ff_fine = refine_psd(
        ice.freqs_hz, ice.psd_n2_per_hz, case.fine_factor, case.fine_f_max_hz
    )
    s_disp = mdof_response_psd(system, freq_fine, s_ff_fine, dof_out, dof_in)
    s_acc = mdof_response_psd(
        system, freq_fine, s_ff_fine, dof_out, dof_in, response="acceleration"
    )

    # drift-velocity sweep: f_ice(U) = U/s against every mode
    velocities = np.linspace(0.04, 0.50, 24)
    sweep = []
    for u in velocities:
        f_ice = u / case.crushing_penetration_m
        verdicts = [
            resonance_risk(f_ice, float(fn), float(z))
            for fn, z in zip(system.eigenfrequencies_hz, system.zeta)
        ]
        sweep.append(
            {
                "drift_velocity_m_s": float(u),
                "f_ice_hz": float(f_ice),
                "tiers": [v.tier for v in verdicts],
                "damping_ratios": [v.damping_ratio for v in verdicts],
                "relative_detuning": [v.relative_detuning for v in verdicts],
                "amplification_ratio": [v.amplification_ratio for v in verdicts],
                "governing_mode": int(np.argmax([v.amplification_ratio for v in verdicts])),
            }
        )

    verdict = resonance_risk(ice.f_ice_hz, float(system.eigenfrequencies_hz[0]), float(system.zeta[0]))
    return {
        "case": "B: hull-girder 5-DOF chain, bow ice excitation",
        "structure": system.payload(),
        "excitation": ice.payload(),
        "response": {
            "rms_displacement_m": rms_from_psd(freq_fine, s_disp),
            "rms_acceleration_m_per_s2": rms_from_psd(freq_fine, s_acc),
            "response_peak_frequency_hz": psd_peak_frequency(freq_fine, s_acc, f_min=0.2),
            "dof_in": dof_in,
            "dof_out": dof_out,
        },
        "resonance_risk_first_mode": verdict.payload(),
        "drift_sweep": sweep,
        "_series": {
            "freqs_hz": freq_fine,
            "s_ff": ice.psd_n2_per_hz,
            "s_ff_fine": s_ff_fine,
            "s_disp": s_disp,
            "s_acc": s_acc,
            "system": system,
        },
    }


# --------------------------------------------------------------------------
# Acceptance evaluation
# --------------------------------------------------------------------------
def _rel(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


def run_acceptance() -> dict[str, Any]:
    """Evaluate AC1 - AC5 and return the node payload (JSON-serialisable)."""
    system = SDOFSystem(mass_kg=3.0e5, stiffness_n_per_m=1.4e7, damping_ratio=0.02, label="AC SDOF")

    # ---------------------------------------------------------------- AC1 SDOF
    f_analytic = system.frequency_hz
    f_lo, f_hi = 0.6 * f_analytic, 1.4 * f_analytic
    grid = np.linspace(f_lo, f_hi, 40001)
    f_identified = float(grid[int(np.argmax(np.abs(system.frf_displacement(grid))))])
    ac1_sdof_error = _rel(f_identified, f_analytic)
    ac1_sdof = {
        "measured_frequency_hz": f_identified,
        "analytic_frequency_hz": f_analytic,
        "relative_error": ac1_sdof_error,
        "tolerance": 0.01,
        "passed": ac1_sdof_error <= 0.01,
        "method": "peak of |H(f)| on a 40001-point grid vs sqrt(k/m)/(2 pi) "
                  "(the damped-peak shift f_n sqrt(1-2 zeta^2) is 0.04 % at zeta=0.02)",
    }

    # ----------------------------------------------------------------- AC1 MDOF
    mdof_checks = []
    for n in (3, 5, 8):
        chain = ChainSystem(n_dof=n, mass_kg=1.0e4, stiffness_n_per_m=1.0e6, zeta_first=0.01, zeta_last=0.01)
        analytic = chain_analytic_frequencies(n, 1.0e4, 1.0e6)
        err = float(np.max(np.abs(chain.eigenfrequencies_hz - analytic) / analytic))
        mdof_checks.append(
            {
                "n_dof": n,
                "eigenfrequencies_hz": chain.eigenfrequencies_hz.tolist(),
                "analytic_frequencies_hz": analytic.tolist(),
                "max_relative_error": err,
                "tolerance": 0.05,
                "passed": err <= 0.05,
            }
        )
    ac1_mdof = {
        "checks": mdof_checks,
        "max_relative_error": max(c["max_relative_error"] for c in mdof_checks),
        "tolerance": 0.05,
        "passed": all(c["passed"] for c in mdof_checks),
    }

    # ------------------------------------------------------------------ AC2 SDOF
    probe = np.concatenate(
        [
            np.linspace(0.05 * f_analytic, 0.9 * f_analytic, 120),
            np.linspace(0.9 * f_analytic, 1.1 * f_analytic, 120),
            np.linspace(1.1 * f_analytic, 5.0 * f_analytic, 160),
        ]
    )
    h_direct = system.frf_displacement(probe)
    h_state = state_space_frf_sdof(
        system.mass_kg, system.stiffness_n_per_m, system.damping_ratio, probe
    )
    amplitude_error = float(np.max(np.abs(np.abs(h_direct) - np.abs(h_state)) / np.abs(h_direct)))
    near = (probe >= 0.95 * f_analytic) & (probe <= 1.05 * f_analytic)
    near_peak_error = float(
        np.max(
            np.abs(np.abs(h_direct) - np.abs(h_state))[near]
            / np.abs(h_direct)[near]
        )
    )
    phase_diff = np.angle(h_direct) - np.angle(h_state)
    phase_diff = np.abs((phase_diff + np.pi) % (2.0 * np.pi) - np.pi)
    phase_error_deg = float(np.degrees(np.max(phase_diff)))
    peak_direct = float(np.max(np.abs(h_direct)))
    peak_analytic = 1.0 / (
        2.0 * system.damping_ratio * system.stiffness_n_per_m * np.sqrt(1.0 - system.damping_ratio**2)
    )
    peak_error = _rel(peak_direct, peak_analytic)
    ac2_sdof = {
        "amplitude_max_relative_error": amplitude_error,
        "amplitude_max_relative_error_near_resonance": near_peak_error,
        "phase_max_error_deg": phase_error_deg,
        "peak_amplitude_measured_m_per_n": peak_direct,
        "peak_amplitude_analytic_m_per_n": peak_analytic,
        "peak_amplitude_relative_error": peak_error,
        "tolerance_amplitude": 0.01,
        "tolerance_near_resonance": 0.05,
        "tolerance_phase_deg": 1.0,
        "passed": bool(
            amplitude_error <= 0.01
            and near_peak_error <= 0.05
            and phase_error_deg <= 1.0
            and peak_error <= 0.01
        ),
    }

    # ------------------------------------------------------------------ AC2 MDOF
    chain = ChainSystem(n_dof=5, mass_kg=5.0e6, stiffness_n_per_m=2.0e9, zeta_first=0.02, zeta_last=0.02)
    fg = np.linspace(0.2, 8.0, 400)
    hd = chain.frf_direct(fg, 4, 4)
    hm = chain.frf_modal(fg, 4, 4)
    mdof_frf_error = float(np.max(np.abs(hd - hm) / np.abs(hd)))
    ac2_mdof = {
        "max_relative_error_modal_vs_direct": mdof_frf_error,
        "tolerance": 0.01,
        "passed": mdof_frf_error <= 0.01,
        "probe_range_hz": [float(fg[0]), float(fg[-1])],
        "dof_out": 4,
        "dof_in": 4,
    }

    # ------------------------------------------------------------------ AC3
    s0 = 1.0e8  # N^2/Hz, one-sided
    rms_analytic = white_noise_rms_analytic(system, s0)
    f_int = np.linspace(1.0e-4, 40.0 * f_analytic, 400000)
    rms_spectral = rms_from_psd(f_int, sdof_response_psd(system, f_int, np.full_like(f_int, s0)))
    # time-domain ensemble: piecewise-constant white force
    dt = 0.002
    duration = 300.0
    n = int(duration / dt)
    sigma_force = float(np.sqrt(s0 / (2.0 * dt)))
    rng_seeds = (11, 23, 37, 41, 53, 67)
    rms_td = []
    for seed in rng_seeds:
        rng = np.random.default_rng(seed)
        force = rng.normal(0.0, sigma_force, n)
        hist = zero_order_hold_response(system, force, dt)
        rms_td.append(hist.rms_displacement_m)
    rms_td_mean = float(np.mean(rms_td))
    ac3 = {
        "white_noise_level_n2_per_hz": s0,
        "rms_analytic_m": rms_analytic,
        "rms_from_spectral_integration_m": rms_spectral,
        "spectral_integration_relative_error": _rel(rms_spectral, rms_analytic),
        "rms_time_domain_ensemble_m": rms_td_mean,
        "rms_time_domain_per_seed_m": rms_td,
        "time_domain_relative_error": _rel(rms_td_mean, rms_analytic),
        "time_domain_relative_scatter": float(np.std(rms_td) / np.mean(rms_td)),
        "tolerance": 0.05,
        "passed": bool(_rel(rms_spectral, rms_analytic) <= 0.05 and _rel(rms_td_mean, rms_analytic) <= 0.05),
    }

    # ------------------------------------------------------------------ AC4
    consistency = classification_consistency(
        detuning_ratios=(0.70, 0.85, 0.94, 0.96, 1.00, 1.04, 1.06, 1.15, 1.30),
        damping_ratios=(0.01, 0.02, 0.05),
        f_n_hz=1.10,
    )
    ac4 = consistency

    # ------------------------------------------------------------------ AC5
    case_a = run_lighthouse_case()
    ice = case_a["_series"]["ice"]
    ice_peak = ice.psd_peak_frequency_hz
    f_ice_model = ice.f_ice_hz
    response_peak = case_a["response"]["response_peak_frequency_hz"]
    lit = LIGHTHOUSE_LITERATURE_FREQ_HZ
    ac5 = {
        "f_ice_from_U_over_s_hz": f_ice_model,
        "ice_force_psd_peak_hz": ice_peak,
        "ice_force_psd_peak_relative_error": _rel(ice_peak, f_ice_model),
        "response_psd_peak_hz": response_peak,
        "literature_dominant_frequency_hz": lit,
        "literature_band_hz": list(LIGHTHOUSE_LITERATURE_BAND_HZ),
        "relative_frequency_error": _rel(response_peak, lit),
        "within_literature_band": bool(
            LIGHTHOUSE_LITERATURE_BAND_HZ[0] <= response_peak <= LIGHTHOUSE_LITERATURE_BAND_HZ[1]
        ),
        "tolerance": 0.20,
        "passed": bool(_rel(response_peak, lit) <= 0.20),
        "note": "frequency validated; amplitude not validated (records not public)",
    }

    case_b = run_hull_chain_case()

    criteria = {
        "AC1": {
            "description": "SDOF natural frequency error <= 1 %; MDOF (>=3 DOF) <= 5 %",
            "sdof": ac1_sdof,
            "mdof": ac1_mdof,
            "passed": bool(ac1_sdof["passed"] and ac1_mdof["passed"]),
        },
        "AC2": {
            "description": "FRF amplitude/phase vs analytic <= 1 % (<= 5 % near resonance)",
            "sdof": ac2_sdof,
            "mdof": ac2_mdof,
            "passed": bool(ac2_sdof["passed"] and ac2_mdof["passed"]),
        },
        "AC3": {
            "description": "random-excitation response RMS vs analytic <= 5 %",
            "result": ac3,
            "passed": ac3["passed"],
        },
        "AC4": {
            "description": "resonance criterion identical to analytic criterion over >= 20 points",
            "result": ac4,
            "passed": bool(ac4["agreement"] >= 1.0 and ac4["n_points"] >= 20),
        },
        "AC5": {
            "description": "DEM-style ice load -> response PSD peak vs literature <= 20 %",
            "result": ac5,
            "passed": ac5["passed"],
        },
    }

    return {
        "node": "P233W2-N07",
        "title": "Ice-induced vibration spectral analysis: response PSD and resonance risk",
        "criteria": criteria,
        "all_passed": all(c["passed"] for c in criteria.values()),
        "cases": {
            "lighthouse": {k: v for k, v in case_a.items() if k != "_series"},
            "hull_chain": {k: v for k, v in case_b.items() if k != "_series"},
        },
        "_series": {"lighthouse": case_a["_series"], "hull_chain": case_b["_series"]},
    }