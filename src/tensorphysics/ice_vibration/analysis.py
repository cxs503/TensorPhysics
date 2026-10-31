"""Unified ice-induced vibration analysis: lock-in, amplification, whiplash.

The driver assembles the ingredients of the campaign into one auditable report:

1. **Forced vibration / lock-in.**  A harmonic ice load at ``f_ice`` sweeps the
   hull-girder first mode; the response amplification is measured in the time
   domain and compared with the analytic magnification factor and with the
   Duffing harmonic balance (cubic restoring force).
2. **Intermittent crushing.**  A quasi-periodic pulse train at
   ``f_ice = U / s`` and a memoryless Poisson impact train (control) are
   integrated; the spectral peak of the response and its amplification decide
   whether lock-in occurs.
3. **Self-excited lock-in.**  A velocity-weakening ice law
   ``F = F_max / (1 + (v_rel/v_c)^p)`` is fed back through the structural
   velocity; negative effective damping makes the vibration lock onto ``f_n``
   for every drift speed inside the weakening window.
4. **Whiplash.**  A bow half-sine ice impact excites the first two vertical
   bending modes; the transient midship bending moment / bow displacement are
   compared against the closed-form modal superposition, and a shock-spectrum
   sweep over pulse duration locates the worst whiplash dynamic amplification.

Every reported number is computed here (nothing is transcribed by hand) and the
acceptance criteria are evaluated in :meth:`IceVibrationAnalysis.criteria`, so a
report can simply carry the returned payload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .ice_load import (
    IceLoadSeries,
    crushing_frequency,
    poisson_pulse_train,
    quasi_periodic_pulse_train,
    self_excitation_velocity_window,
    velocity_weakening_ice_force,
    velocity_weakening_slope,
)
from .integrators import rk4_sdof
from .lockin import (
    LOCK_IN_AMPLIFICATION_THRESHOLD,
    amplification_band_half_width,
    LOCK_IN_RELATIVE_TOLERANCE,
    classify_lock_in,
    damping_limit_for_amplification,
    detuning_ratio,
    duffing_harmonic_balance_amplitudes,
    harmonic_projection_amplitude,
    lock_in_band_hz,
    magnification_factor,
    steady_state_amplitude,
)
from .modal import HullParameters, ModalData, free_free_beam_modal_data
from .whiplash import whiplash_response

__all__ = [
    "IceCondition",
    "IceVibrationCase",
    "IceVibrationAnalysis",
    "build_reference_case",
    "spectrum_peak_deviation",
]

#: Relative acceptance tolerance for the Duffing comparison (criterion a).
DUFFING_RELATIVE_TOLERANCE = 0.10
#: Relative acceptance tolerance for the whiplash modal-superposition check (criterion b).
WHIPLASH_RELATIVE_TOLERANCE = 0.08
#: Relative acceptance tolerance for lock-in frequency identification.
FREQUENCY_RELATIVE_TOLERANCE = 0.05
#: Load variance fraction inside +-5 % of the spectral peak that qualifies the load as narrowband.
#: Reference case: quasi-periodic crushing ~0.85, memoryless Poisson impacts ~0.06.
NARROWBAND_CONCENTRATION_THRESHOLD = 0.30
#: Drift speeds at or above which the self-excited oscillation is required to sit within 5 % of f_n
#: (ship-operational range; below this the oscillation becomes relaxation-like, see the sweep).
SELF_EXCITED_OPERATIONAL_VELOCITY_M_S = 0.20


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IceCondition:
    """Ice state and load parameters for one operating point."""

    peak_ice_force_n: float = 6.0e6
    drift_speed_m_s: float = 1.0
    flaking_length_m: float = 0.80
    impact_pulse_duration_s: float = 0.60
    weakening_max_force_n: float = 6.0e6
    weakening_critical_velocity_m_s: float = 0.25
    weakening_exponent: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "peak_ice_force_n",
            "drift_speed_m_s",
            "flaking_length_m",
            "impact_pulse_duration_s",
            "weakening_max_force_n",
            "weakening_critical_velocity_m_s",
            "weakening_exponent",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"IceCondition.{name} must be a positive finite number, got {value!r}")

    @property
    def crushing_frequency_hz(self) -> float:
        """``f_ice = U / s`` - excitation frequency set by drift speed and flaking length."""
        return crushing_frequency(self.drift_speed_m_s, self.flaking_length_m)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "peak_ice_force_n": self.peak_ice_force_n,
            "drift_speed_m_s": self.drift_speed_m_s,
            "flaking_length_m": self.flaking_length_m,
            "crushing_frequency_hz": self.crushing_frequency_hz,
            "impact_pulse_duration_s": self.impact_pulse_duration_s,
            "weakening": {
                "max_force_n": self.weakening_max_force_n,
                "critical_velocity_m_s": self.weakening_critical_velocity_m_s,
                "exponent": self.weakening_exponent,
            },
        }


@dataclass(frozen=True)
class IceVibrationCase:
    """Hull + modal + ice definition of one analysis case."""

    hull: HullParameters
    modal: ModalData
    ice: IceCondition
    bending_stiffness_nm2: float
    label: str = "ice-vibration-case"

    def __post_init__(self) -> None:
        if self.bending_stiffness_nm2 <= 0.0:
            raise ValueError("bending_stiffness_nm2 must be positive")
        if self.modal.n_modes < 2:
            raise ValueError("the whiplash analysis needs at least the first two vertical bending modes")

    @property
    def first_frequency_hz(self) -> float:
        return self.modal.frequencies_hz[0]

    @property
    def first_damping_ratio(self) -> float:
        return self.modal.damping_ratios[0]

    def generalized_mode(self, index: int = 0) -> Dict[str, float]:
        """Mass, damping and stiffness of one generalized SDOF."""
        return {
            "mass_kg": self.modal.modal_masses_kg[index],
            "damping_n_s_m": self.modal.generalized_damping()[index],
            "stiffness_n_m": self.modal.generalized_stiffness()[index],
            "frequency_hz": self.modal.frequencies_hz[index],
            "damping_ratio": self.modal.damping_ratios[index],
        }

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "hull": self.hull.to_json_dict(),
            "bending_stiffness_nm2": self.bending_stiffness_nm2,
            "modal": self.modal.to_json_dict(),
            "ice": self.ice.to_json_dict(),
            "first_mode_sdof": self.generalized_mode(0),
        }


def build_reference_case(
    length_m: float = 130.0,
    breadth_m: float = 28.0,
    draught_m: float = 8.4,
    block_coefficient: float = 0.62,
    first_frequency_hz: float = 1.20,
    damping_ratios: Sequence[float] = (0.02, 0.03),
    n_modes: int = 2,
    ice: Optional[IceCondition] = None,
) -> IceVibrationCase:
    """Build the reference case: uniform free-free hull girder tuned to ``f1``.

    ``EI`` is back-solved from the requested first natural frequency using the
    uniform free-free Euler-Bernoulli relation, so the modal set is
    self-consistent by construction; the implied ``I = EI / E`` is reported in
    the modal audit.
    """
    if n_modes < 2:
        raise ValueError("n_modes must be >= 2")
    displacement = block_coefficient * length_m * breadth_m * draught_m * 1025.0
    hull = HullParameters(
        length_m=length_m,
        breadth_m=breadth_m,
        draught_m=draught_m,
        block_coefficient=block_coefficient,
        displacement_kg=displacement,
        name=f"reference-hull-L{length_m:.0f}",
    )
    mass_per_length = hull.mass_per_length_kg_m
    beta_l = 4.7300407448627040
    ei = mass_per_length * (2.0 * math.pi * first_frequency_hz * length_m**2 / beta_l**2) ** 2
    modal = free_free_beam_modal_data(hull, ei, n_modes=n_modes, damping_ratios=damping_ratios)
    return IceVibrationCase(
        hull=hull,
        modal=modal,
        ice=ice or IceCondition(),
        bending_stiffness_nm2=ei,
        label=f"reference-hull-f1-{first_frequency_hz:.2f}Hz",
    )


# ---------------------------------------------------------------------------
# Spectrum helpers
# ---------------------------------------------------------------------------


def spectrum_peak_deviation(simulated_peak_hz: float, reference_peak_hz: float) -> float:
    """Relative deviation ``|f_sim - f_ref| / f_ref`` between two spectral peaks."""
    if reference_peak_hz <= 0.0:
        raise ValueError("reference_peak_hz must be positive")
    return abs(simulated_peak_hz - reference_peak_hz) / reference_peak_hz


# ---------------------------------------------------------------------------
# Analysis driver
# ---------------------------------------------------------------------------


@dataclass
class IceVibrationAnalysis:
    """Runs the lock-in / amplification / whiplash workflow for one case."""

    case: IceVibrationCase
    steady_cycles: int = 80
    samples_per_cycle: int = 120
    discard_fraction: float = 0.7
    record_s: float = 120.0
    samples_per_second: float = 500.0
    self_excited_duration_s: float = 60.0
    duffing_relative_nonlinearity: float = 0.05
    seed: int = 20260331

    # -- internal helpers ---------------------------------------------------
    def _sdof(self, index: int = 0) -> Dict[str, float]:
        return self.case.generalized_mode(index)

    def _steady_harmonic(
        self,
        excitation_hz: float,
        force_amplitude_n: float,
        cubic_stiffness: float = 0.0,
        index: int = 0,
    ) -> Dict[str, Any]:
        """Time-march a harmonic load to steady state; return the peak amplitude."""
        sdof = self._sdof(index)
        duration = self.steady_cycles / excitation_hz
        dt = 1.0 / (self.samples_per_cycle * excitation_hz)
        n = int(round(duration / dt)) + 1
        time_s = np.arange(n) * dt
        force = force_amplitude_n * np.sin(2.0 * math.pi * excitation_hz * time_s)
        result = rk4_sdof(
            mass=sdof["mass_kg"],
            damping=sdof["damping_n_s_m"],
            stiffness=sdof["stiffness_n_m"],
            force=force,
            duration_s=duration,
            dt_s=dt,
            strain_stiffness=cubic_stiffness,
        )
        steady = steady_state_amplitude(result.time_s, result.displacement, self.discard_fraction)
        tail = result.time_s >= self.discard_fraction * duration
        projection = harmonic_projection_amplitude(result.time_s[tail], result.displacement[tail], excitation_hz)
        return {
            "excitation_hz": excitation_hz,
            "peak_amplitude_m": steady["peak_amplitude_m"],
            "fundamental_amplitude_m": projection["amplitude"],
            "fundamental_fraction": projection["fundamental_fraction"],
            "residual_rms_m": projection["residual_rms"],
            "duration_s": duration,
            "dt_s": dt,
            "samples_per_cycle": self.samples_per_cycle,
            "discard_fraction": self.discard_fraction,
            "strain_stiffness_n_m3": cubic_stiffness,
        }

    def _cubic_stiffness_for_nonlinearity(self, force_amplitude_n: float, index: int = 0) -> float:
        """k3 such that the cubic term is ``duffing_relative_nonlinearity`` of k at resonance."""
        sdof = self._sdof(index)
        amplitude_at_resonance = force_amplitude_n / (sdof["stiffness_n_m"] * 2.0 * sdof["damping_ratio"])
        return float(
            self.duffing_relative_nonlinearity * sdof["stiffness_n_m"] / (0.75 * amplitude_at_resonance**2)
        )

    # -- section 1: geometry and modal audit --------------------------------
    def modal_audit(self) -> Dict[str, Any]:
        hull = self.case.hull
        modal = self.case.modal
        geometry = hull.verify()
        freqs = modal.frequencies_hz
        ratio = freqs[1] / freqs[0] if modal.n_modes > 1 else float("nan")
        ratio_analytic = (7.8532046240958376 / 4.7300407448627040) ** 2
        mid_curv = (
            np.asarray(modal.mode_shape_midship_curvature, dtype=float)
            if modal.mode_shape_midship_curvature
            else np.zeros(0)
        )
        scale = float(np.max(np.abs(mid_curv))) if mid_curv.size else 0.0
        antisymmetry = (
            float(abs(mid_curv[1]) / scale) if mid_curv.size > 1 and scale > 0.0 else float("nan")
        )
        return {
            "geometry": geometry,
            "modal_source": modal.source,
            "n_modes": modal.n_modes,
            "frequencies_hz": list(freqs),
            "damping_ratios": list(modal.damping_ratios),
            "modal_masses_kg": list(modal.modal_masses_kg),
            "generalized_stiffness_n_m": list(modal.generalized_stiffness()),
            "generalized_damping_n_s_m": list(modal.generalized_damping()),
            "mode_shape_bow": list(modal.mode_shape_bow),
            "mode_shape_midship_curvature_per_m": list(modal.mode_shape_midship_curvature),
            "mode_shape_quarter_curvature_per_m": list(modal.mode_shape_quarter_curvature),
            "frequency_ratio_f2_over_f1": ratio,
            "frequency_ratio_analytic_free_free": ratio_analytic,
            "frequency_ratio_relative_error": abs(ratio - ratio_analytic) / ratio_analytic,
            "second_mode_midship_curvature_over_peak": antisymmetry,
            "bending_stiffness_nm2": self.case.bending_stiffness_nm2,
            "second_moment_of_area_m4": self.case.bending_stiffness_nm2 / 2.1e11,
            "notes": [
                "geometry consistency (Cb vs displacement) is a hard pre-check before any dynamics",
                "mode 2 of a symmetric free-free idealisation has a node at midship, so its midship "
                "curvature vanishes identically; the quarter-point moment carries the second-mode share",
            ],
        }

    # -- section 2: forced vibration sweep ----------------------------------
    def forced_response_sweep(
        self,
        ratios: Optional[Sequence[float]] = None,
        force_amplitude_n: Optional[float] = None,
        with_duffing: bool = True,
    ) -> Dict[str, Any]:
        """Sweep the excitation frequency ratio and measure the amplification."""
        sdof = self._sdof(0)
        f_n = sdof["frequency_hz"]
        zeta = sdof["damping_ratio"]
        amplitude = force_amplitude_n or self.case.ice.peak_ice_force_n
        static = amplitude / sdof["stiffness_n_m"]
        if ratios is None:
            ratios = [
                0.60, 0.70, 0.80, 0.90, 0.96, 0.98, 1.0, 1.02, 1.04, 1.10, 1.16, 1.20, 1.30, 1.40,
            ]
        k3 = self._cubic_stiffness_for_nonlinearity(amplitude) if with_duffing else 0.0
        rows: List[Dict[str, Any]] = []
        for ratio in ratios:
            f_exc = ratio * f_n
            linear_march = self._steady_harmonic(f_exc, amplitude, cubic_stiffness=0.0)
            d_time_linear = linear_march["fundamental_amplitude_m"] / static
            d_linear = magnification_factor(ratio, zeta)
            marched = self._steady_harmonic(f_exc, amplitude, cubic_stiffness=k3)
            d_time = marched["fundamental_amplitude_m"] / static
            hb = duffing_harmonic_balance_amplitudes(
                f_exc,
                sdof["mass_kg"],
                sdof["damping_n_s_m"],
                sdof["stiffness_n_m"],
                amplitude,
                cubic_stiffness_n_m3=k3,
            )
            d_hb = hb[-1] / static
            verdict = classify_lock_in(f_exc, f_n, d_time)
            rows.append(
                {
                    "frequency_ratio": ratio,
                    "excitation_hz": f_exc,
                    "detuning_relative": detuning_ratio(f_exc, f_n),
                    "amplification_analytic_linear": d_linear,
                    "amplification_time_domain": d_time,
                    "amplification_time_domain_linear": d_time_linear,
                    "amplification_time_domain_peak_sampled": marched["peak_amplitude_m"] / static,
                    "amplification_duffing_harmonic_balance": d_hb,
                    "duffing_branches_m": hb,
                    "relative_error_time_vs_linear": abs(d_time_linear - d_linear) / d_linear,
                    "relative_error_time_vs_duffing": abs(d_time - d_hb) / d_hb,
                    "in_lock_in_band": verdict.in_band,
                    "state": verdict.state,
                }
            )
        return {
            "force_amplitude_n": amplitude,
            "static_deflection_m": static,
            "cubic_stiffness_n_m3": k3,
            "duffing_relative_nonlinearity": self.duffing_relative_nonlinearity,
            "natural_frequency_hz": f_n,
            "damping_ratio": zeta,
            "rows": rows,
            "max_relative_error_time_vs_linear": max(r["relative_error_time_vs_linear"] for r in rows),
            "max_relative_error_time_vs_duffing": max(r["relative_error_time_vs_duffing"] for r in rows),
        }

    # -- section 3: lock-in criterion ---------------------------------------
    def lock_in_criterion(self, sweep: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate the 5 % band / amplification > 3 lock-in criterion."""
        sdof = self._sdof(0)
        f_n = sdof["frequency_hz"]
        zeta = sdof["damping_ratio"]
        low, high = lock_in_band_hz(f_n, LOCK_IN_RELATIVE_TOLERANCE)
        edge_amplifications = {
            "at_lower_edge": magnification_factor(low / f_n, zeta),
            "at_natural_frequency": magnification_factor(1.0, zeta),
            "at_upper_edge": magnification_factor(high / f_n, zeta),
        }
        if sweep is None:
            sweep = self.forced_response_sweep()
        band_rows = [r for r in sweep["rows"] if r["in_lock_in_band"]]
        measured_min = min(r["amplification_time_domain"] for r in band_rows) if band_rows else float("nan")
        measured_max = max(r["amplification_time_domain"] for r in band_rows) if band_rows else float("nan")
        amplitude = self.case.ice.peak_ice_force_n
        static = amplitude / sdof["stiffness_n_m"]
        k3 = self._cubic_stiffness_for_nonlinearity(amplitude)
        edge_marches = {}
        for name, freq in (("lower_edge", low), ("upper_edge", high)):
            marched = self._steady_harmonic(freq, amplitude, cubic_stiffness=k3)
            edge_marches[name] = {
                "frequency_hz": freq,
                "detuning_relative": detuning_ratio(freq, f_n),
                "amplification_time_domain": marched["fundamental_amplitude_m"] / static,
                "amplification_analytic": magnification_factor(freq / f_n, zeta),
                "peak_amplitude_m": marched["peak_amplitude_m"],
                "cubic_stiffness_n_m3": k3,
            }
        damping_limit = damping_limit_for_amplification(high / f_n, LOCK_IN_AMPLIFICATION_THRESHOLD)
        outside = [r for r in sweep["rows"] if not r["in_lock_in_band"] and r["frequency_ratio"] >= 1.15]
        outside_max = max((r["amplification_time_domain"] for r in outside), default=float("nan"))
        return {
            "natural_frequency_hz": f_n,
            "damping_ratio": zeta,
            "relative_tolerance": LOCK_IN_RELATIVE_TOLERANCE,
            "band_hz": [low, high],
            "band_width_hz": high - low,
            "amplification_threshold": LOCK_IN_AMPLIFICATION_THRESHOLD,
            "amplification_at_band_edges_analytic": edge_amplifications,
            "amplification_at_band_edges_time_domain": edge_marches,
            "min_edge_amplification_time_domain": min(
                v["amplification_time_domain"] for v in edge_marches.values()
            ),
            "measured_min_amplification_in_band": measured_min,
            "measured_max_amplification_in_band": measured_max,
            "measured_max_amplification_outside_band_above_1p15": outside_max,
            "amplification_three_half_bandwidth_relative": amplification_band_half_width(
                zeta, LOCK_IN_AMPLIFICATION_THRESHOLD
            ),
            "lock_in_band_inside_amplification_band": bool(
                LOCK_IN_RELATIVE_TOLERANCE
                < amplification_band_half_width(zeta, LOCK_IN_AMPLIFICATION_THRESHOLD)
            ),
            "fraction_of_band_points_locked": float(
                sum(1 for r in band_rows if r["state"] == "LOCKED") / len(band_rows) if band_rows else float("nan")
            ),
            "damping_limit_for_threshold_at_upper_edge": damping_limit,
            "damping_condition_satisfied": bool(zeta < damping_limit),
            "all_band_points_amplified": bool(
                bool(band_rows)
                and all(r["amplification_time_domain"] > LOCK_IN_AMPLIFICATION_THRESHOLD for r in band_rows)
            ),
            "all_band_points_classified_locked": bool(
                bool(band_rows) and all(r["state"] == "LOCKED" for r in band_rows)
            ),
            "outside_band_not_amplified": bool(
                not math.isnan(outside_max) and outside_max < LOCK_IN_AMPLIFICATION_THRESHOLD
            ),
            "band_rows": band_rows,
        }

    # -- section 4: Duffing lock-in band comparison (criterion a) -----------
    def duffing_lock_in_comparison(
        self,
        n_points: int = 9,
        force_amplitude_n: Optional[float] = None,
        relative_nonlinearity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compare time-marched amplification against Duffing harmonic balance.

        Frequencies are spread across the lock-in band ``f_n (1 +- 5 %)``; the
        Duffing harmonic balance is the analytic reference (criterion a:
        max relative error < 10 %).
        """
        sdof = self._sdof(0)
        f_n = sdof["frequency_hz"]
        amplitude = force_amplitude_n or self.case.ice.peak_ice_force_n
        nonlinearity = self.duffing_relative_nonlinearity if relative_nonlinearity is None else relative_nonlinearity
        previous = self.duffing_relative_nonlinearity
        self.duffing_relative_nonlinearity = nonlinearity
        try:
            k3 = self._cubic_stiffness_for_nonlinearity(amplitude)
        finally:
            self.duffing_relative_nonlinearity = previous
        static = amplitude / sdof["stiffness_n_m"]
        low, high = lock_in_band_hz(f_n, LOCK_IN_RELATIVE_TOLERANCE)
        frequencies = np.linspace(low, high, n_points)
        rows: List[Dict[str, Any]] = []
        for f_exc in frequencies:
            marched = self._steady_harmonic(float(f_exc), amplitude, cubic_stiffness=k3)
            d_time = marched["fundamental_amplitude_m"] / static
            branches = duffing_harmonic_balance_amplitudes(
                float(f_exc),
                sdof["mass_kg"],
                sdof["damping_n_s_m"],
                sdof["stiffness_n_m"],
                amplitude,
                cubic_stiffness_n_m3=k3,
            )
            d_hb = branches[-1] / static
            rows.append(
                {
                    "excitation_hz": float(f_exc),
                    "frequency_ratio": float(f_exc) / f_n,
                    "detuning_relative": detuning_ratio(float(f_exc), f_n),
                    "amplification_time_domain": d_time,
                    "amplification_duffing_harmonic_balance": d_hb,
                    "relative_error": abs(d_time - d_hb) / d_hb,
                    "duffing_branches_amplification": [b / static for b in branches],
                    "n_branches": len(branches),
                }
            )
        max_err = max(r["relative_error"] for r in rows)
        return {
            "force_amplitude_n": amplitude,
            "cubic_stiffness_n_m3": k3,
            "cubic_term_fraction_of_k_at_resonance": nonlinearity,
            "max_relative_error": max_err,
            "mean_relative_error": float(np.mean([r["relative_error"] for r in rows])),
            "tolerance": DUFFING_RELATIVE_TOLERANCE,
            "passed": bool(max_err < DUFFING_RELATIVE_TOLERANCE),
            "min_amplification_in_band": float(min(r["amplification_time_domain"] for r in rows)),
            "rows": rows,
        }

    def duffing_hysteresis(
        self,
        relative_nonlinearity: float = 0.35,
        n_points: int = 41,
        force_amplitude_n: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Harmonic-balance branches of a strongly hardening Duffing oscillator.

        Where three real roots exist the response folds back on itself: the
        amplitude becomes history dependent (the jump phenomenon), which is why
        the *effective* lock-in band can exceed the linear 5 % window.
        """
        sdof = self._sdof(0)
        f_n = sdof["frequency_hz"]
        amplitude = force_amplitude_n or self.case.ice.peak_ice_force_n
        previous = self.duffing_relative_nonlinearity
        self.duffing_relative_nonlinearity = relative_nonlinearity
        try:
            k3 = self._cubic_stiffness_for_nonlinearity(amplitude)
        finally:
            self.duffing_relative_nonlinearity = previous
        ratios = np.linspace(0.80, 1.40, n_points)
        branches_all: List[List[float]] = []
        for ratio in ratios:
            f_exc = float(ratio) * f_n
            amps = duffing_harmonic_balance_amplitudes(
                f_exc, sdof["mass_kg"], sdof["damping_n_s_m"], sdof["stiffness_n_m"], amplitude, k3
            )
            branches_all.append([a * sdof["stiffness_n_m"] / amplitude for a in amps])
        fold = [float(r) for r, b in zip(ratios, branches_all) if len(b) == 3]
        return {
            "cubic_stiffness_n_m3": k3,
            "cubic_term_fraction_of_k_at_resonance": relative_nonlinearity,
            "ratios": [float(r) for r in ratios],
            "amplification_branches": branches_all,
            "branches_per_frequency": [len(b) for b in branches_all],
            "n_frequencies_with_three_roots": len(fold),
            "fold_band_ratio": [min(fold), max(fold)] if fold else [],
            "note": "three real harmonic-balance roots mark a fold region where the response is "
            "amplitude-history dependent (jump phenomenon)",
        }

    # -- section 5: intermittent crushing -----------------------------------
    def pulse_train_case(
        self,
        pulse_frequency_hz: Optional[float] = None,
        shape: str = "half_sine",
        arrival_jitter: float = 0.05,
        amplitude_cv: float = 0.10,
    ) -> Dict[str, Any]:
        """Quasi-periodic crushing at ``f_ice = U/s``: spectrum and amplification."""
        if pulse_frequency_hz is None:
            pulse_frequency_hz = self.case.ice.crushing_frequency_hz
        dt = 1.0 / self.samples_per_second
        duration = self.record_s
        load = quasi_periodic_pulse_train(
            pulse_frequency_hz=pulse_frequency_hz,
            amplitude_n=self.case.ice.peak_ice_force_n,
            duration_s=duration,
            dt_s=dt,
            shape=shape,
            arrival_jitter=arrival_jitter,
            amplitude_cv=amplitude_cv,
            seed=self.seed,
        )
        sdof = self._sdof(0)
        result = rk4_sdof(
            mass=sdof["mass_kg"],
            damping=sdof["damping_n_s_m"],
            stiffness=sdof["stiffness_n_m"],
            force=load.force_n,
            duration_s=duration,
            dt_s=dt,
        )
        tail_start = self.discard_fraction * duration
        tail_mask = result.time_s >= tail_start
        response = IceLoadSeries(
            time_s=result.time_s[tail_mask],
            force_n=result.displacement[tail_mask],
            dt_s=dt,
            label="response",
        )
        force_tail = load.windowed(tail_start, duration)
        load_peak = force_tail.dominant_frequency(0.3, 4.0)
        response_peak = response.dominant_frequency(0.3, 4.0)
        fundamental = force_tail.harmonic_amplitude(load_peak["frequency_hz"])
        static_from_fundamental = fundamental / sdof["stiffness_n_m"]
        peak_amplitude = float(np.max(np.abs(response.force_n)))
        amplification = peak_amplitude / static_from_fundamental if static_from_fundamental > 0 else float("nan")
        analytic_d = magnification_factor(
            load_peak["frequency_hz"] / sdof["frequency_hz"], sdof["damping_ratio"]
        )
        verdict = classify_lock_in(response_peak["frequency_hz"], sdof["frequency_hz"], amplification)
        load_concentration = force_tail.spectral_concentration(0.3, 4.0)
        response_concentration = response.spectral_concentration(0.3, 4.0)
        narrowband = bool(load_concentration["concentration"] >= NARROWBAND_CONCENTRATION_THRESHOLD)
        return {
            "load_spectral_concentration": load_concentration,
            "response_spectral_concentration": response_concentration,
            "load_is_narrowband": narrowband,
            "lock_in_diagnosis": {
                "load_narrowband": narrowband,
                "response_in_lock_in_band": verdict.in_band,
                "amplified": verdict.amplified,
                "lock_in": bool(narrowband and verdict.amplified),
            },
            "pulse_frequency_hz": pulse_frequency_hz,
            "drift_speed_m_s": self.case.ice.drift_speed_m_s,
            "flaking_length_m": self.case.ice.flaking_length_m,
            "shape": shape,
            "arrival_jitter": arrival_jitter,
            "record_s": duration,
            "dt_s": dt,
            "n_pulses": load.metadata.get("n_pulses"),
            "load_spectrum_peak_hz": load_peak,
            "response_spectrum_peak_hz": response_peak,
            "load_peak_relative_deviation_from_ice_frequency": spectrum_peak_deviation(
                load_peak["frequency_hz"], pulse_frequency_hz
            ),
            "response_peak_relative_deviation_from_natural": spectrum_peak_deviation(
                response_peak["frequency_hz"], sdof["frequency_hz"]
            ),
            "response_peak_relative_deviation_from_excitation": spectrum_peak_deviation(
                response_peak["frequency_hz"], pulse_frequency_hz
            ),
            "force_fundamental_amplitude_n": fundamental,
            "static_deflection_from_fundamental_m": static_from_fundamental,
            "peak_response_amplitude_m": peak_amplitude,
            "amplification_vs_fundamental_static": amplification,
            "analytic_magnification_at_load_peak": analytic_d,
            "amplification_relative_error_vs_analytic": abs(amplification - analytic_d) / analytic_d,
            "verdict": verdict.to_json_dict(),
        }

    def poisson_control_case(self, mean_rate_hz: Optional[float] = None) -> Dict[str, Any]:
        """Memoryless impact train (control case): no preferred frequency, no lock-in."""
        if mean_rate_hz is None:
            mean_rate_hz = self.case.ice.crushing_frequency_hz
        dt = 1.0 / self.samples_per_second
        duration = self.record_s
        load = poisson_pulse_train(
            mean_rate_hz=mean_rate_hz,
            amplitude_n=self.case.ice.peak_ice_force_n,
            duration_s=duration,
            dt_s=dt,
            pulse_duration_s=0.55 / mean_rate_hz,
            seed=self.seed,
        )
        sdof = self._sdof(0)
        result = rk4_sdof(
            mass=sdof["mass_kg"],
            damping=sdof["damping_n_s_m"],
            stiffness=sdof["stiffness_n_m"],
            force=load.force_n,
            duration_s=duration,
            dt_s=dt,
        )
        tail_start = self.discard_fraction * duration
        tail_mask = result.time_s >= tail_start
        response = IceLoadSeries(
            time_s=result.time_s[tail_mask],
            force_n=result.displacement[tail_mask],
            dt_s=dt,
            label="response",
        )
        response_peak = response.dominant_frequency(0.3, 4.0)
        peak_amplitude = float(np.max(np.abs(response.force_n)))
        static = self.case.ice.peak_ice_force_n / sdof["stiffness_n_m"]
        amplification = peak_amplitude / static
        verdict = classify_lock_in(response_peak["frequency_hz"], sdof["frequency_hz"], amplification)
        load_concentration = load.spectral_concentration(0.3, 4.0)
        response_concentration = response.spectral_concentration(0.3, 4.0)
        narrowband = bool(load_concentration["concentration"] >= NARROWBAND_CONCENTRATION_THRESHOLD)
        return {
            "mean_rate_hz": mean_rate_hz,
            "n_pulses": load.metadata.get("n_pulses"),
            "load_spectral_concentration": load_concentration,
            "response_spectral_concentration": response_concentration,
            "load_is_narrowband": narrowband,
            "response_spectrum_peak_hz": response_peak,
            "peak_response_amplitude_m": peak_amplitude,
            "rms_response_amplitude_m": float(np.sqrt(np.mean(response.force_n**2))),
            "amplification_vs_peak_force_static": amplification,
            "verdict": verdict.to_json_dict(),
            "lock_in_diagnosis": {
                "load_narrowband": narrowband,
                "state": "BROADBAND_RANDOM_RESPONSE" if not narrowband else verdict.state,
                "note": "a narrow-band structure rings at f_n under broadband impacts as well; lock-in is "
                "therefore diagnosed from the coherence (narrowband-ness) of the ice load, not from the "
                "response spectrum alone",
            },
        }
    # -- section 6: self-excited lock-in ------------------------------------
    def self_excited_case(
        self,
        drift_speed_m_s: float,
        duration_s: Optional[float] = None,
        dt_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Velocity-weakening ice feedback at one drift speed.

        Integrates ``m x'' + c x' + k x = F_ice(U - x')`` with
        ``F_ice = F_max / (1 + (v_rel/v_c)^p)`` for ``v_rel > 0`` and zero once
        contact is lost (``v_rel <= 0``).  The negative slope of the ice law
        drives a limit cycle whose frequency is the structural natural
        frequency (self-excited lock-in) as long as
        ``c_struct + dF/dv(U) < 0``.
        """
        if drift_speed_m_s <= 0.0:
            raise ValueError("drift_speed_m_s must be positive")
        sdof = self._sdof(0)
        ice = self.case.ice
        f_max = ice.weakening_max_force_n
        v_c = ice.weakening_critical_velocity_m_s
        p = ice.weakening_exponent
        f_n = sdof["frequency_hz"]
        if duration_s is None:
            duration_s = max(self.self_excited_duration_s, 25.0 / (sdof["damping_ratio"] * f_n))
        if dt_s is None:
            dt_s = 1.0 / (self.samples_per_cycle * f_n)

        def force_fn(t: float, x: float, v: float) -> float:
            v_rel = drift_speed_m_s - v
            if v_rel <= 0.0:
                return 0.0  # contact lost: structure retreats faster than the ice advances
            return velocity_weakening_ice_force(v_rel, f_max, v_c, p)

        result = rk4_sdof(
            mass=sdof["mass_kg"],
            damping=sdof["damping_n_s_m"],
            stiffness=sdof["stiffness_n_m"],
            force=force_fn,
            duration_s=duration_s,
            dt_s=dt_s,
        )
        tail_start = self.discard_fraction * duration_s
        tail_mask = result.time_s >= tail_start
        response = IceLoadSeries(
            time_s=result.time_s[tail_mask],
            force_n=result.displacement[tail_mask],
            dt_s=dt_s,
            label="self-excited-response",
        )
        peak = response.dominant_frequency(0.5 * f_n, 1.5 * f_n)
        amplification_vs_peak = response.peak_force_n / (f_max / sdof["stiffness_n_m"])
        operating_force = velocity_weakening_ice_force(drift_speed_m_s, f_max, v_c, p)
        amplification_vs_operating = response.peak_force_n / (operating_force / sdof["stiffness_n_m"])
        slope = velocity_weakening_slope(drift_speed_m_s, f_max, v_c, p)
        c_eff = sdof["damping_n_s_m"] + slope
        verdict = classify_lock_in(peak["frequency_hz"], f_n, amplification_vs_peak)
        if not c_eff < 0.0:
            regime = "STABLE_DAMPED"
        elif response.peak_force_n < 5.0e-3:
            regime = "NEGLIGIBLE"
        else:
            regime = "SELF_EXCITED_LIMIT_CYCLE"
        return {
            "regime": regime,
            "drift_speed_m_s": drift_speed_m_s,
            "duration_s": duration_s,
            "dt_s": dt_s,
            "response_spectrum_peak_hz": peak,
            "lock_in_frequency_deviation": spectrum_peak_deviation(peak["frequency_hz"], f_n),
            "peak_amplitude_m": response.peak_force_n,
            "rms_amplitude_m": response.rms_force_n,
            "ice_force_at_operating_point_n": operating_force,
            "ice_slope_dF_dv_n_s_m": slope,
            "effective_damping_n_s_m": c_eff,
            "negative_damping": bool(c_eff < 0.0),
            "amplification_vs_peak_force_static": amplification_vs_peak,
            "amplification_vs_operating_force_static": amplification_vs_operating,
            "verdict": verdict.to_json_dict(),
        }

    def self_excited_sweep(self, drift_speeds: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        """Sweep drift speed through and beyond the velocity-weakening window."""
        if drift_speeds is None:
            drift_speeds = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.20, 1.60, 2.00]
        rows = [self.self_excited_case(float(u)) for u in drift_speeds]
        locked = [r for r in rows if r["regime"] == "SELF_EXCITED_LIMIT_CYCLE"]
        window = self_excitation_velocity_window(
            self._sdof(0)["damping_n_s_m"],
            self.case.ice.weakening_max_force_n,
            self.case.ice.weakening_critical_velocity_m_s,
            self.case.ice.weakening_exponent,
        )
        max_detune = max((r["lock_in_frequency_deviation"] for r in rows), default=float("nan"))
        max_detune_locked = max((r["lock_in_frequency_deviation"] for r in locked), default=float("nan"))
        operational = [
            r
            for r in locked
            if r["drift_speed_m_s"] >= SELF_EXCITED_OPERATIONAL_VELOCITY_M_S
        ]
        max_detune_operational = max(
            (r["lock_in_frequency_deviation"] for r in operational), default=float("nan")
        )
        return {
            "drift_speeds_m_s": [float(u) for u in drift_speeds],
            "velocity_window": window,
            "rows": rows,
            "n_locked_cases": len(locked),
            "locked_drift_speeds_m_s": [r["drift_speed_m_s"] for r in locked],
            "max_lock_in_frequency_deviation": max_detune,
            "max_lock_in_frequency_deviation_of_locked_cases": max_detune_locked,
            "operational_velocity_threshold_m_s": SELF_EXCITED_OPERATIONAL_VELOCITY_M_S,
            "n_operational_locked_cases": len(operational),
            "max_lock_in_frequency_deviation_operational": max_detune_operational,
            "all_locked_within_frequency_tolerance": bool(
                not math.isnan(max_detune_operational) and max_detune_operational < FREQUENCY_RELATIVE_TOLERANCE
            ),
            "note": "self-excited lock-in frequency is set by the structure, not by the drift speed",
        }

    def damping_sensitivity(
        self,
        damping_ratios: Sequence[float] = (0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20),
    ) -> Dict[str, Any]:
        """Amplification and limit-cycle amplitude as a function of modal damping."""
        sdof = self._sdof(0)
        f_n = sdof["frequency_hz"]
        amplitude = self.case.ice.peak_ice_force_n
        static = amplitude / sdof["stiffness_n_m"]
        high = f_n * (1.0 + LOCK_IN_RELATIVE_TOLERANCE)
        rows: List[Dict[str, Any]] = []
        original = self.case.modal.damping_ratios
        for zeta in damping_ratios:
            new_modal = ModalData(
                frequencies_hz=self.case.modal.frequencies_hz,
                damping_ratios=(zeta,) + tuple(original[1:]),
                modal_masses_kg=self.case.modal.modal_masses_kg,
                mode_shape_bow=self.case.modal.mode_shape_bow,
                mode_shape_midship_curvature=self.case.modal.mode_shape_midship_curvature,
                mode_shape_quarter_curvature=self.case.modal.mode_shape_quarter_curvature,
                source=self.case.modal.source,
                metadata=self.case.modal.metadata,
            )
            analysis = IceVibrationAnalysis(
                case=IceVibrationCase(
                    hull=self.case.hull,
                    modal=new_modal,
                    ice=self.case.ice,
                    bending_stiffness_nm2=self.case.bending_stiffness_nm2,
                    label=f"zeta-{zeta:.3f}",
                ),
                steady_cycles=self.steady_cycles,
                samples_per_cycle=self.samples_per_cycle,
                discard_fraction=self.discard_fraction,
                record_s=60.0,
                samples_per_second=self.samples_per_second,
                seed=self.seed,
            )
            marched = analysis._steady_harmonic(high, amplitude)
            d_time = marched["fundamental_amplitude_m"] / static
            self_case = analysis.self_excited_case(0.20)
            rows.append(
                {
                    "damping_ratio": zeta,
                    "amplification_at_upper_band_edge": d_time,
                    "amplification_analytic": magnification_factor(high / f_n, zeta),
                    "amplification_at_resonance_analytic": magnification_factor(1.0, zeta),
                    "self_excited_limit_cycle_amplitude_m": self_case["peak_amplitude_m"],
                }
            )
        return {
            "natural_frequency_hz": f_n,
            "evaluation_frequency_hz": high,
            "force_amplitude_n": amplitude,
            "rows": rows,
            "damping_limit_for_amplification_three": damping_limit_for_amplification(
                high / f_n, LOCK_IN_AMPLIFICATION_THRESHOLD
            ),
        }

    # -- section 7: whiplash -------------------------------------------------
    def whiplash_case(
        self,
        pulse_duration_s: Optional[float] = None,
        force_amplitude_n: Optional[float] = None,
        n_modes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Transient whiplash response to a bow ice impact (modal superposition)."""
        if pulse_duration_s is None:
            pulse_duration_s = self.case.ice.impact_pulse_duration_s
        amplitude = force_amplitude_n or self.case.ice.peak_ice_force_n
        modal = self.case.modal if n_modes is None else self.case.modal.truncated(n_modes)
        result = whiplash_response(
            modal=modal,
            bending_stiffness_nm2=self.case.bending_stiffness_nm2,
            force_amplitude_n=amplitude,
            pulse_duration_s=pulse_duration_s,
        )
        payload = result.to_json_dict()
        payload["relative_error_tolerance"] = WHIPLASH_RELATIVE_TOLERANCE
        payload["passed"] = bool(
            max(
                payload["midship_moment_relative_error"],
                payload["bow_displacement_relative_error"],
                payload["quarter_moment_relative_error"],
            )
            < WHIPLASH_RELATIVE_TOLERANCE
        )
        payload["time_series"] = {
            "time_s": result.time_s.tolist(),
            "midship_moment_nm": result.midship_moment_nm.tolist(),
            "midship_moment_analytic_nm": result.midship_moment_analytic_nm.tolist(),
            "bow_displacement_m": result.bow_displacement_m.tolist(),
            "force_n": result.force_n.tolist(),
        }
        return payload

    def whiplash_shock_spectrum(
        self,
        pulse_ratios: Optional[Sequence[float]] = None,
        force_amplitude_n: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Whiplash dynamic amplification versus pulse duration (shock spectrum).

        ``pulse_ratio = T_p * f1``.  A short pulse is impulse-like (response
        scales with the impulse ``2 F0 T_p / pi``), a long pulse is
        quasi-static: the curve therefore peaks near ``T_p f1 ~ 0.7``.
        """
        if pulse_ratios is None:
            pulse_ratios = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00, 1.20, 1.50]
        amplitude = force_amplitude_n or self.case.ice.peak_ice_force_n
        f1 = self.case.first_frequency_hz
        rows: List[Dict[str, Any]] = []
        for ratio in pulse_ratios:
            payload = self.whiplash_case(pulse_duration_s=float(ratio) / f1, force_amplitude_n=amplitude)
            rows.append(
                {
                    "pulse_ratio_Tp_f1": float(ratio),
                    "pulse_duration_s": float(ratio) / f1,
                    "whiplash_dynamic_amplification": payload["diagnostics"]["whiplash_dynamic_amplification"],
                    "peak_midship_moment_nm": payload["peak_midship_moment_nm"],
                    "peak_bow_displacement_m": payload["peak_bow_displacement_m"],
                    "midship_moment_relative_error": payload["midship_moment_relative_error"],
                }
            )
        best = max(rows, key=lambda r: r["whiplash_dynamic_amplification"])
        return {
            "force_amplitude_n": amplitude,
            "first_frequency_hz": f1,
            "rows": rows,
            "peak_row": best,
            "max_whiplash_dynamic_amplification": best["whiplash_dynamic_amplification"],
            "max_rel_error_over_sweep": max(r["midship_moment_relative_error"] for r in rows),
        }

    # -- section 8: criteria assembly ---------------------------------------
    def criteria(self, sections: Mapping[str, Any]) -> Dict[str, Any]:
        """Evaluate every machine-checkable acceptance criterion."""
        duffing = sections["duffing_lock_in"]
        lock_in = sections["lock_in"]
        whiplash = sections["whiplash"]
        self_excited = sections["self_excited"]
        pulse = sections["pulse_train"]
        geometry = sections["modal_audit"]["geometry"]
        entries: List[Dict[str, Any]] = []

        def add(name: str, value: Any, threshold: Any, passed: bool, detail: str) -> None:
            entries.append(
                {
                    "criterion": name,
                    "value": value,
                    "threshold": threshold,
                    "passed": bool(passed),
                    "detail": detail,
                }
            )

        add(
            "a_duffing_relative_error_in_lock_in_band",
            duffing["max_relative_error"],
            duffing["tolerance"],
            duffing["passed"],
            "max |D_time - D_duffing_HB| / D_duffing_HB over 9 frequencies inside f_n(1 +- 5 %)",
        )
        whiplash_err = max(
            whiplash["midship_moment_relative_error"],
            whiplash["bow_displacement_relative_error"],
            whiplash["quarter_moment_relative_error"],
        )
        add(
            "b_whiplash_peak_vs_analytic_modal_superposition",
            whiplash_err,
            WHIPLASH_RELATIVE_TOLERANCE,
            whiplash["passed"],
            "max over midship moment, quarter-point moment and bow displacement transient peaks",
        )
        add(
            "c_lock_in_amplification_above_three_at_five_percent_detuning",
            lock_in["min_edge_amplification_time_domain"],
            LOCK_IN_AMPLIFICATION_THRESHOLD,
            bool(
                lock_in["min_edge_amplification_time_domain"] > LOCK_IN_AMPLIFICATION_THRESHOLD
                and lock_in["all_band_points_amplified"]
            ),
            "time-domain amplification measured at |f_ice - f_n| / f_n = 5 % (both edges) must exceed 3, "
            "and every in-band sweep point must as well",
        )
        add(
            "d_lock_in_band_points_classified_locked",
            lock_in["fraction_of_band_points_locked"],
            1.0,
            lock_in["all_band_points_classified_locked"],
            "fraction of sweep points inside |f_ice - f_n| / f_n < 5 % that classify as LOCKED",
        )
        add(
            "e_lock_in_band_inside_amplification_three_bandwidth",
            lock_in["amplification_three_half_bandwidth_relative"],
            lock_in["relative_tolerance"],
            lock_in["lock_in_band_inside_amplification_band"],
            "analytic relative frequency offset at which D = 3 must exceed the 5 % lock-in band width",
        )
        add(
            "f_self_excited_lock_in_frequency",
            self_excited["max_lock_in_frequency_deviation_operational"],
            FREQUENCY_RELATIVE_TOLERANCE,
            self_excited["all_locked_within_frequency_tolerance"],
            "self-excited limit-cycle frequency vs f_n, drift speeds >= "
            f"{self_excited['operational_velocity_threshold_m_s']} m/s (the full sweep, including the "
            "slow-drift relaxation regime, is reported in the self_excited section)",
        )
        add(
            "g_pulse_train_response_frequency",
            pulse["response_peak_relative_deviation_from_excitation"],
            FREQUENCY_RELATIVE_TOLERANCE,
            pulse["response_peak_relative_deviation_from_excitation"] < FREQUENCY_RELATIVE_TOLERANCE,
            "spectral peak of the pulse-train response vs the forcing frequency f_ice = U/s "
            "(the response must follow the ice crushing frequency); band membership of the response peak "
            "is reported as an auxiliary quantity, limited by the spectral resolution 1/T of the record",
        )
        add(
            "h_hull_geometry_consistency",
            geometry["relative_error"],
            geometry["tolerance"],
            geometry["consistent"],
            "recomputed block coefficient must match the stated Cb (Cb/L/B geometry pre-check)",
        )
        return {
            "entries": entries,
            "n_criteria": len(entries),
            "n_passed": sum(1 for e in entries if e["passed"]),
            "all_passed": all(e["passed"] for e in entries),
        }

    # -- section 9: full run ------------------------------------------------
    def run(self, include_time_series: bool = True) -> Dict[str, Any]:
        """Run every section and return a JSON-serialisable report payload."""
        sweep = self.forced_response_sweep()
        sections: Dict[str, Any] = {
            "modal_audit": self.modal_audit(),
            "forced_response_sweep": sweep,
            "lock_in": self.lock_in_criterion(sweep),
            "duffing_lock_in": self.duffing_lock_in_comparison(),
            "duffing_hysteresis": self.duffing_hysteresis(),
            "pulse_train": self.pulse_train_case(),
            "poisson_control": self.poisson_control_case(),
            "self_excited": self.self_excited_sweep(),
            "damping_sensitivity": self.damping_sensitivity(),
            "whiplash": self.whiplash_case(),
            "whiplash_shock_spectrum": self.whiplash_shock_spectrum(),
        }
        if not include_time_series:
            sections["whiplash"].pop("time_series", None)
            for row in sections["forced_response_sweep"]["rows"]:
                row.pop("duffing_branches_m", None)
        sections["criteria"] = self.criteria(sections)
        sections["case"] = self.case.to_json_dict()
        sections["settings"] = {
            "steady_cycles": self.steady_cycles,
            "samples_per_cycle": self.samples_per_cycle,
            "discard_fraction": self.discard_fraction,
            "record_s": self.record_s,
            "samples_per_second": self.samples_per_second,
            "duffing_relative_nonlinearity": self.duffing_relative_nonlinearity,
            "seed": self.seed,
        }
        return sections
