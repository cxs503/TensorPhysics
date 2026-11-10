"""Self-excited ice-induced vibration lock-in on a velocity-weakening ice force.

Physics
-------
Ice crushing against a compliant structure produces a force that *decreases*
as the relative indentation rate grows (ductile-to-brittle transition,
pressure-area softening).  Writing the ice force as

    F(v_rel) = F_max / (1 + (v_rel / v_c)**p),      v_rel = U - x'

with ``U`` the ice drift speed, ``x'`` the structural velocity, ``v_c`` the
velocity scale of the weakening and ``p > 0`` the weakening exponent, and the
single-degree-of-freedom mode as ``m x'' + c_s x' + k x = F(U - x')``, a small
harmonic perturbation ``x = a sin(w t)`` sees

    F ~ F(U) + |dF/dv| a w cos(w t)

i.e. the ice supplies *negative* damping of magnitude ``c_ice = |dF/dv|``.  When

    c_ice > c_s = 2 zeta m w_n

the trivial equilibrium is unstable (self-excitation): the mode extracts energy
from the drifting ice.

**Limit cycle (describing function).**  The amplitude grows until the ice
supplied damping falls back to the structural damping.  With the exact
(not linearised) force evaluated on ``x = a sin(w_n t)`` the first-harmonic
component in phase with velocity is

    c_ice(a) = (1 / (pi a w_n)) * int_0^{2pi} F(U - a w_n cos(theta)) cos(theta) dtheta

and the limit cycle satisfies ``c_ice(a*) = c_s``.  Because the force saturates
(and is clipped at ``v_rel = 0``), ``c_ice`` decreases monotonically with ``a``,
so the root is unique and the cycle is stable.  ``c_ice(a -> 0) = |dF/dv|``
recovers the linear result analytically -- a machine-checkable identity.

**Lock-in and entrainment.**  The ice crushing frequency is set by the drift
speed and the flaking length, ``f_ice = U / s`` (see
:func:`tensorphysics.ice_vibration.ice_load.crushing_frequency`).  Lock-in is
declared when *both*

  (i) ``r = f_ice / f_n`` lies inside the entrainment band (0.8-1.2 by default);
  (ii) the mode is self-excited (``c_ice > c_s``, i.e. a limit cycle exists).

Condition (i) alone is *not* evidence of lock-in: a linear forced response at
``zeta = 0.02`` already reaches a magnification of ``1/(2 zeta) = 25`` near
``r = 1``.  The discriminating signature of the entrained state is that the
response amplitude is set by the ice force *saturation* rather than by
``1/(2 zeta)``, so the amplification is almost independent of the detuning
across the band, and the response frequency equals ``f_n`` regardless of
``f_ice``.  :func:`lock_in_amplification_check` reports both the amplification
level and this insensitivity, so a "resonance passes for lock-in" mistake is
mechanically detectable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ice_load import crushing_frequency
from .lockin import magnification_factor

__all__ = [
    "LockInCase",
    "LockInState",
    "ice_force_velocity_slope",
    "ice_negative_damping",
    "limit_cycle_amplitude",
    "lock_in_state",
    "lockin_scan",
    "self_excited_speed_window",
    "lock_in_amplification_check",
    "time_domain_limit_cycle",
    "DEFAULT_ENTRAINMENT_BAND",
]

#: Default relative band of ``f_ice / f_n`` over which frequency entrainment is
#: assumed (identical to the 5 %-per-side lock-in band used by
#: :mod:`tensorphysics.ice_vibration.lockin`, widened to 20 % because ice
#: crushing pulses are quasi-periodic rather than harmonic).
DEFAULT_ENTRAINMENT_BAND: Tuple[float, float] = (0.8, 1.2)


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LockInCase:
    """Single ice-loaded mode plus its velocity-weakening ice force.

    Parameters
    ----------
    natural_frequency_hz
        Modal natural frequency ``f_n`` [Hz].
    modal_mass_kg
        Modal (generalised) mass [kg].
    damping_ratio
        Structural damping ratio ``zeta`` (material + hydrodynamic).
    ice_force_max_n
        Asymptotic (zero-rate) ice force ``F_max`` [N].
    ice_critical_velocity_m_s
        Velocity scale of the weakening ``v_c`` [m/s].
    flaking_length_m
        Characteristic flaking length ``s`` in ``f_ice = U / s`` [m].
    velocity_exponent
        Weakening exponent ``p`` (>0).
    entrainment_band
        ``(r_lo, r_hi)`` band of ``f_ice / f_n`` for frequency entrainment.
    quadrature_points
        Number of samples per period for the describing-function integral.
    name
        Case label used in reports.
    """

    natural_frequency_hz: float
    modal_mass_kg: float
    damping_ratio: float
    ice_force_max_n: float
    ice_critical_velocity_m_s: float
    flaking_length_m: float
    velocity_exponent: float = 2.0
    entrainment_band: Tuple[float, float] = DEFAULT_ENTRAINMENT_BAND
    quadrature_points: int = 4096
    name: str = "ice lock-in case"

    def __post_init__(self) -> None:
        if self.natural_frequency_hz <= 0.0:
            raise ValueError("natural_frequency_hz must be positive")
        if self.modal_mass_kg <= 0.0:
            raise ValueError("modal_mass_kg must be positive")
        if not 0.0 < self.damping_ratio < 1.0:
            raise ValueError("damping_ratio must be in (0, 1)")
        if self.ice_force_max_n <= 0.0 or self.ice_critical_velocity_m_s <= 0.0:
            raise ValueError("ice force scale and v_c must be positive")
        if self.flaking_length_m <= 0.0:
            raise ValueError("flaking_length_m must be positive")
        if self.velocity_exponent <= 0.0:
            raise ValueError("velocity_exponent must be positive")
        lo, hi = self.entrainment_band
        if not 0.0 < lo < hi:
            raise ValueError("entrainment_band must be a positive increasing pair")
        if self.quadrature_points < 64:
            raise ValueError("quadrature_points must be >= 64")

    # -- derived modal quantities ------------------------------------------
    @property
    def omega_rad_s(self) -> float:
        return 2.0 * math.pi * self.natural_frequency_hz

    @property
    def stiffness_n_m(self) -> float:
        return self.modal_mass_kg * self.omega_rad_s ** 2

    @property
    def structural_damping_ns_m(self) -> float:
        return 2.0 * self.damping_ratio * self.modal_mass_kg * self.omega_rad_s

    def ice_frequency_hz(self, drift_speed_m_s: float) -> float:
        """``f_ice = U / s``."""
        return crushing_frequency(drift_speed_m_s, self.flaking_length_m)

    def drift_speed_for_ratio(self, frequency_ratio: float) -> float:
        """Drift speed producing ``f_ice / f_n = ratio``."""
        if frequency_ratio <= 0.0:
            raise ValueError("frequency_ratio must be positive")
        return frequency_ratio * self.natural_frequency_hz * self.flaking_length_m

    def mean_ice_force_n(self, drift_speed_m_s: float) -> float:
        """Static (zero structural motion) ice force at a drift speed."""
        if drift_speed_m_s <= 0.0:
            raise ValueError("drift_speed_m_s must be positive")
        q = (drift_speed_m_s / self.ice_critical_velocity_m_s) ** self.velocity_exponent
        return self.ice_force_max_n / (1.0 + q)

    def static_deflection_m(self, drift_speed_m_s: float) -> float:
        """Deflection under the static ice force ``F(U) / k``."""
        return self.mean_ice_force_n(drift_speed_m_s) / self.stiffness_n_m


# ---------------------------------------------------------------------------
# Velocity weakening -> negative damping
# ---------------------------------------------------------------------------


def ice_force_velocity_slope(
    drift_speed_m_s: float,
    ice_force_max_n: float,
    critical_velocity_m_s: float,
    exponent: float = 2.0,
) -> float:
    """``dF/dv_rel`` of the velocity-weakening ice force [N s/m] (negative).

    Analytic derivative of ``F = F_max / (1 + (v/v_c)^p)``:
    ``dF/dv = -F_max p (v/v_c)^(p-1) / (v_c (1 + (v/v_c)^p)^2)``.
    """
    if drift_speed_m_s <= 0.0:
        raise ValueError("drift_speed_m_s must be positive")
    u = drift_speed_m_s / critical_velocity_m_s
    denom = critical_velocity_m_s * (1.0 + u ** exponent) ** 2
    return float(
        -ice_force_max_n * exponent * u ** (exponent - 1.0) / denom
    )


def ice_negative_damping(
    case: LockInCase, drift_speed_m_s: float, amplitude_m: float
) -> float:
    """Ice-supplied (negative) damping from the describing function [N s/m].

    For ``amplitude_m -> 0`` this reduces to ``|dF/dv|`` at the drift speed; for
    finite amplitude the force saturation shrinks it, which is what closes the
    limit cycle.  The returned value is positive and enters the equation of
    motion as ``c_eff = c_s - c_ice(a)``.
    """
    if amplitude_m <= 0.0:
        return abs(
            ice_force_velocity_slope(
                drift_speed_m_s,
                case.ice_force_max_n,
                case.ice_critical_velocity_m_s,
                case.velocity_exponent,
            )
        )
    theta = np.linspace(0.0, 2.0 * math.pi, int(case.quadrature_points), endpoint=False)
    velocity = drift_speed_m_s - amplitude_m * case.omega_rad_s * np.cos(theta)
    v_rel = np.maximum(velocity, 0.0)
    force = case.ice_force_max_n / (
        1.0 + (v_rel / case.ice_critical_velocity_m_s) ** case.velocity_exponent
    )
    first_harmonic_in_phase = 2.0 * float(np.mean(force * np.cos(theta)))
    return float(first_harmonic_in_phase / (amplitude_m * case.omega_rad_s))


def limit_cycle_amplitude(
    case: LockInCase,
    drift_speed_m_s: float,
    amplitude_cap_factor: float = 50.0,
) -> Optional[float]:
    """Limit-cycle amplitude ``a*`` [m] where ``c_ice(a*) = c_s``.

    Returns ``None`` when the mode is not self-excited (``c_ice(0) <= c_s``).
    The amplitude search is bracketed geometrically up to
    ``amplitude_cap_factor * U / w_n`` (beyond that the structural velocity
    amplitude exceeds the drift speed by a large factor and saturation has
    already driven ``c_ice`` to zero).
    """
    damping = case.structural_damping_ns_m
    if ice_negative_damping(case, drift_speed_m_s, 0.0) <= damping:
        return None
    scale = drift_speed_m_s / case.omega_rad_s
    lo = 1e-9 * scale
    hi = 1e-3 * scale
    cap = amplitude_cap_factor * scale
    while ice_negative_damping(case, drift_speed_m_s, hi) > damping:
        hi *= 2.0
        if hi > cap:
            return None
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if ice_negative_damping(case, drift_speed_m_s, mid) > damping:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-10:
            break
    return float(math.sqrt(lo * hi))


# ---------------------------------------------------------------------------
# Lock-in state and scan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LockInState:
    """Response of the mode at one drift speed."""

    drift_speed_m_s: float
    ice_frequency_hz: float
    frequency_ratio: float
    locked_in: bool
    amplitude_m: float
    amplification: float
    forced_amplification: float
    force_amplitude_n: float
    force_range_n: float
    ice_damping_ns_m: float
    structural_damping_ns_m: float
    reason: str

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "drift_speed_m_s": self.drift_speed_m_s,
            "ice_frequency_hz": self.ice_frequency_hz,
            "frequency_ratio": self.frequency_ratio,
            "locked_in": self.locked_in,
            "amplitude_m": self.amplitude_m,
            "amplification": self.amplification,
            "forced_amplification": self.forced_amplification,
            "force_amplitude_n": self.force_amplitude_n,
            "force_range_n": self.force_range_n,
            "ice_damping_ns_m": self.ice_damping_ns_m,
            "structural_damping_ns_m": self.structural_damping_ns_m,
            "reason": self.reason,
        }


def lock_in_state(case: LockInCase, drift_speed_m_s: float) -> LockInState:
    """Evaluate the mode response at a drift speed.

    Locked state: amplitude from the limit cycle, dynamic force amplitude
    ``k a`` (the harmonic reaction of the mode).  Unlocked state: the mode is
    driven at ``f_ice`` and the amplification is the linear magnification
    factor at ``r = f_ice / f_n`` (small response away from resonance).
    """
    if drift_speed_m_s <= 0.0:
        raise ValueError("drift_speed_m_s must be positive")
    f_ice = case.ice_frequency_hz(drift_speed_m_s)
    ratio = f_ice / case.natural_frequency_hz
    static = case.static_deflection_m(drift_speed_m_s)
    forced = magnification_factor(ratio, case.damping_ratio)
    ice_damping = ice_negative_damping(case, drift_speed_m_s, 0.0)
    c_s = case.structural_damping_ns_m
    lo, hi = case.entrainment_band
    in_band = lo <= ratio <= hi
    excited = ice_damping > c_s
    if in_band and excited:
        amp = limit_cycle_amplitude(case, drift_speed_m_s)
        if amp is not None:
            force_amp = case.stiffness_n_m * amp
            return LockInState(
                drift_speed_m_s=float(drift_speed_m_s),
                ice_frequency_hz=float(f_ice),
                frequency_ratio=float(ratio),
                locked_in=True,
                amplitude_m=float(amp),
                amplification=float(amp / static),
                forced_amplification=float(forced),
                force_amplitude_n=float(force_amp),
                force_range_n=float(2.0 * force_amp),
                ice_damping_ns_m=float(ice_damping),
                structural_damping_ns_m=float(c_s),
                reason="entrained: f_ice inside band and c_ice > c_s (limit cycle)",
            )
    if not in_band:
        reason = (
            f"no entrainment: f_ice/f_n = {ratio:.3f} outside "
            f"[{lo:.2f}, {hi:.2f}]"
        )
    else:
        reason = "no self-excitation: c_ice <= c_s (ice damping below structural damping)"
    return LockInState(
        drift_speed_m_s=float(drift_speed_m_s),
        ice_frequency_hz=float(f_ice),
        frequency_ratio=float(ratio),
        locked_in=False,
        amplitude_m=float(forced * static),
        amplification=float(forced),
        forced_amplification=float(forced),
        force_amplitude_n=float(case.stiffness_n_m * forced * static),
        force_range_n=float(2.0 * case.stiffness_n_m * forced * static),
        ice_damping_ns_m=float(ice_damping),
        structural_damping_ns_m=float(c_s),
        reason=reason,
    )


def lockin_scan(
    case: LockInCase, drift_speeds_m_s: Sequence[float]
) -> List[LockInState]:
    """Evaluate :func:`lock_in_state` over a drift-speed sweep."""
    return [lock_in_state(case, float(u)) for u in drift_speeds_m_s]


def self_excited_speed_window(case: LockInCase) -> Optional[Tuple[float, float]]:
    """Drift-speed window where ``c_ice(0) = |dF/dv|`` exceeds ``c_s``.

    ``|dF/dv|`` vanishes both at ``U -> 0`` and ``U -> inf`` and peaks at
    ``(U/v_c)**p = 1/(p-1)``, so the window is bounded on both sides (for
    ``p > 1``).  ``None`` when the ice force can never excite the mode.
    """
    c_s = case.structural_damping_ns_m

    def net(u: float) -> float:
        return ice_negative_damping(case, u, 0.0) - c_s

    peak_ratio = (
        1.0 / (case.velocity_exponent - 1.0) if case.velocity_exponent > 1.0 else 0.0
    )
    u_peak = case.ice_critical_velocity_m_s * peak_ratio ** (1.0 / case.velocity_exponent)
    if u_peak <= 0.0 or net(u_peak) <= 0.0:
        return None
    lo_hi = case.ice_critical_velocity_m_s * 1e-3
    hi_hi = case.ice_critical_velocity_m_s * 1e6
    if net(hi_hi) > 0.0:  # pragma: no cover - p <= 1 gives a monotone tail
        return None
    # lower bound
    lo, hi = lo_hi, u_peak
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if net(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi / lo - 1.0 < 1e-12:
            break
    u_lo = math.sqrt(lo * hi)
    # upper bound
    lo, hi = u_peak, hi_hi
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if net(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-12:
            break
    u_hi = math.sqrt(lo * hi)
    return (float(u_lo), float(u_hi))


# ---------------------------------------------------------------------------
# Independent verification: time-domain limit cycle
# ---------------------------------------------------------------------------


def time_domain_limit_cycle(
    case: LockInCase,
    drift_speed_m_s: float,
    n_cycles: int = 300,
    steps_per_period: int = 256,
    warmup_fraction: float = 0.75,
    initial_displacement_m: Optional[float] = None,
) -> Dict[str, float]:
    """Integrate the full nonlinear SDOF and measure the steady-state amplitude.

    This is an *independent* check on :func:`limit_cycle_amplitude` (describing
    function = first-harmonic quadrature vs RK4 time marching of the exact
    nonlinear equation).  The amplitude is taken as half the peak-to-peak of the
    record after ``warmup_fraction`` of the simulated time, and the frequency is
    estimated from the mean zero-crossing spacing of the steady part.
    """
    from .integrators import rk4_sdof

    if n_cycles < 20:
        raise ValueError("n_cycles must be >= 20")
    period = 1.0 / case.natural_frequency_hz
    duration = float(n_cycles) * period
    dt = period / float(steps_per_period)
    u = float(drift_speed_m_s)
    x0 = (
        float(initial_displacement_m)
        if initial_displacement_m is not None
        else 1e-4 * case.static_deflection_m(u)
    )

    def force(_t: float, _x: float, xdot: float) -> float:
        v = max(u - xdot, 0.0)
        return case.ice_force_max_n / (
            1.0 + (v / case.ice_critical_velocity_m_s) ** case.velocity_exponent
        )

    result = rk4_sdof(
        case.modal_mass_kg,
        case.structural_damping_ns_m,
        case.stiffness_n_m,
        force,
        duration,
        dt,
        initial_displacement=x0,
        sample_stride=1,
    )
    t = np.asarray(result.time_s, dtype=float)
    x = np.asarray(result.displacement, dtype=float)
    start = int(warmup_fraction * len(t))
    tail_t, tail_x = t[start:], x[start:]
    amplitude = 0.5 * float(tail_x.max() - tail_x.min())
    # frequency from mean zero-crossing of the mean-removed tail
    y = tail_x - np.mean(tail_x)
    signs = np.sign(y)
    idx = np.where(np.diff(signs) != 0.0)[0]
    crossings = tail_t[idx]
    freq = 0.5 / float(np.mean(np.diff(crossings))) if len(crossings) > 2 else float("nan")
    return {
        "amplitude_m": amplitude,
        "mean_offset_m": float(np.mean(tail_x)),
        "frequency_hz": freq,
        "duration_s": duration,
        "dt_s": dt,
    }


def lock_in_amplification_check(
    case: LockInCase,
    ratios: Optional[Sequence[float]] = None,
    amplification_threshold: float = 3.0,
) -> Dict[str, Any]:
    """C1 check: amplification inside the entrainment band.

    Sweeps ``r = f_ice / f_n`` across the entrainment band, evaluates the locked
    response and returns

    * ``rows`` -- per-ratio state (drift speed, amplitude, amplification),
    * ``min_amplification`` / ``pass`` -- the strictest reading of C1
      (every swept ratio inside the band must amplify by more than the
      threshold),
    * ``entrainment_insensitivity`` -- ratio of the amplification spread of the
      locked response to that of the linear forced response over the same band
      (a value << 1 is the discriminating signature of entrainment; a linear
      resonance would give ~1),
    * ``forced_response_max`` -- the linear magnification maximum in the band,
      quoted to make clear how much of the amplification a purely linear
      resonance *without* any ice feedback would already produce.
    """
    lo, hi = case.entrainment_band
    if ratios is None:
        ratios = list(np.linspace(lo, hi, 9))
    rows: List[Dict[str, Any]] = []
    amps: List[float] = []
    forced: List[float] = []
    all_locked = True
    for ratio in ratios:
        u = case.drift_speed_for_ratio(float(ratio))
        state = lock_in_state(case, u)
        rows.append(state.to_json_dict())
        if state.locked_in:
            amps.append(state.amplification)
        else:
            all_locked = False
        forced.append(state.forced_amplification)
    min_amp = float(min(amps)) if amps else 0.0
    locked_spread = (max(amps) / min(amps)) if len(amps) > 1 and min_amp > 0.0 else 1.0
    forced_spread = (max(forced) / min(forced)) if min(forced) > 0.0 else math.inf
    return {
        "case": case.name,
        "entrainment_band": [lo, hi],
        "amplification_threshold": float(amplification_threshold),
        "rows": rows,
        "all_ratios_locked": bool(all_locked and len(amps) == len(rows)),
        "min_amplification": min_amp,
        "max_amplification": float(max(amps)) if amps else 0.0,
        "pass": bool(all_locked and amps and min_amp > amplification_threshold),
        "locked_amplification_spread": float(locked_spread),
        "forced_amplification_spread": float(forced_spread),
        "entrainment_insensitivity": float(locked_spread / forced_spread)
        if forced_spread
        else None,
        "forced_response_max": float(max(forced)) if forced else 0.0,
    }