"""Voyage-accumulated fatigue damage of an ice-going hull girder.

Scope
-----
This module lifts the fatigue assessment from a **single load window** (one
stress history, one rainflow count, one damage number) to a **whole voyage and
a whole operating life**:

1. **S-N description.**  A power law ``sigma_a**m * N = C`` (``m`` = S-N slope,
   ``C`` = intercept) is the primary curve; it is *fitted* (least squares in
   ``log N`` / ``log sigma_a`` space) to whatever reference data are supplied,
   and :meth:`PowerLawSN.from_dnv` fits it to the DNV-RP-C203 first (steep)
   segment.  The bilinear DNV curve
   (:class:`tensorphysics.fatigue.sn_curve.DNCurve`) is accepted as well, so the
   single-window chain of the fatigue package can be reused unchanged.

2. **Stress response to ice loading.**  For every ice condition the load is a
   train of half-sine crushing pulses (jittered period, lognormal amplitude
   scatter, optional ridge-impact tail).  The midship stress is the sum of

   * the **quasi-static** bow-load bending moment solved on the free-free
     hull-girder beam, ``M_mid = F (L - a) / 2`` for a bow load at ``a <= L/2``,
     ``sigma = M / Z`` with ``Z`` the midship section modulus, and
   * the **ice-induced vibration** residual of the first vertical bending mode,
     ``m1 q'' + c1 q' + k1 q = phi_bow F(t)`` and
     ``sigma_vib = EI phi''_mid [q(t) - phi_bow F(t) / k1] / Z``,

   i.e. only the oscillatory part of the modal response (the modal static
   offset is removed so the quasi-static solution is not counted twice).

   This is the classical "quasi-static + whipping/lock-in" decomposition.  Its
   one approximation is stated openly: the higher vertical bending modes are
   treated as *quasi-static* (their inertia is neglected), because only the
   fundamental carries the ice-induced vibration energy.  For the reference
   hull the first mode carries 37.9 % of the exact static midship moment of a
   bow point load, so the retained vibratory dynamics is the fundamental's
   oscillatory share while the remaining 62.1 % enters through the exact static
   beam solution.  Above the fundamental frequency the mode-1 residual is
   180 deg out of phase with the quasi-static response and therefore *reduces*
   the total stress, which is why the vibratory share (point 4) becomes
   negative there; the share is interpreted inside the sub-critical band
   ``f / f1 < 1`` and for the reference voyage (all legs have ``f / f1 <= 0.9``).

3. **Rainflow + Miner accumulation.**  ASTM E1049-85 four-point rainflow
   counting on the total stress history, using the ASTM-conformant stack
   implementation :func:`tensorphysics.fatigue.rainflow.rainflow_counting_stack`
   (the legacy list variant ``rainflow_counting`` closes the *outer* range of a
   four-point set - a documented legacy behaviour of that module - and
   over-estimates large ranges; see the benchmark for a quantified comparison),
   then ``D = sum_i n_i / N_i``; per-leg damage is scaled
   from a representative window to the full leg duration (linear damage
   scaling, see :func:`leg_damage`), and voyage damage is the sum over legs:
   ``D_voyage = sum_legs D_leg``.

4. **Ice-induced vibration contribution.**  ``f_vib = (D_total - D_static)
   / D_total``; it is bounded in ``(0, 1)`` wherever both components carry
   damage, which is the quantity needed to judge whether vibration matters.

5. **Residual life.**  With a (nearly) constant route severity,
   ``N_total = 1 / D_voyage`` voyages and ``N_remaining = (1 - D) * N_total``,
   so ``D -> 1`` gives ``N_remaining -> 0`` and ``D = 0`` gives
   ``N_remaining = N_total`` exactly (verified in the tests).

References
----------
* ASTM E1049-85, "Standard Practices for Cycle Counting in Fatigue Analysis".
* DNV-RP-C203 (2016), "Fatigue design of offshore steel structures".
* Miner, M.A. (1945), "Cumulative damage in fatigue", J. Appl. Mech. 12,
  A159-A164 (Palmgren-Miner linear rule).
* Karr, D.G. et al. (1985), ice-induced hull vibration and hull fatigue damage.
* Timoshenko & Gere, beam statics: free-free beam influence line used for the
  bow-load midship moment.

Caveats
-------
* Linear elastic, linear-damage framework: no crack growth, no inspection
  credit, no load-sequence interaction, no mean-stress (Goodman) correction.
* The ice load is a pulse-train model; velocity-weakening lock-in feedback is
  *not* modelled, so a leg whose pulse frequency sits close to the hull mode
  over-predicts the vibratory amplitude (reported and bounded in the benchmark).
* Modal response is truncated to the first vertical bending mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..fatigue.rainflow import RainflowResult, rainflow_counting_stack
from ..fatigue.sn_curve import DNCurve
from ..ice_vibration.ice_load import half_sine_pulse
from ..ice_vibration.integrators import rk4_sdof
from ..ice_vibration.modal import HullParameters, free_free_beam_modal_data

__all__ = [
    "PowerLawSN",
    "PowerLawFit",
    "fit_power_law_sn",
    "power_law_from_dnv",
    "DamageBlock",
    "damage_from_ranges",
    "history_damage",
    "n_to_failure_of",
    "rainflow_counting_reference",
    "HullGirder",
    "bow_load_midship_moment",
    "midship_static_stress_mpa",
    "HullGirderVibration",
    "build_hull_girder_vibration",
    "IceForceWindow",
    "ice_force_window",
    "VoyageLeg",
    "LegStressHistory",
    "leg_stress_history",
    "LegDamage",
    "leg_damage",
    "VoyageProfile",
    "VoyageDamage",
    "voyage_damage",
    "DamageTrajectory",
    "accumulate_voyages",
    "voyages_to_failure",
    "RemainingLife",
    "remaining_life",
    "VoyageFatigueCase",
    "build_reference_case",
    "reference_profile",
]

#: Young's modulus of steel (Pa).
YOUNGS_STEEL_PA = 2.06e11

#: Density of sea water (kg/m^3) - only used for displacement cross-checks.
RHO_SEA_WATER = 1025.0


# ===========================================================================
# 1. S-N description
# ===========================================================================


@dataclass(frozen=True)
class PowerLawSN:
    """Power-law S-N curve ``sigma_a**m * N = C``.

    ``N = C sigma_a**(-m)`` with the stress **amplitude** in MPa and ``N`` the
    number of cycles to failure.  Curve formulations stated in terms of the
    stress **range** (DNV-RP-C203 uses ranges) convert through
    ``C_amplitude = K_range / 2**m``; :meth:`n_to_failure_range` does it.
    """

    m: float
    log10_c: float
    name: str = "power_law"

    def __post_init__(self) -> None:
        if not (isinstance(self.m, (int, float)) and math.isfinite(self.m) and self.m > 0.0):
            raise ValueError(f"power-law exponent m must be positive and finite, got {self.m!r}")
        if not (isinstance(self.log10_c, (int, float)) and math.isfinite(self.log10_c)):
            raise ValueError(f"log10(C) must be finite, got {self.log10_c!r}")

    @property
    def c(self) -> float:
        """Intercept ``C`` of ``sigma_a**m * N = C``."""
        return float(10.0 ** self.log10_c)

    def n_to_failure(self, stress_amplitude_mpa: float) -> float:
        """Cycles to failure for a stress amplitude (MPa, >= 0)."""
        sigma_a = float(stress_amplitude_mpa)
        if sigma_a <= 0.0:
            return math.inf
        return float(self.c * sigma_a ** (-self.m))

    def n_to_failure_range(self, stress_range_mpa: float) -> float:
        """Cycles to failure for a stress range (MPa, >= 0)."""
        return self.n_to_failure(0.5 * float(stress_range_mpa))

    def stress_amplitude_for_life(self, n_cycles: float) -> float:
        """Inverse relation: amplitude (MPa) surviving ``n_cycles`` cycles."""
        n = float(n_cycles)
        if n <= 0.0:
            return math.inf
        return float((self.c / n) ** (1.0 / self.m))

    def stress_range_for_life(self, n_cycles: float) -> float:
        return 2.0 * self.stress_amplitude_for_life(n_cycles)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "PowerLawSN",
            "name": self.name,
            "m": float(self.m),
            "log10_C": float(self.log10_c),
            "C": self.c,
            "form": "sigma_a**m * N = C (sigma_a in MPa, amplitude)",
        }

    @classmethod
    def from_dnv(
        cls,
        curve_class: str = "D",
        environment: str = "seawater_cp",
        n_lives: Optional[Sequence[float]] = None,
    ) -> "PowerLawSN":
        """Fit the power law to the steep (first) segment of a DNV-RP-C203 curve.

        ``n_lives`` defaults to log-spaced lives inside the first segment
        (``1e4 ... 1e7`` cycles), where the DNV curve is a straight line in
        ``log N`` / ``log sigma`` space, so the fit is exact up to round-off and
        the returned ``m`` reproduces the code value.
        """
        lives = np.asarray(n_lives if n_lives is not None else np.logspace(4.0, 7.0, 25), dtype=float)
        curve = DNCurve(curve_class=curve_class, environment=environment)
        ranges = np.array([curve.stress_for_life(float(n)) for n in lives], dtype=float)
        fit = fit_power_law_sn(0.5 * ranges, lives)
        return cls(m=fit.m, log10_c=fit.log10_c, name=f"power_law_fit_DNV_{curve_class}_{environment}")


@dataclass(frozen=True)
class PowerLawFit:
    """Least-squares fit of ``log10(N) = log10(C) - m log10(sigma_a)``."""

    m: float
    log10_c: float
    r_squared: float
    n_points: int
    rmse_log10_life: float

    def curve(self, name: str = "power_law_fit") -> PowerLawSN:
        return PowerLawSN(m=self.m, log10_c=self.log10_c, name=name)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "PowerLawFit",
            "m": float(self.m),
            "log10_C": float(self.log10_c),
            "R_squared": float(self.r_squared),
            "n_points": int(self.n_points),
            "rmse_log10_life": float(self.rmse_log10_life),
        }


def fit_power_law_sn(stress_amplitudes_mpa: Sequence[float], n_cycles: Sequence[float]) -> PowerLawFit:
    """Fit ``sigma_a**m * N = C`` by least squares in log-log space.

    Pairs must be positive and finite; at least two distinct amplitudes are
    required.  ``R_squared`` is the coefficient of determination of the linear
    regression of ``log10 N`` on ``log10 sigma_a``.
    """
    sa = np.asarray(stress_amplitudes_mpa, dtype=float)
    nn = np.asarray(n_cycles, dtype=float)
    if sa.shape != nn.shape:
        raise ValueError(f"shape mismatch: {sa.shape} vs {nn.shape}")
    mask = np.isfinite(sa) & np.isfinite(nn) & (sa > 0.0) & (nn > 0.0)
    sa, nn = sa[mask], nn[mask]
    if sa.size < 2:
        raise ValueError("at least two valid (sigma_a, N) pairs are required")
    x = np.log10(sa)
    y = np.log10(nn)
    if np.ptp(x) <= 0.0:
        raise ValueError("stress amplitudes must not all be identical")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return PowerLawFit(
        m=float(-slope),
        log10_c=float(intercept),
        r_squared=float(r_squared),
        n_points=int(sa.size),
        rmse_log10_life=float(np.sqrt(ss_res / sa.size)),
    )


def power_law_from_dnv(curve_class: str = "D", environment: str = "seawater_cp") -> PowerLawSN:
    """:meth:`PowerLawSN.from_dnv` convenience wrapper."""
    return PowerLawSN.from_dnv(curve_class=curve_class, environment=environment)


# ===========================================================================
# 2. Damage primitives (rainflow + Miner)
# ===========================================================================


def n_to_failure_of(curve: Any, stress_range_mpa: float) -> float:
    """Cycles to failure for a *stress range* on any supported curve object.

    Accepts :class:`PowerLawSN` (amplitude based, converted internally),
    :class:`tensorphysics.fatigue.sn_curve.DNCurve` (range based) and any object
    exposing ``n_to_failure_range`` or ``n_to_failure``.
    """
    if hasattr(curve, "n_to_failure_range") and callable(getattr(curve, "n_to_failure_range")):
        return float(curve.n_to_failure_range(stress_range_mpa))  # type: ignore[attr-defined]
    if hasattr(curve, "n_to_failure") and callable(getattr(curve, "n_to_failure")):
        return float(curve.n_to_failure(stress_range_mpa))  # type: ignore[attr-defined]
    raise TypeError(f"curve object {type(curve)!r} exposes neither n_to_failure_range nor n_to_failure")


@dataclass(frozen=True)
class DamageBlock:
    """Miner damage of one (ranges, counts) block."""

    damage: float
    n_cycles: float
    n_full_cycles: float
    n_half_cycles: int
    max_range_mpa: float
    n_blocks: int
    sn_name: str = ""

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "DamageBlock",
            "damage": float(self.damage),
            "n_cycles": float(self.n_cycles),
            "n_full_cycles": float(self.n_full_cycles),
            "n_half_cycles": int(self.n_half_cycles),
            "max_range_mpa": float(self.max_range_mpa),
            "n_blocks": int(self.n_blocks),
            "sn_name": self.sn_name,
        }


def damage_from_ranges(stress_ranges_mpa: Sequence[float], counts: Sequence[float], sn_curve: Any) -> DamageBlock:
    """``D = sum_i n_i / N(ranges_i)`` for one counted block.

    Non-positive ranges contribute nothing (``N = inf``): plain Palmgren-Miner
    with no correction factors.
    """
    ranges = np.asarray(stress_ranges_mpa, dtype=float)
    cnts = np.asarray(counts, dtype=float)
    if ranges.shape != cnts.shape:
        raise ValueError(f"shape mismatch: ranges {ranges.shape} vs counts {cnts.shape}")
    damage = 0.0
    for rng, cnt in zip(ranges, cnts):
        if not (np.isfinite(rng) and np.isfinite(cnt)) or cnt <= 0.0:
            continue
        n_fail = n_to_failure_of(sn_curve, rng)
        if math.isfinite(n_fail) and n_fail > 0.0:
            damage += float(cnt) / n_fail
    n_cycles = float(np.sum(cnts)) if cnts.size else 0.0
    max_range = float(np.max(ranges)) if ranges.size else 0.0
    return DamageBlock(
        damage=float(damage),
        n_cycles=n_cycles,
        n_full_cycles=float(np.count_nonzero(np.isclose(cnts, 1.0))),
        n_half_cycles=int(np.count_nonzero(np.isclose(cnts, 0.5))),
        max_range_mpa=max_range,
        n_blocks=int(ranges.size),
        sn_name=getattr(sn_curve, "name", type(sn_curve).__name__),
    )


def history_damage(stress_history_mpa: Sequence[float], sn_curve: Any) -> DamageBlock:
    """Rainflow-count a stress history and return its Miner damage."""
    history = np.asarray(stress_history_mpa, dtype=float)
    rf: RainflowResult = rainflow_counting_stack(history)
    return damage_from_ranges(rf.ranges, rf.counts, sn_curve)


def _turning_points(signal: np.ndarray) -> np.ndarray:
    """Alternating peaks/valleys, plateau-robust.

    Runs of exactly equal samples (the zero gaps between ice pulses, clipped
    peaks) are compressed to one sample first: otherwise a plateau of length
    ``> 1`` hides the extremum beside it and cycles are silently lost (a defect
    that was found in ``fatigue.rainflow._find_peaks_valleys`` during this work
    and fixed there; the two extraction routines stay independent pieces of
    code on purpose, so that agreement between the counters is evidence).
    """
    if signal.size < 3:
        return signal.copy()
    steps = np.diff(signal)
    keep = np.concatenate(([0], np.nonzero(steps != 0.0)[0] + 1))
    if keep[-1] != signal.size - 1 and signal[-1] != signal[keep[-1]]:
        keep = np.append(keep, signal.size - 1)
    s = signal[keep]
    if s.size < 3:
        return s
    d = np.diff(s)
    turning = [0]
    for i in range(1, len(d)):
        if d[i] * d[i - 1] < 0.0:
            turning.append(i)
    turning.append(s.size - 1)
    return s[np.unique(np.asarray(turning, dtype=int))]


def rainflow_counting_reference(signal: Sequence[float]) -> RainflowResult:
    """Independent ASTM E1049-85 four-point implementation (cross-check).

    Written as an exhaustive "closing check" over the turning-point list
    (repeated full scans with cycle removal) instead of the stack formulation
    used by :func:`tensorphysics.fatigue.rainflow.rainflow_counting_stack`.  It exists
    purely as an *independent* algorithmic reference: agreement between the two
    is evidence rather than a tautology because they share neither code nor data
    structures.  Residual (unclosed) ranges are counted as half cycles, as the
    standard prescribes.
    """
    pts = [float(v) for v in _turning_points(np.asarray(signal, dtype=float))]
    ranges: List[float] = []
    means: List[float] = []
    counts: List[float] = []
    i = 0
    while i + 3 < len(pts):
        y1, y2, y3, y4 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        inner = abs(y3 - y2)
        if inner <= abs(y2 - y1) and inner <= abs(y4 - y3):
            ranges.append(inner)
            means.append(0.5 * (y2 + y3))
            counts.append(1.0)
            del pts[i + 1 : i + 3]
            i = max(i - 2, 0)
        else:
            i += 1
    for j in range(0, len(pts) - 1):
        ranges.append(abs(pts[j + 1] - pts[j]))
        means.append(0.5 * (pts[j + 1] + pts[j]))
        counts.append(0.5)
    return RainflowResult(
        ranges=np.asarray(ranges, dtype=float),
        means=np.asarray(means, dtype=float),
        counts=np.asarray(counts, dtype=float),
        n_full=int(np.count_nonzero(np.isclose(counts, 1.0))),
        n_half=int(np.count_nonzero(np.isclose(counts, 0.5))),
    )


# ===========================================================================
# 3. Hull girder: section properties and static stress
# ===========================================================================


@dataclass(frozen=True)
class HullGirder:
    """Midship hull-girder stiffness and deck section modulus.

    ``section_modulus_m3`` is ``Z = I / (D/2)`` with ``D`` the hull depth, so the
    deck stress for a midship moment ``M`` is ``sigma = M / Z``.
    """

    bending_stiffness_nm2: float
    section_modulus_m3: float
    depth_m: float
    youngs_modulus_pa: float = YOUNGS_STEEL_PA

    def __post_init__(self) -> None:
        for name in ("bending_stiffness_nm2", "section_modulus_m3", "depth_m", "youngs_modulus_pa"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0):
                raise ValueError(f"HullGirder.{name} must be positive and finite, got {value!r}")

    @property
    def moment_of_inertia_m4(self) -> float:
        return float(self.bending_stiffness_nm2) / float(self.youngs_modulus_pa)

    @classmethod
    def from_hull(
        cls,
        hull: HullParameters,
        bending_stiffness_nm2: float,
        depth_m: Optional[float] = None,
        youngs_modulus_pa: float = YOUNGS_STEEL_PA,
    ) -> "HullGirder":
        """Build from a hull and an ``EI``; depth defaults to ``L / 12``."""
        depth = float(depth_m) if depth_m is not None else float(hull.length_m) / 12.0
        inertia = float(bending_stiffness_nm2) / float(youngs_modulus_pa)
        return cls(
            bending_stiffness_nm2=float(bending_stiffness_nm2),
            section_modulus_m3=float(inertia / (0.5 * depth)),
            depth_m=depth,
            youngs_modulus_pa=float(youngs_modulus_pa),
        )

    def stress_from_moment_mpa(self, moment_nm: Any) -> Any:
        """``sigma = |M| / Z`` in MPa (scalar or array in, same shape out)."""
        return np.abs(np.asarray(moment_nm, dtype=float)) / self.section_modulus_m3 / 1.0e6

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "HullGirder",
            "bending_stiffness_nm2": float(self.bending_stiffness_nm2),
            "moment_of_inertia_m4": self.moment_of_inertia_m4,
            "section_modulus_m3": float(self.section_modulus_m3),
            "depth_m": float(self.depth_m),
            "youngs_modulus_pa": float(self.youngs_modulus_pa),
        }


def bow_load_midship_moment(force_n: float, length_m: float, application_x_m: float) -> float:
    """Midship bending moment of a free-free beam under a single point load.

    With the two end reactions of the flotation as supports (standard free-free
    idealisation) a load ``P`` at ``a`` from the aft perpendicular gives

    ``M_mid = P (L - a) / 2``  for ``a <= L/2``  (aft reaction times ``L/2``),
    ``M_mid = P a / 2``        for ``a >= L/2``  (forward reaction times ``L/2``).

    For a bow load at ``a = L/4`` this is ``M_mid = 0.375 P L``.
    """
    L = float(length_m)
    a = float(application_x_m)
    P = float(force_n)
    if L <= 0.0:
        raise ValueError("length_m must be positive")
    if not 0.0 <= a <= L:
        raise ValueError(f"application_x_m={a} outside [0, {L}]")
    if a <= 0.5 * L:
        return P * (L - a) / 2.0
    return P * a / 2.0


def midship_static_stress_mpa(
    force_n: Any,
    hull: HullParameters,
    girder: HullGirder,
    application_x_m: Optional[float] = None,
) -> Any:
    """Quasi-static midship stress of a bow ice load (MPa)."""
    a = float(application_x_m) if application_x_m is not None else 0.25 * float(hull.length_m)
    forces = np.abs(np.asarray(force_n, dtype=float))
    factor = (float(hull.length_m) - a) / 2.0 if a <= 0.5 * float(hull.length_m) else a / 2.0
    return girder.stress_from_moment_mpa(forces * factor)


# ===========================================================================
# 4. Ice-induced hull-girder vibration
# ===========================================================================


@dataclass(frozen=True)
class HullGirderVibration:
    """First vertical bending mode driving the ice-induced vibration stress."""

    frequency_hz: float
    damping_ratio: float
    modal_mass_kg: float
    curvature_midship_per_m: float
    bow_ordinate: float
    bending_stiffness_nm2: float
    section_modulus_m3: float

    def __post_init__(self) -> None:
        if not (isinstance(self.frequency_hz, (int, float)) and math.isfinite(self.frequency_hz) and self.frequency_hz > 0.0):
            raise ValueError("frequency_hz must be positive and finite")
        if not (0.0 <= float(self.damping_ratio) < 1.0):
            raise ValueError("damping_ratio must satisfy 0 <= zeta < 1")
        if not (float(self.modal_mass_kg) > 0.0):
            raise ValueError("modal_mass_kg must be positive")
        if abs(float(self.bow_ordinate)) <= 0.0:
            raise ValueError("bow_ordinate must be non-zero (bow load must excite the mode)")

    @property
    def angular_frequency(self) -> float:
        return 2.0 * math.pi * float(self.frequency_hz)

    @property
    def modal_stiffness(self) -> float:
        return float(self.modal_mass_kg) * self.angular_frequency**2

    @property
    def modal_damping(self) -> float:
        return 2.0 * float(self.damping_ratio) * float(self.modal_mass_kg) * self.angular_frequency

    def stress_per_modal_displacement_mpa(self) -> float:
        """``|EI phi''_mid / Z|`` in MPa per metre of modal amplitude."""
        return abs(
            float(self.bending_stiffness_nm2) * float(self.curvature_midship_per_m)
            / float(self.section_modulus_m3)
        ) / 1.0e6

    def steady_state_amplification(self, excitation_frequency_hz: float) -> float:
        """Analytic harmonic magnification ``|H(r)|`` at ``r = f / f1``."""
        r = float(excitation_frequency_hz) / float(self.frequency_hz)
        return 1.0 / math.sqrt((1.0 - r * r) ** 2 + (2.0 * float(self.damping_ratio) * r) ** 2)

    def vibration_stress_mpa(
        self,
        force_n: Sequence[float],
        dt_s: float,
        static_offset: bool = True,
    ) -> np.ndarray:
        """Vibratory midship stress for a bow ice-force history (MPa, signed).

        Solves ``m1 q'' + c1 q' + k1 q = phi_bow F(t)`` with RK4 and returns
        ``EI phi''_mid q_dyn / Z`` where, with ``static_offset=True``, the modal
        static offset ``phi_bow F(t) / k1`` is removed so that the oscillatory
        (whipping / lock-in) part is isolated and the quasi-static beam solution
        is not counted twice.
        """
        force = np.asarray(force_n, dtype=float)
        if force.size < 2:
            raise ValueError("force history must contain at least two samples")
        if not (dt_s > 0.0 and math.isfinite(dt_s)):
            raise ValueError("dt_s must be positive and finite")
        phi = float(self.bow_ordinate)
        t_end = (force.size - 1) * float(dt_s)
        times = np.arange(force.size, dtype=float) * float(dt_s)

        def modal_force(t: float) -> float:
            return phi * float(np.interp(t, times, force))

        result = rk4_sdof(
            mass=float(self.modal_mass_kg),
            damping=self.modal_damping,
            stiffness=self.modal_stiffness,
            force=modal_force,
            duration_s=t_end,
            dt_s=float(dt_s),
        )
        q = np.asarray(result.displacement, dtype=float)
        if q.size != force.size:  # defensive: integrator may add the final sample
            q = np.interp(times, np.linspace(0.0, t_end, q.size), q)
        q_dyn = q - phi * force / self.modal_stiffness if static_offset else q
        scale = self.stress_per_modal_displacement_mpa()
        return scale * q_dyn


def build_hull_girder_vibration(
    hull: HullParameters,
    girder: HullGirder,
    damping_ratio: float = 0.03,
) -> HullGirderVibration:
    """First vertical bending mode of the equivalent free-free hull girder."""
    modal = free_free_beam_modal_data(
        hull=hull,
        bending_stiffness_nm2=float(girder.bending_stiffness_nm2),
        n_modes=1,
        damping_ratios=(float(damping_ratio),),
    )
    return HullGirderVibration(
        frequency_hz=float(modal.frequencies_hz[0]),
        damping_ratio=float(modal.damping_ratios[0]),
        modal_mass_kg=float(modal.modal_masses_kg[0]),
        curvature_midship_per_m=float(modal.mode_shape_midship_curvature[0]),
        bow_ordinate=float(modal.mode_shape_bow[0]),
        bending_stiffness_nm2=float(girder.bending_stiffness_nm2),
        section_modulus_m3=float(girder.section_modulus_m3),
    )


# ===========================================================================
# 5. Ice load windows
# ===========================================================================


@dataclass(frozen=True)
class IceForceWindow:
    """A representative window of an ice-load time history."""

    time_s: np.ndarray
    force_n: np.ndarray
    peak_force_n: float
    rms_force_n: float
    mean_force_n: float
    n_pulses: int
    dt_s: float
    duration_s: float

    @property
    def peak_factor(self) -> float:
        """Peak / RMS ratio (a narrowband pulse train is well above 1.41)."""
        return self.peak_force_n / self.rms_force_n if self.rms_force_n > 0.0 else 0.0

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "IceForceWindow",
            "duration_s": float(self.duration_s),
            "dt_s": float(self.dt_s),
            "n_samples": int(self.force_n.size),
            "n_pulses": int(self.n_pulses),
            "peak_force_n": float(self.peak_force_n),
            "rms_force_n": float(self.rms_force_n),
            "mean_force_n": float(self.mean_force_n),
            "peak_factor": float(self.peak_factor),
        }


def ice_force_window(
    duration_s: float,
    dt_s: float,
    peak_force_n: float,
    frequency_hz: float,
    pulse_duration_s: float,
    seed: int,
    amplitude_cv: float = 0.35,
    ridge_probability: float = 0.0,
    ridge_factor: float = 1.8,
) -> IceForceWindow:
    """Half-sine crushing pulse train with jittered period and scattered peaks.

    Parameters
    ----------
    duration_s, dt_s
        Window length and sampling step.
    peak_force_n, frequency_hz, pulse_duration_s
        Nominal pulse amplitude, pulse repetition frequency
        (``f = U / s`` for level ice: drift speed over flaking length) and pulse
        width.
    seed
        Seed of the deterministic generator (reproducibility).
    amplitude_cv
        Lognormal coefficient of variation of the pulse amplitude
        (ice crushing is strongly scattered).
    ridge_probability, ridge_factor
        Probability that a pulse is a ridge impact and the amplitude multiplier
        applied to it.
    """
    if not (duration_s > 0.0 and dt_s > 0.0):
        raise ValueError("duration_s and dt_s must be positive")
    if dt_s > pulse_duration_s / 4.0:
        raise ValueError("dt_s must resolve the pulse (<= pulse_duration_s / 4)")
    if not (peak_force_n > 0.0 and frequency_hz > 0.0 and pulse_duration_s > 0.0):
        raise ValueError("peak_force_n, frequency_hz and pulse_duration_s must be positive")
    rng = np.random.default_rng(int(seed))
    n_samples = int(round(duration_s / dt_s)) + 1
    t = np.arange(n_samples, dtype=float) * float(dt_s)
    force = np.zeros(n_samples)
    period = 1.0 / float(frequency_hz)
    n_pulses = int(math.floor(float(duration_s) * float(frequency_hz)))
    for k in range(n_pulses):
        t0 = k * period + float(rng.uniform(-0.15, 0.15)) * period
        amp = float(peak_force_n) * float(np.clip(rng.lognormal(mean=0.0, sigma=float(amplitude_cv)), 0.25, 2.5))
        if ridge_probability > 0.0 and float(rng.random()) < float(ridge_probability):
            amp *= float(ridge_factor)
        if t0 + pulse_duration_s < 0.0:
            continue
        force += half_sine_pulse(t - t0, amp, float(pulse_duration_s))
    return IceForceWindow(
        time_s=t,
        force_n=force,
        peak_force_n=float(np.max(force)),
        rms_force_n=float(np.sqrt(np.mean(force**2))),
        mean_force_n=float(np.mean(force)),
        n_pulses=n_pulses,
        dt_s=float(dt_s),
        duration_s=float(duration_s),
    )


# ===========================================================================
# 6. Voyage legs and voyage damage
# ===========================================================================


@dataclass(frozen=True)
class VoyageLeg:
    """One leg of a voyage: a single ice condition held for ``duration_s``."""

    name: str
    duration_s: float
    peak_force_n: float
    ice_frequency_hz: float
    pulse_duration_s: float
    seed: int
    amplitude_cv: float = 0.35
    ridge_probability: float = 0.0
    ridge_factor: float = 1.8
    window_duration_s: float = 60.0
    dt_s: float = 0.02
    application_x_fraction: float = 0.25
    description: str = ""

    def __post_init__(self) -> None:
        if not (self.duration_s > 0.0 and self.window_duration_s > 0.0):
            raise ValueError(f"leg {self.name}: durations must be positive")
        if self.window_duration_s > self.duration_s:
            raise ValueError(f"leg {self.name}: window longer than the leg itself")
        if not (0.0 < self.application_x_fraction < 1.0):
            raise ValueError(f"leg {self.name}: application_x_fraction must be in (0, 1)")

    @property
    def scale_to_leg(self) -> float:
        """Cycles in the leg / cycles in the representative window."""
        return float(self.duration_s) / float(self.window_duration_s)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "VoyageLeg",
            "name": self.name,
            "duration_s": float(self.duration_s),
            "duration_h": float(self.duration_s) / 3600.0,
            "peak_force_n": float(self.peak_force_n),
            "ice_frequency_hz": float(self.ice_frequency_hz),
            "pulse_duration_s": float(self.pulse_duration_s),
            "amplitude_cv": float(self.amplitude_cv),
            "ridge_probability": float(self.ridge_probability),
            "ridge_factor": float(self.ridge_factor),
            "window_duration_s": float(self.window_duration_s),
            "dt_s": float(self.dt_s),
            "application_x_fraction": float(self.application_x_fraction),
            "seed": int(self.seed),
            "scale_to_leg": float(self.scale_to_leg),
            "description": self.description,
        }


@dataclass(frozen=True)
class LegStressHistory:
    """Representative-window stress history of one leg."""

    leg_name: str
    time_s: np.ndarray
    force_n: np.ndarray
    stress_static_mpa: np.ndarray
    stress_vibration_mpa: np.ndarray
    stress_total_mpa: np.ndarray

    @property
    def peak_total_mpa(self) -> float:
        return float(np.max(np.abs(self.stress_total_mpa)))

    @property
    def peak_static_mpa(self) -> float:
        return float(np.max(np.abs(self.stress_static_mpa)))

    @property
    def peak_vibration_mpa(self) -> float:
        return float(np.max(np.abs(self.stress_vibration_mpa)))

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "LegStressHistory",
            "leg_name": self.leg_name,
            "n_samples": int(self.time_s.size),
            "peak_total_mpa": self.peak_total_mpa,
            "peak_static_mpa": self.peak_static_mpa,
            "peak_vibration_mpa": self.peak_vibration_mpa,
            "rms_total_mpa": float(np.sqrt(np.mean(self.stress_total_mpa**2))),
        }


def leg_stress_history(
    leg: VoyageLeg,
    hull: HullParameters,
    girder: HullGirder,
    vibration: HullGirderVibration,
) -> LegStressHistory:
    """Total midship stress history (quasi-static + ice-induced vibration)."""
    window = ice_force_window(
        duration_s=leg.window_duration_s,
        dt_s=leg.dt_s,
        peak_force_n=leg.peak_force_n,
        frequency_hz=leg.ice_frequency_hz,
        pulse_duration_s=leg.pulse_duration_s,
        seed=leg.seed,
        amplitude_cv=leg.amplitude_cv,
        ridge_probability=leg.ridge_probability,
        ridge_factor=leg.ridge_factor,
    )
    x_apply = leg.application_x_fraction * float(hull.length_m)
    sigma_static = np.asarray(
        midship_static_stress_mpa(window.force_n, hull, girder, application_x_m=x_apply), dtype=float
    )
    sigma_vib = vibration.vibration_stress_mpa(window.force_n, leg.dt_s)
    return LegStressHistory(
        leg_name=leg.name,
        time_s=window.time_s,
        force_n=window.force_n,
        stress_static_mpa=sigma_static,
        stress_vibration_mpa=sigma_vib,
        stress_total_mpa=sigma_static + sigma_vib,
    )


@dataclass(frozen=True)
class LegDamage:
    """Damage of one leg, scaled from its representative window."""

    leg_name: str
    damage_total: float
    damage_static: float
    damage_vibration_only: float
    ice_vibration_fraction: float
    scale_to_leg: float
    window_damage_total: float
    window_cycles: float
    leg_cycles: float
    peak_stress_mpa: float
    peak_vibration_stress_mpa: float
    max_stress_range_mpa: float
    duration_s: float

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "LegDamage",
            "leg_name": self.leg_name,
            "damage_total": float(self.damage_total),
            "damage_static": float(self.damage_static),
            "damage_vibration_only": float(self.damage_vibration_only),
            "ice_vibration_fraction": float(self.ice_vibration_fraction),
            "scale_to_leg": float(self.scale_to_leg),
            "window_damage_total": float(self.window_damage_total),
            "window_cycles": float(self.window_cycles),
            "leg_cycles": float(self.leg_cycles),
            "peak_stress_mpa": float(self.peak_stress_mpa),
            "peak_vibration_stress_mpa": float(self.peak_vibration_stress_mpa),
            "max_stress_range_mpa": float(self.max_stress_range_mpa),
            "duration_s": float(self.duration_s),
        }


def leg_damage(
    leg: VoyageLeg,
    hull: HullParameters,
    girder: HullGirder,
    vibration: HullGirderVibration,
    sn_curve: Any,
    history: Optional[LegStressHistory] = None,
) -> LegDamage:
    """Miner damage of one leg (window damage scaled to the leg duration).

    **Damage scaling.**  Ice loading is statistically stationary *within* a leg,
    so the damage rate of the representative window is applied to the whole leg:
    ``D_leg = D_window * (T_leg / T_window)``.  This is the standard
    representative-block assumption of fatigue design (constant damage rate),
    and it is why the leg duration, not the window duration, enters the voyage
    damage.  The assumption fails only if the ice condition changes faster than
    the window; legs are therefore defined per ice condition.
    """
    hist = history if history is not None else leg_stress_history(leg, hull, girder, vibration)
    block_total = history_damage(hist.stress_total_mpa, sn_curve)
    block_static = history_damage(hist.stress_static_mpa, sn_curve)
    block_vib = history_damage(hist.stress_vibration_mpa, sn_curve)
    scale = leg.scale_to_leg
    d_total = block_total.damage * scale
    d_static = block_static.damage * scale
    d_vib = block_vib.damage * scale
    fraction = (d_total - d_static) / d_total if d_total > 0.0 else 0.0
    return LegDamage(
        leg_name=leg.name,
        damage_total=float(d_total),
        damage_static=float(d_static),
        damage_vibration_only=float(d_vib),
        ice_vibration_fraction=float(fraction),
        scale_to_leg=float(scale),
        window_damage_total=float(block_total.damage),
        window_cycles=float(block_total.n_cycles),
        leg_cycles=float(block_total.n_cycles * scale),
        peak_stress_mpa=hist.peak_total_mpa,
        peak_vibration_stress_mpa=hist.peak_vibration_mpa,
        max_stress_range_mpa=float(block_total.max_range_mpa),
        duration_s=float(leg.duration_s),
    )


@dataclass(frozen=True)
class VoyageProfile:
    """A voyage made of consecutive ice legs plus its calendar duration."""

    name: str
    legs: Tuple[VoyageLeg, ...]
    voyage_duration_days: float = 12.0
    description: str = ""

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("a voyage profile needs at least one leg")
        if not (self.voyage_duration_days > 0.0):
            raise ValueError("voyage_duration_days must be positive")

    @property
    def total_duration_s(self) -> float:
        return float(sum(leg.duration_s for leg in self.legs))

    @property
    def ice_duration_s(self) -> float:
        return self.total_duration_s

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "VoyageProfile",
            "name": self.name,
            "voyage_duration_days": float(self.voyage_duration_days),
            "ice_duration_s": self.total_duration_s,
            "ice_duration_h": self.total_duration_s / 3600.0,
            "legs": [leg.to_json_dict() for leg in self.legs],
            "description": self.description,
        }


@dataclass(frozen=True)
class VoyageDamage:
    """Whole-voyage Miner damage and its decomposition.

    ``ice_vibration_fraction = (D_total - D_quasi_static) / D_total`` is the
    share of the accumulated damage caused by the ice-induced vibration
    residual.  It is strictly inside ``(0, 1)`` for the reference voyage; it can
    turn negative only when the excitation sits *above* the fundamental mode,
    where the modal residual cancels part of the quasi-static stress.
    """

    profile_name: str
    damage_total: float
    damage_static: float
    damage_vibration_only: float
    ice_vibration_fraction: float
    per_leg: Tuple[LegDamage, ...]
    max_single_window_damage: float
    peak_stress_mpa: float
    ice_duration_s: float

    @property
    def worst_leg(self) -> LegDamage:
        return max(self.per_leg, key=lambda d: d.damage_total)

    @property
    def dominates_single_window(self) -> bool:
        """``D_voyage >= D_single_window`` (the voyage contains the window)."""
        return self.damage_total >= self.max_single_window_damage

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "VoyageDamage",
            "profile_name": self.profile_name,
            "damage_total": float(self.damage_total),
            "damage_static": float(self.damage_static),
            "damage_vibration_only": float(self.damage_vibration_only),
            "ice_vibration_fraction": float(self.ice_vibration_fraction),
            "max_single_window_damage": float(self.max_single_window_damage),
            "dominates_single_window": bool(self.dominates_single_window),
            "peak_stress_mpa": float(self.peak_stress_mpa),
            "ice_duration_s": float(self.ice_duration_s),
            "worst_leg": self.worst_leg.leg_name,
            "per_leg": [d.to_json_dict() for d in self.per_leg],
        }


def voyage_damage(
    profile: VoyageProfile,
    hull: HullParameters,
    girder: HullGirder,
    vibration: HullGirderVibration,
    sn_curve: Any,
) -> VoyageDamage:
    """``D_voyage = sum_legs D_leg`` with the ice-vibration share resolved."""
    leg_damages: List[LegDamage] = []
    for leg in profile.legs:
        leg_damages.append(leg_damage(leg, hull, girder, vibration, sn_curve))
    d_total = float(sum(d.damage_total for d in leg_damages))
    d_static = float(sum(d.damage_static for d in leg_damages))
    d_vib = float(sum(d.damage_vibration_only for d in leg_damages))
    fraction = (d_total - d_static) / d_total if d_total > 0.0 else 0.0
    return VoyageDamage(
        profile_name=profile.name,
        damage_total=d_total,
        damage_static=d_static,
        damage_vibration_only=d_vib,
        ice_vibration_fraction=float(fraction),
        per_leg=tuple(leg_damages),
        max_single_window_damage=float(max(d.window_damage_total for d in leg_damages)),
        peak_stress_mpa=float(max(d.peak_stress_mpa for d in leg_damages)),
        ice_duration_s=float(profile.total_duration_s),
    )


# ===========================================================================
# 7. Accumulation over the service life and residual life prediction
# ===========================================================================


@dataclass(frozen=True)
class DamageTrajectory:
    """Damage accumulated voyage by voyage, with the calendar time axis."""

    voyage_index: np.ndarray
    damage: np.ndarray
    time_days: np.ndarray
    voyage_durations_days: np.ndarray

    @property
    def is_monotone(self) -> bool:
        return bool(np.all(np.diff(self.damage) >= -1.0e-15))

    @property
    def final_damage(self) -> float:
        return float(self.damage[-1]) if self.damage.size else 0.0

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "DamageTrajectory",
            "n_voyages": int(self.voyage_index.size - 1),
            "final_damage": self.final_damage,
            "final_time_days": float(self.time_days[-1]) if self.time_days.size else 0.0,
            "is_monotone": self.is_monotone,
        }


def accumulate_voyages(
    damage_per_voyage: float,
    n_voyages: int,
    severity_sequence: Optional[Sequence[float]] = None,
    voyage_duration_days: float = 12.0,
) -> DamageTrajectory:
    """Cumulative damage over ``n_voyages`` voyages.

    ``severity_sequence`` scales the reference per-voyage damage voyage by
    voyage (route/season severity profile); without it every voyage is identical.
    Damage never decreases, because every increment is non-negative - the
    structural content of criterion C4.
    """
    if damage_per_voyage < 0.0:
        raise ValueError("damage_per_voyage must be non-negative")
    n = int(n_voyages)
    if n < 0:
        raise ValueError("n_voyages must be non-negative")
    if severity_sequence is None:
        severities = np.ones(n, dtype=float)
    else:
        severities = np.asarray(severity_sequence, dtype=float)
        if severities.size < n:
            raise ValueError(f"severity_sequence has {severities.size} entries, {n} voyages requested")
        severities = severities[:n]
        if np.any(severities < 0.0):
            raise ValueError("severity_sequence must be non-negative")
    increments = float(damage_per_voyage) * severities
    damage = np.concatenate(([0.0], np.cumsum(increments)))
    index = np.arange(n + 1)
    time_days = index.astype(float) * float(voyage_duration_days)
    return DamageTrajectory(
        voyage_index=index,
        damage=damage,
        time_days=time_days,
        voyage_durations_days=np.full(n, float(voyage_duration_days)),
    )


def voyages_to_failure(
    damage_per_voyage: float,
    severity_sequence: Optional[Sequence[float]] = None,
    damage_critical: float = 1.0,
    max_voyages: int = 1_000_000,
) -> int:
    """Number of voyages until ``D >= damage_critical`` (Miner failure)."""
    if damage_per_voyage <= 0.0:
        return max_voyages
    trajectory = accumulate_voyages(
        damage_per_voyage,
        n_voyages=int(min(max_voyages, max(1, int(math.ceil(damage_critical / damage_per_voyage)) + 2))),
        severity_sequence=severity_sequence,
    )
    hits = np.nonzero(trajectory.damage >= damage_critical)[0]
    return int(hits[0]) if hits.size else int(max_voyages)


@dataclass(frozen=True)
class RemainingLife:
    """Residual life implied by the accumulated damage (linear Miner rule)."""

    damage_used: float
    damage_per_voyage: float
    damage_critical: float
    total_voyages_to_failure: float
    fraction_life_used: float
    remaining_voyages: float
    remaining_operating_days: float
    remaining_operating_years: float

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "RemainingLife",
            "damage_used": float(self.damage_used),
            "damage_per_voyage": float(self.damage_per_voyage),
            "damage_critical": float(self.damage_critical),
            "total_voyages_to_failure": float(self.total_voyages_to_failure),
            "fraction_life_used": float(self.fraction_life_used),
            "remaining_voyages": float(self.remaining_voyages),
            "remaining_operating_days": float(self.remaining_operating_days),
            "remaining_operating_years": float(self.remaining_operating_years),
        }


def remaining_life(
    damage_used: float,
    damage_per_voyage: float,
    voyage_duration_days: float,
    damage_critical: float = 1.0,
) -> RemainingLife:
    """``N_remaining = (1 - D) / D_voyage`` (same route severity each voyage).

    Limits are exact by construction: ``D = 0`` gives ``N_remaining =
    N_total = 1 / D_voyage`` and ``D = D_critical`` gives ``N_remaining = 0``.
    Negative remaining life (over-damaged structure) is clipped to zero and the
    fraction of life used is reported above 1 in that case.
    """
    if damage_per_voyage <= 0.0:
        raise ValueError("damage_per_voyage must be positive for a finite life")
    if damage_critical <= 0.0:
        raise ValueError("damage_critical must be positive")
    d = float(damage_used)
    if d < 0.0:
        raise ValueError("damage_used must be non-negative")
    total = float(damage_critical) / float(damage_per_voyage)
    remaining = max((float(damage_critical) - d), 0.0) / float(damage_per_voyage)
    return RemainingLife(
        damage_used=d,
        damage_per_voyage=float(damage_per_voyage),
        damage_critical=float(damage_critical),
        total_voyages_to_failure=total,
        fraction_life_used=d / float(damage_critical),
        remaining_voyages=float(remaining),
        remaining_operating_days=float(remaining * float(voyage_duration_days)),
        remaining_operating_years=float(remaining * float(voyage_duration_days) / 365.25),
    )


# ===========================================================================
# 8. Reference case (one route, one hull) and life summary
# ===========================================================================


def reference_profile() -> VoyageProfile:
    """Reference Arctic route: five ice legs of increasing severity.

    The legs are ordered as they occur along the route and each carries its own
    ice condition (pulse amplitude, pulse frequency ``U/s``, pulse width and
    ridge tail).  ``lock_in_channel`` deliberately places the pulse frequency
    close to the hull-girder first mode (``f / f1 ~ 0.9``) to expose the
    ice-induced vibration contribution.
    """
    legs = (
        VoyageLeg(
            name="level_ice_transit",
            duration_s=2.0 * 3600.0,
            peak_force_n=2.0e6,
            ice_frequency_hz=0.30,
            pulse_duration_s=0.35,
            seed=101,
            amplitude_cv=0.35,
            description="continuous level-ice crushing at U/s = 0.30 Hz",
        ),
        VoyageLeg(
            name="brash_ice_channel",
            duration_s=1.0 * 3600.0,
            peak_force_n=1.0e6,
            ice_frequency_hz=0.45,
            pulse_duration_s=0.30,
            seed=202,
            amplitude_cv=0.40,
            description="brash-ice channel, high pulse rate, low amplitude",
        ),
        VoyageLeg(
            name="ridge_ramming",
            duration_s=0.5 * 3600.0,
            peak_force_n=3.5e6,
            ice_frequency_hz=0.20,
            pulse_duration_s=0.25,
            seed=303,
            amplitude_cv=0.30,
            ridge_probability=0.20,
            ridge_factor=1.8,
            description="ridge ramming: sparse high-amplitude impacts",
        ),
        VoyageLeg(
            name="lock_in_channel",
            duration_s=1.5 * 3600.0,
            peak_force_n=1.6e6,
            ice_frequency_hz=0.68,
            pulse_duration_s=0.30,
            seed=404,
            amplitude_cv=0.30,
            description="pulse frequency close to f1 -> ice-induced vibration dominant",
        ),
        VoyageLeg(
            name="open_water_background",
            duration_s=5.0 * 3600.0,
            peak_force_n=0.35e6,
            ice_frequency_hz=0.12,
            pulse_duration_s=0.50,
            seed=505,
            amplitude_cv=0.45,
            description="residual open-water / wave-induced load at the same station",
        ),
    )
    return VoyageProfile(
        name="arctic_round_trip_reference",
        legs=legs,
        voyage_duration_days=12.0,
        description="one Arctic round trip: 9.5 h in ice + open water, 12 calendar days",
    )


def reference_hull() -> HullParameters:
    """Reference ice-class tanker (L = 150 m, Delta ~ 26 000 t)."""
    length, breadth, draught, cb = 150.0, 25.0, 9.0, 0.75
    displacement = float(cb * length * breadth * draught * RHO_SEA_WATER)
    return HullParameters(
        length_m=length,
        breadth_m=breadth,
        draught_m=draught,
        block_coefficient=cb,
        displacement_kg=displacement,
        name="reference-ice-class-tanker-150m",
    )


@dataclass(frozen=True)
class VoyageFatigueCase:
    """Complete voyage-fatigue case: hull + girder + route + S-N curve."""

    hull: HullParameters
    girder: HullGirder
    vibration: HullGirderVibration
    profile: VoyageProfile
    curve: Any
    n_voyages: int = 10
    severity_sequence: Optional[Sequence[float]] = None
    _cache: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    # -- cached evaluations -------------------------------------------------
    def voyage_damage(self) -> VoyageDamage:
        cached = self._cache.get("voyage_damage")
        if cached is None:
            cached = voyage_damage(self.profile, self.hull, self.girder, self.vibration, self.curve)
            self._cache["voyage_damage"] = cached
        return cached

    def leg_histories(self) -> Tuple[LegStressHistory, ...]:
        cached = self._cache.get("histories")
        if cached is None:
            cached = tuple(
                leg_stress_history(leg, self.hull, self.girder, self.vibration) for leg in self.profile.legs
            )
            self._cache["histories"] = cached
        return cached

    def trajectory(self, n_voyages: Optional[int] = None) -> DamageTrajectory:
        n = int(self.n_voyages if n_voyages is None else n_voyages)
        key = f"trajectory:{n}"
        cached = self._cache.get(key)
        if cached is None:
            cached = accumulate_voyages(
                self.voyage_damage().damage_total,
                n_voyages=n,
                severity_sequence=self.severity_sequence,
                voyage_duration_days=self.profile.voyage_duration_days,
            )
            self._cache[key] = cached
        return cached

    def life(self, voyages_completed: int) -> RemainingLife:
        return remaining_life(
            damage_used=float(self.trajectory(voyages_completed).damage[-1]),
            damage_per_voyage=self.voyage_damage().damage_total,
            voyage_duration_days=self.profile.voyage_duration_days,
        )

    def life_summary(self) -> Dict[str, Any]:
        vd = self.voyage_damage()
        traj = self.trajectory()
        n_predicted = voyages_to_failure(vd.damage_total)
        life0 = remaining_life(0.0, vd.damage_total, self.profile.voyage_duration_days)
        return {
            "voyage_damage": vd.to_json_dict(),
            "trajectory": traj.to_json_dict(),
            "voyages_to_failure_simulated": int(n_predicted),
            "voyages_to_failure_analytic": float(life0.total_voyages_to_failure),
            "remaining_life_at_zero_damage": life0.to_json_dict(),
            "remaining_life_at_half_life": remaining_life(
                0.5 * vd.damage_total * life0.total_voyages_to_failure,
                vd.damage_total,
                self.profile.voyage_duration_days,
            ).to_json_dict(),
        }

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "VoyageFatigueCase",
            "hull": self.hull.to_json_dict(),
            "girder": self.girder.to_json_dict(),
            "vibration": {
                "frequency_hz": float(self.vibration.frequency_hz),
                "damping_ratio": float(self.vibration.damping_ratio),
                "modal_mass_kg": float(self.vibration.modal_mass_kg),
                "modal_stiffness_nm2": float(self.vibration.modal_stiffness),
                "curvature_midship_per_m": float(self.vibration.curvature_midship_per_m),
                "bow_ordinate": float(self.vibration.bow_ordinate),
                "stress_per_modal_displacement_mpa": float(self.vibration.stress_per_modal_displacement_mpa()),
            },
            "profile": self.profile.to_json_dict(),
            "sn_curve": self.curve.to_json_dict()
            if hasattr(self.curve, "to_json_dict")
            else {"class": type(self.curve).__name__, "name": getattr(self.curve, "name", "")},
            "n_voyages": int(self.n_voyages),
            "severity_sequence": None
            if self.severity_sequence is None
            else [float(v) for v in self.severity_sequence],
        }


def build_reference_case(
    bending_stiffness_nm2: float = 4.0e12,
    damping_ratio: float = 0.03,
    n_voyages: int = 10,
    curve: Optional[Any] = None,
    severity_sequence: Optional[Sequence[float]] = None,
) -> VoyageFatigueCase:
    """Deterministic reference case used by the benchmark and the tests.

    ``EI = 4.0e12 N m^2`` is a plausible midship hull-girder stiffness for a
    150 m double-hull ice-class tanker (``I ~ 19.4 m^4`` at ``E = 206 GPa``,
    deck section modulus ``Z ~ 3.2 m^3``), giving a first vertical bending mode
    near 0.75 Hz.
    """
    hull = reference_hull()
    girder = HullGirder.from_hull(hull, bending_stiffness_nm2=bending_stiffness_nm2)
    vibration = build_hull_girder_vibration(hull, girder, damping_ratio=damping_ratio)
    sn = curve if curve is not None else PowerLawSN.from_dnv("D", "seawater_cp")
    return VoyageFatigueCase(
        hull=hull,
        girder=girder,
        vibration=vibration,
        profile=reference_profile(),
        curve=sn,
        n_voyages=n_voyages,
        severity_sequence=severity_sequence,
    )
