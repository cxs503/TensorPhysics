"""IIW / DNV-RP-C203 S-N curves, log-log fitting and weld-test scatter model.

Design-curve definition used here (IIW Recommendations 2016 / DNV-RP-C203 2016)
------------------------------------------------------------------------------
A weld-detail S-N curve is a bi-linear power law in log-log space

    log10 N = log10 A1 - m1 log10 (dsigma)      for N <= N_knee
    log10 N = log10 A2 - m2 log10 (dsigma)      for N >  N_knee

anchored at the *characteristic* stress range ``dsigma_c`` at ``N = 2e6`` cycles
(the "FAT" number of the class, e.g. FAT90 -> dsigma_c = 90 MPa), with
``m1 = 3`` and, beyond the knee, ``m2 = 5`` (DNV-RP-C203 convention, knee at
``1e7`` cycles).  Hence

    log10 A1 = log10(dsigma_c) + m1 * log10(2e6)

and the design curve corresponds to a survival probability of ~97.7 %
(mean minus two standard deviations of ``log10 N``).

The verification range required by the acceptance criteria is
``1e5 <= N <= 1e7``, i.e. entirely inside the first (single-slope) segment, so
the knee convention does not influence any verified number.

Scatter model (declared, used only to generate *simulated* test series)
-----------------------------------------------------------------------
``sigma_logN = 0.2`` is the reference standard deviation of ``log10 N`` for
welded joints quoted by IIW for as-welded steel details; the design curve is the
mean curve shifted by ``k * sigma_logN`` with ``k = 2`` (97.7 % survival).  This
module never pretends that the simulated series are experiments; the fatigue
correlation checks in the acceptance report are explicitly tagged
``simulated`` where this model is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: IIW characteristic FAT classes (MPa at 2e6 cycles), steel, as-welded.
IIW_FAT_CLASSES: Tuple[float, ...] = (36.0, 40.0, 45.0, 50.0, 56.0, 63.0, 71.0,
                                      80.0, 90.0, 100.0, 112.0, 125.0, 140.0, 160.0)

#: Reference cycle number at which the FAT class is defined.
REFERENCE_CYCLES: float = 2.0e6

#: DNV-RP-C203 curve D in air: m1 = 3, log10 a1 = 12.164 -> dsigma_c = 90 MPa.
#: Coincides with IIW FAT90 at 2e6 cycles (cross-standard anchor).
DNV_D_AIR_LOG_A1: float = 12.164
DNV_D_AIR_M1: float = 3.0


def log_a1_from_fat(fat_class: float, m1: float = 3.0,
                    reference_cycles: float = REFERENCE_CYCLES) -> float:
    """Return ``log10 A1`` of the first S-N segment for a given FAT class."""
    if fat_class <= 0:
        raise ValueError("fat_class must be positive")
    return float(np.log10(reference_cycles) + m1 * np.log10(fat_class))


def fat_from_log_a1(log_a1: float, m1: float = 3.0,
                    reference_cycles: float = REFERENCE_CYCLES) -> float:
    """Inverse of :func:`log_a1_from_fat` (stress range at the reference cycles)."""
    return float(10.0 ** ((log_a1 - np.log10(reference_cycles)) / m1))


@dataclass(frozen=True)
class IIWCurve:
    """Bi-linear IIW / DNV S-N design curve for one FAT class."""

    fat_class: float
    m1: float = 3.0
    m2: float = 5.0
    knee_cycles: float = 1.0e7

    def __post_init__(self) -> None:
        if self.fat_class <= 0:
            raise ValueError("fat_class must be positive")
        if self.m1 <= 0 or self.m2 <= 0:
            raise ValueError("slopes must be positive")
        if self.knee_cycles <= REFERENCE_CYCLES:
            raise ValueError("knee must lie above the reference cycles")

    # ---- parameters -------------------------------------------------------
    @property
    def log_a1(self) -> float:
        return log_a1_from_fat(self.fat_class, self.m1)

    @property
    def stress_at_knee(self) -> float:
        return 10.0 ** ((self.log_a1 - np.log10(self.knee_cycles)) / self.m1)

    @property
    def log_a2(self) -> float:
        return float(np.log10(self.knee_cycles) + self.m2 * np.log10(self.stress_at_knee))

    @property
    def stress_c(self) -> float:
        """Characteristic stress range at 2e6 cycles (the FAT number)."""
        return self.fat_class

    # ---- evaluation -------------------------------------------------------
    def life(self, stress_range: float) -> float:
        """Cycles to failure for a constant-amplitude stress range (MPa)."""
        s = float(np.asarray(stress_range, dtype=float))
        if s <= 0.0:
            return float("inf")
        s_knee = self.stress_at_knee
        if s >= s_knee:
            return float(10.0 ** (self.log_a1 - self.m1 * np.log10(s)))
        return float(10.0 ** (self.log_a2 - self.m2 * np.log10(s)))

    # alias
    n_to_failure = life

    def stress_for_life(self, cycles) -> float:
        """Stress range(s) corresponding to given life/lives (scalar or array)."""
        n = np.asarray(cycles, dtype=float)
        if np.any(n <= 0):
            raise ValueError("cycles must be positive")
        la1, la2 = self.log_a1, self.log_a2
        val = np.where(n <= self.knee_cycles,
                       10.0 ** ((la1 - np.log10(n)) / self.m1),
                       10.0 ** ((la2 - np.log10(n)) / self.m2))
        return float(val) if val.ndim == 0 else val

    def sample(self, n_min: float = 1e5, n_max: float = 1e7,
               n_points: int = 41) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the curve as ``(stress_range, life)`` pairs over a life band."""
        n = np.logspace(np.log10(n_min), np.log10(n_max), n_points)
        return self.stress_for_life(n), n


def fatigue_curve(fat_class: float = 90.0, **kw) -> IIWCurve:
    """Convenience constructor (e.g. ``fatigue_curve(90)`` -> IIW FAT90)."""
    return IIWCurve(fat_class=fat_class, **kw)


# ============================================================================
# log-log fitting
# ============================================================================

@dataclass
class SNLogLogFit:
    """Result of a ``log10 N = log10 A - m log10 (dsigma)`` least-squares fit."""

    slope_m: float          # fatigue exponent m (positive number)
    log_a: float            # intercept log10 A
    r_squared: float        # coefficient of determination in log-log space
    n_points: int
    residual_sd: float      # standard deviation of log10 N residuals
    fat_at_2e6: float       # stress range implied at 2e6 cycles (a "FIT" number)

    def life(self, stress_range: float) -> float:
        s = np.asarray(stress_range, dtype=float)
        return 10.0 ** (self.log_a - self.slope_m * np.log10(s))

    def r2(self) -> float:
        return self.r_squared

    def as_dict(self) -> Dict[str, float]:
        return {
            "slope_m": self.slope_m,
            "log_a": self.log_a,
            "r_squared": self.r_squared,
            "n_points": int(self.n_points),
            "residual_sd_log_n": self.residual_sd,
            "fat_at_2e6": self.fat_at_2e6,
        }


def loglog_fit(stress_range: Sequence[float], life: Sequence[float]) -> SNLogLogFit:
    """Least-squares fit of ``log10 N = log10 A - m log10 (dsigma)``.

    The sign convention keeps ``m`` positive (a physically meaningful fatigue
    exponent).  ``R^2`` is computed in log-log space, which is the space in which
    the S-N correlation is linear.
    """
    s = np.asarray(stress_range, dtype=float).ravel()
    n = np.asarray(life, dtype=float).ravel()
    if s.size != n.size:
        raise ValueError("stress_range and life must have the same length")
    if s.size < 3:
        raise ValueError("at least three points are required for a fit")
    mask = (s > 0) & (n > 0) & np.isfinite(s) & np.isfinite(n)
    x = np.log10(s[mask])
    y = np.log10(n[mask])
    if x.size < 3:
        raise ValueError("fewer than three valid (positive, finite) points")

    # y = c0 + c1 x with c1 = -m
    c1, c0 = np.polyfit(x, y, 1)
    y_hat = c0 + c1 * x
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - y_hat) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = max(x.size - 2, 1)
    resid_sd = float(np.sqrt(ss_res / dof))

    return SNLogLogFit(
        slope_m=float(-c1),
        log_a=float(c0),
        r_squared=float(r2),
        n_points=int(x.size),
        residual_sd=resid_sd,
        fat_at_2e6=fat_from_log_a1(float(c0), m1=float(-c1)),
    )


def reference_fit(curve: IIWCurve, n_min: float = 1e5, n_max: float = 1e7,
                  n_points: int = 41) -> SNLogLogFit:
    """Fit the design curve samples (used to verify the fitting machinery)."""
    stress, life = curve.sample(n_min=n_min, n_max=n_max, n_points=n_points)
    return loglog_fit(stress, life)


# ============================================================================
# Declared scatter model (simulated test series -- NOT experiments)
# ============================================================================

def design_to_mean_factor(sigma_log_n: float = 0.2, k: float = 2.0) -> float:
    """Factor between the probability-of-failure design curve and the mean curve.

    IIW: the design curve is the mean minus ``k`` standard deviations of
    ``log10 N``; with ``sigma_log_n = 0.2`` and ``k = 2`` (97.7 % survival) the
    mean life is ``10**(k * sigma_log_n) = 2.51`` times the design life.
    """
    return float(10.0 ** (k * sigma_log_n))


def simulate_test_series(curve: IIWCurve,
                         stress_levels: Sequence[float],
                         specimens_per_level: int = 12,
                         sigma_log_n: float = 0.2,
                         seed: int = 20260909,
                         on_mean_curve: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a constant-amplitude weld fatigue test series.

    The simulated mean life is the design-curve life multiplied by
    :func:`design_to_mean_factor` (i.e. ``on_mean_curve=True``) or taken directly
    from the design curve (``on_mean_curve=False``); individual specimens are
    lognormal in ``N`` with standard deviation ``sigma_log_n`` in ``log10 N``.

    Returns
    -------
    (stress_range, life) arrays of length ``len(stress_levels)*specimens_per_level``
    """
    rng = np.random.default_rng(seed)
    shift = np.log10(design_to_mean_factor(sigma_log_n)) if on_mean_curve else 0.0
    s_out, n_out = [], []
    for s_level in stress_levels:
        base = curve.life(float(s_level))
        log_n = np.log10(base) + shift + rng.normal(0.0, sigma_log_n, size=specimens_per_level)
        s_out.append(np.full(specimens_per_level, float(s_level)))
        n_out.append(10.0 ** log_n)
    return np.concatenate(s_out), np.concatenate(n_out)


# ============================================================================
# Cross-standard check helpers
# ============================================================================

def dnv_curve_d_air(fat_class: Optional[float] = None) -> IIWCurve:
    """DNV-RP-C203 curve D in air expressed as an :class:`IIWCurve`.

    Curve D in air has ``log10 a1 = 12.164`` and ``m1 = 3``, which corresponds to
    ``dsigma_c = 90 MPa`` at 2e6 cycles, i.e. the same anchor as IIW FAT90.
    The equivalence is verified in the acceptance suite from ``log10 a1`` rather
    than assumed.
    """
    inferred = fat_from_log_a1(DNV_D_AIR_LOG_A1, DNV_D_AIR_M1)
    if fat_class is not None:
        inferred = fat_class
    return IIWCurve(fat_class=inferred, m1=DNV_D_AIR_M1)


def standard_cross_check() -> Dict[str, float]:
    """Compare the IIW FAT90 anchor with DNV-RP-C203 curve D in air."""
    iiw = IIWCurve(fat_class=90.0, m1=3.0)
    dnv_fat = fat_from_log_a1(DNV_D_AIR_LOG_A1, DNV_D_AIR_M1)
    return {
        "iiw_fat90_log_a1": iiw.log_a1,
        "dnv_d_air_log_a1": DNV_D_AIR_LOG_A1,
        "dnv_d_air_dsigma_c": dnv_fat,
        "fat_relative_error_pct": abs(dnv_fat - 90.0) / 90.0 * 100.0,
    }