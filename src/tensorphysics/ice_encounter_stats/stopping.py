"""Closed-form required duration and the reusable stopping rule.

For a stationary record with mean ``mu``, standard deviation ``sigma`` and
integral time scale ``tau_c``, the variance of a window mean over a duration
``T >> tau_c`` is ``2 sigma^2 tau_c / T`` (Bartlett), i.e. the relative standard
error is ``sigma/(mu sqrt(N_eff))`` with ``N_eff = T/(2 tau_c)``.  Requiring a
relative standard error ``eps`` gives the closed form

    T_req = 2 tau_c (sigma/mu)^2 / eps^2 = N_eff_req * 2 tau_c            (5)
    N_eff_req = (CV/eps)^2                                                (6)

Equation (5) is a *one-sigma* statement: a record of length ``T_req`` has a
relative standard error of ``eps``, so a symmetric band of half-width ``eps``
contains the true mean with probability ``1 - 2 Phi(-1)`` ~ 68% when the window
mean is Gaussian, less when its tails are heavier.  A stopping rule that has to
guarantee a band with a stated confidence therefore needs the quantile form

    T_stop = 2 tau_c (CV * zeta / eps)^2,   zeta = q_{1-alpha}(|dmu|/mu)/eps  (7)

where ``zeta`` is the multiplier between the one-sigma scale and the
``1-alpha`` quantile of the relative error of the window mean.  ``zeta = 1.645``
for a Gaussian at 90%, ``1.96`` at 95%; for a spiky ice-resistance record it is
*measured* on an independent calibration set (see
:func:`tail_factor_from_quantile`) and then frozen in the manifest, so the
evaluation set never influences it.

Both forms live in :class:`StoppingRule`; ``tail_factor=1.0`` reproduces the
pure closed form (5), anything larger is the confidence-qualified rule (7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import ConfigError


# --------------------------------------------------------------------------- #
# Gaussian quantile (no scipy dependency)
# --------------------------------------------------------------------------- #
def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |err|<1e-9)."""
    if not (0.0 < p < 1.0):
        raise ConfigError("normal_ppf needs 0 < p < 1")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def tail_factor_from_quantile(relative_errors: Sequence[float], eps: float,
                              alpha: float) -> float:
    """``zeta = q_{1-alpha}(|error|) / eps`` measured on a calibration set.

    The calibration set must be produced at the one-sigma duration ``T_req``;
    the returned factor is then used to lengthen the production run so that the
    ``(1-alpha)`` quantile of the relative error lands exactly on ``eps``.
    """
    err = np.abs(np.asarray(relative_errors, dtype=float))
    err = err[np.isfinite(err)]
    if err.size < 20:
        raise ConfigError("tail-factor calibration needs at least 20 cases")
    if not (0.0 < alpha < 1.0) or eps <= 0.0:
        raise ConfigError("needs 0 < alpha < 1 and eps > 0")
    return float(np.quantile(err, 1.0 - alpha) / eps)


# --------------------------------------------------------------------------- #
# pilot / decision objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PilotStatistics:
    """Pilot estimates needed by the rule (all from a short record)."""

    duration_s: float
    mean: float
    std: float
    tau_c: float

    @property
    def cv(self) -> float:
        return float(self.std / self.mean) if self.mean > 0.0 else float("nan")

    @classmethod
    def from_record(cls, record, tau_c: float) -> "PilotStatistics":
        return cls(duration_s=record.duration_s, mean=record.mean,
                   std=record.std, tau_c=float(tau_c))


@dataclass(frozen=True)
class StopDecision:
    """Outcome of applying the stopping rule to a pilot."""

    eps: float
    tail_factor: float
    required_time_s: float
    required_effective_samples: float
    achieved_relative_standard_error: float
    achieved_quantile_error: float
    converged: bool
    remaining_time_s: float

    def to_dict(self) -> dict:
        return {
            "eps": float(self.eps),
            "tail_factor": float(self.tail_factor),
            "required_time_s": float(self.required_time_s),
            "required_effective_samples": float(self.required_effective_samples),
            "achieved_relative_standard_error": float(self.achieved_relative_standard_error),
            "achieved_quantile_error": float(self.achieved_quantile_error),
            "converged": bool(self.converged),
            "remaining_time_s": float(self.remaining_time_s),
        }


@dataclass(frozen=True)
class StoppingRule:
    """``T_stop = 2 tau_c (CV * tail_factor / eps)^2``.

    ``tail_factor = 1.0`` is the pure one-sigma closed form of equation (5);
    ``tail_factor = normal_ppf(1 - alpha/2)`` reproduces the Gaussian confidence
    rule; any measured value from :func:`tail_factor_from_quantile` is the
    calibration-corrected rule of equation (7).
    """

    eps: float
    tail_factor: float = 1.0
    max_duration_s: Optional[float] = None

    def __post_init__(self) -> None:
        if not (0.0 < self.eps < 1.0):
            raise ConfigError("eps must be a relative standard error in (0, 1)")
        if self.tail_factor <= 0.0:
            raise ConfigError("tail_factor must be positive")

    @classmethod
    def gaussian(cls, eps: float, confidence: float = 0.95,
                 max_duration_s: Optional[float] = None) -> "StoppingRule":
        if not (0.0 < confidence < 1.0):
            raise ConfigError("confidence must lie in (0, 1)")
        return cls(eps=eps, tail_factor=normal_ppf(0.5 + 0.5 * confidence),
                   max_duration_s=max_duration_s)

    # -- closed forms ------------------------------------------------------- #
    def required_effective_samples(self, cv: float) -> float:
        return float((cv * self.tail_factor / self.eps) ** 2)

    def required_time(self, cv: float, tau_c: float) -> float:
        if tau_c <= 0.0:
            raise ConfigError("tau_c must be positive")
        return float(2.0 * tau_c * (cv * self.tail_factor / self.eps) ** 2)

    def relative_standard_error(self, cv: float, tau_c: float, duration_s: float) -> float:
        """One-sigma relative standard error of the window mean."""
        if duration_s <= 0.0:
            raise ConfigError("duration must be positive")
        return float(cv * math.sqrt(2.0 * tau_c / duration_s))

    def quantile_error(self, cv: float, tau_c: float, duration_s: float) -> float:
        """(1-alpha) quantile of the relative error at ``duration_s``."""
        return float(self.tail_factor * self.relative_standard_error(cv, tau_c, duration_s))

    # -- rule application --------------------------------------------------- #
    def assess(self, pilot: PilotStatistics,
               elapsed_s: Optional[float] = None) -> StopDecision:
        """Decide whether a record of length ``elapsed_s`` (default: pilot) suffices."""
        cv = pilot.cv
        if not np.isfinite(cv) or cv <= 0.0:
            raise ConfigError("pilot statistics give a non-positive coefficient of variation")
        t_stop = self.required_time(cv, pilot.tau_c)
        if self.max_duration_s is not None:
            t_stop = min(t_stop, float(self.max_duration_s))
        t_now = float(pilot.duration_s if elapsed_s is None else elapsed_s)
        sem = self.relative_standard_error(cv, pilot.tau_c, t_now)
        return StopDecision(
            eps=self.eps,
            tail_factor=self.tail_factor,
            required_time_s=t_stop,
            required_effective_samples=self.required_effective_samples(cv),
            achieved_relative_standard_error=sem,
            achieved_quantile_error=self.tail_factor * sem,
            converged=bool(self.tail_factor * sem <= self.eps),
            remaining_time_s=float(max(t_stop - t_now, 0.0)),
        )

    def to_dict(self) -> dict:
        return {"eps": float(self.eps), "tail_factor": float(self.tail_factor),
                "max_duration_s": None if self.max_duration_s is None
                else float(self.max_duration_s)}


def rule_from_execution(block, eps_key: str = "eps",
                        tail_factor: Optional[float] = None) -> StoppingRule:
    """Build a rule from an ``execution.*`` block of the manifest."""
    if "eps" not in block and eps_key not in block:
        raise ConfigError("execution block has no eps entry")
    eps = float(block.get(eps_key, block.get("eps")))
    if tail_factor is None:
        tail_factor = float(block.get("tail_factor", 1.0))
    return StoppingRule(eps=eps, tail_factor=tail_factor,
                        max_duration_s=None if block.get("max_duration_s") is None
                        else float(block["max_duration_s"]))


__all__ = ["PilotStatistics", "StopDecision", "StoppingRule", "normal_ppf",
           "rule_from_execution", "tail_factor_from_quantile"]