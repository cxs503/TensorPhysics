"""Metric semantics for acceptance criteria: structured measurement model.

Why this module exists
----------------------
Historical diligence reports encode an acceptance criterion as a *string*, e.g.::

    | 关键量 | 基准 | 误差 | 是否 <3% |
    | ζ_eq  | 试验 0.04-0.06 | 偏差 ~60% | ⚠️ 部分 |

Three pathologies hide in that string and none of them is machine-detectable in the
string form:

* the reference is a *range* (0.04-0.06) with no priority rule  → relative error is
  ambiguous (56% against 0.04, 65% against the midpoint 0.05);
* the tolerance is *plural* (``<3%`` in the header, ``<10% 判据带`` in the cell) →
  a dual standard that lets a 6% deviation be reported as "within the 10% band" after
  being marked "not <3%";
* the reference magnitude goes to *zero* (``基准 = 精确 0``, ``平衡方程本身``) → the
  relative error is unbounded/undefined but the table still prints a number;
* the measured quantity *crosses zero* (``ΔV < 0``) → relative error is asymmetric.

This module promotes the criterion to a structure with explicit *metrology*:
``(y, y_ref, tau, metric, unit, condition)`` plus the auxiliaries needed to decide
whether the criterion is even *judgeable* (noise level ``sigma_noise`` and the
engineering scale ``scale_y``).

Verdicts are three-valued and **explicit**: ``PASS`` / ``FAIL`` / ``WITHHELD``.
There is deliberately no ``PARTIAL`` state: a criterion whose metrology degenerates
must be re-expressed by the adaptive family (:mod:`tensorphysics.metrology.criteria_semantics.adaptive`)
or reported as ``WITHHELD``.  "部分通过" is a measurement artefact, not a verdict.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any


class Metric(str, enum.Enum):
    """Error metric / 口径 of a criterion."""

    RELATIVE = "relative"      # |y - y_ref| / |y_ref|
    ABSOLUTE = "absolute"      # |y - y_ref| in `unit`
    LOG_RATIO = "log_ratio"    # |ln(y / y_ref)|  (requires y, y_ref > 0)
    SIGNED = "signed"          # (y - y_ref) in `unit`, sign preserved
    BOUND = "bound"            # one-sided engineering threshold: PASS iff y <= y_ref (+tau)


class Verdict(str, enum.Enum):
    """Three-valued verdict.  No PARTIAL: partial passing is a metrology bug."""

    PASS = "PASS"
    FAIL = "FAIL"
    WITHHELD = "WITHHELD"

    @property
    def is_judgeable(self) -> bool:
        return self is not Verdict.WITHHELD


class Degeneracy(str, enum.Enum):
    """Degeneracy taxonomy of a criterion's metrology."""

    NONE = "NONE"
    T1_REFERENCE_NEAR_ZERO = "T1_REFERENCE_NEAR_ZERO"
    T2_SIGN_CROSSING = "T2_SIGN_CROSSING"
    T3_MULTI_METRIC_NO_PRIORITY = "T3_MULTI_METRIC_NO_PRIORITY"


@dataclass(frozen=True)
class Measurement:
    """A single measured quantity with its metrological context.

    Parameters
    ----------
    y : float | None
        Measured value.  ``None`` means *not measured* → verdict WITHHELD.
    y_ref : float | None
        Reference value.  ``None`` means *no reference available* → WITHHELD
        (never a pass; that is the "no benchmark" pathology).
    scale_y : float | None
        Engineering scale of the quantity (e.g. ship length, nominal magnitude).
        Used by the magnitude-normalised remedy when ``|y_ref|`` is tiny.
    sigma_noise : float | None
        Noise floor of the measurement chain (same unit as ``y``).
    unit : str
        Engineering unit, for the absolute/signed metrics.  ``"1"`` for dimensionless.
    ref_range : tuple[float, float] | None
        Reference given as an interval with no priority rule (T3 evidence).
    quantity : str
        Human label, carried through to reports.
    """

    y: float | None
    y_ref: float | None
    scale_y: float | None = None
    sigma_noise: float | None = None
    unit: str = "1"
    ref_range: tuple[float, float] | None = None
    quantity: str = ""
    #: True when the source row states that there is no reference at all (empty cell,
    #: "无基准", "不适用", a bare tolerance, or an explicit WITHHELD)
    reference_absent: bool = False
    #: True when ``y_ref`` is a *unit placeholder* (the row reported a dimensionless
    #: residual against a self-referential reference such as "平衡方程本身"),
    #: not a physical reference magnitude
    reference_is_placeholder: bool = False

    def __post_init__(self) -> None:
        for name in ("y", "y_ref", "scale_y", "sigma_noise"):
            v = getattr(self, name)
            if v is not None and not isinstance(v, (int, float)):
                raise TypeError(f"{name} must be float|None, got {type(v).__name__}")
            if isinstance(v, float) and math.isnan(v):
                # NaN is neither a value nor 'missing': treat as missing but keep the flag.
                object.__setattr__(self, name, None)

    # -- derived quantities that are *always bounded* ---------------------------
    def effective_scale(self) -> float:
        """Bounded denominator for magnitude-normalised errors.

        Priority: explicit engineering scale → max(|y|, |y_ref|) → 1.0.
        Never zero, never infinite (this is the invariant the T1 remedy relies on).
        """
        if self.scale_y is not None and self.scale_y != 0.0:
            return abs(self.scale_y)
        mags = [abs(v) for v in (self.y, self.y_ref) if v is not None and v != 0.0]
        if mags:
            return max(mags)
        return 1.0


@dataclass(frozen=True)
class Criterion:
    """An acceptance criterion with explicit metric semantics.

    ``tau`` is expressed in the metric's own space: for RELATIVE it is a fraction
    (``0.03`` = 3 %), for ABSOLUTE/SIGNED in ``unit``, for LOG_RATIO as a ratio on
    ``ln`` (``0.03`` ≈ 3 % symmetric band).
    """

    measurement: Measurement
    tau: float
    metric: Metric = Metric.RELATIVE
    condition: str = ""
    name: str = ""
    #: alternative tolerances that coexist with ``tau`` and have no priority rule (T3 evidence)
    competing_taus: tuple[float, ...] = ()
    #: textual evidence of a sign-sensitive quantity (e.g. "ΔV<0")
    sign_sensitive: bool = False
    #: False when the tolerance is not expressed in the metric's unit space (absolute
    #: residual vs relative tolerance) -> the criterion is not judgeable, never a pass
    tau_available: bool = True
    #: the source row reports an error that is exactly zero (metric-invariant agreement)
    reported_error_is_zero: bool = False
    #: provenance
    source: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tau, (int, float)) or self.tau < 0:
            raise ValueError(f"tau must be a non-negative number, got {self.tau!r}")
        if isinstance(self.tau, float) and math.isnan(self.tau):
            raise ValueError("tau must not be NaN")
        if self.metric is Metric.LOG_RATIO and self.tau <= 0:
            raise ValueError("log-ratio tolerance must be > 0")

    @property
    def all_taus(self) -> tuple[float, ...]:
        return (self.tau,) + tuple(self.competing_taus)


@dataclass(frozen=True)
class Evaluation:
    """Result of evaluating one criterion under one metric family."""

    criterion: Criterion
    metric: Metric
    error: float | None            # error value in the metric's space (None if not computable)
    tau: float
    verdict: Verdict
    reason: str = ""
    degeneracy: Degeneracy = Degeneracy.NONE
    remedy: str = ""
    #: signed value for sign-flip bookkeeping (SIGNED only, else None)
    signed_error: float | None = None
    #: relative error purely for reporting; may be None (never inf/nan by construction)
    relative_error: float | None = None

    @property
    def ratio(self) -> float | None:
        """error / tau — the "how many tolerances off" number used by replay stats."""
        if self.error is None or self.tau <= 0:
            return None
        return self.error / self.tau

    def as_dict(self) -> dict[str, Any]:
        m = self.criterion.measurement
        return {
            "name": self.criterion.name or m.quantity,
            "y": m.y,
            "y_ref": m.y_ref,
            "unit": m.unit,
            "tau": self.tau,
            "metric": self.metric.value,
            "error": self.error,
            "relative_error": self.relative_error,
            "signed_error": self.signed_error,
            "verdict": self.verdict.value,
            "degeneracy": self.degeneracy.value,
            "remedy": self.remedy,
            "reason": self.reason,
        }


def relative_error(y: float | None, y_ref: float | None) -> float | None:
    """Relative error ``|y-y_ref|/|y_ref|``; ``None`` (never ``inf``) when ``y_ref`` is 0.

    This is the *only* sanctioned place where a relative error is produced: it refuses to
    manufacture an unbounded number when the reference magnitude vanishes.  Callers that
    need a bounded number must go through :func:`tensorphysics.metrology.criteria_semantics.adaptive.evaluate`,
    which switches to the magnitude-normalised metric.
    """
    if y is None or y_ref is None:
        return None
    if y_ref == 0.0:
        return None
    out = abs(y - y_ref) / abs(y_ref)
    if not math.isfinite(out):
        return None
    return out