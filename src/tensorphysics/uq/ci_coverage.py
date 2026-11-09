"""Are the quoted 95 % Sobol'-index intervals actually 95 %?

A confidence interval that is reported but never calibrated is decoration.  The
gate of mission IN145-N08 quotes a 95 % interval for every first- and
total-order index, so the interval itself has to be checked against a quantity
whose true value is known: the Ishigami function and Sobol's ``G`` function both
have closed-form indices.

The experiment is a standard repeated-sample coverage study:

for ``r = 1..R`` independent training designs (different sampler seeds)
    fit the chaos surrogate, resample its coefficients ``B`` times
    and record whether the percentile interval of S_i contains the exact S_i

The estimator is the coverage rate ``c_i = (#cover) / R``.  With ``R = 20`` the
binomial standard error at nominal 0.95 is 0.049, so only gross
under-coverage is detectable -- that limitation is stated in the report rather
than hidden behind a wide pass band.  Two machine checks are applied:

* **under-coverage test**: one-sided exact binomial test of ``H0: c >= level``
  against ``H1: c < level``; a p-value below ``alpha`` flags the interval as
  under-covering;
* **floor**: the pooled coverage over all indices must stay above
  ``coverage_floor`` (default 0.80), which is loose enough not to fail on
  Monte-Carlo noise at ``R = 20`` and tight enough to catch the ``N = 128``
  regime where the probe measured 0.67.

Both are *reported*, including the failing regime, so the gate can state the
sample count above which its intervals may be trusted.

Null indices
------------
An index whose exact value is zero (Ishigami's ``S_3``, ``S_12``) has no scale,
so a percentile interval of a numerically-zero coefficient is one-sided: the
estimate is ``~1e-6`` and the bootstrap interval is ``[~0, ~1e-5]``, which
"misses" the exact zero by a rounding error.  Such entries are therefore scored
with a *resolution* criterion instead of a containment criterion -- the interval
has to be consistent with zero up to ``null_tol`` (default ``1e-3``, the same
1e-3 the index criterion uses) -- and the binomial under-coverage test is not
applied to them.  Every entry records which criterion was used
(``criterion`` field), so this is visible, not hidden.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .pce import PCESurrogate, bootstrap_sobol, fit_pce
from .sample_spec import SampleSpec
from .sampler import sample

__all__ = [
    "CoverageEntry",
    "CoverageReport",
    "coverage_study",
    "wilson_interval",
]

DEFAULT_COVERAGE_FLOOR = 0.80
#: |S| below this is treated as a null index (resolution criterion).
DEFAULT_NULL_MATERIAL = 1.0e-2
#: A null index is "resolved" when the interval lower bound is below this.
DEFAULT_NULL_TOL = 1.0e-3


def wilson_interval(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (small-sample safe)."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    z = float(_normal_quantile(0.5 * (1.0 + level)))
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF by bisection on ``math.erf`` (no scipy)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    lo, hi = -10.0, 10.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _covered(truth: float, lo: float, hi: float, null_material: float, null_tol: float) -> bool:
    """Containment for a material index, resolution-to-zero for a null index."""
    if abs(float(truth)) >= null_material:
        return bool(lo <= truth <= hi)
    return bool(lo <= null_tol)


def _binom_lower_pvalue(successes: int, trials: int, p0: float) -> float:
    """One-sided P(X <= successes) for X ~ Binomial(trials, p0)."""
    if trials <= 0:
        return 1.0
    p0 = min(max(float(p0), 0.0), 1.0)
    total = 0.0
    for k in range(0, successes + 1):
        total += math.comb(trials, k) * p0 ** k * (1.0 - p0) ** (trials - k)
    return float(min(1.0, total))


@dataclass(frozen=True)
class CoverageEntry:
    """Coverage of one index by its quoted interval."""

    kind: str
    index: int
    label: str
    analytic: float
    coverage: float
    replications: int
    mean_estimate: float
    mean_halfwidth: float
    interval: tuple[float, float]
    under_coverage_pvalue: float
    criterion: str = "interval"

    @property
    def is_null(self) -> bool:
        return self.criterion == "resolution"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "index": int(self.index),
            "label": self.label,
            "analytic": float(self.analytic),
            "coverage": float(self.coverage),
            "replications": int(self.replications),
            "mean_estimate": float(self.mean_estimate),
            "mean_halfwidth": float(self.mean_halfwidth),
            "wilson95": [float(self.interval[0]), float(self.interval[1])],
            "under_coverage_pvalue": float(self.under_coverage_pvalue),
            "criterion": self.criterion,
        }


@dataclass(frozen=True)
class CoverageReport:
    """Result of a repeated-sample coverage study."""

    n_samples: int
    degree: int
    n_boot: int
    level: float
    replications: int
    entries: tuple[CoverageEntry, ...] = ()
    alpha: float = 0.05
    coverage_floor: float = DEFAULT_COVERAGE_FLOOR
    seed0: int = 20240
    null_material: float = DEFAULT_NULL_MATERIAL
    null_tol: float = DEFAULT_NULL_TOL

    @property
    def pooled_coverage(self) -> float:
        if not self.entries:
            return float("nan")
        return float(np.mean([e.coverage for e in self.entries]))

    @property
    def min_coverage(self) -> float:
        return float(min(e.coverage for e in self.entries)) if self.entries else float("nan")

    @property
    def under_covering(self) -> bool:
        """True when any index shows significant under-coverage."""
        return any(e.under_coverage_pvalue < self.alpha for e in self.entries)

    @property
    def floor_met(self) -> bool:
        return bool(self.pooled_coverage >= self.coverage_floor)

    @property
    def calibrated(self) -> bool:
        return bool(self.floor_met and not self.under_covering)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": int(self.n_samples),
            "degree": int(self.degree),
            "n_boot": int(self.n_boot),
            "level": float(self.level),
            "replications": int(self.replications),
            "alpha": float(self.alpha),
            "coverage_floor": float(self.coverage_floor),
            "seed0": int(self.seed0),
            "null_material": float(self.null_material),
            "null_tol": float(self.null_tol),
            "pooled_coverage": self.pooled_coverage,
            "min_coverage": self.min_coverage,
            "under_covering": self.under_covering,
            "floor_met": self.floor_met,
            "calibrated": self.calibrated,
            "entries": [e.as_dict() for e in self.entries],
        }


def coverage_study(
    spec: SampleSpec,
    model: Any,
    analytic: Mapping[str, Any],
    *,
    n_samples: int = 256,
    degree: int = 8,
    n_boot: int = 200,
    level: float = 0.95,
    replications: int = 20,
    seed0: int = 20240,
    alpha: float = 0.05,
    coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
    null_material: float = DEFAULT_NULL_MATERIAL,
    null_tol: float = DEFAULT_NULL_TOL,
) -> CoverageReport:
    """Repeated-sample coverage of the bootstrap Sobol' intervals.

    ``analytic`` is the closed-form reference (the dict returned by
    :func:`tensorphysics.uq.benchmarks.ishigami_analytic` or
    ``sobol_g_analytic``): ``first`` and ``total`` arrays are required,
    ``second`` (a dict keyed by index tuples) is optional.
    """
    if not isinstance(spec, SampleSpec):
        raise TypeError(f"spec must be a SampleSpec, got {type(spec)!r}")
    if replications < 5:
        raise ValueError(f"replications must be >= 5, got {replications}")
    evaluate = getattr(model, "evaluate", None)
    if callable(evaluate) and not callable(model):
        fn = lambda pts: np.asarray(evaluate(pts), dtype=float).reshape(-1)  # noqa: E731
    elif callable(model):
        fn = lambda pts: np.asarray(model(pts), dtype=float).reshape(-1)  # noqa: E731
    else:  # pragma: no cover - defensive
        raise TypeError(f"model must be callable or expose .evaluate(), got {type(model)!r}")

    truth_first = np.asarray(analytic["first"], dtype=float)
    truth_total = np.asarray(analytic["total"], dtype=float)
    dim = spec.dim
    if truth_first.size != dim or truth_total.size != dim:
        raise ValueError(
            f"analytic index arrays have size {truth_first.size}/{truth_total.size}, "
            f"expected {dim}"
        )

    names = spec.names
    hits_first = np.zeros(dim)
    hits_total = np.zeros(dim)
    est_first = np.zeros(dim)
    est_total = np.zeros(dim)
    half_first = np.zeros(dim)
    half_total = np.zeros(dim)

    for r in range(int(replications)):
        design = sample(spec, int(n_samples), seed=int(seed0) + r)
        points, coords = spec.from_latent(design.latent)
        values = fn(points)
        if values.size != points.shape[0]:
            raise ValueError(
                f"model returned {values.size} values for {points.shape[0]} points"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("model returned non-finite values")
        surrogate = fit_pce(
            coords, values, degree=int(degree), families=spec.families
        )
        indices = bootstrap_sobol(
            surrogate, coords, values, n_boot=int(n_boot), seed=int(seed0) + 1000 + r,
            order=1, level=level,
        )
        lo_f, hi_f = indices.first_ci
        lo_t, hi_t = indices.total_ci
        for i in range(dim):
            est_first[i] += indices.first[i] / replications
            est_total[i] += indices.total[i] / replications
            half_first[i] += 0.5 * (hi_f[i] - lo_f[i]) / replications
            half_total[i] += 0.5 * (hi_t[i] - lo_t[i]) / replications
            if _covered(truth_first[i], lo_f[i], hi_f[i], null_material, null_tol):
                hits_first[i] += 1
            if _covered(truth_total[i], lo_t[i], hi_t[i], null_material, null_tol):
                hits_total[i] += 1

    entries: list[CoverageEntry] = []
    for i in range(dim):
        for kind, truth, hits, estimate, halfwidth in (
            ("first", truth_first[i], hits_first[i], est_first[i], half_first[i]),
            ("total", truth_total[i], hits_total[i], est_total[i], half_total[i]),
        ):
            null = abs(float(truth)) < null_material
            entries.append(
                CoverageEntry(
                    kind=kind,
                    index=i,
                    label=names[i],
                    analytic=float(truth),
                    coverage=float(hits / replications),
                    replications=int(replications),
                    mean_estimate=float(estimate),
                    mean_halfwidth=float(halfwidth),
                    interval=wilson_interval(int(hits), int(replications), level),
                    under_coverage_pvalue=(
                        1.0
                        if null
                        else _binom_lower_pvalue(int(hits), int(replications), level)
                    ),
                    criterion="resolution" if null else "interval",
                )
            )

    return CoverageReport(
        n_samples=int(n_samples),
        degree=int(degree),
        n_boot=int(n_boot),
        level=float(level),
        replications=int(replications),
        entries=tuple(entries),
        alpha=float(alpha),
        coverage_floor=float(coverage_floor),
        seed0=int(seed0),
        null_material=float(null_material),
        null_tol=float(null_tol),
    )