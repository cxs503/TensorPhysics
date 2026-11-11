"""Recalibration of registered error priors from validation residuals.

Registered priors answer "how wrong is model *m* on its calibration set?".  A new
hull is not in that set: heavy-tail blow-ups and regime-dependent growth are
invisible to a two-parameter log-normal law, and the reported confidence
interval then under-covers (see
``docs/MULTIFIDELITY_MODEL_SELECTION_P9003W1_N05.md`` for the measured 72 % vs
90 % gap).

The fix implemented here is assumption-light:

* a residual sample ``r_1..r_n`` (relative error magnitudes of model *m*) is
  turned into an :class:`~tensorphysics.multifidelity.stats.EmpiricalDistribution`;
* the one-sided bound at level ``p`` is the conformal order statistic
  ``k = ceil((n+1) p)``, which has finite-sample coverage ``>= p`` under
  exchangeability -- no log-normal assumption survives the swap;
* residuals may be **stratified** by an axis box (:class:`StratumSpec`); the
  bound for a query is taken from the matching stratum when that stratum has at
  least ``min_n`` samples, otherwise from the pooled sample.  Stratification is
  what removes residual regime bias ("error grows with Froude number") without
  pretending to model it;

Nothing here is domain-specific: axis names and boxes come from configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .model import ErrorView, ModelRegistry, RegistryConfigError
from .stats import EmpiricalDistribution, _check_level


@dataclass(frozen=True)
class StratumSpec:
    """A named axis box used to stratify residual samples."""

    name: str
    bounds: Mapping[str, Tuple[Optional[float], Optional[float]]]

    def __post_init__(self) -> None:
        if not self.name:
            raise RegistryConfigError("stratum needs a name")
        for axis, (lo, hi) in self.bounds.items():
            if lo is not None and hi is not None and lo >= hi:
                raise RegistryConfigError(f"stratum {self.name!r}: axis {axis!r} has lo >= hi")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StratumSpec":
        bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for axis, span in dict(data.get("bounds", {})).items():
            if isinstance(span, Mapping):
                bounds[axis] = (span.get("lo"), span.get("hi"))
            else:
                lo, hi = span
                bounds[axis] = (lo, hi)
        return cls(name=str(data.get("name", "")), bounds=bounds)

    def matches(self, params: Mapping[str, float]) -> bool:
        for axis, (lo, hi) in self.bounds.items():
            if axis not in params or params[axis] is None:
                return False
            value = float(params[axis])
            if lo is not None and value < float(lo):
                return False
            if hi is not None and value > float(hi):
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "bounds": {k: list(v) for k, v in self.bounds.items()}}


class StratifiedEmpiricalError:
    """Empirical (optionally stratified) error law exposed as an ``ErrorView``."""

    __slots__ = ("pooled", "strata", "samples", "min_n", "default_level")
    kind = "empirical"

    def __init__(self, pooled: EmpiricalDistribution,
                 strata: Sequence[StratumSpec] = (),
                 samples: Optional[Mapping[str, Sequence[float]]] = None,
                 min_n: int = 30,
                 default_level: float = 0.90) -> None:
        _check_level(default_level, "default_level")
        if min_n < 2:
            raise ValueError("min_n must be >= 2")
        ordered = list(strata)
        grouped: Dict[str, EmpiricalDistribution] = {}
        for name, values in dict(samples or {}).items():
            grouped[str(name)] = EmpiricalDistribution.from_values(values)
        unknown = set(grouped) - {s.name for s in ordered}
        if unknown:
            raise RegistryConfigError(f"residuals supplied for unknown strata: {sorted(unknown)}")
        self.pooled = pooled
        self.strata = tuple(ordered)
        self.samples: Dict[str, EmpiricalDistribution] = grouped
        self.min_n = int(min_n)
        self.default_level = float(default_level)

    # -- selection of the applicable sample ------------------------------
    def _view_for(self, params: Optional[Mapping[str, float]]) -> Tuple[EmpiricalDistribution, str]:
        if params is not None:
            for spec in self.strata:  # first matching, sufficiently populated stratum wins
                if spec.matches(params):
                    sample = self.samples.get(spec.name)
                    if sample is not None and sample.n >= self.min_n:
                        return sample, spec.name
        return self.pooled, "__pooled__"

    def stratum_name(self, params: Optional[Mapping[str, float]]) -> str:
        return self._view_for(params)[1]

    # -- ErrorView protocol ----------------------------------------------
    def prob_le(self, x: float, params: Optional[Mapping[str, float]] = None) -> float:
        view, _ = self._view_for(params)
        return view.prob_le(x)

    def upper_bound(self, level: Optional[float] = None,
                    params: Optional[Mapping[str, float]] = None) -> float:
        view, _ = self._view_for(params)
        return view.conformal_upper(self.default_level if level is None else level)

    def interval(self, level: float) -> Tuple[float, float]:
        return self.pooled.interval(level)

    @property
    def median(self) -> float:
        return self.pooled.median

    def describe(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind, "pooled": self.pooled.as_dict(),
                               "min_n": self.min_n, "default_level": self.default_level,
                               "strata": {}}
        for spec in self.strata:
            sample = self.samples.get(spec.name)
            out["strata"][spec.name] = {
                "bounds": {k: list(v) for k, v in spec.bounds.items()},
                "n": 0 if sample is None else sample.n,
                "upper_bound_90": None if sample is None else sample.conformal_upper(0.90),
            }
        return out


def _empirical_meta(dist: EmpiricalDistribution, view: StratifiedEmpiricalError) -> Dict[str, Any]:
    return {"kind": "empirical", "n": dist.n,
            "median": dist.median,
            "upper_bound_90": dist.conformal_upper(0.90),
            "digest": dist.digest(),
            "strata": view.describe()["strata"]}


def calibrate_registry(registry: ModelRegistry,
                       residuals: Mapping[str, Sequence[float]],
                       strata: Sequence[StratumSpec] = (),
                       min_n: int = 30,
                       level: float = 0.90) -> ModelRegistry:
    """Pooled recalibration: one residual sample per model, no stratification.

    Returns a new registry; the input object is never mutated.  Models without a
    residual sample keep their registered prior and are flagged
    ``prior_fallback`` in the calibration provenance.
    """
    if not residuals:
        raise RegistryConfigError("calibrate_registry needs at least one residual sample")
    unknown = set(residuals) - set(registry.ids())
    if unknown:
        raise RegistryConfigError(f"residuals for unknown models: {sorted(unknown)}")
    views: Dict[str, ErrorView] = {}
    cal_meta: Dict[str, Dict[str, Any]] = {}
    for model in registry.models:
        values = residuals.get(model.model_id)
        if values is None:
            cal_meta[model.model_id] = {"kind": "prior_fallback",
                                        "reason": "no residual sample supplied"}
            continue
        dist = EmpiricalDistribution.from_values(values)
        view = StratifiedEmpiricalError(pooled=dist, strata=strata, samples={},
                                        min_n=min_n, default_level=level)
        views[model.model_id] = view
        cal_meta[model.model_id] = _empirical_meta(dist, view)
    return registry.with_error_views(views, cal_meta)


def calibrate_registry_stratified(registry: ModelRegistry,
                                  stratified_residuals: Mapping[str, Mapping[str, Sequence[float]]],
                                  strata: Sequence[StratumSpec],
                                  min_n: int = 30,
                                  level: float = 0.90) -> ModelRegistry:
    """Recalibrate with per-stratum residual samples.

    ``stratified_residuals`` maps ``model_id -> {stratum_name: [residuals]}``;
    the pooled sample is the concatenation of the stratum samples, so a stratum
    with fewer than ``min_n`` samples degrades gracefully to the pooled law.
    """
    if not stratified_residuals:
        raise RegistryConfigError("calibrate_registry_stratified needs residual samples")
    unknown = set(stratified_residuals) - set(registry.ids())
    if unknown:
        raise RegistryConfigError(f"residuals for unknown models: {sorted(unknown)}")
    views: Dict[str, ErrorView] = {}
    cal_meta: Dict[str, Dict[str, Any]] = {}
    for model in registry.models:
        groups = stratified_residuals.get(model.model_id)
        if not groups:
            cal_meta[model.model_id] = {"kind": "prior_fallback",
                                        "reason": "no residual sample supplied"}
            continue
        pooled_values = [v for values in groups.values() for v in values]
        dist = EmpiricalDistribution.from_values(pooled_values)
        view = StratifiedEmpiricalError(pooled=dist, strata=strata,
                                        samples={k: list(v) for k, v in groups.items()},
                                        min_n=min_n, default_level=level)
        views[model.model_id] = view
        cal_meta[model.model_id] = _empirical_meta(dist, view)
    return registry.with_error_views(views, cal_meta)