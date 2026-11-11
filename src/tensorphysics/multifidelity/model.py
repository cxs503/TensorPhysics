"""Model registry: accuracy prior, cost prior and validity domain per model.

A registry is pure data (see ``configs/drag_multifidelity.v1.json`` for the
resistance-prediction case shipped with the package).  This module contains no
domain vocabulary: axis names, spans, error/cost priors and provenance strings
all arrive from the configuration, so the same selector serves any quantity.

Three objects carry the contract:

``ErrorView``
    anything that can answer "what is ``P(error <= x)``" and "give me a
    one-sided upper bound at level ``p``".  :class:`~tensorphysics.multifidelity
    .stats.LognormalPrior` is the registered (literature) view;
    :class:`~tensorphysics.multifidelity.calibration.StratifiedEmpiricalError` is
    the recalibrated (data) view.  Because selection only talks to this
    protocol, swapping a prior for a calibrated view is a one-line operation.
``ApplicabilityDomain``
    per-axis validity box (``Re``, ``Fn``, block coefficient, ...).  A model
    outside its box is *not* a candidate, and the refusal reason names the axis
    that was violated.
``ModelSpec`` / ``ModelRegistry``
    identity, fidelity tier, cost prior, evidence tag and provenance notes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .stats import EmpiricalDistribution, LognormalPrior

CONFIG_SCHEMA_VERSION = 1


class RegistryConfigError(ValueError):
    """Raised for malformed registry data (fail loudly, never silently)."""


class ErrorView(Protocol):  # pragma: no cover - structural typing only
    def prob_le(self, x: float, params: Optional[Mapping[str, float]] = None) -> float: ...
    def upper_bound(self, level: float, params: Optional[Mapping[str, float]] = None) -> float: ...
    def interval(self, level: float) -> Tuple[float, float]: ...


class PriorErrorView:
    """Adapter exposing :class:`LognormalPrior` through the ``ErrorView`` protocol."""

    __slots__ = ("prior",)
    kind = "prior"

    def __init__(self, prior: LognormalPrior) -> None:
        self.prior = prior

    @property
    def median(self) -> float:
        return self.prior.median

    def prob_le(self, x: float, params: Optional[Mapping[str, float]] = None) -> float:
        return self.prior.prob_le(x)

    def upper_bound(self, level: float, params: Optional[Mapping[str, float]] = None) -> float:
        return self.prior.upper_bound(level)

    def interval(self, level: float) -> Tuple[float, float]:
        return self.prior.interval(level)

    def describe(self) -> Dict[str, Any]:
        out = {"kind": self.kind}
        out.update(self.prior.as_dict())
        return out


@dataclass(frozen=True)
class ApplicabilityDomain:
    """A closed per-axis box; ``None`` span means "any finite value"."""

    axes: Mapping[str, Tuple[Optional[float], Optional[float]]]
    source: str = ""

    def __post_init__(self) -> None:
        cleaned: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for name, span in self.axes.items():
            lo, hi = span
            for val in (lo, hi):
                if val is not None and not math.isfinite(val):
                    raise RegistryConfigError(f"axis {name!r} bound must be finite or None")
            if lo is not None and hi is not None and lo >= hi:
                raise RegistryConfigError(f"axis {name!r} has lo >= hi ({lo} >= {hi})")
            cleaned[str(name)] = (None if lo is None else float(lo), None if hi is None else float(hi))
        object.__setattr__(self, "axes", cleaned)

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence[Optional[float]]], source: str = "") -> "ApplicabilityDomain":
        if not isinstance(data, Mapping):
            raise RegistryConfigError("domain must be a mapping of axis -> [lo, hi]")
        axes = {}
        for name, span in data.items():
            if isinstance(span, Mapping):
                lo, hi = span.get("lo"), span.get("hi")
            else:
                if len(span) != 2:
                    raise RegistryConfigError(f"axis {name!r}: span must have exactly 2 entries")
                lo, hi = span
            axes[name] = (lo, hi)
        return cls(axes=axes, source=source)

    def admits(self, params: Mapping[str, float]) -> bool:
        return self.violations(params) is None

    def violations(self, params: Mapping[str, float]) -> Optional[Dict[str, Any]]:
        """``None`` when admissible, else a dict naming the first violated axis."""
        for name, (lo, hi) in self.axes.items():
            if name not in params:
                return {"reason": "MISSING_AXIS", "axis": name, "value": None, "span": [lo, hi]}
            value = params[name]
            if value is None or not math.isfinite(float(value)):
                return {"reason": "NON_FINITE_AXIS", "axis": name, "value": value, "span": [lo, hi]}
            value = float(value)
            if lo is not None and value < lo:
                return {"reason": "AXIS_BELOW_DOMAIN", "axis": name, "value": value, "span": [lo, hi]}
            if hi is not None and value > hi:
                return {"reason": "AXIS_ABOVE_DOMAIN", "axis": name, "value": value, "span": [lo, hi]}
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"axes": {k: list(v) for k, v in self.axes.items()}, "source": self.source}


@dataclass(frozen=True)
class CostPrior:
    """Log-normal CPU/facility cost prior, in seconds, parameterised by the median."""

    median_s: float
    sigma: float = 0.0
    provenance: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.median_s) or self.median_s <= 0.0:
            raise RegistryConfigError(f"cost median must be > 0, got {self.median_s!r}")
        if not math.isfinite(self.sigma) or self.sigma < 0.0:
            raise RegistryConfigError(f"cost sigma must be >= 0, got {self.sigma!r}")

    @property
    def mean_s(self) -> float:
        return self.median_s * math.exp(0.5 * self.sigma ** 2)

    def prob_le(self, budget_s: float) -> float:
        if budget_s <= 0.0:
            return 0.0
        if self.sigma == 0.0:
            return 1.0 if budget_s >= self.median_s else 0.0
        return 0.5 * (1.0 + math.erf(math.log(budget_s / self.median_s) / (self.sigma * math.sqrt(2.0))))

    def quantile(self, p: float) -> float:
        from .stats import normal_ppf

        return self.median_s * math.exp(self.sigma * normal_ppf(p))

    def to_dict(self) -> Dict[str, Any]:
        return {"median_s": self.median_s, "sigma": self.sigma, "mean_s": self.mean_s,
                "provenance": self.provenance}


@dataclass(frozen=True)
class ModelSpec:
    """One candidate fidelity level of the same physical quantity."""

    model_id: str
    error: ErrorView
    cost: CostPrior
    domain: ApplicabilityDomain
    fidelity_level: int = 0
    label: str = ""
    evidence: str = "WITHHELD"
    prior: Optional[LognormalPrior] = None
    provenance: str = ""
    notes: str = ""
    calibration: Optional[Dict[str, Any]] = None

    def with_error_view(self, view: ErrorView, calibration: Optional[Dict[str, Any]] = None) -> "ModelSpec":
        return replace(self, error=view, calibration=calibration)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "model_id": self.model_id,
            "label": self.label,
            "fidelity_level": self.fidelity_level,
            "evidence": self.evidence,
            "provenance": self.provenance,
            "notes": self.notes,
            "domain": self.domain.to_dict(),
            "cost": self.cost.to_dict(),
            "error": self.error.describe() if hasattr(self.error, "describe") else {},
        }
        if self.calibration is not None:
            out["calibration"] = self.calibration
        return out


def _lognormal_from_dict(data: Mapping[str, Any], where: str) -> LognormalPrior:
    if "median" not in data:
        raise RegistryConfigError(f"{where}: error prior needs a 'median' field")
    try:
        return LognormalPrior(median=float(data["median"]), sigma=float(data.get("sigma", 0.0)))
    except ValueError as exc:  # re-wrap with location context
        raise RegistryConfigError(f"{where}: {exc}") from exc


class ModelRegistry:
    """Ordered collection of :class:`ModelSpec`, keyed by ``model_id``."""

    def __init__(self, models: Sequence[ModelSpec], meta: Optional[Mapping[str, Any]] = None) -> None:
        if not models:
            raise RegistryConfigError("registry needs at least one model")
        seen = set()
        for m in models:
            if m.model_id in seen:
                raise RegistryConfigError(f"duplicate model_id {m.model_id!r}")
            seen.add(m.model_id)
        self._models: Tuple[ModelSpec, ...] = tuple(models)
        self.meta: Dict[str, Any] = dict(meta or {})

    # -- construction ----------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any], source: str = "<dict>") -> "ModelRegistry":
        if int(data.get("schema_version", CONFIG_SCHEMA_VERSION)) != CONFIG_SCHEMA_VERSION:
            raise RegistryConfigError(f"unsupported schema_version in {source}")
        rows = data.get("models")
        if not rows:
            raise RegistryConfigError(f"{source}: no models registered")
        models: List[ModelSpec] = []
        for idx, row in enumerate(rows):
            where = f"{source}:models[{idx}]"
            if "model_id" not in row:
                raise RegistryConfigError(f"{where}: missing model_id")
            prior = _lognormal_from_dict(row.get("error_prior", {}), where)
            cost_row = row.get("cost", {})
            if "median_s" not in cost_row:
                raise RegistryConfigError(f"{where}: cost needs 'median_s'")
            cost = CostPrior(median_s=float(cost_row["median_s"]),
                             sigma=float(cost_row.get("sigma", 0.0)),
                             provenance=str(cost_row.get("provenance", "")))
            domain = ApplicabilityDomain.from_dict(row.get("domain", {}), source=where)
            models.append(
                ModelSpec(
                    model_id=str(row["model_id"]),
                    error=PriorErrorView(prior),
                    cost=cost,
                    domain=domain,
                    fidelity_level=int(row.get("fidelity_level", idx)),
                    label=str(row.get("label", row["model_id"])),
                    evidence=str(row.get("evidence", "WITHHELD")),
                    prior=prior,
                    provenance=str(row.get("provenance", "")),
                    notes=str(row.get("notes", "")),
                )
            )
        meta = {k: v for k, v in data.items() if k not in {"models"}}
        return cls(models, meta=meta)

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelRegistry":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data, source=str(path))

    # -- access ----------------------------------------------------------
    @property
    def models(self) -> Tuple[ModelSpec, ...]:
        return self._models

    def ids(self) -> List[str]:
        return [m.model_id for m in self._models]

    def get(self, model_id: str) -> ModelSpec:
        for m in self._models:
            if m.model_id == model_id:
                return m
        raise KeyError(f"model {model_id!r} not in registry")

    def admissible(self, params: Mapping[str, float]) -> List[ModelSpec]:
        return [m for m in self._models if m.domain.admits(params)]

    def excluded(self, params: Mapping[str, float]) -> List[Tuple[str, Dict[str, Any]]]:
        out = []
        for m in self._models:
            v = m.domain.violations(params)
            if v is not None:
                out.append((m.model_id, v))
        return out

    def error_floor(self, params: Mapping[str, float], quantile: float = 0.5) -> Optional[float]:
        """Smallest ``quantile`` of the error prior over admissible models.

        This is the *physical* accuracy floor of the query: no model can beat it,
        so an ``epsilon`` below it is unachievable by construction (negative
        control of the selection engine).
        """
        cands = self.admissible(params)
        if not cands:
            return None
        return min(m.error.upper_bound(quantile, params) for m in cands)

    def cost_floor(self, params: Mapping[str, float], quantile: float = 0.5) -> Optional[float]:
        cands = self.admissible(params)
        if not cands:
            return None
        return min(m.cost.quantile(quantile) for m in cands)

    def with_error_views(self, views: Mapping[str, ErrorView],
                         calibration: Optional[Mapping[str, Mapping[str, Any]]] = None) -> "ModelRegistry":
        """New registry with swapped error views (models without a view keep their prior)."""
        cal = calibration or {}
        models = []
        for m in self._models:
            if m.model_id in views:
                models.append(m.with_error_view(views[m.model_id], cal.get(m.model_id)))
            elif m.model_id in cal:  # unchanged view, but record why (prior fallback)
                models.append(replace(m, calibration=cal[m.model_id]))
            else:
                models.append(m)
        return ModelRegistry(models, meta=dict(self.meta))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"schema_version": CONFIG_SCHEMA_VERSION,
                               "models": [m.to_dict() for m in self._models]}
        out.update(self.meta)
        return out


def residuals_from_distribution(prior: LognormalPrior, n: int, rng) -> EmpiricalDistribution:
    """Convenience helper: draw a residual sample from a prior (synthetic registries)."""
    return EmpiricalDistribution.from_values(prior.sample(rng, int(n)))