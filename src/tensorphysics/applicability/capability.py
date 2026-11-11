"""Capability registration: axes, evaluator operator, failure signals, declaration.

A capability is registered as *data*: which callable to evaluate, which keyword
each axis drives, the acceptance criterion, the failure signals and (optionally)
the human-written applicability statement.  The code in this package never
contains a capability name, an axis name or a domain word -- everything comes
from the configuration mapping handed to :meth:`CapabilityRegistry.from_config`.

Configuration schema (version 1)::

    {
      "schema_version": 1,
      "capabilities": [
        {
          "capability_id": "<unique id>",
          "evaluator": {
            "target": "<module>:<callable>",
            "kwargs": {"<keyword>": <json value>, ...},
            "axis_kwargs": {"<axis name>": "<keyword>", ...},
            "value_key": "value", "reference_key": "reference", "flags_key": "flags"
          },
          "axes": [
            {"name": ..., "low": ..., "high": ..., "scale": "linear|log",
             "samples": 17, "distribution": "geometric|uniform",
             "monotone": "none|increasing|decreasing", "monotone_slack": 0.3,
             "sign": 0, "floor": null, "nominal": null}
          ],
          "policy": {
            "criterion": {"kind": "abs|rel", "reference": 0.0, "tolerance": 0.15}
                       | {"kind": "range", "lower": 0.0, "upper": 1.0},
            "pathology": {"flags": {"converged": false}, "nan": true},
            "order_window": {"key": "observed_order", "low": 0.5, "high": 6.0}
          },
          "declared": {
            "<axis name>": {"low": ..., "high": ..., "slack": 0.0,
                            "source": "...", "note": "..."}
          },
          "labels": {"title": "..."}
        }
      ]
    }

Fail-closed rules enforced here:

* unknown keys are rejected (a typo can never silently disable a criterion);
* the evaluator target is imported eagerly, so an unreachable callable fails at
  registration time, not in the middle of a probe;
* an axis driven by the evaluator must exist, and every declared axis must be
  driven (no axis silently evaluated as a constant).
"""
from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .axis import AxisSpec

__all__ = [
    "ApplicabilityConfigError",
    "CapabilitySpec",
    "Criterion",
    "DeclaredAxis",
    "FailurePolicy",
    "InteractionConfig",
    "EvaluatorSpec",
    "CapabilityRegistry",
    "import_target",
]

EVALUATOR_KEYS = ("target", "kwargs", "axis_kwargs", "value_key", "reference_key", "flags_key")
POLICY_KEYS = ("criterion", "pathology", "order_window", "sign")
CRITERION_KEYS = ("kind", "reference", "tolerance", "lower", "upper")
PATHOLOGY_KEYS = ("flags", "nan")
ORDER_KEYS = ("key", "low", "high")
DECLARED_KEYS = ("low", "high", "slack", "source", "note")


class ApplicabilityConfigError(ValueError):
    """A capability declaration is malformed or inconsistent (fail-closed)."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApplicabilityConfigError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ApplicabilityConfigError(f"{name} must be finite, got {value!r}")
    return out


def import_target(target: str) -> Callable[..., Any]:
    """Import ``"package.module:attribute"`` (the only dynamic import in the package)."""
    if not isinstance(target, str) or target.count(":") != 1:
        raise ApplicabilityConfigError(f"target must look like 'module:callable', got {target!r}")
    module_name, attr = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - defensive
        raise ApplicabilityConfigError(f"cannot import {module_name!r}: {exc}") from exc
    try:
        value = getattr(module, attr)
    except AttributeError as exc:
        raise ApplicabilityConfigError(f"{module_name!r} has no attribute {attr!r}") from exc
    if not callable(value):
        raise ApplicabilityConfigError(f"{target!r} is not callable")
    return value


@dataclass(frozen=True)
class Criterion:
    """Acceptance criterion of one evaluation (criterion (a)/(c) of the probe).

    ``abs``   : ``|value - reference| <= tolerance``
    ``rel``   : ``|value - reference| <= tolerance * |reference|``
    ``range`` : ``lower <= value <= upper`` (one-sided bounds may be ``None``)
    """

    kind: str
    reference: float | None = None
    tolerance: float | None = None
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("abs", "rel", "range"):
            raise ApplicabilityConfigError(f"criterion.kind must be abs|rel|range, got {self.kind!r}")
        if self.kind in ("abs", "rel"):
            if self.reference is None or self.tolerance is None:
                raise ApplicabilityConfigError(f"criterion {self.kind!r} needs reference and tolerance")
            object.__setattr__(self, "reference", _number(self.reference, "criterion.reference"))
            object.__setattr__(self, "tolerance", _number(self.tolerance, "criterion.tolerance"))
            if self.tolerance < 0.0:
                raise ApplicabilityConfigError("criterion.tolerance must be >= 0")
        else:
            if self.lower is None and self.upper is None:
                raise ApplicabilityConfigError("criterion 'range' needs at least one bound")
            if self.lower is not None:
                object.__setattr__(self, "lower", _number(self.lower, "criterion.lower"))
            if self.upper is not None:
                object.__setattr__(self, "upper", _number(self.upper, "criterion.upper"))
            if self.lower is not None and self.upper is not None and not self.lower < self.upper:
                raise ApplicabilityConfigError("criterion.lower must be < criterion.upper")

    def check(self, value: float | None, reference: float | None = None) -> tuple[bool, float, str]:
        """Return ``(passes, normalised_margin, detail)``; margin 1 = far inside, 0 = on the limit."""
        if value is None or not math.isfinite(value):
            return False, 0.0, "value is not finite"
        ref = self.reference
        if self.kind in ("abs", "rel"):
            if reference is not None and math.isfinite(reference):
                ref = reference
            assert ref is not None and self.tolerance is not None  # validated in __post_init__
            scale = abs(ref) if self.kind == "rel" else 1.0
            limit = self.tolerance * scale
            deviation = abs(value - ref)
            if limit == 0.0:
                ok = deviation == 0.0
                margin = 1.0 if ok else 0.0
            else:
                ok = deviation <= limit
                margin = max(0.0, min(1.0, (limit - deviation) / limit))
            return ok, margin, f"{self.kind}: |{value:.6g}-{ref:.6g}|={deviation:.3e} vs {limit:.3e}"
        lower, upper = self.lower, self.upper
        ok = (lower is None or value >= lower) and (upper is None or value <= upper)
        span = None
        if lower is not None and upper is not None:
            span = upper - lower
        if span and span > 0.0:
            margin = min(value - lower, upper - value) / span
            margin = max(0.0, min(1.0, margin))
        else:
            margin = 1.0 if ok else 0.0
        return ok, margin, f"range: {value:.6g} not in [{lower}, {upper}]" if not ok else "range: inside"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "reference": self.reference, "tolerance": self.tolerance,
                "lower": self.lower, "upper": self.upper}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Criterion":
        unknown = sorted(set(mapping) - set(CRITERION_KEYS))
        if unknown:
            raise ApplicabilityConfigError(f"unknown criterion keys: {unknown}")
        return cls(**dict(mapping))


@dataclass(frozen=True)
class FailurePolicy:
    """Which observable failures put a sample out of domain."""

    criterion: Criterion
    pathology_flags: Mapping[str, Any] = field(default_factory=dict)
    nan_is_pathological: bool = True
    order_window: tuple[str, float, float] | None = None  # (key, low, high)
    sign: int = 0  # 0 = none, +1 = measured scalar must stay >= 0, -1 = must stay <= 0

    def __post_init__(self) -> None:
        if not isinstance(self.pathology_flags, Mapping):
            raise ApplicabilityConfigError("pathology.flags must be a mapping")
        if self.sign not in (-1, 0, 1):
            raise ApplicabilityConfigError("policy.sign must be -1, 0 or 1")
        if self.order_window is not None:
            key, low, high = self.order_window
            if not isinstance(key, str) or not key:
                raise ApplicabilityConfigError("order_window key must be a non-empty string")
            lo, hi = _number(low, "order_window.low"), _number(high, "order_window.high")
            if not lo < hi:
                raise ApplicabilityConfigError("order_window.low must be < high")
            object.__setattr__(self, "order_window", (key, lo, hi))

    def sign_violation(self, value: float | None) -> str | None:
        """Criterion (b), point form: the measured scalar must keep its declared sign."""
        if self.sign == 0 or value is None or not math.isfinite(value):
            return None
        if self.sign > 0 and value < 0.0:
            return f"sign law: expected a non-negative measure, got {value:.3e}"
        if self.sign < 0 and value > 0.0:
            return f"sign law: expected a non-positive measure, got {value:.3e}"
        return None

    def pathology(self, value: float | None, flags: Mapping[str, Any]) -> tuple[bool, str]:
        """Convergence-order collapse, declared failure flags, and non-finite values."""
        if self.order_window is not None:
            key, low, high = self.order_window
            order = flags.get(key)
            if order is None:
                return True, f"order window declared but {key!r} missing from evaluator flags"
            try:
                order_value = float(order)
            except (TypeError, ValueError):
                return True, f"{key}={order!r} is not a number"
            if not (low < order_value <= high):
                return True, f"order collapse: {key}={order_value:.3g} outside ({low}, {high}]"
        for name, bad in self.pathology_flags.items():
            if name not in flags:
                return True, f"declared failure signal {name!r} absent from evaluator flags"
            if flags[name] == bad:
                return True, f"failure signal {name}={flags[name]!r}"
        if self.nan_is_pathological and (value is None or not math.isfinite(value)):
            return True, "non-finite value"
        return False, ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion.to_dict(),
            "pathology": {"flags": dict(self.pathology_flags), "nan": self.nan_is_pathological},
            "order_window": None if self.order_window is None else {
                "key": self.order_window[0], "low": self.order_window[1], "high": self.order_window[2]},
            "sign": self.sign,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FailurePolicy":
        unknown = sorted(set(mapping) - set(POLICY_KEYS))
        if unknown:
            raise ApplicabilityConfigError(f"unknown policy keys: {unknown}")
        if "criterion" not in mapping:
            raise ApplicabilityConfigError("policy.criterion is required")
        pathology = dict(mapping.get("pathology") or {})
        unknown = sorted(set(pathology) - set(PATHOLOGY_KEYS))
        if unknown:
            raise ApplicabilityConfigError(f"unknown pathology keys: {unknown}")
        order = mapping.get("order_window")
        order_window = None
        if order is not None:
            if not isinstance(order, Mapping):
                raise ApplicabilityConfigError("order_window must be a mapping")
            unknown = sorted(set(order) - set(ORDER_KEYS))
            if unknown:
                raise ApplicabilityConfigError(f"unknown order_window keys: {unknown}")
            order_window = (order.get("key", "observed_order"), order.get("low"), order.get("high"))
        return cls(
            criterion=Criterion.from_mapping(mapping["criterion"]),
            pathology_flags=dict(pathology.get("flags") or {}),
            nan_is_pathological=bool(pathology.get("nan", True)),
            order_window=order_window,
            sign=int(mapping.get("sign", 0)),
        )


@dataclass(frozen=True)
class EvaluatorSpec:
    """How to evaluate the capability at one point of the axis space."""

    target: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    axis_kwargs: Mapping[str, str] = field(default_factory=dict)
    value_key: str = "value"
    reference_key: str = "reference"
    flags_key: str = "flags"

    def __post_init__(self) -> None:
        import_target(self.target)  # eager: unreachable callables fail at registration
        if not isinstance(self.axis_kwargs, Mapping) or not self.axis_kwargs:
            raise ApplicabilityConfigError("evaluator.axis_kwargs must map at least one axis")
        clash = set(self.kwargs) & set(self.axis_kwargs.values())
        if clash:
            raise ApplicabilityConfigError(f"axis kwargs collide with fixed kwargs: {sorted(clash)}")

    def call(self, point: Mapping[str, float]) -> Any:
        function = import_target(self.target)
        kwargs = dict(self.kwargs)
        for axis_name, keyword in self.axis_kwargs.items():
            if axis_name not in point:
                raise ApplicabilityConfigError(f"point is missing axis {axis_name!r}")
            kwargs[keyword] = point[axis_name]
        return function(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "kwargs": dict(self.kwargs),
                "axis_kwargs": dict(self.axis_kwargs), "value_key": self.value_key,
                "reference_key": self.reference_key, "flags_key": self.flags_key}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "EvaluatorSpec":
        unknown = sorted(set(mapping) - set(EVALUATOR_KEYS))
        if unknown:
            raise ApplicabilityConfigError(f"unknown evaluator keys: {unknown}")
        if "target" not in mapping:
            raise ApplicabilityConfigError("evaluator.target is required")
        return cls(**dict(mapping))


@dataclass(frozen=True)
class DeclaredAxis:
    """The human-written applicability statement for one axis (the baseline)."""

    low: float | None = None
    high: float | None = None
    slack: float = 0.0
    source: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.low is None and self.high is None:
            raise ApplicabilityConfigError("declared axis needs at least one bound")
        if self.low is not None:
            object.__setattr__(self, "low", _number(self.low, "declared.low"))
        if self.high is not None:
            object.__setattr__(self, "high", _number(self.high, "declared.high"))
        object.__setattr__(self, "slack", _number(self.slack, "declared.slack"))
        if self.slack < 0.0:
            raise ApplicabilityConfigError("declared.slack must be >= 0")
        if self.low is not None and self.high is not None and not self.low < self.high:
            raise ApplicabilityConfigError("declared.low must be < declared.high")

    def to_dict(self) -> dict[str, Any]:
        return {"low": self.low, "high": self.high, "slack": self.slack,
                "source": self.source, "note": self.note}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "DeclaredAxis":
        unknown = sorted(set(mapping) - set(DECLARED_KEYS))
        if unknown:
            raise ApplicabilityConfigError(f"unknown declared keys: {unknown}")
        return cls(**dict(mapping))


@dataclass(frozen=True)
class InteractionConfig:
    """Knobs of the joint (multi-axis) probe; all values are configuration data.

    ``enabled``
        probe axis pairs jointly at all (ignored for single-axis capabilities);
    ``samples``
        lattice points per axis in the joint scan (``>= 5``), the cell midpoints
        are added on top of it, so the detection resolution is half a spacing;
    ``max_pairs``
        how many axis pairs to scan (declaration order), so the cost of a
        capability with many axes stays bounded and reproducible;
    ``max_rays``
        how many witness rays are bisected for the reported joint boundary (the
        conservative boxes are kept for *every* witness; the bisection is the
        expensive part, so it is capped and the cap is recorded).
    """

    enabled: bool = True
    samples: int = 13
    max_pairs: int = 3
    max_rays: int = 24

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ApplicabilityConfigError("interaction.enabled must be a bool")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 5:
            raise ApplicabilityConfigError("interaction.samples must be an int >= 5")
        if isinstance(self.max_pairs, bool) or not isinstance(self.max_pairs, int) \
                or self.max_pairs < 1:
            raise ApplicabilityConfigError("interaction.max_pairs must be an int >= 1")
        if isinstance(self.max_rays, bool) or not isinstance(self.max_rays, int) \
                or self.max_rays < 0:
            raise ApplicabilityConfigError("interaction.max_rays must be an int >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "samples": self.samples,
                "max_pairs": self.max_pairs, "max_rays": self.max_rays}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "InteractionConfig":
        if mapping is None:
            return cls()
        if not isinstance(mapping, Mapping):
            raise ApplicabilityConfigError("interaction must be a mapping")
        unknown = sorted(set(mapping) - {"enabled", "samples", "max_pairs", "max_rays"})
        if unknown:
            raise ApplicabilityConfigError(f"unknown interaction keys: {unknown}")
        return cls(**dict(mapping))


@dataclass(frozen=True)
class CapabilitySpec:
    """A registered capability: axes + evaluator + failure signals + declaration."""

    capability_id: str
    axes: Sequence[AxisSpec]
    evaluator: EvaluatorSpec
    policy: FailurePolicy
    declared: Mapping[str, DeclaredAxis] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.capability_id, str) or not self.capability_id.strip():
            raise ApplicabilityConfigError("capability_id must be a non-empty string")
        object.__setattr__(self, "axes", tuple(self.axes))
        if len(self.axes) < 1:
            raise ApplicabilityConfigError(f"{self.capability_id}: at least one axis is required")
        names = [a.name for a in self.axes]
        if len(set(names)) != len(names):
            raise ApplicabilityConfigError(f"{self.capability_id}: duplicate axis names {names}")
        driven = set(self.evaluator.axis_kwargs)
        if driven != set(names):
            raise ApplicabilityConfigError(
                f"{self.capability_id}: evaluator drives {sorted(driven)} but axes are {sorted(names)}")
        unknown = sorted(set(self.declared) - set(names))
        if unknown:
            raise ApplicabilityConfigError(f"{self.capability_id}: declared axes {unknown} do not exist")

    def axis(self, name: str) -> AxisSpec:
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise KeyError(name)

    def nominal_point(self) -> dict[str, float]:
        return {axis.name: axis.nominal_value() for axis in self.axes}

    def digest_material(self) -> dict[str, Any]:
        """Everything that shapes the envelope (used for the provenance digest)."""
        return {
            "capability_id": self.capability_id,
            "axes": [axis.to_dict() for axis in self.axes],
            "evaluator": self.evaluator.to_dict(),
            "policy": self.policy.to_dict(),
            "declared": {k: v.to_dict() for k, v in sorted(self.declared.items())},
            "interaction": self.interaction.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        material = self.digest_material()
        material["labels"] = dict(self.labels)
        return material


class CapabilityRegistry:
    """In-memory registry of capability declarations loaded from configuration."""

    def __init__(self, capabilities: Iterable[CapabilitySpec] = ()) -> None:
        self._items: dict[str, CapabilitySpec] = {}
        for spec in capabilities:
            self.register(spec)

    def register(self, spec: CapabilitySpec) -> CapabilitySpec:
        if spec.capability_id in self._items:
            raise ApplicabilityConfigError(f"duplicate capability_id {spec.capability_id!r}")
        self._items[spec.capability_id] = spec
        return spec

    def get(self, capability_id: str) -> CapabilitySpec:
        try:
            return self._items[capability_id]
        except KeyError:
            raise KeyError(f"unknown capability {capability_id!r}") from None

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterable[CapabilitySpec]:
        return iter(sorted(self._items.values(), key=lambda s: s.capability_id))

    def to_dict(self) -> dict[str, Any]:
        return {"capabilities": [spec.to_dict() for spec in self]}

    @classmethod
    def from_config(cls, mapping: Mapping[str, Any]) -> "CapabilityRegistry":
        if not isinstance(mapping, Mapping):
            raise ApplicabilityConfigError("configuration must be a mapping")
        unknown = sorted(set(mapping) - {"schema_version", "capabilities", "labels"})
        if unknown:
            raise ApplicabilityConfigError(f"unknown configuration keys: {unknown}")
        if mapping.get("schema_version") != 1:
            raise ApplicabilityConfigError("configuration schema_version must be 1")
        raw = mapping.get("capabilities")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
            raise ApplicabilityConfigError("configuration must declare a non-empty capabilities list")
        specs = []
        for entry in raw:
            if not isinstance(entry, Mapping):
                raise ApplicabilityConfigError("each capability must be a mapping")
            allowed = {"capability_id", "axes", "evaluator", "policy", "declared", "labels",
                       "interaction"}
            unknown = sorted(set(entry) - allowed)
            if unknown:
                raise ApplicabilityConfigError(f"unknown capability keys: {unknown}")
            for required in ("capability_id", "axes", "evaluator", "policy"):
                if required not in entry:
                    raise ApplicabilityConfigError(f"capability entry is missing {required!r}")
            axes = [AxisSpec.from_mapping(a) for a in entry["axes"]]
            declared = {name: DeclaredAxis.from_mapping(d)
                        for name, d in (entry.get("declared") or {}).items()}
            specs.append(CapabilitySpec(
                capability_id=entry["capability_id"],
                axes=axes,
                evaluator=EvaluatorSpec.from_mapping(entry["evaluator"]),
                policy=FailurePolicy.from_mapping(entry["policy"]),
                declared=declared,
                labels=dict(entry.get("labels") or {}),
                interaction=InteractionConfig.from_mapping(entry.get("interaction")),
            ))
        return cls(specs)

    @classmethod
    def from_config_file(cls, path: str | Path) -> "CapabilityRegistry":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_config(json.loads(text))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"CapabilityRegistry({list(self.ids())})"