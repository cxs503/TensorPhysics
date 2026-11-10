"""Case adapter: configuration -> constraint network.  No domain vocabulary.

A *case* is a JSON document.  It declares parameter axes with their spans, the
constants of the model instance, a list of load conditions (parameter sets), and
the constraints as expression trees.  This module only validates and wires that
data together; it knows nothing about units, physical meaning or which
constraint is "the" limiting one.

Schema (``tensorphysics.envelope.case.v1``)::

    {
      "schema": "tensorphysics.envelope.case.v1",
      "case_id": "case_a",
      "title": "free text",
      "mode": "interval",                       # or "plane"
      "axes": [{"name": "u", "lo": 0.0, "hi": 6.0}, ...],
      "constants": {"k": 12.0},
      "conditions": [{"label": "cond_1", "params": {"p": 11.2}}, ...],
      "constraints": [
        {"id": "c_a", "kind": "inequality", "side": "ge",
         "value": {<expression tree>},
         "active_tolerance": 1e-9,
         "contract": {"slope": {"u": 1}, "convex": true},
         "description": "free text"},
        {"id": "c_d", "kind": "discrete",
         "lattice": {"step": 250.0, "offset": 0.0, "tolerance": 1e-9},
         "value": {<expression tree>}}
      ],
      "anchor": {"u": 3.0},
      "contract": {"min_boundary_order": 1.9, "area_rel_tol": 1e-3,
                   "monotone_rel_tol": 1e-2, "convexity_eig_tolerance": 1e-6,
                   "expect": {"components": 2, "infeasible": false}}
    }

Validation is fail-closed and located: an unknown parameter name, a
non-increasing axis span, a condition that fails to provide a parameter an
expression needs, or a duplicate id all raise :class:`EnvelopeConfigError`
naming the offending path.  A case is hashed (SHA-256 of the canonical JSON) so
an artifact can be tied back to the exact configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Tuple

from .constraints import Constraint, ConstraintContract, ConstraintKind, ConstraintSet, Lattice
from .expressions import ExpressionError, compile_expression, free_names, validate_expression

__all__ = [
    "CaseError",
    "Condition",
    "EnvelopeConfigError",
    "SCHEMA",
    "WindowCase",
    "AxesSpec",
    "canonical_digest",
    "load_case",
    "case_from_dict",
]

SCHEMA = "tensorphysics.envelope.case.v1"


class EnvelopeConfigError(ValueError):
    """Configuration is structurally or semantically invalid."""


CaseError = EnvelopeConfigError


def canonical_digest(payload: Any) -> str:
    """SHA-256 over canonical JSON (sorted keys, no whitespace)."""

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AxesSpec:
    name: str
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not self.name:
            raise EnvelopeConfigError("axis name must be non-empty")
        if not (self.hi > self.lo):
            raise EnvelopeConfigError(
                f"axis {self.name!r}: span must satisfy hi > lo (got lo={self.lo}, hi={self.hi})"
            )

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def clip(self, value: float) -> float:
        return min(max(value, self.lo), self.hi)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "lo": self.lo, "hi": self.hi}


@dataclass(frozen=True)
class Condition:
    label: str
    params: Mapping[str, float]
    expectation: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {"label": self.label, "params": dict(self.params)}
        if self.expectation:
            payload["expect"] = dict(self.expectation)
        return payload


@dataclass(frozen=True)
class ContractPolicy:
    """Thresholds of the contract checker plus optional expectations."""

    min_boundary_order: float = 1.9
    area_rel_tol: float = 1e-3
    monotone_rel_tol: float = 1e-2
    convexity_eig_tolerance: float = 1e-6
    expect_components: int | None = None
    expect_intervals: int | None = None
    expect_infeasible: bool | None = None
    expect_active: Mapping[str, str] = field(default_factory=dict)

    def with_overrides(self, expectation: Mapping[str, Any]) -> "ContractPolicy":
        """Apply a condition-level expectation block on top of the case policy."""

        if not expectation:
            return self
        updates: dict[str, Any] = {}
        mapping = {
            "components": "expect_components",
            "intervals": "expect_intervals",
            "infeasible": "expect_infeasible",
            "active": "expect_active",
        }
        for key, field_name in mapping.items():
            if key in expectation:
                updates[field_name] = expectation[key]
        return replace(self, **updates) if updates else self

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_boundary_order": self.min_boundary_order,
            "area_rel_tol": self.area_rel_tol,
            "monotone_rel_tol": self.monotone_rel_tol,
            "convexity_eig_tolerance": self.convexity_eig_tolerance,
            "expect_components": self.expect_components,
            "expect_intervals": self.expect_intervals,
            "expect_infeasible": self.expect_infeasible,
            "expect_active": dict(self.expect_active),
        }


@dataclass(frozen=True)
class WindowCase:
    case_id: str
    title: str
    mode: str
    axes: Tuple[AxesSpec, ...]
    constants: Mapping[str, float]
    conditions: Tuple[Condition, ...]
    constraints: ConstraintSet
    anchor: Mapping[str, float]
    contract: ContractPolicy
    description: str = ""
    source_digest: str = ""
    extensions: Mapping[str, Any] = field(default_factory=dict)

    # -- parameter resolution ----------------------------------------------
    def condition(self, label: str | None = None) -> Condition:
        if not self.conditions:
            raise EnvelopeConfigError(f"case {self.case_id!r} declares no conditions")
        if label is None:
            return self.conditions[0]
        for condition in self.conditions:
            if condition.label == label:
                return condition
        known = [c.label for c in self.conditions]
        raise EnvelopeConfigError(f"unknown condition {label!r} (known: {known})")

    def policy_for(self, label: str | None = None) -> ContractPolicy:
        """Case policy with the condition's own expectations applied."""

        return self.contract.with_overrides(self.condition(label).expectation)

    def base_params(self, label: str | None = None) -> dict[str, float]:
        params = {k: float(v) for k, v in self.constants.items()}
        params.update({k: float(v) for k, v in self.condition(label).params.items()})
        return params

    def params_with(self, label: str | None, **axes: float) -> dict[str, float]:
        params = self.base_params(label)
        for axis in self.axes:
            if axis.name in axes:
                params[axis.name] = float(axes[axis.name])
        return params

    def anchor_params(self, label: str | None = None) -> dict[str, float]:
        base = self.base_params(label)
        for axis in self.axes:
            base[axis.name] = float(self.anchor.get(axis.name, 0.5 * (axis.lo + axis.hi)))
        return base

    @property
    def window_axes(self) -> Tuple[AxesSpec, ...]:
        return self.axes

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "case_id": self.case_id,
            "title": self.title,
            "description": self.description,
            "mode": self.mode,
            "axes": [a.to_dict() for a in self.axes],
            "constants": dict(self.constants),
            "conditions": [c.to_dict() for c in self.conditions],
            "constraints": self.constraints.describe(),
            "anchor": dict(self.anchor),
            "contract": self.contract.to_dict(),
            "source_digest": self.source_digest,
            "extensions": dict(self.extensions),
        }


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _require(payload: Mapping[str, Any], key: str, where: str, kind: type, default: Any = ...):
    if key not in payload:
        if default is ...:
            raise EnvelopeConfigError(f"{where}: missing required key {key!r}")
        return default
    value = payload[key]
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind):
        raise EnvelopeConfigError(
            f"{where}.{key}: expected {kind.__name__}, got {type(value).__name__}"
        )
    return value


def _parse_lattice(spec: Mapping[str, Any], where: str) -> Lattice:
    if not isinstance(spec, Mapping):
        raise EnvelopeConfigError(f"{where}: lattice must be a mapping")
    values = tuple(float(v) for v in spec.get("values", ()) or ())
    step = spec.get("step")
    try:
        return Lattice(
            step=None if step is None else float(step),
            offset=float(spec.get("offset", 0.0)),
            values=values,
            tolerance=float(spec.get("tolerance", 0.0)),
        )
    except ValueError as exc:
        raise EnvelopeConfigError(f"{where}: {exc}") from exc


def _parse_constraint(spec: Mapping[str, Any], where: str) -> Constraint:
    if not isinstance(spec, Mapping):
        raise EnvelopeConfigError(f"{where}: constraint must be a mapping")
    cid = _require(spec, "id", where, str)
    where = f"{where}.{cid}"
    kind_text = _require(spec, "kind", where, str)
    try:
        kind = ConstraintKind(kind_text)
    except ValueError as exc:
        raise EnvelopeConfigError(
            f"{where}.kind: unknown constraint kind {kind_text!r} "
            f"(known: {[k.value for k in ConstraintKind]})"
        ) from exc
    if "value" not in spec:
        raise EnvelopeConfigError(f"{where}: missing required key 'value'")
    try:
        validate_expression(spec["value"], f"{where}.value")
        evaluator = compile_expression(spec["value"])
    except ExpressionError as exc:
        raise EnvelopeConfigError(str(exc)) from exc
    side_text = spec.get("side", "ge" if kind is not ConstraintKind.EQUALITY else "eq")
    if kind is ConstraintKind.INEQUALITY:
        if side_text not in ("ge", "le"):
            raise EnvelopeConfigError(f"{where}.side: expected 'ge' or 'le', got {side_text!r}")
        side = 1 if side_text == "ge" else -1
    else:
        side = 1
    contract_spec = spec.get("contract", {}) or {}
    if not isinstance(contract_spec, Mapping):
        raise EnvelopeConfigError(f"{where}.contract: expected a mapping")
    slope_spec = contract_spec.get("slope", {}) or {}
    if not isinstance(slope_spec, Mapping):
        raise EnvelopeConfigError(f"{where}.contract.slope: expected a mapping")
    slope: dict[str, int] = {}
    for name, sign in slope_spec.items():
        if sign not in (-1, 0, 1):
            raise EnvelopeConfigError(
                f"{where}.contract.slope.{name}: expected -1, 0 or 1, got {sign!r}"
            )
        slope[str(name)] = int(sign)
    convex = contract_spec.get("convex")
    if convex is not None and not isinstance(convex, bool):
        raise EnvelopeConfigError(f"{where}.contract.convex: expected true/false or null")
    target = spec.get("target", 0.0)
    tolerance = spec.get("tolerance", 0.0)
    active_tolerance = spec.get("active_tolerance", 0.0)
    for name, value in (("target", target), ("tolerance", tolerance),
                        ("active_tolerance", active_tolerance)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise EnvelopeConfigError(f"{where}.{name}: expected a number")
        if name != "target" and float(value) < 0.0:
            raise EnvelopeConfigError(f"{where}.{name}: must be >= 0")
    lattice = None
    if kind is ConstraintKind.DISCRETE:
        if "lattice" not in spec:
            raise EnvelopeConfigError(f"{where}: discrete constraint needs a 'lattice'")
        lattice = _parse_lattice(spec["lattice"], f"{where}.lattice")
    try:
        return Constraint(
            id=cid,
            kind=kind,
            evaluator=evaluator,
            side=side,
            target=float(target),
            tolerance=float(tolerance),
            lattice=lattice,
            active_tolerance=float(active_tolerance),
            contract=ConstraintContract(slope=slope, convex=convex),
            description=str(spec.get("description", "")),
            expression=spec["value"],
        )
    except ValueError as exc:
        raise EnvelopeConfigError(f"{where}: {exc}") from exc


def _parse_expectation(expect: Any, where: str) -> dict[str, Any]:
    """Normalise a structure/active-set expectation block."""

    if expect is None:
        return {}
    if not isinstance(expect, Mapping):
        raise EnvelopeConfigError(f"{where}.expect: expected a mapping")
    normalised: dict[str, Any] = {}
    for key in ("components", "intervals"):
        value = expect.get(key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise EnvelopeConfigError(f"{where}.expect.{key}: expected an integer or null")
        if value is not None:
            normalised[key] = int(value)
    infeasible = expect.get("infeasible")
    if infeasible is not None and not isinstance(infeasible, bool):
        raise EnvelopeConfigError(f"{where}.expect.infeasible: expected true/false or null")
    if infeasible is not None:
        normalised["infeasible"] = bool(infeasible)
    active = expect.get("active", {}) or {}
    if not isinstance(active, Mapping):
        raise EnvelopeConfigError(f"{where}.expect.active: expected a mapping")
    if active:
        normalised["active"] = {str(k): str(v) for k, v in active.items()}
    return normalised


def _parse_contract(spec: Any, where: str) -> ContractPolicy:
    if spec is None:
        return ContractPolicy()
    if not isinstance(spec, Mapping):
        raise EnvelopeConfigError(f"{where}: expected a mapping")
    expect = _parse_expectation(spec.get("expect"), where)
    components = expect.get("components")
    intervals = expect.get("intervals")
    infeasible = expect.get("infeasible")
    active = expect.get("active", {})
    try:
        return ContractPolicy(
            min_boundary_order=float(spec.get("min_boundary_order", 1.9)),
            area_rel_tol=float(spec.get("area_rel_tol", 1e-3)),
            monotone_rel_tol=float(spec.get("monotone_rel_tol", 1e-2)),
            convexity_eig_tolerance=float(spec.get("convexity_eig_tolerance", 1e-6)),
            expect_components=components,
            expect_intervals=intervals,
            expect_infeasible=infeasible,
            expect_active=dict(active),
        )
    except (TypeError, ValueError) as exc:
        raise EnvelopeConfigError(f"{where}: {exc}") from exc


def case_from_dict(payload: Mapping[str, Any], source: str = "<dict>") -> WindowCase:
    if not isinstance(payload, Mapping):
        raise EnvelopeConfigError(f"{source}: case must be a mapping")
    schema = payload.get("schema", SCHEMA)
    if schema != SCHEMA:
        raise EnvelopeConfigError(f"{source}: unsupported schema {schema!r} (expected {SCHEMA!r})")
    case_id = _require(payload, "case_id", source, str)
    where = f"{source}[{case_id}]"
    mode = _require(payload, "mode", where, str, default="interval")
    if mode not in ("interval", "plane"):
        raise EnvelopeConfigError(f"{where}.mode: expected 'interval' or 'plane', got {mode!r}")

    axes_payload = _require(payload, "axes", where, list)
    if not axes_payload:
        raise EnvelopeConfigError(f"{where}.axes: at least one axis is required")
    axes: list[AxesSpec] = []
    for index, spec in enumerate(axes_payload):
        if not isinstance(spec, Mapping):
            raise EnvelopeConfigError(f"{where}.axes[{index}]: expected a mapping")
        name = _require(spec, "name", f"{where}.axes[{index}]", str)
        lo = _require(spec, "lo", f"{where}.axes[{index}]", float)
        hi = _require(spec, "hi", f"{where}.axes[{index}]", float)
        try:
            axes.append(AxesSpec(name=name, lo=lo, hi=hi))
        except EnvelopeConfigError as exc:
            raise EnvelopeConfigError(f"{where}.axes[{index}]: {exc}") from exc
    if len({a.name for a in axes}) != len(axes):
        raise EnvelopeConfigError(f"{where}.axes: duplicate axis names")
    if mode == "interval" and len(axes) != 1:
        raise EnvelopeConfigError(
            f"{where}.axes: interval mode needs exactly one axis, got {len(axes)}"
        )
    if mode == "plane" and len(axes) != 2:
        raise EnvelopeConfigError(
            f"{where}.axes: plane mode needs exactly two axes, got {len(axes)}"
        )

    axis_names = {a.name for a in axes}
    constants_payload = payload.get("constants", {}) or {}
    if not isinstance(constants_payload, Mapping):
        raise EnvelopeConfigError(f"{where}.constants: expected a mapping")
    constants = {}
    for name, value in constants_payload.items():
        if name in axis_names:
            raise EnvelopeConfigError(f"{where}.constants.{name}: name collides with an axis")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise EnvelopeConfigError(f"{where}.constants.{name}: expected a number")
        constants[str(name)] = float(value)

    conditions_payload = _require(payload, "conditions", where, list)
    if not conditions_payload:
        raise EnvelopeConfigError(f"{where}.conditions: at least one condition is required")
    conditions: list[Condition] = []
    labels: set[str] = set()
    for index, spec in enumerate(conditions_payload):
        if not isinstance(spec, Mapping):
            raise EnvelopeConfigError(f"{where}.conditions[{index}]: expected a mapping")
        label = _require(spec, "label", f"{where}.conditions[{index}]", str)
        if label in labels:
            raise EnvelopeConfigError(f"{where}.conditions: duplicate label {label!r}")
        labels.add(label)
        params_payload = spec.get("params", {}) or {}
        if not isinstance(params_payload, Mapping):
            raise EnvelopeConfigError(f"{where}.conditions[{index}].params: expected a mapping")
        params = {}
        for name, value in params_payload.items():
            if name in axis_names:
                raise EnvelopeConfigError(
                    f"{where}.conditions[{index}].params.{name}: name collides with an axis"
                )
            if name in constants:
                raise EnvelopeConfigError(
                    f"{where}.conditions[{index}].params.{name}: name collides with a constant"
                )
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise EnvelopeConfigError(
                    f"{where}.conditions[{index}].params.{name}: expected a number"
                )
            params[str(name)] = float(value)
        expectation = _parse_expectation(
            spec.get("expect"), f"{where}.conditions[{index}]"
        )
        conditions.append(Condition(label=label, params=params, expectation=expectation))

    constraints_payload = _require(payload, "constraints", where, list)
    if not constraints_payload:
        raise EnvelopeConfigError(f"{where}.constraints: at least one constraint is required")
    constraints = tuple(
        _parse_constraint(spec, f"{where}.constraints[{index}]")
        for index, spec in enumerate(constraints_payload)
    )
    try:
        constraint_set = ConstraintSet(constraints)
    except ValueError as exc:
        raise EnvelopeConfigError(f"{where}.constraints: {exc}") from exc

    # every free name must resolve for *every* condition: fail closed, located
    known_common = set(constants) | axis_names
    for index, spec in enumerate(constraints_payload):
        try:
            needed = free_names(spec["value"])
        except ExpressionError as exc:  # pragma: no cover - validated above
            raise EnvelopeConfigError(f"{where}.constraints[{index}]: {exc}") from exc
        missing_common = needed - known_common - set(
            name for condition in conditions for name in condition.params
        )
        if missing_common:
            raise EnvelopeConfigError(
                f"{where}.constraints[{index}] ({spec['id']}): unknown parameter(s) "
                f"{sorted(missing_common)}"
            )
        for condition in conditions:
            available = known_common | set(condition.params)
            missing = needed - available
            if missing:
                raise EnvelopeConfigError(
                    f"{where}.conditions[{condition.label}]: constraint {spec['id']!r} needs "
                    f"parameter(s) {sorted(missing)} not provided"
                )

    anchor_payload = payload.get("anchor", {}) or {}
    if not isinstance(anchor_payload, Mapping):
        raise EnvelopeConfigError(f"{where}.anchor: expected a mapping")
    anchor = {}
    for name, value in anchor_payload.items():
        if name not in axis_names:
            raise EnvelopeConfigError(f"{where}.anchor.{name}: not a declared axis")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise EnvelopeConfigError(f"{where}.anchor.{name}: expected a number")
        anchor[str(name)] = float(value)
    for axis in axes:
        if axis.name in anchor and not (axis.lo <= anchor[axis.name] <= axis.hi):
            raise EnvelopeConfigError(
                f"{where}.anchor.{axis.name}: outside the declared span "
                f"[{axis.lo}, {axis.hi}]"
            )

    contract = _parse_contract(payload.get("contract"), f"{where}.contract")

    explicit_slopes = [
        (c.id, name) for c in constraints for name, sign in c.contract.slope.items() if sign != 0
    ]
    for cid, name in explicit_slopes:
        if name not in axis_names:
            raise EnvelopeConfigError(
                f"{where}.constraints.{cid}.contract.slope.{name}: not a declared axis"
            )

    extensions = payload.get("extensions", {}) or {}
    if not isinstance(extensions, Mapping):
        raise EnvelopeConfigError(f"{where}.extensions: expected a mapping")
    for key in extensions:
        if not isinstance(key, str):
            raise EnvelopeConfigError(f"{where}.extensions: keys must be strings")

    return WindowCase(
        case_id=case_id,
        title=str(payload.get("title", case_id)),
        mode=mode,
        axes=tuple(axes),
        constants=constants,
        conditions=tuple(conditions),
        constraints=constraint_set,
        anchor=anchor,
        contract=contract,
        description=str(payload.get("description", "")),
        source_digest=canonical_digest(payload),
        extensions=dict(extensions),
    )


def load_case(path: str | pathlib.Path) -> WindowCase:
    path = pathlib.Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EnvelopeConfigError(f"{path}: invalid JSON ({exc})") from exc
    return case_from_dict(payload, source=str(path))