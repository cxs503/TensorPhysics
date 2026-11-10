"""Sub-grid feature resolution -- the guarantee behind "no pocket is missed".

A marching scan can only see what its samples see.  The v1 mechanism decided a
segment's feasibility from **one** sample at its midpoint, so

* a feasible pocket narrower than the node spacing was reported as ``feasible``
  over the whole span when the midpoint happened to land inside it (a false
  *positive* of arbitrary size), and
* the same pocket one node to the side was reported as ``infeasible`` -- the
  window vanished without a trace (a false *negative*).

Both verdicts carried ``contracts.ok == True``, because nothing in the pipeline
knew what scale had actually been probed.  This module fixes the mechanism and
states the scale:

**Sampling guarantee.**  The probe spacing is fixed at
``tau = (axis span / scan cells) / samples_per_segment`` -- one scan cell is
probed at ``samples_per_segment`` interior midpoints, whatever the width of the
segment between two bracketed roots.  Any feature (feasible pocket inside an
infeasible span, or infeasible gap inside a feasible one) wider than or equal to
``tau`` therefore contains at least one probe point and cannot hide.  Each span
whose probes disagree is *subdivided* (``sub_cells`` equal parts, roots refined
to machine precision on the sub-nodes) and the parts are re-probed, recursively,
until every remaining span is single-signed, the depth limit is reached, or the
evaluation budget is spent.

**The scale is a number, not a claim.**  Every result carries a
:class:`ResolutionLedger`: ``min_probed_width`` is the smallest span that was
actually sample-tested, ``suspect_spans`` counts disagreements, refined spans are
counted, and any span that could *not* be resolved is listed explicitly.  A
window that still contains an unresolved span is never silently trusted: the
contract checker fails it (``resolution:feature``).

**No domain vocabulary.**  Spans, probe counts, depth and budget are numbers; a
margin is a callable.  A case may declare ``min_feature_width`` in its
``contract.resolution`` block, and the checker then verifies that the engine
probed at least that fine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

from .constraints import Constraint, EvalCounter
from .roots import DEFAULT_MAXITER, DEFAULT_RTOL, RootResult, bracket, linspace, refine_bracket

__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_REFINE_EVALS",
    "DEFAULT_SAMPLES_PER_SEGMENT",
    "DEFAULT_SUB_CELLS",
    "ResolutionLedger",
    "ResolutionPolicy",
    "Span",
    "check_span_partition",
    "resolve",
]

DEFAULT_SAMPLES_PER_SEGMENT = 8
DEFAULT_MAX_DEPTH = 6
DEFAULT_SUB_CELLS = 8
DEFAULT_MAX_REFINE_EVALS = 2_000_000
SAMPLES_LIMIT = 64
DEPTH_LIMIT = 12
SUB_CELLS_LIMIT = 32


@dataclass(frozen=True)
class ResolutionPolicy:
    """How finely a span is probed before its verdict is believed."""

    samples_per_segment: int = DEFAULT_SAMPLES_PER_SEGMENT
    max_depth: int = DEFAULT_MAX_DEPTH
    sub_cells: int = DEFAULT_SUB_CELLS
    max_refine_evals: int = DEFAULT_MAX_REFINE_EVALS
    min_feature_width: float | None = None

    @classmethod
    def from_spec(cls, spec: Any) -> "ResolutionPolicy":
        """Parse and validate a ``contract.resolution`` block.  Domain-free."""

        if spec is None:
            return cls()
        if not isinstance(spec, Mapping):
            raise ValueError("resolution: expected a mapping")
        samples = int(spec.get("samples_per_segment", DEFAULT_SAMPLES_PER_SEGMENT))
        depth = int(spec.get("max_depth", DEFAULT_MAX_DEPTH))
        sub_cells = int(spec.get("sub_cells", DEFAULT_SUB_CELLS))
        budget = int(spec.get("max_refine_evals", DEFAULT_MAX_REFINE_EVALS))
        width = spec.get("min_feature_width")
        return cls(
            samples_per_segment=samples,
            max_depth=depth,
            sub_cells=sub_cells,
            max_refine_evals=budget,
            min_feature_width=None if width is None else float(width),
        ).validated()

    def validated(self) -> "ResolutionPolicy":
        if not (1 <= self.samples_per_segment <= SAMPLES_LIMIT):
            raise ValueError(
                f"resolution.samples_per_segment: {self.samples_per_segment} not in "
                f"[1, {SAMPLES_LIMIT}]"
            )
        if not (0 <= self.max_depth <= DEPTH_LIMIT):
            raise ValueError(
                f"resolution.max_depth: {self.max_depth} not in [0, {DEPTH_LIMIT}]"
            )
        if not (2 <= self.sub_cells <= SUB_CELLS_LIMIT):
            raise ValueError(
                f"resolution.sub_cells: {self.sub_cells} not in [2, {SUB_CELLS_LIMIT}]"
            )
        if self.max_refine_evals < 0:
            raise ValueError("resolution.max_refine_evals: must be non-negative")
        if self.min_feature_width is not None:
            if not (self.min_feature_width > 0.0):
                raise ValueError("resolution.min_feature_width: must be positive")
            if self.min_feature_width != self.min_feature_width:  # NaN
                raise ValueError("resolution.min_feature_width: must be a number")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples_per_segment": self.samples_per_segment,
            "max_depth": self.max_depth,
            "sub_cells": self.sub_cells,
            "max_refine_evals": self.max_refine_evals,
            "min_feature_width": self.min_feature_width,
        }


@dataclass(frozen=True)
class Span:
    """One partition span of the axis with the verdict its probes support."""

    lo: float
    hi: float
    feasible: bool
    resolved: bool
    depth: int = 0

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def to_dict(self) -> dict[str, Any]:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "width": self.width,
            "feasible": self.feasible,
            "resolved": self.resolved,
            "depth": self.depth,
        }


@dataclass(frozen=True)
class ResolutionLedger:
    """What the scan actually probed, and what it could not resolve."""

    policy: ResolutionPolicy
    min_probed_width: float
    spans_tested: int
    suspect_spans: int
    refined_spans: int
    internal_roots: Tuple[Tuple[float, Tuple[str, ...]], ...]
    unresolved_spans: Tuple[Tuple[float, float], ...]
    n_evals: int
    budget_exhausted: bool

    @property
    def resolved(self) -> bool:
        return not self.unresolved_spans and not self.budget_exhausted

    @property
    def guaranteed_width(self) -> float:
        """Features at least this wide cannot hide from the probes."""

        return self.min_probed_width

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "min_probed_width": self.min_probed_width,
            "guaranteed_width": self.guaranteed_width,
            "spans_tested": self.spans_tested,
            "suspect_spans": self.suspect_spans,
            "refined_spans": self.refined_spans,
            "internal_roots": [
                {"x": x, "drivers": list(d)} for x, d in self.internal_roots
            ],
            "n_internal_roots": len(self.internal_roots),
            "unresolved_spans": [list(s) for s in self.unresolved_spans],
            "resolved": self.resolved,
            "n_evals": self.n_evals,
            "budget_exhausted": self.budget_exhausted,
        }


@dataclass
class _Builder:
    policy: ResolutionPolicy
    baseline: int
    counter: EvalCounter
    min_probed_width: float = float("inf")
    spans_tested: int = 0
    suspect_spans: int = 0
    refined_spans: int = 0
    internal_roots: list[tuple[float, Tuple[str, ...]]] = field(default_factory=list)
    unresolved: list[tuple[float, float]] = field(default_factory=list)
    budget_exhausted: bool = False

    def spent(self) -> int:
        return self.counter.count - self.baseline

    def out_of_budget(self) -> bool:
        if self.spent() >= self.policy.max_refine_evals:
            self.budget_exhausted = True
            return True
        return False


def _margins_at(
    constraints: Sequence[Constraint],
    params: dict[str, float],
    axis_name: str,
    x: float,
    counter: EvalCounter,
) -> dict[str, float]:
    params[axis_name] = x
    return {constraint.id: constraint.margin(params, counter) for constraint in constraints}


def _probe_positions(lo: float, hi: float, target: float) -> list[float]:
    """Span endpoints plus interior midpoints at (at most) ``target`` spacing."""

    width = hi - lo
    if not (width > 0.0):
        return [lo]
    count = max(1, int(math.ceil(width / target)))
    step = width / count
    return [lo] + [lo + (index + 0.5) * step for index in range(count)] + [hi]


def _dedupe(
    roots: Sequence[tuple[float, Tuple[str, ...]]], tol: float
) -> list[tuple[float, Tuple[str, ...]]]:
    ordered = sorted(roots, key=lambda item: item[0])
    merged: list[tuple[float, Tuple[str, ...]]] = []
    for x, drivers in ordered:
        if merged and abs(x - merged[-1][0]) <= tol:
            previous = merged[-1]
            merged[-1] = (
                0.5 * (previous[0] + x),
                tuple(dict.fromkeys(previous[1] + drivers)),
            )
        else:
            merged.append((x, drivers))
    return merged


def _roots_on_nodes(
    constraints: Sequence[Constraint],
    params: dict[str, float],
    axis: Any,
    nodes: Sequence[float],
    counter: EvalCounter,
    zero_tol: float,
    dedupe_tol: float,
) -> list[tuple[float, Tuple[str, ...]]]:
    found: list[tuple[float, Tuple[str, ...]]] = []
    for constraint in constraints:
        def margin(x: float, _c: Constraint = constraint) -> float:
            params[axis.name] = x
            return _c.margin(params, counter)

        for a, b in bracket(margin, nodes, tol=zero_tol):
            if a == b:
                result = RootResult(a, margin(a), 0, True, "degenerate")
            else:
                result = refine_bracket(
                    margin, a, b, rtol=DEFAULT_RTOL, maxiter=DEFAULT_MAXITER
                )
            found.append((result.x, (constraint.id,)))
    return _dedupe(found, dedupe_tol)


def _scan_span(
    constraints: Sequence[Constraint],
    params: dict[str, float],
    axis: Any,
    lo: float,
    hi: float,
    depth: int,
    builder: _Builder,
    zero_tol: float,
    dedupe_tol: float,
    target: float,
) -> list[Span]:
    """Probe one span, subdividing it as far as the disagreement reaches.

    The endpoints are sampled together with the interior probes, so a span whose
    interior disagrees with its ends is bracketed and its roots refined -- a
    boundary is never left sitting on a probe position.  The interior verdict
    alone decides a span that shows no disagreement at all.
    """

    policy = builder.policy
    positions = _probe_positions(lo, hi, target)
    values = [
        _margins_at(constraints, params, axis.name, x, builder.counter)
        for x in positions
    ]
    builder.spans_tested += 1
    spacing = (hi - lo) / max(1, len(positions) - 1)
    if spacing < builder.min_probed_width:
        builder.min_probed_width = spacing
    def state(row: Mapping[str, float]) -> int:
        """+1 feasible, -1 infeasible, 0 sitting on the boundary within tol."""

        weakest = min(row.values())
        if weakest >= 0.0:
            return 0 if weakest <= zero_tol else 1
        if weakest >= -zero_tol:
            return 0  # on the boundary within the configured tolerance
        return -1

    states = [state(row) for row in values]
    interior = states[1:-1] if len(states) > 2 else states
    n_feasible = sum(1 for flag in interior if flag > 0)
    n_infeasible = sum(1 for flag in interior if flag < 0)
    uniform_interior = n_feasible > 0 and n_infeasible == 0
    empty_interior = n_infeasible > 0 and n_feasible == 0
    conflict = any(
        states[i] * states[i + 1] < 0 for i in range(len(states) - 1)
    )
    disagreement = conflict or (n_feasible > 0 and n_infeasible > 0)
    inner: list[tuple[float, Tuple[str, ...]]] = []
    if disagreement:
        for constraint in constraints:
            def margin(x: float, _c: Constraint = constraint) -> float:
                params[axis.name] = x
                return _c.margin(params, builder.counter)

            for a, b in bracket(margin, positions, tol=zero_tol):
                if a == b:
                    result = RootResult(a, margin(a), 0, True, "degenerate")
                else:
                    result = refine_bracket(
                        margin, a, b, rtol=DEFAULT_RTOL, maxiter=DEFAULT_MAXITER
                    )
                inner.append((result.x, (constraint.id,)))
        inner = _dedupe(inner, dedupe_tol)
        # samples already in hand cannot decide anything the roots did not
        if not inner and (uniform_interior or empty_interior):
            disagreement = False
    if not disagreement:
        if uniform_interior:
            return [Span(lo, hi, True, True, depth)]
        if empty_interior:
            return [Span(lo, hi, False, True, depth)]
        # a single interior probe that agrees with both ends is all there is
        return [Span(lo, hi, interior[0] > 0, True, depth)]
    builder.suspect_spans += 1
    if depth >= policy.max_depth or builder.out_of_budget() or not (hi > lo):
        builder.unresolved.append((lo, hi))
        return [Span(lo, hi, False, False, depth)]
    builder.refined_spans += 1
    builder.internal_roots.extend(inner)
    nodes = linspace(lo, hi, policy.sub_cells + 1)
    edges = sorted(dict.fromkeys(list(nodes) + [x for x, _ in inner]))
    spans: list[Span] = []
    for left, right in zip(edges, edges[1:]):
        if right <= left:
            continue
        spans.extend(
            _scan_span(
                constraints,
                params,
                axis,
                left,
                right,
                depth + 1,
                builder,
                zero_tol,
                dedupe_tol,
                target,
            )
        )
    if not spans:
        builder.unresolved.append((lo, hi))
        return [Span(lo, hi, False, False, depth)]
    return spans


def resolve(
    constraints: Sequence[Constraint],
    params: dict[str, float],
    axis: Any,
    edges: Sequence[float],
    policy: ResolutionPolicy | None = None,
    counter: EvalCounter | None = None,
    zero_tol: float = 0.0,
    merge_tol: float = 0.0,
    target_spacing: float | None = None,
) -> tuple[tuple[Span, ...], ResolutionLedger]:
    """Partition ``edges`` into probed spans, subdividing every disagreement.

    ``edges`` are the sorted grid roots plus the span ends; the returned spans
    cover the axis in order and each one carries a verdict backed by probes at
    the scale recorded in the ledger.
    """

    policy = (policy or ResolutionPolicy()).validated()
    counter = counter or EvalCounter()
    span = float(edges[-1]) - float(edges[0])
    target = target_spacing if target_spacing and target_spacing > 0.0 else span
    target = max(target, 1e-300)
    builder = _Builder(policy=policy, baseline=counter.count, counter=counter)
    dedupe_tol = max(merge_tol * 1e-3, 1e-15 * max(1.0, abs(edges[-1] - edges[0])))
    ordered = sorted(float(x) for x in edges)
    spans: list[Span] = []
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        spans.extend(
            _scan_span(
                constraints,
                params,
                axis,
                left,
                right,
                0,
                builder,
                zero_tol,
                dedupe_tol,
                target,
            )
        )
    ledger = ResolutionLedger(
        policy=policy,
        min_probed_width=(
            builder.min_probed_width
            if builder.spans_tested and builder.min_probed_width != float("inf")
            else float("nan")
        ),
        spans_tested=builder.spans_tested,
        suspect_spans=builder.suspect_spans,
        refined_spans=builder.refined_spans,
        internal_roots=tuple(builder.internal_roots),
        unresolved_spans=tuple(builder.unresolved),
        n_evals=builder.spent(),
        budget_exhausted=builder.budget_exhausted,
    )
    return tuple(spans), ledger


def check_span_partition(spans: Sequence[Span], axis: Any) -> None:
    """Structural self-check: spans tile the axis, in order, without overlap."""

    if not spans:
        raise ValueError("resolution: empty span partition")
    if abs(spans[0].lo - axis.lo) > 1e-12 * max(1.0, axis.width):
        raise ValueError("resolution: first span does not start at the axis lower end")
    if abs(spans[-1].hi - axis.hi) > 1e-12 * max(1.0, axis.width):
        raise ValueError("resolution: last span does not end at the axis upper end")
    for left, right in zip(spans, spans[1:]):
        if abs(left.hi - right.lo) > 1e-12 * max(1.0, axis.width):
            raise ValueError(
                f"resolution: spans do not tile the axis ({left.hi} != {right.lo})"
            )