"""Stage 2b/3b: joint (multi-axis) probing of an applicability envelope.

Why this module exists
----------------------
The one-axis-at-a-time probe of stages 2/3 varies a single axis while *every
other axis sits at its nominal value*.  That is enough to find a failure that a
single parameter triggers, but it says nothing about a capability that fails only
when two parameters are unfavourable **together**::

    criterion:  |value| <= tolerance
    value     =  k * a * b          (both axes far inside their 1-D intervals)

With the nominal point in a benign corner, every 1-D scan stays inside, the
per-axis envelope certifies the full span, and the *offline* gate
(:func:`tensorphysics.applicability.artifact.gate`, which by design never
re-runs the evaluator) writes ``REAL`` for a point whose measured error is far
outside the tolerance.  That is a fail-closed violation of the very contract the
mechanism exists to enforce, and it is invisible in the reported numbers because
the artifact says ``IN``.

This module closes it without touching the per-axis results:

* a **joint lattice** on a pair of axes (the axis lattices, plus their cell
  midpoints, all other axes at nominal) is evaluated with the *same*
  :func:`~.evaluation.evaluate_point` predicate;
* a node is a **witness** when every axis places it inside the 1-D in-domain
  interval (so the offline gate would certify it) *and* the evaluation fails;
* around each witness a **conservative box** (half the local lattice spacing on
  each axis, clipped to the 1-D in-domain interval and to the searched span) is
  recorded as out of domain.  The box provably contains a failure, so
  classifying its interior ``OUT`` can never certify a bad point; the price is a
  over-refusal of at most one lattice cell, which shrinks quadratically with the
  lattice spacing (measured in the acceptance evidence);
* a **ray refinement** bisects the segment from the nearest passing node to the
  witness down to ``rel_tol`` of the segment, giving the reported boundary point
  and a confidence label (stage 3 semantics applied in the joint plane).

Everything is data driven: axis names, spans, sample counts, refinement depth and
the evaluator all come from the capability configuration.  Nothing in this file
knows what an axis means.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .axis import AxisSpec
from .capability import CapabilitySpec
from .envelope import AxisEnvelope, InteractionBox, InteractionRegion
from .evaluation import CRITERION, LAW, PATHOLOGY, evaluate_point

__all__ = [
    "scan_pair",
    "scan_interactions",
    "interaction_pairs",
    "pair_coordinates",
    "InteractionScanResult",
    "INTERACTION_CASES",
    "InteractionCase",
    "synthetic_interaction_registry",
    "run_interaction_selfcheck",
    "DEFAULT_PAIR_SAMPLES",
]

DEFAULT_PAIR_SAMPLES = 13


@dataclass(frozen=True)
class InteractionScanResult:
    """Outcome of scanning one axis pair (kept separate from the envelope)."""

    region: InteractionRegion
    anchor: Mapping[str, float]
    evaluations: int

    def to_dict(self) -> dict[str, Any]:
        return {"region": self.region.to_dict(), "anchor": dict(self.anchor),
                "evaluations": self.evaluations}


def interaction_pairs(spec: CapabilitySpec, max_pairs: int) -> list[tuple[str, str]]:
    """Axis pairs to probe: the first ``max_pairs`` pairs in declaration order.

    Deterministic by construction (declaration order, not a set iteration), so a
    stored artifact can be reproduced bit for bit.
    """
    names = [axis.name for axis in spec.axes]
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    return pairs[: max(0, int(max_pairs))]


def pair_coordinates(axis: AxisSpec, samples: int) -> tuple[float, ...]:
    """The lattice of ``axis`` re-sampled at ``samples`` points (same law)."""
    from dataclasses import replace

    return replace(axis, samples=max(5, int(samples))).coordinates()


def _midpoints(coords: Sequence[float]) -> list[float]:
    out: list[float] = []
    for a, b in zip(coords, coords[1:]):
        if b > a:
            out.append(0.5 * (a + b))
    return out


def _local_half_spacing(coords: Sequence[float], value: float) -> float:
    """Half the distance to the nearest lattice neighbour (a local cell radius)."""
    below = [c for c in coords if c < value]
    above = [c for c in coords if c > value]
    gaps: list[float] = []
    if below:
        gaps.append(value - max(below))
    if above:
        gaps.append(min(above) - value)
    if not gaps:
        return 0.0
    return 0.5 * min(gaps)


def _inside_axis(axis_envelope: AxisEnvelope, value: float) -> bool:
    """Is ``value`` inside a detected 1-D in-domain interval (closed)?"""
    return any(iv.low <= value <= iv.high for iv in axis_envelope.intervals)


@dataclass
class _Node:
    point: dict[str, float]
    ok: bool
    status: str
    reasons: tuple[str, ...]
    marginal_in: bool


def _evaluate_nodes(spec: CapabilitySpec, pair: tuple[str, str], coords: Mapping[str, Sequence[float]],
                    base: Mapping[str, float], axis_envelopes: Mapping[str, AxisEnvelope]
                    ) -> tuple[dict[tuple[float, float], _Node], int]:
    first, second = pair
    nodes: dict[tuple[float, float], _Node] = {}
    for a in coords[first]:
        for b in coords[second]:
            point = dict(base)
            point[first] = float(a)
            point[second] = float(b)
            evaluation = evaluate_point(spec, point)
            marginal_in = (_inside_axis(axis_envelopes[first], a)
                           and _inside_axis(axis_envelopes[second], b))
            nodes[(float(a), float(b))] = _Node(point=point, ok=evaluation.ok,
                                                status=evaluation.status,
                                                reasons=tuple(evaluation.reasons),
                                                marginal_in=marginal_in)
    return nodes, len(nodes)


def _failure_type(status: str) -> str:
    if status == CRITERION:
        return "criterion"
    if status == PATHOLOGY:
        return "pathology"
    if status == LAW:
        return "law"
    return "error"


def _bisect_ray(spec: CapabilitySpec, inside: Mapping[str, float], outside: Mapping[str, float],
                first: str, second: str, rel_tol: float) -> dict[str, Any]:
    """Bisect the segment ``inside -> outside`` on the point predicate."""
    lo, hi = 0.0, 1.0
    iterations = 0
    while hi - lo > rel_tol and iterations < 80:
        mid = 0.5 * (lo + hi)
        point = {k: float(inside[k]) + mid * (float(outside[k]) - float(inside[k]))
                 for k in inside}
        evaluation = evaluate_point(spec, point)
        if evaluation.ok:
            lo = mid
        else:
            hi = mid
        iterations += 1
    t = 0.5 * (lo + hi)
    boundary = {k: float(inside[k]) + t * (float(outside[k]) - float(inside[k])) for k in inside}
    converged = (hi - lo) <= rel_tol
    return {
        "point": {first: boundary[first], second: boundary[second]},
        "fraction_along_ray": t,
        "width_fraction_along_ray": hi - lo,
        "iterations": iterations,
        "converged": converged,
        "confidence": 1.0 if converged else 0.5,
    }


def scan_pair(spec: CapabilitySpec, pair: tuple[str, str], axis_envelopes: Mapping[str, AxisEnvelope],
              *, samples: int = DEFAULT_PAIR_SAMPLES, rel_tol: float = 1e-3,
              max_rays: int = 24,
              base: Mapping[str, float] | None = None) -> InteractionScanResult:
    """Probe one axis pair jointly and return its conservative out-of-domain cells.

    The pair plane is tiled by the lattice cells of the configured scan.  Each
    cell is sampled at its four corners and its centre (all other axes at
    nominal); a cell is marked out of domain when any of those samples is a
    *witness* -- a point that every 1-D interval places inside yet the evaluation
    rejects.  Because the cells tile the plane, the flagged union has no gaps
    between neighbouring witnesses; its over-refusal is at most one cell thick
    along the joint boundary and shrinks with the lattice spacing (measured in
    the acceptance evidence).
    """
    first, second = pair
    axis_a, axis_b = spec.axis(first), spec.axis(second)
    base = dict(spec.nominal_point() if base is None else base)

    lattice_a = pair_coordinates(axis_a, samples)
    lattice_b = pair_coordinates(axis_b, samples)
    centres_a = _midpoints(lattice_a)
    centres_b = _midpoints(lattice_b)
    coords = {first: tuple(sorted(set(lattice_a) | set(centres_a))),
              second: tuple(sorted(set(lattice_b) | set(centres_b)))}

    nodes, evaluations = _evaluate_nodes(spec, pair, coords, base, axis_envelopes)
    witnesses: dict[tuple[float, float], _Node] = {
        key: node for key, node in nodes.items() if node.marginal_in and not node.ok}
    passing: list[tuple[float, float, _Node]] = [
        (a, b, node) for (a, b), node in nodes.items() if node.marginal_in and node.ok]

    env_a, env_b = axis_envelopes[first], axis_envelopes[second]
    clip_a = (min(iv.low for iv in env_a.intervals), max(iv.high for iv in env_a.intervals)) \
        if env_a.intervals else (axis_a.low, axis_a.high)
    clip_b = (min(iv.low for iv in env_b.intervals), max(iv.high for iv in env_b.intervals)) \
        if env_b.intervals else (axis_b.low, axis_b.high)

    boxes: list[InteractionBox] = []
    for i in range(len(lattice_a) - 1):
        low_a_raw, high_a_raw = lattice_a[i], lattice_a[i + 1]
        for j in range(len(lattice_b) - 1):
            low_b_raw, high_b_raw = lattice_b[j], lattice_b[j + 1]
            cell_witnesses = [
                node for key, node in witnesses.items()
                if low_a_raw <= key[0] <= high_a_raw and low_b_raw <= key[1] <= high_b_raw]
            if not cell_witnesses:
                continue
            # the witness closest to the cell centre is the reported representative
            centre_a = 0.5 * (low_a_raw + high_a_raw)
            centre_b = 0.5 * (low_b_raw + high_b_raw)
            witness = min(cell_witnesses, key=lambda n: math.hypot(
                (n.point[first] - centre_a) / (axis_a.width or 1.0),
                (n.point[second] - centre_b) / (axis_b.width or 1.0)))
            low_a = max(low_a_raw, clip_a[0])
            high_a = min(high_a_raw, clip_a[1])
            low_b = max(low_b_raw, clip_b[0])
            high_b = min(high_b_raw, clip_b[1])
            if high_a < low_a or high_b < low_b:
                continue        # the 1-D intervals already rule this cell out

            boundary: dict[str, Any] | None = None
            anchor_point: dict[str, float] = dict(witness.point)
            if len(boxes) < max_rays and passing:
                anchor, best = None, math.inf
                fa = (witness.point[first] - axis_a.low) / (axis_a.width or 1.0)
                fb = (witness.point[second] - axis_b.low) / (axis_b.width or 1.0)
                for pa, pb, node in passing:
                    da = (pa - axis_a.low) / (axis_a.width or 1.0)
                    db = (pb - axis_b.low) / (axis_b.width or 1.0)
                    distance = math.hypot(da - fa, db - fb)
                    if distance < best:
                        best, anchor = distance, node
                if anchor is not None:
                    anchor_point = dict(anchor.point)
                    boundary = _bisect_ray(spec, anchor.point, witness.point, first, second,
                                           rel_tol)
                    evaluations += int(boundary["iterations"])

            boxes.append(InteractionBox(
                pair=(first, second),
                low={first: float(low_a), second: float(low_b)},
                high={first: float(high_a), second: float(high_b)},
                witness={first: float(witness.point[first]),
                         second: float(witness.point[second])},
                status=witness.status,
                failure_type=_failure_type(witness.status),
                reasons=witness.reasons,
                radius_fraction={
                    first: 0.5 * (high_a_raw - low_a_raw) / (axis_a.width or 1.0),
                    second: 0.5 * (high_b_raw - low_b_raw) / (axis_b.width or 1.0),
                },
                boundary=boundary,
                anchor={first: float(anchor_point[first]), second: float(anchor_point[second])},
            ))

    failure_types: dict[str, int] = {}
    for box in boxes:
        failure_types[box.failure_type] = failure_types.get(box.failure_type, 0) + 1
    notes: list[str] = []
    refined = sum(1 for box in boxes if box.boundary is not None)
    if witnesses:
        notes.append(f"{first}x{second}: {len(witnesses)} interaction witness(es) inside both "
                     f"1-D in-domain intervals; {len(boxes)} flagged cell(s) cover "
                     f"{_covered_fraction(boxes, axis_a, axis_b):.4f} of the searched box")
        if refined < len(witnesses):
            notes.append(f"{first}x{second}: {refined} witness ray(s) bisected "
                         f"(max_rays={max_rays}); every flagged cell is kept regardless")
    if any(box.status == LAW for box in boxes):
        notes.append("point-property law breaks are bisectable; sequence (monotonicity) laws are "
                     "not applied in the joint scan")
    spacing = {first: _max_spacing(lattice_a), second: _max_spacing(lattice_b)}
    notes.append(f"detection resolution: failure regions thinner than one cell "
                 f"({first} {spacing[first]:.6g}, {second} {spacing[second]:.6g}) can be aliased")
    region = InteractionRegion(
        pair=(first, second),
        boxes=tuple(boxes),
        witnesses=len(witnesses),
        nodes_evaluated=evaluations,
        samples=samples,
        spacing=spacing,
        failure_types=failure_types,
        notes=tuple(notes),
    )
    return InteractionScanResult(region=region, anchor=base, evaluations=evaluations)


def _max_spacing(coords: Sequence[float]) -> float:
    return max((b - a for a, b in zip(coords, coords[1:])), default=0.0)


def _covered_fraction(boxes: Iterable[InteractionBox], axis_a: AxisSpec,
                      axis_b: AxisSpec) -> float:
    """Sum of box areas over the searched box area (overlap counted once per box)."""
    total = axis_a.width * axis_b.width
    if total <= 0:
        return 0.0
    area = 0.0
    for box in boxes:
        first, second = box.pair
        area += (box.high[first] - box.low[first]) * (box.high[second] - box.low[second])
    return area / total


def scan_interactions(spec: CapabilitySpec, axis_envelopes: Mapping[str, AxisEnvelope],
                      *, enabled: bool = True, samples: int = DEFAULT_PAIR_SAMPLES,
                      max_pairs: int = 3, rel_tol: float = 1e-3
                      ) -> tuple[InteractionRegion, ...]:
    """Scan every configured axis pair of ``spec`` (empty for single-axis specs)."""
    if not enabled or len(spec.axes) < 2:
        return ()
    regions: list[InteractionRegion] = []
    for pair in interaction_pairs(spec, max_pairs):
        regions.append(scan_pair(spec, pair, axis_envelopes, samples=samples,
                                 rel_tol=rel_tol).region)
    return tuple(regions)


# ---------------------------------------------------------------------------
# synthetic black-box suite with closed-form truth for the joint failure region
# ---------------------------------------------------------------------------

_METRICS = "tensorphysics.applicability.metrics:"


@dataclass(frozen=True)
class InteractionCase:
    """A synthetic two-axis capability whose joint failure set is known exactly."""

    capability: dict
    truth: dict
    note: str = ""

    @property
    def capability_id(self) -> str:
        return self.capability["capability_id"]

    def truth_ok(self, point: Mapping[str, float]) -> bool:
        """Closed-form answer: does ``point`` satisfy the capability criterion?"""
        kind = self.truth["kind"]
        first, second = self.truth["axes"]
        a, b = float(point[first]), float(point[second])
        if kind == "product":
            value = abs(a * b)
        elif kind == "sum":
            value = abs(a + b)
        elif kind == "hyperbola":
            value = abs(a * b) / max(1e-30, a + b)
        else:  # pragma: no cover - defensive
            raise ValueError(kind)
        return value <= self.truth["tolerance"]


def _capability(capability_id: str, target: str, kwargs: dict, axis_kwargs: dict,
                axes: list[dict], criterion: dict, interaction: dict | None = None,
                policy: dict | None = None) -> dict:
    entry = {
        "capability_id": capability_id,
        "evaluator": {"target": _METRICS + target, "kwargs": kwargs,
                      "axis_kwargs": axis_kwargs},
        "axes": axes,
        "policy": dict(policy or {}, criterion=criterion),
    }
    if interaction is not None:
        entry["interaction"] = interaction
    return entry


INTERACTION_CASES: tuple[InteractionCase, ...] = (
    InteractionCase(
        _capability(
            "syn-joint-product-corner", "bilinear", {"amplitude": 1.0},
            {"u": "x", "v": "y"},
            [{"name": "u", "low": 0.5, "high": 4.0, "samples": 11, "nominal": 0.5},
             {"name": "v", "low": 0.5, "high": 8.0, "samples": 11, "nominal": 0.5}],
            {"kind": "abs", "reference": 0.0, "tolerance": 4.0},
        ),
        {"kind": "product", "axes": ("u", "v"), "tolerance": 4.0},
        "value = u*v, tolerance 4, benign nominal (0.5, 0.5): the failure region is the "
        "hyperbolic corner u*v > 4, which neither 1-D scan can see",
    ),
    InteractionCase(
        _capability(
            "syn-joint-sum-diagonal", "additive", {"amplitude": 1.0},
            {"a": "a", "b": "b"},
            [{"name": "a", "low": 0.0, "high": 6.0, "samples": 11, "nominal": 0.0},
             {"name": "b", "low": 0.0, "high": 6.0, "samples": 11, "nominal": 0.0}],
            {"kind": "abs", "reference": 0.0, "tolerance": 6.0},
        ),
        {"kind": "sum", "axes": ("a", "b"), "tolerance": 6.0},
        "value = a+b, tolerance 6, nominal (0, 0): the failure region is the half plane "
        "a+b > 6, a diagonal that lies entirely between the two 1-D intervals",
    ),
    InteractionCase(
        _capability(
            "syn-joint-ratio-hyperbola", "ratio_product", {},
            {"s": "s", "t": "t"},
            [{"name": "s", "low": 0.05, "high": 5.0, "samples": 11, "nominal": 0.05},
             {"name": "t", "low": 0.05, "high": 5.0, "samples": 11, "nominal": 0.05}],
            {"kind": "abs", "reference": 0.0, "tolerance": 1.0},
        ),
        {"kind": "hyperbola", "axes": ("s", "t"), "tolerance": 1.0},
        "value = s*t/(s+t), tolerance 1, nominal (0.05, 0.05): 1-D scans pass everywhere, "
        "while a curved band of the plane fails jointly",
    ),
)


def synthetic_interaction_registry(cases: tuple[InteractionCase, ...] = INTERACTION_CASES
                                   ) -> Any:
    from .capability import CapabilityRegistry

    return CapabilityRegistry.from_config(
        {"schema_version": 1, "capabilities": [case.capability for case in cases]})


def run_interaction_selfcheck(*, samples: int = DEFAULT_PAIR_SAMPLES, rel_tol: float = 1e-3
                              ) -> dict[str, Any]:
    """Probe every interaction case and compare the mechanism with the closed form."""
    from .probe import probe_capability

    registry = synthetic_interaction_registry()
    rows: list[dict[str, Any]] = []
    for case in INTERACTION_CASES:
        report = probe_capability(registry.get(case.capability_id),
                                  rel_tol=rel_tol, interaction_samples=samples)
        envelope = report.envelope
        row: dict[str, Any] = {"capability_id": case.capability_id, "note": case.note,
                               "pair": list(case.truth["axes"]),
                               "axes_envelopes": {name: env.to_dict()
                                                  for name, env in envelope.axes.items()},
                               "interactions": [region.to_dict()
                                                for region in envelope.interactions]}
        rows.append(row)
    return {"samples": samples, "rel_tol": rel_tol, "cases": rows}