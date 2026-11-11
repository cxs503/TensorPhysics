"""Boundary marching: 1-D windows, 2-D boundary tracing, lattice snapping.

Two marching levels share one primitive -- the root refinement of
:mod:`tensorphysics.envelope.roots`:

1-D
    The span of a single axis is sampled; every constraint that changes sign
    between two nodes is bracketed and its root refined to machine precision.
    The refined roots partition the span into segments whose feasibility is
    decided by one evaluation at the segment midpoint.  Because the boundaries
    are *refined roots* and not grid nodes, a boundary is located independently
    of the sampling step (measured order ~2 for the interpolated boundary, see
    :mod:`tensorphysics.envelope.contracts`), and several separate feasible
    intervals (multi-connected 1-D windows) fall out of the same partition.

2-D
    Call the 1-D primitive line by line along ``v`` for a ladder of ``u``
    positions.  Feasible intervals of neighbouring lines are linked into
    components by interval overlap (union-find), so two disjoint feasible
    regions in the plane stay two components instead of fusing into one
    bounding box.  A component's area is the shoelace area of the loop formed by
    its lower branch run forwards and its upper branch run backwards, which is a
    second-order estimate of the enclosed area -- unlike a cell count, which is
    first order and biased.

Discrete feasibility is *not* marched: a lattice has no boundary.  It is applied
as a snapping step after the continuous window exists, and reported separately,
so a continuous boundary can never be silently presented as an attainable
setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from .case import AxesSpec
from .constraints import Constraint, ConstraintSet, EvalCounter, Lattice
from .roots import DEFAULT_MAXITER, DEFAULT_RTOL, RootResult, bracket, linspace, refine_bracket
from .subgrid import ResolutionLedger, ResolutionPolicy, Span, resolve as resolve_spans

__all__ = [
    "BoundaryInterval",
    "Component",
    "DiscreteWindow",
    "PlaneWindow",
    "RootPoint",
    "SettingRun",
    "Window1D",
    "march_plane",
    "snap_lattice",
    "window_1d",
]

MERGE_FRACTION = 1e-12  # roots closer than this fraction of the span are one root
ZERO_FRACTION = 1e-12  # |margin| below this fraction of the grid scale is zero
TOUCH_FRACTION = 1e-9  # interval overlap tolerance, fraction of the span


@dataclass(frozen=True)
class RootPoint:
    x: float
    drivers: Tuple[str, ...]
    residual: float
    converged: bool
    iterations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "drivers": list(self.drivers),
            "residual": self.residual,
            "converged": self.converged,
            "iterations": self.iterations,
        }


@dataclass(frozen=True)
class BoundaryInterval:
    lo: float
    hi: float
    lo_drivers: Tuple[str, ...]
    hi_drivers: Tuple[str, ...]
    lo_converged: bool = True
    hi_converged: bool = True

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def to_dict(self) -> dict[str, Any]:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "width": self.width,
            "lo_drivers": list(self.lo_drivers),
            "hi_drivers": list(self.hi_drivers),
            "lo_converged": self.lo_converged,
            "hi_converged": self.hi_converged,
        }


@dataclass(frozen=True)
class Window1D:
    axis: AxesSpec
    intervals: Tuple[BoundaryInterval, ...]
    roots: Tuple[RootPoint, ...]
    n_evals: int
    attempts: int
    verified: bool
    span_truncated: bool = False
    resolution: ResolutionLedger | None = None

    @property
    def width(self) -> float:
        return sum(interval.width for interval in self.intervals)

    @property
    def feasible(self) -> bool:
        return bool(self.intervals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis.to_dict(),
            "intervals": [i.to_dict() for i in self.intervals],
            "roots": [r.to_dict() for r in self.roots],
            "width": self.width,
            "n_evals": self.n_evals,
            "attempts": self.attempts,
            "verified": self.verified,
            "span_truncated": self.span_truncated,
            "resolution": None if self.resolution is None else self.resolution.to_dict(),
        }


@dataclass(frozen=True)
class SettingRun:
    interval_index: int
    lo: float
    hi: float
    count: int
    step: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval_index": self.interval_index,
            "lo": self.lo,
            "hi": self.hi,
            "count": self.count,
            "step": self.step,
        }


@dataclass(frozen=True)
class DiscreteWindow:
    axis: str
    runs: Tuple[SettingRun, ...]
    n_settings: int
    n_evals: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "runs": [r.to_dict() for r in self.runs],
            "n_settings": self.n_settings,
            "n_evals": self.n_evals,
        }


@dataclass(frozen=True)
class Component:
    index: int
    u_values: Tuple[float, ...]
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]
    lower_drivers: Tuple[Tuple[str, ...], ...]
    upper_drivers: Tuple[Tuple[str, ...], ...]
    area: float
    area_method: str
    closed: bool
    clipped: bool
    complex_topology: bool
    multi_interval_lines: int = 0

    @property
    def points(self) -> Tuple[Tuple[float, float], ...]:
        forward = [(u, lo) for u, lo in zip(self.u_values, self.lower)]
        backward = [(u, hi) for u, hi in zip(reversed(self.u_values), reversed(self.upper))]
        return tuple(forward) + tuple(backward)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "u_values": list(self.u_values),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "lower_drivers": [list(d) for d in self.lower_drivers],
            "upper_drivers": [list(d) for d in self.upper_drivers],
            "area": self.area,
            "area_method": self.area_method,
            "closed": self.closed,
            "clipped": self.clipped,
            "complex_topology": self.complex_topology,
            "multi_interval_lines": self.multi_interval_lines,
        }


@dataclass(frozen=True)
class PlaneWindow:
    axes: Tuple[AxesSpec, AxesSpec]
    components: Tuple[Component, ...]
    lines: Tuple[Tuple[BoundaryInterval, ...], ...]
    u_values: Tuple[float, ...]
    n_evals: int
    resolution: Any = None

    @property
    def area(self) -> float:
        return sum(c.area for c in self.components)

    @property
    def feasible(self) -> bool:
        return bool(self.components)

    def driver_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for component in self.components:
            for drivers in component.lower_drivers + component.upper_drivers:
                for driver in drivers:
                    counts[driver] = counts.get(driver, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": [a.to_dict() for a in self.axes],
            "u_values": list(self.u_values),
            "lines": [[i.to_dict() for i in line] for line in self.lines],
            "components": [c.to_dict() for c in self.components],
            "area": self.area,
            "n_evals": self.n_evals,
            "resolution": self.resolution,
        }


# ---------------------------------------------------------------------------
# 1-D marching
# ---------------------------------------------------------------------------

def _section_constraints(constraints: ConstraintSet) -> Tuple[Constraint, ...]:
    section = constraints.continuous_constraints
    if not section:
        raise ValueError(
            "window detection needs at least one continuous constraint; "
            "a lattice-only case has no boundary to march"
        )
    return section


def _roots_on_grid(
    constraints: Sequence[Constraint],
    params: dict[str, float],
    axis: AxesSpec,
    n: int,
    counter: EvalCounter,
    merge_tol: float,
    zero_tol: float,
) -> Tuple[RootPoint, ...]:
    nodes = linspace(axis.lo, axis.hi, n)
    found: list[RootPoint] = []
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
            found.append(
                RootPoint(
                    x=result.x,
                    drivers=(constraint.id,),
                    residual=result.residual,
                    converged=result.converged,
                    iterations=result.iterations,
                )
            )
    found.sort(key=lambda r: r.x)
    merged: list[RootPoint] = []
    for root in found:
        if merged and abs(root.x - merged[-1].x) <= merge_tol:
            previous = merged[-1]
            drivers = tuple(dict.fromkeys(previous.drivers + root.drivers))
            merged[-1] = RootPoint(
                x=0.5 * (previous.x + root.x),
                drivers=drivers,
                residual=max(abs(previous.residual), abs(root.residual)),
                converged=previous.converged and root.converged,
                iterations=max(previous.iterations, root.iterations),
            )
        else:
            merged.append(root)
    return tuple(merged)


def _edge_lookup(
    edges: Sequence[tuple[float, Tuple[str, ...], bool]], x: float, tol: float
) -> tuple[Tuple[str, ...], bool]:
    """Drivers and convergence flag of the edge that sits at ``x`` (or none)."""

    for ex, drivers, converged in edges:
        if abs(ex - x) <= tol:
            return drivers, converged
    return (), True


def _intervals_from_spans(
    spans: Sequence[Span],
    edges: Sequence[tuple[float, Tuple[str, ...], bool]],
    axis: AxesSpec,
    merge_tol: float,
) -> Tuple[BoundaryInterval, ...]:
    """Merge consecutive resolved feasible spans into boundary intervals.

    An unresolved span (a probe disagreement the budget could not settle) never
    extends a window: it terminates the interval and marks the boundary as not
    converged, so the doubt is visible in the numbers as well as the ledger.
    """

    intervals: list[BoundaryInterval] = []
    open_at: float | None = None
    open_drivers: Tuple[str, ...] = ()
    open_converged = True
    for span in spans:
        if span.feasible and span.resolved:
            if open_at is None:
                open_at = span.lo
                open_drivers, open_converged = _edge_lookup(edges, span.lo, merge_tol)
        else:
            if open_at is not None:
                hi_drivers, hi_converged = _edge_lookup(edges, span.lo, merge_tol)
                intervals.append(
                    BoundaryInterval(
                        lo=open_at,
                        hi=span.lo,
                        lo_drivers=open_drivers,
                        hi_drivers=hi_drivers,
                        lo_converged=open_converged,
                        hi_converged=hi_converged and span.resolved,
                    )
                )
                open_at = None
    if open_at is not None:
        hi_drivers, hi_converged = _edge_lookup(edges, spans[-1].hi, merge_tol)
        intervals.append(
            BoundaryInterval(
                lo=open_at,
                hi=spans[-1].hi,
                lo_drivers=open_drivers,
                hi_drivers=hi_drivers,
                lo_converged=open_converged,
                hi_converged=hi_converged,
            )
        )
    return tuple(intervals)


def _window_once(
    constraints: Sequence[Constraint],
    params: dict[str, float],
    axis: AxesSpec,
    n: int,
    counter: EvalCounter,
    policy: ResolutionPolicy | None = None,
) -> Window1D:
    merge_tol = MERGE_FRACTION * axis.width
    # scale of the constraint values on this grid, used for the "is a node a root" test
    probe = linspace(axis.lo, axis.hi, max(3, n // 4))
    scale = 0.0
    for x in probe:
        params[axis.name] = x
        for constraint in constraints:
            scale = max(scale, abs(constraint.margin(params, counter)))
    zero_tol = ZERO_FRACTION * max(scale, 1e-30)
    start = counter.count
    roots = _roots_on_grid(constraints, params, axis, n, counter, merge_tol, zero_tol)
    edges = (
        [(axis.lo, (), True)]
        + [(r.x, r.drivers, r.converged) for r in roots]
        + [(axis.hi, (), True)]
    )
    cell = axis.width / max(1, n - 1)
    probe_policy = (policy or ResolutionPolicy()).validated()
    spans, ledger = resolve_spans(
        constraints,
        params,
        axis,
        [edge[0] for edge in edges],
        probe_policy,
        counter,
        zero_tol=zero_tol,
        merge_tol=merge_tol,
        target_spacing=cell / probe_policy.samples_per_segment,
    )
    if ledger.internal_roots:
        extra = [(x, drivers, True) for x, drivers in ledger.internal_roots]
        merged = sorted(edges + extra, key=lambda item: item[0])
        deduped: list[tuple[float, Tuple[str, ...], bool]] = []
        for item in merged:
            if deduped and abs(item[0] - deduped[-1][0]) <= merge_tol:
                continue
            deduped.append(item)
        edges = deduped
    intervals = _intervals_from_spans(spans, edges, axis, merge_tol)
    truncated = bool(
        intervals
        and (
            abs(intervals[-1].hi - axis.hi) <= merge_tol
            or abs(intervals[0].lo - axis.lo) <= merge_tol
        )
    )
    return Window1D(
        axis=axis,
        intervals=intervals,
        roots=roots,
        n_evals=counter.count - start,
        attempts=1,
        verified=False,
        span_truncated=truncated,
        resolution=ledger,
    )


def _resolution_clean(*windows: Window1D) -> bool:
    """A window is only trusted when no probe disagreement was left unresolved."""

    for window in windows:
        ledger = window.resolution
        if ledger is not None and not ledger.resolved:
            return False
    return True


def _same_structure(a: Window1D, b: Window1D) -> bool:
    tol = MERGE_FRACTION * a.axis.width
    if len(a.roots) != len(b.roots) or len(a.intervals) != len(b.intervals):
        return False
    for ra, rb in zip(a.roots, b.roots):
        if abs(ra.x - rb.x) > tol:
            return False
    for ia, ib in zip(a.intervals, b.intervals):
        if abs(ia.lo - ib.lo) > tol or abs(ia.hi - ib.hi) > tol:
            return False
    return True


def window_1d(
    constraints: ConstraintSet,
    base_params: Mapping[str, Any],
    axis: AxesSpec,
    n: int = 96,
    counter: EvalCounter | None = None,
    verify: bool = True,
    max_attempts: int = 3,
    policy: ResolutionPolicy | None = None,
) -> Window1D:
    """Feasible intervals of one axis, with refined boundary roots.

    ``verify`` re-runs the whole detection on a grid twice as fine and only
    accepts the answer if the root set and the interval structure reproduce;
    otherwise the grid is refined again, up to ``max_attempts``.  A window that
    cannot be reproduced is reported with ``verified=False`` instead of being
    silently trusted.
    """

    counter = counter or EvalCounter()
    section = _section_constraints(constraints)
    params = {k: float(v) for k, v in base_params.items()}
    n_current = n
    attempts = 0
    current = _window_once(section, params, axis, n_current, counter, policy)
    if not verify:
        return current
    if not _resolution_clean(current):
        # escalating the grid would only move the probe scale without saying so;
        # a span the probes could not settle is reported, not hidden
        return current
    while attempts < max_attempts:
        attempts += 1
        finer = _window_once(section, params, axis, 2 * n_current, counter, policy)
        if _same_structure(current, finer) and _resolution_clean(current, finer):
            return Window1D(
                axis=finer.axis,
                intervals=finer.intervals,
                roots=finer.roots,
                n_evals=current.n_evals + finer.n_evals,
                attempts=attempts,
                verified=True,
                span_truncated=finer.span_truncated,
                resolution=finer.resolution,
            )
        current = finer
        n_current *= 2
    return Window1D(
        axis=current.axis,
        intervals=current.intervals,
        roots=current.roots,
        n_evals=current.n_evals,
        attempts=attempts,
        verified=False,
        span_truncated=current.span_truncated,
        resolution=current.resolution,
    )


# ---------------------------------------------------------------------------
# lattice snapping
# ---------------------------------------------------------------------------

def _lattice_candidates(lattice: Lattice, lo: float, hi: float) -> list[float]:
    return list(lattice.candidates(lo, hi))


def snap_lattice(
    constraints: ConstraintSet,
    base_params: Mapping[str, Any],
    axis: AxesSpec,
    window: Window1D,
    counter: EvalCounter | None = None,
) -> DiscreteWindow | None:
    """Attainable discrete settings inside each continuous feasible interval."""

    discrete = constraints.discrete_constraints
    if not discrete:
        return None
    counter = counter or EvalCounter()
    params = {k: float(v) for k, v in base_params.items()}
    start = counter.count
    runs: list[SettingRun] = []
    total = 0
    for index, interval in enumerate(window.intervals):
        candidates: list[float] | None = None
        for constraint in discrete:
            assert constraint.lattice is not None
            local = _lattice_candidates(constraint.lattice, interval.lo, interval.hi)
            if candidates is None:
                candidates = local
            else:
                tol = max(constraint.lattice.tolerance, 1e-12)
                candidates = [
                    value
                    for value in candidates
                    if any(abs(value - other) <= tol for other in local)
                ]
        allowed: list[float] = []
        for value in sorted(candidates or []):
            params[axis.name] = value
            if all(c.margin(params, counter) >= 0.0 for c in constraints):
                allowed.append(value)
        total += len(allowed)
        base_step = None if len(allowed) < 2 else allowed[1] - allowed[0]
        run_start = 0
        while run_start < len(allowed):
            run_end = run_start
            if base_step is not None:
                while (
                    run_end + 1 < len(allowed)
                    and abs((allowed[run_end + 1] - allowed[run_end]) - base_step)
                    <= 1e-9 * max(abs(base_step), 1.0)
                ):
                    run_end += 1
            step_value = base_step
            runs.append(
                SettingRun(
                    interval_index=index,
                    lo=allowed[run_start],
                    hi=allowed[run_end],
                    count=run_end - run_start + 1,
                    step=step_value,
                )
            )
            run_start = run_end + 1
    return DiscreteWindow(
        axis=axis.name, runs=tuple(runs), n_settings=total, n_evals=counter.count - start
    )


# ---------------------------------------------------------------------------
# 2-D marching
# ---------------------------------------------------------------------------

def _tip_block(
    u_outer: float,
    u_inner: float,
    length_outer: float,
    length_inner: float,
    start: bool,
) -> tuple[float, float, float] | None:
    """Area of an end block of a tapered region, assuming ``L^2`` is linear.

    Near a smooth end the interval length behaves like
    ``L = sqrt(a |u - u_tip|)`` (for a circular arc ``L^2`` is linear to leading
    order), so two knots are enough to locate the tip and integrate the block
    analytically instead of dropping the cap or pushing a square root through a
    trapezoid.  ``start`` selects the taper direction: at the beginning of a
    region the length grows towards the interior, at the end it shrinks.
    Returns ``(block_area, u_tip, a)`` or ``None`` when the knots do not taper.
    """

    if start:
        if not (length_inner > length_outer > 0.0):
            return None
        ratio = (length_outer * length_outer) / (length_inner * length_inner)
        u_tip = (u_outer - ratio * u_inner) / (1.0 - ratio)
        if u_tip >= u_outer:
            return None
        a = length_outer * length_outer / (u_outer - u_tip)
        return (2.0 / 3.0) * a ** 0.5 * (u_inner - u_tip) ** 1.5, u_tip, a
    if not (length_outer > length_inner > 0.0):
        return None
    ratio = (length_outer * length_outer) / (length_inner * length_inner)
    u_tip = (ratio * u_inner - u_outer) / (ratio - 1.0)
    if u_tip <= u_inner:
        return None
    a = length_outer * length_outer / (u_tip - u_outer)
    return (2.0 / 3.0) * a ** 0.5 * (u_tip - u_outer) ** 1.5, u_tip, a


def _trapezoid_area(u_list: Sequence[float], lengths: Sequence[float]) -> float:
    total = 0.0
    for i in range(len(u_list) - 1):
        total += 0.5 * (lengths[i] + lengths[i + 1]) * (u_list[i + 1] - u_list[i])
    return total


def _area_with_tips(
    u_list: Sequence[float], lengths: Sequence[float]
) -> tuple[float, str, tuple[float, ...]]:
    """Area of a closed band: analytic end blocks plus trapezoid in between."""

    m = len(u_list)
    if m < 4:
        return _trapezoid_area(u_list, lengths), "trapezoid", ()
    head = _tip_block(u_list[0], u_list[1], lengths[0], lengths[1], True)
    tail = _tip_block(u_list[-2], u_list[-1], lengths[-2], lengths[-1], False)
    if head is None or tail is None:
        return _trapezoid_area(u_list, lengths), "trapezoid", ()
    total = head[0] + tail[0]
    for i in range(1, m - 2):
        total += 0.5 * (lengths[i] + lengths[i + 1]) * (u_list[i + 1] - u_list[i])
    return total, "tip_model", (head[1], tail[1])


def _overlaps(a: BoundaryInterval, b: BoundaryInterval, tol: float) -> bool:
    return min(a.hi, b.hi) - max(a.lo, b.lo) >= -tol


def march_plane(
    constraints: ConstraintSet,
    base_params: Mapping[str, Any],
    axes: Sequence[AxesSpec],
    n_lines: int = 100,
    n_scan: int = 100,
    counter: EvalCounter | None = None,
    verify_lines: bool = False,
    policy: ResolutionPolicy | None = None,
) -> PlaneWindow:
    """Trace the boundary of every feasible region on a two-axis plane."""

    if len(axes) != 2:
        raise ValueError("march_plane needs exactly two axes")
    counter = counter or EvalCounter()
    u_axis, v_axis = axes[0], axes[1]
    u_values = linspace(u_axis.lo, u_axis.hi, n_lines)
    lines: list[tuple[BoundaryInterval, ...]] = []
    start = counter.count
    ledgers: list[ResolutionLedger] = []
    unresolved: list[dict[str, Any]] = []
    for u in u_values:
        params = {k: float(v) for k, v in base_params.items()}
        params[u_axis.name] = u
        window = window_1d(
            constraints,
            params,
            v_axis,
            n=n_scan,
            counter=counter,
            verify=verify_lines,
            max_attempts=1,
            policy=policy,
        )
        lines.append(window.intervals)
        if window.resolution is not None:
            ledgers.append(window.resolution)
            for lo, hi in window.resolution.unresolved_spans:
                unresolved.append({"u": u, "v_lo": lo, "v_hi": hi})

    # union-find over (line, interval) nodes
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for index, line in enumerate(lines):
        for k in range(len(line)):
            find((index, k))
    touch_tol = TOUCH_FRACTION * v_axis.width
    for index in range(len(lines) - 1):
        for k, left in enumerate(lines[index]):
            for j, right in enumerate(lines[index + 1]):
                if _overlaps(left, right, touch_tol):
                    union((index, k), (index + 1, j))

    clusters: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for node in list(parent):
        clusters.setdefault(find(node), []).append(node)

    components: list[Component] = []
    for cluster_index, members in enumerate(sorted(clusters.values(), key=lambda m: min(m))):
        members.sort()
        per_line: dict[int, list[int]] = {}
        for line_index, position in members:
            per_line.setdefault(line_index, []).append(position)
        ordered = sorted(per_line)
        complex_topology = any(len(per_line[i]) > 1 for i in ordered)
        u_list: list[float] = []
        lo_list: list[float] = []
        hi_list: list[float] = []
        lo_drivers: list[Tuple[str, ...]] = []
        hi_drivers: list[Tuple[str, ...]] = []
        entry_line: list[int] = []
        for line_index in ordered:
            intervals = [lines[line_index][p] for p in per_line[line_index]]
            intervals.sort(key=lambda item: item.lo)
            for item in intervals:
                u_list.append(u_values[line_index])
                lo_list.append(item.lo)
                hi_list.append(item.hi)
                lo_drivers.append(item.lo_drivers)
                hi_drivers.append(item.hi_drivers)
                entry_line.append(line_index)
        clipped = bool(
            ordered and (ordered[0] == 0 or ordered[-1] == len(lines) - 1)
        ) or any(
            abs(lo - v_axis.lo) <= touch_tol or abs(hi - v_axis.hi) <= touch_tol
            for lo, hi in zip(lo_list, hi_list)
        )
        # The area of the component is the integral, along ``u``, of the total
        # feasible ``v``-length on that line -- but the integral is *not* taken
        # over the summed lengths.  One line can cross the same connected region
        # several times (a hole, a notch, a re-entrant boundary); summing the
        # intervals first leaves the integrand with a square-root kink wherever
        # a hole opens or closes, and a trapezoid across that kink is only
        # O(h^1.5) (measured: 1.3e-3 relative area error for a ring at 100
        # lines, worse than the O(h^2) the engine promises).  The region is
        # therefore split into *shells*: the outer shell (last upper bound minus
        # first lower bound) minus one shell per gap between consecutive
        # intervals.  Every shell is a closed chord that tapers like sqrt at its
        # own ends, so the analytic end blocks of :func:`_area_with_tips` apply
        # to each of them and the area error stays second order.
        #
        # The pre-fix behaviour integrated the concatenated per-interval lengths
        # against repeated abscissae, which mixes different v-segments of the
        # same line into one trapezoid: for a ring of analytic area 1.2252 it
        # reported 0.8649 (-29.4 %) with every contract green, because no
        # machine check compared that number with anything.
        per_line_intervals: dict[int, list[tuple[float, float]]] = {}
        for line_index, lo, hi in zip(entry_line, lo_list, hi_list):
            per_line_intervals.setdefault(line_index, []).append((lo, hi))
        multi_interval_lines = sum(
            1 for i in ordered if len(per_line_intervals[i]) > 1
        )
        widest_line = max(len(v) for v in per_line_intervals.values())
        shells: list[list[tuple[float, float]]] = []
        for shell_index in range(widest_line):
            series: list[tuple[float, float]] = []
            for line_index in ordered:
                intervals_on_line = per_line_intervals[line_index]
                if shell_index == 0:
                    chord = intervals_on_line[-1][1] - intervals_on_line[0][0]
                elif len(intervals_on_line) > shell_index:
                    chord = (
                        intervals_on_line[shell_index][0]
                        - intervals_on_line[shell_index - 1][1]
                    )
                else:
                    continue
                series.append((u_values[line_index], chord))
            if series:
                shells.append(series)
        area = 0.0
        shell_methods: list[str] = []
        for shell_index, series in enumerate(shells):
            shell_u = [entry[0] for entry in series]
            shell_chord = [entry[1] for entry in series]
            if clipped or len(series) < 4:
                part, how = _trapezoid_area(shell_u, shell_chord), "trapezoid"
            else:
                part, how, _tips = _area_with_tips(shell_u, shell_chord)
            area += part if shell_index == 0 else -part
            shell_methods.append(how)
        method = (
            "shell_tip_model"
            if shell_methods and all(how == "tip_model" for how in shell_methods)
            else "shell_trapezoid"
        )
        components.append(
            Component(
                index=cluster_index,
                u_values=tuple(u_list),
                lower=tuple(lo_list),
                upper=tuple(hi_list),
                lower_drivers=tuple(lo_drivers),
                upper_drivers=tuple(hi_drivers),
                area=area,
                area_method=method,
                closed=not clipped,
                clipped=clipped,
                complex_topology=complex_topology,
                multi_interval_lines=multi_interval_lines,
            )
        )
    line_spacing = u_axis.width / max(1, len(u_values) - 1)
    single_line = sum(
        1 for c in components if len(set(c.u_values)) < 2
    )
    if ledgers:
        widths = [l.min_probed_width for l in ledgers if l.min_probed_width == l.min_probed_width]
        probe_width = min(widths) if widths else None
        resolution: dict[str, Any] | None = {
            "policy": ledgers[0].policy.to_dict(),
            "min_probed_width": probe_width,
            "line_spacing": line_spacing,
            "single_line_components": single_line,
            "area_reliable": single_line == 0,
            "guaranteed_width": (
                None if probe_width is None else max(probe_width, line_spacing)
            ),
            "lines": len(ledgers),
            "spans_tested": sum(l.spans_tested for l in ledgers),
            "suspect_spans": sum(l.suspect_spans for l in ledgers),
            "refined_spans": sum(l.refined_spans for l in ledgers),
            "unresolved_spans": unresolved,
            "budget_exhausted": any(l.budget_exhausted for l in ledgers),
            "resolved": not unresolved and not any(l.budget_exhausted for l in ledgers),
        }
    else:
        resolution = None
    return PlaneWindow(
        axes=(u_axis, v_axis),
        components=tuple(components),
        lines=tuple(lines),
        u_values=u_values,
        n_evals=counter.count - start,
        resolution=resolution,
    )