"""Contract checker: is the computed window *believable*, not merely computed?

Five independent checks, each of which can fail on its own:

``monotonicity``   a constraint declared to move in one direction along an axis
                   must not bend back.  Central differences at several stations
                   give the sign and a fold count; a declared-but-absent
                   response also fails, so a declaration can never be silently
                   ignored.
``convexity``      a constraint declared to define a convex feasible set
                   (``{margin >= 0}`` convex ``<=>`` margin Hessian negative
                   semidefinite) is verified through a finite-difference Hessian
                   and its eigenvalues, per sample point.
``boundary order`` the interpolated boundary must converge at the declared order
                   (default 1.9) under grid refinement.  This is the check that
                   distinguishes "the boundary moved because the physics moved"
                   from "the boundary moved because the grid moved".
``infeasible semantics``
                   an empty window must be reported as infeasible with a
                   nearest-feasible residual.  No boundary, no area and no
                   extrapolated number is allowed to leak out of an empty
                   window.
``structure``      the number of feasible intervals / components and the
                   identity of the binding constraint on each side can be
                   declared in the configuration and are then verified, which
                   turns the active-set report into a falsifiable statement.

Every check returns :class:`CheckOutcome` with its own metrics dictionary, so a
report cites numbers instead of adjectives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Tuple

from .case import AxesSpec, ContractPolicy
from .constraints import ConstraintSet, EvalCounter
from .grid import NearestPoint

__all__ = [
    "CheckOutcome",
    "ContractReport",
    "boundary_order",
    "check_active_expectation",
    "check_area_consistency",
    "check_boundary_order",
    "check_convexity",
    "check_infeasible_semantics",
    "check_interval_expectation",
    "check_monotonicity",
    "measure_component_boundary_error",
    "symmetric_eigenvalues",
]

REL_STEP = 1e-4  # finite-difference step as a fraction of the axis span


@dataclass(frozen=True)
class CheckOutcome:
    id: str
    kind: str
    passed: bool
    detail: str
    metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "passed": self.passed,
            "detail": self.detail,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class ContractReport:
    outcomes: Tuple[CheckOutcome, ...]

    @property
    def ok(self) -> bool:
        return all(outcome.passed for outcome in self.outcomes)

    @property
    def failed(self) -> Tuple[str, ...]:
        return tuple(o.id for o in self.outcomes if not o.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failed": list(self.failed),
            "outcomes": [o.to_dict() for o in self.outcomes],
        }

    @staticmethod
    def merge(*reports: "ContractReport") -> "ContractReport":
        outcomes: list[CheckOutcome] = []
        for report in reports:
            outcomes.extend(report.outcomes)
        return ContractReport(outcomes=tuple(outcomes))


# ---------------------------------------------------------------------------
# linear algebra helper
# ---------------------------------------------------------------------------

def symmetric_eigenvalues(matrix: Sequence[Sequence[float]], sweeps: int = 60) -> Tuple[float, ...]:
    """Eigenvalues of a small symmetric matrix by cyclic Jacobi rotations."""

    n = len(matrix)
    a = [[float(matrix[i][j]) for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off = max(off, abs(a[i][j]))
        if off <= 1e-300:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p][q]) <= 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
    return tuple(sorted(a[i][i] for i in range(n)))


# ---------------------------------------------------------------------------
# monotonicity
# ---------------------------------------------------------------------------

def check_monotonicity(
    constraints: ConstraintSet,
    base_params: Mapping[str, Any],
    axes: Sequence[AxesSpec],
    policy: ContractPolicy,
    counter: EvalCounter | None = None,
    stations: int = 5,
) -> ContractReport:
    counter = counter or EvalCounter()
    outcomes: list[CheckOutcome] = []
    for constraint in constraints:
        declared = {k: v for k, v in constraint.contract.slope.items() if v != 0}
        for axis in axes:
            if axis.name not in declared:
                continue
            sign = declared[axis.name]
            step = REL_STEP * axis.width
            derivatives: list[float] = []
            station_values: list[float] = []
            for index in range(stations):
                fraction = 0.15 + 0.7 * index / max(1, stations - 1)
                value = axis.lo + fraction * axis.width
                params = dict(base_params)
                for other in axes:
                    params.setdefault(other.name, 0.5 * (other.lo + other.hi))
                params[axis.name] = value - step
                minus = constraint.margin(params, counter)
                params[axis.name] = value + step
                plus = constraint.margin(params, counter)
                derivatives.append((plus - minus) / (2.0 * step))
                station_values.append(value)
            scale = max(abs(d) for d in derivatives) if derivatives else 0.0
            reference = scale if scale > 0.0 else 1.0
            deviations = [max(0.0, -sign * d) / reference for d in derivatives]
            signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in derivatives]
            # a fold is a sign change between consecutive *non-stationary*
            # stations; a station sitting on a stationary point is not a fold
            moving = [sign for sign in signs if sign != 0]
            folds = sum(1 for a, b in zip(moving, moving[1:]) if a != b)
            worst = max(deviations) if deviations else 0.0
            passed = worst <= policy.monotone_rel_tol and scale > 0.0
            detail = (
                f"declared slope {sign:+d} along {axis.name!r}; worst opposite response "
                f"{worst:.3e} of the largest observed slope"
                if scale > 0.0
                else f"declared slope {sign:+d} along {axis.name!r} but the response is flat"
            )
            outcomes.append(
                CheckOutcome(
                    id=f"monotone:{constraint.id}:{axis.name}",
                    kind="monotonicity",
                    passed=passed,
                    detail=detail,
                    metrics={
                        "declared_slope": sign,
                        "stations": list(station_values),
                        "derivatives": derivatives,
                        "slope_scale": scale,
                        "worst_opposite_fraction": worst,
                        "fold_count": folds,
                        "tolerance": policy.monotone_rel_tol,
                    },
                )
            )
    return ContractReport(outcomes=tuple(outcomes))


# ---------------------------------------------------------------------------
# convexity
# ---------------------------------------------------------------------------

def check_convexity(
    constraints: ConstraintSet,
    base_params: Mapping[str, Any],
    axes: Sequence[AxesSpec],
    policy: ContractPolicy,
    counter: EvalCounter | None = None,
    stations: int = 3,
) -> ContractReport:
    counter = counter or EvalCounter()
    outcomes: list[CheckOutcome] = []
    if len(axes) > 4:
        return ContractReport(outcomes=())
    for constraint in constraints:
        if constraint.contract.convex is not True:
            continue
        worst_max_eigen = -math.inf
        worst_point: list[float] | None = None
        eigenvalues_at_points: list[list[float]] = []
        steps = [REL_STEP * axis.width for axis in axes]
        grid_points: list[list[float]] = []
        for index in range(stations):
            fraction = 0.3 + 0.4 * index / max(1, stations - 1)
            grid_points.append([axis.lo + fraction * axis.width for axis in axes])
        for coords in grid_points:
            base = dict(base_params)
            for axis, value in zip(axes, coords):
                base[axis.name] = value
            n = len(axes)
            hessian = [[0.0] * n for _ in range(n)]

            def value_at(offset: Sequence[float]) -> float:
                params = dict(base)
                for axis, delta in zip(axes, offset):
                    params[axis.name] = params[axis.name] + delta
                return constraint.margin(params, counter)

            for i in range(n):
                for j in range(i, n):
                    hi_i = [0.0] * n
                    hi_j = [0.0] * n
                    hi_i[i] = steps[i]
                    hi_j[j] = steps[j]
                    f_pp = value_at([hi_i[k] + hi_j[k] for k in range(n)])
                    f_pm = value_at([hi_i[k] - hi_j[k] for k in range(n)])
                    f_mp = value_at([-hi_i[k] + hi_j[k] for k in range(n)])
                    f_mm = value_at([-hi_i[k] - hi_j[k] for k in range(n)])
                    entry = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
                    hessian[i][j] = entry
                    hessian[j][i] = entry
            eigenvalues = symmetric_eigenvalues(hessian)
            eigenvalues_at_points.append(list(eigenvalues))
            if eigenvalues[-1] > worst_max_eigen:
                worst_max_eigen = eigenvalues[-1]
                worst_point = list(coords)
        magnitude = max(
            (abs(v) for point in eigenvalues_at_points for v in point), default=0.0
        )
        threshold = policy.convexity_eig_tolerance * max(magnitude, 1.0)
        passed = worst_max_eigen <= threshold
        outcomes.append(
            CheckOutcome(
                id=f"convex:{constraint.id}",
                kind="convexity",
                passed=passed,
                detail=(
                    "feasible set {margin >= 0} convex <=> margin Hessian negative semidefinite; "
                    f"max eigenvalue over {len(grid_points)} station(s) = {worst_max_eigen:.3e} "
                    f"(threshold {threshold:.3e})"
                ),
                metrics={
                    "max_eigenvalue": worst_max_eigen,
                    "worst_point": worst_point,
                    "eigenvalues": eigenvalues_at_points,
                    "threshold": threshold,
                },
            )
        )
    return ContractReport(outcomes=tuple(outcomes))


# ---------------------------------------------------------------------------
# boundary order
# ---------------------------------------------------------------------------

def measure_component_boundary_error(
    component: Any,
    analytic_lower: Callable[[float], float],
    analytic_upper: Callable[[float], float],
) -> float:
    """Largest boundary-position error of the *interpolated* boundary.

    The knots themselves sit on refined roots, so a knot error would only
    measure the root solver.  The question the criterion asks is how well the
    reconstructed boundary represents the true one between the knots, so the
    error is evaluated at the midpoint of every chord.
    """

    us = component.u_values
    worst = 0.0
    for i in range(len(us) - 1):
        u_mid = 0.5 * (us[i] + us[i + 1])
        lower = 0.5 * (component.lower[i] + component.lower[i + 1])
        upper = 0.5 * (component.upper[i] + component.upper[i + 1])
        worst = max(worst, abs(lower - analytic_lower(u_mid)), abs(upper - analytic_upper(u_mid)))
    return worst


def boundary_order(step_sizes: Sequence[float], errors: Sequence[float]) -> Tuple[float, ...]:
    """Observed convergence order between consecutive refinements."""

    if len(step_sizes) != len(errors):
        raise ValueError("step_sizes and errors must have the same length")
    orders: list[float] = []
    floor = 1e-300
    for k in range(len(errors) - 1):
        e0, e1 = max(errors[k], floor), max(errors[k + 1], floor)
        h0, h1 = step_sizes[k], step_sizes[k + 1]
        if h0 == h1:
            raise ValueError("step sizes must differ between refinements")
        orders.append(math.log(e0 / e1) / math.log(h0 / h1))
    return tuple(orders)


def check_boundary_order(
    step_sizes: Sequence[float],
    errors: Sequence[float],
    min_order: float,
    label: str = "boundary",
) -> CheckOutcome:
    orders = boundary_order(step_sizes, errors)
    passed = bool(orders) and all(o >= min_order for o in orders)
    return CheckOutcome(
        id=f"boundary_order:{label}",
        kind="boundary_order",
        passed=passed,
        detail=(
            f"observed order(s) {['%.3f' % o for o in orders]} vs required >= {min_order}"
            f"; errors {['%.3e' % e for e in errors]}"
        ),
        metrics={
            "step_sizes": list(step_sizes),
            "errors": list(errors),
            "orders": list(orders),
            "min_order": min_order,
            "min_observed_order": min(orders) if orders else None,
        },
    )


# ---------------------------------------------------------------------------
# semantics and structure
# ---------------------------------------------------------------------------

def check_infeasible_semantics(
    status: str,
    intervals: Sequence[Any],
    components: Sequence[Any],
    area: float | None,
    nearest: NearestPoint | None,
    label: str = "window",
) -> CheckOutcome:
    infeasible = status == "infeasible"
    metrics: dict[str, Any] = {
        "status": status,
        "interval_count": len(intervals),
        "component_count": len(components),
        "area": area,
        "nearest_residual": None if nearest is None else nearest.residual,
        "nearest_found": None if nearest is None else nearest.found,
    }
    problems: list[str] = []
    if infeasible:
        if intervals or components:
            problems.append("infeasible window still carries boundary structure")
        if area not in (None, 0.0):
            problems.append(f"infeasible window carries a non-zero area ({area!r})")
        if nearest is None:
            problems.append("infeasible window has no nearest-feasible report")
        else:
            if not nearest.coords:
                problems.append("nearest-feasible report has no coordinates")
            if not all(math.isfinite(v) for v in nearest.params.values()):
                problems.append("nearest-feasible report contains a non-finite parameter")
            if nearest.found:
                problems.append(
                    "nearest point is reported feasible although the window is empty"
                )
            if not nearest.found and nearest.residual <= 0.0:
                problems.append("no feasible point found but the residual is not positive")
    else:
        if not intervals and not components:
            problems.append("feasible window carries no boundary structure")
        for interval in intervals:
            if not (math.isfinite(interval.lo) and math.isfinite(interval.hi)):
                problems.append("interval endpoint is not finite")
        for component in components:
            if not math.isfinite(component.area):
                problems.append("component area is not finite")
    return CheckOutcome(
        id=f"infeasible_semantics:{label}",
        kind="infeasible_semantics",
        passed=not problems,
        detail="; ".join(problems) if problems else "semantics respected",
        metrics=metrics,
    )


def check_interval_expectation(
    intervals: Sequence[Any], policy: ContractPolicy, label: str = "window"
) -> CheckOutcome | None:
    if policy.expect_intervals is None:
        return None
    observed = len(intervals)
    return CheckOutcome(
        id=f"structure:{label}",
        kind="structure",
        passed=observed == policy.expect_intervals,
        detail=f"declared {policy.expect_intervals} feasible interval(s), observed {observed}",
        metrics={"expected": policy.expect_intervals, "observed": observed},
    )


def check_component_expectation(
    components: Sequence[Any], policy: ContractPolicy, label: str = "window"
) -> CheckOutcome | None:
    if policy.expect_components is None:
        return None
    observed = len(components)
    return CheckOutcome(
        id=f"structure:{label}",
        kind="structure",
        passed=observed == policy.expect_components,
        detail=f"declared {policy.expect_components} feasible component(s), observed {observed}",
        metrics={"expected": policy.expect_components, "observed": observed},
    )


def check_feature_resolution(
    ledgers: Sequence[Any], policy: ContractPolicy, label: str = "window"
) -> CheckOutcome | None:
    """Was the window probed at least as fine as its features demand?

    The window is only believed at the scale that was actually probed.  This
    check fails when a probe disagreement could not be resolved (an unresolved
    span, or an exhausted refinement budget), and when the case declared a
    ``min_feature_width`` coarser than the probed spacing -- in that case a
    feature of the declared size could have slipped between two probes.
    """

    present = [ledger for ledger in ledgers if ledger is not None]
    if not present:
        return None
    declared = policy.resolution.get("min_feature_width")
    declared = None if declared is None else float(declared)

    def field(ledger: Any, name: str, default: Any = None) -> Any:
        if isinstance(ledger, Mapping):
            return ledger.get(name, default)
        return getattr(ledger, name, default)

    widths = [
        field(ledger, "guaranteed_width")
        for ledger in present
        if field(ledger, "guaranteed_width") == field(ledger, "guaranteed_width")
    ]
    probed = min(widths) if widths else None
    unresolved = [span for ledger in present for span in field(ledger, "unresolved_spans")]
    exhausted = any(field(ledger, "budget_exhausted") for ledger in present)
    problems: list[str] = []
    if unresolved:
        problems.append(
            f"{len(unresolved)} span(s) still show a probe disagreement: "
            + ", ".join(f"[{lo:.9g}, {hi:.9g}]" for lo, hi in unresolved[:4])
        )
    if exhausted:
        problems.append("refinement budget exhausted before every span was resolved")
    if declared is not None and probed is not None and probed > declared:
        problems.append(
            f"probed spacing {probed:.9g} is coarser than the declared minimum "
            f"feature width {declared:.9g}"
        )
    return CheckOutcome(
        id=f"resolution:{label}",
        kind="resolution",
        passed=not problems,
        detail="; ".join(problems) if problems else "probed scale covers the declared features",
        metrics={
            "min_probed_width": probed,
            "line_spacing": field(present[0], "line_spacing"),
            "samples_per_segment": policy.resolution_policy().samples_per_segment,
            "sub_cells": policy.resolution_policy().sub_cells,
            "max_depth": policy.resolution_policy().max_depth,
            "suspect_spans": sum(field(ledger, "suspect_spans") for ledger in present),
            "refined_spans": sum(field(ledger, "refined_spans") for ledger in present),
            "unresolved_spans": [list(span) for span in unresolved],
            "budget_exhausted": exhausted,
            "declared_min_feature_width": declared,
        },
    )


def check_active_expectation(
    intervals: Sequence[Any], policy: ContractPolicy, label: str = "window"
) -> CheckOutcome | None:
    if not policy.expect_active:
        return None
    problems: list[str] = []
    observed: dict[str, list[str]] = {"lower": [], "upper": []}
    for side in ("lower", "upper"):
        expected = policy.expect_active.get(side)
        if expected is None:
            continue
        for index, interval in enumerate(intervals):
            drivers = interval.lo_drivers if side == "lower" else interval.hi_drivers
            observed[side].append("|".join(drivers) if drivers else "span-boundary")
            if expected not in drivers:
                problems.append(
                    f"{side} boundary of interval {index} is set by "
                    f"{list(drivers) or 'the span end'}, declared {expected!r}"
                )
    return CheckOutcome(
        id=f"active_set:{label}",
        kind="active_set",
        passed=not problems,
        detail="; ".join(problems) if problems else "declared active sets reproduced",
        metrics={"observed": observed, "declared": dict(policy.expect_active)},
    )


def check_area_consistency(
    area: float,
    reference_area: float,
    rel_tol: float,
    label: str = "area",
) -> CheckOutcome:
    denominator = abs(reference_area) if reference_area != 0.0 else 1.0
    relative = abs(area - reference_area) / denominator
    return CheckOutcome(
        id=f"area_consistency:{label}",
        kind="area_consistency",
        passed=relative <= rel_tol,
        detail=(
            f"marched area {area:.12g} vs independent reference {reference_area:.12g}: "
            f"relative difference {relative:.3e} (tolerance {rel_tol:.0e})"
        ),
        metrics={
            "area": area,
            "reference_area": reference_area,
            "relative_difference": relative,
            "tolerance": rel_tol,
        },
    )


def check_area_vs_grid(
    marched_area: float,
    coarse_area: float,
    fine_area: float,
    rel_tol: float,
    safety: float = 3.0,
    label: str = "window",
) -> CheckOutcome:
    """Marched area against an independent brute-force cell count.

    A uniform feasibility-cell count is a first-order estimate of an area: its
    own discretisation error is not known in advance, so it cannot certify a
    precision tolerance on its own.  What it *can* do is catch a marching
    blunder -- a region integrated over the wrong segment, a hole counted as
    part of the region, a missed component -- which is exactly the failure this
    check exists for (a ring region used to be reported 29.4 % low with every
    contract green).

    Decision rule, declared before looking at any number
    ----------------------------------------------------
    ``band = max(rel_tol * |A_fine|, safety * |A_fine - A_coarse|)``
    ``pass iff |A_marched - A_fine| <= band`` (with ``A = 0`` handled exactly).

    The second term is the grid's own resolution-to-resolution drift, i.e. the
    scale on which the cell count moves when its lattice is refined by two; the
    safety factor covers the fact that the drift is an estimate of the grid
    error, not a bound.  The check is therefore a *blunder detector with a
    resolution-aware band*, not a precision certificate: precision is certified
    against closed-form areas (``check_area_consistency`` with an analytic
    reference).
    """

    denominator = abs(fine_area) if fine_area != 0.0 else 1.0
    difference = abs(marched_area - fine_area)
    drift = abs(fine_area - coarse_area)
    band = max(rel_tol * denominator, safety * drift)
    both_empty = marched_area == 0.0 and fine_area == 0.0 and coarse_area == 0.0
    passed = both_empty or difference <= band
    return CheckOutcome(
        id=f"area_vs_grid:{label}",
        kind="area_vs_grid",
        passed=passed,
        detail=(
            "marched area agrees with the brute-force cell count inside the "
            "resolution-aware band"
            if passed
            else (
                f"marched area {marched_area:.12g} is {difference:.3e} away from the "
                f"cell count {fine_area:.12g}, outside the band {band:.3e} "
                f"(declared tolerance {rel_tol:.0e}, grid drift {drift:.3e}, "
                f"safety factor {safety:g})"
            )
        ),
        metrics={
            "marched_area": marched_area,
            "coarse_area": coarse_area,
            "fine_area": fine_area,
            "difference": difference,
            "grid_drift": drift,
            "band": band,
            "relative_difference": difference / denominator,
            "tolerance": rel_tol,
            "safety_factor": safety,
        },
    )


def check_infeasible_expectation(
    status: str, policy: ContractPolicy, label: str = "window"
) -> CheckOutcome | None:
    if policy.expect_infeasible is None:
        return None
    observed_infeasible = status == "infeasible"
    return CheckOutcome(
        id=f"status:{label}",
        kind="structure",
        passed=observed_infeasible == policy.expect_infeasible,
        detail=(
            f"declared infeasible={policy.expect_infeasible}, observed status {status!r}"
        ),
        metrics={"declared_infeasible": policy.expect_infeasible, "status": status},
    )