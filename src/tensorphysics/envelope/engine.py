"""Envelope engine: case in, window + contract verdict out.

The engine owns the *order* of the five stages and the definition of a window
status.  Nothing else in the package decides what a result means:

1. resolve the condition -> parameter map (configuration);
2. march the continuous window (1-D intervals or 2-D components);
3. snap discrete feasibility onto the continuous window, reported separately;
4. if the window is empty, run the nearest-feasible search -- an empty window
   yields a residual and a point, never a boundary and never an extrapolated
   number;
5. run the contract checks and attach their verdicts to the result.

A result is therefore a triple: *numbers* (boundaries, area, active sets),
*semantics* (status, residual, discretisation), and *verdicts* (per-check
pass/fail with metrics).  Reporting only the first is exactly what this node
exists to prevent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence, Tuple

from .case import AxesSpec, ContractPolicy, WindowCase
from .constraints import EvalCounter
from .contracts import (
    CheckOutcome,
    ContractReport,
    check_active_expectation,
    check_area_consistency,
    check_boundary_order,
    check_component_expectation,
    check_convexity,
    check_infeasible_expectation,
    check_infeasible_semantics,
    check_interval_expectation,
    check_monotonicity,
    measure_component_boundary_error,
)
from .grid import NearestPoint, PlaneScan, nearest_feasible_point, scan_plane
from .marching import (
    BoundaryInterval,
    Component,
    DiscreteWindow,
    PlaneWindow,
    Window1D,
    march_plane,
    snap_lattice,
    window_1d,
)

__all__ = [
    "AnalyticReference",
    "EnvelopeEngine",
    "OrderStudy",
    "WindowResult",
    "WindowStatus",
    "order_study",
]


class WindowStatus(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class AnalyticReference:
    """Closed-form boundary of a window, used only to grade the reconstruction."""

    name: str
    lower: Callable[[float], float]
    upper: Callable[[float], float]


@dataclass(frozen=True)
class WindowResult:
    case_id: str
    condition: str
    mode: str
    status: str
    axes: Tuple[AxesSpec, ...]
    intervals: Tuple[BoundaryInterval, ...]
    components: Tuple[Component, ...]
    area: float | None
    width: float | None
    discrete: DiscreteWindow | None
    nearest: NearestPoint | None
    active_sets: Mapping[str, Any]
    contracts: ContractReport
    n_evals: int
    elapsed_s: float
    source_digest: str
    grid_check: PlaneScan | None = None
    boundary_error: float | None = None

    @property
    def ok(self) -> bool:
        return self.contracts.ok

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "condition": self.condition,
            "mode": self.mode,
            "status": self.status,
            "axes": [a.to_dict() for a in self.axes],
            "intervals": [i.to_dict() for i in self.intervals],
            "components": [c.to_dict() for c in self.components],
            "area": self.area,
            "width": self.width,
            "discrete": None if self.discrete is None else self.discrete.to_dict(),
            "nearest": None if self.nearest is None else self.nearest.to_dict(),
            "active_sets": dict(self.active_sets),
            "contracts": self.contracts.to_dict(),
            "n_evals": self.n_evals,
            "elapsed_s": self.elapsed_s,
            "source_digest": self.source_digest,
            "boundary_error": self.boundary_error,
        }
        if self.grid_check is not None:
            payload["grid_check"] = {
                "cells": self.grid_check.cells.to_dict(),
                "components_4conn": self.grid_check.feasible_components_estimate(),
                "driver_counts": dict(self.grid_check.driver_counts),
                "n_evals": self.grid_check.n_evals,
            }
        return payload


class EnvelopeEngine:
    """Runs a :class:`WindowCase`; the case carries all domain content."""

    def __init__(self, case: WindowCase, policy: ContractPolicy | None = None) -> None:
        self.case = case
        self.policy = policy or case.contract
        self._explicit_policy = policy is not None

    # -- helpers -----------------------------------------------------------
    def _axes_for_mode(self) -> Tuple[AxesSpec, ...]:
        if self.case.mode == "interval":
            if len(self.case.axes) != 1:
                raise ValueError(
                    f"case {self.case.case_id!r}: interval mode needs one axis, "
                    f"got {len(self.case.axes)}"
                )
            return (self.case.axes[0],)
        if len(self.case.axes) != 2:
            raise ValueError(
                f"case {self.case.case_id!r}: plane mode needs two axes, "
                f"got {len(self.case.axes)}"
            )
        return tuple(self.case.axes)

    def run(
        self,
        condition: str | None = None,
        n_scan: int = 96,
        n_lines: int = 100,
        n_grid_check: int = 48,
        verify: bool = True,
        analytic: AnalyticReference | None = None,
        reference_area: float | None = None,
        run_contracts: bool = True,
    ) -> WindowResult:
        started = time.perf_counter()
        counter = EvalCounter()
        # a condition may declare its own structure/active-set expectations; an
        # explicitly supplied policy always wins, so a caller can test a variant
        policy = self.policy if self._explicit_policy else self.case.policy_for(condition)
        axes = self._axes_for_mode()
        base = self.case.base_params(condition)
        contracts: list[ContractReport] = []
        grid_check: PlaneScan | None = None
        boundary_error: float | None = None

        if self.case.mode == "interval":
            axis = axes[0]
            window: Window1D = window_1d(
                self.case.constraints,
                base,
                axis,
                n=n_scan,
                counter=counter,
                verify=verify,
            )
            intervals = window.intervals
            components: Tuple[Component, ...] = ()
            discrete = snap_lattice(self.case.constraints, base, axis, window, counter)
            status = WindowStatus.FEASIBLE if intervals else WindowStatus.INFEASIBLE
            area: float | None = None
            width: float | None = window.width if intervals else None
            nearest = None
            if status is WindowStatus.INFEASIBLE:
                nearest = nearest_feasible_point(
                    self.case.constraints,
                    base,
                    axes,
                    self.case.anchor_params(condition),
                    n_per_axis=65,
                    counter=counter,
                )
            active_sets = {
                axis.name: {
                    "interval_count": len(intervals),
                    "total_width": window.width,
                    "verified": window.verified,
                    "attempts": window.attempts,
                    "truncated_by_span": window.span_truncated,
                    "lower_drivers": [
                        list(i.lo_drivers) or ["span-boundary"] for i in intervals
                    ],
                    "upper_drivers": [
                        list(i.hi_drivers) or ["span-boundary"] for i in intervals
                    ],
                    "roots": [r.to_dict() for r in window.roots],
                }
            }
            if run_contracts:
                contracts.append(
                    check_monotonicity(
                        self.case.constraints, base, axes, policy, counter
                    )
                )
                contracts.append(
                    check_convexity(self.case.constraints, base, axes, policy, counter)
                )
                structure = check_interval_expectation(intervals, policy)
                if structure:
                    contracts.append(ContractReport((structure,)))
                active = check_active_expectation(intervals, policy)
                if active:
                    contracts.append(ContractReport((active,)))
                status_expect = check_infeasible_expectation(status.value, policy)
                if status_expect:
                    contracts.append(ContractReport((status_expect,)))
                if reference_area is not None and width is not None:
                    contracts.append(
                        ContractReport(
                            (
                                check_area_consistency(
                                    width, reference_area, policy.area_rel_tol, "width"
                                ),
                            )
                        )
                    )
        else:
            window_plane: PlaneWindow = march_plane(
                self.case.constraints,
                base,
                axes,
                n_lines=n_lines,
                n_scan=n_scan,
                counter=counter,
            )
            intervals = ()
            components = window_plane.components
            discrete = None
            status = WindowStatus.FEASIBLE if components else WindowStatus.INFEASIBLE
            area: float | None = window_plane.area if components else 0.0
            width = None
            nearest = None
            if run_contracts and n_grid_check >= 2:
                grid_check = scan_plane(
                    self.case.constraints,
                    base,
                    axes,
                    n_grid_check,
                    n_grid_check,
                    counter,
                )
            if status is WindowStatus.INFEASIBLE:
                nearest = nearest_feasible_point(
                    self.case.constraints,
                    base,
                    axes,
                    self.case.anchor_params(condition),
                    n_per_axis=33,
                    counter=counter,
                )
            active_sets = {
                "driver_counts": window_plane.driver_counts(),
                "component_areas": [c.area for c in components],
                "area_methods": [c.area_method for c in components],
                "closed": [c.closed for c in components],
            }
            if run_contracts:
                contracts.append(
                    check_monotonicity(
                        self.case.constraints, base, axes, policy, counter
                    )
                )
                contracts.append(
                    check_convexity(self.case.constraints, base, axes, policy, counter)
                )
                structure = check_component_expectation(components, policy)
                if structure:
                    contracts.append(ContractReport((structure,)))
                status_expect = check_infeasible_expectation(status.value, policy)
                if status_expect:
                    contracts.append(ContractReport((status_expect,)))
                if reference_area is not None:
                    contracts.append(
                        ContractReport(
                            (
                                check_area_consistency(
                                    window_plane.area, reference_area, policy.area_rel_tol, "area"
                                ),
                            )
                        )
                    )
        if analytic is not None and components and self.case.mode == "plane":
            boundary_error = max(
                measure_component_boundary_error(c, analytic.lower, analytic.upper)
                for c in components
            )
        if run_contracts:
            contracts.append(
                ContractReport(
                    (
                        check_infeasible_semantics(
                            status.value, intervals, components, area, nearest
                        ),
                    )
                )
            )
        elapsed = time.perf_counter() - started
        return WindowResult(
            case_id=self.case.case_id,
            condition=self.case.condition(condition).label,
            mode=self.case.mode,
            status=status.value,
            axes=axes,
            intervals=intervals,
            components=components,
            area=area,
            width=width,
            discrete=discrete,
            nearest=nearest,
            active_sets=active_sets,
            contracts=ContractReport.merge(*contracts),
            n_evals=counter.count,
            elapsed_s=elapsed,
            source_digest=self.case.source_digest,
            grid_check=grid_check,
            boundary_error=boundary_error,
        )


@dataclass(frozen=True)
class OrderStudy:
    resolutions: Tuple[int, ...]
    step_sizes: Tuple[float, ...]
    errors: Tuple[float, ...]
    orders: Tuple[float, ...]
    outcome: CheckOutcome
    results: Tuple[WindowResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolutions": list(self.resolutions),
            "step_sizes": list(self.step_sizes),
            "errors": list(self.errors),
            "orders": list(self.orders),
            "outcome": self.outcome.to_dict(),
        }


def order_study(
    case: WindowCase,
    condition: str | None = None,
    resolutions: Sequence[int] = (20, 40, 80),
    n_scan: int = 64,
    analytic: AnalyticReference | None = None,
    min_order: float | None = None,
) -> OrderStudy:
    """Boundary reconstruction order under refinement of the line ladder."""

    if analytic is None:
        raise ValueError("order_study needs an analytic reference boundary")
    engine = EnvelopeEngine(case)
    results = [
        engine.run(
            condition=condition,
            n_lines=n_lines,
            n_scan=n_scan,
            n_grid_check=0,
            verify=False,
            analytic=analytic,
            run_contracts=False,
        )
        for n_lines in resolutions
    ]
    errors = tuple(r.boundary_error or 0.0 for r in results)
    steps = tuple(case.axes[0].width / (n - 1) for n in resolutions)
    threshold = case.contract.min_boundary_order if min_order is None else min_order
    outcome = check_boundary_order(steps, errors, threshold, label=analytic.name)
    from .contracts import boundary_order

    return OrderStudy(
        resolutions=tuple(resolutions),
        step_sizes=steps,
        errors=errors,
        orders=boundary_order(steps, errors),
        outcome=outcome,
        results=tuple(results),
    )