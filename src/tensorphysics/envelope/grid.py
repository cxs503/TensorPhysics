"""Parameter-plane grid scan: feasibility mask, active set, cell area.

The grid scan is the *cheap* view of a window: it answers "which cells are
feasible", how many separate feasible regions the eye would see, and which
constraint is the binding one in each cell.  Its area estimate is a raw cell
count, therefore first order in the cell size -- it is used as an independent
cross-check of the marched area (:mod:`tensorphysics.envelope.marching`), never
as the reported boundary.

Everything here is a pure function of a :class:`ConstraintSet` plus a parameter
map; the axes are just names and spans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from .case import AxesSpec
from .constraints import ConstraintSet, EvalCounter
from .roots import linspace

__all__ = [
    "AxisScan",
    "CellCount",
    "NearestPoint",
    "PlaneScan",
    "brute_cell_area",
    "nearest_feasible_point",
    "scan_axis",
    "scan_plane",
]


@dataclass(frozen=True)
class ScanPoint:
    coords: Tuple[float, ...]
    min_margin: float
    feasible: bool
    driver: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "coords": list(self.coords),
            "min_margin": self.min_margin,
            "feasible": self.feasible,
            "driver": self.driver,
        }


@dataclass(frozen=True)
class AxisScan:
    """One-dimensional scan along a single axis."""

    axis: str
    values: Tuple[float, ...]
    points: Tuple[ScanPoint, ...]
    n_evals: int

    @property
    def feasible_values(self) -> Tuple[float, ...]:
        return tuple(p.coords[0] for p in self.points if p.feasible)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "n_evals": self.n_evals,
            "n_feasible": len(self.feasible_values),
            "points": [p.to_dict() for p in self.points],
        }


@dataclass(frozen=True)
class NearestPoint:
    """Outcome of the grid search used when no feasible point exists.

    ``found`` is an assertion about the search, not about the physics: if a
    feasible point *is* found the window would not have been empty in the first
    place, and that contradiction is reported instead of being hidden.
    """

    params: Mapping[str, float]
    coords: Tuple[float, ...]
    residual: float
    found: bool
    distance_to_anchor: float
    searched: int
    n_evals: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": dict(self.params),
            "coords": list(self.coords),
            "residual": self.residual,
            "found": self.found,
            "distance_to_anchor": self.distance_to_anchor,
            "searched": self.searched,
            "n_evals": self.n_evals,
        }


@dataclass(frozen=True)
class CellCount:
    feasible: int
    total: int
    area: float

    def to_dict(self) -> dict[str, Any]:
        return {"feasible_cells": self.feasible, "total_cells": self.total, "area": self.area}


@dataclass(frozen=True)
class PlaneScan:
    """Two-dimensional feasibility mask on a uniform lattice."""

    axes: Tuple[AxesSpec, AxesSpec]
    u_values: Tuple[float, ...]
    v_values: Tuple[float, ...]
    feasible: Tuple[Tuple[bool, ...], ...]
    min_margins: Tuple[Tuple[float, ...], ...]
    drivers: Tuple[Tuple[str, ...], ...]
    cell_area: float
    cells: CellCount
    driver_counts: Mapping[str, int]
    n_evals: int

    def feasible_components_estimate(self) -> int:
        """4-connected component count of the cell mask (topology cross-check)."""

        seen = [[False] * len(self.v_values) for _ in self.u_values]
        components = 0
        for i in range(len(self.u_values)):
            for j in range(len(self.v_values)):
                if not self.feasible[i][j] or seen[i][j]:
                    continue
                components += 1
                stack = [(i, j)]
                seen[i][j] = True
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < len(self.u_values) and 0 <= nb < len(self.v_values):
                            if self.feasible[na][nb] and not seen[na][nb]:
                                seen[na][nb] = True
                                stack.append((na, nb))
        return components

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": [a.to_dict() for a in self.axes],
            "u_values": list(self.u_values),
            "v_values": list(self.v_values),
            "feasible": [list(row) for row in self.feasible],
            "min_margins": [list(row) for row in self.min_margins],
            "drivers": [list(row) for row in self.drivers],
            "cells": self.cells.to_dict(),
            "driver_counts": dict(self.driver_counts),
            "n_evals": self.n_evals,
            "components_4conn": self.feasible_components_estimate(),
        }


def scan_axis(
    constraints: ConstraintSet,
    base_params: Mapping[str, Any],
    axis: AxesSpec,
    n: int,
    counter: EvalCounter,
    fixed: Mapping[str, float] | None = None,
) -> AxisScan:
    params = dict(base_params)
    if fixed:
        params.update({k: float(v) for k, v in fixed.items()})
    start = counter.count
    points = []
    for value in linspace(axis.lo, axis.hi, n):
        params[axis.name] = value
        evaluation = constraints.evaluate(params, counter)
        points.append(
            ScanPoint(
                coords=(value,),
                min_margin=evaluation.min_margin,
                feasible=evaluation.feasible,
                driver=evaluation.driver,
            )
        )
    return AxisScan(
        axis=axis.name,
        values=tuple(p.coords[0] for p in points),
        points=tuple(points),
        n_evals=counter.count - start,
    )


def scan_plane(
    constraints: ConstraintSet,
    base_params: Mapping[str, Any],
    axes: Sequence[AxesSpec],
    nu: int,
    nv: int,
    counter: EvalCounter,
) -> PlaneScan:
    if len(axes) != 2:
        raise ValueError("scan_plane needs exactly two axes")
    u_axis, v_axis = axes[0], axes[1]
    start = counter.count
    params = dict(base_params)
    u_values = linspace(u_axis.lo, u_axis.hi, nu)
    v_values = linspace(v_axis.lo, v_axis.hi, nv)
    feasible: list[list[bool]] = []
    margins: list[list[float]] = []
    drivers: list[list[str]] = []
    driver_counts: dict[str, int] = {}
    for u in u_values:
        params[u_axis.name] = u
        row_f: list[bool] = []
        row_m: list[float] = []
        row_d: list[str] = []
        for v in v_values:
            params[v_axis.name] = v
            evaluation = constraints.evaluate(params, counter)
            row_f.append(evaluation.feasible)
            row_m.append(evaluation.min_margin)
            row_d.append(evaluation.driver)
            if evaluation.feasible:
                driver_counts[evaluation.driver] = driver_counts.get(evaluation.driver, 0) + 1
        feasible.append(row_f)
        margins.append(row_m)
        drivers.append(row_d)
    du = u_axis.width / (nu - 1)
    dv = v_axis.width / (nv - 1)
    cell = du * dv
    feasible_cells = sum(1 for row in feasible for flag in row if flag)
    return PlaneScan(
        axes=(u_axis, v_axis),
        u_values=u_values,
        v_values=v_values,
        feasible=tuple(tuple(row) for row in feasible),
        min_margins=tuple(tuple(row) for row in margins),
        drivers=tuple(tuple(row) for row in drivers),
        cell_area=cell,
        cells=CellCount(feasible=feasible_cells, total=nu * nv, area=feasible_cells * cell),
        driver_counts=driver_counts,
        n_evals=counter.count - start,
    )


def brute_cell_area(
    constraints: ConstraintSet,
    base_params: Mapping[str, Any],
    axes: Sequence[AxesSpec],
    nu: int,
    nv: int,
    counter: EvalCounter,
) -> CellCount:
    """Raw cell-count area (first order in the cell size); independent cross-check."""

    scan = scan_plane(constraints, base_params, axes, nu, nv, counter)
    return scan.cells

def nearest_feasible_point(
    constraints: ConstraintSet,
    base_params: Mapping[str, Any],
    axes: Sequence[AxesSpec],
    anchor: Mapping[str, float],
    n_per_axis: int = 33,
    counter: EvalCounter | None = None,
) -> NearestPoint:
    """Grid search for the least-violating point, plus the residual at it.

    The residual is the remaining violation ``max(0, -min_margin)`` at the best
    point found; it is a *statement about the search*, therefore it can only be
    quoted together with the search resolution (``n_per_axis``) and the number
    of points examined.  No coordinate is extrapolated outside the declared
    spans.
    """

    counter = counter or EvalCounter()
    start = counter.count
    grids: list[Tuple[float, ...]] = [linspace(axis.lo, axis.hi, n_per_axis) for axis in axes]
    best_violation_point: tuple[list[float], float] | None = None
    best_feasible_point: tuple[list[float], float] | None = None
    anchor_coords = [float(anchor.get(axis.name, 0.5 * (axis.lo + axis.hi))) for axis in axes]
    widths = [axis.width for axis in axes]
    searched = 0

    def distance(coords: Sequence[float]) -> float:
        return sum(
            ((coords[i] - anchor_coords[i]) / widths[i]) ** 2 for i in range(len(coords))
        ) ** 0.5

    def walk(depth: int, coords: list[float]) -> None:
        nonlocal best_violation_point, best_feasible_point, searched
        if depth == len(grids):
            params = dict(base_params)
            for axis, value in zip(axes, coords):
                params[axis.name] = value
            evaluation = constraints.evaluate(params, counter)
            searched += 1
            if evaluation.feasible:
                dist = distance(coords)
                if best_feasible_point is None or dist < best_feasible_point[1]:
                    best_feasible_point = (list(coords), dist)
            else:
                violation = -evaluation.min_margin
                if best_violation_point is None or violation < best_violation_point[1]:
                    best_violation_point = (list(coords), violation)
            return
        for value in grids[depth]:
            coords.append(value)
            walk(depth + 1, coords)
            coords.pop()

    walk(0, [])

    if best_feasible_point is not None:
        coords, dist = best_feasible_point
        params = dict(base_params)
        params.update({axis.name: value for axis, value in zip(axes, coords)})
        evaluation = constraints.evaluate(params, counter)
        return NearestPoint(
            params=params,
            coords=tuple(coords),
            residual=max(0.0, -evaluation.min_margin),
            found=True,
            distance_to_anchor=dist,
            searched=searched,
            n_evals=counter.count - start,
        )
    if best_violation_point is None:  # pragma: no cover - grids are never empty
        raise ValueError("nearest_feasible_point: empty search grid")
    coords, violation = best_violation_point
    params = dict(base_params)
    params.update({axis.name: value for axis, value in zip(axes, coords)})
    return NearestPoint(
        params=params,
        coords=tuple(coords),
        residual=violation,
        found=False,
        distance_to_anchor=distance(coords),
        searched=searched,
        n_evals=counter.count - start,
    )
