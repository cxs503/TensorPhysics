"""Conservative, monotonicity-preserving scalar-field remapping FEM -> LBM lattice.

The pipeline implemented here is the classical three-stage conservative remap
(Farrell & Maddison 2011; Liska & Sistek 2008), specialised to a P1 source field:

1. **source integration** -- the P1 field is integrated exactly over every element
   (``int_T phi dA = |T| * mean(nodal values)``), and over every element/lattice-cell
   intersection (``|C cap T| * phi(centroid of C cap T)``, exact because ``phi`` is
   linear inside an element);
2. **overlap-volume weighting** -- the target (lattice) value is the cell average of
   the source field, ``phi_j = (int_{C_j} phi dA) / |C_j cap Omega|``; because the
   lattice cells partition the source domain, ``sum_j |C_j| phi_j = int_Omega phi dA``
   *identically* (discrete conservation is an algebraic identity of the operator,
   not a numerical tolerance);
3. **monotonicity limiting + conservative redistribution** -- a second-order
   limited reconstruction (Barth-Jespersen-type per-cell factor, bounds taken from
   the *overlapping source elements*) restores second-order accuracy in smooth
   regions while provably producing no new extrema; the resulting integral defect
   is then removed by a bounded, iterative redistribution that moves mass only
   into cells that still have slack towards the local bound, so conservation and
   monotonicity hold *simultaneously*.

Every operator is linear in the nodal values except the limiter and the repair,
and both are audited (:func:`conservation_report`, :func:`monotonicity_report`)
so that "conservative" and "monotone" are measured numbers, not claims.

Ownership: remapping numerics only (no FEM assembly, no LBM kernel).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .lattice import CartesianLattice, OverlapTable, build_overlap
from .mesh import TriMesh

__all__ = [
    "ScalarMapResult",
    "map_scalar_field",
    "conservative_repair",
    "conservation_report",
    "monotonicity_report",
    "exact_cell_average",
    "l2_error_vs_exact",
    "make_map",
]


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
@dataclass
class ScalarMapResult:
    mode: str
    phi: np.ndarray  # (n_cells,) target values
    coverage: np.ndarray  # (n_cells,) mapped area
    coverage_fraction: np.ndarray
    bounds_lo: np.ndarray
    bounds_hi: np.ndarray
    source_integral: float
    target_integral: float
    repair: dict = field(default_factory=dict)

    @property
    def covered(self) -> np.ndarray:
        return self.coverage_fraction > 1e-12

    def grid(self, lattice: CartesianLattice) -> np.ndarray:
        return lattice.field_to_grid(self.phi)


# ---------------------------------------------------------------------------
# local bounds from the overlapping source elements
# ---------------------------------------------------------------------------
def _local_bounds(mesh: TriMesh, phi_nodes: np.ndarray,
                  table: OverlapTable) -> tuple[np.ndarray, np.ndarray]:
    """Per-target-cell ``[min, max]`` of the source field *over its overlap region*.

    Because a P1 field attains its extrema on the element vertices, the exact range
    of the field on ``C_j cap Omega`` is the nodal range over the elements that
    overlap ``C_j``.  Clipping a reconstruction to these bounds is therefore the
    discrete local maximum principle ("the target cell may not hold a temperature
    the source field does not have there"), and it automatically enforces the
    global source range as well.
    """
    n = table.n_cells
    lo = np.full(n, np.inf)
    hi = np.full(n, -np.inf)
    for j in range(n):
        e = table.overlap_elems[j]
        if e.size == 0:
            continue
        nodes = mesh.tri[e].ravel()
        lo[j] = float(phi_nodes[nodes].min())
        hi[j] = float(phi_nodes[nodes].max())
    return lo, hi


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
def _cell_average(table: OverlapTable, phi_nodes: np.ndarray) -> np.ndarray:
    num = table.M @ np.asarray(phi_nodes, dtype=float)
    cov = table.coverage
    out = np.zeros(table.n_cells)
    ok = cov > 1e-300
    out[ok] = num[ok] / cov[ok]
    return out


def _linear_reconstruction(mesh: TriMesh, table: OverlapTable,
                           phi_nodes: np.ndarray) -> np.ndarray:
    """Second-order reconstruction to the lattice centroids.

    For each target cell the reconstruction is the overlap-area-weighted mean of
    the *element-wise linear* extrapolations ``phi_T + grad phi_T . (x_j - x_T)``.
    Exact for linear source fields (consistency), O(h^2) for smooth fields, and
    unbounded near extrema -- which is precisely what the limiter is for.
    """
    cell = mesh.cell_values(phi_nodes)
    grad = mesh.element_gradient_of_field(phi_nodes)
    cen_e = mesh.centroids()
    cen_c = table.lattice.centroids()
    num = np.zeros(table.n_cells)
    den = np.zeros(table.n_cells)
    for j in range(table.n_cells):
        e = table.overlap_elems[j]
        if e.size == 0:
            continue
        w = table.overlap_areas[j]
        dc = cen_c[j] - cen_e[e]  # (k,2)
        vals = cell[e] + np.einsum("kd,kd->k", grad[e], dc)
        num[j] = float(w @ vals)
        den[j] = float(w.sum())
    ok = den > 1e-300
    out = np.zeros(table.n_cells)
    out[ok] = num[ok] / den[ok]
    return out


def _barth_jespersen(phi_avg: np.ndarray, phi_lin: np.ndarray,
                     lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell limiting factor ``psi in [0,1]`` that keeps the value in bounds."""
    d = phi_lin - phi_avg
    psi = np.ones_like(phi_avg)
    up = d > 0.0
    dn = d < 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        psi_up = np.where(up & (d != 0.0), (hi - phi_avg) / np.where(d == 0.0, 1.0, d), 1.0)
        psi_dn = np.where(dn & (d != 0.0), (lo - phi_avg) / np.where(d == 0.0, 1.0, d), 1.0)
    psi[up] = np.minimum(psi[up], psi_up[up])
    psi[dn] = np.minimum(psi[dn], psi_dn[dn])
    psi = np.clip(psi, 0.0, 1.0)
    return psi, psi < 1.0 - 1e-15


def conservative_repair(
    phi: np.ndarray,
    area: np.ndarray,
    target_integral: float,
    lo: np.ndarray,
    hi: np.ndarray,
    max_iter: int = 200,
    rtol: float = 1e-14,
) -> tuple[np.ndarray, dict]:
    """Remove the integral defect of ``phi`` without leaving ``[lo, hi]``.

    Mass is pushed as a *uniform increment per unit area* into the cells that
    still have slack in the required direction; cells whose slack is smaller than
    the increment saturate at their bound and the remainder is redistributed in
    the next sweep.  The residual therefore decreases monotonically and the
    bounds are never violated, so the two requirements (conservation, no new
    extrema) are met at the same time -- which a plain clip cannot do (clipping
    is what creates the defect in the first place).
    """
    phi = np.array(phi, dtype=float, copy=True)
    area = np.asarray(area, dtype=float)
    scale = max(abs(target_integral), 1e-300)
    r0 = float(phi @ area) - target_integral
    it = 0
    capacity = float("nan")
    for it in range(1, max_iter + 1):
        resid = float(phi @ area) - target_integral
        if abs(resid) <= rtol * scale:
            break
        need = -resid
        sgn = 1.0 if need > 0.0 else -1.0
        slack = np.where(sgn > 0.0, np.maximum(hi - phi, 0.0), np.maximum(phi - lo, 0.0))
        weights = slack * area  # available "mass" towards the relaxed bound
        capacity = float(weights.sum())
        if capacity <= 0.0:
            break
        # exact single sweep when the capacity suffices; best effort otherwise
        beta = min(1.0, abs(need) / capacity)
        phi = phi + sgn * beta * slack
        if beta >= 1.0:
            break
    resid_after = float(phi @ area) - target_integral
    info = {
        "residual_before": r0,
        "residual_after": resid_after,
        "rel_residual_before": abs(r0) / scale,
        "rel_residual_after": abs(resid_after) / scale,
        "iterations": int(it),
        "capacity_used": capacity,
        "feasible": bool(abs(resid_after) <= max(rtol * scale, 1e-12 * scale)),
        "max_bound_violation": float(max(np.max(phi - hi), np.max(lo - phi))),
    }
    return phi, info


def map_scalar_field(
    mesh: TriMesh,
    phi_nodes: np.ndarray,
    table: OverlapTable,
    mode: str = "bounded",
    repair: bool = True,
    rtol: float = 1e-13,
) -> ScalarMapResult:
    """Map a P1 nodal scalar field onto the Cartesian lattice.

    ``mode``
        ``"average"``  first-order overlap-volume weighting (conservative and
                       monotone by construction; the conservative baseline);
        ``"linear"``   second-order element-wise linear reconstruction, *no*
                       limiter (the non-monotone negative control);
        ``"bounded"``  second-order reconstruction + Barth-Jespersen limiter
                       (+ optional conservative repair of the limiter defect).
    """
    phi_nodes = np.asarray(phi_nodes, dtype=float)
    lo, hi = _local_bounds(mesh, phi_nodes, table)
    phi_avg = _cell_average(table, phi_nodes)
    source_integral = float(mesh.cell_integrals(phi_nodes).sum())

    info: dict = {}
    if mode == "average":
        phi = phi_avg
    elif mode == "linear":
        phi = _linear_reconstruction(mesh, table, phi_nodes)
    elif mode == "bounded":
        phi_lin = _linear_reconstruction(mesh, table, phi_nodes)
        psi, limited = _barth_jespersen(phi_avg, phi_lin, lo, hi)
        phi = phi_avg + psi * (phi_lin - phi_avg)
        info = {"n_limited_cells": int(np.count_nonzero(limited)),
                "min_psi": float(psi.min())}
        if repair:
            area = table.lattice.areas()
            cov = table.coverage
            phi_rep, rep_info = conservative_repair(
                phi, area, source_integral, lo, hi, rtol=rtol * max(1.0, abs(source_integral)))
            phi = phi_rep
            info["repair"] = rep_info
    else:
        raise ValueError(f"unknown remap mode {mode!r}")

    area = table.lattice.areas()
    covered = table.coverage_fraction() > 1e-12
    target_integral = float((phi * area * covered).sum())
    return ScalarMapResult(
        mode=mode,
        phi=phi,
        coverage=table.coverage,
        coverage_fraction=table.coverage_fraction(),
        bounds_lo=lo,
        bounds_hi=hi,
        source_integral=source_integral,
        target_integral=target_integral,
        repair=info,
    )


def make_map(mesh: TriMesh, lattice: CartesianLattice,
             phi_nodes: np.ndarray, **kw) -> tuple[ScalarMapResult, OverlapTable]:
    """Convenience: build the overlap table and run :func:`map_scalar_field`."""
    table = build_overlap(mesh, lattice)
    return map_scalar_field(mesh, phi_nodes, table, **kw), table


# ---------------------------------------------------------------------------
# audits
# ---------------------------------------------------------------------------
def conservation_report(result: ScalarMapResult, area: np.ndarray | None = None) -> dict:
    """Relative integral defect of the target field against the source integral."""
    area = result.coverage if area is None else area
    src = result.source_integral
    tgt = result.target_integral
    denom = max(abs(src), 1e-300)
    return {
        "mode": result.mode,
        "source_integral": src,
        "target_integral": tgt,
        "abs_defect": tgt - src,
        "rel_defect": abs(tgt - src) / denom,
        "min_coverage_fraction": float(result.coverage_fraction.min()),
        "n_undercovered_cells": int(np.count_nonzero(result.coverage_fraction < 1.0 - 1e-10)),
    }


def monotonicity_report(mesh: TriMesh, phi_nodes: np.ndarray,
                        result: ScalarMapResult) -> dict:
    """No-new-extrema audit against the *source nodal* and *source cell-mean* ranges."""
    phi_nodes = np.asarray(phi_nodes, dtype=float)
    cell = mesh.cell_values(phi_nodes)
    covered = result.coverage_fraction > 1e-12
    tgt = result.phi[covered]
    nodal_lo, nodal_hi = float(phi_nodes.min()), float(phi_nodes.max())
    cell_lo, cell_hi = float(cell.min()), float(cell.max())
    span = max(nodal_hi - nodal_lo, 1e-300)
    tol = 1e-12 * span  # round-off band: counting at 1e-17 would measure float noise
    under = float(nodal_lo - tgt.min()) if tgt.min() < nodal_lo - tol else 0.0
    over = float(tgt.max() - nodal_hi) if tgt.max() > nodal_hi + tol else 0.0
    return {
        "mode": result.mode,
        "source_nodal_min": nodal_lo,
        "source_nodal_max": nodal_hi,
        "source_cell_mean_min": cell_lo,
        "source_cell_mean_max": cell_hi,
        "target_min": float(tgt.min()),
        "target_max": float(tgt.max()),
        "undershoot": under,
        "overshoot": over,
        "overshoot_rel": max(under, over) / span,
        "bounded": bool(under == 0.0 and over == 0.0),
        "n_cells_above_nodal_max": int(np.count_nonzero(result.phi[covered] > nodal_hi + tol)),
        "n_cells_below_nodal_min": int(np.count_nonzero(result.phi[covered] < nodal_lo - tol)),
        "range_tolerance": tol,
    }


# ---------------------------------------------------------------------------
# error metrics against an analytic field
# ---------------------------------------------------------------------------
def exact_cell_average(lattice: CartesianLattice, fn: Callable[[np.ndarray], np.ndarray],
                       n_gauss: int = 5) -> np.ndarray:
    """High-order composite Gauss reference of the analytic cell average."""
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    cen = lattice.centroids()
    hx, hy = lattice.dx, lattice.dy
    acc = np.zeros(lattice.n_cells)
    for a in range(n_gauss):
        for b in range(n_gauss):
            px = cen[:, 0] + 0.5 * hx * xg[a]
            py = cen[:, 1] + 0.5 * hy * xg[b]
            vals = np.asarray(fn(np.stack([px, py], axis=1)), dtype=float)
            acc += 0.25 * wg[a] * wg[b] * vals
    return acc


def l2_error_vs_exact(lattice: CartesianLattice, phi_cells: np.ndarray,
                      reference: np.ndarray) -> dict:
    """Discrete L2 error of the target cell values against a cell-average reference."""
    covered = np.isfinite(phi_cells)
    diff = phi_cells[covered] - reference[covered]
    area = lattice.cell_area()
    l2 = float(np.sqrt(area * np.sum(diff ** 2)) / np.sqrt(area * covered.sum()))
    ref_rms = float(np.sqrt(area * np.sum(reference[covered] ** 2)) / np.sqrt(area * covered.sum()))
    return {
        "l2_abs": l2,
        "l2_rel": l2 / max(ref_rms, 1e-300),
        "linf_abs": float(np.max(np.abs(diff))),
        "reference_rms": ref_rms,
    }