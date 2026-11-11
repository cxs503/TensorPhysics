"""Conservative remapping operators between two unstructured triangular meshes.

Three projection modes for scalar fields:

``"average"``     first-order overlap-area weighting (conservative and monotone
                  by construction; the conservative baseline);
``"p1_recon"``    second-order reconstruction using source cell gradients
                  (consistent for linear fields, but may produce new extrema);
``"bounded"``     second-order reconstruction + Barth-Jespersen limiter +
                  conservative repair of the limiter defect.

For vector (momentum) fields, each component is projected independently with
the same conservative operator.

All operators satisfy the algebraic conservation identity:

    sum_j phi_tgt[j] * |T_j| = sum_i phi_src[i] * |S_i|

up to machine precision (for the "average" mode) or up to the repair tolerance
(for the "bounded" mode).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .overlap import OverlapTable, TriMesh, polygon_centroid

__all__ = [
    "RemapResult",
    "remap_cell_average",
    "remap_p1_nodal",
    "remap_vector",
    "conservation_report",
    "monotonicity_report",
    "l2_error",
    "symmetry_roundtrip",
]


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
@dataclass
class RemapResult:
    mode: str
    phi: np.ndarray          # (n_tgt,) target values
    coverage: np.ndarray     # (n_tgt,) mapped area
    source_integral: float
    target_integral: float
    repair: dict = field(default_factory=dict)

    @property
    def covered(self) -> np.ndarray:
        return self.coverage > 1e-12


# ---------------------------------------------------------------------------
# first-order: overlap-area weighting (conservative baseline)
# ---------------------------------------------------------------------------
def remap_cell_average(
    table: OverlapTable,
    phi_src: np.ndarray,
) -> RemapResult:
    """Conservative projection of cell-averaged scalar field.

    ``phi_tgt[j] = sum_i |T_j cap S_i| * phi_src[i] / |T_j|``

    This is conservative by construction because the overlap areas partition
    each source cell exactly (when the meshes cover the same domain).
    """
    phi_src = np.asarray(phi_src, dtype=float)
    num = table.M @ phi_src
    tgt_areas = table.tgt_areas
    covered = table.coverage > 1e-300
    phi_tgt = np.zeros(table.n_tgt)
    phi_tgt[covered] = num[covered] / tgt_areas[covered]

    src_integral = float(phi_src @ table.src_areas)
    tgt_integral = float((phi_tgt * tgt_areas * covered).sum())

    return RemapResult(
        mode="average",
        phi=phi_tgt,
        coverage=table.coverage,
        source_integral=src_integral,
        target_integral=tgt_integral,
    )


# ---------------------------------------------------------------------------
# second-order: P1 reconstruction with gradient
# ---------------------------------------------------------------------------
def _compute_cell_gradients(mesh: TriMesh, phi_cell: np.ndarray) -> np.ndarray:
    """Least-squares gradient per cell from cell-centred values.

    For each cell T, solve the overdetermined system:
        phi_N - phi_T = grad . (x_N - x_T)
    for all neighbours N of T, in the least-squares sense.
    """
    n = mesh.n_elements
    centroids = mesh.centroids()
    grad = np.zeros((n, 2))

    # Build cell adjacency from shared edges
    cell_neighbours: dict[int, list[int]] = {e: [] for e in range(n)}
    edge_cells: dict[tuple[int, int], list[int]] = {}
    for e in range(n):
        for a in range(3):
            n1 = int(mesh.tri[e, a])
            n2 = int(mesh.tri[e, (a + 1) % 3])
            key = (min(n1, n2), max(n1, n2))
            if key not in edge_cells:
                edge_cells[key] = []
            edge_cells[key].append(e)

    for cells in edge_cells.values():
        if len(cells) == 2:
            e1, e2 = cells
            cell_neighbours[e1].append(e2)
            cell_neighbours[e2].append(e1)

    for e in range(n):
        neighbours = cell_neighbours[e]
        if not neighbours:
            continue
        xc = centroids[e]
        phi_c = phi_cell[e]
        # Build least-squares system: A @ g = b
        # where A has rows (x_N - x_c) and b has entries phi_N - phi_c
        dx = centroids[neighbours] - xc  # (k, 2)
        dphi = phi_cell[neighbours] - phi_c  # (k,)
        # Normal equations: (A^T A) g = A^T b
        ATA = dx.T @ dx  # (2, 2)
        ATb = dx.T @ dphi  # (2,)
        # Solve with small regularisation for robustness
        ATA += 1e-20 * np.eye(2)
        grad[e] = np.linalg.solve(ATA, ATb)

    return grad


def remap_p1_reconstruction(
    table: OverlapTable,
    phi_src: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Second-order reconstruction using source cell gradients.

    For each target cell centroid, evaluate the linear reconstruction from
    overlapping source cells, weighted by overlap area.

    Returns (phi_tgt, info_dict).
    """
    phi_src = np.asarray(phi_src, dtype=float)
    grad = _compute_cell_gradients(table.source, phi_src)
    src_centroids = table.source.centroids()
    tgt_centroids = table.target.centroids()

    num = np.zeros(table.n_tgt)
    den = np.zeros(table.n_tgt)

    # Build overlap element lists from the sparse matrix
    M_coo = table.M.tocoo()
    overlap_map: dict[int, list[tuple[int, float]]] = {}
    for r, c, v in zip(M_coo.row, M_coo.col, M_coo.data):
        overlap_map.setdefault(int(r), []).append((int(c), float(v)))

    for j in range(table.n_tgt):
        if j not in overlap_map:
            continue
        xj = tgt_centroids[j]
        w_sum = 0.0
        v_sum = 0.0
        for i, w in overlap_map[j]:
            dc = xj - src_centroids[i]
            val = phi_src[i] + np.dot(grad[i], dc)
            v_sum += w * val
            w_sum += w
        if w_sum > 1e-300:
            num[j] = v_sum
            den[j] = w_sum

    covered = den > 1e-300
    phi_tgt = np.zeros(table.n_tgt)
    phi_tgt[covered] = num[covered] / den[covered]

    info = {"n_reconstructed": int(covered.sum())}
    return phi_tgt, info


def _barth_jespersen_1d(
    phi_avg: np.ndarray,
    phi_lin: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell limiting factor psi in [0,1] that keeps the value in bounds."""
    d = phi_lin - phi_avg
    psi = np.ones_like(phi_avg)
    up = d > 0.0
    dn = d < 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        psi_up = np.where(up & (np.abs(d) > 1e-300), (hi - phi_avg) / d, 1.0)
        psi_dn = np.where(dn & (np.abs(d) > 1e-300), (lo - phi_avg) / d, 1.0)
    psi[up] = np.minimum(psi[up], psi_up[up])
    psi[dn] = np.minimum(psi[dn], psi_dn[dn])
    psi = np.clip(psi, 0.0, 1.0)
    return psi, psi < 1.0 - 1e-15


def _conservative_repair_1d(
    phi: np.ndarray,
    area: np.ndarray,
    target_integral: float,
    lo: np.ndarray,
    hi: np.ndarray,
    max_iter: int = 200,
    rtol: float = 1e-14,
) -> tuple[np.ndarray, dict]:
    """Remove integral defect without leaving [lo, hi]."""
    phi = np.array(phi, dtype=float, copy=True)
    area = np.asarray(area, dtype=float)
    scale = max(abs(target_integral), 1e-300)
    r0 = float(phi @ area) - target_integral
    it = 0
    for it in range(1, max_iter + 1):
        resid = float(phi @ area) - target_integral
        if abs(resid) <= rtol * scale:
            break
        need = -resid
        sgn = 1.0 if need > 0.0 else -1.0
        slack = np.where(sgn > 0.0, np.maximum(hi - phi, 0.0), np.maximum(phi - lo, 0.0))
        weights = slack * area
        capacity = float(weights.sum())
        if capacity <= 0.0:
            break
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
        "feasible": bool(abs(resid_after) <= max(rtol * scale, 1e-12 * scale)),
    }
    return phi, info


def remap_p1_nodal(
    table: OverlapTable,
    phi_src: np.ndarray,
    mode: str = "bounded",
    repair: bool = True,
) -> RemapResult:
    """Remap a cell-averaged scalar field with optional second-order reconstruction.

    Parameters
    ----------
    table : OverlapTable
        Geometric overlap between source and target meshes.
    phi_src : (n_src,) array
        Cell-averaged source field.
    mode : str
        ``"average"``: first-order overlap weighting.
        ``"p1_recon"``: second-order reconstruction, no limiter.
        ``"bounded"``: second-order + Barth-Jespersen limiter + repair.
    repair : bool
        Whether to apply conservative repair after limiting (only for "bounded").
    """
    phi_src = np.asarray(phi_src, dtype=float)
    src_integral = float(phi_src @ table.src_areas)

    if mode == "average":
        result = remap_cell_average(table, phi_src)
        result.mode = "average"
        return result

    # Compute bounds from overlapping source cells
    M_coo = table.M.tocoo()
    lo = np.full(table.n_tgt, np.inf)
    hi = np.full(table.n_tgt, -np.inf)
    for r, c, v in zip(M_coo.row, M_coo.col, M_coo.data):
        r, c = int(r), int(c)
        lo[r] = min(lo[r], phi_src[c])
        hi[r] = max(hi[r], phi_src[c])
    lo = np.where(np.isfinite(lo), lo, 0.0)
    hi = np.where(np.isfinite(hi), hi, 0.0)

    phi_avg = np.zeros(table.n_tgt)
    covered = table.coverage > 1e-300
    phi_avg[covered] = (table.M @ phi_src)[covered] / table.tgt_areas[covered]

    if mode == "p1_recon":
        phi_lin, info = remap_p1_reconstruction(table, phi_src)
        phi_tgt = phi_lin
        repair_info = {}
    elif mode == "bounded":
        phi_lin, info = remap_p1_reconstruction(table, phi_src)
        psi, limited = _barth_jespersen_1d(phi_avg, phi_lin, lo, hi)
        phi_tgt = phi_avg + psi * (phi_lin - phi_avg)
        repair_info = {"n_limited_cells": int(np.count_nonzero(limited)),
                       "min_psi": float(psi.min())}
        if repair:
            phi_tgt, rep_info = _conservative_repair_1d(
                phi_tgt, table.tgt_areas, src_integral, lo, hi)
            repair_info["repair"] = rep_info
    else:
        raise ValueError(f"unknown mode {mode!r}")

    tgt_integral = float((phi_tgt * table.tgt_areas * covered).sum())

    return RemapResult(
        mode=mode,
        phi=phi_tgt,
        coverage=table.coverage,
        source_integral=src_integral,
        target_integral=tgt_integral,
        repair=repair_info,
    )


# ---------------------------------------------------------------------------
# vector (momentum) remapping
# ---------------------------------------------------------------------------
def remap_vector(
    table: OverlapTable,
    vec_src: np.ndarray,  # (n_src, d)
    mode: str = "average",
) -> RemapResult:
    """Conservative remapping of a vector field (e.g., momentum).

    Each component is projected independently with the same conservative operator.
    The result is stored as a flattened array; use ``.reshape(n_tgt, d)`` to recover.
    """
    vec_src = np.asarray(vec_src, dtype=float)
    if vec_src.ndim == 1:
        vec_src = vec_src[:, None]
    d = vec_src.shape[1]

    results = []
    for k in range(d):
        r = remap_cell_average(table, vec_src[:, k])
        results.append(r.phi)

    phi_tgt = np.column_stack(results)  # (n_tgt, d)
    src_integral = float(np.sum(vec_src * table.src_areas[:, None]))
    tgt_integral = float(np.sum(phi_tgt * table.tgt_areas[:, None]))

    return RemapResult(
        mode=mode,
        phi=phi_tgt.ravel(),
        coverage=table.coverage,
        source_integral=src_integral,
        target_integral=tgt_integral,
    )


# ---------------------------------------------------------------------------
# audits
# ---------------------------------------------------------------------------
def conservation_report(result: RemapResult) -> dict:
    """Relative integral defect of the target field against the source integral."""
    src = result.source_integral
    tgt = result.target_integral
    denom = max(abs(src), 1e-300)
    return {
        "mode": result.mode,
        "source_integral": src,
        "target_integral": tgt,
        "abs_defect": tgt - src,
        "rel_defect": abs(tgt - src) / denom,
    }


def monotonicity_report(
    table: OverlapTable,
    phi_src: np.ndarray,
    result: RemapResult,
) -> dict:
    """Check whether the target field stays within the source range."""
    phi_src = np.asarray(phi_src, dtype=float)
    phi_tgt = result.phi
    src_min = float(phi_src.min())
    src_max = float(phi_src.max())
    tgt_min = float(phi_tgt.min())
    tgt_max = float(phi_tgt.max())
    overshoot = max(0.0, tgt_max - src_max)
    undershoot = max(0.0, src_min - tgt_min)
    return {
        "source_min": src_min,
        "source_max": src_max,
        "target_min": tgt_min,
        "target_max": tgt_max,
        "overshoot": overshoot,
        "undershoot": undershoot,
        "overshoot_rel": overshoot / max(abs(src_max), 1e-300),
        "n_cells_above_max": int(np.count_nonzero(phi_tgt > src_max + 1e-12)),
        "n_cells_below_min": int(np.count_nonzero(phi_tgt < src_min - 1e-12)),
    }


def l2_error(
    table: OverlapTable,
    phi_tgt: np.ndarray,
    phi_ref: np.ndarray,
) -> dict:
    """L2 and Linf error of the target field against a reference."""
    phi_tgt = np.asarray(phi_tgt, dtype=float)
    phi_ref = np.asarray(phi_ref, dtype=float)
    diff = phi_tgt - phi_ref
    areas = table.tgt_areas
    l2 = float(np.sqrt(np.sum(diff ** 2 * areas)))
    l2_rel = l2 / max(float(np.sqrt(np.sum(phi_ref ** 2 * areas))), 1e-300)
    linf = float(np.max(np.abs(diff)))
    linf_rel = linf / max(float(np.max(np.abs(phi_ref))), 1e-300)
    return {"l2_abs": l2, "l2_rel": l2_rel, "linf_abs": linf, "linf_rel": linf_rel}


# ---------------------------------------------------------------------------
# symmetry roundtrip test
# ---------------------------------------------------------------------------
def symmetry_roundtrip(
    table_ab: OverlapTable,
    table_ba: OverlapTable,
    phi_a: np.ndarray,
) -> dict:
    """Test A -> B -> A roundtrip: the field should return close to the original.

    This tests the symmetry of the conservative projection: if the overlap
    computation is correct and the projection is truly conservative, the
    roundtrip should preserve the integral exactly and the field shape closely.
    """
    # A -> B
    res_ab = remap_cell_average(table_ab, phi_a)
    # B -> A
    res_ba = remap_cell_average(table_ba, res_ab.phi)

    phi_a_rt = res_ba.phi
    src_integral = float(phi_a @ table_ab.src_areas)
    rt_integral = float(phi_a_rt @ table_ba.tgt_areas)  # table_ba.src = target of ba = A

    diff = phi_a_rt - phi_a
    l2 = float(np.sqrt(np.sum(diff ** 2 * table_ab.src_areas)))
    l2_rel = l2 / max(float(np.sqrt(np.sum(phi_a ** 2 * table_ab.src_areas))), 1e-300)

    return {
        "source_integral": src_integral,
        "roundtrip_integral": rt_integral,
        "integral_rel_error": abs(rt_integral - src_integral) / max(abs(src_integral), 1e-300),
        "l2_abs": l2,
        "l2_rel": l2_rel,
        "linf_abs": float(np.max(np.abs(diff))),
    }
