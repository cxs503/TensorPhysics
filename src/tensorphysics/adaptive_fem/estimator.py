"""Zienkiewicz–Zhu (ZZ) a-posteriori error estimation.

Recovery
--------
``sigma_h`` for a T3 (CST) discretisation is piecewise constant and therefore
discontinuous; it is a poor approximation of the exact stress.  ZZ replace it by
a recovered *smooth* field ``sigma*`` obtained by a patch-wise moving least
squares fit (SPR — superconvergent patch recovery): for every node ``a`` the
patch is the set of elements containing ``a``, the sampling points are the
element centroids and the sampled values are the element stresses.  A bilinear
polynomial ``p(x) = c0 + c1 (x - x_a) + c2 (y - y_a)`` is fitted per stress
component in the least-squares sense with element area weights; ``sigma*(x_a)``
is the recovered nodal value.  Patches that are rank-deficient (fewer than
three non-collinear samples, e.g. a corner node with a single element) fall back
to the area-weighted mean, which is the standard SPR safeguard.

Estimator
---------
``eta_K^2 = |K| t (sigma*_K - sigma_h,K)^T D^-1 (sigma*_K - sigma_h,K)``
with ``sigma*_K`` the mean of the nodal recovered values of ``K``.  The global
estimate is ``eta = (sum_K eta_K^2)^(1/2)``, an estimate of the error in the
energy norm ``||sigma - sigma_h||_E`` (no factor 1/2, i.e. twice the square root
of the error in strain energy).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elasticity2d import Material2D, PlaneSolution, element_energy_norm_sq, element_stress
from .mesh2d import TriMesh, signed_areas

__all__ = [
    "ZzEstimate",
    "spr_recover_nodal_stress",
    "nodal_average_stress",
    "zz_estimate",
    "element_energy_norm",
    "actual_energy_error",
    "effectivity_index",
]


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------
def spr_recover_nodal_stress(
    nodes: np.ndarray, tris: np.ndarray, elem_stress: np.ndarray, elem_area: np.ndarray | None = None
) -> np.ndarray:
    """SPR recovery of a continuous stress field at the mesh nodes, ``(n,3)``."""
    nodes = np.asarray(nodes, float)
    tris = np.asarray(tris, np.int64)
    m = len(tris)
    if elem_area is None:
        elem_area = np.abs(signed_areas(nodes, tris))
    cent = nodes[tris].mean(axis=1)

    nd = tris.reshape(-1)  # node id per (element, corner) pair, row-major
    el = np.repeat(np.arange(m), 3)
    dx = cent[el, 0] - nodes[nd, 0]
    dy = cent[el, 1] - nodes[nd, 1]
    a = np.stack([np.ones_like(dx), dx, dy], axis=1)  # (3m, 3) local polynomial basis
    w = elem_area[el]
    v = elem_stress[el]

    n = len(nodes)
    M = np.zeros((n, 3, 3))
    R = np.zeros((n, 3, 3))
    np.add.at(M, nd, w[:, None, None] * a[:, :, None] * a[:, None, :])
    np.add.at(R, nd, w[:, None, None] * a[:, :, None] * v[:, None, :])

    out = np.zeros((n, 3))
    coef, det = _solve3(M, R)
    tr = np.trace(M, axis1=1, axis2=2)
    scale = np.maximum(np.abs(tr), 1e-300)
    ok = np.isfinite(det) & (np.abs(det) > 1e-12 * scale**3)

    if np.any(ok):
        out[ok] = coef[ok][:, 0, :]
    bad = ~ok
    if np.any(bad):
        wsum = M[bad][:, 0, 0]
        wsum = np.where(np.abs(wsum) > 0.0, wsum, 1.0)
        out[bad] = R[bad][:, 0, :] / wsum[:, None]
    return out


def _solve3(M: np.ndarray, R: np.ndarray):
    """Closed-form solution of ``n`` stacked 3x3 systems.

    ``np.linalg.solve`` on stacked 3x3 systems falls through to LAPACK per
    matrix and is two orders of magnitude slower than this Cramer/adjugate
    formulation on the patched BLAS of the target environment, which matters
    because it is called once per mesh node.
    """
    a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    d, e, f = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
    g, h, i = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    inv = np.empty_like(M)
    inv[:, 0, 0] = e * i - f * h
    inv[:, 0, 1] = c * h - b * i
    inv[:, 0, 2] = b * f - c * e
    inv[:, 1, 0] = f * g - d * i
    inv[:, 1, 1] = a * i - c * g
    inv[:, 1, 2] = c * d - a * f
    inv[:, 2, 0] = d * h - e * g
    inv[:, 2, 1] = b * g - a * h
    inv[:, 2, 2] = a * e - b * d
    safe = np.where(np.abs(det) > 0.0, det, 1.0)
    inv /= safe[:, None, None]
    return np.einsum("nij,njk->nik", inv, R), det


def nodal_average_stress(
    nodes: np.ndarray, tris: np.ndarray, elem_stress: np.ndarray, elem_area: np.ndarray | None = None
) -> np.ndarray:
    """Plain area-weighted nodal averaging (SP1), the classical ZZ recovery."""
    if elem_area is None:
        elem_area = np.abs(signed_areas(nodes, tris))
    n = len(nodes)
    acc = np.zeros((n, 3))
    wacc = np.zeros(n)
    for j in range(3):
        np.add.at(acc, tris[:, j], elem_area[:, None] * elem_stress)
        np.add.at(wacc, tris[:, j], elem_area)
    wacc = np.where(wacc > 0.0, wacc, 1.0)
    return acc / wacc[:, None]


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------
@dataclass
class ZzEstimate:
    eta_elem: np.ndarray  # (m,) element indicators (energy norm units)
    eta_total: float
    smoothed_stress: np.ndarray  # (m,3) recovered stress, element-wise
    nodal_stress: np.ndarray  # (n,3)
    recovery: str

    def dorfler_sorted(self) -> np.ndarray:
        """Element indices ordered by decreasing indicator."""
        return np.argsort(-self.eta_elem)


def zz_estimate(sol: PlaneSolution, *, recovery: str = "spr") -> ZzEstimate:
    """Compute ZZ indicators for a solved plane-elasticity state."""
    mat: Material2D = sol.material
    nodes = sol.mesh.nodes
    tris = sol.mesh.tris
    if recovery == "spr":
        nodal = spr_recover_nodal_stress(nodes, tris, sol.stress, sol.area)
    elif recovery == "average":
        nodal = nodal_average_stress(nodes, tris, sol.stress, sol.area)
    else:  # pragma: no cover - guard
        raise ValueError("recovery must be 'spr' or 'average'")

    smoothed = nodal[tris].mean(axis=1)
    diff = smoothed - sol.stress
    dens = element_energy_norm_sq(diff, sol.area, mat)
    eta_elem = np.sqrt(np.abs(dens) * sol.area * mat.thickness)
    eta_total = float(np.sqrt(np.sum(eta_elem**2)))
    return ZzEstimate(
        eta_elem=eta_elem,
        eta_total=eta_total,
        smoothed_stress=smoothed,
        nodal_stress=nodal,
        recovery=recovery,
    )


# ---------------------------------------------------------------------------
# error metrics against a reference
# ---------------------------------------------------------------------------
def element_energy_norm(stress_diff: np.ndarray, area: np.ndarray, mat: Material2D) -> np.ndarray:
    dens = element_energy_norm_sq(stress_diff, area, mat)
    return np.sqrt(np.abs(dens) * area * mat.thickness)


def actual_energy_error(sol: PlaneSolution, reference_stress: np.ndarray) -> float:
    """``||sigma_ref - sigma_h||_E`` on the mesh of ``sol`` (reference at centroids)."""
    diff = reference_stress - sol.stress
    return float(np.sqrt(np.sum(element_energy_norm(diff, sol.area, sol.material) ** 2)))


def effectivity_index(eta_total: float, actual: float) -> float:
    """``eta_eff = eta_estimated / eta_actual`` (>0, target ~1)."""
    if actual <= 0.0:
        raise ValueError("actual error must be positive")
    return float(eta_total / actual)


def rel_error(actual: float, solution_norm: float) -> float:
    return float(actual / solution_norm) if solution_norm > 0.0 else float("nan")


def stress_at_centroids(sol: PlaneSolution) -> np.ndarray:
    return sol.stress


def reinterpolate_stress(
    sol: PlaneSolution, ref_nodes: np.ndarray, ref_tris: np.ndarray, ref_stress: np.ndarray
) -> "ReferenceStressField":
    return ReferenceStressField(ref_nodes, ref_tris, ref_stress)


class ReferenceStressField:
    """Locate-and-sample a piecewise-constant reference stress field.

    Used to measure the *actual* discretisation error of any mesh (uniform or
    adaptively graded) against a high-resolution reference solution that lives
    on a different, non-nested mesh.
    """

    def __init__(self, nodes: np.ndarray, tris: np.ndarray, elem_stress: np.ndarray, k: int = 12):
        from scipy.spatial import cKDTree

        self.nodes = np.asarray(nodes, float)
        self.tris = np.asarray(tris, np.int64)
        self.stress = np.asarray(elem_stress, float)
        self.cent = self.nodes[self.tris].mean(axis=1)
        self.tree = cKDTree(self.cent)
        self.k = int(min(k, len(self.tris)))

    def at(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float)
        _, idx = self.tree.query(pts, k=self.k)
        idx = np.atleast_2d(idx)
        cand = idx  # (p, k)
        p = self.nodes[self.tris[cand]]  # (p, k, 3, 2)
        q = pts[:, None, :]
        # barycentric coordinates via edge functions
        v0 = p[:, :, 1] - p[:, :, 0]
        v1 = p[:, :, 2] - p[:, :, 0]
        v2 = q - p[:, :, 0]
        den = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            l1 = (v2[..., 0] * v1[..., 1] - v1[..., 0] * v2[..., 1]) / den
            l2 = (v0[..., 0] * v2[..., 1] - v2[..., 0] * v0[..., 1]) / den
        l0 = 1.0 - l1 - l2
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9) & np.isfinite(den)
        first = np.where(inside.any(axis=1), np.argmax(inside, axis=1), 0)
        rows = np.arange(len(pts))
        return self.stress[cand[rows, first]]