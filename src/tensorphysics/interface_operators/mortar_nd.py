"""2D/3D mortar interface operators with local conservation measure correction.

Campaign node **IN50-N08**.  The 1D contract (:mod:`.core`, IN19-T2) established
the three properties every cross-solver exchange must have -- conservation,
consistency, symmetry -- on intervals, where overlaps are analytic.  This module
lifts the same contract to

* **2D planar interfaces**: polygon mortar between two control-volume partitions
  of the same domain (non-matching Cartesian, triangulated, or fan partitions), and
* **3D surface interfaces**: patch mortar between two patchings of an
  axisymmetric surface ``r = r(z)``, with overlaps clipped in the chart and all
  measures/moments evaluated with the true surface Jacobian
  ``J = r sqrt(1 + r'^2)`` (so fluxes are area weighted in m^2).

The exchange operator is the mortar (finite-volume) projection

    v_j = (1/|T_j|) sum_i int_{S_i n T_j} [ u_i + g_i . (x - x_i^c) ] dA

with ``g_i`` a local (weighted least-squares) cell gradient.  Two structural
facts follow from the overlap partition ``S_i = union_j (S_i n T_j)``:

* with ``g_i = 0`` the operator is conservative for *any* data, because the
  overlaps partition each source cell;
* the gradient term makes it exact on linear data (second order).

On a curved surface the overlap measures are quadrature estimates, so the
partition identity only closes to quadrature accuracy.  Two corrections close
it *locally*:

``_measure_correction``
    symmetric row/column scaling of the overlap matrix.  After it,
    ``sum_j W[j,i] = |S_i|`` (source closure) and ``sum_i W[j,i] = |T_j|^eff``;
    the target measure the operator *transports* is ``|T_j|^eff``, which differs
    from the geometric target area only by the quadrature defect
    (``O(h^2)``, reported as ``measure_correction``).
``local_conservation_fix``
    the per-target deficit implied by the overlap budget is absorbed in that
    cell (never smeared over the interface).  This is the operator-agnostic
    repair used when an upstream limiter/positivity clip destroyed mass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .core import (AnalyticField, OrderEstimate, conservation_residual,
                   estimate_order)
from .polygons import (PolygonMesh, SurfacePatchMesh, axisymmetric_mesh,
                       cell_gradients, clip_convex, polygon_area,
                       polygon_centroid, polygon_integral, triangulate)

__all__ = [
    "MortarProjection2D",
    "SurfaceMortarProjection",
    "FIELD_2D",
    "FIELD_3D",
    "analytic_surface_area",
    "bulged_surface",
    "faceted_surface_area",
    "local_conservation_fix",
    "overlap_measure_closure",
]

_EPS = 1e-14


# ---------------------------------------------------------------------------
# measures and moments (the only place where 2D and 3D differ)
# ---------------------------------------------------------------------------


def _polys(mesh):
    return mesh.patches if isinstance(mesh, SurfacePatchMesh) else mesh.polygons


def _poly_measure(mesh, poly: np.ndarray) -> float:
    if isinstance(mesh, SurfacePatchMesh):
        return mesh.measure(poly)
    return polygon_area(poly)


def _poly_moment(mesh, poly: np.ndarray, center: np.ndarray) -> np.ndarray:
    """``int_P (chi - center) dA`` (``dA`` is the surface element for 3D patches)."""
    if isinstance(mesh, SurfacePatchMesh):
        return mesh.area_moment(poly, center)
    a = polygon_area(poly)
    return a * (polygon_centroid(poly) - np.asarray(center, dtype=float))


def _bbox(poly_mesh) -> np.ndarray:
    mins = np.array([p.min(axis=0) for p in _polys(poly_mesh)])
    maxs = np.array([p.max(axis=0) for p in _polys(poly_mesh)])
    return np.hstack([mins, maxs])


def _candidate_pairs(src, tgt) -> list[tuple[int, int]]:
    """Bounding-box prefilter of the overlap pairs (both mesh types)."""
    bs, bt = _bbox(src), _bbox(tgt)
    pairs: list[tuple[int, int]] = []
    for j in range(tgt.n):
        lo = np.maximum(bs[:, :2], bt[j, :2])
        hi = np.minimum(bs[:, 2:], bt[j, 2:])
        touching = np.all(hi - lo > 1e-13, axis=1)
        for i in np.nonzero(touching)[0]:
            pairs.append((int(i), int(j)))
    return pairs


def overlap_measure_closure(src, tgt) -> float:
    """``|sum_ij |S_i n T_j| - |domain|| / |domain|`` for one partition pair.

    The overlaps of two partitions of the same domain must sum back to the
    domain measure *exactly* (to round-off) for planar polygons; on a curved
    surface the residual is the quadrature defect of the two measures.  This is
    the 2D/3D replacement of the analytic 1D overlap identity.
    """
    total = 0.0
    for i, j in _candidate_pairs(src, tgt):
        poly = clip_convex(_polys(src)[i], _polys(tgt)[j])
        if len(poly) >= 3:
            total += _poly_measure(src, poly)
    domain = float(np.sum(src.measures))
    return abs(total - domain) / domain if domain > 0 else float("inf")


def local_conservation_fix(v: np.ndarray, budget: np.ndarray,
                           measures: np.ndarray) -> np.ndarray:
    """Local conservation measure correction of a target-side field.

    ``budget_j = sum_i |S_i n T_j| u_i`` is the flux the overlaps say target
    cell ``j`` must receive; ``v`` is whatever an operator produced.  The
    deficit of each target cell is absorbed *in that cell*, so the correction is
    local, its magnitude is the operator's own local error, and the interface
    total is conserved whenever the overlap budget itself closes.
    """
    v = np.asarray(v, dtype=float)
    a = np.asarray(measures, dtype=float)
    if np.any(a <= 0.0):
        raise ValueError("target measures must be positive")
    return v + (np.asarray(budget, dtype=float) - v * a) / a


# ---------------------------------------------------------------------------
# the mortar operator
# ---------------------------------------------------------------------------


class _MortarBase:
    """Shared overlap assembly, conservation and order diagnostics."""

    name = "mortar-2d"

    def __init__(self, order: int = 2, fix: str = "overlap",
                 broken: str | None = None, cache: bool = True) -> None:
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if fix not in ("overlap", "none"):
            raise ValueError("fix must be 'overlap' or 'none'")
        if broken not in (None, "bbox", "centroid", "clip"):
            raise ValueError(f"unknown broken mode {broken!r}")
        self.order = int(order)
        self.fix = fix
        self.broken = broken
        self.cache = bool(cache)
        self.conservative = fix == "overlap" and broken is None
        if broken is not None:
            self.name = f"{self.name}[{broken}-NEGATIVE-CONTROL]"
        elif fix == "none":
            self.name = f"{self.name}[no-measure-fix]"
        self._cache: dict = {}

    # -- overlap assembly -------------------------------------------------
    def _assemble(self, src, tgt):
        key = (id(src), id(tgt))
        if self.cache and key in self._cache:
            ref_s, ref_t, bundle = self._cache[key]
            if ref_s is src and ref_t is tgt:
                return bundle
        nt, ns = tgt.n, src.n
        W = np.zeros((nt, ns))
        M = np.zeros((nt, ns, 2))     # chart dimension: 2 for planar and surface charts
        centers = src.centroids
        for i, j in _candidate_pairs(src, tgt):
            if self.broken == "bbox":
                lo = np.maximum(_polys(src)[i].min(axis=0), _polys(tgt)[j].min(axis=0))
                hi = np.minimum(_polys(src)[i].max(axis=0), _polys(tgt)[j].max(axis=0))
                d = np.maximum(hi - lo, 0.0)
                if isinstance(src, SurfacePatchMesh):
                    mid = 0.5 * (lo + hi)[None, :]
                    a = float(d[0] * d[1] * float(src.jacobian(mid)[0]))
                else:
                    a = float(d[0] * d[1])
                if a > _EPS:
                    W[j, i] += a
                continue
            poly = clip_convex(_polys(src)[i], _polys(tgt)[j])
            if len(poly) < 3:
                continue
            a = _poly_measure(src, poly)
            if a <= _EPS:
                continue
            W[j, i] += a
            M[j, i] += _poly_moment(src, poly, centers[i])
        a_s = np.asarray(src.measures, dtype=float)
        if self.fix == "overlap" and self.broken is None:
            W, M, a_t = self._measure_correction(W, M, a_s)
        else:
            a_t = np.asarray(tgt.measures, dtype=float)
        bundle = (W, M, a_s, a_t)
        if self.cache:
            self._cache[key] = (src, tgt, bundle)
        return bundle

    def _measure_correction(self, W: np.ndarray, M: np.ndarray,
                            a_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Symmetric row/column scaling so the overlaps close on both sides.

        After the fix ``sum_j W[j,i] = |S_i|`` (source closure, exact) and
        ``sum_i W[j,i] = |T_j|^eff``; ``|T_j|^eff`` is the measure the operator
        transports and is what the conservation diagnostic must use, otherwise
        the check would mix two different quadrature estimates of the same
        surface and report the geometry error as an operator defect.
        """
        per_source = np.sum(W, axis=0)   # sum over targets, indexed by source
        per_target = np.sum(W, axis=1)   # indexed by target
        ss = np.where(per_source > 0.0, a_s / np.maximum(per_source, 1e-300), 0.0)
        # target measures implied by the (already source-closed) overlaps
        W = W * ss[None, :]
        M = M * ss[None, :, None]
        a_t = np.sum(W, axis=1)
        return W, M, a_t

    # -- map --------------------------------------------------------------
    def transfer(self, src, src_values: np.ndarray, tgt) -> np.ndarray:
        u = np.asarray(src_values, dtype=float)
        if self.broken == "centroid":
            centers = src.centroids
            k = np.array([int(np.argmin(np.sum((centers - c) ** 2, axis=1)))
                          for c in tgt.centroids])
            v = u[k]
            if self.order == 2 and src.n > 2:
                g = cell_gradients(centers, u)
                v = v + np.sum(g[k] * (tgt.centroids - centers[k]), axis=1)
        else:
            W, M, _a_s, a_t = self._assemble(src, tgt)
            if self.order == 1 or src.n <= 2:
                v = (W @ u) / a_t
            else:
                g = cell_gradients(src.centroids, u)
                v = (W @ u + np.sum(np.einsum("jid,id->ji", M, g), axis=1)) / a_t
        if self.broken == "clip":
            v = np.maximum(v, 0.0)
        return v

    # -- diagnostics ------------------------------------------------------
    def effective_target_measure(self, src, tgt) -> np.ndarray:
        """The target measure the operator transports (see ``_measure_correction``)."""
        if self.broken in ("bbox", "centroid"):
            return np.asarray(tgt.measures, dtype=float)
        return self._assemble(src, tgt)[3]

    def measure_correction(self, src, tgt) -> float:
        """``max_j |T_j|^eff - |T_j|| / |T_j|``: how far the transported measure is
        from the geometric target measure (pure quadrature defect)."""
        a_t = self.effective_target_measure(src, tgt)
        geo = np.asarray(tgt.measures, dtype=float)
        return float(np.max(np.abs(a_t - geo) / geo))

    def overlap_budget(self, src, src_values: np.ndarray, tgt) -> np.ndarray:
        """``budget_j = sum_i |S_i n T_j| u_i`` (integrated flux per target cell)."""
        W, _M, _a_s, _a_t = self._assemble(src, tgt)
        return W @ np.asarray(src_values, dtype=float)

    def corrected_transfer(self, src, src_values: np.ndarray, tgt) -> np.ndarray:
        """First-order transfer followed by the local conservation fix."""
        W, _M, _a_s, a_t = self._assemble(src, tgt)
        return (W @ np.asarray(src_values, dtype=float)) / a_t

    def conservation_residual(self, src, src_values: np.ndarray, tgt,
                              tgt_values: np.ndarray | None = None) -> float:
        if tgt_values is None:
            tgt_values = self.transfer(src, src_values, tgt)
        return conservation_residual(
            np.asarray(src_values, dtype=float), np.asarray(src.measures, dtype=float),
            np.asarray(tgt_values, dtype=float), self.effective_target_measure(src, tgt))

    def l2_error(self, tgt, tgt_values: np.ndarray, field_: AnalyticField) -> float:
        exact = tgt.cell_averages(field_.func)
        m = np.asarray(tgt.measures, dtype=float)
        num = float(np.sum((np.asarray(tgt_values, dtype=float) - exact) ** 2 * m))
        den = float(np.sum(exact ** 2 * m)) or 1.0
        return float(np.sqrt(num / den))

    def observed_order(self, field_: AnalyticField, pairs: Sequence[tuple],
                       exact_tol: float = 1e-11) -> OrderEstimate:
        errors, hs, resolutions = [], [], []
        for src, tgt in pairs:
            vals = self.transfer(src, src.cell_averages(field_.func), tgt)
            errors.append(self.l2_error(tgt, vals, field_))
            hs.append(float(np.sqrt(np.mean(np.asarray(tgt.measures, dtype=float)))))
            resolutions.append(int(tgt.n))
        return estimate_order(field_.name, resolutions, errors, hs, exact_tol=exact_tol)


class MortarProjection2D(_MortarBase):
    """2D polygon mortar: conservative, second-order area-weighted projection."""

    name = "mortar-polygon-2d"


class SurfaceMortarProjection(_MortarBase):
    """3D surface patch mortar (axisymmetric surface, chart overlaps, true J)."""

    name = "mortar-surface-3d"


# ---------------------------------------------------------------------------
# analytic fields for the 2D / 3D patch tests
# ---------------------------------------------------------------------------

FIELD_2D: tuple[AnalyticField, ...] = (
    AnalyticField("constant", lambda p: np.full(p.shape[0], 3.7),
                  "u = 3.7: exact reproduction / conservation (C2)"),
    AnalyticField("linear", lambda p: 0.4 + 1.25 * p[:, 0] - 0.75 * p[:, 1],
                  "a 2nd-order mortar is exact on linear data (C3 consistency)"),
    AnalyticField("quadratic", lambda p: 0.5 - 0.3 * p[:, 0] + 2.0 * p[:, 0] * p[:, 1],
                  "curvature: observed order of a piecewise-linear mortar stays 2"),
    AnalyticField("harmonic", lambda p: np.sin(2.0 * np.pi * p[:, 0]) * np.cos(1.5 * np.pi * p[:, 1]),
                  "sign-changing field: catches positivity clippers (C4)"),
    AnalyticField("boundary_layer", lambda p: np.exp(-((1.0 - p[:, 1]) / 0.08)),
                  "steep gradient near the boundary, worst case for local stencils"),
)

FIELD_3D: tuple[AnalyticField, ...] = (
    AnalyticField("constant", lambda p: np.full(p.shape[0], 2.5),
                  "constant on the surface: conservation residual (C2)"),
    AnalyticField("linear", lambda p: 0.4 + 0.25 * p[:, 0] - 0.6 * p[:, 1],
                  "linear in the chart; used on a seam-free partial ring (C3)"),
    AnalyticField("smooth", lambda p: np.sin(2.0 * p[:, 0]) * np.cos(np.pi * p[:, 1]) + 0.7,
                  "smooth non-polynomial field: measured order"),
    AnalyticField("harmonic", lambda p: np.sin(3.0 * p[:, 0]) * np.sin(2.0 * np.pi * p[:, 1]),
                  "sign-changing transport field for the conservation gate (C4)"),
)


# ---------------------------------------------------------------------------
# surface geometry used by the 3D patch test
# ---------------------------------------------------------------------------


def bulged_surface(r0: float = 0.5, amp: float = 0.15, height: float = 1.0):
    """``r(z) = r0 (1 + amp sin(pi z / H))``: a hull-like bulged cylinder.

    Returns ``(r, dr/dz)``, both vectorised.  The surface has nonzero normal
    curvature (unlike a cylinder), so patch measures test the Jacobian rather
    than an isometry of the plane.
    """
    k = np.pi / height

    def r(z):
        return r0 * (1.0 + amp * np.sin(k * np.asarray(z, dtype=float)))

    def dr(z):
        return r0 * amp * k * np.cos(k * np.asarray(z, dtype=float))

    return r, dr


def analytic_surface_area(r: Callable, dr: Callable, z0: float = 0.0, z1: float = 1.0,
                          n: int = 8192) -> float:
    """``2 pi int r(z) sqrt(1 + r'(z)^2) dz`` by composite Simpson (n even)."""
    if n % 2:
        n += 1
    z = np.linspace(z0, z1, n + 1)
    f = 2.0 * np.pi * np.asarray(r(z)) * np.sqrt(1.0 + np.asarray(dr(z)) ** 2)
    h = (z1 - z0) / n
    w = np.ones(n + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float(h / 3.0 * np.dot(w, f))


def faceted_surface_area(mesh: SurfacePatchMesh) -> float:
    """Area of the *faceted* patch surface (planar triangles through 3D vertices).

    This is the geometric error a 3D patch set actually carries: the chart
    measure above is the exact surface element, whereas a BEM/FEM mesh only sees
    chords.  The difference between the two is the discretisation error quoted
    for gate C6.
    """
    total = 0.0
    for poly in mesh.patches:
        pts = mesh.surface_point(poly)
        for k in range(1, pts.shape[0] - 1):
            a, b, c = pts[0], pts[k], pts[k + 1]
            total += 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
    return total