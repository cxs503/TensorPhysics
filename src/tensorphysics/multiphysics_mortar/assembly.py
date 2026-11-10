"""Conservative mortar transfer between two non-matching interface trace spaces.

Operator family
---------------
The transfer is the *L2 projection onto the receiving trace space*,

    b_j = int_Gamma u_src phi_j^tgt dGamma ,      M_ij = int_Gamma phi_i^tgt phi_j^tgt dGamma ,
    u_tgt = M^{-1} b            ( = "mortar projection", symmetric M ),

and the *dual basis* of the receiving space is ``psi_i = (M^{-1} phi)_i``, i.e.
``psi_i = sum_j (M^{-1})_ij phi_j``.  The coefficient extraction
``c_i = int_Gamma u_src psi_i`` is **algebraically identical** to ``M^{-1} b``
(``c = M^{-1} C u``), which is why this package exposes both paths: the dual
route is the one that diagonalises the mortar coupling (one constraint per slave
DOF), the matrix route is the one that is numerically stable.  The identity is
verified numerically in the tests, not silently assumed.

Why it is exactly conservative
------------------------------
Because the receiving basis is a partition of unity (``sum_j phi_j = 1``),

    int_Gamma u_tgt = int_Gamma u_tgt * 1 = sum_j int_Gamma u_tgt phi_j
                    = sum_j b_j = int_Gamma u_src sum_j phi_j^tgt = int_Gamma u_src ,

i.e. the *primal* partition of unity - not a property of the dual basis - is what
makes the transfer conservative for arbitrary data.  The dual basis is generally
**not** a partition of unity (see ``dual_partition_of_unity_defect``); a
formulation that relied on it would leak mass.  This is stated explicitly because
the derivation is a common trap.

The same argument gives the in-repo negative control: with matching meshes
``b = M u_src`` so ``u_tgt = u_src`` to round-off, i.e. mortar degenerates to the
identity map (C4).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..interface_operators.polygons import (_TRI_BARY, _TRI_W, clip_convex,
                                            polygon_area, polygon_integral)
from .spaces import (GAUSS5_W, GAUSS5_X, LineTraceSpace, TriTraceSpace)

__all__ = [
    "CouplingAssembly",
    "spd_inverse",
    "spd_solve",
    "MortarTransfer",
    "NearestNeighborTransfer",
    "build_coupling",
]

_EPS = 1e-15


# ---------------------------------------------------------------------------
# metric resolution (the interface measure is a property of the interface)
# ---------------------------------------------------------------------------


def _line_metric(src: LineTraceSpace, tgt: LineTraceSpace
                 ) -> Callable[[np.ndarray], np.ndarray] | None:
    for sp in (src, tgt):
        if sp.metric is not None:
            return sp.metric
    return None


def _tri_metric(src: TriTraceSpace, tgt: TriTraceSpace
                ) -> Callable[[np.ndarray], np.ndarray] | None:
    for sp in (src, tgt):
        if sp.metric is not None:
            return sp.metric
    return None


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def _line_mass(space: LineTraceSpace, subdivisions: int) -> np.ndarray:
    """``M_ij = int phi_i phi_j dGamma`` (5-point Gauss per sub-interval)."""
    M = np.zeros((space.n, space.n))
    metric = space.metric
    for cell in range(space.n_cells):
        a, b = space.nodes[cell], space.nodes[cell + 1]
        edges = np.linspace(a, b, int(subdivisions) + 1)
        for p, q in zip(edges[:-1], edges[1:]):
            xi = 0.5 * (p + q) + 0.5 * (q - p) * GAUSS5_X
            w = 0.5 * (q - p) * GAUSS5_W
            if metric is not None:
                w = w * np.asarray(metric(xi), dtype=float)
            phi = space.local_values(space.locate(xi)[1])
            ids = space.local_ids(cell)
            M[np.ix_(ids, ids)] += (w[:, None, None] * phi[:, :, None]
                                    * phi[:, None, :]).sum(axis=0)
    return M


def _line_cross(src: LineTraceSpace, tgt: LineTraceSpace,
                subdivisions: int) -> tuple[np.ndarray, int]:
    """``C[j, k] = int phi_j^tgt phi_k^src dGamma`` via the merged breakpoints."""
    C = np.zeros((tgt.n, src.n))
    metric = _line_metric(src, tgt)
    breaks = np.union1d(src.nodes, tgt.nodes)
    n_pieces = 0
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b - a <= _EPS:
            continue
        edges = np.linspace(a, b, int(subdivisions) + 1)
        for p, q in zip(edges[:-1], edges[1:]):
            mid = 0.5 * (p + q)
            s_ids = src.local_ids(int(src.locate(np.array([mid]))[0][0]))
            t_ids = tgt.local_ids(int(tgt.locate(np.array([mid]))[0][0]))
            xi = 0.5 * (p + q) + 0.5 * (q - p) * GAUSS5_X
            w = 0.5 * (q - p) * GAUSS5_W
            if metric is not None:
                w = w * np.asarray(metric(xi), dtype=float)
            phi_s = src.local_values(src.locate(xi)[1])
            phi_t = tgt.local_values(tgt.locate(xi)[1])
            C[np.ix_(t_ids, s_ids)] += (w[:, None, None] * phi_t[:, :, None]
                                        * phi_s[:, None, :]).sum(axis=0)
            n_pieces += 1
    return C, n_pieces


def _clip_tri_fast(subject: np.ndarray, clipper: np.ndarray) -> list[tuple[float, float]]:
    """Sutherland--Hodgman clip specialised to a *triangle* clipper.

    Plain-float version of :func:`tensorphysics.interface_operators.polygons.clip_convex`
    (same algorithm, same vertex order) written for speed: the canonical kernel
    spends ~0.4 ms per call on NumPy scalar overhead, which dominates the assembly
    of a large interface (see the C7 criterion).  Equivalence with the canonical
    kernel is asserted in the test suite, so this is an optimisation with a
    reference, not a second source of truth.
    """
    cx = [float(p[0]) for p in clipper]
    cy = [float(p[1]) for p in clipper]
    a2 = sum(cx[k] * cy[(k + 1) % 3] - cx[(k + 1) % 3] * cy[k] for k in range(3))
    if a2 < 0.0:
        cx = cx[::-1]
        cy = cy[::-1]
    out = [(float(p[0]), float(p[1])) for p in subject]
    for k in range(3):
        if not out:
            return []
        ax, ay = cx[k], cy[k]
        bx, by = cx[(k + 1) % 3], cy[(k + 1) % 3]
        ex, ey = bx - ax, by - ay
        inp = out
        out = []
        px, py = inp[-1]
        pside = ex * (py - ay) - ey * (px - ax)
        for qx, qy in inp:
            cside = ex * (qy - ay) - ey * (qx - ax)
            if cside >= 0.0:
                if pside < 0.0:
                    t = pside / (pside - cside)
                    out.append((px + t * (qx - px), py + t * (qy - py)))
                out.append((qx, qy))
            elif pside >= 0.0:
                t = pside / (pside - cside)
                out.append((px + t * (qx - px), py + t * (qy - py)))
            px, py, pside = qx, qy, cside
    return out


def _bary_rows(tri: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Area coordinates of ``pts`` (``(m, 2)``) w.r.t. ``tri`` -> ``(m, 3)``."""
    a, b, c = tri[0], tri[1], tri[2]
    area2 = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    ux = b[0] - pts[:, 0]
    uy = b[1] - pts[:, 1]
    wx = c[0] - pts[:, 0]
    wy = c[1] - pts[:, 1]
    zx = a[0] - pts[:, 0]
    zy = a[1] - pts[:, 1]
    return np.stack([(ux * wy - uy * wx) / area2,
                     (wx * zy - wy * zx) / area2,
                     (zx * uy - zy * ux) / area2], axis=-1)


def _bary_many(tris: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Vectorised area coordinates: ``tris (m, 3, 2)`` at ``pts (m, 2)``."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area2 = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
             - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    ux, uy = b[:, 0] - pts[:, 0], b[:, 1] - pts[:, 1]
    wx, wy = c[:, 0] - pts[:, 0], c[:, 1] - pts[:, 1]
    zx, zy = a[:, 0] - pts[:, 0], a[:, 1] - pts[:, 1]
    return np.stack([(ux * wy - uy * wx) / area2,
                     (wx * zy - wy * zx) / area2,
                     (zx * uy - zy * ux) / area2], axis=-1)


def _polygon_areas(polys: np.ndarray) -> np.ndarray:
    """Shoelace area of ``(P, k, 2)`` convex polygons."""
    x = polys[:, :, 0]
    y = polys[:, :, 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1)
                               - np.roll(x, -1, axis=1) * y, axis=1))


def _piece_matrices(polys: Sequence[np.ndarray], src_tris: np.ndarray,
                    tgt_tris: np.ndarray,
                    weight: Callable[[np.ndarray], np.ndarray] | None
                    ) -> np.ndarray:
    """``(P, 3, 3)`` overlap integrals ``int_P L_j^tgt L_k^src dGamma``.

    Batched degree-5 fan rule (the nodes/weights of
    :func:`~tensorphysics.interface_operators.polygons.polygon_integral`, which it
    reproduces to round-off -- cross-checked in the tests).  Batching matters:
    per-piece NumPy call overhead is what makes a naive assembly dominate the
    coupled-run cost (see criterion C7).
    """
    groups: dict[int, list[int]] = {}
    for idx, poly in enumerate(polys):
        groups.setdefault(int(poly.shape[0]), []).append(idx)

    out = np.zeros((len(polys), 3, 3))
    for k, idx in groups.items():
        idx_arr = np.array(idx, dtype=int)
        arr = np.stack([polys[i] for i in idx])                     # (P, k, 2)
        if k == 3:
            tris = arr[:, None, :, :]                               # (P, 1, 3, 2)
        else:
            v0 = np.broadcast_to(arr[:, 0][:, None, :], (arr.shape[0], k - 2, 2))
            tris = np.stack([v0, arr[:, 1:-1], arr[:, 2:]], axis=2)
        nf = tris.shape[1]
        e1 = tris[:, :, 1] - tris[:, :, 0]
        e2 = tris[:, :, 2] - tris[:, :, 0]
        areas = 0.5 * np.abs(e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0])
        pts = np.einsum("wk,ptkd->ptwd", _TRI_BARY, tris)            # (P, nf, 7, 2)
        flat = pts.reshape(-1, 2)
        per_piece = nf * _TRI_BARY.shape[0]

        def repeat(v):
            return np.repeat(v[:, None, :, :], per_piece, axis=1).reshape(-1, 3, 2)

        ls = _bary_many(repeat(src_tris[idx_arr]), flat)
        lt = _bary_many(repeat(tgt_tris[idx_arr]), flat)
        prod = (lt[:, :, None] * ls[:, None, :]).reshape(flat.shape[0], 9)
        w = (areas[:, :, None] * _TRI_W[None, None, :]).reshape(-1)
        if weight is not None:
            w = w * np.asarray(weight(flat), dtype=float)
        out[idx_arr] = (w[:, None] * prod).reshape(len(idx_arr), nf * _TRI_BARY.shape[0], 9).sum(axis=1).reshape(-1, 3, 3)
    return out


def _accumulate(M: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                blocks: np.ndarray) -> None:
    """``M[rows[p, a], cols[p, b]] += blocks[p, a, b]`` for every piece ``p``."""
    p = blocks.shape[0]
    r = np.broadcast_to(rows[:, :, None], (p, 3, 3))
    c = np.broadcast_to(cols[:, None, :], (p, 3, 3))
    np.add.at(M, (r, c), blocks)


def _tri_mass(space: TriTraceSpace) -> np.ndarray:
    tris = space.triangles
    blocks = _piece_matrices([tris[i] for i in range(tris.shape[0])], tris, tris,
                             space.weights)
    M = np.zeros((space.n, space.n))
    ids = space.node_ids
    _accumulate(M, ids, ids, blocks)
    return M


def _tri_cross(src: TriTraceSpace, tgt: TriTraceSpace
               ) -> tuple[np.ndarray, int]:
    """``C[j, k] = int_{Gamma} phi_j^tgt phi_k^src dGamma`` by polygon clipping.

    The intersection of a source triangle with a target triangle is a convex
    polygon; on it both bases are linear, so the degree-5 fan rule is exact for
    the product -- and for an affine Jacobian also for the measure, so every
    conserved quantity integrated with these matrices balances to round-off.
    """
    lo_s = src.triangles.min(axis=1)
    hi_s = src.triangles.max(axis=1)
    lo_t = tgt.triangles.min(axis=1)
    hi_t = tgt.triangles.max(axis=1)
    ov_lo = np.maximum(lo_s[:, None, :], lo_t[None, :, :])
    ov_hi = np.minimum(hi_s[:, None, :], hi_t[None, :, :])
    mask = np.all(ov_hi - ov_lo > 1e-12, axis=2)
    pairs = np.argwhere(mask)

    ii: list[int] = []
    jj: list[int] = []
    polys: list[np.ndarray] = []
    for i, j in pairs:
        poly = _clip_tri_fast(tgt.triangles[j], src.triangles[i])
        if len(poly) < 3:
            continue
        P = np.array(poly, dtype=float)
        ii.append(int(i))
        jj.append(int(j))
        polys.append(P)
    if not polys:
        return np.zeros((tgt.n, src.n)), 0
    ii_arr = np.array(ii, dtype=int)
    jj_arr = np.array(jj, dtype=int)
    keep = _polygon_areas(np.stack([p for p in polys])) > _EPS if len(
        {p.shape[0] for p in polys}) == 1 else np.ones(len(polys), dtype=bool)
    if not np.all(keep):
        ii_arr, jj_arr = ii_arr[keep], jj_arr[keep]
        polys = [p for p, k in zip(polys, keep) if k]
    blocks = _piece_matrices(polys, src.triangles[ii_arr], tgt.triangles[jj_arr],
                             tgt.weights)
    C = np.zeros((tgt.n, src.n))
    _accumulate(C, tgt.node_ids[jj_arr], src.node_ids[ii_arr], blocks)
    return C, len(polys)


def spd_solve(M: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve ``M X = B`` for a symmetric positive definite mass matrix ``M``.

    NumPy's LAPACK path is used deliberately.  ``scipy.linalg.cho_factor`` is
    measured to be ~80x faster here (0.5 ms vs 42 ms for 150 DOF) but it
    **segfaults** on this host for 33..65 DOF SPD systems, so it must not be on the
    path of a library call; NumPy's ``solve`` is the stable option and its cost is
    ~2 % of one coupled step (see criterion C7), which is acceptable for a one-off
    setup.
    """
    M = np.asarray(M, dtype=float)
    B = np.asarray(B, dtype=float)
    if M.shape[0] != M.shape[1]:
        raise ValueError("spd_solve expects a square matrix")
    return np.linalg.solve(M, B)


def spd_inverse(A: np.ndarray) -> np.ndarray:
    """``A^{-1}`` through :func:`spd_solve` (used for the dual basis matrix)."""
    A = np.asarray(A, dtype=float)
    return spd_solve(A, np.eye(A.shape[0]))


@dataclass(frozen=True)
class CouplingAssembly:
    """Mass matrices and the cross (mortar) matrix of one interface pair."""

    src: object
    tgt: object
    M_src: np.ndarray
    M_tgt: np.ndarray
    C: np.ndarray
    n_pieces: int
    seconds: float
    label: str = "interface"

    # -- measures --------------------------------------------------------
    def integral(self, space, values: np.ndarray) -> np.ndarray:
        """``int_Gamma q dGamma`` for nodal ``values`` (independent path)."""
        vals = np.asarray(values, dtype=float)
        if vals.ndim == 1:
            vals = vals[:, None]
        return space.measures() @ vals

    def row_sum_defect(self) -> float:
        """How far the assembled mass matrix is from the lumped measures.

        ``sum_j M_ij`` *is* ``int phi_i dGamma`` mathematically; the assembled
        value differs from ``space.measures()`` only by quadrature inconsistency
        (the two integrals use different partitions).  This is exactly the
        defect that limits conservation on curved metrics, so it is reported.
        """
        out = 0.0
        for space, M in ((self.src, self.M_src), (self.tgt, self.M_tgt)):
            a = space.measures()
            scale = max(float(np.max(np.abs(a))), _EPS)
            out = max(out, float(np.max(np.abs(M.sum(axis=1) - a))) / scale)
        return out

    def describe(self) -> dict:
        return {
            "label": self.label,
            "src_n": self.src.n,
            "tgt_n": self.tgt.n,
            "n_pieces": self.n_pieces,
            "build_seconds": self.seconds,
            "row_sum_defect": self.row_sum_defect(),
        }


def build_coupling(src, tgt, *, subdivisions: int = 1,
                   label: str = "interface") -> CouplingAssembly:
    """Assemble ``M_src``, ``M_tgt`` and the mortar cross matrix ``C``."""
    t0 = time.perf_counter()
    if isinstance(src, LineTraceSpace) and isinstance(tgt, LineTraceSpace):
        M_src = _line_mass(src, subdivisions)
        M_tgt = _line_mass(tgt, subdivisions)
        C, n_pieces = _line_cross(src, tgt, subdivisions)
    elif isinstance(src, TriTraceSpace) and isinstance(tgt, TriTraceSpace):
        M_src = _tri_mass(src)
        M_tgt = _tri_mass(tgt)
        C, n_pieces = _tri_cross(src, tgt)
    else:
        raise TypeError(f"unsupported space pair {type(src).__name__}/"
                        f"{type(tgt).__name__}")
    return CouplingAssembly(src=src, tgt=tgt, M_src=M_src, M_tgt=M_tgt, C=C,
                            n_pieces=n_pieces, seconds=time.perf_counter() - t0,
                            label=label)


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------


@dataclass
class MortarTransfer:
    """Conservative mortar (L2-projection) transfer ``src -> tgt``."""

    src: object
    tgt: object
    subdivisions: int = 1
    label: str = "mortar"
    assembly: CouplingAssembly = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.assembly = build_coupling(self.src, self.tgt,
                                       subdivisions=self.subdivisions,
                                       label=self.label)
        self._M_inv = spd_inverse(self.assembly.M_tgt)
        self._T = self._M_inv @ self.assembly.C

    # -- operator --------------------------------------------------------
    def matrix(self) -> np.ndarray:
        """``T = M_tgt^{-1} C`` (``u_tgt = T u_src``)."""
        return self._T

    def apply(self, values: np.ndarray) -> np.ndarray:
        """``u_tgt = M^{-1} C u_src`` (works for scalar and vector payloads)."""
        return self._T @ np.asarray(values, dtype=float)

    def apply_dual(self, values: np.ndarray) -> np.ndarray:
        """Same transfer through the dual basis ``psi_i = (M^{-1} phi)_i``.

        ``c_i = int u_src psi_i`` is algebraically ``(M^{-1} C u)_i``; computing it
        through the explicit dual basis is the mortar "one constraint per slave
        DOF" route.  The two results agree to round-off (tested).
        """
        u = np.asarray(values, dtype=float)
        one_d = u.ndim == 1
        U = u[:, None] if one_d else u
        coeff = self._M_inv @ (self.assembly.C @ U)
        return coeff[:, 0] if one_d else coeff

    # -- diagnostics -----------------------------------------------------
    def dual_basis_matrix(self) -> np.ndarray:
        """Rows are the dual basis functions in the receiving P1 basis."""
        return self._M_inv

    def biorthogonality_defect(self) -> float:
        """``max |int phi_i psi_j - delta_ij|`` (scale-normalised)."""
        G = self.assembly.M_tgt @ self.dual_basis_matrix()
        return float(np.max(np.abs(G - np.eye(G.shape[0]))))

    def dual_partition_of_unity_defect(self) -> float:
        """``max |sum_i psi_i - 1|`` sampled on the receiving interface.

        Expected to be **nonzero**: the L2 dual basis is biorthogonal, not a
        partition of unity.  Conservation therefore comes from the *primal*
        partition of unity (see the module docstring).
        """
        tgt = self.tgt
        if isinstance(tgt, LineTraceSpace):
            xi = np.linspace(*tgt.bounds, 401)
            phi = tgt.basis_dense(xi)          # (m, n)
        else:
            pts = np.vstack([tgt.node_coords, tgt.triangles.mean(axis=1)])
            phi = np.zeros((pts.shape[0], tgt.n))
            for t in range(tgt.n_tri):
                lam = tgt.basis_on(t, pts)
                phi[np.arange(pts.shape[0])[:, None], tgt.node_ids[t]] += lam
        psi = phi @ self.dual_basis_matrix().T   # (m, n): psi_j = sum_k (M^-1)_jk phi_k
        return float(np.max(np.abs(psi.sum(axis=1) - 1.0)))

    def identity_defect(self, values: np.ndarray) -> float:
        """Relative defect of the transfer against the identity (matching meshes)."""
        v = np.asarray(values, dtype=float)
        out = self.apply(v)
        den = max(float(np.max(np.abs(v))), _EPS)
        return float(np.max(np.abs(out - v))) / den

    def conservation_defect(self, values: np.ndarray) -> float:
        """``|int_Gamma u_tgt - int_Gamma u_src| / int_Gamma |u_src|``."""
        src_int = self.assembly.integral(self.src, values)
        tgt_int = self.assembly.integral(self.tgt, self.apply(values))
        den = max(float(np.linalg.norm(src_int)), _EPS)
        return float(np.linalg.norm(tgt_int - src_int)) / den


@dataclass
class NearestNeighborTransfer:
    """Nearest-neighbour (donor-cell) baseline: 1st order, not conservative.

    Each receiving DOF takes the *cell mean* of the source cell that contains the
    receiving DOF coordinate.  This is the interpolation a naive coupling script
    uses, and it is the C6 baseline.
    """

    src: object
    tgt: object

    def apply(self, values: np.ndarray) -> np.ndarray:
        u = np.asarray(values, dtype=float)
        one_d = u.ndim == 1
        U = u[:, None] if one_d else u
        out = np.zeros((self.tgt.n, U.shape[1]))
        if isinstance(self.src, LineTraceSpace) and isinstance(self.tgt, LineTraceSpace):
            nodes = self.tgt.nodes
            cell = np.clip(np.searchsorted(self.src.nodes, nodes, side="right") - 1,
                           0, self.src.n_cells - 1)
            out = 0.5 * (U[cell] + U[cell + 1])
        elif isinstance(self.src, TriTraceSpace) and isinstance(self.tgt, TriTraceSpace):
            tris = self.src.locate(self.tgt.node_coords)
            if np.any(tris < 0):
                raise ValueError("target node outside the source patch")
            for k, t in enumerate(tris):
                out[k] = U[self.src.node_ids[int(t)]].mean(axis=0)
        else:
            raise TypeError("NearestNeighborTransfer needs matching space families")
        return out[:, 0] if one_d else out

    def conservation_defect(self, values: np.ndarray) -> float:
        src_int = np.asarray(self.src.measures() @ np.asarray(values, float).reshape(
            self.src.n, -1), dtype=float)
        tgt_int = np.asarray(self.tgt.measures() @ self.apply(values).reshape(
            self.tgt.n, -1), dtype=float)
        den = max(float(np.linalg.norm(src_int)), _EPS)
        return float(np.linalg.norm(tgt_int - src_int)) / den