"""Structured Q1 finite-element discretisation for the three verification cases.

All three physics keep the same spatial discretisation (trilinear/quadrilateral
Q1 isoparametric elements on a tensor-product grid) so that the parallel
infrastructure -- assembly, halo exchange, PCG -- is exercised by one code path
and only the element kernel changes:

``poisson``
    ``-Laplace(u) = f`` (2-D/3-D), bilinear ``Q1`` stiffness.
``heat``
    ``-div(k(x) grad T) = f`` with a spatially varying conductivity ``k``
    evaluated at the Gauss points (variable-coefficient diffusion).
``elasticity``
    Plane-strain / 3-D linear elasticity ``-div(sigma(u)) = f`` with the
    isotropic elasticity matrix ``D``; ``ncomp = ndim``.

Element matrices are integrated with ``2**ndim`` Gauss points on the reference
cell ``[-1,1]**ndim`` and mapped with the diagonal Cartesian Jacobian, so the
element kernel is exact for the polynomial integrands of a rectangular cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "StructuredGrid",
    "ElementKernel",
]

CORNERS_CACHE: Dict[int, np.ndarray] = {}


def reference_corners(ndim: int) -> np.ndarray:
    """Corner coordinates of the reference cell in ``[-1, 1]**ndim``."""
    if ndim not in CORNERS_CACHE:
        CORNERS_CACHE[ndim] = np.array(list(product((-1.0, 1.0), repeat=ndim)))
    return CORNERS_CACHE[ndim]


def _corner_index(corner: Sequence[int]) -> int:
    """Local node number of a ``0/1`` corner tuple (last dim fastest)."""
    idx = 0
    ndim = len(corner)
    for d, c in enumerate(corner):
        idx += int(c) << (ndim - 1 - d)
    return idx


def q1_gauss(ndim: int):
    """2-point Gauss rule on the reference cell: points and weights."""
    g = 1.0 / np.sqrt(3.0)
    pts = np.array(list(product((-g, g), repeat=ndim)))
    wts = np.ones(pts.shape[0])
    return pts, wts


def q1_shape_grad(pts: np.ndarray, ndim: int):
    """Q1 shape functions and reference-space gradients at each Gauss point.

    Returns ``(N, dN_dxi)`` with shapes ``(ngp, 2**ndim)`` and
    ``(ngp, 2**ndim, ndim)``.
    """
    corners = reference_corners(ndim)
    ngp = pts.shape[0]
    nloc = corners.shape[0]
    N = np.ones((ngp, nloc))
    dN = np.zeros((ngp, nloc, ndim))
    for a in range(nloc):
        for d in range(ndim):
            s = corners[a, d]
            N[:, a] *= 0.5 * (1.0 + s * pts[:, d])
        for d in range(ndim):
            val = np.full(ngp, 0.5 * corners[a, d])
            for e in range(ndim):
                if e == d:
                    continue
                val = val * 0.5 * (1.0 + corners[a, e] * pts[:, e])
            dN[:, a, d] = val
    return N, dN


@dataclass(frozen=True)
class StructuredGrid:
    """Tensor-product grid: ``n[d]`` nodes per dimension, ``lengths[d]`` extent."""

    n: Tuple[int, ...]
    lengths: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", tuple(int(v) for v in self.n))
        lengths = self.lengths or tuple(1.0 for _ in self.n)
        object.__setattr__(self, "lengths", tuple(float(v) for v in lengths))
        if len(self.n) != len(self.lengths):
            raise ValueError("n and lengths must have the same length")
        if any(v < 2 for v in self.n):
            raise ValueError(f"need at least 2 nodes per dim, got {self.n}")
        strides = []
        s = 1
        for v in self.n:
            strides.append(s)
            s *= v
        object.__setattr__(self, "strides", tuple(int(v) for v in strides))

    @property
    def ndim(self) -> int:
        return len(self.n)

    @property
    def num_nodes(self) -> int:
        return int(np.prod(self.n))

    @property
    def nel(self) -> Tuple[int, ...]:
        return tuple(v - 1 for v in self.n)

    @property
    def h(self) -> Tuple[float, ...]:
        return tuple(self.lengths[d] / (self.n[d] - 1) for d in range(self.ndim))

    @property
    def nloc(self) -> int:
        return 2 ** self.ndim

    def gid(self, multi: Sequence[np.ndarray]) -> np.ndarray:
        out = np.zeros(np.shape(multi[0]), dtype=np.int64)
        for d, arr in enumerate(multi):
            out = out + np.asarray(arr, dtype=np.int64) * self.strides[d]
        return out

    def multi(self, gids: np.ndarray):
        gids = np.asarray(gids, dtype=np.int64)
        out = []
        for d in range(self.ndim):
            out.append((gids // self.strides[d]) % self.n[d])
        return out

    def coords(self, gids: np.ndarray) -> np.ndarray:
        """Physical coordinates ``(n, ndim)`` of the given global node ids."""
        multi = self.multi(gids)
        h = self.h
        return np.stack([multi[d] * h[d] for d in range(self.ndim)], axis=1)

    def corner_positions(self, multi: Sequence[np.ndarray], corner) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-index + inside-domain mask of element-lower-corner ``+ corner``."""
        inside = np.ones(np.shape(multi[0]), dtype=bool)
        out = []
        for d in range(self.ndim):
            pos = np.asarray(multi[d], dtype=np.int64) + int(corner[d])
            inside &= (pos >= 0) & (pos <= self.n[d] - 1)
            out.append(pos)
        return out, inside


@dataclass
class ElementKernel:
    """Element-level matrices for one physics type on a rectangular cell."""

    kind: str
    ndim: int
    h: Tuple[float, ...]
    material: Dict[str, float] = field(default_factory=dict)
    source: Optional[Callable[[np.ndarray], np.ndarray]] = None
    conductivity: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __post_init__(self) -> None:
        self.nloc = 2 ** self.ndim
        self.ncomp = self.ndim if self.kind == "elasticity" else 1
        self._pts, self._wts = q1_gauss(self.ndim)
        self._N, self._dNref = q1_shape_grad(self._pts, self.ndim)
        # reference -> physical gradient scaling and Jacobian determinant
        scale = np.array([2.0 / self.h[d] for d in range(self.ndim)])
        self._dN = self._dNref * scale  # (ngp, nloc, ndim)
        self._detJ = float(np.prod([self.h[d] / 2.0 for d in range(self.ndim)]))
        self._weights = self._wts * self._detJ
        centroid = np.array([self.h[d] / 2.0 for d in range(self.ndim)])
        self._gp_coords = centroid + self._pts * np.array(self.h) / 2.0

        if self.kind == "elasticity":
            self._D = self._elasticity_matrix()

    # -- helpers -----------------------------------------------------------
    def _elasticity_matrix(self) -> np.ndarray:
        E = float(self.material.get("E", 210e9))
        nu = float(self.material.get("nu", 0.3))
        if self.ndim == 2:
            mode = self.material.get("mode", "plane_strain")
            if mode == "plane_stress":
                c = E / (1.0 - nu ** 2)
                return c * np.array(
                    [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
                )
            c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
            return c * np.array(
                [[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]]
            )
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        D = np.zeros((6, 6))
        D[:3, :3] = lam
        D[np.arange(3), np.arange(3)] += 2.0 * mu
        D[np.arange(3, 6), np.arange(3, 6)] = mu
        return D

    def _B_matrix(self, gp: int) -> np.ndarray:
        """Strain-displacement matrix at Gauss point ``gp`` (dof ordering node-major)."""
        nloc, ncomp = self.nloc, self.ncomp
        dN = self._dN[gp]  # (nloc, ndim)
        if self.kind != "elasticity":
            return dN.reshape(1, nloc)
        if self.ndim == 2:
            B = np.zeros((3, nloc * 2))
            for a in range(nloc):
                B[0, 2 * a] = dN[a, 0]
                B[1, 2 * a + 1] = dN[a, 1]
                B[2, 2 * a] = dN[a, 1]
                B[2, 2 * a + 1] = dN[a, 0]
            return B
        B = np.zeros((6, nloc * 3))
        for a in range(nloc):
            dx, dy, dz = dN[a]
            B[0, 3 * a] = dx
            B[1, 3 * a + 1] = dy
            B[2, 3 * a + 2] = dz
            B[3, 3 * a] = dy
            B[3, 3 * a + 1] = dx
            B[4, 3 * a + 1] = dz
            B[4, 3 * a + 2] = dy
            B[5, 3 * a] = dz
            B[5, 3 * a + 2] = dx
        return B

    # -- public API --------------------------------------------------------
    @property
    def is_constant(self) -> bool:
        """True when one element matrix serves every cell (constant coefficients)."""
        return self.conductivity is None

    def element_matrix(self) -> np.ndarray:
        """Local element matrix ``(nloc*ncomp) x (nloc*ncomp)`` for an interior cell."""
        return self.element_matrices(np.zeros((1, self.ndim)))[0]

    def element_matrices(self, centers: np.ndarray) -> np.ndarray:
        """Per-element matrices for the cells with the given centres.

        Shape ``(n_elem, ndof, ndof)``; for a constant-coefficient physics the
        entries of the first axis are identical (and :meth:`is_constant` lets
        the assembler use a single matrix instead of the replicated stack).
        """
        centers = np.asarray(centers, dtype=float).reshape(-1, self.ndim)
        n = centers.shape[0]
        ndof = self.nloc * self.ncomp
        h = np.array(self.h, dtype=float)
        Ke = np.zeros((n, ndof, ndof))
        for gp in range(self._pts.shape[0]):
            w = self._weights[gp]
            if self.kind == "elasticity":
                B = self._B_matrix(gp)
                Ke += w * (B.T @ self._D @ B)
                continue
            dN = self._dN[gp]  # (nloc, ndim)
            G = dN @ dN.T  # (nloc, nloc)
            if self.conductivity is None:
                Ke += w * G
            else:
                xg = centers + self._pts[gp] * h / 2.0
                k = np.asarray(self.conductivity(xg), dtype=float).ravel()
                Ke += w * k[:, None, None] * G[None, :, :]
        return Ke

    def element_load(self) -> np.ndarray:
        """Consistent load vector of one (first) cell -- convenience wrapper."""
        return self.element_loads(np.zeros((1, self.ndim)))[0]

    def element_loads(self, centers: np.ndarray) -> np.ndarray:
        """Consistent load vectors ``int N_a f dV`` at the given cell centres."""
        centers = np.asarray(centers, dtype=float).reshape(-1, self.ndim)
        n = centers.shape[0]
        ncomp = self.ncomp
        fe = np.zeros((n, self.nloc * ncomp))
        if self.source is None:
            return fe
        h = np.array(self.h, dtype=float)
        for gp in range(self._pts.shape[0]):
            w = self._weights[gp]
            xg = centers + self._pts[gp] * h / 2.0
            f = np.asarray(self.source(xg), dtype=float).reshape(n, -1)
            if f.shape[1] == 1 and ncomp > 1:
                f = np.repeat(f, ncomp, axis=1)
            contrib = w * self._N[gp][None, :, None] * f[:, None, :]
            fe += contrib.reshape(n, self.nloc * ncomp)
        return fe
