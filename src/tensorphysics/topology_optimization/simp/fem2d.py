"""Plane-stress Q4 finite-element kernel for the SIMP verification suite.

Deliberately small and readable (numpy + scipy.sparse only): the point of this
module is to be an auditable anchor for the SIMP density-filter and adjoint
sensitivity surfaces, not a production FE code.  Element numbering follows the
classic 88-line topopt convention (row-major, bottom-left origin), so results
can be compared with the published MBB / cantilever benchmarks.

The kernel is verified independently in the test suite against the exact
constant-strain patch response and rigid-body modes (zero energy).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "DEFAULT_NU",
    "Q4Mesh",
    "assemble_stiffness",
    "compliance_and_displacements",
    "plane_stress_D",
    "q4_unit_stiffness",
    "solve_dirichlet",
]

DEFAULT_NU = 0.3
_GAUSS = 1.0 / math.sqrt(3.0)
_GP = ((-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS))


def plane_stress_D(E: float, nu: float) -> np.ndarray:
    """Isotropic plane-stress constitutive matrix (3x3)."""
    return E / (1.0 - nu * nu) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]],
        dtype=float,
    )


def q4_unit_stiffness(hx: float = 1.0, hy: float = 1.0, nu: float = DEFAULT_NU) -> np.ndarray:
    """8x8 stiffness of a rectangular Q4 element with ``E = 1``.

    ``hx``/``hy`` are the element edge lengths (default 1 -- the SIMP meshes
    work in element units).  Integrated with the standard 2x2 Gauss rule,
    which is exact for this (bilinear, constant-Jacobian) element.
    """
    D = plane_stress_D(1.0, nu)
    k = np.zeros((8, 8), dtype=float)
    for xi, eta in _GP:
        # dN/d(xi, eta) for the four bilinear shape functions
        dN_dxi = 0.25 * np.array(
            [
                [-(1.0 - eta), -(1.0 - xi)],
                [(1.0 - eta), -(1.0 + xi)],
                [(1.0 + eta), (1.0 + xi)],
                [-(1.0 + eta), (1.0 - xi)],
            ]
        )
        # J is diagonal for an axis-aligned rectangle: invert analytically
        # (a general 2x2 inverse here costs more than the whole element loop).
        dN_dx = dN_dxi / np.array([0.5 * hx, 0.5 * hy])  # (4, 2)
        B = np.zeros((3, 8), dtype=float)
        B[0, 0::2] = dN_dx[:, 0]
        B[1, 1::2] = dN_dx[:, 1]
        B[2, 0::2] = dN_dx[:, 1]
        B[2, 1::2] = dN_dx[:, 0]
        k += B.T @ D @ B * (0.25 * hx * hy)  # det(J) * weight(=1)
    return k


class Q4Mesh:
    """Structured ``nelx x nely`` plane-stress mesh in element-length units."""

    def __init__(self, nelx: int, nely: int, hx: float = 1.0, hy: float = 1.0,
                 nu: float = DEFAULT_NU) -> None:
        if nelx < 1 or nely < 1:
            raise ValueError("nelx and nely must be >= 1")
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.hx = float(hx)
        self.hy = float(hy)
        self.nu = float(nu)

    # -- geometry -----------------------------------------------------------
    @property
    def n_elems(self) -> int:
        return self.nelx * self.nely

    @property
    def shape(self) -> tuple[int, int]:
        """Density array shape (``nely``, ``nelx``) -- image convention."""
        return (self.nely, self.nelx)

    @property
    def n_nodes(self) -> int:
        return (self.nelx + 1) * (self.nely + 1)

    @property
    def n_dof(self) -> int:
        return 2 * self.n_nodes

    def node_index(self, ix: int, iy: int) -> int:
        return iy * (self.nelx + 1) + ix

    def node_dofs(self, ix: int, iy: int) -> tuple[int, int]:
        n = self.node_index(ix, iy)
        return (2 * n, 2 * n + 1)

    def element_dofs(self) -> np.ndarray:
        """``(n_elems, 8)`` DOF table, counter-clockwise node ordering."""
        ys, xs = np.meshgrid(np.arange(self.nely), np.arange(self.nelx), indexing="ij")
        ex = xs.ravel()
        ey = ys.ravel()
        n1 = ey * (self.nelx + 1) + ex
        n2 = n1 + 1
        n3 = n2 + (self.nelx + 1)
        n4 = n1 + (self.nelx + 1)
        n = np.column_stack([n1, n2, n3, n4])
        dofs = np.empty((n.shape[0], 8), dtype=np.int64)
        dofs[:, 0::2] = 2 * n
        dofs[:, 1::2] = 2 * n + 1
        return dofs

    def element_mask(self, ix: int, iy: int) -> np.ndarray:
        """Boolean ``(n_elems,)`` mask of the elements inside an ``(ix, iy)`` box."""
        ys, xs = np.meshgrid(np.arange(self.nely), np.arange(self.nelx), indexing="ij")
        return (xs.ravel() <= ix) & (ys.ravel() <= iy)

    @property
    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Nodal coordinates ``(x, y)`` in element units."""
        iy, ix = np.meshgrid(np.arange(self.nely + 1), np.arange(self.nelx + 1), indexing="ij")
        return ix.ravel() * self.hx, iy.ravel() * self.hy

    def unit_stiffness(self) -> np.ndarray:
        cached = getattr(self, "_unit_k", None)
        if cached is None:
            cached = q4_unit_stiffness(self.hx, self.hy, self.nu)
            self._unit_k = cached
        return cached


def assemble_stiffness(mesh: Q4Mesh, E_elem: np.ndarray | float) -> sp.csr_matrix:
    """Assemble the global stiffness from per-element moduli.

    ``E_elem`` is either a scalar or a length-``mesh.n_elems`` array ordered
    like ``mesh.element_dofs()`` (row-major, y outer).
    """
    E_elem = np.asarray(E_elem, dtype=float)
    if E_elem.ndim == 0:
        E_elem = np.full(mesh.n_elems, float(E_elem))
    if E_elem.size != mesh.n_elems:
        raise ValueError(f"expected {mesh.n_elems} element moduli, got {E_elem.size}")

    dofs = mesh.element_dofs()
    k = mesh.unit_stiffness()
    ke = (E_elem[:, None, None] * k[None, :, :]).ravel()  # (n_elems*64,)
    rows = np.repeat(dofs, 8, axis=1).ravel()
    cols = np.tile(dofs, (1, 8)).ravel()
    K = sp.coo_matrix((ke, (rows, cols)), shape=(mesh.n_dof, mesh.n_dof))
    return K.tocsr()


def solve_dirichlet(K: sp.csr_matrix, F: np.ndarray, fixed: Sequence[int]) -> np.ndarray | None:
    """Solve ``K u = F`` with homogeneous Dirichlet data on ``fixed``.

    Returns ``None`` when the reduced system is singular (e.g. a design that
    disintegrates into disconnected islands under a black-and-white
    interpretation) -- callers report that as non-physical rather than
    silently propagating garbage.
    """
    fixed = np.asarray(sorted(set(int(d) for d in fixed)), dtype=np.int64)
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    Kr = K[free][:, free].tocsc()
    Fr = np.asarray(F, dtype=float)[free]
    try:
        with np.errstate(all="raise"):
            ur = spla.spsolve(Kr, Fr)
    except Exception:  # singular reduced system -> treated as non-physical
        return None
    if not np.all(np.isfinite(ur)):
        return None
    # A singular (or numerically rank-deficient, e.g. disconnected island)
    # system can still return finite garbage -- verify the residual instead of
    # trusting the solver.
    scale = max(float(np.linalg.norm(Fr)), 1e-300)
    if float(np.linalg.norm(Kr @ ur - Fr)) > 1e-6 * scale:
        return None
    u = np.zeros(K.shape[0], dtype=float)
    u[free] = ur
    return u


def compliance_and_displacements(
    mesh: Q4Mesh,
    rho: np.ndarray,
    F: np.ndarray,
    fixed: Iterable[int],
    *,
    penal: float = 3.0,
    E0: float = 1.0,
    Emin: float = 1e-9,
) -> tuple[float | None, np.ndarray | None]:
    """SIMP compliance ``C = F^T u`` and the displacement field.

    ``rho`` is flattened row-major (y outer) -- use ``rho.ravel()`` on the
    ``(nely, nelx)`` image array.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    E_elem = Emin + np.power(rho, penal) * (E0 - Emin)
    K = assemble_stiffness(mesh, E_elem)
    u = solve_dirichlet(K, F, fixed)
    if u is None:
        return None, None
    return float(F @ u), u