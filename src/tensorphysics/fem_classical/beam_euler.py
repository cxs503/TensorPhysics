"""Euler-Bernoulli beam element (2D, 2 DOF per node: w, θ).

A cantilever beam is discretised into N equal-length Euler-Bernoulli beam
elements.  Each node carries transverse displacement *w* and rotation *θ*.
The element stiffness matrix is the classical cubic-Hermite beam:

    K_e = (EI / L³) · [[ 12,  6L, -12,  6L],
                        [ 6L, 4L², -6L, 2L²],
                        [-12, -6L,  12, -6L],
                        [ 6L, 2L², -6L, 4L²]]

The consistent load vector for a uniform load *q* on one element is:

    f_e = [qL/2, qL²/12, qL/2, −qL²/12]ᵀ

A concentrated end load *P* is applied directly as a nodal force at the
free-end node.

Boundary conditions for a cantilever: w = 0 and θ = 0 at the fixed end
(node 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["EulerBernoulliBeam2D"]


@dataclass(frozen=True)
class EulerBernoulliBeam2D:
    """Euler-Bernoulli cantilever beam solver.

    Parameters
    ----------
    L : float
        Total beam length [m].
    E : float
        Young's modulus [Pa].
    I : float
        Second moment of area [m⁴].
    n_elem : int
        Number of equal-length beam elements.
    """

    L: float
    E: float
    I: float
    n_elem: int

    @property
    def le(self) -> float:
        """Element length."""
        return self.L / self.n_elem

    @property
    def n_nodes(self) -> int:
        return self.n_elem + 1

    @property
    def ndof(self) -> int:
        """Total DOFs (2 per node: w, θ)."""
        return 2 * self.n_nodes

    def _element_stiffness(self) -> np.ndarray:
        """4×4 element stiffness matrix."""
        L = self.le
        EI = self.E * self.I
        c = EI / L ** 3
        return c * np.array([
            [12,  6 * L, -12,  6 * L],
            [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2],
        ])

    def _element_load_udl(self, q: float) -> np.ndarray:
        """Consistent load vector for uniform load q on one element."""
        L = self.le
        return q * np.array([L / 2, L ** 2 / 12, L / 2, -L ** 2 / 12])

    def assemble_stiffness(self) -> sp.csr_matrix:
        """Assemble global stiffness matrix (sparse)."""
        Ke = self._element_stiffness()
        ndof = self.ndof
        rows, cols, vals = [], [], []
        for e in range(self.n_elem):
            dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            for i in range(4):
                for j in range(4):
                    rows.append(dofs[i])
                    cols.append(dofs[j])
                    vals.append(Ke[i, j])
        K = sp.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))
        return K

    def assemble_load_udl(self, q: float) -> np.ndarray:
        """Assemble global load vector for uniform load q."""
        f = np.zeros(self.ndof)
        fe = self._element_load_udl(q)
        for e in range(self.n_elem):
            dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            for i in range(4):
                f[dofs[i]] += fe[i]
        return f

    def solve(self, load: np.ndarray) -> np.ndarray:
        """Solve Ku = f with cantilever BCs (w=0, θ=0 at node 0).

        Parameters
        ----------
        load : ndarray
            Global load vector (ndof,).

        Returns
        -------
        ndarray
            Displacement vector (ndof,).
        """
        K = self.assemble_stiffness()
        # Fixed DOFs: node 0 → DOFs 0 (w) and 1 (θ)
        fixed = np.array([0, 1])
        free = np.setdiff1d(np.arange(self.ndof), fixed)

        Kff = K[np.ix_(free, free)].tocsc()
        ff = load[free]
        u_free = spla.spsolve(Kff, ff)

        u = np.zeros(self.ndof)
        u[free] = u_free
        return u

    def solve_point_load(self, P: float) -> np.ndarray:
        """Solve for a concentrated end load P at the free tip.

        Parameters
        ----------
        P : float
            Point load at free end [N] (positive downward).

        Returns
        -------
        ndarray
            Displacement vector (ndof,).
        """
        f = np.zeros(self.ndof)
        f[-2] = P  # w at last node
        return self.solve(f)

    def solve_udl(self, q: float) -> np.ndarray:
        """Solve for uniform distributed load q.

        Parameters
        ----------
        q : float
            Distributed load intensity [N/m] (positive downward).

        Returns
        -------
        ndarray
            Displacement vector (ndof,).
        """
        f = self.assemble_load_udl(q)
        return self.solve(f)

    def tip_deflection(self, u: np.ndarray) -> float:
        """Extract tip transverse displacement."""
        return u[-2]

    def tip_rotation(self, u: np.ndarray) -> float:
        """Extract tip rotation."""
        return u[-1]

    def condition_number_no_bc(self) -> float:
        """Condition number of the stiffness matrix WITHOUT boundary conditions.

        Used as a negative control: a structure with no constraints must have
        a singular (or near-singular) stiffness matrix.
        """
        K = self.assemble_stiffness().toarray()
        # Use SVD for robust condition number
        s = np.linalg.svd(K, compute_uv=False)
        return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
