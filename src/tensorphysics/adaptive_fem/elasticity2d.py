"""Plane linear-elasticity finite elements (T3/CST) used by the adaptive driver.

The discretisation is deliberately minimal — constant-strain triangles — because
the scientific question here is *mesh adaptivity*, not element technology: a
low-order element makes the ZZ recovery and the adaptive benefit visible without
extra machinery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from .mesh2d import TriMesh, boundary_edges, nodal_weights, signed_areas

__all__ = [
    "Material2D",
    "PlaneSolution",
    "cst_b_matrices",
    "assemble_stiffness",
    "solve_dirichlet",
]


@dataclass(frozen=True)
class Material2D:
    """Isotropic linear elasticity in 2D (plane stress or plane strain)."""

    E: float
    nu: float
    thickness: float = 1.0
    plane: str = "stress"  # "stress" | "strain"

    def __post_init__(self) -> None:
        if self.plane not in ("stress", "strain"):
            raise ValueError("plane must be 'stress' or 'strain'")
        if self.E <= 0.0:
            raise ValueError("E must be positive")
        if not -1.0 < self.nu < 0.5:
            raise ValueError("nu must lie in (-1, 0.5)")

    @property
    def D(self) -> np.ndarray:
        E, nu = self.E, self.nu
        if self.plane == "stress":
            c = E / (1.0 - nu * nu)
            return c * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]])
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return c * np.array(
            [[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)]]
        )

    @property
    def compliance(self) -> np.ndarray:
        """``D^{-1}`` — the strain-energy (compliance) metric on stress space."""
        E, nu = self.E, self.nu
        if self.plane == "stress":
            return (1.0 / E) * np.array(
                [[1.0, -nu, 0.0], [-nu, 1.0, 0.0], [0.0, 0.0, 2.0 * (1.0 + nu)]]
            )
        c = (1.0 + nu) / E
        return c * np.array(
            [[1.0 - nu, -nu, 0.0], [-nu, 1.0 - nu, 0.0], [0.0, 0.0, 2.0]]
        )

    @property
    def E_eff(self) -> float:
        """Effective modulus used by 2-D crack-tip relations (``E'``)."""
        if self.plane == "stress":
            return self.E
        return self.E / (1.0 - self.nu * self.nu)

    @property
    def shear_modulus(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def kolosov(self) -> float:
        """Kolosov constant ``kappa`` (plane stress: (3-nu)/(1+nu), strain: 3-4nu)."""
        if self.plane == "stress":
            return (3.0 - self.nu) / (1.0 + self.nu)
        return 3.0 - 4.0 * self.nu


def cst_b_matrices(nodes: np.ndarray, tris: np.ndarray):
    """Constant-strain ``B`` matrices ``(m,3,6)`` and element areas ``(m,)``."""
    x = nodes[tris, 0]
    y = nodes[tris, 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    area2 = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    area = 0.5 * area2
    if np.any(area <= 0.0):
        raise ValueError("all triangles must be CCW with positive area")
    inv2a = 1.0 / area2
    B = np.zeros((len(tris), 3, 6))
    B[:, 0, 0::2] = b * inv2a[:, None]
    B[:, 1, 1::2] = c * inv2a[:, None]
    B[:, 2, 0::2] = c * inv2a[:, None]
    B[:, 2, 1::2] = b * inv2a[:, None]
    return B, area


def assemble_stiffness(
    nodes: np.ndarray, tris: np.ndarray, mat: Material2D
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Assemble the global stiffness matrix; returns ``(K, B, area)``."""
    B, area = cst_b_matrices(nodes, tris)
    D = mat.D
    ke = np.einsum("mki,kl,mlj->mij", B, D, B) * (area * mat.thickness)[:, None, None]
    dofs = np.stack([2 * tris, 2 * tris + 1], axis=2).reshape(len(tris), 6)
    rows = np.repeat(dofs, 6, axis=1).ravel()
    cols = np.tile(dofs, (1, 6)).ravel()
    K = coo_matrix((ke.ravel(), (rows, cols)), shape=(2 * len(nodes), 2 * len(nodes))).tocsr()
    return K, B, area


def solve_dirichlet(K: csr_matrix, f: np.ndarray, fixed: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Solve ``K u = f`` with inhomogeneous Dirichlet data on ``fixed`` dofs."""
    n = K.shape[0]
    u = np.zeros(n, dtype=float)
    fixed = np.asarray(fixed, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    if fixed.size:
        u[fixed] = values
    free = np.setdiff1d(np.arange(n), fixed, assume_unique=False)
    if free.size == 0:
        return u
    Kff = K[free][:, free].tocsc()
    rhs = f[free] - K[free][:, fixed] @ values if fixed.size else f[free]
    u[free] = spsolve(Kff, rhs)
    return u


def element_stress(B: np.ndarray, u: np.ndarray, tris: np.ndarray, mat: Material2D) -> np.ndarray:
    """Element stress in Voigt form ``(sigma_xx, sigma_yy, sigma_xy)``, shape ``(m,3)``."""
    ue = u[np.stack([2 * tris, 2 * tris + 1], axis=2).reshape(len(tris), 6)]
    strain = np.einsum("mij,mj->mi", B, ue)
    return strain @ mat.D.T


@dataclass
class PlaneSolution:
    """Result of a plane-elasticity solve on one mesh."""

    mesh: TriMesh
    material: Material2D
    u: np.ndarray
    stress: np.ndarray  # (m,3) element stress, Voigt
    area: np.ndarray  # (m,) element area
    B: np.ndarray

    @property
    def nodal_weights(self) -> np.ndarray:
        return nodal_weights(self.mesh.nodes, self.mesh.tris)

    def energy_norm_sq(self, stress: np.ndarray | None = None) -> float:
        """``||sigma||_E^2 = int sigma^T D^-1 sigma dA`` (thickness included)."""
        s = self.stress if stress is None else stress
        dens = element_energy_norm_sq(s, self.area, self.material)
        return float(np.sum(dens * self.area) * self.material.thickness)

    def energy_error_sq(self, other_stress: np.ndarray) -> float:
        """``||sigma_h - other||_E^2`` (thickness included)."""
        dens = element_energy_norm_sq(self.stress - other_stress, self.area, self.material)
        return float(np.sum(dens * self.area) * self.material.thickness)

    def strain_energy(self) -> float:
        """Total strain energy ``0.5 * int sigma^T D^-1 sigma dA``."""
        return 0.5 * self.energy_norm_sq()


def element_energy_norm_sq(stress: np.ndarray, area: np.ndarray, mat: Material2D) -> np.ndarray:
    """Per-element integrand of ``||sigma||_E^2 = int sigma^T D^-1 sigma dA`` (w/o thickness)."""
    S = mat.compliance
    return np.einsum("mi,ij,mj->m", stress, S, stress)