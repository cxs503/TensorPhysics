"""Stage 1a -- Euler-Bernoulli FEM beam, bending stress field and failure faces.

This is a real displacement-based finite element model (cubic Hermite shape
functions, 2-node beam elements, 2 DOF per node).  For a cantilever with a tip
load the FE solution is nodally exact for ``w`` and ``theta`` and the bending
moment field ``M(x) = P (L - x)`` is reproduced element-wise, so the analytic
Euler-Bernoulli reference ``delta = P L^3 / (3 E I)``, ``U = P^2 L^3 / (6 E I)``
can be used as truth.

The module also extracts the **failure face** the DEM stage consumes: position,
outward normal, area, traction profile ``t(y) = sigma_xx(x_f, y)`` across the
section and the fixed-grip LEFM energy-release rate
``G = M^2 / (2 E I b)`` (per unit crack area), the quantity that decides whether
a crack can propagate for a given fracture energy ``G_f``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BeamSection",
    "BeamSolution",
    "FailureFace",
    "element_stiffness",
    "solve_cantilever",
    "extract_failure_face",
    "flexural_strength_load",
]


@dataclass(frozen=True)
class BeamSection:
    """Rectangular beam cross-section (a plate strip of width ``b``)."""

    E: float
    b: float
    h: float

    @property
    def A(self) -> float:
        return self.b * self.h

    @property
    def I(self) -> float:
        return self.b * self.h**3 / 12.0

    @property
    def EI(self) -> float:
        return self.E * self.I

    @property
    def section_modulus(self) -> float:
        return self.I / (self.h / 2.0)

    def max_bending_stress(self, moment: float) -> float:
        return abs(moment) / self.section_modulus


@dataclass
class BeamSolution:
    section: BeamSection
    L: float
    P: float
    n_elem: int
    x_nodes: np.ndarray
    w: np.ndarray
    theta: np.ndarray
    x_elem: np.ndarray
    M_elem: np.ndarray
    U_elem: np.ndarray
    M_left: np.ndarray = None  # type: ignore[assignment]
    M_right: np.ndarray = None  # type: ignore[assignment]

    @property
    def tip_deflection(self) -> float:
        return float(self.w[-1])

    @property
    def U(self) -> float:
        return float(self.U_elem.sum())

    @property
    def M_root(self) -> float:
        """Bending moment at the clamp (``M_elem[0]`` is the element average)."""
        return float(self.moment_at(0.0)[0])

    @property
    def max_stress(self) -> float:
        return self.section.max_bending_stress(self.M_root)

    @property
    def energy_release_rate(self) -> float:
        """Fixed-grip LEFM release rate at the root, per unit crack area."""
        return self.M_root**2 / (2.0 * self.section.EI * self.section.b)

    def analytic_tip_deflection(self) -> float:
        return self.P * self.L**3 / (3.0 * self.section.EI)

    def analytic_strain_energy(self) -> float:
        return self.P**2 * self.L**3 / (6.0 * self.section.EI)

    def moment_at(self, x):
        """Bending moment by FE interpolation (element-wise linear)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        le = self.L / self.n_elem
        idx = np.clip((x / le).astype(int), 0, self.n_elem - 1)
        xi = np.clip(x / le - idx, 0.0, 1.0)
        if self.M_left is None:
            m0 = self.M_elem[idx]
            m1 = self.M_elem[np.clip(idx + 1, 0, self.n_elem - 1)]
        else:
            m0 = self.M_left[idx]
            m1 = self.M_right[idx]
        return m0 + (m1 - m0) * xi

    def curvature_at(self, x):
        return self.moment_at(x) / self.section.EI

    def stress(self, x, y):
        """Bending stress ``sigma_xx = M(x) y / I``."""
        return np.asarray(self.moment_at(x)) * np.asarray(y) / self.section.I

    def deflection_at(self, x):
        """Deflection by Hermite interpolation of the nodal solution."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        le = self.L / self.n_elem
        idx = np.clip((x / le).astype(int), 0, self.n_elem - 1)
        xi = np.clip(x / le - idx, 0.0, 1.0)
        w1 = self.w[idx]
        t1 = self.theta[idx]
        w2 = self.w[np.clip(idx + 1, 0, self.n_elem)]
        t2 = self.theta[np.clip(idx + 1, 0, self.n_elem)]
        n1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
        n2 = le * (xi - 2.0 * xi**2 + xi**3)
        n3 = 3.0 * xi**2 - 2.0 * xi**3
        n4 = le * (-xi**2 + xi**3)
        return n1 * w1 + n2 * t1 + n3 * w2 + n4 * t2


def element_stiffness(EI: float, le: float) -> np.ndarray:
    """Classic 4x4 Euler-Bernoulli element stiffness matrix."""
    c = EI / le**3
    return c * np.array(
        [
            [12.0, 6.0 * le, -12.0, 6.0 * le],
            [6.0 * le, 4.0 * le**2, -6.0 * le, 2.0 * le**2],
            [-12.0, -6.0 * le, 12.0, -6.0 * le],
            [6.0 * le, 2.0 * le**2, -6.0 * le, 4.0 * le**2],
        ]
    )


def _shape_second_derivative(xi: float, le: float) -> np.ndarray:
    return np.array(
        [
            (-6.0 + 12.0 * xi) / le**2,
            (-4.0 + 6.0 * xi) / le,
            (6.0 - 12.0 * xi) / le**2,
            (-2.0 + 6.0 * xi) / le,
        ]
    )


def flexural_strength_load(
    section: BeamSection, L: float, sigma_f: float, load_factor: float = 1.0
) -> float:
    """Tip load giving the flexural strength at the clamp, times ``load_factor``."""
    moment = sigma_f * section.section_modulus
    return load_factor * moment / L


def solve_cantilever(
    section: BeamSection, L: float, P: float, n_elem: int = 200
) -> BeamSolution:
    """Solve a cantilever under a transverse tip load ``P`` with beam FEs."""
    le = L / n_elem
    ndof = 2 * (n_elem + 1)
    K = np.zeros((ndof, ndof))
    ke = element_stiffness(section.EI, le)
    for e in range(n_elem):
        dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(dofs, dofs)] += ke

    F = np.zeros(ndof)
    F[2 * n_elem] = P

    free = np.arange(2, ndof)  # clamp node 0: w = theta = 0
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

    w = u[0::2]
    theta = u[1::2]

    # element moments and strain energy by 3-point Gauss quadrature
    gauss_xi = 0.5 * (1.0 + np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]))
    gauss_w = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])
    M_elem = np.zeros(n_elem)
    U_elem = np.zeros(n_elem)
    M_left = np.zeros(n_elem)
    M_right = np.zeros(n_elem)
    for e in range(n_elem):
        d = u[2 * e : 2 * e + 4]
        N2 = np.array([_shape_second_derivative(x, le) for x in gauss_xi])  # (3,4)
        kappa = N2 @ d
        M_elem[e] = section.EI * float(np.dot(gauss_w, kappa))
        U_elem[e] = 0.5 * section.EI * le * float(np.dot(gauss_w, kappa**2))
        M_left[e] = section.EI * float(_shape_second_derivative(0.0, le) @ d)
        M_right[e] = section.EI * float(_shape_second_derivative(1.0, le) @ d)

    x_elem = (np.arange(n_elem) + 0.5) * le
    return BeamSolution(
        section=section,
        L=L,
        P=P,
        n_elem=n_elem,
        x_nodes=np.arange(n_elem + 1) * le,
        w=w,
        theta=theta,
        x_elem=x_elem,
        M_elem=M_elem,
        U_elem=U_elem,
        M_left=M_left,
        M_right=M_right,
    )


@dataclass(frozen=True)
class FailureFace:
    """A fracture plane handed from FEM to DEM."""

    face_id: int
    x: float
    normal: tuple  # outward normal of the crack plane (unit)
    area: float
    moment: float
    max_stress: float
    strain_energy: float
    energy_release_rate: float
    y_fibres: np.ndarray
    traction: np.ndarray  # sigma_xx(y) on the face, Pa (tension > 0)

    def as_dict(self) -> dict:
        return {
            "face_id": int(self.face_id),
            "x": float(self.x),
            "normal": [float(v) for v in self.normal],
            "area": float(self.area),
            "moment": float(self.moment),
            "max_stress": float(self.max_stress),
            "strain_energy": float(self.strain_energy),
            "energy_release_rate": float(self.energy_release_rate),
            "y_fibres": [float(v) for v in self.y_fibres],
            "traction": [float(v) for v in self.traction],
        }


def extract_failure_face(sol: BeamSolution, n_fibres: int = 21) -> FailureFace:
    """Extract the critical section (max |sigma|) as a fracture face."""
    face_x = float(sol.x_nodes[int(np.argmax(np.abs(sol.M_elem)))])
    moment = float(sol.moment_at(face_x)[0])
    h = sol.section.h
    y = np.linspace(-h / 2.0, h / 2.0, n_fibres)
    traction = np.full_like(y, moment) * y / sol.section.I
    return FailureFace(
        face_id=0,
        x=face_x,
        normal=(-1.0, 0.0, 0.0),
        area=sol.section.A,
        moment=moment,
        max_stress=float(np.max(np.abs(traction))),
        strain_energy=sol.U,
        energy_release_rate=moment**2 / (2.0 * sol.section.EI * sol.section.b),
        y_fibres=y,
        traction=traction,
    )