"""Small-strain constitutive kernels: isotropic elasticity and J2 plasticity.

Voigt convention throughout: strain ``[exx, eyy, ezz, gxy, gyz, gxz]`` with
*engineering* shear strains, stress ``[sxx, syy, szz, sxy, syz, sxz]``.

J2 (von Mises) plasticity with linear isotropic hardening
``sigma_y(alpha) = sigma_y0 + H alpha``:

* implicit backward-Euler *radial return mapping* in the deviatoric plane;
* the *algorithmic (consistent) tangent* is derived analytically and checked
  against central differences of the return mapping itself
  (:func:`numerical_tangent`), which is what gives quadratic Newton
  convergence in the global equilibrium iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

__all__ = [
    "M_VEC",
    "iso_inner",
    "deviatoric_projector",
    "deviatoric_strain_map",
    "isotropic_elasticity",
    "lame_parameters",
    "PlasticState",
    "J2Material",
    "numerical_tangent",
    "uniaxial_tension_analytic",
]


M_VEC = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def iso_inner(a: np.ndarray, b: np.ndarray) -> float:
    """Tensor inner product of two *stress-like* Voigt vectors ``a : b``.

    For engineering-shear strain vectors the plain dot product is the tensor
    contraction instead; use this helper only for stress-like quantities.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b + a[3] * b[3] + a[4] * b[4] + a[5] * b[5])


def deviatoric_projector() -> np.ndarray:
    """Deviatoric projector acting on *stress-like* Voigt vectors.

    Shear components of a stress Voigt vector are the tensor components
    (``tau = sigma_xy``), so they are only de-traced, never scaled.
    """
    return np.eye(6) - np.outer(M_VEC, M_VEC) / 3.0


def deviatoric_strain_map() -> np.ndarray:
    """Map engineering-shear strain Voigt -> deviatoric stress / (2 mu).

    ``dev(eps)`` leaves shear *strains* untouched, but the stress produced by
    ``2 mu dev(eps)`` carries ``tau_xy = mu gamma_xy`` -- i.e. the shear
    diagonal of this map is 1/2, which is the standard engineering-Voigt
    elasticity layout ``C[3,3] = G``.
    """
    P = np.eye(6) - np.outer(M_VEC, M_VEC) / 3.0
    P[3:, 3:] *= 0.5
    return P


def lame_parameters(E: float, nu: float) -> tuple[float, float, float]:
    """Return ``(lambda, mu, K)`` for Young's modulus ``E`` and Poisson ratio ``nu``."""
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"Poisson ratio must satisfy -1 < nu < 0.5, got {nu}")
    if E <= 0.0:
        raise ValueError(f"Young's modulus must be positive, got {E}")
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    bulk = E / (3.0 * (1.0 - 2.0 * nu))
    return lam, mu, bulk


def isotropic_elasticity(E: float, nu: float) -> np.ndarray:
    """Isotropic elasticity matrix (6x6, engineering shear)."""
    lam, mu, bulk = lame_parameters(E, nu)
    return 2.0 * mu * deviatoric_strain_map() + bulk * np.outer(M_VEC, M_VEC)


@dataclass
class PlasticState:
    """Per-Gauss-point history: plastic strain and equivalent plastic strain."""

    eps_p: np.ndarray = field(default_factory=lambda: np.zeros(6))
    alpha: float = 0.0

    def copy(self) -> "PlasticState":
        return PlasticState(eps_p=self.eps_p.copy(), alpha=float(self.alpha))


@dataclass
class J2Material:
    """Isotropic elastic / J2 plastic material with linear isotropic hardening."""

    E: float
    nu: float
    sigma_y0: float
    hardening: float = 0.0
    name: str = "j2"

    def __post_init__(self) -> None:
        self.lam, self.mu, self.bulk = lame_parameters(self.E, self.nu)
        if self.sigma_y0 <= 0.0:
            raise ValueError("yield stress must be positive")
        if self.hardening < 0.0:
            raise ValueError("hardening slope must be non-negative")

    # ------------------------------------------------------------------
    @property
    def elastic_matrix(self) -> np.ndarray:
        return isotropic_elasticity(self.E, self.nu)

    def yield_stress(self, alpha: float) -> float:
        return self.sigma_y0 + self.hardening * alpha

    # ------------------------------------------------------------------
    def return_map(
        self, strain: np.ndarray, state: PlasticState
    ) -> tuple[np.ndarray, PlasticState, np.ndarray, bool]:
        """Radial return for a total strain increment (total-strain formulation).

        Returns ``(stress, new_state, consistent_tangent, plastic)``.
        """
        strain = np.asarray(strain, dtype=float)
        C_el = self.elastic_matrix
        P_dev = deviatoric_projector()
        G = self.mu
        K = self.bulk
        H = self.hardening

        sig_tr = C_el @ (strain - state.eps_p)
        s_tr = P_dev @ sig_tr
        q_tr = np.sqrt(1.5 * iso_inner(s_tr, s_tr))
        sigma_y_n = self.yield_stress(state.alpha)
        f_tr = q_tr - sigma_y_n

        if f_tr <= 0.0 or q_tr <= 0.0:
            return sig_tr, state.copy(), C_el, False

        dlam = f_tr / (3.0 * G + H)
        # a very large strain increment can in principle drive the trial point
        # through the origin; clip so the returned von Mises stress stays >= 0
        dlam = min(dlam, q_tr / (3.0 * G))
        theta = 1.0 - 3.0 * G * dlam / q_tr
        s_new = theta * s_tr
        q_new = q_tr - 3.0 * G * dlam
        if q_new < 0.0:  # pragma: no cover - only for absurd hardening data
            raise ValueError("return mapping produced negative von Mises stress")

        stress = K * float(M_VEC @ strain) * M_VEC + s_new
        # flow rule  d(eps_p) = dlam * (3/2) s/q  in *tensor* form; the stored
        # plastic strain uses engineering shear components, hence the factor 2
        if q_new > 1e-12 * self.sigma_y0:
            flow = dlam * 1.5 * s_new / q_new
        else:  # stress driven exactly to the origin by a huge increment
            flow = np.zeros(6)
        flow[3:] *= 2.0
        eps_p_new = state.eps_p + flow
        new_state = PlasticState(eps_p=eps_p_new, alpha=state.alpha + dlam)

        # consistent (algorithmic) tangent
        M = s_tr / q_tr
        coef = 9.0 * G * G * (1.0 / (3.0 * G + H) - dlam / q_tr)
        C_ep = (
            K * np.outer(M_VEC, M_VEC)
            + 2.0 * G * theta * deviatoric_strain_map()
            - coef * np.outer(M, M)
        )
        return stress, new_state, 0.5 * (C_ep + C_ep.T), True

    # ------------------------------------------------------------------
    def uniaxial_stress_curve(
        self, strain_xx: np.ndarray, d_strain: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reference uniaxial::stress-strain curve (analytic, linear hardening)."""
        strain_xx = np.asarray(strain_xx, dtype=float)
        return strain_xx, uniaxial_tension_analytic(
            strain_xx, self.E, self.sigma_y0, self.hardening
        )


def uniaxial_tension_analytic(
    strain: np.ndarray, E: float, sigma_y0: float, hardening: float
) -> np.ndarray:
    """Analytic uniaxial *stress* response of a J2 solid with linear hardening.

    ``eps = sigma/E + (sigma - sigma_y0)/H`` for ``sigma > sigma_y0`` gives
    ``sigma = (eps + sigma_y0/H) / (1/E + 1/H)``.
    """
    eps = np.asarray(strain, dtype=float)
    eps_y = sigma_y0 / E
    out = E * eps
    if hardening > 0.0:
        plastic = eps > eps_y
        out = np.where(
            plastic,
            (eps + sigma_y0 / hardening) / (1.0 / E + 1.0 / hardening),
            out,
        )
    else:
        out = np.minimum(out, sigma_y0)
    return out


def numerical_tangent(
    material: J2Material,
    strain: np.ndarray,
    state: PlasticState,
    h: float = 1e-7,
) -> np.ndarray:
    """Central-difference tangent of the return mapping (verification aid)."""
    strain = np.asarray(strain, dtype=float)
    C = np.zeros((6, 6))
    for j in range(6):
        ds = np.zeros(6)
        ds[j] = h * max(1.0, abs(strain[j]))
        sp = material.return_map(strain + ds, state)[0]
        sm = material.return_map(strain - ds, state)[0]
        C[:, j] = (sp - sm) / (2.0 * ds[j])
    return C


def with_hardening(material: J2Material, hardening: float) -> J2Material:
    """Return a copy of ``material`` with a different hardening slope."""
    return replace(material, hardening=float(hardening))