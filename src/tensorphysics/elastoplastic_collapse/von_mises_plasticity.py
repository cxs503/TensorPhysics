"""Von Mises elastoplastic constitutive law with isotropic hardening.

Implements the classical J2 flow theory with linear isotropic hardening:

    σ_y(ε_p) = σ_y0 + H ε_p

where:
- σ_y0 is the initial yield stress
- H is the plastic (hardening) modulus
- ε_p is the accumulated equivalent plastic strain

The return mapping algorithm solves the consistency condition:
    Φ = σ_eq - σ_y(ε_p) = 0

and returns the consistent (algorithmic) tangent stiffness tensor C_ep
for use in Newton-Raphson iterations.

Conventions
-----------
- 3D stress/strain in Voigt notation: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]
- Plane stress reduction available for thin plates/shells
- Small strain assumption (additive decomposition ε = ε_e + ε_p)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

__all__ = [
    "VonMisesMaterial",
    "yield_function",
    "return_mapping",
    "elastoplastic_tangent",
]


@dataclass(frozen=True)
class VonMisesMaterial:
    """J2 elastoplastic material with linear isotropic hardening.

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio [-].
    sigma_y0 : float
        Initial yield stress [Pa].
    H : float
        Plastic (hardening) modulus [Pa]. H=0 gives elastic-perfectly-plastic.
    """

    E: float
    nu: float
    sigma_y0: float
    H: float = 0.0

    def __post_init__(self) -> None:
        if self.E <= 0.0:
            raise ValueError("Young's modulus must be positive")
        if not (0.0 <= self.nu < 0.5):
            raise ValueError("Poisson's ratio must be in [0, 0.5)")
        if self.sigma_y0 <= 0.0:
            raise ValueError("Yield stress must be positive")
        if self.H < 0.0:
            raise ValueError("Hardening modulus must be non-negative")

    @property
    def shear_modulus(self) -> float:
        """Shear modulus G = E / (2(1+ν)) [Pa]."""
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus K = E / (3(1-2ν)) [Pa]."""
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    @property
    def elastic_matrix_3d(self) -> np.ndarray:
        """3D isotropic elastic stiffness tensor (6x6 Voigt) [Pa]."""
        lam = self.nu * self.E / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        mu = self.shear_modulus
        C = np.array([
            [lam + 2*mu, lam,         lam,         0,   0,   0  ],
            [lam,         lam + 2*mu, lam,         0,   0,   0  ],
            [lam,         lam,         lam + 2*mu, 0,   0,   0  ],
            [0,           0,           0,           mu,  0,   0  ],
            [0,           0,           0,           0,   mu, 0  ],
            [0,           0,           0,           0,   0,  mu],
        ])
        return C

    @property
    def elastic_matrix_plane_stress(self) -> np.ndarray:
        """Plane stress elastic stiffness (3x3 Voigt: xx, yy, xy) [Pa]."""
        c = self.E / (1.0 - self.nu**2)
        return c * np.array([
            [1.0, self.nu, 0.0],
            [self.nu, 1.0, 0.0],
            [0.0, 0.0, 0.5*(1.0 - self.nu)],
        ])


def yield_function(sigma: np.ndarray, sigma_y: float) -> np.ndarray:
    """Von Mises yield function Φ = σ_eq - σ_y.

    Parameters
    ----------
    sigma : ndarray, shape (..., 3) or (..., 6)
        Stress tensor in Voigt notation (plane stress or 3D).
    sigma_y : float or ndarray
        Current yield stress [Pa].

    Returns
    -------
    Phi : ndarray
        Yield function value. Φ > 0 means plastic loading.
    """
    sigma = np.asarray(sigma)
    if sigma.shape[-1] == 3:
        # Plane stress: [σ_xx, σ_yy, σ_xy]
        sxx, syy, sxy = sigma[..., 0], sigma[..., 1], sigma[..., 2]
        # σ_eq = sqrt(σ_xx² - σ_xx σ_yy + σ_yy² + 3 σ_xy²)
        sigma_eq = np.sqrt(np.maximum(
            sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2, 0.0))
    elif sigma.shape[-1] == 6:
        # 3D: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]
        sxx = sigma[..., 0]
        syy = sigma[..., 1]
        szz = sigma[..., 2]
        syz = sigma[..., 3]
        sxz = sigma[..., 4]
        sxy = sigma[..., 5]
        sigma_eq = np.sqrt(np.maximum(
            0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2
                   + 6.0*(syz**2 + sxz**2 + sxy**2)), 0.0))
    else:
        raise ValueError(f"Expected stress shape (..., 3) or (..., 6), got {sigma.shape}")
    return sigma_eq - sigma_y


def return_mapping(
    strain: np.ndarray,
    eps_p_old: np.ndarray,
    alpha_old: np.ndarray,
    material: VonMisesMaterial,
    plane_stress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Radial return mapping for J2 plasticity with isotropic hardening.

    Solves the consistency condition using bisection on the plastic multiplier.

    Parameters
    ----------
    strain : ndarray, shape (..., 3) or (..., 6)
        Total strain tensor (Voigt).
    eps_p_old : ndarray, same shape as strain
        Plastic strain from previous step.
    alpha_old : ndarray, shape (...)
        Accumulated equivalent plastic strain from previous step.
    material : VonMisesMaterial
        Material parameters.
    plane_stress : bool
        If True, use plane stress elastic matrix (3-component Voigt).

    Returns
    -------
    sigma : ndarray, same shape as strain
        Updated stress tensor.
    C_ep : ndarray, shape (..., n, n) where n = 3 or 6
        Consistent tangent stiffness.
    eps_p_new : ndarray, same shape as strain
        Updated plastic strain.
    alpha_new : ndarray, shape (...)
        Updated accumulated plastic strain.
    """
    strain = np.asarray(strain, dtype=float)
    eps_p_old = np.asarray(eps_p_old, dtype=float)
    alpha_old = np.asarray(alpha_old, dtype=float)

    if plane_stress:
        C_e = material.elastic_matrix_plane_stress
        n_comp = 3
    else:
        C_e = material.elastic_matrix_3d
        n_comp = 6

    # Elastic trial stress
    sigma_tr = np.einsum("ij,...j->...i", C_e, strain - eps_p_old)

    # Current yield stress
    sigma_y_current = material.sigma_y0 + material.H * alpha_old

    # Yield function at trial state
    Phi_tr = yield_function(sigma_tr, sigma_y_current)

    # Initialize outputs
    sigma_new = sigma_tr.copy()
    eps_p_new = eps_p_old.copy()
    alpha_new = alpha_old.copy()
    C_ep = np.broadcast_to(C_e, sigma_tr.shape + (n_comp,)).copy()

    # Identify plastic points
    plastic = Phi_tr > 1e-12 * material.sigma_y0
    if not np.any(plastic):
        return sigma_new, C_ep, eps_p_new, alpha_new

    # Compute deviatoric trial stress and flow direction
    if plane_stress:
        # Plane stress deviator: s = σ - (1/3) tr(σ) I
        # For plane stress σ_zz = 0, so tr(σ) = σ_xx + σ_yy
        tr_sigma = sigma_tr[..., 0] + sigma_tr[..., 1]
        s_tr = sigma_tr.copy()
        s_tr[..., 0] -= tr_sigma / 3.0
        s_tr[..., 1] -= tr_sigma / 3.0
        # σ_zz component (implicit) = -tr_sigma/3
        s_zz = -tr_sigma / 3.0

        # Equivalent stress from deviator
        sigma_eq_tr = np.sqrt(np.maximum(
            1.5 * (s_tr[..., 0]**2 + s_tr[..., 1]**2 + s_zz**2
                   + 2.0 * s_tr[..., 2]**2), 1e-30))

        # Flow direction n = (3/2) s / σ_eq
        n_flow = 1.5 * s_tr / sigma_eq_tr[..., None]
        # Include σ_zz contribution in flow direction for plane stress
        # (not stored but needed for tangent)

    else:
        # 3D deviator
        tr_sigma = sigma_tr[..., 0] + sigma_tr[..., 1] + sigma_tr[..., 2]
        s_tr = sigma_tr.copy()
        s_tr[..., 0] -= tr_sigma / 3.0
        s_tr[..., 1] -= tr_sigma / 3.0
        s_tr[..., 2] -= tr_sigma / 3.0

        sigma_eq_tr = np.sqrt(np.maximum(
            1.5 * (s_tr[..., 0]**2 + s_tr[..., 1]**2 + s_tr[..., 2]**2
                   + 2.0*(s_tr[..., 3]**2 + s_tr[..., 4]**2 + s_tr[..., 5]**2)),
            1e-30))

        n_flow = 1.5 * s_tr / sigma_eq_tr[..., None]

    # Bisection on plastic multiplier Δγ
    # Consistency: σ_eq(σ_tr - Δγ C n) = σ_y0 + H (α + Δγ)
    # For plane stress, this is not closed-form, so we use bisection

    # Upper bound: Δγ_max such that σ_eq = 0
    # σ_eq ≈ σ_eq_tr - 3G Δγ for von Mises, so Δγ_max ≈ σ_eq_tr / (3G)
    G = material.shear_modulus
    if plane_stress:
        # More conservative upper bound for plane stress
        dg_max = sigma_eq_tr / (2.0 * G * (1.0 - material.nu))
    else:
        dg_max = sigma_eq_tr / (3.0 * G)

    dg_lo = np.zeros_like(alpha_old)
    dg_hi = np.where(plastic, np.maximum(dg_max, 1e-20), 0.0)

    # Bisection iterations
    for _ in range(80):
        dg_mid = 0.5 * (dg_lo + dg_hi)

        # Trial stress at dg_mid
        Cn = np.einsum("ij,...j->...i", C_e, n_flow)
        sigma_mid = sigma_tr - dg_mid[..., None] * Cn

        # Check yield function
        sigma_y_mid = material.sigma_y0 + material.H * (alpha_old + dg_mid)
        Phi_mid = yield_function(sigma_mid, sigma_y_mid)

        # Update bounds
        too_high = plastic & (Phi_mid > 0)
        dg_lo = np.where(too_high, dg_mid, dg_lo)
        dg_hi = np.where(plastic & ~too_high, dg_mid, dg_hi)

    # Final plastic multiplier
    dg = 0.5 * (dg_lo + dg_hi)
    dg = np.where(plastic, dg, 0.0)

    # Update stress and internal variables
    Cn = np.einsum("ij,...j->...i", C_e, n_flow)
    sigma_new = sigma_tr - dg[..., None] * Cn
    eps_p_new = eps_p_old + dg[..., None] * n_flow
    alpha_new = alpha_old + dg

    # Consistent tangent stiffness
    # C_ep = C_e - (C_e n) ⊗ (C_e n) / (H + n^T C_e n)
    Cn = np.einsum("ij,...j->...i", C_e, n_flow)
    nCn = np.einsum("...i,...i->...", n_flow, Cn)
    denom = material.H + nCn
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)

    for i in range(n_comp):
        for j in range(n_comp):
            C_ep[..., i, j] = np.where(
                plastic,
                C_e[i, j] - Cn[..., i] * Cn[..., j] / denom,
                C_e[i, j]
            )

    return sigma_new, C_ep, eps_p_new, alpha_new


def elastoplastic_tangent(
    material: VonMisesMaterial,
    sigma: np.ndarray,
    alpha: float,
    plastic: bool,
) -> np.ndarray:
    """Compute the consistent tangent stiffness at a given stress state.

    This is a convenience function for single-point evaluation.

    Parameters
    ----------
    material : VonMisesMaterial
        Material parameters.
    sigma : ndarray, shape (3,) or (6,)
        Current stress state (Voigt).
    alpha : float
        Accumulated plastic strain.
    plastic : bool
        Whether the point is currently yielding.

    Returns
    -------
    C_ep : ndarray, shape (3, 3) or (6, 6)
        Consistent tangent stiffness.
    """
    sigma = np.asarray(sigma)
    n_comp = sigma.shape[0]

    if not plastic:
        if n_comp == 3:
            return material.elastic_matrix_plane_stress.copy()
        else:
            return material.elastic_matrix_3d.copy()

    # Reshape for return_mapping
    strain_dummy = np.zeros_like(sigma)
    eps_p_dummy = np.zeros_like(sigma)
    alpha_dummy = np.array([alpha])

    # We need to find the strain that gives this stress state
    # This is an inverse problem; for tangent evaluation we use the flow direction
    if n_comp == 3:
        C_e = material.elastic_matrix_plane_stress
        tr_sigma = sigma[0] + sigma[1]
        s = sigma.copy()
        s[0] -= tr_sigma / 3.0
        s[1] -= tr_sigma / 3.0
        s_zz = -tr_sigma / 3.0
        sigma_eq = np.sqrt(max(
            1.5 * (s[0]**2 + s[1]**2 + s_zz**2 + 2.0*s[2]**2), 1e-30))
        n_flow = 1.5 * s / sigma_eq
    else:
        C_e = material.elastic_matrix_3d
        tr_sigma = sigma[0] + sigma[1] + sigma[2]
        s = sigma.copy()
        s[0] -= tr_sigma / 3.0
        s[1] -= tr_sigma / 3.0
        s[2] -= tr_sigma / 3.0
        sigma_eq = np.sqrt(max(
            1.5 * (s[0]**2 + s[1]**2 + s[2]**2
                   + 2.0*(s[3]**2 + s[4]**2 + s[5]**2)), 1e-30))
        n_flow = 1.5 * s / sigma_eq

    Cn = C_e @ n_flow
    nCn = n_flow @ Cn
    denom = material.H + nCn

    C_ep = C_e - np.outer(Cn, Cn) / denom
    return C_ep
