"""Von Mises elastoplastic constitutive law with isotropic hardening.

Implements the classical J2 flow theory with linear isotropic hardening:

    σ_y(ε̄_p) = σ_y0 + H ε̄_p

where:
- σ_y0 is the initial yield stress
- H is the plastic (hardening) modulus
- ε̄_p is the accumulated equivalent plastic strain

Return mapping
--------------
The consistency condition

    Φ = σ_eq - σ_y(ε̄_p) ≤ 0

is solved by the standard **radial return** (closest-point projection).  For
linear isotropic hardening and an elastic trial state the projection is closed
form:

    Δγ = Φ_tr / (3G + H),      σ = σ_tr - Δγ C_e n

with n = (3/2) s/σ_eq the deviatoric flow normal, because the trial stress is
purely elastic and therefore maps radially onto the shifted yield surface.

Plane stress
------------
For plane stress the return cannot be radial inside the (σ_xx, σ_yy, τ_xy)
subspace: the out-of-plane strain ε_zz is not known a priori.  It is obtained
from the local scalar condition σ_zz(ε_zz) = 0, solved by a safeguarded Newton
iteration wrapping the exact 3D return mapping.  The algorithmic (consistent)
tangent for plane stress is the static condensation of the 3D consistent
tangent onto the in-plane strain subspace, exact for the constrained problem by
the implicit function theorem:

    C_ps = C_aa - C_ab C_bb^{-1} C_ba,   a = (xx, yy, xy), b = (zz, yz, xz)

History variables
-----------------
Only the accumulated equivalent plastic strain ε̄_p is required for the stress
update (isotropic hardening).  The plastic strain vector is recomputed from the
converged state and returned for post-processing.

Voigt conventions
-----------------
- Ordering: [xx, yy, zz, yz, xz, xy] (3D) and [xx, yy, xy] (2D).
- Strain-like quantities store **engineering** shear (γ_ij = 2 ε_ij); the
  elastic matrices are written for that convention.
- Plastic strain increments are returned in the same engineering convention.
- Small strains, additive decomposition ε = ε_e + ε_p.

Verified anchors (see ``tests/test_elastoplastic_verification_p167w2_n03.py``)
-----------------------------------------------------------------------------
- 3D and plane-stress uniaxial tension reproduce the closed-form solution
  σ = σ_y0 + H ε̄_p, ε = σ/E + ε̄_p.
- The consistent tangent matches a central finite difference of the return
  mapping (both 3D and plane stress).
- Plastic flow is isochoric: tr(ε_p) = 0.

References
----------
- Simo, J.C. & Hughes, T.J.R. (1998). Computational Inelasticity. Springer.
- Simo, J.C. & Taylor, R.L. (1986). "A return mapping algorithm for plane
  stress elastoplasticity." Int. J. Numer. Methods Eng., 22, 649-670.
- de Souza Neto, E.A., Perić, D. & Owen, D.R.J. (2008). Computational Methods
  for Plasticity. Wiley.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "VonMisesMaterial",
    "yield_function",
    "equivalent_stress",
    "return_mapping",
    "return_mapping_3d",
    "plane_stress_tangent",
    "elastoplastic_tangent",
]

_TOL_YIELD = 1.0e-12


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
        return np.array([
            [lam + 2*mu, lam,         lam,         0,   0,   0  ],
            [lam,         lam + 2*mu, lam,         0,   0,   0  ],
            [lam,         lam,         lam + 2*mu, 0,   0,   0  ],
            [0,           0,           0,           mu,  0,   0  ],
            [0,           0,           0,           0,   mu, 0  ],
            [0,           0,           0,           0,   0,  mu],
        ])

    @property
    def elastic_matrix_plane_stress(self) -> np.ndarray:
        """Plane stress elastic stiffness (3x3 Voigt: xx, yy, xy) [Pa]."""
        c = self.E / (1.0 - self.nu**2)
        return c * np.array([
            [1.0, self.nu, 0.0],
            [self.nu, 1.0, 0.0],
            [0.0, 0.0, 0.5*(1.0 - self.nu)],
        ])

    def yield_stress(self, alpha: float) -> float:
        """Current yield stress σ_y = σ_y0 + H ε̄_p [Pa]."""
        return self.sigma_y0 + self.H * float(alpha)


# ---------------------------------------------------------------------------
# invariants and flow normal
# ---------------------------------------------------------------------------
def _deviator_and_eq(sigma: np.ndarray, plane_stress: bool):
    """Deviatoric stress (Voigt, full 6 slots) and von Mises equivalent stress.

    ``sigma`` is a single Voigt vector (3 or 6 components).
    """
    sig = np.asarray(sigma, dtype=float).ravel()
    if plane_stress:
        tr = sig[0] + sig[1]
        s = np.array([sig[0] - tr / 3.0,
                      sig[1] - tr / 3.0,
                      -tr / 3.0,
                      0.0, 0.0,
                      sig[2]])
        factor = 1.5 * (s[0]**2 + s[1]**2 + s[2]**2 + 2.0 * s[5]**2)
    else:
        tr = sig[0] + sig[1] + sig[2]
        s = np.array([sig[0] - tr / 3.0,
                      sig[1] - tr / 3.0,
                      sig[2] - tr / 3.0,
                      sig[3], sig[4], sig[5]])
        factor = 1.5 * (s[0]**2 + s[1]**2 + s[2]**2
                        + 2.0 * (s[3]**2 + s[4]**2 + s[5]**2))
    return s, float(np.sqrt(max(factor, 1e-30)))


def _flow_normal(s: np.ndarray, seq: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (n_tensor, n_engineering) flow normals for a deviator s.

    n_tensor = (3/2) s / σ_eq is the tensor-valued normal; n_engineering has the
    shear entries doubled so that ``Δε_p = Δγ n_engineering`` is written in the
    engineering-shear Voigt convention used for strain vectors.
    """
    n_tensor = 1.5 * s / seq
    n_eng = n_tensor.copy()
    n_eng[3:] *= 2.0
    return n_tensor, n_eng


def _consistent_tangent_3d(
    material: VonMisesMaterial,
    n_tensor: np.ndarray,
    seq_tr: float,
    dgamma: float,
) -> np.ndarray:
    """Algorithmic (consistent) tangent of the radial return, 3D.

    Differentiation of  σ = σ_tr − 2G Δγ n(σ_tr)  with the engineering-Voigt
    convention used for strain vectors gives

        dσ/dε = C_e − 2G Δγ [ (3/(2σ_eq^tr)) P C_e − n ⊗ (dσ_eq^tr/dε)/σ_eq^tr ]
                     − (2G n) ⊗ (dσ_eq^tr/dε) / (3G + H)

    with P the deviatoric projector acting on stress Voigt vectors and
    dσ_eq^tr/dε = C_e n_eng (n_eng = n with doubled shear entries).  This is the
    *algorithmic* tangent: it differs from the continuum elastoplastic modulus
    C_e − (C_e n_eng)⊗(C_e n_eng)/(3G+H) by the term proportional to Δγ that
    comes from the rotation of the flow normal during the increment.
    """
    G = material.shear_modulus
    C_e = material.elastic_matrix_3d
    P = np.eye(6)
    P[:3, :3] -= 1.0 / 3.0
    n_eng = n_tensor.copy()
    n_eng[3:] *= 2.0
    g = C_e @ n_eng                       # dσ_eq^tr/dε  (6,)
    seq = max(seq_tr, 1e-30)
    dn_de = 1.5 / seq * (P @ C_e) - np.outer(n_tensor, g) / seq
    C_ep = (C_e
            - 2.0 * G * dgamma * dn_de
            - np.outer(2.0 * G * n_tensor, g) / (3.0 * G + material.H))
    return 0.5 * (C_ep + C_ep.T)


def equivalent_stress(sigma: np.ndarray, plane_stress: Optional[bool] = None) -> float:
    """von Mises equivalent stress of a single Voigt stress vector [Pa]."""
    arr = np.asarray(sigma, dtype=float).ravel()
    if plane_stress is None:
        plane_stress = arr.size == 3
    _, seq = _deviator_and_eq(arr, bool(plane_stress))
    return seq


def yield_function(sigma: np.ndarray, sigma_y: float) -> np.ndarray:
    """Von Mises yield function Φ = σ_eq - σ_y.

    Parameters
    ----------
    sigma : ndarray, shape (..., 3) or (..., 6)
        Stress tensor in Voigt notation (plane stress or 3D).
    sigma_y : float
        Current yield stress [Pa].

    Returns
    -------
    Phi : float or ndarray
        Yield function value. Φ > 0 means plastic loading.
    """
    arr = np.asarray(sigma, dtype=float)
    if arr.ndim == 1:
        return equivalent_stress(arr) - sigma_y
    flat = arr.reshape(-1, arr.shape[-1])
    out = np.array([equivalent_stress(row) - sigma_y for row in flat])
    return out.reshape(arr.shape[:-1])


# ---------------------------------------------------------------------------
# 3D return mapping (closed form)
# ---------------------------------------------------------------------------
def return_mapping_3d(
    strain: np.ndarray,
    alpha_old: float,
    material: VonMisesMaterial,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Exact radial return for a 3D strain state.

    Parameters
    ----------
    strain : ndarray, shape (6,)
        Total strain in Voigt engineering notation.
    alpha_old : float
        Accumulated equivalent plastic strain at the start of the increment.
    material : VonMisesMaterial
        Material parameters.

    Returns
    -------
    sigma : ndarray, shape (6,)
        Updated stress (Voigt, tensor shear stress components).
    C_ep : ndarray, shape (6, 6)
        Consistent (algorithmic) tangent stiffness.
    eps_p : ndarray, shape (6,)
        Total plastic strain (engineering shear convention).
    alpha : float
        Updated accumulated equivalent plastic strain.
    """
    strain = np.asarray(strain, dtype=float).ravel()
    if strain.size != 6:
        raise ValueError("return_mapping_3d expects a 6-component strain")
    C_e = material.elastic_matrix_3d
    sigma_tr = C_e @ strain
    s_tr, seq_tr = _deviator_and_eq(sigma_tr, plane_stress=False)
    n_tensor, n_eng = _flow_normal(s_tr, seq_tr)

    sigma_y_old = material.yield_stress(alpha_old)
    phi_tr = seq_tr - sigma_y_old

    if phi_tr <= _TOL_YIELD * material.sigma_y0:
        eps_e = strain
        eps_p = np.zeros(6)
        return sigma_tr, C_e.copy(), eps_p, float(alpha_old)

    G = material.shear_modulus
    dgamma = phi_tr / (3.0 * G + material.H)
    Cn = C_e @ n_eng                      # = 2G * n_tensor in Voigt slots
    sigma = sigma_tr - dgamma * Cn
    eps_p = dgamma * n_eng
    alpha = float(alpha_old + dgamma)

    C_ep = _consistent_tangent_3d(material, n_tensor, seq_tr, dgamma)
    return sigma, C_ep, eps_p, alpha


# ---------------------------------------------------------------------------
# plane stress return mapping (ε_zz condensation)
# ---------------------------------------------------------------------------
def _eps_zz_for_plane_stress(
    strain_inplane: np.ndarray,
    alpha_old: float,
    material: VonMisesMaterial,
    max_iter: int = 60,
    tol: float = 1.0e-10,
) -> Tuple[np.ndarray, float, float, int]:
    """Find ε_zz such that σ_zz = 0 (safeguarded Newton on the exact 3D return).

    Returns (sigma6, alpha, eps_zz, iterations).
    """
    ex, ey, gxy = strain_inplane
    nu = material.nu
    # elastic initial guess (plane stress elasticity: ε_zz = -ν(ε_xx+ε_yy)/(1-ν))
    ezz = -nu / (1.0 - nu) * (ex + ey)
    target = tol * material.sigma_y0

    def state(ezz_val):
        st = np.array([ex, ey, ezz_val, 0.0, 0.0, gxy])
        return return_mapping_3d(st, alpha_old, material)

    sigma, C_ep, _, alpha = state(ezz)
    if abs(sigma[2]) <= target:
        return sigma, alpha, ezz, 0

    for it in range(1, max_iter + 1):
        res = sigma[2]
        slope = C_ep[2, 2]
        if abs(slope) < 1e-12 * material.E:
            break
        step = -res / slope
        # damp the step to keep the local iteration stable near yield onset
        max_step = 1.0e-3
        if abs(step) > max_step:
            step = np.sign(step) * max_step
        ezz_new = ezz + step
        sigma_new, C_new, _, alpha_new = state(ezz_new)
        if abs(sigma_new[2]) < abs(res):
            ezz, sigma, C_ep, alpha = ezz_new, sigma_new, C_new, alpha_new
        else:  # bisection fallback between the two bracketing strains
            lo, hi = sorted((ezz, ezz_new))
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                s_m, C_m, _, a_m = state(mid)
                if abs(s_m[2]) <= target:
                    ezz, sigma, C_ep, alpha = mid, s_m, C_m, a_m
                    break
                s_lo, _, _, _ = state(lo)
                if s_lo[2] * s_m[2] > 0:
                    lo = mid
                else:
                    hi = mid
            else:
                mid = 0.5 * (lo + hi)
                sigma, C_ep, _, alpha = state(mid)
                ezz = mid
        if abs(sigma[2]) <= target:
            return sigma, alpha, ezz, it
    return sigma, alpha, ezz, max_iter


def plane_stress_tangent(C_ep6: np.ndarray) -> np.ndarray:
    """Static condensation of a 3D consistent tangent onto plane stress."""
    idx_a = [0, 1, 5]
    idx_b = [2, 3, 4]
    C_aa = C_ep6[np.ix_(idx_a, idx_a)]
    C_ab = C_ep6[np.ix_(idx_a, idx_b)]
    C_ba = C_ep6[np.ix_(idx_b, idx_a)]
    C_bb = C_ep6[np.ix_(idx_b, idx_b)]
    C_ps = C_aa - C_ab @ np.linalg.solve(C_bb, C_ba)
    return 0.5 * (C_ps + C_ps.T)


def _plane_stress_state(strain: np.ndarray, alpha_old: float,
                        material: VonMisesMaterial):
    """Converged plane-stress state: (sigma3, C_ep3, eps_p3, alpha)."""
    sigma6, alpha, ezz, _it = _eps_zz_for_plane_stress(strain, alpha_old, material)
    C_e = material.elastic_matrix_3d
    strain6 = np.array([strain[0], strain[1], ezz, 0.0, 0.0, strain[2]])
    sigma_tr = C_e @ strain6
    s_tr, seq_tr = _deviator_and_eq(sigma_tr, plane_stress=False)
    n_tensor, _n_eng = _flow_normal(s_tr, seq_tr)
    phi_tr = seq_tr - material.yield_stress(alpha_old)
    if phi_tr > 1.0e-12 * material.sigma_y0:
        dgamma = phi_tr / (3.0 * material.shear_modulus + material.H)
        C_ep6 = _consistent_tangent_3d(material, n_tensor, seq_tr, dgamma)
        C_ps = plane_stress_tangent(C_ep6)
    else:
        C_ps = material.elastic_matrix_plane_stress.copy()
    eps_e6 = np.linalg.solve(C_e, sigma6)
    eps_p6 = np.array([strain[0], strain[1], ezz, 0.0, 0.0, strain[2]]) - eps_e6
    sigma3 = np.array([sigma6[0], sigma6[1], sigma6[5]])
    eps_p3 = np.array([eps_p6[0], eps_p6[1], eps_p6[5]])
    return sigma3, C_ps, eps_p3, alpha


def _return_mapping_point(
    strain: np.ndarray,
    alpha_old: float,
    material: VonMisesMaterial,
    plane_stress: bool,
):
    """Single-point driver returning (sigma, C_ep, eps_p, alpha)."""
    strain = np.asarray(strain, dtype=float).ravel()
    if plane_stress:
        if strain.size != 3:
            raise ValueError("plane stress expects a 3-component strain")
        return _plane_stress_state(strain, alpha_old, material)
    sigma6, C_ep6, eps_p6, alpha = return_mapping_3d(strain, alpha_old, material)
    return sigma6, C_ep6, eps_p6, alpha


def return_mapping(
    strain: np.ndarray,
    eps_p_old: Optional[np.ndarray],
    alpha_old: np.ndarray,
    material: VonMisesMaterial,
    plane_stress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """J2 return mapping for a (batched) strain state.

    Backward-compatible API of the ``elastoplastic_collapse`` package: the
    stress update depends only on the total strain and on the accumulated
    equivalent plastic strain α (isotropic hardening); ``eps_p_old`` is accepted
    for signature stability and is *not* used to recompute the stress.

    Parameters
    ----------
    strain : ndarray, shape (n,) or (..., n)
        Total strain (Voigt engineering shear), n = 3 (plane stress) or 6 (3D).
    eps_p_old : ndarray or None
        Previous plastic strain (kept for API compatibility).
    alpha_old : float or ndarray
        Previous accumulated equivalent plastic strain.
    material : VonMisesMaterial
        Material parameters.
    plane_stress : bool
        Plane stress (3 components) or 3D (6 components).

    Returns
    -------
    sigma, C_ep, eps_p_new, alpha_new
        Shapes: stress (..., n), tangent (n, n), plastic strain (..., n),
        accumulated plastic strain (...,).
    """
    arr = np.asarray(strain, dtype=float)
    n_comp = 3 if plane_stress else 6
    if arr.shape[-1] != n_comp:
        raise ValueError(
            f"expected {n_comp} strain components, got {arr.shape[-1]}")

    single = arr.ndim == 1
    batch = arr.reshape(-1, n_comp)
    alpha_arr = np.atleast_1d(np.asarray(alpha_old, dtype=float)).ravel()
    if alpha_arr.size == 1:
        alpha_batch = np.full(batch.shape[0], float(alpha_arr[0]))
    else:
        alpha_batch = alpha_arr.reshape(batch.shape[0])

    sig_out = np.empty_like(batch)
    ep_out = np.empty_like(batch)
    alpha_out = np.empty(batch.shape[0])
    C_out = np.empty((n_comp, n_comp))

    for i in range(batch.shape[0]):
        sig, C_ep, eps_p, alpha = _return_mapping_point(
            batch[i], alpha_batch[i], material, plane_stress)
        sig_out[i] = sig
        ep_out[i] = eps_p
        alpha_out[i] = alpha
        if i == 0:
            C_out = C_ep

    if single:
        return (sig_out[0], C_out, ep_out[0], np.array([alpha_out[0]]))
    return (sig_out, C_out, ep_out, alpha_out)


# ---------------------------------------------------------------------------
# single-state tangent helper
# ---------------------------------------------------------------------------
def elastoplastic_tangent(
    material: VonMisesMaterial,
    sigma: np.ndarray,
    alpha: float,
    plastic: bool,
) -> np.ndarray:
    """Consistent tangent stiffness at a given stress state.

    Parameters
    ----------
    material : VonMisesMaterial
        Material parameters.
    sigma : ndarray, shape (3,) or (6,)
        Current stress state (Voigt).
    alpha : float
        Accumulated equivalent plastic strain.
    plastic : bool
        Whether the point is currently yielding.

    Returns
    -------
    C_ep : ndarray, shape (3, 3) or (6, 6)
        Consistent tangent stiffness [Pa].
    """
    sigma = np.asarray(sigma, dtype=float).ravel()
    plane_stress = sigma.size == 3
    n_comp = 3 if plane_stress else 6
    if not plastic:
        return (material.elastic_matrix_plane_stress.copy() if plane_stress
                else material.elastic_matrix_3d.copy())

    if plane_stress:
        sigma6 = np.array([sigma[0], sigma[1], 0.0, 0.0, 0.0, sigma[2]])
    else:
        sigma6 = sigma
    s, seq = _deviator_and_eq(sigma6, plane_stress=plane_stress)
    n_tensor, n_eng = _flow_normal(s, seq)
    C_ep6 = _consistent_tangent_3d(material, n_tensor, seq, 0.0)
    if plane_stress:
        return plane_stress_tangent(C_ep6)
    return C_ep6