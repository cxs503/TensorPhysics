"""J2 (von Mises) elastoplastic kernels for the ice-indentation node.

The finite-element part of this package does *not* re-implement constitutive
physics: it reuses :class:`tensorphysics.solid_fem.material.J2Material`
(implicit radial return mapping + analytic consistent tangent).  What this
module adds on top of that single-Gauss-point kernel is

* a **vectorised batch twin** of :meth:`J2Material.return_map`
  (:func:`batch_return_map`) so that a whole mesh (~10^4 Gauss points) is
  integrated in one numpy call per Newton iteration.  The batch twin is
  verified against the scalar kernel to machine precision by
  ``tests/test_ice_indentation.py::test_batch_return_map_matches_scalar``;
* the **material-point verification drivers** that the acceptance criteria
  need:

  - :func:`uniaxial_stress_response` -- uniaxial *stress* path
    (:math:`\\sigma_{yy}=\\sigma_{zz}=0`) integrated with the very same
    return mapping that the FE solver uses, to be compared with the closed
    form of J2 + linear isotropic hardening (AC1);
  - :func:`shear_yield_stress` -- first yield of the pure-shear path
    :math:`\\gamma_{xy}`, to be compared with :math:`\\sigma_y/\\sqrt3`
    (AC2);
  - :func:`tangent_error` -- analytic consistent tangent vs a central
    difference of the return mapping.

Voigt convention (identical to :mod:`tensorphysics.solid_fem.material`):
strain ``[exx, eyy, ezz, gxy, gyz, gxz]`` with *engineering* shear strains,
stress ``[sxx, syy, szz, sxy, syz, sxz]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..solid_fem.material import (
    J2Material,
    PlasticState,
    deviatoric_projector,
    deviatoric_strain_map,
    iso_inner,
    numerical_tangent,
    uniaxial_tension_analytic,
)

__all__ = [
    "batch_return_map",
    "BatchResponse",
    "uniaxial_stress_response",
    "UniaxialResponse",
    "shear_yield_stress",
    "tangent_error",
    "hydrostatic_strain_map",
]

_H_MAT = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _iso_inner_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched tensor contraction ``a : b`` of stress-like Voigt rows.

    Identical to :func:`tensorphysics.solid_fem.material.iso_inner`: the plain
    row dot product already counts the shear products once, so the shear part
    must be counted a second time to obtain the full tensor contraction
    ``s_ij s_ij``.
    """
    return (a * b).sum(axis=1) + (a[:, 3:] * b[:, 3:]).sum(axis=1)


# --------------------------------------------------------------------------- #
# Batched radial return mapping (same algorithm as J2Material.return_map)
# --------------------------------------------------------------------------- #
@dataclass
class BatchResponse:
    """Result of :func:`batch_return_map` for ``n`` Gauss points."""

    stress: np.ndarray        # (n, 6)
    eps_p: np.ndarray         # (n, 6) engineering-shear plastic strain
    alpha: np.ndarray         # (n,)   equivalent plastic strain
    tangent: np.ndarray       # (n, 6, 6) consistent tangent
    plastic: np.ndarray       # (n,) bool

    @property
    def mises(self) -> np.ndarray:
        s = self.stress - (self.stress.sum(axis=1) / 3.0)[:, None] * _H_MAT
        return np.sqrt(1.5 * _iso_inner_batch(s, s))

    @property
    def hydrostatic(self) -> np.ndarray:
        return self.stress[:, :3].sum(axis=1) / 3.0

    def plastic_dissipation_increment(self, sigma_y0: float, hardening: float,
                                      alpha_old: np.ndarray) -> np.ndarray:
        """Exact dissipation increment ``int sigma_y dalpha`` per point.

        With ``sigma_y(alpha) = sigma_y0 + H alpha`` the integral is
        ``sigma_y0*dalpha + H/2*(alpha_new^2 - alpha_old^2)``.
        """
        d_alpha = self.alpha - np.asarray(alpha_old, dtype=float)
        return sigma_y0 * d_alpha + 0.5 * hardening * (
            self.alpha ** 2 - np.asarray(alpha_old, dtype=float) ** 2
        )


def batch_return_map(
    material: J2Material,
    strain: np.ndarray,
    eps_p: np.ndarray,
    alpha: np.ndarray,
) -> BatchResponse:
    """Vectorised backward-Euler radial return for ``n`` Gauss points.

    Parameters
    ----------
    material : :class:`J2Material`
    strain : ``(n, 6)`` total strain increment (total-strain formulation, the
        history is carried by ``eps_p``/``alpha``)
    eps_p : ``(n, 6)`` plastic strain at the start of the increment
    alpha : ``(n,)`` equivalent plastic strain at the start of the increment
    """
    eps = np.atleast_2d(np.asarray(strain, dtype=float))
    eps_p = np.atleast_2d(np.asarray(eps_p, dtype=float))
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float)).astype(float)
    if eps.shape != eps_p.shape:
        raise ValueError(f"strain {eps.shape} and eps_p {eps_p.shape} must match")
    n = eps.shape[0]

    E, mu, bulk, H = material.E, material.mu, material.bulk, material.hardening
    G = mu
    C_el = material.elastic_matrix
    P_dev = deviatoric_projector()
    P_strain = deviatoric_strain_map()

    sig_tr = (eps - eps_p) @ C_el
    s_tr = sig_tr @ P_dev
    q_tr = np.sqrt(1.5 * _iso_inner_batch(s_tr, s_tr))
    sigma_y_n = material.sigma_y0 + H * alpha
    f_tr = q_tr - sigma_y_n
    plastic = (f_tr > 0.0) & (q_tr > 1e-300)

    # elastic branch by default
    stress = sig_tr.copy()
    eps_p_new = eps_p.copy()
    alpha_new = alpha.copy()
    tangent = np.repeat(C_el[None, :, :], n, axis=0)

    if not np.any(plastic):
        return BatchResponse(stress=stress, eps_p=eps_p_new, alpha=alpha_new,
                             tangent=tangent, plastic=plastic)

    idx = np.nonzero(plastic)[0]
    q = q_tr[idx]
    dlam = f_tr[idx] / (3.0 * G + H)
    dlam = np.minimum(dlam, q / (3.0 * G))          # clip at the deviatoric origin
    theta = 1.0 - 3.0 * G * dlam / q
    s_new = theta[:, None] * s_tr[idx]
    vol = bulk * eps[idx][:, :3].sum(axis=1)   # trace: normal components only
    stress[idx] = vol[:, None] * _H_MAT + s_new

    q_new = q - 3.0 * G * dlam
    flow = np.zeros_like(s_new)
    live = q_new > 1e-12 * material.sigma_y0
    flow[live] = (dlam[live] * 1.5 / q_new[live])[:, None] * s_new[live]
    flow[:, 3:] *= 2.0                               # engineering shear storage
    eps_p_new[idx] = eps_p[idx] + flow
    alpha_new[idx] = alpha[idx] + dlam

    # analytic consistent (algorithmic) tangent
    M = s_tr[idx] / q[:, None]
    coef = 9.0 * G * G * (1.0 / (3.0 * G + H) - dlam / q)
    C_ep = (
        bulk * np.broadcast_to(np.outer(_H_MAT, _H_MAT), (idx.size, 6, 6))
        + (2.0 * G * theta)[:, None, None] * P_strain[None, :, :]
        - coef[:, None, None] * np.einsum("gi,gj->gij", M, M)
    )
    tangent[idx] = 0.5 * (C_ep + np.transpose(C_ep, (0, 2, 1)))

    return BatchResponse(stress=stress, eps_p=eps_p_new, alpha=alpha_new,
                         tangent=tangent, plastic=plastic)


def hydrostatic_strain_map() -> np.ndarray:
    """Map engineering-Voigt strain to its volumetric part ``tr(eps)/3``."""
    return np.outer(_H_MAT, _H_MAT) / 3.0


# --------------------------------------------------------------------------- #
# Material-point verification drivers (AC1 / AC2)
# --------------------------------------------------------------------------- #
@dataclass
class UniaxialResponse:
    """Uniaxial *stress* path response (lateral stresses driven to zero)."""

    strain_xx: np.ndarray
    lateral_strain: np.ndarray
    stress_xx: np.ndarray
    alpha: np.ndarray
    analytic: np.ndarray

    @property
    def max_relative_error(self) -> float:
        """``max |sigma_num - sigma_analytic| / sigma_analytic`` over the path.

        The zero-strain point (where the analytic stress vanishes) is excluded
        from the relative measure.
        """
        scale = np.abs(self.analytic)
        mask = scale > 1e-12 * max(float(np.max(scale)), 1e-30)
        if not np.any(mask):
            return 0.0
        return float(np.max(np.abs(self.stress_xx - self.analytic)[mask]
                            / scale[mask]))

    def plastic_onset_strain(self) -> float | None:
        idx = np.nonzero(self.alpha > 1e-14)[0]
        return None if idx.size == 0 else float(self.strain_xx[idx[0]])


def _lateral_strain_zero_lateral_stress(
    material: J2Material, eps_xx: float, state: PlasticState,
    xtol: float = 1e-16, rtol: float = 1e-14,
) -> tuple[float, np.ndarray, PlasticState]:
    """Solve ``sigma_yy(e_lat) = sigma_zz(e_lat) = 0`` for one strain increment."""

    def residual(e_lat: float) -> float:
        strain = np.array([eps_xx, e_lat, e_lat, 0.0, 0.0, 0.0])
        return material.return_map(strain, state)[0][1]

    span = 0.75 * abs(eps_xx) + 1e-12
    lo, hi = -span, span
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo > 0.0:  # both ends already tensile -> widen once
        lo, f_lo = lo - span, residual(lo - span)
    if f_hi < 0.0:
        hi, f_hi = hi + span, residual(hi + span)
    if f_lo > 0.0 or f_hi < 0.0:  # pragma: no cover - defensive
        raise RuntimeError("could not bracket the zero-lateral-stress state")
    e_lat = brentq(residual, lo, hi, xtol=xtol, rtol=rtol, maxiter=200)
    strain = np.array([eps_xx, e_lat, e_lat, 0.0, 0.0, 0.0])
    stress, new_state, _C, _pl = material.return_map(strain, state)
    return e_lat, stress, new_state


def uniaxial_stress_response(
    material: J2Material,
    max_strain: float,
    n_steps: int = 400,
) -> UniaxialResponse:
    """Integrate a monotonic uniaxial *stress* path in ``n_steps`` increments.

    At every increment the lateral strain is solved so that
    ``sigma_yy = sigma_zz = 0`` exactly (to ``brentq`` tolerance), which is the
    classical tension/compression test condition.  The returned curve is
    compared with :func:`uniaxial_tension_analytic` (closed form for J2 +
    linear isotropic hardening), making AC1 a statement about the constitutive
    kernel and not about the textbook formula.
    """
    if max_strain == 0.0:
        raise ValueError("max_strain must be non-zero")
    strains = np.linspace(0.0, max_strain, int(n_steps) + 1)
    state = PlasticState()
    lateral = np.zeros_like(strains)
    stress_xx = np.zeros_like(strains)
    alpha = np.zeros_like(strains)
    for i, eps_xx in enumerate(strains):
        if i == 0:
            continue
        e_lat, stress, state = _lateral_strain_zero_lateral_stress(
            material, float(eps_xx), state
        )
        lateral[i] = e_lat
        stress_xx[i] = stress[0]
        alpha[i] = state.alpha
    # the J2 uniaxial-stress response is odd in the strain, so the tension
    # closed form is applied to |eps| and the sign is restored afterwards
    # (this also covers the *compression* branch, where the perfect-plastic
    # solution saturates at -sigma_y)
    analytic = np.sign(strains) * uniaxial_tension_analytic(
        np.abs(strains), material.E, material.sigma_y0, material.hardening)
    return UniaxialResponse(strain_xx=strains, lateral_strain=lateral,
                            stress_xx=stress_xx, alpha=alpha, analytic=analytic)


def shear_yield_stress(material: J2Material, tol: float = 1e-13) -> float:
    """Applied ``sigma_xy`` at first yield of the pure-shear path.

    The trial state for a pure engineering-shear strain ``gamma_xy`` is
    ``s = [0,0,0,G gamma,0,0]``, hence ``q = sqrt(3) G gamma``; the material
    therefore yields at ``sigma_xy = sigma_y / sqrt(3)``.  The value returned
    here is measured from the actual return mapping (bisection on the first
    increment that turns plastic), so it also validates the implementation.
    """
    G = material.mu

    def plastic(gamma: float) -> bool:
        strain = np.array([0.0, 0.0, 0.0, gamma, 0.0, 0.0])
        return bool(material.return_map(strain, PlasticState())[3])

    lo, hi = 0.0, 2.0 * material.sigma_y0 / G + 1e-12
    if not plastic(hi):  # pragma: no cover - defensive
        raise RuntimeError("pure-shear path did not yield within 2 sigma_y/G")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if plastic(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(hi, 1e-30):
            break
    return G * hi


def tangent_error(
    material: J2Material, strain: np.ndarray, state: PlasticState, h: float = 1e-7
) -> float:
    """Max abs difference between the analytic and finite-difference tangent."""
    C_ana = material.return_map(np.asarray(strain, dtype=float), state)[2]
    C_num = numerical_tangent(material, np.asarray(strain, dtype=float), state, h=h)
    scale = np.max(np.abs(C_ana))
    return float(np.max(np.abs(C_ana - C_num)) / scale)