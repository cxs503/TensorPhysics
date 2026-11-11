r"""Elastic half-space (continuum) foundation for an infinite Kirchhoff plate.

Model
-----
The Winkler and Pasternak foundations are *local*: the subgrade pressure at a
point depends only on the deflection (and its curvature) there.  The physically
complete one-parameter family is the **elastic half-space** (:math:`E_s`,
:math:`\nu_s`), for which the Boussinesq solution gives the surface deflection
under an axisymmetric pressure :math:`p(r)`::

    w(r) = (1 - nu_s^2)/(pi E_s) * int int_A p(rho) rho drho dtheta / R.

Hankel transforming (``int_0^inf . J_0(alpha r) r dr``) turns the Boussinesq
kernel into the algebraic factor ``2 (1 - nu_s^2)/(E_s alpha)``, so the coupled
plate/half-space problem has the *exact* spectral solution::

    [D alpha^4 + C alpha] W(alpha) = Q(alpha),   C = E_s / [2 (1 - nu_s^2)],

    w(r) = int_0^inf Q(alpha) J_0(alpha r)/(D alpha^3 + C) d alpha,

with ``Q = P J_1(alpha a)/(pi a)`` for a uniform patch pressure and ``Q = P/(2 pi)``
for a point load.  Note the fractional power ``alpha^1`` of the foundation
operator: this is precisely what the two-parameter Pasternak kernel
``k + G_p alpha^2`` can only approximate (see
:func:`tensorphysics.elastic_foundation.pasternak_plate.match_pasternak_to_halfspace`).

Consequences that the verification surface exercises
----------------------------------------------------
* **No finite point-load deflection.**  As the patch radius ``a -> 0``,
  ``w(0)`` diverges logarithmically, ``d w(0)/d ln a = -P/(pi C)``; a continuum
  foundation therefore always needs a finite contact patch, whereas Winkler /
  Pasternak give a finite ``w(0)`` (:func:`pasternak_plate_center_deflection`).
* **Plate-free (Boussinesq) limit.**  ``D -> 0`` recovers the classical
  Boussinesq closed forms :func:`boussinesq_patch_deflection`,
  :func:`boussinesq_point_deflection` and :func:`boussinesq_punch_deflection`.
* **Stiff-plate law.**  For ``a << l_hs = (D/C)^(1/3)`` the peak deflection obeys
  ``w_max ~ D^(-1/3)`` ; the exponent is an exact limit law of the spectral
  solution.
* **Far field.**  At ``r >> a`` the plate term dies and ``w -> P (1 - nu_s^2)/
  (pi E_s r)`` (Boussinesq point load), independent of ``D``.

Conventions
-----------
``w`` [m] and ``p`` [Pa] positive downwards; all lengths in metres.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk, j0, j1

from .hankel import oscillatory_quad

__all__ = [
    "halfspace_foundation_modulus",
    "halfspace_plate_char_length",
    "halfspace_plate_patch_load",
    "halfspace_plate_center_deflection",
    "halfspace_plate_moment_r",
    "halfspace_plate_spectral_integrand",
    "boussinesq_patch_deflection",
    "boussinesq_point_deflection",
    "boussinesq_punch_deflection",
]


# ---------------------------------------------------------------------------
# continuum constants
# ---------------------------------------------------------------------------
def halfspace_foundation_modulus(E_s: float, nu_s: float) -> float:
    """Boussinesq foundation modulus ``C = E_s/[2 (1 - nu_s^2)]`` [Pa]."""
    E_s, nu_s = float(E_s), float(nu_s)
    if E_s <= 0.0:
        raise ValueError("E_s must be positive")
    if not (-1.0 < nu_s < 0.5):
        raise ValueError("nu_s out of range")
    return E_s / (2.0 * (1.0 - nu_s ** 2))


def halfspace_plate_char_length(D: float, C: float) -> float:
    """Continuum plate-foundation length ``l_hs = (D/C)^(1/3)`` [m]."""
    D, C = float(D), float(C)
    if D <= 0.0 or C <= 0.0:
        raise ValueError("D and C must be positive")
    return (D / C) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# spectral solution
# ---------------------------------------------------------------------------
def halfspace_plate_spectral_integrand(alpha, r: float, a: float, D: float,
                                      C: float) -> float:
    """Spectral amplitude ``A(alpha) J_0(alpha r)`` of the deflection [m/Pa].

    ``A(alpha) = Q(alpha)/(D alpha^3 + C)`` with the pressure transform
    ``Q = J_1(alpha a)/(pi a alpha)`` for a uniform patch (``a > 0``) and
    ``Q = 1/(2 pi)`` for a point load (``a == 0``); the total load is applied by
    the callers.  The extra ``1/alpha`` (relative to the algebraic kernel)
    comes from ``alpha d alpha`` of the Hankel inversion.
    """
    if a > 0.0:
        num = j1(alpha * a) * j0(alpha * r) / (np.pi * a * alpha)
    else:
        num = j0(alpha * r) / (2.0 * np.pi)
    return num / (D * alpha ** 3 + C)


def halfspace_plate_patch_load(r, P: float, a: float, D: float, E_s: float,
                              nu_s: float):
    """Infinite plate on an elastic half-space, uniform pressure over ``r <= a``.

    ``w(r) = P int_0^inf J_1(alpha a) J_0(alpha r)/[pi a alpha (D alpha^3 + C)]
    d alpha``.
    """
    D = float(D)
    a = float(a)
    if a <= 0.0:
        raise ValueError("a must be positive (the point-load limit diverges)")
    C = halfspace_foundation_modulus(E_s, nu_s)
    l_hs = halfspace_plate_char_length(D, C)
    alpha_max = 100.0 / l_hs
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.empty(rr.shape)
    for i, rv in enumerate(rr):
        def integrand(alpha, _r=rv):
            return halfspace_plate_spectral_integrand(alpha, _r, a, D, C)

        # only the Bessel factors oscillate; the algebraic decay is resolved by
        # the geometrically graded panels of oscillatory_quad
        out[i] = float(P) * float(oscillatory_quad(integrand, alpha_max,
                                                   max(a, rv)))
    return out


def halfspace_plate_center_deflection(P: float, a: float, D: float, E_s: float,
                                      nu_s: float) -> float:
    """Centre deflection of the patch-loaded plate on an elastic half-space [m]."""
    return float(halfspace_plate_patch_load(0.0, P, a, D, E_s, nu_s)[0])


def halfspace_plate_moment_r(r, P: float, a: float, D: float, E_s: float,
                             nu_s: float, nu: float):
    """Radial bending moment ``M_r`` of the patch-loaded plate [N m/m].

    From ``w = int A(alpha) J_0(alpha r) d alpha``,
    ``w'' + nu w'/r = -int A alpha^2 [J_0(alpha r) - (1 - nu) J_1(alpha r)/
    (alpha r)] d alpha`` and ``M_r = -D (w'' + nu w'/r)``.
    """
    D = float(D)
    a = float(a)
    nu = float(nu)
    if a <= 0.0:
        raise ValueError("a must be positive")
    C = halfspace_foundation_modulus(E_s, nu_s)
    l_hs = halfspace_plate_char_length(D, C)
    alpha_max = 100.0 / l_hs
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.empty(rr.shape)
    for i, rv in enumerate(rr):
        if rv <= 0.0:
            out[i] = np.nan
            continue

        def integrand(alpha, _r=rv):
            amp = (j1(alpha * a) / (np.pi * a * alpha)) / (D * alpha ** 3 + C)
            return D * alpha ** 2 * amp * (
                j0(alpha * _r) - (1.0 - nu) * j1(alpha * _r) / (alpha * _r))

        val = oscillatory_quad(integrand, alpha_max, max(a, rv))
        out[i] = float(P) * float(val)
    return out


# ---------------------------------------------------------------------------
# Boussinesq closed forms (the D -> 0 / no-plate limit)
# ---------------------------------------------------------------------------
def boussinesq_patch_deflection(r, p: float, a: float, E_s: float, nu_s: float):
    """Exact surface deflection of a half-space under uniform pressure ``p`` on
    ``r <= a`` (Boussinesq 1885).

    ``r <= a``: ``w = 4 p a (1 - nu^2)/(pi E) * E_ell(m = (r/a)^2)`` with
    ``E_ell`` the complete elliptic integral of the second kind in the
    *parameter* convention of ``scipy.special.ellipe`` (``E_ell(0) = pi/2`` ->
    ``w(0) = 2 p a (1 - nu^2)/E``; ``E_ell(1) = 1`` -> ``w(a) = 4 p a (1 - nu^2)/
    (pi E)``).
    ``r > a``: ``w = 4 (1 - nu^2) p/(pi E) * int_0^a K(m) rho drho/(r + rho)``
    with ``m = 4 r rho/(r + rho)^2`` (smooth 1-D quadrature; the far field is
    :func:`boussinesq_point_deflection`).
    """
    p, a = float(p), float(a)
    E_s, nu_s = float(E_s), float(nu_s)
    if a <= 0.0:
        raise ValueError("a must be positive")
    pref = 4.0 * p * a * (1.0 - nu_s ** 2) / (np.pi * E_s)
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.empty(rr.shape)
    nodes, weights = np.polynomial.legendre.leggauss(240)
    for i, rv in enumerate(rr):
        if rv <= a:
            out[i] = pref * float(ellipe((rv / a) ** 2))
        else:
            rho = 0.5 * a * (nodes + 1.0)
            wts = 0.5 * a * weights
            m = 4.0 * rv * rho / (rv + rho) ** 2
            integ = ellipk(m) * rho / (rv + rho)
            out[i] = pref / a * float(np.dot(wts, integ))
    return out


def boussinesq_point_deflection(P: float, r: float, E_s: float, nu_s: float) -> float:
    """Boussinesq point-load surface deflection ``P (1 - nu^2)/(pi E r)`` [m]."""
    r = float(r)
    if r <= 0.0:
        raise ValueError("the point-load deflection diverges at r = 0")
    return float(P) * (1.0 - float(nu_s) ** 2) / (np.pi * float(E_s) * r)


def boussinesq_punch_deflection(P: float, a: float, E_s: float, nu_s: float) -> float:
    """Flat-ended rigid punch: ``w = P (1 - nu^2)/(2 E a)`` [m] (Boussinesq)."""
    a = float(a)
    if a <= 0.0:
        raise ValueError("a must be positive")
    return float(P) * (1.0 - float(nu_s) ** 2) / (2.0 * float(E_s) * a)