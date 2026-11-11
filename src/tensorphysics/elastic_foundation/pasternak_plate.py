r"""Infinite plate on a Pasternak (two-parameter) and Vlasov elastic foundation.

Model
-----
The Pasternak foundation reacts with

``p(x, y) = k w(x, y) - G_p lap(w)(x, y)``   [Pa],

i.e. a Winkler spring bed ``k`` [Pa/m] in parallel with a shear layer of
modulus ``G_p`` [N/m] (Pasternak 1954; Kerr 1964).  Substitution into the
Kirchhoff plate equation gives

``D lap^4 w - G_p lap^2 w + k w = q``,   ``D = E h^3 / [12 (1 - nu^2)]``,

with ``w`` and ``q`` positive downwards.  ``G_p = 0`` recovers Winkler exactly,
and ``k = G_p = 0`` recovers the free (foundation-free) plate.

Closed-form Green function
--------------------------
Hankel transforming (``W(alpha) = int_0^inf w(r) J_0(alpha r) r dr``) turns the
operator into the quartic ``D alpha^4 + G_p alpha^2 + k``, whose ``alpha^2``
roots are ``m_1, m_2`` of ``D m^2 + G_p m + k = 0``.  Partial fractions and
``int_0^inf alpha J_0(alpha r)/(alpha^2 + b^2) d alpha = K_0(b r)`` give the
*exact* Green function used here::

    w(r) = P / [2 pi D (m_1 - m_2)] * [K_0(mu_1 r) - K_0(mu_2 r)],
    mu_i = sqrt(-m_i),  Re(mu_i) > 0,
    w(0) = P / [2 pi D (m_1 - m_2)] * ln(mu_2 / mu_1)   (finite),

which reduces *identically* to Hetényi's ``w = -P kei(r/l)/(2 pi k l^2)`` for
``G_p = 0`` (checked to machine precision in the tests).  The point-load
curvature, and hence ``M_r``, diverges logarithmically at ``r = 0`` — a
property of the point load, not of the foundation.

Vlasov elastic half-space parameters
------------------------------------
Vlasov & Leontiev (1966) identify the two Pasternak constants with a
compressible elastic layer of thickness ``H`` (modulus ``E_s``, Poisson ratio
``nu_s``) and a linear displacement shape function ``phi(z) = 1 - z/H``::

    k   = E_s / [(1 - nu_s^2) H],     G_p = E_s H / [6 (1 + nu_s)] .

``H -> inf`` is the "elastic half-space" limit: ``k -> 0`` and ``G_p -> inf``.
The two constants are also fitted directly to the exact half-space kernel
``C alpha`` (see :mod:`tensorphysics.elastic_foundation.halfspace`) by
:func:`match_pasternak_to_halfspace`.

Conventions
-----------
``w`` [m] and ``q`` [Pa] positive downwards; lengths in metres; results are
real (imaginary parts are round-off).
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0, j1, kv

from .hankel import oscillatory_quad

__all__ = [
    "pasternak_plate_roots",
    "pasternak_plate_point_load",
    "pasternak_plate_center_deflection",
    "pasternak_plate_point_load_hankel",
    "pasternak_plate_patch_load",
    "pasternak_plate_moment_r",
    "pasternak_plate_patch_moment_r",
    "pasternak_plate_bilaplacian_residual",
    "pasternak_plate_char_length",
    "vlasov_foundation_params",
    "vlasov_layer_depth",
    "match_pasternak_to_halfspace",
]

_GAMMA_EULER = 0.5772156649015329


# ---------------------------------------------------------------------------
# spectral roots
# ---------------------------------------------------------------------------
def pasternak_plate_roots(D: float, k: float, G_p: float = 0.0):
    """Roots of ``D m^2 + G_p m + k = 0`` and the wave numbers ``mu_i``.

    Returns
    -------
    (m1, m2, mu1, mu2) : tuple of complex
        ``Re(mu_i) > 0`` selects the decaying branch.  ``G_p = 0`` gives
        ``mu = exp(+-i pi/4)/l`` with ``l = (D/k)^(1/4)`` (Kelvin functions).
    """
    D = float(D)
    k = float(k)
    G_p = float(G_p)
    if D <= 0.0:
        raise ValueError("D must be positive")
    if k <= 0.0:
        raise ValueError("k must be positive")
    if G_p < 0.0:
        raise ValueError("G_p must be non-negative")
    disc = complex(G_p * G_p - 4.0 * D * k)
    if abs(disc) <= 1e-12 * (G_p * G_p + 4.0 * D * k):
        raise ValueError("degenerate Pasternak roots (G_p = 2 sqrt(D k)); "
                         "perturb G_p")
    root = np.sqrt(disc)
    m1 = (-G_p + root) / (2.0 * D)
    m2 = (-G_p - root) / (2.0 * D)
    mu1 = np.sqrt(-m1)
    mu2 = np.sqrt(-m2)
    if mu1.real < 0.0:
        mu1 = -mu1
    if mu2.real < 0.0:
        mu2 = -mu2
    if abs(m1 - m2) < 1e-11 * max(abs(m1), abs(m2)):
        raise ValueError("degenerate Pasternak roots (G_p = 2 sqrt(D k)); "
                         "perturb G_p")
    return m1, m2, mu1, mu2


def pasternak_plate_char_length(D: float, k: float) -> float:
    """Winkler characteristic length ``l = (D/k)^(1/4)`` [m] (shear-free limit)."""
    return (float(D) / float(k)) ** 0.25


# ---------------------------------------------------------------------------
# closed-form point load
# ---------------------------------------------------------------------------
def _kernel_scale(D: float, k: float, G_p: float) -> float:
    """Largest spectral wavenumber ``max(|mu_1|, |mu_2|)`` of the kernel [1/m].

    The Hankel integrand varies on the scale ``1/scale`` and has decayed beyond
    ``alpha ~ 60 scale``; both are used to size the quadrature range/panels.
    """
    mu = pasternak_plate_roots(D, k, G_p)[2:]
    return float(max(abs(mu[0]), abs(mu[1])))


def _k0_derivs(z, order: int):
    """``K_0`` and its first ``order`` derivatives with respect to its argument."""
    if np.iscomplexobj(z):
        z = complex(z)
        out = [kv(0, z)]
        k0, k1 = out[0], kv(1, z)
        if order >= 1:
            out.append(-k1)
        if order >= 2:
            out.append(k0 + k1 / z)
        if order >= 3:
            out.append(-(k1 + k0 / z + 2.0 * k1 / z ** 2))
        if order >= 4:
            out.append(k0 * (1.0 + 3.0 / z ** 2) + k1 * (2.0 / z + 6.0 / z ** 3))
    else:
        z = np.asarray(z, dtype=float)
        k0, k1 = kv(0, z), kv(1, z)
        out = [k0]
        if order >= 1:
            out.append(-k1)
        if order >= 2:
            out.append(k0 + k1 / z)
        if order >= 3:
            out.append(-(k1 + k0 / z + 2.0 * k1 / z ** 2))
        if order >= 4:
            out.append(k0 * (1.0 + 3.0 / z ** 2) + k1 * (2.0 / z + 6.0 / z ** 3))
    return out


def pasternak_plate_point_load(r, P: float, D: float, k: float,
                               G_p: float = 0.0, deriv: int = 0):
    """Exact infinite-plate Pasternak Green function and its r-derivatives.

    ``w(r) = P / [2 pi D (m1 - m2)] [K_0(mu1 r) - K_0(mu2 r)]``.

    Parameters
    ----------
    r : array_like
        Radius [m] (load at the origin).
    P : float
        Point load [N], positive downwards.
    D, k, G_p : float
        Flexural rigidity [N m], subgrade modulus [Pa/m], shear-layer modulus
        [N/m].
    deriv : int
        0 -> ``w``, 1 -> ``w'``, 2 -> ``w''``, 3 -> ``w'''``, 4 -> ``w''''``.
        Odd derivatives vanish *at* ``r = 0`` by symmetry.

    Notes
    -----
    ``w(0)`` is finite; ``w''`` (hence the bending moment) diverges
    logarithmically at the load point, so derivatives at ``r < 1e-6 (D/k)^(1/4)``
    are returned as ``nan``.
    """
    if deriv not in (0, 1, 2, 3, 4):
        raise ValueError("deriv must be 0..4")
    m1, m2, mu1, mu2 = pasternak_plate_roots(D, k, G_p)
    C = float(P) / (2.0 * np.pi * float(D) * (m1 - m2))
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    l = pasternak_plate_char_length(D, k)
    out = np.empty(rr.shape, dtype=complex)
    for i, rv in enumerate(rr):
        if rv < 0.0:
            raise ValueError("r must be non-negative")
        if rv == 0.0:
            if deriv == 0:
                out[i] = C * np.log(mu2 / mu1)
            else:
                out[i] = 0.0 if deriv % 2 == 1 else np.nan
        else:
            d1 = _k0_derivs(mu1 * rv, deriv)[deriv]
            d2 = _k0_derivs(mu2 * rv, deriv)[deriv]
            out[i] = C * (mu1 ** deriv * d1 - mu2 ** deriv * d2)
    res = np.real(out)
    if deriv % 2 == 1:
        res = np.where(rr == 0.0, 0.0, res)
    tiny = rr < 1e-6 * l
    if deriv >= 2:
        res = np.where(tiny, np.nan, res)
    return res


def pasternak_plate_center_deflection(P: float, D: float, k: float,
                                     G_p: float = 0.0) -> float:
    """Centre deflection ``w(0) = P ln(mu2/mu1) / [2 pi D (m1 - m2)]`` [m]."""
    m1, m2, mu1, mu2 = pasternak_plate_roots(D, k, G_p)
    C = float(P) / (2.0 * np.pi * float(D) * (m1 - m2))
    return float(np.real(C * np.log(mu2 / mu1)))


def pasternak_plate_point_load_hankel(r, P: float, D: float, k: float,
                                      G_p: float = 0.0) -> float:
    """Point-load deflection from the defining Hankel integral (independent).

    ``w(r) = P/(2 pi) int_0^inf alpha J_0(alpha r)/(D alpha^4 + G_p alpha^2 + k)
    d alpha`` — no Bessel-function identity is used, so it is an independent
    re-derivation of :func:`pasternak_plate_point_load`.
    """
    D, k, G_p = float(D), float(k), float(G_p)
    rho = float(r)
    if rho < 0.0:
        raise ValueError("r must be non-negative")

    def integrand(alpha):
        return alpha * j0(alpha * rho) / (D * alpha ** 4 + G_p * alpha ** 2 + k)

    scale = _kernel_scale(D, k, G_p)
    alpha_max = 60.0 * scale
    val = oscillatory_quad(integrand, alpha_max, max(rho, 1e-3 * scale))
    return float(P / (2.0 * np.pi) * float(val))


def pasternak_plate_patch_load(r, P: float, a: float, D: float, k: float,
                               G_p: float = 0.0):
    """Uniform pressure over a circular patch of radius ``a`` on Pasternak.

    ``w(r) = P/(pi a) int_0^inf J_1(alpha a) J_0(alpha r) /
    (D alpha^4 + G_p alpha^2 + k) d alpha``.
    """
    D, k, G_p = float(D), float(k), float(G_p)
    a = float(a)
    if a <= 0.0:
        raise ValueError("a must be positive")
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    scale = _kernel_scale(D, k, G_p)
    alpha_max = 60.0 * scale
    out = np.empty(rr.shape)
    for i, rv in enumerate(rr):
        def integrand(alpha, _r=rv):
            return (j1(alpha * a) * j0(alpha * _r)
                    / (D * alpha ** 4 + G_p * alpha ** 2 + k))

        val = oscillatory_quad(integrand, alpha_max, max(a, rv))
        out[i] = float(P / (np.pi * a) * float(val))
    return out


def pasternak_plate_moment_r(r, P: float, D: float, k: float, nu: float,
                             G_p: float = 0.0):
    """Radial bending moment ``M_r = -D (w'' + nu w'/r)`` [N m/m]."""
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    d1 = pasternak_plate_point_load(rr, P, D, k, G_p, deriv=1)
    d2 = pasternak_plate_point_load(rr, P, D, k, G_p, deriv=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -float(D) * (d2 + float(nu) * d1 / rr)


def pasternak_plate_patch_moment_r(r, P: float, a: float, D: float, k: float,
                                   nu: float, G_p: float = 0.0):
    """Radial bending moment of the patch-loaded infinite Pasternak plate.

    With the spectral amplitude ``A(alpha) = P alpha J_1(alpha a)/
    [pi a (D alpha^4 + G_p alpha^2 + k)]`` and ``w = int A J_0(alpha r) d alpha``,

    ``M_r = D int A alpha^2 [J_0(alpha r) - (1 - nu) J_1(alpha r)/(alpha r)] d alpha``

    which is smooth at ``r = 0`` (finite patch pressure), unlike the point-load
    moment.
    """
    D, a, nu, G_p = float(D), float(a), float(nu), float(G_p)
    if a <= 0.0:
        raise ValueError("a must be positive")
    scale = _kernel_scale(D, k, G_p)
    alpha_max = 60.0 * scale
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.empty(rr.shape)
    for i, rv in enumerate(rr):

        def integrand(alpha, _r=rv):
            amp = alpha * j1(alpha * a) / (np.pi * a * (D * alpha ** 4 + G_p * alpha ** 2 + k))
            if _r == 0.0:
                # lim_{r->0} J_1(alpha r)/(alpha r) = 1/2
                bracket = 1.0 - (1.0 - nu) / 2.0
            else:
                bracket = j0(alpha * _r) - (1.0 - nu) * j1(alpha * _r) / (alpha * _r)
            return D * amp * alpha ** 2 * bracket

        val = oscillatory_quad(integrand, alpha_max, max(a, rv))
        out[i] = float(P) * float(val)
    return out


def pasternak_plate_bilaplacian_residual(r, P: float, D: float, k: float,
                                         G_p: float = 0.0):
    """Relative residual ``(D lap^4 w - G_p lap^2 w + k w)/(k w)``.

    Zero for an exact solution away from the load; it proves that the closed
    form satisfies the Pasternak plate equation.
    """
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    d = [pasternak_plate_point_load(rr, P, D, k, G_p, deriv=n) for n in range(5)]
    w, d1, d2, d3, d4 = d
    lap = d2 + d1 / rr
    lap4 = d4 + 2.0 * d3 / rr - d2 / rr ** 2 + d1 / rr ** 3
    return (float(D) * lap4 - float(G_p) * lap + float(k) * w) / (float(k) * w)


# ---------------------------------------------------------------------------
# Vlasov identification with an elastic layer / half-space
# ---------------------------------------------------------------------------
def vlasov_foundation_params(E_s: float, nu_s: float, H: float):
    """Vlasov two-parameter constants of a compressible layer of depth ``H``.

    ``k = E_s/[(1 - nu_s^2) H]``, ``G_p = E_s H/[6 (1 + nu_s)]`` [Pa/m, N/m].
    """
    E_s, nu_s, H = float(E_s), float(nu_s), float(H)
    if E_s <= 0.0 or H <= 0.0:
        raise ValueError("E_s and H must be positive")
    if not (-1.0 < nu_s < 0.5):
        raise ValueError("nu_s out of range")
    k = E_s / ((1.0 - nu_s ** 2) * H)
    G_p = E_s * H / (6.0 * (1.0 + nu_s))
    return k, G_p


def vlasov_layer_depth(E_s: float, nu_s: float, k: float) -> float:
    """Depth ``H`` whose Vlasov subgrade modulus equals ``k`` [m]."""
    E_s, nu_s, k = float(E_s), float(nu_s), float(k)
    if k <= 0.0:
        raise ValueError("k must be positive")
    return E_s / ((1.0 - nu_s ** 2) * k)


def match_pasternak_to_halfspace(D: float, E_s: float, nu_s: float,
                                 alpha_ref: float | None = None):
    """Two-Pasternak-constant fit of the exact half-space kernel ``C alpha``.

    The elastic half-space supplies the foundation kernel ``C alpha`` with
    ``C = E_s/[2 (1 - nu_s^2)]`` (Boussinesq), whereas Pasternak supplies
    ``k + G_p alpha^2``.  Matching value *and* slope at the plate-relevant
    wavenumber ``alpha_ref`` (default ``1/l_hs``, ``l_hs = (D/C)^(1/3)``)
    gives ``G_p = C/(2 alpha_ref)`` and ``k = C alpha_ref/2``.

    Returns ``(k, G_p, alpha_ref)``.
    """
    C = float(E_s) / (2.0 * (1.0 - float(nu_s) ** 2))
    l_hs = (float(D) / C) ** (1.0 / 3.0)
    if alpha_ref is None:
        alpha_ref = 1.0 / l_hs
    alpha_ref = float(alpha_ref)
    G_p = C / (2.0 * alpha_ref)
    k = C * alpha_ref / 2.0
    return k, G_p, alpha_ref