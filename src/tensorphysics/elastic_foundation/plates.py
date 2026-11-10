r"""Infinite and finite-radius plates on a Winkler foundation.

The plate equation (deflection ``w`` positive downwards) is::

    D grad^4 w + k w = q(x, y),      D = E h^3 / [12 (1 - nu^2)].

Closed-form point-load solution (Hetényi 1946)
----------------------------------------------
With the characteristic length ``l = (D/k)^(1/4)`` the Green function of the
infinite plate is (Hankel-transform derivation, see
:func:`winkler_plate_point_load_hankel`)::

    w(r) = (P / (2 pi k l^2)) * int_0^inf  u J_0(u r/l) / (1 + u^4) du
         = - P * kei(r/l) / (2 pi k l^2)  =  (P/(8 k l^2)) * (-4/pi) * kei(r/l)

so that ``w(0) = P/(8 k l^2) = P/(8 sqrt(k D))``, the classical Hetényi value
(``kei(0) = -pi/4``).  ``ker`` and ``kei`` are the Kelvin functions, i.e. the
real and imaginary parts of ``K_0(r e^{i pi/4}/l)``.

Two frequently mis-transcribed variants are *not* used here and are kept only
for falsification:

* ``w = P/(8 k l^2) [ker(r/l) + kei(r/l)]`` — ``ker(r/l)`` diverges
  logarithmically as ``r -> 0`` so the centre deflection is infinite, and the
  ``kei`` amplitude is off by the factor ``4/pi``;
* any expression containing ``ker`` for a *point* load, because ``ker`` is a
  singular homogeneous solution that must be excluded by regularity at the
  origin.

Finite radius and boundary effects
----------------------------------
For a circular plate of radius ``R`` on a Winkler foundation, held at the edge
(clamped: ``w = w' = 0``; simply supported: ``w = 0``, ``M_r = 0``), the exact
solution is the sum of the singular particular solution and the two *regular*
homogeneous solutions, which are the real and imaginary parts of
``I_0(r e^{i pi/4}/l)``::

    W(r) = -P kei(r/l)/(2 pi k l^2) + c1 Re I_0(q r) + c2 Im I_0(q r),
    q = e^{i pi/4}/l.

Because ``|I_0(q r)| ~ exp(r/(sqrt(2) l))``, the boundary perturbation
``c1, c2`` and hence the relative centre-deflection error decay like
``exp(-R/(sqrt(2) l))`` — this is the exponential boundary-effect law verified
numerically in the report.

Axisymmetric finite element
---------------------------
Conforming ring elements with two degrees of freedom per node ``(w, w')`` and
the strain energy

``U = pi D int [ w''^2 + (w'/r)^2 + 2 nu w'' w'/r ] r dr + pi k int w^2 r dr``

so that the element stiffness has the three contributions
``2 pi D int ( r N'' N''^T + N' N'^T / r + nu (N'' N'^T + N' N''^T) ) dr``
and ``2 pi k int r N N^T dr``.  At ``r = 0`` the rotational degree of freedom is
removed (axisymmetry demands ``w'(0) = 0``), which also removes the ``1/r``
singularity of the integrand.

References
----------
* Hetényi, M. (1946). *Beams on Elastic Foundation*, Univ. of Michigan Press.
* Timoshenko, S. & Woinowsky-Krieger, S. (1959). *Theory of Plates and Shells*,
  2nd ed., Art. 61 (plates on elastic foundation).
* Kerr, A.D. (1975). CRREL Report 75-1.
* Abramowitz & Stegun (1964), Ch. 9.9 (Kelvin functions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warnings

from scipy.integrate import quad
from scipy.special import iv, j0, j1, kv

__all__ = [
    "winkler_plate_flexural_rigidity",
    "winkler_plate_char_length",
    "winkler_plate_point_load",
    "winkler_plate_point_load_hankel",
    "winkler_plate_patch_load",
    "winkler_plate_patch_moment_r",
    "kelvin_k0_derivatives",
    "winkler_plate_load_recovery",
    "winkler_plate_moment_r",
    "winkler_plate_biharmonic_residual",
    "solve_axisym_plate_fem",
    "AxisymPlateFEMResult",
    "finite_plate_solution",
    "FinitePlateSolution",
    "finite_plate_center_deflection",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def winkler_plate_flexural_rigidity(E: float, h: float, nu: float) -> float:
    """Plate flexural rigidity ``D = E h^3 / [12 (1 - nu^2)]`` [N m]."""
    return float(E) * float(h) ** 3 / (12.0 * (1.0 - float(nu) ** 2))


def winkler_plate_char_length(D: float, k: float) -> float:
    """Characteristic (rigidity) length ``l = (D/k)^(1/4)`` [m]."""
    D = float(D)
    k = float(k)
    if D <= 0 or k <= 0:
        raise ValueError("D and k must be positive")
    return (D / k) ** 0.25


def kelvin_k0_derivatives(r, l: float, order: int = 2):
    """``K_0`` and its first ``order`` r-derivatives at ``z = r exp(i pi/4)/l``.

    ``Re K_0 = ker(r/l)``, ``Im K_0 = kei(r/l)``.  For ``order = 2`` the
    tuple ``(F, Fprime, Fsecond)`` is returned; for larger ``order`` the
    list ``[F, Fprime, ...]``.  Derivatives follow from the Bessel
    identities ``K0prime = -K1`` and ``K1prime = -K0 - K1/z``.
    """
    """``K_0``, ``d/dr K_0``, ``d^2/dr^2 K_0`` at the complex argument of the
    Kelvin functions, i.e. ``z = r exp(i pi/4)/l``.

    ``Re K_0 = ker(r/l)``, ``Im K_0 = kei(r/l)``.
    """
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    # K_0 has a logarithmic singularity at z = 0 while its imaginary part
    # (kei) stays finite there: kei(0) = -pi/4.  Nudging the origin by
    # 1e-30 l keeps the exact imaginary part and avoids log(0).
    rr = np.where(rr > 0.0, rr, 1e-30 * l)
    c = np.exp(1j * np.pi / 4) / l
    z = rr * c
    k0 = kv(0, z)
    k1 = kv(1, z)
    derivs = [k0, -c * k1]
    if order >= 2:
        derivs.append(c ** 2 * (k0 + k1 / z))
    if order >= 3:
        derivs.append(c ** 3 * (-k1 - k0 / z - 2.0 * k1 / z ** 2))
    if order >= 4:
        derivs.append(c ** 4 * (k0 * (1.0 + 3.0 / z ** 2)
                                + k1 * (2.0 / z + 6.0 / z ** 3)))
    if order > 4:
        raise ValueError("only derivatives up to order 4 are implemented")
    if order == 2:
        return tuple(derivs)
    return derivs


# ---------------------------------------------------------------------------
# closed-form infinite-plate solutions
# ---------------------------------------------------------------------------
def winkler_plate_point_load(r, P: float, D: float, k: float, deriv: int = 0):
    """Infinite Winkler plate, point load ``P`` at the origin (Hetényi/Kelvin).

    ``w(r) = -P kei(r/l) / (2 pi k l^2)``, with ``l = (D/k)^(1/4)`` and
    ``w(0) = P/(8 k l^2) = P/(8 sqrt(kD))``.

    ``deriv`` 0 returns ``w``, 1 returns ``w'``, 2 returns ``w''``.
    """
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    r0 = (rr == 0.0)
    l = winkler_plate_char_length(D, k)
    dfs = kelvin_k0_derivatives(rr, l, order=4)
    scale = -P / (2.0 * np.pi * k * l ** 2)
    if deriv > 4:  # pragma: no cover
        raise ValueError("deriv must be 0..4")
    out = np.real(scale * dfs[deriv].imag)
    if deriv in (1, 3):
        out = np.where(r0, 0.0, out)     # odd terms vanish at r = 0
    return out


def winkler_plate_point_load_hankel(r, P: float, D: float, k: float,
                                    limit: int = 200) -> float:
    """Infinite-plate point-load deflection from the *defining* Hankel integral.

    ``w(r) = (P/(2 pi k l^2)) int_0^inf u J_0(u r/l)/(1+u^4) du``.
    This is the independent re-derivation of :func:`winkler_plate_point_load`
    (no Kelvin-function identities involved).
    """
    l = winkler_plate_char_length(D, k)
    rho = float(r) / l

    def integrand(u):
        return u * j0(u * rho) / (1.0 + u ** 4)

    # the integrand oscillates with period pi/rho while decaying like u^-3:
    # split the range into half-period panels so that QUADPACK converges
    # (a single call over [0, inf) hits its subdivision limit).
    u_max = 60.0 + 40.0 * abs(rho)
    n_panel = 1 if abs(rho) < 1e-3 else int(np.ceil(u_max * abs(rho) / np.pi))
    n_panel = int(min(max(n_panel, 1), 240))
    edges = np.linspace(0.0, u_max, n_panel + 1)
    val = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        val += quad(integrand, lo, hi, limit=200)[0]
    # tail panel (integrand ~ u^-3 there, so the infinite range converges)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val += quad(integrand, edges[-1], np.inf, limit=400)[0]
    return P / (2.0 * np.pi * k * l ** 2) * val


def winkler_plate_load_recovery(P: float, D: float, k: float) -> float:
    """Ratio ``2 pi k int_0^inf w r dr / P`` for the infinite point-loaded plate.

    Equilibrium requires every applied load to be carried by the foundation, so
    this ratio must be 1: it is an amplitude check of the Kelvin solution that
    does not rely on any Kelvin-function identity.
    """
    l = winkler_plate_char_length(D, k)

    def integrand(r):
        return 2.0 * np.pi * k * float(winkler_plate_point_load(r, P, D, k)[0]) * r

    val, _err = quad(integrand, 0.0, 30.0 * l, limit=400)
    return val / P


def winkler_plate_moment_r(r, P: float, D: float, k: float, nu: float):
    """Radial bending moment of the infinite point-loaded plate [N m/m]."""
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    d1 = winkler_plate_point_load(rr, P, D, k, deriv=1)
    d2 = winkler_plate_point_load(rr, P, D, k, deriv=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = -D * (d2 + nu * d1 / rr)
    if np.any(rr == 0.0):
        val = np.where(rr > 0.0, val, -D * (1.0 + nu) * d2)
    return val


def winkler_plate_biharmonic_residual(r, P: float, D: float, k: float):
    """Relative residual ``(D lap^4 w + k w) / (k w)`` of the Kelvin solution.

    Identically zero for an exact solution; it proves that the closed form
    really solves the plate equation.  Polar form of the bilaplacian:
    ``lap^4 w = wIV + 2 wIII/r - wII/r^2 + wI/r^3``.
    """
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    d1 = winkler_plate_point_load(rr, P, D, k, deriv=1)
    d2 = winkler_plate_point_load(rr, P, D, k, deriv=2)
    d3 = winkler_plate_point_load(rr, P, D, k, deriv=3)
    d4 = winkler_plate_point_load(rr, P, D, k, deriv=4)
    w = winkler_plate_point_load(rr, P, D, k, deriv=0)
    lap4 = d4 + 2.0 * d3 / rr - d2 / rr ** 2 + d1 / rr ** 3
    return (D * lap4 + k * w) / (k * w)


def _quad_oscillatory(func, rho: float, u_max: float | None = None) -> float:
    """Integrate ``func(u)`` over ``[0, inf)`` with half-period panel splitting.

    Bessel-type integrands oscillate with period ``pi/|rho|`` and decay only
    algebraically, so a single adaptive call over ``[0, inf)`` hits QUADPACK's
    subdivision limit; splitting into panels keeps full machine accuracy.
    """
    u_max = 60.0 + 40.0 * abs(rho) if u_max is None else u_max
    n_panel = 1 if abs(rho) < 1e-3 else int(np.ceil(u_max * abs(rho) / np.pi))
    n_panel = int(min(max(n_panel, 1), 240))
    edges = np.linspace(0.0, u_max, n_panel + 1)
    val = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        val += quad(func, lo, hi, limit=200)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val += quad(func, edges[-1], np.inf, limit=400)[0]
    return val


def winkler_plate_patch_load(r, P: float, a: float, D: float, k: float):
    """Infinite Winkler plate, uniform pressure over a circular patch of radius ``a``.

    ``w(r) = int_0^inf q a J_1(xi a) J_0(xi r) / (D xi^4 + k) dxi`` with
    ``q = P/(pi a^2)``.  For ``a -> 0`` this tends to the point-load Kelvin
    solution (:func:`winkler_plate_point_load`).
    """
    l = winkler_plate_char_length(D, k)
    q = P / (np.pi * a * a)
    aa = float(a) / l
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.empty_like(rr)
    for i, rv in enumerate(rr):
        rho = rv / l

        def integrand(u, _rho=rho):
            return q * a * j1(u * aa) * j0(u * _rho) / (D * (u / l) ** 4 + k) / l

        out[i] = _quad_oscillatory(integrand, rho)
    return out


def winkler_plate_patch_moment_r(r, P: float, a: float, D: float, k: float,
                                 nu: float):
    """Radial bending moment of the patch-loaded infinite plate [N m/m].

    Derivatives are taken under the Hankel integral
    (``J_0' = -J_1``, ``J_0'' = -J_0 + J_1/(xi r)``)::

        w'  = int q a J_1(xi a) (-xi J_1(xi r)) / (D xi^4 + k) d xi
        w'' = int q a J_1(xi a) (-xi^2 J_0(xi r) + xi J_1(xi r)/r)
              / (D xi^4 + k) d xi
        M_r = -D (w'' + nu w'/r)

    At ``r = 0`` the limit ``M_r = M_t = -D (1+nu) w''(0)`` is used.
    """
    l = winkler_plate_char_length(D, k)
    q = P / (np.pi * a * a)
    aa = float(a) / l
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.empty_like(rr)
    for i, rv in enumerate(rr):
        rho = rv / l

        def wpp(u, _rho=rho):
            den = D * (u / l) ** 4 + k
            xi = u / l
            j0v = j0(u * _rho)
            j1v = j1(u * _rho)
            if rho == 0.0:
                # xi J_1(xi r)/r -> xi^2 / 2 as r -> 0
                tail = xi * xi * 0.5
            else:
                tail = xi * j1v / rv
            return q * a * j1(u * aa) * (-xi * xi * j0v + tail) / den / l

        def wp(u, _rho=rho):
            den = D * (u / l) ** 4 + k
            xi = u / l
            return q * a * j1(u * aa) * (-xi * j1(u * _rho)) / den / l

        w2 = _quad_oscillatory(wpp, rho)
        if rho == 0.0:
            out[i] = -D * (1.0 + nu) * w2
        else:
            w1 = _quad_oscillatory(wp, rho)
            out[i] = -D * (w2 + nu * w1 / rv)
    return out


def winkler_plate_load_recovery(P: float, D: float, k: float) -> float:
    """Ratio ``2 pi k int_0^inf w r dr / P`` for the infinite point-loaded plate.

    Equilibrium requires every applied load to be carried by the foundation, so
    this ratio must be 1: it is an amplitude check of the Kelvin solution that
    does not rely on any Kelvin-function identity.
    """
    l = winkler_plate_char_length(D, k)

    def integrand(r):
        return 2.0 * np.pi * k * float(winkler_plate_point_load(r, P, D, k)[0]) * r

    val, _err = quad(integrand, 0.0, 30.0 * l, limit=400)
    return val / P


def winkler_plate_moment_r(r, P: float, D: float, k: float, nu: float):
    """Radial bending moment of the infinite point-loaded plate [N m/m]."""
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    d1 = winkler_plate_point_load(rr, P, D, k, deriv=1)
    d2 = winkler_plate_point_load(rr, P, D, k, deriv=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = -D * (d2 + nu * d1 / rr)
    if np.any(rr == 0.0):
        val = np.where(rr > 0.0, val, -D * (1.0 + nu) * d2)
    return val


def winkler_plate_biharmonic_residual(r, P: float, D: float, k: float):
    """Relative residual ``(D lap^4 w + k w) / (k w)`` of the Kelvin solution.

    Identically zero for an exact solution; it proves that the closed form
    really solves the plate equation.  Polar form of the bilaplacian:
    ``lap^4 w = wIV + 2 wIII/r - wII/r^2 + wI/r^3``.
    """
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    d1 = winkler_plate_point_load(rr, P, D, k, deriv=1)
    d2 = winkler_plate_point_load(rr, P, D, k, deriv=2)
    d3 = winkler_plate_point_load(rr, P, D, k, deriv=3)
    d4 = winkler_plate_point_load(rr, P, D, k, deriv=4)
    w = winkler_plate_point_load(rr, P, D, k, deriv=0)
    lap4 = d4 + 2.0 * d3 / rr - d2 / rr ** 2 + d1 / rr ** 3
    return (D * lap4 + k * w) / (k * w)


def winkler_plate_patch_load(r, P: float, a: float, D: float, k: float):
    """Infinite Winkler plate, uniform pressure over a circular patch of radius ``a``.

    ``w(r) = int_0^inf q a J_1(xi a) J_0(xi r) / (D xi^4 + k) dxi`` with
    ``q = P/(pi a^2)``.  For ``r >> a`` this tends to the point-load Kelvin
    solution (:func:`winkler_plate_point_load`).
    """
    l = winkler_plate_char_length(D, k)
    q = P / (np.pi * a * a)
    aa = float(a) / l
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.empty_like(rr)
    for i, rv in enumerate(rr):
        rho = rv / l

        def integrand(u, _rho=rho):
            # u = xi l is dimensionless, d xi = du / l
            denom = D * (u / l) ** 4 + k
            return q * a * j1(u * aa) * j0(u * _rho) / denom / l

        out[i], _err = quad(integrand, 0.0, np.inf, limit=400)
    return out


# ---------------------------------------------------------------------------
# axisymmetric finite element
# ---------------------------------------------------------------------------
def _hermite(xi, le):
    """Hermite cubics and their r-derivatives at local coordinate ``xi``."""
    N = np.array([1.0 - 3.0 * xi ** 2 + 2.0 * xi ** 3,
                  le * (xi - 2.0 * xi ** 2 + xi ** 3),
                  3.0 * xi ** 2 - 2.0 * xi ** 3,
                  le * (-xi ** 2 + xi ** 3)])
    dN = np.array([(-6.0 * xi + 6.0 * xi ** 2) / le,
                   1.0 - 4.0 * xi + 3.0 * xi ** 2,
                   (6.0 * xi - 6.0 * xi ** 2) / le,
                   -2.0 * xi + 3.0 * xi ** 2])
    d2N = np.array([(-6.0 + 12.0 * xi) / le ** 2,
                    (-4.0 + 6.0 * xi) / le,
                    (6.0 - 12.0 * xi) / le ** 2,
                    (-2.0 + 6.0 * xi) / le])
    return N, dN, d2N


@dataclass
class AxisymPlateFEMResult:
    """Axisymmetric Kirchhoff ring-element solution on a Winkler foundation."""

    r: np.ndarray
    w: np.ndarray
    theta: np.ndarray
    D: float
    nu: float
    k: float
    bc: str
    load_radius: float = 0.0

    def _elem(self, rq: float) -> int:
        i = int(np.searchsorted(self.r, rq) - 1)
        return min(max(i, 0), len(self.r) - 2)

    def evaluate(self, rq, deriv: int = 0):
        """Interpolate ``w`` and its r-derivatives at arbitrary radii."""
        rq = np.atleast_1d(np.asarray(rq, dtype=float))
        out = np.empty_like(rq)
        for k, rv in enumerate(rq):
            e = self._elem(rv)
            r1, r2 = self.r[e], self.r[e + 1]
            le = r2 - r1
            xi = (rv - r1) / le
            d = np.array([self.w[e], self.theta[e],
                          self.w[e + 1], self.theta[e + 1]])
            N, dN, d2N = _hermite(xi, le)
            if deriv == 0:
                out[k] = float(N @ d)
            elif deriv == 1:
                out[k] = float(dN @ d)
            elif deriv == 2:
                out[k] = float(d2N @ d)
            else:  # pragma: no cover
                raise ValueError("deriv must be 0..2")
        return out

    def moment_r(self, rq):
        """Radial bending moment ``M_r = -D (w'' + nu w'/r)`` [N m/m]."""
        rq = np.atleast_1d(np.asarray(rq, dtype=float))
        w2 = self.evaluate(rq, 2)
        w1 = self.evaluate(rq, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = -self.D * (w2 + self.nu * w1 / rq)
        if np.any(rq == 0.0):
            val = np.where(rq > 0.0, val, -self.D * (1.0 + self.nu) * w2)
        return val

    def moment_t(self, rq):
        """Tangential bending moment ``M_t = -D (w'/r + nu w'')`` [N m/m]."""
        rq = np.atleast_1d(np.asarray(rq, dtype=float))
        w2 = self.evaluate(rq, 2)
        w1 = self.evaluate(rq, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = -self.D * (w1 / rq + self.nu * w2)
        if np.any(rq == 0.0):
            val = np.where(rq > 0.0, val, -self.D * (1.0 + self.nu) * w2)
        return val


def solve_axisym_plate_fem(D: float, nu: float, k: float, R_out: float,
                           P: float = 0.0, load_radius: float = 0.0,
                           n_center: int = 12, n_outer: int = 240,
                           grading: float = 2.0, bc: str = "free"):
    """Axisymmetric Kirchhoff ring-element solution on a Winkler foundation.

    Parameters
    ----------
    D, nu, k : float
        Flexural rigidity [N m], Poisson ratio, subgrade modulus [Pa/m].
    R_out : float
        Outer radius [m].
    P : float
        Total transverse load [N] (point load at ``r = 0`` when
        ``load_radius == 0``, otherwise uniform pressure over ``r <= load_radius``).
    load_radius : float
        Radius of the loaded patch [m]; the mesh is built so that a node sits
        exactly on ``r = load_radius``.
    n_center, n_outer : int
        Number of elements in ``[0, load_radius]`` and ``[load_radius, R_out]``.
    grading : float
        Power-law grading exponent of the outer mesh (``r = a + (R-a)(j/n)^p``),
        used to cluster elements near the load.
    bc : {'free', 'simply_supported', 'clamped'}
        Outer-edge condition.
    """
    if bc not in ("free", "simply_supported", "clamped"):
        raise ValueError("bc must be free / simply_supported / clamped")
    k = float(k)
    D = float(D)
    nu = float(nu)
    if load_radius > 0.0:
        a = float(load_radius)
        inner = np.linspace(0.0, a, n_center + 1)[:-1]
        outer = a + (R_out - a) * (np.arange(n_outer + 1) / n_outer) ** grading
        r = np.concatenate([inner, outer])
    else:
        r = R_out * (np.arange(n_outer + 1) / n_outer) ** grading
        a = 0.0
    nnode = r.size
    ndof = 2 * nnode
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    # 4-point Gauss is exact for the polynomial integrands of the first
    # element once the singular centre rotation is removed.
    gp, gw = np.polynomial.legendre.leggauss(6)
    for e in range(nnode - 1):
        r1, r2 = r[e], r[e + 1]
        le = r2 - r1
        ke = np.zeros((4, 4))
        fe = np.zeros(4)
        for xg, wg in zip(gp, gw):
            xi = 0.5 * (xg + 1.0)
            rr = r1 + xi * le
            jac = 0.5 * le * wg
            N, dN, d2N = _hermite(xi, le)
            ke += 2.0 * np.pi * D * (
                rr * np.outer(d2N, d2N)
                + np.outer(dN, dN) / rr
                + nu * (np.outer(d2N, dN) + np.outer(dN, d2N))
            ) * jac
            ke += 2.0 * np.pi * k * rr * np.outer(N, N) * jac
            if a > 0.0 and r2 <= a * (1.0 + 1e-12):
                q = P / (np.pi * a * a)
                fe += 2.0 * np.pi * q * rr * N * jac
        if r1 == 0.0:
            # axisymmetry: w'(0) = 0 -> remove the singular rotational DOF
            ke[1, :] = 0.0
            ke[:, 1] = 0.0
            fe[1] = 0.0
        idx = np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[np.ix_(idx, idx)] += ke
        F[idx] += fe

    if a == 0.0 and P != 0.0:
        F[0] += P                      # point load on the centre node

    # essential boundary conditions
    fixed = [1]                                 # w'(0) = 0
    last = nnode - 1
    if bc in ("simply_supported", "clamped"):
        fixed.append(2 * last)                  # w(R) = 0
    if bc == "clamped":
        fixed.append(2 * last + 1)              # w'(R) = 0
    free = np.setdiff1d(np.arange(ndof), fixed)
    sol = np.zeros(ndof)
    sol[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

    return AxisymPlateFEMResult(
        r=r, w=sol[0::2], theta=sol[1::2], D=D, nu=nu, k=k, bc=bc,
        load_radius=a,
    )


# ---------------------------------------------------------------------------
# exact finite-radius solution (boundary-effect law)
# ---------------------------------------------------------------------------
@dataclass
class FinitePlateSolution:
    """Exact finite-radius Winkler-plate solution (Kelvin particular + regular
    growing homogeneous pair)."""

    R: float
    l: float
    bc: str
    c1: float
    c2: float
    w_center: float
    P: float
    D: float
    k: float
    nu: float

    def deflection(self, r):
        rr = np.atleast_1d(np.asarray(r, dtype=float))
        w_p = winkler_plate_point_load(rr, self.P, self.D, self.k)
        q = np.exp(1j * np.pi / 4) / self.l
        G = iv(0, rr * q)
        return w_p + self.c1 * G.real + self.c2 * G.imag


def finite_plate_solution(R: float, P: float, D: float, k: float, nu: float,
                          bc: str = "clamped") -> FinitePlateSolution:
    """Exact centre-loaded Winkler plate of finite radius ``R`` with a rigid edge.

    ``bc = 'clamped'`` sets ``w(R) = w'(R) = 0``; ``bc = 'simply_supported'``
    sets ``w(R) = 0`` and ``M_r(R) = 0``.  The solution is

    ``W(r) = -P kei(r/l)/(2 pi k l^2) + c1 Re I_0(q r) + c2 Im I_0(q r)``,
    ``q = exp(i pi/4)/l``,

    with ``(c1, c2)`` from the two edge conditions.  As ``R -> inf`` both
    constants vanish and the infinite-plate Kelvin solution is recovered.
    """
    if bc not in ("clamped", "simply_supported"):
        raise ValueError("bc must be clamped or simply_supported")
    l = winkler_plate_char_length(D, k)
    q = np.exp(1j * np.pi / 4) / l

    def parts(r):
        k0, d1, d2 = kelvin_k0_derivatives(np.array([r]), l)
        scale = -P / (2.0 * np.pi * k * l ** 2)
        wp = scale * k0.imag[0]
        dwp = scale * d1.imag[0]
        d2wp = scale * d2.imag[0]
        if r == 0.0:
            G0 = 1.0 + 0.0j
            G1 = 0.0 + 0.0j
            G2 = q * q * 0.5          # limit of I_1(z)/z as z -> 0
        else:
            z = r * q
            G0 = iv(0, z)
            G1 = q * iv(1, z)
            G2 = q * q * (iv(0, z) - iv(1, z) / z)
        return (wp, dwp, d2wp), (G0, G1, G2)

    (wp, dwp, d2wp), (G0, G1, G2) = parts(R)
    # row 1: W(R) = 0
    A = np.array([[G0.real, G0.imag]])
    b = np.array([-wp])
    if bc == "clamped":
        A = np.vstack([A, [G1.real, G1.imag]])
        b = np.append(b, -dwp)
    else:
        # M_r = -D (W'' + nu W'/R) = 0
        A = np.vstack([A, [G2.real + nu * G1.real / R,
                           G2.imag + nu * G1.imag / R]])
        b = np.append(b, -(d2wp + nu * dwp / R))
    c1, c2 = np.linalg.solve(A, b)
    (wp0, _d, _dd), (G00, _g, _gg) = parts(0.0)
    w_center = wp0 + c1 * G00.real + c2 * G00.imag
    return FinitePlateSolution(R=R, l=l, bc=bc, c1=float(c1), c2=float(c2),
                               w_center=float(w_center), P=P, D=D, k=k, nu=nu)


def finite_plate_center_deflection(R: float, P: float, D: float, k: float,
                                   nu: float, bc: str = "clamped") -> float:
    """Centre deflection of a finite-radius, centre-loaded Winkler plate."""
    return finite_plate_solution(R, P, D, k, nu, bc).w_center