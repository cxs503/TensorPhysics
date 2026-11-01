"""Infinite plate on a fluid (Winkler) foundation: Hetenyi series vs. discrete solve.

Static problem
--------------
For a Kirchhoff plate resting on a buoyant (Winkler) foundation of modulus
``k_f = rho_w g`` the equilibrium equation is

    D lap^4 w + rho_w g w = p(x, y).

For a concentrated downward load ``P`` the closed-form solution (Hertenyi 1946,
 1966) follows from the Hankel transform

    w(r) = P / (2 pi k_f l_c^2) * kei(r / l_c),      l_c = (D / k_f)^(1/4),

with ``kei`` the Kelvin function of zero order (imaginary part).  Since
``kei(0) = -pi/4`` this reproduces the classical centre deflection
``w_max = P l_c^2 / (8 D)``.  Sign convention here: ``w > 0`` means deflection
*downwards* (in the direction of the load).

The numerical channel :func:`solve_biharmonic_fft` solves the same problem on a
periodic grid with the discrete biharmonic operator.  The classical 13-point
stencil is exactly ``(lap_h)^2`` with the 5-point Laplacian, so it is
diagonalised by the discrete Fourier transform with symbol

    S(k) = (2 cos(k_x dx) + 2 cos(k_y dy) - 4)^2 / dx^4   ->   k^4 as dx -> 0.

Agreement with the Kelvin-function solution therefore measures the spatial
discretisation error of the plate operator (verification, not validation).
"""

from __future__ import annotations

import numpy as np
from scipy.special import kei

from .dispersion import GRAVITY, RHO_WATER, characteristic_length

__all__ = [
    "hetenyi_point_load",
    "hetenyi_profile",
    "hetenyi_center_deflection",
    "solve_biharmonic_fft",
    "biharmonic_symbol",
    "hankel_gauss_deflection",
    "hankel_gauss_stress_r",
]


def hetenyi_point_load(
    P: float,
    D: float,
    r,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> np.ndarray:
    """Hertenyi closed-form deflection of an infinite plate under a point load.

    ``w(r)`` is positive downwards (load direction).  ``P`` [N], ``D`` [N m].
    """
    if D <= 0.0:
        raise ValueError("D must be positive")
    lc = characteristic_length(D, rho_w, g)
    rr = np.asarray(r, dtype=float)
    scale = P / (2.0 * np.pi * rho_w * g * lc**2)
    out = np.empty_like(rr)
    pos = rr > 0.0
    out[pos] = -scale * kei(rr[pos] / lc)
    if np.any(~pos):
        # kei(0) = -pi / 4  =>  w(0) = P l^2 / (8 D)
        out[~pos] = scale * np.pi / 4.0
    return out if out.ndim else float(out)


def hetenyi_profile(
    r,
    P: float,
    D: float,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> np.ndarray:
    """Alias of :func:`hetenyi_point_load` with a radius-first signature."""
    return hetenyi_point_load(P, D, r, rho_w, g)


def hetenyi_center_deflection(
    P: float, D: float, rho_w: float = RHO_WATER, g: float = GRAVITY
) -> float:
    """``w(0) = P l_c^2 / (8 D)`` [m], positive downwards."""
    lc = characteristic_length(D, rho_w, g)
    return P * lc**2 / (8.0 * D)


def biharmonic_symbol(nx: int, ny: int, dx: float, dy: float | None = None,
                      kind: str = "finite_difference") -> np.ndarray:
    """Fourier symbol of the biharmonic operator used by the static solver."""
    dy = dx if dy is None else dy
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    if kind == "spectral":
        return (KX**2 + KY**2) ** 2
    if kind != "finite_difference":
        raise ValueError(f"unknown symbol kind {kind!r}")
    lap_h = (2.0 * np.cos(KX * dx) + 2.0 * np.cos(KY * dy) - 4.0) / dx**2
    return lap_h**2


def solve_biharmonic_fft(
    load: np.ndarray,
    dx: float,
    D: float,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
    dy: float | None = None,
    kind: str = "finite_difference",
    zero_mean: bool = True,
) -> np.ndarray:
    """Solve ``D lap^4 w + rho_w g w = -p`` on a periodic grid by FFT.

    ``load`` is the downward pressure [Pa] (positive downwards, e.g. a
    discretised point load ``P / (dx dy)`` in a single cell).  Returns the
    downward deflection field [m].  ``zero_mean`` removes the k = 0 mode, which
    carries no physical information for an infinite plate (see the module
    docstring of :mod:`tensorphysics.ice.dispersion`).
    """
    load = np.asarray(load, dtype=float)
    if load.ndim != 2:
        raise ValueError("load must be a 2-D array")
    ny, nx = load.shape
    S = biharmonic_symbol(nx, ny, dx, dy, kind)
    # Sign: with w positive downwards, D lap^4 w + k_f w = p (downward load).
    denom = D * S + rho_w * g
    P_hat = np.fft.fft2(load)
    if zero_mean:
        P_hat[0, 0] = 0.0
    W_hat = P_hat / denom
    W_hat[0, 0] = 0.0
    return np.real(np.fft.ifft2(W_hat))

def hankel_gauss_deflection(
    r,
    P: float,
    sigma: float,
    D: float,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
    u_max: float = 25.0,
    n_nodes: int = 20001,
) -> np.ndarray:
    """Semi-analytic deflection of an infinite plate under a Gaussian patch load.

    The Gaussian load ``p(r) = P/(2 pi sigma^2) exp(-r^2/(2 sigma^2))`` has the
    Hankel transform ``P exp(-kappa^2 sigma^2/2)``, so with ``kappa = u / sigma``

        w(r) = P sigma^2/(2 pi) int_0^inf u J_0(u r/sigma) exp(-u^2/2)
               / (D u^4 + rho_w g sigma^4) d u,

    which is well conditioned (the Gaussian factor bounds the range).  This is an
    independent, grid-free reference for the discrete static solver.
    """
    from scipy.integrate import simpson
    from scipy.special import j0

    if D <= 0.0:
        raise ValueError("D must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    kf_s4 = rho_w * g * sigma**4
    pref = P * sigma**2 / (2.0 * np.pi)
    # composite Simpson on a fine grid: the integrand is smooth and Gaussian
    # damped but oscillatory (J_0(u r/sigma)), where adaptive quad is unreliable;
    # ~1e3 nodes per oscillation give ~1e-10 relative quadrature error.
    n = int(n_nodes) if n_nodes else 20001
    n += (n + 1) % 2
    u = np.linspace(0.0, u_max, n)

    def integrand(rr: float) -> np.ndarray:
        return u * j0(u * rr / sigma) * np.exp(-0.5 * u * u) / (D * u**4 + kf_s4)

    rr = np.atleast_1d(np.asarray(r, dtype=float))
    out = pref * np.array([simpson(integrand(float(rt)), x=u) for rt in rr])
    return out if out.size > 1 else float(out[0])


def hankel_gauss_stress_r(
    r,
    P: float,
    sigma: float,
    D: float,
    E: float,
    h: float,
    nu: float = 0.3,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> dict:
    """Radial/circumferential *bottom* stress of the Gaussian patch solution.

    Derivatives of the Hankel integral are taken by central differences, giving a
    grid-free reference for the spectral stress recovery:

        sigma_r = -C (w_rr + nu w_r / r),   sigma_t = -C (w_r/r + nu w_rr),
        C = E h / (2 (1 - nu^2)),           w > 0 downwards (tension positive).
    """
    C = E * h / (2.0 * (1.0 - nu**2))
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    dr = max(2.0e-3 * sigma, 1.0e-4)
    w0 = np.atleast_1d(hankel_gauss_deflection(rr, P, sigma, D, rho_w, g))
    wp = np.atleast_1d(hankel_gauss_deflection(rr + dr, P, sigma, D, rho_w, g))
    wm = np.atleast_1d(hankel_gauss_deflection(np.maximum(rr - dr, 0.0), P, sigma, D, rho_w, g))
    w_r = (wp - wm) / (2.0 * dr)
    w_rr = (wp - 2.0 * w0 + wm) / dr**2
    safe_r = np.where(rr > 0.0, rr, np.inf)
    sigma_r = -C * (w_rr + nu * w_r / safe_r)
    sigma_t = -C * (w_r / safe_r + nu * w_rr)
    at_zero = rr <= 0.0
    if np.any(at_zero):
        w2 = np.atleast_1d(hankel_gauss_deflection(np.array([2.0 * dr]), P, sigma, D, rho_w, g))[0]
        curv0 = (w2 - 2.0 * w0[at_zero][0] + w0[at_zero][0]) / (2.0 * dr) ** 2
        sigma_r[at_zero] = -C * (1.0 + nu) * curv0
        sigma_t[at_zero] = -C * (1.0 + nu) * curv0
    return {"sigma_r_Pa": sigma_r, "sigma_t_Pa": sigma_t, "w_m": w0}
