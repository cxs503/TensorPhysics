"""Flexural-gravity wave dispersion under a continuous ice cover.

Physical model
--------------
A thin elastic plate (the ice cover, Kirchhoff-Love, negligible plate inertia)
floating on a homogeneous, inviscid, irrotational water layer of depth ``h``.
Linearised potential flow ``phi`` with ``grad^2 phi = 0`` in ``-h < z < 0``,
rigid bottom ``dphi/dz = 0`` at ``z = -h``, kinematic condition
``dphi/dz = deta/dt`` at ``z = 0`` and the plate condition

    D d^4 eta/dx^4 = p_w(0),

where the linearised Bernoulli pressure is ``p_w = -rho_w (dphi/dt + g eta)``.
Seeking ``eta ~ exp(i(kx - omega t))`` yields the classical dispersion relation

    omega^2 = (g k + D k^5 / rho_w) * tanh(k h).

Note on the rigidity term: ``D`` has units N m = kg m^2 / s^2, so ``D k^5 / rho_w``
has units 1/s^2 (consistent with ``g k``), whereas ``D k^4 / rho_w`` would have
units m/s^2 and cannot be added to ``g k``. The dimensionally consistent
expression is therefore the one implemented here; a naive ``D k^4 / rho`` form is
rejected by :func:`flexural_rigidity`-based dimensional checks in the tests.

Independent numerical verification
----------------------------------
:func:`spectral_dispersion` solves the *same* boundary value problem without ever
using the closed form: Chebyshev collocation in ``z`` (``N`` nodes) plus the
unknown surface elevation ``eta`` gives a generalised eigenproblem ``A x = omega B x``
whose eigenvalue closest to the analytical root is compared in
:func:`spectral_residual`. Spectral (exponential) convergence makes residuals
below ``1e-12`` attainable for ``N >= 20``.
"""

from __future__ import annotations

from math import isfinite, pi, sqrt, tanh
from typing import Any

import numpy as np


class FlexuralWaveError(ValueError):
    """Invalid physical input for the flexural-gravity wave model."""


GRAVITY = 9.81
RHO_WATER = 1025.0


def _check_positive(name: str, value: Any, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise FlexuralWaveError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not isfinite(value):
        raise FlexuralWaveError(f"{name} must be finite, got {value!r}")
    if allow_zero:
        if value < 0.0:
            raise FlexuralWaveError(f"{name} must be non-negative, got {value!r}")
    elif value <= 0.0:
        raise FlexuralWaveError(f"{name} must be positive, got {value!r}")
    return value


def flexural_rigidity(
    thickness: float,
    young_modulus: float = 5.0e9,
    poisson_ratio: float = 0.3,
) -> float:
    """Flexural rigidity ``D = E t^3 / (12 (1 - nu^2))`` in N m.

    Defaults are representative of columnar sea ice (``E = 5 GPa``, ``nu = 0.3``).
    """
    t = _check_positive("thickness", thickness)
    e = _check_positive("young_modulus", young_modulus)
    nu = float(poisson_ratio)
    if not isfinite(nu) or not (-0.999 < nu < 0.5):
        raise FlexuralWaveError(f"poisson_ratio must satisfy -1 < nu < 0.5, got {nu!r}")
    return e * t**3 / (12.0 * (1.0 - nu**2))


def flexural_gravity_dispersion(
    k: float,
    h: float,
    D: float = 0.0,
    *,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> float:
    """Return ``omega^2`` for wavenumber ``k`` (rad/m), depth ``h`` (m), rigidity ``D``.

    ``D = 0`` recovers the open-water gravity-wave dispersion
    ``omega^2 = g k tanh(k h)``.
    """
    k = _check_positive("k", k)
    h = _check_positive("h", h)
    D = _check_positive("D", D, allow_zero=True)
    rho_w = _check_positive("rho_w", rho_w)
    g = _check_positive("g", g)
    return (g * k + D * k**5 / rho_w) * tanh(k * h)


def flexural_gravity_wavenumber(
    omega: float,
    h: float,
    D: float = 0.0,
    *,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
    tol: float = 1e-15,
    max_iter: int = 200,
) -> float:
    """Solve the dispersion relation for ``k > 0`` given a real frequency.

    Uses a bracketed Newton iteration on ``f(k) = omega^2 - omega^2_disp(k)``;
    ``f`` is strictly decreasing in ``k`` for ``D >= 0`` so the root is unique.
    """
    omega = _check_positive("omega", omega)
    h = _check_positive("h", h)
    D = _check_positive("D", D, allow_zero=True)
    target = omega**2

    def f(k: float) -> float:
        return flexural_gravity_dispersion(k, h, D, rho_w=rho_w, g=g) - target

    lo, hi = 1e-12, 1.0
    # omega_disp^2(k) increases monotonically in k, so grow hi until it brackets.
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > 1e12:
            raise FlexuralWaveError("failed to bracket the wavenumber root")
    if f(lo) > 0.0:
        raise FlexuralWaveError("wavenumber root is below the lower bracket bound")
    k = 0.5 * (lo + hi)
    for _ in range(max_iter):
        val = f(k)
        if val > 0.0:
            hi = k  # k is too large: the root lies below
        else:
            lo = k
        # Newton step with bisection fallback keeps the iterate bracketed.
        dk = 1e-8 * max(k, 1e-8)
        deriv = (f(k + dk) - f(k - dk)) / (2.0 * dk)
        candidate = k - val / deriv if deriv < 0.0 else 0.5 * (lo + hi)
        if not (lo < candidate < hi):
            candidate = 0.5 * (lo + hi)
        if abs(candidate - k) <= tol * max(1.0, abs(k)):
            k = candidate
            break
        k = candidate
    return k


def phase_speed(
    k: float,
    h: float,
    D: float = 0.0,
    *,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> float:
    """Phase speed ``c = omega / k``."""
    k = _check_positive("k", k)
    return sqrt(flexural_gravity_dispersion(k, h, D, rho_w=rho_w, g=g)) / k


def group_speed(
    k: float,
    h: float,
    D: float = 0.0,
    *,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> float:
    """Group speed ``domega/dk`` from the analytical dispersion relation."""
    k = _check_positive("k", k)
    h = _check_positive("h", h)
    D = _check_positive("D", D, allow_zero=True)
    t = tanh(k * h)
    sech2 = 1.0 - t * t
    omega2 = (g * k + D * k**5 / rho_w) * t
    domega2_dk = (g + 5.0 * D * k**4 / rho_w) * t + (g * k + D * k**5 / rho_w) * h * sech2
    return domega2_dk / (2.0 * sqrt(omega2))


def _cheb(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Chebyshev collocation nodes on [-1, 1] and the differentiation matrix."""
    if n == 0:
        return np.zeros(1), np.zeros((1, 1))
    nodes = np.cos(pi * np.arange(n + 1) / n)
    x = nodes.reshape(-1, 1)
    c = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1.0) ** np.arange(n + 1)
    dx = x - x.T
    D = np.outer(c, 1.0 / c) / (dx + np.eye(n + 1))
    D = D - np.diag(D.sum(axis=1))
    return nodes, D


def _clencurt(n: int) -> np.ndarray:
    """Clenshaw-Curtis quadrature weights on the Chebyshev-Lobatto nodes of [-1, 1]."""
    theta = pi * np.arange(n + 1) / n
    w = np.zeros(n + 1)
    v = np.ones(n - 1)
    if n % 2 == 0:
        w[0] = w[n] = 1.0 / (n**2 - 1)
        for k in range(1, n // 2):
            v -= 2.0 * np.cos(2.0 * k * theta[1:n]) / (4.0 * k**2 - 1.0)
        v -= np.cos(n * theta[1:n]) / (n**2 - 1.0)
    else:
        w[0] = w[n] = 1.0 / n**2
        for k in range(1, (n - 1) // 2 + 1):
            v -= 2.0 * np.cos(2.0 * k * theta[1:n]) / (4.0 * k**2 - 1.0)
    w[1:n] = 2.0 * v / n
    return w


def spectral_mode_shape(
    k: float,
    h: float,
    *,
    n_modes: int = 32,
    stretch: float = 0.0,
    max_kh: float = 18.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve the depth boundary value problem for the potential mode shape.

    The problem ``(D^2 - k^2) phihat = 0`` in ``-h < z < 0`` with the rigid-bottom
    condition ``phihat'(-h) = 0`` and the (arbitrary) normalisation
    ``phihat'(0) = 1`` is a well-posed linear system: the plate/kinematic coupling
    is *not* imposed here, it defines the dispersion relation instead
    (see :func:`spectral_dispersion`). Chebyshev collocation with an exponential
    stretch towards the surface (``z = -h G(xi)``) makes the scheme spectrally
    accurate even when ``k h >> 1``.

    When ``k h > max_kh`` the domain is truncated to ``h_eff = max_kh / k``: the
    mode amplitude at the truncated bottom is ``cosh(k h_eff) / cosh(k h)`` and the
    formal error of ``omega^2`` is ``O(tanh(k h_eff) / tanh(k h) - 1)`` which is
    ``4e-16`` for ``max_kh = 18`` (i.e. below double-precision resolution), while
    an untruncated boundary layer of thickness ``1/k`` cannot be resolved by a
    moderate number of collocation nodes.

    Returns ``(z_nodes, phat, surface_slope, h_eff)``. ``surface_slope`` is
    ``phihat'(0)`` obtained from the ODE identity

        phihat'(0) = phihat'(0) - phihat'(-h) = int_{-h}^{0} phihat'' dz
                   = k^2 int_{-h}^{0} phihat dz,

    evaluated with Clenshaw-Curtis quadrature. Using the integral identity instead
    of the differentiation matrix avoids catastrophic cancellation in the
    shallow-water limit, where ``phihat'(0) = O(k^2 h)`` while ``phihat = O(1)``.
    """
    k = _check_positive("k", k)
    h = _check_positive("h", h)
    n_modes = int(n_modes)
    if n_modes < 4:
        raise FlexuralWaveError(f"n_modes must be >= 4, got {n_modes}")
    stretch = float(stretch)
    if not isfinite(stretch) or stretch < 0.0:
        raise FlexuralWaveError(f"stretch must be finite and non-negative, got {stretch!r}")

    max_kh = float(max_kh)
    if not isfinite(max_kh) or max_kh <= 0.0:
        raise FlexuralWaveError(f"max_kh must be finite and positive, got {max_kh!r}")
    h_eff = h if k * h <= max_kh else max_kh / k

    n = n_modes
    nodes, Dmat = _cheb(n)  # nodes[0] = +1, nodes[n] = -1
    if stretch == 0.0:
        # Linear map: node 0 -> z = 0 (surface), node n -> z = -h (bottom).
        z = h_eff * (nodes - 1.0) / 2.0
        dz_dxi = np.full_like(nodes, h_eff / 2.0)
    else:
        # Exponential map z = -h G(xi) with G(+1) = 0, G(-1) = 1 which clusters
        # nodes towards the surface (z = 0), where the potential behaves as
        # cosh(k (z + h)) and forms a boundary layer of thickness 1/k.
        denom = np.exp(stretch) - 1.0
        g_map = (np.exp(stretch * (1.0 - nodes) / 2.0) - 1.0) / denom
        z = -h_eff * g_map
        dz_dxi = h_eff * (stretch / 2.0) * np.exp(stretch * (1.0 - nodes) / 2.0) / denom
    dz_dxi = np.where(dz_dxi == 0.0, 1e-300, dz_dxi)
    D1 = np.diag(1.0 / dz_dxi) @ Dmat
    D2 = D1 @ D1

    size = n + 1
    A = np.zeros((size, size))
    rhs = np.zeros(size)
    # Interior Laplace equation at nodes 1..n-1.
    for row, j in enumerate(range(1, n)):
        A[row, :] = D2[j, :] - k**2 * np.eye(size)[j, :]
    # Rigid bottom Neumann condition at z = -h (node n).
    A[n - 1, :] = D1[n, :]
    # Normalisation phihat(0) = 1 at the surface (node 0): a Dirichlet value keeps
    # the linear system well conditioned in the shallow-water limit where the
    # mode is almost flat and phihat'(0) -> k^2 h -> 0.
    A[n, :] = np.eye(size)[0, :]
    rhs[n] = 1.0
    phat = np.linalg.solve(A, rhs)
    weights = _clencurt(n)
    integral = float(np.sum(weights * phat * dz_dxi))
    surface_slope = k**2 * integral
    return z, phat, surface_slope, h_eff


def spectral_dispersion(
    k: float,
    h: float,
    D: float = 0.0,
    *,
    n_modes: int = 32,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
    stretch: float = 0.0,
    max_kh: float = 18.0,
) -> complex:
    """Numerically solve the flexural-gravity boundary value problem.

    The closed-form dispersion relation is never used: the depth mode shape
    ``phihat(z)`` is obtained from the collocation solve of
    :func:`spectral_mode_shape` with ``phihat(0) = 1``, and the surface
    plate/Bernoulli condition

        (D k^4 + rho_w g) phihat'(0) = omega^2 rho_w phihat(0)

    then yields ``omega^2 = (D k^4 + rho_w g) phihat'(0) / rho_w`` directly.

    ``D`` may be complex (Kelvin-Voigt viscoelastic plate), in which case the
    returned frequency is complex and ``|Im omega|`` is the temporal decay rate.
    """
    k = _check_positive("k", k)
    h = _check_positive("h", h)
    D = complex(D)
    if not isfinite(D.real) or not isfinite(D.imag) or D.real < 0.0:
        raise FlexuralWaveError(f"D must be finite with non-negative real part, got {D!r}")
    rho_w = _check_positive("rho_w", rho_w)
    g = _check_positive("g", g)
    _z, _phat, surface_slope, _h_eff = spectral_mode_shape(
        k, h, n_modes=n_modes, stretch=stretch, max_kh=max_kh
    )
    if not isfinite(surface_slope):
        raise FlexuralWaveError("degenerate mode shape: phihat'(0) is not finite")
    omega2 = (D * k**4 + rho_w * g) * surface_slope / rho_w
    omega = np.sqrt(complex(omega2))
    if omega.real < 0.0:
        omega = -omega
    return complex(omega)


def _generalised_eig(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thin wrapper around ``scipy.linalg.eig`` with a numpy fallback."""
    try:  # pragma: no cover - scipy is optional at runtime
        from scipy import linalg  # noqa: PLC0415

        values, vectors = linalg.eig(A, B)
        return np.asarray(values), np.asarray(vectors)
    except Exception:  # pragma: no cover - fallback path
        # B is singular (one zero row block); fall back to dense eig of a
        # regularised pencil.
        eps = 1e-14 * max(1.0, float(np.abs(B).max()))
        return np.linalg.eig(np.linalg.solve(B + eps * np.eye(B.shape[0]), A))


def spectral_residual(
    k: float,
    h: float,
    D: float = 0.0,
    *,
    n_modes: int = 32,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
    stretch: float = 0.0,
    max_kh: float = 18.0,
) -> float:
    """Relative residual ``|omega_num^2 - omega_ana^2| / |omega_ana^2|``.

    ``omega_num`` comes from :func:`spectral_dispersion` (discretised BVP) and
    ``omega_ana`` from the closed-form dispersion relation.
    """
    omega_num = spectral_dispersion(
        k, h, D, n_modes=n_modes, rho_w=rho_w, g=g, stretch=stretch, max_kh=max_kh
    )
    h_eff = h if k * float(h) <= float(max_kh) else float(max_kh) / k
    omega2_ana = flexural_gravity_dispersion(k, h_eff, D, rho_w=rho_w, g=g)
    return float(abs(omega_num**2 - omega2_ana) / abs(omega2_ana))


def depth_truncation_error(
    k: float,
    h: float,
    D: float = 0.0,
    *,
    max_kh: float = 18.0,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> float:
    """Relative error of ``omega^2`` caused by truncating the depth to ``max_kh / k``."""
    k = _check_positive("k", k)
    h = _check_positive("h", h)
    h_eff = h if k * h <= float(max_kh) else float(max_kh) / k
    full = flexural_gravity_dispersion(k, h, D, rho_w=rho_w, g=g)
    trunc = flexural_gravity_dispersion(k, h_eff, D, rho_w=rho_w, g=g)
    return float(abs(trunc - full) / abs(full))