"""Two-level analysis (TLA) of the geometric multigrid V-cycle.

For the constant-coefficient Poisson model problem and a *diagonal-symbol*
smoother (damped Jacobi), local Fourier analysis (LFA) gives the exact
spectrum of the two-grid operator acting on the space spanned by the four fine
harmonics that alias to one coarse harmonic.

Harmonics
---------
With standard coarsening (``H = 2h``) the four fine frequencies

    omega_mn = theta + (m*pi, n*pi),   m, n in {0, 1},   theta in [0, pi/2)^2

alias onto the single coarse frequency ``2*theta``.  In this basis the
two-grid operator ``M = S^{nu2} (I - P A_c^{-1} R A) S^{nu1}`` has the 4x4
symbol matrix

    M_ij(theta) = s_i^{nu2} [ delta_ij - 4 P_i(theta) r_j a_j / a_c ] s_j^{nu1}

with

    a_j    = 4 - 2 cos(omega_jx) - 2 cos(omega_jy)          (fine 5-point symbol)
    a_c    = 4 - 2 cos(2 theta_x) - 2 cos(2 theta_y)        (coarse symbol)
    r_j    = (1 + cos omega_jx)(1 + cos omega_jy) / 4       (full weighting)
    P_i    = prod over d of  px_{m_i}(theta_d)              (bilinear prolongation)
             px_0 = (1 + cos theta_d)/2,  px_1 = (1 - cos theta_d)/2
    s_j    = 1 - omega_J a_j / 4                            (damped Jacobi)

The factor 4 in front of ``P_i r_j a_j / a_c`` comes from the grid scaling
``A_c^{-1} = 4 h^2 / a_c`` and ``A = a/h^2``.  The formula is validated against
the explicitly assembled periodic two-grid operator in
:func:`periodic_two_grid_spectral_radius` (agreement ~1e-14, see
``tests/test_multigrid_poisson.py``).

Because ``M`` is ``diag(d) - u v^T`` (a rank-one update of a diagonal matrix),
its eigenvalues are the roots of the explicit quartic

    prod_j (lambda - d_j) - sum_j u_j v_j prod_{k != j} (lambda - d_k) = 0,

which is the closed form used in :func:`tla_characteristic_coefficients`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "harmonic_set",
    "fine_symbol",
    "coarse_symbol",
    "restriction_symbol",
    "prolongation_weights",
    "jacobi_symbol",
    "tla_symbol_matrix",
    "tla_spectral_radius",
    "tla_characteristic_coefficients",
    "tla_closed_form_radius",
    "tla_two_grid_factor",
    "periodic_two_grid_spectral_radius",
    "periodic_lfa_max",
    "smoothing_factor",
]


# ---------------------------------------------------------------------------
# symbols
# ---------------------------------------------------------------------------
def harmonic_set(theta: Sequence[float]) -> List[Tuple[float, float]]:
    """The four fine frequencies aliasing to the coarse frequency ``2*theta``."""
    tx, ty = float(theta[0]), float(theta[1])
    return [
        (tx, ty),
        (tx + np.pi, ty),
        (tx, ty + np.pi),
        (tx + np.pi, ty + np.pi),
    ]


def fine_symbol(omega: Sequence[float]) -> float:
    """Dimensionless 5-point symbol ``a(omega) = 4 - 2cos wx - 2cos wy``."""
    return 4.0 - 2.0 * np.cos(omega[0]) - 2.0 * np.cos(omega[1])


def coarse_symbol(theta: Sequence[float]) -> float:
    """Coarse-grid 5-point symbol at coarse frequency ``2*theta``."""
    return 4.0 - 2.0 * np.cos(2.0 * theta[0]) - 2.0 * np.cos(2.0 * theta[1])


def restriction_symbol(omega: Sequence[float]) -> float:
    """Full-weighting restriction symbol ``(1+cos wx)(1+cos wy)/4``."""
    return 0.25 * (1.0 + np.cos(omega[0])) * (1.0 + np.cos(omega[1]))


def prolongation_weights(theta: Sequence[float]) -> List[float]:
    """Bilinear prolongation weights on the four fine harmonics.

    Order matches :func:`harmonic_set`: ``[(0,0), (1,0), (0,1), (1,1)]``.
    """
    px = (0.5 * (1.0 + np.cos(theta[0])), 0.5 * (1.0 - np.cos(theta[0])))
    py = (0.5 * (1.0 + np.cos(theta[1])), 0.5 * (1.0 - np.cos(theta[1])))
    # order matches harmonic_set: (m, n) = (0,0), (1,0), (0,1), (1,1)
    return [px[m] * py[n] for n in (0, 1) for m in (0, 1)]


def jacobi_symbol(omega: Sequence[float], omega_j: float) -> float:
    """Damped-Jacobi smoothing symbol ``s = 1 - omega_J a(omega)/4``."""
    return 1.0 - 0.25 * omega_j * fine_symbol(omega)


# ---------------------------------------------------------------------------
# two-grid symbol matrix
# ---------------------------------------------------------------------------
def _symbol_data(theta: Sequence[float], omega_j: float):
    """Symbols at the four harmonics, with the ``theta -> 0`` limit handled.

    As ``theta -> 0`` the coarse symbol ``a_c`` vanishes and the ratios
    ``r_j a_j / a_c`` acquire finite limits; a vanishing denominator is
    therefore replaced by a ``1e-12`` relative offset of ``theta``.
    """
    th = np.asarray(theta, dtype=float)
    if coarse_symbol(th) < 1e-8:
        # the theta -> 0 limit is direction dependent, but rho(M(theta)) -> 0
        # there (the smooth component is corrected exactly), so the value at a
        # small offset is representative and no maximum is lost.
        th = np.asarray(th, dtype=float) + 1e-3
    omegas = harmonic_set(th)
    a = np.array([fine_symbol(w) for w in omegas])
    r = np.array([restriction_symbol(w) for w in omegas])
    s = np.array([jacobi_symbol(w, omega_j) for w in omegas])
    p = np.array(prolongation_weights(th))
    a_c = coarse_symbol(th)
    return th, a, r, s, p, a_c


def tla_symbol_matrix(
    theta: Sequence[float], nu1: int = 1, nu2: int = 1, omega_j: float = 2.0 / 3.0
) -> np.ndarray:
    """4x4 symbol matrix of the two-grid operator at low frequency ``theta``."""
    _, a, r, s, p, a_c = _symbol_data(theta, omega_j)
    M = np.diag(s ** (nu1 + nu2))
    M -= np.outer(s**nu2 * 4.0 * p, r * a * s**nu1) / a_c
    return M


def tla_spectral_radius(
    theta: Sequence[float], nu1: int = 1, nu2: int = 1, omega_j: float = 2.0 / 3.0
) -> float:
    """Spectral radius of the 4x4 symbol matrix at ``theta`` (eigen-solve)."""
    M = tla_symbol_matrix(theta, nu1, nu2, omega_j)
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def tla_characteristic_coefficients(
    theta: Sequence[float], nu1: int = 1, nu2: int = 1, omega_j: float = 2.0 / 3.0
) -> np.ndarray:
    """Coefficients ``[c4, c3, c2, c1, c0]`` of the closed-form quartic.

    ``M = diag(d) - u v^T`` (rank-one update of a diagonal matrix) with
    ``d_i = s_i^{nu1+nu2}``, ``u_i = 4 s_i^{nu2} P_i``,
    ``v_j = r_j a_j s_j^{nu1} / a_c``.  Its characteristic polynomial is

        det(lambda I - M) = prod_j (lambda - d_j)
                            + sum_j u_j v_j prod_{k != j} (lambda - d_k),
    """
    _, a, r, s, p, a_c = _symbol_data(theta, omega_j)
    d = s ** (nu1 + nu2)
    u = 4.0 * s**nu2 * p
    v = r * a * s**nu1 / a_c
    w = u * v
    coeffs = np.polynomial.polynomial.polyfromroots(d)  # ascending, monic
    corr = np.zeros(5)
    for j in range(4):
        pj = np.polynomial.polynomial.polyfromroots(np.delete(d, j))
        corr[:4] += w[j] * pj
    coeffs = coeffs + corr
    return coeffs[::-1]


def tla_closed_form_radius(
    theta: Sequence[float], nu1: int = 1, nu2: int = 1, omega_j: float = 2.0 / 3.0
) -> float:
    """Largest-magnitude root of the closed-form quartic (analytical prediction)."""
    c = tla_characteristic_coefficients(theta, nu1, nu2, omega_j)
    roots = np.polynomial.polynomial.polyroots(c[::-1])
    return float(np.max(np.abs(roots)))


def tla_two_grid_factor(
    nu1: int = 1,
    nu2: int = 1,
    omega_j: float = 2.0 / 3.0,
    n_theta: int = 64,
    method: str = "eig",
) -> Dict[str, object]:
    """Two-grid convergence factor ``rho_TLA = max_theta rho(M(theta))``.

    ``theta`` is sampled over ``[0, pi/2)^2`` (the low-frequency half of the
    fine frequency range); the maximum of the symbol spectral radius is the
    LFA prediction of the two-grid convergence factor.
    """
    thetas = np.linspace(0.0, 0.5 * np.pi, n_theta, endpoint=False)
    best = 0.0
    best_theta = (0.0, 0.0)
    fn = tla_spectral_radius if method == "eig" else tla_closed_form_radius
    for tx in thetas:
        for ty in thetas:
            r = fn((tx, ty), nu1, nu2, omega_j)
            if r > best:
                best, best_theta = r, (float(tx), float(ty))
    return {
        "rho_tla": float(best),
        "theta_max": best_theta,
        "nu1": nu1,
        "nu2": nu2,
        "omega_j": omega_j,
        "method": method,
    }


# ---------------------------------------------------------------------------
# independent cross-check: explicitly assembled periodic two-grid operator
# ---------------------------------------------------------------------------
def _periodic_laplacian(n: int, h: float) -> np.ndarray:
    m = n * n
    A = np.zeros((m, m))
    idx = lambda i, j: (i % n) * n + (j % n)
    for i in range(n):
        for j in range(n):
            k = idx(i, j)
            A[k, k] = 4.0 / (h * h)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                A[k, idx(i + di, j + dj)] = -1.0 / (h * h)
    return A


def _periodic_restriction(n: int) -> np.ndarray:
    """Full weighting from an ``n x n`` periodic grid to ``n/2 x n/2``."""
    nc = n // 2
    R = np.zeros((nc * nc, n * n))
    idxf = lambda i, j: (i % n) * n + (j % n)
    idxc = lambda i, j: (i % nc) * nc + (j % nc)
    for i in range(nc):
        for j in range(nc):
            k = idxc(i, j)
            I, J = 2 * i, 2 * j
            R[k, idxf(I, J)] += 4.0 / 16.0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                R[k, idxf(I + di, J + dj)] += 2.0 / 16.0
            for di in (1, -1):
                for dj in (1, -1):
                    R[k, idxf(I + di, J + dj)] += 1.0 / 16.0
    return R


def _periodic_prolongation(n: int) -> np.ndarray:
    nc = n // 2
    P = np.zeros((n * n, nc * nc))
    idxf = lambda i, j: (i % n) * n + (j % n)
    idxc = lambda i, j: (i % nc) * nc + (j % nc)
    for i in range(nc):
        for j in range(nc):
            k = idxc(i, j)
            I, J = 2 * i, 2 * j
            P[idxf(I, J), k] += 1.0
            P[idxf(I + 1, J), k] += 0.5
            P[idxf(I - 1, J), k] += 0.5
            P[idxf(I, J + 1), k] += 0.5
            P[idxf(I, J - 1), k] += 0.5
            for di in (1, -1):
                for dj in (1, -1):
                    P[idxf(I + di, J + dj), k] += 0.25
    return P


def periodic_two_grid_spectral_radius(
    n: int = 16, nu1: int = 1, nu2: int = 1, omega_j: float = 2.0 / 3.0
) -> Dict[str, object]:
    """Spectral radius of an explicitly assembled periodic two-grid operator.

    Independent numerical reference for the LFA symbol formula: ``A``, ``A_c``,
    ``R`` and ``P`` are assembled as dense matrices on an ``n x n`` periodic
    grid and ``rho(S^{nu2}(I - P A_c^{-1} R A) S^{nu1})`` is obtained from a
    dense eigenvalue solve.  The constant mode (eigenvalue 1, undamped because
    ``A 1 = 0``) is projected out, matching the LFA convention that excludes
    the exactly-zero frequency.
    """
    h = 1.0 / n
    A = _periodic_laplacian(n, h)
    Ac = _periodic_laplacian(n // 2, 2.0 * h)
    R = _periodic_restriction(n)
    P = _periodic_prolongation(n)
    S = np.eye(n * n) - (omega_j / (4.0 / (h * h))) * A
    # the periodic operator is singular (constants in its null space) -> pinv
    C = np.eye(n * n) - P @ np.linalg.pinv(Ac) @ (R @ A)
    M = np.linalg.matrix_power(S, nu2) @ C @ np.linalg.matrix_power(S, nu1)
    # projection onto the orthogonal complement of the constant vector
    m = n * n
    e = np.ones(m) / np.sqrt(m)
    Mperp = M - np.outer(M @ e, e) - np.outer(e, e @ M) + np.outer(e, e) * (e @ M @ e)
    eigs = np.linalg.eigvals(Mperp)
    return {
        "rho": float(np.max(np.abs(eigs))),
        "rho_including_constant": float(np.max(np.abs(np.linalg.eigvals(M)))),
        "n": n,
        "matrix": M,
        "eigs": eigs,
    }


def periodic_lfa_max(
    n: int = 16, nu1: int = 1, nu2: int = 1, omega_j: float = 2.0 / 3.0
) -> Dict[str, object]:
    """LFA prediction restricted to the *discrete* harmonics of an ``n x n``
    periodic grid (so it can be compared one-to-one with
    :func:`periodic_two_grid_spectral_radius`)."""
    best, best_theta = 0.0, None
    for m in range(0, n // 4 + 1):
        for l in range(0, n // 4 + 1):
            theta = (2.0 * np.pi * m / n, 2.0 * np.pi * l / n)
            r = tla_spectral_radius(theta, nu1, nu2, omega_j)
            if r > best:
                best, best_theta = r, theta
    return {"rho_lfa": float(best), "theta_max": best_theta}


def smoothing_factor(omega_j: float = 2.0 / 3.0, n_theta: int = 200) -> float:
    """Smoothing factor: worst damping of high-frequency harmonics.

    ``mu = max_{omega in high} |1 - omega_J a(omega)/4|`` with the *high*
    range ``a in [4, 8]`` (2-D 5-point Laplacian, ``d = 2``).
    """
    vals = []
    for a in np.linspace(4.0, 8.0, n_theta):
        vals.append(abs(1.0 - 0.25 * omega_j * a))
    return float(max(vals))