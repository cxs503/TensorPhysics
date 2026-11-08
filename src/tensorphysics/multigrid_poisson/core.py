"""Geometric multigrid for the 2-D Poisson model problem.

Model problem
-------------
    -Laplacian u = f      on  Omega = (0,1)^2,   u = 0 on dOmega
    u*(x,y) = sin(pi x) sin(pi y),  f(x,y) = 2 pi^2 sin(pi x) sin(pi y)

Discretisation: vertex-centred 5-point finite differences on a uniform grid
with ``n`` cells per direction (``h = 1/n``), ``(n-1)^2`` interior unknowns.
The discrete operator is written as ``A u = (4 u_ij - sum(neighbours)) / h^2``,
i.e. ``A`` is the negative Laplacian.

Hierarchy: ``n_fine = 2^L`` coarsened by factor 2 down to ``n_coarse = 4``
(``h = 1/4``), so the number of levels is ``L - 1``.

Components
----------
* smoothers: damped Jacobi (diagonal, Fourier-symbol smooth, used for the
  two-level analysis) and red-black Gauss-Seidel (production smoother),
* transfer: full-weighting restriction (tensor product of [1/4, 1/2, 1/4])
  and bilinear prolongation (tensor product of [1/2, 1, 1/2]),
* cycles: recursive V-cycle and W-cycle, Dirichlet boundary values are held
  at zero on every level (homogeneous correction problems).

All routines are pure NumPy (no SciPy dependency in the solver path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "analytic_solution",
    "poisson_rhs",
    "discrete_operator",
    "residual",
    "restrict_full_weighting",
    "prolong_bilinear",
    "jacobi_sweeps",
    "red_black_gs_sweeps",
    "coarse_matrix",
    "dense_direct_solve",
    "MultigridSolver",
    "jacobi_solver",
    "convergence_history",
]


# ---------------------------------------------------------------------------
# model problem data
# ---------------------------------------------------------------------------
def analytic_solution(n: int) -> np.ndarray:
    """Exact solution ``u* = sin(pi x) sin(pi y)`` sampled on the grid."""
    x = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


def poisson_rhs(n: int) -> np.ndarray:
    """Right-hand side ``f = -Laplacian u* = 2 pi^2 sin(pi x) sin(pi y)``."""
    return 2.0 * np.pi**2 * analytic_solution(n)


# ---------------------------------------------------------------------------
# discrete operator / residual
# ---------------------------------------------------------------------------
def discrete_operator(u: np.ndarray, h: float) -> np.ndarray:
    """``A u = (4 u_ij - sum of 4 neighbours) / h^2`` on the interior.

    Boundary entries of the returned array are zero (``u = 0`` on dOmega).
    """
    au = np.zeros_like(u)
    au[1:-1, 1:-1] = (
        4.0 * u[1:-1, 1:-1]
        - u[:-2, 1:-1]
        - u[2:, 1:-1]
        - u[1:-1, :-2]
        - u[1:-1, 2:]
    ) / (h * h)
    return au


def residual(u: np.ndarray, f: np.ndarray, h: float) -> np.ndarray:
    """Residual ``f - A u`` (zero on the boundary)."""
    r = np.zeros_like(u)
    r[1:-1, 1:-1] = f[1:-1, 1:-1] - discrete_operator(u, h)[1:-1, 1:-1]
    return r


# ---------------------------------------------------------------------------
# grid transfers
# ---------------------------------------------------------------------------
def restrict_full_weighting(r_fine: np.ndarray) -> np.ndarray:
    """Full-weighting restriction of a *residual* from ``n`` to ``n/2`` cells.

    Tensor product of the 1-D stencil ``[1/4, 1/2, 1/4]`` -> 2-D weights
    ``4/16`` (centre), ``2/16`` (edge neighbours), ``1/16`` (corner
    neighbours).  The fine grid has ``n + 1`` nodes, the coarse grid has
    ``n/2 + 1`` nodes; coarse boundary values are zero.
    """
    n = r_fine.shape[0] - 1
    if n % 2 or n < 2:
        raise ValueError(f"restriction requires an even number of cells, got {n}")
    nc = n // 2
    r_c = np.zeros((nc + 1, nc + 1))
    c = r_fine[0::2, 0::2]
    e1 = r_fine[0::2, 1::2]  # first  fine index even, second odd
    e2 = r_fine[1::2, 0::2]
    corner = r_fine[1::2, 1::2]
    # centre: (2i, 2j)
    r_c[:] = 4.0 * c
    # edge neighbours of the coarse point (2i, 2j): (2i, 2j +- 1), (2i +- 1, 2j)
    # e1[k, l] = r(2k, 2l+1): neighbour of coarse (k, l) and (k, l+1)
    r_c[:, :-1] += 2.0 * e1
    r_c[:, 1:] += 2.0 * e1
    r_c[:-1, :] += 2.0 * e2
    r_c[1:, :] += 2.0 * e2
    # corner neighbours (2i +- 1, 2j +- 1)
    r_c[:-1, :-1] += corner
    r_c[:-1, 1:] += corner
    r_c[1:, :-1] += corner
    r_c[1:, 1:] += corner
    r_c /= 16.0
    # Dirichlet boundary: no correction on the boundary
    r_c[0, :] = 0.0
    r_c[-1, :] = 0.0
    r_c[:, 0] = 0.0
    r_c[:, -1] = 0.0
    return r_c


def prolong_bilinear(e_coarse: np.ndarray) -> np.ndarray:
    """Bilinear prolongation from ``n/2`` to ``n`` cells (tensor product of
    ``[1/2, 1, 1/2]``).  Boundary values of the coarse array are assumed zero."""
    nc = e_coarse.shape[0] - 1
    n = 2 * nc
    e = np.zeros((n + 1, n + 1))
    e[0::2, 0::2] = e_coarse
    e[0::2, 1:-1:2] = 0.5 * (e_coarse[:, :-1] + e_coarse[:, 1:])
    e[1:-1:2, 0::2] = 0.5 * (e_coarse[:-1, :] + e_coarse[1:, :])
    e[1:-1:2, 1:-1:2] = 0.25 * (
        e_coarse[:-1, :-1] + e_coarse[:-1, 1:] + e_coarse[1:, :-1] + e_coarse[1:, 1:]
    )
    return e


# ---------------------------------------------------------------------------
# smoothers
# ---------------------------------------------------------------------------
def jacobi_sweeps(
    u: np.ndarray, f: np.ndarray, h: float, omega: float = 2.0 / 3.0, sweeps: int = 1
) -> np.ndarray:
    """``sweeps`` damped-Jacobi iterations ``u <- u + omega/4 (h^2 f - A h^2 u)``."""
    if omega <= 0.0:
        raise ValueError("jacobi_sweeps: omega must be positive")
    h2 = h * h
    u = u.copy()
    for _ in range(sweeps):
        u[1:-1, 1:-1] += (omega / 4.0) * (
            h2 * f[1:-1, 1:-1]
            - 4.0 * u[1:-1, 1:-1]
            + u[:-2, 1:-1]
            + u[2:, 1:-1]
            + u[1:-1, :-2]
            + u[1:-1, 2:]
        )
    return u


def red_black_gs_sweeps(u: np.ndarray, f: np.ndarray, h: float, sweeps: int = 1) -> np.ndarray:
    """``sweeps`` red-black Gauss-Seidel sweeps (in place on a copy)."""
    h2f = h * h * f
    u = u.copy()
    ii, jj = np.meshgrid(
        np.arange(1, u.shape[0] - 1), np.arange(1, u.shape[1] - 1), indexing="ij"
    )
    red = (ii + jj) % 2 == 0
    black = ~red
    for _ in range(sweeps):
        for mask in (red, black):
            core = u[1:-1, 1:-1]
            nb = (
                u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
            )
            core[mask] = 0.25 * (h2f[1:-1, 1:-1][mask] + nb[mask])
    return u


# ---------------------------------------------------------------------------
# coarse-grid direct solve
# ---------------------------------------------------------------------------
def coarse_matrix(n: int, h: float) -> np.ndarray:
    """Dense matrix of the 5-point operator on an ``n``-cell grid (small ``n``)."""
    m = n - 1
    if m <= 0:
        return np.zeros((0, 0))
    A = np.zeros((m * m, m * m))
    for i in range(m):
        for j in range(m):
            k = i * m + j
            A[k, k] = 4.0 / (h * h)
            if i > 0:
                A[k, k - m] = -1.0 / (h * h)
            if i < m - 1:
                A[k, k + m] = -1.0 / (h * h)
            if j > 0:
                A[k, k - 1] = -1.0 / (h * h)
            if j < m - 1:
                A[k, k + 1] = -1.0 / (h * h)
    return A


def dense_direct_solve(f: np.ndarray, h: float) -> np.ndarray:
    """Exact solve of the coarse-grid system by dense LU (small grids only)."""
    n = f.shape[0] - 1
    m = n - 1
    u = np.zeros_like(f)
    if m <= 0:
        return u
    A = coarse_matrix(n, h)
    sol = np.linalg.solve(A, f[1:-1, 1:-1].reshape(-1))
    u[1:-1, 1:-1] = sol.reshape(m, m)
    return u


# ---------------------------------------------------------------------------
# multigrid cycles
# ---------------------------------------------------------------------------
Smoother = Callable[[np.ndarray, np.ndarray, float, int], np.ndarray]


@dataclass
class MultigridSolver:
    """Geometric multigrid V-/W-cycle for the 2-D Poisson problem.

    Parameters
    ----------
    n_fine
        Number of cells per direction on the finest grid (``h = 1/n_fine``);
        must be a power of two.
    nu1, nu2
        Number of pre- and post-smoothing sweeps.
    smoother
        ``"jacobi"`` (damped, diagonal symbol -> usable in the two-level
        analysis) or ``"gs"`` (red-black Gauss-Seidel).
    omega
        Damping parameter for the Jacobi smoother.
    n_coarse
        Number of cells per direction on the coarsest grid (default 4 ->
        ``h = 1/4``); the coarsest system is solved directly.
    cycle
        ``"V"`` or ``"W"``.
    """

    n_fine: int = 64
    nu1: int = 1
    nu2: int = 1
    smoother: str = "gs"
    omega: float = 2.0 / 3.0
    n_coarse: int = 4
    cycle: str = "V"

    _levels: List[int] = field(init=False, default_factory=list)
    cycles_applied: int = field(init=False, default=0)
    _coarse_cache: Dict[int, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        n = int(self.n_fine)
        if n & (n - 1):
            raise ValueError(f"n_fine must be a power of two, got {n}")
        if self.n_coarse & (self.n_coarse - 1) or self.n_coarse < 2:
            raise ValueError("n_coarse must be a power of two >= 2")
        if n < self.n_coarse:
            raise ValueError("n_fine must be >= n_coarse")
        if self.smoother not in ("jacobi", "gs"):
            raise ValueError("smoother must be 'jacobi' or 'gs'")
        if self.cycle.upper() not in ("V", "W"):
            raise ValueError("cycle must be 'V' or 'W'")
        self.cycle = self.cycle.upper()
        levels = []
        k = n
        while k >= self.n_coarse:
            levels.append(k)
            k //= 2
        if levels[-1] != self.n_coarse:
            raise ValueError("n_fine must be an integer multiple of n_coarse by powers of two")
        self._levels = levels

    # -- public API ---------------------------------------------------------
    @property
    def levels(self) -> List[int]:
        return list(self._levels)

    @property
    def n_levels(self) -> int:
        return len(self._levels)

    def smooth(self, u: np.ndarray, f: np.ndarray, h: float, sweeps: int) -> np.ndarray:
        if sweeps <= 0:
            return u
        if self.smoother == "jacobi":
            return jacobi_sweeps(u, f, h, omega=self.omega, sweeps=sweeps)
        return red_black_gs_sweeps(u, f, h, sweeps=sweeps)

    def coarse_solve(self, f: np.ndarray, h: float, level: int) -> np.ndarray:
        """Exact coarse-grid solve with a cached factorisation."""
        n = f.shape[0] - 1
        if level not in self._coarse_cache:
            self._coarse_cache[level] = np.linalg.inv(coarse_matrix(n, h))
        u = np.zeros_like(f)
        u[1:-1, 1:-1] = (self._coarse_cache[level] @ f[1:-1, 1:-1].reshape(-1)).reshape(n - 1, n - 1)
        return u

    def solve_level(self, u: np.ndarray, f: np.ndarray, h: float, level: int) -> np.ndarray:
        """One multigrid cycle on ``level`` (in place on ``u``, returns ``u``)."""
        n = self._levels[level]
        if level == self.n_levels - 1:
            return self.coarse_solve(f, h, level)
        u = self.smooth(u, f, h, self.nu1)
        r = residual(u, f, h)
        f_c = restrict_full_weighting(r)
        e_c = np.zeros((n // 2 + 1, n // 2 + 1))
        h_c = h * 2.0
        e_c = self.solve_level(e_c, f_c, h_c, level + 1)
        if self.cycle == "W":
            # second coarse-grid visit (W-cycle)
            r_c = residual(e_c, f_c, h_c)
            e2 = np.zeros_like(e_c)
            e2 = self.solve_level(e2, r_c, h_c, level + 1)
            e_c = e_c + e2
        u = u + prolong_bilinear(e_c)
        u = self.smooth(u, f, h, self.nu2)
        return u

    def cycle_once(
        self, u: np.ndarray, f: np.ndarray, n: Optional[int] = None
    ) -> np.ndarray:
        """Apply one cycle on the ``n_fine`` grid; ``u`` is not modified."""
        n = self.n_fine if n is None else n
        self.cycles_applied += 1
        return self.solve_level(u.copy(), f, 1.0 / n, 0)

    def solve(
        self,
        f: np.ndarray,
        tol: float = 1e-10,
        max_cycles: int = 100,
        u0: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """Iterate cycles until the discrete ``L_inf`` residual is below ``tol``."""
        n = f.shape[0] - 1
        h = 1.0 / n
        u = np.zeros_like(f) if u0 is None else u0.copy()
        hist = [float(np.abs(residual(u, f, h)).max())]
        for k in range(max_cycles):
            u = self.cycle_once(u, f, n)
            hist.append(float(np.abs(residual(u, f, h)).max()))
            if hist[-1] <= tol * max(hist[0], 1e-300):
                break
        return {
            "u": u,
            "residual_history": hist,
            "cycles": k + 1,
            "converged": bool(hist[-1] <= tol * max(hist[0], 1e-300)),
            "residual": hist[-1],
        }

    # -- linear-operator view (for spectral radius measurements) ------------
    def operator_matrix(self, n: Optional[int] = None) -> np.ndarray:
        """Explicit matrix of one cycle in the homogeneous (``f = 0``) case.

        The cycle is linear in ``u`` when ``f = 0``; its matrix is obtained by
        applying it to the canonical unit vectors of the interior unknowns.
        """
        n = self.n_fine if n is None else n
        m = n - 1
        f0 = np.zeros((n + 1, n + 1))
        M = np.zeros((m * m, m * m))
        for col in range(m * m):
            e = np.zeros(m * m)
            e[col] = 1.0
            u = np.zeros((n + 1, n + 1))
            u[1:-1, 1:-1] = e.reshape(m, m)
            M[:, col] = self.cycle_once(u, f0, n)[1:-1, 1:-1].reshape(-1)
        return M


def jacobi_solver(
    f: np.ndarray, omega: float = 1.0, max_iter: int = 200, tol: float = 1e-10
) -> Dict[str, object]:
    """Plain (undamped by default) Jacobi iteration -- the *baseline* solver.

    Returns the residual history, the iteration count and the iterate.
    """
    n = f.shape[0] - 1
    h = 1.0 / n
    u = np.zeros_like(f)
    hist = [float(np.abs(residual(u, f, h)).max())]
    it = 0
    for it in range(1, max_iter + 1):
        u = jacobi_sweeps(u, f, h, omega=omega, sweeps=1)
        hist.append(float(np.abs(residual(u, f, h)).max()))
        if hist[-1] <= tol * hist[0]:
            break
    return {
        "u": u,
        "residual_history": hist,
        "iterations": it,
        "converged": bool(hist[-1] <= tol * hist[0]),
    }


def convergence_history(
    solver: MultigridSolver,
    n: Optional[int] = None,
    cycles: int = 40,
    norm: str = "inf",
    seed: int = 0,
) -> Dict[str, object]:
    """Homogeneous-problem error decay of a stationary iteration.

    The cycle is applied repeatedly to a fixed pseudo-random initial guess with
    ``f = 0``; the per-cycle norm ratios converge to the spectral radius of the
    cycle operator (power iteration).
    """
    n = solver.n_fine if n is None else n
    h = 1.0 / n
    rng = np.random.default_rng(seed)
    u = np.zeros((n + 1, n + 1))
    u[1:-1, 1:-1] = rng.standard_normal((n - 1, n - 1))
    f0 = np.zeros_like(u)
    norms = [float(np.abs(u).max())]
    for _ in range(cycles):
        u = solver.cycle_once(u, f0, n)
        norms.append(float(np.abs(u).max()) if norm == "inf" else float(np.linalg.norm(u)))
    ratios = [norms[k + 1] / norms[k] for k in range(len(norms) - 1)]
    tail = ratios[-min(10, len(ratios)) :]
    return {
        "norms": norms,
        "ratios": ratios,
        "rho_asymptotic": float(np.exp(np.mean(np.log(tail)))),
        "h": h,
    }