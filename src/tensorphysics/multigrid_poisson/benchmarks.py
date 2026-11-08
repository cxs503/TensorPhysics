"""Benchmarks and quantitative verification helpers for the multigrid solver.

Provides

``two_grid_operator_spectral_radius``
    Exact spectral radius of the *two-grid* operator (coarse solve done
    directly) on the Dirichlet grid, obtained from an explicitly assembled
    matrix -- the numerical reference for the two-level analysis.
``jacobi_rate_analytic`` / ``jacobi_rate_measured``
    Classical Jacobi convergence rate ``cos(pi h)`` for the 2-D 5-point
    Laplacian vs. the measured asymptotic rate (an independent analytic
    cross-check of the discrete operator, and the negative control C6).
``lowest_mode_decay``
    Decay of the pure lowest eigenmode ``sin(pi x) sin(pi y)`` under one
    stationary iteration -- quantifies low-frequency stagnation (C7).
``complexity_study``
    Wall-clock work per cycle as a function of the number of unknowns and the
    measured speed-up of the V-cycle over Jacobi at ``n = 256`` (C3).
``parameter_sweep``
    Convergence factors over a smoother/damping/cycle parameter grid, with the
    two-level-analysis prediction for every damped-Jacobi case.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .core import (
    MultigridSolver,
    analytic_solution,
    convergence_history,
    jacobi_solver,
    jacobi_sweeps,
    poisson_rhs,
    residual,
)
from .tla import tla_two_grid_factor

__all__ = [
    "two_grid_operator_spectral_radius",
    "per_cycle_time",
    "work_units_per_cycle",
    "jacobi_rate_analytic",
    "jacobi_rate_measured",
    "jacobi_iteration_matrix_spectral_radius",
    "lowest_mode_decay",
    "complexity_study",
    "parameter_sweep",
    "discretisation_error_study",
]


# ---------------------------------------------------------------------------
def two_grid_operator_spectral_radius(
    n: int = 32, nu1: int = 1, nu2: int = 1, smoother: str = "jacobi", omega: float = 2.0 / 3.0
) -> Dict[str, object]:
    """Exact spectral radius of the two-grid operator on the Dirichlet grid.

    The hierarchy is truncated to two levels (``n_coarse = n/2``), so the
    coarse-grid equation is solved directly; the cycle matrix is assembled
    explicitly and its eigenvalues are computed.  This is the numerical
    counterpart of the two-level analysis and has no smoothing-factor
    approximations.
    """
    solver = MultigridSolver(
        n_fine=n, nu1=nu1, nu2=nu2, smoother=smoother, omega=omega,
        n_coarse=n // 2, cycle="V",
    )
    M = solver.operator_matrix(n)
    eigs = np.linalg.eigvals(M)
    return {
        "rho": float(np.max(np.abs(eigs))),
        "eigs_abs_max": float(np.max(np.abs(eigs))),
        "n": n,
        "nu1": nu1,
        "nu2": nu2,
        "smoother": smoother,
        "omega": omega,
        "n_unknowns": (n - 1) ** 2,
    }


# ---------------------------------------------------------------------------
def jacobi_rate_analytic(n: int) -> float:
    """Spectral radius of the undamped Jacobi iteration for the 2-D 5-point
    Laplacian with Dirichlet BC: ``rho = cos(pi h)``."""
    return float(np.cos(np.pi / n))


def jacobi_rate_measured(n: int, cycles: int = 60, seed: int = 0) -> Dict[str, object]:
    """Measured asymptotic decay rate of undamped Jacobi on the homogeneous
    problem (power iteration from a pseudo-random start)."""
    h = 1.0 / n
    rng = np.random.default_rng(seed)
    u = np.zeros((n + 1, n + 1))
    u[1:-1, 1:-1] = rng.standard_normal((n - 1, n - 1))
    f0 = np.zeros_like(u)
    norms = [float(np.abs(u).max())]
    for _ in range(cycles):
        u = jacobi_sweeps(u, f0, h, omega=1.0, sweeps=1)
        norms.append(float(np.abs(u).max()))
    ratios = [norms[k + 1] / norms[k] for k in range(len(norms) - 1)]
    tail = ratios[-10:]
    return {
        "rho_measured": float(np.exp(np.mean(np.log(tail)))),
        "rho_analytic": jacobi_rate_analytic(n),
        "rel_error": float(abs(np.exp(np.mean(np.log(tail))) - jacobi_rate_analytic(n))
                           / jacobi_rate_analytic(n)),
        "norms": norms,
    }


# ---------------------------------------------------------------------------
def jacobi_iteration_matrix_spectral_radius(n: int = 16) -> Dict[str, object]:
    """Exact spectral radius of the undamped-Jacobi iteration matrix.

    ``S = I - A h^2/4`` is assembled densely on an ``n x n`` Dirichlet grid and
    its eigenvalues are computed; the analytic value is ``cos(pi h)``.
    """
    from .core import coarse_matrix

    h = 1.0 / n
    A = coarse_matrix(n, h)
    S = np.eye(A.shape[0]) - (h * h / 4.0) * A
    rho = float(np.max(np.abs(np.linalg.eigvals(S))))
    exact = jacobi_rate_analytic(n)
    return {
        "n": n,
        "rho_numeric": rho,
        "rho_analytic": exact,
        "rel_error": float(abs(rho - exact) / exact),
        "rho_analytic_256": jacobi_rate_analytic(256),
    }


def lowest_mode_decay(
    n: int = 256, iterations: int = 100, mg_cycles: int = 10,
    nu1: int = 1, nu2: int = 1, omega: float = 0.8,
) -> Dict[str, object]:
    """Decay of the pure lowest eigenvalue mode under Jacobi and under MG.

    Initial error ``e0 = sin(pi x) sin(pi y)`` (the slowest mode of the
    discrete Laplacian, eigenvalue ``8 sin^2(pi h / 2) / h^2``).  Jacobi
    reduces it at the rate ``1 - pi^2 h^2 / 2`` (stagnation), the multigrid
    cycle should damp it like any other smooth mode.
    """
    h = 1.0 / n
    e0 = analytic_solution(n)
    f0 = np.zeros_like(e0)
    # Jacobi
    e = e0.copy()
    for _ in range(iterations):
        e = jacobi_sweeps(e, f0, h, omega=1.0, sweeps=1)
    jacobi_ratio = float(np.abs(e).max() / np.abs(e0).max())
    jacobi_rate = jacobi_ratio ** (1.0 / iterations)
    # multigrid
    mg = MultigridSolver(n_fine=n, nu1=nu1, nu2=nu2, smoother="jacobi", omega=omega)
    e = e0.copy()
    for _ in range(mg_cycles):
        e = mg.cycle_once(e, f0, n)
    mg_ratio = float(np.abs(e).max() / np.abs(e0).max())
    mg_rate = mg_ratio ** (1.0 / mg_cycles)
    lam = 8.0 * np.sin(np.pi * h / 2.0) ** 2 / (h * h)
    return {
        "n": n,
        "iterations_jacobi": iterations,
        "jacobi_rate": float(jacobi_rate),
        "jacobi_rate_analytic_slow_mode": float(1.0 - lam * h * h / 4.0),
        "jacobi_error_after": jacobi_ratio,
        "mg_cycles": mg_cycles,
        "mg_rate": float(mg_rate),
        "mg_error_after": mg_ratio,
    }


# ---------------------------------------------------------------------------
def per_cycle_time(
    n: int, cycles: int = 20, nu1: int = 1, nu2: int = 2, omega: float = 0.8,
    n_coarse: int = 4,
) -> float:
    """Measured mean wall-clock time of one V-cycle on an ``n`` grid."""
    solver = MultigridSolver(n_fine=n, nu1=nu1, nu2=nu2, smoother="jacobi",
                             omega=omega, n_coarse=n_coarse)
    u = np.zeros((n + 1, n + 1))
    f0 = np.zeros_like(u)
    solver.cycle_once(u, f0, n)  # warm-up (factorisation cache)
    best = float("inf")
    for _ in range(3):  # min of repeats: robust against machine load
        t0 = time.perf_counter()
        for _ in range(cycles):
            u = solver.cycle_once(u, f0, n)
        best = min(best, (time.perf_counter() - t0) / cycles)
    return best


def work_units_per_cycle(nu1: int, nu2: int, n_levels: int) -> Dict[str, float]:
    """Deterministic work accounting in fine-grid-equivalent relaxation sweeps.

    A V-cycle performs ``nu1 + nu2`` sweeps on every level; the grid sizes
    shrink geometrically (``4^-l``), so the smoothing cost per cycle is
    ``(nu1+nu2) * (1 - 4^-L)/(1 - 1/4)`` fine-grid sweeps (transfers are
    ``O(1)`` fine-grid sweeps per level and are neglected).
    """
    factor = (1.0 - 4.0 ** (-n_levels)) / (1.0 - 0.25)
    return {
        "smoothing_fine_sweeps_per_cycle": float((nu1 + nu2) * factor),
        "n_levels": int(n_levels),
    }


def complexity_study(
    sizes: Sequence[int] = (64, 128, 256),
    tol: float = 1e-10,
    jacobi_probe_iterations: int = 2000,
    nu1: int = 1,
    nu2: int = 1,
    omega: float = 0.8,
) -> Dict[str, object]:
    """Measured work of the V-cycle vs Jacobi.

    For each grid the multigrid solver is run to convergence (relative
    ``L_inf`` residual ``<= tol``) and timed.  Jacobi is *probed* for
    ``jacobi_probe_iterations`` sweeps: its measured asymptotic rate
    ``rho_J`` is fitted to ``cos(pi h)`` and the iteration count needed to
    reach the same tolerance is extrapolated, which gives the speed-up.  The
    per-cycle cost is also reported to expose the ``O(#unknowns)`` scaling.
    """
    rows: List[Dict[str, object]] = []
    for n in sizes:
        f = poisson_rhs(n)
        mg = MultigridSolver(n_fine=n, nu1=nu1, nu2=nu2, smoother="jacobi", omega=omega)
        t0 = time.perf_counter()
        res = mg.solve(f, tol=tol, max_cycles=200)
        t_mg = time.perf_counter() - t0
        # Jacobi probe
        h = 1.0 / n
        u = np.zeros_like(f)
        t0 = time.perf_counter()
        for _ in range(jacobi_probe_iterations):
            u = jacobi_sweeps(u, f, h, omega=1.0, sweeps=1)
        t_jacobi_probe = time.perf_counter() - t0
        rho_j = jacobi_rate_analytic(n)
        needed = int(np.ceil(np.log(tol) / np.log(rho_j)))
        t_jacobi_est = t_jacobi_probe / jacobi_probe_iterations * needed
        t_cyc = per_cycle_time(n, cycles=20, nu1=nu1, nu2=nu2, omega=omega)
        rows.append({
            "n": n,
            "n_unknowns": (n - 1) ** 2,
            "mg_cycles": res["cycles"],
            "mg_time_s": t_mg,
            "mg_time_per_cycle_s": t_cyc,
            "mg_residual": res["residual"],
            "mg_converged": res["converged"],
            "jacobi_rate": rho_j,
            "jacobi_iterations_estimated": needed,
            "jacobi_time_per_sweep_s": t_jacobi_probe / jacobi_probe_iterations,
            "jacobi_time_estimated_s": t_jacobi_est,
            "speedup_vs_jacobi": t_jacobi_est / t_mg,
            "work_units_per_cycle": work_units_per_cycle(
                nu1, nu2, len(MultigridSolver(n_fine=n, n_coarse=4).levels)
            )["smoothing_fine_sweeps_per_cycle"],
            "jacobi_fine_sweeps_estimated": needed,
        })
    # O(#unknowns) scaling of the per-cycle cost (least-squares slope in log-log)
    n_unk = np.array([r["n_unknowns"] for r in rows], dtype=float)
    t_cyc = np.array([r["mg_time_per_cycle_s"] for r in rows], dtype=float)
    if len(rows) > 1:
        slope = float(np.polyfit(np.log(n_unk), np.log(t_cyc), 1)[0])
    else:
        slope = float("nan")
    return {"rows": rows, "per_cycle_scaling_exponent": slope}


# ---------------------------------------------------------------------------
def parameter_sweep(
    sizes: Sequence[int] = (64, 128, 256),
    omegas: Iterable[float] = (0.5, 2.0 / 3.0, 0.8, 0.9),
    nus: Iterable[Tuple[int, int]] = ((1, 1), (2, 2)),
    cycles_kind: Iterable[str] = ("V", "W"),
    cycles: int = 40,
) -> List[Dict[str, object]]:
    """Convergence factors over a damping / smoothing / cycle parameter grid."""
    rows: List[Dict[str, object]] = []
    for n in sizes:
        for om in omegas:
            for nu1, nu2 in nus:
                tla = tla_two_grid_factor(nu1, nu2, om)["rho_tla"]
                for kind in cycles_kind:
                    s = MultigridSolver(
                        n_fine=n, nu1=nu1, nu2=nu2, smoother="jacobi", omega=om, cycle=kind
                    )
                    hist = convergence_history(s, n, cycles=cycles)
                    rows.append({
                        "n": n,
                        "smoother": "jacobi",
                        "omega": float(om),
                        "nu1": nu1,
                        "nu2": nu2,
                        "cycle": kind,
                        "rho_measured": hist["rho_asymptotic"],
                        "rho_tla": tla,
                        "rel_error_vs_tla": float(
                            abs(hist["rho_asymptotic"] - tla) / tla
                        ),
                    })
    # red-black Gauss-Seidel rows (no scalar LFA symbol -> no TLA column)
    for n in sizes:
        for nu1, nu2 in nus:
            for kind in cycles_kind:
                s = MultigridSolver(n_fine=n, nu1=nu1, nu2=nu2, smoother="gs", cycle=kind)
                hist = convergence_history(s, n, cycles=cycles)
                rows.append({
                    "n": n,
                    "smoother": "gs",
                    "omega": None,
                    "nu1": nu1,
                    "nu2": nu2,
                    "cycle": kind,
                    "rho_measured": hist["rho_asymptotic"],
                    "rho_tla": None,
                    "rel_error_vs_tla": None,
                })
    return rows


# ---------------------------------------------------------------------------
def discretisation_error_study(sizes: Sequence[int] = (32, 64, 128, 256)) -> Dict[str, object]:
    """Discretisation error of the discrete solution against ``u*`` and the
    observed order of convergence (sanity check that the accuracy recovered by
    the solver is the *discretisation* error, not a solver error)."""
    rows = []
    prev = None
    for n in sizes:
        mg = MultigridSolver(n_fine=n, nu1=1, nu2=1, smoother="jacobi", omega=0.8)
        res = mg.solve(poisson_rhs(n), tol=1e-10, max_cycles=200)
        err = float(np.abs(res["u"] - analytic_solution(n)).max())
        order = None if prev is None else float(np.log(prev / err) / np.log(2.0))
        rows.append({"n": n, "h": 1.0 / n, "l_inf_error": err,
                     "residual": res["residual"], "cycles": res["cycles"],
                     "observed_order": order})
        prev = err
    return {"rows": rows}