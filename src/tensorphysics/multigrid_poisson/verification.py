"""Machine-checkable verification driver for the multigrid Poisson solver.

Runs the acceptance criteria C1--C8 of the campaign node and writes a JSON
record with every number, threshold and pass/fail flag.

Criteria
--------
C1  V-cycle convergence factor vs. the two-level-analysis (TLA) closed form
C2  W-cycle: ``rho_W <= rho_V`` and W-cycle vs. TLA
C3  complexity: measured speed-up of the multigrid cycle over Jacobi (N=256)
C4  spectral radius: numerical vs. analytical (closed-form quartic and
    explicitly assembled periodic two-grid operator)
C5  grid independence of ``rho_V`` for N = 64, 128, 256
C6  negative control: Jacobi stagnates (rate ``cos(pi h)``) / over-damped
    Jacobi diverges
C7  negative control: low-frequency stagnation of Jacobi vs. multigrid
C8  accuracy recovery: multigrid solution vs. an independent sparse direct
    solve of the same discrete system
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Dict, List, Optional

import numpy as np

from . import benchmarks as bm
from .core import (
    MultigridSolver,
    discrete_operator,
    analytic_solution,
    convergence_history,
    dense_direct_solve,
    jacobi_solver,
    jacobi_sweeps,
    poisson_rhs,
    residual,
)
from .tla import (
    harmonic_set,
    periodic_lfa_max,
    periodic_two_grid_spectral_radius,
    smoothing_factor,
    tla_characteristic_coefficients,
    tla_closed_form_radius,
    tla_spectral_radius,
    tla_two_grid_factor,
)

__all__ = ["PRIMARY_CONFIG", "run_verification", "sparse_direct_solve",
           "cg_reference_solve"]

PRIMARY_CONFIG = {
    "n_fine": 256,
    "nu1": 1,
    "nu2": 1,
    "smoother": "jacobi",
    "omega": 0.8,
    "cycle": "V",
    "n_coarse": 4,
}

# thresholds of the campaign node
THRESHOLDS = {
    "C1_rel_err_tla": 0.05,
    "C2_rel_err_tla": 0.05,
    "C3_speedup": 8.0,
    "C4_abs_err": 1e-6,
    "C5_rel_variation": 0.05,
    "C6_jacobi_rate_abs_err": 1e-6,
    "C6_jacobi_fail_budget": 200,
    "C7_jacobi_min_error_after": 0.95,
    "C7_mg_max_rate": 0.30,
    "C8_abs_err": 1e-6,
}


def _check(name: str, value, threshold, passed: bool, detail: str = "") -> Dict[str, object]:
    return {
        "criterion": name,
        "value": value,
        "threshold": threshold,
        "passed": bool(passed),
        "detail": detail,
    }


def cg_reference_solve(f: np.ndarray, tol: float = 1e-13, max_iter: int = 200000) -> np.ndarray:
    """Independent reference solution by Jacobi-preconditioned CG (NumPy only).

    Fallback for :func:`sparse_direct_solve` when SciPy is unavailable; the
    Krylov iteration is a structurally different algorithm, so it still serves
    as an independent reference for the multigrid solution.
    """
    n = f.shape[0] - 1
    h = 1.0 / n
    x = np.zeros_like(f)
    r = f.copy()
    z = r / (4.0 / (h * h))
    p = z.copy()
    rz = float((r[1:-1, 1:-1] * z[1:-1, 1:-1]).sum())
    rhs_norm = np.sqrt(float((f[1:-1, 1:-1] ** 2).sum()))
    for _ in range(max_iter):
        if np.sqrt(float((r[1:-1, 1:-1] ** 2).sum())) <= tol * rhs_norm:
            break
        Ap = discrete_operator(p, h)
        alpha = rz / float((p[1:-1, 1:-1] * Ap[1:-1, 1:-1]).sum())
        x += alpha * p
        r -= alpha * Ap
        z = r / (4.0 / (h * h))
        rz_new = float((r[1:-1, 1:-1] * z[1:-1, 1:-1]).sum())
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def sparse_direct_solve(f: np.ndarray) -> np.ndarray:
    """Independent reference solution of the same discrete system.

    Sparse direct LU (``scipy.sparse.linalg.spsolve``); the discretisation is
    the same 5-point stencil, so the difference to the multigrid solution
    isolates the *solver* error.  Falls back to Jacobi-preconditioned CG when
    SciPy is not installed.
    """
    try:
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve
    except Exception:  # pragma: no cover - SciPy missing
        return cg_reference_solve(f)

    n = f.shape[0] - 1
    h = 1.0 / n
    m = n - 1
    A = lil_matrix((m * m, m * m))
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
    sol = spsolve(A.tocsr(), f[1:-1, 1:-1].reshape(-1))
    u = np.zeros_like(f)
    u[1:-1, 1:-1] = sol.reshape(m, m)
    return u


# ---------------------------------------------------------------------------
def run_verification(
    n_fine: int = 256,
    omega: float = 0.8,
    nu1: int = 1,
    nu2: int = 1,
    out_dir: Optional[str] = None,
    fast: bool = False,
) -> Dict[str, object]:
    """Run criteria C1--C8 and write ``verification.json``.

    Parameters
    ----------
    n_fine
        Finest grid (default 256, i.e. ``h = 1/256``).
    omega, nu1, nu2
        Damped-Jacobi smoother parameters of the production configuration.
    fast
        Skip the more expensive parameter sweep / complexity probe (used by
        the unit tests; the acceptance record uses ``fast=False``).
    """
    t_start = time.perf_counter()
    sizes = (64, 128, 256) if not fast else (64,)
    checks: List[Dict[str, object]] = []

    # ---------------- C1/C2: cycle factors vs TLA --------------------------
    tla_primary = tla_two_grid_factor(nu1, nu2, omega)
    rho_tla = float(tla_primary["rho_tla"])
    tla_closed = tla_two_grid_factor(nu1, nu2, omega, method="cf")
    cycle_rates: Dict[str, Dict[str, float]] = {}
    for kind in ("V", "W"):
        s = MultigridSolver(
            n_fine=n_fine, nu1=nu1, nu2=nu2, smoother="jacobi", omega=omega, cycle=kind
        )
        hist = convergence_history(s, n_fine, cycles=40 if not fast else 25)
        cycle_rates[kind] = {
            "rho": hist["rho_asymptotic"],
            "rel_error_vs_tla": abs(hist["rho_asymptotic"] - rho_tla) / rho_tla,
        }
    two_grid_n = 16 if fast else 32
    two_grid = bm.two_grid_operator_spectral_radius(
        n=two_grid_n, nu1=nu1, nu2=nu2, smoother="jacobi", omega=omega
    )

    checks.append(_check(
        "C1", cycle_rates["V"]["rho"], THRESHOLDS["C1_rel_err_tla"],
        cycle_rates["V"]["rel_error_vs_tla"] <= THRESHOLDS["C1_rel_err_tla"],
        f"rho_V(jacobi omega={omega}, nu=({nu1},{nu2}), N={n_fine})="
        f"{cycle_rates['V']['rho']:.6f} vs rho_TLA={rho_tla:.6f}, "
        f"relative error {cycle_rates['V']['rel_error_vs_tla']*100:.2f}% <= 5%",
    ))
    checks.append(_check(
        "C2", cycle_rates["W"]["rho"], THRESHOLDS["C2_rel_err_tla"],
        (cycle_rates["W"]["rho"] <= cycle_rates["V"]["rho"])
        and (cycle_rates["W"]["rel_error_vs_tla"] <= THRESHOLDS["C2_rel_err_tla"]),
        f"rho_W={cycle_rates['W']['rho']:.6f} <= rho_V={cycle_rates['V']['rho']:.6f}; "
        f"W vs TLA relative error {cycle_rates['W']['rel_error_vs_tla']*100:.2f}% <= 5%",
    ))
    checks.append(_check(
        "C1-supplement-two-grid", two_grid["rho"], rho_tla,
        abs(two_grid["rho"] - rho_tla) / rho_tla <= THRESHOLDS["C1_rel_err_tla"],
        f"exact two-grid operator on the Dirichlet grid (N={two_grid_n}) rho={two_grid['rho']:.6f} "
        f"vs rho_TLA={rho_tla:.6f}, relative error "
        f"{abs(two_grid['rho']-rho_tla)/rho_tla*100:.3f}%",
    ))

    # ---------------- C3: complexity / speed-up ---------------------------
    complexity = bm.complexity_study(
        sizes=(64, 128, 256, 512) if not fast else (64, 128),
        nu1=nu1, nu2=nu2, omega=omega,
        jacobi_probe_iterations=500 if fast else 1000,
    )
    n256 = [r for r in complexity["rows"] if r["n"] == 256]
    ref_row = n256[0] if n256 else complexity["rows"][-1]
    speedup = ref_row["speedup_vs_jacobi"]
    checks.append(_check(
        "C3", float(speedup), THRESHOLDS["C3_speedup"], speedup >= THRESHOLDS["C3_speedup"],
        "wall-clock speed-up of the multigrid cycle over Jacobi at N=256 to a relative "
        f"residual tolerance 1e-10: {speedup:.1f}x (Jacobi needs "
        f"{ref_row['jacobi_iterations_estimated']} sweeps at rate "
        f"cos(pi h)={ref_row['jacobi_rate']:.8f}); per-cycle cost scales "
        f"as (N_unknowns)^{complexity['per_cycle_scaling_exponent']:.2f}",
    ))

    # ---------------- C4: numerical vs analytical spectral radius ---------
    theta_grid = np.linspace(1e-3, 0.5 * np.pi, 24, endpoint=False)
    worst_quart = 0.0
    worst_theta = (0.0, 0.0)
    for tx in theta_grid:
        for ty in theta_grid:
            r_num = tla_spectral_radius((tx, ty), nu1, nu2, omega)
            r_cf = tla_closed_form_radius((tx, ty), nu1, nu2, omega)
            if abs(r_num - r_cf) > worst_quart:
                worst_quart = abs(r_num - r_cf)
                worst_theta = (float(tx), float(ty))
    periodic = periodic_two_grid_spectral_radius(16, nu1, nu2, omega)
    lfa_discrete = periodic_lfa_max(16, nu1, nu2, omega)
    worst_periodic = abs(periodic["rho"] - lfa_discrete["rho_lfa"])
    checks.append(_check(
        "C4", float(max(worst_quart, worst_periodic)), THRESHOLDS["C4_abs_err"],
        max(worst_quart, worst_periodic) <= THRESHOLDS["C4_abs_err"],
        f"|rho_eig(4x4 symbol) - rho_quartic(closed form)| <= {worst_quart:.2e} over 576 "
        f"theta points (worst at {np.round(worst_theta,4).tolist()}); "
        f"|rho_explicit_periodic_operator - rho_LFA| = {worst_periodic:.2e} (n=16)",
    ))

    # ---------------- C5: grid independence --------------------------------
    grid_rows = []
    for n in sizes:
        s = MultigridSolver(n_fine=n, nu1=nu1, nu2=nu2, smoother="jacobi", omega=omega)
        hist = convergence_history(s, n, cycles=40)
        grid_rows.append({"n": n, "rho_V": hist["rho_asymptotic"],
                          "rel_error_vs_tla": abs(hist["rho_asymptotic"] - rho_tla) / rho_tla})
    rhos = [r["rho_V"] for r in grid_rows]
    variation = (max(rhos) - min(rhos)) / float(np.mean(rhos)) if len(rhos) > 1 else 0.0
    checks.append(_check(
        "C5", float(variation), THRESHOLDS["C5_rel_variation"],
        variation <= THRESHOLDS["C5_rel_variation"],
        f"rho_V over N={list(sizes)}: {[round(r,6) for r in rhos]}, relative variation "
        f"{(max(rhos)-min(rhos))/float(np.mean(rhos))*100:.2f}% <= 5%",
    ))

    # ---------------- C6: negative control, Jacobi ------------------------
    n_jac = 16
    jac_mats = bm.jacobi_iteration_matrix_spectral_radius(n_jac)
    jac_run = jacobi_solver(poisson_rhs(256), omega=1.0,
                            max_iter=THRESHOLDS["C6_jacobi_fail_budget"])
    # over-damped control (omega > 1): |1 - omega a/4| > 1 for the high modes
    n_div = 64
    rng = np.random.default_rng(0)
    e = np.zeros((n_div + 1, n_div + 1))
    e[1:-1, 1:-1] = rng.standard_normal((n_div - 1, n_div - 1))
    f0 = np.zeros_like(e)
    n0 = float(np.abs(e).max())
    growth = []
    for _ in range(50):
        e = jacobi_sweeps(e, f0, 1.0 / n_div, omega=1.1, sweeps=1)
        growth.append(float(np.abs(e).max()) / n0)
    div_rate = growth[-1] ** (1.0 / 50.0)
    rel_red = jac_run["residual_history"][-1] / jac_run["residual_history"][0]
    c6_pass = (
        (not jac_run["converged"])
        and jac_mats["rel_error"] <= THRESHOLDS["C6_jacobi_rate_abs_err"]
        and div_rate > 1.0
    )
    checks.append(_check(
        "C6", jac_mats["rho_numeric"], THRESHOLDS["C6_jacobi_rate_abs_err"], c6_pass,
        f"Jacobi (omega=1) iteration matrix on N={n_jac}: exact spectral radius "
        f"{jac_mats['rho_numeric']:.10f} vs analytic cos(pi h)={jac_mats['rho_analytic']:.10f} "
        f"(relative error {jac_mats['rel_error']:.2e} <= 1e-6); at N=256 "
        f"{THRESHOLDS['C6_jacobi_fail_budget']} sweeps leave {rel_red*100:.1f}% of the initial "
        f"residual (rho=1-pi^2h^2/2={jac_mats['rho_analytic_256']:.8f}, tolerance 1e-10 NOT "
        f"reached -> practical non-convergence); over-damped control omega=1.1 with a random "
        f"start grows at {div_rate:.4f} per sweep > 1 (divergence)",
    ))

    # ---------------- C7: low-frequency stagnation ------------------------
    slow = bm.lowest_mode_decay(n=128 if fast else 256, iterations=100, mg_cycles=10,
                                nu1=nu1, nu2=nu2, omega=omega)
    c7_pass = (slow["jacobi_error_after"] >= THRESHOLDS["C7_jacobi_min_error_after"]
               and slow["mg_rate"] <= THRESHOLDS["C7_mg_max_rate"])
    checks.append(_check(
        "C7", slow["jacobi_error_after"], THRESHOLDS["C7_jacobi_min_error_after"], c7_pass,
        f"pure lowest mode sin(pi x)sin(pi y), N={slow['n']}: after 100 Jacobi sweeps "
        f"{slow['jacobi_error_after']*100:.2f}% of the error remains (rate "
        f"{slow['jacobi_rate']:.8f} vs analytic 1-lambda h^2/4 = "
        f"{slow['jacobi_rate_analytic_slow_mode']:.8f}); one multigrid cycle rate "
        f"{slow['mg_rate']:.4f} -> after 10 cycles {slow['mg_error_after']:.2e}",
    ))

    # ---------------- C8: accuracy vs independent direct solve ------------
    n8 = 128 if fast else n_fine
    f8 = poisson_rhs(n8)
    mg8 = MultigridSolver(n_fine=n8, nu1=nu1, nu2=nu2, smoother="jacobi", omega=omega)
    res8 = mg8.solve(f8, tol=1e-10, max_cycles=200)
    u_ref = sparse_direct_solve(f8)
    err_solver = float(np.abs(res8["u"] - u_ref).max())
    err_disc = float(np.abs(res8["u"] - analytic_solution(n8)).max())
    checks.append(_check(
        "C8", err_solver, THRESHOLDS["C8_abs_err"], err_solver <= THRESHOLDS["C8_abs_err"],
        f"N={n8}: |u_MG - u_direct|_inf = {err_solver:.3e} <= 1e-6 (solver error); "
        f"|u_MG - u*|_inf = {err_disc:.3e} (discretisation error, O(h^2))",
    ))

    # ---------------- supplementary tables --------------------------------
    sweep = bm.parameter_sweep(
        sizes=sizes,
        omegas=(0.5, 2.0 / 3.0, 0.8, 0.9),
        nus=((1, 1), (2, 2)),
        cycles_kind=("V", "W"),
        cycles=40 if not fast else 20,
    )
    disc = bm.discretisation_error_study(sizes=(32, 64, 128, 256) if not fast else (32, 64))

    # solution / error fields for the figures (stored as JSON-friendly lists)
    n_field = 64 if fast else 128
    mg_field = MultigridSolver(n_fine=n_field, nu1=nu1, nu2=nu2, smoother="jacobi", omega=omega)
    res_field = mg_field.solve(poisson_rhs(n_field), tol=1e-10, max_cycles=200)

    record = {
        "schema": "tensorphysics.multigrid_poisson.verification/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_fine": n_fine,
            "nu1": nu1,
            "nu2": nu2,
            "smoother": "damped Jacobi",
            "omega": omega,
            "n_coarse": 4,
            "levels": MultigridSolver(n_fine=n_fine, n_coarse=4).levels,
            "cycle_rates": cycle_rates,
            "rho_tla": rho_tla,
            "rho_tla_closed_form": tla_closed["rho_tla"],
            "theta_max": tla_primary["theta_max"],
            "smoothing_factor": smoothing_factor(omega),
            "two_grid_operator_rho_dirichlet_N32": two_grid["rho"],
        },
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "grid_independence": grid_rows,
        "complexity": complexity,
        "parameter_sweep": sweep,
        "discretisation_error": disc,
        "lowest_mode": slow,
        "jacobi_probe": {
            "rho_numeric": jac_mats["rho_numeric"],
            "rho_analytic": jac_mats["rho_analytic"],
            "rel_error": jac_mats["rel_error"],
            "n": jac_mats["n"],
            "rho_analytic_256": jac_mats["rho_analytic_256"],
        },
        "negative_control": {
            "jacobi_omega_1_residual_after_200": jac_run["residual_history"][-1],
            "jacobi_omega_1_relative_reduction": rel_red,
            "jacobi_omega_1_rate_exact": jac_mats["rho_numeric"],
            "jacobi_omega_1_converged": jac_run["converged"],
            "jacobi_omega_1.1_rate_per_sweep": float(div_rate),
        },
        "accuracy": {
            "n": n8,
            "solver_error_vs_direct": err_solver,
            "discretisation_error_vs_exact": err_disc,
        },
        "field": {
            "n": n_field,
            "solution": np.asarray(res_field["u"]).tolist(),
            "analytic": analytic_solution(n_field).tolist(),
        },
        "runtime_s": time.perf_counter() - t_start,
    }

    if out_dir:
        path = pathlib.Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "verification.json").write_text(json.dumps(record, indent=2, default=float))
        record["verification_json"] = str(path / "verification.json")
    return record


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI helper
    import argparse

    ap = argparse.ArgumentParser(description="multigrid Poisson verification (C1-C8)")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--omega", type=float, default=0.8)
    ap.add_argument("--out", default="artifacts/multigrid_poisson")
    args = ap.parse_args(argv)
    rec = run_verification(n_fine=args.n, omega=args.omega, out_dir=args.out)
    for c in rec["checks"]:
        print(f"{c['criterion']:24s} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    print("all_passed:", rec["all_passed"], " runtime:", round(rec["runtime_s"], 1), "s")
    return 0 if rec["all_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())