"""Scalar metrics used to fill the applicability probe.

Two families live here, both free of domain vocabulary:

``analytic``
    Closed-form scalars with a *known* boundary, used to black-box test the
    mechanism itself (each has an algebraically known in/out boundary, so the
    detected boundary can be checked against the truth).

``numerical``
    Reference metrics over this repository's own neutral numerical kernels
    (stationary iteration, geometric multigrid).  Every function-symbol, grid
    size, iteration budget and tolerance is a *required* keyword argument: the
    wiring comes from configuration, never from a default buried in code.

All metrics return a mapping readable by
:class:`~.capability.EvaluatorSpec`: ``{"value": float, "reference": float,
"flags": {...}}``.  Where a failure signal is mechanical the metric reports it in
``flags`` (``finite``, ``converged``, ``observed_order``) and the capability
declaration decides whether that signal is fatal.
"""
from __future__ import annotations

import math
from .capability import import_target

__all__ = [
    "power_law",
    "bilinear",
    "additive",
    "ratio_product",
    "exponential_deviation",
    "reciprocal",
    "log_value",
    "flag_gate",
    "order_report",
    "hyperbolic_excess",
    "stationary_solver_error",
    "stationary_iteration_rate",
    "multigrid_cycle_rate",
]


# ---------------------------------------------------------------------------
# analytic / algebraic scalars (known boundary, used by the black-box tests)
# ---------------------------------------------------------------------------

def power_law(x: float, *, amplitude: float, exponent: float, offset: float = 0.0) -> dict:
    """``value = amplitude * x**exponent + offset`` (monotone for positive exponent)."""
    value = amplitude * float(x) ** float(exponent) + offset
    return {"value": float(value), "reference": float(offset),
            "flags": {"finite": math.isfinite(value)}}


def bilinear(x: float, y: float, *, amplitude: float) -> dict:
    """``value = amplitude * x * y`` -- two independent axes, known boundary."""
    value = amplitude * float(x) * float(y)
    return {"value": float(value), "reference": 0.0,
            "flags": {"finite": math.isfinite(value)}}


def additive(a: float, b: float, *, amplitude: float = 1.0) -> dict:
    """``value = amplitude * (a + b)`` -- joint failure on a diagonal of the plane."""
    value = float(amplitude) * (float(a) + float(b))
    return {"value": float(value), "reference": 0.0,
            "flags": {"finite": math.isfinite(value)}}


def ratio_product(s: float, t: float) -> dict:
    """``value = s * t / (s + t)`` -- joint failure on a curved band of the plane."""
    total = float(s) + float(t)
    value = (float(s) * float(t) / total) if total != 0.0 else float("nan")
    return {"value": float(value), "reference": 0.0,
            "flags": {"finite": math.isfinite(value)}}


def exponential_deviation(x: float, *, amplitude: float, rate: float, target: float = 0.0) -> dict:
    """``value = |amplitude * exp(-rate * x) - target|`` -- decays, then floors out."""
    value = abs(amplitude * math.exp(-float(rate) * float(x)) - target)
    return {"value": float(value), "reference": 0.0, "flags": {"finite": math.isfinite(value)}}


def reciprocal(x: float, *, amplitude: float) -> dict:
    """``value = amplitude / x`` -- defined only for ``x > 0`` (raises otherwise)."""
    value = amplitude / float(x)
    return {"value": float(value), "reference": 0.0, "flags": {"finite": math.isfinite(value)}}


def log_value(x: float, *, scale: float) -> dict:
    """``value = log(x / scale)`` -- non-finite/raising for ``x <= 0``."""
    ratio = float(x) / float(scale)
    if ratio <= 0.0:
        return {"value": float("nan"), "reference": 0.0, "flags": {"finite": False}}
    return {"value": math.log(ratio), "reference": 0.0, "flags": {"finite": True}}


def hyperbolic_excess(x: float, *, critical: float, amplitude: float,
                      epsilon: float = 0.0) -> dict:
    """``value = amplitude * x / (critical - x)`` -- diverges at ``critical``.

    ``epsilon`` widens the divergence guard so a floating-point sample that lands
    numerically *next to* the singularity is still reported as pathological
    instead of as a huge finite number.
    """
    gap = float(critical) - float(x)
    if abs(gap) <= float(epsilon):
        raise ZeroDivisionError(f"metric diverges at {critical!r} (gap={gap:.3e})")
    value = amplitude * float(x) / gap
    return {"value": float(value), "reference": 0.0,
            "flags": {"finite": math.isfinite(value)}}


def flag_gate(x: float, *, limit: float, signal: str) -> dict:
    """Pass the axis value through and report a mechanical ``signal`` flag."""
    value = float(x)
    return {"value": value, "reference": 0.0,
            "flags": {signal: value <= float(limit), "finite": math.isfinite(value)}}


def order_report(x: float, *, key: str) -> dict:
    """Report the axis value as an *observed convergence order* (collapse detector)."""
    value = float(x)
    return {"value": value, "reference": 0.0,
            "flags": {key: value, "finite": math.isfinite(value)}}


# ---------------------------------------------------------------------------
# reference metrics over this repository's neutral numerical kernels
# ---------------------------------------------------------------------------

def _numpy():
    import numpy as np

    return np


def stationary_solver_error(grid_n: float, *, solver: str, rhs: str, exact: str,
                            max_iter: int, tol: float) -> dict:
    """Relative ``L_inf`` error of a stationary solver against a closed-form solution.

    ``solver``/``rhs``/``exact`` are ``"module:callable"`` targets supplied by the
    configuration; ``grid_n`` is the axis.  The iteration budget and tolerance are
    part of the declaration: an error measured with a fixed budget *is* the
    quantity whose applicability we want to know.
    """
    np = _numpy()
    solve = import_target(solver)
    make_rhs = import_target(rhs)
    make_exact = import_target(exact)
    n = int(round(float(grid_n)))
    if n < 2:
        return {"value": float("nan"), "reference": 0.0,
                "flags": {"converged": False, "finite": False}, "n": n}
    f = np.asarray(make_rhs(n), dtype=float)
    result = solve(f, max_iter=int(max_iter), tol=float(tol))
    u = np.asarray(result["u"], dtype=float)
    reference = np.asarray(make_exact(n), dtype=float)
    scale = float(np.abs(reference).max())
    finite = bool(np.all(np.isfinite(u)))
    if not finite or scale == 0.0:
        return {"value": float("nan") if not finite else 0.0, "reference": 0.0,
                "flags": {"converged": bool(result.get("converged", False)), "finite": finite},
                "n": n}
    err = float(np.abs(u - reference).max()) / scale
    return {"value": err, "reference": 0.0,
            "flags": {"converged": bool(result.get("converged", False)),
                      "finite": True},
            "n": n,
            "iterations": int(result.get("iterations", 0)),
            "relative_residual": float(result["residual_history"][-1] / result["residual_history"][0])
            if result.get("residual_history") else None}


def stationary_iteration_rate(relaxation: float, *, sweeper: str, grid_n: int, sweeps: int,
                              seed: int) -> dict:
    """Per-sweep growth factor of a stationary iteration (power iteration, homogeneous problem).

    ``rho > 1`` means the iteration diverges: the classic stability boundary of a
    damped stationary scheme.
    """
    np = _numpy()
    sweep = import_target(sweeper)
    n = int(grid_n)
    h = 1.0 / n
    rng = np.random.default_rng(int(seed))
    u = np.zeros((n + 1, n + 1))
    u[1:-1, 1:-1] = rng.standard_normal((n - 1, n - 1))
    f = np.zeros_like(u)
    norms = [float(np.abs(u).max())]
    for _ in range(int(sweeps)):
        u = np.asarray(sweep(u, f, h, omega=float(relaxation), sweeps=1), dtype=float)
        norms.append(float(np.abs(u).max()))
        if not math.isfinite(norms[-1]):
            break
    tail = norms[-min(10, len(norms) - 1):]
    ratios = [tail[i + 1] / tail[i] for i in range(len(tail) - 1) if tail[i] > 0.0]
    if not ratios:
        return {"value": float("inf"), "reference": 1.0, "flags": {"finite": False}}
    if any(not math.isfinite(r) for r in ratios):
        return {"value": float("inf"), "reference": 1.0, "flags": {"finite": False}}
    rho = float(math.exp(sum(math.log(r) for r in ratios) / len(ratios)))
    return {"value": rho, "reference": 1.0,
            "flags": {"finite": math.isfinite(rho), "diverging": rho > 1.0}}


def multigrid_cycle_rate(damping: float, *, solver: str, history: str, n_fine: int,
                         n_coarse: int, cycles: int, seed: int, smoother: str) -> dict:
    """Asymptotic per-cycle contraction factor of a multigrid cycle vs. smoother damping."""
    solver_cls = import_target(solver)
    history_fn = import_target(history)
    instance = solver_cls(n_fine=int(n_fine), n_coarse=int(n_coarse), smoother=str(smoother),
                          omega=float(damping))
    result = history_fn(instance, n=int(n_fine), cycles=int(cycles), seed=int(seed))
    rho = result.get("rho_asymptotic")
    if rho is None:
        return {"value": float("nan"), "reference": 1.0,
                "flags": {"finite": False}, "keys": sorted(result)}
    rho = float(rho)
    return {"value": rho, "reference": 1.0,
            "flags": {"finite": math.isfinite(rho), "diverging": rho > 1.0},
            "n_fine": int(n_fine), "n_coarse": int(n_coarse)}