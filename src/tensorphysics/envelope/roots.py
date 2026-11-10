"""One-dimensional root finding: bracketing, bisection, Brent.

Pure standard library so the engine has no runtime dependency.  ``brent`` is the
classic inverse-quadratic-interpolation + secant + bisection hybrid (zeroin);
it is only ever called on a *bracketed* sign change, so the returned root can be
driven to machine precision -- which is what the migration-equivalence criterion
(engine boundary vs closed-form boundary, relative difference ~1e-15) relies on.

The reported ``iterations`` and ``converged`` flag are part of the evidence
chain: a silently unconverged root must be visible, never averaged away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

__all__ = [
    "RootResult",
    "bisect",
    "bracket",
    "brent",
    "linspace",
    "refine_bracket",
]

DEFAULT_RTOL = 4.0 * 2.220446049250313e-16  # ~ 4 eps
DEFAULT_MAXITER = 200


@dataclass(frozen=True)
class RootResult:
    x: float
    residual: float
    iterations: int
    converged: bool
    method: str

    def to_dict(self) -> dict[str, object]:
        return {
            "x": self.x,
            "residual": self.residual,
            "iterations": self.iterations,
            "converged": self.converged,
            "method": self.method,
        }


def linspace(lo: float, hi: float, n: int) -> tuple[float, ...]:
    if n < 2:
        raise ValueError("linspace needs n >= 2")
    step = (hi - lo) / (n - 1)
    return tuple(lo + i * step for i in range(n))


def bisect(
    f: Callable[[float], float],
    a: float,
    b: float,
    rtol: float = DEFAULT_RTOL,
    maxiter: int = DEFAULT_MAXITER,
) -> RootResult:
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return RootResult(a, fa, 0, True, "bisect")
    if fb == 0.0:
        return RootResult(b, fb, 0, True, "bisect")
    if (fa > 0.0) == (fb > 0.0):
        return RootResult(a, fa, 0, False, "bisect")
    lo, hi, f_lo = (a, b, fa) if fa < 0.0 else (b, a, fb)
    for iteration in range(1, maxiter + 1):
        x = 0.5 * (lo + hi)
        fx = f(x)
        if fx == 0.0:
            return RootResult(x, fx, iteration, True, "bisect")
        if (fx < 0.0) == (f_lo < 0.0):
            lo, f_lo = x, fx
        else:
            hi = x
        if abs(hi - lo) <= rtol * max(abs(lo), abs(hi), 1.0) or fx == 0.0:
            break
    x = 0.5 * (lo + hi)
    fx = f(x)
    converged = abs(hi - lo) <= 8.0 * rtol * max(abs(lo), abs(hi), 1.0)
    return RootResult(x, fx, maxiter, converged, "bisect")


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    rtol: float = DEFAULT_RTOL,
    maxiter: int = DEFAULT_MAXITER,
) -> RootResult:
    """Brent's method on a bracket ``[a, b]`` with ``f(a) * f(b) <= 0``."""

    fa, fb = f(a), f(b)
    if fa == 0.0:
        return RootResult(a, fa, 0, True, "brent")
    if fb == 0.0:
        return RootResult(b, fb, 0, True, "brent")
    if (fa > 0.0) == (fb > 0.0):
        return RootResult(a, fa, 0, False, "brent")
    c, fc = a, fa
    d = e = b - a
    for iteration in range(1, maxiter + 1):
        if (fb > 0.0) == (fc > 0.0):
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2.0 * rtol * max(abs(b), 1.0) + 1e-300
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0.0:
            return RootResult(b, fb, iteration, True, "brent")
        if abs(e) < tol or abs(fa) <= abs(fb):
            d = e = m
        else:
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m
        a, fa = b, fb
        b = b + (d if abs(d) > tol else (tol if m > 0.0 else -tol))
        fb = f(b)
    return RootResult(
        b, fb, maxiter, abs(0.5 * (c - b)) <= 8.0 * rtol * max(abs(b), 1.0), "brent"
    )


def bracket(
    f: Callable[[float], float], xs: Sequence[float], tol: float = 0.0
) -> tuple[tuple[float, float], ...]:
    """Sign-change brackets on an ordered sample list.

    A sample whose value is within ``tol`` of zero counts as a root and is
    returned as a degenerate bracket ``(x, x)`` so a boundary that lands exactly
    on a scan node is never lost.
    """

    values = [f(x) for x in xs]
    brackets: list[tuple[float, float]] = []
    for i in range(len(xs) - 1):
        fa, fb = values[i], values[i + 1]
        if abs(fa) <= tol:
            # a root sitting exactly on a node: report it once, as a degenerate
            # bracket, and never as an endpoint of the neighbouring segments
            brackets.append((xs[i], xs[i]))
            continue
        if abs(fb) <= tol:
            continue
        if (fa < 0.0) != (fb < 0.0):
            brackets.append((xs[i], xs[i + 1]))
    if xs and abs(values[-1]) <= tol:
        brackets.append((xs[-1], xs[-1]))
    return tuple(brackets)


def refine_bracket(
    f: Callable[[float], float],
    a: float,
    b: float,
    rtol: float = DEFAULT_RTOL,
    maxiter: int = DEFAULT_MAXITER,
    fallback: bool = True,
) -> RootResult:
    """Brent first; if it fails to converge, bisection on the same bracket."""

    if a == b:
        return RootResult(a, f(a), 0, True, "degenerate")
    result = brent(f, a, b, rtol=rtol, maxiter=maxiter)
    if result.converged or not fallback:
        return result
    return bisect(f, a, b, rtol=rtol, maxiter=maxiter)


def assert_finite(value: float, what: str = "value") -> float:
    if not math.isfinite(value):
        raise ValueError(f"{what} is not finite ({value!r})")
    return value