"""Panelled quadrature for Hankel-type integrals on ``[0, inf)``.

The foundation Green functions of this sub-package are all of the form

``w(r) = int_0^inf  A(alpha) J_0(alpha r) d alpha``

with an algebraic decay ``A ~ alpha^(-p)`` (``p = 3`` for a two-parameter
foundation, ``p = 2`` for an elastic half-space).  Two length scales compete in
``A``: the algebraic features (poles/decay of the foundation kernel, widths from
``alpha ~ 1e-8`` up to ``alpha ~ 60 alpha_c``) and the Bessel oscillation of
period ``2 pi / max(r, a)``.  The range is therefore split into

* **geometrically graded panels** (ratio 1.1) from ``1e-8 alpha_max`` up to the
  wavenumber where the panel width would reach one twelfth of a Bessel period, so
  every decade of the algebraic kernel is resolved;
* **uniform panels** of width ``period/12`` from there to ``alpha_max``, so the
  oscillation is resolved;

each panel integrated with a 12-point Gauss-Legendre rule (machine precision on
smooth panels).  The remaining tail ``[alpha_max, inf)`` goes to adaptive
quadrature and its magnitude is reported, so callers (and tests) can prove that
``alpha_max`` was chosen large enough.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import quad

__all__ = ["oscillatory_quad", "OscillatoryResult"]


class OscillatoryResult(float):
    """``float`` result carrying the estimated tail magnitude as attribute."""

    tail: float = 0.0
    n_panel: int = 0


def oscillatory_quad(func, alpha_max: float, freq: float, n_per_period: int = 12,
                     growth: float = 1.1, n_panel_max: int = 200000,
                     tail: bool = True):
    """Integrate ``func`` over ``[0, alpha_max]`` (+ decaying tail).

    Parameters
    ----------
    func : callable
        Real-valued integrand ``f(alpha)``.
    alpha_max : float
        End of the panelled range [1/m]; choose it where the algebraic decay has
        made the integrand negligible.
    freq : float
        Wavenumber of the oscillatory factor of ``func`` [m]; the periodic part
        of the integrand oscillates with period ``2 pi / freq``.
    n_per_period : int
        Gauss-Legendre nodes per oscillation period (uniform panels).
    growth : float
        Panel ratio of the geometrically graded low-wavenumber range.
    tail : bool
        Add ``quad(func, alpha_max, inf)`` and record its magnitude.
    """
    alpha_max = float(alpha_max)
    freq = max(float(freq), 1e-300)
    if alpha_max <= 0.0:
        raise ValueError("alpha_max must be positive")
    period = 2.0 * np.pi / freq
    # panel width of the geometric range must stay below period/n_per_period
    alpha_t = min(alpha_max, period / (n_per_period * (growth - 1.0)))
    lo = max(1e-8 * alpha_max, 1e-12)
    edges = [lo]
    while edges[-1] < alpha_t and len(edges) < n_panel_max:
        edges.append(edges[-1] * growth)
    if edges[-1] < alpha_t:
        edges.append(alpha_t)
    if alpha_t < alpha_max:
        width = period / n_per_period
        n_uni = int(np.ceil((alpha_max - alpha_t) / width))
        n_uni = max(n_uni, 1)
        edges.extend(np.linspace(alpha_t, alpha_max, n_uni + 1)[1:].tolist())
    edges = np.asarray(edges, dtype=float)
    if edges.size > n_panel_max:
        raise RuntimeError("panel limit exceeded; reduce alpha_max or freq")

    nodes, weights = np.polynomial.legendre.leggauss(n_per_period)
    val = 0.0
    for lo_e, hi_e in zip(np.concatenate(([0.0], edges[:-1])), edges):
        half = 0.5 * (hi_e - lo_e)
        mid = 0.5 * (hi_e + lo_e)
        val += half * float(np.dot(weights,
                                   [func(mid + half * float(x)) for x in nodes]))
    tail_val = 0.0
    n_tail = 0
    if tail:
        # cheap magnitude bound for an (at least) alpha^-2 integrand; the
        # adaptive call is only made when the tail can matter, which keeps
        # QUADPACK away from slow oscillatory ranges it cannot resolve.
        bound = abs(func(alpha_max)) / alpha_max * 2.0
        if bound > 1e-9 * max(abs(val), 1e-300):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tail_val = quad(func, alpha_max, np.inf, limit=200)[0]
            n_tail = 1
    out = OscillatoryResult(val + tail_val)
    out.tail = float(abs(tail_val))
    out.tail_done = n_tail
    out.n_panel = int(edges.size)
    return out