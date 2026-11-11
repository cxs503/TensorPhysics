r"""Ring-crack spacing and first-crack load of a bending-failing ice plate.

Physics
-------
A level ice sheet on a foundation fails in bending when the radial bending
moment reaches the flexural strength moment

``M_f = sigma_f h^2 / 6``   [N m/m]

(``sigma_f`` flexural strength, ``h`` ice thickness).  Because the moment field
of a concentrated or patch load oscillates while it decays, the ring cracks that
appear as the load grows are spaced by the *moment-extrema spacing* rather than
by an arbitrary length.  For the infinite plate on a Winkler foundation the
moment is a Kelvin (``kei``) function whose extrema sit at fixed multiples of the
characteristic length ``l = (D/k)^(1/4)``, so the spacing follows the scaling law

``s = C_s * l = C_s (D/k)^(1/4)``   =>   ``alpha = d ln s / d ln k = -1/4``.

That exponent is the "theoretical value" against which the numerical moment
fields (closed form and axisymmetric FEM) are checked here.  For the Pasternak
foundation the spacing is no longer a pure power of ``k`` (two spectral scales
``mu_1, mu_2`` enter); its *local* exponent is obtained from the analytic radii
by finite differences and used as the reference for the measured exponent.

Definitions used below (all machine-checkable)
----------------------------------------------
* extremum radii: local maxima of ``|M_r(r)|`` on a dense grid, refined by a
  parabolic fit (a ring crack nucleates at a moment extremum);
* crack spacing ``s``: mean distance between *consecutive* extremum radii
  inside the first ``n_keep`` extrema;
* first-crack load ``P_c``: ``M_f / max_r |M_r(r; P = 1)|`` — the load at which
  the strongest moment extremum reaches the strength moment;
* bearing-capacity ratio: ``P_c`` of the Pasternak / half-space model divided by
  the Winkler value (dimensionless, foundation-model sensitivity).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "flexural_strength_moment",
    "pasternak_ring_crack_period",
    "local_log_slope",
    "moment_extrema_radii",
    "crack_spacing",
    "spacing_scaling_exponent",
    "first_crack_load",
    "bearing_capacity_ratio",
]


def flexural_strength_moment(sigma_f: float, h: float) -> float:
    """Flexural strength moment ``M_f = sigma_f h^2 / 6`` [N m/m]."""
    sigma_f, h = float(sigma_f), float(h)
    if sigma_f <= 0.0 or h <= 0.0:
        raise ValueError("sigma_f and h must be positive")
    return sigma_f * h * h / 6.0


def pasternak_ring_crack_period(D: float, k: float, G_p: float) -> float:
    """Oscillation period ``2 pi / Im(mu)`` of the Pasternak plate moment [m].

    For ``G_p < 2 sqrt(D k)`` the spectral roots are complex conjugate pairs
    ``mu_1 = conj(mu_2)``; the moment and deflection fields then oscillate with
    the period ``2 pi/Im(mu_1)``, so ring-crack radii are spaced by a constant
    fraction of it.  Raising for the two-real-root regime (``G_p > 2 sqrt(D k)``)
    is deliberate: there the moment field is monotonic and no oscillatory crack
    pattern exists.
    """
    from .pasternak_plate import pasternak_plate_roots

    mu1, mu2 = pasternak_plate_roots(D, k, G_p)[2:]
    if abs(mu1.imag) < 1e-12 * abs(mu1):
        raise ValueError("no oscillatory regime (G_p >= 2 sqrt(D k)): the "
                         "moment field is monotonic, ring-crack spacing undefined")
    return float(2.0 * np.pi / abs(mu1.imag))


def local_log_slope(func, x0: float, eps: float = 1e-4) -> float:
    """Local exponent ``d ln f / d ln x`` at ``x0`` by central differences."""
    f_hi = float(func(x0 * (1.0 + eps)))
    f_lo = float(func(x0 * (1.0 - eps)))
    if f_hi <= 0.0 or f_lo <= 0.0:
        raise ValueError("func must be positive on both sides of x0")
    return float(np.log(f_hi / f_lo) / np.log((1.0 + eps) / (1.0 - eps)))


def moment_extrema_radii(moment, r_max: float, n: int = 4001,
                         n_keep: int | None = None):
    """Radii of the local maxima of ``|M_r(r)|`` on ``(0, r_max]``.

    Parameters
    ----------
    moment : callable
        ``moment(r) -> ndarray`` bending moment [N m/m]; ``r`` positive.
    r_max : float
        Upper radius of the scan [m].
    n : int
        Number of grid points.
    n_keep : int, optional
        Keep only the first ``n_keep`` extrema.

    Returns
    -------
    ndarray
        Extremum radii [m] in increasing order, refined by a parabolic fit
        through the three grid points around each maximum.
    """
    r_max = float(r_max)
    if r_max <= 0.0:
        raise ValueError("r_max must be positive")
    r = np.linspace(r_max / n, r_max, n)
    m = np.abs(np.asarray(moment(r), dtype=float))
    if m.shape != r.shape:
        raise ValueError("moment() must return one value per radius")
    idx = np.where((m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:]))[0] + 1
    if idx.size == 0:
        return np.empty(0)
    rr = []
    for i in idx:
        # parabolic vertex through (i-1, i, i+1)
        y0, y1, y2 = m[i - 1], m[i], m[i + 1]
        denom = (y0 - 2.0 * y1 + y2)
        shift = 0.0 if abs(denom) < 1e-300 else 0.5 * (y0 - y2) / denom
        dr = r[1] - r[0]
        rr.append(r[i] + shift * dr)
    rr = np.asarray(rr)
    if n_keep is not None:
        rr = rr[:int(n_keep)]
    return rr


def crack_spacing(moment, r_max: float, n: int = 4001, n_keep: int = 6) -> float:
    """Mean spacing between consecutive ring-crack radii [m]."""
    rr = moment_extrema_radii(moment, r_max, n=n, n_keep=n_keep)
    if rr.size < 2:
        raise ValueError("fewer than two moment extrema found; enlarge r_max")
    return float(np.mean(np.diff(rr)))


def spacing_scaling_exponent(values, spacings):
    """Least-squares exponent ``alpha`` in ``s = C * value^alpha``.

    ``alpha = d ln s / d ln value``; returns ``(alpha, C)``.
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(spacings, dtype=float)
    if v.size != s.size or v.size < 2:
        raise ValueError("need at least two (value, spacing) pairs")
    if np.any(v <= 0.0) or np.any(s <= 0.0):
        raise ValueError("values and spacings must be positive")
    lv = np.log(v)
    ls = np.log(s)
    alpha, lnC = np.polyfit(lv, ls, 1)
    return float(alpha), float(np.exp(lnC))


def first_crack_load(moment_per_load, M_f: float, r_max: float,
                     n: int = 4001) -> float:
    """Load at which the strongest moment extremum reaches ``M_f`` [N].

    ``moment_per_load(r)`` returns ``M_r`` for a unit load.
    """
    rr = np.linspace(r_max / n, r_max, n)
    m = np.abs(np.asarray(moment_per_load(rr), dtype=float))
    m_max = float(np.nanmax(m))
    if m_max <= 0.0:
        raise ValueError("moment field vanishes")
    return float(M_f) / m_max


def bearing_capacity_ratio(P_c_model: float, P_c_winkler: float) -> float:
    """``P_c(model)/P_c(Winkler)`` — sensitivity of the bearing capacity."""
    if float(P_c_winkler) <= 0.0:
        raise ValueError("Winkler bearing capacity must be positive")
    return float(P_c_model) / float(P_c_winkler)