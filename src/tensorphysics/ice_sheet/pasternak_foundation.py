r"""Pasternak (two-parameter) elastic foundation for floating ice plates.

Why this module
---------------
The level-ice bending-failure chain (:mod:`tensorphysics.ice_sheet.
kirchhoff_plate_fem`) historically rested the ice sheet on a *Winkler* (one
parameter) foundation ``p = k w``, i.e. an array of independent springs.  A
Winkler subgrade transmits no shear, so every point of the sheet answers only to
the load directly above it.  The physical deficiency is registered as
``GAP-FEM-P0``; the Pasternak model closes it by adding the shear-interaction
term

.. math::

    p(x, y) = k\\, w(x, y) - G_p \\nabla^2 w(x, y),

so that the plate equation becomes

.. math::

    D \\nabla^4 w - G_p \\nabla^2 w + k\\, w = q(x, y),
    \\qquad D = \\frac{E h^3}{12 (1 - \\nu^2)} ,

with ``w`` positive downwards, ``k`` the subgrade reaction modulus [Pa/m] and
``G_p`` the shear-layer modulus [N/m] (= shear modulus x layer thickness, a
force per unit length).  ``G_p = 0`` degenerates exactly to Winkler, and
``k = G_p = 0`` to the free (unfounded) plate.

Closed-form anchor (Hetényi / Vlasov-Kerr Green function)
--------------------------------------------------------
For an infinite plate the operator factors as
``D (lap^2 - m1)(lap^2 - m2)`` with ``m1, m2`` the roots of
``D m^2 - G_p m + k = 0``.  Using the 2-D fundamental solution
``(lap^2 - m) K_0(sqrt(m) r) = -2 pi delta(r)`` the point-load (Green) function is

.. math::

    w(r) = \\frac{P}{2\\pi\\sqrt{\\Delta}}
           \\big[ K_0(b r) - K_0(a r) \\big], \\quad
    a = \\sqrt{m_1},\\; b = \\sqrt{m_2},\\;
    \\Delta = G_p^2 - 4 D k .

The logarithmic singularities of the two Bessel terms cancel, so ``w(0)`` is
finite:

.. math::

    w(0) = \\mathrm{Re}\\Big[ \\frac{P}{2\\pi\\sqrt{\\Delta}} \\ln\\frac{a}{b} \\Big] .

Two regimes follow from the sign of ``Delta`` (equivalently of the dimensionless
shear coupling ``gamma = G_p / \\sqrt{D k}``):

* ``gamma < 2`` (``Delta < 0``): ``a`` and ``b`` are complex conjugates, the
  response is the damped-oscillatory Hetényi pattern (``K_0`` on an oblique
  complex argument -> Kelvin-type ``ker/kei`` when ``gamma = 0``);
* ``gamma > 2`` (``Delta > 0``): two real decay lengths, ``1/b`` being the long
  one that governs the far field; the shear layer spreads the load over
  ``~ \\sqrt{G_p / k}``.

``gamma = 0`` reproduces the classical Hetényi/Kelvin solution
``w = -(P l^2 / 2 pi D) kei(r/l)``, ``l = (D/k)^{1/4}`` used by
:func:`tensorphysics.ice_sheet.kirchhoff_plate_fem.kelvin_point_load`; this
module returns that function verbatim for ``G_p == 0`` so that the Winkler
degeneration is exact to the last bit.

Parameter calibration for ice
-----------------------------
``k`` comes from buoyancy, ``k = rho_w g`` (Archimedes restoring pressure per
unit deflection).  For the shear layer the usual soil-mechanics estimate
``G_p = G H`` (shear modulus x participating thickness) is applied to the ice
cover, ``G_ice = E / [2 (1 + nu)]``, ``G_p = beta G_ice h``.  The resulting
dimensionless coupling ``gamma = G_p / sqrt(D k)`` is *huge* for a 1 m ice sheet
(``gamma ~ 10^3`` at ``beta = 1``), i.e. the naive calibration overshoots the
regime where shear and bending compete (``gamma ~ 1``) by three orders of
magnitude.  :func:`ice_foundation_calibration` therefore reports ``gamma``, the
shear length ``sqrt(G_p/k)`` and the far-field decay length alongside the raw
``G_p``, and :func:`calibration_table` tabulates the sensitivity so the choice of
``beta`` is auditable instead of silent.  The physics caveat (open water cannot
carry static shear -> ``beta -> 0``; ``beta > 0`` stands for a compliant or
grounded subgrade: rubble, shoal, grounded ice) is discussed in the campaign
report.

References
----------
- Hetényi, M. (1946). *Beams on Elastic Foundation*. Univ. of Michigan Press.
- Pasternak, P.L. (1954). On a new method of analysis of an elastic foundation
  by means of two foundation constants. Gos. Izd. Lit. po Stroit. i Arkh.
- Vlasov, V.Z. & Leont'ev, N.N. (1966). *Beams, Plates and Shells on Elastic
  Foundations*. Israel Program for Scientific Translations.
- Kerr, A.D. (1975). The bearing capacity of floating ice plates subjected to
  static or quasi-static loads. CRREL Report 75-1.
- Selvadurai, A.P.S. (1979). *Elastic Analysis of Soil-Foundation Interaction*.
- Squire, V.A. et al. (1996). Of ocean waves and sea ice. *Annu. Rev. Fluid
  Mech.* 28, 115-168.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import k0 as _scipy_k0
from scipy.special import k1 as _scipy_k1

from .kirchhoff_plate_fem import kelvin_point_load, kelvin_point_load_moments

__all__ = [
    "PasternakFoundation",
    "pasternak_roots",
    "pasternak_point_load",
    "pasternak_point_load_moments",
    "pasternak_critical_point_load",
    "pasternak_patch_field",
    "pasternak_critical_patch_load",
    "pasternak_patch_load",
    "ice_foundation_calibration",
    "calibration_table",
    "shear_coupling",
    "bessel_k_reference",
    "WINKLER_COEFF",
]

#: Buoyancy modulus of sea water, ``rho_w g`` [Pa/m] (k for a floating sheet).
WINKLER_COEFF: float = 1025.0 * 9.81

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PasternakFoundation:
    """Two-parameter foundation ``p = k w - G_p lap^2 w`` [SI units].

    Attributes
    ----------
    k_w : float
        Subgrade reaction modulus [Pa/m]; must be positive (``k_w = 0`` is only
        meaningful as a limit, use the solver's ``foundation_model="none"``).
    g_p : float
        Shear-layer modulus [N/m]; ``g_p = 0`` degenerates to Winkler.  Must be
        non-negative: ``G_p < 0`` makes the foundation energy unbounded below.
    """

    k_w: float = WINKLER_COEFF
    g_p: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.k_w) or self.k_w <= 0.0:
            raise ValueError(f"k_w must be positive and finite, got {self.k_w!r}")
        if not np.isfinite(self.g_p) or self.g_p < 0.0:
            raise ValueError(f"g_p must be non-negative and finite, got {self.g_p!r}")

    @property
    def is_winkler(self) -> bool:
        """True when the shear layer is absent (exact Winkler degeneration)."""
        return self.g_p == 0.0

    def char_length(self, D: float) -> float:
        """Hetényi characteristic length ``l = (D / k_w)^{1/4}`` [m]."""
        if not np.isfinite(D) or D <= 0.0:
            raise ValueError(f"D must be positive and finite, got {D!r}")
        return (D / self.k_w) ** 0.25

    def shear_length(self) -> float:
        """Pasternak shear spreading length ``sqrt(G_p / k_w)`` [m] (0 for Winkler)."""
        if self.g_p == 0.0:
            return 0.0
        return math.sqrt(self.g_p / self.k_w)

    def coupling(self, D: float) -> float:
        """Dimensionless shear coupling ``gamma = G_p / sqrt(D k_w)`` [-].

        ``gamma = 0`` Winkler; ``gamma < 2`` complex (oscillatory) roots;
        ``gamma > 2`` real roots; ``gamma ~ 1`` is where the shear layer and the
        plate bending compete at the same length scale.
        """
        if not np.isfinite(D) or D <= 0.0:
            raise ValueError(f"D must be positive and finite, got {D!r}")
        return self.g_p / math.sqrt(D * self.k_w)


# ---------------------------------------------------------------------------
# roots of the foundation/plate dispersion relation
# ---------------------------------------------------------------------------
def pasternak_roots(D: float, k_w: float, g_p: float) -> Dict[str, object]:
    """Roots of ``D m^2 - G_p m + k = 0`` and the derived wave numbers.

    Returns a dict with ``m1, m2`` (complex), ``a = sqrt(m1)``, ``b = sqrt(m2)``
    (both with positive real part), the discriminant ``delta``, the regime tag
    (``"winkler"`` / ``"complex"`` / ``"real"``), the dimensionless coupling
    ``gamma`` and the ``decay_length_m`` = ``1 / Re(a)`` (slowest decay; the two
    roots share their real part in the complex regime).
    """
    if not np.isfinite(D) or D <= 0.0:
        raise ValueError(f"D must be positive and finite, got {D!r}")
    if not np.isfinite(k_w) or k_w <= 0.0:
        raise ValueError(f"k_w must be positive and finite, got {k_w!r}")
    if not np.isfinite(g_p) or g_p < 0.0:
        raise ValueError(f"g_p must be non-negative and finite, got {g_p!r}")
    delta = complex(g_p * g_p - 4.0 * D * k_w, 0.0)
    sq = np.sqrt(delta)
    if sq.real < 0.0:  # keep the branch with positive real part
        sq = -sq
    m1 = (g_p + sq) / (2.0 * D)
    m2 = (g_p - sq) / (2.0 * D)
    a = np.sqrt(m1)
    b = np.sqrt(m2)
    if a.real < 0.0:
        a = -a
    if b.real < 0.0:
        b = -b
    if g_p == 0.0:
        regime = "winkler"
    elif delta.real < 0.0:
        regime = "complex"
    else:
        regime = "real"
    return {
        "D": D,
        "k_w": k_w,
        "g_p": g_p,
        "delta": delta,
        "m1": m1,
        "m2": m2,
        "a": complex(a),
        "b": complex(b),
        "gamma": g_p / math.sqrt(D * k_w),
        "regime": regime,
        # far-field (slow) decay length: the smaller of the two wave numbers
        "decay_length_m": float(1.0 / min(a.real, b.real)),
        "shear_length_m": math.sqrt(g_p / k_w) if g_p > 0.0 else 0.0,
        "char_length_m": (D / k_w) ** 0.25,
    }


def shear_coupling(D: float, k_w: float, g_p: float) -> float:
    """Dimensionless Pasternak coupling ``G_p / sqrt(D k_w)`` (see :func:`pasternak_roots`)."""
    return float(pasternak_roots(D, k_w, g_p)["gamma"])


# ---------------------------------------------------------------------------
# complex-order Bessel K (mpmath fallback for non-real arguments)
# ---------------------------------------------------------------------------
_EULER_GAMMA = 0.57721566490153286060651209008240243
_Z_SERIES_MAX = 8.5   # ascending-series / asymptotic switch in |z|
_Z_TERMS = 16         # terms in the large-|z| asymptotic expansion


def _bessel_k_series(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ascending series for ``K_0`` and ``K_1`` for |z| <= 6 (vectorised).

    ``K_0(z) = -[ln(z/2)+gamma] I_0(z) + sum_{k>=1} (z^2/4)^k/(k!)^2 H_k``

    ``K_1 = -K_0'`` is differentiated term by term from the same series (this
    avoids the easily mis-transcribed DLMF 10.31.1 digamma sum):

    ``K_1(z) = I_0(z)/z + [ln(z/2)+gamma] I_1(z)
               - (2/z) sum_{k>=1} k (z^2/4)^k H_k/(k!)^2``

    with ``H_k`` the harmonic numbers and ``I_0, I_1`` the modified Bessel
    functions of the first kind (their own (absolutely convergent) series).
    """
    z = np.asarray(z, dtype=complex)
    t = 0.25 * z * z
    nmax = 64
    i0 = np.ones_like(z)
    i1 = 0.5 * z                   # k = 0 term of I_1
    s0 = np.zeros_like(z)
    s0k = np.zeros_like(z)
    t0 = np.ones_like(z)          # (z^2/4)^k/(k!)^2
    hk = 0.0
    for k in range(1, nmax):
        hk += 1.0 / k             # H_k
        t0 = t0 * t / (k * k)
        i0 = i0 + t0
        i1 = i1 + 0.5 * z * t0 / (k + 1)  # (z/2)^{2k+1}/(k!(k+1)!)
        s0 = s0 + t0 * hk
        s0k = s0k + k * t0 * hk
    log_half = np.log(0.5 * z)
    k0 = -(log_half + _EULER_GAMMA) * i0 + s0
    k1 = i0 / z + (log_half + _EULER_GAMMA) * i1 - 2.0 * s0k / z
    return k0, k1


def _bessel_k_asymptotic(z: np.ndarray, nu: int) -> np.ndarray:
    """Large-|z| expansion ``K_nu(z) ~ sqrt(pi/2z) e^-z sum_k a_k/z^k`` (DLMF 10.40.2)."""
    z = np.asarray(z, dtype=complex)
    mu = 4.0 * nu * nu
    total = np.ones_like(z)
    term = np.ones_like(z)
    for k in range(1, _Z_TERMS + 1):
        term = term * (mu - (2 * k - 1) ** 2) / (8.0 * k * z)
        total = total + term
    return np.sqrt(np.pi / (2.0 * z)) * np.exp(-z) * total


def _bessel_k0_k1(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``K_0`` and ``K_1`` for a (possibly complex) array, vectorised.

    Real arguments use :mod:`scipy.special`; complex arguments (oblique Kelvin
    arguments arising for ``0 < gamma < 2``) use the ascending series below
    ``|z| = 6`` and the asymptotic expansion above it.  Accuracy is ~1e-10
    relative, verified against an ``mpmath`` reference
    (:func:`bessel_k_reference`) in ``tests/test_pasternak_foundation.py``.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=complex))
    out0 = np.empty(x.shape, dtype=complex)
    out1 = np.empty(x.shape, dtype=complex)
    if x.size == 0:
        return out0, out1
    real = np.abs(x.imag) <= 1e-13 * np.maximum(np.abs(x.real), 1e-300)
    if np.all(real):
        xr = x.real
        if np.any(xr <= 0.0):
            raise ValueError("Bessel K arguments must have positive real part")
        out0[...] = _scipy_k0(xr)
        out1[...] = _scipy_k1(xr)
        return out0, out1
    if np.any(x.real <= 0.0):
        raise ValueError("Bessel K arguments must have positive real part")
    small = np.abs(x) <= _Z_SERIES_MAX
    if np.any(small):
        k0s, k1s = _bessel_k_series(x[small])
        out0[small], out1[small] = k0s, k1s
    big = ~small
    if np.any(big):
        out0[big] = _bessel_k_asymptotic(x[big], 0)
        out1[big] = _bessel_k_asymptotic(x[big], 1)
    return out0, out1


def bessel_k_reference(x, order: int = 0) -> np.ndarray:
    """``mpmath`` reference for ``K_order`` (30 significant digits; test anchor).

    Slow (about 15 ms per point) and therefore only used to certify the fast
    vectorised :func:`_bessel_k0_k1` kernel, never in a production solve.
    """
    import mpmath as mp

    arr = np.atleast_1d(np.asarray(x, dtype=complex))
    out = np.empty(arr.shape, dtype=complex)
    with mp.workdps(30):
        for idx, val in np.ndenumerate(arr):
            out[idx] = complex(mp.besselk(int(order), mp.mpc(val.real, val.imag)))
    return out


# ---------------------------------------------------------------------------
# Green function: point load on an infinite Pasternak plate
# ---------------------------------------------------------------------------
def pasternak_point_load(r, P: float, D: float, k_w: float, g_p: float) -> np.ndarray:
    """Hetényi/Vlasov closed-form deflection of an infinite plate under a point load.

    ``D lap^4 w - G_p lap^2 w + k w = P delta(r)`` for a total load ``P`` [N];
    ``w`` is positive downwards [m].  ``g_p = 0`` returns
    :func:`tensorphysics.ice_sheet.kirchhoff_plate_fem.kelvin_point_load`
    unchanged (exact Winkler degeneration).
    """
    if not np.isfinite(P):
        raise ValueError("P must be finite")
    if g_p == 0.0:
        return kelvin_point_load(r, P, D, k_w)
    roots = pasternak_roots(D, k_w, g_p)
    a, b = roots["a"], roots["b"]
    delta = roots["delta"]
    C = P / (_TWO_PI * np.sqrt(delta))
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    if np.any(rr < 0.0):
        raise ValueError("r must be non-negative")
    out = np.empty(rr.shape, dtype=complex)
    positive = rr > 0.0
    if np.any(positive):
        rp = rr[positive]
        k0b, _ = _bessel_k0_k1(b * rp)
        k0a, _ = _bessel_k0_k1(a * rp)
        out[positive] = C * (k0b - k0a)
    if np.any(~positive):
        out[~positive] = C * np.log(a / b)  # finite limit at the load point
    out = np.real(out)
    return out if out.size > 1 else float(out[0])


def pasternak_point_load_moments(
    r, P: float, D: float, k_w: float, g_p: float, nu: float = 0.3
) -> Dict[str, np.ndarray]:
    """Radial/circumferential moment field of a point load on a Pasternak plate.

    With ``w = C [K_0(b r) - K_0(a r)]``, ``C = P / (2 pi sqrt(delta))`` and the
    Bessel identities ``K_0' = -K_1``, ``K_0'' = K_0 + K_1/x``:

        w,r  = C [a K_1(a r) - b K_1(b r)]
        w,rr = C [a^2 (K_0(ar) + K_1(ar)/(ar)) - b^2 (K_0(br) + K_1(br)/(br))]
        M_r  = -D (w,rr + nu w,r / r),    M_t = -D (w,r / r + nu w,rr)

    ``M > 0`` sagging, ``M < 0`` hogging (top-surface tension = ring-crack
    driver).  ``g_p = 0`` reproduces
    :func:`...kirchhoff_plate_fem.kelvin_point_load_moments`.
    """
    if g_p == 0.0:
        return kelvin_point_load_moments(r, P, D, k_w, nu)
    roots = pasternak_roots(D, k_w, g_p)
    a, b = roots["a"], roots["b"]
    C = P / (_TWO_PI * np.sqrt(roots["delta"]))
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    if np.any(rr <= 0.0):
        raise ValueError("r must be positive (the moment field is evaluated off the load point)")
    k0a, k1a = _bessel_k0_k1(a * rr)
    k0b, k1b = _bessel_k0_k1(b * rr)
    w = np.real(C * (k0b - k0a))
    w_r = np.real(C * (a * k1a - b * k1b))
    w_rr = np.real(
        C * (b * b * (k0b + k1b / (b * rr)) - a * a * (k0a + k1a / (a * rr)))
    )
    return {
        "w": w,
        "w_r": w_r,
        "w_rr": w_rr,
        "M_r": -D * (w_rr + nu * w_r / rr),
        "M_t": -D * (w_r / rr + nu * w_rr),
    }


def pasternak_critical_point_load(
    M_f: float,
    D: float,
    k_w: float,
    g_p: float,
    nu: float = 0.3,
    r_lo: float = 0.02,
    r_hi: float = 8.0,
    n: int = 8001,
) -> Dict[str, float]:
    """Point load at which the first ring crack forms on a Pasternak plate.

    The response is linear in ``P``, so the critical load is
    ``P_crit = M_f / |min_r M_r(P=1)|``.  Returns also the radius of the peak
    hogging moment (normalised by the characteristic length ``l``), the peak
    moment per unit load and the foundation diagnostics.
    """
    if not np.isfinite(M_f) or M_f <= 0.0:
        raise ValueError("M_f must be positive and finite")
    l = (D / k_w) ** 0.25
    rr = np.linspace(r_lo * l, r_hi * l, int(n))
    mom = pasternak_point_load_moments(rr, 1.0, D, k_w, g_p, nu)
    i = int(np.argmin(mom["M_r"]))
    m_peak = float(mom["M_r"][i])
    roots = pasternak_roots(D, k_w, g_p)
    return {
        "P_crit_N": float(M_f / abs(m_peak)),
        "r_peak_m": float(rr[i]),
        "r_peak_over_l": float(rr[i] / l),
        "M_peak_per_unit_load": m_peak,
        "w_center_per_unit_load_m_per_N": float(
            np.real(
                (1.0 / (_TWO_PI * np.sqrt(roots["delta"])))
                * np.log(roots["a"] / roots["b"])
            )
        ),
        "char_length_m": float(l),
        "decay_length_m": float(roots["decay_length_m"]),
        "gamma": float(roots["gamma"]),
        "regime": roots["regime"],
        # Diagnostics: for every G_p >= 0 the *hogging* moment has an interior
        # minimum (finite first ring-crack load), whereas the *sagging* moment
        # diverges logarithmically at r = 0 (+inf, top/bottom surface tension
        # driver).  An argmin on the search boundary would flag a degenerate
        # case; it does not occur for the admissible parameter range.
        "interior_minimum": bool(0 < i < rr.size - 1),
        "argmin_at_search_bound": bool(i == 0 or i == rr.size - 1),
    }


def pasternak_patch_load(
    r,
    P: float,
    radius: float,
    D: float,
    k_w: float,
    g_p: float,
    n_radial: int = 32,
    n_theta: int = 48,
) -> np.ndarray:
    """Deflection of a Pasternak plate under a uniform pressure on a disc.

    ``P`` is the *total* load [N] spread uniformly over a disc of ``radius`` [m]
    (a ship-bow contact patch): ``w(x) = q ∫∫ G(|x - x'|) dA'`` evaluated with a
    polar Gauss-Legendre rule (the Green function is only logarithmically
    singular, so the quadrature converges).  For ``radius -> 0`` this tends to
    :func:`pasternak_point_load`; it is used as a non-singular cross-check of the
    FEM patch load.
    """
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    if not np.isfinite(P) or P == 0.0:
        raise ValueError("P must be finite and non-zero")
    q = P / (math.pi * radius * radius)
    gr, gw = np.polynomial.legendre.leggauss(int(n_radial))
    gt, gwt = np.polynomial.legendre.leggauss(int(n_theta))
    s = 0.5 * (gr + 1.0)                     # (0, 1)
    ws = 0.5 * gw * radius                   # dr' = radius/2 dxi
    th = math.pi * (gt + 1.0)                # (0, 2 pi)
    wth = math.pi * gwt
    rp = radius * s                          # radial stations
    dA = (rp[:, None] * ws[:, None]) * wth[None, :]  # r dr dtheta
    xp = rp[:, None] * np.cos(th)[None, :]
    yp = rp[:, None] * np.sin(th)[None, :]
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.zeros(rr.shape)
    for i, ri in enumerate(rr):
        d = np.hypot(ri - xp, yp)
        g = np.atleast_1d(pasternak_point_load(d.ravel(), 1.0, D, k_w, g_p))
        out[i] = q * float(np.sum(g.reshape(d.shape) * dA))
    return out if out.size > 1 else float(out[0])


# ---------------------------------------------------------------------------
# finite contact patch: ring-integrated Green function (no singularity)
# ---------------------------------------------------------------------------
def _bessel_i0_i1(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``I_0``, ``I_1`` by their (absolutely convergent) ascending series."""
    z = np.asarray(z, dtype=complex)
    t = 0.25 * z * z
    i0 = np.ones_like(z)
    i1 = 0.5 * z
    t0 = np.ones_like(z)
    for k in range(1, 96):
        t0 = t0 * t / (k * k)
        i0 = i0 + t0
        i1 = i1 + 0.5 * z * t0 / (k + 1)
    return i0, i1


def _ring_green(a: complex, r: np.ndarray, rp: np.ndarray) -> Tuple[np.ndarray, ...]:
    r"""Ring-averaged kernel ``(1/2pi) int_0^2pi K_0(a |x - x'|) dtheta`` and derivatives.

    Uses the Graf addition theorem for the axisymmetric average

    .. math:: \frac{1}{2\pi}\int_0^{2\pi} K_0(a\rho)\,d\theta
              = I_0(a r_<) K_0(a r_>), \quad
              \rho = \sqrt{r^2 + r'^2 - 2 r r' \cos\theta},

    with ``I_0' = I_1``, ``I_0'' = I_0 - I_1/z``, ``K_0' = -K_1``,
    ``K_0'' = K_0 + K_1/z``.  ``r`` (field) and ``rp`` (source) broadcast; the
    inner/outer branch is selected elementwise, so the source quadrature never
    touches the singular point.
    """
    r = np.asarray(r, dtype=float)
    rp = np.asarray(rp, dtype=float)
    smaller = r < rp
    r_lo = np.where(smaller, r, rp)
    r_hi = np.where(smaller, rp, r)
    a_lo = a * r_lo
    a_hi = a * r_hi
    i0_lo, i1_lo = _bessel_i0_i1(a_lo)
    k0_hi, k1_hi = _bessel_k0_k1(a_hi)
    safe_lo = np.where(np.abs(a_lo) > 0.0, a_lo, 1.0)
    base = i0_lo * k0_hi                       # value of the ring average
    # d/dr of the field radial coordinate
    d_inner = a * i1_lo * k0_hi                # r < rp : d/dr I_0(ar)
    d_outer = -a * i0_lo * k1_hi               # r > rp : d/dr K_0(ar)
    d1 = np.where(smaller, d_inner, d_outer)
    dd_inner = a * a * (i0_lo - i1_lo / safe_lo) * k0_hi
    dd_outer = a * a * i0_lo * (k0_hi + k1_hi / np.where(np.abs(a_hi) > 0.0, a_hi, 1.0))
    d2 = np.where(smaller, dd_inner, dd_outer)
    return base, d1, d2


def pasternak_patch_field(
    P: float,
    radius: float,
    D: float,
    k_w: float,
    g_p: float,
    nu: float = 0.3,
    r_max: Optional[float] = None,
    n_field: int = 601,
    n_source: int = 48,
) -> Dict[str, np.ndarray]:
    """Axisymmetric deflection/moment field of a *uniform disc* load (Pasternak plate).

    A total load ``P`` [N] is spread uniformly over a disc of ``radius`` [m]
    (ship-bow contact patch).  The field follows from the ring-averaged Green
    function of :func:`_ring_green`, so the logarithmic point-load singularity is
    integrated out.  This matters for the *sagging* (bottom-tension) moment,
    which diverges at a mathematical point load; the hogging first ring-crack
    load is controlled by an interior moment peak and therefore agrees with the
    point-load result to a few 1e-4 (see the campaign report).

    Returns ``r, w, w_r, w_rr, M_r, M_t`` on a uniform radial grid
    ``0 < r <= r_max`` (default ``8 l``); ``M > 0`` is sagging.
    """
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    if not np.isfinite(P):
        raise ValueError("P must be finite")
    l = (D / k_w) ** 0.25
    r_max = 8.0 * l if r_max is None else float(r_max)
    if r_max <= 0.0:
        raise ValueError("r_max must be positive")
    roots = pasternak_roots(D, k_w, g_p)
    a, b = roots["a"], roots["b"]
    C1 = 1.0 / (_TWO_PI * np.sqrt(roots["delta"]))     # unit point load
    q = P / (math.pi * radius * radius)
    r = np.linspace(r_max / n_field, r_max, int(n_field))
    gx, gw = np.polynomial.legendre.leggauss(int(n_source))
    rp = 0.5 * radius * (gx + 1.0)                    # source stations (0, Rc)
    wp = 0.5 * radius * gw * rp                       # r' dr'
    R = r[:, None]
    P_ = rp[None, :]
    kb0, kb1, kb2 = _ring_green(b, R, P_)
    ka0, ka1, ka2 = _ring_green(a, R, P_)
    ring = (kb0 - ka0)
    ring_r = (kb1 - ka1)
    ring_rr = (kb2 - ka2)
    fac = (_TWO_PI * C1 * wp)[None, :]
    w = q * np.sum(ring * fac, axis=1).real
    w_r = q * np.sum(ring_r * fac, axis=1).real
    w_rr = q * np.sum(ring_rr * fac, axis=1).real
    M_r = -D * (w_rr + nu * w_r / r)
    M_t = -D * (w_r / r + nu * w_rr)
    return {
        "r": r,
        "w": w,
        "w_r": w_r,
        "w_rr": w_rr,
        "M_r": M_r,
        "M_t": M_t,
        "radius_m": float(radius),
        "gamma": float(roots["gamma"]),
        "regime": roots["regime"],
    }


def pasternak_critical_patch_load(
    M_f: float,
    D: float,
    k_w: float,
    g_p: float,
    radius: float,
    nu: float = 0.3,
    r_max: Optional[float] = None,
    n_field: int = 601,
    n_source: int = 48,
) -> Dict[str, float]:
    """First-crack load of a plate on a Pasternak foundation under a disc load.

    ``P_crit = M_f / max_r [-M_r]`` with ``-M_r`` the hogging (top-tension) peak
    of the finite-disc field (:func:`pasternak_patch_field`), i.e. the physical
    contact-pressure problem rather than the singular point idealisation.
    Returns the peak location, the peak moment per unit load and the foundation
    diagnostics; for the admissible range the result agrees with
    :func:`pasternak_critical_point_load` to ~1e-3.
    """
    if not np.isfinite(M_f) or M_f <= 0.0:
        raise ValueError("M_f must be positive and finite")
    field = pasternak_patch_field(
        1.0, radius, D, k_w, g_p, nu=nu, r_max=r_max, n_field=n_field, n_source=n_source
    )
    hog = -field["M_r"]
    i = int(np.argmax(hog))
    peak = float(hog[i])
    l = (D / k_w) ** 0.25
    roots = pasternak_roots(D, k_w, g_p)
    return {
        "P_crit_N": float(M_f / peak),
        "M_hog_peak_per_unit_load": peak,
        "r_peak_m": float(field["r"][i]),
        "r_peak_over_l": float(field["r"][i] / l),
        "r_peak_over_radius": float(field["r"][i] / radius),
        "radius_m": float(radius),
        "gamma": float(roots["gamma"]),
        "regime": roots["regime"],
        "w_center_m_per_unit_load": float(field["w"][0]),
    }


# ---------------------------------------------------------------------------
# ice parameter calibration
# ---------------------------------------------------------------------------
def ice_foundation_calibration(
    h: float,
    E: float = 5.0e9,
    nu: float = 0.3,
    rho_w: float = 1025.0,
    g: float = 9.81,
    beta: float = 1.0,
) -> Dict[str, float]:
    """Calibrated ``(k, G_p)`` for an ice sheet on a Pasternak foundation.

    ``k = rho_w g`` (buoyancy) and ``G_p = beta * G_ice * h`` with
    ``G_ice = E / [2 (1 + nu)]`` the ice shear modulus, following the standard
    Pasternak soil estimate ``G_p = G H`` with ``H = h``.  ``beta`` is an
    explicit, auditable calibration factor (``beta = 0`` == Winkler).

    Returns a dict with the inputs, ``D``, ``k_w``, ``g_p``, ``G_ice``, the
    characteristic/shear/decay lengths, the dimensionless coupling ``gamma`` and
    the root regime.
    """
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(f"h must be positive and finite, got {h!r}")
    if not np.isfinite(E) or E <= 0.0:
        raise ValueError(f"E must be positive and finite, got {E!r}")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError(f"beta must be non-negative and finite, got {beta!r}")
    D = E * h**3 / (12.0 * (1.0 - nu * nu))
    k_w = rho_w * g
    G_ice = E / (2.0 * (1.0 + nu))
    g_p = beta * G_ice * h
    roots = pasternak_roots(D, k_w, g_p)
    return {
        "h_m": float(h),
        "E_Pa": float(E),
        "nu": float(nu),
        "rho_w_kg_m3": float(rho_w),
        "g_m_s2": float(g),
        "beta": float(beta),
        "D_Nm": float(D),
        "G_ice_Pa": float(G_ice),
        "k_w_Pa_per_m": float(k_w),
        "g_p_N_per_m": float(g_p),
        "char_length_m": float(roots["char_length_m"]),
        "shear_length_m": float(roots["shear_length_m"]),
        "decay_length_m": float(roots["decay_length_m"]),
        "gamma": float(roots["gamma"]),
        "regime": roots["regime"],
        "w_center_over_P_m_per_N": float(
            np.real(
                (1.0 / (_TWO_PI * np.sqrt(roots["delta"])))
                * np.log(roots["a"] / roots["b"])
            )
        ),
    }


def calibration_table(
    h: float,
    betas: Sequence[float] = (0.0, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 0.1, 0.3, 1.0),
    E: float = 5.0e9,
    nu: float = 0.3,
    rho_w: float = 1025.0,
    g: float = 9.81,
    P: float = 1.0e6,
    sigma_f: float = 0.7e6,
    contact_radius: float = 0.5,
) -> List[Dict[str, object]]:
    """Sensitivity table ``beta -> (G_p, gamma, lengths, w(0), P_crit)``.

    Each row is a :func:`ice_foundation_calibration` result plus the unit-load
    centre deflection ``w(0)`` [m per N], its ratio to the Winkler value, and the
    first-crack load ``P_crit`` of a disc load of ``contact_radius`` [m] with its
    ratio/deviation against Winkler (``M_f = sigma_f h^2 / 6``).  Ratios make the
    shear-layer effect comparable across ``beta`` without any hand-copied number.
    """
    base = ice_foundation_calibration(h, E=E, nu=nu, rho_w=rho_w, g=g, beta=0.0)
    M_f = sigma_f * h * h / 6.0
    p_crit_w = float(
        pasternak_critical_patch_load(
            M_f, base["D_Nm"], base["k_w_Pa_per_m"], 0.0, contact_radius, nu
        )["P_crit_N"]
    )
    w0_w = float(base["w_center_over_P_m_per_N"])
    rows: List[Dict[str, object]] = []
    for beta in betas:
        cal = ice_foundation_calibration(h, E=E, nu=nu, rho_w=rho_w, g=g, beta=beta)
        D, k_w, g_p = cal["D_Nm"], cal["k_w_Pa_per_m"], cal["g_p_N_per_m"]
        crit = pasternak_critical_patch_load(M_f, D, k_w, g_p, contact_radius, nu)
        w0 = cal["w_center_over_P_m_per_N"]
        p_crit = float(crit["P_crit_N"])
        rows.append(
            {
                "beta": float(beta),
                "g_p_N_per_m": g_p,
                "gamma": cal["gamma"],
                "regime": cal["regime"],
                "shear_length_m": cal["shear_length_m"],
                "decay_length_m": cal["decay_length_m"],
                "char_length_m": cal["char_length_m"],
                "w_center_per_unit_load_m_per_N": w0,
                "w_center_ratio_vs_winkler": w0 / w0_w,
                "centre_deflection_m_at_P": float(w0 * P),
                "P_crit_N": p_crit,
                "P_crit_ratio_vs_winkler": p_crit / p_crit_w,
                "P_crit_deviation_vs_winkler_pct": 100.0 * (p_crit / p_crit_w - 1.0),
                "r_peak_over_l": float(crit["r_peak_over_l"]),
                "contact_radius_m": float(contact_radius),
            }
        )
    return rows


def _spearman(xs: Iterable[float], ys: Iterable[float]) -> float:
    """Spearman rank correlation without a scipy.stats dependency."""
    x = np.asarray(list(xs), dtype=float)
    y = np.asarray(list(ys), dtype=float)

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(v.size, dtype=float)
        ranks[order] = np.arange(v.size, dtype=float)
        # average ties
        vals, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        for k, c in enumerate(counts):
            if c > 1:
                ranks[inv == k] = ranks[inv == k].mean()
        return ranks

    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float(np.sum(rx * rx)) * float(np.sum(ry * ry)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)