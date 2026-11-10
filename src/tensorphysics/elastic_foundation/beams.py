r"""Infinite beams on Winkler / Pasternak elastic foundations.

Governing equation (Euler-Bernoulli beam, deflection ``w`` positive downwards,
distributed load ``q`` positive downwards)::

    EI w'''' - G_p w'' + k w = q(x)

``G_p = 0`` recovers the Winkler (one-parameter, independent springs) model;
``k = G_p = 0`` the free beam.  ``EI`` is the bending stiffness [N m^2],
``k`` the subgrade modulus [Pa/m] = [N/m^2], ``G_p`` the shear-layer modulus
[N/m] (= shear modulus x layer thickness).

Closed-form Green function (point load ``P`` at the origin)
-----------------------------------------------------------
Factoring :math:`EI (D^2-s_1)(D^2-s_2)` with ``s_1, s_2`` the roots of
``EI s^2 - G_p s + k = 0`` and taking the branches ``a = sqrt(s_1)``,
``b = sqrt(s_2)`` with positive real part, the decaying Green function of
``(D^2-s_1)(D^2-s_2) w = P/EI delta(x)`` is

.. math::

    w(x) = \frac{P}{2\,EI\,(b^2-a^2)}
           \left[\frac{e^{-a|x|}}{a} - \frac{e^{-b|x|}}{b}\right] .

For ``G_p = 0`` the roots are purely imaginary, ``a = (1+i)/l_b``,
``b = conj(a)`` with ``l_b = (4 EI/k)^{1/4}``, and the expression above reduces
to Hetényi's classical infinite-beam solution

.. math::

    w(x) = \frac{P}{2 k l_b}\, e^{-|x|/l_b}
           \left[\cos(|x|/l_b) + \sin(|x|/l_b)\right],

which the test-suite checks to machine precision.  The *cosine-only* variant
``w = P/(2 k l_b) e^{-|x|/l_b} cos(x/l_b)`` quoted in some handbooks satisfies
the homogeneous equation but carries only **half** the point-load source
strength (its shear jump is ``P/(2EI)`` instead of ``P/EI``); this is checked by
:func:`point_load_shear_jump` and reported as a falsified formula in the
verification report.

Finite element cross-check
--------------------------
Two-node Hermite cubic elements (``w``, ``theta = w'`` per node) with the
consistent foundation operators

* Winkler:  ``K_k = k * le/420 * [156, 22le, 54, -13le; ...]``
* Pasternak: ``K_G = G_p/(30 le) * [36, 3le, -36, 3le; ...]``

(the second being the standard consistent matrix for ``int (w')^2 dx``), free
end conditions at ``x = +-L`` and a nodal point load.

References
----------
* Hetényi, M. (1946). *Beams on Elastic Foundation*. Univ. of Michigan Press.
* Selvadurai, A.P.S. (2000). *Partial Differential Equations in Mechanics 2*,
  and *Elastic Analysis of Soil-Foundation Interaction* — infinite beam on a
  two-parameter (Pasternak/Vlasov) foundation.
* Kerr, A.D. (1975). The bearing capacity of floating ice plates subjected to
  static or quasi-static loads. CRREL Report 75-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "beam_foundation_roots",
    "hetenyi_beam_point_load",
    "cosine_only_beam_point_load",
    "pasternak_beam_point_load",
    "point_load_shear_jump",
    "beam_char_length",
    "solve_beam_foundation_fem",
    "BeamFEMResult",
]


# ---------------------------------------------------------------------------
# analytical Green functions
# ---------------------------------------------------------------------------
def beam_foundation_roots(EI: float, k: float, G_p: float = 0.0):
    """Complex wavenumbers ``a, b`` (Re > 0) of the Pasternak beam operator.

    ``s_1, s_2`` solve ``EI s^2 - G_p s + k = 0``; ``a = sqrt(s_1)``,
    ``b = sqrt(s_2)`` on the branch with positive real part (decaying waves).

    Two regimes follow from :math:`\\gamma = G_p/\\sqrt{EI k}`:

    * ``gamma < 2`` (discriminant < 0): ``a, b`` complex conjugates ->
      damped-oscillatory response with decay ``alpha = Re a`` and wave number
      ``beta = Im a``;
    * ``gamma > 2``: two real decay lengths, the long one ``1/min(a, b)``.
    """
    EI = float(EI)
    k = float(k)
    G_p = float(G_p)
    if EI <= 0.0:
        raise ValueError("EI must be positive")
    if k <= 0.0:
        raise ValueError("k must be positive")
    if G_p < 0.0:
        raise ValueError("G_p must be non-negative")
    disc = complex(G_p * G_p - 4.0 * EI * k)
    root = np.sqrt(disc)                       # principal branch
    s1 = (G_p + root) / (2.0 * EI)
    s2 = (G_p - root) / (2.0 * EI)
    a = np.sqrt(s1)
    b = np.sqrt(s2)
    if a.real < 0.0:
        a = -a
    if b.real < 0.0:
        b = -b
    return a, b


def pasternak_beam_point_load(x, P: float, EI: float, k: float,
                              G_p: float = 0.0, deriv: int = 0):
    """Closed-form response of an infinite Pasternak beam to a point load.

    Parameters
    ----------
    x : array_like
        Abscissa [m] (load at ``x = 0``).
    P : float
        Point load [N], positive downwards.
    EI, k, G_p :
        Bending stiffness [N m^2], subgrade modulus [N/m^2], shear-layer
        modulus [N/m].
    deriv : int
        Order of the derivative returned: 0 -> ``w``, 1 -> ``w'``,
        2 -> ``w''``, 3 -> ``w'''``, 4 -> ``w''''``.
        ``sign(0) = 0`` makes ``w'`` and ``w'''`` vanish *at* the load point,
        where they are discontinuous.

    Returns
    -------
    numpy.ndarray
        Real part of the requested derivative (imaginary part is zero for the
        Hermitian problem).
    """
    xx = np.atleast_1d(np.asarray(x, dtype=float))
    a, b = beam_foundation_roots(EI, k, G_p)
    coeff = P / (2.0 * EI * (b * b - a * a))
    ax = np.abs(xx)
    sg = np.sign(xx)
    ea = np.exp(-a * ax)
    eb = np.exp(-b * ax)
    if deriv == 0:
        val = coeff * (ea / a - eb / b)
    elif deriv == 1:
        val = coeff * sg * (-ea + eb)
    elif deriv == 2:
        val = coeff * (a * ea - b * eb)
    elif deriv == 3:
        val = coeff * sg * (-a * a * ea + b * b * eb)
    elif deriv == 4:
        val = coeff * (a ** 3 * ea - b ** 3 * eb)
    else:  # pragma: no cover - guarded by tests
        raise ValueError("deriv must be 0..4")
    return np.real(val)


def beam_char_length(EI: float, k: float) -> float:
    """Winkler beam characteristic length ``l_b = (4 EI/k)^(1/4)`` [m]."""
    return (4.0 * float(EI) / float(k)) ** 0.25


def hetenyi_beam_point_load(x, P: float, EI: float, k: float, deriv: int = 0):
    """Hetényi (1946) infinite Winkler beam solution, explicit trig form.

    ``w = P/(2 k l_b) e^{-|x|/l_b} [cos(|x|/l_b) + sin(|x|/l_b)]``.
    Provided as an independent implementation for the ``G_p = 0`` cross-check.
    """
    xx = np.atleast_1d(np.asarray(x, dtype=float))
    lb = beam_char_length(EI, k)
    u = np.abs(xx) / lb
    amp = P / (2.0 * k * lb)
    sg = np.sign(xx)
    e = np.exp(-u)
    c = np.cos(u)
    s = np.sin(u)
    if deriv == 0:
        return amp * e * (c + s)
    if deriv == 1:
        return sg * (-2.0 * amp * e * s) / lb
    if deriv == 2:
        return 2.0 * amp * e * (s - c) / lb ** 2
    if deriv == 3:
        return sg * 4.0 * amp * e * c / lb ** 3
    if deriv == 4:
        return -4.0 * amp * e * (c + s) / lb ** 4
    raise ValueError("deriv must be 0..4")


def cosine_only_beam_point_load(x, P: float, EI: float, k: float,
                                deriv: int = 0):
    """Cosine-only (task-brief) variant ``P/(2 k l_b) e^{-|x|/l} cos(x/l)``.

    Kept :mod:`only` so that the verification suite can *falsify* it: it solves
    the homogeneous equation but its shear jump is ``P/(2 EI)`` instead of
    ``P/EI``, i.e. it represents a point load of half the requested magnitude.
    """
    xx = np.atleast_1d(np.asarray(x, dtype=float))
    lb = beam_char_length(EI, k)
    u = np.abs(xx) / lb
    amp = P / (2.0 * k * lb)
    e = np.exp(-u)
    c = np.cos(u)
    s = np.sin(u)
    if deriv == 0:
        return amp * e * c
    if deriv == 1:
        return -np.sign(xx) * amp * e * (c + s) / lb
    if deriv == 2:
        return amp * 2.0 * e * s / lb ** 2
    if deriv == 3:
        return np.sign(xx) * 2.0 * amp * e * (c - s) / lb ** 3
    if deriv == 4:
        return -4.0 * amp * e * c / lb ** 4
    raise ValueError("deriv must be 0..4")


def point_load_shear_jump(w_deriv3, P: float, EI: float, x_plus: float = 1e-9):
    """Shear-jump criterion ``EI [w''']_{0-}^{0+} = P`` for a point load.

    Returns ``EI * (w'''(x_plus) - w'''(-x_plus)) / P``, which must equal 1.0
    for a *correct* point-load solution.  Works for any callable
    ``w_deriv3(x)``.
    """
    plus = float(np.atleast_1d(w_deriv3(x_plus))[0])
    minus = float(np.atleast_1d(w_deriv3(-x_plus))[0])
    return EI * (plus - minus) / P


# ---------------------------------------------------------------------------
# finite element cross-check
# ---------------------------------------------------------------------------
@dataclass
class BeamFEMResult:
    """Container for a Hermite-beam-on-foundation FE solution."""

    x: np.ndarray          # node coordinates [m]
    w: np.ndarray          # nodal deflection (positive down) [m]
    theta: np.ndarray      # nodal slope dw/dx [-]
    EI: float
    k: float
    G_p: float
    elem_len: np.ndarray = None

    # -- evaluation --------------------------------------------------------
    def _elem_index(self, xq: float) -> int:
        i = int(np.searchsorted(self.x, xq) - 1)
        return min(max(i, 0), len(self.x) - 2)

    def evaluate(self, xq, deriv: int = 0):
        """Evaluate ``w`` (or its derivatives) at arbitrary ``x`` by Hermite."""
        xq = np.atleast_1d(np.asarray(xq, dtype=float))
        out = np.empty_like(xq)
        for k, xv in enumerate(xq):
            e = self._elem_index(xv)
            x1, x2 = self.x[e], self.x[e + 1]
            le = x2 - x1
            xi = (xv - x1) / le
            d = np.array([self.w[e], self.theta[e],
                          self.w[e + 1], self.theta[e + 1]])
            if deriv == 0:
                N = np.array([1 - 3 * xi ** 2 + 2 * xi ** 3,
                              le * (xi - 2 * xi ** 2 + xi ** 3),
                              3 * xi ** 2 - 2 * xi ** 3,
                              le * (-xi ** 2 + xi ** 3)])
            elif deriv == 1:
                N = np.array([(-6 * xi + 6 * xi ** 2) / le,
                              1 - 4 * xi + 3 * xi ** 2,
                              (6 * xi - 6 * xi ** 2) / le,
                              -2 * xi + 3 * xi ** 2])
            elif deriv == 2:
                N = np.array([(-6 + 12 * xi) / le ** 2,
                              (-4 + 6 * xi) / le,
                              (6 - 12 * xi) / le ** 2,
                              (-2 + 6 * xi) / le])
            elif deriv == 3:
                N = np.array([12 / le ** 3, 6 / le ** 2,
                              -12 / le ** 3, 6 / le ** 2])
            else:  # pragma: no cover
                raise ValueError("deriv must be 0..3")
            out[k] = float(N @ d)
        return out

    def moment(self, xq):
        """Bending moment ``M = -EI w''`` [N m] (sagging positive-down)."""
        return -self.EI * self.evaluate(xq, deriv=2)

    def shear(self, xq):
        """Shear force ``V = -EI w'''`` [N]."""
        return -self.EI * self.evaluate(xq, deriv=3)


def _beam_element_matrices(EI: float, k: float, G_p: float, le: float):
    """Consistent 4x4 element matrices for the Pasternak beam element."""
    kb = np.array([
        [12.0, 6 * le, -12.0, 6 * le],
        [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
        [-12.0, -6 * le, 12.0, -6 * le],
        [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2],
    ]) * (EI / le ** 3)
    kf = np.array([
        [156.0, 22 * le, 54.0, -13 * le],
        [22 * le, 4 * le ** 2, 13 * le, -3 * le ** 2],
        [54.0, 13 * le, 156.0, -22 * le],
        [-13 * le, -3 * le ** 2, -22 * le, 4 * le ** 2],
    ]) * (k * le / 420.0)
    kg = np.array([
        [36.0, 3 * le, -36.0, 3 * le],
        [3 * le, 4 * le ** 2, -3 * le, -le ** 2],
        [-36.0, -3 * le, 36.0, -3 * le],
        [3 * le, -le ** 2, -3 * le, 4 * le ** 2],
    ]) * (G_p / (30.0 * le))
    return kb + kf + kg


def solve_beam_foundation_fem(EI: float, k: float, P: float, half_length: float,
                              n_elem: int, G_p: float = 0.0,
                              x_load: float = 0.0):
    """Hermite-beam FE solution for an infinite beam (truncated at +-L).

    The infinite beam is approximated by a free-free beam of half-length
    ``half_length`` with a point load at ``x_load``; because the closed-form
    decay length is finite, the truncation error decays exponentially in
    ``half_length / l_b``.

    Returns
    -------
    BeamFEMResult
    """
    if n_elem % 2 != 0:
        raise ValueError("n_elem must be even so that x = 0 is a node")
    x = np.linspace(-half_length, half_length, n_elem + 1)
    ndof = 2 * (n_elem + 1)
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)
    elem_len = np.diff(x)
    for e in range(n_elem):
        le = elem_len[e]
        ke = _beam_element_matrices(EI, k, G_p, le)
        idx = np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[np.ix_(idx, idx)] += ke
    # point load at the nearest node (x = 0 is a node by construction)
    node = int(np.argmin(np.abs(x - x_load)))
    F[2 * node] += P
    w = np.linalg.solve(K, F)
    return BeamFEMResult(
        x=x,
        w=w[0::2],
        theta=w[1::2],
        EI=float(EI),
        k=float(k),
        G_p=float(G_p),
        elem_len=elem_len,
    )