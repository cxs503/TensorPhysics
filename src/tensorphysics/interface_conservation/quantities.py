"""Conserved interface quantities for cross-solver coupling gates.

The gate in this package is defined on **interface fluxes**, not on state
variables, because a conserved quantity is only meaningful as a flux across a
control surface.  Three quantities are defined on a straight interface whose
unit normal is ``n`` (axis aligned, ``n = e_x`` for all cases shipped here):

``mass flux``
    ``Phi_m = integral rho (u . n) dA``
``normal momentum flux``
    ``Phi_p = integral [ rho (u . n) (u . n) + p - tau_nn ] dA``
    i.e. the ``n``-component of the momentum flux vector across the surface.
    In the incompressible formulation used here the dynamic pressure enters as
    ``p``; the hydrostatic reference of the receiving (lattice) solver,
    ``rho_0 c_s^2``, is *excluded* (see :func:`lbm_normal_momentum_flux`), so both
    sides are compared in the same dynamic-pressure gauge.
``energy flux``
    ``Phi_e = integral [ (p + E) (u . n) - u . (tau . n) ] dA`` with
    ``E = rho |u|^2 / 2`` the resolved kinetic energy.  For the isothermal
    incompressible limit this is the mechanical-energy flux; internal energy is
    constant and drops out.

The unit system is that of the finite-volume (FVM) side: lengths by the cavity
side ``L``, velocities by the lid velocity ``U``, density by ``rho``, so
pressures and stresses are in ``rho U^2``.  The lattice side is converted into
the same units with the two scale factors returned by
:func:`lattice_scales`; every conversion factor is therefore explicit and no
hidden unit assumption remains in the residual.

Residual definition (the machine criterion)::

    R_q = |Phi_q^A - Phi_q^B| / max(|Phi_q^A|, |Phi_q^B|, q_ref)

with ``q_ref`` a per-quantity reference magnitude supplied by the caller
(``rho U^2 L`` for pressure-type quantities) so that a residual is never
divided by a near-zero flux.  Two aggregation levels are reported:

``integral``
    the residual of the surface integrals above -- this is the statement
    "nothing is lost at the coupling interface".
``local``
    the ``L2``-normalised residual of the *pointwise* (per-face) fluxes, which
    measures whether the two discretisations agree locally.  This is the level
    at which a boundary closure error is visible and the level on which the
    observed order of accuracy is measured.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "InterfaceFlux",
    "lattice_scales",
    "flux_residual",
    "local_flux_residual",
    "fit_order",
    "energy_flux_from_fields",
]

#: D2Q9 sound speed squared.
CS2 = 1.0 / 3.0


@dataclass(frozen=True)
class InterfaceFlux:
    """Surface-integrated conserved fluxes through one straight interface.

    All members are per unit depth (2D), in the FVM non-dimensionalisation
    (``rho = L = U = 1``).  ``momentum`` is the normal component of the momentum
    flux vector (``x``-component for an ``x``-normal interface).
    """

    mass: float
    momentum: float
    energy: float

    def scaled(self, mass: float = 1.0, momentum: float = 1.0,
               energy: float = 1.0) -> "InterfaceFlux":
        return InterfaceFlux(self.mass * mass, self.momentum * momentum,
                             self.energy * energy)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.mass, self.momentum, self.energy)


def lattice_scales(u_lat: float, h: float) -> tuple[float, float]:
    """Return ``(s_u, s_p)`` converting lattice units to FVM units.

    ``u_lat`` is the reference (lid) velocity in lattice units, ``h`` the
    physical size of one lattice cell.  A lattice velocity ``u`` maps to
    ``u * s_u`` and a lattice stress/pressure to ``s_p`` times itself.

    Derivation: one lattice time step advances ``dt = h u_lat / U`` in physical
    time (``U = 1`` is the FVM velocity unit), hence the velocity scale is
    ``U/u_lat`` and the stress scale is its square.
    """
    if not (0.0 < u_lat < 0.5):
        raise ValueError("lattice reference velocity must satisfy 0 < u_lat < 0.5")
    if h <= 0.0:
        raise ValueError("h must be positive")
    s_u = 1.0 / u_lat
    return s_u, s_u * s_u


def flux_residual(a: InterfaceFlux, b: InterfaceFlux,
                  ref: InterfaceFlux | None = None) -> InterfaceFlux:
    """Relative residual of the surface-integrated fluxes ``a`` vs ``b``.

    ``ref`` supplies the denominator floor per quantity (``rho U^2 L`` for the
    momentum and energy channels by default); it prevents a near-zero flux pair
    from producing a meaningless ratio.
    """
    if ref is None:
        ref = InterfaceFlux(mass=1.0, momentum=1.0, energy=1.0)
    out = []
    for x, y, r in zip(a.as_tuple(), b.as_tuple(), ref.as_tuple()):
        if r <= 0.0:
            raise ValueError("reference magnitudes must be positive")
        denom = max(abs(x), abs(y), r)
        out.append(abs(x - y) / denom)
    return InterfaceFlux(*out)


def local_flux_residual(points_a: Sequence[float], points_b: Sequence[float],
                        ref_scale: float) -> float:
    """``L2`` residual of the pointwise interface flux distributions.

    ``R_local = ||q_a - q_b||_2 / (sqrt(N) * ref_scale)``; with ``ref_scale`` the
    same reference magnitude used by :func:`flux_residual` so the two levels are
    directly comparable (the local level carries ``1/sqrt(N)`` normalisation, as
    is standard for a grid-function norm).
    """
    qa = np.asarray(points_a, dtype=float)
    qb = np.asarray(points_b, dtype=float)
    if qa.shape != qb.shape:
        raise ValueError("pointwise flux arrays must have the same shape")
    if qa.size == 0:
        raise ValueError("empty interface")
    if ref_scale <= 0.0:
        raise ValueError("ref_scale must be positive")
    return float(np.linalg.norm(qa - qb) / (math.sqrt(qa.size) * ref_scale))


def fit_order(h: Iterable[float], err: Iterable[float]) -> tuple[float, float]:
    """Least-squares slope of ``log(err)`` against ``log(h)`` and its ``R^2``.

    The slope is the observed order of accuracy ``p`` in ``err ~ C h^p``.
    """
    hs = np.asarray(list(h), dtype=float)
    es = np.asarray(list(err), dtype=float)
    if hs.size != es.size:
        raise ValueError("h and err must have the same length")
    if hs.size < 2:
        raise ValueError("at least two refinement levels are needed")
    mask = (hs > 0.0) & (es > 0.0)
    if mask.sum() < 2:
        raise ValueError("need at least two positive (h, err) pairs")
    x = np.log(hs[mask])
    y = np.log(es[mask])
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return float(slope), float(r2)


def energy_flux_from_fields(u_n: np.ndarray, u_t: np.ndarray, p: np.ndarray,
                            tau_nn: np.ndarray, tau_nt: np.ndarray,
                            rho: float = 1.0) -> float:
    """Mechanical-energy flux ``int [(p + E) u_n - u_n tau_nn - u_t tau_nt] dA``.

    ``u_n``/``u_t``/``p``/``tau`` are pointwise samples on the interface (equal
    weight per sample; the caller supplies the surface measure through the
    sample spacing).
    """
    un = np.asarray(u_n, dtype=float)
    ut = np.asarray(u_t, dtype=float)
    pr = np.asarray(p, dtype=float)
    tnn = np.asarray(tau_nn, dtype=float)
    tnt = np.asarray(tau_nt, dtype=float)
    kinetic = 0.5 * rho * (un * un + ut * ut)
    return float(np.sum((pr + kinetic) * un - un * tnn - ut * tnt))