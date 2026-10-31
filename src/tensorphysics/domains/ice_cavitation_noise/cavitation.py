"""Cavitation mass-transfer models and single-bubble (Rayleigh-Plesset) dynamics.

Physical model
--------------
Cavitation in the ice-water mixture is modelled as a homogeneous mixture of
liquid water and vapour with the vapour volume fraction ``alpha`` transported
by the flow.  Mass exchange between the phases is supplied by a barotropic
cavitation model evaluated on the local pressure:

* Schnerr-Sauer (2001)::

      R   = ( alpha / (1 - alpha) * 3 / (4 pi n0) )^(1/3)
      mdot = sign(p_v - p) * C * (rho_l rho_v / rho_m) * (3 alpha (1-alpha) / R)
             * sqrt( 2 |p_v - p| / (3 rho_l) )

* Zwart-Gerber-Belamri (2004), with the ``C_dest``/``C_prod``
  (50 / 0.01) asymmetry between destruction and production.

Both models are derived from the Rayleigh-Plesset equation in the
inertia-controlled limit, ``R d2R/dt2 + 3/2 (dR/dt)^2 = (p_v - p)/rho_l``,
whose collapse time from rest is the classical Rayleigh result

    t_c = 0.91468 R0 sqrt(rho_l / dp)

All constants are SI (Pa, kg/m^3, m, s); ``RHO_L``/``P_V`` are taken at 20 degC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Water / vapour properties at 20 degC (SI)
# ---------------------------------------------------------------------------
RHO_L: float = 998.2       # liquid density                [kg/m^3]
RHO_V: float = 0.0173      # vapour density               [kg/m^3]
P_V: float = 2339.0        # saturation (vapour) pressure [Pa]
SIGMA_L: float = 0.0728    # surface tension              [N/m]
MU_L: float = 1.002e-3     # dynamic viscosity            [Pa s]
C_L: float = 1482.0        # liquid sound speed           [m/s]
RAYLEIGH_COLLAPSE_CONST: float = 0.91468


def _rayleigh_drive(
    dp: np.ndarray,
    radius: np.ndarray,
    prm: "CavitationParams",
    regularize: bool = False,
) -> np.ndarray:
    """Inertial drive ``sqrt(2 |dp| / (3 rho_l))`` of the Rayleigh-Plesset limit.

    ``regularize=True`` floors ``|dp|`` at the Blake surface-tension threshold
    ``2 sigma / R`` of the current bubble.  The *source* itself must stay
    continuous in ``p`` (it switches sign at ``p = p_v``), so the floor is used
    only where a pressure derivative is needed -- the linearisation of the
    pressure-correction equation in :mod:`flow1d`.
    """
    dp_abs = np.abs(dp)
    if regularize:
        dp_abs = np.maximum(dp_abs, 2.0 * prm.sigma / np.maximum(radius, 1.0e-9))
    return np.sqrt(2.0 * dp_abs / (3.0 * prm.rho_l))


def rayleigh_drive_regularized(
    dp: float | np.ndarray,
    radius: float | np.ndarray,
    params: CavitationParams | None = None,
) -> np.ndarray:
    """Blake-regularised inertial drive (numerical linearisation only)."""
    prm = params or CavitationParams()
    return _rayleigh_drive(np.asarray(dp, dtype=float), np.asarray(radius, dtype=float), prm, regularize=True)


@dataclass(frozen=True)
class CavitationParams:
    """Tunables of the barotropic cavitation models.

    ``n0`` is the nuclei number density, ``alpha_nuc`` the initial vapour
    volume fraction (dissolved/gaseous nuclei that seed the mass transfer),
    ``R_B`` the fixed bubble radius used by Zwart-Gerber-Belamri.
    """

    n0: float = 1.0e10         # nuclei per m^3 (10 per cm^3, bubbly seawater)
    alpha_nuc: float = 5.0e-4  # initial vapour volume fraction
    C: float = 1.0             # Schnerr-Sauer calibration constant
    C_dest: float = 50.0       # ZGB vapour destruction (collapse) coefficient
    C_prod: float = 0.01       # ZGB vapour production (growth) coefficient
    R_B: float = 1.0e-6        # ZGB reference bubble radius [m]
    model: str = "schnerr_sauer"
    sigma: float = SIGMA_L
    limit_to_equilibrium: bool = True
    alpha_saturation: float = 0.5
    rho_l: float = RHO_L
    rho_v: float = RHO_V
    p_v: float = P_V

    @property
    def alpha_floor(self) -> float:
        """Smallest admissible vapour fraction (nuclei background)."""
        return 1.0e-12


def radius_from_alpha(alpha: float | np.ndarray, n0: float) -> np.ndarray:
    """Bubble radius implied by a vapour volume fraction (Schnerr-Sauer)."""
    a = np.clip(np.asarray(alpha, dtype=float), 1.0e-12, 1.0 - 1.0e-9)
    cells = 3.0 / (4.0 * np.pi * n0)
    return np.cbrt(a / (1.0 - a) * cells)


def alpha_from_radius(radius: float | np.ndarray, n0: float) -> np.ndarray:
    """Vapour volume fraction from a bubble radius (inverse of the above)."""
    r = np.clip(np.asarray(radius, dtype=float), 1.0e-12, None)
    void = n0 * 4.0 / 3.0 * np.pi * r ** 3
    return void / (1.0 + void)


def schnerr_sauer_source(
    p: float | np.ndarray,
    alpha: float | np.ndarray,
    params: CavitationParams | None = None,
) -> np.ndarray:
    """Vapour mass production rate ``mdot`` [kg m^-3 s^-1] (Schnerr-Sauer).

    Positive ``mdot`` means liquid -> vapour (cavitation), negative means
    vapour -> liquid (collapse).  The rate vanishes at ``p = p_v`` and is zero
    wherever there are no vapour nuclei (``alpha -> 0``), which is why a
    non-zero ``alpha_nuc`` background is required.
    """
    prm = params or CavitationParams()
    p = np.asarray(p, dtype=float)
    a = np.clip(np.asarray(alpha, dtype=float), prm.alpha_floor, 1.0 - 1.0e-9)

    radius = radius_from_alpha(a, prm.n0)
    rho_m = (1.0 - a) * prm.rho_l + a * prm.rho_v
    dp = prm.p_v - p
    drive = _rayleigh_drive(dp, radius, prm)
    # Growth acts on all available vapour; collapse may only remove vapour in
    # excess of the non-condensable nuclei background (dissolved gas nuclei do
    # not condense), otherwise a pressurised quiescent mixture would be
    # destroyed at an unphysically large rate.
    alpha_active = np.where(dp > 0.0, a, np.maximum(a - prm.alpha_nuc, 0.0))
    rate = (
        np.sign(dp)
        * prm.C
        * (prm.rho_l * prm.rho_v / rho_m)
        * (3.0 * alpha_active * (1.0 - a) / radius)
        * drive
    )
    return _limit_to_equilibrium(rate, p, a, prm)


def zwart_gerber_belamri_source(
    p: float | np.ndarray,
    alpha: float | np.ndarray,
    params: CavitationParams | None = None,
) -> np.ndarray:
    """Vapour mass production rate ``mdot`` [kg m^-3 s^-1] (Zwart-Gerber-Belamri)."""
    prm = params or CavitationParams()
    p = np.asarray(p, dtype=float)
    a = np.clip(np.asarray(alpha, dtype=float), prm.alpha_floor, 1.0 - 1.0e-9)

    dp = prm.p_v - p
    drive = _rayleigh_drive(dp, np.full_like(a, prm.R_B), prm)
    growth = (
        prm.C_prod
        * 3.0 * a * (1.0 - a) * prm.rho_v / prm.R_B * drive
    )
    collapse = (
        prm.C_dest
        * 3.0 * prm.alpha_nuc * (1.0 - a) * prm.rho_v / prm.R_B * drive
    )
    rate = np.where(dp > 0.0, growth, -collapse)
    return _limit_to_equilibrium(rate, p, a, prm)


def cavitation_source(
    p: float | np.ndarray,
    alpha: float | np.ndarray,
    params: CavitationParams | None = None,
) -> np.ndarray:
    """Dispatch to the model selected by ``params.model``."""
    prm = params or CavitationParams()
    if prm.model == "schnerr_sauer":
        return schnerr_sauer_source(p, alpha, prm)
    if prm.model in ("zwart_gerber_belamri", "zgb"):
        return zwart_gerber_belamri_source(p, alpha, prm)
    raise ValueError(f"unknown cavitation model {prm.model!r}")


def _limit_to_equilibrium(
    rate: np.ndarray, p: np.ndarray, alpha: np.ndarray, prm: "CavitationParams"
) -> np.ndarray:
    """Suppress vaporisation once the vapour fraction reaches static equilibrium.

    The barotropic model has no built-in equilibrium: below ``p_v`` it would
    keep vaporising without bound.  Physically the vapour content saturates at
    the static (Blake) equilibrium value of the local pressure,
    ``alpha_eq(p)`` -- beyond it the mixture cannot hold more vapour at that
    pressure, and the excess volume is what pushes the pressure back up.  This
    limiter is what keeps a cavitating solution physical instead of
    "super-vaporising" into a negative-pressure state.
    """
    if not prm.limit_to_equilibrium:
        return rate
    # Volume-limited cavity: the reduced model has no interface tracking, so the
    # vapour fraction saturates smoothly at ``alpha_saturation`` instead of
    # growing without bound.  The source stays C1 in alpha and p, which matters
    # for the implicit steady march.
    width = 0.25 * prm.alpha_saturation
    factor = np.clip((prm.alpha_saturation - alpha) / width, 0.0, 1.0)
    return np.where(rate > 0.0, rate * factor, rate)


def equilibrium_alpha(
    p: float | np.ndarray, params: CavitationParams | None = None
) -> np.ndarray:
    """Equilibrium vapour fraction of a nuclei population at pressure ``p``.

    Static (Blake) equilibrium of a bubble without permanent gas gives
    ``p_v - p = 2 sigma / R``; the nuclei population ``n0`` then has

        alpha_eq = f( R = 2 sigma / (p_v - p) ).

    This is the *equilibrium* (infinitely fast mass transfer) closure.  Its
    derivative ``d alpha_eq / d p`` is the mixture compressibility that keeps a
    pressure-correction scheme bounded when a vapour cavity forms: without it
    the pressure would dive through ``p_v`` without limit.
    """
    prm = params or CavitationParams()
    p = np.asarray(p, dtype=float)
    dp = np.maximum(prm.p_v - p, 1.0)
    radius = 2.0 * prm.sigma / dp
    alpha = alpha_from_radius(radius, prm.n0)
    return np.where(p < prm.p_v, np.maximum(alpha, prm.alpha_nuc), prm.alpha_nuc)


def rayleigh_collapse_time(r0: float, dp: float, rho: float = RHO_L) -> float:
    """Rayleigh collapse time of a spherical bubble from rest [s]."""
    if dp <= 0.0:
        raise ValueError("dp must be positive for a collapse")
    return RAYLEIGH_COLLAPSE_CONST * r0 * np.sqrt(rho / dp)


def bubble_radius_forced(
    t: np.ndarray,
    p_drive: np.ndarray,
    r0: float,
    rho: float = RHO_L,
    p_v: float = P_V,
    sigma: float = SIGMA_L,
    mu: float = MU_L,
    polytropic: float = 1.4,
    r_min_factor: float = 0.02,
) -> np.ndarray:
    """Forced Rayleigh-Plesset response of one nucleus to a pressure history.

    A permanent-gas content is included (polytropic exponent ``polytropic``)
    with its initial partial pressure chosen so that the nucleus is in static
    equilibrium at ``p_v``:

        p_g0 = p_v + 2 sigma / r0

    Without it the collapse terminates in a singular point and the rebounding
    bubble cloud (the part of the spectrum that carries the broadband noise)
    cannot be represented.  The radius is floored at
    ``r_min_factor * r0`` which is where the present (isothermal, no
    compressibility) model stops being meaningful.

    Returns the radius history on the input time grid (classical RK4).
    """
    t = np.asarray(t, dtype=float)
    p_drive = np.asarray(p_drive, dtype=float)
    r = float(r0)
    rdot = 0.0
    out = np.empty_like(t)
    out[0] = r
    p_g0 = p_v + 2.0 * sigma / r0
    r_min = r_min_factor * r0

    def accel(rad, rad_dot, p_inf):
        rad_safe = max(rad, r_min)
        p_gas = p_g0 * (r0 / rad_safe) ** (3.0 * polytropic)
        return (
            (p_gas - p_inf - 2.0 * sigma / rad_safe) / rho
            - 4.0 * mu * rad_dot / (rho * rad_safe)
            - 1.5 * rad_dot ** 2
        ) / rad_safe

    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        p_inf = p_drive[i]
        k1r = rdot
        k1d = accel(r, rdot, p_inf)
        k2r = rdot + 0.5 * dt * k1d
        k2d = accel(r + 0.5 * dt * k1r, rdot + 0.5 * dt * k1d, p_inf)
        k3r = rdot + 0.5 * dt * k2d
        k3d = accel(r + 0.5 * dt * k2r, rdot + 0.5 * dt * k2d, p_inf)
        k4r = rdot + dt * k3d
        k4d = accel(r + dt * k3r, rdot + dt * k3d, p_inf)
        r_new = r + dt / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        rdot_new = rdot + dt / 6.0 * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
        if r_new <= r_min:
            # rigid-core bounce: the isothermal model cannot resolve the final
            # compression stage, so the wall is stopped at the floor radius
            r = r_min
            rdot = abs(rdot_new) * 0.5
        else:
            r = r_new
            rdot = rdot_new
        out[i] = r
    return out


def rayleigh_growth_velocity(dp: float, rho: float = RHO_L) -> float:
    """Asymptotic bubble-wall speed ``sqrt(2 dp / (3 rho))`` for inertial growth."""
    return float(np.sqrt(2.0 * dp / (3.0 * rho)))


def _rp_rhs(
    r: float,
    rdot: float,
    p_inf: float,
    p_v: float,
    rho: float,
    sigma: float,
    mu: float,
) -> tuple[float, float]:
    """Right-hand side of the Rayleigh-Plesset equation."""
    r_safe = max(r, 1.0e-12)
    pressure_term = (p_v - p_inf) / rho
    surface = 2.0 * sigma / (rho * r_safe)
    viscous = 4.0 * mu * rdot / (rho * r_safe)
    rddot = (pressure_term - surface - viscous - 1.5 * rdot * rdot) / r_safe
    return rdot, rddot


def bubble_radius_history(
    r0: float,
    p_inf: float,
    t_end: float,
    dt: float,
    sigma: float = 0.0,
    mu: float = 0.0,
    rho: float = RHO_L,
    p_v: float = P_V,
    r_stop: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the Rayleigh-Plesset equation with classical RK4.

    Returns ``(t, R)``.  Integration stops early once ``R <= r_stop`` (used to
    time a collapse without resolving the singular final approach).
    """
    t_values = [0.0]
    r_values = [float(r0)]
    r = float(r0)
    rdot = 0.0
    t = 0.0
    n_steps = int(np.ceil(t_end / dt))
    for _ in range(max(n_steps, 1)):
        k1r, k1d = _rp_rhs(r, rdot, p_inf, p_v, rho, sigma, mu)
        k2r, k2d = _rp_rhs(
            r + 0.5 * dt * k1r, rdot + 0.5 * dt * k1d, p_inf, p_v, rho, sigma, mu
        )
        k3r, k3d = _rp_rhs(
            r + 0.5 * dt * k2r, rdot + 0.5 * dt * k2d, p_inf, p_v, rho, sigma, mu
        )
        k4r, k4d = _rp_rhs(
            r + dt * k3r, rdot + dt * k3d, p_inf, p_v, rho, sigma, mu
        )
        r_new = r + dt / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        rdot = rdot + dt / 6.0 * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
        t = t + dt
        r = r_new
        t_values.append(t)
        r_values.append(r)
        if r <= 0.0:
            break
        if r_stop is not None and r <= r_stop:
            break
    return np.asarray(t_values), np.asarray(r_values)