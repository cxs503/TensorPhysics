"""Dilute ice-particle phase and its coupling to the liquid.

Model
-----
Ice floes/crystals are carried as a dilute, statistically steady population of
rigid spherical particles of mean diameter ``d_p`` and density ``rho_ice``.
With a *volumetric ice concentration* ``c`` (0 <= c < 1) the number density is

    n_p = c / (pi d_p^3 / 6)          [particles per m^3 of mixture]

The drag on one particle follows Schiller-Naumann::

    C_D = 24/Re (1 + 0.15 Re^0.687)   for Re < 1000,   0.44 otherwise
    F   = 0.5 C_D rho_l A_p |w| w     (w = u_l - u_p slip velocity)

so that the momentum coupling source (force per unit mixture volume) is
``S = n_p F``.  The particle momentum response time including the added-mass of
the displaced liquid is

    tau_p = d_p^2 (rho_ice + rho_l/2) / (18 mu)

Because ice particles are large (Stokes number >> 1) they do not follow the
turbulent pressure field; each particle sheds drag fluctuations that appear as
a local pressure fluctuation of order

    p'_rms = kappa * c * n_p / c * F_drag / d_p^2 = kappa * F_drag / d_p^2 * c

written explicitly as ``pressure_fluctuation_rms`` below (``kappa`` is an O(1)
model coefficient; its sensitivity is reported in the validation report).
These fluctuations are the mechanism by which ice raises the *apparent*
cavitation inception number: the local instantaneous pressure drops below the
vapour pressure before the time-mean pressure does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cavitation import MU_L, RHO_L


@dataclass(frozen=True)
class IcePhaseParams:
    """Geometrical and material parameters of the ice phase."""

    rho_ice: float = 916.7        # density of (model) ice  [kg/m^3]
    d_p: float = 5.0e-4           # mean particle diameter [m]
    kappa: float = 1.0            # O(1) pressure-fluctuation coefficient
    strouhal: float = 0.2         # Strouhal number of particle wake shedding
    rho_l: float = RHO_L
    mu_l: float = MU_L


def number_density(concentration: float, d_p: float) -> float:
    """Particle number density [m^-3] for a volumetric ice concentration."""
    if concentration < 0.0:
        raise ValueError("ice concentration must be non-negative")
    volume = np.pi * d_p ** 3 / 6.0
    return float(concentration / volume)


def drag_coefficient(re_p: float | np.ndarray) -> np.ndarray:
    """Schiller-Naumann drag coefficient with the Newton-regime plateau."""
    re = np.maximum(np.asarray(re_p, dtype=float), 1.0e-12)
    stokes = 24.0 / re * (1.0 + 0.15 * re ** 0.687)
    return np.where(re < 1000.0, stokes, 0.44)


def particle_response_time(
    d_p: float, rho_ice: float = 916.7, rho_l: float = RHO_L, mu_l: float = MU_L
) -> float:
    """Particle momentum relaxation time including added mass [s]."""
    return d_p ** 2 * (rho_ice + 0.5 * rho_l) / (18.0 * mu_l)


def drag_force_per_particle(
    slip: float,
    d_p: float,
    rho_l: float = RHO_L,
    mu_l: float = MU_L,
) -> float:
    """Drag force magnitude on a single particle [N]."""
    area = np.pi * d_p ** 2 / 4.0
    re_p = rho_l * abs(slip) * d_p / mu_l
    cd = float(drag_coefficient(re_p))
    return float(0.5 * cd * rho_l * area * slip * abs(slip))


def drag_force_per_volume(
    slip: float,
    concentration: float,
    params: IcePhaseParams | None = None,
) -> float:
    """Momentum coupling source ``S_p`` [N/m^3] exerted by liquid on ice.

    Sign convention: positive when the liquid moves faster than the ice, i.e.
    the liquid is decelerated by this source.
    """
    prm = params or IcePhaseParams()
    if concentration <= 0.0:
        return 0.0
    n_p = number_density(concentration, prm.d_p)
    return n_p * drag_force_per_particle(slip, prm.d_p, prm.rho_l, prm.mu_l)


def pressure_fluctuation_rms(
    slip: float,
    concentration: float,
    params: IcePhaseParams | None = None,
) -> float:
    """RMS of the particle-induced local pressure fluctuation [Pa].

    Derivation: one particle exerts ``F`` on the liquid; the reaction is spread
    over the particle-scale area ``d_p^2``, and in a suspension of number
    density ``n_p`` the *coherent* part survives over the inter-particle volume
    ``1/n_p``.  The resulting pressure scale is ``n_p F d_p`` (force per unit
    area of the particle layer), which for a volumetric concentration ``c``
    equals ``c F / (d_p^2 * pi/6)``.  We keep the O(1) coefficient ``kappa``
    explicit rather than hiding it inside a fitted constant.
    """
    prm = params or IcePhaseParams()
    if concentration <= 0.0:
        return 0.0
    force = abs(drag_force_per_particle(slip, prm.d_p, prm.rho_l, prm.mu_l))
    return float(prm.kappa * concentration * force / (np.pi / 6.0 * prm.d_p ** 2))


def fluctuating_drag_force(
    t: np.ndarray,
    u_liquid: np.ndarray,
    u_particle_mean: float,
    concentration: float,
    params: IcePhaseParams | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Time series of the fluctuating liquid->ice momentum source [N/m^3].

    Deterministic pseudo-random superposition of wake-shedding events at the
    Strouhal frequency ``f_s = St |w| / d_p`` of each particle class; the
    amplitude of every component is set so that the resulting RMS equals
    ``drag_force_per_volume`` of the local slip.  This gives a *reproducible*
    broadband forcing (fixed seed) instead of hard-coded noise.
    """
    prm = params or IcePhaseParams()
    t = np.asarray(t, dtype=float)
    u_liquid = np.asarray(u_liquid, dtype=float)
    if concentration <= 0.0:
        return np.zeros_like(t)

    slip = u_liquid - u_particle_mean
    mean_source = np.array([drag_force_per_volume(s, concentration, prm) for s in slip])
    rng = np.random.default_rng(seed)
    f_shed = prm.strouhal * np.abs(slip) / prm.d_p
    f_shed = np.clip(f_shed, 1.0, None)

    # 24 random phases per mode: the superposition has zero mean and a
    # broadband envelope centred on the shedding frequency.
    n_modes = 24
    out = np.zeros_like(t)
    for _ in range(n_modes):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        weight = rng.uniform(0.5, 1.5)
        out += weight * np.sin(2.0 * np.pi * f_shed * t + phase)
    out *= np.sqrt(2.0 / n_modes)          # unit RMS per mode, total RMS ~ 1
    out /= max(np.std(out), 1.0e-30)
    return out * mean_source