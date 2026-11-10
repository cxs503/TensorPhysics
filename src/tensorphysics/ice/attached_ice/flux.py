"""Frazil transport and impact flux to submerged surfaces.

Physical model
==============
Frazil ice particles suspended in the water column are transported to the
hull surface by a combination of:

1. **Turbulent diffusion** in the boundary layer, characterised by the
   friction velocity ``u_star``.
2. **Gravitational settling** (Stokes regime for small frazil crystals).

The net deposition velocity is the quadrature sum of the two mechanisms,
and the mass flux is::

    J = C_f * rho_ice * v_deposition

where ``C_f`` is the frazil volume concentration (dimensionless) and
``v_deposition`` is the effective deposition velocity [m/s].

The turbulent transport velocity is modelled as a fraction of the friction
velocity (following deposition literature for fine particles in turbulent
boundary layers)::

    v_turb = beta * u_star

with ``beta`` an empirical coefficient (~0.1 for sub-millimetre particles).

The settling velocity in the Stokes regime is::

    v_settle = (rho_ice - rho_water) * g * d^2 / (18 * mu_water)

References
----------
* Camfield (1980): frazil ice accumulation on structures.
* Daly & Colbeck (1986): frazil ice transport and deposition.
* Stokes settling for small particles in viscous flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import (
    RHO_ICE, RHO_WATER, MU_WATER, GRAVITY, DEFAULT_FRAZIL_DIAMETER_M,
)

# Empirical turbulent deposition coefficient for fine particles
BETA_TURB = 0.10


@dataclass(frozen=True)
class FrazilFluxState:
    """Decomposition of the frazil impact flux at a wall location."""

    #: Frazil volume concentration (dimensionless)
    c_f: float
    #: Friction velocity [m/s]
    u_star: float
    #: Deposition velocity [m/s]
    v_deposition: float
    #: Turbulent transport velocity component [m/s]
    v_turb: float
    #: Gravitational settling velocity component [m/s]
    v_settle: float
    #: Mass flux [kg/(m^2*s)]
    mass_flux: float

    def as_dict(self) -> dict:
        return {
            "c_f": self.c_f,
            "u_star_m_s": self.u_star,
            "v_deposition_m_s": self.v_deposition,
            "v_turb_m_s": self.v_turb,
            "v_settle_m_s": self.v_settle,
            "mass_flux_kg_m2s": self.mass_flux,
        }


def settling_velocity_stokes(
    particle_diameter_m: float = DEFAULT_FRAZIL_DIAMETER_M,
    rho_ice: float = RHO_ICE,
    rho_water: float = RHO_WATER,
    mu_water: float = MU_WATER,
    gravity: float = GRAVITY,
) -> float:
    """Stokes settling velocity of a frazil particle [m/s].

    Valid for particle Reynolds number Re_p < 1 (Stokes regime).
    For d = 0.1 mm in seawater at 0 degC, Re_p ~ 0.01, well within range.

    Parameters
    ----------
    particle_diameter_m : float
        Equivalent spherical diameter of the frazil particle [m].
    rho_ice, rho_water, mu_water, gravity : float
        Physical constants.

    Returns
    -------
    float
        Settling velocity [m/s].
    """
    d = float(particle_diameter_m)
    if d <= 0.0:
        raise ValueError("particle diameter must be positive")
    return float(
        (rho_ice - rho_water) * gravity * d**2 / (18.0 * mu_water)
    )


def deposition_velocity(
    u_star: float,
    particle_diameter_m: float = DEFAULT_FRAZIL_DIAMETER_M,
    beta: float = BETA_TURB,
) -> float:
    """Effective deposition velocity combining turbulent and gravitational transport.

    Parameters
    ----------
    u_star : float
        Friction velocity [m/s].
    particle_diameter_m : float
        Frazil particle diameter [m].
    beta : float
        Turbulent deposition coefficient.

    Returns
    -------
    float
        Deposition velocity [m/s].
    """
    u_s = float(u_star)
    if u_s < 0.0:
        raise ValueError("friction velocity must be non-negative")
    v_t = beta * u_s
    v_s = settling_velocity_stokes(particle_diameter_m)
    # Quadrature sum: independent transport mechanisms
    return float(np.sqrt(v_t**2 + v_s**2))


def frazil_impact_flux(
    c_f: float,
    u_star: float,
    particle_diameter_m: float = DEFAULT_FRAZIL_DIAMETER_M,
    rho_ice: float = RHO_ICE,
    beta: float = BETA_TURB,
) -> FrazilFluxState:
    """Compute the frazil impact flux state.

    Parameters
    ----------
    c_f : float
        Frazil volume concentration (dimensionless, typically 1e-6 to 1e-3).
    u_star : float
        Friction velocity [m/s].
    particle_diameter_m : float
        Frazil particle diameter [m].
    rho_ice : float
        Ice density [kg/m^3].
    beta : float
        Turbulent deposition coefficient.

    Returns
    -------
    FrazilFluxState
    """
    c_f = float(c_f)
    if c_f < 0.0:
        raise ValueError("frazil concentration must be non-negative")
    if c_f > 1.0:
        raise ValueError("frazil concentration must be <= 1")

    v_t = beta * float(u_star)
    v_s = settling_velocity_stokes(particle_diameter_m)
    v_dep = float(np.sqrt(v_t**2 + v_s**2))
    mass_flux = c_f * rho_ice * v_dep

    return FrazilFluxState(
        c_f=c_f,
        u_star=float(u_star),
        v_deposition=v_dep,
        v_turb=v_t,
        v_settle=v_s,
        mass_flux=mass_flux,
    )
