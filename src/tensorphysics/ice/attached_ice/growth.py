"""Interface energy-balance ODE for underwater attached ice growth.

Physical model
==============
The ice layer grows by freezing of frazil particles that impact and stick
to the surface, and by conductive freezing of the water film at the ice-water
interface. The governing energy balance at the interface is::

    rho_ice * L_f * dh/dt = q_cond - q_conv + q_frazil

where:

* ``q_cond = k_ice * (T_f - T_surface) / h_ice`` is the conductive heat
  flux through the ice layer (positive when heat flows out of the interface).
* ``q_conv = h_c * (T_water - T_f)`` is the convective heat flux from the
  water to the interface (positive when water is warmer than freezing).
* ``q_frazil = eta_stick * J_mass * L_f`` is the latent heat release from
  frazil particles that stick to the surface.

The convective heat transfer coefficient is estimated from the friction
velocity using a Reynolds analogy::

    h_c = rho_water * C_p * u_star / Pr_t

where ``Pr_t`` is the turbulent Prandtl number (~1 for water).

Stefan limit
------------
When frazil flux is zero and convection is negligible (large undercooling),
the ODE reduces to the classical Stefan problem::

    h(t) = sqrt(2 * k_ice * dT * t / (rho_ice * L_f))

This analytic solution is used as a verification benchmark (C2: <3% error).

Negative control
----------------
When the sticking efficiency ``eta_stick = 0`` and there is no conductive
growth (``dT = 0``), the ice thickness must remain identically zero
(C3: <1e-15).

Detachment limiting
-------------------
During integration, if the wall shear stress exceeds the adhesive strength
of the ice-surface bond, the ice layer is stripped and growth resets to zero.
This ensures the monotonicity property (C4): higher velocity → lower steady
state thickness.

Numerical integration
---------------------
The ODE is integrated using forward Euler (numpy-only). The integration
stops or resets when detachment occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import (
    RHO_ICE, RHO_WATER, K_ICE, L_FUSION, C_P_WATER, T_FREEZE_FW,
    DEFAULT_STICKING_EFFICIENCY, DEFAULT_BOND_STRENGTH_PA,
)
from .flux import frazil_impact_flux, FrazilFluxState
from .detach import wall_shear_stress, adhesive_strength


@dataclass(frozen=True)
class GrowthResult:
    """Result of ice growth integration."""

    #: Time array [s]
    t: np.ndarray
    #: Ice thickness array [m]
    h_ice: np.ndarray
    #: Energy residual array [W/m^2] (should be ~0 at each step)
    energy_residual: np.ndarray
    #: Final thickness [m]
    h_final: float
    #: Time to steady state or detachment [s] (inf if not reached)
    t_steady: float
    #: Whether detachment occurred during integration
    detached: bool
    #: Time of first detachment [s] (inf if never)
    t_detach: float

    def as_dict(self) -> dict:
        return {
            "t_s": self.t.tolist(),
            "h_ice_m": self.h_ice.tolist(),
            "h_final_m": self.h_final,
            "t_steady_s": self.t_steady,
            "detached": self.detached,
            "t_detach_s": self.t_detach,
        }


def stefan_analytic(
    undercooling_k: float,
    time_s: float,
    k_ice: float = K_ICE,
    rho_ice: float = RHO_ICE,
    l_fusion: float = L_FUSION,
) -> float:
    """Analytic Stefan solution for ice thickness [m].

    ``h(t) = sqrt(2 * k * dT * t / (rho * L))``

    Parameters
    ----------
    undercooling_k : float
        Temperature difference T_freeze - T_surface [K].
    time_s : float
        Time [s].
    k_ice, rho_ice, l_fusion : float
        Physical constants.

    Returns
    -------
    float
        Ice thickness [m].
    """
    dT = float(undercooling_k)
    t = float(time_s)
    if dT < 0.0 or t < 0.0:
        raise ValueError("undercooling and time must be non-negative")
    if dT == 0.0 or t == 0.0:
        return 0.0
    return float(np.sqrt(2.0 * k_ice * dT * t / (rho_ice * l_fusion)))


def energy_residual(
    h_ice: float,
    dh_dt: float,
    undercooling_k: float,
    convective_flux: float,
    frazil_flux_kg_m2s: float,
    eta_stick: float,
    k_ice: float = K_ICE,
    rho_ice: float = RHO_ICE,
    l_fusion: float = L_FUSION,
) -> float:
    """Compute the energy balance residual [W/m^2].

    Residual = rho*L*dh/dt - (q_cond - q_conv + q_frazil)
    Should be ~0 when the ODE is satisfied.

    Parameters
    ----------
    h_ice : float
        Current ice thickness [m].
    dh_dt : float
        Growth rate [m/s].
    undercooling_k : float
        T_freeze - T_surface [K].
    convective_flux : float
        Convective heat flux from water [W/m^2].
    frazil_flux_kg_m2s : float
        Frazil mass flux [kg/(m^2*s)].
    eta_stick : float
        Sticking efficiency.
    k_ice, rho_ice, l_fusion : float
        Physical constants.

    Returns
    -------
    float
        Energy residual [W/m^2].
    """
    if h_ice <= 0.0:
        q_cond = 0.0
    else:
        q_cond = k_ice * undercooling_k / h_ice

    q_frazil = eta_stick * frazil_flux_kg_m2s * l_fusion

    lhs = rho_ice * l_fusion * dh_dt
    rhs = q_cond - convective_flux + q_frazil

    return float(lhs - rhs)


class IceGrowthODE:
    """Ice growth ODE: dh/dt = f(h, params).

    The ODE right-hand side is::

        dh/dt = (q_cond - q_conv + q_frazil) / (rho_ice * L_f)

    where q_cond depends on h (nonlinear), making this a nonlinear ODE.

    Detachment is checked at each step: if wall shear stress exceeds
    adhesive strength, the ice layer is stripped (h reset to 0).
    """

    def __init__(
        self,
        undercooling_k: float,
        u_star: float,
        c_f: float,
        eta_stick: float = DEFAULT_STICKING_EFFICIENCY,
        t_water_k: Optional[float] = None,
        particle_diameter_m: float = 1.0e-4,
        k_ice: float = K_ICE,
        rho_ice: float = RHO_ICE,
        l_fusion: float = L_FUSION,
        c_p_water: float = C_P_WATER,
        rho_water: float = RHO_WATER,
        tau_bond: float = DEFAULT_BOND_STRENGTH_PA,
    ):
        """Initialise the growth ODE.

        Parameters
        ----------
        undercooling_k : float
            T_freeze - T_surface [K].
        u_star : float
            Friction velocity [m/s].
        c_f : float
            Frazil volume concentration.
        eta_stick : float
            Sticking efficiency (0 to 1).
        t_water_k : float, optional
            Water temperature [K]. If None, assumed = T_freeze.
        particle_diameter_m : float
            Frazil particle diameter [m].
        k_ice, rho_ice, l_fusion, c_p_water, rho_water : float
            Physical constants.
        tau_bond : float
            Bond strength [Pa] for detachment check.
        """
        self.undercooling_k = float(undercooling_k)
        self.u_star = float(u_star)
        self.c_f = float(c_f)
        self.eta_stick = float(eta_stick)
        self.t_water_k = t_water_k if t_water_k is not None else T_FREEZE_FW
        self.particle_diameter_m = float(particle_diameter_m)
        self.k_ice = float(k_ice)
        self.rho_ice = float(rho_ice)
        self.l_fusion = float(l_fusion)
        self.c_p_water = float(c_p_water)
        self.rho_water = float(rho_water)
        self.tau_bond = float(tau_bond)

        # Pre-compute frazil flux
        self._flux_state = frazil_impact_flux(
            c_f=self.c_f,
            u_star=self.u_star,
            particle_diameter_m=self.particle_diameter_m,
        )
        self._frazil_mass_flux = self._flux_state.mass_flux

        # Convective heat transfer coefficient (Reynolds analogy, Pr_t ~ 1)
        self._h_c = self.rho_water * self.c_p_water * self.u_star

        # Convective flux (positive when water is warmer than freezing)
        delta_t_conv = self.t_water_k - T_FREEZE_FW
        self._convective_flux = self._h_c * delta_t_conv

        # Wall shear stress (constant for given u_star)
        self._tau_wall = wall_shear_stress(self.u_star, self.rho_water)

    def _check_detachment(self, h_ice: float) -> bool:
        """Check if ice detaches at given thickness."""
        tau_adh = adhesive_strength(h_ice, self.tau_bond)
        return self._tau_wall > tau_adh

    def rhs(self, h_ice: float) -> float:
        """Right-hand side of the growth ODE: dh/dt [m/s].

        Parameters
        ----------
        h_ice : float
            Current ice thickness [m].

        Returns
        -------
        float
            Growth rate [m/s].
        """
        h = float(h_ice)
        if h < 0.0:
            raise ValueError("ice thickness must be non-negative")

        # Check detachment first
        if self._check_detachment(h):
            return 0.0  # no growth when detached

        # Conductive flux through ice
        if h <= 1e-15:
            q_cond = 0.0
        else:
            q_cond = self.k_ice * self.undercooling_k / h

        # Frazil contribution
        q_frazil = self.eta_stick * self._frazil_mass_flux * self.l_fusion

        # Net energy flux
        q_net = q_cond - self._convective_flux + q_frazil

        return float(q_net / (self.rho_ice * self.l_fusion))

    def solve_euler(
        self,
        t_max: float,
        n_steps: int = 1000,
        h_init: float = 0.0,
    ) -> GrowthResult:
        """Solve the ODE using forward Euler (numpy-only, no scipy).

        Parameters
        ----------
        t_max : float
            Maximum integration time [s].
        n_steps : int
            Number of time steps.
        h_init : float
            Initial ice thickness [m].

        Returns
        -------
        GrowthResult
        """
        t = np.linspace(0.0, float(t_max), int(n_steps) + 1)
        dt = t[1] - t[0]
        h = np.zeros_like(t)
        h[0] = float(h_init)
        res = np.zeros_like(t)
        detached = False
        t_detach = float("inf")

        for i in range(n_steps):
            # Check detachment before growth
            if self._check_detachment(h[i]) and h[i] > 1e-15:
                h[i + 1] = 0.0  # strip ice
                if not detached:
                    detached = True
                    t_detach = float(t[i])
                res[i] = 0.0
                continue

            dh = self.rhs(h[i]) * dt
            h[i + 1] = max(h[i] + dh, 0.0)  # enforce non-negativity

            # Compute energy residual at this step
            if h[i] > 1e-15:
                dh_dt = (h[i + 1] - h[i]) / dt if dt > 0 else 0.0
                res[i] = energy_residual(
                    h_ice=h[i],
                    dh_dt=dh_dt,
                    undercooling_k=self.undercooling_k,
                    convective_flux=self._convective_flux,
                    frazil_flux_kg_m2s=self._frazil_mass_flux,
                    eta_stick=self.eta_stick,
                    k_ice=self.k_ice,
                    rho_ice=self.rho_ice,
                    l_fusion=self.l_fusion,
                )
            else:
                res[i] = 0.0

        # Final residual
        if h[-1] > 1e-15:
            dh_dt = self.rhs(h[-1])
            res[-1] = energy_residual(
                h_ice=h[-1],
                dh_dt=dh_dt,
                undercooling_k=self.undercooling_k,
                convective_flux=self._convective_flux,
                frazil_flux_kg_m2s=self._frazil_mass_flux,
                eta_stick=self.eta_stick,
                k_ice=self.k_ice,
                rho_ice=self.rho_ice,
                l_fusion=self.l_fusion,
            )

        # Estimate steady-state time
        initial_rate = self.rhs(1e-10) if self.rhs(1e-10) > 0 else self.rhs(1e-6)
        if initial_rate > 0:
            threshold = 0.01 * initial_rate
            rates = np.array([self.rhs(hi) for hi in h])
            steady_idx = np.where(rates <= threshold)[0]
            t_steady = float(t[steady_idx[0]]) if len(steady_idx) > 0 else float(t_max)
        else:
            t_steady = 0.0

        return GrowthResult(
            t=t,
            h_ice=h,
            energy_residual=res,
            h_final=float(h[-1]),
            t_steady=t_steady,
            detached=detached,
            t_detach=t_detach,
        )


def solve_growth(
    undercooling_k: float,
    u_star: float,
    c_f: float,
    t_max: float,
    eta_stick: float = DEFAULT_STICKING_EFFICIENCY,
    n_steps: int = 1000,
    **kwargs,
) -> GrowthResult:
    """Convenience function to solve the ice growth ODE.

    Parameters
    ----------
    undercooling_k : float
        T_freeze - T_surface [K].
    u_star : float
        Friction velocity [m/s].
    c_f : float
        Frazil volume concentration.
    t_max : float
        Maximum integration time [s].
    eta_stick : float
        Sticking efficiency.
    n_steps : int
        Number of time steps.

    Returns
    -------
    GrowthResult
    """
    ode = IceGrowthODE(
        undercooling_k=undercooling_k,
        u_star=u_star,
        c_f=c_f,
        eta_stick=eta_stick,
        **kwargs,
    )
    return ode.solve_euler(t_max=t_max, n_steps=n_steps)
