"""Pressure melting – refreezing bond cycle for ice–structure contact.

Physical model
--------------
When ice slides against a structure under high contact pressure, the melting
point is depressed by the Clausius–Clapeyron relation.  If the environmental
temperature exceeds the depressed melting point, a thin water film forms.
This film reduces the effective contact force (lubrication).  When the pressure
is released, the water refreezes and creates a new adhesive bond.

State machine
-------------
The contact cycles through four states::

    solid ──(P > P_melt)──► melting ──(P < P_melt)──► refreezing ──(σ_bond ≥ σ_thresh)──► bonded
      ▲                                                                                    │
      └────────────────────(bond broken by load)───────────────────────────────────────────┘

Governing equations
-------------------
1. Pressure melting threshold (Clausius–Clapeyron):

       P_melt = (L_f / Δv) · ln(T_env / T_m0)

   where Δv = v_water - v_ice < 0 is the specific volume change on fusion.
   This is the exact inverse of T(p) = T_m0 · exp(Δv · p / L_f).

2. Water film evolution (melting state):

       dh_water/dt = (P - P_melt) / (ρ_ice · L_f) · v_sliding - h_water / τ_drain

   The first term is the melting rate driven by excess pressure; the second
   term is drainage out of the contact zone.

3. Contact force correction:

       F = k_n · δ_n · (1 - h_water / h_critical)

   The water film reduces the effective normal stiffness.  When h_water ≥
   h_critical the contact is fully lubricated (F → 0).

4. Refreezing (refreezing state):

       dh_water/dt = -h_water / τ_refreeze
       σ_bond,refreeze = σ_bond,0 · (1 - exp(-t_refreeze / τ_refreeze))

   The refreezing time constant τ_refreeze depends on the environmental
   temperature: colder environments freeze faster.

References
----------
* Clausius–Clapeyron relation for ice Ih (standard thermodynamics).
* Stefan problem for solidification (see :mod:`.stefan`).
* Ice adhesion strength model (see :mod:`.adhesion`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .properties import IceWaterProperties

__all__ = [
    "ContactStateEnum",
    "PressureMeltingCycle",
    "PressureMeltingCycleParams",
    "melt_rate",
    "refreeze_rate",
    "refreeze_time_constant",
    "bond_strength",
    "contact_force",
    "pressure_melting_threshold_linear",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ContactStateEnum(str, Enum):
    """State of the ice–structure contact interface."""
    SOLID = "solid"
    MELTING = "melting"
    REFREEZING = "refreezing"
    BONDED = "bonded"


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PressureMeltingCycleParams:
    """Parameters for the pressure melting–refreezing cycle model.

    All parameters have physically reasonable defaults for fresh-water ice
    sliding against carbon steel.
    """
    # Thermophysical
    props: IceWaterProperties = field(default_factory=IceWaterProperties)

    # Contact mechanics
    k_n: float = 1.0e9              # N/m³, normal contact stiffness per unit area
    h_critical: float = 1.0e-6      # m, critical film thickness for full lubrication
    tau_drain: float = 1.0e-3       # s, drainage time constant

    # Refreezing
    sigma_bond_0: float = 0.5e6     # Pa, maximum refrozen bond strength
    tau_refreeze_base: float = 0.1  # s, base refreezing time constant at ΔT = 1 K
    # Refreezing accelerates with supercooling: τ_refreeze = τ_base / (ΔT / 1K)
    # where ΔT = T_m0 - T_env

    # Bond breaking
    bond_break_strain: float = 1.0e-4  # dimensionless, strain at which bond breaks

    def __post_init__(self) -> None:
        for name in ("k_n", "h_critical", "tau_drain",
                     "sigma_bond_0", "tau_refreeze_base", "bond_break_strain"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a real number")
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")


# ---------------------------------------------------------------------------
# Core physics functions
# ---------------------------------------------------------------------------

def pressure_melting_threshold_linear(
    ice_temperature: float,
    props: Optional[IceWaterProperties] = None,
) -> float:
    """Pressure melting threshold [Pa] from the Clausius–Clapeyron relation.

    Uses the exact inverse of the Clapeyron integral:

        P_melt = (L_f / Δv) · ln(T_env / T_m0)

    where Δv = v_water - v_ice < 0 is the specific volume change on fusion.
    This agrees with the analytic Clapeyron solution to machine precision.
    """
    p = props if props is not None else IceWaterProperties()
    if ice_temperature <= 0.0:
        raise ValueError("ice_temperature must be a positive absolute temperature")
    if ice_temperature >= p.melting_temperature:
        return 0.0
    # P_melt = (L_f / Δv) · ln(T_env / T_m0)
    return (p.latent_heat_fusion / p.specific_volume_change) * math.log(
        ice_temperature / p.melting_temperature
    )


def melt_rate(
    pressure: float,
    p_melt: float,
    v_sliding: float,
    rho_ice: float,
    latent_heat: float,
    h_water: float,
    tau_drain: float,
) -> float:
    """Rate of water film thickness change [m/s] in the melting state.

    dh_water/dt = (P - P_melt) / (ρ_ice · L_f) · v_sliding - h_water / τ_drain
    """
    if pressure <= p_melt:
        # No melting; only drainage
        return -h_water / tau_drain
    melting_term = (pressure - p_melt) / (rho_ice * latent_heat) * v_sliding
    drainage_term = h_water / tau_drain
    return melting_term - drainage_term


def refreeze_rate(
    h_water: float,
    tau_refreeze: float,
) -> float:
    """Rate of water film thickness change [m/s] in the refreezing state.

    dh_water/dt = -h_water / τ_refreeze
    """
    return -h_water / tau_refreeze


def refreeze_time_constant(
    ice_temperature: float,
    props: Optional[IceWaterProperties] = None,
    tau_refreeze_base: float = 0.1,
) -> float:
    """Refreezing time constant [s] as a function of environmental temperature.

    τ_refreeze = τ_base / (ΔT / 1 K)  where ΔT = T_m0 - T_env

    Colder environments → smaller τ_refreeze → faster refreezing.
    """
    p = props if props is not None else IceWaterProperties()
    delta_t = p.melting_temperature - ice_temperature
    if delta_t <= 0.0:
        # At or above melting point: no refreezing
        return float("inf")
    return tau_refreeze_base / delta_t


def bond_strength(
    t_refreeze: float,
    tau_refreeze: float,
    sigma_bond_0: float,
) -> float:
    """Refrozen bond strength [Pa].

    σ_bond,refreeze = σ_bond,0 · (1 - exp(-t_refreeze / τ_refreeze))

    Properties:
    - t_refreeze = 0 → σ = 0
    - t_refreeze → ∞ → σ → σ_bond,0
    """
    if tau_refreeze <= 0.0 or tau_refreeze == float("inf"):
        return 0.0
    if t_refreeze <= 0.0:
        return 0.0
    return sigma_bond_0 * (1.0 - math.exp(-t_refreeze / tau_refreeze))


def contact_force(
    k_n: float,
    delta_n: float,
    h_water: float,
    h_critical: float,
) -> float:
    """Contact force per unit area [Pa] with water film correction.

    F = k_n · δ_n · (1 - h_water / h_critical)

    Properties:
    - h_water = 0 → F = k_n · δ_n (dry contact)
    - h_water → h_critical → F → 0 (fully lubricated)
    - h_water ≥ h_critical → F = 0 (no negative force)
    - Monotonically non-increasing with h_water
    """
    if h_water <= 0.0:
        return k_n * delta_n
    reduction = 1.0 - h_water / h_critical
    if reduction <= 0.0:
        return 0.0
    return k_n * delta_n * reduction


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

@dataclass
class PressureMeltingCycle:
    """State machine for the pressure melting–refreezing bond cycle.

    Tracks the contact state and evolves the water film thickness and bond
    strength over time.
    """
    params: PressureMeltingCycleParams = field(default_factory=PressureMeltingCycleParams)

    # State variables
    state: ContactStateEnum = ContactStateEnum.SOLID
    h_water: float = 0.0          # m, current water film thickness
    t_refreeze: float = 0.0       # s, accumulated refreezing time
    sigma_bond: float = 0.0       # Pa, current bond strength
    t_in_state: float = 0.0       # s, time spent in current state

    # History for analysis
    history: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        """Reset to initial solid state."""
        self.state = ContactStateEnum.SOLID
        self.h_water = 0.0
        self.t_refreeze = 0.0
        self.sigma_bond = 0.0
        self.t_in_state = 0.0
        self.history.clear()

    def _p_melt(self, ice_temperature: float) -> float:
        """Pressure melting threshold for the current temperature."""
        return pressure_melting_threshold_linear(ice_temperature, self.params.props)

    def _tau_refreeze(self, ice_temperature: float) -> float:
        """Refreezing time constant for the current temperature."""
        return refreeze_time_constant(
            ice_temperature, self.params.props, self.params.tau_refreeze_base
        )

    def step(
        self,
        dt: float,
        pressure: float,
        v_sliding: float,
        ice_temperature: float,
        delta_n: float,
    ) -> dict:
        """Advance the state machine by one time step.

        Parameters
        ----------
        dt : float
            Time step [s].
        pressure : float
            Current contact pressure [Pa].
        v_sliding : float
            Sliding velocity [m/s].
        ice_temperature : float
            Ice temperature [K].
        delta_n : float
            Normal overlap/displacement [m].

        Returns
        -------
        dict with state variables after the step.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        p_melt = self._p_melt(ice_temperature)
        tau_refreeze = self._tau_refreeze(ice_temperature)
        p = self.params

        # Determine target state based on pressure
        if pressure > p_melt and self.state != ContactStateEnum.MELTING:
            # Transition to melting
            if self.state == ContactStateEnum.BONDED:
                # Bond breaks when pressure exceeds melting threshold
                pass  # bond is broken by the transition
            self.state = ContactStateEnum.MELTING
            self.t_in_state = 0.0
        elif pressure <= p_melt and self.state == ContactStateEnum.MELTING:
            # Transition to refreezing
            self.state = ContactStateEnum.REFREEZING
            self.t_in_state = 0.0
            self.t_refreeze = 0.0
        elif self.state == ContactStateEnum.REFREEZING:
            # Check if bond has formed
            self.t_refreeze += dt
            new_sigma = bond_strength(self.t_refreeze, tau_refreeze, p.sigma_bond_0)
            if new_sigma >= 0.99 * p.sigma_bond_0:
                self.state = ContactStateEnum.BONDED
                self.sigma_bond = new_sigma
                self.t_in_state = 0.0
            else:
                self.sigma_bond = new_sigma

        # Evolve water film
        if self.state == ContactStateEnum.MELTING:
            dh = melt_rate(
                pressure, p_melt, v_sliding,
                p.props.rho_ice, p.props.latent_heat_fusion,
                self.h_water, p.tau_drain,
            )
            self.h_water = max(0.0, self.h_water + dh * dt)
            self.t_in_state += dt
        elif self.state == ContactStateEnum.REFREEZING:
            dh = refreeze_rate(self.h_water, tau_refreeze)
            self.h_water = max(0.0, self.h_water + dh * dt)
            self.t_in_state += dt
        elif self.state == ContactStateEnum.SOLID:
            self.h_water = 0.0
            self.t_in_state += dt
        elif self.state == ContactStateEnum.BONDED:
            # Bonded state: check if bond breaks
            # Bond breaks if the applied stress exceeds the bond strength
            applied_stress = p.k_n * delta_n
            if applied_stress > self.sigma_bond:
                self.state = ContactStateEnum.SOLID
                self.h_water = 0.0
                self.sigma_bond = 0.0
                self.t_refreeze = 0.0
                self.t_in_state = 0.0
            else:
                self.t_in_state += dt

        # Compute contact force
        f_contact = contact_force(p.k_n, delta_n, self.h_water, p.h_critical)

        result = {
            "state": self.state.value,
            "h_water": self.h_water,
            "sigma_bond": self.sigma_bond,
            "t_refreeze": self.t_refreeze,
            "p_melt": p_melt,
            "tau_refreeze": tau_refreeze,
            "f_contact": f_contact,
            "t_in_state": self.t_in_state,
        }
        self.history.append(result)
        return result

    def run_timeseries(
        self,
        times: list[float],
        pressures: list[float],
        v_sliding: float,
        ice_temperature: float,
        delta_n: float,
    ) -> list[dict]:
        """Run the state machine over a time series.

        Parameters
        ----------
        times : list[float]
            Time points [s].
        pressures : list[float]
            Pressure at each time point [Pa].
        v_sliding : float
            Sliding velocity [m/s].
        ice_temperature : float
            Ice temperature [K].
        delta_n : float
            Normal overlap [m].

        Returns
        -------
        list of state dictionaries.
        """
        if len(times) != len(pressures):
            raise ValueError("times and pressures must have the same length")
        if len(times) < 2:
            raise ValueError("need at least 2 time points")

        results = []
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            if dt <= 0.0:
                raise ValueError("times must be strictly increasing")
            r = self.step(dt, pressures[i], v_sliding, ice_temperature, delta_n)
            r["time"] = times[i]
            r["pressure"] = pressures[i]
            results.append(r)
        return results
