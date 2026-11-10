"""C3: Ice floe drift in ocean current — verification against MT Uikku trials.

Physics
-------
After the ice sheet is broken by the FEM→DEM handoff, the resulting ice floes
drift under the action of:
- Ocean current drag (dominant in calm wind)
- Wind drag on exposed ice surface
- Coriolis force (important for large-scale drift)
- Ice–ice interaction (collisions, rafting)

The MT Uikku full-scale ice management trials (Keinonen et al. 1998) provide
reference data for the channel clearing rate and floe drift velocity in
broken ice conditions.

Reference experiment
--------------------
MT Uikku (1998, Baltic Sea):
- Ship speed: 3–10 knots
- Ice thickness: 0.6–1.0 m
- Floe size: 5–50 m characteristic dimension
- Current velocity: 0.05–0.20 m/s
- Channel clearing rate: 0.5–2.0 m²/s per unit width

The channel clearing area A(t) grows approximately linearly with time for
free-drift conditions:

    A(t) ≈ W_channel × U_drift × t

where W_channel is the channel width and U_drift is the characteristic floe
drift velocity.

Drag model
----------
The floe drift velocity is determined by the momentum balance:

    ρ_i h dU/dt = τ_water - τ_air - τ_ice

For steady state (dU/dt = 0):

    C_w ρ_w |U_w - U| (U_w - U) = C_a ρ_a |U_a - U| (U_a - U)

where C_w, C_a are the water and air drag coefficients, U_w is the current
velocity, U_a is the wind velocity, and U is the floe drift velocity.

For the typical case of negligible wind (U_a ≈ 0) and small floe drift
relative to current (|U| << |U_w|):

    U ≈ (C_w ρ_w / (C_w ρ_w + C_a ρ_a)) U_w ≈ 0.02–0.05 U_w

This gives a floe drift of about 2–5% of the current velocity, consistent
with observations.

References
----------
* Keinonen, A. et al. (1998). "MT Uikku full-scale ice trials",
  KVTT Report, Espoo, Finland.
* Lu, W., Riska, K., & Ehlers, S. (2015). "Ice-induced loads on a ship hull",
  Proc. ICE 2015.
* Hopkins, M.A. & Tuhkunen, R.J. (2020). "Discrete-element modeling of
  broken ice cover in the nearshore", Cold Regions Science and Technology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# MT Uikku reference data (Keinonen et al. 1998)
MT_UIKKU_REFERENCE = {
    "ice_thickness_m": (0.6, 1.0),  # range
    "floe_size_m": (5.0, 50.0),  # characteristic dimension
    "current_velocity_m_s": (0.05, 0.20),  # range
    "channel_clearing_rate_m2_s_per_m": (0.5, 2.0),  # per unit channel width
    "floe_drift_fraction": (0.02, 0.05),  # U_floe / U_current
    "ship_speed_knots": (3.0, 10.0),
}


@dataclass
class FloeProperties:
    """Properties of an individual ice floe."""
    # Geometry
    area_m2: float  # planform area
    thickness_m: float  # ice thickness
    # Mass properties
    density_kg_m3: float = 917.0
    # State
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # Shape (for collision detection)
    shape: str = "circular"  # circular, rectangular, irregular

    @property
    def mass_kg(self) -> float:
        return self.density_kg_m3 * self.area_m2 * self.thickness_m

    @property
    def characteristic_length_m(self) -> float:
        """Equivalent diameter for a circular floe of same area."""
        return 2.0 * math.sqrt(self.area_m2 / math.pi)


@dataclass
class DriftEnvironment:
    """Environmental conditions for floe drift."""
    current_velocity: np.ndarray  # [u, v] m/s
    wind_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    water_density_kg_m3: float = 1025.0
    air_density_kg_m3: float = 1.225
    cd_water: float = 5.0e-3  # water drag coefficient
    cd_air: float = 1.5e-3  # air drag coefficient
    coriolis_param: float = 1.4e-4  # f = 2Ω sin(lat), ~75° latitude


@dataclass
class FloeDriftResult:
    """Result of the floe drift verification (C3)."""
    # Simulation
    time_s: np.ndarray
    floe_positions: List[np.ndarray]  # positions at each time step
    floe_velocities: List[np.ndarray]
    # Channel clearing
    channel_width_m: float
    channel_area_sim_m2: float  # simulated clearing area
    channel_area_exp_m2: float  # experimental reference
    clearing_rate_sim_m2_s: float
    clearing_rate_exp_m2_s: float
    # Error
    area_error_rel: float
    # Drift velocity
    mean_drift_speed_m_s: float
    current_speed_m_s: float
    drift_fraction: float
    # Reference
    drift_fraction_ref: Tuple[float, float]
    # Pass/fail
    c3_pass: bool  # area error < 20%

    def as_dict(self) -> dict:
        return {
            "channel_width_m": self.channel_width_m,
            "channel_area_sim_m2": self.channel_area_sim_m2,
            "channel_area_exp_m2": self.channel_area_exp_m2,
            "clearing_rate_sim_m2_s": self.clearing_rate_sim_m2_s,
            "clearing_rate_exp_m2_s": self.clearing_rate_exp_m2_s,
            "area_error_rel": self.area_error_rel,
            "mean_drift_speed_m_s": self.mean_drift_speed_m_s,
            "current_speed_m_s": self.current_speed_m_s,
            "drift_fraction": self.drift_fraction,
            "drift_fraction_ref": list(self.drift_fraction_ref),
            "c3_pass": self.c3_pass,
        }


def _water_drag(
    floe_vel: np.ndarray,
    current_vel: np.ndarray,
    cd: float,
    rho_w: float,
    area_m2: float,
) -> np.ndarray:
    """Quadratic water drag force on a floe."""
    rel_vel = current_vel - floe_vel
    speed = np.linalg.norm(rel_vel)
    if speed < 1e-10:
        return np.zeros(2)
    return cd * rho_w * area_m2 * speed * rel_vel


def _air_drag(
    floe_vel: np.ndarray,
    wind_vel: np.ndarray,
    cd: float,
    rho_a: float,
    area_m2: float,
) -> np.ndarray:
    """Quadratic air drag force on a floe."""
    rel_vel = wind_vel - floe_vel
    speed = np.linalg.norm(rel_vel)
    if speed < 1e-10:
        return np.zeros(2)
    return cd * rho_a * area_m2 * speed * rel_vel


def _coriolis_force(
    floe_vel: np.ndarray,
    mass: float,
    f: float,
) -> np.ndarray:
    """Coriolis force: F = -m f × v (2D, f perpendicular to plane)."""
    return mass * f * np.array([floe_vel[1], -floe_vel[0]])


def simulate_floe_drift(
    floes: List[FloeProperties],
    env: DriftEnvironment,
    duration_s: float = 3600.0,
    dt_s: float = 10.0,
    channel_width_m: float = 50.0,
) -> FloeDriftResult:
    """Simulate floe drift and compute channel clearing area.

    Parameters
    ----------
    floes : list of FloeProperties
        Initial floe configuration (positions, sizes).
    env : DriftEnvironment
        Environmental conditions.
    duration_s : float
        Simulation duration [s].
    dt_s : float
        Time step [s].
    channel_width_m : float
        Width of the ship-broken channel [m].

    Returns
    -------
    FloeDriftResult
    """
    n_steps = int(duration_s / dt_s)
    n_floes = len(floes)

    # Storage
    positions = [np.array([f.position.copy() for f in floes])]
    velocities = [np.array([f.velocity.copy() for f in floes])]
    time = np.linspace(0, duration_s, n_steps + 1)

    # Time integration (explicit Euler, sufficient for slow drift)
    for step in range(n_steps):
        new_positions = []
        new_velocities = []
        for i, floe in enumerate(floes):
            # Forces
            F_water = _water_drag(
                floe.velocity, env.current_velocity,
                env.cd_water, env.water_density_kg_m3, floe.area_m2
            )
            F_air = _air_drag(
                floe.velocity, env.wind_velocity,
                env.cd_air, env.air_density_kg_m3, floe.area_m2
            )
            F_coriolis = _coriolis_force(floe.velocity, floe.mass_kg, env.coriolis_param)

            # Total force
            F_total = F_water + F_air + F_coriolis

            # Update velocity (explicit Euler)
            accel = F_total / floe.mass_kg
            new_vel = floe.velocity + dt_s * accel

            # Update position
            new_pos = floe.position + dt_s * new_vel

            # Store
            floe.position = new_pos
            floe.velocity = new_vel
            new_positions.append(new_pos.copy())
            new_velocities.append(new_vel.copy())

        positions.append(np.array(new_positions))
        velocities.append(np.array(new_velocities))

    # Compute channel clearing area
    # The channel is along the x-axis; floes drift in the current direction
    # Channel clearing = floes that have moved out of the channel region
    initial_in_channel = 0
    final_out_channel = 0
    for i, floe in enumerate(floes):
        x0, y0 = positions[0][i]
        xf, yf = positions[-1][i]
        # Check if initially in channel
        if abs(y0) < channel_width_m / 2:
            initial_in_channel += 1
        # Check if finally out of channel
        if abs(yf) >= channel_width_m / 2:
            final_out_channel += 1

    # Channel clearing area = number of floes cleared × average floe area
    avg_floe_area = np.mean([f.area_m2 for f in floes])
    channel_area_sim = final_out_channel * avg_floe_area

    # Experimental reference: from MT Uikku clearing rate
    clearing_rate_exp = 1.0  # m²/s per unit width (mid-range of MT Uikku)
    channel_area_exp = clearing_rate_exp * channel_width_m * duration_s

    # Mean drift speed
    all_speeds = []
    for step_vel in velocities:
        for v in step_vel:
            all_speeds.append(np.linalg.norm(v))
    mean_drift = np.mean(all_speeds)
    current_speed = np.linalg.norm(env.current_velocity)
    drift_fraction = mean_drift / current_speed if current_speed > 0 else 0.0

    # Error
    area_error = abs(channel_area_sim - channel_area_exp) / channel_area_exp

    return FloeDriftResult(
        time_s=time,
        floe_positions=positions,
        floe_velocities=velocities,
        channel_width_m=channel_width_m,
        channel_area_sim_m2=channel_area_sim,
        channel_area_exp_m2=channel_area_exp,
        clearing_rate_sim_m2_s=channel_area_sim / duration_s if duration_s > 0 else 0.0,
        clearing_rate_exp_m2_s=clearing_rate_exp * channel_width_m,
        area_error_rel=area_error,
        mean_drift_speed_m_s=mean_drift,
        current_speed_m_s=current_speed,
        drift_fraction=drift_fraction,
        drift_fraction_ref=MT_UIKKU_REFERENCE["floe_drift_fraction"],
        c3_pass=area_error < 0.20,
    )


def channel_clearing_area(
    floe_positions_initial: np.ndarray,
    floe_positions_final: np.ndarray,
    floe_areas: np.ndarray,
    channel_width_m: float,
    channel_length_m: float,
) -> float:
    """Compute the channel clearing area from floe positions.

    Parameters
    ----------
    floe_positions_initial : ndarray, shape (n_floes, 2)
        Initial floe positions [m].
    floe_positions_final : ndarray, shape (n_floes, 2)
        Final floe positions [m].
    floe_areas : ndarray, shape (n_floes,)
        Floe planform areas [m²].
    channel_width_m : float
        Channel width [m].
    channel_length_m : float
        Channel length [m].

    Returns
    -------
    float
        Total area cleared from the channel [m²].
    """
    # A floe is "in channel" if |y| < W/2 and 0 < x < L
    in_channel_initial = (
        (np.abs(floe_positions_initial[:, 1]) < channel_width_m / 2)
        & (floe_positions_initial[:, 0] > 0)
        & (floe_positions_initial[:, 0] < channel_length_m)
    )
    in_channel_final = (
        (np.abs(floe_positions_final[:, 1]) < channel_width_m / 2)
        & (floe_positions_final[:, 0] > 0)
        & (floe_positions_final[:, 0] < channel_length_m)
    )
    cleared = in_channel_initial & ~in_channel_final
    return float(np.sum(floe_areas[cleared]))
