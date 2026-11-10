"""Cargo shift assessment per IMO CSS Code Annex 13.

Implements cargo sliding and overturning checks for ice-going vessels
under ice-zone roll/pitch accelerations.

References
----------
- IMO CSS Code (Code of Safe Practice for Cargo Stowage and Securing),
  Annex 13: Securing of cargo units on deck
- IMO MSC/Circ.1228: Guidance on securing arrangements
- DNV-GL: Cargo securing in ice conditions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

__all__ = [
    "CargoShiftCriteria",
    "CargoShiftResult",
    "imo_css_sliding_check",
    "imo_css_overturning_check",
    "transverse_acceleration",
    "longitudinal_acceleration",
]

# Standard gravity
G = 9.81  # m/s^2


@dataclass
class CargoShiftCriteria:
    """Parameters for cargo shift assessment.

    Attributes
    ----------
    cargo_mass_kg : float
        Mass of cargo unit (kg).
    cargo_width_m : float
        Width of cargo (transverse direction, m).
    cargo_height_m : float
        Height of cargo (m).
    cargo_length_m : float
        Length of cargo (longitudinal direction, m).
    friction_coefficient : float
        Coefficient of friction between cargo and deck.
        Typical: steel-on-steel ≈ 0.3, timber-on-steel ≈ 0.4.
    securing_strength_kn : float
        Total securing strength (kN) in the direction of interest.
    lashing_angle_deg : float
        Angle of lashing from horizontal (degrees).
    """
    cargo_mass_kg: float
    cargo_width_m: float
    cargo_height_m: float
    cargo_length_m: float
    friction_coefficient: float = 0.3
    securing_strength_kn: float = 0.0
    lashing_angle_deg: float = 30.0


def transverse_acceleration(
    roll_amplitude_deg: float,
    roll_period_s: float,
    vcg_above_roll_axis_m: float,
    transverse_offset_m: float = 0.0,
) -> float:
    """Compute maximum transverse acceleration from roll motion.

    a_y = ω² · r
    where ω = 2π/T and r = √(vcg² + offset²)

    Parameters
    ----------
    roll_amplitude_deg : float
        Roll amplitude (degrees).
    roll_period_s : float
        Roll period (s).
    vcg_above_roll_axis_m : float
        Vertical distance from roll axis to cargo CG (m).
    transverse_offset_m : float
        Transverse offset of cargo from roll axis (m).

    Returns
    -------
    float
        Maximum transverse acceleration (m/s^2).
    """
    roll_rad = np.radians(roll_amplitude_deg)
    omega = 2.0 * np.pi / roll_period_s
    r = np.sqrt(vcg_above_roll_axis_m**2 + transverse_offset_m**2)
    return omega**2 * r * roll_rad


def longitudinal_acceleration(
    pitch_amplitude_deg: float,
    pitch_period_s: float,
    vcg_above_pitch_axis_m: float,
    longitudinal_offset_m: float = 0.0,
    surge_acceleration_m_s2: float = 0.0,
) -> float:
    """Compute maximum longitudinal acceleration from pitch + surge.

    a_x = ω² · r + a_surge

    Parameters
    ----------
    pitch_amplitude_deg : float
        Pitch amplitude (degrees).
    pitch_period_s : float
        Pitch period (s).
    vcg_above_pitch_axis_m : float
        Vertical distance from pitch axis to cargo CG (m).
    longitudinal_offset_m : float
        Longitudinal offset of cargo from pitch axis (m).
    surge_acceleration_m_s2 : float
        Surge (heave-induced) acceleration (m/s^2).

    Returns
    -------
    float
        Maximum longitudinal acceleration (m/s^2).
    """
    pitch_rad = np.radians(pitch_amplitude_deg)
    omega = 2.0 * np.pi / pitch_period_s
    r = np.sqrt(vcg_above_pitch_axis_m**2 + longitudinal_offset_m**2)
    return omega**2 * r * pitch_rad + surge_acceleration_m_s2


@dataclass
class CargoShiftResult:
    """Result of cargo shift assessment.

    Attributes
    ----------
    transverse_accel_m_s2 : float
        Transverse acceleration (m/s^2).
    longitudinal_accel_m_s2 : float
        Longitudinal acceleration (m/s^2).
    sliding_force_transverse_n : float
        Transverse inertial force (N).
    sliding_force_longitudinal_n : float
        Longitudinal inertial force (N).
    friction_resistance_n : float
        Friction resistance force (N).
    securing_contribution_n : float
        Securing device contribution (N).
    sliding_margin_transverse : float
        Sliding safety margin (transverse): (resistance - force) / force.
    sliding_margin_longitudinal : float
        Sliding safety margin (longitudinal).
    is_sliding_safe : bool
        True if sliding margins > 0 in both directions.
    overturning_moment_n_m : float
        Overturning moment (N·m).
    restoring_moment_n_m : float
        Restoring moment from weight (N·m).
    overturning_margin : float
        Overturning safety margin: (restoring - overturning) / overturning.
    is_overturning_safe : bool
        True if overturning margin > 0.
    """
    transverse_accel_m_s2: float
    longitudinal_accel_m_s2: float
    sliding_force_transverse_n: float
    sliding_force_longitudinal_n: float
    friction_resistance_n: float
    securing_contribution_n: float
    sliding_margin_transverse: float
    sliding_margin_longitudinal: float
    is_sliding_safe: bool
    overturning_moment_n_m: float
    restoring_moment_n_m: float
    overturning_margin: float
    is_overturning_safe: bool


def imo_css_sliding_check(
    cargo: CargoShiftCriteria,
    transverse_accel: float,
    longitudinal_accel: float,
) -> Tuple[bool, float, float]:
    """IMO CSS Code sliding check.

    The sliding force must be resisted by friction + securing:
        F_slide ≤ μ · (m·g - F_vertical) + F_securing

    For simplicity, we neglect vertical acceleration component here
    (conservative assumption).

    Parameters
    ----------
    cargo : CargoShiftCriteria
        Cargo parameters.
    transverse_accel : float
        Transverse acceleration (m/s^2).
    longitudinal_accel : float
        Longitudinal acceleration (m/s^2).

    Returns
    -------
    (is_safe, margin_transverse, margin_longitudinal)
        margin = (resistance - force) / force; positive = safe.
    """
    m = cargo.cargo_mass_kg
    mu = cargo.friction_coefficient
    g = G

    # Inertial forces
    F_y = m * transverse_accel
    F_x = m * longitudinal_accel

    # Friction resistance (conservative: no vertical accel reduction)
    F_friction = mu * m * g

    # Securing contribution (projected along direction)
    lashing_rad = np.radians(cargo.lashing_angle_deg)
    F_sec = cargo.securing_strength_kn * 1000.0  # kN → N
    F_sec_y = F_sec * np.cos(lashing_rad)
    F_sec_x = F_sec * np.sin(lashing_rad)

    # Margins
    total_resistance_y = F_friction + F_sec_y
    total_resistance_x = F_friction + F_sec_x

    if F_y > 0:
        margin_y = (total_resistance_y - F_y) / F_y
    else:
        margin_y = np.inf

    if F_x > 0:
        margin_x = (total_resistance_x - F_x) / F_x
    else:
        margin_x = np.inf

    is_safe = margin_y > 0 and margin_x > 0

    return is_safe, margin_y, margin_x


def imo_css_overturning_check(
    cargo: CargoShiftCriteria,
    transverse_accel: float,
    longitudinal_accel: float,
) -> Tuple[bool, float]:
    """IMO CSS Code overturning check.

    Overturning moment from inertial forces:
        M_ot = F_y · h/2 + F_x · h/2

    Restoring moment from weight:
        M_rest = m·g · min(w/2, l/2)

    Parameters
    ----------
    cargo : CargoShiftCriteria
        Cargo parameters.
    transverse_accel : float
        Transverse acceleration (m/s^2).
    longitudinal_accel : float
        Longitudinal acceleration (m/s^2).

    Returns
    -------
    (is_safe, margin)
        margin = (M_rest - M_ot) / M_ot; positive = safe.
    """
    m = cargo.cargo_mass_kg
    g = G
    h = cargo.cargo_height_m
    w = cargo.cargo_width_m
    l = cargo.cargo_length_m

    # Inertial forces
    F_y = m * transverse_accel
    F_x = m * longitudinal_accel

    # Overturning moment (about the edge)
    M_ot = F_y * h / 2.0 + F_x * h / 2.0

    # Restoring moment (about the same edge)
    # Use the smaller lever arm for conservative estimate
    lever = min(w / 2.0, l / 2.0)
    M_rest = m * g * lever

    if M_ot > 0:
        margin = (M_rest - M_ot) / M_ot
    else:
        margin = np.inf

    is_safe = margin > 0

    return is_safe, margin


def assess_cargo_shift(
    cargo: CargoShiftCriteria,
    roll_amplitude_deg: float,
    roll_period_s: float,
    pitch_amplitude_deg: float,
    pitch_period_s: float,
    vcg_above_roll_axis_m: float = 5.0,
    vcg_above_pitch_axis_m: float = 5.0,
    surge_acceleration_m_s2: float = 0.0,
) -> CargoShiftResult:
    """Full cargo shift assessment.

    Parameters
    ----------
    cargo : CargoShiftCriteria
        Cargo parameters.
    roll_amplitude_deg : float
        Roll amplitude (degrees).
    roll_period_s : float
        Roll period (s).
    pitch_amplitude_deg : float
        Pitch amplitude (degrees).
    pitch_period_s : float
        Pitch period (s).
    vcg_above_roll_axis_m : float
        VCG above roll axis (m).
    vcg_above_pitch_axis_m : float
        VCG above pitch axis (m).
    surge_acceleration_m_s2 : float
        Surge acceleration (m/s^2).

    Returns
    -------
    CargoShiftResult
    """
    # Compute accelerations
    a_y = transverse_acceleration(
        roll_amplitude_deg, roll_period_s, vcg_above_roll_axis_m
    )
    a_x = longitudinal_acceleration(
        pitch_amplitude_deg, pitch_period_s, vcg_above_pitch_axis_m,
        surge_acceleration_m_s2=surge_acceleration_m_s2,
    )

    # Sliding check
    slide_safe, margin_y, margin_x = imo_css_sliding_check(cargo, a_y, a_x)

    # Overturning check
    ot_safe, margin_ot = imo_css_overturning_check(cargo, a_y, a_x)

    # Forces
    F_y = cargo.cargo_mass_kg * a_y
    F_x = cargo.cargo_mass_kg * a_x
    F_friction = cargo.friction_coefficient * cargo.cargo_mass_kg * G

    lashing_rad = np.radians(cargo.lashing_angle_deg)
    F_sec = cargo.securing_strength_kn * 1000.0
    F_sec_contribution = F_sec * np.cos(lashing_rad)

    # Moments
    M_ot = F_y * cargo.cargo_height_m / 2.0 + F_x * cargo.cargo_height_m / 2.0
    lever = min(cargo.cargo_width_m / 2.0, cargo.cargo_length_m / 2.0)
    M_rest = cargo.cargo_mass_kg * G * lever

    return CargoShiftResult(
        transverse_accel_m_s2=a_y,
        longitudinal_accel_m_s2=a_x,
        sliding_force_transverse_n=F_y,
        sliding_force_longitudinal_n=F_x,
        friction_resistance_n=F_friction,
        securing_contribution_n=F_sec_contribution,
        sliding_margin_transverse=margin_y,
        sliding_margin_longitudinal=margin_x,
        is_sliding_safe=slide_safe,
        overturning_moment_n_m=M_ot,
        restoring_moment_n_m=M_rest,
        overturning_margin=margin_ot,
        is_overturning_safe=ot_safe,
    )
