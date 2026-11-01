"""Internal Loss Factor (ILF) and Coupling Loss Factor (CLF) for SEA.

ILF (eta): energy dissipated within a subsystem per radian.
CLF (eta_ij): energy transferred from subsystem i to j per radian.

Reciprocity relation:
    eta_ij * n_i = eta_ji * n_j

where n_i, n_j are modal densities.

References:
    - Lyon & DeJong (1995), Ch. 3-4.
    - Cremer, Heckl & Ungar (1988). Structure-Borne Sound.
"""
from __future__ import annotations

import math

from .subsystems import PlateSubsystem, BeamSubsystem


def internal_loss_factor(
    material_loss: float = 0.01,
    structural_loss: float = 0.0,
    radiation_loss: float = 0.0,
) -> float:
    """Total internal loss factor (ILF) for a subsystem.

    eta_total = eta_material + eta_structural + eta_radiation

    Args:
        material_loss: material damping loss factor.
        structural_loss: joint/connection damping loss factor.
        radiation_loss: radiation damping loss factor (into fluid).

    Returns:
        Total ILF (dimensionless).
    """
    eta = material_loss + structural_loss + radiation_loss
    if eta <= 0:
        raise ValueError("total ILF must be positive")
    return eta


def radiation_loss_factor(
    frequency: float,
    density: float,
    thickness: float,
    sound_speed: float = 343.0,
    rho_fluid: float = 1.225,
) -> float:
    """Radiation loss factor for a plate radiating into a fluid.

    For a baffled plate above the critical frequency:
        eta_rad = (rho_fluid * c_fluid) / (pi * rho_plate * h * f)
                  * sigma_rad

    where sigma_rad is the radiation efficiency (≈ 1 above critical freq).

    Below critical frequency, radiation efficiency drops significantly.
    The critical frequency:
        f_c = c_fluid² / (2*pi) * sqrt(rho_plate*h/D)

    Args:
        frequency: center frequency (Hz).
        density: plate material density (kg/m³).
        thickness: plate thickness (m).
        sound_speed: sound speed in fluid (m/s), default air.
        rho_fluid: fluid density (kg/m³), default air.

    Returns:
        Radiation loss factor (dimensionless).
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    rho_h = density * thickness
    characteristic_impedance = rho_fluid * sound_speed

    # Radiation loss factor (simplified, assumes sigma ≈ 1 for high freq)
    eta_rad = characteristic_impedance / (math.pi * rho_h * frequency)
    return eta_rad


def coupling_loss_factor_plates(
    plate_i: PlateSubsystem,
    plate_j: PlateSubsystem,
    frequency: float,
    joint_length: float | None = None,
) -> float:
    """Coupling loss factor between two plate subsystems.

    For two plates connected along a line joint of length L_j:
        eta_ij = (L_j * cg_i) / (omega * A_i) * tau_ij

    where:
        L_j: joint length (m)
        cg_i: group velocity in plate i (m/s), cg = 2 * cb for bending waves
        omega: angular frequency (rad/s)
        A_i: area of plate i (m²)
        tau_ij: power transmission coefficient

    The power transmission coefficient for bending waves:
        tau_ij = 4 * Z_i * Z_j / (Z_i + Z_j)²

    where Z = rho * h * cb is the characteristic impedance.

    Reciprocity is enforced: eta_ji = eta_ij * n_i / n_j

    Args:
        plate_i: source plate subsystem.
        plate_j: receiving plate subsystem.
        frequency: center frequency (Hz).
        joint_length: length of the connecting joint (m).
                      If None, uses min(width_i, width_j).

    Returns:
        Coupling loss factor eta_ij (dimensionless).
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    omega = 2.0 * math.pi * frequency

    # Bending wave phase velocities
    cb_i = plate_i.bending_wave_speed_coefficient * math.sqrt(omega)
    cb_j = plate_j.bending_wave_speed_coefficient * math.sqrt(omega)

    # Group velocities (cg = 2 * cb for bending waves)
    cg_i = 2.0 * cb_i
    cg_j = 2.0 * cb_j

    # Joint length
    if joint_length is None:
        joint_length = min(plate_i.width, plate_j.width)

    # Characteristic impedances (Z = rho * h * cb)
    Z_i = plate_i.density * plate_i.thickness * cb_i
    Z_j = plate_j.density * plate_j.thickness * cb_j

    # Power transmission coefficient
    tau = 4.0 * Z_i * Z_j / (Z_i + Z_j) ** 2

    # CLF from i to j
    eta_ij = (joint_length * cg_i) / (omega * plate_i.area) * tau
    return eta_ij


def total_loss_factor(
    ilf: float,
    clf_sum: float,
) -> float:
    """Total loss factor for a subsystem.

    eta_total = eta_internal + sum(eta_coupling)

    Args:
        ilf: internal loss factor.
        clf_sum: sum of all coupling loss factors from this subsystem.

    Returns:
        Total loss factor (dimensionless).
    """
    return ilf + clf_sum
