"""Statistical Energy Analysis (SEA) for ice-induced high-frequency vibration and noise.

Implements subsystem energy flow, coupling loss factors, and ice-load PSD input
for predicting structural vibration (>500 Hz) and radiated noise of ship hulls
under ice loading.

Core modules:
    - subsystems: plate, beam, cavity subsystem definitions
    - modal_density: analytical modal density n(f) for each subsystem type
    - loss_factors: internal loss factor (ILF) and coupling loss factor (CLF)
    - power_input: ice load PSD → input power Pin(f)
    - sea_solver: SEA energy balance equation solver
    - response: vibration velocity level, sound power level, sound pressure level
"""
from .subsystems import PlateSubsystem, BeamSubsystem, CavitySubsystem, Subsystem
from .modal_density import (
    plate_modal_density, beam_modal_density, cavity_modal_density,
    modal_density_wave_theory,
)
from .loss_factors import (
    internal_loss_factor, coupling_loss_factor_plates,
    radiation_loss_factor, total_loss_factor,
)
from .power_input import ice_load_psd, input_power_from_psd
from .sea_solver import SEASystem, solve_sea
from .response import (
    vibration_velocity_level, sound_power_level, sound_pressure_level,
)

__all__ = [
    "PlateSubsystem", "BeamSubsystem", "CavitySubsystem", "Subsystem",
    "plate_modal_density", "beam_modal_density", "cavity_modal_density",
    "modal_density_wave_theory",
    "internal_loss_factor", "coupling_loss_factor_plates",
    "radiation_loss_factor", "total_loss_factor",
    "ice_load_psd", "input_power_from_psd",
    "SEASystem", "solve_sea",
    "vibration_velocity_level", "sound_power_level", "sound_pressure_level",
]
