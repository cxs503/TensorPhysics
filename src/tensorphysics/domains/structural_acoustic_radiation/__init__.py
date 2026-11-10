"""Structural acoustic radiation: vibration → sound power → far-field pressure.

This module implements the classical chain for predicting underwater sound
radiation from a vibrating structure:

1. Structural vibration: harmonic excitation F(t) = F_0 e^{iωt}, mean-square
   velocity <v²> = v_0²/2.
2. Radiated sound power: W_rad = σ_rad · ρ · c · S · <v²>, where σ_rad(ω)
   is the frequency-dependent radiation efficiency.
3. Far-field sound pressure: p(r) = √(ρcW_rad/(4πr²)) · e^{-ikr}.

Analytical anchors (closed-form solutions):
- Pulsating sphere: exact radiation impedance and efficiency.
- Circular piston in infinite baffle: exact radiation efficiency via Bessel
  functions.

References:
- Morse & Ingard, "Theoretical Acoustics" (1968).
- Junger & Feit, "Sound, Structures, and Their Interaction" (1986).
- Williams, "Engineering Noise Analysis" (2013).
"""

from .radiation import (
    PulsatingSphere,
    CircularPiston,
    RadiationChain,
    dipole_radiation_efficiency,
    far_field_pressure,
    piston_radiation_efficiency,
    pulsating_sphere_radiation_efficiency,
    radiated_power,
    semi_empirical_radiation_efficiency,
    sound_pressure_level,
)
from .vibration import HarmonicVibrator, chain_provenance, vibration_radiation_chain

__all__ = [
    "PulsatingSphere",
    "CircularPiston",
    "RadiationChain",
    "HarmonicVibrator",
    "pulsating_sphere_radiation_efficiency",
    "piston_radiation_efficiency",
    "dipole_radiation_efficiency",
    "semi_empirical_radiation_efficiency",
    "far_field_pressure",
    "sound_pressure_level",
    "radiated_power",
    "vibration_radiation_chain",
    "chain_provenance",
]
