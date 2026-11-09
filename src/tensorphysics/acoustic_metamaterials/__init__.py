"""
Acoustic metamaterials module for elastic wave bandgap analysis.

This module implements verification surfaces for locally resonant acoustic
metamaterial plates, including:
- Bloch-Floquet periodic unit cell dispersion analysis
- Local resonance frequency anchoring
- Bandgap detection and characterization
- Finite plate transmission loss analysis
"""

from .plate_theory import PlateTheory, ThinPlateTheory
from .bloch_floquet import BlochFloquetAnalyzer
from .metamaterial_plate import MetamaterialPlate, Resonator
from .transmission import TransmissionAnalyzer

__all__ = [
    'PlateTheory',
    'ThinPlateTheory',
    'BlochFloquetAnalyzer',
    'MetamaterialPlate',
    'Resonator',
    'TransmissionAnalyzer',
]
