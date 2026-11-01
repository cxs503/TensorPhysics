"""SEA subsystem definitions for ship hull structures.

Each subsystem represents a group of modes that are statistically
equivalent within a frequency band. Typical subsystems:
- PlateSubsystem: hull plating, deck plating, bulkhead panels
- BeamSubsystem: stiffeners, frames, girders
- CavitySubsystem: enclosed fluid volumes (cargo holds, tanks)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class PlateSubsystem:
    """Flat or curved plate subsystem.

    Attributes:
        name: subsystem identifier
        length: plate length in x-direction (m)
        width: plate width in y-direction (m)
        thickness: plate thickness (m)
        young_modulus: Young's modulus (Pa)
        density: material density (kg/m³)
        poisson_ratio: Poisson's ratio (dimensionless)
        loss_factor: structural damping loss factor (dimensionless)
        boundary_condition: 'simply_supported' | 'clamped' | 'free'
        curvature_radius: radius of curvature (m); None for flat plate
    """
    name: str
    length: float
    width: float
    thickness: float
    young_modulus: float
    density: float
    poisson_ratio: float
    loss_factor: float = 0.01
    boundary_condition: Literal["simply_supported", "clamped", "free"] = "simply_supported"
    curvature_radius: float | None = None

    @property
    def area(self) -> float:
        return self.length * self.width

    @property
    def bending_stiffness(self) -> float:
        """Flexural rigidity D = E h³ / [12(1-ν²)]."""
        return (self.young_modulus * self.thickness ** 3
                / (12.0 * (1.0 - self.poisson_ratio ** 2)))

    @property
    def bending_wave_speed_coefficient(self) -> float:
        """Coefficient alpha in cb = alpha * sqrt(f), where cb is bending wave speed."""
        import math
        return (2.0 * math.pi * self.bending_stiffness
                / (self.density * self.thickness)) ** 0.25

    def __post_init__(self) -> None:
        for attr in ("length", "width", "thickness", "young_modulus", "density"):
            val = getattr(self, attr)
            if not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(f"{attr} must be positive, got {val}")
        if not (0 < self.poisson_ratio < 0.5):
            raise ValueError(f"poisson_ratio must be in (0, 0.5), got {self.poisson_ratio}")
        if not (0 < self.loss_factor < 1):
            raise ValueError(f"loss_factor must be in (0, 1), got {self.loss_factor}")


@dataclass(frozen=True)
class BeamSubsystem:
    """Beam subsystem (stiffener, frame, girder).

    Attributes:
        name: subsystem identifier
        length: beam length (m)
        cross_section_area: cross-sectional area (m²)
        second_moment: second moment of area about bending axis (m⁴)
        young_modulus: Young's modulus (Pa)
        density: material density (kg/m³)
        loss_factor: structural damping loss factor
    """
    name: str
    length: float
    cross_section_area: float
    second_moment: float
    young_modulus: float
    density: float
    loss_factor: float = 0.01

    @property
    def mass_per_length(self) -> float:
        return self.density * self.cross_section_area

    def __post_init__(self) -> None:
        for attr in ("length", "cross_section_area", "second_moment",
                      "young_modulus", "density"):
            val = getattr(self, attr)
            if not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(f"{attr} must be positive, got {val}")


@dataclass(frozen=True)
class CavitySubsystem:
    """Acoustic cavity subsystem (enclosed fluid volume).

    Attributes:
        name: subsystem identifier
        volume: cavity volume (m³)
        sound_speed: speed of sound in fluid (m/s)
        density: fluid density (kg/m³)
        loss_factor: acoustic damping loss factor
    """
    name: str
    volume: float
    sound_speed: float
    density: float
    loss_factor: float = 0.001

    def __post_init__(self) -> None:
        for attr in ("volume", "sound_speed", "density"):
            val = getattr(self, attr)
            if not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(f"{attr} must be positive, got {val}")


Subsystem = PlateSubsystem | BeamSubsystem | CavitySubsystem
