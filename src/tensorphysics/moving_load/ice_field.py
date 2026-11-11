"""Reduced discrete floe stream that drives the hull girder moving load.

The ice load in the ship-fixed frame is a sequence of contact pulses that
sweep aft along the hull: as the ship advances at ``V``, floe ``j`` touches the
bow at ``t_j``, is crushed while it travels the contact zone over
``D_j / V`` seconds, and leaves.  In the hull-girder equation this is exactly a
*moving* load ``F(x, t)`` with ``x(t) = x_entry - V (t - t_j)``.

Modelling level (declared, not implied): this is a **discrete floe stream**
reduced model of a DEM ice field.  Each floe is a discrete entity with its own
diameter, gap and crushing force drawn from a seeded population, and the
contact force history is fully time-resolved -- but floe-floe collisions,
floe rigid-body rotation and the 3-D geometry of the bow are *not* resolved.
For hull-girder excitation the missing physics is second order (the force
magnitude and its sweep speed dominate), and the limitation is stated in the
report.

Crushing force: pressure-area law (Korzhavin/Riska family)

    p(A) = p0 (A / A0)^e,   A = D h,   F = k1 k2 p(A) A

with ``p0 = 1.0 MPa`` at ``A0 = 1 m^2`` and the customary negative exponent
``e = -0.10``; ``k1`` is the indentation factor and ``k2`` the contact-shape
factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .beam import HullGirderBeam
from .paths import LoadPath

__all__ = [
    "DEFAULT_PRESSURE_AREA_P0_PA",
    "DEFAULT_PRESSURE_AREA_A0_M2",
    "DEFAULT_PRESSURE_AREA_EXPONENT",
    "IceFloeField",
    "sample_floe_field",
    "reference_floe_field",
    "korzhavin_contact_force",
    "floe_stream_paths",
    "encounter_frequency_hz",
]

#: Reference crushing pressure at the reference contact area (Pa).
DEFAULT_PRESSURE_AREA_P0_PA = 1.0e6
#: Reference contact area for the pressure-area law (m^2).
DEFAULT_PRESSURE_AREA_A0_M2 = 1.0
#: Pressure-area exponent (negative: pressure decreases with area).
DEFAULT_PRESSURE_AREA_EXPONENT = -0.10


@dataclass(frozen=True)
class IceFloeField:
    """Sampled floe population travelling with the ship.

    Attributes
    ----------
    diameters_m
        Floe diameter of each floe (m).
    gaps_m
        Clear gap between consecutive floes (m); ``gaps_m[0]`` is the lead-in
        distance before the first floe.
    thickness_m
        Level-ice thickness (m), taken uniform over the population.
    speed_m_s
        Ship speed relative to the ice (m/s).
    """

    diameters_m: np.ndarray
    gaps_m: np.ndarray
    thickness_m: float
    speed_m_s: float
    seed: int = 0
    indentation_factor: float = 1.0
    shape_factor: float = 0.90

    def __post_init__(self) -> None:
        d = np.asarray(self.diameters_m, dtype=float)
        g = np.asarray(self.gaps_m, dtype=float)
        if d.ndim != 1 or d.size < 1:
            raise ValueError("diameters_m must be a non-empty 1-D array")
        if g.size != d.size:
            raise ValueError("gaps_m must have the same length as diameters_m")
        if np.any(d <= 0.0) or np.any(g < 0.0):
            raise ValueError("diameters must be positive and gaps non-negative")
        if self.thickness_m <= 0.0:
            raise ValueError("thickness_m must be positive")
        if self.speed_m_s <= 0.0:
            raise ValueError("speed_m_s must be positive")

    @property
    def n_floes(self) -> int:
        return int(np.asarray(self.diameters_m).size)

    @property
    def mean_spacing_m(self) -> float:
        """Mean centre-to-centre spacing (m): floe radius sum plus clear gap."""
        d = np.asarray(self.diameters_m, dtype=float)
        if self.n_floes < 2:
            return float(d[0])
        g = np.asarray(self.gaps_m, dtype=float)[1:]
        centres = 0.5 * (d[:-1] + d[1:]) + g
        return float(np.mean(centres))

    @property
    def mean_diameter_m(self) -> float:
        return float(np.mean(self.diameters_m))

    @property
    def track_length_m(self) -> float:
        return float(np.sum(self.diameters_m) + np.sum(self.gaps_m))

    def as_dict(self) -> dict:
        return {
            "n_floes": self.n_floes,
            "thickness_m": float(self.thickness_m),
            "speed_m_s": float(self.speed_m_s),
            "seed": int(self.seed),
            "mean_diameter_m": self.mean_diameter_m,
            "mean_spacing_m": self.mean_spacing_m,
            "track_length_m": self.track_length_m,
            "diameters_m": [float(v) for v in np.asarray(self.diameters_m)],
            "gaps_m": [float(v) for v in np.asarray(self.gaps_m)],
            "indentation_factor": float(self.indentation_factor),
            "shape_factor": float(self.shape_factor),
        }


def sample_floe_field(
    n_floes: int = 14,
    speed_m_s: float = 5.0,
    thickness_m: float = 1.0,
    mean_diameter_m: float = 10.0,
    diameter_cov: float = 0.35,
    mean_gap_m: float = 6.0,
    lead_in_m: float = 6.0,
    seed: int = 20261111,
    indentation_factor: float = 1.0,
    shape_factor: float = 0.90,
) -> IceFloeField:
    """Draw a reproducible floe population (lognormal diameters, exponential gaps)."""
    if n_floes < 1:
        raise ValueError("n_floes must be >= 1")
    rng = np.random.default_rng(int(seed))
    mu = math.log(mean_diameter_m) - 0.5 * math.log(1.0 + diameter_cov ** 2)
    sigma = math.sqrt(math.log(1.0 + diameter_cov ** 2))
    diameters = np.exp(rng.normal(mu, sigma, size=int(n_floes)))
    gaps = rng.exponential(mean_gap_m, size=int(n_floes))
    gaps[0] = float(lead_in_m)
    return IceFloeField(
        diameters_m=diameters,
        gaps_m=gaps,
        thickness_m=float(thickness_m),
        speed_m_s=float(speed_m_s),
        seed=int(seed),
        indentation_factor=float(indentation_factor),
        shape_factor=float(shape_factor),
    )


def reference_floe_field(speed_m_s: float = 5.0, thickness_m: float = 1.0) -> IceFloeField:
    """The reference ice field of the acceptance study (deterministic seed)."""
    return sample_floe_field(speed_m_s=speed_m_s, thickness_m=thickness_m)


def korzhavin_contact_force(
    diameter_m: float | np.ndarray,
    thickness_m: float,
    p0_pa: float = DEFAULT_PRESSURE_AREA_P0_PA,
    a0_m2: float = DEFAULT_PRESSURE_AREA_A0_M2,
    exponent: float = DEFAULT_PRESSURE_AREA_EXPONENT,
    indentation_factor: float = 1.0,
    shape_factor: float = 0.90,
) -> np.ndarray | float:
    """Contact force of a floe of diameter ``D`` and thickness ``h`` (N).

    ``F = k1 k2 p0 (A/A0)^e A`` with ``A = D h``.  Scalar in, float out.
    """
    d = np.asarray(diameter_m, dtype=float)
    area = np.maximum(d * float(thickness_m), 1e-12)
    pressure = p0_pa * (area / a0_m2) ** exponent
    force = float(indentation_factor) * float(shape_factor) * pressure * area
    if np.isscalar(diameter_m):
        return float(force)
    return force


def encounter_frequency_hz(speed_m_s: float, mean_spacing_m: float) -> float:
    """Floe encounter frequency ``V / s`` (Hz)."""
    if mean_spacing_m <= 0.0:
        raise ValueError("mean_spacing_m must be positive")
    return float(speed_m_s) / float(mean_spacing_m)


def floe_stream_paths(
    beam: HullGirderBeam,
    field: IceFloeField,
    x_contact_start_m: float | None = None,
    pulse_shape: str = "half_sine",
) -> list:
    """Convert a floe population into moving load paths on the beam.

    The contact point starts at ``x_contact_start_m`` (default ``0.92 L``: the
    bow shoulder, i.e. inside the end support of the equivalent SS beam, where
    the mode shapes are non-zero) and sweeps aft by one floe diameter per
    contact.
    """
    if x_contact_start_m is None:
        x_contact_start_m = 0.92 * beam.length_m
    x0 = float(x_contact_start_m)
    if not 0.0 < x0 < beam.length_m:
        raise ValueError("x_contact_start_m must lie inside the span")

    forces = korzhavin_contact_force(
        field.diameters_m,
        field.thickness_m,
        indentation_factor=field.indentation_factor,
        shape_factor=field.shape_factor,
    )
    paths = []
    t = 0.0
    for d, gap, force in zip(np.asarray(field.diameters_m), np.asarray(field.gaps_m), np.asarray(forces)):
        t += float(gap) / field.speed_m_s
        duration = float(d) / field.speed_m_s
        x_exit = max(x0 - float(d), 0.05 * beam.length_m)
        paths.append(
            LoadPath(
                magnitude_n=float(force),
                x_entry_m=x0,
                x_exit_m=x_exit,
                t_entry_s=t,
                duration_s=duration,
                shape=pulse_shape,
            )
        )
        t += duration
    return paths