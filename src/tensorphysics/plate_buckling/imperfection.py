"""Initial imperfections for hull plating (welding distortions + residual stress).

The initial deflection field follows the classical plate-imperfection model

.. math:: w_0(x,y) = w_{0,\\max}\\,\\sin(m\\pi x/a)\\,\\sin(n\\pi y/b),

with the mode orders ``m``, ``n`` taken either from the plate critical mode
(``m ~ a/b``, ``n = 1``) or from an explicitly requested buckle pattern.  The
amplitude levels are the three imperfection categories used for plating in the
Paik/ALPS and IACS practice:

===========  =================
level        ``w0max``
===========  =================
``slight``   ``0.025 beta^2 t``
``average``  ``0.1   beta^2 t``
``severe``   ``0.3   beta^2 t``
===========  =================

The longitudinal welding residual stress is modelled as a self-equilibrated
two-strip pattern: tension :math:`\\sigma_r` over a strip of width
``edge_fraction * b`` at both unloaded edges, balanced by uniform compression in
the plate middle, i.e. ``int_0^b sigma_x dy = 0`` (checked in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .core import PlateGeometry

__all__ = [
    "ImperfectionSpec",
    "critical_mode_orders",
    "deflection_amplitude",
    "apply_imperfection",
    "residual_stress_field",
]

#: imperfection amplitude factors ``w0max / (beta^2 t)``
AMPLITUDE_LEVELS = {"slight": 0.025, "average": 0.1, "severe": 0.3}


def critical_mode_orders(geom: PlateGeometry) -> Dict[str, int]:
    """Buckle half-wave numbers that minimise the elastic critical stress.

    For a plate simply supported on all edges the elastic critical stress is
    minimised by ``m = round(a/b)`` half waves in the loaded direction and a
    single half wave across the breadth.
    """
    m = max(1, int(round(geom.aspect)))
    return {"m": m, "n": 1}


def deflection_amplitude(geom: PlateGeometry, level: str = "average") -> float:
    """Initial deflection amplitude ``w0max`` in metres for a standard level."""
    if level not in AMPLITUDE_LEVELS:
        raise ValueError(f"level must be one of {sorted(AMPLITUDE_LEVELS)}")
    return AMPLITUDE_LEVELS[level] * geom.beta**2 * geom.t


@dataclass
class ImperfectionSpec:
    """Initial geometry / residual stress specification of a plate panel."""

    w0max: float
    m: int = 1
    n: int = 1
    sigma_r: float = 0.0            # residual stress amplitude (Pa)
    edge_fraction: float = 0.2      # width fraction of each tension strip
    level: str = "custom"

    @classmethod
    def standard(cls, geom: PlateGeometry, level: str = "average",
                 sigma_r_ratio: float = 0.1, mode: Optional[Dict[str, int]] = None
                 ) -> "ImperfectionSpec":
        """Standard imperfection set: critical mode shape + residual stress."""
        orders = critical_mode_orders(geom) if mode is None else dict(mode)
        return cls(
            w0max=deflection_amplitude(geom, level),
            m=orders["m"],
            n=orders["n"],
            sigma_r=sigma_r_ratio * geom.sigma_y,
            level=level,
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "w0max": float(self.w0max),
            "m": int(self.m),
            "n": int(self.n),
            "sigma_r": float(self.sigma_r),
            "edge_fraction": float(self.edge_fraction),
            "level": self.level,
        }


def apply_imperfection(model, spec: ImperfectionSpec) -> None:
    """Apply a :class:`ImperfectionSpec` to a :class:`PlateModel` in place."""
    model.apply_initial_deflection(w0=spec.w0max, m=spec.m, n=spec.n)
    if spec.sigma_r:
        model.set_residual_stress(spec.sigma_r, edge_fraction=spec.edge_fraction)


def residual_stress_field(sigma_r: float, b: float, t: float,
                          edge_fraction: float = 0.2, n_samples: int = 2001
                          ) -> Dict[str, np.ndarray]:
    """Discrete ``sigma_x(y)`` pattern over the breadth (for checks/plots)."""
    y = np.linspace(0.0, b, n_samples)
    sigma = np.empty_like(y)
    inside = (y <= edge_fraction * b) | (y >= (1.0 - edge_fraction) * b)
    sigma[inside] = sigma_r
    sigma[~inside] = -sigma_r * (2.0 * edge_fraction) / (1.0 - 2.0 * edge_fraction)
    axial_force = np.trapezoid(sigma, y) * t
    return {"y": y, "sigma_x": sigma, "axial_force_per_width": np.array([axial_force])}