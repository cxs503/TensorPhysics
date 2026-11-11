"""Uniform Euler-Bernoulli beam that carries the hull-girder moving-load model.

Why a simply supported beam?
----------------------------
The moving-load problem has a small set of *closed-form* anchors, and they exist
for the simply supported (SS) beam:

* static midspan deflection ``P L^3 / 48 EI`` (exact),
* undamped modal solution for a point force traversing the span (Fryba 1972),
* the critical (resonant) velocity ``V_cr = omega_1 L / pi``.

Keeping the verification model exactly SS therefore buys three independent
analytic references at zero extra cost.  The real hull girder is closer to
free-free; the difference is declared as a modelling limitation in the report
and does not affect the acceptance criteria, which are all stated against the
SS reference problem.

Geometry is validated before any dynamics is run (block coefficient,
length/beam and beam/draught ratios, positive stiffness and mass), following
the platform rule "verify the geometry before trusting the numbers".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "STEEL_E_MODULUS_PA",
    "STEEL_DENSITY_KG_M3",
    "RHO_SEA_WATER",
    "HullGirderBeam",
    "hull_girder_beam_from_particulars",
    "reference_hull_girder",
]

#: Young's modulus of shipbuilding steel (Pa).
STEEL_E_MODULUS_PA = 2.10e11
#: Steel density (kg/m^3).
STEEL_DENSITY_KG_M3 = 7850.0
#: Sea-water density (kg/m^3).
RHO_SEA_WATER = 1025.0


@dataclass(frozen=True)
class HullGirderBeam:
    """Uniform Euler-Bernoulli beam: length, bending stiffness, line mass.

    Parameters
    ----------
    length_m
        Span ``L`` of the beam (m).
    ei_nm2
        Bending stiffness ``EI`` (N*m^2).
    rho_a_kg_m
        Mass per unit length ``rho*A`` including entrained water (kg/m).

    Notes
    -----
    Mode shapes are the mass-unnormalised SS set ``phi_n(x) = sin(n pi x / L)``
    with modal mass ``M_n = rho*A*L/2`` and modal stiffness
    ``K_n = M_n*omega_n^2``.  Shear deformation and rotary inertia are
    neglected: the beam is slender and the first ten modes sit far below the
    shear cut-off (checked in the report).
    """

    length_m: float
    ei_nm2: float
    rho_a_kg_m: float

    def __post_init__(self) -> None:
        for name in ("length_m", "ei_nm2", "rho_a_kg_m"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value!r}")

    # -- basic properties ---------------------------------------------------
    @property
    def flexural_wave_speed(self) -> float:
        """``sqrt(EI / (rho A))`` (m^2/s): the dispersion-relation scale."""
        return math.sqrt(self.ei_nm2 / self.rho_a_kg_m)

    def omega(self, n: int) -> float:
        """Natural circular frequency of mode ``n`` (rad/s), ``n`` from 1."""
        if n < 1:
            raise ValueError("mode number n must be >= 1")
        return (n * math.pi / self.length_m) ** 2 * self.flexural_wave_speed

    def frequency(self, n: int) -> float:
        """Natural frequency of mode ``n`` (Hz)."""
        return self.omega(n) / (2.0 * math.pi)

    def natural_frequencies(self, n_modes: int = 10) -> np.ndarray:
        """Vector of the first ``n_modes`` natural frequencies (Hz)."""
        return np.array([self.frequency(n) for n in range(1, int(n_modes) + 1)], dtype=float)

    def mode_shape(self, n: int, x: float) -> float:
        """``phi_n(x) = sin(n pi x / L)``."""
        return math.sin(n * math.pi * float(x) / self.length_m)

    def mode_shapes(self, n_modes: int, x: np.ndarray) -> np.ndarray:
        """Mode-shape matrix ``(n_modes, len(x))``."""
        x = np.asarray(x, dtype=float)
        n = np.arange(1, int(n_modes) + 1, dtype=float)[:, None]
        return np.sin(n * math.pi * x[None, :] / self.length_m)

    def modal_mass(self, n: int) -> float:
        """Modal mass ``M_n = rho A L / 2`` (kg) for ``sin(n pi x / L)``."""
        return 0.5 * self.rho_a_kg_m * self.length_m

    def modal_stiffness(self, n: int) -> float:
        """Modal stiffness ``K_n = M_n omega_n^2`` (N/m)."""
        return self.modal_mass(n) * self.omega(n) ** 2

    def generalized_excitation_frequency(self, n: int, speed_m_s: float) -> float:
        """Modal forcing frequency ``Omega_n = n pi V / L`` (rad/s)."""
        return n * math.pi * float(speed_m_s) / self.length_m

    def critical_velocity(self) -> float:
        """Critical (first-mode resonant) velocity ``V_cr = omega_1 L / pi``."""
        return self.omega(1) * self.length_m / math.pi

    # -- statics ------------------------------------------------------------
    def static_midspan_deflection(self, load_n: float) -> float:
        """Exact static midspan deflection of an SS beam, ``P L^3 / 48 EI``.

        Positive downwards (i.e. in the direction of the applied load).
        """
        return float(load_n) * self.length_m ** 3 / (48.0 * self.ei_nm2)

    def static_deflection_profile(self, load_n: float, x_sequence: Sequence[float]) -> np.ndarray:
        """Exact static deflection curve for a point load at midspan.

        For ``x <= L/2``: ``P x (3 L^2 - 4 x^2) / (48 EI)``.
        """
        x = np.asarray(x_sequence, dtype=float)
        p, L, ei = float(load_n), self.length_m, self.ei_nm2
        xs = np.minimum(x, L - x)
        return p * xs * (3.0 * L ** 2 - 4.0 * xs ** 2) / (48.0 * ei)

    def static_modal_solution(self, load_n: float, x_load: float, n_modes: int = 10) -> np.ndarray:
        """Modal-truncated static deflection curve (nodal loads, no time stepping)."""
        x = np.linspace(0.0, self.length_m, 201)
        acc = np.zeros_like(x)
        for n in range(1, int(n_modes) + 1):
            q = float(load_n) * self.mode_shape(n, x_load) / self.modal_stiffness(n)
            acc += q * np.sin(n * math.pi * x / self.length_m)
        return acc


def hull_girder_beam_from_particulars(
    length_m: float = 150.0,
    breadth_m: float = 22.0,
    draught_m: float = 8.0,
    block_coefficient: float = 0.70,
    midship_inertia_m4: float = 18.0,
    added_mass_ratio: float = 0.20,
    youngs_modulus_pa: float = STEEL_E_MODULUS_PA,
    water_density: float = RHO_SEA_WATER,
) -> HullGirderBeam:
    """Build the equivalent hull-girder beam from principal particulars.

    The line mass combines the displaced water mass per unit length with an
    entrained-water (added-mass) fraction, which is the standard way of
    writing the wet hull-girder frequency:

    ``rho A = rho_w * Cb * B * T * (1 + added_mass_ratio)``.

    Geometry is validated first: ``0 < Cb < 1``, ``4 <= L/B <= 12`` and
    ``2 <= B/T <= 5`` -- the envelope in which the thin-beam idealisation is
    defensible.
    """
    L, B, T, Cb = float(length_m), float(breadth_m), float(draught_m), float(block_coefficient)
    if L <= 0 or B <= 0 or T <= 0:
        raise ValueError("L, B and T must be positive")
    if not 0.0 < Cb < 1.0:
        raise ValueError(f"block coefficient must lie in (0, 1), got {Cb}")
    if not 4.0 <= L / B <= 12.0:
        raise ValueError(f"L/B = {L / B:.2f} outside the thin-beam envelope [4, 12]")
    if not 2.0 <= B / T <= 5.0:
        raise ValueError(f"B/T = {B / T:.2f} outside the hull envelope [2, 5]")
    if float(midship_inertia_m4) <= 0.0:
        raise ValueError("midship section inertia must be positive")
    if float(added_mass_ratio) < 0.0:
        raise ValueError("added mass ratio must be non-negative")

    rho_a = water_density * Cb * B * T * (1.0 + float(added_mass_ratio))
    ei = float(youngs_modulus_pa) * float(midship_inertia_m4)
    return HullGirderBeam(length_m=L, ei_nm2=ei, rho_a_kg_m=rho_a)


def reference_hull_girder() -> HullGirderBeam:
    """The reference 150 m hull girder used by the acceptance study.

    ``L = 150 m``, ``B = 22 m``, ``T = 8 m``, ``Cb = 0.70``, midship section
    inertia ``I = 18 m^4`` (steel, E = 210 GPa) and 20 % entrained water.
    """
    return hull_girder_beam_from_particulars()