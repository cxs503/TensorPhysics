"""Flexible model-scale barge: beam modes, modal generalised force, whipping stress.

The beam is the *downstream* stage of the chain.  It reuses the repository's
existing hull-girder machinery instead of duplicating a beam solver:

* ``hull_hydroelastic.timoshenko``  - free-free Timoshenko dry modes (REAL FE),
* ``hull_hydroelastic.wet``         - weighted (wet) modes in the dry basis,
* ``hull_hydroelastic.sections``    - 2-D sectional added mass from a Dirichlet
  boundary-value problem (validated in-repo against ``rho pi a^2``).

What is *new* here is only the chain adapter: the slamming pressure field is
projected onto the (wet) mode shapes, the Duhamel convolution is driven by that
generalised force, and the midship whipping stress is assembled.

Evidence levels
---------------
REAL              dry/wet modes, modal superposition, convolution, stress from
                  curvature.
MODEL-INPUT       ``EI`` of the flexible barge model and the modal damping ratio
                  (structural welding damping order 2 %) are inputs.
APPROXIMATE-STRIP the sectional added mass is 2-D; three-dimensional end effects
                  of the finite-length barge are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hull_hydroelastic.sections import heave_added_mass, superellipse_section
from ..hull_hydroelastic.timoshenko import BeamProperties, TimoshenkoBeam
from ..hull_hydroelastic.wet import wet_modes as solve_wet_modes

__all__ = ["FlexibleBarge", "ModalBeam"]


@dataclass(frozen=True)
class FlexibleBarge:
    """Uniform flexible barge model (all SI)."""

    length: float = 10.0
    breadth: float = 1.0
    draught: float = 0.15
    depth: float = 0.6
    mass: float = 1537.5          # ballasted to the design draught (rho L B d)
    ei: float = 8.858e7           # steel box girder, 2 mm plating -> I = 4.3e-4 m^4
    young: float = 2.06e11
    rho: float = 1025.0
    zeta_structural: float = 0.02
    n_elements: int = 160
    section_exponent: float = 8.0

    @property
    def mass_per_length(self) -> float:
        return float(self.mass / self.length)

    @property
    def second_moment(self) -> float:
        """Second moment of area implied by ``EI`` and the material modulus."""
        return float(self.ei / self.young)

    @property
    def section_modulus(self) -> float:
        """Elastic section modulus ``I / (D/2)`` [m^3]."""
        return float(self.second_moment / (0.5 * self.depth))

    def beam_properties(self) -> BeamProperties:
        """Uniform Timoshenko beam properties on ``n_elements + 1`` nodes."""
        x = np.linspace(-0.5 * self.length, 0.5 * self.length, int(self.n_elements) + 1)
        n = x.size
        m = np.full(n, self.mass_per_length)
        ei = np.full(n, self.ei)
        # shear stiffness: G As with As = (5/6) * web area of the box girder
        shear_modulus = self.young / (2.0 * (1.0 + 0.3))
        area = 2.0 * self.breadth * 0.002 + 2.0 * self.depth * 0.002   # 2 mm plating
        gas = np.full(n, shear_modulus * (5.0 / 6.0) * area)
        ri = np.full(n, self.mass_per_length * self.depth ** 2 / 12.0)
        return BeamProperties(x=x, EI=ei, GAs=gas, m=m, rI=ri)

    def dry_modes(self, n_modes: int = 6) -> dict:
        """Free-free dry bending modes (Timoshenko FE)."""
        beam = TimoshenkoBeam(self.beam_properties())
        dry = beam.dry_modes(int(n_modes))
        return {
            "x": np.asarray(dry.x),
            "shapes": np.asarray(dry.shapes),
            "omega": np.asarray(dry.omega),
            "frequencies_hz": np.asarray(dry.frequencies_hz),
            "curvature": np.asarray(dry.curvature),
            "beam": beam,
        }

    def sectional_added_mass(self, n_panels: int = 160) -> float:
        """2-D heave added mass per unit length [kg/m] from the sectional BVP."""
        section = superellipse_section(
            half_beam=0.5 * self.breadth,
            draught=self.draught,
            exponent=self.section_exponent,
        )
        return float(heave_added_mass(section, rho=self.rho, n_panels=n_panels))

    def wet_beam(self, n_modes: int = 6, n_panels: int = 160) -> "ModalBeam":
        """Wet (added-mass weighted) modes of the barge in the dry modal basis."""
        dry = self.dry_modes(n_modes)
        x = dry["x"]
        a33 = np.full(x.size, self.sectional_added_mass(n_panels))
        b_wp = np.full(x.size, self.breadth)
        wet = solve_wet_modes(
            dry["beam"],
            a33,
            n_modes=int(n_modes),
            zeta_structural=self.zeta_structural,
            b_waterplane=b_wp,
            rho=self.rho,
        )
        return ModalBeam(
            x=np.asarray(wet.x),
            shapes=np.asarray(wet.wet_shapes),
            omega=np.asarray(wet.wet_omega),
            zeta=np.asarray(wet.zeta),
            curvature=np.asarray(wet.wet_curvature),
            dry_omega=np.asarray(wet.dry_omega),
            added_mass_ratio=np.asarray(wet.added_mass_ratio),
            barge=self,
        )


@dataclass
class ModalBeam:
    """Mass-normalised wet modes plus the chain-level modal post-processing."""

    x: np.ndarray
    shapes: np.ndarray        # (n_modes, n_x)
    omega: np.ndarray         # (n_modes,) wet angular frequencies
    zeta: np.ndarray          # (n_modes,)
    curvature: np.ndarray     # (n_modes, n_x)
    dry_omega: np.ndarray
    added_mass_ratio: np.ndarray
    barge: FlexibleBarge

    @property
    def n_modes(self) -> int:
        return int(self.omega.size)

    @property
    def midship_index(self) -> int:
        return int(np.argmin(np.abs(self.x)))

    @property
    def stress_shape(self) -> np.ndarray:
        """``s_n = E (D/2) phi_n''(x_mid)`` [Pa per unit modal displacement]."""
        i = self.midship_index
        return self.barge.young * (0.5 * self.barge.depth) * self.curvature[:, i]

    def project_line_load(self, line_load: np.ndarray) -> np.ndarray:
        """``Q_n = int f(x) phi_n(x) dx`` for a sampled line load."""
        f = np.asarray(line_load, dtype=float)
        if f.size != self.x.size:
            raise ValueError("line load must be sampled on the beam grid")
        return np.trapezoid(self.shapes * f[None, :], self.x, axis=1)

    def stress_history(self, q: np.ndarray) -> np.ndarray:
        """Midship bending stress ``sigma(t) = sum_n s_n q_n(t)`` [Pa]."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        return self.stress_shape @ q

    def summary(self) -> dict:
        return {
            "n_modes": self.n_modes,
            "dry_hz": (self.dry_omega / (2.0 * np.pi)).tolist(),
            "wet_hz": (self.omega / (2.0 * np.pi)).tolist(),
            "zeta": self.zeta.tolist(),
            "added_mass_ratio": self.added_mass_ratio.tolist(),
            "stress_shape_Pa_per_m": self.stress_shape.tolist(),
            "midship_x_m": float(self.x[self.midship_index]),
            "section_modulus_m3": self.barge.section_modulus,
            "second_moment_m4": self.barge.second_moment,
        }