"""Fragment kinetic energy computation.

After fracture, the ice fragments gain kinetic energy from the release of
elastic strain energy.  The kinetic energy of each fragment is:

    E_k_i = ½ m_i v_i²

where m_i is the fragment mass and v_i is its velocity.

The total kinetic energy is:

    E_k_total = Σ_i ½ m_i v_i²

For energy conservation, the kinetic energy should be bounded by the
available energy from the fracture process:

    E_k_total ≤ U_bend - G_f - E_dissipated

References
----------
* Lubbad, R. & Løset, S. (2011). "A numerical model for real-time
  simulation of ship-ice interaction", Cold Regions Science and Technology,
  65(2), 111-127.
* Tuhkuri, J. & Lensu, M. (2009). "Full-scale observations of
  ice-ship interaction", Cold Regions Science and Technology, 57(1-2), 1-15.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class IceFragment:
    """A single ice fragment after fracture."""
    fragment_id: int
    mass: float           # mass [kg]
    velocity: np.ndarray  # velocity vector [m/s], shape (3,)
    position: np.ndarray  # position [m], shape (3,)

    @property
    def kinetic_energy(self) -> float:
        """Kinetic energy ½ m v² [J]."""
        v_sq = np.dot(self.velocity, self.velocity)
        return 0.5 * self.mass * v_sq


@dataclass
class FragmentKineticEnergy:
    """Fragment kinetic energy computation result."""
    n_fragments: int
    total_mass: float          # total fragment mass [kg]
    E_k_total: float           # total kinetic energy [J]
    E_k_max_possible: float    # maximum possible kinetic energy [J]
    velocity_rms: float        # RMS velocity [m/s]
    c5_pass: bool              # negative control pass flag

    def as_dict(self) -> dict:
        return {
            "n_fragments": self.n_fragments,
            "total_mass_kg": self.total_mass,
            "E_k_total_J": self.E_k_total,
            "E_k_max_possible_J": self.E_k_max_possible,
            "velocity_rms_m_s": self.velocity_rms,
            "c5_pass": self.c5_pass,
        }


def compute_fragment_kinetic_energy(
    fragments: List[IceFragment],
    E_available: float,
) -> FragmentKineticEnergy:
    """Compute the total kinetic energy of ice fragments.

    Parameters
    ----------
    fragments : list of IceFragment
        The ice fragments after fracture.
    E_available : float
        Available energy for kinetic energy [J].
        This is U_bend - G_f - E_dissipated.

    Returns
    -------
    FragmentKineticEnergy
    """
    n_fragments = len(fragments)
    total_mass = sum(f.mass for f in fragments)
    E_k_total = sum(f.kinetic_energy for f in fragments)

    # RMS velocity
    if n_fragments > 0 and total_mass > 0:
        v_sq_avg = sum(f.mass * np.dot(f.velocity, f.velocity) for f in fragments) / total_mass
        velocity_rms = math.sqrt(v_sq_avg)
    else:
        velocity_rms = 0.0

    # Maximum possible kinetic energy (all available energy goes to kinetic)
    E_k_max_possible = E_available

    # Negative control: if no fracture, kinetic energy should be near zero
    c5_pass = E_k_total <= 0.01 * E_available if E_available > 0 else True

    return FragmentKineticEnergy(
        n_fragments=n_fragments,
        total_mass=total_mass,
        E_k_total=E_k_total,
        E_k_max_possible=E_k_max_possible,
        velocity_rms=velocity_rms,
        c5_pass=c5_pass,
    )


def create_fragments_from_fracture(
    fracture_area: float,
    h: float,
    rho_ice: float = 917.0,
    n_fragments: int = 8,
    velocity_scale: float = 0.5,
) -> List[IceFragment]:
    """Create ice fragments from a fracture surface.

    Parameters
    ----------
    fracture_area : float
        Total fracture area [m²].
    h : float
        Plate thickness [m].
    rho_ice : float
        Ice density [kg/m³].
    n_fragments : int
        Number of fragments.
    velocity_scale : float
        Characteristic velocity scale [m/s].

    Returns
    -------
    list of IceFragment
    """
    # Total ice volume in the failure region
    # Assume the failure region is roughly circular with area proportional to fracture_area
    failure_radius = math.sqrt(fracture_area / math.pi)
    failure_volume = math.pi * failure_radius ** 2 * h
    total_mass = rho_ice * failure_volume
    fragment_mass = total_mass / n_fragments

    fragments = []
    for i in range(n_fragments):
        # Random velocity direction with characteristic magnitude
        angle = 2.0 * math.pi * i / n_fragments
        vx = velocity_scale * math.cos(angle)
        vy = velocity_scale * math.sin(angle)
        vz = velocity_scale * 0.1  # small vertical component

        # Position on a circle
        px = failure_radius * 0.5 * math.cos(angle)
        py = failure_radius * 0.5 * math.sin(angle)
        pz = h / 2.0

        fragment = IceFragment(
            fragment_id=i,
            mass=fragment_mass,
            velocity=np.array([vx, vy, vz]),
            position=np.array([px, py, pz]),
        )
        fragments.append(fragment)

    return fragments
