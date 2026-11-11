"""DEM bond fracture energy computation.

When the FEM plate fractures, the fracture surface is transferred to the DEM
solver which creates bonds between fragments.  The bond fracture energy is:

    E_bond = G_f × A_fragment

where G_f is the fracture energy per unit area and A_fragment is the area
of the fragment surface that was bonded.

For a DEM model with discrete bonds, the total bond energy is the sum over
all broken bonds:

    E_bond_total = Σ_i G_f × A_bond_i

where A_bond_i is the representative area associated with bond i.

References
----------
* Potyondy, D.O. & Cundall, P.A. (2004). "A bonded-particle model for rock",
  International Journal of Rock Mechanics and Mining Sciences, 41(8), 1329-1364.
* Scholtès, L. & Donzé, F.V. (2012). "A DEM model for soft and hard rocks",
  International Journal of Rock Mechanics and Mining Sciences, 60, 1-12.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DEMBond:
    """A single DEM bond between two fragments."""
    bond_id: int
    fragment_i: int
    fragment_j: int
    bond_area: float       # representative area [m²]
    bond_strength: float   # bond strength [Pa]
    broken: bool = False

    @property
    def bond_energy(self) -> float:
        """Energy to break this bond [J]."""
        return self.bond_strength * self.bond_area if self.broken else 0.0


@dataclass
class DEMBondEnergy:
    """DEM bond fracture energy computation result."""
    n_bonds_total: int
    n_bonds_broken: int
    G_f: float                # fracture energy per unit area [J/m²]
    total_bond_area: float    # total area of broken bonds [m²]
    E_bond_total: float       # total bond fracture energy [J]
    E_bond_expected: float    # expected energy = G_f × A_fracture [J]
    bond_energy_error_rel: float  # |E_bond - E_expected| / E_expected
    c3_pass: bool

    def as_dict(self) -> dict:
        return {
            "n_bonds_total": self.n_bonds_total,
            "n_bonds_broken": self.n_bonds_broken,
            "G_f_J_m2": self.G_f,
            "total_bond_area_m2": self.total_bond_area,
            "E_bond_total_J": self.E_bond_total,
            "E_bond_expected_J": self.E_bond_expected,
            "bond_energy_error_rel": self.bond_energy_error_rel,
            "c3_pass": self.c3_pass,
        }


def compute_dem_bond_energy(
    bonds: List[DEMBond],
    G_f: float,
    fracture_area: float,
) -> DEMBondEnergy:
    """Compute the DEM bond fracture energy.

    Parameters
    ----------
    bonds : list of DEMBond
        The DEM bonds (some may be broken).
    G_f : float
        Fracture energy per unit area [J/m²].
    fracture_area : float
        Total fracture area from FEM [m²].

    Returns
    -------
    DEMBondEnergy
    """
    n_total = len(bonds)
    broken_bonds = [b for b in bonds if b.broken]
    n_broken = len(broken_bonds)

    total_bond_area = sum(b.bond_area for b in broken_bonds)
    E_bond_total = sum(b.bond_energy for b in broken_bonds)

    # Expected energy based on G_f and fracture area
    E_bond_expected = G_f * fracture_area

    # Error: compare actual bond energy vs expected
    if E_bond_expected > 0:
        error_rel = abs(E_bond_total - E_bond_expected) / E_bond_expected
    else:
        error_rel = 0.0 if E_bond_total == 0 else float("inf")

    # C3 pass: bond energy error ≤ 1%
    c3_pass = error_rel <= 0.01

    return DEMBondEnergy(
        n_bonds_total=n_total,
        n_bonds_broken=n_broken,
        G_f=G_f,
        total_bond_area=total_bond_area,
        E_bond_total=E_bond_total,
        E_bond_expected=E_bond_expected,
        bond_energy_error_rel=error_rel,
        c3_pass=c3_pass,
    )


def create_dem_bonds_from_fracture(
    fracture_area: float,
    G_f: float,
    n_bonds: int = 100,
    bond_area_fraction: float = 0.95,
) -> List[DEMBond]:
    """Create DEM bonds from a fracture surface.

    This simulates the transfer of the FEM fracture surface to DEM bonds.
    The total bond area should match the fracture area (within tolerance).

    Parameters
    ----------
    fracture_area : float
        Total fracture area from FEM [m²].
    G_f : float
        Fracture energy per unit area [J/m²].
    n_bonds : int
        Number of DEM bonds to create.
    bond_area_fraction : float
        Fraction of fracture area covered by bonds (should be close to 1).

    Returns
    -------
    list of DEMBond
    """
    total_bond_area = fracture_area * bond_area_fraction
    bond_area = total_bond_area / n_bonds
    bond_strength = G_f  # bond strength = fracture energy per unit area

    bonds = []
    for i in range(n_bonds):
        bond = DEMBond(
            bond_id=i,
            fragment_i=i // 2,
            fragment_j=i // 2 + 1,
            bond_area=bond_area,
            bond_strength=bond_strength,
            broken=True,  # all bonds broken in this scenario
        )
        bonds.append(bond)

    return bonds
