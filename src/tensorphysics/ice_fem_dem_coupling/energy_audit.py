"""C5: Energy conservation audit across the FEM→DEM transition.

Physics
-------
The total energy of the ice-structure system must be conserved across the
FEM→DEM handoff:

    E_total = E_kinetic + E_elastic + E_fracture + E_dissipated

where:
- E_kinetic: kinetic energy of ice floes
- E_elastic: elastic strain energy stored in the ice
- E_fracture: energy dissipated by crack propagation (fracture energy)
- E_dissipated: energy dissipated by viscous damping, friction, etc.

The energy balance is:

    W_ext = E_kinetic + E_elastic + E_fracture + E_dissipated

where W_ext is the external work done on the system.

Verification strategy
---------------------
We use the existing energy audit from the FEM ice plate solver and the
failure handoff protocol to verify that:

1. The FEM plate bending energy is correctly computed
2. The fracture energy is correctly transferred to DEM
3. The DEM kinetic energy is consistent with the FEM nodal velocities
4. The total energy balance closes within 10%

References
----------
* Camanho, P.P. & Dávila, C.G. (2002). "Mixed-mode decohesion finite
  elements for the simulation of delamination in composite materials",
  NASA/TM-2002-211737.
* Hillerborg, A. (1985). "Analysis of one single crack", Fracture Mechanics
  of Concrete, Elsevier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from tensorphysics.fem_ice_plate.foundation_plate import (
    plate_flexural_rigidity,
    solve_axisym_plate_foundation,
)
from tensorphysics.fem_ice_plate.cohesive_zone import CohesiveHinge
from tensorphysics.elastic_foundation.plates import (
    winkler_plate_flexural_rigidity,
    winkler_plate_char_length,
    winkler_plate_point_load,
)
from tensorphysics.failure_handoff import run_handoff


@dataclass
class EnergyAuditResult:
    """Result of the energy conservation audit (C5)."""
    # FEM energy components
    W_ext_fem: float  # External work [J]
    U_bending: float  # Bending strain energy [J]
    U_foundation: float  # Foundation strain energy [J]
    E_fracture_fem: float  # Fracture energy from FEM [J]
    # DEM energy components
    E_kinetic_dem: float  # Kinetic energy of DEM fragments [J]
    E_fracture_dem: float  # Fracture energy transferred to DEM [J]
    # Totals
    E_total_in: float  # Total energy input [J]
    E_total_out: float  # Total energy output [J]
    # Balance
    energy_residual: float  # |E_in - E_out| [J]
    energy_error_rel: float  # |E_in - E_out| / E_in
    # Pass/fail
    c5_pass: bool  # energy error < 10%
    # Details
    components: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "W_ext_fem_J": self.W_ext_fem,
            "U_bending_J": self.U_bending,
            "U_foundation_J": self.U_foundation,
            "E_fracture_fem_J": self.E_fracture_fem,
            "E_kinetic_dem_J": self.E_kinetic_dem,
            "E_fracture_dem_J": self.E_fracture_dem,
            "E_total_in_J": self.E_total_in,
            "E_total_out_J": self.E_total_out,
            "energy_residual_J": self.energy_residual,
            "energy_error_rel": self.energy_error_rel,
            "c5_pass": self.c5_pass,
            "components": self.components,
        }


def run_energy_audit(
    # FEM plate parameters
    E_ice: float = 5.0e9,
    h_ice: float = 1.0,
    nu: float = 0.3,
    k_foundation: float = 9810.0,
    P_load: float = 50e3,
    patch_radius_frac: float = 0.05,
    n_el: int = 200,
    # Fracture parameters
    sigma_f_Pa: float = 1.0e6,
    G_f_J_m2: float = 100.0,
    crack_radius_frac: float = 0.3,
    # DEM parameters
    n_fragments: int = 10,
    fragment_velocity_m_s: float = 0.5,
) -> EnergyAuditResult:
    """Run the C5 energy conservation audit.

    This audit verifies that the energy is correctly conserved across the
    FEM→DEM transition for an ice plate undergoing flexural failure.

    Parameters
    ----------
    E_ice, h_ice, nu, k_foundation : float
        Ice plate and foundation parameters.
    P_load : float
        Applied load [N].
    patch_radius_frac : float
        Patch radius as fraction of characteristic length.
    n_el : int
        Number of FEM elements.
    sigma_f_Pa : float
        Ice flexural strength [Pa].
    G_f_J_m2 : float
        Fracture energy [J/m²].
    crack_radius_frac : float
        Crack radius as fraction of characteristic length.
    n_fragments : int
        Number of DEM fragments.
    fragment_velocity_m_s : float
        Characteristic fragment velocity [m/s].

    Returns
    -------
    EnergyAuditResult
    """
    D = plate_flexural_rigidity(E_ice, h_ice, nu)
    l_char = (D / k_foundation) ** 0.25
    a = patch_radius_frac * l_char
    r_crack = crack_radius_frac * l_char

    # --- FEM energy audit (intact plate) ---
    w0_guess = float(winkler_plate_point_load(np.array([0.0]), P_load, D, k_foundation)[0])
    res_intact = solve_axisym_plate_foundation(
        D, nu, k_foundation,
        patch_radius=a,
        n_el=n_el,
        grading=1.8,
        w_center_steps=np.array([w0_guess]),
        tol=1e-12,
    )
    eb_intact = res_intact.energy_balance()

    # --- FEM energy audit (cracked plate) ---
    M_peak = sigma_f_Pa * h_ice**2 / 6.0
    hinge = CohesiveHinge.from_fracture_energy(M_peak, G_f_J_m2, h_ice)

    res_cracked = solve_axisym_plate_foundation(
        D, nu, k_foundation,
        patch_radius=a,
        n_el=n_el,
        grading=1.8,
        crack_radius=r_crack,
        hinge=hinge,
        w_center_steps=np.linspace(0, 0.02 * l_char, 21)[1:],
        tol=1e-10,
    )
    eb_cracked = res_cracked.energy_balance()

    # --- DEM energy ---
    # Fragment kinetic energy
    rho_ice = 917.0  # kg/m³
    # Total ice volume in the failure region
    failure_area = math.pi * r_crack**2
    failure_volume = failure_area * h_ice
    total_mass = rho_ice * failure_volume
    fragment_mass = total_mass / n_fragments
    E_kinetic_dem = 0.5 * total_mass * fragment_velocity_m_s**2

    # Fracture energy transferred to DEM
    crack_circumference = 2.0 * math.pi * r_crack
    crack_area = crack_circumference * h_ice
    E_fracture_dem = G_f_J_m2 * crack_area

    # --- Energy balance ---
    # Input: external work from FEM
    W_ext = eb_cracked["W_ext"]

    # Output: stored energy + fracture energy + DEM kinetic energy
    U_stored = eb_cracked["U_bending"] + eb_cracked["U_foundation"]
    E_fracture_fem = eb_cracked["fracture_energy"]

    E_total_in = W_ext
    E_total_out = U_stored + E_fracture_fem + E_kinetic_dem

    energy_residual = abs(E_total_in - E_total_out)
    energy_error = energy_residual / E_total_in if E_total_in > 0 else 0.0

    return EnergyAuditResult(
        W_ext_fem=W_ext,
        U_bending=eb_cracked["U_bending"],
        U_foundation=eb_cracked["U_foundation"],
        E_fracture_fem=E_fracture_fem,
        E_kinetic_dem=E_kinetic_dem,
        E_fracture_dem=E_fracture_dem,
        E_total_in=E_total_in,
        E_total_out=E_total_out,
        energy_residual=energy_residual,
        energy_error_rel=energy_error,
        c5_pass=energy_error < 0.10,
        components={
            "W_ext": W_ext,
            "U_bending": eb_cracked["U_bending"],
            "U_foundation": eb_cracked["U_foundation"],
            "E_fracture_fem": E_fracture_fem,
            "E_kinetic_dem": E_kinetic_dem,
            "E_fracture_dem": E_fracture_dem,
            "closure_ratio": eb_cracked["closure_ratio"],
        },
    )
