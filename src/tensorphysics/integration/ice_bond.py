"""``ice-bond-v1`` receiver: TensorDEM bond data -> TensorPhysics cohesive law.

The two repositories describe ice strength at different scales.

* TensorDEM resolves **bonds**: a lattice of particles of spacing ``a`` with a
  bond stiffness ``k`` [N/m] and a breaking strain
  ``eps_b = sigma_b / E`` (``tensordem.ice_beam_bending.lattice_elastic_properties``).
  The bond carries a *force* criterion, ``k a eps_b``, not a stress.
* TensorPhysics consumes a **continuum cohesive law**
  (:func:`tensorphysics.ice.degradation.cohesive_law`): a peak traction
  ``tau_p`` [Pa] and a fracture energy ``G_c`` [J/m^2], with the failure
  separation ``delta_c = 2 G_c / tau_p``.

Under-determined interface
--------------------------
``(sigma_b, eps_b)`` alone do **not** determine ``G_c``: an energy needs a
length.  The contract therefore carries a characteristic length ``ell`` (the
lattice spacing -- one bond spans one spacing) and the mapping is

    tau_p = sigma_b,        delta_c = eps_b * ell,        G_c = 0.5 * tau_p * delta_c

The factor ``1/2`` is the area of the bilinear traction-separation triangle;
``ell`` is the *only* dimensional bridge between the repos.  Dropping it (or
assuming ``ell = 1 m``) changes ``G_c`` by ``1/ell`` -- for a 12.5 mm lattice
that is a factor of 80 -- which is why the receiver rejects payloads that do
not pin the length scale (see ``tests/test_p27w3_integration_contracts.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..ice.degradation import cohesive_law
from ..ice_sheet.kirchhoff_plate_fem import limiting_moment
from .interchange import ContractViolation, IceBondTransfer

__all__ = [
    "BondMapping",
    "cohesive_energy_error",
    "dem_bond_to_cohesive",
    "dem_failure_load",
    "tensorphysics_failure_load",
]


@dataclass(frozen=True)
class BondMapping:
    """Continuum cohesive parameters implied by a DEM lattice bond."""

    tau_peak_Pa: float
    delta_c_m: float
    G_c_J_m2: float
    characteristic_length_m: float
    bond_strength_Pa: float
    breaking_strain: float
    bond_stiffness_N_m: float

    def as_dict(self) -> dict:
        return {
            "tau_peak_Pa": self.tau_peak_Pa,
            "delta_c_m": self.delta_c_m,
            "G_c_J_m2": self.G_c_J_m2,
            "characteristic_length_m": self.characteristic_length_m,
            "bond_strength_Pa": self.bond_strength_Pa,
            "breaking_strain": self.breaking_strain,
            "bond_stiffness_N_m": self.bond_stiffness_N_m,
            "bond_force_at_break_N": self.bond_stiffness_N_m
            * self.characteristic_length_m * self.breaking_strain,
        }


def dem_bond_to_cohesive(transfer: IceBondTransfer,
                         characteristic_length_m: float | None = None) -> BondMapping:
    """Map the DEM lattice bond onto the TensorPhysics cohesive law.

    ``characteristic_length_m`` defaults to the DEM lattice spacing, which is
    the physical gauge length of one bond.  A non-positive value is rejected
    instead of silently producing a degenerate fracture energy.
    """
    ell = float(transfer.spacing_m if characteristic_length_m is None
                else characteristic_length_m)
    if not np.isfinite(ell) or ell <= 0.0:
        raise ContractViolation("ice-bond-v1: characteristic length must be positive")
    sigma_b = transfer.bond_strength_Pa
    eps_b = float(transfer.lattice["breaking_strain"])
    if sigma_b <= 0.0 or eps_b <= 0.0:                        # pragma: no cover
        raise ContractViolation("ice-bond-v1: bond strength/strain must be positive")
    delta_c = eps_b * ell
    return BondMapping(
        tau_peak_Pa=sigma_b,
        delta_c_m=delta_c,
        G_c_J_m2=0.5 * sigma_b * delta_c,
        characteristic_length_m=ell,
        bond_strength_Pa=sigma_b,
        breaking_strain=eps_b,
        bond_stiffness_N_m=float(transfer.lattice["bond_stiffness_N_m"]),
    )


def cohesive_energy_error(mapping: BondMapping, n: int = 20001) -> dict:
    """Relative error of the *integrated* bilinear law against ``G_c``.

    The traction is obtained from :func:`tensorphysics.ice.degradation.cohesive_law`
    (MPa interface), so this checks the library path the ice modules actually
    call, not a re-derived formula.
    """
    slips = np.linspace(0.0, mapping.delta_c_m, n)
    tau_mpa = cohesive_law(mapping.tau_peak_Pa * 1.0e-6, slips,
                           mapping.G_c_J_m2)
    energy = float(np.trapezoid(tau_mpa * 1.0e6, slips))
    rel = abs(energy - mapping.G_c_J_m2) / mapping.G_c_J_m2
    return {
        "G_c_target_J_m2": mapping.G_c_J_m2,
        "G_c_integrated_J_m2": energy,
        "rel_dev": rel,
        "n_samples": n,
    }


def dem_failure_load(transfer: IceBondTransfer) -> float:
    """First-fibre failure load of the DEM specimen (sender value if present)."""
    if "failure_load_N" in transfer.beam_theory:
        return float(transfer.beam_theory["failure_load_N"])
    s = transfer.spec
    b, h, L = (float(s["width_m"]), float(s["thickness_m"]), float(s["length_m"]))
    return 2.0 * transfer.bond_strength_Pa * b * h * h / (3.0 * L)


def tensorphysics_failure_load(transfer: IceBondTransfer) -> dict:
    """Failure load rebuilt through the icy-plate limiting moment.

    ``M_lim = sigma_b h^2 / 6`` comes from
    :func:`tensorphysics.ice_sheet.kirchhoff_plate_fem.limiting_moment`; a
    simply supported strip of width ``b`` under a central line load reaches it
    at ``P = 4 M_lim b / L``.  The result should be identical to the DEM
    envelope ``2 sigma_b b h^2 / (3L)``; the check therefore verifies that the
    two repositories use the *same* strength semantics (first-fibre bending
    failure, no plastic reserve), not just the same numbers.
    """
    s = transfer.spec
    b, h, L = (float(s["width_m"]), float(s["thickness_m"]), float(s["length_m"]))
    M_lim = float(limiting_moment(transfer.bond_strength_Pa, h))
    P = 4.0 * M_lim * b / L
    dem = dem_failure_load(transfer)
    return {
        "limiting_moment_Nm_per_m": M_lim,
        "failure_load_N": P,
        "dem_failure_load_N": dem,
        "rel_dev": abs(P - dem) / abs(dem),
        "second_moment_m4": b * h ** 3 / 12.0,
        "section_modulus_m3": b * h * h / 6.0,
    }