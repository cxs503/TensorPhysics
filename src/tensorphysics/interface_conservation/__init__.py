"""Cross-solver interface conservation gate (campaign node P189-N01).

Three layers, each independently checkable:

``quantities``
    the conserved interface quantities (mass, normal momentum, mechanical
    energy), their residual definitions (integral and pointwise), the lattice /
    finite-volume unit conversion, and the observed-order estimator.
``fvm``
    a staggered finite-volume incompressible solver (MAC + projection) that
    provides an *independent* discretisation with face-exact mass fluxes.
``lbm``
    a D2Q9 lattice-Boltzmann solver with Zou & He closures and a measured
    momentum-exchange accounting for moving/internal solid links.
``coupling``
    the partitioned FVM-LBM cavity, the interface sampling and the residual
    history (experimental: the two-way partitioned time integration is *not*
    claimed to be stable, see ``docs/INTERFACE_CONSERVATION_P189N01.md``).
``transfer``
    the machine-checkable transfer gate: conservative (flux-derived) versus
    state (velocity) exchange, on a channel with a verified-exact inlet closure,
    including the refinement study.
"""
from __future__ import annotations

from .quantities import (
    CS2,
    InterfaceFlux,
    energy_flux_from_fields,
    fit_order,
    flux_residual,
    lattice_scales,
    local_flux_residual,
)
from .lbm import D2Q9
from .fvm import StaggeredFVM
from .coupling import CoupledCavity, CoupledSample, DOMAIN, refinement_study, run_coupled_cavity
from .transfer import (
    CHANNEL_DOMAIN,
    TransferSample,
    run_channel_transfer,
    transfer_refinement,
)

__all__ = [
    "CS2",
    "InterfaceFlux",
    "lattice_scales",
    "flux_residual",
    "local_flux_residual",
    "fit_order",
    "energy_flux_from_fields",
    "D2Q9",
    "StaggeredFVM",
    "CoupledCavity",
    "CoupledSample",
    "DOMAIN",
    "run_coupled_cavity",
    "refinement_study",
    "CHANNEL_DOMAIN",
    "TransferSample",
    "run_channel_transfer",
    "transfer_refinement",
]
