"""Aeroelastic module: LBM-free (thin-aerofoil) FSI for 2-D flutter verification.

Sub-modules
-----------
:mod:`~tensorphysics.aeroelastic.uvlm`
    Glauert-spectral linearised unsteady thin-aerofoil vortex lattice (fluid).
:mod:`~tensorphysics.aeroelastic.section`
    plunge/pitch typical section + Newmark integrator (structure).
:mod:`~tensorphysics.aeroelastic.interface`
    panel-pressure -> modal-force transfer with momentum/energy gates.
:mod:`~tensorphysics.aeroelastic.coupling`
    explicit staggered vs implicit sub-iterated partitioned coupling.
:mod:`~tensorphysics.aeroelastic.theodorsen`
    Theodorsen function, frequency-domain forces, p-k flutter reference.
:mod:`~tensorphysics.aeroelastic.flutter`
    time-domain onset search and theory comparison.
"""

from .coupling import CoupledRun, run_coupled
from .flutter import (Case, envelope_growth, flutter_frequency, flutter_speed,
                      theodorsen_reference, time_domain_run)
from .interface import (generalised_force, interface_gate, momentum_residual,
                        panel_grid, work_conjugacy_residual)
from .section import SectionState, TypicalSection
from .theodorsen import (aero_force_harmonic, aero_operators,
                         divergence_speed, flutter_boundary_tracked,
                         flutter_determinant, pk_eigen_tracked,
                         pk_eigenvalues, pk_flutter, theodorsen_c)
from .uvlm import AeroState, GlauertVortexLattice
from .wagner import A_POLES, BETA_POLES, FIT_ERROR, RationalAero, c_rational

__all__ = [
    "GlauertVortexLattice", "AeroState",
    "RationalAero", "c_rational", "A_POLES", "BETA_POLES", "FIT_ERROR",
    "TypicalSection", "SectionState",
    "panel_grid", "generalised_force", "momentum_residual",
    "work_conjugacy_residual", "interface_gate",
    "run_coupled", "CoupledRun",
    "theodorsen_c", "aero_force_harmonic", "aero_operators",
    "pk_eigenvalues", "pk_flutter", "pk_eigen_tracked",
    "flutter_boundary_tracked", "flutter_determinant", "divergence_speed",
    "Case", "time_domain_run", "envelope_growth", "flutter_speed",
    "flutter_frequency", "theodorsen_reference",
]