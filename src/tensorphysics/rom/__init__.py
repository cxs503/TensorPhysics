"""
POD-Galerkin-DEIM Reduced Order Model (ROM) module.

Provides:
- FOM: 2D Navier-Stokes solver (projection method, periodic domain)
- POD: Proper Orthogonal Decomposition via SVD
- Galerkin: Projection with skew-symmetric operators
- DEIM: Discrete Empirical Interpolation Method
- Energy: Stability verification utilities
- ErrorBound: Certified a posteriori error bounds for ROM solutions
"""

from .pod import PODDecomposition
from .galerkin import GalerkinProjector
from .deim import DEIMInterpolator
from .fom_ns2d import NS2DSolver
from .rom_integrator import ROMIntegrator
from .energy import EnergyVerifier
from .error_bound import ResidualErrorBound
from .certified_bound import (
    ConsistentPODGalerkinROM,
    PosterioriResidualBound,
    true_error,
)

__all__ = [
    'PODDecomposition',
    'GalerkinProjector', 
    'DEIMInterpolator',
    'NS2DSolver',
    'ROMIntegrator',
    'EnergyVerifier',
    'ResidualErrorBound',
    'ConsistentPODGalerkinROM',
    'PosterioriResidualBound',
    'true_error',
]
