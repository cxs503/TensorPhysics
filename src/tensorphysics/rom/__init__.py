"""
POD-Galerkin-DEIM Reduced Order Model (ROM) module.

Provides:
- FOM: 2D Navier-Stokes solver (projection method, periodic domain)
- POD: Proper Orthogonal Decomposition via SVD
- Galerkin: Projection with skew-symmetric operators
- DEIM: Discrete Empirical Interpolation Method
- Energy: Stability verification utilities
"""

from .pod import PODDecomposition
from .galerkin import GalerkinProjector
from .deim import DEIMInterpolator
from .fom_ns2d import NS2DSolver
from .rom_integrator import ROMIntegrator
from .energy import EnergyVerifier

__all__ = [
    'PODDecomposition',
    'GalerkinProjector', 
    'DEIMInterpolator',
    'NS2DSolver',
    'ROMIntegrator',
    'EnergyVerifier',
]
