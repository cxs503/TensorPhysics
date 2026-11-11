"""1D Transient Heat Conduction in Ice Sheets.

Implements analytical and numerical solutions for transient heat conduction
in ice sheets with surface cooling, providing the temperature field that
drives thermal stress development.

Physical Setup:
  - Ice sheet of thickness H, initially at uniform temperature T_init
  - Top surface (z=0) suddenly cooled to T_surface < T_init
  - Bottom surface (z=H) insulated or at fixed temperature
  - Temperature field T(z,t) evolves by diffusion equation

Analytical Solution (Neumann-type for semi-infinite solid):
  For short times (before bottom boundary feels the cooling):
    T(z,t) = T_surface + (T_init - T_surface) * erf(z / (2*sqrt(alpha*t)))
  
  This is valid when t << H²/(4*alpha), i.e., thermal penetration depth << H.

Numerical Solution:
  Implicit finite difference for full domain with arbitrary BCs.
  Crank-Nicolson scheme for second-order accuracy in time.

Validation:
  Numerical solution validated against analytical for semi-infinite case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erf


# ── Ice thermal properties ─────────────────────────────────────────────
K_ICE = 2.1         # Thermal conductivity (W/(m·K))
RHO_ICE = 917.0     # Density (kg/m³)
CP_ICE = 2090.0     # Specific heat capacity (J/(kg·K))
ALPHA_ICE = K_ICE / (RHO_ICE * CP_ICE)  # Thermal diffusivity (m²/s) ≈ 1.1e-6


@dataclass
class IceThermalConduction:
    """1D transient heat conduction solver for ice sheets.
    
    Parameters
    ----------
    H : float
        Ice sheet thickness (m).
    T_init : float
        Initial uniform temperature (°C).
    T_surface : float
        Surface temperature after cooling (°C).
    bc_bottom : {'insulated', 'fixed'}
        Bottom boundary condition type.
    T_bottom : float, optional
        Bottom temperature if bc_bottom='fixed' (°C).
    nz : int
        Number of vertical grid points.
    dt : float
        Time step (s).
    """
    
    H: float
    T_init: float
    T_surface: float
    bc_bottom: Literal['insulated', 'fixed'] = 'insulated'
    T_bottom: float = 0.0
    nz: int = 101
    dt: float = 60.0  # 1 minute default
    
    def __post_init__(self):
        # Allow both cooling (T_surface < T_init) and warming (T_surface > T_init)
        # Cooling creates tensile stress at surface; warming creates compression
        
        self.dz = self.H / (self.nz - 1)
        self.z = np.linspace(0, self.H, self.nz)
        
        # Initialize temperature field
        self.T = np.full(self.nz, self.T_init)
        self.T[0] = self.T_surface  # Surface BC
        if self.bc_bottom == 'fixed':
            self.T[-1] = self.T_bottom
        
        self.time = 0.0
        
        # Precompute coefficient matrix for implicit scheme
        r = ALPHA_ICE * self.dt / self.dz**2
        
        # Tridiagonal matrix (banded storage for scipy)
        # ab[0, :] = upper diagonal
        # ab[1, :] = main diagonal
        # ab[2, :] = lower diagonal
        n = self.nz
        self._ab = np.zeros((3, n))
        self._ab[0, 1:] = -r / 2.0       # upper (Crank-Nicolson)
        self._ab[2, :-1] = -r / 2.0      # lower (Crank-Nicolson)
        self._ab[1, :] = 1.0 + r         # main diagonal
        
        # Boundary conditions
        self._ab[1, 0] = 1.0             # Dirichlet at surface
        self._ab[0, 1] = 0.0             # No upper coupling from BC
        
        if self.bc_bottom == 'insulated':
            # dT/dz = 0 at bottom: T[n-2] = T[n]
            self._ab[1, -1] = 1.0 + r / 2.0
            self._ab[2, -1] = -r / 2.0
        else:  # fixed
            self._ab[1, -1] = 1.0
            self._ab[2, -1] = 0.0
    
    def step(self, n_steps: int = 1) -> None:
        """Advance solution by n_steps using Crank-Nicolson scheme."""
        r = ALPHA_ICE * self.dt / self.dz**2
        
        for _ in range(n_steps):
            # Right-hand side (explicit part of Crank-Nicolson)
            rhs = np.zeros(self.nz)
            
            # Interior points: (1-r)*T[i] + r/2*(T[i-1] + T[i+1])
            rhs[1:-1] = (1.0 - r) * self.T[1:-1] + (r / 2.0) * (self.T[:-2] + self.T[2:])
            
            # Boundary conditions
            rhs[0] = self.T_surface
            
            if self.bc_bottom == 'insulated':
                rhs[-1] = (1.0 - r / 2.0) * self.T[-1] + (r / 2.0) * self.T[-2]
            else:
                rhs[-1] = self.T_bottom
            
            # Solve tridiagonal system
            self.T = solve_banded((1, 1), self._ab, rhs)
            
            # Enforce BCs
            self.T[0] = self.T_surface
            if self.bc_bottom == 'fixed':
                self.T[-1] = self.T_bottom
            
            self.time += self.dt
    
    def temperature_profile(self) -> np.ndarray:
        """Return current temperature profile T(z)."""
        return self.T.copy()
    
    def thermal_penetration_depth(self) -> float:
        """Estimate thermal penetration depth: delta ~ sqrt(alpha*t)."""
        return math.sqrt(ALPHA_ICE * self.time) if self.time > 0 else 0.0


def transient_temperature_profile(
    z: np.ndarray,
    t: float,
    T_surface: float,
    T_init: float,
    alpha: float = ALPHA_ICE,
) -> np.ndarray:
    """Analytical solution for semi-infinite solid cooling.
    
    T(z,t) = T_surface + (T_init - T_surface) * erf(z / (2*sqrt(alpha*t)))
    
    Valid for t << H²/(4*alpha), i.e., before bottom boundary effect.
    
    Parameters
    ----------
    z : array
        Depth coordinates (m), z=0 at surface.
    t : float
        Time (s).
    T_surface : float
        Surface temperature (°C).
    T_init : float
        Initial temperature (°C).
    alpha : float
        Thermal diffusivity (m²/s).
    
    Returns
    -------
    T : array
        Temperature profile (°C).
    """
    if t <= 0:
        return np.full_like(z, T_init)
    
    eta = z / (2.0 * math.sqrt(alpha * t))
    return T_surface + (T_init - T_surface) * erf(eta)


def cooling_rate(
    z: float,
    t: float,
    T_surface: float,
    T_init: float,
    alpha: float = ALPHA_ICE,
) -> float:
    """Analytical cooling rate dT/dt at depth z and time t.
    
    dT/dt = (T_init - T_surface) * z / (4*sqrt(pi*alpha*t³)) * exp(-z²/(4*alpha*t))
    
    Parameters
    ----------
    z : float
        Depth (m).
    t : float
        Time (s).
    T_surface : float
        Surface temperature (°C).
    T_init : float
        Initial temperature (°C).
    alpha : float
        Thermal diffusivity (m²/s).
    
    Returns
    -------
    dTdt : float
        Cooling rate (°C/s).
    """
    if t <= 0:
        return 0.0
    
    delta_T = T_init - T_surface
    prefactor = delta_T * z / (4.0 * math.sqrt(math.pi * alpha * t**3))
    exponential = math.exp(-z**2 / (4.0 * alpha * t))
    
    return -prefactor * exponential  # Negative because cooling
