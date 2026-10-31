"""Thermodynamic submodel for Multi-Year Ice (MYI).

Extends the classical Stefan problem with salinity-dependent phase change
temperature and porosity-dependent effective thermal conductivity.

Key relations:
  - Melting temperature: T_m(s) = T_0 - mu * s
  - Effective conductivity: k_eff(phi) = k_ice * (1 - phi)^1.5
  - Brine volume fraction: v_b from liquidus equilibrium

The Stefan solver uses an enthalpy method with proper latent heat tracking,
validated against the Neumann analytical solution for the one-phase
solidification problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.special import erf


# ── Physical constants for sea ice ──────────────────────────────────────
T_0 = 0.0          # Reference freezing point of pure water (°C)
MU = 0.054          # Liquidus slope (°C/ppt)
K_ICE = 2.1         # Thermal conductivity of pure ice (W/(m·K))
RHO_ICE = 917.0     # Density of ice (kg/m³)
RHO_BRINE = 1025.0  # Density of brine (kg/m³)
CP_ICE = 2090.0     # Specific heat capacity of ice (J/(kg·K))
L_FUSION = 334000.0 # Latent heat of fusion for ice (J/kg)
ALPHA_ICE = K_ICE / (RHO_ICE * CP_ICE)  # Thermal diffusivity (m²/s)
H_LATENT = RHO_ICE * L_FUSION  # Volumetric latent heat (J/m³)


def melting_temperature(salinity: float | np.ndarray) -> float | np.ndarray:
    """Salinity-dependent melting (phase change) temperature.

    T_m(s) = T_0 - mu * s
    """
    return T_0 - MU * salinity


def effective_thermal_conductivity(porosity: float | np.ndarray) -> float | np.ndarray:
    """Porosity-dependent effective thermal conductivity.

    k_eff(phi) = k_ice * (1 - phi)^1.5
    """
    return K_ICE * (1.0 - porosity) ** 1.5


def brine_volume_fraction(salinity: float, temperature: float) -> float:
    """Brine volume fraction from liquidus equilibrium.

    v_b = (s_ice / s_brine) * (rho_ice / rho_brine)
    where s_brine = |T| / mu from the liquidus condition.
    """
    if temperature > 0:
        raise ValueError(f"Temperature must be <= 0°C, got {temperature}")
    if salinity < 0:
        raise ValueError(f"Salinity must be >= 0 ppt, got {salinity}")
    if salinity == 0:
        return 0.0
    T_abs = abs(temperature)
    if T_abs < 1e-10:
        return 1.0
    s_brine = T_abs / MU
    v_b = (salinity / s_brine) * (RHO_ICE / RHO_BRINE)
    return min(max(v_b, 0.0), 1.0)


# ── Neumann analytical solution ────────────────────────────────────────

def _neumann_lambda(T_surface: float, T_melt: float = 0.0) -> float:
    """Solve for the Neumann similarity parameter lambda.

    lambda * erf(lambda) * exp(lambda^2) = St / sqrt(pi)
    where St = c_p * (T_melt - T_surface) / L_fusion.
    """
    delta_T = T_melt - T_surface
    if delta_T <= 0:
        raise ValueError("T_surface must be below T_melt for ice growth")
    St = CP_ICE * delta_T / L_FUSION
    rhs = St / math.sqrt(math.pi)

    def equation(lam: float) -> float:
        return lam * math.erf(lam) * math.exp(lam ** 2) - rhs

    return brentq(equation, 1e-12, 10.0, xtol=1e-14)


def neumann_interface_position(t: float, T_surface: float,
                                T_melt: float = 0.0) -> float:
    """Analytical Neumann solution: s(t) = 2*lambda*sqrt(alpha*t)."""
    lam = _neumann_lambda(T_surface, T_melt)
    return 2.0 * lam * math.sqrt(ALPHA_ICE * t)


def neumann_temperature_profile(x: float | np.ndarray, t: float,
                                 T_surface: float,
                                 T_melt: float = 0.0) -> float | np.ndarray:
    """Analytical Neumann temperature profile."""
    lam = _neumann_lambda(T_surface, T_melt)
    eta = x / (2.0 * math.sqrt(ALPHA_ICE * t))
    return T_surface + (T_melt - T_surface) * erf(eta) / erf(lam)


# ── Numerical Stefan solver (enthalpy method) ──────────────────────────

@dataclass
class StefanSolver:
    """1-D Stefan problem solver using the enthalpy method.

    Uses an implicit scheme for heat conduction with enthalpy-based
    latent heat tracking. Uses scipy.linalg.solve_banded for efficient
    tridiagonal solves.

    Enthalpy-temperature relationship:
      - Solid (T < T_melt):  H = rho*cp*(T - T_melt)          [H < 0]
      - Mushy (T = T_melt):  H in [0, rho*L]                  [phase change]
      - Liquid (T > T_melt): H = rho*cp*(T - T_melt) + rho*L  [H > rho*L]

    Parameters
    ----------
    L : float
        Domain length in meters.
    nx : int
        Number of grid cells.
    T_surface : float
        Fixed surface (Dirichlet) temperature in °C.
    T_initial : float
        Initial temperature in °C.
    T_melt : float
        Phase change temperature in °C.
    salinity : float
        Bulk salinity in ppt.
    porosity : float
        Porosity for effective conductivity.
    dt : float
        Time step in seconds.
    """

    L: float
    nx: int
    T_surface: float
    T_initial: float = 0.0
    T_melt: float = 0.0
    salinity: float = 0.0
    porosity: float = 0.0
    dt: float = 1.0

    def __post_init__(self):
        if self.salinity > 0 and self.T_melt == 0.0:
            self.T_melt = melting_temperature(self.salinity)
        self.dx = self.L / self.nx
        self.k_eff = effective_thermal_conductivity(self.porosity)

        # Initialize: all liquid at T_initial
        self.T = np.full(self.nx, self.T_initial, dtype=float)
        self.T[0] = self.T_surface
        self.H = self._T_to_H_vec(self.T)
        self.time = 0.0

        # Precompute tridiagonal band matrix (constant coefficients)
        alpha = self.k_eff / (RHO_ICE * CP_ICE)
        r = alpha * self.dt / self.dx ** 2
        n = self.nx

        # Banded matrix storage for scipy.linalg.solve_banded
        # ab[0, :] = upper diagonal (c_i = -r), shifted right by 1
        # ab[1, :] = main diagonal (b_i)
        # ab[2, :] = lower diagonal (a_i = -r), shifted left by 1
        self._ab = np.zeros((3, n))
        self._ab[0, 1:] = -r       # upper diagonal
        self._ab[2, :-1] = -r      # lower diagonal
        self._ab[1, :] = 1.0 + 2.0 * r  # main diagonal (interior)

        # Boundary modifications
        self._ab[1, 0] = 1.0       # Dirichlet left
        self._ab[0, 1] = 0.0       # no upper coupling from BC cell
        self._ab[1, -1] = 1.0 + r  # Neumann right
        self._ab[2, -1] = -r       # lower coupling for last cell

        # BC enthalpy
        self._H_bc_left = float(
            RHO_ICE * CP_ICE * (self.T_surface - self.T_melt)
            if self.T_surface < self.T_melt
            else RHO_ICE * CP_ICE * (self.T_surface - self.T_melt) + H_LATENT
        )

    def _T_to_H_vec(self, T: np.ndarray) -> np.ndarray:
        """Convert temperature array to enthalpy array (vectorized)."""
        return np.where(
            T < self.T_melt,
            RHO_ICE * CP_ICE * (T - self.T_melt),
            np.where(
                T > self.T_melt,
                RHO_ICE * CP_ICE * (T - self.T_melt) + H_LATENT,
                H_LATENT
            )
        )

    def _H_to_T_vec(self, H: np.ndarray) -> np.ndarray:
        """Convert enthalpy array to temperature array (vectorized)."""
        return np.where(
            H < 0,
            self.T_melt + H / (RHO_ICE * CP_ICE),
            np.where(
                H > H_LATENT,
                self.T_melt + (H - H_LATENT) / (RHO_ICE * CP_ICE),
                self.T_melt
            )
        )

    def step(self, n_steps: int = 1) -> None:
        """Advance the solver by n_steps using implicit enthalpy method."""
        ab = self._ab
        H_bc = self._H_bc_left

        for _ in range(n_steps):
            # Right-hand side: current temperature
            d = self.T.copy()
            d[0] = self.T_surface  # Dirichlet BC

            # Solve tridiagonal system
            T_star = solve_banded((1, 1), ab, d)

            # Enthalpy update: H_new = H_old + rho*cp*(T* - T_old)
            self.H += RHO_ICE * CP_ICE * (T_star - self.T)

            # Recover temperature from enthalpy
            self.T = self._H_to_T_vec(self.H)

            # Enforce boundary conditions
            self.T[0] = self.T_surface
            self.H[0] = H_bc

            self.time += self.dt

    def interface_position(self) -> float:
        """Estimate the phase interface position from the enthalpy field.

        The interface is where H crosses from solid (H < 0) to
        mushy/liquid (H >= 0).
        """
        solid_mask = self.H < 0
        if not np.any(solid_mask):
            return 0.0
        last_solid = int(np.max(np.where(solid_mask)[0]))

        if last_solid >= self.nx - 1:
            return self.L

        H_left = self.H[last_solid]
        H_right = self.H[last_solid + 1]
        denom = H_right - H_left
        if abs(denom) < 1e-30:
            frac = 0.5
        else:
            frac = (0.0 - H_left) / denom
        return (last_solid + frac) * self.dx

    def energy_residual(self) -> float:
        """Compute normalized energy balance residual."""
        total_H = np.sum(self.H) * self.dx
        if self.T_initial < self.T_melt:
            H_init_val = RHO_ICE * CP_ICE * (self.T_initial - self.T_melt)
        elif self.T_initial > self.T_melt:
            H_init_val = RHO_ICE * CP_ICE * (self.T_initial - self.T_melt) + H_LATENT
        else:
            H_init_val = H_LATENT
        total_H_init = H_init_val * self.L
        return abs(total_H - total_H_init) / (H_LATENT * self.L + 1e-30)
