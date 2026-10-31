"""Coupled Multi-Year Ice (MYI) Thermodynamic-Constitutive Model.

Integrates the thermodynamic Stefan solver with the constitutive
mechanical property models to provide a unified framework for MYI
analysis. Supports coupling with prop-724-01 (Weibull strength) and
prop-726-P1 (thermodynamic framework) through the FieldStore interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .thermodynamics import (
    ALPHA_ICE,
    StefanSolver,
    brine_volume_fraction,
    effective_thermal_conductivity,
    melting_temperature,
    neumann_interface_position,
)
from .constitutive import (
    WeibullStrengthDistribution,
    compressive_strength,
    elastic_modulus,
    fracture_toughness,
    hall_petch_slope,
)


@dataclass
class MYIModel:
    """Coupled Multi-Year Ice thermodynamic-constitutive model.

    Provides a unified interface for computing MYI thermal evolution
    and mechanical properties. The model couples:
      1. Stefan thermodynamic solver (salinity-dependent phase change)
      2. Constitutive properties (E, sigma_c, K_IC)
      3. Weibull strength distribution (prop-724-01 integration)

    Parameters
    ----------
    salinity : float
        Bulk ice salinity in ppt.
    temperature_surface : float
        Surface temperature in °C.
    porosity : float or None
        Override porosity; if None, computed from salinity and temperature.
    grain_size : float
        Average grain diameter in meters (default 5mm).
    domain_length : float
        Ice thickness / domain length in meters.
    nx : int
        Number of grid cells for Stefan solver.
    dt : float
        Time step in seconds for Stefan solver.
    """

    salinity: float = 5.0
    temperature_surface: float = -20.0
    porosity: float | None = None
    grain_size: float = 5.0e-3  # 5 mm
    domain_length: float = 2.0
    nx: int = 200
    dt: float = 100.0

    # Internal state
    _solver: StefanSolver | None = field(default=None, init=False, repr=False)
    _time: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        T_melt = melting_temperature(self.salinity)
        phi = self.porosity if self.porosity is not None else 0.0
        self._solver = StefanSolver(
            L=self.domain_length,
            nx=self.nx,
            T_surface=self.temperature_surface,
            T_initial=T_melt,
            T_melt=T_melt,
            salinity=self.salinity,
            porosity=phi,
            dt=self.dt,
        )

    def compute_porosity(self, temperature: float) -> float:
        """Compute porosity from salinity and temperature."""
        if self.porosity is not None:
            return self.porosity
        return brine_volume_fraction(self.salinity, temperature)

    def elastic_modulus_at(self, temperature: float,
                           porosity: float | None = None) -> float:
        """Compute elastic modulus at given conditions."""
        phi = porosity if porosity is not None else self.compute_porosity(temperature)
        return float(elastic_modulus(temperature, phi))

    def compressive_strength_at(self, temperature: float,
                                 porosity: float | None = None) -> float:
        """Compute compressive strength at given conditions."""
        phi = porosity if porosity is not None else self.compute_porosity(temperature)
        return float(compressive_strength(temperature, phi))

    def fracture_toughness_at(self, temperature: float,
                               porosity: float | None = None,
                               grain_size: float | None = None) -> float:
        """Compute fracture toughness at given conditions."""
        phi = porosity if porosity is not None else self.compute_porosity(temperature)
        d = grain_size if grain_size is not None else self.grain_size
        return float(fracture_toughness(temperature, phi, d))

    def weibull_distribution(self, temperature: float,
                              porosity: float | None = None,
                              modulus: float = 5.0) -> WeibullStrengthDistribution:
        """Create Weibull strength distribution from constitutive model."""
        phi = porosity if porosity is not None else self.compute_porosity(temperature)
        return WeibullStrengthDistribution.from_constitutive(
            temperature, phi, modulus=modulus
        )

    def advance_thermal(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance the thermal solver and return diagnostics."""
        self._solver.step(n_steps)
        self._time = self._solver.time
        return {
            "time": self._solver.time,
            "interface_position": self._solver.interface_position(),
            "energy_residual": self._solver.energy_residual(),
            "T_min": float(np.min(self._solver.T)),
            "T_max": float(np.max(self._solver.T)),
        }

    def validate_stefan(self, t_final: float) -> dict[str, Any]:
        """Validate the Stefan solver against the Neumann analytical solution.

        Creates a fresh solver instance to ensure clean initial conditions.
        Uses the effective thermal diffusivity consistent with the solver's
        porosity setting for the analytical comparison.

        Parameters
        ----------
        t_final : float
            Total simulation time in seconds.

        Returns
        -------
        dict
            Validation results.
        """
        T_melt = melting_temperature(self.salinity)
        phi = self.porosity if self.porosity is not None else 0.0

        # Create a fresh solver for validation
        solver = StefanSolver(
            L=self.domain_length,
            nx=self.nx,
            T_surface=self.temperature_surface,
            T_initial=T_melt,
            T_melt=T_melt,
            salinity=self.salinity,
            porosity=phi,
            dt=self.dt,
        )
        n_steps = int(t_final / solver.dt)
        solver.step(n_steps)

        s_numerical = solver.interface_position()

        # Analytical solution using the same effective diffusivity
        k_eff = effective_thermal_conductivity(phi)
        alpha_eff = k_eff / (917.0 * 2090.0)
        from .thermodynamics import _neumann_lambda
        lam = _neumann_lambda(self.temperature_surface, T_melt)
        s_analytical = 2.0 * lam * np.sqrt(alpha_eff * t_final)

        relative_error = abs(s_numerical - s_analytical) / (abs(s_analytical) + 1e-30)
        energy_res = solver.energy_residual()

        return {
            "time": t_final,
            "s_numerical": s_numerical,
            "s_analytical": s_analytical,
            "relative_error": relative_error,
            "energy_residual": energy_res,
            "passes_3pct": relative_error < 0.03,
        }

    def parameter_scan_elastic(self,
                                temperatures: np.ndarray,
                                porosities: np.ndarray) -> np.ndarray:
        """2-D parameter scan of elastic modulus."""
        T_grid, phi_grid = np.meshgrid(temperatures, porosities)
        E = elastic_modulus(T_grid, phi_grid)
        return E / 1e9  # Convert to GPa

    def parameter_scan_strength(self,
                                 temperatures: np.ndarray,
                                 porosities: np.ndarray) -> np.ndarray:
        """2-D parameter scan of compressive strength."""
        T_grid, phi_grid = np.meshgrid(temperatures, porosities)
        sigma = compressive_strength(T_grid, phi_grid)
        return sigma / 1e6  # Convert to MPa

    def parameter_scan_toughness(self,
                                  temperatures: np.ndarray,
                                  grain_sizes: np.ndarray,
                                  porosity: float = 0.05) -> np.ndarray:
        """2-D parameter scan of fracture toughness."""
        T_grid, d_grid = np.meshgrid(temperatures, grain_sizes)
        phi_grid = np.full_like(T_grid, porosity)
        K = fracture_toughness(T_grid, phi_grid, d_grid)
        return K / 1e3  # Convert to kPa*sqrt(m)

    def parameter_scan_weibull(self,
                                temperatures: np.ndarray,
                                porosities: np.ndarray,
                                modulus: float = 5.0) -> np.ndarray:
        """2-D parameter scan of Weibull characteristic strength."""
        T_grid, phi_grid = np.meshgrid(temperatures, porosities)
        sigma = compressive_strength(T_grid, phi_grid)
        return sigma / 1e6  # Scale parameter = sigma_c in MPa
