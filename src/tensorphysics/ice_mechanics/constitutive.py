"""Constitutive submodel for Multi-Year Ice (MYI).

Implements temperature- and porosity-dependent mechanical properties:

  - Elastic modulus: E(T, phi) = E_0 * exp(-alpha*phi) * (1 + beta*DeltaT)
  - Compressive strength: sigma_c(T, phi) = sigma_0(T) * (1 - phi/phi_crit)^gamma
  - Fracture toughness: K_IC(T, phi, d) with Hall-Petch grain-size relation
  - Weibull strength distribution for statistical fracture modeling

Calibrated against:
  - Timco & Frederking (2007): E at T=-2°C, phi=0.05
  - Schulson (1999): sigma_c at T=-10°C, s=5ppt
  - Dempsey et al. (1999): K_IC Hall-Petch slope
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ── Reference constants ────────────────────────────────────────────────

# Elastic modulus parameters
E_0 = 10.5e9        # Reference elastic modulus at T_ref, phi=0 (Pa)
ALPHA_E = 4.0        # Porosity sensitivity exponent
BETA_E = -0.015      # Temperature sensitivity coefficient (1/°C)
T_REF = 0.0          # Reference temperature (°C)

# Compressive strength parameters
SIGMA_REF = 5.0e6    # Reference compressive strength at T_ref (Pa)
KAPPA_SIGMA = 0.065  # Temperature sensitivity for sigma_0 (1/°C)
PHI_CRIT = 0.40      # Critical porosity (dimensionless)
GAMMA_SIGMA = 0.50   # Porosity exponent

# Fracture toughness parameters (Hall-Petch)
K_IC_0 = 0.1e6       # Base fracture toughness (Pa*sqrt(m))
K_D = 0.015e6        # Hall-Petch slope coefficient (Pa*m)
# K_IC = K_IC_0 + K_D / sqrt(d), d in meters

# Weibull distribution parameters
WEIBULL_MODULUS_DEFAULT = 5.0  # Typical for sea ice (m)


def elastic_modulus(temperature: float | np.ndarray,
                    porosity: float | np.ndarray) -> float | np.ndarray:
    """Temperature- and porosity-dependent elastic modulus for MYI.

    E(T, phi) = E_0 * exp(-alpha * phi) * (1 + beta * (T - T_ref))

    The temperature term (1 + beta * DeltaT) increases modulus for colder
    ice since beta < 0 and DeltaT = T - T_ref < 0 for sub-freezing temps.

    Parameters
    ----------
    temperature : float or ndarray
        Ice temperature in °C (typically negative).
    porosity : float or ndarray
        Total porosity (brine + air) as volume fraction [0, 1).

    Returns
    -------
    float or ndarray
        Elastic modulus in Pa.

    References
    ----------
    Timco, G.W. & Frederking, R.M.W. (2007). Cold Regions Science and
    Technology, 47, 299-311.
    """
    delta_T = temperature - T_REF
    E = E_0 * np.exp(-ALPHA_E * porosity) * (1.0 + BETA_E * delta_T)
    # Ensure non-negative modulus
    return np.maximum(E, 0.0)


def compressive_strength(temperature: float | np.ndarray,
                         porosity: float | np.ndarray) -> float | np.ndarray:
    """Temperature- and porosity-dependent compressive strength for MYI.

    sigma_c(T, phi) = sigma_0(T) * (1 - phi/phi_crit)^gamma

    where sigma_0(T) = sigma_ref * (1 + kappa * |T - T_ref|)

    Parameters
    ----------
    temperature : float or ndarray
        Ice temperature in °C (typically negative).
    porosity : float or ndarray
        Total porosity (brine + air) as volume fraction [0, phi_crit).

    Returns
    -------
    float or ndarray
        Compressive strength in Pa.

    References
    ----------
    Schulson, E.M. (1999). Journal of Glaciology, 45(150), 406-416.
    """
    delta_T_abs = np.abs(temperature - T_REF)
    sigma_0 = SIGMA_REF * (1.0 + KAPPA_SIGMA * delta_T_abs)
    ratio = np.clip(porosity / PHI_CRIT, 0.0, 1.0 - 1e-12)
    sigma_c = sigma_0 * (1.0 - ratio) ** GAMMA_SIGMA
    return sigma_c


def fracture_toughness(temperature: float | np.ndarray,
                        porosity: float | np.ndarray,
                        grain_size: float | np.ndarray) -> float | np.ndarray:
    """Fracture toughness with Hall-Petch grain-size relation.

    K_IC(T, phi, d) = [K_IC_0 + K_D / sqrt(d)] * f(T, phi)

    where f(T, phi) = exp(-alpha_k * phi) * (1 + beta_k * |DeltaT|)
    accounts for temperature and porosity effects on toughness.

    Parameters
    ----------
    temperature : float or ndarray
        Ice temperature in °C.
    porosity : float or ndarray
        Total porosity as volume fraction.
    grain_size : float or ndarray
        Average grain diameter in meters.

    Returns
    -------
    float or ndarray
        Fracture toughness K_IC in Pa*sqrt(m).

    References
    ----------
    Dempsey, J.P. et al. (1999). Engineering Fracture Mechanics, 64, 305-323.
    Schulson, E.M. & Duval, P. (2009). Creep and Fracture of Ice. Cambridge.
    """
    # Hall-Petch base term
    K_hp = K_IC_0 + K_D / np.sqrt(np.maximum(grain_size, 1e-6))

    # Temperature and porosity correction
    alpha_k = 3.0   # Porosity sensitivity for toughness
    beta_k = 0.01   # Temperature sensitivity for toughness (1/°C)
    delta_T_abs = np.abs(temperature - T_REF)
    f_correction = np.exp(-alpha_k * porosity) * (1.0 + beta_k * delta_T_abs)

    return K_hp * f_correction


def hall_petch_slope(temperatures: np.ndarray,
                     porosities: np.ndarray,
                     grain_sizes: np.ndarray) -> float:
    """Estimate the Hall-Petch slope from parameter arrays.

    Fits K_IC vs 1/sqrt(d) to extract the slope K_D, which should
    match the literature value.

    Parameters
    ----------
    temperatures : ndarray
        Temperature array in °C.
    porosities : ndarray
        Porosity array.
    grain_sizes : ndarray
        Grain size array in meters.

    Returns
    -------
    float
        Estimated Hall-Petch slope K_D in Pa*m.
    """
    K_values = fracture_toughness(temperatures, porosities, grain_sizes)
    x = 1.0 / np.sqrt(grain_sizes)

    # Linear regression: K_IC = K_0 + K_D * (1/sqrt(d))
    # Use least squares
    A = np.column_stack([np.ones_like(x), x])
    result = np.linalg.lstsq(A, K_values, rcond=None)
    K_D_estimated = result[0][1]
    return K_D_estimated


@dataclass
class WeibullStrengthDistribution:
    """Weibull strength distribution for statistical ice fracture.

    P_f(sigma) = 1 - exp[-(sigma / sigma_0)^m]

    This integrates with the prop-724-01 Weibull framework for
    probabilistic failure assessment of MYI.

    Parameters
    ----------
    modulus : float
        Weibull modulus m (shape parameter). Typical sea ice: 3-7.
    scale : float
        Scale parameter sigma_0 in Pa (characteristic strength).
    volume : float
        Reference volume in m³.
    """

    modulus: float = WEIBULL_MODULUS_DEFAULT
    scale: float = 1.0e6
    volume: float = 1.0

    def probability_of_failure(self, stress: float | np.ndarray) -> float | np.ndarray:
        """Compute probability of failure at given stress.

        P_f = 1 - exp[-(sigma / sigma_0)^m]

        Parameters
        ----------
        stress : float or ndarray
            Applied stress in Pa.

        Returns
        -------
        float or ndarray
            Failure probability [0, 1].
        """
        ratio = np.maximum(stress / self.scale, 0.0)
        return 1.0 - np.exp(-(ratio ** self.modulus))

    def characteristic_strength(self, pf: float = 0.632) -> float:
        """Stress at which probability of failure equals pf.

        Parameters
        ----------
        pf : float
            Target failure probability (default: 1 - 1/e ≈ 0.632).

        Returns
        -------
        float
            Characteristic strength in Pa.
        """
        if pf <= 0 or pf >= 1:
            raise ValueError("pf must be in (0, 1)")
        return self.scale * (-np.log(1.0 - pf)) ** (1.0 / self.modulus)

    def mean_strength(self) -> float:
        """Mean strength from the Weibull distribution.

        E[sigma] = sigma_0 * Gamma(1 + 1/m)

        Returns
        -------
        float
            Mean strength in Pa.
        """
        return self.scale * math.gamma(1.0 + 1.0 / self.modulus)

    @classmethod
    def from_constitutive(cls, temperature: float, porosity: float,
                          volume: float = 1.0,
                          modulus: float = WEIBULL_MODULUS_DEFAULT
                          ) -> "WeibullStrengthDistribution":
        """Create a Weibull distribution calibrated from constitutive model.

        Uses the deterministic compressive strength as the scale parameter.

        Parameters
        ----------
        temperature : float
            Ice temperature in °C.
        porosity : float
            Total porosity.
        volume : float
            Reference volume in m³.
        modulus : float
            Weibull modulus.

        Returns
        -------
        WeibullStrengthDistribution
        """
        sigma_c = float(compressive_strength(temperature, porosity))
        return cls(modulus=modulus, scale=sigma_c, volume=volume)
