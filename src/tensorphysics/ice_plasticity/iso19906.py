"""ISO 19906 ice pressure-area relationships.

ISO 19906 (Arctic offshore structures) specifies ice pressure as:
  p(A) = p0 · A^(-ex)

where:
  p0 : reference pressure at A = 1 m² [MPa]
  ex : pressure-area exponent (typically 0.45–0.55 for first-year ice)
  A  : contact area [m²]

For ice-belt plating analysis, the pressure is applied over the panel area.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class ISO19906Pressure:
    """ISO 19906 ice pressure model."""
    p0: float   # reference pressure at A=1 m² [MPa]
    ex: float   # pressure-area exponent (default 0.45)
    A_min: float = 0.1   # minimum area [m²]
    A_max: float = 100.0  # maximum area [m²]

    def pressure(self, A: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute ice pressure for given contact area(s).

        Parameters
        ----------
        A : float or array
            Contact area(s) [m²]

        Returns
        -------
        p : float or array
            Ice pressure [MPa]. Returns same type/shape as input.
        """
        A_arr = np.atleast_1d(np.asarray(A, dtype=float))
        A_clipped = np.clip(A_arr, self.A_min, self.A_max)
        result = self.p0 * A_clipped ** (-self.ex)
        
        # Return scalar if input was scalar
        if np.ndim(A) == 0:
            return float(result[0])
        return result

    def pressure_Pa(self, A: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Ice pressure in Pa."""
        p = self.pressure(A)
        if isinstance(p, (int, float)):
            return float(p) * 1e6
        return np.asarray(p) * 1e6


@dataclass
class CyclicIcePressure:
    """Cyclic ice pressure block generator.

    Generates pressure time histories from ISO 19906 pressure-area family.
    """
    p0: float   # reference pressure [MPa]
    ex: float   # exponent
    A_ref: float = 1.0  # reference area [m²]

    def generate_block(self, n_cycles: int, A_range: Tuple[float, float],
                       load_ratio: float = 0.0) -> np.ndarray:
        """Generate cyclic pressure block.

        Parameters
        ----------
        n_cycles : int
            Number of cycles
        A_range : (A_min, A_max)
            Contact area range [m²]
        load_ratio : float
            R = p_min / p_max (0 for pulsating, -1 for fully reversed)

        Returns
        -------
        p_history : array of shape (n_cycles, 2)
            [p_min, p_max] for each cycle [MPa]
        """
        A_min, A_max = A_range
        p_max = self.p0 * A_min ** (-self.ex)
        p_min = load_ratio * p_max

        history = np.full((n_cycles, 2), [p_min, p_max])
        return history

    def generate_spectrum(self, n_cycles: int, A_values: np.ndarray,
                          load_ratio: float = 0.0) -> np.ndarray:
        """Generate pressure spectrum from area distribution.

        Parameters
        ----------
        n_cycles : int
            Total number of cycles
        A_values : array
            Contact area values [m²]
        load_ratio : float
            R = p_min / p_max

        Returns
        -------
        p_history : array of shape (n_cycles, 2)
        """
        pressures = self.pressure(A_values)
        p_max_vals = np.atleast_1d(np.asarray(pressures, dtype=float))
        p_min_vals = load_ratio * p_max_vals

        # Distribute cycles across pressure levels
        n_per_level = max(1, n_cycles // len(A_values))
        history = []
        for p_max, p_min in zip(p_max_vals, p_min_vals):
            for _ in range(n_per_level):
                history.append([p_min, p_max])

        history = np.array(history[:n_cycles])
        return history


def typical_ice_belt_params() -> dict:
    """Return typical ice-belt plating parameters for Arctic merchant vessels.

    Based on ISO 19906 and IACS Polar Class requirements.
    """
    return {
        # Ice pressure parameters
        "p0": 3.0,       # MPa, reference pressure for first-year ice
        "ex": 0.45,      # pressure-area exponent
        # Plate geometry
        "plate_thickness": 0.025,  # 25 mm
        "stiffener_spacing": 0.6,  # 600 mm
        "frame_spacing": 2.4,      # 2400 mm
        # Material (marine steel)
        "E": 210e9,      # Pa
        "sigma_y": 355e6,  # Pa (AH36)
        "nu": 0.3,
    }
