"""Ice load model: half-sine pulse with IACS UR I3 peak pressure.

Ice pressure model
------------------
The ice impact is modelled as a half-sine pulse::

    p_ice(t) = p_0 * sin(pi * t / T),  0 <= t <= T

where:
- p_0: peak pressure from IACS UR I3
- T: pulse duration (typically 0.1-0.5 s for ice impact)

IACS UR I3 peak pressure
-------------------------
For polar class ships, the design ice pressure is computed from::

    p_0 = CF * CA * (AR)^(-0.3) * PC^0.7    [MPa]

where:
- CF: class factor (depends on ice class PC1-PC7)
- CA: area correction factor
- AR: aspect ratio of loaded area
- PC: polar class number (1-7)

For simplified analysis, we use the IACS UR I3 formula for local ice pressure
on shell plating::

    p_0 = 3.0 * PC^0.7    [MPa]  (simplified, for AR = 1)

The actual pressure on a panel of area A is::

    p(A) = p_0 * A^(-0.3)    [MPa]
"""

import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple

# IACS UR I3 class factors for polar classes PC1-PC7
# CF values from IACS Unified Requirement I3
IACS_PC_CLASS_FACTOR = {
    1: 1.00,  # PC1 (highest)
    2: 0.85,
    3: 0.72,
    4: 0.58,
    5: 0.45,
    6: 0.35,
    7: 0.25,  # PC7 (lowest)
}

# Reference pressure at A=1 m² for each PC class [MPa]
# Based on IACS UR I3: p = CF * 3.0 * AR^(-0.3)
IACS_PC_REF_PRESSURE = {
    pc: 3.0 * cf for pc, cf in IACS_PC_CLASS_FACTOR.items()
}


@dataclass(frozen=True)
class IACSUR_I3_Pressure:
    """IACS UR I3 ice pressure model for polar class ships.

    Parameters
    ----------
    pc : int
        Polar class (1-7, PC1 = highest ice class)
    AR : float
        Aspect ratio of loaded area (length/width), default 1.0
    A : float
        Loaded area [m²], default 1.0
    """
    pc: int
    AR: float = 1.0
    A: float = 1.0

    def __post_init__(self):
        if self.pc not in IACS_PC_CLASS_FACTOR:
            raise ValueError(f"PC must be 1-7, got {self.pc}")
        if self.AR <= 0:
            raise ValueError("AR must be positive")
        if self.A <= 0:
            raise ValueError("A must be positive")

    @property
    def CF(self) -> float:
        """Class factor from IACS UR I3."""
        return IACS_PC_CLASS_FACTOR[self.pc]

    @property
    def p_ref(self) -> float:
        """Reference pressure at A=1 m², AR=1 [MPa]."""
        return IACS_PC_REF_PRESSURE[self.pc]

    def peak_pressure(self, A: float | None = None, AR: float | None = None) -> float:
        """Peak ice pressure [MPa] for given area and aspect ratio.

        IACS UR I3: p = CF * CA * AR^(-0.3) * PC_factor
        Simplified: p = p_ref * (A/1)^(-0.3) * (AR/1)^(-0.3)
        """
        A = A if A is not None else self.A
        AR = AR if AR is not None else self.AR

        # Area correction: CA = A^(-0.3)
        CA = A ** (-0.3)
        # Aspect ratio correction
        AR_corr = AR ** (-0.3)

        p = self.p_ref * CA * AR_corr
        return float(p)

    def peak_pressure_Pa(self, A: float | None = None, AR: float | None = None) -> float:
        """Peak ice pressure in Pa."""
        return self.peak_pressure(A, AR) * 1e6


@dataclass(frozen=True)
class HalfSineIcePulse:
    """Half-sine pulse ice load model.

    p(t) = p_0 * sin(pi * t / T),  0 <= t <= T

    Parameters
    ----------
    p0 : float
        Peak pressure [Pa]
    T : float
        Pulse duration [s]
    """
    p0: float  # Pa
    T: float   # s

    def pressure(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Ice pressure at time t [Pa].

        Returns 0 for t < 0 or t > T.
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        result = np.zeros_like(t_arr)
        mask = (t_arr >= 0) & (t_arr <= self.T)
        result[mask] = self.p0 * np.sin(np.pi * t_arr[mask] / self.T)

        if np.ndim(t) == 0:
            return float(result[0])
        return result

    def impulse(self) -> float:
        """Total impulse per unit area [Pa·s].

        I = integral_0^T p_0 * sin(pi*t/T) dt = 2*p_0*T/pi
        """
        return 2.0 * self.p0 * self.T / np.pi

    def peak_force(self, area: float) -> float:
        """Peak force on area [N]."""
        return self.p0 * area

    def time_to_peak(self) -> float:
        """Time to reach peak pressure [s]."""
        return self.T / 2.0

    def time_history(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate time history arrays.

        Returns
        -------
        t : array
            Time values [s]
        p : array
            Pressure values [Pa]
        """
        t = np.linspace(0, self.T, n_points)
        p = self.pressure(t)
        return t, p


def compute_ice_load(pc: int, panel_area: float, panel_AR: float,
                     pulse_duration: float = 0.2) -> dict:
    """Compute ice load parameters for a panel.

    Parameters
    ----------
    pc : int
        Polar class (1-7)
    panel_area : float
        Panel loaded area [m²]
    panel_AR : float
        Panel aspect ratio
    pulse_duration : float
        Ice pulse duration [s], default 0.2

    Returns
    -------
    result : dict
        Ice load parameters including peak pressure, pulse model, and impulse
    """
    ice_model = IACSUR_I3_Pressure(pc=pc, A=panel_area, AR=panel_AR)
    p0_MPa = ice_model.peak_pressure()
    p0_Pa = p0_MPa * 1e6

    pulse = HalfSineIcePulse(p0=p0_Pa, T=pulse_duration)

    return {
        "pc": pc,
        "panel_area_m2": panel_area,
        "panel_AR": panel_AR,
        "p0_MPa": p0_MPa,
        "p0_Pa": p0_Pa,
        "pulse_duration_s": pulse_duration,
        "impulse_Pa_s": pulse.impulse(),
        "peak_force_N": pulse.peak_force(panel_area),
        "time_to_peak_s": pulse.time_to_peak(),
    }
