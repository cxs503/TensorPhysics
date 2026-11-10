"""Core shared constants and environment for underwater acoustics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants (underwater acoustics)
# ---------------------------------------------------------------------------
RHO_WATER: float = 1025.0   # density of seawater [kg/m³]
C_WATER: float = 1500.0     # reference sound speed in seawater [m/s]
P_REF: float = 1.0e-6       # reference pressure for underwater SPL [Pa] (1 μPa)


def TL(p: float | np.ndarray, p_ref: float = P_REF) -> float | np.ndarray:
    """Transmission loss in dB.

    TL = 20 log10(|p| / p_ref)

    Parameters
    ----------
    p : float or array
        Acoustic pressure amplitude [Pa].
    p_ref : float, optional
        Reference pressure [Pa]. Default: 1 μPa (underwater standard).

    Returns
    -------
    float or array
        Transmission loss [dB re 1 μPa].
    """
    p = np.asarray(p, dtype=float)
    # Guard against zero / negative pressure
    p_abs = np.abs(p)
    p_abs = np.where(p_abs < 1e-30, 1e-30, p_abs)
    return 20.0 * np.log10(p_abs / p_ref)


@dataclass
class UnderwaterAcousticEnvironment:
    """Range-independent underwater acoustic environment.

    Parameters
    ----------
    depth : float
        Water column depth [m].
    c0 : float
        Reference sound speed in water [m/s].
    c1 : float
        Sound speed in bottom (semi-infinite half-space) [m/s].
    rho1 : float
        Bottom density [kg/m³].
    alpha : float
        Volume absorption coefficient [dB/km].
    """

    depth: float = 5000.0
    c0: float = C_WATER
    c1: float = 1600.0
    rho1: float = 2000.0
    alpha: float = 0.0

    def __post_init__(self):
        if self.depth <= 0:
            raise ValueError(f"Water depth must be positive, got {self.depth}")
        if self.c0 <= 0 or self.c1 <= 0:
            raise ValueError("Sound speeds must be positive")


@dataclass
class UnderwaterAcousticResult:
    """Container for a full underwater acoustic propagation result."""

    environment: UnderwaterAcousticEnvironment
    frequency: float
    source_depth: float
    receiver_range: np.ndarray
    receiver_depth: float
    tl_pekeris: np.ndarray | None = None
    tl_munk: np.ndarray | None = None
    tl_ray: np.ndarray | None = None
    tl_normal_mode: np.ndarray | None = None
    modes: list = field(default_factory=list)
    eigenrays: list = field(default_factory=list)
    energy_closure: float | None = None
    metadata: dict = field(default_factory=dict)
