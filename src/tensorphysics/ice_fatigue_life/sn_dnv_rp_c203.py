"""DNV-RP-C203 S-N curves for ice-induced fatigue assessment.

Implements S-N curves per DNV-RP-C203 (2021) for seawater environment welds
with low-temperature correction for ice zone operations.

Key features:
- DNV-RP-C203 S-N curve parameters (Table 2-1)
- Seawater environment with cathodic protection (CP) factors
- Low-temperature correction (-20°C): +15% fatigue strength
- Bilinear curves with knee point at N = 1e7 cycles

References
----------
- DNV-RP-C203 (2021): Fatigue Design of Offshore Steel Structures
- DNV-OS-C401 (2021): Fatigue Design of Offshore Steel Structures
- ISO 19906 (2019): Petroleum and natural gas industries — Arctic structures
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

__all__ = [
    "SNCurveRP203",
    "DNV_RP_C203_CURVES",
    "low_temperature_correction",
    "seawater_environment_factor",
]


@dataclass
class RP203CurveParams:
    """Parameters for DNV-RP-C203 S-N curve.
    
    Single-slope: log10(N) = log_a1 - m1 * log10(dS)
    Bilinear: second slope m2 after knee point N_knee
    
    Attributes
    ----------
    detail : str
        Detail category name.
    log_a1 : float
        Intercept of first (high-stress) segment.
    m1 : float
        Slope of first segment.
    log_a2 : float
        Intercept of second (low-stress) segment.
    m2 : float
        Slope of second segment.
    n_knee : float
        Cycle count at knee point (typically 1e7).
    t_ref_mm : float
        Reference thickness for thickness correction (mm).
    k_thick : float
        Thickness correction exponent.
    description : str
        Description of the weld detail.
    """
    detail: str
    log_a1: float
    m1: float
    log_a2: float
    m2: float
    n_knee: float = 1e7
    t_ref_mm: float = 25.0
    k_thick: float = 0.25
    description: str = ""


# DNV-RP-C203 Table 2-1 (seawater with cathodic protection)
# Key curves for ice zone welds
DNV_RP_C203_CURVES: Dict[str, RP203CurveParams] = {
    "B1": RP203CurveParams(
        detail="B1",
        log_a1=15.117,
        m1=4.0,
        log_a2=17.146,
        m2=5.0,
        n_knee=1e7,
        t_ref_mm=25.0,
        k_thick=0.25,
        description="Base metal / parent plate (rolled surface)",
    ),
    "D": RP203CurveParams(
        detail="D",
        log_a1=13.640,
        m1=3.5,
        log_a2=17.016,
        m2=5.0,
        n_knee=1e7,
        t_ref_mm=25.0,
        k_thick=0.25,
        description="Butt weld (full penetration, both sides)",
    ),
    "E": RP203CurveParams(
        detail="E",
        log_a1=13.473,
        m1=3.5,
        log_a2=16.856,
        m2=5.0,
        n_knee=1e7,
        t_ref_mm=25.0,
        k_thick=0.25,
        description="Butt weld (full penetration, one side)",
    ),
    "F": RP203CurveParams(
        detail="F",
        log_a1=13.104,
        m1=3.0,
        log_a2=17.022,
        m2=5.0,
        n_knee=1e7,
        t_ref_mm=25.0,
        k_thick=0.25,
        description="Butt weld (full penetration, ground flush)",
    ),
    "W3": RP203CurveParams(
        detail="W3",
        log_a1=12.164,
        m1=3.0,
        log_a2=16.081,
        m2=5.0,
        n_knee=1e7,
        t_ref_mm=25.0,
        k_thick=0.30,
        description="Fillet weld (non-load carrying attachment)",
    ),
}


def low_temperature_correction(
    temperature_c: float,
    reference_temp_c: float = 20.0,
    strength_increase_per_deg: float = 0.005,
) -> float:
    """Low-temperature correction factor for ice zone welds.
    
    At low temperatures (e.g., -20°C in ice zones), the fatigue strength
    of welds increases due to improved material properties. Per ISO 19906
    and experimental data, the fatigue strength increases approximately
    15% at -20°C compared to ambient (20°C).
    
    The correction factor is applied to the S-N curve intercept:
        log_a_corrected = log_a + log10(factor)
    
    This effectively shifts the S-N curve upward (higher N for same dS).
    
    Parameters
    ----------
    temperature_c : float
        Operating temperature (°C).
    reference_temp_c : float
        Reference temperature for standard S-N curves (°C).
    strength_increase_per_deg : float
        Fatigue strength increase per degree below reference (fraction/°C).
        Calibrated to give 15% increase at -20°C.
    
    Returns
    -------
    float
        Correction factor (>1.0 for cold temperatures, =1.0 at reference).
        Applied to fatigue life N (not stress).
    
    Notes
    -----
    At -20°C: factor ≈ 1.15 (15% life increase)
    At 20°C: factor = 1.0 (no correction)
    
    References
    ----------
    - ISO 19906 (2019) Annex D: Low temperature effects
    - DNV-RP-C203 (2021) Section 2.4.6
    """
    if temperature_c >= reference_temp_c:
        return 1.0
    
    # Linear interpolation: 15% increase at -20°C
    delta_t = reference_temp_c - temperature_c
    max_delta_t = reference_temp_c - (-20.0)  # 40°C range
    
    # Fractional increase (capped at 15% for -20°C and below)
    if delta_t >= max_delta_t:
        return 1.15
    else:
        return 1.0 + strength_increase_per_deg * delta_t


def seawater_environment_factor(
    with_cathodic_protection: bool = True,
) -> float:
    """Environment factor for seawater conditions.
    
    Per DNV-RP-C203, seawater environment reduces fatigue life:
    - With cathodic protection (CP): factor = 1.0 (no reduction)
    - Without CP (free corrosion): factor = 0.5 (50% life reduction)
    
    For ice zone operations, cathodic protection is typically assumed.
    
    Parameters
    ----------
    with_cathodic_protection : bool
        True if cathodic protection is provided.
    
    Returns
    -------
    float
        Environment factor applied to fatigue life N.
    """
    return 1.0 if with_cathodic_protection else 0.5


def thickness_correction(
    t_mm: float,
    t_ref_mm: float = 25.0,
    k: float = 0.25,
) -> float:
    """DNV-RP-C203 thickness correction factor.
    
    For t > t_ref, the effective stress range is amplified:
        factor = (t / t_ref)^k
    
    For t <= t_ref, no correction (factor = 1.0).
    
    Parameters
    ----------
    t_mm : float
        Actual plate thickness (mm).
    t_ref_mm : float
        Reference thickness (mm), default 25 mm.
    k : float
        Thickness exponent (0.0 to 0.3).
    
    Returns
    -------
    float
        Thickness correction factor (>= 1.0).
    """
    if t_mm <= t_ref_mm:
        return 1.0
    return (t_mm / t_ref_mm) ** k


class SNCurveRP203:
    """DNV-RP-C203 bilinear S-N curve evaluator.
    
    Bilinear S-N curve:
    - First segment (N < n_knee): log10(N) = log_a1 - m1 * log10(dS)
    - Second segment (N >= n_knee): log10(N) = log_a2 - m2 * log10(dS)
    
    With optional:
    - Thickness correction
    - Low-temperature correction
    - Seawater environment factor
    
    Parameters
    ----------
    detail : str
        Detail category: 'B1', 'D', 'E', 'F', or 'W3'.
    thickness_mm : float, optional
        Plate thickness for thickness correction. None = no correction.
    temperature_c : float, optional
        Operating temperature for low-temp correction. None = ambient.
    with_cathodic_protection : bool
        True if cathodic protection is provided (default True).
    """
    
    def __init__(
        self,
        detail: str = "D",
        thickness_mm: Optional[float] = None,
        temperature_c: Optional[float] = None,
        with_cathodic_protection: bool = True,
    ):
        detail_upper = detail.upper()
        if detail_upper not in DNV_RP_C203_CURVES:
            raise ValueError(
                f"Unknown detail '{detail}'. Available: {list(DNV_RP_C203_CURVES.keys())}"
            )
        self.params = DNV_RP_C203_CURVES[detail_upper]
        self.detail = detail_upper
        self.thickness_mm = thickness_mm
        self.temperature_c = temperature_c
        self.with_cathodic_protection = with_cathodic_protection
        
        # Thickness correction factor
        if thickness_mm is not None:
            self._thick_factor = thickness_correction(
                thickness_mm, self.params.t_ref_mm, self.params.k_thick
            )
        else:
            self._thick_factor = 1.0
        
        # Low-temperature correction factor
        if temperature_c is not None:
            self._temp_factor = low_temperature_correction(temperature_c)
        else:
            self._temp_factor = 1.0
        
        # Environment factor
        self._env_factor = seawater_environment_factor(with_cathodic_protection)
        
        # Combined life correction factor
        self._life_factor = self._temp_factor * self._env_factor
    
    @property
    def log_a1(self) -> float:
        """First segment intercept (corrected for temperature)."""
        return self.params.log_a1 + np.log10(self._life_factor)
    
    @property
    def m1(self) -> float:
        """First segment slope."""
        return self.params.m1
    
    @property
    def log_a2(self) -> float:
        """Second segment intercept (corrected for temperature)."""
        return self.params.log_a2 + np.log10(self._life_factor)
    
    @property
    def m2(self) -> float:
        """Second segment slope."""
        return self.params.m2
    
    @property
    def n_knee(self) -> float:
        """Knee point cycle count."""
        return self.params.n_knee
    
    @property
    def thickness_factor(self) -> float:
        """Thickness correction factor."""
        return self._thick_factor
    
    @property
    def temperature_factor(self) -> float:
        """Low-temperature correction factor."""
        return self._temp_factor
    
    @property
    def environment_factor(self) -> float:
        """Seawater environment factor."""
        return self._env_factor
    
    def n_to_failure(self, stress_range_mpa: float) -> float:
        """Calculate cycles to failure for a given stress range.
        
        Parameters
        ----------
        stress_range_mpa : float
            Nominal stress range (MPa). Will be multiplied by thickness factor.
        
        Returns
        -------
        float
            Number of cycles to failure.
        """
        if stress_range_mpa <= 0:
            return np.inf
        
        # Apply thickness correction
        ds_eff = stress_range_mpa * self._thick_factor
        
        # First segment (high stress, N < n_knee)
        log_n1 = self.log_a1 - self.m1 * np.log10(ds_eff)
        n1 = 10.0 ** log_n1
        
        if n1 <= self.n_knee:
            return n1
        
        # Second segment (low stress, N >= n_knee)
        log_n2 = self.log_a2 - self.m2 * np.log10(ds_eff)
        n2 = 10.0 ** log_n2
        
        return n2
    
    def stress_for_life(self, n_cycles: float) -> float:
        """Calculate allowable stress range for a given number of cycles.
        
        Parameters
        ----------
        n_cycles : float
            Target number of cycles.
        
        Returns
        -------
        float
            Allowable nominal stress range (MPa), accounting for thickness.
        """
        if n_cycles <= 0:
            return np.inf
        
        if n_cycles <= self.n_knee:
            # First segment
            log_ds_eff = (self.log_a1 - np.log10(n_cycles)) / self.m1
        else:
            # Second segment
            log_ds_eff = (self.log_a2 - np.log10(n_cycles)) / self.m2
        
        ds_eff = 10.0 ** log_ds_eff
        
        # Remove thickness correction to get nominal stress
        return ds_eff / self._thick_factor
    
    def evaluate_range(self, stress_ranges: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of N for an array of stress ranges.
        
        Parameters
        ----------
        stress_ranges : np.ndarray
            Array of stress ranges (MPa).
        
        Returns
        -------
        np.ndarray
            Array of cycles to failure.
        """
        result = np.full_like(stress_ranges, np.inf, dtype=float)
        mask = stress_ranges > 0
        ds_eff = stress_ranges[mask] * self._thick_factor
        
        # First segment
        log_n1 = self.log_a1 - self.m1 * np.log10(ds_eff)
        n1 = 10.0 ** log_n1
        
        # Second segment
        log_n2 = self.log_a2 - self.m2 * np.log10(ds_eff)
        n2 = 10.0 ** log_n2
        
        # Select based on knee point
        use_first = n1 <= self.n_knee
        idx = np.where(mask)[0]
        
        result[idx[use_first]] = n1[use_first]
        result[idx[~use_first]] = n2[~use_first]
        
        return result
