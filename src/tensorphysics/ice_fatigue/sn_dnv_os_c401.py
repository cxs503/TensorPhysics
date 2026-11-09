"""DNV-OS-C401 S-N curves for fatigue assessment of welded ship structures.

Implements the S-N curves from DNV-OS-C401 Table 2-1:
- B1 detail (base metal / parent plate): log a = 15.117, m = 4.0
- F detail (butt weld): log a = 13.640, m = 3.5
- W3 detail (fillet weld): log a = 12.164, m = 3.0

The S-N curve form is:
    log10(N) = log10(a) - m * log10(ΔS)

with thickness correction:
    ΔS_eff = ΔS * (t / t_ref)^k,  k = 0.2 for m <= 3, k = 0.3 for m > 3
    (t_ref = 25 mm per DNV-OS-C401)

Also provides a bilinear variant with fatigue limit at N = 1e7 cycles.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

__all__ = [
    "SNCurveOS401",
    "DNV_OS_C401_CURVES",
    "thickness_correction",
    "fatigue_limit_stress",
]


@dataclass(frozen=True)
class OS401CurveParams:
    """Parameters for a DNV-OS-C401 S-N curve.

    Attributes
    ----------
    detail : str
        Detail category name (e.g. 'B1', 'F', 'W3').
    log_a : float
        Intercept parameter log10(a).
    m : float
        Inverse slope of the S-N curve.
    t_ref_mm : float
        Reference thickness for thickness correction (mm).
    k_thick : float
        Thickness correction exponent.
    description : str
        Human-readable description.
    """
    detail: str
    log_a: float
    m: float
    t_ref_mm: float = 25.0
    k_thick: float = 0.25
    description: str = ""


# DNV-OS-C401 Table 2-1 S-N curve parameters
DNV_OS_C401_CURVES = {
    "B1": OS401CurveParams(
        detail="B1",
        log_a=15.117,
        m=4.0,
        t_ref_mm=25.0,
        k_thick=0.2,  # k=0.2 for m<=4 per DNV-OS-C401 §2.5
        description="Base metal / parent plate (rolled surface)",
    ),
    "F": OS401CurveParams(
        detail="F",
        log_a=13.640,
        m=3.5,
        t_ref_mm=25.0,
        k_thick=0.25,  # k=0.25 for 3<m<4
        description="Butt weld (full penetration, ground flush)",
    ),
    "W3": OS401CurveParams(
        detail="W3",
        log_a=12.164,
        m=3.0,
        t_ref_mm=25.0,
        k_thick=0.3,  # k=0.3 for m<=3
        description="Fillet weld (non-load carrying attachment)",
    ),
}


def thickness_correction(
    t_mm: float,
    t_ref_mm: float = 25.0,
    k: float = 0.25,
) -> float:
    """DNV-OS-C401 thickness correction factor.

    For t > t_ref, the effective stress range is amplified:
        factor = (t / t_ref)^k

    For t <= t_ref, no correction (factor = 1.0).

    Parameters
    ----------
    t_mm : float
        Actual plate thickness (mm).
    t_ref_mm : float
        Reference thickness (mm), default 25 mm per DNV-OS-C401.
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


def fatigue_limit_stress(log_a: float, m: float, n_transition: float = 1e7) -> float:
    """Compute the fatigue limit stress range at N = n_transition.

    From log10(N) = log_a - m * log10(ΔS):
        ΔS = (10^log_a / N)^(1/m)

    Parameters
    ----------
    log_a : float
        S-N curve intercept.
    m : float
        S-N curve slope.
    n_transition : float
        Transition cycle count.

    Returns
    -------
    float
        Stress range at the fatigue limit (MPa).
    """
    return (10.0 ** log_a / n_transition) ** (1.0 / m)


class SNCurveOS401:
    """DNV-OS-C401 S-N curve evaluator.

    Single-slope S-N curve: log10(N) = log_a - m * log10(ΔS)

    With optional bilinear knee at N = 1e7 and thickness correction.

    Parameters
    ----------
    detail : str
        Detail category: 'B1', 'F', or 'W3'.
    thickness_mm : float, optional
        Plate thickness for thickness correction. None = no correction.
    bilinear : bool
        If True, add a fatigue limit knee at N = 1e7 (horizontal segment).
    """

    def __init__(
        self,
        detail: str = "F",
        thickness_mm: Optional[float] = None,
        bilinear: bool = False,
    ):
        detail_upper = detail.upper()
        if detail_upper not in DNV_OS_C401_CURVES:
            raise ValueError(
                f"Unknown detail '{detail}'. Available: {list(DNV_OS_C401_CURVES.keys())}"
            )
        self.params = DNV_OS_C401_CURVES[detail_upper]
        self.detail = detail_upper
        self.thickness_mm = thickness_mm
        self.bilinear = bilinear

        # Thickness correction factor
        if thickness_mm is not None:
            self._thick_factor = thickness_correction(
                thickness_mm, self.params.t_ref_mm, self.params.k_thick
            )
        else:
            self._thick_factor = 1.0

        # Fatigue limit for bilinear curve
        if bilinear:
            self._fatigue_limit = fatigue_limit_stress(
                self.params.log_a, self.params.m
            )
        else:
            self._fatigue_limit = 0.0

    @property
    def log_a(self) -> float:
        return self.params.log_a

    @property
    def m(self) -> float:
        return self.params.m

    @property
    def thickness_factor(self) -> float:
        return self._thick_factor

    def n_to_failure(self, stress_range_mpa: float) -> float:
        """Calculate cycles to failure for a given stress range.

        Parameters
        ----------
        stress_range_mpa : float
            Nominal stress range (MPa). Will be multiplied by thickness factor.

        Returns
        -------
        float
            Number of cycles to failure. Returns inf if below fatigue limit.
        """
        if stress_range_mpa <= 0:
            return np.inf

        # Apply thickness correction
        ds_eff = stress_range_mpa * self._thick_factor

        # Check fatigue limit for bilinear curve
        if self.bilinear and ds_eff < self._fatigue_limit:
            return np.inf

        # log10(N) = log_a - m * log10(ΔS)
        log_n = self.params.log_a - self.params.m * np.log10(ds_eff)
        return 10.0 ** log_n

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

        # log10(ΔS_eff) = (log_a - log10(N)) / m
        log_ds_eff = (self.params.log_a - np.log10(n_cycles)) / self.params.m
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

        if self.bilinear:
            mask2 = ds_eff >= self._fatigue_limit
            ds_valid = ds_eff[mask2]
            log_n = self.params.log_a - self.params.m * np.log10(ds_valid)
            idx = np.where(mask)[0][mask2]
            result[idx] = 10.0 ** log_n
        else:
            log_n = self.params.log_a - self.params.m * np.log10(ds_eff)
            result[mask] = 10.0 ** log_n

        return result
