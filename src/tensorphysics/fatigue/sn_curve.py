"""DNV-RP-C203 S-N curves for fatigue assessment.

Implements D and F curves for:
- Seawater with cathodic protection (CP)
- Seawater free corrosion

Bilinear S-N curve with transition at N = 10^7 cycles.
Ensures continuity at the transition point.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SNCurveParams:
    """Parameters for a bilinear S-N curve per DNV-RP-C203.

    For N <= N_transition:
        log10(N) = log10(K1) - m1 * log10(S)

    For N > N_transition:
        log10(N) = log10(K2) - m2 * log10(S)

    Continuity at transition is ensured by:
        log10(K2) = log10(K1) + (m2 - m1) * log10(S_transition)
    where S_transition is the stress at N_transition on the first segment.
    """
    m1: float          # slope exponent for first segment (typically 3-5)
    log_K1: float      # log10 of intercept constant for first segment
    m2: float          # slope exponent for second segment (m1 + 2)
    log_K2: float      # log10 of intercept constant for second segment
    n_transition: float  # transition cycle count (typically 1e7)


# DNV-RP-C203 standard curve parameters
# The log_K2 values are computed to ensure continuity at N=1e7

def _compute_log_K2(m1, log_K1, m2, n_transition):
    """Compute log_K2 to ensure continuity at transition.

    At transition: N = n_transition, S = S_t
    From segment 1: log10(S_t) = (log_K1 - log10(n_transition)) / m1
    For segment 2: log_K2 = log10(n_transition) + m2 * log10(S_t)
    """
    log_S_t = (log_K1 - np.log10(n_transition)) / m1
    log_K2 = np.log10(n_transition) + m2 * log_S_t
    return log_K2


DNV_CURVES = {
    # D-curve: seawater with cathodic protection
    ("D", "seawater_cp"): SNCurveParams(
        m1=3.0,
        log_K1=12.102,  # DNV D-curve
        m2=5.0,
        log_K2=None,  # Will be computed
        n_transition=1e7,
    ),
    # D-curve: seawater free corrosion
    ("D", "seawater_free"): SNCurveParams(
        m1=3.0,
        log_K1=12.102,
        m2=5.0,
        log_K2=None,
        n_transition=1e7,
    ),
    # F-curve: seawater with cathodic protection
    ("F", "seawater_cp"): SNCurveParams(
        m1=3.0,
        log_K1=11.644,  # DNV F-curve
        m2=5.0,
        log_K2=None,
        n_transition=1e7,
    ),
    # F-curve: seawater free corrosion
    ("F", "seawater_free"): SNCurveParams(
        m1=3.0,
        log_K1=11.644,
        m2=5.0,
        log_K2=None,
        n_transition=1e7,
    ),
    # C-curve (for reference)
    ("C", "seawater_cp"): SNCurveParams(
        m1=3.0,
        log_K1=12.402,
        m2=5.0,
        log_K2=None,
        n_transition=1e7,
    ),
    ("C", "seawater_free"): SNCurveParams(
        m1=3.0,
        log_K1=12.402,
        m2=5.0,
        log_K2=None,
        n_transition=1e7,
    ),
    # B-curve (for reference)
    ("B", "seawater_cp"): SNCurveParams(
        m1=4.0,
        log_K1=15.307,
        m2=6.0,
        log_K2=None,
        n_transition=1e7,
    ),
    ("B", "seawater_free"): SNCurveParams(
        m1=4.0,
        log_K1=15.307,
        m2=6.0,
        log_K2=None,
        n_transition=1e7,
    ),
}

# Compute log_K2 for all curves to ensure continuity
for key, params in DNV_CURVES.items():
    if params.log_K2 is None:
        params.log_K2 = _compute_log_K2(
            params.m1, params.log_K1, params.m2, params.n_transition
        )


@dataclass
class SNCurveResult:
    """Result of S-N curve evaluation."""
    stress_range: float    # MPa
    n_to_failure: float    # cycles to failure
    curve_class: str       # DNV curve class
    environment: str       # environment type


class SNCurve:
    """Single-slope S-N curve for closed-form comparison.

    S^m * N = K  =>  N = K / S^m
    """

    def __init__(self, m: float, log_K: float):
        self.m = m
        self.log_K = log_K
        self.K = 10 ** log_K

    def n_to_failure(self, stress_range: float) -> float:
        if stress_range <= 0:
            return np.inf
        return self.K / (stress_range ** self.m)

    def stress_for_life(self, n_cycles: float) -> float:
        if n_cycles <= 0:
            return np.inf
        return (self.K / n_cycles) ** (1.0 / self.m)


class DNCurve:
    """DNV-RP-C203 S-N curve evaluator.

    Parameters
    ----------
    curve_class : str
        DNV curve class: 'B', 'C', 'D', 'F', etc.
    environment : str
        'seawater_cp' for cathodic protection,
        'seawater_free' for free corrosion.
    """

    def __init__(self, curve_class: str = "D", environment: str = "seawater_cp"):
        key = (curve_class.upper(), environment)
        if key not in DNV_CURVES:
            raise ValueError(f"Unknown DNV curve: {key}. Available: {list(DNV_CURVES.keys())}")
        self.params = DNV_CURVES[key]
        self.curve_class = curve_class.upper()
        self.environment = environment

    def n_to_failure(self, stress_range: float) -> float:
        """Calculate cycles to failure for a given stress range.

        Parameters
        ----------
        stress_range : float
            Stress range in MPa.

        Returns
        -------
        float
            Number of cycles to failure.
        """
        if stress_range <= 0:
            return np.inf

        # First segment: N <= N_transition
        # log10(N) = log_K1 - m1 * log10(S)
        log_S = np.log10(stress_range)
        log_N1 = self.params.log_K1 - self.params.m1 * log_S
        n1 = 10 ** log_N1

        if n1 <= self.params.n_transition:
            return n1

        # Second segment: N > N_transition
        # log10(N) = log_K2 - m2 * log10(S)
        log_N2 = self.params.log_K2 - self.params.m2 * log_S
        n2 = 10 ** log_N2
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
            Allowable stress range in MPa.
        """
        if n_cycles <= 0:
            return np.inf

        if n_cycles <= self.params.n_transition:
            # First segment: log10(N) = log_K1 - m1 * log10(S)
            # log10(S) = (log_K1 - log10(N)) / m1
            log_S = (self.params.log_K1 - np.log10(n_cycles)) / self.params.m1
            return 10 ** log_S
        else:
            # Second segment: log10(N) = log_K2 - m2 * log10(S)
            log_S = (self.params.log_K2 - np.log10(n_cycles)) / self.params.m2
            return 10 ** log_S

    def evaluate(self, stress_range: float) -> SNCurveResult:
        """Full evaluation of S-N curve at a stress range.

        Parameters
        ----------
        stress_range : float
            Stress range in MPa.

        Returns
        -------
        SNCurveResult
        """
        return SNCurveResult(
            stress_range=stress_range,
            n_to_failure=self.n_to_failure(stress_range),
            curve_class=self.curve_class,
            environment=self.environment,
        )
