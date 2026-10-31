"""Full fatigue damage chain for ice-loaded ship hull assessment.

Integrates:
1. Ice load time series → rainflow counting
2. Nominal stress → hot-spot stress (Kt)
3. S-N curve evaluation (DNV-RP-C203)
4. Miner cumulative damage
5. Long-term extrapolation (Weibull/Gumbel)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

from .rainflow import rainflow_counting, RainflowResult, rainflow_histogram
from .sn_curve import DNCurve, SNCurveResult
from .miner import miner_damage, MinerResult
from .extrapolation import weibull_fit, rayleigh_fit, gumbel_extreme
from .hotspot import hotspot_stress, HotSpotResult, kt_from_detail


@dataclass
class ChainResult:
    """Complete fatigue damage chain result."""
    # Rainflow results
    rainflow: RainflowResult
    n_bins: int

    # Stress results
    hot_spot_results: list  # List of HotSpotResult
    kt_factor: float

    # S-N results
    sn_curve_class: str
    environment: str

    # Miner damage
    miner: MinerResult
    annual_damage: float
    design_life_years: float
    total_damage: float

    # Extrapolation
    weibull_shape: float
    weibull_scale: float
    ks_pvalue: float
    extreme_20yr: float

    # Inspection recommendation
    recommended_inspection_interval: float  # years
    is_acceptable: bool

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class FatigueDamageChain:
    """Complete fatigue damage assessment chain.

    Parameters
    ----------
    stress_time_series : np.ndarray
        Time series of nominal stress (MPa).
    kt_factor : float
        Stress concentration factor.
    sn_curve_class : str
        DNV S-N curve class ('B', 'C', 'D', 'F').
    environment : str
        Environment: 'seawater_cp' or 'seawater_free'.
    design_life_years : float
        Design life in years (default 20).
    annual_load_events : float
        Number of load events per year for extrapolation.
    n_bins : int
        Number of bins for rainflow histogram.
    """

    def __init__(
        self,
        stress_time_series: np.ndarray,
        kt_factor: float = 1.2,
        sn_curve_class: str = "D",
        environment: str = "seawater_cp",
        design_life_years: float = 20.0,
        annual_load_events: float = 1000.0,
        n_bins: int = 20,
        thickness: Optional[float] = None,
    ):
        self.stress_time_series = np.asarray(stress_time_series)
        self.kt_factor = kt_factor
        self.sn_curve_class = sn_curve_class
        self.environment = environment
        self.design_life_years = design_life_years
        self.annual_load_events = annual_load_events
        self.n_bins = n_bins
        self.thickness = thickness

        self.sn_curve = DNCurve(curve_class=sn_curve_class, environment=environment)

    def run(self) -> ChainResult:
        """Execute the full fatigue damage chain.

        Returns
        -------
        ChainResult
        """
        # Step 1: Rainflow counting
        rf_result = rainflow_counting(self.stress_time_series)

        # Step 2: Hot-spot stress conversion
        hot_spot_results = []
        hot_spot_ranges = []
        for sr, mean in zip(rf_result.ranges, rf_result.means):
            hs = hotspot_stress(
                nominal_stress=sr,
                kt_factor=self.kt_factor,
                thickness=self.thickness,
            )
            hot_spot_results.append(hs)
            hot_spot_ranges.append(hs.hot_spot_stress)

        hot_spot_ranges = np.array(hot_spot_ranges)

        # Step 3: Bin the hot-spot stress ranges
        bin_edges, bin_counts, bin_centers = rainflow_histogram(
            RainflowResult(
                ranges=hot_spot_ranges,
                means=rf_result.means,
                counts=rf_result.counts,
            ),
            n_bins=self.n_bins,
        )

        # Step 4: Miner damage calculation
        # Scale bin counts to annual (assuming the time series represents one year)
        # If the time series is shorter, we need to scale appropriately
        # For now, assume the input represents annual loading
        annual_counts = bin_counts  # cycles per year

        miner = miner_damage(
            stress_ranges=bin_centers,
            n_cycles=annual_counts,
            sn_curve=self.sn_curve,
        )

        annual_damage = miner.total_damage
        total_damage = annual_damage * self.design_life_years

        # Step 5: Weibull/Rayleigh fitting of stress ranges
        if len(hot_spot_ranges) > 0 and np.all(hot_spot_ranges > 0):
            wb_result = weibull_fit(hot_spot_ranges)
            weibull_shape = wb_result.shape
            weibull_scale = wb_result.scale
            ks_pvalue = wb_result.ks_pvalue
        else:
            weibull_shape = 2.0  # default Rayleigh
            weibull_scale = np.mean(hot_spot_ranges) if len(hot_spot_ranges) > 0 else 0.0
            ks_pvalue = 1.0

        # Step 6: Gumbel extreme value extrapolation
        # Generate block maxima from the Weibull fit
        n_blocks = max(int(len(hot_spot_ranges) / 100), 10)
        if len(hot_spot_ranges) >= n_blocks:
            block_size = len(hot_spot_ranges) // n_blocks
            block_maxima = np.array([
                np.max(hot_spot_ranges[i * block_size:(i + 1) * block_size])
                for i in range(n_blocks)
            ])
            if len(block_maxima) >= 3:
                gumbel = gumbel_extreme(
                    block_maxima,
                    return_period=self.design_life_years,
                    events_per_year=self.annual_load_events,
                )
                extreme_20yr = gumbel.extreme_value
            else:
                extreme_20yr = np.max(hot_spot_ranges) * 1.5
        else:
            extreme_20yr = np.max(hot_spot_ranges) * 1.5 if len(hot_spot_ranges) > 0 else 0.0

        # Step 7: Inspection interval recommendation
        # Based on damage accumulation rate
        if annual_damage > 0:
            years_to_failure = 1.0 / annual_damage
            # Recommend inspection at 50% of remaining life or 5 years, whichever is less
            recommended_interval = min(
                years_to_failure * 0.5,
                5.0,
            )
            recommended_interval = max(recommended_interval, 1.0)  # minimum 1 year
        else:
            recommended_interval = 5.0

        is_acceptable = total_damage < 1.0

        return ChainResult(
            rainflow=rf_result,
            n_bins=self.n_bins,
            hot_spot_results=hot_spot_results,
            kt_factor=self.kt_factor,
            sn_curve_class=self.sn_curve_class,
            environment=self.environment,
            miner=miner,
            annual_damage=annual_damage,
            design_life_years=self.design_life_years,
            total_damage=total_damage,
            weibull_shape=weibull_shape,
            weibull_scale=weibull_scale,
            ks_pvalue=ks_pvalue,
            extreme_20yr=extreme_20yr,
            recommended_inspection_interval=recommended_interval,
            is_acceptable=is_acceptable,
            metadata={
                "n_full_cycles": rf_result.n_full,
                "n_half_cycles": rf_result.n_half,
                "total_cycles": rf_result.total_cycles,
                "max_stress_range": float(np.max(self.stress_time_series) - np.min(self.stress_time_series)),
            },
        )
