"""Complete ice-induced fatigue life assessment chain.

Integrates all components for ice zone structural fatigue assessment:

1. Ice load spectrum → stress time history (via structural response)
2. Hot-spot stress at plate-beam weld toe (Peterson K_t)
3. Rainflow counting → stress range histogram
4. S-N curve (DNV-RP-C203 D curve, seawater, -20°C correction)
5. Miner cumulative damage → fatigue life
6. Validation against published examples

References
----------
- DNV-RP-C203 (2021): Fatigue Design of Offshore Steel Structures
- ISO 19906 (2019): Petroleum and natural gas industries — Arctic structures
- DNV-GL (2015): Recommended Practice DNVGL-RP-0001
- Su, B. et al. (2010): Ice-induced fatigue of offshore structures
- Liu, L. et al. (2015): Fatigue assessment of ice-going vessels
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

from .sn_dnv_rp_c203 import SNCurveRP203, DNV_RP_C203_CURVES
from .hotspot_stress import (
    hot_spot_stress_from_nominal,
    plate_beam_kt_peterson,
    fem_submodel_convergence_study,
)
from .miner_accumulation import (
    miner_cumulative_damage,
    fatigue_life_from_damage,
    MinerResult,
)
from .ice_load_spectrum import (
    generate_ice_load_time_series,
    rainflow_counting,
    stress_range_histogram,
    IceLoadSpectrum,
    RainflowResult,
)

__all__ = [
    "IceFatigueAssessment",
    "IceFatigueResult",
    "run_ice_fatigue_assessment",
    "validate_against_reference",
]


@dataclass
class IceFatigueResult:
    """Complete ice-induced fatigue assessment result.
    
    Attributes
    ----------
    sn_detail : str
        S-N curve detail category used.
    m_sn : float
        S-N curve slope (first segment).
    log_a_sn : float
        S-N curve intercept (first segment, corrected).
    temperature_c : float
        Operating temperature (°C).
    temp_correction_factor : float
        Low-temperature correction factor.
    kt_geometric : float
        Geometric stress concentration factor.
    kt_total : float
        Total K_t (geometric × weld profile).
    hot_spot_stress_range_mpa : float
        Representative hot-spot stress range (MPa).
    n_rainflow_cycles : int
        Total rainflow cycles counted.
    miner_damage : float
        Total Miner cumulative damage D.
    fatigue_life_years : float
        Estimated fatigue life (years).
    is_acceptable : bool
        True if fatigue life > design life.
    design_life_years : float
        Required design life (years).
    safety_factor : float
        Fatigue life / design life.
    metadata : dict
        Additional assessment details.
    """
    sn_detail: str
    m_sn: float
    log_a_sn: float
    temperature_c: float
    temp_correction_factor: float
    kt_geometric: float
    kt_total: float
    hot_spot_stress_range_mpa: float
    n_rainflow_cycles: int
    miner_damage: float
    fatigue_life_years: float
    is_acceptable: bool
    design_life_years: float
    safety_factor: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IceFatigueAssessment:
    """Ice-induced fatigue assessment configuration and executor.
    
    Parameters
    ----------
    sn_detail : str
        S-N curve detail ('B1', 'D', 'E', 'F', 'W3').
    plate_thickness_mm : float
        Plate thickness (mm).
    beam_spacing_mm : float
        Stiffener spacing (mm).
    weld_toe_radius_mm : float
        Weld toe radius (mm). Default 0.5.
    joint_type : str
        Joint type for K_t ('stiffener', 'transverse', 'bracket').
    temperature_c : float
        Operating temperature (°C). Default -20.
    design_life_years : float
        Required design life (years). Default 25.
    with_cathodic_protection : bool
        Cathodic protection provided. Default True.
    ultimate_strength_mpa : float
        Material ultimate strength (MPa). Default 450.
    """
    sn_detail: str = "D"
    plate_thickness_mm: float = 20.0
    beam_spacing_mm: float = 600.0
    weld_toe_radius_mm: float = 0.5
    joint_type: str = "stiffener"
    temperature_c: float = -20.0
    design_life_years: float = 25.0
    with_cathodic_protection: bool = True
    ultimate_strength_mpa: float = 450.0
    
    def run(
        self,
        ice_load: Optional[IceLoadSpectrum] = None,
        stress_conversion_factor: float = 1.0e-7,
        duration_s: float = 600.0,
        sampling_hz: float = 10.0,
        mean_force_n: float = 3.0e6,
        dynamic_amplitude_n: float = 1.5e6,
        primary_freq_hz: float = 0.5,
        seed: int = 20261110,
        annual_exposure_hours: float = 2000.0,
    ) -> IceFatigueResult:
        """Run complete fatigue assessment.
        
        Parameters
        ----------
        ice_load : IceLoadSpectrum, optional
            Pre-computed ice load. If None, generates synthetic load.
        stress_conversion_factor : float
            Force to stress conversion (MPa/N).
        duration_s : float
            Load record duration (s).
        sampling_hz : float
            Sampling frequency (Hz).
        mean_force_n : float
            Mean ice force (N).
        dynamic_amplitude_n : float
            Dynamic amplitude (N).
        primary_freq_hz : float
            Primary excitation frequency (Hz).
        seed : int
            Random seed.
        annual_exposure_hours : float
            Annual hours of ice exposure.
        
        Returns
        -------
        IceFatigueResult
            Complete assessment result.
        """
        # Step 1: Generate or use ice load
        if ice_load is None:
            ice_load = generate_ice_load_time_series(
                duration_s=duration_s,
                sampling_hz=sampling_hz,
                mean_force_n=mean_force_n,
                dynamic_amplitude_n=dynamic_amplitude_n,
                primary_freq_hz=primary_freq_hz,
                seed=seed,
            )
        
        # Step 2: Convert force to nominal stress
        nominal_stress = ice_load.force_n * stress_conversion_factor
        
        # Step 3: Hot-spot stress at weld toe
        hs_result = hot_spot_stress_from_nominal(
            nominal_stress_mpa=float(np.std(nominal_stress)),
            plate_thickness_mm=self.plate_thickness_mm,
            beam_spacing_mm=self.beam_spacing_mm,
            weld_toe_radius_mm=self.weld_toe_radius_mm,
            joint_type=self.joint_type,
            ultimate_strength_mpa=self.ultimate_strength_mpa,
        )
        
        # Apply K_t to full time series for rainflow
        hot_spot_stress_ts = nominal_stress * hs_result.kt_total * hs_result.thickness_factor
        
        # Step 4: Rainflow counting
        rf_result = rainflow_counting(hot_spot_stress_ts)
        
        # Step 5: Create stress range histogram
        bin_centers, bin_counts = stress_range_histogram(rf_result, n_bins=30)
        
        # Step 6: S-N curve
        sn_curve = SNCurveRP203(
            detail=self.sn_detail,
            thickness_mm=self.plate_thickness_mm,
            temperature_c=self.temperature_c,
            with_cathodic_protection=self.with_cathodic_protection,
        )
        
        # Step 7: Miner damage
        miner_result = miner_cumulative_damage(
            stress_ranges_mpa=bin_centers,
            cycle_counts=bin_counts,
            sn_curve=sn_curve,
        )
        
        # Step 8: Scale to annual damage
        record_duration_years = duration_s / (365.25 * 24 * 3600)
        if record_duration_years > 0:
            annual_damage = miner_result.total_damage / record_duration_years
        else:
            annual_damage = 0.0
        
        # Account for annual exposure (not continuous)
        exposure_fraction = annual_exposure_hours / (365.25 * 24)
        effective_annual_damage = annual_damage * exposure_fraction
        
        # Step 9: Fatigue life
        fatigue_life_years = fatigue_life_from_damage(effective_annual_damage, 1.0)
        
        # Step 10: Acceptance check
        safety_factor = fatigue_life_years / self.design_life_years if self.design_life_years > 0 else np.inf
        is_acceptable = safety_factor >= 1.0
        
        return IceFatigueResult(
            sn_detail=self.sn_detail,
            m_sn=sn_curve.m1,
            log_a_sn=sn_curve.log_a1,
            temperature_c=self.temperature_c,
            temp_correction_factor=sn_curve.temperature_factor,
            kt_geometric=hs_result.kt_geometric,
            kt_total=hs_result.kt_total,
            hot_spot_stress_range_mpa=float(np.std(hot_spot_stress_ts)),
            n_rainflow_cycles=rf_result.n_full + rf_result.n_half,
            miner_damage=miner_result.total_damage,
            fatigue_life_years=fatigue_life_years,
            is_acceptable=is_acceptable,
            design_life_years=self.design_life_years,
            safety_factor=safety_factor,
            metadata={
                "plate_thickness_mm": self.plate_thickness_mm,
                "beam_spacing_mm": self.beam_spacing_mm,
                "weld_toe_radius_mm": self.weld_toe_radius_mm,
                "joint_type": self.joint_type,
                "thickness_factor": hs_result.thickness_factor,
                "env_factor": sn_curve.environment_factor,
                "n_full_cycles": rf_result.n_full,
                "n_half_cycles": rf_result.n_half,
                "record_duration_s": duration_s,
                "annual_exposure_hours": annual_exposure_hours,
                "effective_annual_damage": effective_annual_damage,
                "mean_ice_force_n": ice_load.mean_force_n,
                "peak_freq_hz": ice_load.peak_frequency_hz,
            },
        )


def run_ice_fatigue_assessment(
    sn_detail: str = "D",
    plate_thickness_mm: float = 20.0,
    beam_spacing_mm: float = 600.0,
    temperature_c: float = -20.0,
    design_life_years: float = 25.0,
    **kwargs,
) -> IceFatigueResult:
    """Convenience function to run ice fatigue assessment.
    
    Parameters
    ----------
    sn_detail : str
        S-N curve detail.
    plate_thickness_mm : float
        Plate thickness (mm).
    beam_spacing_mm : float
        Stiffener spacing (mm).
    temperature_c : float
        Operating temperature (°C).
    design_life_years : float
        Design life (years).
    **kwargs
        Additional parameters passed to IceFatigueAssessment.run().
    
    Returns
    -------
    IceFatigueResult
    """
    assessment = IceFatigueAssessment(
        sn_detail=sn_detail,
        plate_thickness_mm=plate_thickness_mm,
        beam_spacing_mm=beam_spacing_mm,
        temperature_c=temperature_c,
        design_life_years=design_life_years,
    )
    return assessment.run(**kwargs)


def validate_against_reference(
    reference_life_years: float,
    computed_life_years: float,
    tolerance: float = 0.30,
) -> Dict[str, Any]:
    """Validate computed fatigue life against published reference.
    
    Parameters
    ----------
    reference_life_years : float
        Reference fatigue life from published example (years).
    computed_life_years : float
        Computed fatigue life (years).
    tolerance : float
        Acceptable relative error (default 30%).
    
    Returns
    -------
    dict
        Validation result with error metrics.
    """
    if reference_life_years <= 0:
        raise ValueError("Reference life must be positive")
    
    relative_error = abs(computed_life_years - reference_life_years) / reference_life_years
    
    return {
        "reference_life_years": reference_life_years,
        "computed_life_years": computed_life_years,
        "relative_error": relative_error,
        "tolerance": tolerance,
        "is_valid": relative_error <= tolerance,
        "error_pct": relative_error * 100.0,
    }
