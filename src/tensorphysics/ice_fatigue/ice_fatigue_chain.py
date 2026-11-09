"""Complete ice-induced structural fatigue assessment chain.

Implements the full pipeline from ice load to fatigue life:

1. Ice load time series → FFT → load PSD S_FF(f)
2. Structural FRF → response PSD S_yy(f)
3. Response PSD → stress time series (via inverse spectral synthesis)
4. Rainflow counting → cycle histogram (Δσ_i, n_i)
5. S-N curve (DNV-OS-C401) → N_i for each Δσ_i
6. Miner cumulative damage: D = Σ n_i / N_i
7. DNV-OS-C401 fatigue check with design fatigue factor (DFL)

References
----------
- DNV-OS-C401 (2021): Fatigue Design of Offshore Steel Structures
- DNV-RP-C203 (2021): Fatigue Design of Offshore Steel Structures (S-N curves)
- ASTM E1049-85: Standard Practices for Cycle Counting in Fatigue Analysis
- ISO 19906 (2019): Petroleum and natural gas industries —
  Fixed steel offshore structures
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np

from .sn_dnv_os_c401 import SNCurveOS401, DNV_OS_C401_CURVES, thickness_correction
from .spectrum import (
    IceLoadSpectrum, StructuralFRF, ResponseSpectrum,
    compute_load_psd, compute_frf, compute_response_psd,
    ice_load_time_series, response_time_series_from_psd,
    spectral_fatigue_assessment,
)

__all__ = [
    "IceFatigueChain",
    "IceFatigueResult",
    "DNVCheckResult",
    "run_ice_fatigue_assessment",
]


@dataclass
class DNVCheckResult:
    """Result of DNV-OS-C401 fatigue check.

    Attributes
    ----------
    design_fatigue_factor : float
        DFL (2-10 depending on inspectability).
    required_life_years : float
        T_req = T_design × DFL.
    computed_life_years : float
        T_life = T_service / D (or 1/damage_rate).
    utilisation : float
        U = T_req / T_life. Must be ≤ 1.0 for acceptance.
    is_acceptable : bool
        True if utilisation ≤ 1.0.
    margin : float
        Safety margin = T_life / T_req - 1.
    """
    design_fatigue_factor: float
    required_life_years: float
    computed_life_years: float
    utilisation: float
    is_acceptable: bool
    margin: float


@dataclass
class IceFatigueResult:
    """Complete ice-induced fatigue assessment result.

    Attributes
    ----------
    load_spectrum : IceLoadSpectrum
        Ice load spectral analysis result.
    frf : StructuralFRF
        Structural frequency response function.
    response_spectrum : ResponseSpectrum
        Structural response spectral analysis result.
    stress_time_series : np.ndarray
        Synthesized stress time history (MPa).
    time_vector : np.ndarray
        Time vector for stress history (s).
    rainflow_ranges : np.ndarray
        Stress ranges from rainflow counting (MPa).
    rainflow_counts : np.ndarray
        Cycle counts from rainflow counting.
    sn_detail : str
        S-N curve detail category used.
    miner_damage : float
        Total Miner cumulative damage D.
    fatigue_life_years : float
        Computed fatigue life (years).
    service_period_years : float
        Service period for damage calculation (years).
    dnv_check : DNVCheckResult
        DNV-OS-C401 fatigue check result.
    spectral_damage : float
        Narrow-band spectral fatigue damage (for comparison).
    metadata : dict
        Additional metadata.
    """
    load_spectrum: IceLoadSpectrum
    frf: StructuralFRF
    response_spectrum: ResponseSpectrum
    stress_time_series: np.ndarray
    time_vector: np.ndarray
    rainflow_ranges: np.ndarray
    rainflow_counts: np.ndarray
    sn_detail: str
    miner_damage: float
    fatigue_life_years: float
    service_period_years: float
    dnv_check: DNVCheckResult
    spectral_damage: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class IceFatigueChain:
    """Complete ice-induced structural fatigue assessment chain.

    Parameters
    ----------
    ice_load_time : np.ndarray
        Time vector for ice load (s).
    ice_load_force : np.ndarray
        Ice force time series (N).
    structural_mass_kg : float
        Modal mass of structure (kg).
    structural_stiffness_n_m : float
        Modal stiffness (N/m).
    structural_damping_ratio : float
        Damping ratio ζ (typical 0.01-0.05 for ships).
    stress_conversion_factor : float
        Factor to convert displacement to stress (MPa/m).
        For a beam: σ = E·y·κ where κ is curvature.
    sn_detail : str
        DNV-OS-C401 S-N curve detail: 'B1', 'F', or 'W3'.
    plate_thickness_mm : float, optional
        Plate thickness for thickness correction (mm).
    design_life_years : float
        Design fatigue life (years). Default 20.
    design_fatigue_factor : float
        DFL per DNV-OS-C401 §4.5 (2-10). Default 3.0.
    hotspot_kt : float
        Hot-spot stress concentration factor. Default 1.0.
    service_period_years : float
        Service period for damage scaling (years). Default 1.0.
    """

    def __init__(
        self,
        ice_load_time: np.ndarray,
        ice_load_force: np.ndarray,
        structural_mass_kg: float,
        structural_stiffness_n_m: float,
        structural_damping_ratio: float = 0.02,
        stress_conversion_factor: float = 1.0e-6,
        sn_detail: str = "F",
        plate_thickness_mm: Optional[float] = None,
        design_life_years: float = 20.0,
        design_fatigue_factor: float = 3.0,
        hotspot_kt: float = 1.0,
        service_period_years: float = 1.0,
    ):
        self.ice_load_time = np.asarray(ice_load_time)
        self.ice_load_force = np.asarray(ice_load_force)
        self.mass = structural_mass_kg
        self.stiffness = structural_stiffness_n_m
        self.damping_ratio = structural_damping_ratio
        self.stress_factor = stress_conversion_factor
        self.sn_detail = sn_detail
        self.thickness_mm = plate_thickness_mm
        self.design_life = design_life_years
        self.dfl = design_fatigue_factor
        self.kt = hotspot_kt
        self.service_years = service_period_years

        # Derived
        self.damping_coeff = 2.0 * structural_damping_ratio * np.sqrt(
            structural_stiffness_n_m * structural_mass_kg
        )

    def run(self) -> IceFatigueResult:
        """Execute the complete ice fatigue assessment chain.

        Returns
        -------
        IceFatigueResult
        """
        # Import rainflow from existing fatigue module
        from tensorphysics.fatigue.rainflow import rainflow_counting_stack

        # Step 1: Ice load spectrum analysis
        load_spectrum = compute_load_psd(
            self.ice_load_time, self.ice_load_force
        )

        # Step 2: Structural FRF
        frf = compute_frf(
            load_spectrum.frequencies,
            self.mass,
            self.stiffness,
            self.damping_coeff,
        )

        # Step 3: Response spectrum
        response_spectrum = compute_response_psd(load_spectrum, frf)

        # Step 4: Synthesize stress time series from response PSD
        duration = float(self.ice_load_time[-1] - self.ice_load_time[0])
        fs = 1.0 / np.mean(np.diff(self.ice_load_time))

        time_resp, disp_resp = response_time_series_from_psd(
            response_spectrum, duration_s=duration, fs_hz=fs
        )

        # Convert displacement to stress (MPa)
        stress_mpa = disp_resp * self.stress_factor * self.kt

        # Step 5: Rainflow counting
        rf_result = rainflow_counting_stack(stress_mpa)

        # Step 6: S-N curve evaluation and Miner damage
        sn_curve = SNCurveOS401(
            detail=self.sn_detail,
            thickness_mm=self.thickness_mm,
        )

        # Compute Miner damage
        if len(rf_result.ranges) > 0:
            n_f = sn_curve.evaluate_range(rf_result.ranges)
            # Avoid division by zero
            valid = n_f < np.inf
            if np.any(valid):
                damage_per_cycle = np.zeros_like(rf_result.counts)
                damage_per_cycle[valid] = rf_result.counts[valid] / n_f[valid]
                miner_damage = float(np.sum(damage_per_cycle))
            else:
                miner_damage = 0.0
        else:
            miner_damage = 0.0

        # Scale damage to service period
        # The time series represents `duration` seconds of loading
        # Scale to annual damage, then to service period
        seconds_per_year = 365.25 * 24 * 3600
        if duration > 0:
            annual_damage = miner_damage * seconds_per_year / duration
        else:
            annual_damage = 0.0

        total_damage = annual_damage * self.service_years

        # Fatigue life
        if annual_damage > 0:
            fatigue_life_years = 1.0 / annual_damage
        else:
            fatigue_life_years = np.inf

        # Step 7: Spectral fatigue (narrow-band, for comparison)
        spectral_result = spectral_fatigue_assessment(
            response_spectrum,
            sn_curve_m=sn_curve.m,
            sn_curve_log_a=sn_curve.log_a,
            duration_s=duration,
        )
        spectral_annual = spectral_result["damage_rate"] * seconds_per_year
        spectral_total = spectral_annual * self.service_years

        # Step 8: DNV-OS-C401 check
        required_life = self.design_life * self.dfl
        if fatigue_life_years > 0 and fatigue_life_years < np.inf:
            utilisation = required_life / fatigue_life_years
        else:
            utilisation = 0.0

        dnv_check = DNVCheckResult(
            design_fatigue_factor=self.dfl,
            required_life_years=required_life,
            computed_life_years=fatigue_life_years,
            utilisation=utilisation,
            is_acceptable=utilisation <= 1.0,
            margin=fatigue_life_years / required_life - 1.0 if required_life > 0 else np.inf,
        )

        return IceFatigueResult(
            load_spectrum=load_spectrum,
            frf=frf,
            response_spectrum=response_spectrum,
            stress_time_series=stress_mpa,
            time_vector=time_resp,
            rainflow_ranges=rf_result.ranges,
            rainflow_counts=rf_result.counts,
            sn_detail=self.sn_detail,
            miner_damage=total_damage,
            fatigue_life_years=fatigue_life_years,
            service_period_years=self.service_years,
            dnv_check=dnv_check,
            spectral_damage=spectral_total,
            metadata={
                "duration_s": duration,
                "sampling_hz": fs,
                "n_rainflow_cycles": rf_result.total_cycles,
                "n_full_cycles": rf_result.n_full,
                "n_half_cycles": rf_result.n_half,
                "peak_load_freq_hz": load_spectrum.peak_frequency,
                "peak_response_freq_hz": response_spectrum.peak_frequency,
                "natural_freq_hz": frf.resonance_freq,
                "bandwidth_alpha2": response_spectrum.bandwidth,
                "thickness_factor": sn_curve.thickness_factor,
                "hotspot_kt": self.kt,
                "annual_damage": annual_damage,
            },
        )


def run_ice_fatigue_assessment(
    duration_s: float = 600.0,
    fs_hz: float = 10.0,
    mean_ice_force_n: float = 3.0e6,
    dynamic_amplitude_n: float = 1.5e6,
    primary_freq_hz: float = 0.5,
    structural_mass_kg: float = 5.0e6,
    natural_freq_hz: float = 1.2,
    damping_ratio: float = 0.02,
    stress_factor: float = 1.0e-6,
    sn_detail: str = "F",
    plate_thickness_mm: Optional[float] = None,
    design_life_years: float = 20.0,
    dfl: float = 3.0,
    hotspot_kt: float = 1.0,
    service_years: float = 1.0,
    seed: int = 20261109,
) -> IceFatigueResult:
    """Convenience function to run a complete ice fatigue assessment.

    Generates synthetic ice load, performs spectral analysis,
    rainflow counting, Miner damage, and DNV-OS-C401 check.

    Parameters
    ----------
    duration_s : float
        Duration of ice load record (s).
    fs_hz : float
        Sampling frequency (Hz).
    mean_ice_force_n : float
        Mean ice force (N).
    dynamic_amplitude_n : float
        Dynamic amplitude of ice force (N).
    primary_freq_hz : float
        Primary ice excitation frequency (Hz).
    structural_mass_kg : float
        Modal mass (kg).
    natural_freq_hz : float
        Structural natural frequency (Hz).
    damping_ratio : float
        Damping ratio.
    stress_factor : float
        Displacement-to-stress conversion (MPa/m).
    sn_detail : str
        S-N curve detail ('B1', 'F', 'W3').
    plate_thickness_mm : float, optional
        Plate thickness (mm).
    design_life_years : float
        Design life (years).
    dfl : float
        Design fatigue factor.
    hotspot_kt : float
        Hot-spot stress concentration factor.
    service_years : float
        Service period (years).
    seed : int
        Random seed.

    Returns
    -------
    IceFatigueResult
    """
    # Generate ice load
    time_ice, force_ice = ice_load_time_series(
        duration_s=duration_s,
        fs_hz=fs_hz,
        mean_force_n=mean_ice_force_n,
        amplitude_n=dynamic_amplitude_n,
        primary_freq_hz=primary_freq_hz,
        seed=seed,
    )

    # Structural properties
    omega_n = 2.0 * np.pi * natural_freq_hz
    stiffness = structural_mass_kg * omega_n**2

    # Run chain
    chain = IceFatigueChain(
        ice_load_time=time_ice,
        ice_load_force=force_ice,
        structural_mass_kg=structural_mass_kg,
        structural_stiffness_n_m=stiffness,
        structural_damping_ratio=damping_ratio,
        stress_conversion_factor=stress_factor,
        sn_detail=sn_detail,
        plate_thickness_mm=plate_thickness_mm,
        design_life_years=design_life_years,
        design_fatigue_factor=dfl,
        hotspot_kt=hotspot_kt,
        service_period_years=service_years,
    )

    return chain.run()
