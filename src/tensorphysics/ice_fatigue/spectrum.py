"""Ice-induced vibration spectral analysis.

Implements the frequency-domain chain from ice load to structural response:

1. Ice load time series F_ice(t) → FFT → load PSD S_FF(f)
2. Structural frequency response function H(f) for SDOF:
   H(f) = 1 / (k - m·ω² + i·c·ω)
3. Response PSD: S_yy(f) = |H(f)|² · S_FF(f)
4. Response time series via inverse FFT (or modal superposition)
5. Spectral moments and characteristic frequencies

References
----------
- DNV-OS-C401 §3: Dynamic analysis
- ISO 19906 §7: Ice actions on structures
- Metrikine & Vrouwenvelder (2001): Out-of-plane vibration of beam under moving load
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from scipy.signal import welch

__all__ = [
    "IceLoadSpectrum",
    "StructuralFRF",
    "ResponseSpectrum",
    "compute_load_psd",
    "compute_frf",
    "compute_response_psd",
    "spectral_moments",
    "ice_load_time_series",
    "response_time_series_from_psd",
    "spectral_fatigue_assessment",
]


@dataclass
class IceLoadSpectrum:
    """Result of ice load spectral analysis.

    Attributes
    ----------
    frequencies : np.ndarray
        Frequency vector (Hz).
    psd : np.ndarray
        Power spectral density of ice load (N²/Hz).
    peak_frequency : float
        Frequency of maximum PSD (Hz).
    m0 : float
        Zeroth spectral moment (variance, N²).
    rms_force : float
        RMS ice force (N).
    """
    frequencies: np.ndarray
    psd: np.ndarray
    peak_frequency: float
    m0: float
    rms_force: float


@dataclass
class StructuralFRF:
    """Structural frequency response function.

    Attributes
    ----------
    frequencies : np.ndarray
        Frequency vector (Hz).
    h_real : np.ndarray
        Real part of FRF (m/N).
    h_imag : np.ndarray
        Imaginary part of FRF (m/N).
    h_mag : np.ndarray
        Magnitude of FRF (m/N).
    resonance_freq : float
        Natural frequency (Hz).
    peak_magnification : float
        Peak magnification factor (1/(2ζ)).
    """
    frequencies: np.ndarray
    h_real: np.ndarray
    h_imag: np.ndarray
    h_mag: np.ndarray
    resonance_freq: float
    peak_magnification: float


@dataclass
class ResponseSpectrum:
    """Result of structural response spectral analysis.

    Attributes
    ----------
    frequencies : np.ndarray
        Frequency vector (Hz).
    response_psd : np.ndarray
        Response PSD (m²/Hz or MPa²/Hz).
    peak_frequency : float
        Frequency of maximum response PSD (Hz).
    m0 : float
        Zeroth spectral moment.
    m1 : float
        First spectral moment.
    m2 : float
        Second spectral moment.
    m4 : float
        Fourth spectral moment.
    rms_response : float
        RMS response (m or MPa).
    bandwidth : float
        Bandwidth parameter α₂ = m2²/(m0·m4).
    """
    frequencies: np.ndarray
    response_psd: np.ndarray
    peak_frequency: float
    m0: float
    m1: float
    m2: float
    m4: float
    rms_response: float
    bandwidth: float


def compute_load_psd(
    time: np.ndarray,
    force: np.ndarray,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
) -> IceLoadSpectrum:
    """Compute power spectral density of ice load time series.

    Uses Welch's method for spectral estimation.

    Parameters
    ----------
    time : np.ndarray
        Time vector (s).
    force : np.ndarray
        Ice force time series (N).
    nperseg : int, optional
        Segment length for Welch method. Default = len(force)//8.
    noverlap : int, optional
        Overlap length. Default = nperseg//2.

    Returns
    -------
    IceLoadSpectrum
    """
    dt = np.mean(np.diff(time))
    fs = 1.0 / dt

    if nperseg is None:
        nperseg = min(len(force) // 4, 256)
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = welch(force, fs=fs, nperseg=nperseg, noverlap=noverlap,
                       window='hann', scaling='density')

    # Spectral moments
    m0 = float(np.trapezoid(psd, freqs))
    peak_idx = np.argmax(psd)
    peak_freq = float(freqs[peak_idx])

    return IceLoadSpectrum(
        frequencies=freqs,
        psd=psd,
        peak_frequency=peak_freq,
        m0=m0,
        rms_force=np.sqrt(m0),
    )


def compute_frf(
    frequencies: np.ndarray,
    mass_kg: float,
    stiffness_n_m: float,
    damping_n_s_m: float,
) -> StructuralFRF:
    """Compute structural frequency response function for SDOF system.

    H(f) = 1 / (k - m·ω² + i·c·ω)

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency vector (Hz).
    mass_kg : float
        Modal mass (kg).
    stiffness_n_m : float
        Modal stiffness (N/m).
    damping_n_s_m : float
        Modal damping coefficient (N·s/m).

    Returns
    -------
    StructuralFRF
    """
    omega = 2.0 * np.pi * frequencies

    # H(f) = 1 / (k - m*omega^2 + i*c*omega)
    denominator = stiffness_n_m - mass_kg * omega**2 + 1j * damping_n_s_m * omega
    h_complex = 1.0 / denominator

    # Natural frequency and damping ratio
    fn = np.sqrt(stiffness_n_m / mass_kg) / (2.0 * np.pi)
    zeta = damping_n_s_m / (2.0 * np.sqrt(stiffness_n_m * mass_kg))
    peak_mag = 1.0 / (2.0 * zeta * stiffness_n_m) if zeta > 0 else np.inf

    return StructuralFRF(
        frequencies=frequencies,
        h_real=h_complex.real,
        h_imag=h_complex.imag,
        h_mag=np.abs(h_complex),
        resonance_freq=float(fn),
        peak_magnification=float(1.0 / (2.0 * zeta)) if zeta > 0 else np.inf,
    )


def compute_response_psd(
    load_psd: IceLoadSpectrum,
    frf: StructuralFRF,
) -> ResponseSpectrum:
    """Compute structural response PSD.

    S_yy(f) = |H(f)|² · S_FF(f)

    Parameters
    ----------
    load_psd : IceLoadSpectrum
        Ice load PSD.
    frf : StructuralFRF
        Structural FRF.

    Returns
    -------
    ResponseSpectrum
    """
    # Interpolate FRF to load PSD frequencies if needed
    if len(load_psd.frequencies) != len(frf.frequencies):
        h_mag_interp = np.interp(load_psd.frequencies, frf.frequencies, frf.h_mag)
        freqs = load_psd.frequencies
    else:
        h_mag_interp = frf.h_mag
        freqs = load_psd.frequencies

    # Response PSD
    response_psd = h_mag_interp**2 * load_psd.psd

    # Spectral moments
    m0 = float(np.trapezoid(response_psd, freqs))
    m1 = float(np.trapezoid(response_psd * freqs, freqs))
    m2 = float(np.trapezoid(response_psd * freqs**2, freqs))
    m4 = float(np.trapezoid(response_psd * freqs**4, freqs))

    # Peak frequency
    peak_idx = np.argmax(response_psd)
    peak_freq = float(freqs[peak_idx])

    # Bandwidth parameter
    if m0 > 0 and m4 > 0:
        alpha2 = m2**2 / (m0 * m4)
    else:
        alpha2 = 1.0

    return ResponseSpectrum(
        frequencies=freqs,
        response_psd=response_psd,
        peak_frequency=peak_freq,
        m0=m0,
        m1=m1,
        m2=m2,
        m4=m4,
        rms_response=np.sqrt(m0),
        bandwidth=alpha2,
    )


def spectral_moments(
    frequencies: np.ndarray,
    psd: np.ndarray,
    orders: Tuple[int, ...] = (0, 1, 2, 4),
) -> dict:
    """Compute spectral moments of a PSD.

    m_n = ∫ f^n · S(f) df

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency vector (Hz).
    psd : np.ndarray
        Power spectral density.
    orders : tuple of int
        Moment orders to compute.

    Returns
    -------
    dict
        {order: moment_value}
    """
    result = {}
    for n in orders:
        result[n] = float(np.trapezoid(psd * frequencies**n, frequencies))
    return result


def ice_load_time_series(
    duration_s: float = 600.0,
    fs_hz: float = 10.0,
    mean_force_n: float = 3.0e6,
    amplitude_n: float = 1.5e6,
    primary_freq_hz: float = 0.5,
    secondary_freq_hz: float = 1.2,
    secondary_amplitude_ratio: float = 0.3,
    noise_ratio: float = 0.1,
    seed: int = 20261109,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic ice load time series.

    Models quasi-periodic ice crushing with:
    - Primary frequency from ice-structure interaction
    - Secondary frequency from flexural cracking
    - Broadband noise component

    Parameters
    ----------
    duration_s : float
        Duration of time series (s). Default 600s (10 min).
    fs_hz : float
        Sampling frequency (Hz).
    mean_force_n : float
        Mean ice force (N).
    amplitude_n : float
        Peak dynamic amplitude (N).
    primary_freq_hz : float
        Primary excitation frequency (Hz).
    secondary_freq_hz : float
        Secondary excitation frequency (Hz).
    secondary_amplitude_ratio : float
        Ratio of secondary to primary amplitude.
    noise_ratio : float
        Ratio of noise amplitude to primary amplitude.
    seed : int
        Random seed.

    Returns
    -------
    time : np.ndarray
        Time vector (s).
    force : np.ndarray
        Ice force time series (N).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * fs_hz)
    time = np.arange(n_samples) / fs_hz

    # Primary harmonic (ice crushing frequency)
    f1 = amplitude_n * np.sin(2 * np.pi * primary_freq_hz * time)

    # Secondary harmonic (flexural cracking)
    f2 = amplitude_n * secondary_amplitude_ratio * np.sin(
        2 * np.pi * secondary_freq_hz * time + rng.uniform(0, 2 * np.pi)
    )

    # Broadband noise
    f_noise = amplitude_n * noise_ratio * rng.standard_normal(n_samples)

    # Combine
    force = mean_force_n + f1 + f2 + f_noise

    return time, force


def response_time_series_from_psd(
    response_spectrum: ResponseSpectrum,
    duration_s: float = 600.0,
    fs_hz: float = 10.0,
    seed: int = 20261109,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize Gaussian response time series from response PSD.

    Uses the spectral representation method (Shinozuka).

    Parameters
    ----------
    response_spectrum : ResponseSpectrum
        Response PSD.
    duration_s : float
        Duration (s).
    fs_hz : float
        Sampling frequency (Hz).
    seed : int
        Random seed.

    Returns
    -------
    time : np.ndarray
        Time vector (s).
    response : np.ndarray
        Response time series.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * fs_hz)
    time = np.arange(n_samples) / fs_hz

    freqs = response_spectrum.frequencies
    psd = response_spectrum.response_psd

    # Frequency resolution
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    # Amplitude spectrum
    amplitude = np.sqrt(2.0 * psd * df)

    # Random phases
    phases = rng.uniform(0, 2 * np.pi, len(freqs))

    # Synthesize time series
    response = np.zeros(n_samples)
    for i, (f, a, phi) in enumerate(zip(freqs, amplitude, phases)):
        if f > 0 and a > 0:
            response += a * np.cos(2 * np.pi * f * time + phi)

    # Scale to match target RMS
    target_rms = response_spectrum.rms_response
    actual_rms = np.std(response)
    if actual_rms > 0:
        response *= target_rms / actual_rms

    return time, response


def spectral_fatigue_assessment(
    response_spectrum: ResponseSpectrum,
    sn_curve_m: float,
    sn_curve_log_a: float,
    duration_s: float = 600.0,
) -> dict:
    """Narrow-band spectral fatigue assessment.

    Uses the Rayleigh distribution of peaks for narrow-band processes.
    Expected damage rate:
        E[D] = ν₀ · E[ΔS^m] / K
    where ν₀ = √(m2/m0) is the mean zero-crossing rate,
    and E[ΔS^m] = (√2)^m · Γ(1+m/2) · m0^(m/2) for Rayleigh.

    Parameters
    ----------
    response_spectrum : ResponseSpectrum
        Response PSD.
    sn_curve_m : float
        S-N curve slope.
    sn_curve_log_a : float
        S-N curve intercept log10(a).
    duration_s : float
        Duration of loading (s).

    Returns
    -------
    dict
        Fatigue assessment results.
    """
    from scipy.special import gamma as gamma_func

    m0 = response_spectrum.m0
    m2 = response_spectrum.m2

    if m0 <= 0 or m2 <= 0:
        return {
            "damage_rate": 0.0,
            "total_damage": 0.0,
            "fatigue_life_s": np.inf,
            "mean_zero_crossing_rate": 0.0,
        }

    # Mean zero-crossing rate
    nu_0 = np.sqrt(m2 / m0)

    # Expected range^m for Rayleigh distribution
    # E[S^m] = (2*m0)^(m/2) * Γ(1 + m/2)
    e_sm = (2.0 * m0) ** (sn_curve_m / 2.0) * gamma_func(1.0 + sn_curve_m / 2.0)

    # Damage rate
    K = 10.0 ** sn_curve_log_a
    damage_rate = nu_0 * e_sm / K

    # Total damage over duration
    total_damage = damage_rate * duration_s

    # Fatigue life
    if damage_rate > 0:
        fatigue_life_s = 1.0 / damage_rate
    else:
        fatigue_life_s = np.inf

    return {
        "damage_rate": float(damage_rate),
        "total_damage": float(total_damage),
        "fatigue_life_s": float(fatigue_life_s),
        "mean_zero_crossing_rate": float(nu_0),
        "spectral_m0": m0,
        "spectral_m2": m2,
        "bandwidth_alpha2": response_spectrum.bandwidth,
    }
