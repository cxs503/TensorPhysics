"""Ice load spectrum generation and rainflow counting for fatigue assessment.

Generates synthetic ice-induced load time histories based on DEM contact
force patterns, then extracts stress amplitude spectra via rainflow counting.

Key features:
- Ice load time series from crushing/breaking patterns
- Rainflow counting (ASTM E1049-85 four-point method)
- Stress range histogram generation
- Spectral moments and bandwidth parameters

References
----------
- ASTM E1049-85 (2017): Standard Practices for Cycle Counting in Fatigue Analysis
- ISO 19906 (2019) Annex C: Ice load models
- Suyuthi, A. et al. (2013): Ice-induced vibrations in ice-structure interaction
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

__all__ = [
    "IceLoadSpectrum",
    "RainflowResult",
    "generate_ice_load_time_series",
    "rainflow_counting",
    "stress_range_histogram",
    "spectral_moments",
]


@dataclass
class IceLoadSpectrum:
    """Ice load spectral characterization.
    
    Attributes
    ----------
    time_s : np.ndarray
        Time vector (s).
    force_n : np.ndarray
        Ice force time history (N).
    mean_force_n : float
        Mean ice force (N).
    std_force_n : float
        Standard deviation of ice force (N).
    peak_frequency_hz : float
        Dominant frequency from FFT (Hz).
    duration_s : float
        Record duration (s).
    sampling_hz : float
        Sampling frequency (Hz).
    """
    time_s: np.ndarray
    force_n: np.ndarray
    mean_force_n: float
    std_force_n: float
    peak_frequency_hz: float
    duration_s: float
    sampling_hz: float


@dataclass
class RainflowResult:
    """Result of rainflow cycle counting.
    
    Attributes
    ----------
    ranges_mpa : np.ndarray
        Stress ranges (MPa).
    means_mpa : np.ndarray
        Mean stresses (MPa).
    counts : np.ndarray
        Cycle counts (0.5 for half-cycles, 1.0 for full).
    n_full : int
        Number of full cycles.
    n_half : int
        Number of half-cycles.
    total_cycles : float
        Total equivalent full cycles.
    """
    ranges_mpa: np.ndarray
    means_mpa: np.ndarray
    counts: np.ndarray
    n_full: int = 0
    n_half: int = 0
    
    @property
    def total_cycles(self) -> float:
        return float(np.sum(self.counts))


def generate_ice_load_time_series(
    duration_s: float = 600.0,
    sampling_hz: float = 10.0,
    mean_force_n: float = 3.0e6,
    dynamic_amplitude_n: float = 1.5e6,
    primary_freq_hz: float = 0.5,
    crushing_freq_hz: float = 2.0,
    seed: int = 20261110,
) -> IceLoadSpectrum:
    """Generate synthetic ice load time series.
    
    Models ice-induced loading as a combination of:
    1. Quasi-static mean load (ice resistance)
    2. Low-frequency oscillation (global structural response)
    3. High-frequency crushing pulses (local ice failure)
    4. Random component (turbulence, ridging)
    
    Parameters
    ----------
    duration_s : float
        Record duration (s).
    sampling_hz : float
        Sampling frequency (Hz).
    mean_force_n : float
        Mean ice force (N).
    dynamic_amplitude_n : float
        Dynamic amplitude of oscillation (N).
    primary_freq_hz : float
        Primary oscillation frequency (Hz).
    crushing_freq_hz : float
        Ice crushing pulse frequency (Hz).
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    IceLoadSpectrum
        Ice load time series with spectral characterization.
    """
    rng = np.random.RandomState(seed)
    
    n_samples = int(duration_s * sampling_hz)
    time = np.linspace(0, duration_s, n_samples, endpoint=False)
    dt = 1.0 / sampling_hz
    
    # 1. Quasi-static mean
    force = np.full(n_samples, mean_force_n)
    
    # 2. Low-frequency oscillation (structural response to ice)
    phase1 = rng.uniform(0, 2 * np.pi)
    force += dynamic_amplitude_n * np.sin(2 * np.pi * primary_freq_hz * time + phase1)
    
    # 3. High-frequency crushing pulses (intermittent)
    n_pulses = int(duration_s * crushing_freq_hz)
    pulse_times = rng.uniform(0, duration_s, n_pulses)
    pulse_amplitudes = rng.exponential(0.3 * dynamic_amplitude_n, n_pulses)
    
    for pt, pa in zip(pulse_times, pulse_amplitudes):
        # Exponential decay pulse
        pulse_duration = 0.1 + 0.05 * rng.random()
        mask = (time >= pt) & (time < pt + pulse_duration)
        t_local = time[mask] - pt
        force[mask] += pa * np.exp(-10 * t_local / pulse_duration)
    
    # 4. Random component (broadband)
    force += 0.1 * dynamic_amplitude_n * rng.randn(n_samples)
    
    # Ensure non-negative force (ice can only push)
    force = np.maximum(force, 0.0)
    
    # Spectral characterization
    fft_vals = np.fft.rfft(force - np.mean(force))
    freqs = np.fft.rfftfreq(n_samples, dt)
    psd = np.abs(fft_vals) ** 2
    
    # Peak frequency (exclude DC)
    if len(freqs) > 1:
        peak_idx = np.argmax(psd[1:]) + 1
        peak_freq = freqs[peak_idx]
    else:
        peak_freq = primary_freq_hz
    
    return IceLoadSpectrum(
        time_s=time,
        force_n=force,
        mean_force_n=float(np.mean(force)),
        std_force_n=float(np.std(force)),
        peak_frequency_hz=float(peak_freq),
        duration_s=duration_s,
        sampling_hz=sampling_hz,
    )


def _find_peaks_valleys(signal: np.ndarray) -> np.ndarray:
    """Extract alternating peaks and valleys from a signal."""
    if len(signal) < 3:
        return signal.copy()
    
    # Collapse plateaus
    changed = np.nonzero(np.diff(signal) != 0.0)[0] + 1
    keep = np.concatenate(([0], changed))
    if keep[-1] != len(signal) - 1 and signal[-1] != signal[keep[-1]]:
        keep = np.append(keep, len(signal) - 1)
    compressed = signal[keep]
    
    if len(compressed) < 3:
        return compressed
    
    # Find extrema
    diff = np.diff(compressed)
    extrema_indices = [0]
    for i in range(1, len(diff)):
        if diff[i] * diff[i - 1] < 0:
            extrema_indices.append(i)
    extrema_indices.append(len(compressed) - 1)
    
    return compressed[extrema_indices]


def rainflow_counting(signal: np.ndarray) -> RainflowResult:
    """ASTM E1049-85 four-point rainflow counting.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series (stress or load).
    
    Returns
    -------
    RainflowResult
        Cycle counting result with ranges, means, and counts.
    """
    if len(signal) < 3:
        return RainflowResult(
            ranges_mpa=np.array([]),
            means_mpa=np.array([]),
            counts=np.array([]),
        )
    
    # Extract peaks and valleys
    pv = _find_peaks_valleys(signal)
    
    if len(pv) < 3:
        return RainflowResult(
            ranges_mpa=np.array([]),
            means_mpa=np.array([]),
            counts=np.array([]),
        )
    
    # Four-point rainflow algorithm
    ranges = []
    means = []
    counts = []
    n_full = 0
    n_half = 0
    
    # Working stack
    stack = list(pv)
    extracted = []
    
    i = 0
    while i < len(stack) - 3:
        # Get four consecutive points
        s0, s1, s2, s3 = stack[i], stack[i+1], stack[i+2], stack[i+3]
        
        # Inner ranges
        r1 = abs(s1 - s2)
        r2 = abs(s0 - s1)
        r3 = abs(s2 - s3)
        
        # Four-point check: extract if inner range is smallest
        if r1 <= r2 and r1 <= r3:
            extracted.append((r1, (s1 + s2) / 2.0, 1.0))
            n_full += 1
            # Remove the two inner points
            stack.pop(i + 2)
            stack.pop(i + 1)
            # Don't increment i; re-check from same position
        else:
            i += 1
    
    # Remaining points: count as half-cycles
    for j in range(len(stack) - 1):
        r = abs(stack[j + 1] - stack[j])
        m = (stack[j] + stack[j + 1]) / 2.0
        extracted.append((r, m, 0.5))
        n_half += 1
    
    if len(extracted) > 0:
        extracted_arr = np.array(extracted)
        ranges = extracted_arr[:, 0]
        means = extracted_arr[:, 1]
        counts = extracted_arr[:, 2]
    else:
        ranges = np.array([])
        means = np.array([])
        counts = np.array([])
    
    return RainflowResult(
        ranges_mpa=ranges,
        means_mpa=means,
        counts=counts,
        n_full=n_full,
        n_half=n_half,
    )


def stress_range_histogram(
    rainflow_result: RainflowResult,
    n_bins: int = 20,
    max_stress_mpa: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create stress range histogram from rainflow result.
    
    Parameters
    ----------
    rainflow_result : RainflowResult
        Rainflow counting result.
    n_bins : int
        Number of histogram bins.
    max_stress_mpa : float, optional
        Maximum stress for binning. Default: max range × 1.1.
    
    Returns
    -------
    bin_centers : np.ndarray
        Stress range bin centers (MPa).
    bin_counts : np.ndarray
        Weighted cycle counts per bin.
    """
    if len(rainflow_result.ranges_mpa) == 0:
        return np.array([]), np.array([])
    
    ranges = rainflow_result.ranges_mpa
    counts = rainflow_result.counts
    
    if max_stress_mpa is None:
        max_stress_mpa = np.max(ranges) * 1.1
    
    bins = np.linspace(0, max_stress_mpa, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_counts = np.zeros(n_bins)
    
    for r, c in zip(ranges, counts):
        idx = np.searchsorted(bins[1:], r)
        if idx < n_bins:
            bin_counts[idx] += c
    
    return bin_centers, bin_counts


def spectral_moments(
    signal: np.ndarray,
    dt: float,
    n_moments: int = 4,
) -> np.ndarray:
    """Compute spectral moments m_0, m_1, ..., m_n.
    
    m_n = ∫ f^n × S(f) df
    
    Parameters
    ----------
    signal : np.ndarray
        Time series.
    dt : float
        Time step (s).
    n_moments : int
        Number of moments to compute (0 to n_moments-1).
    
    Returns
    -------
    np.ndarray
        Array of spectral moments [m_0, m_1, ..., m_{n-1}].
    """
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, dt)
    psd = np.abs(fft_vals) ** 2 * dt / n
    
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    
    moments = np.zeros(n_moments)
    for k in range(n_moments):
        moments[k] = np.sum(freqs ** k * psd) * df
    
    return moments
