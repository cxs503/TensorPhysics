"""ASTM E1049-85 four-point rainflow counting algorithm.

Implements the standard rainflow cycle counting method with explicit
boundary handling (0.5 cycle accounting for residual ranges).
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class RainflowResult:
    """Result of rainflow cycle counting."""
    ranges: np.ndarray       # stress ranges (half-cycles counted as 0.5)
    means: np.ndarray        # mean stress for each cycle
    counts: np.ndarray       # cycle counts (0.5 for half-cycles, 1.0 for full)
    n_full: int = 0          # number of full cycles
    n_half: int = 0          # number of half-cycles

    @property
    def total_cycles(self) -> float:
        return float(np.sum(self.counts))


def _find_peaks_valleys(signal: np.ndarray) -> np.ndarray:
    """Extract alternating peaks and valleys from a signal.

    Removes monotonic segments, keeping only local extrema.
    """
    if len(signal) < 3:
        return signal.copy()

    # Compute differences
    diff = np.diff(signal)

    # Find sign changes (peaks and valleys)
    extrema_indices = [0]  # always include first point
    for i in range(1, len(diff)):
        if diff[i] * diff[i - 1] < 0:
            extrema_indices.append(i)
    extrema_indices.append(len(signal) - 1)  # always include last point

    return signal[extrema_indices]


def rainflow_counting(signal: np.ndarray) -> RainflowResult:
    """Perform ASTM E1049-85 four-point rainflow counting.

    Parameters
    ----------
    signal : np.ndarray
        Time series of stress (or load) values.

    Returns
    -------
    RainflowResult
        Contains ranges, means, and counts for all identified cycles.
        Half-cycles are counted as 0.5, full cycles as 1.0.
    """
    if len(signal) < 2:
        return RainflowResult(
            ranges=np.array([]),
            means=np.array([]),
            counts=np.array([]),
        )

    # Step 1: Extract peaks and valleys
    pv = _find_peaks_valleys(signal)

    if len(pv) < 3:
        # Not enough extrema for any cycle
        return RainflowResult(
            ranges=np.array([]),
            means=np.array([]),
            counts=np.array([]),
        )

    # Step 2: Rainflow counting using four-point method
    working = list(pv)
    ranges_list = []
    means_list = []
    counts_list = []

    i = 0
    while i + 3 < len(working):
        A = working[i]
        B = working[i + 1]
        C = working[i + 2]
        D = working[i + 3]

        range_BC = abs(C - B)
        range_AB = abs(B - A)
        range_CD = abs(D - C)

        if range_BC >= range_AB and range_BC >= range_CD:
            # B-C forms a cycle
            stress_range = range_BC
            mean_stress = (B + C) / 2.0
            ranges_list.append(stress_range)
            means_list.append(mean_stress)
            counts_list.append(1.0)  # full cycle

            # Remove B and C
            del working[i + 1]
            del working[i + 1]

            # Step back to check if A can now form a cycle
            if i > 0:
                i -= 1
        else:
            i += 1

    # Step 3: Handle residue (remaining points form half-cycles)
    n_full = len(ranges_list)
    n_half = 0

    if len(working) >= 2:
        for j in range(len(working) - 1):
            stress_range = abs(working[j + 1] - working[j])
            mean_stress = (working[j + 1] + working[j]) / 2.0
            if stress_range > 0:
                ranges_list.append(stress_range)
                means_list.append(mean_stress)
                counts_list.append(0.5)  # half cycle
                n_half += 1

    return RainflowResult(
        ranges=np.array(ranges_list),
        means=np.array(means_list),
        counts=np.array(counts_list),
        n_full=n_full,
        n_half=n_half,
    )


def rainflow_histogram(result: RainflowResult, n_bins: int = 20,
                       range_max: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a histogram of rainflow cycle ranges.

    Uses weighted mean of actual ranges in each bin for better accuracy.

    Parameters
    ----------
    result : RainflowResult
        Output from rainflow_counting.
    n_bins : int
        Number of bins for the histogram.
    range_max : float, optional
        Maximum range for binning. If None, uses max of result.ranges.

    Returns
    -------
    bin_edges, bin_counts, bin_means
        bin_means is the weighted mean of actual ranges in each bin,
        weighted by cycle count.
    """
    if len(result.ranges) == 0:
        return np.array([]), np.array([]), np.array([])

    if range_max is None:
        range_max = np.max(result.ranges)

    bin_edges = np.linspace(0, range_max, n_bins + 1)
    bin_counts = np.zeros(n_bins)
    bin_weighted_sums = np.zeros(n_bins)  # sum of (range * count) per bin

    for r, c in zip(result.ranges, result.counts):
        idx = np.searchsorted(bin_edges, r, side='right') - 1
        idx = min(idx, n_bins - 1)
        if idx >= 0:
            bin_counts[idx] += c
            bin_weighted_sums[idx] += r * c

    # Compute weighted mean for each bin
    bin_means = np.zeros(n_bins)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_means[i] = bin_weighted_sums[i] / bin_counts[i]
        else:
            bin_means[i] = (bin_edges[i] + bin_edges[i + 1]) / 2.0

    return bin_edges, bin_counts, bin_means
