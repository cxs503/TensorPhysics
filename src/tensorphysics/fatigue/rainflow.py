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

    Plateaus (runs of exactly equal samples, e.g. the zero gaps between
    intermittent ice-crushing pulses or a clipped signal) are collapsed to a
    single sample *before* the sign change test.  Without that step a plateau
    hides the extremum next to it - ``diff`` is zero inside the plateau, so
    ``diff[i] * diff[i-1] < 0`` never fires - and extrema are silently lost,
    which under-counts cycles.  For signals without repeats the result is
    bit-identical to the plain sign-change scan.
    """
    if len(signal) < 3:
        return signal.copy()

    # Collapse plateaus: keep the first sample of every run of equal values.
    # The last sample is appended only when it ends a non-constant step; if the
    # record ends *inside* a plateau the plateau's first sample already
    # represents it and appending another copy would create a spurious
    # zero-range residue cycle.
    changed = np.nonzero(np.diff(signal) != 0.0)[0] + 1
    keep = np.concatenate(([0], changed))
    if keep[-1] != len(signal) - 1 and signal[-1] != signal[keep[-1]]:
        keep = np.append(keep, len(signal) - 1)
    compressed = signal[keep]
    if len(compressed) < 3:
        return compressed

    # Find sign changes (peaks and valleys)
    diff = np.diff(compressed)
    extrema_indices = [0]  # always include first point
    for i in range(1, len(diff)):
        if diff[i] * diff[i - 1] < 0:
            extrema_indices.append(i)
    extrema_indices.append(len(compressed) - 1)  # always include last point

    return compressed[extrema_indices]


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


def rainflow_counting_stack(signal: np.ndarray) -> RainflowResult:
    """O(N) stack implementation of the ASTM E1049-85 four-point rule.

    Numerically identical to :func:`rainflow_counting` (same ranges, means and
    0.5/1.0 cycle weights) but linear in the number of extrema instead of
    quadratic: ``rainflow_counting`` deletes from the middle of a Python list,
    which is fine for a few thousand points but impractical for the 1e6-sample
    records used in spectral-fatigue Monte-Carlo studies.

    Algorithm
    ---------
    Push each extremum.  For the last four stack entries (A, B, C, D) the middle
    range B-C is a closed cycle when ``|C-B| <= |B-A|`` and ``|C-B| <= |D-C|``:
    record the full cycle and drop B and C.  Repeat.  Whatever remains at the
    end is counted as half cycles between consecutive points (residue).
    """
    if len(signal) < 2:
        return RainflowResult(ranges=np.array([]), means=np.array([]),
                              counts=np.array([]))

    pv = _find_peaks_valleys(np.asarray(signal, dtype=float))
    if len(pv) < 3:
        return RainflowResult(ranges=np.array([]), means=np.array([]),
                              counts=np.array([]))

    stack = []
    ranges = []
    means = []
    counts = []
    for x in pv:
        stack.append(float(x))
        while len(stack) >= 4:
            a, b, c, d = stack[-4], stack[-3], stack[-2], stack[-1]
            r_bc = abs(c - b)
            if r_bc <= abs(b - a) and r_bc <= abs(d - c):
                ranges.append(r_bc)
                means.append(0.5 * (b + c))
                counts.append(1.0)
                del stack[-3:-1]
            else:
                break

    n_full = len(ranges)
    n_half = 0
    for j in range(len(stack) - 1):
        r = abs(stack[j + 1] - stack[j])
        if r > 0:
            ranges.append(r)
            means.append(0.5 * (stack[j + 1] + stack[j]))
            counts.append(0.5)
            n_half += 1

    return RainflowResult(
        ranges=np.array(ranges),
        means=np.array(means),
        counts=np.array(counts),
        n_full=n_full,
        n_half=n_half,
    )
