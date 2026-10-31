"""Palmgren-Miner linear cumulative damage rule with bilinear correction.

Implements:
- Standard Miner's rule: D = sum(n_i / N_i)
- Bilinear correction for high-cycle regime (DNV approach)
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .sn_curve import DNCurve, SNCurve


@dataclass
class MinerResult:
    """Result of Miner damage calculation."""
    total_damage: float
    damage_per_block: np.ndarray
    n_cycles_per_block: np.ndarray
    n_to_failure_per_block: np.ndarray
    damage_ratio: float  # D / D_critical (typically D_critical = 1.0)
    is_safe: bool        # True if total_damage < 1.0


def miner_damage(
    stress_ranges: np.ndarray,
    n_cycles: np.ndarray,
    sn_curve: DNCurve,
    bilinear: bool = True,
) -> MinerResult:
    """Calculate cumulative fatigue damage using Palmgren-Miner rule.

    Parameters
    ----------
    stress_ranges : np.ndarray
        Stress ranges for each load block (MPa).
    n_cycles : np.ndarray
        Number of cycles at each stress range.
    sn_curve : DNCurve
        S-N curve evaluator.
    bilinear : bool
        If True, use bilinear S-N curve (DNV approach).
        If False, use single-slope S-N curve.

    Returns
    -------
    MinerResult
        Contains total damage and per-block breakdown.
    """
    if len(stress_ranges) != len(n_cycles):
        raise ValueError("stress_ranges and n_cycles must have same length")

    damage_per_block = np.zeros(len(stress_ranges))
    n_to_failure_per_block = np.zeros(len(stress_ranges))

    for i, (sr, nc) in enumerate(zip(stress_ranges, n_cycles)):
        if sr <= 0:
            n_to_failure = np.inf
        else:
            n_to_failure = sn_curve.n_to_failure(sr)

        n_to_failure_per_block[i] = n_to_failure

        if n_to_failure > 0 and n_to_failure < np.inf:
            damage_per_block[i] = nc / n_to_failure
        else:
            damage_per_block[i] = 0.0

    total_damage = float(np.sum(damage_per_block))

    return MinerResult(
        total_damage=total_damage,
        damage_per_block=damage_per_block,
        n_cycles_per_block=n_cycles,
        n_to_failure_per_block=n_to_failure_per_block,
        damage_ratio=total_damage,
        is_safe=total_damage < 1.0,
    )


def miner_two_block(
    stress_range_1: float, n1: float,
    stress_range_2: float, n2: float,
    sn_curve: DNCurve,
) -> float:
    """Calculate Miner damage for two-block loading.

    Returns D = n1/N1 + n2/N2.

    Parameters
    ----------
    stress_range_1 : float
        Stress range for block 1 (MPa).
    n1 : float
        Number of cycles in block 1.
    stress_range_2 : float
        Stress range for block 2 (MPa).
    n2 : float
        Number of cycles in block 2.
    sn_curve : DNCurve
        S-N curve evaluator.

    Returns
    -------
    float
        Total Miner damage.
    """
    N1 = sn_curve.n_to_failure(stress_range_1)
    N2 = sn_curve.n_to_failure(stress_range_2)

    d1 = n1 / N1 if N1 > 0 and N1 < np.inf else 0.0
    d2 = n2 / N2 if N2 > 0 and N2 < np.inf else 0.0

    return d1 + d2


def miner_two_block_single_slope(
    stress_range_1: float, n1: float,
    stress_range_2: float, n2: float,
    m: float, log_K: float,
) -> float:
    """Calculate Miner damage for two-block loading using single-slope S-N.

    This matches the closed_form_two_block function exactly.

    Parameters
    ----------
    stress_range_1 : float
        Stress range for block 1.
    n1 : float
        Cycles in block 1.
    stress_range_2 : float
        Stress range for block 2.
    n2 : float
        Cycles in block 2.
    m : float
        S-N curve slope.
    log_K : float
        log10 of S-N curve intercept.

    Returns
    -------
    float
        Total Miner damage.
    """
    sn = SNCurve(m=m, log_K=log_K)
    N1 = sn.n_to_failure(stress_range_1)
    N2 = sn.n_to_failure(stress_range_2)

    d1 = n1 / N1 if N1 > 0 and N1 < np.inf else 0.0
    d2 = n2 / N2 if N2 > 0 and N2 < np.inf else 0.0

    return d1 + d2


def closed_form_two_block(
    stress_range_1: float, n1: float,
    stress_range_2: float, n2: float,
    m: float, log_K: float,
) -> float:
    """Closed-form Miner damage for single-slope S-N curve.

    D = n1 * S1^m / 10^logK + n2 * S2^m / 10^logK

    Parameters
    ----------
    stress_range_1 : float
        Stress range for block 1.
    n1 : float
        Cycles in block 1.
    stress_range_2 : float
        Stress range for block 2.
    n2 : float
        Cycles in block 2.
    m : float
        S-N curve slope.
    log_K : float
        log10 of S-N curve intercept.

    Returns
    -------
    float
        Total Miner damage.
    """
    K = 10 ** log_K
    d1 = n1 * (stress_range_1 ** m) / K
    d2 = n2 * (stress_range_2 ** m) / K
    return d1 + d2
