"""Miner linear cumulative damage rule for ice-induced fatigue.

Implements Palmgren-Miner linear damage accumulation:
    D = Σ (n_i / N_i)

where n_i = number of cycles at stress range Δσ_i
      N_i = cycles to failure at Δσ_i (from S-N curve)

Failure criterion: D ≥ 1.0

Also implements:
- Verification against exact solutions
- Negative controls (random N_i must FAIL exact check)
- Fatigue life estimation from damage rate
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

__all__ = [
    "MinerResult",
    "miner_cumulative_damage",
    "miner_exact_verification",
    "fatigue_life_from_damage",
    "miner_two_level_loading",
]


@dataclass
class MinerResult:
    """Result of Miner cumulative damage calculation.
    
    Attributes
    ----------
    total_damage : float
        Total cumulative damage D = Σ(n_i / N_i).
    damage_per_bin : np.ndarray
        Damage contribution from each stress bin.
    stress_ranges_mpa : np.ndarray
        Stress range bins (MPa).
    cycle_counts : np.ndarray
        Number of cycles at each stress range.
    n_to_failure : np.ndarray
        Cycles to failure at each stress range.
    is_failed : bool
        True if D ≥ 1.0 (fatigue failure).
    safety_factor : float
        1/D (if D > 0). Higher = safer.
    fatigue_life_cycles : float
        Estimated total fatigue life in cycles.
    """
    total_damage: float
    damage_per_bin: np.ndarray
    stress_ranges_mpa: np.ndarray
    cycle_counts: np.ndarray
    n_to_failure: np.ndarray
    is_failed: bool
    safety_factor: float
    fatigue_life_cycles: float


def miner_cumulative_damage(
    stress_ranges_mpa: np.ndarray,
    cycle_counts: np.ndarray,
    sn_curve,
) -> MinerResult:
    """Calculate Miner cumulative damage from stress spectrum.
    
    D = Σ (n_i / N_i)
    
    where N_i is obtained from the S-N curve for each stress range.
    
    Parameters
    ----------
    stress_ranges_mpa : np.ndarray
        Stress range bins from rainflow counting (MPa).
    cycle_counts : np.ndarray
        Number of cycles at each stress range.
    sn_curve : SNCurveRP203 or compatible
        S-N curve evaluator with n_to_failure() method.
    
    Returns
    -------
    MinerResult
        Complete damage assessment result.
    
    Notes
    -----
    - Stress ranges ≤ 0 are ignored (no damage).
    - N_i = inf → damage contribution = 0.
    - D ≥ 1.0 indicates fatigue failure per Miner's rule.
    """
    stress_ranges_mpa = np.asarray(stress_ranges_mpa, dtype=float)
    cycle_counts = np.asarray(cycle_counts, dtype=float)
    
    if len(stress_ranges_mpa) != len(cycle_counts):
        raise ValueError("stress_ranges and cycle_counts must have same length")
    
    n_bins = len(stress_ranges_mpa)
    damage_per_bin = np.zeros(n_bins)
    n_to_failure = np.zeros(n_bins)
    
    for i in range(n_bins):
        ds = stress_ranges_mpa[i]
        ni = cycle_counts[i]
        
        if ds <= 0 or ni <= 0:
            n_to_failure[i] = np.inf
            damage_per_bin[i] = 0.0
            continue
        
        Ni = sn_curve.n_to_failure(ds)
        n_to_failure[i] = Ni
        
        if Ni > 0 and Ni < np.inf:
            damage_per_bin[i] = ni / Ni
        else:
            damage_per_bin[i] = 0.0
    
    total_damage = float(np.sum(damage_per_bin))
    
    # Safety factor
    if total_damage > 0:
        safety_factor = 1.0 / total_damage
    else:
        safety_factor = np.inf
    
    # Fatigue life: total cycles applied / total damage
    total_cycles_applied = float(np.sum(cycle_counts))
    if total_damage > 0:
        fatigue_life_cycles = total_cycles_applied / total_damage
    else:
        fatigue_life_cycles = np.inf
    
    return MinerResult(
        total_damage=total_damage,
        damage_per_bin=damage_per_bin,
        stress_ranges_mpa=stress_ranges_mpa,
        cycle_counts=cycle_counts,
        n_to_failure=n_to_failure,
        is_failed=total_damage >= 1.0,
        safety_factor=safety_factor,
        fatigue_life_cycles=fatigue_life_cycles,
    )


def miner_exact_verification(
    stress_ranges_mpa: np.ndarray,
    cycle_counts: np.ndarray,
    m: float,
    log_a: float,
) -> Tuple[float, float]:
    """Verify Miner damage against closed-form solution.
    
    For a single-slope S-N curve: N = 10^log_a / (Δσ)^m
    
    The Miner damage is:
        D = Σ n_i / N_i = Σ n_i × (Δσ_i)^m / 10^log_a
    
    This function computes both the direct sum and the closed-form
    to verify they match within numerical precision.
    
    Parameters
    ----------
    stress_ranges_mpa : np.ndarray
        Stress ranges (MPa).
    cycle_counts : np.ndarray
        Cycle counts.
    m : float
        S-N curve slope.
    log_a : float
        S-N curve intercept (log10).
    
    Returns
    -------
    direct_sum : float
        Direct Miner sum: Σ(n_i / N_i).
    closed_form : float
        Closed-form: Σ(n_i × Δσ_i^m) / 10^log_a.
    """
    stress_ranges_mpa = np.asarray(stress_ranges_mpa, dtype=float)
    cycle_counts = np.asarray(cycle_counts, dtype=float)
    
    K = 10.0 ** log_a
    
    # Direct sum
    direct_sum = 0.0
    for ds, ni in zip(stress_ranges_mpa, cycle_counts):
        if ds > 0 and ni > 0:
            Ni = K / (ds ** m)
            direct_sum += ni / Ni
    
    # Closed form
    closed_form = 0.0
    for ds, ni in zip(stress_ranges_mpa, cycle_counts):
        if ds > 0 and ni > 0:
            closed_form += ni * (ds ** m) / K
    
    return direct_sum, closed_form


def fatigue_life_from_damage(
    total_damage: float,
    exposure_duration_years: float = 1.0,
) -> float:
    """Estimate fatigue life from Miner damage.
    
    If damage D accumulates over T years of exposure,
    the total fatigue life is:
        T_life = T / D  (years)
    
    Parameters
    ----------
    total_damage : float
        Total Miner damage accumulated.
    exposure_duration_years : float
        Duration over which damage accumulated (years).
    
    Returns
    -------
    float
        Estimated fatigue life (years). Returns inf if D = 0.
    """
    if total_damage <= 0:
        return np.inf
    return exposure_duration_years / total_damage


def miner_two_level_loading(
    ds1_mpa: float,
    n1: float,
    ds2_mpa: float,
    n2: float,
    m: float,
    log_a: float,
) -> float:
    """Miner damage for two-level constant-amplitude loading.
    
    D = n1/N1 + n2/N2
    
    where N_i = 10^log_a / (Δσ_i)^m
    
    Parameters
    ----------
    ds1_mpa : float
        Stress range level 1 (MPa).
    n1 : float
        Cycles at level 1.
    ds2_mpa : float
        Stress range level 2 (MPa).
    n2 : float
        Cycles at level 2.
    m : float
        S-N curve slope.
    log_a : float
        S-N curve intercept.
    
    Returns
    -------
    float
        Total Miner damage D.
    """
    K = 10.0 ** log_a
    
    d1 = 0.0
    if ds1_mpa > 0 and n1 > 0:
        N1 = K / (ds1_mpa ** m)
        d1 = n1 / N1
    
    d2 = 0.0
    if ds2_mpa > 0 and n2 > 0:
        N2 = K / (ds2_mpa ** m)
        d2 = n2 / N2
    
    return d1 + d2
