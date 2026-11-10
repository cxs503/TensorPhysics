"""Palmgren-Miner linear cumulative damage + variable-amplitude verification.

The module provides

* ``miner_damage``            -- D = sum(n_i / N_i) for any block spectrum;
* ``two_block_blocks_to_failure`` -- exact inversion of D(lambda) = 1 for a
  two-level block spectrum (closed form, used as an independent cross-check of
  the accumulation loop);
* ``sequence_effect_probe``   -- demonstrates the *documented* limitation of
  Miner's rule (no load-sequence dependence);
* ``rainflow_block_damage``   -- ASTM E1049-85 counting (re-using
  :mod:`tensorphysics.fatigue.rainflow`) followed by Miner accumulation, i.e.
  the variable-amplitude path;
* ``CORTEN_DOLAN_STATUS``     -- explicit honest declaration that a non-linear
  cumulative-damage rule (Corten-Dolan) is **not** part of this capability yet
  (acceptance criterion C7).

Everything here is pure arithmetic on top of an S-N curve object exposing
``life(stress_range)``; that keeps the damage rule testable against closed-form
solutions to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ..rainflow import rainflow_counting


# ============================================================================
# Declarations (acceptance criterion C7 -- honest limitation gate)
# ============================================================================

#: Non-linear cumulative damage is deliberately *not* implemented/validated.
CORTEN_DOLAN_STATUS: Dict[str, str] = {
    "model": "Corten-Dolan",
    "status": "not_implemented",
    "reason": ("load-sequence / interaction effects are outside Miner's rule; a "
               "Corten-Dolan (or double-linear / damage-curve) rule requires "
               "calibration against variable-amplitude test data, which is not "
               "available in this repository"),
    "impact": ("Miner gives sequence-independent damage: strongly skewed spectra "
               "(high-low vs low-high) are mispredicted in the direction "
               "quantified by sequence_effect_probe()"),
    "action": "pending: implement + calibrate against variable-amplitude weld tests",
}

#: Further documented limitations of the present capability.
KNOWN_LIMITATIONS: Dict[str, str] = {
    "linear_damage": "Miner's rule ignores load-sequence and interaction effects.",
    "mean_stress": "no mean-stress correction (Haigh / Goodman) is applied.",
    "thickness_effect": "no plate-thickness correction of the S-N curve (t_eff rule).",
    "post_weld_treatment": "only as-welded FAT classes; TIG dressing / peening not covered.",
    "crack_growth": "no fracture-mechanics (da/dN) path; initiation/design-curve only.",
}


# ============================================================================
# Miner accumulation
# ============================================================================

@dataclass
class MinerDamage:
    """Miner damage for a block spectrum."""

    total_damage: float
    per_block_damage: np.ndarray
    life_per_block: np.ndarray
    stress_ranges: np.ndarray
    cycles: np.ndarray
    blocks_to_failure: float

    @property
    def d(self) -> float:
        return self.total_damage

    def is_safe(self, d_crit: float = 1.0) -> bool:
        return self.total_damage < d_crit

    def as_dict(self) -> Dict[str, object]:
        return {
            "total_damage": self.total_damage,
            "blocks_to_failure": self.blocks_to_failure,
            "per_block_damage": self.per_block_damage.tolist(),
            "life_per_block": self.life_per_block.tolist(),
            "stress_ranges": self.stress_ranges.tolist(),
            "cycles": self.cycles.tolist(),
        }


def miner_damage(curve, stress_ranges: Sequence[float],
                 cycles: Sequence[float]) -> MinerDamage:
    """Accumulate Palmgren-Miner damage ``D = sum(n_i / N_i)``.

    Parameters
    ----------
    curve : object with ``life(stress_range) -> float``
        S-N curve (e.g. :class:`tensorphysics.fatigue.detail_maturity.sn_iiw.IIWCurve`).
    stress_ranges : sequence of float
        Constant-amplitude stress ranges of the block spectrum (MPa).
    cycles : sequence of float
        Cycle counts of each block.
    """
    s = np.atleast_1d(np.asarray(stress_ranges, dtype=float))
    n = np.atleast_1d(np.asarray(cycles, dtype=float))
    if s.size != n.size:
        raise ValueError("stress_ranges and cycles must have the same length")
    if np.any(n < 0):
        raise ValueError("cycle counts must be non-negative")

    life = np.array([curve.life(float(si)) for si in s], dtype=float)
    dmg = np.zeros_like(s)
    finite = np.isfinite(life) & (life > 0)
    dmg[finite] = n[finite] / life[finite]
    total = float(np.sum(dmg))
    return MinerDamage(
        total_damage=total,
        per_block_damage=dmg,
        life_per_block=life,
        stress_ranges=s,
        cycles=n,
        blocks_to_failure=float("inf") if total <= 0 else 1.0 / total,
    )


def blocks_to_failure(curve, stress_ranges: Sequence[float],
                      cycles: Sequence[float], d_critical: float = 1.0) -> float:
    """Number of repetitions of a block spectrum until ``D = d_critical``."""
    d = miner_damage(curve, stress_ranges, cycles)
    if d.total_damage <= 0:
        return float("inf")
    return float(d_critical / d.total_damage)


def two_block_blocks_to_failure(curve, ds1: float, n1: float,
                                ds2: float, n2: float,
                                d_critical: float = 1.0) -> float:
    """Exact closed form for a two-level block spectrum.

    ``D = n1/N1 + n2/N2`` for one repetition, so the number of repetitions to
    failure is ``lambda = d_critical / (n1/N1 + n2/N2)``.
    """
    life1 = curve.life(float(ds1))
    life2 = curve.life(float(ds2))
    d1 = 0.0 if not np.isfinite(life1) else n1 / life1
    d2 = 0.0 if not np.isfinite(life2) else n2 / life2
    denom = d1 + d2
    if denom <= 0:
        return float("inf")
    return float(d_critical / denom)


def power_law_two_block_life(ds1: float, n1: float, ds2: float, n2: float,
                             log_a: float, m: float,
                             d_critical: float = 1.0) -> float:
    """Independent single-slope closed form ``N_i = 10**log_a * dsigma_i**-m``.

    Used as a second, curve-object-free implementation so that the accumulation
    loop can be checked against pure algebra (not against itself).
    """
    n1_life = 10.0 ** (log_a - m * np.log10(ds1))
    n2_life = 10.0 ** (log_a - m * np.log10(ds2))
    d = n1 / n1_life + n2 / n2_life
    return float(d_critical / d)


# ============================================================================
# Variable amplitude: block history + rainflow
# ============================================================================

def block_history(stress_ranges: Sequence[float], cycles: Sequence[float],
                  order: str = "high_low", n_ramp: int = 2,
                  mean: float = 0.0) -> np.ndarray:
    """Build a stress history alternating between the block levels.

    Each block is a triangular wave oscillating between ``mean - dsigma/2`` and
    ``mean + dsigma/2`` with ``2*cycles`` reversals; blocks are concatenated in
    the requested order.

    Parameters
    ----------
    order : {'high_low', 'low_high'}
        Sequence in which the blocks are applied.
    n_ramp : int
        Extra reversals per block (kept >= 1 so that every block contains at
        least one full cycle).
    """
    s = np.atleast_1d(np.asarray(stress_ranges, dtype=float))
    n = np.atleast_1d(np.asarray(cycles, dtype=float))
    if order not in ("high_low", "low_high"):
        raise ValueError("order must be 'high_low' or 'low_high'")
    idx = np.argsort(-s if order == "high_low" else s)
    sig: List[float] = []
    for i in idx:
        n_cyc = int(round(float(n[i])))
        if n_cyc < 1:
            continue
        lo = mean - 0.5 * float(s[i])
        hi = mean + 0.5 * float(s[i])
        rev = [hi, lo] * n_cyc
        rev = rev[: max(2, 2 * n_cyc - 1 + n_ramp)]
        sig.extend(rev)
    return np.asarray(sig, dtype=float)


def rainflow_block_damage(curve, signal: Sequence[float]) -> Dict[str, object]:
    """Rainflow-count a stress history and accumulate Miner damage.

    Returns a dictionary with the total damage, the number of extracted cycles
    and the (range, count) spectrum.
    """
    rf = rainflow_counting(np.asarray(signal, dtype=float))
    ranges = np.asarray(rf.ranges, dtype=float)
    counts = np.asarray(rf.counts, dtype=float)
    keep = (ranges > 0) & (counts > 0)
    d = miner_damage(curve, ranges[keep], counts[keep]) if keep.any() else None
    return {
        "total_damage": 0.0 if d is None else d.total_damage,
        "n_cycles": float(np.sum(counts[keep])) if keep.any() else 0.0,
        "n_full": int(rf.n_full),
        "n_half": int(rf.n_half),
        "ranges": ranges[keep].tolist(),
        "counts": counts[keep].tolist(),
    }


# ============================================================================
# Sequence-effect probe (documents Miner's blindness -- criterion C7)
# ============================================================================

@dataclass
class SequenceEffectProbe:
    """Miner damage of the same two blocks applied in opposite order."""

    damage_high_low: float
    damage_low_high: float

    @property
    def relative_difference(self) -> float:
        ref = max(self.damage_high_low, 1e-30)
        return abs(self.damage_low_high - self.damage_high_low) / ref

    @property
    def is_sequence_independent(self) -> bool:
        return self.relative_difference < 1e-12


def sequence_effect_probe(curve, ds1: float, n1: float,
                          ds2: float, n2: float) -> SequenceEffectProbe:
    """Damage of ``(ds1, n1)`` then ``(ds2, n2)`` and of the reversed order.

    Under Miner's rule the two are *identical by construction*; the probe makes
    the limitation explicit and machine-checkable instead of hiding it in prose.
    """
    d_hl = miner_damage(curve, [ds1, ds2], [n1, n2]).total_damage
    d_lh = miner_damage(curve, [ds2, ds1], [n2, n1]).total_damage
    return SequenceEffectProbe(damage_high_low=d_hl, damage_low_high=d_lh)


# ============================================================================
# Design margin helpers
# ============================================================================

def scatter_band_factor(sigma_log_n: float = 0.2, coverage: float = 0.90) -> float:
    """Two-sided ``coverage`` factor of a log-normal life scatter band.

    ``coverage=0.90`` with the IIW reference ``sigma_log_n = 0.2`` gives
    ``10**(1.645*0.2) = 2.13`` -- i.e. the *experimental* scatter band of a single
    specimen is wider than a factor 2, which must not be confused with a model
    error.
    """
    from scipy.stats import norm
    z = float(norm.ppf(0.5 + 0.5 * coverage))
    return float(10.0 ** (z * sigma_log_n))


def predicted_vs_test_error(predicted_life: float, test_life: float) -> float:
    """Error factor ``max(ratio, 1/ratio)`` (>= 1) between predicted and test life."""
    if predicted_life <= 0 or test_life <= 0:
        return float("inf")
    r = predicted_life / test_life
    return float(max(r, 1.0 / r))


def error_factor_of_series(predicted_life: float, test_lives: Sequence[float],
                           quantile: float = 0.9) -> Dict[str, float]:
    """Predicted/test error factors of a test series, with a quantile.

    Returns the median factor and the factor at the requested two-sided
    confidence level (default 90 %: the 5 %/95 % envelope).
    """
    t = np.asarray(test_lives, dtype=float)
    ratios = predicted_life / t
    factors = np.maximum(ratios, 1.0 / ratios)
    lo = (1.0 - quantile) / 2.0
    hi = 1.0 - lo
    # central comparison: prediction against the median test life (a factor-2
    # acceptance on a *life prediction* is a statement about central values; the
    # per-specimen spread is a property of the data, reported separately)
    central = float(predicted_life / np.median(t))
    return {
        "central_ratio": central,
        "central_factor": float(max(central, 1.0 / central)),
        "median_factor": float(np.median(factors)),
        "p05_factor": float(np.quantile(factors, lo)),
        "p95_factor": float(np.quantile(factors, hi)),
        "envelope_factor": float(np.quantile(factors, hi)),
        "n_specimens": int(t.size),
    }