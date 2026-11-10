"""S-N curve (DNV-RP-C203 F3, seawater with cathodic protection) + Goodman + Miner.

Kept deliberately small and self-contained so that the *anchor* reproduction of
the ship-platform ice-fatigue case (DAG IN32-N07, mission m5528) does not depend
on the shape of the general-purpose :mod:`tensorphysics.fatigue.sn_curve` table
(which is anchored on the D/F classes and does not carry F3/CP).

Curve form (DNV-RP-C203 Table 2-1/2-2, characteristic values, Δσ in MPa)::

    N = 10^{log a1} Δσ^{-m1},   N <= N_knee        (m1 = 3)
    N = 10^{log a2} Δσ^{-m2},   N >  N_knee        (m2 = 5, log a2 from knee continuity)

F3 in air: Δσ(N=1e7) = 31.86 MPa -> log a1 = 11.510.  Seawater with cathodic
protection multiplies the life by 2.5 (log a -> log a - log10 2.5 = -0.39794),
so log a1(CP) = 11.11206.

Two corrections are applied, in this order, before the curve is evaluated:

Knee handling (important): the cathodic-protection factor multiplies the life,
so *both* log a values are shifted by -log10(2.5) while the knee stress range
stays at the air value Delta-sigma_knee = 31.86 MPa.  Equivalently the knee cycle
count becomes N_knee/2.5 = 4e6.  Deriving log a2 "from continuity at N_knee=1e7"
would instead place the knee at 23.48 MPa, which over-penalises the low-stress
(m2) branch by a factor ~2 and was caught by the m5528 negative control (a
+6.4 % damage bias before the fix).

* **Thickness** (DNV-RP-C203 2.3): Δσ_used = Δσ_corr / (t/t_ref)^k with
  t_ref = 25 mm, k = 0.2 (only for t > 25 mm) -- i.e. the S-N curve is entered
  with the thickness-reduced range, which is the same thing as reading the
  corrected curve at the uncorrected range.
* **Goodman mean-stress**: σ_a,eff = σ_a / (1 - σ_m/σ_UTS), Δσ_eff = 2 σ_a,eff.

The thickness direction was chosen to match the m5528 reference implementation
bit for bit (reproduction of its published damage is a hard acceptance
criterion, see :mod:`tensorphysics.fatigue.combined.ice_anchor`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DNV_F3_LOG_A1_AIR",
    "DNV_F3_LOG_A1_CP",
    "DNV_F3_LOG_A2_AIR",
    "DNV_F3_RANGE_KNEE_MPA",
    "CP_LIFE_FACTOR",
    "SNParams",
    "goodman_effective_range",
    "SNCurveF3",
    "miner_damage",
    "BASELINE_SN",
]

DNV_F3_LOG_A1_AIR = 11.510
CP_LIFE_FACTOR = 2.5
DNV_F3_LOG_A1_CP = DNV_F3_LOG_A1_AIR - math.log10(CP_LIFE_FACTOR)  # 11.11206


DNV_F3_LOG_A2_AIR = 14.516
DNV_F3_RANGE_KNEE_MPA = 31.86  # Delta-sigma at N = 1e7 (air), unchanged under CP


@dataclass(frozen=True)
class SNParams:
    """Bilinear S-N description, Delta-sigma in MPa."""

    m1: float = 3.0
    log_a1: float = DNV_F3_LOG_A1_CP
    m2: float = 5.0
    log_a2: float = DNV_F3_LOG_A2_AIR - math.log10(CP_LIFE_FACTOR)
    range_knee: float = DNV_F3_RANGE_KNEE_MPA
    n_cutoff: float = 1.0e16
    t_ref_mm: float = 25.0
    thickness_exp: float = 0.2
    #: ``"standard"`` (DNV: knee range kept at the air value, N_knee/2.5) or
    #: ``"legacy_m5528"`` (knee range recomputed as 10^((log a1 - 7)/m1) from the
    #: *CP-shifted* log a1, which moves the knee down to 23.48 MPa).  The legacy
    #: rule is only used to reproduce the m5528 reference bit for bit.
    knee_rule: str = "standard"
    #: ``"legacy_m5528"`` divides the stress range by the thickness factor (so a
    #: thicker plate gets a *longer* life), ``"dnv"`` multiplies it
    #: (DNV-RP-C203 2.3: the corrected curve sees the enlarged range).  The
    #: legacy direction is non-conservative by (t/25)^(0.2 m); it is kept as the
    #: default only because the m5528 reference uses it.
    thickness_rule: str = "legacy_m5528"

    @property
    def range_knee_effective(self) -> float:
        if self.knee_rule == "standard":
            return self.range_knee
        if self.knee_rule == "legacy_m5528":
            return float(10.0 ** ((self.log_a1 - math.log10(1.0e7)) / self.m1))
        raise ValueError(f"unknown knee_rule {self.knee_rule!r}")

    @property
    def n_knee(self) -> float:
        """Cycles at the knee of *this* curve (1e7 air, 4e6 with CP)."""
        return float(10.0 ** (self.log_a1 - self.m1 * math.log10(self.range_knee_effective)))

    def thickness_factor(self, thickness_mm: Optional[float]) -> float:
        if thickness_mm is None or thickness_mm <= self.t_ref_mm:
            return 1.0
        return (float(thickness_mm) / self.t_ref_mm) ** self.thickness_exp


BASELINE_SN = SNParams()


def goodman_effective_range(
    stress_range: float, mean_stress: float, s_uts: float
) -> float:
    """Δσ_eff = Δσ / (1 - σ_m/σ_UTS); ``inf`` when the tensile mean exhausts UTS."""
    if s_uts <= 0.0:
        raise ValueError("s_uts must be positive")
    denom = 1.0 - float(mean_stress) / float(s_uts)
    if denom <= 0.0:
        return math.inf
    return abs(float(stress_range)) / denom


class SNCurveF3:
    """DNV-RP-C203 F3 (seawater + cathodic protection) with thickness effect."""

    def __init__(self, params: SNParams = BASELINE_SN) -> None:
        self.params = params

    # -- life ---------------------------------------------------------------
    def life(self, stress_range: float, thickness_mm: Optional[float] = None) -> float:
        """Cycles to failure at ``stress_range`` (MPa), thickness-corrected.

        Bilinear branch: the m1 segment (steep-stress, N <= N_knee) applies while
        the first-segment estimate is below the knee, i.e. for
        ``ds > ds_knee = 10^((log a1 - log10 N_knee)/m1)``; below that the m2
        segment is used.
        """
        tf = self.params.thickness_factor(thickness_mm)
        ds = abs(float(stress_range)) / tf if self.params.thickness_rule == "legacy_m5528" \
            else abs(float(stress_range)) * tf
        if ds <= 0.0:
            return math.inf
        p = self.params
        if ds > p.range_knee_effective:
            n = 10.0 ** (p.log_a1 - p.m1 * math.log10(ds))
        else:
            n = 10.0 ** (p.log_a2 - p.m2 * math.log10(ds))
        return float(min(n, p.n_cutoff))

    # -- equivalence helpers used by the tests ------------------------------
    def damage(
        self, stress_ranges: Sequence[float], counts: Sequence[float],
        means: Optional[Sequence[float]] = None, s_uts: Optional[float] = None,
        thickness_mm: Optional[float] = None, exposure_scale: float = 1.0,
    ) -> float:
        ranges = np.asarray(stress_ranges, dtype=float)
        cnt = np.asarray(counts, dtype=float)
        if means is not None and s_uts is not None:
            means_arr = np.asarray(means, dtype=float)
            denom = 1.0 - means_arr / float(s_uts)
            ranges = np.where(denom > 0.0, ranges / np.where(denom > 0.0, denom, 1.0), np.inf)
        total = 0.0
        for r, c in zip(ranges, cnt):
            n_allow = self.life(r, thickness_mm)
            if math.isfinite(n_allow) and n_allow > 0.0:
                total += c * exposure_scale / n_allow
        return float(total)

    def single_slope_constant(self, m: float = 3.0) -> float:
        """``C`` such that ``N = C S^{-m}`` matches the F3/CP curve above the knee.

        Used as the *law-study* S-N (one slope, no knee) so that the spectral
        closures (Rayleigh/Dirlik/Tovo-Benasciutti, all of the form
        ``D = nu E[S^m] / C``) are exactly comparable to the rainflow truth.
        """
        if abs(m - self.params.m1) > 1e-12:
            raise ValueError("single-slope C is only defined for m = m1 = 3")
        return 10.0 ** self.params.log_a1


def miner_damage(
    curve: SNCurveF3,
    stress_ranges: Sequence[float],
    counts: Sequence[float],
    means: Optional[Sequence[float]] = None,
    s_uts: Optional[float] = None,
    thickness_mm: Optional[float] = None,
    exposure_scale: float = 1.0,
) -> Dict[str, float]:
    """Σ n_i / N_i with optional Goodman correction, thickness and exposure scaling."""
    D = curve.damage(stress_ranges, counts, means, s_uts, thickness_mm, exposure_scale)
    return {
        "damage": D,
        "exposure_scale": float(exposure_scale),
        "n_cycles": float(np.sum(np.asarray(counts, dtype=float)) * exposure_scale),
    }