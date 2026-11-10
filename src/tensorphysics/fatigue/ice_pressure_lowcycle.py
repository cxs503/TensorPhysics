"""Ice-pressure cyclic load spectrum to low-cycle fatigue damage.

Chain
-----
1. **Excitation.**  At ice-induced vibration lock-in the mode oscillates at
   ``f_n`` with a limit-cycle amplitude ``a`` (see
   :mod:`tensorphysics.ice_vibration.lockin_coupling`).  The harmonic reaction
   of the mode is ``F_amp = k a``, so the load *range* imposed on the loaded
   detail is ``dF = 2 k a`` and the number of cycles in an exposure ``T`` hours
   is ``n = [f_n * 3600 * T]``.
2. **Structural transfer.**  For a simply supported ice-belt frame of span
   ``L`` loaded at mid-span, ``M = dF L / 4``; the nominal (remote) stress range
   at the detail is ``dS = M / W`` with ``W`` the section modulus.  The transfer
   is purely geometric, so no empirical factor enters.
3. **Notch.**  ``dS`` is amplified to the notch root by the Neuber rule against
   the material's cyclic curve (:mod:`tensorphysics.fatigue.strain_life`), which
   conserves ``(K_t dS)^2/E = dsigma deps`` and therefore correctly bounds the
   notch strain when the root yields.
4. **Damage.**  Each pulse amplitude is turned into ``N_f`` on the
   epsilon-N curve and accumulated with Palmgren-Miner, ``D = sum n_i/N_f,i``.
   Blocks with ``N_f < N_T`` are LCF (strain-controlled), the rest HCF
   (stress-controlled); the split is reported because only the LCF part needs
   the full strain-life treatment.
5. **Design output.**  The allowable exposure ``T_allow`` for ``D = 1`` follows
   directly from the damage rate ``dD/dT``.

Ice crushing pulses are not of constant height; the pulse amplitude is modelled
as a unit-mean Weibull variate whose coefficient of variation is a declared
case input (a Rayleigh variate, ``CoV = 0.523``, is the default).  Because
``N_f ~ amplitude**(-1/b)`` is strongly convex, the amplitude spread raises the
accumulated damage above the constant-amplitude value -- the "spectrum effect"
that a deterministic single-amplitude check would miss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .strain_life import (
    DamageBlock,
    StrainLifeDamageResult,
    StrainLifeMaterial,
    cycles_to_failure,
    neuber_notch_strain,
    notch_factor_neuber,
    notch_factor_peterson,
    strain_life_damage,
    transition_life,
)

__all__ = [
    "RAYLEIGH_COV",
    "cycles_from_exposure",
    "nominal_stress_range_MPa",
    "weibull_shape_from_cov",
    "pulse_amplitude_blocks",
    "IcePressureLowCycleResult",
    "ice_pressure_low_cycle_assessment",
]

#: Coefficient of variation of a Rayleigh distributed pulse amplitude.
RAYLEIGH_COV = math.sqrt(4.0 / math.pi - 1.0)


def cycles_from_exposure(lockin_frequency_hz: float, exposure_hours: float) -> float:
    """Number of load cycles ``n = f * 3600 * T``."""
    if lockin_frequency_hz <= 0.0 or exposure_hours < 0.0:
        raise ValueError("frequency must be positive and exposure non-negative")
    return float(lockin_frequency_hz * 3600.0 * exposure_hours)


def nominal_stress_range_MPa(
    force_range_n: float, span_m: float, section_modulus_m3: float
) -> float:
    """Nominal stress range at mid-span of a simply supported beam [MPa].

    ``dS = (dF * L / 4) / W`` -- the classical mid-span moment of a central
    point load divided by the section modulus.
    """
    if force_range_n <= 0.0:
        raise ValueError("force_range_n must be positive")
    if span_m <= 0.0 or section_modulus_m3 <= 0.0:
        raise ValueError("span and section modulus must be positive")
    moment_range = force_range_n * span_m / 4.0
    return float(moment_range / section_modulus_m3 / 1.0e6)


def weibull_shape_from_cov(cov: float) -> float:
    """Weibull shape ``k`` whose coefficient of variation equals ``cov``.

    ``CoV^2 = Gamma(1 + 2/k) / Gamma(1 + 1/k)^2 - 1``; solved by bisection on
    ``log k`` (monotone decreasing in ``k``).  ``CoV -> 1`` corresponds to the
    exponential, ``k = 1``.
    """
    if not 0.0 < cov < 5.0:
        raise ValueError("cov must be in (0, 5)")

    def f(k: float) -> float:
        g1 = math.gamma(1.0 + 1.0 / k)
        g2 = math.gamma(1.0 + 2.0 / k)
        return math.sqrt(max(g2 / (g1 * g1) - 1.0, 0.0)) - cov

    lo, hi = 1e-2, 1e3
    # CoV decreases with k: CoV(lo) large, CoV(hi) small
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-12:
            break
    return float(math.sqrt(lo * hi))


def pulse_amplitude_blocks(
    cycles: float, cov: float = RAYLEIGH_COV, n_blocks: int = 6
) -> List[Tuple[float, float]]:
    """Discretise a unit-mean Weibull pulse-amplitude distribution.

    Returns ``[(amplitude_multiplier, cycles_in_block), ...]`` with equal
    probability mass per block and the block representative taken at the
    mid-quantile.  ``cov = RAYLEIGH_COV`` reproduces the Rayleigh amplitude
    distribution of a narrow-band Gaussian process.
    """
    if cycles < 0.0:
        raise ValueError("cycles must be non-negative")
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    k = weibull_shape_from_cov(cov)
    # unit-mean Weibull scale: mean = lambda * Gamma(1 + 1/k) = 1
    lam = 1.0 / math.gamma(1.0 + 1.0 / k)
    edges = np.linspace(0.0, 1.0, int(n_blocks) + 1)
    blocks: List[Tuple[float, float]] = []
    for i in range(int(n_blocks)):
        f_mid = 0.5 * (edges[i] + edges[i + 1])
        amp = lam * (-math.log(1.0 - min(f_mid, 1.0 - 1e-12))) ** (1.0 / k)
        blocks.append((float(amp), float(cycles / n_blocks)))
    return blocks


@dataclass
class IcePressureLowCycleResult:
    """Result of the ice-pressure -> notch strain -> epsilon-N chain."""

    material: str
    force_range_n: float
    lockin_frequency_hz: float
    exposure_hours: float
    cycles: float
    nominal_stress_range_MPa: float
    notch_radius_mm: float
    K_t: float
    K_f_neuber: Optional[float]
    K_f_peterson: Optional[float]
    notch: Dict[str, Any]
    damage: StrainLifeDamageResult
    transition_life: float
    damage_rate_per_hour: float
    allowable_hours: float
    primary_N_f: Optional[float]
    primary_regime: str
    pulse_cov: float
    blocks_multiplier_range: Tuple[float, float]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "material": self.material,
            "force_range_n": self.force_range_n,
            "lockin_frequency_hz": self.lockin_frequency_hz,
            "exposure_hours": self.exposure_hours,
            "cycles": self.cycles,
            "nominal_stress_range_MPa": self.nominal_stress_range_MPa,
            "notch_radius_mm": self.notch_radius_mm,
            "K_t": self.K_t,
            "K_f_neuber": self.K_f_neuber,
            "K_f_peterson": self.K_f_peterson,
            "notch": self.notch,
            "damage": self.damage.to_json_dict(),
            "transition_life": self.transition_life,
            "damage_rate_per_hour": self.damage_rate_per_hour,
            "allowable_hours": self.allowable_hours,
            "primary_N_f": self.primary_N_f,
            "primary_regime": self.primary_regime,
            "pulse_cov": self.pulse_cov,
            "blocks_multiplier_range": list(self.blocks_multiplier_range),
        }


def ice_pressure_low_cycle_assessment(
    material: StrainLifeMaterial,
    force_range_n: float,
    lockin_frequency_hz: float,
    exposure_hours: float,
    span_m: float,
    section_modulus_m3: float,
    K_t: float = 3.0,
    notch_radius_mm: float = 2.0,
    mean_stress_MPa: float = 0.0,
    pulse_cov: float = RAYLEIGH_COV,
    n_blocks: int = 6,
    damage_target: float = 1.0,
    runout_reversals: float = 2.0e8,
) -> IcePressureLowCycleResult:
    """Run the full ice-pressure low-cycle fatigue chain.

    The *primary* block is the unit-amplitude pulse (``multiplier = 1``, the
    mean pulse); the remaining blocks discretise the pulse-height distribution.
    ``allowable_hours`` is the exposure at which the accumulated damage reaches
    ``damage_target`` (1.0 for the Palmgren-Miner design check).
    """
    if exposure_hours < 0.0:
        raise ValueError("exposure_hours must be non-negative")
    base_stress = nominal_stress_range_MPa(force_range_n, span_m, section_modulus_m3)
    notch_primary = neuber_notch_strain(material, K_t, base_stress)
    total_cycles = cycles_from_exposure(lockin_frequency_hz, exposure_hours)

    blocks: List[DamageBlock] = []
    for amp_mult, n_cycles in pulse_amplitude_blocks(total_cycles, pulse_cov, n_blocks):
        notch = neuber_notch_strain(material, K_t, base_stress * amp_mult)
        blocks.append(
            DamageBlock(
                strain_amplitude=notch.strain_amplitude,
                cycles=n_cycles,
                mean_stress_MPa=mean_stress_MPa,
                label=f"pulse x{amp_mult:.3f}",
            )
        )
    damage = strain_life_damage(material, blocks, runout_reversals=runout_reversals)
    n_t = transition_life(material)
    damage_rate = damage.damage_total / exposure_hours if exposure_hours > 0.0 else math.inf
    allowable = (
        float(damage_target / damage_rate) if damage_rate > 0.0 else math.inf
    )
    primary_nf = cycles_to_failure(material, notch_primary.strain_amplitude, mean_stress_MPa)
    multipliers = [amp for amp, _ in pulse_amplitude_blocks(1.0, pulse_cov, n_blocks)]
    const = material.notch_constant_mm
    kf_neuber = (
        notch_factor_neuber(K_t, notch_radius_mm, const) if const is not None else None
    )
    kf_peterson = (
        notch_factor_peterson(K_t, notch_radius_mm, const) if const is not None else None
    )
    return IcePressureLowCycleResult(
        material=material.name,
        force_range_n=float(force_range_n),
        lockin_frequency_hz=float(lockin_frequency_hz),
        exposure_hours=float(exposure_hours),
        cycles=float(total_cycles),
        nominal_stress_range_MPa=float(base_stress),
        notch_radius_mm=float(notch_radius_mm),
        K_t=float(K_t),
        K_f_neuber=float(kf_neuber) if kf_neuber is not None else None,
        K_f_peterson=float(kf_peterson) if kf_peterson is not None else None,
        notch=notch_primary.to_json_dict(),
        damage=damage,
        transition_life=float(n_t),
        damage_rate_per_hour=float(damage_rate),
        allowable_hours=float(allowable),
        primary_N_f=float(primary_nf),
        primary_regime="LCF" if primary_nf < n_t else "HCF",
        pulse_cov=float(pulse_cov),
        blocks_multiplier_range=(min(multipliers), max(multipliers)),
    )