"""Pipe-span stress from foundation acceleration, rainflow and DNV fatigue life.

Chain step 3-4.  The pipe run is idealised as a simply supported span of length
``L`` carrying its own steel mass plus the contained medium; the equipment
foundation acceleration (output of :mod:`.transfer`) is applied as a *base
excitation*, i.e. as the equivalent uniform inertial load ``q(x, t) = m a_g(t)``.
Modal superposition of the simply supported bending modes
``phi_n = sin(n pi x / L)`` gives the complex stress transfer

.. math::

    T_\\sigma(f) = \\frac{\\sigma(f)}{a_g(f)}
    = -E \\frac{D}{2} \\sum_n \\frac{(n\\pi/L)^2 \\sin(n\\pi x_d/L)\\,
      \\int_0^L m\\phi_n\\,dx / M_n}{\\omega_n^2-\\omega^2+2 i \\zeta \\omega_n \\omega}

with ``M_n = m L/2`` and ``omega_n = (n pi/L)^2 sqrt(EI/m)``.  Two properties are
used as machine checks in the tests:

* the static limit ``T_sigma(0) = m L^2 / (8 Z)`` (mid-span moment of a uniform
  load), so the modal sum is validated against the closed-form static solution;
* the first span mode ``f_1 = (pi/2) (1/L^2) sqrt(EI/m)/... `` lands inside the
  ice band (typically 8-20 Hz), which is why pipe runs are the classic fatigue
  detail of a vibration environment.

The stress PSD follows as ``S_sigma(f) = |T_sigma(f)|^2 S_a(f)``.  A stationary
Gaussian time history with that PSD is synthesised (random-phase harmonic
superposition, deterministic seed), counted with the ASTM E1049-85 four-point
rainflow algorithm of :mod:`tensorphysics.fatigue.rainflow`, and accumulated with
Palmgren-Miner against the DNV-RP-C203 D curve in seawater with cathodic
protection, including the SCF of the welded detail and the DNV thickness
correction ``S = S0 (t/t_ref)^k``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..fatigue.miner import miner_damage
from ..fatigue.rainflow import rainflow_counting
from ..fatigue.sn_curve import DNCurve

__all__ = [
    "PipeSpan",
    "SECONDS_PER_YEAR",
    "fatigue_life",
    "narrowband_damage_rate",
    "synthesise_stress_history",
]

SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


@dataclass
class PipeSpan:
    """Simply supported pipe span with contents (SI units)."""

    length_m: float = 8.0
    outer_diameter_m: float = 0.4064
    wall_thickness_m: float = 0.0127
    rho_steel_kg_m3: float = 7850.0
    content_density_kg_m3: float = 1000.0
    young_pa: float = 2.06e11
    zeta: float = 0.010
    n_modes: int = 7
    scf: float = 1.80
    thickness_exponent: float = 0.20
    reference_thickness_m: float = 0.025
    detail_fraction_of_span: float = 0.5

    def __post_init__(self) -> None:
        if self.length_m <= 0.0 or self.outer_diameter_m <= 0.0 or self.wall_thickness_m <= 0.0:
            raise ValueError("pipe span dimensions must be positive")
        if not 2.0 * self.wall_thickness_m < self.outer_diameter_m:
            raise ValueError("wall thickness must be less than half the outer diameter")
        if not 0.0 < self.zeta < 1.0:
            raise ValueError("pipe damping ratio must satisfy 0 < zeta < 1")
        if self.n_modes < 1:
            raise ValueError("at least one mode is required")

    # --------------------------------------------------------------- section
    @property
    def inner_diameter_m(self) -> float:
        return self.outer_diameter_m - 2.0 * self.wall_thickness_m

    @property
    def steel_area_m2(self) -> float:
        return float(np.pi / 4.0 * (self.outer_diameter_m**2 - self.inner_diameter_m**2))

    @property
    def bore_area_m2(self) -> float:
        return float(np.pi / 4.0 * self.inner_diameter_m**2)

    @property
    def second_moment_m4(self) -> float:
        return float(np.pi / 64.0 * (self.outer_diameter_m**4 - self.inner_diameter_m**4))

    @property
    def section_modulus_m3(self) -> float:
        return float(2.0 * self.second_moment_m4 / self.outer_diameter_m)

    @property
    def mass_per_length_kg_m(self) -> float:
        return float(
            self.rho_steel_kg_m3 * self.steel_area_m2 + self.content_density_kg_m3 * self.bore_area_m2
        )

    @property
    def stiffness_ei_nm2(self) -> float:
        return float(self.young_pa * self.second_moment_m4)

    def omega_n(self, n: int) -> float:
        """Simply supported bending eigenvalue ``omega_n`` [rad/s]."""
        return float((n * np.pi / self.length_m) ** 2 * np.sqrt(self.stiffness_ei_nm2 / self.mass_per_length_kg_m))

    @property
    def first_frequency_hz(self) -> float:
        return float(self.omega_n(1) / (2.0 * np.pi))

    @property
    def detail_station_m(self) -> float:
        return float(self.detail_fraction_of_span * self.length_m)

    @property
    def static_stress_per_acceleration(self) -> float:
        """Closed-form static limit ``m L^2 / (8 Z)`` [Pa per m/s^2]."""
        return float(self.mass_per_length_kg_m * self.length_m**2 / (8.0 * self.section_modulus_m3))

    # ------------------------------------------------------------ transfer
    def stress_transfer(self, freqs: np.ndarray, x_detail: float | None = None) -> np.ndarray:
        """Complex stress transfer [Pa per m/s^2] from base acceleration."""
        f = np.asarray(freqs, dtype=float)
        xd = self.detail_station_m if x_detail is None else float(x_detail)
        omega = 2.0 * np.pi * f
        m = self.mass_per_length_kg_m
        L = self.length_m
        acc = np.zeros(f.size, dtype=complex)
        for n in range(1, self.n_modes + 1):
            wn = self.omega_n(n)
            q = n * np.pi / L
            modal_force = m * (1.0 - np.cos(n * np.pi)) / q        # int_0^L m phi_n dx
            modal_mass = m * L / 2.0
            denom = wn**2 - omega**2 + 2j * self.zeta * wn * omega
            acc += q**2 * np.sin(q * xd) * (modal_force / modal_mass) / denom
        return -(self.young_pa * self.outer_diameter_m / 2.0) * acc

    def peak_frequency_hz(self, s_acceleration: np.ndarray, freqs: np.ndarray) -> float:
        s_sigma = np.abs(self.stress_transfer(freqs)) ** 2 * np.asarray(s_acceleration, dtype=float)
        return float(np.asarray(freqs, dtype=float)[int(np.argmax(s_sigma))])

    # ------------------------------------------------------------- detail
    def effective_stress_range(self, stress_range_pa: np.ndarray | float) -> np.ndarray:
        """SCF + DNV-RP-C203 thickness correction, returned in MPa.

        ``S_eff = S * SCF * (t / t_ref)^k`` (DNV-RP-C203 thickness correction for
        the detail thickness ``t``); the DNV S-N curves consume MPa.
        """
        factor = self.scf * (self.wall_thickness_m / self.reference_thickness_m) ** self.thickness_exponent
        return np.asarray(stress_range_pa, dtype=float) * factor / 1.0e6

    def as_dict(self) -> dict[str, Any]:
        return {
            "length_m": float(self.length_m),
            "outer_diameter_m": float(self.outer_diameter_m),
            "wall_thickness_m": float(self.wall_thickness_m),
            "inner_diameter_m": float(self.inner_diameter_m),
            "z_m3": float(self.section_modulus_m3),
            "i_m4": float(self.second_moment_m4),
            "mass_per_length_kg_m": float(self.mass_per_length_kg_m),
            "content_density_kg_m3": float(self.content_density_kg_m3),
            "EI_Nm2": float(self.stiffness_ei_nm2),
            "first_frequency_hz": float(self.first_frequency_hz),
            "zeta": float(self.zeta),
            "n_modes": int(self.n_modes),
            "scf": float(self.scf),
            "thickness_exponent": float(self.thickness_exponent),
            "reference_thickness_m": float(self.reference_thickness_m),
            "static_stress_per_acceleration_Pa_per_m_s2": float(self.static_stress_per_acceleration),
            "detail_station_m": float(self.detail_station_m),
        }


def synthesise_stress_history(
    freqs: np.ndarray,
    s_stress_pa2_per_hz: np.ndarray,
    duration_s: float,
    dt_s: float,
    seed: int = 12345,
    f_max_hz: float | None = None,
) -> np.ndarray:
    """Random-phase harmonic superposition of a one-sided stress PSD.

    ``sigma(t) = sum_k sqrt(2 S(f_k) df) cos(2 pi f_k t + phi_k)`` with uniform
    random phases -- the standard Shinozuka stationary-Gaussian realisation, whose
    PSD converges to the target PSD as ``df -> 0``.  Deterministic for a given
    ``seed``.  The DC bin is dropped (zero-mean stress process).
    """
    f = np.asarray(freqs, dtype=float)
    s = np.asarray(s_stress_pa2_per_hz, dtype=float)
    if f.shape != s.shape:
        raise ValueError("freqs and stress PSD must have the same shape")
    n_steps = int(round(duration_s / dt_s))
    if n_steps < 1024:
        raise ValueError("history too short for rainflow counting")
    f_hi = float(f[-1]) if f_max_hz is None else min(float(f_max_hz), float(f[-1]))
    df = float(f[1] - f[0]) if f.size > 1 else 0.0
    mask = (f > 0.0) & (f <= f_hi)
    fk = f[mask]
    amp = np.sqrt(2.0 * np.maximum(s[mask], 0.0) * df)
    rng = np.random.default_rng(int(seed))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=fk.size)
    t = np.arange(n_steps) * dt_s
    # accumulate in blocks to bound the temporary memory of the outer product
    sigma = np.zeros(n_steps)
    block = 256
    for start in range(0, fk.size, block):
        sl = slice(start, start + block)
        sigma += np.cos(2.0 * np.pi * np.outer(t, fk[sl]) + phase[sl]) @ amp[sl]
    return sigma


def narrowband_damage_rate(
    sigma_rms_pa: float,
    zero_crossing_hz: float,
    sn_curve: DNCurve,
    scf_factor: float = 1.0,
) -> float:
    """Damage per second of a narrowband Gaussian process (Rayleigh ranges).

    Closed-form cross-check for the rainflow result.  For a narrowband process the
    peaks ``a`` are Rayleigh distributed with parameter ``sigma``; the *stress
    range* is ``S = 2a`` and therefore Rayleigh distributed with parameter
    ``2 sigma``, i.e. ``p(S) = S/(4 sigma^2) exp(-S^2/(8 sigma^2))``.  The damage
    rate is ``f_z * integral_0^inf p(S) / N(S) dS``.
    """
    if sigma_rms_pa <= 0.0 or zero_crossing_hz <= 0.0:
        return 0.0
    sigma_r = 2.0 * sigma_rms_pa
    s = np.linspace(1e-3, 12.0 * sigma_r, 4000)
    p = s / sigma_r**2 * np.exp(-(s**2) / (2.0 * sigma_r**2))   # Rayleigh pdf of the range
    ranges_mpa = s * scf_factor / 1.0e6
    vector = np.vectorize(lambda v: sn_curve.n_to_failure(float(v)))
    n_fail = vector(ranges_mpa)
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where(np.isfinite(n_fail) & (n_fail > 0.0), p / n_fail, 0.0)
    return float(zero_crossing_hz * np.trapezoid(integrand, s))


def fatigue_life(
    s_stress_pa2_per_hz: np.ndarray,
    freqs: np.ndarray,
    span: PipeSpan,
    sn_curve: DNCurve,
    duration_s: float,
    dt_s: float,
    operating_fraction: float = 0.25,
    seed: int = 12345,
) -> dict[str, Any]:
    """Rainflow + S-N + Miner damage and the resulting fatigue life.

    Returns the damage *rate* (per second and per year of operation), the
    equivalent constant-amplitude stress range at 1e7 cycles, the rainflow
    histogram summary, and the cross-check against the narrowband closed form.
    """
    sigma = synthesise_stress_history(freqs, s_stress_pa2_per_hz, duration_s, dt_s, seed=seed)
    rf = rainflow_counting(sigma)
    scf_factor = float(
        span.scf * (span.wall_thickness_m / span.reference_thickness_m) ** span.thickness_exponent
    )
    ranges_mpa = span.effective_stress_range(rf.ranges)
    keep = ranges_mpa > 0.0
    miner = miner_damage(ranges_mpa[keep], rf.counts[keep], sn_curve) if np.any(keep) else None
    damage = float(miner.total_damage) if miner is not None else 0.0
    damage_rate = damage / float(duration_s)
    damage_per_year = damage_rate * SECONDS_PER_YEAR * float(operating_fraction)
    life_years = float("inf") if damage_per_year <= 0.0 else float(1.0 / damage_per_year)

    # spectral moments of the (already SCF-corrected) stress process
    f = np.asarray(freqs, dtype=float)
    s = np.asarray(s_stress_pa2_per_hz, dtype=float)
    m0 = float(np.trapezoid(s, f))
    m2 = float(np.trapezoid((2.0 * np.pi * f) ** 2 * s, f))
    n_target = 1.0e7
    sigma_rms = float(np.sqrt(max(m0, 0.0)))
    zero_cross = float(np.sqrt(m2 / m0) / (2.0 * np.pi)) if m0 > 0.0 else 0.0
    nb_rate = narrowband_damage_rate(sigma_rms, zero_cross, sn_curve, scf_factor=scf_factor)
    nb_life = float("inf") if nb_rate <= 0.0 else 1.0 / (nb_rate * SECONDS_PER_YEAR * operating_fraction)

    # equivalent constant-amplitude stress range at 1e7 cycles (DNV "equivalent
    # stress"): D = N_eq / N(s_eq) with N_eq = 1e7  =>  s_eq = N^-1(1e7 / D)
    s_eq_mpa = float(sn_curve.stress_for_life(n_target / damage)) if damage > 0.0 else None

    # coarse rainflow histogram for the report figure (30 bins, MPa)
    hist_counts, hist_edges = (
        np.histogram(ranges_mpa[keep], bins=30) if np.any(keep) else (np.zeros(30), np.zeros(31))
    )
    return {
        "rainflow_histogram": {
            "range_MPa": [float(v) for v in 0.5 * (hist_edges[1:] + hist_edges[:-1])],
            "cycles": [float(v) for v in hist_counts],
        },
        "stress_rms_MPa": float(sigma_rms * scf_factor / 1.0e6),
        "stress_rms_raw_MPa": float(sigma_rms / 1.0e6),
        "zero_crossing_hz": zero_cross,
        "max_stress_range_MPa": float(np.max(ranges_mpa)) if ranges_mpa.size else 0.0,
        "rainflow_cycles": float(rf.total_cycles),
        "rainflow_n_ranges": int(ranges_mpa.size),
        "damage": damage,
        "damage_rate_per_s": damage_rate,
        "damage_per_year": damage_per_year,
        "life_years": life_years,
        "operating_fraction": float(operating_fraction),
        "equivalent_stress_range_1e7_MPa": s_eq_mpa,
        "narrowband_crosscheck": {
            "damage_per_year": float(nb_rate * SECONDS_PER_YEAR * operating_fraction),
            "life_years": nb_life,
            "rainflow_to_narrowband_ratio": float(damage_per_year / (nb_rate * SECONDS_PER_YEAR * operating_fraction))
            if nb_rate > 0.0
            else float("nan"),
        },
        "scf_factor": scf_factor,
        "synthesis": {
            "duration_s": float(duration_s),
            "dt_s": float(dt_s),
            "samples": int(sigma.size),
            "seed": int(seed),
            "sigma_of_history_MPa": float(np.std(sigma) / 1.0e6),
        },
    }