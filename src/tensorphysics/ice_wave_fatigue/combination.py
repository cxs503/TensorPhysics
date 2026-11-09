"""Ice + wave combined fatigue: combination law, spectral methods, whipping bookkeeping.

Physical questions of node ``P82W4-N06`` and how each is answered here
---------------------------------------------------------------------
(a) **Is the combined damage the sum of the two damages?**  Two stationary
    stress components (an ice-crushing band at ``f_ice`` and a wave band at
    ``f_w``) are superposed and counted with rainflow.  The ratio
    ``rho = D_comb/(D_ice + D_wave)`` is compared against the predeclared band
    ``[0.97, 1.03]`` as a function of the band-separation ratio
    ``f_ice/f_w in {0.3, 1, 10, 100}``.  Where the sum rule fails the code
    supplies a *one-parameter nonlinear combination law* interpolating between
    the two exact limits

        separated bands (rainflow cycles do not interact):  D = D1 + D2
        merged bands (amplitudes add inside one cycle):    D = D_TB(S1 + S2)

    with the weight given by the *spectral overlap* of the two bands
    (:func:`overlap_weight`).  The mechanism is amplitude superposition versus
    cycle additivity, i.e. the variance-sum convexity of ``D ~ sigma^m`` for
    ``m > 1``.
(b) **Spectral-method capability**: Dirlik (``tensorphysics.fatigue``),
    Tovo-Benasciutti and the narrow-band Rayleigh formula are compared with the
    rainflow truth of histories synthesised from *the same* band-limited PSD, on
    three declared spectral-width levels plus a wide bimodal case.
(c) **Whipping bookkeeping**: the m5528 whipping share ``s`` (dynamic part of the
    ice stress) is related to the damage bias obtained when the high-frequency
    whipping component is counted twice, ``A = 1 + a``, ``a = (s/(1-s))**m``.
(d) **Rainflow commutability**: segmentation with residue carry-over, and
    permutation of closed excursion blocks.
(e) **Negative control**: waves off must reproduce the m5528 (IN32-N07) damage.

Conventions
-----------
* Stress in MPa (hot-spot, ``scf`` applied) unless a name says otherwise.
* Single-slope S-N ``N = C S^-m`` with ``C = 1``, ``m = 3`` for every model
  comparison (rainflow, Dirlik, Tovo-Benasciutti, Rayleigh), so that only the
  *counting* differs; ``D`` then carries units of MPa^3 and only ratios of it
  are ever reported.
* Physical (Level B) cases use the DNV-RP-C203 bilinear curve + Goodman from
  :mod:`sn`.  The two damage models are never mixed inside one comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tensorphysics.fatigue.spectral import (
    benasciutti_tovo_damage_rate,
    dirlik_damage_rate,
    moments_from_dict,
    rayleigh_damage_rate,
    spectral_moments as _lib_spectral_moments,
    synthesize_gaussian_history,
)

from . import sn as snmod
from .beam import (
    BeamModes,
    modal_stress_history,
    quasi_static_moment_free_free,
    whipping_share,
)
from .dem import (
    CALIBRATED_M5528,
    IceContactParams,
    generate_ice_force,
    load_statistics,
    scaled_drift_rate,
)
from .rainflow_stream import (
    StreamingRainflow,
    damage_from_cycles,
    equivalent_stress_range,
    rainflow_cycles,
    turning_points,
)
from .waves import (
    bimodal_f,
    jonswap_f,
    normalised_psd,
    spectral_width_measures,
    wave_stress_psd,
)

#: Default ``kappa = 1 + a r^b N^c`` constants, produced by
#: :func:`calibrate_combination_law` on the m7361 calibration grid
#: (band ratios 10 and 100, four amplitude ratios each).  Overwritten at run time
#: by the pipeline with the freshly calibrated values that are stored in
#: ``artifacts/m7361/results.json``.
KAPPA_A_DEFAULT = 1.1406
KAPPA_B_DEFAULT = 1.3253
KAPPA_C_DEFAULT = 0.3850

__all__ = [
    "HullGirderCase",
    "SeaState",
    "ice_stress_history",
    "stress_spectrum_shape",
    "gaussian_band_shape",
    "component_grid",
    "physical_wave_component",
    "equal_damage_sigma",
    "overlap_weight",
    "combination_case",
    "combination_law",
    "calibrate_combination_law",
    "evaluate_combination_law",
    "kappa_enlargement",
    "upcrossing_rate",
    "stretched_ice_stress",
    "harmonic_ice_stress",
    "riding_amplification",
    "KAPPA_A_DEFAULT",
    "KAPPA_B_DEFAULT",
    "KAPPA_C_DEFAULT",
    "mechanism_diagnostic",
    "band_separation_scan",
    "spectral_method_benchmark",
    "WIDTH_LEVELS",
    "whipping_accounting",
    "segmentation_invariance",
    "m5528_reproduction",
]


# ---------------------------------------------------------------------------
# Case definition (m5528 IN32-N07 hull girder + DNV F3 detail)
# ---------------------------------------------------------------------------
@dataclass
class HullGirderCase:
    """Hull-girder and detail parameters of the m5528 reference case."""

    length: float = 250.0
    EI: float = 5.0e13
    mass_per_length: float = 5.2e5
    z_deck: float = 10.5
    E: float = 2.06e11
    n_modes: int = 10
    damping_ratio: float = 0.015
    x_load_frac: float = 0.05
    x_detail_frac: float = 0.30
    scf: float = 2.5
    thickness: float = 28.0
    s_uts: float = 490.0
    sn_curve: str = "DNV-F3-sw-cp"
    ice_hours_per_year: float = 2500.0
    breadth: float = 40.0
    draught: float = 14.0
    ice_band_hz: float = 0.43
    _modes: Optional[BeamModes] = field(default=None, repr=False, compare=False)

    @property
    def x_load(self) -> float:
        return self.x_load_frac * self.length

    @property
    def x_detail(self) -> float:
        return self.x_detail_frac * self.length

    @property
    def modes(self) -> BeamModes:
        if self._modes is None:
            self._modes = BeamModes(self.length, self.EI, self.mass_per_length,
                                    n_modes=self.n_modes,
                                    damping_ratio=self.damping_ratio)
        return self._modes

    @property
    def f1_hz(self) -> float:
        return float(self.modes.freq[0])

    @property
    def f_cut_hz(self) -> float:
        return 0.5 * self.f1_hz

    @property
    def z_deck_from_EI(self) -> float:
        return self.EI / self.E / self.z_deck

    def curve(self) -> snmod.SNCurve:
        return snmod.get_curve(self.sn_curve)


@dataclass
class SeaState:
    """JONSWAP or bimodal sea state (``tp`` is the wind-sea peak period)."""

    hs: float = 4.0
    tp: float = 9.0
    gamma: float = 3.3
    swell_fraction: float = 0.5
    peak_separation: float = 2.0
    swell_gamma: float = 6.0
    bimodal: bool = False

    def elevation(self) -> Callable[[np.ndarray], np.ndarray]:
        if self.bimodal:
            return lambda f: bimodal_f(f, self.hs, self.tp, self.gamma,
                                       swell_energy_fraction=self.swell_fraction,
                                       peak_separation=self.peak_separation,
                                       swell_gamma=self.swell_gamma)
        return lambda f: jonswap_f(f, self.hs, self.tp, self.gamma)

    @property
    def f_peak_hz(self) -> float:
        return 1.0 / (self.tp * (self.peak_separation if self.bimodal else 1.0))

    def label(self) -> str:
        if self.bimodal:
            return (f"bimodal(Hs={self.hs:g}m,Tp={self.tp:g}s,"
                    f"r_p={self.peak_separation:g},swell_frac={self.swell_fraction:g})")
        return f"JONSWAP(Hs={self.hs:g}m,Tp={self.tp:g}s,gamma={self.gamma:g})"


# ---------------------------------------------------------------------------
# Ice leg
# ---------------------------------------------------------------------------
def ice_stress_history(
    case: HullGirderCase,
    params: IceContactParams = CALIBRATED_M5528,
    duration: Optional[float] = None,
    dt: Optional[float] = None,
    settle_time: float = 60.0,
) -> Dict[str, Any]:
    """Ice force -> modal superposition -> hot-spot stress history [MPa].

    ``settle_time`` drops the start-up transient exactly as in m5528: a constant
    crushing load step rings the 1.5 %-damped first mode for ~19 s, and without
    this cut the whipping share reads 70-100 % (the m5528 trap).
    """
    dt = float(dt if dt is not None else 2.0e-3)
    duration = float(duration if duration is not None else 300.0)
    t, F = generate_ice_force(params, duration=duration, dt=dt)
    res = modal_stress_history(case.modes, F, dt, case.x_load, case.x_detail,
                               case.z_deck, case.E)
    skip = min(int(settle_time / dt), max(0, F.size // 2))
    sigma = res["sigma"][skip:] * case.scf / 1e6
    sigma_qs = res["sigma_qs"][skip:] * case.scf / 1e6
    ws = whipping_share(sigma, sigma_qs)
    st = load_statistics(F[skip:], dt, case.f_cut_hz)
    m_static = quasi_static_moment_free_free(np.array([case.x_detail]), st["peak"],
                                             case.x_load, case.length)[0]
    return {
        "t": t[skip:],
        "dt": dt,
        "sigma_mpa": sigma,
        "sigma_qs_mpa": sigma_qs,
        "whipping": ws,
        "load_statistics": st,
        "force_peak_MN": st["peak"] / 1e6,
        "force_mean_MN": st["mean"] / 1e6,
        "sigma_rms_mpa": float(np.std(sigma)),
        "sigma_peak_mpa": float(np.max(np.abs(sigma))),
        "static_moment_MNm": float(m_static) / 1e6,
        "static_stress_mpa": float(m_static) / case.z_deck / 1e6,
        "f_pulse_hz": params.pulse_rate_hz(),
        "pulse_rate_hz": st["pulse_rate"],
    }# ---------------------------------------------------------------------------
# Stress-space spectra and synthesis (Level A: controlled band position/width)
# ---------------------------------------------------------------------------
def stress_spectrum_shape(kind: str, f_peak: float, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """Unit-variance stress-PSD shape (JONSWAP-shaped or bimodal).

    The shape is normalised so that ``int shape df = 1``; multiplying by
    ``sigma**2`` gives a stress PSD with ``m0 = sigma^2``.  This is the
    "RAO amplitude folded into the spectrum" representation: the hull-girder RAO
    is flat across the wave band for ``f_w << f_1``, so a JONSWAP-shaped *stress*
    spectrum is the physically relevant idealisation, and it lets band position
    and width be controlled independently of the sea state.  The physical chain
    with the frequency-dependent RAO is exercised by
    :func:`physical_wave_component`.
    """
    if f_peak <= 0:
        raise ValueError("f_peak must be positive")
    tp = 1.0 / f_peak
    f_max = max(12.0 * f_peak, 2.5)
    fg = np.linspace(1e-6, f_max, 40001)
    if kind == "jonswap":
        raw = jonswap_f(fg, hs=4.0, tp=tp, gamma=float(kwargs.get("gamma", 3.3)))
    elif kind == "bimodal":
        raw = bimodal_f(fg, hs=4.0, tp=tp, gamma=float(kwargs.get("gamma", 3.3)),
                        swell_energy_fraction=float(kwargs.get("swell_fraction", 0.5)),
                        peak_separation=float(kwargs.get("peak_separation", 2.0)),
                        swell_gamma=float(kwargs.get("swell_gamma", 6.0)))
    else:
        raise ValueError("kind must be 'jonswap' or 'bimodal'")
    m0 = float(np.trapezoid(raw, fg))
    if m0 <= 0:
        raise ValueError("degenerate spectrum shape")
    unit = raw / m0

    def shape(f):
        return np.interp(np.asarray(f, dtype=float), fg, unit, left=0.0, right=0.0)

    return shape


def gaussian_band_shape(f_peak: float, rel_bandwidth: float = 0.15,
                        n_grid: int = 20001) -> Callable[[np.ndarray], np.ndarray]:
    """Unit-variance Gaussian band stress-PSD shape centred on ``f_peak``.

    Represents the ice-crushing band: a quasi-harmonic process whose irregularity
    is set by ``rel_bandwidth = b / f_peak`` (``b`` = Gaussian sigma in Hz).
    """
    if f_peak <= 0 or rel_bandwidth <= 0:
        raise ValueError("f_peak and rel_bandwidth must be positive")
    b = rel_bandwidth * f_peak
    f_max = f_peak + 8.0 * b
    fg = np.linspace(1e-6, f_max, n_grid)
    raw = np.exp(-((fg - f_peak) ** 2) / (2.0 * b ** 2))
    m0 = float(np.trapezoid(raw, fg))
    unit = raw / m0

    def shape(f):
        return np.interp(np.asarray(f, dtype=float), fg, unit, left=0.0, right=0.0)

    return shape


def _grid_from_callable(psd: Callable, n_samples: int, fs: float,
                        f_hi: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample a PSD callable on the rFFT grid of the record (one-sided).

    ``tensorphysics.fatigue.spectral.discrete_psd_grid`` accepts an array or a
    spectrum *object* exposing ``band()``; a plain callable is rejected there, so
    the grid is built here explicitly.  ``f = 0`` and the Nyquist bin are zeroed
    (no DC, no half-cosine at fs/2) exactly like the library routine.
    """
    df = fs / n_samples
    freqs = np.arange(n_samples // 2 + 1, dtype=float) * df
    s = np.asarray(psd(freqs), dtype=float)
    if s.shape != freqs.shape:
        raise ValueError("PSD callable must return one value per frequency")
    s = np.where(np.isfinite(s), s, 0.0)
    s[0] = 0.0
    if f_hi is not None:
        s = np.where(freqs <= float(f_hi), s, 0.0)
    if n_samples % 2 == 0:
        s[-1] = 0.0
    return freqs, s, df


def component_grid(shape: Callable, sigma_rms: float, n_samples: int,
                   fs: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Band-limited PSD on the rFFT grid of a record (``m0 = sigma^2``)."""
    psd = lambda f: sigma_rms ** 2 * shape(f)
    return _grid_from_callable(psd, n_samples, fs)


def equal_damage_sigma(f_peak: float, damage_rate_target: float, m: float = 3.0) -> float:
    """Component rms that yields a narrow-band target damage rate.

    ``D_rate = nu0 sigma^m (2 sqrt 2)^m Gamma(1 + m/2)`` with ``nu0 ~ f_peak``.
    Puts the two bands on an *equal per-component damage* footing, which is what
    makes the sum-rule test informative: with equal rms the low-frequency band
    would always lose on ``sigma^m``.
    """
    gamma_f = math.gamma(1.0 + 0.5 * m)
    return float((damage_rate_target
                  / (f_peak * (2.0 * math.sqrt(2.0)) ** m * gamma_f)) ** (1.0 / m))


def overlap_weight(freqs: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> float:
    """Spectral overlap of two stress components in ``[0, 1]``.

    ``beta = int min(S1, S2) df / min(int S1 df, int S2 df)``: 0 for disjoint
    bands (rainflow cycles stay separate), 1 for identical bands (amplitudes add
    inside one cycle).
    """
    f = np.asarray(freqs, float)
    a = np.asarray(s1, float)
    b = np.asarray(s2, float)
    inter = float(np.trapezoid(np.minimum(a, b), f))
    denom = min(float(np.trapezoid(a, f)), float(np.trapezoid(b, f)))
    return 0.0 if denom <= 0 else float(np.clip(inter / denom, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Physical wave leg (Level B: sea state -> RAO -> stress history)
# ---------------------------------------------------------------------------
def physical_wave_component(
    case: HullGirderCase,
    sea: SeaState,
    n_samples: int,
    fs: float,
    target_rms: Optional[float] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Wave-induced detail stress for a real sea state, with the actual RAO.

    ``target_rms=None`` keeps the strip-theory amplitude (reported as
    ``raw_rms_mpa`` together with the implied gain if a target is imposed), so the
    difference between the strip level and a design level stays visible.
    """
    f_max = min(0.5 * fs, 2.0)
    freqs = np.linspace(1e-4, f_max, 2001)
    transfer = wave_stress_psd(freqs, case.modes, sea.elevation(), case.x_detail,
                               case.z_deck, case.E, case.length, case.breadth, case.draught)
    s_sigma = transfer["s_sigma"] / 1e12                       # Pa^2/Hz -> MPa^2/Hz
    raw_rms = math.sqrt(max(float(np.trapezoid(s_sigma, freqs)), 0.0))
    gain = 1.0
    if target_rms is not None:
        s_sigma, gain = normalised_psd(freqs, s_sigma, target_rms)
    m0 = float(np.trapezoid(s_sigma, freqs))
    unit = s_sigma / m0 if m0 > 0 else s_sigma * 0.0

    def shape(f):
        return np.interp(np.asarray(f, dtype=float), freqs, unit, left=0.0, right=0.0)

    fgrid, s_band, df = _grid_from_callable(lambda f: m0 * shape(f), n_samples, fs)
    hist = synthesize_gaussian_history(s_band, n_samples, fs, seed=seed)
    exposure = n_samples / fs
    r, mn, c = rainflow_cycles(turning_points(hist))
    return {
        "sigma_mpa": hist,
        "freqs_hz": fgrid,
        "psd_mpa2_hz": s_band,
        "sigma_rms_mpa": float(np.std(hist)),
        "psd_sigma_rms_mpa": float(math.sqrt(max(float(np.trapezoid(s_band, fgrid)), 0.0))),
        "raw_rms_mpa": raw_rms,
        "gain": gain,
        "width": spectral_width_measures(fgrid, s_band),
        "duration_s": exposure,
        "n_cycles": float(np.sum(c)),
        "damage_rate_m3": damage_from_cycles(r, c, 3.0) / exposure,
        "transfer_rms": float(math.sqrt(max(float(np.trapezoid(
            np.abs(transfer["transfer_pa_per_m"]) ** 2, freqs)), 0.0))),
        "sea_state": sea.label(),
        "f_peak_hz": sea.f_peak_hz,
    }# ---------------------------------------------------------------------------
# (a) one realisation of the superposition and the band-separation scan
# ---------------------------------------------------------------------------
def combination_case(
    case: HullGirderCase,
    ratio: float,
    wave_kind: str = "jonswap",
    wave_kwargs: Optional[Dict[str, float]] = None,
    amplitude_mode: str = "equal_damage",
    seed: int = 0,
    duration: float = 1800.0,
    fs: float = 40.0,
    sigma_ratio: float = 0.6,
    d_ice_rate: float = 1.0,
    m: float = 3.0,
) -> Dict[str, Any]:
    """Superpose an ice band and a wave band with ``f_ice / f_w = ratio``.

    Both components are Gaussian, band-limited and zero-mean; the combined
    history is their pointwise sum.  For a linear structural system the
    superposition itself is exact (the response to a summed load equals the sum
    of the responses), so what is under test is only the *damage* combination
    rule, not the stress superposition.
    """
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    f_ice = case.ice_band_hz
    f_w = f_ice / float(ratio)
    n = int(round(duration * fs))
    wk = dict(wave_kwargs or {})

    ice_shape = gaussian_band_shape(f_ice, 0.15)
    wave_shape = stress_spectrum_shape(wave_kind, f_w, **wk)

    sigma_ice = equal_damage_sigma(f_ice, d_ice_rate, m)
    if amplitude_mode == "equal_damage":
        sigma_wave = equal_damage_sigma(f_w, d_ice_rate, m)
    elif amplitude_mode == "equal_rms":
        sigma_wave = sigma_ratio * sigma_ice
    else:
        raise ValueError("amplitude_mode must be 'equal_damage' or 'equal_rms'")

    grid, s_ice, _ = component_grid(ice_shape, sigma_ice, n, fs)
    _, s_wave, _ = component_grid(wave_shape, sigma_wave, n, fs)
    h_ice = synthesize_gaussian_history(s_ice, n, fs, seed=seed)
    h_wave = synthesize_gaussian_history(s_wave, n, fs, seed=seed + 1000)
    h_comb = h_ice + h_wave
    exposure = n / fs

    def _rate(h):
        r, mn, c = rainflow_cycles(turning_points(h))
        return damage_from_cycles(r, c, m) / exposure, r, mn, c

    d_ice = _rate(h_ice)[0]
    d_wave = _rate(h_wave)[0]
    d_comb, r_c, mn_c, cnt_c = _rate(h_comb)

    mom_comb = _lib_spectral_moments(grid, s_ice + s_wave)
    mc = moments_from_dict(mom_comb)
    d_tb = benasciutti_tovo_damage_rate(mc, 1.0, m)
    beta = overlap_weight(grid, s_ice, s_wave)

    return {
        "ratio_f_ice_over_f_wave": float(ratio),
        "f_ice_hz": f_ice,
        "f_wave_hz": f_w,
        "wave_kind": wave_kind,
        "wave_kwargs": wk,
        "amplitude_mode": amplitude_mode,
        "seed": int(seed),
        "sigma_ice_mpa": float(np.std(h_ice)),
        "sigma_wave_mpa": float(np.std(h_wave)),
        "sigma_comb_mpa": float(np.std(h_comb)),
        "d_ice_rate": d_ice,
        "d_wave_rate": d_wave,
        "d_comb_rate": d_comb,
        "rho": d_comb / (d_ice + d_wave) if (d_ice + d_wave) > 0 else float("nan"),
        "d_tb_rate": d_tb,
        "rho_tb_model": d_tb / (d_ice + d_wave) if (d_ice + d_wave) > 0 else float("nan"),
        "overlap_beta": beta,
        "combined_width": spectral_width_measures(grid, s_ice + s_wave),
        "ice_width": spectral_width_measures(grid, s_ice),
        "wave_width": spectral_width_measures(grid, s_wave),
        "n_cycles_comb": float(np.sum(cnt_c)),
        "range_max_comb": float(np.max(r_c)) if r_c.size else 0.0,
        "duration_s": exposure,
        "fs_hz": fs,
    }


def riding_amplification(r: float, m: float = 3.0) -> float:
    """Carrier-cycle damage amplification caused by a rider band, ``r = s_r/s_c``.

    Mechanism (see ``docs/ICE_WAVE_COMBINED_FATIGUE.md``): in the rainflow count
    of a superposed history the *rider* (high-frequency) cycles are counted at
    their own amplitude, while the *carrier* (low-frequency) cycles are counted
    from the extreme values of the sum, i.e. their range grows by the rider
    amplitude standing at the carrier crest/trough.  Both families therefore
    contain the rider amplitude, which is what breaks damage additivity.

    For a Rayleigh carrier (rms ``s_c``) and a rider whose amplitude at the
    carrier extremes is Rayleigh with rms ``s_r``::

        E[R]   = 2 sqrt(2) s_c Gamma(3/2) = 2.506 s_c
        E[delta] = 2 sqrt(pi/2) s_r       = 2.506 s_r   ->  E[delta]/E[R] = r
        kappa(r) = E[(R+delta)^m] / E[R^m]
                 = sum_{k=0}^{m} C(m,k) delta^k E[R^{m-k}] / E[R^m]

    with ``R`` Rayleigh, ``E[R^j] = (2 sqrt2 s_c)^j Gamma(1+j/2)``.  ``kappa -> 1``
    as ``r -> 0`` (a negligible rider cannot enlarge the carrier), and
    ``kappa > 1`` otherwise, independently of the band separation.
    """
    if r < 0:
        raise ValueError("r must be non-negative")

    def e_r(j: float) -> float:
        return (2.0 * math.sqrt(2.0)) ** j * math.gamma(1.0 + 0.5 * j)

    mm = int(round(m))
    if abs(m - mm) > 1e-12:
        raise ValueError("m must be integral for this closed form")
    delta = 2.0 * r * math.sqrt(math.pi / 2.0)
    num = sum(math.comb(mm, k) * delta ** k * e_r(mm - k) for k in range(mm + 1))
    return float(num / e_r(mm))


def _rider_carrier(row: Dict[str, Any]) -> Tuple[str, float, float, float]:
    """Identify rider (higher frequency) and carrier, return ``(name, r, N, f_ratio)``."""
    f_ice = float(row["f_ice_hz"])
    f_w = float(row["f_wave_hz"])
    s_ice = float(row["sigma_ice_mpa"])
    s_wave = float(row["sigma_wave_mpa"])
    if f_ice >= f_w:
        name, sigma_r, sigma_c, N = "ice", s_ice, s_wave, f_ice / f_w
    else:
        name, sigma_r, sigma_c, N = "wave", s_wave, s_ice, f_w / f_ice
    r = sigma_r / sigma_c if sigma_c > 0 else float("inf")
    return name, r, N, N


def kappa_enlargement(r: float, N: float, a: float = KAPPA_A_DEFAULT,
                      b: float = KAPPA_B_DEFAULT, c: float = KAPPA_C_DEFAULT) -> float:
    """Carrier damage amplification by a rider band: ``kappa = 1 + a r^b N^c``.

    ``r = sigma_rider / sigma_carrier``, ``N = f_rider / f_carrier >= 1``.
    The three constants are calibrated by :func:`calibrate_combination_law` on the
    *separated-band* subset of the scan (the branch the law uses where the bands do
    not overlap); ``kappa -> 1`` as ``r -> 0`` reproduces the classic additive
    limit.  The mechanism behind ``kappa > 1`` is measured by
    :func:`mechanism_diagnostic`: the rider amplitude standing at the carrier's
    crest and trough enlarges the carrier range, so the rider amplitude is counted
    twice -- once in its own cycles and once inside the carrier cycles.
    """
    if r < 0 or N < 0:
        raise ValueError("r and N must be non-negative")
    if r == 0.0:
        return 1.0
    return float(1.0 + a * r ** b * max(N, 1.0) ** c)


def combination_law(row: Dict[str, Any], params: Optional[Dict[str, float]] = None,
                    m: float = 3.0) -> Dict[str, float]:
    """Nonlinear ice+wave damage combination law.

    ``D_law = (1 - beta) * [D_rider + kappa(r, N) * D_carrier] + beta * D_TB(S_comb)``

    * ``beta`` -- spectral overlap of the two bands (0 = disjoint, 1 = merged).
      No fitted constant: it is computed from the two PSDs.
    * ``kappa`` -- carrier-range enlargement by the rider
      (:func:`kappa_enlargement`), calibrated on the separated-band subset.
    * ``D_TB`` -- Tovo-Benasciutti damage rate of the *merged* PSD, the correct
      limit when the bands overlap.

    ``rho_law = D_law / (D_ice + D_wave)`` is the predicted damage ratio.
    """
    params = params or {"a": KAPPA_A_DEFAULT, "b": KAPPA_B_DEFAULT, "c": KAPPA_C_DEFAULT}
    name, r, N, _ = _rider_carrier(row)
    if name == "ice":
        d_rider, d_carrier = row["d_ice_rate"], row["d_wave_rate"]
    else:
        d_rider, d_carrier = row["d_wave_rate"], row["d_ice_rate"]
    kappa = kappa_enlargement(r, N, params["a"], params["b"], params["c"])
    d_sep = d_rider + kappa * d_carrier
    beta = float(row["overlap_beta"])
    d_law = (1.0 - beta) * d_sep + beta * row["d_tb_rate"]
    d_sum = row["d_ice_rate"] + row["d_wave_rate"]
    return {
        "rider": name,
        "r_rider_over_carrier": r,
        "N_freq_ratio": N,
        "kappa_carrier_amplification": kappa,
        "beta_overlap": beta,
        "d_separated_law": d_sep,
        "d_law_rate": d_law,
        "rho_law": d_law / d_sum if d_sum > 0 else float("nan"),
        "d_sum_rate": d_sum,
    }


def calibrate_combination_law(rows: Sequence[Dict[str, Any]],
                              training_ratios: Sequence[float] = (10.0, 100.0),
                              beta_max: float = 0.05) -> Dict[str, Any]:
    """Least-squares calibration of ``kappa = 1 + a r^b N^c``.

    Uses only rows of the *separated-band* regime (``beta <= beta_max``) inside
    ``training_ratios``; the ``kappa`` implied by each row is
    ``(D_comb - D_rider) / D_carrier`` read off the rainflow truth, so the fit
    isolates the carrier-enlargement mechanism from the overlap blending.
    """
    pts = []
    for row in rows:
        if row["overlap_beta"] > beta_max:
            continue
        if float(row["ratio"]) not in {float(x) for x in training_ratios}:
            continue
        name, r, N, _ = _rider_carrier(row)
        if name == "ice":
            d_rider, d_carrier = row["d_ice_rate"], row["d_wave_rate"]
        else:
            d_rider, d_carrier = row["d_wave_rate"], row["d_ice_rate"]
        if d_carrier <= 0 or r <= 0:
            continue
        kappa = (row["d_comb_rate"] - d_rider) / d_carrier
        if kappa > 1.0:
            pts.append((r, N, kappa))
    if len(pts) < 3:
        raise ValueError("not enough separated-band points to calibrate the law")
    X = np.array([[1.0, math.log(r), math.log(N)] for r, N, _ in pts])
    y = np.log(np.array([k - 1.0 for _, _, k in pts]))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = float(math.exp(coef[0]))
    b = float(coef[1])
    c = float(coef[2])
    pred = np.array([kappa_enlargement(r, N, a, b, c) for r, N, _ in pts])
    meas = np.array([k for _, _, k in pts])
    resid = pred / meas - 1.0
    return {
        "a": a, "b": b, "c": c,
        "n_points": len(pts),
        "training_ratios": [float(x) for x in training_ratios],
        "points": [{"r": r, "N": N, "kappa_measured": k} for r, N, k in pts],
        "fit_rel_err_max_pct": float(np.max(np.abs(resid)) * 100.0),
        "fit_rel_err_rms_pct": float(np.sqrt(np.mean(resid ** 2)) * 100.0),
    }


def evaluate_combination_law(rows: Sequence[Dict[str, Any]],
                             params: Optional[Dict[str, float]] = None,
                             holdout_ratios: Sequence[float] = (0.3, 3.0),
                             m: float = 3.0) -> Dict[str, Any]:
    """Predict ``rho`` with the law and score it against the rainflow truth.

    ``holdout_ratios`` marks the band ratios whose predictions are *not* used in
    the calibration (predictive check); ``ratio = 1`` is the merged-band regime
    where the ``beta`` branch dominates and is reported separately.
    """
    params = params or {"a": KAPPA_A_DEFAULT, "b": KAPPA_B_DEFAULT, "c": KAPPA_C_DEFAULT}
    holdout = {float(x) for x in holdout_ratios}
    scored = []
    for row in rows:
        law = combination_law(row, params, m)
        truth = float(row["rho_mean"])
        scored.append({
            "amplitude_mode": row["amplitude_mode"],
            "ratio": float(row["ratio"]),
            "truth_rho": truth,
            "sum_rule_rho": 1.0,
            "sum_rule_err_pct": 100.0 * (1.0 / truth - 1.0),
            "law_rho": float(law["rho_law"]),
            "law_err_pct": 100.0 * (law["rho_law"] / truth - 1.0),
            "tb_only_err_pct": 100.0 * (row["rho_tb_model_mean"] / truth - 1.0),
            "kappa": float(law["kappa_carrier_amplification"]),
            "r_rider": float(law["r_rider_over_carrier"]),
            "N_freq": float(law["N_freq_ratio"]),
            "beta": float(law["beta_overlap"]),
            "split": "merged_band" if abs(float(row["ratio"]) - 1.0) < 1e-9
                     else ("holdout" if float(row["ratio"]) in holdout else "train"),
        })
    def _stats(key):
        e = np.array([abs(s[key]) for s in scored], float)
        return {"max_abs_pct": float(e.max()), "mean_abs_pct": float(e.mean())}
    holdout_errs = [abs(s["law_err_pct"]) for s in scored if s["split"] == "holdout"]
    return {
        "params": params,
        "rows": scored,
        "law_stats_all": _stats("law_err_pct"),
        "law_stats_holdout": {"max_abs_pct": float(max(holdout_errs)) if holdout_errs else float("nan"),
                              "mean_abs_pct": float(np.mean(holdout_errs)) if holdout_errs else float("nan")},
        "sum_rule_stats_all": _stats("sum_rule_err_pct"),
        "tb_stats_all": _stats("tb_only_err_pct"),
        "holdout_ratios": list(holdout_ratios),
    }


def mechanism_diagnostic(case: Optional[HullGirderCase] = None,
                         ratio: float = 10.0, seed: int = 0,
                         duration: float = 1800.0, fs: float = 40.0,
                         m: float = 3.0) -> Dict[str, Any]:
    """Measure the carrier-range enlargement predicted by the law.

    The carrier family is separated from the rider family by a range threshold
    (``4 sigma_rider``).  The measured enlargement of the mean carrier range is
    compared with the prediction ``E[delta] = 2 sqrt(pi/2) sigma_rider``.
    """
    case = case or HullGirderCase()
    one = combination_case(case, ratio, seed=seed, duration=duration, fs=fs, m=m)
    f_ice, f_w = one["f_ice_hz"], one["f_wave_hz"]
    n = int(round(duration * fs))
    ice_shape = gaussian_band_shape(f_ice, 0.15)
    wave_shape = stress_spectrum_shape(one["wave_kind"], f_w, **one["wave_kwargs"])
    grid, s_ice, _ = component_grid(ice_shape, one["sigma_ice_mpa"], n, fs)
    _, s_wave, _ = component_grid(wave_shape, one["sigma_wave_mpa"], n, fs)
    h_ice = synthesize_gaussian_history(s_ice, n, fs, seed=seed)
    h_wave = synthesize_gaussian_history(s_wave, n, fs, seed=seed + 1000)
    h_comb = h_ice + h_wave
    sigma_ice = float(np.std(h_ice))
    sigma_wave = float(np.std(h_wave))
    if f_ice >= f_w:
        carrier = h_wave
        sigma_rider, sigma_carrier = sigma_ice, sigma_wave
    else:
        carrier = h_ice
        sigma_rider, sigma_carrier = sigma_wave, sigma_ice
    r_car, _, _ = rainflow_cycles(turning_points(carrier))
    r_comb, _, _ = rainflow_cycles(turning_points(h_comb))
    thresh = 4.0 * sigma_rider
    large = r_comb[r_comb > thresh]
    delta_meas = float(large.mean() - r_car.mean()) if large.size else float("nan")
    delta_pred = 2.0 * sigma_rider * math.sqrt(math.pi / 2.0)
    return {
        "ratio": float(ratio),
        "f_rider_hz": float(max(f_ice, f_w)),
        "f_carrier_hz": float(min(f_ice, f_w)),
        "sigma_rider_mpa": sigma_rider,
        "sigma_carrier_mpa": sigma_carrier,
        "r_rider_over_carrier": sigma_rider / sigma_carrier,
        "mean_range_carrier_alone_mpa": float(r_car.mean()),
        "mean_range_carrier_in_sum_mpa": float(large.mean()) if large.size else float("nan"),
        "n_carrier_cycles_classified": int(large.size),
        "delta_measured_mpa": delta_meas,
        "delta_predicted_mpa": delta_pred,
        "delta_prediction_error_pct": 100.0 * (delta_pred / delta_meas - 1.0)
        if delta_meas and math.isfinite(delta_meas) and delta_meas != 0 else float("nan"),
        "kappa_measured": (1.0 + delta_meas / float(r_car.mean())) ** m
        if large.size and r_car.mean() > 0 else float("nan"),
        "kappa_predicted": riding_amplification(sigma_rider / sigma_carrier, m),
    }


def band_separation_scan(
    case: Optional[HullGirderCase] = None,
    ratios: Sequence[float] = (0.3, 1.0, 10.0, 100.0),
    seeds: Sequence[int] = tuple(range(8)),
    wave_kind: str = "jonswap",
    wave_kwargs: Optional[Dict[str, float]] = None,
    amplitude_modes: Sequence[str] = ("equal_damage", "equal_rms"),
    sigma_ratios: Sequence[float] = (0.6,),
    **kwargs: Any,
) -> Dict[str, Any]:
    """Scan ``f_ice / f_w`` and test ``rho`` against the predeclared band."""
    case = case or HullGirderCase()
    rows: List[Dict[str, Any]] = []
    for mode in amplitude_modes:
        srs = sigma_ratios if mode == "equal_rms" else (sigma_ratios[0],)
        for ratio in ratios:
          for sigma_ratio in srs:
            per_seed = [
                combination_case(case, ratio, wave_kind=wave_kind,
                                 wave_kwargs=wave_kwargs, amplitude_mode=mode,
                                 sigma_ratio=float(sigma_ratio), seed=s, **kwargs)
                for s in seeds
            ]
            rho = np.array([r["rho"] for r in per_seed], float)
            rho_model = np.array([r["rho_tb_model"] for r in per_seed], float)
            beta = float(np.mean([r["overlap_beta"] for r in per_seed]))
            d_merged = float(np.mean([r["d_tb_rate"] for r in per_seed]))
            ref = per_seed[0]
            law_row = {
                "ratio": float(ratio),
                "f_ice_hz": ref["f_ice_hz"],
                "f_wave_hz": ref["f_wave_hz"],
                "sigma_ice_mpa": float(np.mean([r["sigma_ice_mpa"] for r in per_seed])),
                "sigma_wave_mpa": float(np.mean([r["sigma_wave_mpa"] for r in per_seed])),
                "d_ice_rate": float(np.mean([r["d_ice_rate"] for r in per_seed])),
                "d_wave_rate": float(np.mean([r["d_wave_rate"] for r in per_seed])),
                "d_comb_rate": float(np.mean([r["d_comb_rate"] for r in per_seed])),
                "d_tb_rate": d_merged,
                "overlap_beta": beta,
            }
            law = combination_law(law_row)
            rows.append({
                "amplitude_mode": mode,
                "sigma_ratio": float(sigma_ratio),
                "ratio": float(ratio),
                "f_ice_hz": ref["f_ice_hz"],
                "f_wave_hz": ref["f_wave_hz"],
                "sigma_ice_mpa": float(np.mean([r["sigma_ice_mpa"] for r in per_seed])),
                "sigma_wave_mpa": float(np.mean([r["sigma_wave_mpa"] for r in per_seed])),
                "d_ice_rate": float(np.mean([r["d_ice_rate"] for r in per_seed])),
                "d_wave_rate": float(np.mean([r["d_wave_rate"] for r in per_seed])),
                "d_comb_rate": float(np.mean([r["d_comb_rate"] for r in per_seed])),
                "rho_mean": float(rho.mean()),
                "rho_std": float(rho.std(ddof=1)) if rho.size > 1 else 0.0,
                "rho_min": float(rho.min()),
                "rho_max": float(rho.max()),
                "rho_per_seed": rho.tolist(),
                "rho_tb_model_mean": float(rho_model.mean()),
                "d_tb_rate": d_merged,
                "rho_law": float(law["rho_law"]),
                "law_kappa": float(law["kappa_carrier_amplification"]),
                "law_r_rider": float(law["r_rider_over_carrier"]),
                "law_dev_pct": 100.0 * abs(float(law["rho_law"]) - float(rho.mean())) / float(rho.mean()),
                "overlap_beta": beta,
                "waves_frac_of_sum": ref["d_wave_rate"] / (ref["d_ice_rate"] + ref["d_wave_rate"]),
                "combined_eps": float(np.mean([r["combined_width"]["eps"] for r in per_seed])),
                "combined_alpha2": float(np.mean([r["combined_width"]["alpha2"] for r in per_seed])),
                "n_seeds": int(rho.size),
                "seeds_in_band": int(np.sum((rho >= 0.97) & (rho <= 1.03))),
                "mean_in_band": bool(0.97 <= float(rho.mean()) <= 1.03),
            })
    return {
        "rows": rows,
        "ratios": list(ratios),
        "seeds": list(seeds),
        "band": [0.97, 1.03],
        "wave_kind": wave_kind,
        "wave_kwargs": wave_kwargs or {},
    }


# ---------------------------------------------------------------------------
# (b) spectral-method benchmark
# ---------------------------------------------------------------------------
WIDTH_LEVELS: Dict[str, Dict[str, Any]] = {
    # the three declared spectral-width levels of criterion (b)
    "narrow_gamma6": {"wave_kind": "jonswap", "wave_kwargs": {"gamma": 6.0}},
    "medium_gamma3p3": {"wave_kind": "jonswap", "wave_kwargs": {"gamma": 3.3}},
    "wide_bimodal_rp3": {"wave_kind": "bimodal",
                         "wave_kwargs": {"peak_separation": 3.0, "swell_fraction": 0.5,
                                         "gamma": 3.3, "swell_gamma": 6.0}},
    # supplementary width levels (reported, not part of the declared three)
    "wide_gamma1": {"wave_kind": "jonswap", "wave_kwargs": {"gamma": 1.0}},
    "medium_bimodal_rp1p6": {"wave_kind": "bimodal",
                             "wave_kwargs": {"peak_separation": 1.6, "swell_fraction": 0.5,
                                             "gamma": 3.3, "swell_gamma": 6.0}},
}


def spectral_method_benchmark(
    case: Optional[HullGirderCase] = None,
    ratios: Sequence[float] = (10.0, 1.0),
    seeds: Sequence[int] = tuple(range(6)),
    levels: Optional[Sequence[str]] = None,
    duration: float = 1800.0,
    fs: float = 40.0,
    m: float = 3.0,
    d_ice_rate: float = 1.0,
) -> Dict[str, Any]:
    """Dirlik / Tovo-Benasciutti / narrow-band vs rainflow truth on the same PSD.

    The three spectral models and the rainflow counter see the *same* band-limited
    combined PSD: the PSD is sampled on the rFFT grid of the record, ``m0..m4`` are
    integrated on those very bins and the histories are synthesised from them, so
    the only difference left is the counting model.
    """
    case = case or HullGirderCase()
    names = list(levels) if levels is not None else list(WIDTH_LEVELS)
    rows: List[Dict[str, Any]] = []
    for name in names:
        cfg = WIDTH_LEVELS[name]
        for ratio in ratios:
            f_ice = case.ice_band_hz
            f_w = f_ice / float(ratio)
            n = int(round(duration * fs))
            ice_shape = gaussian_band_shape(f_ice, 0.15)
            wave_shape = stress_spectrum_shape(cfg["wave_kind"], f_w, **cfg["wave_kwargs"])
            sigma_ice = equal_damage_sigma(f_ice, d_ice_rate, m)
            sigma_wave = equal_damage_sigma(f_w, d_ice_rate, m)
            freqs, s_ice, _ = component_grid(ice_shape, sigma_ice, n, fs)
            _, s_wave, _ = component_grid(wave_shape, sigma_wave, n, fs)
            s_comb = s_ice + s_wave
            mom = moments_from_dict(_lib_spectral_moments(freqs, s_comb))
            d_tb = benasciutti_tovo_damage_rate(mom, 1.0, m)
            d_dirlik = dirlik_damage_rate(mom, 1.0, m)
            d_nb = rayleigh_damage_rate(mom, 1.0, m)
            truth = []
            for s in seeds:
                x = synthesize_gaussian_history(s_comb, n, fs, seed=s + 500)
                r, mn, c = rainflow_cycles(turning_points(x))
                truth.append(damage_from_cycles(r, c, m) / (n / fs))
            truth_a = np.array(truth, float)
            mean_truth = float(truth_a.mean())
            rows.append({
                "level": name,
                "wave_kind": cfg["wave_kind"],
                "wave_kwargs": cfg["wave_kwargs"],
                "ratio": float(ratio),
                "f_wave_hz": f_w,
                "alpha2": mom.alpha2,
                "nu0_hz": mom.nu0,
                "nup_hz": mom.nup,
                "sigma_comb_mpa": float(math.sqrt(mom.m0)),
                "d_rainflow_mean": mean_truth,
                "d_rainflow_std": float(truth_a.std(ddof=1)) if truth_a.size > 1 else 0.0,
                "d_dirlik": d_dirlik,
                "d_tb": d_tb,
                "d_nb": d_nb,
                "err_dirlik_pct": 100.0 * (d_dirlik / mean_truth - 1.0),
                "err_tb_pct": 100.0 * (d_tb / mean_truth - 1.0),
                "err_nb_pct": 100.0 * (d_nb / mean_truth - 1.0),
                "declared_level": name in ("narrow_gamma6", "medium_gamma3p3", "wide_bimodal_rp3"),
            })
    declared = [r for r in rows if r["declared_level"]]
    worst_dirlik = max(abs(r["err_dirlik_pct"]) for r in declared)
    worst_tb = max(abs(r["err_tb_pct"]) for r in declared)
    return {
        "rows": rows,
        "declared_levels": ["narrow_gamma6", "medium_gamma3p3", "wide_bimodal_rp3"],
        "threshold_pct": 10.0,
        "worst_err_dirlik_pct": worst_dirlik,
        "worst_err_tb_pct": worst_tb,
        "pass": bool(worst_dirlik <= 10.0 and worst_tb <= 10.0),
        "seeds": list(seeds),
        "duration_s": duration,
        "fs_hz": fs,
        "m": m,
    }# ---------------------------------------------------------------------------
# (c) whipping bookkeeping
# ---------------------------------------------------------------------------
def harmonic_ice_stress(case: "HullGirderCase", f_hz: float = 0.05,
                        force_amplitude_MN: float = 10.6, duration: float = 900.0,
                        dt: float = 2.0e-3, settle_time: float = 60.0) -> Dict[str, Any]:
    """Harmonically varying ice force -> stress (the m5528 ``f = 0.05 Hz`` check).

    A slowly varying crushing force at ``f_hz`` is the cleanest quasi-static
    reference: the first hull-girder mode (0.559 Hz) is then barely excited, so
    the whipping share must collapse.  ``force_amplitude_MN`` defaults to the
    m5528 record mean force (10.59 MN).
    """
    n = int(round(duration / dt))
    t = np.arange(n) * dt
    F = force_amplitude_MN * 1e6 * (1.0 + np.sin(2.0 * np.pi * f_hz * t))
    res = modal_stress_history(case.modes, F, dt, case.x_load, case.x_detail,
                               case.z_deck, case.E)
    skip = min(int(settle_time / dt), n // 3)
    sigma = res["sigma"][skip:] * case.scf / 1e6
    sigma_qs = res["sigma_qs"][skip:] * case.scf / 1e6
    return {
        "sigma_mpa": sigma,
        "sigma_qs_mpa": sigma_qs,
        "dt": dt,
        "whipping": whipping_share(sigma, sigma_qs),
        "f_pulse_hz": float(f_hz),
        "harmonic": True,
    }


def upcrossing_rate(series: np.ndarray, dt: float) -> float:
    """Mean upward zero-crossing rate [Hz] of a detrended series."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    if x.size < 3 or dt <= 0:
        return 0.0
    s = np.sign(x)
    s[s == 0] = 1.0
    n_up = int(np.sum((s[:-1] < 0) & (s[1:] > 0)))
    return float(n_up / (x.size * dt))


def stretched_ice_stress(case: "HullGirderCase", stretch: float,
                         params: IceContactParams = CALIBRATED_M5528,
                         duration: float = 600.0, dt: float = 2.0e-3,
                         settle_time: float = 60.0) -> Dict[str, Any]:
    """Ice stress of a *quasi-static* drift: identical DEM waveform, stretched time.

    Multiplying the sampling interval by ``stretch`` keeps the sample values (the
    same crushing events) while lengthening every physical time scale, i.e. the
    load and the ice's own unloading both slow down by ``stretch``.  This is the
    quasi-static limit referred to by the m5528 ``f = 0.05 Hz`` check; the
    alternative (only the indentation rate slows while the crushing decay stays at
    0.15 s) is kept as a sensitivity because it leaves a fast unloading step that
    still excites the first hull-girder mode.
    """
    dt_s = float(dt) * float(stretch)
    t, F = generate_ice_force(params, duration=duration, dt=float(dt))
    res = modal_stress_history(case.modes, F, dt_s, case.x_load, case.x_detail,
                               case.z_deck, case.E)
    skip = min(int(settle_time / dt_s), max(0, F.size // 3))
    sigma = res["sigma"][skip:] * case.scf / 1e6
    sigma_qs = res["sigma_qs"][skip:] * case.scf / 1e6
    st = load_statistics(F[skip:], dt_s, case.f_cut_hz)
    return {
        "sigma_mpa": sigma,
        "sigma_qs_mpa": sigma_qs,
        "dt": dt_s,
        "whipping": whipping_share(sigma, sigma_qs),
        "load_statistics": st,
        "f_pulse_hz": params.pulse_rate_hz() / float(stretch),
        "stretch": float(stretch),
    }


def whipping_accounting(
    case: Optional[HullGirderCase] = None,
    m: Optional[float] = None,
) -> Dict[str, Any]:
    """Relate the whipping share to the damage bias of counting it twice.

    Bookkeeping
    -----------
    ``s_peak``        ``max|sigma - sigma_qs| / max|sigma|`` (the m5528 caliber).
    ``s_rms``         ``rms(sigma - sigma_qs) / rms(sigma)``.
    ``D_qs``/``D_whip``  damage of the quasi-static series and of the residual
                      (dynamic) series ``sigma - sigma_qs`` (DNV F3 bilinear +
                      Goodman, annualised over the ice season).
    ``a = D_whip/D_qs``  the ratio the double-counting bias rides on:
                      ``A = D_mis/D_true = 1 + a`` with ``D_mis = D(sigma) +
                      D(sigma - sigma_qs)`` -- the classic error of adding an
                      ice-vibration fatigue increment to a history that already
                      contains the vibration.
    ``nu_qs``/``nu_whip``  mean up-crossing rates of the two series.

    Two mappings from "share" to ``A`` are scored:

    * **peak caliber (m5528)**: ``a = (s_peak/(1-s_peak))**m`` -- treats the
      peak share as if it carried all the damage.
    * **damage caliber (this study)**: narrow-band Miner on each family,
      ``a = (rms_whip/rms_qs)**m * (nu_whip/nu_qs)``; inverted, it gives
      ``s_rms_implied`` to be compared with ``s_rms``.

    Regimes: the m5528 crushing condition, the quasi-static drift
    (``f_pulse = 0.05 Hz``, time-stretched so that the m5528 "<8 %" check
    applies), a rate-only slowed drift as a sensitivity, and a fast drift for
    contrast.
    """
    case = case or HullGirderCase()
    m_exp = float(m if m is not None else 3.0)
    curve = case.curve()
    regimes = [
        ("crushing_m5528", CALIBRATED_M5528, 300.0, 2.0e-3, 1.0),
        ("harmonic_0p05Hz", None, 900.0, 2.0e-3, 1.0),
        ("quasi_static_stretched_0p05Hz", CALIBRATED_M5528, 600.0, 2.0e-3,
         CALIBRATED_M5528.pulse_rate_hz() / 0.05),
        ("slow_rate_only_0p05Hz", scaled_drift_rate(0.05), 1200.0, 2.0e-3, 1.0),
        ("fast_drift_1Hz", CALIBRATED_M5528.with_v_indent(1.0 * 0.8 / 3.0), 300.0, 2.0e-3, 1.0),
    ]
    rows: List[Dict[str, Any]] = []
    for name, params, duration, dt, stretch in regimes:
        if params is None:
            s = harmonic_ice_stress(case, f_hz=0.05, duration=duration, dt=dt)
        elif stretch == 1.0:
            s = ice_stress_history(case, params, duration=duration, dt=dt)
        else:
            s = stretched_ice_stress(case, stretch, params, duration=duration, dt=dt)
            s["duration_s"] = duration - 60.0
        dt_eff = s["dt"]
        exposure = s["sigma_mpa"].size * dt_eff
        scale = case.ice_hours_per_year * 3600.0 / exposure

        def _bilinear(series):
            r, mn, c = rainflow_cycles(turning_points(series))
            return snmod.miner_damage(r, mn, c, curve, s_uts=case.s_uts,
                                      thickness=case.thickness, exposure_scale=scale)

        total = _bilinear(s["sigma_mpa"])
        quasi = _bilinear(s["sigma_qs_mpa"])
        resid = _bilinear(s["sigma_mpa"] - s["sigma_qs_mpa"])
        d_true = float(total["damage"])
        d_qs = float(quasi["damage"])
        d_whip = float(resid["damage"])
        a_qs = d_whip / d_qs if d_qs > 0 else float("nan")
        a_meas = d_whip / d_true if d_true > 0 else float("nan")
        share_peak = float(s["whipping"]["share_peak"])
        share_rms = float(s["whipping"]["share_rms"])
        rms_qs = float(np.std(s["sigma_qs_mpa"]))
        rms_whip = float(np.std(s["sigma_mpa"] - s["sigma_qs_mpa"]))
        nu_qs = upcrossing_rate(s["sigma_qs_mpa"], dt_eff)
        nu_whip = upcrossing_rate(s["sigma_mpa"] - s["sigma_qs_mpa"], dt_eff)
        # peak caliber (m5528 reading)
        a_peak = (share_peak / (1.0 - share_peak)) ** m_exp if share_peak < 1.0 else float("inf")
        s_implied_peak = (a_meas ** (1.0 / m_exp)) / (1.0 + a_meas ** (1.0 / m_exp)) if a_meas > 0 else 0.0
        # damage caliber: rms ratio + cycle-rate ratio
        rate_ratio = nu_whip / nu_qs if nu_qs > 0 else float("nan")
        a_damage = (rms_whip / rms_qs) ** m_exp * rate_ratio if rms_qs > 0 else float("nan")
        rms_whip_implied = ((a_meas / rate_ratio) ** (1.0 / m_exp)) * rms_qs if rate_ratio > 0 else float("nan")
        s_rms_implied = (rms_whip_implied
                         / math.sqrt(rms_qs ** 2 + rms_whip_implied ** 2)) if rms_whip_implied > 0 else 0.0
        rows.append({
            "regime": name,
            "f_pulse_hz": s["f_pulse_hz"],
            "duration_s": exposure,
            "dt_s": dt_eff,
            "stretch": float(stretch),
            "share_peak": share_peak,
            "share_rms": share_rms,
            "rms_qs_mpa": rms_qs,
            "rms_whip_mpa": rms_whip,
            "nu_qs_hz": nu_qs,
            "nu_whip_hz": nu_whip,
            "nu_ratio": rate_ratio,
            "damage_annual_true": d_true,
            "damage_annual_qs": d_qs,
            "damage_annual_whip": d_whip,
            "whip_damage_share": d_whip / d_true if d_true > 0 else float("nan"),
            "a_over_qs": a_qs,
            "a_measured": a_meas,
            "amplification_A": 1.0 + a_meas,
            "amplification_a_over_qs": 1.0 + a_qs,
            "damage_share_whip": float(d_whip / (d_whip + d_qs)) if (d_whip + d_qs) > 0 else float("nan"),
            "a_predicted_peak_caliber": a_peak,
            "share_implied_peak_caliber": s_implied_peak,
            "deviation_peak_caliber_pp": 100.0 * abs(s_implied_peak - share_peak),
            "A_predicted_peak_caliber": 1.0 + a_peak,
            "A_predicted_rms_caliber": 1.0 + (share_rms / (1.0 - share_rms)) ** m_exp
            if share_rms < 1.0 else float("inf"),
            "A_predicted_rms_x_rate": 1.0 + share_rms ** m_exp * (nu_whip / upcrossing_rate(s["sigma_mpa"], dt_eff))
            if nu_whip > 0 else float("nan"),
            "a_predicted_damage_caliber": a_damage,
            "share_implied_damage_caliber_rms": s_rms_implied,
            "deviation_damage_caliber_pp": 100.0 * abs(s_rms_implied - share_rms),
        })
    harmonic = [r for r in rows if r["regime"].startswith("harmonic")]
    pulse_slow = [r for r in rows if r["regime"] in
                  ("quasi_static_stretched_0p05Hz", "slow_rate_only_0p05Hz")]
    design = [r for r in rows if r["regime"] == "crushing_m5528"]
    dev_peak_design = design[0]["deviation_peak_caliber_pp"] if design else float("nan")
    sub = {
        # the m5528 "<8 % at 0.05 Hz" statement, harmonic (quasi-static) reading
        "c1_harmonic_slow_share_below_8pct": bool(harmonic and harmonic[0]["share_peak"] < 0.08),
        "c1_share_peak": harmonic[0]["share_peak"] if harmonic else None,
        # ... and the DEM pulse-train reading at the same pulse rate
        "c2_pulse_train_slow_share_below_8pct": bool(
            pulse_slow and all(r["share_peak"] < 0.08 for r in pulse_slow)),
        "c2_share_peak_stretched": pulse_slow[0]["share_peak"] if pulse_slow else None,
        "c2_share_peak_rate_only": pulse_slow[1]["share_peak"] if len(pulse_slow) > 1 else None,
        # peak caliber -> damage amplification consistency at the design point
        "c3_peak_caliber_within_2pp": bool(dev_peak_design <= 2.0),
        "c3_deviation_pp_design": dev_peak_design,
        "c3_A_measured_design": design[0]["amplification_A"] if design else None,
        "c3_A_predicted_peak_caliber": design[0]["A_predicted_peak_caliber"] if design else None,
    }
    return {
        "rows": rows,
        "m_exponent": m_exp,
        "subchecks": sub,
        "damage_share_whip_range": [min(r["damage_share_whip"] for r in rows),
                                    max(r["damage_share_whip"] for r in rows)],
        "A_range": [min(r["amplification_A"] for r in rows),
                    max(r["amplification_A"] for r in rows)],
        "max_deviation_peak_caliber_pp": max(r["deviation_peak_caliber_pp"] for r in rows),
        "max_deviation_damage_caliber_pp": max(r["deviation_damage_caliber_pp"] for r in rows),
        "tolerance_pp": 2.0,
        "pass": bool(sub["c1_harmonic_slow_share_below_8pct"]
                     and sub["c2_pulse_train_slow_share_below_8pct"]
                     and sub["c3_peak_caliber_within_2pp"]),
        "peak_caliber_pass": bool(sub["c3_peak_caliber_within_2pp"]),
    }


# ---------------------------------------------------------------------------
# (d) rainflow commutability
# ---------------------------------------------------------------------------
def _closed_burst_blocks(n_blocks: int = 24, block_s: float = 24.0, fs: float = 40.0,
                         seed: int = 7) -> List[np.ndarray]:
    """Blocks that start and end at exactly 0 (closed excursions).

    Each block is a Hann-tapered multi-frequency burst, hence a genuine closed
    excursion with internal cycles rather than a single triangle.
    """
    rng = np.random.default_rng(seed)
    r = int(round(block_s * fs))
    t = np.linspace(0.0, 1.0, r, endpoint=False)
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * t)          # Hann: exactly 0 at both ends
    blocks = []
    for _ in range(n_blocks):
        amp = float(rng.uniform(5.0, 25.0))
        f1 = float(rng.uniform(2.0, 6.0))
        f2 = float(rng.uniform(8.0, 14.0))
        ph = rng.uniform(0.0, 2.0 * np.pi, size=2)
        wav = (0.7 * np.sin(2.0 * np.pi * f1 * t + ph[0])
               + 0.3 * np.sin(2.0 * np.pi * f2 * t + ph[1]))
        blocks.append(amp * win * wav)
    return blocks


def segmentation_invariance(
    n_blocks: int = 24,
    seeds: Sequence[int] = tuple(range(20)),
    m: float = 3.0,
) -> Dict[str, Any]:
    """Rainflow commutability tests (criterion d).

    ``seg``    the same history fed to the streaming counter in contiguous pieces
               with the residue stack carried across the cuts (cut points placed
               anywhere, including inside peaks) -> must equal the whole-record
               damage bit for bit.
    ``perm``   closed excursion blocks (each starts and ends at exactly 0) counted
               per block and summed; permuting the block order must not change the
               damage (only floating-point summation order does).
    ``naive``  *trap*: the whole-record counter applied to a permuted
               concatenation of the same blocks -- the damage then depends on the
               order, which is why the closed-excursion decomposition above is the
               correct combination practice.
    """
    blocks = _closed_burst_blocks(n_blocks=n_blocks)
    hist = np.concatenate(blocks)
    n = hist.size

    whole = StreamingRainflow().feed(hist).finalize()
    r_w, _, c_w = whole.cycles()
    d_whole = damage_from_cycles(r_w, c_w, m)

    seg_reports = []
    for n_chunks in (2, 7, 31, 97):
        rf = StreamingRainflow()
        bounds = np.linspace(0, n, n_chunks + 1).astype(int)
        for a, b in zip(bounds[:-1], bounds[1:]):
            rf.feed(hist[a:b])
        r_s, _, c_s = rf.finalize().cycles()
        d_seg = damage_from_cycles(r_s, c_s, m)
        seg_reports.append({
            "n_chunks": int(n_chunks),
            "d_whole": d_whole,
            "d_segmented": d_seg,
            "abs_dev": abs(d_seg - d_whole),
            "rel_dev": abs(d_seg - d_whole) / d_whole if d_whole else 0.0,
            "n_cycles_whole": float(np.sum(c_w)),
            "n_cycles_seg": float(np.sum(c_s)),
        })

    def _closed_block_damage(blk: np.ndarray) -> float:
        r, _, c = rainflow_cycles(turning_points(blk))
        return damage_from_cycles(r, c, m)

    d_blocks = np.array([_closed_block_damage(b) for b in blocks], float)
    d_sum = float(d_blocks.sum())
    perm_devs = []
    perm_naive = []
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        order = rng.permutation(len(blocks))
        perm_devs.append(abs(float(d_blocks[order].sum()) - d_sum) / d_sum)
        perm_hist = np.concatenate([blocks[i] for i in order])
        rf = StreamingRainflow().feed(perm_hist).finalize()
        r_p, _, c_p = rf.cycles()
        perm_naive.append(damage_from_cycles(r_p, c_p, m))

    naive_a = np.array(perm_naive, float)
    naive_spread = float((naive_a.max() - naive_a.min()) / naive_a.mean())

    return {
        "n_blocks": int(len(blocks)),
        "block_samples": int(blocks[0].size),
        "fs_hz": 40.0,
        "d_whole": d_whole,
        "n_cycles_whole": float(np.sum(c_w)),
        "segmentation": seg_reports,
        "max_seg_rel_dev": max(r["rel_dev"] for r in seg_reports),
        "closed_blocks_damage_sum": d_sum,
        "permutation_max_rel_dev": float(max(perm_devs)),
        "permutation_tolerance": 1e-9,
        "permutation_pass": bool(max(perm_devs) <= 1e-9),
        "naive_permutation_spread_rel": naive_spread,
        "naive_permutation_min": float(naive_a.min()),
        "naive_permutation_max": float(naive_a.max()),
        "segmentation_pass": bool(max(r["rel_dev"] for r in seg_reports) <= 1e-9),
    }


# ---------------------------------------------------------------------------
# (e) negative control: reproduce m5528
# ---------------------------------------------------------------------------
#: Published m5528 (IN32-N07) reference values -- read from
#: ``artifacts/ice_fatigue/results.json`` of mission m5528, quoted here so the
#: negative control does not need the platform repository.
M5528_REFERENCE: Dict[str, float] = {
    "damage_annual_goodman": 0.3764367799916392,
    "life_years": 2.656488561033304,
    "whipping_share_peak": 0.35108652124027145,
    "whipping_share_rms": 0.7591470731209626,
    "hotspot_peak_MPa": 73.96962417792953,
    "hotspot_rms_MPa": 9.314118642407184,
    "nominal_peak_MPa": 29.587849671171814,
    "dsigma_eq_MPa": 18.752560299707003,
    "n_window_cycles": 305.0,
    "f1_Hz": 0.5586672438138549,
    "force_peak_MN": 19.19124139841869,
    "force_mean_MN": 10.585594582643434,
    "mean_peak": 0.5515846715114354,
    "rise_median_s": 0.219,
    "qs_energy_frac": 0.932543769465116,
}


def m5528_reproduction(case: Optional[HullGirderCase] = None,
                       tolerance_pct: float = 2.0) -> Dict[str, Any]:
    """Negative control (criterion e): waves off -> the m5528 numbers."""
    case = case or HullGirderCase()
    s = ice_stress_history(case, CALIBRATED_M5528, duration=300.0, dt=2.0e-3)
    curve = case.curve()
    r, mn, c = rainflow_cycles(turning_points(s["sigma_mpa"]))
    exposure = 300.0 - 60.0
    scale = case.ice_hours_per_year * 3600.0 / exposure
    dmg = snmod.miner_damage(r, mn, c, curve, s_uts=case.s_uts,
                            thickness=case.thickness, exposure_scale=scale)
    d_annual = float(dmg["damage"])
    life = snmod.fatigue_life(d_annual, 1.0)
    computed = {
        "damage_annual_goodman": d_annual,
        "life_years": life,
        "whipping_share_peak": float(s["whipping"]["share_peak"]),
        "whipping_share_rms": float(s["whipping"]["share_rms"]),
        "hotspot_peak_MPa": s["sigma_peak_mpa"],
        "hotspot_rms_MPa": s["sigma_rms_mpa"],
        "nominal_peak_MPa": s["sigma_peak_mpa"] / case.scf,
        "dsigma_eq_MPa": equivalent_stress_range(r, c, 3.0),
        "n_window_cycles": float(np.sum(c)),
        "f1_Hz": case.f1_hz,
        "force_peak_MN": s["force_peak_MN"],
        "force_mean_MN": s["force_mean_MN"],
        "mean_peak": s["load_statistics"]["mean_peak"],
        "rise_median_s": s["load_statistics"]["rise_median"],
        "qs_energy_frac": s["load_statistics"]["qs_energy_frac"],
    }
    comparisons = {}
    for key, ref in M5528_REFERENCE.items():
        val = computed[key]
        rel = abs(val - ref) / abs(ref) * 100.0 if ref != 0 else abs(val) * 100.0
        comparisons[key] = {"computed": val, "m5528": ref, "rel_err_pct": rel,
                            "within_tol": bool(rel <= tolerance_pct)}
    worst = max(v["rel_err_pct"] for v in comparisons.values())
    return {
        "computed": computed,
        "comparisons": comparisons,
        "tolerance_pct": tolerance_pct,
        "worst_rel_err_pct": worst,
        "pass": bool(worst <= tolerance_pct),
    }


# ---------------------------------------------------------------------------
# Level B: physically anchored combined case (DEM ice + real sea state)
# ---------------------------------------------------------------------------
def physical_combination(
    case: Optional[HullGirderCase] = None,
    sea: Optional[SeaState] = None,
    duration: float = 3600.0,
    dt: float = 1.0e-2,
    wave_target_rms: Optional[float] = None,
    wave_scale: float = 1.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Level B: DEM-calibrated ice history + real-sea wave history, combined.

    The wave amplitude can be (i) the strip-theory level (``wave_target_rms=None``),
    (ii) scaled to a target rms, or (iii) multiplied by ``wave_scale``.  Both
    single-load damages and the combined damage are computed with the *same*
    bilinear DNV F3 + Goodman accounting, i.e. the physically relevant model.
    """
    case = case or HullGirderCase()
    sea = sea or SeaState()
    fs = 1.0 / dt

    ice = ice_stress_history(case, CALIBRATED_M5528, duration=duration + 60.0, dt=dt)
    wave = physical_wave_component(case, sea, n_samples=ice["sigma_mpa"].size, fs=fs,
                                  target_rms=wave_target_rms, seed=seed)
    h_wave = wave["sigma_mpa"] * wave_scale
    h_ice = ice["sigma_mpa"]
    h_comb = h_ice + h_wave
    qs_comb = ice["sigma_qs_mpa"] + (h_wave - h_wave.mean())
    curve = case.curve()
    exposure = h_ice.size / fs
    scale = case.ice_hours_per_year * 3600.0 / exposure

    def _bilinear(series):
        r, mn, c = rainflow_cycles(turning_points(series))
        return snmod.miner_damage(r, mn, c, curve, s_uts=case.s_uts,
                                  thickness=case.thickness, exposure_scale=scale)

    def _single(series, m=3.0):
        r, mn, c = rainflow_cycles(turning_points(series))
        return damage_from_cycles(r, c, m) / exposure, r, mn, c

    d_ice_b, d_wave_b, d_comb_b = _bilinear(h_ice), _bilinear(h_wave), _bilinear(h_comb)
    r3_ice, _, _, _ = _single(h_ice)
    r3_wave, _, _, _ = _single(h_wave)
    r3_comb, r_c, mn_c, cnt_c = _single(h_comb)
    ws = whipping_share(h_comb, qs_comb)
    share_ice = ice["whipping"]["share_peak"]
    resid = h_comb - qs_comb
    d_resid_b = _bilinear(resid)["damage"]
    d_qs_b = _bilinear(qs_comb)["damage"]
    a_qs = d_resid_b / d_qs_b if d_qs_b > 0 else float("nan")
    d_true_b = float(d_comb_b["damage"])
    a_meas = d_resid_b / d_true_b if d_true_b > 0 else float("nan")
    total_combined_series = h_comb
    nu_whip = upcrossing_rate(resid, dt)
    nu_total = upcrossing_rate(total_combined_series, dt)
    return {
        "sea_state": sea.label(),
        "duration_s": duration,
        "dt_s": dt,
        "n_samples": int(h_ice.size),
        "ice": {
            "sigma_rms_mpa": float(np.std(h_ice)),
            "sigma_peak_mpa": float(np.max(np.abs(h_ice))),
            "whipping_share_peak": share_ice,
            "f_pulse_hz": ice["pulse_rate_hz"],
        },
        "wave": {
            "sigma_rms_mpa": float(np.std(h_wave)),
            "sigma_peak_mpa": float(np.max(np.abs(h_wave))),
            "strip_rms_mpa": wave["raw_rms_mpa"],
            "gain": wave["gain"],
            "wave_scale": wave_scale,
            "f_peak_hz": wave["f_peak_hz"],
            "width": wave["width"],
            "n_cycles": wave["n_cycles"],
        },
        "damage": {
            "bilinear_ice": float(d_ice_b["damage"]),
            "bilinear_wave": float(d_wave_b["damage"]),
            "bilinear_combined": float(d_comb_b["damage"]),
            "bilinear_ratio_comb_over_sum": float(d_comb_b["damage"])
            / float(d_ice_b["damage"] + d_wave_b["damage"]),
            "single_slope_ice_rate": r3_ice,
            "single_slope_wave_rate": r3_wave,
            "single_slope_combined_rate": r3_comb,
            "single_slope_ratio": r3_comb / (r3_ice + r3_wave),
            "life_years_combined": snmod.fatigue_life(float(d_comb_b["damage"]), 1.0),
        },
        "whipping": {
            "share_peak_combined": float(ws["share_peak"]),
            "share_rms_combined": float(ws["share_rms"]),
            "a_over_qs": a_qs,
            "a_measured": a_meas,
            "amplification_A": 1.0 + a_meas,
            "amplification_a_over_qs": 1.0 + a_qs,
            "damage_share_whip": float(d_resid_b / (d_resid_b + d_qs_b)) if (d_resid_b + d_qs_b) > 0 else float("nan"),
            "nu_whip_hz": nu_whip,
            "nu_total_hz": nu_total,
            "share_implied_from_A": (a_meas ** (1.0 / 3.0)) / (1.0 + a_meas ** (1.0 / 3.0))
            if a_meas > 0 else 0.0,
            "share_deviation_pp": 100.0 * abs(
                ((a_meas ** (1.0 / 3.0)) / (1.0 + a_meas ** (1.0 / 3.0)) if a_meas > 0 else 0.0)
                - float(ws["share_peak"])),
        },
        "n_cycles_combined": float(np.sum(cnt_c)),
        "range_max_combined": float(np.max(r_c)) if r_c.size else 0.0,
        "f_ratio_ice_over_wave": float(ice["pulse_rate_hz"] / wave["f_peak_hz"]),
    }