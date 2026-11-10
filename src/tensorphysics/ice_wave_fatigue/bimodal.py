"""Bimodal (ice-band + wave-band) spectral fatigue closures.

Node ``P82W5-N09`` -- follow-up of ``P82W4-N06`` (``m7361``), which established
that the ice+wave damage is *not* the sum of the two component damages but left
two machine criteria open (spectral-method accuracy in the separated-band
regime, whipping-share caliber).  This module carries the follow-up work.

What is added here
------------------
1. :func:`band_split` / :func:`band_case` -- one reproducible construction of a
   two-band stress PSD (ice band at ``f_ice``, wave band at ``f_ice/N``) that
   returns, from *the same* band-limited PSD, the spectral band damages and the
   rainflow truth of the synthesised history.  Everything downstream (kappa
   calibration, benchmark) uses this single function, so no comparison ever
   mixes PSD definitions.
2. :func:`two_band_riding_damage` -- a **two-band spectral closure**
   ``D = D_rider + kappa * D_carrier`` with a regime switch to the classical
   single-peak Dirlik damage when the bands are not resolvable as separate
   cycle families.  The switch sits at ``N = N_SEPARATED = 5``, the separation
   threshold already documented in :mod:`tensorphysics.fatigue.spectral`
   (Torhaug 1990 / Jiao & Moan 1990), i.e. it is *declared*, not fitted.
3. :func:`fit_kappa_closure` -- the closed form of the carrier amplification
   ``kappa(r, N)`` (``r = sigma_rider/sigma_carrier``, ``N = f_rider/f_carrier``)
   calibrated on a Level-A rainflow grid.  Residuals and out-of-sample
   behaviour are reported, never hidden.
4. :func:`sum_rule_envelope` -- the *decidable* statement criterion (a) asks
   for: for which ``(r, N)`` the plain sum rule ``D = D1 + D2`` stays inside
   +-3 %.
5. :func:`bimodal_benchmark` -- criterion (b): Dirlik, Tovo-Benasciutti and the
   narrow-band formula on the *combined* spectrum versus rainflow truth on the
   three declared spectral widths, plus the two-band closure.

Conventions (identical to :mod:`tensorphysics.ice_wave_fatigue.combination`)
----------------------------------------------------------------------------
* Stress in MPa, single-slope S-N ``N = C S^-m`` with ``C = 1``, ``m = 3`` for
  every model comparison, so only the *counting* differs between models.
* ``N`` always means the band-separation ratio ``f_rider / f_carrier >= 1``;
  ``r`` always means ``sigma_rider / sigma_carrier``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tensorphysics.fatigue.spectral import (
    SpectralMoments,
    benasciutti_tovo_damage_rate,
    dirlik_damage_rate,
    moments_from_dict,
    rayleigh_damage_rate,
    spectral_moments as _lib_moments,
    synthesize_gaussian_history,
)

from .combination import (
    HullGirderCase,
    WIDTH_LEVELS,
    component_grid,
    gaussian_band_shape,
    overlap_weight,
    stress_spectrum_shape,
)
from .rainflow_stream import damage_from_cycles, rainflow_cycles, turning_points

__all__ = [
    "N_SEPARATED",
    "BandSplit",
    "KappaClosure",
    "equal_damage_amplitude_ratio",
    "band_split",
    "band_case",
    "riding_kappa_convolution",
    "fit_kappa_closure",
    "two_band_riding_damage",
    "sum_rule_envelope",
    "kappa_calibration_grid",
    "bimodal_benchmark",
]

#: Band-separation ratio above which the rainflow ranges may be split into two
#: independent families (Torhaug 1990; Jiao & Moan 1990; see the module docstring
#: of :mod:`tensorphysics.fatigue.spectral`).  Declared, not fitted.
N_SEPARATED = 5.0


def equal_damage_amplitude_ratio(n_ratio: float, m: float = 3.0) -> float:
    """``r`` that puts the two bands on an equal per-component damage footing.

    ``D ~ nu sigma^m`` with ``nu ~ f``, so equal component damage needs
    ``r = (f_carrier/f_rider)**(1/m) = N**(-1/m)``.
    """
    return float(max(n_ratio, 1.0) ** (-1.0 / float(m)))


# ---------------------------------------------------------------------------
# Band decomposition of a two-band PSD
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BandSplit:
    """Valley split of a two-band PSD into a carrier and a rider band."""

    freqs: np.ndarray
    s_comb: np.ndarray
    f_valley_hz: float
    f_carrier_hz: float
    f_rider_hz: float
    mom_carrier: SpectralMoments
    mom_rider: SpectralMoments
    mom_comb: SpectralMoments
    beta_overlap: float

    @property
    def sigma_carrier(self) -> float:
        return math.sqrt(max(self.mom_carrier.m0, 0.0))

    @property
    def sigma_rider(self) -> float:
        return math.sqrt(max(self.mom_rider.m0, 0.0))

    @property
    def r(self) -> float:
        sc = self.sigma_carrier
        return float(self.sigma_rider / sc) if sc > 0 else 0.0

    @property
    def n_ratio(self) -> float:
        return float(self.f_rider_hz / self.f_carrier_hz)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "f_valley_hz": self.f_valley_hz,
            "f_carrier_hz": self.f_carrier_hz,
            "f_rider_hz": self.f_rider_hz,
            "r_rider_over_carrier": self.r,
            "N_freq_ratio": self.n_ratio,
            "sigma_carrier_mpa": self.sigma_carrier,
            "sigma_rider_mpa": self.sigma_rider,
            "m0_carrier": self.mom_carrier.m0,
            "m0_rider": self.mom_rider.m0,
            "alpha2_comb": float(self.mom_comb.alpha2),
            "alpha1_comb": float(self.mom_comb.alpha1),
            "nu0_carrier_hz": float(self.mom_carrier.nu0),
            "nu0_rider_hz": float(self.mom_rider.nu0),
            "nup_comb_hz": float(self.mom_comb.nup),
            "beta_overlap": self.beta_overlap,
        }


def band_split(freqs: Sequence[float], s_comb: Sequence[float],
               f_ice: float, f_wave: float) -> BandSplit:
    """Split a two-band PSD at the valley between the two band peaks.

    The valley is the minimum of the summed PSD strictly between the two band
    peak frequencies; if no interior minimum exists (merged bands, e.g.
    ``f_ice == f_wave``) the geometric mean of the two frequencies is used,
    which reproduces the "one wide band" limit.  The rider band is the
    higher-frequency one by construction.
    """
    f = np.asarray(freqs, dtype=float)
    s = np.asarray(s_comb, dtype=float)
    lo, hi = float(min(f_ice, f_wave)), float(max(f_ice, f_wave))
    f_valley = math.sqrt(max(lo * hi, 1e-12))
    inside = np.where((f > 1.02 * lo) & (f < 0.98 * hi))[0]
    if inside.size > 2:
        f_valley = float(f[inside[int(np.argmin(s[inside]))]])
    s_carrier = np.where(f <= f_valley, s, 0.0)
    s_rider = np.where(f > f_valley, s, 0.0)
    mom_c = moments_from_dict(_lib_moments(f, s_carrier))
    mom_r = moments_from_dict(_lib_moments(f, s_rider))
    mom_all = moments_from_dict(_lib_moments(f, s))
    if f_ice >= f_wave:
        f_carrier, f_rider = float(f_wave), float(f_ice)
    else:
        f_carrier, f_rider = float(f_ice), float(f_wave)
    return BandSplit(
        freqs=f, s_comb=s, f_valley_hz=f_valley,
        f_carrier_hz=f_carrier, f_rider_hz=f_rider,
        mom_carrier=mom_c, mom_rider=mom_r, mom_comb=mom_all,
        beta_overlap=overlap_weight(f, s_carrier, s_rider),
    )


def _band_psd(case: HullGirderCase, f_ice: float, f_wave: float, sigma_ice: float,
              sigma_wave: float, wave_kind: str, wave_kwargs: Dict[str, float],
              n: int, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ice_shape = gaussian_band_shape(f_ice, 0.15)
    wave_shape = stress_spectrum_shape(wave_kind, f_wave, **wave_kwargs)
    freqs, s_ice, _ = component_grid(ice_shape, sigma_ice, n, fs)
    _, s_wave, _ = component_grid(wave_shape, sigma_wave, n, fs)
    return freqs, s_ice, s_wave


def _rainflow_rate(series: np.ndarray, dt: float, m: float) -> float:
    r, _means, c = rainflow_cycles(turning_points(series))
    return float(damage_from_cycles(r, c, m) / (series.size * dt))


def band_case(
    case: Optional[HullGirderCase] = None,
    n_ratio: float = 10.0,
    r: Optional[float] = None,
    wave_kind: str = "jonswap",
    wave_kwargs: Optional[Dict[str, float]] = None,
    seed: int = 0,
    duration: float = 1200.0,
    fs: float = 40.0,
    m: float = 3.0,
    sigma_carrier: float = 1.0,
) -> Dict[str, Any]:
    """One two-band case: PSD data, spectral band damages, and rainflow truth.

    ``n_ratio = f_ice / f_wave`` (``> 1`` puts the ice band at the higher
    frequency, ``< 1`` the other way round).  ``r`` is
    ``sigma_rider / sigma_carrier``; ``r=None`` puts the two bands on an equal
    per-component damage footing (``r = N**(-1/m)``), the convention of the
    node-``P82W4-N06`` benchmark.
    """
    case = case or HullGirderCase()
    if n_ratio <= 0:
        raise ValueError("n_ratio must be positive")
    f_ice = case.ice_band_hz
    f_wave = f_ice / float(n_ratio)
    n_sep = float(max(f_ice, f_wave) / min(f_ice, f_wave))
    rr = equal_damage_amplitude_ratio(n_sep, m) if r is None else float(r)
    sigma_rider = sigma_carrier * rr
    if f_ice >= f_wave:
        sigma_ice, sigma_wave = sigma_rider, sigma_carrier
    else:
        sigma_ice, sigma_wave = sigma_carrier, sigma_rider
    n = int(round(duration * fs))
    freqs, s_ice, s_wave = _band_psd(case, f_ice, f_wave, sigma_ice, sigma_wave,
                                     wave_kind, dict(wave_kwargs or {}), n, fs)
    s_comb = s_ice + s_wave
    split = band_split(freqs, s_comb, f_ice, f_wave)
    d_carrier = dirlik_damage_rate(split.mom_carrier, 1.0, m)
    d_rider = dirlik_damage_rate(split.mom_rider, 1.0, m)
    d_comb_dirlik = dirlik_damage_rate(split.mom_comb, 1.0, m)
    d_comb_tb = benasciutti_tovo_damage_rate(split.mom_comb, 1.0, m)
    d_comb_nb = rayleigh_damage_rate(split.mom_comb, 1.0, m)
    x = synthesize_gaussian_history(s_comb, n, fs, seed=int(seed) + 500)
    d_truth = _rainflow_rate(x, 1.0 / fs, m)
    s_c = np.where(freqs <= split.f_valley_hz, s_comb, 0.0)
    s_r = np.where(freqs > split.f_valley_hz, s_comb, 0.0)
    d_c_rf = _rainflow_rate(synthesize_gaussian_history(s_c, n, fs, seed=int(seed) + 900),
                            1.0 / fs, m)
    d_r_rf = _rainflow_rate(synthesize_gaussian_history(s_r, n, fs, seed=int(seed) + 1300),
                            1.0 / fs, m)
    # isolated *component* rainflow damages (the plain sum-rule denominator of
    # criterion (a): D_ice + D_wave, each from its own component PSD)
    d_ice_rf = _rainflow_rate(synthesize_gaussian_history(s_ice, n, fs, seed=int(seed) + 1700),
                              1.0 / fs, m)
    d_wave_rf = _rainflow_rate(synthesize_gaussian_history(s_wave, n, fs, seed=int(seed) + 2100),
                               1.0 / fs, m)
    return {
        "n_ratio": float(n_sep),
        "n_ratio_ice_over_wave": float(n_ratio),
        "f_ice_hz": f_ice,
        "f_wave_hz": f_wave,
        "r_rider_over_carrier": float(split.r),
        "sigma_ice_mpa": float(sigma_ice),
        "sigma_wave_mpa": float(sigma_wave),
        "seed": int(seed),
        "duration_s": float(duration),
        "fs_hz": float(fs),
        "m": float(m),
        "wave_kind": wave_kind,
        "wave_kwargs": dict(wave_kwargs or {}),
        "split": split.as_dict(),
        "d_carrier_dirlik": d_carrier,
        "d_rider_dirlik": d_rider,
        "d_split_sum": d_carrier + d_rider,
        "d_comb_dirlik": d_comb_dirlik,
        "d_comb_tb": d_comb_tb,
        "d_comb_narrowband": d_comb_nb,
        "d_truth": d_truth,
        "d_carrier_rainflow": d_c_rf,
        "d_rider_rainflow": d_r_rf,
        "rho_truth": float(d_truth / (d_c_rf + d_r_rf)) if (d_c_rf + d_r_rf) > 0 else float("nan"),
        "d_ice_component_rainflow": d_ice_rf,
        "d_wave_component_rainflow": d_wave_rf,
        "rho_components": float(d_truth / (d_ice_rf + d_wave_rf))
        if (d_ice_rf + d_wave_rf) > 0 else float("nan"),
        "kappa_truth_dirlik_basis": ((d_truth - d_rider) / d_carrier) if d_carrier > 0 else float("nan"),
        "freqs": freqs,
        "s_comb": s_comb,
    }


# ---------------------------------------------------------------------------
# Carrier amplification closure
# ---------------------------------------------------------------------------
def riding_kappa_convolution(sigma_carrier: float, sigma_rider: float,
                             m: float = 3.0, n_grid: int = 400) -> float:
    """Pure-theory carrier amplification ``E[(R + delta)^m] / E[R^m]``.

    ``R`` is the carrier rainflow range (Rayleigh, scale ``sqrt(2) sigma_c``) and
    ``delta = |x(t_max)| + |x(t_min)|`` is the *enlargement* of that range by the
    rider: at the carrier maximum and at the carrier minimum the rider sits near
    its own local extrema, so each of the two magnitudes is Rayleigh distributed
    with mean ``sigma_r sqrt(pi/2)`` and ``E[delta] = 2 sqrt(pi/2) sigma_r``
    -- the enlargement measured in node ``P82W4-N06``.  For ``m = 1`` this gives
    the closed form ``kappa = 1 + sqrt(2) r`` (pinned by a test).

    Kept as the *theory anchor* the calibrated closure is scored against; it is
    explicitly **not** what the module uses for the closure, because it ignores
    the rider cycles that rainflow merges into the enlarged carrier excursions
    (measured: the anchor over-predicts the truth by ~60 % at ``r = 0.465``,
    because enlarging every carrier range *and* keeping all rider cycles counts
    the same amplitude twice -- the very double counting the node measures).
    """
    if sigma_carrier <= 0 or sigma_rider <= 0:
        return 1.0
    R = np.linspace(1e-12, 14.0 * sigma_carrier, n_grid)
    pR = R / (2.0 * sigma_carrier ** 2) * np.exp(-R ** 2 / (4.0 * sigma_carrier ** 2))
    x = np.linspace(1e-12, 9.0 * sigma_rider, n_grid)
    px = x / sigma_rider ** 2 * np.exp(-x ** 2 / (2.0 * sigma_rider ** 2))
    dd = float(x[1] - x[0])
    pd = np.convolve(px * dd, px * dd) / dd          # pdf of |x1| + |x2|
    delta = np.arange(pd.size) * dd
    pd = pd / np.trapezoid(pd, delta)
    inner = np.trapezoid(pd[None, :] * (R[:, None] + delta[None, :]) ** m, delta, axis=1)
    num = float(np.trapezoid(pR * inner, R))
    den = float(np.trapezoid(pR * R ** m, R))
    return num / den if den > 0 else 1.0


@dataclass(frozen=True)
class KappaClosure:
    """Calibrated carrier amplification ``kappa(r, N) = 1 + A N^b r^p / (1 + C N^c r^q)``.

    ``r = sigma_rider/sigma_carrier``, ``N = f_rider/f_carrier >= 1``.  Both the
    numerator and the denominator carry an ``N`` power so the closure can
    reproduce the measured structure: ``kappa - 1`` grows with ``N`` at small
    ``r`` (weak riding enlarges the carrier ranges without disturbing the
    carrier cycle family) and shrinks with ``N`` at large ``r`` (strong riding
    takes over the counting and the carrier family dissolves).  It is a
    *semi-empirical* closure: calibrated on Level-A rainflow truth, and labelled
    as such wherever it is used.
    """

    A: float = 1.0
    b: float = 0.3
    p: float = 1.0
    C: float = 0.5
    c: float = 0.5
    q: float = 2.0
    n_points: int = 0
    rms_rel_err_pct: float = float("nan")
    max_rel_err_pct: float = float("nan")

    def kappa(self, r: float, n_ratio: float) -> float:
        r = max(float(r), 0.0)
        N = max(float(n_ratio), 1.0)
        num = self.A * N ** self.b * r ** self.p
        den = 1.0 + self.C * N ** self.c * r ** self.q
        return float(1.0 + num / den)

    def as_dict(self) -> Dict[str, float]:
        return {
            "A": self.A, "b": self.b, "p": self.p,
            "C": self.C, "c": self.c, "q": self.q,
            "n_points": self.n_points,
            "rms_rel_err_pct": self.rms_rel_err_pct,
            "max_rel_err_pct": self.max_rel_err_pct,
        }


def fit_kappa_closure(rows: Sequence[Dict[str, Any]],
                      max_nfev: int = 6000) -> KappaClosure:
    """Least-squares fit of :class:`KappaClosure` to measured ``kappa`` values.

    The residual is taken on ``log(kappa - 1)`` because ``kappa`` spans 1.1 to
    12; that weights both ends of the range instead of letting the large values
    dominate.
    """
    from scipy.optimize import least_squares

    r = np.array([row["r_rider_over_carrier"] for row in rows], float)
    N = np.array([row["n_ratio"] for row in rows], float)
    k = np.array([row["kappa_truth_dirlik_basis"] for row in rows], float)
    keep = np.isfinite(k) & (k > 1.0) & (r > 0)
    r, N, k = r[keep], N[keep], k[keep]
    if r.size < 6:
        raise ValueError("not enough calibration points")
    y = np.log(k - 1.0)

    def model(theta, rr, nn):
        A, b, p, C, c, q = theta
        num = np.maximum(A, 1e-9) * nn ** b * rr ** p
        den = 1.0 + np.maximum(C, 1e-9) * nn ** c * rr ** q
        return np.log(np.maximum(num / den, 1e-12))

    start = np.array([1.0, 0.3, 1.0, 0.5, 0.5, 2.0])
    lower = np.array([1e-3, -2.0, 0.05, 1e-3, -2.0, 0.05])
    upper = np.array([1e3, 3.0, 6.0, 1e3, 3.0, 6.0])
    sol = least_squares(lambda th: model(th, r, N) - y, start,
                        bounds=(lower, upper), max_nfev=max_nfev)
    A, b, p, C, c, q = [float(v) for v in sol.x]
    pred = 1.0 + np.exp(model(sol.x, r, N))
    rel = pred / k - 1.0
    return KappaClosure(A=A, b=b, p=p, C=C, c=c, q=q, n_points=int(r.size),
                        rms_rel_err_pct=float(np.sqrt(np.mean(rel ** 2)) * 100.0),
                        max_rel_err_pct=float(np.max(np.abs(rel)) * 100.0))


def two_band_riding_damage(case_row: Dict[str, Any],
                           closure: Optional[KappaClosure] = None,
                           m: Optional[float] = None,
                           n_separated: float = N_SEPARATED) -> Dict[str, Any]:
    """Two-band spectral damage closure for one :func:`band_case` row.

    Regime switch (declared, :data:`N_SEPARATED`):

    * ``N < n_separated``: the bands are not resolvable as separate cycle
      families, the process is a wide-band single-peak process, and the combined
      Dirlik damage is used unchanged;
    * ``N >= n_separated``: ``D = D_rider + kappa(r, N) * D_carrier``, where both
      family damages are the Dirlik damage of that band's own moments and
      ``kappa`` is the only free ingredient.
    """
    m = float(m if m is not None else case_row.get("m", 3.0))
    split = case_row["split"]
    N = float(split["N_freq_ratio"])
    r = float(split["r_rider_over_carrier"])
    d_dirlik = float(case_row["d_comb_dirlik"])
    d_rider = float(case_row["d_rider_dirlik"])
    d_carrier = float(case_row["d_carrier_dirlik"])
    if N < n_separated or closure is None:
        return {"regime": "merged_band_dirlik", "kappa": 1.0, "damage": d_dirlik,
                "d_rider": d_rider, "d_carrier": d_carrier, "n_ratio": N, "r": r}
    kappa = closure.kappa(r, N)
    return {"regime": "two_band_riding", "kappa": kappa,
            "damage": d_rider + kappa * d_carrier,
            "d_rider": d_rider, "d_carrier": d_carrier, "n_ratio": N, "r": r}


# ---------------------------------------------------------------------------
# Criterion (a): where is the plain sum rule allowed?
# ---------------------------------------------------------------------------
def sum_rule_envelope(closure: KappaClosure, tol: float = 0.03, n_grid: int = 400,
                      d_carrier_over_rider: float = 1.0) -> Dict[str, Any]:
    """Largest riding amplitude ratio ``r`` for which the sum rule holds +-tol.

    ``rho - 1 = (kappa - 1) / (1 + D_carrier/D_rider)``.  With the two band
    damage rates available from the spectra the sum rule is therefore
    *decidable* without any time-domain counting.  ``d_carrier_over_rider = 1``
    is the equal-component-damage convention of the benchmark; a case with known
    component damages inserts its own ratio.
    """
    rows = []
    for N in (2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
        r = np.linspace(1e-4, 3.0, n_grid)
        kappa = np.array([closure.kappa(float(x), N) for x in r])
        rho = 1.0 + (kappa - 1.0) / (1.0 + d_carrier_over_rider)
        ok = np.abs(rho - 1.0) <= tol
        rows.append({
            "N_freq_ratio": N,
            "r_max_sum_rule": float(r[ok].max()) if ok.any() else 0.0,
            "sum_rule_at_r_tiny": float(rho[0]),
            "kappa_at_r_0p05": closure.kappa(0.05, N),
        })
    return {
        "tol": tol,
        "d_carrier_over_rider": float(d_carrier_over_rider),
        "rows": rows,
        "statement": (
            "D_comb/(D1+D2) stays inside 1+-tol while the riding amplitude ratio "
            "r stays below r_max(N); kappa(r,N) makes the check decidable from "
            "the two band spectra alone."
        ),
    }


def kappa_calibration_grid(
    case: Optional[HullGirderCase] = None,
    n_ratios: Sequence[float] = (5.0, 10.0, 20.0, 50.0, 100.0),
    r_values: Sequence[float] = (0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.5),
    seeds: Sequence[int] = (0, 1, 2),
    duration: float = 1200.0,
    fs: float = 40.0,
    m: float = 3.0,
    wave_kind: str = "jonswap",
    wave_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Level-A rainflow grid of ``kappa`` used to calibrate the closure."""
    case = case or HullGirderCase()
    rows: List[Dict[str, Any]] = []
    for N in n_ratios:
        for r in r_values:
            kappas, truths, splits, rhos = [], [], [], []
            for s in seeds:
                out = band_case(case, n_ratio=N, r=r, wave_kind=wave_kind,
                                wave_kwargs=wave_kwargs, seed=int(s),
                                duration=duration, fs=fs, m=m)
                kappas.append(out["kappa_truth_dirlik_basis"])
                truths.append(out["d_truth"])
                splits.append(out["d_split_sum"])
                rhos.append(out["rho_truth"])
            karr = np.array(kappas, float)
            rows.append({
                "n_ratio": float(N),
                "r_rider_over_carrier": float(r),
                "kappa_truth_dirlik_basis": float(np.mean(karr)),
                "kappa_seed_std": float(np.std(karr, ddof=1)) if karr.size > 1 else 0.0,
                "d_truth_mean": float(np.mean(truths)),
                "d_split_sum_mean": float(np.mean(splits)),
                "rho_truth": float(np.mean(rhos)),
                "seeds": [int(s) for s in seeds],
            })
    closure = fit_kappa_closure(rows)
    for row in rows:
        row["kappa_closure"] = closure.kappa(row["r_rider_over_carrier"], row["n_ratio"])
        row["closure_rel_err_pct"] = 100.0 * (row["kappa_closure"]
                                              / row["kappa_truth_dirlik_basis"] - 1.0)
        row["kappa_convolution_theory"] = riding_kappa_convolution(
            1.0, row["r_rider_over_carrier"], m)
    return {
        "rows": rows,
        "closure": closure.as_dict(),
        "grid": {"n_ratios": [float(x) for x in n_ratios],
                 "r_values": [float(x) for x in r_values],
                 "seeds": [int(s) for s in seeds], "duration_s": duration,
                 "fs_hz": fs, "m": m, "wave_kind": wave_kind,
                 "wave_kwargs": dict(wave_kwargs or {})},
    }


# ---------------------------------------------------------------------------
# Criterion (b): spectral-method benchmark
# ---------------------------------------------------------------------------
def bimodal_benchmark(
    case: Optional[HullGirderCase] = None,
    levels: Optional[Sequence[str]] = None,
    n_ratios: Sequence[float] = (0.3, 1.0, 10.0, 100.0),
    seeds: Sequence[int] = tuple(range(6)),
    closure: Optional[KappaClosure] = None,
    duration: float = 1200.0,
    fs: float = 40.0,
    m: float = 3.0,
    threshold_pct: float = 10.0,
) -> Dict[str, Any]:
    """Dirlik / TB / narrow-band / two-band closure vs rainflow truth.

    Every model and the rainflow counter see the same band-limited combined PSD
    (``m0..m4`` integrated on the very rFFT bins the histories are synthesised
    from), so the only difference left is the counting model.  Components are
    put on an equal per-component damage footing (``r = N**(-1/m)``), which makes
    the benchmark independent of any absolute stress scale.
    """
    case = case or HullGirderCase()
    names = list(levels) if levels is not None else ["narrow_gamma6", "medium_gamma3p3",
                                                     "wide_bimodal_rp3"]
    rows: List[Dict[str, Any]] = []
    for name in names:
        cfg = WIDTH_LEVELS[name]
        for n_ratio in n_ratios:
            truths, cands = [], []
            dur = duration * max(1.0, min(3.0, float(n_ratio) / 20.0))
            for s in seeds:
                out = band_case(case, n_ratio=n_ratio, wave_kind=cfg["wave_kind"],
                                wave_kwargs=cfg["wave_kwargs"], seed=int(s),
                                duration=dur, fs=fs, m=m)
                truths.append(out["d_truth"])
                cands.append(out)
            t = float(np.mean(truths))
            sd = float(np.std(truths, ddof=1)) if len(truths) > 1 else 0.0
            first = cands[0]
            d_dirlik = float(np.mean([c["d_comb_dirlik"] for c in cands]))
            d_tb = float(np.mean([c["d_comb_tb"] for c in cands]))
            d_nb = float(np.mean([c["d_comb_narrowband"] for c in cands]))
            two_band = [two_band_riding_damage(c, closure, m) for c in cands]
            d_two = float(np.mean([tb["damage"] for tb in two_band]))
            d_split = float(np.mean([c["d_split_sum"] for c in cands]))
            rows.append({
                "level": name,
                "wave_kind": cfg["wave_kind"],
                "wave_kwargs": cfg["wave_kwargs"],
                "n_ratio": float(first["n_ratio"]),
                "n_ratio_ice_over_wave": float(n_ratio),
                "r_rider_over_carrier": float(first["r_rider_over_carrier"]),
                "alpha2": float(first["split"]["alpha2_comb"]),
                "alpha1": float(first["split"]["alpha1_comb"]),
                "sigma_comb_mpa": float(math.sqrt(first["split"]["m0_carrier"]
                                                  + first["split"]["m0_rider"])),
                "beta_overlap": float(first["split"]["beta_overlap"]),
                "regime_two_band": two_band[0]["regime"],
                "kappa_used": float(two_band[0]["kappa"]),
                "d_truth_mean": t,
                "d_truth_rel_std_pct": 100.0 * sd / t if t > 0 else float("nan"),
                "d_dirlik": d_dirlik,
                "d_tb": d_tb,
                "d_narrowband": d_nb,
                "d_split_sum": d_split,
                "d_two_band": d_two,
                "err_dirlik_pct": 100.0 * (d_dirlik / t - 1.0) if t > 0 else float("nan"),
                "err_tb_pct": 100.0 * (d_tb / t - 1.0) if t > 0 else float("nan"),
                "err_narrowband_pct": 100.0 * (d_nb / t - 1.0) if t > 0 else float("nan"),
                "err_split_sum_pct": 100.0 * (d_split / t - 1.0) if t > 0 else float("nan"),
                "err_two_band_pct": 100.0 * (d_two / t - 1.0) if t > 0 else float("nan"),
            })
    def _worst(key):
        return float(max(abs(r[key]) for r in rows))

    separated = [r for r in rows if r["n_ratio"] >= N_SEPARATED]
    merged = [r for r in rows if r["n_ratio"] < N_SEPARATED]
    worst_two_sep = max((abs(r["err_two_band_pct"]) for r in separated), default=float("nan"))
    worst_two = _worst("err_two_band_pct")
    return {
        "rows": rows,
        "levels": names,
        "n_ratios": [float(x) for x in n_ratios],
        "seeds": [int(s) for s in seeds],
        "duration_s": duration,
        "fs_hz": fs,
        "m": m,
        "threshold_pct": threshold_pct,
        "worst_err_dirlik_pct": _worst("err_dirlik_pct"),
        "worst_err_tb_pct": _worst("err_tb_pct"),
        "worst_err_narrowband_pct": _worst("err_narrowband_pct"),
        "worst_err_split_sum_pct": _worst("err_split_sum_pct"),
        "worst_err_two_band_pct": worst_two,
        "worst_err_two_band_separated_pct": worst_two_sep,
        "n_separated": N_SEPARATED,
        "n_separated_band_points": len(separated),
        "n_merged_band_points": len(merged),
        "pass_dirlik": bool(_worst("err_dirlik_pct") <= threshold_pct),
        "pass_tb": bool(_worst("err_tb_pct") <= threshold_pct),
        "pass_two_band": bool(worst_two <= threshold_pct),
    }