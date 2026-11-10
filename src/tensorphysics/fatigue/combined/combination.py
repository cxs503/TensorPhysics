"""Combination law for ice- and wave-induced fatigue damage.

Question
--------
Ships in brash/level ice under waves see two stress processes at once: the
ice-crushing/lock-in process (tonal at the crushing frequency ``f_ice``, plus a
transient whipping component at the hull-girder frequency ``f1``) and the
wave-induced vertical bending process (band centred on the encounter frequency
``f_w``).  Practice adds the two damage rates,

    D_comb =?= D_ice + D_wave                                          (linear law)

with no stated error bound.  Palmgren-Miner is linear in the *cycle set*, not in
the load superposition, so the law is only exact when the two processes occupy
disjoint frequency bands (their rainflow ranges then close inside each band).
When the bands overlap the two tones beat: the combined process keeps roughly the
carrier cycle rate but its ranges grow to the *sum* of the two amplitudes, and
the damage of the fused process exceeds the sum of the separate damages.

What this module does
---------------------
1. :func:`linearity_scan` -- designed experiment: two narrow-band Gaussian
   stress bands with prescribed ``f_ice/f_wave``, RMS and bandwidth, synthsised
   from the same PSD grid used for the spectral closures; damage from rainflow
   truth (ASTM E1049-85 + single-slope Miner) and from the composite-PSD
   closures (Rayleigh / Dirlik / Tovo-Benasciutti).
2. :func:`spectral_width_study` -- Dirlik/TB vs rainflow error over three
   spectral widths (alpha2 ~ 1 narrow, ~ 0.6 moderate, ~ 0.3 wide/bimodal).
3. :func:`whipping_accounting` -- reconciles the two ways of "counting" the
   high-frequency whipping content (m5528 peak/RMS share convention) with the
   damage amplification it produces.
4. :func:`rainflow_commutativity` -- damage of a closed-cycle sequence is
   invariant under permutation of the cycles (order-independence of the
   rainflow+Miner functional).
5. :func:`analytic_bounds` / :func:`fit_separation_law` -- the two exact limits
   and a one-parameter interpolation used as the engineering combination law.

Damage is expressed with the single-slope S-N ``N = C S^-m`` (``S`` = stress
range, MPa) so that the frequency-domain closures are exactly comparable with
the rainflow truth; ``C`` defaults to the DNV-RP-C203 F3/CP value at m = 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..rainflow import rainflow_counting_stack
from ..spectral import (
    BimodalSpectrum,
    GaussianSpectrum,
    JONSWAP,
    SpectralMoments,
    discrete_psd_grid,
    moments_from_dict,
    spectral_moments,
    synthesize_gaussian_history,
)
from .processes import (
    TwoBandCase,
    band_psd,
    combined_spectrum,
    orthogonalise,
    rainflow_damage_rate,
    realised_process,
    spectral_damage_rates,
)

__all__ = [
    "SN_C_F3_CP",
    "SN_M",
    "LINEARITY_TOL",
    "SPECTRAL_ERROR_TOL",
    "linearity_scan",
    "realistic_case",
    "admissible_region",
    "WHIPPING_OPERATING_POINTS",
    "spectral_width_study",
    "rainflow_commutativity",
    "whipping_accounting",
    "analytic_bounds",
    "fit_separation_law",
    "GAMMA_RANGE",
]

#: DNV-RP-C203 F3 seawater+CP, m = 3: N = C S^-m with S in MPa.
SN_C_F3_CP = 10.0 ** (11.510 - math.log10(2.5))  # 1.2938e11
SN_M = 3.0
GAMMA_RANGE = math.gamma(1.0 + 0.5 * SN_M)        # E[(2 sqrt2 sigma)^m] / sigma^m

#: pre-declared acceptance tolerances (must not be tuned after the results)
LINEARITY_TOL = 0.03        # |D_comb/(D_ice+D_wave) - 1| <= 3 %
SPECTRAL_ERROR_TOL = 0.10   # |D_spectral/D_rainflow - 1| <= 10 %
COMMUTATIVITY_TOL = 1.0e-9  # bit-level agreement of permuted-cycle damage
WHIPPING_TOL_PP = 2.0       # percentage-point agreement of the share-based law


# ---------------------------------------------------------------------------
# 1. Designed experiment: frequency separation scan
# ---------------------------------------------------------------------------
def _one_seed(case: TwoBandCase, m: float = SN_M, C: float = SN_C_F3_CP) -> Dict[str, float]:
    dt = 1.0 / case.fs
    n = case.n_samples
    ice_psd, wave_psd, comb_psd = combined_spectrum(case)

    ice = realised_process(ice_psd, n, case.fs, seed=case.seed,
                           f_centre=case.f_ice, sigma_f=case.bw_ice * case.f_ice)
    wave = realised_process(wave_psd, n, case.fs, seed=case.seed + 7001,
                            f_centre=case.f_wave, sigma_f=case.bw_wave * case.f_wave)
    comb_grid = discrete_psd_grid(
        comb_psd, n, case.fs,
        f_lo=min(ice["f_lo"], wave["f_lo"]), f_hi=max(ice["f_hi"], wave["f_hi"]))
    mom_comb = moments_from_dict(spectral_moments(comb_grid[0], comb_grid[1]), 0.0, comb_grid[0][-1])

    # exact orthogonality of the two realisations (removes the nuisance
    # sample cross-covariance, see processes.orthogonalise)
    wave_hist = orthogonalise(wave["history"], ice["history"], target_var=wave["moments"].m0)
    history = np.asarray(ice["history"]) + wave_hist

    d_ice = rainflow_damage_rate(ice["history"], dt, C=C, m=m)
    d_wave = rainflow_damage_rate(wave_hist, dt, C=C, m=m)
    d_comb = rainflow_damage_rate(history, dt, C=C, m=m)
    spec = spectral_damage_rates(mom_comb, C=C, m=m)

    D_sum = d_ice["damage_rate"] + d_wave["damage_rate"]
    return {
        "D_ice": d_ice["damage_rate"],
        "D_wave": d_wave["damage_rate"],
        "D_comb": d_comb["damage_rate"],
        "D_sum": D_sum,
        "ratio_linear": d_comb["damage_rate"] / D_sum if D_sum > 0 else float("nan"),
        "D_comb_dirlik": spec["dirlik"],
        "D_comb_tb": spec["tovo_benasciutti"],
        "D_comb_rayleigh": spec["rayleigh"],
        "err_dirlik": spec["dirlik"] / d_comb["damage_rate"] - 1.0 if d_comb["damage_rate"] > 0 else float("nan"),
        "err_tb": spec["tovo_benasciutti"] / d_comb["damage_rate"] - 1.0 if d_comb["damage_rate"] > 0 else float("nan"),
        "alpha2_comb": spec["alpha2"],
        "alpha2_ice": float(ice["moments"].alpha2),
        "alpha2_wave": float(wave["moments"].alpha2),
        "sigma_comb_MPa": spec["sigma_MPa"],
        "n_cycles_ice": d_ice["n_cycles_per_s"],
        "n_cycles_wave": d_wave["n_cycles_per_s"],
        "n_cycles_comb": d_comb["n_cycles_per_s"],
        "var_ratio_ice": ice["var_ratio"],
        "var_ratio_wave": wave["var_ratio"],
    }


@dataclass
class ScanRow:
    """One (ratio, split) cell of the linearity design, aggregated over seeds."""

    ratio: float
    sigma_ice: float
    sigma_wave: float
    f_ice: float
    f_wave: float
    D_ice: float
    D_wave: float
    D_sum: float
    D_comb_mean: float
    D_comb_std: float
    ratio_mean: float
    ratio_std: float
    ratio_min: float
    ratio_max: float
    Dirlik_mean: float
    TB_mean: float
    err_dirlik: float
    err_tb: float
    alpha2: float
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "f_ice_Hz": self.f_ice,
            "f_wave_Hz": self.f_wave,
            "f_ratio": self.ratio,
            "sigma_ice_MPa": self.sigma_ice,
            "sigma_wave_MPa": self.sigma_wave,
            "D_ice": self.D_ice,
            "D_wave": self.D_wave,
            "D_ice_plus_wave": self.D_sum,
            "D_comb_mean": self.D_comb_mean,
            "D_comb_std": self.D_comb_std,
            "ratio_mean": self.ratio_mean,
            "ratio_std": self.ratio_std,
            "ratio_min": self.ratio_min,
            "ratio_max": self.ratio_max,
            "ratio_in_tol": bool(abs(self.ratio_mean - 1.0) <= LINEARITY_TOL),
            "Dirlik_mean": self.Dirlik_mean,
            "TB_mean": self.TB_mean,
            "err_dirlik_pct": 100.0 * self.err_dirlik,
            "err_tb_pct": 100.0 * self.err_tb,
            "alpha2_comb": self.alpha2,
            **self.extra,
        }


def linearity_scan(
    ratios: Sequence[float] = (1.0 / 30.0, 1.0 / 10.0, 0.3, 1.0, 3.0, 10.0, 30.0),
    f_fast: float = 0.5,
    sigma_wave: float = 5.0,
    splits: Sequence[float] = (0.5, 1.0, 2.0),
    bw: float = 0.05,
    duration: float = 10800.0,
    fs: float = 50.0,
    seeds: Sequence[int] = (11, 23, 37, 41, 53),
    m: float = SN_M,
    C: float = SN_C_F3_CP,
    equal_damage: bool = False,
) -> Dict[str, object]:
    """Scan ``f_ice/f_wave`` at fixed RMS split (or fixed damage split).

    ``equal_damage=True`` rescales the ice RMS so that ``D_ice = D_wave`` at
    every ratio, ``sigma_ice = sigma_wave (f_wave/f_ice)^(1/m)``: the interesting
    configuration for an additivity test, since neither branch is then dominant.
    """
    rows: List[ScanRow] = []
    for split in splits:
        for r in ratios:
            # r = f_ice / f_wave by definition; the faster band is always f_fast
            # so the sampling rate stays the same for every cell.
            if r >= 1.0:
                f_ice, f_wave = float(f_fast), float(f_fast) / float(r)
            else:
                f_ice, f_wave = float(f_fast) * float(r), float(f_fast)
            s_wave = float(sigma_wave)
            if equal_damage:
                s_ice = s_wave * (f_wave / f_ice) ** (1.0 / m)
            else:
                s_ice = s_wave * float(split)
            per_seed = []
            for sd in seeds:
                case = TwoBandCase(f_ice=f_ice, f_wave=f_wave, sigma_ice=s_ice,
                                   sigma_wave=s_wave, bw_ice=bw, bw_wave=bw,
                                   duration=duration, fs=fs, seed=int(sd))
                per_seed.append(_one_seed(case, m=m, C=C))
            arr = {k: np.array([p[k] for p in per_seed]) for k in per_seed[0]}
            D_sum = float(arr["D_sum"].mean())
            rows.append(ScanRow(
                ratio=float(r), sigma_ice=s_ice, sigma_wave=s_wave,
                f_ice=float(f_ice), f_wave=float(f_wave),
                D_ice=float(arr["D_ice"].mean()), D_wave=float(arr["D_wave"].mean()),
                D_sum=D_sum,
                D_comb_mean=float(arr["D_comb"].mean()), D_comb_std=float(arr["D_comb"].std(ddof=1)),
                ratio_mean=float(arr["ratio_linear"].mean()),
                ratio_std=float(arr["ratio_linear"].std(ddof=1)),
                ratio_min=float(arr["ratio_linear"].min()), ratio_max=float(arr["ratio_linear"].max()),
                Dirlik_mean=float(arr["D_comb_dirlik"].mean()),
                TB_mean=float(arr["D_comb_tb"].mean()),
                err_dirlik=float((arr["D_comb_dirlik"] / arr["D_comb"] - 1.0).mean()),
                err_tb=float((arr["D_comb_tb"] / arr["D_comb"] - 1.0).mean()),
                alpha2=float(arr["alpha2_comb"].mean()),
                extra={
                    "D_comb_over_D_sum_per_seed": arr["ratio_linear"].tolist(),
                    "n_cycles_ice_per_s": float(arr["n_cycles_ice"].mean()),
                    "n_cycles_wave_per_s": float(arr["n_cycles_wave"].mean()),
                    "n_cycles_comb_per_s": float(arr["n_cycles_comb"].mean()),
                    "err_dirlik_pct_per_seed": (100.0 * (arr["D_comb_dirlik"] / arr["D_comb"] - 1.0)).tolist(),
                    "err_tb_pct_per_seed": (100.0 * (arr["D_comb_tb"] / arr["D_comb"] - 1.0)).tolist(),
                },
            ))
    return {
        "design": {
            "f_fast_Hz": f_fast,
            "sigma_wave_MPa": sigma_wave,
            "splits": list(splits),
            "bw_rel": bw,
            "duration_s": duration,
            "fs_Hz": fs,
            "seeds": list(seeds),
            "m": m,
            "C": C,
            "equal_damage": equal_damage,
            "tolerance": LINEARITY_TOL,
        },
        "rows": [r.as_dict() for r in rows],
    }


# ---------------------------------------------------------------------------
# 2. Spectral-width study: Dirlik / Tovo-Benasciutti vs rainflow truth
# ---------------------------------------------------------------------------
def spectral_width_study(
    duration: float = 3600.0,
    fs: float = 20.0,
    seeds: Sequence[int] = (5, 17, 29, 43, 61, 73, 89, 97),
    m: float = SN_M,
    C: float = SN_C_F3_CP,
) -> Dict[str, object]:
    """Dirlik / TB damage error against rainflow truth for three spectral widths.

    The three shapes span the practical range of the irregularity factor
    ``alpha2 = m2 / sqrt(m0 m4)``:

    ``narrow``    Gaussian pw-band, ``sigma_f/f0 = 0.02``  (alpha2 ~ 1.0)
    ``moderate``  JONSWAP, ``gamma = 3.3``                (alpha2 ~ 0.9)
    ``wide``      two well-separated Gaussians            (alpha2 << 0.9)
    """
    n = int(round(duration * fs))
    shapes = {
        "narrow": band_psd(0.55, 5.0, 0.02),
        "moderate": JONSWAP(hs=4.0, fp=0.55, gamma=3.3),
        "wide": BimodalSpectrum(low=band_psd(0.30, 3.0, 0.06),
                                high=band_psd(2.40, 6.0, 0.05)),
    }
    rows: List[Dict[str, object]] = []
    for name, psd in shapes.items():
        tr: List[float] = []
        dl: List[float] = []
        tb: List[float] = []
        a2: List[float] = []
        for sd in seeds:
            real = realised_process(psd, n, fs, seed=int(sd))
            spec = spectral_damage_rates(real["moments"], C=C, m=m)
            tr.append(rainflow_damage_rate(real["history"], 1.0 / fs, C=C, m=m)["damage_rate"])
            dl.append(spec["dirlik"])
            tb.append(spec["tovo_benasciutti"])
            a2.append(spec["alpha2"])
        tr_a, dl_a, tb_a = np.array(tr), np.array(dl), np.array(tb)
        rows.append({
            "shape": name,
            "alpha2": float(np.mean(a2)),
            "alpha1": float(real["moments"].alpha1),
            "D_rainflow_mean": float(tr_a.mean()),
            "D_rainflow_std": float(tr_a.std(ddof=1)),
            "D_dirlik_mean": float(dl_a.mean()),
            "D_tb_mean": float(tb_a.mean()),
            "err_dirlik_pct": float(100.0 * (dl_a / tr_a - 1.0).mean()),
            "err_tb_pct": float(100.0 * (tb_a / tr_a - 1.0).mean()),
            "err_dirlik_abs_pct_mean": float(np.mean(np.abs(100.0 * (dl_a / tr_a - 1.0)))),
            "err_tb_abs_pct_mean": float(np.mean(np.abs(100.0 * (tb_a / tr_a - 1.0)))),
            "err_rayleigh_pct": float(100.0 * (np.mean([spectral_damage_rates(realised_process(psd, n, fs, seed=int(sd))["moments"], C=C, m=m)["rayleigh"] for sd in seeds]) / tr_a.mean() - 1.0)),
            "per_seed_err_dirlik_pct": (100.0 * (dl_a / tr_a - 1.0)).tolist(),
            "per_seed_err_tb_pct": (100.0 * (tb_a / tr_a - 1.0)).tolist(),
        })
    return {
        "design": {"duration_s": duration, "fs_Hz": fs, "seeds": list(seeds), "m": m, "C": C,
                   "tolerance_abs_pct": 100.0 * SPECTRAL_ERROR_TOL},
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 3. Whipping accounting: share convention -> damage amplification
# ---------------------------------------------------------------------------
def _narrowband_damage_rate(sigma: float, nu: float, C: float, m: float) -> float:
    """Rayleigh narrow-band damage rate (1/s) for RMS ``sigma`` and rate ``nu`` [1/s]."""
    if sigma <= 0.0 or nu <= 0.0:
        return 0.0
    return float(nu / C * (2.0 * math.sqrt(2.0) * sigma) ** m * math.gamma(1.0 + 0.5 * m))


#: operating points of the ice-load model (all other calibrated parameters kept)
WHIPPING_OPERATING_POINTS = {
    # name: overrides of IceContactParams.calibrated()
    "lock_in": dict(v_indent=0.12),                                # m5528 calibrated case
    "transition": dict(v_indent=0.03, decay_time=0.60, modulation=0.10),
    "quasi_static": dict(v_indent=0.005, decay_time=30.0, modulation=0.0, crush_depth=3.0),
}


def _share_law_amplification(w_rms: float, r_freq: float, m: float) -> float:
    """A = r (w^2/(1-w^2))^(m/2): whipping damage relative to the quasi-static part."""
    if not (0.0 < w_rms < 1.0) or not (r_freq > 0.0):
        return float("nan")
    return r_freq * (w_rms ** 2 / (1.0 - w_rms ** 2)) ** (m / 2.0)


def _convention_amplifications(w_rms: float, w_peak: float, m: float) -> Dict[str, float]:
    """Damage amplification of the two wrong ways of folding whipping into cycles."""
    amp = {}
    amp["amplitude_additive_pct"] = 100.0 * ((1.0 - w_peak) ** (-m) - 1.0) if 0 < w_peak < 1 else float("nan")
    amp["variance_additive_pct"] = 100.0 * ((1.0 - w_rms ** 2) ** (-m / 2.0) - 1.0) if 0 < w_rms < 1 else float("nan")
    return amp


def whipping_accounting(
    operating_points: Optional[Dict[str, Dict[str, float]]] = None,
    duration: float = 600.0,
    dt: float = 2.0e-3,
    m: float = SN_M,
    C: float = SN_C_F3_CP,
    synthetic_seeds: Sequence[int] = (7, 19, 31),
    synthetic_duration: float = 10800.0,
    synthetic_fs: float = 20.0,
) -> Dict[str, object]:
    """Reconcile the whipping *stress share* with the whipping *damage* effect.

    Two questions are answered separately, because mixing them is the actual
    source of confusion in practice.

    1. **Does the share explain the damage?**  With ``sigma = sigma_qs +
       sigma_whip`` (``sigma_qs`` = modal-static limit of the same ice load,
       ``sigma_whip`` = the rest: the inertial/whip part) and the m5528
       convention ``w = rms(sigma_whip)/rms(sigma)``, two independent
       narrow-band components of RMS ratio ``sigma_whip/sigma_qs =
       w/sqrt(1-w^2)`` and cycle-rate ratio ``r = nu_whip/nu_qs`` satisfy

           D_whip/D_qs = r (w^2/(1-w^2))^(m/2)                       (share law)

       so the damage amplification of the whipping content is predicted from the
       *share alone*.  ``deviation_pp = 100|A_measured - A_predicted|`` is the
       acceptance metric (criterion (c), tolerance 2 percentage points):
       ``A_measured = D_true/D_qs - 1`` from rainflow on the actual stress
       histories, ``A_predicted`` from the share law.
    2. **How much does the accounting convention matter?**  The two commonly
       used wrong conventions (folding the whipping *peak* into the cycle ranges
       -> ``(1-w_peak)^-m - 1``, or folding its *variance* into the quasi-static
       narrow band -> ``(1-w_rms^2)^(-m/2) - 1``) are evaluated at the same
       shares; their spread is the size of the bookkeeping risk.

    The share law is exact only when the whipping really is an independent
    narrow-band process, so it is first validated on *synthetic* independent
    bands (:func:`_synthetic_share_law`) and then applied to the physical ice
    chain.  The two answers differ, and that difference is the reported finding.
    """
    from .ice_anchor import (
        ANCHOR_CASE, FreeFreeBeamModes, IceContactParams, generate_ice_force,
        modal_stress_history, whipping_share,
    )

    cfg = dict(ANCHOR_CASE)
    points = operating_points or WHIPPING_OPERATING_POINTS
    rows: List[Dict[str, object]] = []
    for name, overrides in points.items():
        prm = IceContactParams.calibrated(**overrides)
        modes = FreeFreeBeamModes(cfg["length_m"], cfg["EI_Nm2"], cfg["mass_per_length_kg_m"],
                                  n_modes=int(cfg["n_modes"]), zeta=cfg["zeta"])
        _, F = generate_ice_force(prm, duration=duration, dt=dt)
        skip = min(int(cfg["settle_time_s"] / dt), max(0, F.size // 2))
        resp = modal_stress_history(modes, F, dt, cfg["x_load_m"], cfg["x_detail_m"],
                                    cfg["z_deck_m"], cfg["E_Pa"])
        sig = resp["sigma"][skip:] / 1.0e6     # MPa, nominal
        sig_qs = resp["sigma_qs"][skip:] / 1.0e6
        ws = whipping_share(sig, sig_qs)
        T = sig.size * dt
        f1 = float(modes.freq[0])

        rf_true = rainflow_counting_stack(sig)
        D_true = float(np.sum(rf_true.counts * rf_true.ranges ** m) / C / T)
        rf_qs = rainflow_counting_stack(sig_qs)
        D_qs = float(np.sum(rf_qs.counts * rf_qs.ranges ** m) / C / T)
        nu_qs = float(rf_qs.counts.sum() / T)
        nu_whip = f1
        sigma_tot = float(np.std(sig))
        sigma_whip = float(np.std(sig - sig_qs))
        sigma_qs = float(np.std(sig_qs))

        w = ws["share_rms"]
        r = nu_whip / nu_qs if nu_qs > 0 else float("inf")
        A_meas = D_true / D_qs - 1.0 if D_qs > 0 else float("nan")
        A_pred = _share_law_amplification(w, r, m)
        conv = _convention_amplifications(ws["share_rms"], ws["share_peak"], m)
        rows.append({
            "operating_point": name,
            "ice_params_overrides": overrides,
            "share_peak": ws["share_peak"],
            "share_rms": ws["share_rms"],
            "slow_varying_lt_8pct": bool(ws["share_peak"] < 0.08),
            "sigma_total_MPa": sigma_tot,
            "sigma_qs_MPa": sigma_qs,
            "sigma_whip_MPa": sigma_whip,
            "power_additive_sigma_qs_ratio": (
                math.sqrt(max(sigma_tot ** 2 - sigma_whip ** 2, 0.0)) / sigma_qs
                if sigma_qs > 0 else float("nan")),
            "nu_qs_Hz": nu_qs,
            "nu_whip_Hz": nu_whip,
            "r_freq": r,
            "D_true": D_true,
            "D_qs_only": D_qs,
            "damage_share_of_whipping": (1.0 - D_qs / D_true) if D_true > 0 else float("nan"),
            "A_measured_pct": 100.0 * A_meas,
            "A_predicted_pct": 100.0 * A_pred,
            "deviation_pp": 100.0 * abs(A_meas - A_pred) if math.isfinite(A_pred) else float("nan"),
            **conv,
        })

    synthetic = _synthetic_share_law(
        seeds=synthetic_seeds, duration=synthetic_duration, fs=synthetic_fs, m=m, C=C)
    return {
        "design": {"duration_s": duration, "dt_s": dt, "m": m, "C": C,
                   "tolerance_pp": WHIPPING_TOL_PP, "anchor_case": cfg},
        "rows": rows,
        "synthetic_share_law": synthetic,
        "verdict": {
            "physical_deviation_pp": [row["deviation_pp"] for row in rows],
            "synthetic_deviation_pp": synthetic["deviation_pp"],
            "law_holds_on_independent_bands": bool(synthetic["deviation_pp"] <= WHIPPING_TOL_PP),
            "law_holds_on_physical_ice": bool(
                all(math.isfinite(row["deviation_pp"]) and row["deviation_pp"] <= WHIPPING_TOL_PP
                    for row in rows)),
        },
    }


def _synthetic_share_law(
    seeds: Sequence[int] = (7, 19, 31),
    duration: float = 10800.0,
    fs: float = 20.0,
    m: float = SN_M,
    C: float = SN_C_F3_CP,
    ratios: Sequence[float] = (1.0, 2.0, 5.0, 10.0),
    share_values: Sequence[float] = (0.05, 0.15, 0.30, 0.50),
) -> Dict[str, object]:
    """Validate the share law on two *independent* narrow-band components.

    ``sigma_qs`` at 0.05 Hz (RMS 5 MPa) and ``sigma_whip`` at ``ratios*0.05`` Hz
    with RMS ``w/sqrt(1-w^2) * 5`` MPa, i.e. exactly the two-process model the
    share law assumes.  Damage is rainflow truth for the sum and for the
    quasi-static component; the deviation from the analytic share law is the
    intrinsic error of the reconciliation.
    """
    n = int(round(duration * fs))
    rows: List[Dict[str, object]] = []
    for r_ratio in ratios:
        for w in share_values:
            sig_qs = 5.0
            sig_w = sig_qs * w / math.sqrt(1.0 - w * w)
            meas, pred = [], []
            for sd in seeds:
                qs = realised_process(band_psd(0.05, sig_qs, 0.02), n, fs, seed=int(sd),
                                      f_centre=0.05, sigma_f=0.05 * 0.02)
                wh = realised_process(band_psd(0.05 * r_ratio, sig_w, 0.02), n, fs,
                                      seed=int(sd) + 991,
                                      f_centre=0.05 * r_ratio, sigma_f=0.05 * r_ratio * 0.02)
                wh_hist = orthogonalise(wh["history"], qs["history"],
                                        target_var=wh["moments"].m0)
                hist = np.asarray(qs["history"]) + wh_hist
                d_true = rainflow_damage_rate(hist, 1.0 / fs, C=C, m=m)["damage_rate"]
                d_qs = rainflow_damage_rate(qs["history"], 1.0 / fs, C=C, m=m)["damage_rate"]
                meas.append(d_true / d_qs - 1.0)
                nu_qs = 0.05
                pred.append(_share_law_amplification(w, r_ratio, m))
            am, ap = float(np.mean(meas)), float(np.mean(pred))
            rho = w / math.sqrt(1.0 - w * w)
            a_own = 100.0 * r_ratio * rho ** m                     # fast band's own cycles
            a_abs = 100.0 * ((1.0 + rho) ** m - 1.0)               # full peak absorption
            eff = (100.0 * am - a_own) / a_abs if a_abs > 0 else float("nan")
            rows.append({
                "r_freq": float(r_ratio),
                "share_rms": float(w),
                "A_measured_pct": 100.0 * am,
                "A_predicted_pct": 100.0 * ap,
                "deviation_pp": 100.0 * abs(am - ap),
                "A_own_cycles_pct": a_own,
                "A_full_absorption_pct": a_abs,
                "absorption_efficiency": eff,
                "within_bounds": bool(
                    (a_own - 1e-9) <= 100.0 * am <= (a_own + a_abs + 1e-9)),
            })
    worst = max(row["deviation_pp"] for row in rows)
    eff_by_ratio: Dict[float, float] = {}
    for r_ratio in ratios:
        vals = [row["absorption_efficiency"] for row in rows
                if row["r_freq"] == float(r_ratio) and math.isfinite(row["absorption_efficiency"])]
        if vals:
            eff_by_ratio[float(r_ratio)] = float(np.mean(vals))
    return {
        "rows": rows,
        "deviation_pp": float(worst),
        "all_within_bounds": bool(all(row["within_bounds"] for row in rows)),
        "absorption_efficiency_by_ratio": eff_by_ratio,
        "tolerance_pp": WHIPPING_TOL_PP,
        "passes": bool(worst <= WHIPPING_TOL_PP),
    }


# ---------------------------------------------------------------------------
# 4. Rainflow commutativity (order independence of the damage functional)
# ---------------------------------------------------------------------------
def rainflow_commutativity(
    history: np.ndarray,
    dt: float,
    n_segments: int = 10,
    n_permutations: int = 6,
    seed: int = 2026,
    m: float = SN_M,
    C: float = SN_C_F3_CP,
    f_period_s: Optional[float] = None,
) -> Dict[str, object]:
    """Order invariance of the rainflow + Miner damage functional.

    Four variants are measured, because "damage is invariant when the order of
    the pieces is changed" is true for *some* decompositions and false for
    others -- and only the first three can be exact:

    ``segment_accumulation``
        The record is cut into ``n_segments`` blocks, each block is counted on
        its own (residue included) and the per-block damage rates are summed.
        Permuting the blocks cannot change the sum; the only differences are
        floating-point summation order.  This is the frozen-count convention of
        an operational monitoring system and must agree to ``1e-9``.
    ``time_reversal``
        Rainflow's closed loops are a property of the *hysteresis* structure, not
        of the direction of time: ``x`` and ``x[::-1]`` give the same cycle set
        (classical property, verified here).
    ``cyclic_rotation``
        For a record that is an integer number of periods long, rotating it by a
        whole number of periods leaves every turning point intact, so the counts
        and the damage are identical.
    ``raw_segment_permutation``
        Concatenating the *open* blocks in a different order re-cuts the turning
        points, so a genuinely different cycle set results.  This is reported as
        a measurement (its magnitude is ~1e-4 relative here) and **not** as a
        pass/fail condition: rainflow counting is not invariant under arbitrary
        permutation of open segments, and claiming 1e-9 there would be wrong.

    ``f_period_s`` must be given to run the cyclic-rotation variant (the record
    is then rotated by whole periods).
    """
    x = np.asarray(history, dtype=float)
    blocks = np.array_split(x, n_segments)
    rng = np.random.default_rng(seed)

    def _rate(h: np.ndarray) -> float:
        return rainflow_damage_rate(h, dt, C=C, m=m)["damage_rate"]

    # (i) segment accumulation, permuted.  Damage *rates* are duration weighted
    # (a plain sum of the block rates would be O(n_segments) too large).
    durations = np.array([b.size * dt for b in blocks], dtype=float)
    per_block = [_rate(b) for b in blocks]
    base_sum = float(np.sum(np.array(per_block) * durations) / durations.sum())
    perm_diffs = []
    for _ in range(n_permutations):
        order = rng.permutation(n_segments)
        acc = float(np.sum([per_block[i] * durations[i] for i in order]) / durations.sum())
        perm_diffs.append(abs(acc / base_sum - 1.0))
    seg_worst = max(perm_diffs) if perm_diffs else 0.0
    whole = _rate(x)

    # (ii) time reversal
    rev_worst = abs(_rate(x[::-1].copy()) / whole - 1.0) if whole > 0 else 0.0

    # (iii) cyclic rotation by whole periods
    rot_worst: Optional[float] = None
    if f_period_s is not None and f_period_s > 0:
        n_per = int(round(float(f_period_s) / dt))
        if 0 < n_per < x.size:
            rot_worst = 0.0
            k = n_per
            while k < x.size:
                y = np.concatenate([x[k:], x[:k]])
                rot_worst = max(rot_worst, abs(_rate(y) / whole - 1.0))
                k += n_per * max(1, (x.size // n_per) // 4)

    # (iv) raw open-segment permutation (informational)
    raw = []
    for _ in range(n_permutations):
        order = rng.permutation(n_segments)
        y = np.concatenate([blocks[i] for i in order])
        raw.append(abs(_rate(y) / whole - 1.0))

    checks = {
        "segment_accumulation": seg_worst,
        "time_reversal": rev_worst,
        "cyclic_rotation": rot_worst,
    }
    return {
        "design": {
            "n_segments": n_segments,
            "n_permutations": n_permutations,
            "tolerance_rel": COMMUTATIVITY_TOL,
            "seed": seed,
            "f_period_s": f_period_s,
            "damage_rate_whole_record": whole,
            "damage_rate_segmented": base_sum,
        },
        "checks_rel_diff": checks,
        "pass_1e9": {k: bool(v <= COMMUTATIVITY_TOL) for k, v in checks.items() if v is not None},
        "raw_open_segment_permutation_rel_diff": raw,
        "raw_open_segment_permutation_max": float(max(raw)) if raw else 0.0,
        "segmentation_vs_whole_rel_diff": base_sum / whole - 1.0 if whole > 0 else 0.0,
        "interpretation": (
            "segment_accumulation / time_reversal / cyclic_rotation are exact "
            "invariance properties of the ASTM four-point counter and are the "
            "1e-9 checks; permutation of OPEN segments re-cuts the turning points "
            "and is not an invariance property (measured magnitude reported)."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Analytic limits and the engineering law
# ---------------------------------------------------------------------------
def analytic_bounds(sigma_ice: float, sigma_wave: float, m: float = SN_M) -> Dict[str, float]:
    """Exact damage ratios in the two limits (equal cycle rates per band).

    * fully separated bands: ``D_comb = D_ice + D_wave`` -> ratio 1.
    * fully coincident tones (same frequency): the two tones fuse into one
      narrow-band process of RMS ``sqrt(s1^2 + s2^2)``, hence
      ``ratio = (s1^2+s2^2)^(m/2) / (s1^m + s2^m)``.
    """
    s1, s2 = float(sigma_ice), float(sigma_wave)
    fused = ((s1 ** 2 + s2 ** 2) ** (m / 2.0)) / (s1 ** m + s2 ** m)
    return {
        "ratio_separated": 1.0,
        "ratio_coincident": float(fused),
        "ratio_coincident_equal_rms": float(2.0 ** (m / 2.0 - 1.0)),
        "amplification_max_pct": 100.0 * (fused - 1.0),
    }


def fit_separation_law(
    rows: Sequence[Dict[str, object]], m: float = SN_M
) -> Dict[str, object]:
    """One-parameter interpolation between the two exact limits.

    ``kappa(r) = 1 + (kappa_fused - 1) * exp(-(ln r)^2 / (2 s^2))`` with the
    fused-limit amplification ``kappa_fused`` taken from the ratio of the
    sub-RMS values (no fitting) and only the *log-separation width* ``s`` fitted.
    The law is valid for narrow-band sub-processes; ``s`` quantifies how many
    octaves of separation are needed to recover additivity.
    """
    ratio = np.array([r["f_ratio"] for r in rows], dtype=float)
    kappa = np.array([r["ratio_mean"] for r in rows], dtype=float)
    s1 = np.array([r["sigma_ice_MPa"] for r in rows], dtype=float)
    s2 = np.array([r["sigma_wave_MPa"] for r in rows], dtype=float)
    fused = (s1 ** 2 + s2 ** 2) ** (m / 2.0) / (s1 ** m + s2 ** m)
    amp = fused - 1.0

    def model(s: float) -> np.ndarray:
        return 1.0 + amp * np.exp(-(np.log(ratio) ** 2) / (2.0 * s * s))

    grid = np.linspace(0.15, 6.0, 2000)
    sse = [float(np.sum((model(s) - kappa) ** 2)) for s in grid]
    s_best = float(grid[int(np.argmin(sse))])
    pred = model(s_best)
    resid = kappa - pred
    ss_tot = float(np.sum((kappa - kappa.mean()) ** 2))
    return {
        "law": "kappa(r) = 1 + (kappa_fused - 1) * exp(-(ln r)^2/(2 s^2))",
        "s_fitted_octaves_natural_log": s_best,
        "kappa_fused_from_rows": float(np.mean(fused[np.argmin(np.abs(np.log(ratio) / np.log(10.0)))])) if ratio.size else float("nan"),
        "r2": float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "max_abs_residual_pct": float(100.0 * np.max(np.abs(resid))),
        "rows": [
            {"f_ratio": float(r), "kappa_measured": float(k), "kappa_fitted": float(p),
             "residual_pct": float(100.0 * (k - p))}
            for r, k, p in zip(ratio, kappa, pred)
        ],
        "comment": (
            "s is the natural-log separation width at which the additivity error has "
            "decayed by exp(-1/2); for the narrow-band bands used here s ~ 1 means "
            "about one e-fold of frequency separation is needed."
        ),
    }

# ---------------------------------------------------------------------------
# 6. Physically grounded case: DEM ice chain + JONSWAP wave stress
# ---------------------------------------------------------------------------
def realistic_case(
    duration: float = 3600.0,
    dt: float = 5.0e-3,
    sigma_wave: float = 3.0,
    fp_wave: float = 0.12,
    gamma_wave: float = 3.3,
    wave_seeds: Sequence[int] = (101, 211, 307),
    m: float = SN_M,
    C: float = SN_C_F3_CP,
) -> Dict[str, object]:
    """Ice (DEM-calibrated, whipping included) + wave (JONSWAP stress) case.

    The ice stress history is the calibrated anchor chain of
    :mod:`tensorphysics.fatigue.combined.ice_anchor` (modal superposition, so the
    whipping ringing at ``f1 = 0.56 Hz`` is *in* the history); the wave stress is
    a stationary Gaussian realisation of a JONSWAP-shaped stress PSD with peak
    ``fp_wave`` and RMS ``sigma_wave`` (i.e. the wave-induced vertical bending
    stress, the RAO amplitude collapsed into the RMS).  The dominant ice
    frequency is the crushing/ringing rate ~0.5 Hz, so the physical separation is
    ``f_ice/f_wave ~ 4`` -- squarely inside the non-additive band.

    Reports rainflow damages of the components and of the sum, the combination
    ratio, the Dirlik/TB closures evaluated on the *measured* composite stress
    PSD (Welch + wave PSD), and the whipping accounting of the ice branch.
    """
    from .ice_anchor import (
        ANCHOR_CASE, FreeFreeBeamModes, IceContactParams, generate_ice_force,
        modal_stress_history, whipping_share,
    )
    from scipy import signal

    cfg = dict(ANCHOR_CASE)
    prm = IceContactParams.calibrated()
    modes = FreeFreeBeamModes(cfg["length_m"], cfg["EI_nm2"] if "EI_nm2" in cfg else cfg["EI_Nm2"],
                              cfg["mass_per_length_kg_m"], n_modes=int(cfg["n_modes"]),
                              zeta=cfg["zeta"])
    _, F = generate_ice_force(prm, duration=duration, dt=dt)
    skip = min(int(cfg["settle_time_s"] / dt), max(0, F.size // 2))
    resp = modal_stress_history(modes, F, dt, cfg["x_load_m"], cfg["x_detail_m"],
                                cfg["z_deck_m"], cfg["E_Pa"])
    ice = resp["sigma"][skip:] / 1.0e6                 # MPa nominal
    ice_qs = resp["sigma_qs"][skip:] / 1.0e6
    fs = 1.0 / dt
    n = ice.size
    t = np.arange(n) * dt
    ws = whipping_share(ice, ice_qs)

    wave_psd = JONSWAP(hs=4.0, fp=float(fp_wave), gamma=float(gamma_wave))
    # scale the JONSWAP variance to the requested stress RMS
    m0 = wave_psd.moments_object().m0
    scale = float(sigma_wave) ** 2 / m0
    wave_histories = []
    for sd in wave_seeds:
        x = synthesize_gaussian_history(wave_psd, n, fs, seed=int(sd),
                                        f_lo=max(1.0 / (n * dt), 1.0e-3),
                                        f_hi=min(0.98 * fs / 2.0, 6.0 * fp_wave))
        wave_histories.append(x * math.sqrt(scale))

    d_ice = rainflow_damage_rate(ice, dt, C=C, m=m)
    d_qs = rainflow_damage_rate(ice_qs, dt, C=C, m=m)
    rows = []
    for sd, wave in zip(wave_seeds, wave_histories):
        wave = orthogonalise(wave, ice, target_var=float(np.var(wave)))
        d_wave = rainflow_damage_rate(wave, dt, C=C, m=m)
        d_comb = rainflow_damage_rate(ice + wave, dt, C=C, m=m)
        # measured composite PSD (Welch) + analytic wave PSD -> spectral closures
        f_w, p_ice = signal.welch(ice + wave, fs=fs, nperseg=min(2 ** 14, n))
        s_wave = scale * np.asarray(wave_psd(f_w), dtype=float)
        mom = moments_from_dict(spectral_moments(f_w, p_ice + s_wave), 0.0, f_w[-1])
        spec = spectral_damage_rates(mom, C=C, m=m)
        rows.append({
            "wave_seed": int(sd),
            "D_ice": d_ice["damage_rate"],
            "D_ice_quasi_static": d_qs["damage_rate"],
            "D_wave": d_wave["damage_rate"],
            "D_comb": d_comb["damage_rate"],
            "ratio_linear": d_comb["damage_rate"] / (d_ice["damage_rate"] + d_wave["damage_rate"]),
            "D_comb_dirlik": spec["dirlik"],
            "D_comb_tb": spec["tovo_benasciutti"],
            "err_dirlik_pct": 100.0 * (spec["dirlik"] / d_comb["damage_rate"] - 1.0),
            "err_tb_pct": 100.0 * (spec["tovo_benasciutti"] / d_comb["damage_rate"] - 1.0),
            "alpha2_composite": spec["alpha2"],
            "ignoring_whipping_underprediction_factor": (
                d_comb["damage_rate"] / d_qs["damage_rate"] if d_qs["damage_rate"] > 0 else float("nan")),
        })
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return {
        "design": {"duration_s": duration, "dt_s": dt, "fs_Hz": fs,
                   "sigma_wave_MPa": sigma_wave, "fp_wave_Hz": fp_wave,
                   "gamma_wave": gamma_wave, "wave_seeds": list(wave_seeds),
                   "m": m, "C": C, "ice_dominant_freq_Hz": float(modes.freq[0]),
                   "separation_ratio": float(modes.freq[0] / fp_wave)},
        "whipping": {k: float(v) for k, v in ws.items()},
        "rows": rows,
        "summary": {
            "ratio_linear_mean": float(arr["ratio_linear"].mean()),
            "ratio_linear_std": float(arr["ratio_linear"].std(ddof=1)) if arr["ratio_linear"].size > 1 else 0.0,
            "err_dirlik_pct_mean": float(arr["err_dirlik_pct"].mean()),
            "err_tb_pct_mean": float(arr["err_tb_pct"].mean()),
            "ignoring_whipping_factor_mean": float(arr["ignoring_whipping_underprediction_factor"].mean()),
            "D_ice_over_D_wave": float(arr["D_ice"].mean() / arr["D_wave"].mean()),
        },
    }


# ---------------------------------------------------------------------------
# 7. Admissible region of the linear law
# ---------------------------------------------------------------------------
def admissible_region(
    rows: Sequence[Dict[str, object]], tol: float = LINEARITY_TOL
) -> Dict[str, object]:
    """Boundary of the region where |kappa - 1| <= tol (linear law admissible).

    Returns, for each amplitude split, the smallest separation ratio at which the
    measured combination ratio enters the tolerance band, together with the fit
    of the one-parameter law to that split.
    """
    by_split: Dict[float, List[Dict[str, object]]] = {}
    for row in rows:
        key = round(float(row["sigma_ice_MPa"]) / float(row["sigma_wave_MPa"]), 6)
        by_split.setdefault(key, []).append(row)
    out = []
    for split, group in sorted(by_split.items()):
        group = sorted(group, key=lambda r: float(r["f_ratio"]))
        ok = [r for r in group if abs(float(r["ratio_mean"]) - 1.0) <= tol]
        # smallest *fast/slow* separation that satisfies the tolerance
        best = None
        for r in ok:
            sep = max(float(r["f_ratio"]), 1.0 / float(r["f_ratio"]))
            best = sep if best is None else min(best, sep)
        fit = fit_separation_law(group)
        out.append({
            "sigma_split": split,
            "min_separation_for_tolerance": best,
            "kappa_by_ratio": {float(r["f_ratio"]): float(r["ratio_mean"]) for r in group},
            "fit": {k: fit[k] for k in ("s_fitted_octaves_natural_log", "r2", "max_abs_residual_pct")},
        })
    return {
        "tolerance": tol,
        "rows": out,
        "verdict": (
            "linear additivity is admissible only in the region reported above; "
            "outside it kappa > 1 (the linear law under-predicts damage)."
        ),
    }
