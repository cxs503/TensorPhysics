"""Node ``P82W5-N09`` driver: ice+wave combination law, bimodal spectra, whipping.

Runs every machine criterion of the node on the m5528 (IN32-N07) hull-girder
case and returns a JSON-serialisable results dictionary:

``(a)`` **sum rule versus the required band ratios.**  ``rho = D_comb/(D_ice+D_wave)``
       measured by rainflow for ``f_ice/f_w in {0.3, 1, 10, 100}``, three
       spectral widths and two amplitude conventions, with the seed scatter, and
       scored against the predeclared ``[0.97, 1.03]`` band; where the sum rule
       fails the two-band riding closure of :mod:`.bimodal` is scored as the
       replacement law, together with the decidable envelope
       :func:`~.bimodal.sum_rule_envelope`.
``(b)`` **spectral-method capability.**  Dirlik, Tovo-Benasciutti, narrow-band and
       the two-band closure against rainflow truth on the same band-limited PSD,
       three declared spectral widths, four band ratios.
``(c)`` **whipping calibers.**  Peak / RMS / damage calibers over the declared
       loading plane, with the regime boundary of the m5528 ``< 8 %`` claim.
``(d)`` **rainflow commutability.**  Streaming blocks with residue carry-over on
       the two-band records plus closed-excursion-block permutation.
``(e)`` **negative control.**  Waves off must reproduce the m5528 damage.

The two heavy machinery pieces are reused, not duplicated:
``combination.m5528_reproduction`` for (e), ``combination.segmentation_invariance``
for the permutation half of (d), ``combination.WIDTH_LEVELS`` for the declared
widths.
"""

from __future__ import annotations

import math
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from tensorphysics.fatigue.spectral import synthesize_gaussian_history

from . import sn as snmod  # noqa: F401  (kept for provenance symmetry)
from .bimodal import (
    N_SEPARATED,
    KappaClosure,
    band_case,
    bimodal_benchmark,
    equal_damage_amplitude_ratio,
    kappa_calibration_grid,
    sum_rule_envelope,
    two_band_riding_damage,
)
from .combination import (
    HullGirderCase,
    WIDTH_LEVELS,
    m5528_reproduction,
    segmentation_invariance,
)
from .rainflow_stream import StreamingRainflow, damage_from_cycles, rainflow_cycles, turning_points
from .whipping_regime import whipping_regime_scan, regime_consistency

__all__ = [
    "CRITERION_A_RATIOS",
    "sum_rule_scan",
    "commutation_check",
    "run_w5n09_study",
    "summarise_criteria",
]

#: Band ratios demanded by the node's machine criterion (a).
CRITERION_A_RATIOS = (0.3, 1.0, 10.0, 100.0)
#: Predeclared acceptance band of the plain sum rule.
SUM_RULE_BAND = (0.97, 1.03)


def _git_commit() -> Optional[str]:
    """Short commit of the checkout the module lives in (``None`` outside a repo)."""
    import pathlib
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            try:
                return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                               cwd=str(parent), stderr=subprocess.DEVNULL,
                                               text=True).strip()
            except Exception:
                return None
    return None


def _environment() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_commit": _git_commit(),
    }


def sum_rule_scan(
    case: Optional[HullGirderCase] = None,
    widths: Sequence[str] = ("narrow_gamma6", "medium_gamma3p3", "wide_bimodal_rp3"),
    ratios: Sequence[float] = CRITERION_A_RATIOS,
    seeds: Sequence[int] = (0, 1, 2, 3),
    closure: Optional[KappaClosure] = None,
    duration: float = 1200.0,
    fs: float = 40.0,
    m: float = 3.0,
    sigma_carrier_mpa: float = 10.0,
    equal_rms_ratios: Sequence[float] = (0.6, 1.0, 2.0),
) -> Dict[str, Any]:
    """Criterion (a): measure ``rho`` and score the sum rule / the closure.

    Two amplitude conventions per band ratio:

    * ``equal_damage`` -- ``r = N**(-1/m)``, the two bands carrying equal
      per-component narrow-band damage (the ``P82W4-N06`` convention);
    * ``equal_rms`` -- fixed ``r`` from ``equal_rms_ratios``, so that the sum
      rule can be probed at riding amplitudes well below and well above the
      carrier.
    """
    case = case or HullGirderCase()
    rows: List[Dict[str, Any]] = []
    for width in widths:
        cfg = WIDTH_LEVELS[width]
        for ratio in ratios:
            r_specs = [("equal_damage", None)] + [("equal_rms", float(x)) for x in equal_rms_ratios]
            for mode, r_val in r_specs:
                # a carrier at f_ice/N has only N/T1 cycles per period: scale the
                # record length so that every band ratio keeps >= ~10 carrier cycles
                dur = duration * max(1.0, min(3.0, float(ratio) / 20.0))
                outs = [band_case(case, n_ratio=ratio, r=r_val,
                                  wave_kind=cfg["wave_kind"], wave_kwargs=cfg["wave_kwargs"],
                                  seed=int(s), duration=dur, fs=fs, m=m,
                                  sigma_carrier=sigma_carrier_mpa)
                        for s in seeds]
                rho = np.array([o["rho_components"] for o in outs], float)
                truth = np.array([o["d_truth"] for o in outs], float)
                two = [two_band_riding_damage(o, closure, m) for o in outs]
                d_two = np.array([t["damage"] for t in two], float)
                d_sum = np.array([o["d_ice_component_rainflow"] + o["d_wave_component_rainflow"]
                                  for o in outs], float)
                first = outs[0]
                N = float(first["n_ratio"])
                rho_mean = float(rho.mean())
                rows.append({
                    "width": width,
                    "wave_kind": cfg["wave_kind"],
                    "wave_kwargs": cfg["wave_kwargs"],
                    "ratio_f_ice_over_wave": float(ratio),
                    "n_ratio": N,
                    "amplitude_mode": mode,
                    "r_rider_over_carrier": float(first["r_rider_over_carrier"]),
                    "sigma_carrier_mpa": sigma_carrier_mpa,
                    "rho_mean": rho_mean,
                    "rho_std": float(rho.std(ddof=1)) if rho.size > 1 else 0.0,
                    "rho_min": float(rho.min()),
                    "rho_max": float(rho.max()),
                    "sum_rule_inside_band": bool(SUM_RULE_BAND[0] <= rho_mean <= SUM_RULE_BAND[1]),
                    "sum_rule_rel_err_pct": 100.0 * (1.0 / rho_mean - 1.0),
                    "d_truth_mean": float(truth.mean()),
                    "d_sum_mean": float(d_sum.mean()),
                    "d_two_band_mean": float(d_two.mean()),
                    "two_band_err_pct": 100.0 * (float(d_two.mean()) / float(truth.mean()) - 1.0),
                    "two_band_regime": two[0]["regime"],
                    "kappa_used": float(two[0]["kappa"]),
                    "band_separated": bool(N >= N_SEPARATED),
                    "seeds": [int(s) for s in seeds],
                })
    separated = [r for r in rows if r["ratio_f_ice_over_wave"] >= 10.0
                 and r["amplitude_mode"] == "equal_damage"]
    inside = [r for r in separated if r["sum_rule_inside_band"]]
    return {
        "rows": rows,
        "band": list(SUM_RULE_BAND),
        "n_points": len(rows),
        "separated_equal_damage_points": len(separated),
        "separated_equal_damage_inside_band": len(inside),
        "sum_rule_all_max_abs_err_pct": float(max(abs(r["sum_rule_rel_err_pct"]) for r in rows)),
        "sum_rule_separated_max_abs_err_pct": float(max(abs(r["sum_rule_rel_err_pct"])
                                                         for r in separated)),
        "two_band_worst_err_pct": float(max(abs(r["two_band_err_pct"]) for r in rows)),
        "two_band_worst_err_separated_pct": float(max(abs(r["two_band_err_pct"])
                                                      for r in separated)),
        "rho_range": [float(min(r["rho_mean"] for r in rows)),
                      float(max(r["rho_mean"] for r in rows))],
        "pass_sum_rule": bool(len(inside) == len(separated) and len(separated) > 0),
        "pass_two_band_law": bool(max(abs(r["two_band_err_pct"]) for r in rows) <= 10.0),
        "seeds": [int(s) for s in seeds],
        "duration_s": duration,
        "fs_hz": fs,
        "m": m,
        "equal_rms_ratios": [float(x) for x in equal_rms_ratios],
    }


def commutation_check(
    case: Optional[HullGirderCase] = None,
    n_ratio: float = 10.0,
    seed: int = 3,
    duration: float = 1200.0,
    fs: float = 40.0,
    m: float = 3.0,
    block_counts: Sequence[int] = (2, 7, 31, 97),
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """Criterion (d), streaming half: block-wise == whole-record, bit for bit.

    The two-band record is synthesised once, then fed to
    :class:`~.rainflow_stream.StreamingRainflow` in ``k`` blocks (the residue
    stack carries the open excursions across the boundary) and compared with the
    whole-record rainflow damage of the same samples.
    """
    case = case or HullGirderCase()
    out = band_case(case, n_ratio=n_ratio, seed=seed, duration=duration, fs=fs, m=m,
                    sigma_carrier=10.0)
    x = synthesize_gaussian_history(out["s_comb"], int(round(duration * fs)), fs,
                                    seed=seed + 500)
    r_whole, _mn, c_whole = rainflow_cycles(turning_points(x))
    d_ref = damage_from_cycles(r_whole, c_whole, m)
    rows = []
    for k in block_counts:
        counter = StreamingRainflow()
        for block in np.array_split(x, int(k)):
            counter.feed(block)
        counter.finalize()
        r, _m, c = counter.cycles()
        d = damage_from_cycles(r, c, m)
        rows.append({
            "blocks": int(k),
            "damage": d,
            "rel_dev": float(abs(d / d_ref - 1.0)) if d_ref > 0 else float("nan"),
            "n_ranges_equal": bool(r.size == r_whole.size),
        })
    return {
        "rows": rows,
        "d_whole_record": d_ref,
        "max_rel_dev": float(max(r["rel_dev"] for r in rows)),
        "tolerance": tolerance,
        "pass": bool(max(r["rel_dev"] for r in rows) <= tolerance
                     and all(r["n_ranges_equal"] for r in rows)),
        "n_ratio": float(out["n_ratio"]),
        "duration_s": duration,
        "fs_hz": fs,
        "seed": int(seed),
    }


def run_w5n09_study(
    case: Optional[HullGirderCase] = None,
    quick: bool = False,
    duration: float = 1200.0,
    fs: float = 40.0,
    m: float = 3.0,
    seeds_a: Sequence[int] = (0, 1, 2, 3),
    seeds_b: Sequence[int] = (0, 1, 2),
    cal_seeds: Sequence[int] = (0, 1, 2),
) -> Dict[str, Any]:
    """Run all five criteria of node ``P82W5-N09``.

    ``quick=True`` shrinks every record length (used by the unit tests); the
    numbers of the published report come from ``quick=False``.
    """
    case = case or HullGirderCase()
    if quick:
        duration = min(duration, 240.0)
        seeds_a = tuple(seeds_a)[:2] or (0,)
        seeds_b = tuple(seeds_b)[:2] or (0,)
        cal_seeds = tuple(cal_seeds)[:2] or (0,)
    cal = kappa_calibration_grid(case=case, seeds=cal_seeds, duration=duration, fs=fs, m=m)
    closure = KappaClosure(**cal["closure"])
    scan_a = sum_rule_scan(case, seeds=seeds_a, closure=closure, duration=duration,
                           fs=fs, m=m)
    bench_b = bimodal_benchmark(case, seeds=seeds_b, closure=closure,
                                duration=duration, fs=fs, m=m)
    whip_scan = whipping_regime_scan(case, m=m)
    whip_cons = regime_consistency(whip_scan)
    comm_stream = commutation_check(case, duration=duration, fs=fs, m=m)
    comm_perm = segmentation_invariance()
    neg = m5528_reproduction(case)
    envelope = sum_rule_envelope(closure, tol=0.03)
    result: Dict[str, Any] = {
        "node": "P82W5-N09",
        "title": "冰-波浪组合疲劳：组合律裁决、双峰谱方法与 whipping 口径的可判定边界",
        "environment": _environment(),
        "case": {
            "length_m": case.length,
            "breadth_m": case.breadth,
            "draught_m": case.draught,
            "EI_Nm2": case.EI,
            "mass_per_length_kg_m": case.mass_per_length,
            "z_detail_m": case.z_deck,
            "scf": case.scf,
            "damping_ratio": case.damping_ratio,
            "n_modes": case.n_modes,
            "f1_hz": case.f1_hz,
            "f_ice_band_hz": case.ice_band_hz,
            "sn_curve": case.sn_curve,
            "sn_utS_mpa": case.s_uts,
            "ice_hours_per_year": case.ice_hours_per_year,
        },
        "settings": {
            "duration_s": duration, "fs_hz": fs, "m": m,
            "seeds_a": [int(s) for s in seeds_a],
            "seeds_b": [int(s) for s in seeds_b],
            "calibration_seeds": [int(s) for s in cal_seeds],
            "width_levels": {k: WIDTH_LEVELS[k] for k in
                             ("narrow_gamma6", "medium_gamma3p3", "wide_bimodal_rp3")},
            "n_separated": N_SEPARATED,
            "sum_rule_band": list(SUM_RULE_BAND),
            "spectral_threshold_pct": 10.0,
            "whipping_tolerance_pp": whip_cons["tolerance_pp"],
        },
        "calibration": cal,
        "criterion_a": scan_a,
        "criterion_b": bench_b,
        "criterion_c": {"scan": whip_scan, "consistency": whip_cons},
        "criterion_d": {"streaming": comm_stream, "block_permutation": comm_perm},
        "criterion_e": neg,
        "envelope": envelope,
    }
    result["criteria"] = summarise_criteria(result)
    return result


def summarise_criteria(result: Dict[str, Any]) -> Dict[str, Any]:
    """One-line verdict per machine criterion, with the deciding numbers."""
    a = result["criterion_a"]
    b = result["criterion_b"]
    c = result["criterion_c"]["consistency"]
    d = result["criterion_d"]
    e = result["criterion_e"]
    d_pass = bool(d["streaming"]["pass"]
                  and d["block_permutation"].get("permutation_max_rel_dev", 1.0) <= 1e-9)
    e_pass = bool(e.get("pass", False)) if isinstance(e, dict) else False
    return {
        "a_sum_rule_rejected": {
            "verdict": "sum rule rejected" if not a["pass_sum_rule"] else "sum rule holds",
            "rho_range": a["rho_range"],
            "separated_points_inside_band": f"{a['separated_equal_damage_inside_band']}"
                                            f"/{a['separated_equal_damage_points']}",
            "law_two_band_worst_err_pct": a["two_band_worst_err_pct"],
            "pass_as_law": a["pass_two_band_law"],
        },
        "b_spectral_methods": {
            "verdict_dirlik": "PASS" if b["pass_dirlik"] else "FAIL",
            "verdict_tb": "PASS" if b["pass_tb"] else "FAIL",
            "verdict_two_band": "PASS" if b["pass_two_band"] else "FAIL",
            "worst_err_dirlik_pct": b["worst_err_dirlik_pct"],
            "worst_err_tb_pct": b["worst_err_tb_pct"],
            "worst_err_two_band_pct": b["worst_err_two_band_pct"],
            "threshold_pct": b["threshold_pct"],
            "pass": bool(b["pass_two_band"]),
        },
        "c_whipping_caliber": {
            "verdict": "scoped PASS" if c["pass"] else "FAIL",
            "c1_quasi_static": c["c1_quasi_static_share_and_peak_caliber"],
            "c1_dev_peak_pp_max": c["c1_dev_peak_pp_max"],
            "c2_claim_not_transferable": c["c2_claim_does_not_transfer_outside_quasi_static"],
            "c2_share_peak_min_outside": c["c2_share_peak_min_outside"],
            "c3_caliber_ranking": c["c3_caliber_ranking"],
            "caliber_max_dev_pp": c["caliber_max_dev_pp"],
            "pass": c["pass"],
        },
        "d_commutation": {
            "verdict": "PASS" if d_pass else "FAIL",
            "streaming_max_rel_dev": d["streaming"]["max_rel_dev"],
            "permutation_max_rel_dev": d["block_permutation"].get("permutation_max_rel_dev"),
            "tolerance": 1e-9,
            "pass": d_pass,
        },
        "e_negative_control": {
            "verdict": "PASS" if e_pass else "FAIL",
            "worst_rel_err_pct": e.get("worst_rel_err_pct") if isinstance(e, dict) else None,
            "tolerance_pct": e.get("tolerance_pct") if isinstance(e, dict) else None,
            "pass": e_pass,
        },
    }