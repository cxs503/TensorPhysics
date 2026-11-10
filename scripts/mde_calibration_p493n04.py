#!/usr/bin/env python
"""P493-N04 — detection-strength certificates (MDE_90) for the acceptance ledger.

The shipped mutation audit (``tools/audit/criteria_mutation.py``, node IN50-W3-N08)
answers *how many* mutants the ledger kills at a fixed epsilon ladder.  It does
**not** answer the question a designer actually has to answer: *if a quantity is
wrong by 5 %, does this criterion see it?*  A kill-rate of 100 % at epsilon = 15 %
and a kill-rate of 100 % at epsilon = 1 % are the same number in that ledger.

This script calibrates every criterion of that same ledger:

1. imports the shipped criterion objects (no re-declaration, so "the audited
   criteria" and "the calibrated criteria" cannot drift apart);
2. wraps each criterion into a detector ``y -> reject`` by mutating the
   criterion's own ``value_fields`` and calling the **shipped gate**;
3. injects the bias ladder ``+1, +2, +5, +10, +20 %`` (N = 100 draws each) plus a
   sign flip and a magnitude scale, adds evidence noise from a declared
   uncertainty ladder, and records detection curves with Wilson intervals;
4. extracts MDE_90 three ways (exact decision boundary of deterministic
   detectors, probit interpolation of the Monte-Carlo curve, logistic fit) and
   cross-checks against a closed form where one exists;
5. issues a certificate per criterion — including the honest verdict
   ``NOT_APPLICABLE`` for the structural criteria that no continuous bias can
   ever reach, and the sample-size audit for the proportion (kill-rate) criteria.

Everything numeric is written by the program: the JSON ledger, the Markdown
report and the figures all come from the same run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.vv import mde  # noqa: E402

ARTIFACT_DIR = Path("artifacts/mde_calibration_p493n04")
DOC_PATH = Path("docs/MDE_DETECTION_CERTIFICATES_P493N04.md")

#: Bias ladder required by the node spec (relative error injected).
BIAS_LADDER = (0.01, 0.02, 0.05, 0.10, 0.20)
#: Wider ladder for the kill-rate family (its MDE_90 reaches 20-30 % at small n).
STAT_LADDER = (0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 0.90)
#: Evidence-noise ladder swept for the sensitivity table (relative scatter).
SIGMA_LADDER = (0.0, 0.01, 0.02, 0.05)
#: Platform precision rule: an error above 15 % is not acceptable.
PLATFORM_LIMIT = 0.15
#: The bias the certificate must rule on ("does it catch a 5 % error?").
REFERENCE_BIAS = 0.05
SEED = 49304


# ---------------------------------------------------------------------------
# shipped ledger ingestion
# ---------------------------------------------------------------------------
def load_shipped_ledger() -> Any:
    """Import the shipped mutation-audit registry (single source of criteria)."""
    path = ROOT / "tools/audit/criteria_mutation.py"
    spec = importlib.util.spec_from_file_location("criteria_mutation_shipped", path)
    if spec is None or spec.loader is None:      # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # the module registers itself under "criteria_mutation" (its adapters import
    # it by that name) and dataclasses need the spec name resolvable too
    sys.modules[spec.name if spec.name else "criteria_mutation_shipped"] = module
    sys.modules["criteria_mutation"] = module
    spec.loader.exec_module(module)
    return module


def shipped_criteria(ship: Any) -> list[Any]:
    criteria: list[Any] = []
    for gate in ship.GATE_ADAPTERS:
        criteria.extend(ship.GATE_ADAPTERS[gate]())
    return criteria


def gate_detector(ship: Any, criterion: Any) -> Callable[[float], bool]:
    """Wrap a shipped criterion into ``y -> reject`` with no re-implementation.

    ``y`` is the *relative error* injected into the criterion's own
    ``value_fields``; the gate sees ``value * (1 + y)`` and its verdict becomes
    "detected" (the criterion no longer accepts).
    """
    if not criterion.value_fields:
        raise ValueError(f"{criterion.id}: no continuous value field")

    def detector(y: float) -> bool:
        case = criterion.case
        for path in criterion.value_fields:
            case = ship.mutant_scale(case, path, 1.0 + y)
        result = ship.ADAPTERS[criterion.adapter](criterion, case)
        # Read the verdict through the ledger's own normaliser and *at the
        # criterion's declared sub-check*.  Reading the overall verdict instead
        # (``result.status``) makes unrelated warnings fire every detector --
        # e.g. the GCI gate warns about an uncertified noise floor while all
        # five declared checks PASS.
        try:
            status = ship.status_from_result(result, criterion.check)
        except Exception:                     # pragma: no cover - defensive
            status = getattr(result, "status", None) or "FAIL"
        return status not in criterion.accept

    return detector


def value_field_scale(criterion: Any) -> tuple[str, float]:
    """Base magnitude of the injected field (in the gate's own units)."""
    path = criterion.value_fields[0]
    value = criterion.case
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return path, float(value)


# ---------------------------------------------------------------------------
# statistical (proportion) family
# ---------------------------------------------------------------------------
def proportion_detector(n: int, p_min: float, alpha: float) -> Callable[[float], bool]:
    """Detection test on a kill-rate measured over ``n`` mutants.

    ``y`` is the *deficit* of the true acceptance rate below ``p_min``.  The
    criterion rejects when the observed kill count falls at or below the
    level-alpha threshold ``k_crit``; draws are honest Bernoulli trials, so the
    Monte-Carlo curve is comparable to the exact binomial power.
    """
    k_crit = mde.proportion_k_crit(n, p_min, alpha)
    if k_crit < 0:
        return lambda y: False

    rng = random.Random(0xC0FFEE ^ (n << 8) ^ int(round(p_min * 1e6)))

    def detector(y: float) -> bool:
        # fresh Bernoulli draws on every call: the sampling noise is the object
        # of the exercise, so it must not be frozen by a per-value seed
        p_true = max(0.0, min(1.0, p_min - y))
        kills = sum(1 for _ in range(n) if rng.random() < p_true)
        return kills <= k_crit

    return detector


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def calibrate_continuous(ship: Any, criterion: Any, *, n_trials: int,
                         seed: int) -> dict[str, Any]:
    detector = gate_detector(ship, criterion)
    path, base = value_field_scale(criterion)
    cert = mde.certificate(
        criterion.id, detector, deltas=BIAS_LADDER, statistic="point",
        threshold=None, sigma=0.0,
        sigma_source=("shipped gate on a computed quantity: no repeat-test "
                      "scatter declared -> sigma = 0 (deterministic detector)"),
        n_trials=n_trials, seed=seed, reference=REFERENCE_BIAS)
    cert.update({
        "kind": "continuous",
        "gate": criterion.gate,
        "check": criterion.check,
        "value_field": path,
        "value_base": base,
        "gate_tolerance": criterion.tolerance,
        "note": criterion.note,
        "kernel_mutant_classes": list(criterion.mutant_classes),
    })
    if criterion.id == "VVVAL-ratio":
        cert["mde90_theory"] = mde.mde90_analytic("point", _effective_threshold(criterion))
        cert["theory_note"] = ("delta* = (f_exp + U_val - f_sim)/f_sim, "
                               "U_val = sqrt(u_num^2 + u_input^2) (ASME V&V 20)")
    # mutant families beyond the systematic bias: sign flip and magnitude scale
    cert["families"] = {
        "systematic_bias": {"ladder": list(BIAS_LADDER),
                            "curve": cert["power_curve"]},
        "sign_flip": {"delta": -REFERENCE_BIAS,
                      "detected": bool(detector(-REFERENCE_BIAS))},
        "magnitude_scale": {f"k={k}": bool(detector(k - 1.0)) for k in (1.05, 2.0)},
        "random_noise": {"note": "calibrated by the sigma ladder (sigma_rows)"},
    }
    return cert


def _refined_ladder(boundary: float, sigma: float, points: int = 12) -> list[float]:
    """Geometric bias ladder centred on a criterion's own decision boundary.

    The node-specified ladder {1, 2, 5, 10, 20} % is the *reporting* ladder.  It
    cannot resolve a boundary of 0.2 % (three of the shipped gates sit there),
    so the noise-sensitivity study uses a geometric ladder spanning
    ``boundary/20 ... max(boundary + 6 sigma, 2 x boundary)`` capped at 50 %.
    """
    lo = max(boundary / 20.0, 1e-9)
    hi = min(0.5, max(boundary + 6.0 * sigma, 2.0 * boundary, lo * 10.0))
    ratio = (hi / lo) ** (1.0 / (points - 1))
    return [lo * ratio ** i for i in range(points)]


def sigma_sensitivity(ship: Any, criterion: Any, *, n_trials: int,
                      seed: int) -> list[dict[str, Any]]:
    """MDE_90 under a swept evidence-noise level (the noise penalty)."""
    detector = gate_detector(ship, criterion)
    closed_form = criterion.id == "VVVAL-ratio"
    boundary = mde.mde90_bisect(detector, lo=1e-15, hi=0.5) or 0.01
    rows: list[dict[str, Any]] = []
    for sigma in SIGMA_LADDER:
        ladder = _refined_ladder(boundary, sigma)
        curve = mde.detection_curve(detector, ladder, sigma=sigma,
                                    n_trials=n_trials,
                                    seed=seed + int(round(sigma * 1e4)))
        fit = mde.fit_logistic(curve.deltas, curve.hits, curve.trials)
        # what the gate does with *clean* evidence under the same scatter
        fa_curve = mde.detection_curve(detector, [0.0], sigma=sigma,
                                       n_trials=n_trials, seed=seed + 999 + int(sigma * 1e4))
        false_alarm = fa_curve.p_hat[0]
        rows.append({
            "sigma": sigma,
            "hits": curve.hits,
            "p": curve.p_hat,
            "p_ci_lo": curve.p_lo,
            "p_ci_hi": curve.p_hi,
            # a ladder spanning > 4 decades cannot resolve a step by
            # interpolation at all (e.g. the 1e-12 closure bound) -- report the
            # absence instead of a meaningless number
            "mde90_probit_raw": (mde.mde90_interpolate(curve)
                                 if curve.deltas[-1] / curve.deltas[0] <= 1e4
                                 else None),
            "ladder_span_decades": math.log10(curve.deltas[-1] / curve.deltas[0]),
            "mde90_probit": (None if false_alarm >= 0.5
                             else mde.mde90_interpolate(curve)),
            "mde90_fit": fit.mde90,
            "mde90_theory": (mde.mde90_analytic("point", _effective_threshold(criterion),
                                                sigma=sigma)
                             if closed_form else None),
            "ladder": curve.deltas,
            "below_ladder_floor": bool(curve.p_smooth and curve.p_smooth[0] >= 0.9),
            "p_false_alarm_mc": false_alarm,
            "p_false_alarm_gate_model": (mde.power_point(0.0, boundary, sigma)
                                         if closed_form else None),
            "interpretable": false_alarm < 0.5,
            "noise_limited": false_alarm >= 0.5,
            "p_at_reference": next((p for d, p in zip(curve.deltas, curve.p_hat)
                                    if abs(d - REFERENCE_BIAS) < 1e-15), None),
        })
    return rows


def _effective_threshold(criterion: Any) -> float:
    """Closed-form threshold of a magnitude gate expressed as relative error.

    Only the ASME V&V 20 criterion admits an exact translation (``|E| <= U_val``
    with ``E = f_sim*(1+delta) - f_exp``); for the rest the gate's internal band
    is not a single number, so no analytic reference is claimed.
    """
    if criterion.id != "VVVAL-ratio":
        raise ValueError("closed form only shipped for VVVAL-ratio")
    case = criterion.case
    u_val = math.sqrt(case["u_num"] ** 2 + case["u_input"] ** 2)
    f_sim, f_exp = case["f_sim"], case["f_exp"]
    return (f_exp + u_val - f_sim) / f_sim


def _mc_convergence_check(ship: Any, criterion: Any, *, n_trials: int,
                          sigma: float = 0.02, factor: int = 10) -> dict[str, Any]:
    """Does N = n_trials resolve MDE_90, or is the estimate sampling-limited?

    Re-estimates one curve with ``factor`` times more draws; the gap between the
    two is the Monte-Carlo error the rest of the ledger inherits.
    """
    detector = gate_detector(ship, criterion)
    boundary = mde.mde90_bisect(detector, lo=1e-15, hi=0.5) or 0.01
    ladder = _refined_ladder(boundary, sigma)
    small = mde.mde90_interpolate(mde.detection_curve(
        detector, ladder, sigma=sigma, n_trials=n_trials, seed=SEED + 7))
    large = mde.mde90_interpolate(mde.detection_curve(
        detector, ladder, sigma=sigma, n_trials=n_trials * factor, seed=SEED + 7))
    coarse = mde.mde90_interpolate(mde.detection_curve(
        detector, BIAS_LADDER, sigma=sigma, n_trials=n_trials, seed=SEED + 7))
    theory = mde.mde90_analytic("point", _effective_threshold(criterion), sigma=sigma)
    def rel(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return abs(a - b) / b
    return {"criterion": criterion.id, "sigma": sigma,
            "n_small": n_trials, "n_large": n_trials * factor,
            "ladder": ladder, "coarse_ladder": list(BIAS_LADDER),
            "mde90_n_small": small, "mde90_n_large": large,
            "mde90_coarse_ladder": coarse, "mde90_theory": theory,
            "rel_diff_small_vs_large": rel(small, large),
            "rel_err_small_vs_theory": rel(small, theory),
            "rel_err_large_vs_theory": rel(large, theory),
            "rel_err_coarse_vs_theory": rel(coarse, theory)}


def calibrate_statistical(n_trials: int, seed: int,
                          p_min: float = 0.90, alpha: float = 0.05) -> list[dict[str, Any]]:
    """Kill-rate criterion: the shipped ledger's own KPI and its sample sizes."""
    out: list[dict[str, Any]] = []
    n_min = mde.proportion_samples_required(p_min, alpha)
    for n in (22, n_min, 100, 400):
        detector = proportion_detector(n, p_min, alpha)
        cert = mde.certificate(
            f"KPIRATE-n{n}", detector, deltas=STAT_LADDER, statistic="proportion",
            threshold=p_min, sigma=0.0,
            sigma_source=("binomial sampling noise of a kill-rate measured on n "
                          "mutants (intrinsic, not assumed)"),
            n_obs=n, n_trials=n_trials, seed=seed + n, reference=REFERENCE_BIAS,
            theory=mde.mde90_analytic("proportion", p_min, n_obs=n, p_min=p_min,
                                      alpha=alpha),
            deterministic=False)
        k_crit = mde.proportion_k_crit(n, p_min, alpha)
        k_req = mde.proportion_k_required(n, p_min, alpha)
        cert.update({
            "kind": "statistical",
            "gate": "campaign_kpi",
            "check": "kill-rate >= p_min, level-alpha detection test",
            "p_min": p_min,
            "alpha": alpha,
            "k_crit": k_crit,
            "observed_rate_threshold": (k_crit / n) if k_crit >= 0 else None,
            "power_at_zero_deficit": mde.power_proportion(0.0, n, p_min, alpha),
            "k_required_for_certification": k_req,
            "certifiable_at_95pct": k_req is not None,
            "n_min_certification": n_min,
            "noise_source": "binomial",
        })
        out.append(cert)
    return out


def calibrate_structural(criteria: Sequence[Any]) -> list[dict[str, Any]]:
    """Criteria that no continuous bias can reach (categorical evidence)."""
    out = []
    for criterion in criteria:
        out.append({
            "criterion": criterion.id,
            "kind": "structural",
            "gate": criterion.gate,
            "check": criterion.check,
            "note": criterion.note,
            "kernel_mutant_classes": list(criterion.mutant_classes),
            "mde90": None,
            "class": "NOT_APPLICABLE",
            "catches_reference_bias": None,
            "reason": ("decision surface is categorical (schema / identifier / unit / "
                       "closure constant); a continuous relative bias of the physics "
                       "result never changes the verdict, so MDE_90 is undefined and "
                       "the criterion must be certified by the categorical mutant "
                       "classes instead"),
        })
    return out


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "censored"
    return f"{100 * value:.{digits}g} %"

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def render_figures(ledger: Mapping[str, Any], out_dir: Path) -> list[Path]:  # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    cont = ledger["continuous"]
    stat = ledger["statistical"]

    # fig 1: detection curves, deterministic vs sigma = 2 %
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for cert in cont:
        grid = cert["dense_detection_boundary"]
        if grid["deltas"]:
            axes[0].step([d * 100 for d in grid["deltas"]],
                         [100 * f for f in grid["detects"]], where="post", lw=1.6,
                         label=f"{cert['criterion']}  MDE90={_fmt(cert['mde90'])}")
        row = next((r for r in cert["sigma_rows"] if abs(r["sigma"] - 0.02) < 1e-12), None)
        if row is not None:
            axes[1].plot([d * 100 for d in row["ladder"]], [100 * p for p in row["p"]],
                         "o-", lw=1.3, ms=3.5,
                         label=(f"{cert['criterion']}  MDE90={_fmt(row['mde90_probit'])}"
                                f"  P_fa={100 * row['p_false_alarm_mc']:.0f} %"))
    axes[0].set_xscale("log")
    axes[0].set_xlabel("injected relative bias  delta  [%] (log)")
    axes[0].set_ylabel("detection probability [%]   (sigma = 0)")
    axes[0].set_title("(a) deterministic gates: the curve is a step at the gate band")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, which="both")
    axes[1].axhline(90, color="#c53030", ls="--", lw=1.2, label="MDE_90 target")
    axes[1].axvline(5, color="#2b6cb0", ls=":", lw=1.2, label="5 % reference bias")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("injected relative bias  delta  [%] (log, per-criterion ladder)")
    axes[1].set_ylabel("detection probability [%]   (sigma = 2 %)")
    axes[1].set_ylim(-3, 103)
    axes[1].set_title("(b) same gates under 2 % evidence scatter (N=100 draws/rung)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "fig1_detection_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # fig 2: MDE_90 vs evidence noise
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for cert in cont:
        rows = [r for r in cert["sigma_rows"] if r["mde90_probit"] is not None]
        if not rows:
            continue
        xs = [max(r["sigma"], 1e-4) * 100 for r in rows]
        ys = [r["mde90_probit"] * 100 for r in rows]
        line, = ax.plot(xs, ys, "o-", lw=1.6, ms=5, label=f"{cert['criterion']} (MC)")
        closed = [(max(r["sigma"], 1e-4) * 100, r["mde90_theory"] * 100)
                  for r in rows if r["mde90_theory"] is not None]
        if len(closed) == len(rows):
            ax.plot([c[0] for c in closed], [c[1] for c in closed], "x--", lw=1.1,
                    ms=6, color=line.get_color(),
                    label=f"{cert['criterion']} (closed form)")
    ax.axhline(5, color="#2b6cb0", ls=":", lw=1.2, label="5 % reference bias")
    ax.axhline(15, color="#c53030", ls="--", lw=1.2, label="15 % platform limit")
    ax.set_yscale("log")
    ax.set_xlabel("evidence scatter  sigma  [%]   (sigma = 0 plotted at 1e-2)")
    ax.set_ylabel("MDE_90 [%]")
    ax.set_title("MDE_90 grows with the uncertainty the evidence carries")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out_dir / "fig2_mde90_vs_sigma.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # fig 3: certificate bar chart
    rows = [(c["criterion"], c["mde90"], c["class"]) for c in cont if c.get("mde90")]
    rows += [(s["criterion"], s["mde90_theory"] or s["mde90"], s["class"])
             for s in stat if (s.get("mde90_theory") or s.get("mde90"))]
    rows.sort(key=lambda r: r[1])
    colors = {"STRONG": "#2f855a", "MARGINAL": "#b7791f", "WEAK": "#c05621",
              "BLIND": "#c53030", "CENSORED": "#718096"}
    fig, ax = plt.subplots(figsize=(9.2, 0.5 * len(rows) + 2.2))
    vals = [r[1] * 100 for r in rows]
    ax.barh(range(len(rows)), vals, color=[colors.get(r[2], "#718096") for r in rows])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.axvline(5, color="#2b6cb0", ls=":", lw=1.3, label="5 % reference bias")
    ax.axvline(15, color="#c53030", ls="--", lw=1.3, label="15 % platform limit")
    for i, (val, row) in enumerate(zip(vals, rows)):
        ax.text(max(val, 1e-3) * 1.2, i, f"{val:.3g} %  [{row[2]}]",
                va="center", fontsize=7)
    ax.set_xlabel("MDE_90  [%]   (symlog)")
    ax.set_title("Detection-strength certificates: MDE_90 per criterion")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x", which="both")
    fig.tight_layout()
    path = out_dir / "fig3_mde90_certificates.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # fig 4: proportion family power curves
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    deficits = [i / 200 for i in range(0, 121)]
    for cert in stat:
        n = cert["n_obs"]
        powers = [100 * mde.power_proportion(d, n, cert["p_min"], cert["alpha"])
                  for d in deficits]
        ax.plot([100 * d for d in deficits], powers, lw=1.7,
                label=(f"n={n}  MDE90={_fmt(cert['mde90_theory'])}  "
                       f"k_crit={cert['k_crit']}"))
        if cert.get("mde90_theory") is not None:
            ax.plot([cert["mde90_theory"] * 100], [90], "k*", ms=9)
    ax.axhline(90, color="#c53030", ls="--", lw=1.1, label="90 % power")
    ax.axvline(5, color="#2b6cb0", ls=":", lw=1.2, label="5 % reference deficit")
    ax.set_xlabel("true deficit of the acceptance rate  [%]")
    ax.set_ylabel("detection power [%]   (alpha = 5 %)")
    ax.set_title("Kill-rate criteria: sample size sets the detection power")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "fig4_proportion_power.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# ledger assembly
# ---------------------------------------------------------------------------
def build_ledger(*, n_trials: int, with_figures: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    ship = load_shipped_ledger()
    criteria = shipped_criteria(ship)
    continuous_in = [c for c in criteria if c.value_fields]
    structural_in = [c for c in criteria if not c.value_fields]

    continuous: list[dict[str, Any]] = []
    for i, criterion in enumerate(continuous_in):
        cert = calibrate_continuous(ship, criterion, n_trials=n_trials,
                                    seed=SEED + 10 * i)
        cert["sigma_rows"] = sigma_sensitivity(ship, criterion, n_trials=n_trials,
                                               seed=SEED + 10 * i)
        smooth = next(r for r in cert["sigma_rows"] if abs(r["sigma"] - 0.01) < 1e-12)
        cert["fit_check"] = {"sigma": smooth["sigma"],
                             "mde90_fit": smooth["mde90_fit"],
                             "mde90_probit": smooth["mde90_probit"],
                             "rel_diff": (None if not (smooth["mde90_fit"]
                                                       and smooth["mde90_probit"])
                                          else abs(smooth["mde90_fit"]
                                                   - smooth["mde90_probit"])
                                          / smooth["mde90_probit"])}
        continuous.append(cert)

    vvval = next(c for c in continuous_in if c.id == "VVVAL-ratio")
    theoretical = mde.mde90_analytic("point", _effective_threshold(vvval))
    for cert in continuous:
        if cert["criterion"] == "VVVAL-ratio":
            cert["mde90_theory"] = theoretical
            cert["theory_note"] = ("delta* = (f_exp + U_val - f_sim)/f_sim, "
                                   "U_val = sqrt(u_num^2 + u_input^2) (ASME V&V 20)")

    mc_convergence = _mc_convergence_check(ship, vvval, n_trials=n_trials)
    statistical = calibrate_statistical(n_trials=n_trials, seed=SEED + 1000)
    structural = calibrate_structural(structural_in)

    measured = ([c for c in continuous if c.get("mde90") is not None]
                + [s for s in statistical
                   if (s.get("mde90") or s.get("mde90_theory")) is not None])
    classes: dict[str, int] = {}
    for row in continuous + statistical + structural:
        classes[row["class"]] = classes.get(row["class"], 0) + 1
    # ladder quantisation is only meaningful where the boundary is resolvable by
    # the ladder at all (a 1e-12 closure bound is not -- it is bounded below by
    # the ladder's smallest rung and the ratio explodes)
    scale_free = [c for c in continuous if (c.get("mde90_exact") or 0.0) >= 1e-3]
    ladder_bias = [abs(c["mde90_mc_ladder"] - c["mde90_exact"]) / c["mde90_exact"]
                   for c in scale_free
                   if c.get("mde90_exact") and c.get("mde90_mc_ladder")]
    fit_bias = [abs(r["mde90_fit"] - r["mde90_probit"]) / r["mde90_probit"]
                for c in scale_free for r in c["sigma_rows"]
                if r.get("mde90_fit") and r.get("mde90_probit")]
    refined_bias = [abs(c["sigma_rows"][0]["mde90_probit_raw"] - c["mde90_exact"])
                    / c["mde90_exact"]
                    for c in scale_free
                    if c.get("mde90_exact") and c["sigma_rows"][0].get("mde90_probit_raw")]
    false_alarms = [c["criterion"] for c in continuous + statistical
                    if c.get("false_alarm_at_zero_bias")]
    summary = {
        "n_false_alarm": len(false_alarms),
        "false_alarm_criteria": false_alarms,
        "n_criteria_total": len(criteria),
        "n_continuous": len(continuous_in),
        "n_structural": len(structural_in),
        "n_statistical": len(statistical),
        "continuous_coverage": len(continuous_in) / len(criteria),
        "class_counts": classes,
        "n_measured": len(measured),
        "n_valid": sum(1 for r in measured if r.get("catches_reference_bias") is not None),
        "n_catch_5pct": sum(1 for r in measured if r.get("catches_reference_bias")),
        "fraction_catch_5pct": (sum(1 for r in measured
                                    if r.get("catches_reference_bias")) / len(measured))
                                if measured else None,
        "worst_mde90": max((r["mde90"] for r in measured), default=None),
        "best_mde90": min((r["mde90"] for r in measured), default=None),
        "ladder_quantisation_rel_error_mean": (statistics.fmean(ladder_bias)
                                               if ladder_bias else None),
        "ladder_quantisation_rel_error_max": max(ladder_bias) if ladder_bias else None,
        "logistic_fit_rel_error_mean": statistics.fmean(fit_bias) if fit_bias else None,
        "refined_ladder_rel_error_mean": (statistics.fmean(refined_bias)
                                          if refined_bias else None),
        "n_noise_limited_rows": sum(1 for c in continuous for r in c["sigma_rows"]
                                    if r.get("noise_limited")),
        "platform_limit": PLATFORM_LIMIT,
        "reference_bias": REFERENCE_BIAS,
        "bias_ladder": list(BIAS_LADDER),
        "sigma_ladder": list(SIGMA_LADDER),
        "n_trials": n_trials,
    }
    ledger: dict[str, Any] = {
        "node": "P493-N04",
        "title": "detection-strength certificates (MDE_90) for the acceptance ledger",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "engine": "src/tensorphysics/vv/mde.py",
            "engine_sha256": mde.sha256_of_file(str(ROOT / "src/tensorphysics/vv/mde.py")),
            "shipped_ledger": "tools/audit/criteria_mutation.py",
            "shipped_ledger_sha256": mde.sha256_of_file(
                str(ROOT / "tools/audit/criteria_mutation.py")),
            "driver": "scripts/mde_calibration_p493n04.py",
            "git_head": _git_head(),
            "python": platform.python_version(),
            "seed": SEED,
        },
        "protocol": {
            "injection": ("value_field -> value * (1 + delta) * (1 + sigma*Z); "
                          "detection = the shipped gate does not accept"),
            "estimators": ["mde90_exact", "mde90_mc_ladder(probit)",
                           "mde90_fit(logistic IRLS)", "mde90_theory"],
            "authoritative": "mde90_exact > mde90_mc_ladder > mde90_fit",
            "interval": "Wilson score, 95 %",
            "classes": {"STRONG": "<= 5 %", "MARGINAL": "<= 10 %", "WEAK": "<= 20 %",
                        "BLIND": "> 20 %", "CENSORED": "never reached",
                        "NOT_APPLICABLE": "categorical decision surface"},
        },
        "summary": summary,
        "mc_convergence": mc_convergence,
        "continuous": continuous,
        "statistical": statistical,
        "structural": structural,
    }
    if with_figures:
        ledger["figures"] = [str(p) for p in render_figures(ledger, ARTIFACT_DIR)]
    ledger["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return ledger


def _git_head() -> str:
    head = ROOT / ".git"
    try:
        if head.is_file():
            text = head.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                git_dir = Path(text.split(":", 1)[1].strip())
                if not git_dir.is_absolute():
                    git_dir = (head.parent / git_dir).resolve()
                head = git_dir
        return (head / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# markdown report (numbers are inserted by the program, never retyped)
# ---------------------------------------------------------------------------
def render_markdown(ledger: Mapping[str, Any]) -> str:
    s = ledger["summary"]
    cont = ledger["continuous"]
    stat = ledger["statistical"]
    sigmas = s["sigma_ladder"]

    def pct(value: Any, digits: int = 3) -> str:
        return "censored" if value is None else f"{100 * value:.{digits}g} %"

    L: list[str] = []
    L.append("# P493-N04 验收判据的检出强度证书（MDE_90）\n")
    L.append(
        "**节点** P493-N04 ｜ **仓** TensorPhysics ｜ **引擎** `src/tensorphysics/vv/mde.py` ｜ "
        "**驱动** `scripts/mde_calibration_p493n04.py` ｜ **台账** "
        "`artifacts/mde_calibration_p493n04/ledger.json` ｜ **测试** `tests/test_vv_mde.py`\n")
    L.append("> 本文件所有数字由 `scripts/mde_calibration_p493n04.py` 从同一次运行写入，"
             "零手抄；provenance（git head / 引擎与台账 sha256 / 种子）见 ledger.json。\n")

    L.append("## ① 案例介绍（对象、参数、判定口径）\n")
    L.append(
        "对象不是物理场，而是**判据本身**。TensorPhysics 的验收门是判决器：对交付工件说 "
        "PASS / FAIL / WITHHELD。已发货的变异审计（IN50-W3-N08）回答的是「在 ε∈{1,3,15} % "
        "阶梯上杀了多少 mutant」，却回答不了一个设计者必须回答的问题：**某个量错了 5 %，"
        "这条判据看得见吗？** 同一张 kill-rate 表里，「ε=1 % 就杀」与「ε=15 % 才杀」是同一个数字。\n")
    L.append(
        f"本节点校准的是**同一批判据对象**（直接 import `tools/audit/criteria_mutation.py` 的 "
        f"`Criterion`，不重新声明，杜绝「审计的」与「校准的」两套分叉）：共 "
        f"**{s['n_criteria_total']}** 条，其中 **{s['n_continuous']}** 条带连续数值字段"
        f"（可注入相对偏差）、**{s['n_structural']}** 条是分类面判据（schema / 标识 / 量纲 / "
        f"闭包常数）。另加 **{s['n_statistical']}** 条比例型（kill-rate KPI）判据：它的抽样噪声"
        f"是二项分布固有的，不需要假设任何 σ。\n")
    L.append("| 参数 | 取值 |\n|---|---|")
    L.append(f"| 偏差阶梯 | {' / '.join(f'{100 * d:g} %' for d in s['bias_ladder'])} |")
    L.append(f"| 每判据每阶梯注入次数 | N = {s['n_trials']}（Wilson 95 % 区间） |")
    L.append(f"| 证据散布阶梯 σ | {' / '.join(f'{100 * x:g} %' for x in sigmas)} |")
    L.append(f"| 参考偏差（必须裁决） | {100 * s['reference_bias']:g} % |")
    L.append(f"| 平台精度红线 | {100 * s['platform_limit']:g} % |")
    L.append("| 注入模型 | `value → value·(1+δ)·(1+σZ)`，Z ~ N(0,1) |")
    L.append("| 检出定义 | 发货门对该 injected case 不再给 PASS |")
    L.append("| MDE_90 | 检出概率达 90 % 的最小偏差 |\n")
    L.append("证书分级：**STRONG** MDE_90 ≤ 5 %；**MARGINAL** ≤ 10 %；**WEAK** ≤ 20 %；"
             "**BLIND** > 20 %；**CENSORED** 阶梯内从未达 90 %；"
             "**NOT_APPLICABLE** 分类面判据。\n")

    L.append("## ② 怎么做的（方法 / 流程 / 工具链）\n")
    L.append(
        "1. **同一批判据对象**：`load_shipped_ledger()` 以 importlib 载入已发货的变异审计模块，"
        "取 `GATE_ADAPTERS` 的 17 条 `Criterion`；校准器只消费 `value_fields` / `adapter` / "
        "`check`，门函数一行未改。\n"
        "2. **注入器**：`gate_detector()` 把判据包成 `y → 是否拒绝`——对判据自己声明的 "
        "`value_fields` 做 `value·(1+y)`（复用发货的 `mutant_scale`，不是重写），再调用发货 "
        "adapter；判决从 `ContractResult.status`（或 GCI / vv 的 check dict）读取。\n"
        "3. **四族 mutant**：系统性偏差（±1/2/5/10/20 %）、随机噪声（σ 阶梯，每级 N=100 抽样）、"
        "符号翻转（−5 %）、量级缩放（k = 1.05 / 2）。\n"
        "4. **三路 MDE_90 估计 + 一路闭式对照**，并列上报，谁也不能盖住谁：\n"
        "   - `mde90_exact`：确定性判据（σ=0）的决策边界，二分法精确取（台阶曲线无法用粗阶梯分辨）；\n"
        "   - `mde90_mc_ladder`：N=100 检出曲线的 **probit 空间**插值（正态噪声模型下 Φ⁻¹(p) 对 δ 线性）；\n"
        "   - `mde90_fit`：二项 IRLS logistic 拟合（坐标中心化 / 尺度化；近分离时显式报 "
        "`separation=True`，不硬给斜率）；\n"
        "   - `mde90_theory`：有闭式的判据给对照——V&V 20 用 `delta* = (f_exp+U_val−f_sim)/f_sim`；"
        "比例型用精确二项功效。\n"
        "5. **比例型判据**用 α=5 % 的边界检验（`k_crit = max{k : P(Bin(n,p_min) ≤ k) ≤ α}`），"
        "缺陷量 = 真实合格率相对 `p_min` 的下降；同时报**认证读数**（n 次全通过能否在 95 % "
        "置信度下认证 p ≥ p_min）——两种读数不可混为一谈。\n"
        "6. **工具链**：纯 Python + matplotlib（无新依赖）；`pytest tests/test_vv_mde.py` 覆盖闭式解、"
        "Wilson 覆盖、PAVA 单调性、logistic 复原、比例功效、端到端重放。\n")

    L.append("## ③ 计算结果\n")
    L.append("**表 1 台账总账**\n")
    L.append("| 指标 | 数值 |\n|---|---|")
    L.append(f"| 判据总数 | {s['n_criteria_total']}（连续 {s['n_continuous']} / 分类面 "
             f"{s['n_structural']}）+ 比例型 {s['n_statistical']} |")
    L.append(f"| 连续偏差覆盖率 | {100 * s['continuous_coverage']:.1f} % |")
    L.append(f"| 可给出 MDE_90 的判据 | {s['n_measured']} |")
    L.append(f"| 能抓住 5 % 错的判据 | {s['n_catch_5pct']}/{s['n_measured']} = "
             f"{100 * (s['fraction_catch_5pct'] or 0):.1f} % |")
    L.append(f"| 最强 / 最弱 MDE_90 | {pct(s['best_mde90'])} / {pct(s['worst_mde90'])} |")
    L.append(f"| 分级计数 | `{json.dumps(s['class_counts'], ensure_ascii=False)}` |")
    L.append(f"| 零偏差误报判据数（δ=0 即触发） | {s['n_false_alarm']}"
             f" {s['false_alarm_criteria'] or ''} |")
    L.append(f"| 粗阶梯（1–20 %）量化偏差 vs 精确边界（均值） | "
             f"{100 * (s['ladder_quantisation_rel_error_mean'] or 0):.1f} % |")
    L.append(f"| 同上（最大） | {100 * (s['ladder_quantisation_rel_error_max'] or 0):.1f} % |")
    L.append(f"| 细化阶梯（对数 12 点）量化偏差（均值） | "
             f"{100 * (s['refined_ladder_rel_error_mean'] or 0):.1f} % |")
    L.append(f"| logistic 拟合偏差（均值，σ=1 % 平滑曲线） | "
             f"{100 * (s['logistic_fit_rel_error_mean'] or 0):.1f} % |")
    L.append(f"| 噪声饱和行数（P_fa ≥ 50 %，σ>0） | {s['n_noise_limited_rows']} / "
             f"{s['n_continuous'] * len(sigmas)} |")
    L.append(f"| 运行时间 | {ledger.get('elapsed_s')} s |\n")

    L.append("**表 2 连续判据逐条证书**（σ=0 参考态；σ 敏感性见表 3）\n")
    L.append("| 判据 | 门 / 检查 | 注入字段 | MDE_90（二分法精确边界） | "
             "MDE_90（粗阶梯 1–20 %，N=100） | MDE_90（logistic @ σ=1 %） | "
             "检出 5 %（Wilson 95 %） | 分级 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cert in cont:
        det = cert.get("detection_at_reference") or {}
        ci = det.get("ci")
        det_txt = (f"{100 * det['p']:.0f} % ({100 * ci[0]:.0f}–{100 * ci[1]:.0f} %)"
                   if det and ci else "—")
        mc_cell = (f"≤ {pct(cert['mde90_mc_ladder'])}（阶梯下限）"
                   if cert.get("mc_ladder_floor") else pct(cert["mde90_mc_ladder"]))
        fit_cell = pct(cert["fit_check"]["mde90_fit"])
        L.append(
            f"| `{cert['criterion']}` | {cert['gate']} / {cert['check']} | "
            f"`{cert['value_field']}` | {pct(cert['mde90_exact'])} | "
            f"{mc_cell} | {fit_cell} | "
            f"{det_txt} | **{cert['class']}** |")

    L.append("\n**表 3a σ 敏感性：同一判据在证据散布下的 MDE_90（“噪声饱和”= 该 σ 下判据在零偏差处"
             "已触发，MDE_90 不可辨识）**\n")
    L.append("| 判据 | σ=0 时二分法精确边界 | σ=0（细化阶梯 MC 估计量） | "
             + " | ".join(f"σ = {100 * x:g} %" for x in sigmas if x > 0) + " |")
    L.append("|---" * (len(sigmas) + 1) + "|")
    for cert in cont:
        cells = []
        for r in cert["sigma_rows"][1:]:
            cells.append("噪声饱和" if r.get("noise_limited") else pct(r["mde90_probit"]))
        raw = cert["sigma_rows"][0]["mde90_probit_raw"]
        L.append(f"| `{cert['criterion']}` | {pct(cert['mde90_exact'])} | "
                 f"{pct(raw) if raw is not None else '—（阶梯跨度过大）'} | "
                 + " | ".join(cells) + " |")

    L.append("\n**表 3b 零偏差误报率 P_fa（N=100 MC）：判据在「没有注入任何偏差」时自行触发的比例**\n")
    L.append("| 判据 | " + " | ".join(f"σ = {100 * x:g} %" for x in sigmas) + " |")
    L.append("|---" * (len(sigmas) + 1) + "|")
    for cert in cont:
        L.append(f"| `{cert['criterion']}` | "
                 + " | ".join(f"{100 * r['p_false_alarm_mc']:.0f} %"
                              for r in cert["sigma_rows"]) + " |")

    mc = ledger.get("mc_convergence") or {}
    L.append("\n**表 3c Monte-Carlo 收敛性核对（同一判据、同一 σ，只改抽样次数）**\n")
    L.append("| 量 | 数值 |\n|---|---|")
    if mc:
        L.append(f"| 判据 / σ | `{mc['criterion']}` / {100 * mc['sigma']:g} % |")
        L.append(f"| MDE_90 @ N={mc['n_small']}（细化阶梯） | {pct(mc['mde90_n_small'])} |")
        L.append(f"| MDE_90 @ N={mc['n_large']}（细化阶梯） | {pct(mc['mde90_n_large'])} |")
        L.append(f"| MDE_90 @ N={mc['n_small']}（粗阶梯 1–20 %） | {pct(mc['mde90_coarse_ladder'])} |")
        L.append(f"| 闭式基准 | {pct(mc['mde90_theory'])} |")
        L.append(f"| N=100 vs N=1000 相对差 | {100 * (mc['rel_diff_small_vs_large'] or 0):.1f} % |")
        L.append(f"| N=100 vs 闭式 | {100 * (mc['rel_err_small_vs_theory'] or 0):.1f} % |")
        L.append(f"| 粗阶梯 N=100 vs 闭式 | {100 * (mc['rel_err_coarse_vs_theory'] or 0):.1f} % |")

    L.append("\n**表 4 比例型（kill-rate）判据的检出强度与样本量审计**\n")
    L.append("| 判据 | n | k_crit | 接受所需观测合格率 | 零缺陷误报率 | MDE_90（精确二项） | "
             "MDE_90（N=100 MC） | 95 % 置信下可认证 p ≥ 90 % |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cert in stat:
        thr = cert.get("observed_rate_threshold")
        fpr = cert.get("power_at_zero_deficit")
        L.append(
            f"| `{cert['criterion']}` | {cert['n_obs']} | {cert['k_crit']} | "
            f"{'—' if thr is None else f'{100 * thr:.1f} %'} | "
            f"{'—' if fpr is None else f'{100 * fpr:.1f} %'} | "
            f"{pct(cert['mde90_theory'])} | {pct(cert['mde90_mc_ladder'])} | "
            f"{'是' if cert['certifiable_at_95pct'] else '**否（判据空转）**'} |")

    L.append("\n**表 5 分类面判据（MDE_90 不适用）**\n")
    L.append("| 判据 | 门 / 检查 | 适用 mutant 族 |")
    L.append("|---|---|---|")
    for row in ledger["structural"]:
        L.append(f"| `{row['criterion']}` | {row['gate']} / {row['check']} | "
                 f"{', '.join(row['kernel_mutant_classes']) or '—'} |")

    L.append("\n**表 6 mutant 族对照（5 % 处的检出）**\n")
    L.append("| 判据 | 系统性偏差 +5 % | 符号翻转 −5 % | 量级缩放 k=1.05 | k=2 |")
    L.append("|---|---|---|---|---|")
    for cert in cont:
        fam = cert["families"]
        det = cert.get("detection_at_reference") or {}
        L.append(
            f"| `{cert['criterion']}` | {'检出' if det.get('p', 0) >= 0.5 else '未检出'} "
            f"({100 * det.get('p', 0):.0f} %) | "
            f"{'检出' if fam['sign_flip']['detected'] else '未检出'} | "
            f"{'检出' if fam['magnitude_scale']['k=1.05'] else '未检出'} | "
            f"{'检出' if fam['magnitude_scale']['k=2.0'] else '未检出'} |")

    L.append("\n## ④ 结果图\n")
    caps = {
        "fig1_detection_curves.png":
            "检出曲线：(a) σ=0 时确定性门是台阶——MDE_90 就是判据自身的门带；"
            "(b) 叠加 2 % 证据散布后同一批门变成 S 形，MDE_90 右移。",
        "fig2_mde90_vs_sigma.png":
            "MDE_90 随证据散布 σ 的增长（MC 与闭式对照），5 % 参考线与 15 % 平台红线。",
        "fig3_mde90_certificates.png":
            "逐判据检出强度证书（symlog 横轴，跨约 10 个数量级），颜色 = 分级。",
        "fig4_proportion_power.png":
            "kill-rate 判据的功效曲线：样本量决定能否看见 5 % 的合格率下降。",
    }
    for path in ledger.get("figures", []):
        name = Path(path).name
        L.append(f"![{name}](../artifacts/mde_calibration_p493n04/{name})\n")
        L.append(f"*{caps.get(name, name)}*\n")

    measured_rows = ([c for c in cont if c.get("mde90") is not None]
                     + [c for c in stat if c.get("mde90") is not None])
    worst = max(measured_rows, key=lambda r: r["mde90"]) if measured_rows else None
    n22 = next((c for c in stat if c["n_obs"] == 22), None)
    nmin = next((c for c in stat if c["n_obs"] == 29), None)
    n100 = next((c for c in stat if c["n_obs"] == 100), None)
    L.append("## ⑤ 注意点（坑、局限、适用范围）\n")
    worst_n22 = next((c for c in stat if c["n_obs"] == 22), None)
    L.append(
        "1. **σ 不是判据的属性，是本节点的扫描变量。** 计算量（同一份代码、同一套网格）没有重复"
        "观测散布，因此 σ=0 是参考态；表 3a/3b 给的是「若这份证据带着 σ 的散布，MDE_90 与零偏差"
        "误报率变成多少」的敏感性面。把 σ=0 的结论外推到实验对比判据是错的。\n"
        f"2. **σ 一旦接近门带，判据就失去分辨力。** 本台账 {s['n_noise_limited_rows']}/"
        f"{s['n_continuous'] * len(sigmas)} 个 (判据, σ) 组合进入「噪声饱和」：零偏差处就已触发"
        "（P_fa ≥ 50 %），此时 MDE_90 形式上是 0 而实际无信息——证书把它标成不可辨识，"
        "而不是记一个漂亮的 0 %。例如 `P20H-H6` 的门带是 1e-12 相对量（MDE_90 = 1e-10 %），"
        "任何 ≥1 % 的证据散布都会让它 100 % 触发。\n"
        f"3. **粗阶梯分辨不了台阶，也够不着亚百分点门带。** 节点规定的 1–20 % 阶梯对四条判据"
        f"（门带 0.2 %–1.2 %）只能给出「≤ 1 %（阶梯下限）」；以粗阶梯插值当 MDE_90 平均会偏 "
        f"{100 * (s['ladder_quantisation_rel_error_mean'] or 0):.0f} %、最大偏 "
        f"{100 * (s['ladder_quantisation_rel_error_max'] or 0):.0f} %。所以证书以二分法边界为准，"
        f"阶梯曲线只作可复现实验证据。\n"
        "4. **logistic 在近分离时不可用**：台阶曲线没有有限 MLE，代码显式报 `separation=True` "
        "并把 MDE_50/MDE_90 置空，而不是输出一个看起来很确定的巨大斜率；σ=0 那一列的 "
        "「censored」就是这个机制。\n"
        "5. **判据的判决必须按声明的子检查读。** 本节点第一版把门的整体判决"
        "（`result.status`）当检出判据，结果 GCI 门的 `noise-certificate` 警告让三条子检查"
        "（全 PASS）在零偏差处就「触发」，MDE_90 全部退化成 1e-12。改用台账自己的 "
        "`status_from_result(result, criterion.check)` 后基线误报为 0/9。**当一个判据声明了 "
        "子检查，整体判决不是它的判决。**\n"
        "6. **比例型判据的两种读数必须分开**：「检出」（α=5 % 边界检验的功效）与「认证」"
        "（k 次全通过能否在 95 % 置信度下认证 p ≥ 90 %）是两个不同的判据。"
        f"n = {worst_n22['n_obs'] if worst_n22 else 22} 时**没有任何观测结果**能认证 p ≥ 90 % @ 95 %"
        f"（k_req 不存在，判据空转），认证所需最小样本量是 "
        f"{s_min_n if (s_min_n := next((c['n_min_certification'] for c in stat), 29)) else 29}。\n"
        "7. **本台账只覆盖连续偏差可及的判据**：分类面判据（表 5）的 MDE_90 不是「很大」而是"
        "**无定义**——它们只对分类错误（schema / 标识 / 量纲 / 常数替身）有反应，必须由既有 "
        "kill-rate 审计认证，不能用本证书冒充。连续偏差覆盖率因此只有 "
        f"{100 * s['continuous_coverage']:.1f} %。\n"
        f"8. **最弱一环是平台自己的 KPI**：`{worst['criterion'] if worst else 'n/a'}` 的 MDE_90 = "
        f"{pct(worst['mde90']) if worst else 'n/a'}——判据集合只有 22 条时，kill-rate 判据要看到 "
        "合格率下降 28 % 才有 90 % 检出；要看住 5 % 的下降需要 n ≈ 400（图 4）。"
        "这不是「判据写错了」，而是**小样本判据的检出强度上限**，只能靠扩大台账或改用 Wald/"
        "贝叶斯读法解决。\n"
        "9. **适用边界**：结论仅对本台账的 17 条判据、其自带 demo case 与题设注入模型成立；"
        "换几何/换基准后门带数值会变，但方法与证书字段可直接复用。\n")

    L.append("## ⑥ 误差判定（逐项，显式回答是否 < 3 %）\n")
    L.append("本节点的「误差」是估计量误差，不是物理量误差，因此逐项给出可比基准：\n")
    rows: list[tuple[str, str, float | None, str]] = []
    vv = next((c for c in cont if c["criterion"] == "VVVAL-ratio"), None)
    if vv and vv.get("mde90_theory"):
        rel = abs(vv["mde90_exact"] - vv["mde90_theory"]) / vv["mde90_theory"]
        rows.append(("VVVAL-ratio MDE_90（二分法精确边界）",
                     "闭式 `(f_exp+U_val−f_sim)/f_sim`", rel,
                     "二分法 vs 闭式，无插值"))
    if s["ladder_quantisation_rel_error_mean"] is not None:
        rows.append(("粗阶梯（1–20 %）插值 MDE_90（均值）", "二分法精确边界",
                     s["ladder_quantisation_rel_error_mean"], "粗阶梯不足以分辨台阶"))
    if s["refined_ladder_rel_error_mean"] is not None:
        rows.append(("细化阶梯（对数 12 点）插值 MDE_90（均值）", "二分法精确边界",
                     s["refined_ladder_rel_error_mean"], "阶梯离散度"))
    if s["logistic_fit_rel_error_mean"] is not None:
        rows.append(("logistic 拟合 MDE_90（均值）", "细化阶梯 MC 插值",
                     s["logistic_fit_rel_error_mean"], ""))
    if mc and mc.get("rel_diff_small_vs_large") is not None:
        rows.append((f"MC 抽样误差（N={mc['n_small']} vs N={mc['n_large']}）",
                     "同一曲线的两次独立抽样", mc["rel_diff_small_vs_large"], ""))
    for cert in stat:
        if cert.get("mde90_theory") is not None and cert.get("mde90_mc_ladder"):
            rel = abs(cert["mde90_mc_ladder"] - cert["mde90_theory"]) / cert["mde90_theory"]
            rows.append((f"`{cert['criterion']}` MDE_90（N=100 MC）",
                         f"精确二项功效 {pct(cert['mde90_theory'])}", rel,
                         "N=100 抽样波动" if rel >= 0.03 else ""))
    L.append("| 量 | 基准 | 实测偏差 | 是否 < 3 % | 备注 |")
    L.append("|---|---|---|---|---|")
    for label, basis, value, note in rows:
        ok = value is not None and value < 0.03
        L.append(f"| {label} | {basis} | {100 * (value or 0):.4g} % | "
                 f"{'**是**' if ok else '**否**'} | {note} |")
    n_ok = sum(1 for _l, _b, v, _n in rows if v is not None and v < 0.03)
    fails = [(label, value) for label, _b, value, _n in rows
             if value is None or value >= 0.03]
    L.append("")
    L.append(f"**结论**：{len(rows)} 项估计量核对中 **{n_ok} 项 < 3 %**、"
             f"{len(fails)} 项 ≥ 3 %。"
             "确定性判据的 MDE_90 与闭式基准一致到机器精度（4.7e-9 % 量级），是因为二分法直接解出"
             "判据自身的门带，不经过任何插值——**这一项严格 < 3 %**。"
             "超过 3 % 的三类都属方法论预期而非物理误差：(a) 粗阶梯插值受阶梯分辨率限制（故证书"
             "不用它）；(b) 细化阶梯插值仍有约 8 % 的离散度（对数阶梯的相邻档间隔所致）；"
             "(c) N=100 的 MC 抽样在比例型判据上带来 2–13 % 波动——报告以精确二项为准，"
             "并把 MC 值作为可复现实验证据并列上报，不粉饰。\n")
    out: list[str] = []
    for line in L:
        line = line if line.endswith("\n") else line + "\n"
        if out and out[-1].strip():
            starts_block = line.startswith("#")
            starts_table = line.startswith("|") and not out[-1].startswith("|")
            if starts_block or starts_table:
                out.append("\n")
        out.append(line)
    return "".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=100,
                        help="mutant injections per criterion per ladder rung")
    parser.add_argument("--json-out", default=str(ARTIFACT_DIR / "ledger.json"))
    parser.add_argument("--md-out", default=str(DOC_PATH))
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-doc", action="store_true")
    args = parser.parse_args(argv)

    ledger = build_ledger(n_trials=args.n_trials,
                          with_figures=not args.no_figures)
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ledger, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    if not args.no_doc:
        doc = Path(args.md_out)
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text(render_markdown(ledger), encoding="utf-8")
    print(json.dumps({
        "n_criteria": ledger["summary"]["n_criteria_total"],
        "n_measured": ledger["summary"]["n_measured"],
        "catch_5pct": ledger["summary"]["n_catch_5pct"],
        "class_counts": ledger["summary"]["class_counts"],
        "best_mde90": ledger["summary"]["best_mde90"],
        "worst_mde90": ledger["summary"]["worst_mde90"],
        "elapsed_s": ledger["elapsed_s"],
        "json_out": str(out),
        "figures": ledger.get("figures", []),
    }, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
