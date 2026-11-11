"""End-to-end verification of the multi-fidelity selection engine.

One call (:func:`run_verification`) does what the campaign node asks for:

1. load the registered model set (accuracy prior, cost prior, validity domain);
2. simulate a validation campaign -- residual samples of every model drawn from a
   *deployment truth* that the recalibration never sees;
3. build four registry variants: registered priors only, conformal pooled,
   conformal stratified on the declared partition, and conformal stratified on a
   deliberately **wrong** partition (falsification probe);
4. backtest every variant against both a prior-consistent truth (mechanism
   control) and the deployment truth (2000 independent queries, fixed seeds);
5. evaluate the acceptance criteria (accuracy, cost efficiency, interval
   coverage, negative control) against the numbers in the configuration.

The returned object carries the machine-readable artifact plus the per-query
records needed by :mod:`tensorphysics.multifidelity.figures`.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .backtest import (AxisSpec, RegimeRule, TruthGenerator, TruthSpec,
                       generate_validation_residuals, run_backtest, sample_queries)
from .calibration import StratumSpec, calibrate_registry, calibrate_registry_stratified
from .model import ModelRegistry
from .selection import Budget, MIN_COST, PASS, pareto_front, frontier_curve, select_model

NODE_ID = "p9003w1_n05"
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "drag_multifidelity.v1.json"


# ---------------------------------------------------------------------------
# configuration plumbing
# ---------------------------------------------------------------------------
def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    config["__path__"] = str(path)
    config["__digest__"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config


def build_registry(config: Mapping[str, Any]) -> ModelRegistry:
    payload = {k: v for k, v in config.items() if not k.startswith("__")}
    payload.pop("axes", None)
    return ModelRegistry.from_dict(payload, source=config.get("__path__", "<config>"))


def build_axes(config: Mapping[str, Any]) -> Dict[str, AxisSpec]:
    return {name: AxisSpec.from_dict(spec) for name, spec in config["axes"].items()}


def build_strata(config: Mapping[str, Any]) -> List[StratumSpec]:
    return [StratumSpec.from_dict(row) for row in config.get("strata", [])]


def build_mispartition(config: Mapping[str, Any]) -> List[StratumSpec]:
    """Deliberately wrong partition on a different axis (falsification probe)."""
    design = config.get("stratification_design", {})
    probe = design.get("mispartition_probe", "")
    # The probe is declared in the config text; the boxes below are its machine form.
    return [StratumSpec(name="cb_high", bounds={"Cb": (0.75, 0.90)}),
            StratumSpec(name="cb_low", bounds={"Cb": (0.48, 0.68)})] if probe else []


def build_truths(config: Mapping[str, Any]) -> Dict[str, TruthGenerator]:
    out: Dict[str, TruthGenerator] = {}
    for row in config["truths"]:
        rules = tuple(RegimeRule.from_dict(r) for r in row.get("regime_rules", []))
        spec = TruthSpec(
            label=str(row["label"]),
            sigma_inflation=float(row.get("sigma_inflation", 1.0)),
            jump_prob=float(row.get("jump_prob", 0.0)),
            jump_median=float(row.get("jump_median", 2.2)),
            jump_sigma=float(row.get("jump_sigma", 0.4)),
            cost_sigma_inflation=float(row.get("cost_sigma_inflation", 1.0)),
            cost_median_inflation=float(row.get("cost_median_inflation", 1.0)),
            regime_rules=rules,
        )
        out[spec.label] = TruthGenerator(spec=spec)
    return out


# ---------------------------------------------------------------------------
# registry variants
# ---------------------------------------------------------------------------
def build_registry_variants(config: Mapping[str, Any],
                            registry: ModelRegistry,
                            deployment_truth: TruthGenerator) -> tuple[Dict[str, ModelRegistry], Dict[str, Any]]:
    """Prior-only, pooled conformal, stratified conformal, mispartitioned conformal."""
    axes = build_axes(config)
    strata = build_strata(config)
    mispartition = build_mispartition(config)
    validation = config.get("validation", {})
    val_seed = int(validation.get("seed", 11))
    n_stratum = int(validation.get("n_per_stratum", 250))
    n_pooled = int(validation.get("n_pooled_min", 400))
    min_n = int(validation.get("min_n_per_stratum", 30))
    level = float(validation.get("interval_level", 0.90))

    res_declared = generate_validation_residuals(
        registry, deployment_truth, axes, strata=strata, n_per_stratum=n_stratum,
        n_pooled_min=n_pooled, seed=val_seed)
    res_mis = generate_validation_residuals(
        registry, deployment_truth, axes, strata=mispartition, n_per_stratum=n_stratum,
        n_pooled_min=n_pooled, seed=val_seed + 2) if mispartition else {}

    pooled = {mid: groups["__pooled__"] for mid, groups in res_declared.items()}
    declared_groups = {mid: {k: v for k, v in groups.items() if k != "__pooled__"}
                      for mid, groups in res_declared.items()}
    mis_groups = {mid: {k: v for k, v in groups.items() if k != "__pooled__"}
                  for mid, groups in res_mis.items()}

    variants: Dict[str, ModelRegistry] = {"prior_only": registry}
    variants["conformal_pooled"] = calibrate_registry(
        registry, pooled, strata=(), min_n=min_n, level=level)
    variants["conformal_stratified"] = calibrate_registry_stratified(
        registry, declared_groups, strata, min_n=min_n, level=level)
    if mis_groups:
        variants["conformal_mispartitioned"] = calibrate_registry_stratified(
            registry, mis_groups, mispartition, min_n=min_n, level=level)

    meta = {
        "validation_seed": val_seed,
        "n_per_stratum": n_stratum,
        "n_pooled_min": n_pooled,
        "declared_strata": [s.to_dict() for s in strata],
        "mispartition_probe": [s.to_dict() for s in mispartition],
        "residuals": {mid: {"n": len(values)} for mid, values in pooled.items()},
        "calibration": {name: {mid: reg.get(mid).calibration for mid in reg.ids()}
                        for name, reg in variants.items()},
    }
    return variants, meta

# ---------------------------------------------------------------------------
# backtests and acceptance
# ---------------------------------------------------------------------------
@dataclass
class VerificationResult:
    artifact: Dict[str, Any]
    records: Dict[str, Dict[str, List[Dict[str, Any]]]] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)


def evaluate_acceptance(metrics: Mapping[str, Any],
                        acceptance: Mapping[str, Any],
                        negative_controls: Sequence[Mapping[str, Any]],
                        pytest_summary: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Turn measured rates into pass/fail checks against the configured thresholds."""
    checks: Dict[str, Any] = {}

    def _check(key: str, value: Optional[float], threshold: float, comparison: str,
               note: str = "") -> None:
        if value is None:
            checks[key] = {"value": None, "threshold": threshold, "pass": False,
                           "note": note or "metric unavailable"}
            return
        ok = value >= threshold if comparison == ">=" else value <= threshold
        checks[key] = {"value": value, "threshold": threshold, "comparison": comparison,
                       "pass": bool(ok), "note": note}

    _check("a_recommendation_accuracy", metrics.get("accuracy"),
           float(acceptance["accuracy_min"]), ">=",
           "P(realised error <= epsilon) over recommended queries")
    _check("b_cost_efficiency", metrics.get("cost_efficiency"),
           float(acceptance["cost_efficiency_min"]), ">=",
           "P(realised cost <= C) over recommended queries")
    _check("c_interval_coverage", metrics.get("coverage"),
           float(acceptance["coverage_min"]), ">=",
           "empirical coverage of the reported one-sided bound")
    # negative control: a sub-1 % accuracy budget must be refused
    ctrl_rows = list(negative_controls)
    ctrl_ok = all(row["status"] != PASS and row["reason_code"] == "EPSILON_BELOW_MODEL_FLOOR"
                  for row in ctrl_rows)
    checks["d_negative_control"] = {
        "value": [row["epsilon"] for row in ctrl_rows if row["status"] != PASS],
        "threshold": "every epsilon < 1 % must return FAIL/EPSILON_BELOW_MODEL_FLOOR",
        "pass": bool(ctrl_ok and ctrl_rows),
        "detail": ctrl_rows,
    }
    if pytest_summary is not None:
        n_tests = int(pytest_summary.get("n_tests", 0))
        checks["e_pytest"] = {"value": n_tests, "threshold": 10, "comparison": ">=",
                              "pass": n_tests >= 10, "detail": dict(pytest_summary)}
    coverage_worst = metrics.get("coverage_worst_stratum")
    checks["f_coverage_worst_stratum"] = {
        "value": coverage_worst, "threshold": float(acceptance["coverage_min"]),
        "comparison": ">=", "pass": bool(coverage_worst is not None
                                         and coverage_worst >= float(acceptance["coverage_min"])),
        "note": "reported for transparency: stratification must not leave a stratum uncovered"}
    checks["all_pass"] = all(v["pass"] for k, v in checks.items() if k != "f_coverage_worst_stratum")
    return checks


def run_negative_controls(registry: ModelRegistry, config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Queries with ``epsilon < 1 %`` must be refused, not silently mis-served."""
    acceptance = config["acceptance"]
    params = {"Fn": 0.22, "Cb": 0.70, "Re": 1.0e9}
    out: List[Dict[str, Any]] = []
    for eps in acceptance["negative_control_epsilons"]:
        budget = Budget(epsilon=float(eps), cost=1.0e7,
                        error_confidence=float(config["budgets"]["error_confidence"]),
                        cost_confidence=float(config["budgets"]["cost_confidence"]),
                        interval_level=float(config["validation"]["interval_level"]))
        selection = select_model(registry, params, budget, objective=acceptance.get("objective", MIN_COST))
        out.append({"epsilon": float(eps), "status": selection.status,
                    "reason_code": selection.reason_code, "reason": selection.reason,
                    "model_id": selection.model_id,
                    "error_floor": registry.error_floor(params)})
    return out


def physical_anchor() -> Dict[str, Any]:
    """Evaluate the registered quantity at a public hull to anchor the budget space.

    Uses the repository's Holtrop-1984 kernel on the KRISO KCS particulars; the
    purpose is to show that the ``(Fn, Cb, Re)`` box and the ``epsilon`` values
    correspond to real resistance magnitudes, **not** to claim an EFD benchmark.
    """
    from tensorphysics.physics_kernels_holtrop1984 import HullGeometry, holtrop1984_resistance

    geom = HullGeometry(name="KCS", l=230.0, b=32.2, t=10.8, cb=0.6505, cm=0.9770,
                        cwp=0.8596, lcb_pct=-1.48, cstern=0.0)
    rows = []
    for knots in (18.0, 21.0, 24.0):
        speed = knots * 0.514444
        res = holtrop1984_resistance(geom, speed)
        rows.append({
            "speed_kn": knots,
            "speed_ms": speed,
            "Fn": res.fn,
            "Re": res.re,
            "R_T_N": res.rt,
            "eps_1pct_kN": 0.01 * res.rt / 1000.0,
            "eps_2pct_kN": 0.02 * res.rt / 1000.0,
            "P_E_kW": res.pe_kw,
        })
    return {
        "hull": {"name": "KCS (KRISO container ship, public geometry)", "Lpp_m": geom.l,
                 "B_m": geom.b, "T_m": geom.t, "Cb": geom.cb, "Cm": geom.cm, "Cwp": geom.cwp,
                 "LCB_pct": geom.lcb_pct, "nabla_m3": geom.vol},
        "kernel": "tensorphysics.physics_kernels_holtrop1984.holtrop1984_resistance",
        "rows": rows,
        "claim_boundary": "resistance magnitudes and (Fn, Re) ranges only; no EFD comparison is claimed here",
    }


def run_confidence_sweep(config: Mapping[str, Any],
                         variants: Mapping[str, ModelRegistry],
                         truth: TruthGenerator,
                         axes: Mapping[str, AxisSpec],
                         strata: Sequence[StratumSpec],
                         alphas: Sequence[float] = (0.80, 0.90, 0.95),
                         variant_names: Sequence[str] = ("prior_only", "conformal_stratified"),
                         n_queries: Optional[int] = None,
                         seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Trace the guarantee/price trade-off by sweeping the confidence level.

    The same queries are re-scored with ``error_confidence = cost_confidence =
    alpha``: a lower demanded confidence admits cheaper models, and the ex-post
    oracle ratio says how much that costs in wrong (too conservative) decisions.
    """
    from dataclasses import replace as _replace

    acceptance = config["acceptance"]
    budgets = config["budgets"]
    n_queries = int(n_queries or acceptance["n_backtest_queries"])
    seed = int(seed or acceptance["backtest_seed"])
    base = sample_queries(axes, n=n_queries, seed=seed,
                          epsilon_range=tuple(budgets["epsilon_range"]),
                          cost_range=tuple(budgets["cost_range_s"]),
                          error_confidence=float(budgets["error_confidence"]),
                          cost_confidence=float(budgets["cost_confidence"]),
                          interval_level=float(config["validation"]["interval_level"]))
    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        queries = [{"index": q["index"], "params": q["params"],
                    "budget": _replace(q["budget"], error_confidence=float(alpha),
                                       cost_confidence=float(alpha))} for q in base]
        for name in variant_names:
            bt = run_backtest(variants[name], truth, queries,
                              objective=str(acceptance.get("objective", MIN_COST)),
                              label=name, latent_seed=int(acceptance["latent_seed"]),
                              strata=strata)
            m = bt.metrics()
            rows.append({
                "alpha": float(alpha),
                "variant": name,
                "accuracy": m["accuracy"],
                "coverage": m["coverage"],
                "recommendation_rate": m["recommendation_rate"],
                "oracle_optimal_rate": m["oracle_optimal_rate"],
                "median_cost_ratio_vs_oracle": m["median_cost_ratio_vs_oracle"],
                "cost_efficiency": m["cost_efficiency"],
                "mean_bound": m["mean_bound"],
                "recommendation_histogram": m["recommendation_histogram"],
            })
    return rows


def run_verification(config_path: Optional[str | Path] = None,
                     n_queries: Optional[int] = None,
                     seed: Optional[int] = None,
                     pytest_summary: Optional[Mapping[str, Any]] = None) -> VerificationResult:
    config = load_config(config_path)
    acceptance = config["acceptance"]
    budgets = config["budgets"]
    objective = str(acceptance.get("objective", MIN_COST))
    n_queries = int(n_queries or acceptance["n_backtest_queries"])
    seed = int(seed or acceptance["backtest_seed"])

    registry = build_registry(config)
    axes = build_axes(config)
    truths = build_truths(config)
    strata = build_strata(config)
    deployment_truth = truths["deployment"]
    variants, calibration_meta = build_registry_variants(config, registry, deployment_truth)

    queries = sample_queries(
        axes, n=n_queries, seed=seed,
        epsilon_range=tuple(budgets["epsilon_range"]),
        cost_range=tuple(budgets["cost_range_s"]),
        error_confidence=float(budgets["error_confidence"]),
        cost_confidence=float(budgets["cost_confidence"]),
        interval_level=float(config["validation"]["interval_level"]))

    metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for truth_label, truth in truths.items():
        metrics[truth_label] = {}
        records[truth_label] = {}
        for variant_name, reg in variants.items():
            bt = run_backtest(reg, truth, queries, objective=objective, label=variant_name,
                              latent_seed=int(acceptance["latent_seed"]), strata=strata)
            metrics[truth_label][variant_name] = bt.metrics()
            records[truth_label][variant_name] = [r.to_dict() for r in bt.records]

    primary_truth = "deployment"
    primary_variant = "conformal_stratified"
    negative_controls = run_negative_controls(variants[primary_variant], config)
    checks = evaluate_acceptance(metrics[primary_truth][primary_variant], acceptance,
                                negative_controls, pytest_summary)

    probe_params = {"Fn": 0.30, "Cb": 0.72, "Re": 1.5e9}
    pareto = {
        "query": probe_params,
        "frontier": frontier_curve(variants[primary_variant], probe_params),
        "front_models": pareto_front(variants[primary_variant], probe_params),
        "all_models": [
            {"model_id": m.model_id,
             "cost_median_s": m.cost.median_s,
             "cost_upper_s": m.cost.quantile(0.90),
             "error_median": m.error.upper_bound(0.5, probe_params),
             "error_upper": m.error.upper_bound(0.90, probe_params)}
            for m in registry.models if m.domain.admits(probe_params)
        ],
        "budget": {"epsilon": 0.04, "cost": 2.0e4},
    }

    sweep = run_confidence_sweep(config, variants, deployment_truth, axes, strata,
                                n_queries=n_queries, seed=seed)

    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:  # pragma: no cover - git may be absent
            return "unknown"

    commit = _git("rev-parse", "HEAD")
    # the last commit that touched this module: a stable identity, unlike HEAD which
    # advances when the produced artifact is committed afterwards
    source_commit = _git("log", "-1", "--format=%H", "--", "src/tensorphysics/multifidelity")

    artifact: Dict[str, Any] = {
        "node": NODE_ID,
        "module": "tensorphysics.multifidelity",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_source_commit": source_commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config_path": config.get("__path__"),
        "config_sha256": config.get("__digest__"),
        "case": config.get("case", {}),
        "registry": registry.to_dict(),
        "axes": {k: v.to_dict() for k, v in axes.items()},
        "calibration": calibration_meta,
        "truths": {k: v.spec.to_dict() for k, v in truths.items()},
        "protocol": {"n_queries": n_queries, "query_seed": seed,
                     "latent_seed": int(acceptance["latent_seed"]),
                     "objective": objective,
                     "interval_level": float(config["validation"]["interval_level"]),
                     "epsilon_range": list(budgets["epsilon_range"]),
                     "cost_range_s": list(budgets["cost_range_s"]),
                     "error_confidence": float(budgets["error_confidence"]),
                     "cost_confidence": float(budgets["cost_confidence"])},
        "backtests": metrics,
        "primary": {"truth": primary_truth, "variant": primary_variant},
        "negative_controls": negative_controls,
        "acceptance": checks,
        "pareto": pareto,
        "confidence_sweep": sweep,
        "physical_anchor": physical_anchor(),
        # per-query audit trail: the primary truth only (2.7 MB otherwise); the other
        # scenarios keep their metrics, which is what the ablation is read from.
        "records_compact": {
            "deployment": {variant: _compact_records_from(recs)
                           for variant, recs in records["deployment"].items()},
        },
        "records_compact_columns": ["epsilon", "cost_budget", "realised_error", "realised_cost",
                                    "error_upper", "model_index"],
    }
    return VerificationResult(artifact=artifact, records=records, metrics=metrics)


def _compact_records_from(recs: Sequence[Mapping[str, Any]]) -> List[List[Optional[float]]]:
    model_ids = sorted({r["model_id"] for r in recs if r["model_id"] is not None})
    index = {mid: float(i) for i, mid in enumerate(model_ids)}
    rows: List[List[Optional[float]]] = []
    for r in recs:
        rows.append([r["epsilon"], r["cost_budget"], r["realised_error"], r["realised_cost"],
                     r["error_upper"], None if r["model_id"] is None else index[r["model_id"]]])
    return rows
