"""Mission IN57-N07 acceptance harness: math gate vs the six machine criteria.

Runs the six criteria of the mission on evidence that is either manufactured
(F1, F2, F4) or produced by real solver repositories (F3, F5, F6), and writes a
single machine-readable record plus figures::

    artifacts/vv_gate_n07/acceptance.json
    artifacts/vv_gate_n07/figures/*.png

Criteria
--------
F1  positive control ``f(h) = f_exact + C h^2`` on three grids: ``p = 2`` within
    1e-2, ``|f_exact_error| < 1e-6``, GCI decreasing as the refinement ratio
    grows.
F2  negative controls: oscillatory / diverging / stagnant sequences FAIL; 1 %
    random noise never yields a silent PASS (WARN or FAIL only).
F3  real cross-repository run: TensorLBM's own DG-advection manufactured case on
    3+ grids, ``|p - 2| < 0.2``; TensorFEM's ``h_convergence_gci`` evidence is
    adjudicated by the same gate and cross-checked against that repository's
    independent Richardson/GCI implementation.
F4  ASME V&V 20 table: ten ``(f_sim, f_exp, U_num, U_input)`` cases against
    hand-computed ``U_val`` and verdicts, 100 % agreement.
F5  adapters: >= 2 repositories' real benchmark documents parsed by the same
    CLI, with the numbers carried through unchanged (``diff < 1e-12``).
F6  zero intrusion: no solver repository file is modified; one existing repo
    benchmark re-adjudicated with the new gate (comparison table).

Usage::

    python tools/vv_acceptance_n07.py [--skip-coverage]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.vv import adapters  # noqa: E402
from tensorphysics.vv.gci import (  # noqa: E402
    GatePolicy,
    Observation,
    evaluate,
    sweep_refinement_ratio,
)
from tensorphysics.vv.validation import (  # noqa: E402
    evaluate_validation_table,
    validate,
)

ART = ROOT / "artifacts" / "vv_gate_n07"
FIG = ART / "figures"
HS3 = (1.0, 0.5, 0.25)
HS5 = (1.0, 0.5, 0.25, 0.125, 0.0625)
F_EXACT, COEFF = 3.0, 0.5


def quad(h: float) -> float:
    return F_EXACT + COEFF * h * h


def _levels(hs, f) -> list[Observation]:
    return [Observation(h=h, f=f(h), label=f"h={h:g}") for h in hs]


# ---------------------------------------------------------------------------
# F1 - manufactured positive control
# ---------------------------------------------------------------------------


def f1_positive_control() -> dict[str, Any]:
    levels = _levels(HS3, quad)
    verdict = evaluate(levels, quantity="manufactured_f", policy=GatePolicy(expected_order=2.0))
    default_status = verdict.status
    declared = evaluate(
        [Observation(h=l.h, f=l.f, sigma=1e-12) for l in levels],
        quantity="manufactured_f",
        policy=GatePolicy(expected_order=2.0),
    )
    study = verdict.study
    sweep = sweep_refinement_ratio(quad, h_coarse=1.0, p=2.0, ratios=(2.0, 3.0, 4.0, 8.0),
                                   safety_factor=1.25)
    gci_r = [{"r": r, "gci": g} for r, g in sweep]
    gci_decreasing = all(a["gci"] > b["gci"] for a, b in zip(gci_r, gci_r[1:]))
    five = evaluate(_levels(HS5, quad), quantity="manufactured_f",
                    policy=GatePolicy(expected_order=2.0))
    checks = {
        "order_p_vs_2_within_0.01": abs(study.order_p - 2.0) <= 1e-2,
        "f_exact_error_below_1e-6": abs(study.f_extrapolated - F_EXACT) < 1e-6,
        "gci_decreases_with_r": gci_decreasing,
        "five_grid_noise_certified_pass": five.status == "PASS" and five.study.noise_certified,
    }
    return {
        "case": "f(h) = 3 + 0.5 h^2 on h = 1, 0.5, 0.25 (r = 2)",
        "order_p": study.order_p,
        "order_error_vs_theory": abs(study.order_p - 2.0),
        "f_extrapolated": study.f_extrapolated,
        "f_extrapolated_abs_error": abs(study.f_extrapolated - F_EXACT),
        "gci_fine": study.gci_fine,
        "gci_vs_r": gci_r,
        "verdict_default_policy": default_status,
        "verdict_with_declared_sigma": declared.status,
        "five_grid": {
            "order_p": five.study.order_p,
            "fit_residual_rel": five.study.fit_residual_rel,
            "status": five.status,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# F2 - negative controls
# ---------------------------------------------------------------------------


def _noisy(hs, sigma_rel: float, rng: random.Random, declare: bool) -> list[Observation]:
    out = []
    for h in hs:
        base = quad(h)
        noise = base * (1.0 + sigma_rel * rng.uniform(-1.0, 1.0))
        out.append(Observation(h=h, f=noise, sigma=abs(base) * sigma_rel if declare else None))
    return out


def f2_negative_controls(trials: int = 30) -> dict[str, Any]:
    seq = {
        "oscillatory": (1.0, 1.3, 1.15),
        "diverging": (1.0, 1.2, 1.6),
        "stagnant": (1.0, 1.2, 1.4),
    }
    structural: dict[str, Any] = {}
    for name, values in seq.items():
        verdict = evaluate([Observation(h=h, f=v) for h, v in zip(HS3, values)])
        structural[name] = {
            "convergence_type": verdict.study.convergence_type,
            "status": verdict.status,
            "reasons": list(verdict.reasons)[:2],
        }
    noise_cases = {
        "three_grid_declared_sigma_1pct": {"hs": HS3, "declare": True},
        "three_grid_undeclared_sigma_1pct": {"hs": HS3, "declare": False},
        "five_grid_undeclared_sigma_1pct": {"hs": HS5, "declare": False},
    }
    noise: dict[str, Any] = {}
    for name, cfg in noise_cases.items():
        statuses: dict[str, int] = {"PASS": 0, "WARN": 0, "FAIL": 0}
        legacy_pass = 0
        orders: list[float] = []
        for seed in range(trials):
            rng = random.Random(seed)
            levels = _noisy(cfg["hs"], 0.01, rng, cfg["declare"])
            verdict = evaluate(levels, quantity="q", policy=GatePolicy(expected_order=2.0))
            statuses[verdict.status] += 1
            if verdict.study.order_p is not None:
                orders.append(verdict.study.order_p)
            legacy = evaluate(levels, quantity="q",
                              policy=GatePolicy(expected_order=2.0,
                                                pass_requires_noise_certificate=False))
            if legacy.status == "PASS":
                legacy_pass += 1
        noise[name] = {
            "trials": trials,
            "status_counts": statuses,
            "silent_pass": statuses["PASS"],
            "legacy_policy_pass": legacy_pass,
            "order_p_range": [min(orders), max(orders)] if orders else None,
        }
    checks = {
        "structural_non_asymptotic_fail": all(v["status"] == "FAIL" for v in structural.values()),
        "noise_never_silently_passes": all(v["silent_pass"] == 0 for v in noise.values()),
    }
    return {
        "structural": structural,
        "noise": noise,
        "note": (
            "the legacy column counts PASSes that the pre-hardening policy "
            "(pass_requires_noise_certificate=False) would have emitted: the "
            "three-grid undeclared-noise case is exactly the silent PASS that "
            "the default policy now caps at WARN"
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# F3 - real cross-repository runs
# ---------------------------------------------------------------------------


def _gate_record(native: Path, *, repo: str, expected_order: float | None = None,
                 value_key: str | None = None, study: str | None = None,
                 extra: tuple[str, ...] = ()) -> dict[str, Any]:
    """Run the shipped CLI in a subprocess and return its JSON record."""
    cmd = [sys.executable, "-m", "tensorphysics.vv.gci", "--benchmark", str(native),
           "--repo", repo, "--format", "json"]
    if expected_order is not None:
        cmd += ["--expected-order", str(expected_order)]
    if value_key:
        cmd += ["--value-key", value_key]
    if study:
        cmd += ["--study", study]
    cmd += list(extra)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                          env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    record = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else {}
    record["_exit_code"] = proc.returncode
    record["_cmd"] = " ".join(cmd[1:])
    if proc.returncode == 64:
        record["_stderr"] = proc.stderr.strip()
    return record


def f3_crossrepo() -> dict[str, Any]:
    lbm_native = ART / "tensorlbm_dgmms" / "tensorlbm_dg_mms_native.json"
    fem_evidence = ART / "tensorfem_hconv" / "evidence.json"
    out: dict[str, Any] = {"runs": {}}
    checks: dict[str, bool] = {}

    if lbm_native.exists():
        rec = _gate_record(lbm_native, repo="tensorlbm", expected_order=2.0)
        rec_abs = _gate_record(lbm_native, repo="tensorlbm", expected_order=2.0,
                               extra=("--gci-absolute",))
        study = rec["study"]
        p = study["order_p"]
        checks["tensorlbm_mms_p_within_0.2_of_2"] = abs(p - 2.0) < 0.2
        checks["tensorlbm_mms_gci_monotone"] = bool(study["gci_monotone"])
        checks["tensorlbm_mms_levels_ge_3"] = study["n_levels"] >= 3
        out["runs"]["tensorlbm_dg_mms"] = {
            "case": "nodal P1 DG advection of sin(x), one period, CFL 0.2, SSP-RK3",
            "levels": study["levels"],
            "order_p": p,
            "order_error_vs_theory": abs(p - 2.0),
            "f_extrapolated": study["f_extrapolated"],
            "gci_fine": study["gci_fine"],
            "convergence_ratio": study["convergence_ratio"],
            "gci_pairs": study["gci_pairs"],
            "gci_fine_absolute": rec_abs["study"]["gci_fine"],
            "gci_absolute_note": (
                "an error norm converges to zero, so its relative GCI is unbounded "
                "by construction; the absolute band is the meaningful one"
            ),
            "status": rec["status"],
            "exit_code": rec["_exit_code"],
            "cmd": rec["_cmd"],
        }
    else:  # pragma: no cover - artifact generation is a prerequisite
        checks["tensorlbm_mms_p_within_0.2_of_2"] = False
        out["runs"]["tensorlbm_dg_mms"] = {"error": f"missing {lbm_native}"}

    if fem_evidence.exists():
        fem = json.loads(fem_evidence.read_text())
        rec = _gate_record(fem_evidence, repo="tensorfem", expected_order=2.0,
                           value_key="strain_energy", study="manufactured")
        rec_err = _gate_record(fem_evidence, repo="tensorfem", expected_order=2.0,
                               value_key="relative_strain_energy_error", study="manufactured")
        study = rec["study"]
        energy = fem["manufactured"]["energy_study"]
        their_order = energy.get("finest_order", energy.get("mean_observed_order"))
        their_gci = (energy.get("gci") or {}).get("relative_gci")
        order_diff = None if their_order is None else abs(study["order_p"] - their_order)
        gci_diff = None if their_gci is None else abs(study["gci_fine"] - their_gci)
        checks["tensorfem_p_within_0.2_of_2"] = abs(study["order_p"] - 2.0) < 0.2
        checks["tensorfem_independent_implementation_agrees"] = (
            order_diff is not None and order_diff < 1e-9 and gci_diff is not None and gci_diff < 1e-9
        )
        out["runs"]["tensorfem_manufactured"] = {
            "case": "HEX8 manufactured elasticity, relative strain-energy error",
            "levels": study["levels"],
            "order_p": study["order_p"],
            "order_p_error_norm_observable": rec_err["study"]["order_p"],
            "tensorfem_own_order": their_order,
            "order_diff_vs_tensorfem": order_diff,
            "gci_fine": study["gci_fine"],
            "gci_is_relative": study["gci_is_relative"],
            "tensorfem_own_gci": their_gci,
            "gci_diff_vs_tensorfem": gci_diff,
            "status": rec["status"],
            "exit_code": rec["_exit_code"],
        }
    else:  # pragma: no cover
        checks["tensorfem_p_within_0.2_of_2"] = False
        out["runs"]["tensorfem_manufactured"] = {"error": f"missing {fem_evidence}"}

    out["checks"] = checks
    out["passed"] = all(checks.values()) if checks else False
    return out


# ---------------------------------------------------------------------------
# F4 - ASME V&V 20 table (hand-computed expectations)
# ---------------------------------------------------------------------------

#: label, f_sim, f_exp, U_num, U_input, hand U_val, hand verdict
VV20_ROWS = [
    ("pass_inside_band", 1.0, 1.02, 0.02, 0.01, 0.022360679774997898, "PASS"),
    ("fail_outside_band", 1.0, 1.05, 0.03, 0.03, 0.042426406871192854, "FAIL"),
    ("boundary_exact", 1.05, 1.0, 0.03, 0.04, 0.05, "PASS"),
    ("degenerate_zero", 0.0, 0.0, 0.0, 0.0, 0.0, "PASS"),
    ("large_error", 2.0, 1.0, 0.5, 0.5, 0.7071067811865476, "FAIL"),
    ("boundary_symmetric", 100.0, 100.5, 0.3, 0.4, 0.5, "PASS"),
    ("boundary_integers", 10.0, 9.0, 0.6, 0.8, 1.0, "PASS"),
    ("just_outside_band", 10.0, 9.0, 0.6, 0.79, 0.9920181449953424, "FAIL"),
    ("tiny_band_zero_error", 1.0, 1.0, 0.0, 1e-9, 1e-9, "PASS"),
    ("negative_values", -3.0, -3.2, 0.1, 0.1, 0.14142135623730953, "FAIL"),
]


def f4_validation_table() -> dict[str, Any]:
    rows = [{"label": r[0], "f_sim": r[1], "f_exp": r[2], "u_num": r[3], "u_input": r[4]}
            for r in VV20_ROWS]
    results = evaluate_validation_table(rows)
    detail = []
    agree = 0
    for (label, f_sim, f_exp, u_num, u_input, u_hand, verdict_hand), res in zip(VV20_ROWS, results):
        u_ok = abs(res.u_val - u_hand) <= 1e-12 * max(1.0, abs(u_hand))
        v_ok = res.status == verdict_hand
        agree += int(u_ok and v_ok)
        detail.append({
            "label": label, "f_sim": f_sim, "f_exp": f_exp, "u_num": u_num, "u_input": u_input,
            "u_val_gate": res.u_val, "u_val_hand": u_hand, "u_val_match": u_ok,
            "error": res.error, "ratio": res.ratio,
            "status_gate": res.status, "status_hand": verdict_hand, "status_match": v_ok,
        })
    # the gate's own validation path used from the CLI table mode
    checks = {"ten_of_ten_agree": agree == len(VV20_ROWS)}
    return {"rows": detail, "n_agree": agree, "n_rows": len(VV20_ROWS),
            "checks": checks, "passed": all(checks.values())}


# ---------------------------------------------------------------------------
# F5 - cross-repository adapters on real documents
# ---------------------------------------------------------------------------


def _native_value(item: dict[str, Any], keys) -> float | None:
    for k in keys:
        if k in item and isinstance(item[k], (int, float)):
            return float(item[k])
    return None


def f5_adapters() -> dict[str, Any]:
    docs = {
        "tensorlbm": {
            "path": ART / "tensorlbm_dgmms" / "tensorlbm_dg_mms_native.json",
            "kwargs": {},
            "value_keys": ("value", "l2_error"),
            "length_keys": ("h", "dx", "cell_size", "resolution"),
        },
        "tensorfem": {
            "path": ART / "tensorfem_hconv" / "evidence.json",
            "kwargs": {"study": "manufactured", "value_key": "relative_strain_energy_error"},
            "value_keys": ("relative_strain_energy_error",),
            "length_keys": ("mesh_size",),
        },
        "tensorbem": {
            "path": Path("/opt/mm-verify/TensorBEM/docs/hydrostatic_verification.json"),
            "kwargs": {"study": "hemisphere_convergence", "value_key": "volume_relative_error"},
            "value_keys": ("volume_relative_error",),
            "length_keys": ("panel_size",),
        },
    }
    out: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for repo, cfg in docs.items():
        path: Path = cfg["path"]
        if not path.exists():  # pragma: no cover - optional repo checkout
            out[repo] = {"skipped": f"missing {path}"}
            continue
        payload = json.loads(path.read_text())
        series = adapters.extract_series(payload, repo=repo, source_path=str(path), **cfg["kwargs"])
        # field conservation: adapted value == native value, exactly (0 diff here)
        max_diff = 0.0
        compared = 0
        for adapted, raw in zip(series.levels, series.raw_levels):
            native_f = _native_value(raw, cfg["value_keys"])
            if native_f is None:  # pragma: no cover
                continue
            max_diff = max(max_diff, abs(adapted.f - native_f))
            compared += 1
        rec = _gate_record(path, repo=repo, value_key=cfg["kwargs"].get("value_key"),
                           study=cfg["kwargs"].get("study"))
        finest = rec.get("study", {}).get("f_fine")
        finest_diff = None if finest is None else abs(finest - series.levels[-1].f)
        out[repo] = {
            "source": str(path).replace(str(ROOT) + "/", ""),
            "study": series.study,
            "n_levels": len(series.levels),
            "quantity": series.quantity,
            "levels": [lvl.as_dict() for lvl in series.levels],
            "levels_compared": compared,
            "field_conservation_max_abs_diff": max_diff,
            "cli_finest_value_diff": finest_diff,
            "cli_exit_code": rec["_exit_code"],
            "cli_order_p": rec.get("study", {}).get("order_p"),
            "cli_status": rec.get("status"),
        }
        checks[f"{repo}_parsed"] = compared >= 2
        checks[f"{repo}_field_conservation_1e-12"] = max_diff < 1e-12
    repositories = [k for k in ("tensorlbm", "tensorfem", "tensorbem") if checks.get(f"{k}_parsed")]
    checks["at_least_two_repositories"] = len(repositories) >= 2
    checks["at_least_three_profiles_registered"] = len(adapters.REPO_PROFILES) >= 3
    return {
        "repositories_parsed": repositories,
        "registered_profiles": sorted(adapters.REPO_PROFILES),
        "documents": out,
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# F6 - zero intrusion + re-adjudicated existing benchmark
# ---------------------------------------------------------------------------


def f6_zero_intrusion() -> dict[str, Any]:
    repos = {
        "TensorLBM": Path("/opt/mm-verify/TensorLBM"),
        "TensorFEM": Path("/opt/mm-verify/TensorFEM"),
        "TensorBEM": Path("/opt/mm-verify/TensorBEM"),
    }
    clean: dict[str, Any] = {}
    for name, path in repos.items():
        if not path.exists():  # pragma: no cover
            clean[name] = {"present": False}
            continue
        proc = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                              capture_output=True, text=True)
        dirty = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        clean[name] = {"present": True, "dirty_entries": len(dirty), "sample": dirty[:3]}

    # an existing benchmark re-run through the new gate: TensorLBM Poiseuille
    poiseuille = {}
    for observable in ("rel_l2_error", "flow_rate_error", "peak_velocity_error"):
        path = ROOT / "artifacts" / "vv_gate" / f"obs_tensorlbm_poiseuille_{observable}.json"
        if not path.exists():  # pragma: no cover
            continue
        rec = _gate_record(path, repo="tensorphysics", expected_order=2.0)
        study = rec.get("study", {})
        p = study.get("order_p")
        hs = [l["h"] for l in study.get("levels", [])]  # coarse -> fine
        ratio = (hs[0] / hs[-1]) ** (1.0 / max(1, len(hs) - 1)) if hs else None
        poiseuille[observable] = {
            "levels": study.get("levels"),
            "order_p": p,
            "deviation_from_2": None if p is None else abs(p - 2.0),
            "gci_fine": study.get("gci_fine"),
            "mean_refinement_ratio": ratio,
            "repo_own_criterion": "refinement ratio > 1.5 (i.e. p > 0.58) is accepted",
            "repo_own_verdict": None if ratio is None else ("PASS" if ratio > 1.5 else "FAIL"),
            "gate_verdict_vs_theory_2": rec.get("status"),
            "gate_exit_code": rec["_exit_code"],
        }
    checks = {
        "solver_repositories_clean": all(v.get("dirty_entries", 0) == 0
                                         for v in clean.values() if v.get("present")),
        "existing_benchmark_readjudicated": len(poiseuille) >= 3,
        "new_gate_is_stricter_than_repo_criterion": all(
            v["repo_own_verdict"] == "PASS" and v["order_p"] is not None and abs(v["order_p"] - 2.0) > 0.2
            for v in poiseuille.values()
        ) if poiseuille else False,
    }
    return {"repositories": clean, "poiseuille": poiseuille,
            "checks": checks, "passed": all(checks.values())}


# ---------------------------------------------------------------------------
# gate payloads consumed by the test-suite and by the CLI examples
# ---------------------------------------------------------------------------


def emit_payloads() -> dict[str, str]:
    """Write the observation payloads that the tests and the CLI examples read.

    ``artifacts/vv_gate/obs_tensorfem_manufactured.json``  -- strain-energy QoI of
    TensorFEM's manufactured study with a declared round-off sigma (float64,
    1e-9 relative), so the noise floor is certified and the gate can PASS;
    ``artifacts/vv_gate_n07/tensorlbm_dgmms/obs_tensorlbm_dg_mms.json`` -- the
    adapted TensorLBM DG-MMS series with the same round-off declaration.
    """
    written: dict[str, str] = {}
    fem_evidence = ART / "tensorfem_hconv" / "evidence.json"
    if fem_evidence.exists():
        payload = {
            "schema": "tensorphysics.vv.observations/1",
            "quantity": "relative_strain_energy_error",
            "expected_order": 2.0,
            "source": {"repo": "tensorfem", "path": str(fem_evidence),
                       "study": "manufactured", "qoi": "strain_energy"},
            "levels": [
                {"h": lvl["mesh_size"], "f": lvl["relative_strain_energy_error"],
                 "label": lvl.get("label", ""),
                 "sigma": 1e-9 * abs(lvl["strain_energy"] / lvl["exact_strain_energy"])}
                for lvl in json.loads(fem_evidence.read_text())["manufactured"]["levels"]
            ],
        }
        out = ROOT / "artifacts" / "vv_gate" / "obs_tensorfem_manufactured.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written["tensorfem_manufactured"] = str(out.relative_to(ROOT))

    lbm_native = ART / "tensorlbm_dgmms" / "tensorlbm_dg_mms_native.json"
    if lbm_native.exists():
        series = adapters.extract_series(json.loads(lbm_native.read_text()), repo="tensorlbm",
                                         source_path=str(lbm_native))
        payload = adapters.to_gate_payload(
            adapters.AdapterSeries(
                repo=series.repo, quantity="l2_error",
                levels=tuple(Observation(h=l.h, f=l.f, label=l.label,
                                         sigma=1e-9 * abs(l.f)) for l in series.levels),
                source_path=series.source_path, study=series.study,
                metadata=series.metadata, raw_levels=series.raw_levels,
            ),
            expected_order=2.0,
        )
        out = ART / "tensorlbm_dgmms" / "obs_tensorlbm_dg_mms.json"
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written["tensorlbm_dg_mms"] = str(out.relative_to(ROOT))
    return written


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(record: dict[str, Any]) -> list[str]:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    FIG.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1. positive control + GCI vs r
    f1 = record["criteria"]["F1_positive_control"]
    hs = [1.0, 0.5, 0.25]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    ax = axes[0]
    ax.plot(hs, [quad(h) for h in hs], "o-", color="#1f4e79", label="discrete levels")
    ax.axhline(f1["f_extrapolated"], color="#c00000", ls="--",
               label=f"Richardson f_exact={f1['f_extrapolated']:.6f}")
    ax.set_xscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("f")
    ax.set_title(f"positive control: p={f1['order_p']:.6f} (theory 2)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax = axes[1]
    rs = [g["r"] for g in f1["gci_vs_r"]]
    gs = [g["gci"] for g in f1["gci_vs_r"]]
    ax.plot(rs, gs, "s-", color="#2e7d32")
    ax.set_xlabel("refinement ratio r")
    ax.set_ylabel("GCI_fine")
    ax.set_yscale("log")
    ax.set_title("GCI decreases as the refinement ratio grows")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = FIG / "fig_f1_positive_control.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p.relative_to(ROOT)))

    # 2. negative controls: noise + structural sequences
    f2 = record["criteria"]["F2_negative_controls"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    ax = axes[0]
    rng = random.Random(3)
    noisy = _noisy(HS3, 0.01, rng, declare=True)
    ax.plot(HS3, [quad(h) for h in HS3], "-", color="k", lw=1.0, label="exact power law")
    ax.errorbar([l.h for l in noisy], [l.f for l in noisy],
                yerr=[l.sigma for l in noisy], fmt="o", color="#c00000",
                label="1 % noise, declared sigma")
    ax.set_xscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("f")
    ax.set_title("noise case never passes silently "
                 f"(default policy: {f2['noise']['three_grid_declared_sigma_1pct']['status_counts']})",
                 fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax = axes[1]
    labels, counts = [], []
    for name, cfg in f2["noise"].items():
        sc = cfg["status_counts"]
        labels.append(name.replace("_", "\n"))
        counts.append((sc["PASS"], sc["WARN"], sc["FAIL"]))
    import numpy as np  # noqa: PLC0415

    arr = np.array(counts)
    bottom = np.zeros(len(labels))
    for i, (label, color) in enumerate((("PASS", "#c62828"), ("WARN", "#f9a825"), ("FAIL", "#2e7d32"))):
        ax.bar(labels, arr[:, i], bottom=bottom, label=label, color=color)
        bottom += arr[:, i]
    ax.set_ylabel(f"verdicts over {f2['noise']['three_grid_declared_sigma_1pct']['trials']} seeds")
    ax.set_title("1 % noise: PASS count must be 0", fontsize=9)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "fig_f2_negative_controls.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p.relative_to(ROOT)))

    # 3. cross-repo observed order vs theory
    f3 = record["criteria"]["F3_crossrepo_real_runs"]
    names, orders = [], []
    for key, run in f3["runs"].items():
        if "order_p" in run:
            names.append(key.replace("_", "\n"))
            orders.append(run["order_p"])
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.bar(names, orders, color="#2e75b6")
    ax.axhline(2.0, color="k", ls="--", label="theory p = 2")
    ax.axhspan(1.8, 2.2, color="#2e7d32", alpha=0.15, label="acceptance |p-2| < 0.2")
    for i, o in enumerate(orders):
        ax.text(i, o + 0.02, f"{o:.4f}", ha="center", fontsize=8)
    ax.set_ylabel("observed order p")
    ax.set_title("real cross-repository MMS runs through the gate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "fig_f3_crossrepo_orders.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p.relative_to(ROOT)))

    # 4. V&V 20 table ratios
    f4 = record["criteria"]["F4_vv20_table"]
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    labels = [r["label"] for r in f4["rows"]]
    ratios = [r["ratio"] for r in f4["rows"]]
    colors = ["#2e7d32" if r["status_gate"] == "PASS" else "#c62828" for r in f4["rows"]]
    ax.bar(labels, ratios, color=colors)
    ax.axhline(1.0, color="k", ls="--", label="|E|/U_val = 1")
    ax.set_ylabel("|E| / U_val")
    ax.set_title(f"ASME V&V 20 table: {f4['n_agree']}/{f4['n_rows']} agree with the hand computation")
    ax.tick_params(axis="x", labelsize=6, rotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "fig_f4_vv20_table.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p.relative_to(ROOT)))

    # 5. TensorLBM Poiseuille: repo criterion vs gate
    f6 = record["criteria"]["F6_zero_intrusion"]
    if f6["poiseuille"]:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        obs = list(f6["poiseuille"])
        vals = [f6["poiseuille"][o]["order_p"] for o in obs]
        ax.bar([o.replace("_error", "") for o in obs], vals, color="#8e24aa")
        ax.axhline(2.0, color="k", ls="--", label="theory p = 2 (Poiseuille)")
        ax.axhline(0.58, color="#c62828", ls=":", label="repo test threshold p = 0.58")
        ax.set_ylabel("observed order p")
        ax.set_title("existing TensorLBM benchmark re-adjudicated by the new gate")
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = FIG / "fig_f5_lbm_poiseuille_readjudication.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p.relative_to(ROOT)))
    return written


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_coverage() -> dict[str, Any] | None:
    cov_json = ART / "coverage.json"
    cmd = [sys.executable, "-m", "pytest",
           "tests/test_vv_gci_core.py", "tests/test_vv_validation.py",
           "tests/test_vv_adapters_cli.py", "tests/test_vv_cli_formats.py",
           "--cov=src/tensorphysics/vv", f"--cov-report=json:{cov_json}", "-q"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if not cov_json.exists():  # pragma: no cover
        return {"error": proc.stdout[-2000:] + proc.stderr[-2000:]}
    data = json.loads(cov_json.read_text())
    return {
        "percent_covered": data["totals"]["percent_covered"],
        "files": {Path(k).name: v["summary"]["percent_covered"]
                  for k, v in data["files"].items()},
        "tests_returncode": proc.returncode,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-coverage", action="store_true")
    ap.add_argument("--trials", type=int, default=30,
                    help="random-noise seeds per noise configuration")
    args = ap.parse_args(argv)
    ART.mkdir(parents=True, exist_ok=True)
    payloads = emit_payloads()

    criteria = {
        "F1_positive_control": f1_positive_control(),
        "F2_negative_controls": f2_negative_controls(trials=args.trials),
        "F3_crossrepo_real_runs": f3_crossrepo(),
        "F4_vv20_table": f4_validation_table(),
        "F5_crossrepo_adapters": f5_adapters(),
        "F6_zero_intrusion": f6_zero_intrusion(),
    }
    record: dict[str, Any] = {
        "mission": "IN57-N07",
        "module": "tensorphysics.vv",
        "criteria": criteria,
        "passed": {k: bool(v.get("passed")) for k, v in criteria.items()},
    }
    record["all_pass"] = all(record["passed"].values())
    record["payloads"] = payloads
    record["figures"] = make_figures(record)
    if not args.skip_coverage:
        record["coverage"] = run_coverage()
    (ART / "acceptance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    for key, ok in record["passed"].items():
        print(f"{key:<26} {'PASS' if ok else 'FAIL'}")
    print(f"all_pass = {record['all_pass']}")
    if "coverage" in record and "percent_covered" in record["coverage"]:
        print(f"coverage  = {record['coverage']['percent_covered']:.2f} %")
    print(f"record    = {ART / 'acceptance.json'}")
    return 0 if record["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())