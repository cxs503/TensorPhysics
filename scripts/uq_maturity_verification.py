"""P142W3-N10 verification: F1-F7 numbers, figures and the evidence JSON.

Run from the repository root::

    python scripts/uq_maturity_verification.py

Outputs (all machine-generated, nothing hand-typed):

``artifacts/uq_maturity/p142w3_n10_verification.json``
    acceptance items F1-F7 with the numbers that decide them
``artifacts/uq_maturity/fig*.png``
    index comparison, Holton convergence, fail-safe degradation, budget stack

The script is deliberately *additive*: it imports the package under test and
never re-implements a formula, so a defect in the library shows up as a failed
acceptance item instead of being masked by a parallel implementation here.
"""
from __future__ import annotations

import ast
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.uq import (  # noqa: E402
    DEFAULT_Q2_MIN,
    Marginal,
    PCESurrogate,
    SampleSpec,
    assess_failsafe,
    combine_budget,
    fit_pce,
    gate_with_benchmark,
    holton_analytic,
    ishigami,
    ishigami_analytic,
    ishigami_spec,
    loo_q2,
    make_holton,
    parse_benchmarks,
    run_uq_study,
    run_with_failsafe,
    saltelli_indices,
    sample,
    sobol_g,
    sobol_g_analytic,
    sobol_g_spec,
)

OUT = ROOT / "artifacts" / "uq_maturity"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "cross_repo_benchmarks"

START = time.time()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# ---------------------------------------------------------------------------
# F1 - Ishigami anchor
# ---------------------------------------------------------------------------
def f1_ishigami() -> dict[str, Any]:
    spec = ishigami_spec()
    analytic = ishigami_analytic()
    study = run_uq_study(
        ishigami, spec, n_samples=512, degree=8, seed=20240,
        run_mc=True, n_base_mc=1 << 15, require_cross_check=True,
    )
    pce = study.pce_sobol
    mc = study.mc
    labels = ["S_1", "S_2", "S_3", "S_T1", "S_T2", "S_T3"]
    ref = np.concatenate([analytic["first"], analytic["total"]])
    pce_vals = np.concatenate([pce.first, pce.total])
    mc_vals = np.concatenate([mc.first, mc.total])
    pce_err = np.abs(pce_vals - ref)
    mc_err = np.abs(mc_vals - ref)
    # seed spread of the sampled route: it is a statistical estimator, so the
    # 2 % acceptance is applied to the (deterministic) surrogate indices that the
    # tool reports, while the sampled cross-check is held to the 5 % statistical
    # tolerance used for sampled estimates elsewhere (F2).
    def _evaluate_mc(unit: np.ndarray) -> np.ndarray:
        phys, _ = spec.transform(unit)
        return ishigami(phys)

    seed_spread = []
    for sd in (20240, 20241, 20242):
        rep = saltelli_indices(_evaluate_mc, 3, n_base=1 << 13, seed=sd)
        err = float(np.max(np.abs(np.concatenate([rep.first, rep.total]) - ref)))
        seed_spread.append({"seed": sd, "max_abs_error": err})
    return {
        "criterion": "S_1,S_2,S_3,S_T1,S_T2,S_T3 within 2% of the closed form (reported "
                     "surrogate indices); sampled cross-check within 5%",
        "tolerance": 0.02,
        "tolerance_sampled": 0.05,
        "sampled_seed_spread_n_base_8192": seed_spread,
        "analytic": dict(zip(labels, ref.tolist())),
        "pce": dict(zip(labels, pce_vals.tolist())),
        "mc": dict(zip(labels, mc_vals.tolist())),
        "pce_abs_error": dict(zip(labels, pce_err.tolist())),
        "mc_abs_error": dict(zip(labels, mc_err.tolist())),
        "max_abs_error_pce": float(pce_err.max()),
        "max_abs_error_mc": float(mc_err.max()),
        "pass": bool(pce_err.max() <= 0.02 and mc_err.max() <= 0.05),
        "study_status": study.status,
        "cross_check_agreement": bool(study.cross_check.agreement) if study.cross_check else None,
        "n_base_mc": int(study.mc.n_base) if study.mc is not None else None,
        "n_model_evaluations": int(study.metadata["n_model_evaluations"]),
    }


# ---------------------------------------------------------------------------
# F2 - Holton anchor
# ---------------------------------------------------------------------------
def f2_holton() -> dict[str, Any]:
    dim = 8
    model = make_holton(dim)
    spec = model.spec()
    analytic = model.analytic()

    design = sample(spec, 512, rule="sobol", seed=31)
    points, coords = spec.from_latent(design.latent)
    surrogate = fit_pce(coords, model(points), degree=2, families=spec.families)
    pce = surrogate.sobol()
    pce_err = float(
        max(np.max(np.abs(pce.first - analytic["first"])),
            np.max(np.abs(pce.total - analytic["total"])))
    )

    def evaluate(unit: np.ndarray) -> np.ndarray:
        phys, _ = spec.transform(unit)
        return model(phys)

    convergence = []
    for n_base in (512, 1024, 2048, 4096, 8192):
        mc = saltelli_indices(evaluate, dim, n_base=n_base, seed=20240)
        first_err = float(np.max(np.abs(mc.first - analytic["first"])))
        total_err = float(np.max(np.abs(mc.total - analytic["total"])))
        convergence.append(
            {
                "n_base": n_base,
                "n_model_evaluations": int(mc.n_model_evaluations),
                "max_first_abs_error": first_err,
                "max_total_abs_error": total_err,
                "max_abs_error": max(first_err, total_err),
            }
        )
    worst = convergence[-1]["max_abs_error"]
    return {
        "criterion": "high-dimensional sampled indices within 5% of the closed form",
        "tolerance": 0.05,
        "dim": dim,
        "analytic_first": analytic["first"].tolist(),
        "analytic_total": analytic["total"].tolist(),
        "interaction_variance_share": float(analytic["interaction_variance_share"]),
        "total_index_sum": float(analytic["total_sum"]),
        "pce_degree2_max_abs_error": pce_err,
        "pce_exact": bool(pce_err < 1e-6),
        "convergence": convergence,
        "max_abs_error_at_finest": float(worst),
        "pass": bool(worst <= 0.05 and pce_err < 1e-6),
    }


# ---------------------------------------------------------------------------
# F3 - surrogate quality (Q^2)
# ---------------------------------------------------------------------------
def f3_pce_quality(f1: dict[str, Any]) -> dict[str, Any]:
    spec = ishigami_spec()
    design = sample(spec, 512, rule="sobol", seed=20240)
    points, coords = spec.from_latent(design.latent)
    surrogate = fit_pce(coords, ishigami(points), degree=8, families=spec.families)
    q2_ishigami = loo_q2(surrogate)

    holton = make_holton(8)
    hspec = holton.spec()
    hdesign = sample(hspec, 512, rule="sobol", seed=31)
    hpoints, hcoords = hspec.from_latent(hdesign.latent)
    hsur = fit_pce(hcoords, holton(hpoints), degree=2, families=hspec.families)
    q2_holton = loo_q2(hsur)

    gobs = sobol_g_spec(4)
    a = [0.0, 0.5, 1.5, 4.0]
    gdesign = sample(gobs, 512, rule="sobol", seed=41)
    gpoints, gcoords = gobs.from_latent(gdesign.latent)
    gsur = fit_pce(gcoords, sobol_g(gpoints, a=a), degree=3, families=gobs.families)
    q2_g = loo_q2(gsur)

    # negative control: the Sobol' G function has a kink at x = 1/2, so a
    # low-degree chaos expansion cannot reach Q^2 = 0.95 even when the LOO
    # score is far from catastrophic (0.80).  The gate must therefore WITHHOLD
    # (UQ-W03-LOO) rather than certify an index set it cannot support.
    g_study = run_uq_study(
        lambda p: sobol_g(p, a=[0.0, 0.5, 1.5, 4.0]), gobs, n_samples=512,
        degree=3, seed=41, run_mc=False,
    )
    values = {"ishigami": q2_ishigami, "holton": q2_holton}
    return {
        "criterion": "leave-one-out Q^2 >= 0.95 on the smooth anchors; a non-smooth "
                     "QoI must fall below the floor and trigger the fail-safe",
        "tolerance": 0.95,
        "q2": values,
        "eps_loo": {"ishigami": float(surrogate.eps_loo), "holton": float(hsur.eps_loo)},
        "negative_control": {
            "case": "sobol G-function, degree 3, kinked at x = 1/2",
            "q2": q2_g,
            "eps_loo": float(gsur.eps_loo),
            "gate_status": g_study.status,
            "gate_codes": list(g_study.verdict.reason_codes),
        },
        "pass": bool(
            all(v is not None and v >= 0.95 for v in values.values())
            and q2_g is not None
            and q2_g < 0.95
            and g_study.status == "WITHHELD"
            and "UQ-W03-LOO" in g_study.verdict.reason_codes
        ),
    }


# ---------------------------------------------------------------------------
# F4 - fail-safe degradation
# ---------------------------------------------------------------------------
def f4_failsafe() -> dict[str, Any]:
    analytic = ishigami_analytic()
    # (a) starved design: 10 runs cannot support a degree-8 truncation
    starved = run_with_failsafe(
        ishigami, ishigami_spec(), n_samples=10, degree=8, seed=7,
        failsafe_n_base=4096, failsafe_budget_n_mc=4096,
    )
    first_err = float(np.max(np.abs(starved.indices.first - analytic["first"])))
    total_err = float(np.max(np.abs(starved.indices.total - analytic["total"])))
    starved_err = max(first_err, total_err)

    # (b) low-quality surrogate on a design that is large enough
    def rough(points: np.ndarray) -> np.ndarray:
        x = np.asarray(points)
        return np.sin(12.0 * x[:, 0]) + 0.5 * np.cos(17.0 * x[:, 1]) + 0.25 * x[:, 2]

    rough_spec = SampleSpec(
        (
            Marginal.uniform("x1", -1.0, 1.0),
            Marginal.uniform("x2", -1.0, 1.0),
            Marginal.uniform("x3", -1.0, 1.0),
        )
    )
    low = run_with_failsafe(
        rough, rough_spec, n_samples=64, degree=2, seed=5,
        failsafe_n_base=4096, failsafe_budget_n_mc=2048,
    )
    return {
        "criterion": "automatic switch to quasi-MC when Q^2 < 0.5 / samples are short; "
        "degraded indices within 10% of the closed form",
        "tolerance": 0.10,
        "q2_floor": DEFAULT_Q2_MIN,
        "starved_design": {
            "n_samples": 10,
            "mode": starved.mode,
            "triggered": starved.decision.triggered,
            "reasons": list(starved.decision.reasons),
            "q2": starved.decision.q2,
            "max_index_abs_error": starved_err,
            "status": starved.status,
            "n_mc_evaluations": starved.metadata["n_mc_evaluations"],
            "index_source": "montecarlo" if starved.mode == "montecarlo" else "pce",
        },
        "low_quality_surrogate": {
            "n_samples": 64,
            "mode": low.mode,
            "triggered": low.decision.triggered,
            "q2": low.decision.q2,
            "reasons": list(low.decision.reasons),
            "status": low.status,
        },
        "pass": bool(
            starved.mode == "montecarlo"
            and starved_err <= 0.10
            and low.mode == "montecarlo"
            and low.decision.q2 is not None
            and low.decision.q2 < DEFAULT_Q2_MIN
        ),
    }


# ---------------------------------------------------------------------------
# F5 - maturity gate: coverage, docs, CLI modes
# ---------------------------------------------------------------------------
def f5_maturity_gate(f7: dict[str, Any]) -> dict[str, Any]:
    cov_path = OUT / "coverage.json"
    coverage = None
    if cov_path.is_file():
        payload = json.loads(cov_path.read_text(encoding="utf-8"))
        totals = payload.get("totals", {})
        coverage = {
            "percent_covered": float(totals.get("percent_covered", 0.0)),
            "num_statements": int(totals.get("num_statements", 0)),
            "missing_lines": int(totals.get("missing_lines", 0)),
            "per_module": {
                name: round(v.get("summary", {}).get("percent_covered", 0.0), 1)
                for name, v in payload.get("files", {}).items()
            },
            "measured_by": "python -m coverage run --source=src/tensorphysics/uq -m pytest tests/test_uq_*.py",
        }
    docs = {
        "api_reference": (ROOT / "docs" / "UQ_API_REFERENCE.md").is_file(),
        "migration_guide": (ROOT / "docs" / "UQ_MIGRATION_GUIDE.md").is_file(),
        "uncertainty_budget": (ROOT / "docs" / "UQ_UNCERTAINTY_BUDGET.md").is_file(),
    }
    cli_modes = {}
    for flag in ("--pce", "--sobol", "--montecarlo"):
        cmd = [sys.executable, "-m", "tensorphysics.uq.cli", flag, "--benchmark", "ishigami",
               "--n", "48", "--degree", "6", "--n-base", "256", "--budget-n-mc", "256",
               "--quiet", "--json", str(OUT / f"cli_{flag.strip('-')}.json")]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              env={**__import__("os").environ, "PYTHONPATH": str(SRC)})
        cli_modes[flag] = {"exit": proc.returncode, "stderr": proc.stderr[-300:]}
    coverage_ok = coverage is not None and coverage["percent_covered"] >= 90.0
    docs_ok = all(docs.values())
    cli_ok = all(v["exit"] == 0 for v in cli_modes.values())
    return {
        "criterion": "coverage >= 90%, API + migration docs present, CLI three modes usable",
        "coverage": coverage,
        "coverage_ok": bool(coverage_ok),
        "docs": docs,
        "docs_ok": bool(docs_ok),
        "cli_modes": cli_modes,
        "cli_ok": bool(cli_ok),
        "pytest_items": f7["n_test_functions"],
        "pass": bool(coverage_ok and docs_ok and cli_ok),
    }


# ---------------------------------------------------------------------------
# F6 - cross-repository adapters
# ---------------------------------------------------------------------------
def f6_cross_repo() -> dict[str, Any]:
    files = {
        "TensorLBM": DATA / "tensorlbm_sdaa_wigley_hydrostatics.json",
        "TensorFVM-profile": DATA / "tensorfvm_lid_cavity_summary.json",
        "TensorFVM-mesh": DATA / "tensorfvm_mesh_quality_results.json",
        "TensorFEM": DATA / "tensorfem_finite_width_hole_audit.json",
    }
    parsed = {}
    solvers = set()
    for label, path in files.items():
        artifact, records = parse_benchmarks(path)
        solvers.add(artifact.solver)
        rel = [r.relative_error for r in records if r.relative_error is not None]
        parsed[label] = {
            "solver": artifact.solver,
            "path": str(path.relative_to(ROOT)),
            "sha256": artifact.sha256,
            "sha256_matches_disk": artifact.sha256 == digest(path),
            "n_records": len(records),
            "first_qoi": records[0].qoi_id,
            "worst_relative_error": None if not rel else float(max(abs(v) for v in rel)),
        }
    return {
        "criterion": ">= 2 solver-repository benchmark schemas parsed correctly",
        "solver_repositories": sorted(solvers),
        "n_repositories": len(solvers),
        "artifacts": parsed,
        "pass": bool(len(solvers) >= 2 and all(v["n_records"] > 0 and v["sha256_matches_disk"]
                                                for v in parsed.values())),
    }


# ---------------------------------------------------------------------------
# F7 - pytest suite
# ---------------------------------------------------------------------------
NEW_TESTS = (
    "tests/test_uq_holton.py",
    "tests/test_uq_failsafe.py",
    "tests/test_uq_solver_outputs.py",
    "tests/test_uq_cli.py",
    "tests/test_uq_gate_matrix.py",
)


def f7_pytest() -> dict[str, Any]:
    counts = {}
    total = 0
    for rel in NEW_TESTS:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        n = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
        counts[rel] = n
        total += n
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *NEW_TESTS, "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return {
        "criterion": ">= 20 pytest items incl. analytic anchors, negative controls, fail-safe trigger",
        "n_test_functions": total,
        "per_file": counts,
        "collected_and_run": tail,
        "exit_code": proc.returncode,
        "pass": bool(total >= 20 and proc.returncode == 0),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figures(f1, f2, f4, f6) -> list[Path]:
    made: list[Path] = []

    # fig1: Ishigami indices, analytic vs PCE vs sampled
    labels = list(f1["analytic"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - 0.27, [f1["analytic"][k] for k in labels], width=0.27, label="analytic")
    ax.bar(x, [f1["pce"][k] for k in labels], width=0.27, label="PCE (degree 8)")
    ax.bar(x + 0.27, [f1["mc"][k] for k in labels], width=0.27, label="Saltelli/Jansen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sobol' index")
    ax.set_title(
        f"Ishigami anchor: max |error| = {f1['max_abs_error_pce']:.2e} (PCE), "
        f"{f1['max_abs_error_mc']:.2e} (sampled)"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig1_ishigami_indices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)

    # fig2: Holton convergence
    conv = f2["convergence"]
    n = [c["n_base"] for c in conv]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.loglog(n, [c["max_first_abs_error"] for c in conv], "o-", label="$\\max_i|S_i-S_i^{ref}|$")
    ax.loglog(n, [c["max_total_abs_error"] for c in conv], "s-", label="$\\max_i|S_{Ti}-S_{Ti}^{ref}|$")
    ax.axhline(f2["tolerance"], color="crimson", ls="--", label="5 % acceptance")
    ax.set_xlabel("pick-freeze base sample size $n_{base}$")
    ax.set_ylabel("max |index error|")
    ax.set_title(f"Holton anchor (dim={f2['dim']}, interaction share "
                 f"{100 * f2['interaction_variance_share']:.0f} %)")
    ax.legend(frameon=False)
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig2_holton_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)

    # fig3: fail-safe decision map
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cases = [
        ("starved design\n(10 runs, degree 8)", f4["starved_design"]["q2"],
         f4["starved_design"]["mode"]),
        ("oscillatory QoI\n(64 runs, degree 2)", f4["low_quality_surrogate"]["q2"],
         f4["low_quality_surrogate"]["mode"]),
    ]
    xs = np.arange(len(cases))
    q2 = [c[1] if c[1] is not None else 0.0 for c in cases]
    ax.bar(xs, q2, color=["#d95f02" if c[2] == "montecarlo" else "#1b9e77" for c in cases])
    ax.axhline(DEFAULT_Q2_MIN, color="crimson", ls="--", label="predeclared floor $q^2_{min}$ = 0.5")
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in cases])
    ax.set_ylabel("$Q^2$ (leave-one-out)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Fail-safe route decision (orange = degraded to quasi-MC)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig3_failsafe_decision.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)

    # fig4: budget stack from the parsed benchmark artefacts
    _, fem_records = parse_benchmarks(DATA / "tensorfem_finite_width_hole_audit.json")
    uq_width = 0.05
    budget = combine_budget(uq_width, fem_records, threshold=0.15)
    comps = budget.components[:6]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    names = [c.name.replace("benchmark:TensorFEM:", "FEM ").replace("uq_band", "UQ band")
             for c in comps]
    vals = [c.relative_standard for c in comps]
    ax.barh(names[::-1], vals[::-1], color=["#1b9e77"] + ["#7570b3"] * (len(vals) - 1))
    ax.axvline(budget.expanded_relative, color="crimson", ls="--",
               label=f"combined k=2: {budget.expanded_relative:.4f}")
    ax.axvline(budget.threshold, color="black", ls=":", label=f"threshold {budget.threshold}")
    ax.set_xlabel("relative standard uncertainty component")
    ax.set_title("GUM budget: UQ band + TensorFEM benchmark errors (quadrature)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig4_budget_components.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)
    return made


def main() -> int:
    f7 = f7_pytest()
    f1 = f1_ishigami()
    f2 = f2_holton()
    f3 = f3_pce_quality(f1)
    f4 = f4_failsafe()
    f5 = f5_maturity_gate(f7)
    f6 = f6_cross_repo()
    figs = figures(f1, f2, f4, f6)

    acceptance = {
        "F1_ishigami_anchor": f1["pass"],
        "F2_holton_anchor": f2["pass"],
        "F3_pce_quality": f3["pass"],
        "F4_failsafe": f4["pass"],
        "F5_maturity_gate": f5["pass"],
        "F6_cross_repo": f6["pass"],
        "F7_pytest": f7["pass"],
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    payload = {
        "node": "P142W3-N10",
        "title": "Uncertainty budget as a first-class pre-evidence gate: "
                 "non-intrusive PCE + Sobol' variance decomposition, fail-safe, cross-repo",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - START, 1),
        "git_commit": commit,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "acceptance": acceptance,
        "all_passed": all(acceptance.values()),
        "items": {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F6": f6, "F7": f7},
        "figures": [str(p.relative_to(ROOT)) for p in figs],
        "inputs": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [DATA / "PROVENANCE.json", *sorted(DATA.glob("*.json"))]
        },
    }
    path = OUT / "p142w3_n10_verification.json"
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(jsonable(acceptance), indent=2))
    print(f"written {path.relative_to(ROOT)}; elapsed {time.time() - START:.1f}s")
    return 0 if all(acceptance.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())