"""W3-N04 benchmark: ISO 19906 ice-belt structural reliability (FORM / SORM / system).

Runs the full verification and design campaign documented in
``docs/reports/w3n04/structural_reliability_report.md``:

* criteria (a) closed-form linear / exponential limit states,
* criterion (b) FORM vs crude Monte Carlo (1e6 samples) on the three ice-belt
  limit states (yield, elastic-plastic buckling, Miner fatigue),
* criterion (c) Breitung SORM vs Monte Carlo on a strongly nonlinear surface,
* criterion (d) design-point sensitivity vs central finite differences (h=1e-4),
* criterion (e) the pytest suite of ``tensorphysics.structural``,
* Ditlevsen series-system bounds with a simulated union cross-check,
* back-calculation of the required as-built plate thickness for target beta.

Every number in the report is read back from the JSON written here (no manual
copying). Usage::

    PYTHONPATH=src python3 benchmarks/structural_reliability/run_w3n04_reliability.py
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from math import exp, sqrt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensorphysics.structural import (  # noqa: E402
    BUCKLING,
    FATIGUE,
    YIELD,
    NatafTransform,
    NormalDistribution,
    PolarMidshipReliability,
    ReliabilityProblem,
    VariableSet,
    norm_cdf,
    norm_pdf,
    norm_ppf,
)
from tensorphysics.structural._batch import batch_u_to_x  # noqa: E402
from tensorphysics.structural.distributions import ExponentialDistribution  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "docs", "reports", "w3n04")
FIG_DIR = os.path.join(OUT_DIR, "figures")
RUN_DIR = os.path.join(REPO_ROOT, "outputs", "w3n04_reliability")
MODES = (YIELD, BUCKLING, FATIGUE)
TARGET_BETA = 4.0


def _simpson(f, a: float, b: float, n: int = 20000) -> float:
    if n % 2:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4.0 if i % 2 else 2.0) * f(a + i * h)
    return total * h / 3.0


def beta_from_pf(pf: float) -> float:
    if pf <= 0.0:
        return float("inf")
    return -norm_ppf(min(max(pf, 1e-300), 1.0 - 1e-16))


# --------------------------------------------------------------------------- #
# 1. Closed-form verification cases (criterion a) and the SORM case (c)
# --------------------------------------------------------------------------- #
def closed_form_cases() -> dict:
    cases = {}

    transform = NatafTransform([NormalDistribution(400.0, 40.0), NormalDistribution(200.0, 30.0)])
    form = ReliabilityProblem(transform, lambda x: x[0] - x[1], name="linear").form()
    analytic = 200.0 / sqrt(40.0**2 + 30.0**2)
    cases["linear"] = {
        "limit_state": "g = R - S, R~N(400,40), S~N(200,30)",
        "reference": "beta = (mu_R - mu_S)/sqrt(sR^2 + sS^2)",
        "reference_beta": analytic,
        "form_beta": form.beta,
        "form_pf": form.pf,
        "relative_error": abs(form.beta - analytic) / analytic,
        "design_point": form.design_point_x,
        "iterations": form.iterations,
        "converged": form.converged,
    }

    transform = NatafTransform([NormalDistribution(1.0, 0.5)])
    form = ReliabilityProblem(transform, lambda x: exp(2.0) - exp(x[0]), name="exponential-lsf").form()
    analytic = (2.0 - 1.0) / 0.5
    cases["exponential_lsf"] = {
        "limit_state": "g = exp(2) - exp(X), X~N(1,0.5)  (failure iff X > 2)",
        "reference": "beta = (ln(a)/b - mu_X)/sigma_X",
        "reference_beta": analytic,
        "form_beta": form.beta,
        "form_pf": form.pf,
        "relative_error": abs(form.beta - analytic) / analytic,
        "design_point": form.design_point_x,
        "iterations": form.iterations,
        "converged": form.converged,
    }

    transform = NatafTransform([NormalDistribution(3.0, 0.3), ExponentialDistribution(1.0)])
    problem = ReliabilityProblem(transform, lambda x: x[0] - x[1], name="exponential-marginal")
    form = problem.form()
    reference_pf = _simpson(lambda e: exp(-e) * norm_cdf((e - 3.0) / 0.3), 0.0, 30.0, 12000)
    reference_beta = beta_from_pf(reference_pf)
    mc = problem.monte_carlo_pf(samples=1_000_000, seed=2026)
    cases["exponential_marginal"] = {
        "limit_state": "g = R - E, R~N(3,0.3), E~Exp(rate=1)",
        "reference": "Pf = int_0^inf exp(-e) Phi((e-3)/0.3) de  (Simpson, 12000 panels)",
        "reference_beta": reference_beta,
        "reference_pf": reference_pf,
        "form_beta": form.beta,
        "form_pf": form.pf,
        "relative_error": abs(form.beta - reference_beta) / reference_beta,
        "mc_beta": mc.beta,
        "mc_pf": mc.pf,
        "mc_relative_error": abs(mc.beta - reference_beta) / reference_beta,
        "iterations": form.iterations,
        "converged": form.converged,
    }
    return cases


def sorm_case(a: float = 3.0, c: float = 1.0) -> dict:
    """SORM verification on g = a - u1 + c (u2^2 - 1), a strongly curved surface."""
    import numpy as np

    def scalar(x):
        return a - x[0] + c * (x[1] ** 2 - 1.0)

    def batch(x):
        array = np.asarray(x, dtype=float)
        return a - array[:, 0] + c * (array[:, 1] ** 2 - 1.0)

    transform = NatafTransform([NormalDistribution(0.0, 1.0), NormalDistribution(0.0, 1.0)])
    problem = ReliabilityProblem(transform, scalar, name="parabolic", batch_limit_state=batch)
    form = problem.form()
    sorm = problem.sorm(form)
    mc = problem.monte_carlo_pf(samples=1_000_000, seed=2026)
    reference_pf = _simpson(
        lambda u: norm_pdf(u) * norm_cdf(-(a + c * (u * u - 1.0))), -10.0, 10.0, 40000
    )
    reference_beta = beta_from_pf(reference_pf)
    return {
        "limit_state": f"g = {a} - u1 + {c}*(u2^2 - 1) in standard normal space",
        "reference": "Pf = int phi(u2) Phi(-(a + c(u2^2-1))) du2  (Simpson, 40000 panels)",
        "reference_beta": reference_beta,
        "reference_pf": reference_pf,
        "form_beta": form.beta,
        "form_pf": form.pf,
        "sorm_beta": sorm.beta_sorm,
        "sorm_pf": sorm.pf_sorm,
        "sorm_curvatures": sorm.principal_curvatures,
        "sorm_valid": sorm.valid,
        "form_relative_error": abs(form.beta - reference_beta) / reference_beta,
        "sorm_relative_error": abs(sorm.beta_sorm - reference_beta) / reference_beta,
        "sorm_vs_mc_relative_difference": abs(sorm.beta_sorm - mc.beta) / mc.beta,
        "form_vs_mc_relative_difference": abs(form.beta - mc.beta) / mc.beta,
        "mc_beta": mc.beta,
        "mc_pf": mc.pf,
        "mc_samples": mc.samples,
        "design_point_u": form.design_point_u,
        "a": a,
        "c": c,
    }


# --------------------------------------------------------------------------- #
# 2. Ice-belt component / system analysis
# --------------------------------------------------------------------------- #
def ice_analysis() -> tuple[PolarMidshipReliability, dict]:
    import numpy as np

    analysis = PolarMidshipReliability()
    forms = analysis.form_all()
    components: dict[str, dict] = {}
    for name in MODES:
        form = forms[name]
        sorm = analysis.problems[name].sorm(form)
        mc = analysis.problems[name].monte_carlo_pf(samples=1_000_000, seed=2026)
        isample = analysis.problems[name].importance_sampling_pf(
            samples=200_000, seed=7, form_result=form
        )
        components[name] = {
            "form_beta": form.beta,
            "form_pf": form.pf,
            "sorm_beta": sorm.beta_sorm,
            "sorm_pf": sorm.pf_sorm,
            "sorm_curvatures": sorm.principal_curvatures,
            "sorm_valid": sorm.valid,
            "mc_beta": mc.beta,
            "mc_pf": mc.pf,
            "mc_samples": mc.samples,
            "mc_failures": mc.failures,
            "mc_std_error": mc.std_error,
            "form_vs_mc_relative": abs(form.beta - mc.beta) / mc.beta,
            "sorm_vs_mc_relative": abs(sorm.beta_sorm - mc.beta) / mc.beta,
            "is_beta": isample.beta,
            "is_pf": isample.pf,
            "is_std_error": isample.std_error,
            "is_vs_form_relative": abs(isample.beta - form.beta) / form.beta,
            "design_point_x": form.design_point_x,
            "design_point_u": form.design_point_u,
            "alpha": form.alpha,
            "g_design": form.g_design,
            "iterations": form.iterations,
            "converged": form.converged,
            "gradient_norm": form.gradient_norm,
            "history": form.history,
        }

    system = analysis.system_bounds(forms)
    rng = np.random.default_rng(2027)
    samples = 1_000_000
    x = batch_u_to_x(analysis.transform, rng.standard_normal((samples, 4)))
    union = np.zeros(len(x), dtype=bool)
    mode_fail = {}
    for name, function in analysis.geometry.batch_limit_states().items():
        failed = function(x) < 0.0
        mode_fail[name] = float(failed.mean())
        union |= failed
    pf_sim = float(union.mean())
    system_payload = system.to_dict()
    system_payload.update(
        {
            "simulated_union_pf": pf_sim,
            "simulated_union_beta": beta_from_pf(pf_sim),
            "simulated_samples": samples,
            "simulated_component_pf": mode_fail,
            "bounds_contain_simulation_10pct": bool(
                system.pf_lower * 0.9 <= pf_sim <= system.pf_upper * 1.1
            ),
            "simulation_vs_upper_bound_relative": (pf_sim - system.pf_upper) / system.pf_upper,
        }
    )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "geometry": {
            "t_nom_mm": analysis.geometry.t_nom,
            "stiffener_spacing_mm": analysis.geometry.stiffener_spacing,
            "frame_spacing_mm": analysis.geometry.frame_spacing,
            "alpha_bending": analysis.geometry.alpha_bending,
            "k_buckling": analysis.geometry.k_buckling,
            "E_MPa": analysis.geometry.E,
            "nu": analysis.geometry.nu,
            "sigma_global_MPa": analysis.geometry.sigma_global,
            "psi_membrane": analysis.geometry.psi_membrane,
            "sn_log_a": analysis.geometry.sn_log_a,
            "sn_slope": analysis.geometry.sn_slope,
            "cycles": analysis.geometry.cycles,
            "pressure_amplitude_MPa": analysis.geometry.pressure_amplitude,
            "damage_allow": analysis.geometry.damage_allow,
        },
        "variables": analysis.variables.describe(),
        "correlation": [list(row) for row in analysis.variables.correlation],
        "natanf_rho0": [list(row) for row in analysis.transform.rho0],
        "nominal_stresses": analysis.nominal_stresses(),
        "components": components,
        "system": system_payload,
        "sensitivities": {name: analysis.sensitivities(name) for name in MODES},
    }
    return analysis, payload


# --------------------------------------------------------------------------- #
# 3. Design curves, thickness back-calculation, correlation study
# --------------------------------------------------------------------------- #
def design_sweep(analysis: PolarMidshipReliability) -> dict:
    thicknesses = [28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 50.0, 55.0, 60.0]
    curves = {name: [] for name in MODES}
    system_curve = []
    for t in thicknesses:
        for name in MODES:
            curves[name].append(analysis.beta_of_thickness(t, name))
        system_curve.append(analysis.system_beta_of_thickness(t))
    design_table = {}
    for target in (3.0, 3.5, 4.0, 4.5):
        row = {}
        for name in MODES:
            outcome = analysis.required_thickness_for(name, target_beta=target)
            row[name] = {
                "t_required": outcome["t_required"],
                "beta_achieved": outcome["beta_achieved"],
                "iterations": outcome["iterations"],
            }
        system_outcome = analysis.required_thickness_for(None, target_beta=target)
        row["system"] = {
            "t_required": system_outcome["t_required"],
            "beta_achieved": system_outcome["beta_achieved"],
            "iterations": system_outcome["iterations"],
        }
        row["plt_gain"] = {
            key: row[key]["t_required"] / analysis.geometry.t_nom - 1.0
            for key in (*MODES, "system")
        }
        design_table[f"{target:.1f}"] = row

    correlation_curve = []
    for rho in (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75):
        variables = VariableSet(
            marginals=analysis.variables.marginals,
            names=analysis.variables.names,
            correlation=(
                (1.0, 0.0, 0.0, rho),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (rho, 0.0, 0.0, 1.0),
            ),
            units=analysis.variables.units,
        )
        variant = PolarMidshipReliability(geometry=analysis.geometry, variables=variables)
        variant_forms = variant.form_all()
        correlation_curve.append(
            {
                "rho": rho,
                "beta_yield": variant_forms[YIELD].beta,
                "beta_buckling": variant_forms[BUCKLING].beta,
                "beta_fatigue": variant_forms[FATIGUE].beta,
                "natanf_rho0_load_pair": variant.transform.rho0[0][3],
            }
        )
    return {
        "thicknesses": thicknesses,
        "beta_curves": curves,
        "system_curve": system_curve,
        "target_beta": TARGET_BETA,
        "design_table": design_table,
        "correlation_curve": correlation_curve,
    }


def fatigue_damage_sampling(analysis: PolarMidshipReliability, samples: int = 1_000_000) -> dict:
    import numpy as np

    rng = np.random.default_rng(31337)
    x = batch_u_to_x(analysis.transform, rng.standard_normal((samples, 4)))
    geometry = analysis.geometry
    t_net = geometry.t_nom - x[:, 2]
    dsigma = (
        x[:, 3]
        * geometry.alpha_bending
        * geometry.pressure_amplitude
        * (geometry.stiffener_spacing / t_net) ** 2
    )
    damage = geometry.cycles * dsigma**geometry.sn_slope / geometry.sn_capacity()
    return {
        "damage_samples": damage,
        "mean": float(damage.mean()),
        "p95": float(np.quantile(damage, 0.95)),
        "p99": float(np.quantile(damage, 0.99)),
        "pf_exceed": float((damage > geometry.damage_allow).mean()),
        "damage_allow": geometry.damage_allow,
        "samples": samples,
    }


def pytest_summary() -> dict:
    targets = [
        "tests/test_structural_distributions.py",
        "tests/test_structural_reliability.py",
        "tests/test_structural_ice_reliability.py",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *targets, "-q", "--tb=no"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    match_pass = re.search(r"(\d+) passed", proc.stdout)
    match_fail = re.search(r"(\d+) failed", proc.stdout)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return {
        "command": "python -m pytest " + " ".join(targets) + " -q --tb=no",
        "passed": int(match_pass.group(1)) if match_pass else 0,
        "failed": int(match_fail.group(1)) if match_fail else 0,
        "returncode": proc.returncode,
        "summary_line": tail,
        "targets": targets,
    }


# --------------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------------- #
def make_figures(analysis, payload, design, sorm_payload, damage_info) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(FIG_DIR, exist_ok=True)
    figures: list[dict] = []
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True})
    geometry = analysis.geometry
    forms = {name: analysis.problems[name].form() for name in MODES}

    # fig 1 -- safety-margin fields ----------------------------------------
    pressure = np.linspace(1.0, 8.0, 220)
    thickness = np.linspace(26.0, 52.0, 220)
    P, T = np.meshgrid(pressure, thickness)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    for ax, name in zip(axes, MODES):
        dx = forms[name].design_point_x
        corrosion, k_mod, fy = dx[2], dx[3], dx[1]
        t_net = T - corrosion
        if name == YIELD:
            g = fy - k_mod * geometry.alpha_bending * P * (geometry.stiffener_spacing / t_net) ** 2
            unit = "[MPa]"
        elif name == BUCKLING:
            coeff = geometry.k_buckling * math.pi**2 * geometry.E / (12.0 * (1.0 - geometry.nu**2))
            sigma_e = coeff * (t_net / geometry.frame_spacing) ** 2
            capacity = np.where(sigma_e <= 0.5 * fy, sigma_e, fy * (1.0 - fy / (4.0 * sigma_e)))
            g = capacity - k_mod * (geometry.sigma_global + geometry.psi_membrane * P)
            unit = "[MPa]"
        else:
            dsigma = k_mod * geometry.alpha_bending * geometry.pressure_amplitude * (
                geometry.stiffener_spacing / t_net
            ) ** 2
            g = geometry.damage_allow - geometry.cycles * dsigma**geometry.sn_slope / geometry.sn_capacity()
            unit = "[-]"
        contour = ax.contourf(P, T, g, levels=22, cmap="RdYlBu")
        ax.contour(P, T, g, levels=[0.0], colors="k", linewidths=1.7)
        plt.colorbar(contour, ax=ax, label=f"g {unit}")
        ax.axhline(geometry.t_nom, color="k", ls=":", lw=0.9)
        ax.plot([dx[0]], [geometry.t_nom], "k*", markersize=14, label="design point")
        ax.set_xlabel("ice pressure p [MPa]")
        ax.set_ylabel("as-built thickness t_nom [mm]")
        ax.set_title(f"{name}: safety margin g(p, t)")
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("W3-N04: limit-state safety margins in the (ice pressure, plate thickness) plane")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_limit_state_fields.png")
    fig.savefig(path)
    plt.close(fig)
    figures.append(
        {
            "name": "fig1_limit_state_fields.png",
            "path": os.path.relpath(path, REPO_ROOT),
            "caption": "Safety-margin fields g(p_ice, t_nom) for the three limit states with the remaining "
            "random variables frozen at their design-point values; the black curve is g = 0, the star is "
            "the design point projected on the nominal thickness line.",
        }
    )

    # fig 2 -- beta vs thickness -------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for name in MODES:
        ax.plot(design["thicknesses"], design["beta_curves"][name], "o-", label=name)
    ax.plot(design["thicknesses"], design["system_curve"], "k--", label="series system (Ditlevsen)")
    ax.axhline(TARGET_BETA, color="r", ls=":", label="ISO 19906 class target beta = 4.0")
    ax.axvline(geometry.t_nom, color="grey", ls="-.", lw=1.0, label=f"nominal t = {geometry.t_nom:.0f} mm")
    t_req = design["design_table"][f"{TARGET_BETA:.1f}"]["system"]["t_required"]
    ax.axvline(t_req, color="r", ls="-.", lw=1.0, label=f"required t = {t_req:.1f} mm")
    ax.set_xlabel("as-built plate thickness t_nom [mm]")
    ax.set_ylabel("reliability index beta")
    ax.set_title("Reliability index vs ice-belt plate thickness")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_beta_vs_thickness.png")
    fig.savefig(path)
    plt.close(fig)
    figures.append(
        {
            "name": "fig2_beta_vs_thickness.png",
            "path": os.path.relpath(path, REPO_ROOT),
            "caption": "Component and series-system reliability index versus as-built plating thickness; the "
            "intersection with the beta = 4.0 target defines the required thickness.",
        }
    )

    # fig 3 -- FORM / SORM / MC --------------------------------------------
    labels = list(MODES) + ["parabolic case"]
    form_b = [payload["components"][m]["form_beta"] for m in MODES] + [sorm_payload["form_beta"]]
    sorm_b = [payload["components"][m]["sorm_beta"] for m in MODES] + [sorm_payload["sorm_beta"]]
    mc_b = [payload["components"][m]["mc_beta"] for m in MODES] + [sorm_payload["mc_beta"]]
    reference = [None, None, None, sorm_payload["reference_beta"]]
    xpos = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.bar(xpos - width, form_b, width, label="FORM")
    ax.bar(xpos, sorm_b, width, label="SORM (Breitung)")
    ax.bar(xpos + width, mc_b, width, label="Monte Carlo (1e6)")
    for i, ref in enumerate(reference):
        if ref is not None:
            ax.hlines(ref, i - 1.5 * width, i + 1.5 * width, colors="k", linestyles="--",
                      label="quadrature reference")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("reliability index beta")
    ax.set_title("FORM vs SORM vs simulation (annotations: relative difference to MC)")
    ax.legend(fontsize=7)
    for i, (f_, s_, m_) in enumerate(zip(form_b, sorm_b, mc_b)):
        ax.annotate(f"{abs(f_ - m_) / m_ * 100:.2f}%", (i - width, f_), ha="center", va="bottom", fontsize=6)
        ax.annotate(f"{abs(s_ - m_) / m_ * 100:.2f}%", (i, s_), ha="center", va="bottom", fontsize=6)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_form_sorm_mc.png")
    fig.savefig(path)
    plt.close(fig)
    figures.append(
        {
            "name": "fig3_form_sorm_mc.png",
            "path": os.path.relpath(path, REPO_ROOT),
            "caption": "Reliability indices from FORM, Breitung SORM and Monte Carlo (1e6 samples) for the three "
            "ice-belt limit states and the strongly nonlinear parabolic verification case.",
        }
    )

    # fig 4 -- convergence --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name in MODES:
        history = payload["components"][name]["history"]
        ax.semilogy([h["iteration"] for h in history], [abs(h["g"]) + 1e-16 for h in history], "o-", label=name)
    ax.set_xlabel("radial Hasofer-Lind iteration")
    ax.set_ylabel("|g| at the iterate [MPa or -]")
    ax.set_title("Design-point iteration convergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    figures.append(
        {
            "name": "fig4_convergence.png",
            "path": os.path.relpath(path, REPO_ROOT),
            "caption": "Convergence of the radial HL-RF iteration: |g| falls to the 1e-13 level within a few "
            "iterations for all three limit states.",
        }
    )

    # fig 5 -- sensitivities ------------------------------------------------
    table = payload["sensitivities"][YIELD]
    keys = list(table.keys())
    ypos = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.barh(ypos - 0.2, [table[k]["analytic"] for k in keys], 0.4, label="design-point (analytic)")
    ax.barh(ypos + 0.2, [table[k]["finite_difference"] for k in keys], 0.4,
            label="central difference h = 1e-4")
    ax.set_yticks(ypos)
    ax.set_yticklabels(keys, fontsize=7)
    ax.set_xlabel("d beta / d theta_i")
    ax.set_title("Yield mode: sensitivity of beta to the distribution parameters")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig5_sensitivity.png")
    fig.savefig(path)
    plt.close(fig)
    figures.append(
        {
            "name": "fig5_sensitivity.png",
            "path": os.path.relpath(path, REPO_ROOT),
            "caption": "Design-point sensitivities of the yield-mode reliability index with respect to all eight "
            "distribution parameters, analytic versus central differences.",
        }
    )

    # fig 6 -- correlation --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    rhos = [row["rho"] for row in design["correlation_curve"]]
    for key, style in (("beta_yield", "o-"), ("beta_buckling", "s-"), ("beta_fatigue", "^-")):
        ax.plot(rhos, [row[key] for row in design["correlation_curve"]], style, label=key.replace("beta_", ""))
    ax.set_xlabel("correlation rho between ice pressure and load-model factor")
    ax.set_ylabel("reliability index beta")
    ax.set_title("Influence of the load correlation structure (Nataf)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig6_correlation.png")
    fig.savefig(path)
    plt.close(fig)
    figures.append(
        {
            "name": "fig6_correlation.png",
            "path": os.path.relpath(path, REPO_ROOT),
            "caption": "Reliability index versus the physical correlation between the ice-pressure extreme and "
            "the load-model bias; the Nataf transformation re-solves the Gaussian-copula correlation at every point.",
        }
    )

    # fig 7 -- fatigue damage -----------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    damage = damage_info["damage_samples"]
    upper = min(float(np.quantile(damage, 0.9995)) * 1.4, 3.0)
    ax.hist(damage, bins=140, range=(0.0, upper), color="steelblue", alpha=0.85,
            label="Monte Carlo damage D")
    ax.axvline(damage_info["damage_allow"], color="r", ls="--",
               label=f"D_allow = {damage_info['damage_allow']}")
    ax.set_xlabel("Miner damage D [-]")
    ax.set_ylabel("count")
    ax.set_title("Miner damage distribution at the nominal thickness (1e6 samples)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig7_fatigue_damage.png")
    fig.savefig(path)
    plt.close(fig)
    figures.append(
        {
            "name": "fig7_fatigue_damage.png",
            "path": os.path.relpath(path, REPO_ROOT),
            "caption": "Monte Carlo distribution of the Miner damage at the nominal thickness; the tail beyond "
            "D_allow is the fatigue failure probability.",
        }
    )

    # fig 8 -- SORM geometry ------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    a, c = sorm_payload["a"], sorm_payload["c"]
    u1 = np.linspace(0.0, 7.0, 400)
    u2 = np.linspace(-3.0, 3.0, 400)
    U2, U1 = np.meshgrid(u2, u1)
    g = a - U1 + c * (U2**2 - 1.0)
    contour = ax.contourf(U1, U2, g, levels=24, cmap="RdYlBu")
    ax.contour(U1, U2, g, levels=[0.0], colors="k", linewidths=1.8)
    star = sorm_payload["design_point_u"]
    ax.plot([star[0]], [star[1]], "k*", markersize=15, label="FORM design point")
    beta = sorm_payload["form_beta"]
    ax.plot([beta, beta], [-3.0, 3.0], "k--", lw=1.2, label="FORM tangent (linearised)")
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_title("SORM verification: parabolic limit-state surface")
    plt.colorbar(contour, ax=ax, label="g [-]")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig8_sorm_geometry.png")
    fig.savefig(path)
    plt.close(fig)
    figures.append(
        {
            "name": "fig8_sorm_geometry.png",
            "path": os.path.relpath(path, REPO_ROOT),
            "caption": "Parabolic limit-state surface in standard normal space: FORM replaces the surface by its "
            "tangent at the design point, SORM corrects with the principal curvature kappa = 2c.",
        }
    )
    return figures


# --------------------------------------------------------------------------- #
# 5. Driver
# --------------------------------------------------------------------------- #
def criteria_block(payload, closed, sorm_payload, design, pytest_info) -> dict:
    comp = payload["components"]
    sens_errors = []
    for name in MODES:
        for key, entry in payload["sensitivities"][name].items():
            if abs(entry["finite_difference"]) >= 1e-9:
                sens_errors.append(entry["relative_error"])
    items = []

    def add(identifier, criterion, value, limit, passed, detail=""):
        items.append(
            {
                "id": identifier,
                "criterion": criterion,
                "value": value,
                "limit": limit,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    worst_closed = max(closed[k]["relative_error"] for k in closed)
    add("a", "closed-form linear/exponential beta vs analytic (max rel. error)",
        f"{worst_closed*100:.4f}%", "< 0.5%", worst_closed < 0.005,
        "; ".join(f"{k}: {closed[k]['relative_error']*100:.4f}%" for k in closed))
    worst_form_mc = max(comp[m]["form_vs_mc_relative"] for m in MODES)
    add("b", "FORM beta vs Monte Carlo 1e6 (max rel. difference over 3 limit states)",
        f"{worst_form_mc*100:.4f}%", "< 2%", worst_form_mc < 0.02,
        "; ".join(f"{m}: {comp[m]['form_vs_mc_relative']*100:.2f}%" for m in MODES))
    add("c1", "SORM beta vs Monte Carlo 1e6 on strongly nonlinear surface",
        f"{sorm_payload['sorm_vs_mc_relative_difference']*100:.4f}%", "< 3%",
        sorm_payload["sorm_vs_mc_relative_difference"] < 0.03,
        f"FORM error {sorm_payload['form_relative_error']*100:.3f}% vs SORM "
        f"{sorm_payload['sorm_relative_error']*100:.3f}% against the quadrature reference")
    add("c2", "SORM better than FORM on the nonlinear surface",
        f"SORM {sorm_payload['sorm_relative_error']*100:.3f}% vs FORM "
        f"{sorm_payload['form_relative_error']*100:.3f}%", "SORM < FORM",
        sorm_payload["sorm_relative_error"] < sorm_payload["form_relative_error"])
    worst_sens = max(sens_errors) if sens_errors else 0.0
    add("d", "design-point d beta / d theta vs central difference (h = 1e-4)",
        f"{worst_sens*100:.4f}%", "< 1%", worst_sens < 0.01,
        f"{len(sens_errors)} non-zero parameter sensitivities over 3 limit states")
    add("e", "pytest suite of the structural reliability module",
        f"{pytest_info['passed']} passed, {pytest_info['failed']} failed", ">= 18 passed",
        pytest_info["passed"] >= 18 and pytest_info["failed"] == 0,
        pytest_info["summary_line"])
    system = payload["system"]
    add("design", "target beta = 4.0 back-solved plate thickness (system)",
        f"{design['design_table']['4.0']['system']['t_required']:.2f} mm "
        f"(achieved beta {design['design_table']['4.0']['system']['beta_achieved']:.3f})",
        "|beta - 4.0| <= 0.01",
        abs(design["design_table"]["4.0"]["system"]["beta_achieved"] - 4.0) <= 0.01,
        f"nominal thickness {payload['geometry']['t_nom_mm']:.1f} mm gives beta "
        f"{payload['system']['beta_point']:.3f}")
    add("system", "Ditlevsen bounds bracket the simulated 1e6-sample union probability",
        f"bounds [{system['pf_lower']:.4e}, {system['pf_upper']:.4e}] vs simulated "
        f"{system['simulated_union_pf']:.4e}", "within 10%",
        system["bounds_contain_simulation_10pct"],
        f"simulation / upper bound - 1 = {system['simulation_vs_upper_bound_relative']*100:+.2f}%")
    return {
        "items": items,
        "all_passed": all(item["passed"] for item in items),
        "count": len(items),
    }


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)

    print("[1/6] closed-form verification cases")
    closed = closed_form_cases()
    print("[2/6] ice-belt component / system analysis")
    analysis, payload = ice_analysis()
    print("[3/6] thickness design curves")
    design = design_sweep(analysis)
    print("[4/6] SORM verification case")
    sorm_payload = sorm_case()
    print("[5/6] fatigue damage sampling + figures")
    damage_info = fatigue_damage_sampling(analysis)
    figures = make_figures(analysis, payload, design, sorm_payload, damage_info)
    print("[6/6] pytest + report")
    pytest_info = pytest_summary()
    criteria = criteria_block(payload, closed, sorm_payload, design, pytest_info)

    results = {
        "node": "W3-N04",
        "title": "Structural reliability of a polar ship midship ice belt: FORM/SORM with ISO 19906 limit states",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": "TensorPhysics",
        "module": "tensorphysics.structural",
        "branch": "agent/m791",
        "criteria": criteria,
        "closed_form": closed,
        "sorm_case": sorm_payload,
        "design": design,
        "fatigue_damage": {
            "mean": damage_info["mean"],
            "p95": damage_info["p95"],
            "p99": damage_info["p99"],
            "pf_exceed": damage_info["pf_exceed"],
            "damage_allow": damage_info["damage_allow"],
            "samples": damage_info["samples"],
        },
        "pytest": pytest_info,
        "payload": payload,
        "figures": figures,
    }

    target = os.path.join(OUT_DIR, "reliability_results.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=False)
    with open(os.path.join(RUN_DIR, "reliability_results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=False)
    print(f"wrote {os.path.relpath(target, REPO_ROOT)}")

    from w3n04_report import build_report  # local import: report assembly

    markdown = build_report(results)
    report_path = os.path.join(OUT_DIR, "structural_reliability_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"wrote {os.path.relpath(report_path, REPO_ROOT)}")

    print(json.dumps({item["id"]: item["passed"] for item in criteria["items"]}, indent=2))
    return 0 if criteria["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())