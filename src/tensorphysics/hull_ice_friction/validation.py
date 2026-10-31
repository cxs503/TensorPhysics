"""端到端标定-验证流程：数值实验 → 参数辨识 → 幂律拟合 → 参考数据比对 → 收敛性。

产物（默认写入 ``artifacts/hull_ice_friction/``）：

* ``calibration_report.json``  判据数字、参数、逐点误差、收敛性表（数字全部程序化生成）
* ``figures/*.png``            μ(T) 曲线、机理分解、收敛性、速度敏感性

所有判据阈值在此显式给出，通过/不通过由程序判定，不接受人工改数。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace as dc_replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .calibration import (
    compare_with_reference,
    fit_power_law,
    identify_real_contact_pressure,
)
from .contact_law import IceSteelContactParameters
from .experiment import (
    AsperityEnsembleParameters,
    FrictionTestConfig,
    build_ensemble_bank,
    evaluate_ensemble_bank,
    run_temperature_scan,
    run_velocity_sweep,
)
from .ice_materials import DEFAULT_ICE, IceMaterialCorrelations
from .reference_data import (
    PROVENANCE,
    reference_mu,
    reference_table,
    reference_temperatures,
    reference_uncertainties,
    role_targets,
)
from .thermal_contact import (
    STEEL_THERMAL,
    fd_grid_convergence,
    interface_heat_partition,
)

#: 验收阈值（机器可算判据）
CRITERIA = {
    "power_law_r2_min": 0.95,
    "per_point_relative_error_max": 0.15,
    "fd_grid_relative_error_max": 0.01,
    "fixed_point_relative_change_max": 1e-6,
    "ensemble_size_relative_change_max": 0.02,
}

DEFAULT_TEMPERATURES_C = tuple(reference_temperatures())


# ---------------------------------------------------------------------------
# 收敛性研究
# ---------------------------------------------------------------------------


def material_property_table(
    temperatures_C=DEFAULT_TEMPERATURES_C,
    material: IceMaterialCorrelations = DEFAULT_ICE,
    salinity_psu: float = 0.0,
) -> list[dict]:
    """冰力学性质的温度相关性表（E、σ_c、τ_s、硬度）。"""
    rows = []
    for temperature in temperatures_C:
        theta = float(material.undercooling(temperature, salinity_psu))
        rows.append(
            {
                "temperature_C": float(temperature),
                "undercooling_K": theta,
                "elastic_modulus_GPa": float(material.elastic_modulus_GPa(theta)),
                "compressive_strength_MPa": float(material.compressive_strength_MPa(theta)),
                "shear_strength_MPa": float(material.shear_strength_Pa(theta) * 1e-6),
                "hardness_MPa": float(material.hardness_Pa(theta) * 1e-6),
                "melting_temperature_C": float(material.melting_temperature(salinity_psu)),
            }
        )
    return rows


def ensemble_size_convergence(
    config: FrictionTestConfig,
    temperature_C: float = -15.0,
    sizes: tuple[int, ...] = (32, 64, 128, 256, 512, 1024),
) -> list[dict]:
    """微凸体数量 N 的收敛性：μ(N) 相对最细 N 的偏差。"""
    rows: list[dict] = []
    reference_mu_value: float | None = None
    for size in sizes:
        sub = config.replace(ensemble=dc_replace(config.ensemble, n_asperities=int(size)))
        bank = build_ensemble_bank([temperature_C], sub)
        points = evaluate_ensemble_bank(bank, sub)
        mu = points[0]["mu_mean"]
        stderr = points[0]["mu_stderr"]
        rows.append({"n_asperities": int(size), "mu": mu, "mu_stderr": stderr})
    if rows:
        reference_mu_value = rows[-1]["mu"]
        for row in rows:
            row["relative_deviation"] = (row["mu"] - reference_mu_value) / reference_mu_value
    return rows


def repetition_convergence(
    config: FrictionTestConfig,
    temperature_C: float = -15.0,
    max_repetitions: int = 40,
) -> dict:
    """重复次数收敛性：累计均值与标准误随重复次数的衰减。"""
    sub = config.replace(n_repetitions=int(max_repetitions))
    bank = build_ensemble_bank([temperature_C], sub)
    points = evaluate_ensemble_bank(bank, sub)
    reps = np.array(points[0]["mu_repetitions"], dtype=float)
    cumulative_mean = np.cumsum(reps) / np.arange(1, reps.size + 1)
    cumulative_stderr = np.array(
        [
            reps[: k + 1].std(ddof=1) / np.sqrt(k + 1) if k > 0 else 0.0
            for k in range(reps.size)
        ]
    )
    final = float(cumulative_mean[-1])
    return {
        "temperature_C": float(temperature_C),
        "max_repetitions": int(max_repetitions),
        "cumulative_mean": [float(x) for x in cumulative_mean],
        "cumulative_stderr": [float(x) for x in cumulative_stderr],
        "mean_at_n": final,
        "stderr_at_n": float(cumulative_stderr[-1]),
        "max_deviation_from_final_mean": float(np.max(np.abs(cumulative_mean - final))),
    }


def fixed_point_convergence(
    config: FrictionTestConfig, temperatures_C=DEFAULT_TEMPERATURES_C
) -> dict:
    """接触律局部不动点求解的收敛证据（两种求解器对比）。

    * ``bisection``（生产求解器）：区间宽度逐次减半，无条件收敛；
    * ``picard``（ω=0.5 的松弛迭代）：冷端出现极限环（不收敛），说明为何生产求解器
      选择对称二分——这是本次数值实验的一个真实发现，不做粉饰。
    """
    summary = {}
    for solver in ("bisection", "picard"):
        scan = run_temperature_scan(
            temperatures_C, config, collect_history=True, solver=solver
        )
        history = scan["fixed_point_residual_history"]
        per_temperature = {}
        for point in scan["points"]:
            key = f"{point['temperature_C']:g}"
            hist = history.get(key, [])
            per_temperature[key] = {
                "iterations": point["iterations_max"],
                "converged_all": point["converged_all"],
                "final_residual": hist[-1] if hist else None,
                "residual_history": hist,
            }
        summary[solver] = per_temperature
    return summary


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    report: dict
    output_dir: Path


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    return obj


def run_validation(
    output_dir: str | Path = "artifacts/hull_ice_friction",
    base_config: FrictionTestConfig | None = None,
    iterations_note: str | None = None,
    make_figures: bool = True,
) -> ValidationResult:
    """执行完整标定-验证流程并写出 JSON/PNG 产物。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    if make_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    config = base_config or FrictionTestConfig()
    temperatures = list(DEFAULT_TEMPERATURES_C)

    # 0) 接触热物理（Blok 界面热分配等）
    partition = interface_heat_partition()
    thermophysics = {
        "zeta_ice": partition.zeta_ice,
        "zeta_steel": partition.zeta_steel,
        "effusivity_sum_J_m2_K_sqrt": partition.effusivity_sum,
        "ice_effusivity": float(config.material.effusivity),
        "ice_thermal_diffusivity_m2_s": float(config.material.thermal_diffusivity_m2_s),
        "steel_effusivity": float(STEEL_THERMAL.effusivity),
        "note": "Blok interface temperature: dT = 2*q/S*sqrt(t_c/pi), S = effusivity sum",
    }

    # 1) 材料性质温度相关性
    material_table = material_property_table(temperatures, config.material, config.salinity_psu)

    # 2) 唯一本构参数 p_real 在标定子集上辨识
    calibration_targets = role_targets("calibration")
    identification = identify_real_contact_pressure(calibration_targets, config)
    identified_config = config.replace(
        contact=config.contact.replace(real_contact_pressure_Pa=identification.real_contact_pressure_Pa)
    )

    # 3) 温度扫描（每温度 10 重复）
    scan = run_temperature_scan(temperatures, identified_config)

    # 4) 幂律拟合 μ(θ)=μ_0+α θ^β
    theta = np.array([p["undercooling_K"] for p in scan["points"]], dtype=float)
    mu_sim = np.array([p["mu_mean"] for p in scan["points"]], dtype=float)
    fit = fit_power_law(theta, mu_sim)

    # 5) 与文献锚定参考数据比对
    comparison = compare_with_reference(
        scan["points"], reference_mu(), config.salinity_psu, config.material
    )
    calibration_set = {float(t) for t in calibration_targets}
    for row in comparison:
        row["role"] = "calibration" if float(row["temperature_C"]) in calibration_set else "validation"
        row["within_15pct"] = bool(row["abs_relative_error"] < CRITERIA["per_point_relative_error_max"])

    # 6) 收敛性研究
    fd_table = fd_grid_convergence(0.25, node_counts=(41, 81, 161, 321, 641))
    ensemble_table = ensemble_size_convergence(identified_config)
    repetition_table = repetition_convergence(identified_config, temperature_C=-15.0)
    fixed_point = fixed_point_convergence(identified_config, temperatures)

    # 7) 速度敏感性（机理检验：高滑移速度下融水润滑应增强）
    velocity_sweep = run_velocity_sweep((0.25, 0.5, 1.0, 2.0, 4.0), (-5.0, -15.0, -25.0), identified_config)

    # 8) 判据判定
    fd_finest_error = min(row["abs_rel_error"] for row in fd_table)
    ensemble_by_size = {row["n_asperities"]: row for row in ensemble_table}
    production_size = identified_config.ensemble.n_asperities
    finest_size = max(ensemble_by_size)
    ensemble_worst = abs(
        (ensemble_by_size[production_size]["mu"] - ensemble_by_size[finest_size]["mu"])
        / ensemble_by_size[finest_size]["mu"]
    )
    bisection_stats = fixed_point["bisection"]
    picard_stats = fixed_point["picard"]
    fp_ok = all(v["converged_all"] for v in bisection_stats.values())
    fp_residual = max((v["final_residual"] or 0.0) for v in bisection_stats.values())
    picard_converged = all(v["converged_all"] for v in picard_stats.values())
    max_rel_error = max(row["abs_relative_error"] for row in comparison)
    criteria = {
        "criterion_a_power_law_r2": {
            "threshold": CRITERIA["power_law_r2_min"],
            "value": fit.r2,
            "passed": bool(fit.r2 > CRITERIA["power_law_r2_min"]),
            "definition": "R^2 of mu(theta)=mu_0+alpha*theta^beta fitted to the 5 simulated points",
        },
        "criterion_b_per_point_relative_error": {
            "threshold": CRITERIA["per_point_relative_error_max"],
            "value": max_rel_error,
            "passed": bool(max_rel_error < CRITERIA["per_point_relative_error_max"]),
            "per_point": [
                {
                    "temperature_C": row["temperature_C"],
                    "role": row["role"],
                    "relative_error": row["relative_error"],
                    "passed": row["within_15pct"],
                }
                for row in comparison
            ],
            "definition": "max |mu_sim - mu_ref| / mu_ref over the 5 reference temperatures",
        },
        "criterion_c_convergence": {
            "fd_grid_finest_abs_rel_error": fd_finest_error,
            "fd_grid_passed": bool(fd_finest_error < CRITERIA["fd_grid_relative_error_max"]),
            "fixed_point_all_converged": bool(fp_ok),
            "fixed_point_max_final_residual": fp_residual,
            "fixed_point_passed": bool(fp_ok and fp_residual < CRITERIA["fixed_point_relative_change_max"]),
            "fixed_point_solver": "bisection (guaranteed convergence, bracket halving)",
            "picard_relaxed_all_converged": bool(picard_converged),
            "picard_relaxed_note": "omega=0.5 damped Picard develops a limit cycle for hot contact spots (a>=1 mm, p>=30 MPa); the production run uses the unconditionally convergent bisection solver",
            "ensemble_size_production_vs_finest_relative_deviation": ensemble_worst,
            "ensemble_size_passed": bool(ensemble_worst < CRITERIA["ensemble_size_relative_change_max"]),
            "repetition_stderr_at_40": repetition_table["stderr_at_n"],
            "passed": bool(
                fd_finest_error < CRITERIA["fd_grid_relative_error_max"]
                and fp_ok
                and fp_residual < CRITERIA["fixed_point_relative_change_max"]
                and ensemble_worst < CRITERIA["ensemble_size_relative_change_max"]
            ),
        },
        "criterion_d_pytest": {
            "definition": "see tests/test_hull_ice_friction.py; count recorded in report",
            "value": None,
            "passed": None,
        },
    }
    crit_a = criteria["criterion_a_power_law_r2"]["passed"]
    crit_b = criteria["criterion_b_per_point_relative_error"]["passed"]
    crit_c = criteria["criterion_c_convergence"]["passed"]

    report = {
        "node": "W3-N07",
        "title": "Hull-ice friction coefficient temperature dependence: calibration and validation",
        "repository": "TensorPhysics",
        "module": "tensorphysics.hull_ice_friction",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_data_provenance": PROVENANCE,
        "reference_table": reference_table(),
        "reference_uncertainties": {str(k): v for k, v in reference_uncertainties().items()},
        "contact_thermophysics": thermophysics,
        "material_property_table": material_table,
        "constitutive_identification": _json_safe(identification.as_dict()),
        "experiment": {
            "config": scan["config"],
            "n_repetitions": scan["n_repetitions"],
            "points": scan["points"],
        },
        "macro_fit": fit.as_dict(),
        "reference_comparison": comparison,
        "convergence": {
            "fd_grid": fd_table,
            "ensemble_size": ensemble_table,
            "repetitions": repetition_table,
            "fixed_point": fixed_point,
        },
        "velocity_sensitivity": velocity_sweep,
        "criteria": criteria,
        "verdict": {
            "criterion_a": bool(crit_a),
            "criterion_b": bool(crit_b),
            "criterion_c": bool(crit_c),
            "all_machine_criteria_passed": bool(crit_a and crit_b and crit_c),
        },
        "notes": iterations_note,
    }
    report = _json_safe(report)

    if make_figures:
        report["figures"] = _make_figures(report, figures_dir)

    json_path = output_dir / "calibration_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return ValidationResult(report=report, output_dir=output_dir)


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------


def _make_figures(report: dict, figures_dir: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: list[dict] = []
    points = report["experiment"]["points"]
    temperatures = np.array([p["temperature_C"] for p in points], dtype=float)
    mu_sim = np.array([p["mu_mean"] for p in points], dtype=float)
    mu_err = np.array([p["mu_stderr"] for p in points], dtype=float)
    theta = np.array([p["undercooling_K"] for p in points], dtype=float)
    ref_temp, ref_mu, ref_unc = (
        np.array([row["temperature_C"] for row in report["reference_table"]]),
        np.array([row["mu_reference"] for row in report["reference_table"]]),
        np.array([row["uncertainty"] for row in report["reference_table"]]),
    )
    fit = report["macro_fit"]

    # --- 图 1：μ(T) 仿真 vs 幂律拟合 vs 文献锚定参考
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    theta_dense = np.linspace(theta.min() * 0.8, theta.max() * 1.15, 200)
    ax.plot(
        -theta_dense,
        fit["mu_0"] + fit["alpha"] * theta_dense ** fit["beta"],
        "-",
        color="#1f4e79",
        lw=1.8,
        label=f"power law fit: $\\mu={fit['mu_0']:.4f}+{fit['alpha']:.5f}\\,\\theta^{{{fit['beta']:.3f}}}$ (R²={fit['r2']:.4f})",
    )
    ax.errorbar(
        temperatures,
        mu_sim,
        yerr=mu_err,
        fmt="o",
        color="#c00000",
        ms=6,
        capsize=3,
        label="numerical experiment (10 reps, mean ± s.e.)",
    )
    ax.errorbar(
        ref_temp,
        ref_mu,
        yerr=ref_unc,
        fmt="s",
        mfc="none",
        color="#404040",
        ms=7,
        capsize=3,
        label="literature-anchored reference (±cross-study scatter)",
    )
    calib_temps = [
        row["temperature_C"] for row in report["reference_comparison"] if row["role"] == "calibration"
    ]
    for t in calib_temps:
        ax.axvline(t, color="#c00000", alpha=0.07, lw=6)
    ax.set_xlabel("temperature T (°C)")
    ax.set_ylabel("friction coefficient μ")
    ax.set_title("Hull-ice friction coefficient vs temperature\n(translucent bands = calibration subset)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    path = figures_dir / "mu_vs_temperature.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(
        {
            "name": "mu_vs_temperature",
            "path": str(path),
            "caption": "μ(T) from the numerical friction experiment (red, 10 repetitions ±s.e.), the fitted power law μ=μ0+α|T−T_melt|^β (blue) and the literature-anchored reference medians with cross-study scatter (grey).",
        }
    )

    # --- 图 2：机理分解
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    dry = np.array([p["dry_friction"] for p in points])
    lam = np.array([p["lambda_wet_mean"] for p in points])
    wet = np.array([p["wet_friction"] for p in points])
    axes[0].plot(temperatures, dry, "o--", color="#1f4e79", label="dry branch $μ_{dry}=τ_s(θ)/p_{real}$")
    axes[0].plot(temperatures, mu_sim, "o-", color="#c00000", label="mixed-law result μ")
    axes[0].plot(temperatures, np.full_like(temperatures, wet[0]), ":", color="#2e7d32", label="wet branch $μ_{wet}$")
    axes[0].fill_between(
        temperatures, mu_sim, dry, color="#c00000", alpha=0.12, label="lubrication reduction $λ(μ_{dry}-μ_{wet})$"
    )
    axes[0].set_xlabel("temperature T (°C)")
    axes[0].set_ylabel("friction coefficient (-)")
    axes[0].set_title("friction branches")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(temperatures, lam, "s-", color="#7030a0", label="meltwater coverage λ (area weighted)")
    ax2.set_xlabel("temperature T (°C)")
    ax2.set_ylabel("λ (-)")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    flash_mean = np.array([p["flash_temperature_K"]["mean_K"] for p in points])
    flash_p90 = np.array([p["flash_temperature_K"]["p90_K"] for p in points])
    ax2b.plot(temperatures, flash_mean, "^--", color="#e07b00", label="interface flash temp (area mean)")
    ax2b.plot(temperatures, flash_p90, "v:", color="#b35c00", label="flash temp p90")
    ax2b.plot(temperatures, theta, "-", color="black", lw=1.0, label="undercooling θ=|T−T_melt|")
    ax2b.set_ylabel("flash temperature rise (K)")
    ax2b.legend(fontsize=7.5, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_title("meltwater coverage and contact flash temperature")
    fig.tight_layout()
    path = figures_dir / "mechanism_branches.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(
        {
            "name": "mechanism_branches",
            "path": str(path),
            "caption": "Left: dry branch τ_s(θ)/p_real, mixed-law result and the meltwater-lubrication reduction. Right: area-weighted meltwater coverage λ, interface flash temperature (mean/p90) and the undercooling θ.",
        }
    )

    # --- 图 3：收敛性
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    fd = report["convergence"]["fd_grid"]
    nodes = np.array([row["n_nodes"] for row in fd], dtype=float)
    err = np.array([row["abs_rel_error"] for row in fd], dtype=float)
    axes[0, 0].loglog(nodes, err, "o-", color="#1f4e79")
    axes[0, 0].loglog(nodes, err[0] * (nodes[0] / nodes) ** 2, "--", color="#808080", label="slope −2 reference")
    axes[0, 0].set_xlabel("FD nodes")
    axes[0, 0].set_ylabel("|relative error| vs $2\\sqrt{Fo/\\pi}$")
    axes[0, 0].set_title("(a) transient conduction FD vs analytic")
    axes[0, 0].grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=8)

    ens = report["convergence"]["ensemble_size"]
    sizes = np.array([row["n_asperities"] for row in ens], dtype=float)
    dev = np.array([abs(row["relative_deviation"]) for row in ens])
    axes[0, 1].loglog(sizes, np.maximum(dev, 1e-8), "s-", color="#c00000", label="|μ(N)−μ(N_max)|/μ(N_max)")
    axes[0, 1].axhline(CRITERIA["ensemble_size_relative_change_max"], color="k", ls=":", label="2% threshold")
    axes[0, 1].set_xlabel("asperities per realisation N")
    axes[0, 1].set_ylabel("relative deviation")
    axes[0, 1].set_title("(b) ensemble-size convergence (T = -15 °C)")
    axes[0, 1].grid(alpha=0.3, which="both")
    axes[0, 1].legend(fontsize=8)

    fixed = report["convergence"]["fixed_point"]
    for key, entry in fixed["bisection"].items():
        hist = np.array(entry["residual_history"], dtype=float)
        if hist.size:
            axes[1, 0].semilogy(
                np.arange(1, hist.size + 1), np.maximum(hist, 1e-16), "-", label=f"bisection T={key} °C"
            )
    for key, entry in fixed["picard"].items():
        hist = np.array(entry["residual_history"], dtype=float)
        if hist.size:
            axes[1, 0].semilogy(
                np.arange(1, hist.size + 1),
                np.maximum(hist, 1e-16),
                ":",
                lw=1.0,
                alpha=0.8,
                label=f"picard ω=0.5 T={key} °C",
            )
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel("max |Δμ_i| (-)")
    axes[1, 0].set_title("(c) local fixed point: bisection vs damped Picard")
    axes[1, 0].grid(alpha=0.3, which="both")
    axes[1, 0].legend(fontsize=5.5, ncol=2)

    rep = report["convergence"]["repetitions"]
    cmean = np.array(rep["cumulative_mean"], dtype=float)
    cse = np.array(rep["cumulative_stderr"], dtype=float)
    n = np.arange(1, cmean.size + 1)
    axes[1, 1].plot(n, cmean, "-", color="#2e7d32", label="cumulative mean μ")
    axes[1, 1].plot(n, cmean + cse, "--", color="#808080", lw=0.8)
    axes[1, 1].plot(n, cmean - cse, "--", color="#808080", lw=0.8, label="± s.e.")
    axes[1, 1].set_xlabel("repetitions")
    axes[1, 1].set_ylabel("μ (T = -15 °C)")
    axes[1, 1].set_title("(d) repetition convergence")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / "convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(
        {
            "name": "convergence",
            "path": str(path),
            "caption": "Convergence evidence: (a) FD transient conduction vs analytic 2√(Fo/π) under grid refinement (observed order ≈2), (b) μ vs asperity count, (c) local fixed-point residual: bracket halving of the production bisection solver (solid) versus the damped Picard iteration (dotted, limit cycle at cold temperatures), (d) cumulative mean and standard error over repetitions.",
        }
    )

    # --- 图 4：速度敏感性
    rows = report["velocity_sensitivity"]["rows"]
    velocities = sorted({row["velocity_m_s"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for temperature in sorted({row["temperature_C"] for row in rows}):
        subset = [row for row in rows if row["temperature_C"] == temperature]
        axes[0].plot(
            [row["velocity_m_s"] for row in subset],
            [row["mu_mean"] for row in subset],
            "o-",
            label=f"T={temperature:g} °C",
        )
        axes[1].plot(
            [row["velocity_m_s"] for row in subset],
            [row["lambda_wet_mean"] for row in subset],
            "o-",
            label=f"T={temperature:g} °C",
        )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("sliding velocity v (m/s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("μ")
    axes[0].set_title("friction coefficient vs sliding velocity")
    axes[1].set_ylabel("meltwater coverage λ")
    axes[1].set_title("meltwater coverage vs sliding velocity")
    fig.tight_layout()
    path = figures_dir / "velocity_sensitivity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(
        {
            "name": "velocity_sensitivity",
            "path": str(path),
            "caption": "Sliding-velocity sensitivity (v = 0.25…4 m/s): frictional heating raises λ and lowers μ, reproducing the experimental trend that higher sliding speeds push ice friction into the meltwater-lubricated regime even at cold ambient temperatures.",
        }
    )

    return figures


__all__ = [
    "CRITERIA",
    "DEFAULT_TEMPERATURES_C",
    "ValidationResult",
    "ensemble_size_convergence",
    "fixed_point_convergence",
    "material_property_table",
    "repetition_convergence",
    "run_validation",
]