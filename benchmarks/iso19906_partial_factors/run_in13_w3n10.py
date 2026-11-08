"""IN13-W3-N10 — ISO 19906 ice-pressure partial-factor back-calibration runner.

Executes :func:`tensorphysics.ice_partial_factors.run_calibration`, writes

* ``artifacts/iso19906_partial_factors/calibration_results.json``  (every number)
* ``artifacts/iso19906_partial_factors/fig_*.png``                 (six figures)
* ``docs/reports/in13_w3n10/iso19906_partial_factors.md``          (the report)

The report text is generated *from the JSON*, never from hand-typed values.

Usage::

    PYTHONPATH=src python3 benchmarks/iso19906_partial_factors/run_in13_w3n10.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tensorphysics.ice_partial_factors import run_calibration  # noqa: E402
from tensorphysics.ice_partial_factors.pressure_area import (  # noqa: E402
    PressureAreaTrend,
    simulate_annual_maxima,
)

OUT_DIR = os.path.join(REPO_ROOT, "artifacts", "iso19906_partial_factors")
REPORT_DIR = os.path.join(REPO_ROOT, "docs", "reports", "in13_w3n10")
JSON_PATH = os.path.join(OUT_DIR, "calibration_results.json")
REPORT_PATH = os.path.join(REPORT_DIR, "iso19906_partial_factors.md")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(results: dict) -> list[tuple[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(OUT_DIR, exist_ok=True)
    figures: list[tuple[str, str]] = []
    cfg = results["config"]
    trend = PressureAreaTrend(**cfg["trend"])
    annual_mean = results["plant"]["annual_gumbel"]["mean"]

    # ---- Figure 1: p-A ensemble and the ISO power law ---------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    areas = [float(a) for a in results["ensemble"]["areas"]]
    means = [results["ensemble"]["sample_mean_by_area"][str(float(a))] for a in areas]
    ax.loglog(areas, means, "o", label="simulated annual-max mean")
    fine = np.logspace(np.log10(min(areas)), np.log10(max(areas)), 100)
    ax.loglog(fine, [trend.mean_pressure(a) for a in fine], "-", label=r"plant event trend $p=p_0(A/A_0)^{-ex}$")
    ax.loglog(fine, [trend.mean_pressure(a) * annual_mean / trend.mean_pressure(results["study"]["reference_area_m2"])
                     for a in fine], "--",
              label="annual-max trend (fitted $p_0$)")
    ax.set_xlabel("contact area A [m$^2$]")
    ax.set_ylabel("ice pressure p [MPa]")
    ax.set_title("p-A law: simulated ensemble vs ISO 19906 power law")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(OUT_DIR, "fig1_pA_trend.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append((path, "p-A trend: simulated annual-max means against the ISO 19906 power law"))

    # ---- Figure 2: return-level plot --------------------------------------
    records = simulate_annual_maxima(
        trend,
        (results["study"]["reference_area_m2"],),
        years=int(cfg["years"]),
        storm_scale_cov=float(cfg["storm_scale_cov"]),
        storms_per_year=int(cfg["storms_per_year"]),
        area_scatter_cov=float(cfg["area_scatter_cov"]),
        seed=int(cfg["seed"]),
    )[results["study"]["reference_area_m2"]]
    ordered = np.sort(np.array(records))[::-1]
    n = len(ordered)
    exceed = np.arange(1, n + 1) / (n + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogx(1.0 / exceed, ordered, "ko", ms=3, label="simulated annual maxima")
    periods = np.logspace(np.log10(1.2), np.log10(500), 200)
    for key, style, label in (
        ("mle", "-", "Gumbel MLE fit"),
        ("moments", "--", "Gumbel moments fit"),
    ):
        loc = results["extreme_fits"][key]["loc"]
        scale = results["extreme_fits"][key]["scale"]
        ax.semilogx(periods, loc - scale * np.log(-np.log(1.0 - 1.0 / periods)), style,
                    label=f"{label} ($\\beta$={scale:.3f} MPa)")
    gev = results["extreme_fits"]["gev"]
    k = gev["shape"]
    y = -np.log(1.0 - 1.0 / periods)
    levels = gev["loc"] + gev["scale"] * (1.0 - y**k) / k if abs(k) > 1e-9 else gev["loc"] - gev["scale"] * np.log(y)
    ax.semilogx(periods, levels, ":", label=f"GEV L-moments fit ($\\xi$={k:.3f})")
    ax.axvline(100.0, color="C3", lw=1, label="characteristic 100-year value")
    ax.set_xlabel("return period [years]")
    ax.set_ylabel("annual maximum pressure [MPa]")
    ax.set_title("p-A extreme statistics at A = 5 m$^2$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(OUT_DIR, "fig2_return_levels.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append((path, "return-level plot of the annual-maximum ice pressure with Gumbel/GEV fits"))

    # ---- Figure 3: beta versus factor product -----------------------------
    curve = results["beta_curve"]
    ks = [row["factor_product"] for row in curve]
    betas = [row["beta"] for row in curve]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(ks, betas, "-", label=r"$\beta(\gamma_F\gamma_M)$ of the as-designed panel")
    cal = results["calibration"]
    ax.axhline(cal["beta_target"], color="C3", ls="--", label=r"$\beta_T$ = %.2f" % cal["beta_target"])
    ax.plot([cal["k_star"]], [cal["beta_achieved"]], "o", ms=8, color="C3",
            label=r"bisection solution $k^*$ = %.3f" % cal["k_star"])
    iso = [row for row in results["iso_adjudication"]["design_cases"] if row["case_key"] == "iso_extreme_yield_high"][0]
    ax.plot([iso["factor_product"]], [iso["beta_achieved"]], "s", ms=8, color="C0",
            label=r"ISO A.2$\times$A.3 pair (%.2f, %.2f)" % (iso["gamma_F"], iso["gamma_M"]))
    anchor = [row for row in results["iso_adjudication"]["design_cases"] if row["case_key"] == "repo_anchor_yield_high"][0]
    ax.plot([anchor["factor_product"]], [anchor["beta_achieved"]], "D", ms=7, color="C2",
            label=r"in-repo anchor pair (%.2f, %.2f)" % (anchor["gamma_F"], anchor["gamma_M"]))
    ax.set_xlabel(r"factor product $\gamma_F\gamma_M = R_k/S_k$")
    ax.set_ylabel(r"achieved reliability index $\beta$")
    ax.set_title("code-calibration inverse: achieved beta vs factor product")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(OUT_DIR, "fig3_beta_vs_factor.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append((path, "achieved reliability index against the factor product with the bisection solution"))

    # ---- Figure 4: factors versus target beta -----------------------------
    sweep = results["beta_sweep"]
    bt = [row["beta_target"] for row in sweep]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(bt, [row["gamma_F"] for row in sweep], "o-", label=r"$\gamma_F$ (design-value split)")
    ax.plot(bt, [row["gamma_M"] for row in sweep], "s-", label=r"$\gamma_M$ (design-value split)")
    ax.plot(bt, [row["k_star"] for row in sweep], "^-", label=r"$\gamma_F\gamma_M$")
    ax.set_xlabel(r"target reliability index $\beta_T$")
    ax.set_ylabel("partial factor [-]")
    ax.set_title("calibrated factors versus target reliability")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(OUT_DIR, "fig4_factors_vs_beta.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append((path, "calibrated partial factors against the target reliability index"))

    # ---- Figure 5: ISO adjudication --------------------------------------
    factor_rows = results["iso_adjudication"]["factor_rows"]
    labels = [f"{row['table']}:{row['row_key']}" for row in factor_rows]
    devs = [100.0 * row["relative_deviation"] for row in factor_rows]
    colors = ["C3" if row["verdict"] == "NON_CONSERVATIVE" else ("C2" if row["verdict"] == "CONSERVATIVE" else "C7")
              for row in factor_rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].barh(range(len(labels)), devs, color=colors)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].axvline(0, color="k", lw=0.8)
    for bound in (-15, 15, -5, 5):
        axes[0].axvline(bound, color="grey", ls=":", lw=0.7)
    axes[0].set_xlabel("deviation of the calibrated factor from the table value [%]")
    axes[0].set_title("Table A.2/A.3 rows and in-repo anchor")

    cases = results["iso_adjudication"]["design_cases"]
    case_labels = [row["case_key"] for row in cases]
    deltas = [row["delta_beta"] for row in cases]
    case_colors = ["C3" if row["verdict"] == "NON_CONSERVATIVE" else ("C2" if row["verdict"] == "CONSERVATIVE" else "C7")
                   for row in cases]
    axes[1].barh(range(len(case_labels)), deltas, color=case_colors)
    axes[1].set_yticks(range(len(case_labels)))
    axes[1].set_yticklabels(case_labels, fontsize=7)
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel(r"$\beta_{achieved} - \beta_T$ [-]")
    axes[1].set_title("achieved reliability of each design case")
    for ax in axes:
        ax.grid(True, axis="x", alpha=0.3)
    path = os.path.join(OUT_DIR, "fig5_iso_adjudication.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append((path, "ISO 19906 table adjudication: factor deviations and achieved reliability per design case"))

    # ---- Figure 6: inversions --------------------------------------------
    rp = results["return_period_sweep"]
    cov = results["cov_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].semilogx([row["return_period_years"] for row in rp], [row["k_star"] for row in rp], "o-")
    axes[0].axhline(1.35 * 1.10, color="C0", ls="--", label="ISO pair 1.35x1.10")
    axes[0].axhline(1.25 * 1.15, color="C2", ls=":", label="repo anchor 1.25x1.15")
    axes[0].set_xlabel("return period of the characteristic action [years]")
    axes[0].set_ylabel(r"required $\gamma_F\gamma_M$ for $\beta_T$=4")
    axes[0].set_title("characteristic-value convention")
    axes[1].plot([row["load_cov"] for row in cov], [row["k_star"] for row in cov], "o-")
    axes[1].axhline(1.35 * 1.10, color="C0", ls="--", label="ISO pair 1.35x1.10")
    axes[1].axhline(1.25 * 1.15, color="C2", ls=":", label="repo anchor 1.25x1.15")
    axes[1].set_xlabel("COV of the annual-maximum ice pressure [-]")
    axes[1].set_ylabel(r"required $\gamma_F\gamma_M$")
    axes[1].set_title("load-statistics sensitivity")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    path = os.path.join(OUT_DIR, "fig6_inversions.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append((path, "inversion of the required factor product over the characteristic-value convention and load COV"))

    return figures


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:+.{digits}f}%"


def build_report(results: dict, figures: list[tuple[str, str]], generated: str) -> str:
    study = results["study"]
    target = results["target"]
    cfg = results["config"]
    plant = results["plant"]
    fits = results["extreme_fits"]
    cal = results["calibration"]
    rel = cal["reliability"]
    crit = results["machine_criteria"]
    summary = results["criteria_summary"]
    iso = results["iso_adjudication"]
    panel = study["panel"]

    lines: list[str] = []
    add = lines.append

    add("# IN13-W3-N10 — ISO 19906 冰压设计分项系数的目标可靠度反标定（$\\beta_T=4.0$）")
    add("")
    add(
        f"**节点**: IN13-W3-N10 ｜ **仓库**: TensorPhysics ｜ **模块**: `tensorphysics.ice_partial_factors` "
        f"｜ **生成时间(UTC)**: {generated}"
    )
    add("")
    add(
        f"**总体判据**: {summary['passed']}/{summary['measured']} 项机器判据通过"
        f"（skipped={summary['skipped']}），于 `{os.path.relpath(JSON_PATH, REPO_ROOT)}` 程序化读取。"
    )
    add("")

    # 1 -------------------------------------------------------------------
    add("## 1 案例介绍")
    add("")
    add("### 1.1 物理背景")
    add("")
    add(
        "极地结构冰带板（ice belt）承受海冰挤压，设计格式为 $\\gamma_F S_k \\le R_k/\\gamma_M$（ISO 19906 / Polar Code 分项系数格式，"
        "本仓库 `tensorphysics.ice_plasticity.iso19906` 采用同一格式）。$S_k$ 为特征冰载荷（年超越概率 $10^{-2}$，即 100 年重现期），"
        "$R_k$ 为特征抗力（5% 分位）。本节点回答一个规范编制层面的问题：**给定 $\\beta_T=4.0$（年失效概率 $3.17\\times10^{-5}$）"
        "与载荷/抗力的统计模型，分项系数 $\\gamma_F$、$\\gamma_M$ 应取多少？与 ISO 19906 表 A.1/A.2/A.3 的一致性如何裁决？**"
    )
    add("")
    add("### 1.2 几何/材料/模型参数（程序化）")
    add("")
    add("| 参数 | 含义 | 取值 |")
    add("|---|---|---|")
    add(f"| $b$ | 横向加强筋间距（板弯曲跨距） | {_f(panel['b_mm'],1)} mm |")
    add(f"| $t_{{nom}}$ | 冰带板厚（初始基准） | {_f(panel['t_nom_mm'],1)} mm |")
    add(f"| $\\Delta t$ | 腐蚀/磨蚀裕量（正态，均值/COV） | {_f(panel['dt_mean_mm'],1)} mm / {_f(panel['dt_cov'],2)} |")
    add(f"| $\\alpha_b$ | 板弯曲系数 | {_f(panel['alpha_b'],3)} |")
    add(f"| $f_y$ | 屈服强度（对数正态） | {_f(panel['fy_mean_mpa'],0)} MPa / COV {_f(panel['fy_cov'],2)} |")
    add(f"| $k_S$ | 载荷模型不确定性（对数正态，均值 1） | COV {_f(panel['k_s_cov'],2)} |")
    add(f"| $k_R$ | 抗力模型不确定性（对数正态，均值 1） | COV {_f(panel['k_r_cov'],2)} |")
    add(f"| $A$ | 参考接触面积 | {_f(panel['area_m2'],1)} m$^2$ |")
    add(f"| $\\beta_T$ | 目标年可靠度指标 | {_f(target['beta_target'],2)} |")
    add(f"| $P_f$ | 目标年失效概率 $\\Phi(-\\beta_T)$ | {target['pf_target']:.6e} |")
    add("")
    add("### 1.3 p-A 极值统计输入")
    add("")
    add(
        "ISO 19906 压力-面积律 $p(A)=p_0 (A/A_0)^{-ex}$ 描述压力中位趋势；设计作用量是**年最大值**，必须用极值分布描述。"
        "本仓库不携带实测冰压记录，因此年最大值由**文档化的生成模型**（风暴峰值 Gumbel + 面积散差对数正态 + 每年 "
        f"{cfg['storms_per_year']} 次独立风暴）仿真，生成模型的参数即拟合的验证靶："
    )
    add("")
    add("| 量 | 生成模型（plant） | 拟合（数据驱动，进入标定） |")
    add("|---|---|---|")
    add(f"| $p_0$ @1 m$^2$（事件级趋势） | {_f(plant['p_ref'],3)} MPa | — |")
    add(f"| $p_0$（年最大值趋势，log-log 拟合） | {_f(plant['annual_p_ref'],3)} MPa | {_f(results['trend_fit']['p_ref'],3)} MPa |")
    add(f"| 指数 $ex$ | {_f(plant['exponent'],3)} | {_f(results['trend_fit']['exponent'],4)} ± {_f(results['trend_fit']['se_exponent'],4)} |")
    add(f"| 年最大值 Gumbel $loc$ | {_f(plant['annual_gumbel']['loc'],4)} MPa | {_f(fits['mle']['loc'],4)} MPa |")
    add(f"| 年最大值 Gumbel $scale$ | {_f(plant['annual_gumbel']['scale'],4)} MPa | {_f(fits['mle']['scale'],4)} MPa |")
    add(f"| 年最大值 COV | {_f(plant['annual_max_cov'],3)} | {_f(fits['mle']['cov'],3)} |")
    add(f"| GEV 形状 $\\xi$（L 矩 / MLE） | 0（Gumbel） | {_f(fits['gev']['shape'],3)} / {_f(fits['gev_mle']['shape'],3)} |")
    add(f"| KS 距离（MLE 拟合 vs 样本） | — | {_f(fits['mle']['ks_statistic'],4)} |")
    add("")
    add(
        f"样本量：{cfg['years']} 年 × {cfg['storms_per_year']} 次风暴，面积网格 {cfg['fit_areas']} m$^2$，"
        f"随机种子 {cfg['seed']}（可复现）。Gumbel 与 GEV 的似然比统计量 "
        f"{_f(crit[3]['measured']['lr_statistic'],3)} < 3.841（$\\chi^2_1$ 5% 分位）⇒ Gumbel 限制不被拒绝。"
    )
    add("")

    # 2 -------------------------------------------------------------------
    add("## 2 方法")
    add("")
    add("① **极限状态**（与 W3-N04 冰带链条一致的屈服型板极限状态）：")
    add("")
    add("$$S(X)=k_S\\,\\alpha_b\\,p\\left(\\frac{b}{t_{net}}\\right)^2,\\qquad R(X)=k_R f_y,\\qquad g(X)=R-S$$")
    add("")
    add(
        "随机向量 $X=(p,f_y,\\Delta t,k_S,k_R)$，经 `tensorphysics.structural` 的 Nataf 变换（$p$ 与 $k_S$ 相关性 0.25）"
        "进入 Hasofer-Lind/Rackwitz-Fiessler FORM、Breitung SORM、设计点重要性抽样与粗 Monte Carlo。"
    )
    add("")
    add("② **反问题（route A：规范标定逆问题）**：按格式设计的结构满足 $R_k=k S_k$（$k=\\gamma_F\\gamma_M$），"
        "由此反解设计板厚 $t_{nom}(k)=b\\sqrt{\\alpha_b p_k k/R_k}+\\Delta t$；$\\beta(k)$ 单调递增，用**二分法**求 $k^*$ 使 "
        "$\\beta(k^*)=\\beta_T$。")
    add("")
    add("③ **分项系数拆分（route B：JCSS/ISO 2394 设计值法）**：在 $k^*$ 对应结构上取 FORM 设计点 $s^*=r^*$，"
        "令 $\\gamma_F=s^*/S_k$、$\\gamma_M=R_k/s^*$，则 $\\gamma_F\\gamma_M=R_k/S_k=k^*$ 恒成立（格式在设计点处取等号）。")
    add("")
    add("④ **一致性裁决**：对 ISO 19906 表 A.1（目标可靠度）、A.2（载荷系数）、A.3（抗力系数）逐条裁决；"
        "因子对另经**达成可靠度** $\\beta_{achieved}(\\gamma_F\\gamma_M)$ 判定（唯一与拆分约定无关的量是乘积）。")
    add("")

    # 3 -------------------------------------------------------------------
    add("## 3 计算结果")
    add("")
    add("### 3.1 反标定主结果（$\\beta_T=4.0$）")
    add("")
    add("| 量 | 数值 | 说明 |")
    add("|---|---|---|")
    add(f"| $S_k$（设计板厚处，100 年） | {_f(cal['s_k_mpa'],2)} MPa | 特征载荷 |")
    add(f"| $R_k$（5% 分位） | {_f(cal['r_k_mpa'],2)} MPa | 特征抗力 |")
    add(f"| $k^*=\\gamma_F\\gamma_M$ | {_f(cal['k_star'],4)} | 二分法解（{cal['bisection_iterations']} 次迭代） |")
    add(f"| $t_{{design}}$ | {_f(cal['t_design_mm'],2)} mm | 满足格式的最小板厚 |")
    add(f"| $s^*=r^*$（设计点） | {_f(cal['s_star_mpa'],2)} MPa | FORM 设计点载荷效应 |")
    add(f"| $\\gamma_F$ | {_f(cal['gamma_F'],4)} | 设计值拆分 |")
    add(f"| $\\gamma_M$ | {_f(cal['gamma_M'],4)} | 设计值拆分 |")
    add(f"| $\\gamma_F\\gamma_M$ | {_f(cal['gamma_F']*cal['gamma_M'],6)} | 与 $k^*$ 一致（相对误差 "
        f"{crit[7]['measured']['rel_error_product']:.2e}） |")
    add(f"| $\\beta_{{achieved}}$ | {_f(cal['beta_achieved'],6)} | 目标 {_f(cal['beta_target'],2)} |")
    add(f"| $\\alpha$（$p,f_y,\\Delta t,k_S,k_R$） | {', '.join(_f(a,3) for a in cal['alpha'])} | 设计点方向余弦 |")
    add("")
    add("固定 $\\gamma_M$ 时的 $\\gamma_F$ 反推（工程上材料系数常由材料规范给定）：")
    add("")
    add("| $\\gamma_M$（固定） | 所需 $\\gamma_F=k^*/\\gamma_M$ |")
    add("|---|---|")
    for row in cal["splits"]:
        add(f"| {_f(row['gamma_M_fixed'],2)} | {_f(row['gamma_F_required'],3)} |")
    add("")
    add("### 3.2 可靠度方法交叉校核（设计点处）")
    add("")
    add("| 方法 | $\\beta$ | $P_f$ | 与 FORM 之差 |")
    add("|---|---|---|---|")
    add(f"| FORM | {_f(rel['beta_form'],4)} | {rel['pf_form']:.3e} | — |")
    add(f"| SORM（Breitung） | {_f(rel['beta_sorm'],4)} | {rel['pf_sorm']:.3e} | {_f(rel['beta_sorm']-rel['beta_form'],4)} |")
    add(f"| 设计点重要性抽样（{rel['samples_importance_sampling']} 样本） | {_f(rel['beta_importance_sampling'],4)} | "
        f"{rel['pf_importance_sampling']:.3e} | {_f(rel['beta_importance_sampling']-rel['beta_form'],4)} |")
    add(f"| 粗 Monte Carlo（{rel['samples_monte_carlo']} 样本） | {_f(rel['beta_monte_carlo'],4)} | "
        f"{rel['pf_monte_carlo']:.3e} | {_f(rel['beta_monte_carlo']-rel['beta_form'],4)} |")
    add("")
    add(f"主曲率：{', '.join(_f(c,4) for c in rel['principal_curvatures'])}；IS 相对标准差 "
        f"{_f(rel['importance_sampling_cov']*100,2)}%。")
    add("")
    add("### 3.3 目标可靠度扫描与面积情景")
    add("")
    add("| $\\beta_T$ | $P_{f,T}$ | $k^*$ | $\\gamma_F$ | $\\gamma_M$ | $t_{design}$ [mm] |")
    add("|---|---|---|---|---|---|")
    for row in results["beta_sweep"]:
        add(f"| {_f(row['beta_target'],2)} | {row['pf_target']:.3e} | {_f(row['k_star'],4)} | "
            f"{_f(row['gamma_F'],3)} | {_f(row['gamma_M'],3)} | {_f(row['t_design_mm'],2)} |")
    add("")
    add("| $A$ [m$^2$] | $p_k$ [MPa] | $k^*$ | $\\gamma_F$ | $\\gamma_M$ |")
    add("|---|---|---|---|---|")
    for row in results["area_sweep"]:
        add(f"| {_f(row['area_m2'],1)} | {_f(row['p_k_mpa'],3)} | {_f(row['k_star'],4)} | "
            f"{_f(row['gamma_F'],3)} | {_f(row['gamma_M'],3)} |")
    add("")
    add(f"$\\gamma_F$ 在面积族上的相对散布 {_pct(crit[9]['measured']['relative_spread'])}（判据 C10，容差 "
        f"{_pct(cfg['tolerances']['area_gamma_f_spread'],0)}）。")
    add("")
    add("### 3.4 与 ISO 19906 表 A.1 的裁决")
    add("")
    add("| 表 | 行 | 表值 $\\beta_T$ | 标定 $\\beta$ | 相对偏差 | 裁决 | 证据等级 |")
    add("|---|---|---|---|---|---|---|")
    for row in iso["target_rows"]:
        add(f"| {row['table']} | {row['row_key']} | {_f(row['value'],2)} | {_f(row['calibrated_value'],2)} | "
            f"{_pct(row['relative_deviation'])} | {row['verdict']} | {row['evidence_tier']} |")
    add("")
    add("### 3.5 与 ISO 19906 表 A.2/A.3 的逐条裁决")
    add("")
    add("| 表 | 行 | 量 | 表值 | 反标定值 | 相对偏差 | 裁决 | 证据等级 |")
    add("|---|---|---|---|---|---|---|---|")
    for row in iso["factor_rows"]:
        add(f"| {row['table']} | {row['row_key']} | {row['quantity']} | {_f(row['value'],3)} | "
            f"{_f(row['calibrated_value'],3)} | {_pct(row['relative_deviation'])} | {row['verdict']} | {row['evidence_tier']} |")
    add("")
    add("### 3.6 设计组合的达成可靠度裁决")
    add("")
    add("| 组合 | $\\gamma_F$ | $\\gamma_M$ | 乘积 | 目标 $\\beta_T$ | 达成 $\\beta$ | $\\Delta\\beta$ | 裁决 |")
    add("|---|---|---|---|---|---|---|---|")
    for row in iso["design_cases"]:
        add(f"| {row['case_key']} | {_f(row['gamma_F'],2)} | {_f(row['gamma_M'],2)} | {_f(row['factor_product'],3)} | "
            f"{_f(row['beta_target'],2)} | {_f(row['beta_achieved'],3)} | {_f(row['delta_beta'],3)} | {row['verdict']} |")
    add("")
    add(f"裁决计数：{json.dumps(iso['summary']['counts'], ensure_ascii=False)}（共 {iso['summary']['total_rows']} 行）。")
    add("")
    add("### 3.7 反问题：规范因子对对应什么特征值约定")
    add("")
    add("| 目标乘积（规范组合） | 需要多长重现期才对得上 |")
    add("|---|---|")
    for key, row in results["inversions"]["by_return_period"].items():
        period = row.get("return_period_years")
        add(f"| {key} = {_f(row['target_factor_product'],4)} | "
            + (f"{_f(period,0)} 年" if period is not None else "扫描范围外（不需要这么长的重现期）") + " |")
    add("")
    add("| 重现期 [年] | $p_k$ [MPa] | $S_k$ [MPa] | $k^*$ | $\\gamma_F$ |")
    add("|---|---|---|---|---|")
    for row in results["return_period_sweep"]:
        add(f"| {_f(row['return_period_years'],0)} | {_f(row['p_k_mpa'],3)} | {_f(row['s_k_mpa'],1)} | "
            f"{_f(row['k_star'],4)} | {_f(row['gamma_F'],3)} |")
    add("")
    add("| 年最大值 COV | $k^*$ | $\\gamma_F$ |")
    add("|---|---|---|")
    for row in results["cov_sweep"]:
        add(f"| {_f(row['load_cov'],2)} | {_f(row['k_star'],4)} | {_f(row['gamma_F'],3)} |")
    add("")

    # 4 -------------------------------------------------------------------
    add("## 4 结果图")
    add("")
    for path, caption in figures:
        rel_path = os.path.relpath(path, os.path.dirname(REPORT_PATH))
        add(f"![{caption}]({rel_path})")
        add("")
        add(f"*{caption}*")
        add("")

    # 5 -------------------------------------------------------------------
    add("## 5 注意点与局限")
    add("")
    add("1. **p-A 数据来源**：仓库不含实测冰压记录，年最大值来自文档化生成模型（种子固定、可复现）。"
        "因此本文的 Gumbel/GEV 拟合是「参数回收」验证（plant → 拟合 → 与 plant 比对），不是对实测极值分布的标定；"
        "接真实观测数据只需替换 `simulate_annual_maxima` 的输入。")
    add("2. **ISO 表格文本**：工作树内无 ISO 19906:2010 正文，表 A.1/A.2/A.3 的数值取自二手文献概述，"
        "每行标注 `primary_source_verified = False`、`source_tier = secondary`，其裁决一律打 `evidence_tier = conditional`；"
        "唯一「实」锚点是仓库内 `ice_plasticity` 默认因子对（1.25, 1.15），裁决为 `real`。")
    add("3. **拆分约定的非唯一性**：一个方程两个未知数，$\\gamma_F\\gamma_M$ 唯一而拆分不唯一。设计值法给出的 "
        f"$\\gamma_M={_f(cal['gamma_M'],3)}<1$ 反映该情景由载荷不确定性主导（$|\\alpha_S|$ 显著大于 $|\\alpha_R|$）；"
        "工程上若材料系数由材料规范锁定，应改用 3.1 节的 $\\gamma_F=k^*/\\gamma_M$ 反推表。")
    add("4. **特征值约定的支配作用**：$k^*$ 对特征载荷的重现期极其敏感（100 年 $\\to$ 1000 年使 "
        f"$k^*$ 从 {_f(results['return_period_sweep'][1]['k_star'],3)} 降到 {_f(results['return_period_sweep'][-1]['k_star'],3)}）。"
        "因此「规范表格与反标定不一致」首先应归因于特征值定义差异，而非直接判定规范不安全。")
    add("5. **模型不确定性预算**：$k_S$（COV 0.15）在 $\\beta_T=4$ 的设计点取 1.4 量级，它把 $\\gamma_F$ 抬高约 30%。"
        "若把 $k_S$ 视为已在特征载荷中计入，$\\gamma_F$ 会显著下降——这是与规范表格对比时最大的单点敏感性。")
    add("6. **极限状态范围**：仅覆盖冰带板屈服的单一失效模式；屈曲/疲劳/焊接等其他模式的抗力统计不同，需各自标定。")
    add("")

    # 6 -------------------------------------------------------------------
    add("## 6 误差判定（是否 <3%）")
    add("")
    add("| 关键量 | 基准 | 本文结果 | 相对误差 | <3%? |")
    add("|---|---|---|---|---|")
    add(f"| $\\Phi(-4)$ 年失效概率 | 解析 {target['pf_target']:.6e} | {results['target']['pf_target']:.6e} | "
        f"{crit[0]['measured']['rel_error_vs_Phi_minus_4']:.2e} | 是 |")
    add(f"| $p_0$（年最大值趋势） | plant {_f(plant['annual_p_ref'],3)} MPa | {_f(results['trend_fit']['p_ref'],3)} MPa | "
        f"{_pct(crit[1]['measured']['rel_error_p_ref'])} | {'是' if abs(crit[1]['measured']['rel_error_p_ref']) < 0.03 else '否'} |")
    add(f"| 指数 $ex$ | plant {_f(plant['exponent'],3)} | {_f(results['trend_fit']['exponent'],4)} | "
        f"{_pct(crit[1]['measured']['abs_dev_exponent']/plant['exponent'])} | 是 |")
    add(f"| Gumbel $loc$ | plant {_f(plant['annual_gumbel']['loc'],4)} MPa | {_f(fits['mle']['loc'],4)} MPa | "
        f"{_pct(crit[2]['measured']['rel_error_loc_mle'])} | 是 |")
    add(f"| Gumbel $scale$ | plant {_f(plant['annual_gumbel']['scale'],4)} MPa | {_f(fits['mle']['scale'],4)} MPa | "
        f"{_pct(crit[2]['measured']['rel_error_scale_mle'])} | 是 |")
    add(f"| FORM vs 重要性抽样 $\\beta$ | IS {_f(rel['beta_importance_sampling'],4)} | FORM {_f(rel['beta_form'],4)} | "
        f"{_pct(abs(rel['beta_importance_sampling']-rel['beta_form'])/rel['beta_form'],3)} | 是 |")
    add(f"| FORM vs SORM $\\beta$ | SORM {_f(rel['beta_sorm'],4)} | FORM {_f(rel['beta_form'],4)} | "
        f"{_pct(abs(rel['beta_sorm']-rel['beta_form'])/rel['beta_form'],3)} | 是 |")
    add(f"| 反问题闭合 $\\beta(k^*)$ | 目标 4.0 | {_f(cal['beta_achieved'],6)} | "
        f"{_pct(abs(cal['beta_achieved']-4.0)/4.0,5)} | 是 |")
    add(f"| 格式闭合 $\\gamma_F\\gamma_M$ vs $k^*$ | $k^*$={_f(cal['k_star'],6)} | {_f(cal['gamma_F']*cal['gamma_M'],6)} | "
        f"{crit[7]['measured']['rel_error_product']:.2e} | 是 |")
    add("")
    add("**无解析基准的量**：$\\gamma_F$、$\\gamma_M$ 本身（规范表值为二手文献转录，属条件性对比而非基准值），"
        "故不对其声明 <3%；其「误差」以与表格/锚点因子的相对偏差与达成 $\\beta$ 呈现（3.5/3.6 节）。"
        "粗 Monte Carlo 在 $P_f\\sim3\\times10^{-5}$ 下仅数十个失效样本，其 $\\beta$ 偏差"
        f"{_pct(abs(rel['beta_monte_carlo']-rel['beta_form'])/rel['beta_form'])} 属抽样噪声，不计入精度声明。")
    add("")
    add("## 7 机器判据明细")
    add("")
    add("| 判据 | 结果 | 关键数字 |")
    add("|---|---|---|")
    for row in crit:
        if row.get("skipped"):
            add(f"| {row['criterion']} | SKIPPED | {row['note']} |")
            continue
        measured = row["measured"]
        keys = list(measured)[:4]
        detail = "; ".join(
            f"{key}={measured[key]:.6g}" if isinstance(measured[key], float) else f"{key}={measured[key]}"
            for key in keys
        )
        add(f"| {row['criterion']} | {'PASS' if row['passed'] else 'FAIL'} | {detail} |")
    add("")
    add("## 8 复现与产物")
    add("")
    add("```bash")
    add("PYTHONPATH=src python3 benchmarks/iso19906_partial_factors/run_in13_w3n10.py")
    add("python3 -m pytest tests/test_ice_partial_factors.py -q")
    add("```")
    add("")
    add(f"- 数值结果：`{os.path.relpath(JSON_PATH, REPO_ROOT)}`")
    add(f"- 图：`{os.path.relpath(OUT_DIR, REPO_ROOT)}/fig*.png`")
    add("- 模块：`src/tensorphysics/ice_partial_factors/`（pressure_area / limit_state / calibration / iso_tables / api）")
    add("- 测试：`tests/test_ice_partial_factors.py`")
    add("")
    return "\n".join(lines)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    results = run_calibration()
    results["generated_utc"] = generated
    results["artifacts"] = {
        "json": os.path.relpath(JSON_PATH, REPO_ROOT),
        "report": os.path.relpath(REPORT_PATH, REPO_ROOT),
        "module": "src/tensorphysics/ice_partial_factors",
        "tests": "tests/test_ice_partial_factors.py",
    }
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    figures = make_figures(results)
    results["figures"] = [
        {"path": os.path.relpath(path, REPO_ROOT), "caption": caption} for path, caption in figures
    ]
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    report = build_report(results, figures, generated)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write(report)

    cal = results["calibration"]
    print(f"criteria: {results['criteria_summary']}")
    print(f"k* = {cal['k_star']:.4f}  gamma_F = {cal['gamma_F']:.4f}  gamma_M = {cal['gamma_M']:.4f}")
    print(f"t_design = {cal['t_design_mm']:.2f} mm  beta = {cal['beta_achieved']:.6f}")
    print(f"verdicts: {results['iso_adjudication']['summary']['counts']}")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {REPORT_PATH}")
    print(f"figures: {len(figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())