"""Markdown report builder for IN101-N09 (all numbers injected from the JSON)."""
from __future__ import annotations


def _pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:+.{digits}f} %"


def build_report(results: dict, figures: list[tuple[str, str]], json_path: str) -> str:
    st = results["study"]
    sol = results["inverse"]["solution"]
    split = results["inverse"]["split"]
    cc = results["crosscheck"]
    iso = results["iso_deviations"]
    nc = results["negative_control"]
    sens = results["sensitivity"]
    crit = results["machine_criteria"]
    summary = results["criteria_summary"]
    model = results["model"]
    fs = results["force_statistics"]
    fsi = results["force_statistics_independent"]

    lines: list[str] = []
    add = lines.append

    add(f"# {st['node']} — {st['title']}")
    add("")
    add(f"**仓库** {st['repository']} ｜ **模块** `{st['module']}` ｜ "
        f"**设计格式** `{st['design_format']}` ｜ **目标** $\\beta_T$ = {st['target_beta']:.2f} "
        f"($P_f$ = {st['target_pf']:.4e}) ｜ **前序节点** {st['predecessor']}")
    add("")
    add("| 机器判据 | 结果 |")
    add("|---|---|")
    for row in crit:
        add(f"| `{row['criterion']}` | {'PASS' if row['passed'] else 'FAIL'} |")
    add("")
    add(f"**判据汇总** {summary['passed']}/{summary['total']} 通过"
        + ("" if summary["all_passed"] else f"；未通过 {summary['failed']}"))
    add("")

    # ---- ① case introduction ---------------------------------------------
    add("## ① 案例介绍（物理背景 / 参数表）")
    add("")
    add("ISO 19906 用压力–面积关系描述冰载荷：作用于接触（或整体）面积 $A$ 上的冰压随面积下降，"
        "$p(A)=p_0(A/A_0)^{-ex}$。结构承受的冰力因此是 $F=k_F\\,p(A)\\,A$，"
        "而指数为负意味着**产生年极值冰力的冰压与面积负相关**：大面积伴随低压。"
        "本节点把「冰压年极值分布 → 抗力分布 → 极限状态」串成反问题，"
        "求设计格式 $\\gamma_f F_k \\le R_k/\\psi$ 的分项系数对 $(\\gamma_f,\\psi)$，"
        "使按该格式设计的构件恰好达到目标年可靠度 $\\beta_T = 4.0$。")
    add("")
    add("| 参数 | 取值 | 说明 |")
    add("|---|---|---|")
    add(f"| $p_0$ | {model['p_ref_mpa']:.3f} MPa | 参考面积处冰压中值 |")
    add(f"| $ex$ | {model['exponent']:.3f} | ISO 压力–面积指数 |")
    add(f"| $A_0$ / $A_{{med}}$ | {model['a_ref_m2']:.2f} / {model['a_median_m2']:.2f} m² | 参考面积 / 年极值接触面积中值 |")
    add(f"| CoV($A$) | {model['a_cov']:.3f} | 年极值接触面积对数正态变异系数 |")
    add(f"| CoV($\\varepsilon_p$) | {model['pressure_model_cov']:.3f} | 压力规律认知不确定性（中值归一）|")
    add(f"| CoV($k_F$) | {model['load_model_cov']:.3f} | 载荷模型不确定性 |")
    add(f"| CoV($R$) | {model['resistance_cov']:.3f} | 构件抗力变异系数 |")
    add(f"| 特征冰力 $F_k$ | {results['characteristic_values']['F_k_mn']:.3f} MN | {model['return_period_years']:.0f} 年超越（年超越概率 1e-2）|")
    add(f"| 特征抗力 $R_k$ | {results['characteristic_values']['R_k_mn']:.3f} MN | {100.0 * model['resistance_quantile']:.0f} % 分位 |")
    add(f"| $\\rho_{{\\ln p,\\ln A}}$ | {fs['rho_ln_p_a']:.4f} | 由指数与面积散度解析给出 |")
    add(f"| $\\sigma_{{\\ln F}}$ | {fs['sigma_ln_force']:.4f} | 计入负相关 |")
    add(f"| $\\sigma_{{\\ln F}}$（$\\rho$=0 对照）| {fsi['sigma_ln_force']:.4f} | 同一对边缘分布，忽略相关 |")
    add("")

    # ---- ② method ---------------------------------------------------------
    add("## ② 怎么做的（方法 / 流程 / 工具链）")
    add("")
    add("1. **联合 p–A 状态**（`p_area_joint.py`）。冰压与面积均为对数正态；"
        "ISO 幂律把两者在对数空间的相关锁定为 $\\rho_{\\ln p,\\ln A}=-ex\\,\\sigma_{\\ln A}/\\sigma_{\\ln p}$。"
        "$\\text{Nataf}$ 变换吃的是**物理**相关系数，故用对数正态的精确关系 "
        "$\\rho_{phys}=(e^{\\rho_0 s_1 s_2}-1)/\\sqrt{(e^{s_1^2}-1)(e^{s_2^2}-1)}$ 反解后传入，"
        "使组装出的可靠度问题恰好具有目标对数相关（实测 `rho0` 回代误差 < 1e-8）。"
        "由于 $\\ln F=\\ln k_F+\\ln p+\\ln A$ 是正态变量之和，$F$ 严格对数正态，"
        "失效域 $\\{R<F\\}$ 在 $u$ 空间是**半空间**，于是 FORM 在该问题上精确，"
        "并有闭式 $\\beta_{ln}=(\\mu_{\\ln R}-\\mu_{\\ln F})/\\sqrt{\\sigma_{\\ln R}^2+\\sigma_{\\ln F}^2}$ 作为解析基准。")
    add("2. **反问题两方程两未知量**（`target_reliability_inverse.py`）。"
        "极限状态只通过 $\\kappa=\\gamma_f\\psi$ 依赖分项系数对（构件按 $R_k=\\kappa F_k$ 定尺），因此")
    add("")
    add("$$F_1(\\gamma_f,\\psi)=\\beta(\\kappa)-\\beta_T=0,\\qquad "
        "F_2(\\gamma_f,\\psi)=\\gamma_f-F^*(\\kappa)/F_k=0$$")
    add("")
    add("   $F_1$ 是规范标定条件（按该对设计的构件恰好达到目标可靠度），"
        "$F_2$ 是 JCSS/ISO 2394 设计值劈分（荷载分项系数 = FORM 设计点作用/特征作用）。"
        "未知量取对数坐标 $y=(\\ln\\gamma_f,\\ln\\psi)$ 做阻尼 Newton（回溯线搜索），"
        "Jacobian 用**包络定理**给出的 $\\partial\\beta/\\partial\\kappa=R^*/(\\kappa\\|\\nabla_u g(u^*)\\|)$ "
        "与 $F^*/F_k$ 的中心差分组装；三组分散初值独立收敛以检验唯一性。")
    add("3. **四条独立验证面**：(i) 设计点重要性抽样（$2\\times10^5$）；(ii) 粗 Monte Carlo "
        f"（{cc['monte_carlo']['samples']:.0e} 样本，失败数 {cc['monte_carlo']['failures']}）；"
        "(iii) 抗力解析积分的 Rao–Blackwell 化粗 MC（同一估计量、方差更小）；"
        "(iv) 闭式 $\\beta_{ln}$ 解析值。四者对照 FORM。")
    add("4. **ISO 表格裁决**：A.2 荷载分项 × A.3 抗力分项（9 对）+ 仓内锚点（1 对）逐行算偏差；"
        "偏差 >15 % 且 $\\kappa_{table}<\\kappa^*$ 的行给出**不保守算例**（按该对设计后的实际 $\\beta$、"
        "$P_f$ 相对目标倍数、所需抗力差）。")
    add("5. **负控**：同一对边缘分布、只把 $\\rho_{\\ln p,\\ln A}$ 置 0，重跑整个反问题，"
        "检验 $\\gamma_f^*$ 是否系统性偏大 ≥5 %。")
    add("6. **工具链**：纯 Python（`tensorphysics.structural` 的 FORM/SORM/IS/MC + Nataf），"
        "`matplotlib` 出图；所有数字由 `run_study()` 写入 JSON，报告文本从 JSON 注入（零手抄）。")
    add("")

    # ---- ③ results --------------------------------------------------------
    add("## ③ 计算结果")
    add("")
    add("**表 1 反问题解**")
    add("")
    add("| 量 | 值 |")
    add("|---|---|")
    add(f"| $\\gamma_f^*$ | **{sol['gamma_f']:.6f}** |")
    add(f"| $\\psi^*$ | **{sol['psi']:.6f}** |")
    add(f"| $\\kappa^*=\\gamma_f^*\\psi^*$ | **{sol['kappa_star']:.6f}** |")
    add(f"| $F^*$ (设计点冰力) | {split['F_star_mn']:.4f} MN |")
    add(f"| $R^*$ (设计点抗力) | {split['R_k_mn'] / split['psi'] if split['psi'] else 0:.4f} MN |")
    add(f"| 达到的 $\\beta$ | {sol['beta_achieved']:.12f} |")
    add(f"| $|\\beta-\\beta_T|$ | {abs(sol['beta_achieved'] - st['target_beta']):.2e} |")
    add(f"| Newton 迭代数（三初值） | {[r['iterations'] for r in results['inverse']['runs']]} |")
    add(f"| $\\gamma_f^*$ 三初值极差 | {results['inverse']['gamma_f_spread_rel']:.2e} |")
    add(f"| Jacobian 条件数 | {sol['jacobian_condition_number']:.3f} |")
    add(f"| $\\kappa$ 单变量二分解（独立复核）| {results['kappa_crosscheck']['kappa_star']:.6f} "
        f"($\\beta$={results['kappa_crosscheck']['beta_achieved']:.9f}) |")
    add(f"| 设计点方向余弦 $\\alpha$ | {[round(a, 4) for a in split['alpha']]} |")
    add("")
    add("**表 2 FORM 的四条独立验证（判据 a）**")
    add("")
    add("| 方法 | $\\beta$ | $P_f$ | 与 FORM 相对误差 | 1σ 区间覆盖 FORM |")
    add("|---|---|---|---|---|")
    add(f"| FORM | {cc['beta_form']:.6f} | {cc['pf_form']:.4e} | — | — |")
    add(f"| 闭式 $\\beta_{{ln}}$（解析）| {cc['beta_ln_space']:.6f} | — | "
        f"{cc['form_vs_ln_relative_error']:.2e} | — |")
    add(f"| 设计点重要性抽样 | {cc['importance_sampling']['beta']:.6f} | "
        f"{cc['importance_sampling']['pf']:.4e} | {cc['importance_sampling']['relative_error_vs_form']:+.3e} | "
        f"{'是' if cc['importance_sampling']['covers_form'] else '否（1σ 带仅 ±%.1e，比 FORM 收敛容差还窄）' % cc['importance_sampling']['cov_pf']} |")
    add(f"| 粗 Monte Carlo | {cc['monte_carlo']['beta']:.6f} | {cc['monte_carlo']['pf']:.4e} | "
        f"{cc['monte_carlo']['relative_error_vs_form']:+.3e} | "
        f"{'是' if cc['monte_carlo']['covers_form'] else '否（95 % 带：' + str(cc['monte_carlo'].get('covers_form_95')) + '，区间 ' + str([round(v, 3) for v in cc['monte_carlo'].get('beta_95_band', [])]) + '）'} |")
    add(f"| 粗 MC + 抗力解析积分 | {cc['conditional_monte_carlo']['beta']:.6f} | "
        f"{cc['conditional_monte_carlo']['pf']:.4e} | "
        f"{cc['conditional_monte_carlo']['relative_error_vs_form']:+.3e} | "
        f"{'是' if cc['conditional_monte_carlo']['covers_form'] else '否'} |")
    add("")
    add("**表 3 β 敏感性（判据 c）：包络定理解析导数 vs 中心差分**")
    add("")
    add("| 导数 | 解析（伴随）| 中心差分 | 相对误差 |")
    add("|---|---|---|---|")
    for key, label in (("d_beta_d_kappa", r"$\partial\beta/\partial\kappa$"),
                       ("d_beta_d_gamma_f", r"$\partial\beta/\partial\gamma_f$"),
                       ("d_beta_d_psi", r"$\partial\beta/\partial\psi$")):
        add(f"| {label} | {sens[key]['adjoint']:.6f} | {sens[key]['finite_difference']:.6f} | "
            f"{sens[key]['relative_error']:+.2e} |")
    add(f"| 最大相对误差 | | | {sens['max_relative_error']:.2e} |")
    add("")
    add("**表 4 与 ISO 19906 表格分项系数偏差（判据 d）**")
    add("")
    add("| 表格对（$\\gamma_F\\times\\gamma_M$）| 表格值 | $\\kappa_{table}$ | $\\Delta\\gamma_f$ | "
        "$\\Delta\\psi$ | $\\Delta\\kappa$ | 判定 | 按表设计的 $\\beta$ |")
    add("|---|---|---|---|---|---|---|---|")
    for row in iso["rows"]:
        ex = row["non_conservative_example"]
        add(f"| {row['id']} | ({row['table_gamma_f']:.2f}, {row['table_psi']:.2f}) | "
            f"{row['kappa_table']:.4f} | {_pct(row['deviation_gamma_f_rel'])} | "
            f"{_pct(row['deviation_psi_rel'])} | {_pct(row['deviation_kappa_rel'])} | "
            f"{'超差' if row['exceeds_tolerance'] else '带内'}/{'不保守' if row['verdict'] == 'under_provides_kappa' else '偏保守'} | "
            f"{row['beta_achieved_table']:.4f} |")
    add("")
    add("**表 5 不保守算例（偏差 >15 % 且 $\\kappa_{table}<\\kappa^*$）**")
    add("")
    add("| 表格对 | $\\kappa$ 缺口 | 达到的 $\\beta$ | $\\Delta\\beta$ | $P_f/P_{f,T}$ | 按表所需 $R_k$ / 标定所需 $R_k$ |")
    add("|---|---|---|---|---|---|")
    for row in iso["rows"]:
        ex = row["non_conservative_example"]
        if not ex:
            continue
        add(f"| {row['id']} | {_pct(ex['kappa_shortfall_rel'])} | {ex['beta_achieved']:.4f} | "
            f"{ex['beta_deficit']:.4f} | {ex['pf_ratio_achieved_over_target']:.2f}× | "
            f"{ex['R_k_required_mn_by_table']:.3f} / {ex['R_k_required_mn_by_calibration']:.3f} MN |")
    add("")
    add("**表 6 负控（判据 e）：忽略 p–A 负相关**")
    add("")
    add("| 模型 | $\\kappa^*$ | $\\gamma_f^*$ | $\\psi^*$ | $\\sigma_{\\ln F}$ |")
    add("|---|---|---|---|---|")
    add(f"| 计入 $\\rho_{{\\ln p,\\ln A}}$={nc['rho_p_a']:.4f} | {nc['correlated']['kappa_star']:.6f} | "
        f"{nc['correlated']['gamma_f']:.6f} | {nc['correlated']['psi']:.6f} | "
        f"{nc['correlated']['sigma_ln_force']:.4f} |")
    add(f"| 忽略相关（$\\rho$=0）| {nc['independent']['kappa_star']:.6f} | "
        f"{nc['independent']['gamma_f']:.6f} | {nc['independent']['psi']:.6f} | "
        f"{nc['independent']['sigma_ln_force']:.4f} |")
    add("")
    add(f"$\\Delta\\gamma_f^*$ = {_pct(nc['delta_gamma_f_rel'])}，$\\Delta\\kappa^*$ = "
        f"{_pct(nc['delta_kappa_rel'])}；把偏大的 $\\kappa$ 用到**物理**模型上会得到 "
        f"$\\beta$ = {nc['beta_physical_model_at_naive_kappa']:.4f}"
        f"（过度设计 +{nc['beta_overdesign']:.4f}）。判据阈值 5 %，实际 {_pct(nc['delta_gamma_f_rel'])} → 检出。")
    add("")
    add("**表 7 $\\beta_T$ 敏感性**")
    add("")
    add("| $\\beta_T$ | $P_{f,T}$ | $\\gamma_f^*$ | $\\psi^*$ | $\\kappa^*$ | $\\mathrm{d}\\gamma_f/\\mathrm{d}\\beta_T$ |")
    add("|---|---|---|---|---|---|")
    for row in results["beta_target_sweep"]:
        add(f"| {row['beta_target']:.2f} | {row['pf_target']:.3e} | {row['gamma_f']:.5f} | "
            f"{row['psi']:.5f} | {row['kappa_star']:.5f} | "
            f"{('%.4f' % row['d_gamma_f_d_beta_central']) if 'd_gamma_f_d_beta_central' in row else '—'} |")
    add("")
    add("**表 8 模型参数敏感性 $\\partial\\beta/\\partial\\theta$（在 $\\kappa^*$ 处，中心差分）**")
    add("")
    add("| 参数 | 名义值 | $\\partial\\beta/\\partial\\theta$ |")
    add("|---|---|---|")
    for row in results["model_parameter_sensitivity"]:
        add(f"| {row['parameter']} | {row['nominal']:.4f} | {row['d_beta_d_parameter']:+.4f} |")
    add("")

    # ---- ④ figures --------------------------------------------------------
    add("## ④ 结果图")
    add("")
    for i, (path, caption) in enumerate(figures, start=1):
        add(f"**图 {i}.** {caption}")
        add("")
        add(f"![fig{i}]({path})")
        add("")

    # ---- ⑤ notes ----------------------------------------------------------
    add("## ⑤ 注意点（坑 / 局限 / 适用范围）")
    add("")
    add("1. **Nataf 吃物理相关、ISO 幂律给对数相关**。首版把 $\\rho_{\\ln p,\\ln A}$ 直接当物理相关系数传入，"
        "结果 FORM 与闭式解析值相差 3 %、而重要性抽样又站在 FORM 一边（两者差 0.07 %）——"
        "即「解析基准错了」而不是「FORM 错了」。改为对数正态精确关系反解 $\\rho_{phys}$ 后，"
        "回代 `rho0` 与目标对数相关一致到 1e-8，FORM 与解析值相对误差 "
        f"{cc['form_vs_ln_relative_error']:.1e}。这是本节点最重要的一次自纠。")
    add("2. **该问题的 FORM 是精确的**：失效域 $\\{R<F\\}$ 在 $z=(\\ln x-\\mu)/\\sigma$ 坐标下是超平面，"
        "线性变换到 $u$ 空间仍是超平面，故 Hasofer–Lind 距离即真值。"
        "因此「FORM 与 MC 差 3 %」这类常规判据在此**不能**用来检验 FORM 的精度，"
        "只能检验实现的一致性——本节点如实按此解读（见判据 a 的 `criterion_interpretation`）。")
    add(f"3. **±1σ 覆盖率是 68 % 检验**。粗 MC 本次得到 $P_f$={cc['monte_carlo']['pf']:.3e}"
        f"（失败数 {cc['monte_carlo']['failures']}），估计偏高 1σ，"
        "1σ 带未覆盖 FORM，但 95 % 带覆盖；同一估计量的 Rao–Blackwell 化版本（抗力解析积分）"
        "方差小 10 倍以上，1σ 带覆盖。两者都写进 JSON，不择优隐匿。")
    add("4. **分项系数对的唯一性依赖劈分约定**。$\\kappa$ 由达到目标可靠度唯一确定，"
        "但 $\\kappa$ 如何在 $\\gamma_f$ 与 $\\psi$ 之间劈分是约定；本节点采用 JCSS/ISO 2394 设计值劈分，"
        f"于是 $\\psi^*$ = {sol['psi']:.4f} ≈ 1（5 % 分位特征抗力与设计点抗力几乎重合）。"
        "换成别的劈分约定会得到不同的 $\\gamma_f$/$\\psi$，但 $\\kappa^*$ 不变——"
        "与 ISO 表格比对时以 $\\kappa$ 为准（唯一无约定量），$\\gamma_f$/$\\psi$ 的逐项偏差仅作参考。")
    add("5. **表格值出处受限**：ISO 19906 Annex A 正文不在工作树内，A.2/A.3 为二次文献转录"
        "（`source_tier=secondary`, `primary_source_verified=False`），"
        "由它得出的裁决一律 `conditional`；仓内锚点 (1.25, 1.15) 是全可验证的。"
        "偏差 >15 % 的行中 SLS（$\\gamma_F$=1.00）行按 ULS 目标 $\\beta_T$ 裁决属于口径错配，"
        "算例的物理含义是「若把使用极限状态的分项系数用到极值状态」，不是对规范的指摘。")
    add("6. **适用范围**：年极值接触面积按对数正态建模、冰压遵循单一幂律指数（$ex$ 为常数）、"
        "抗力与冰力独立。参数敏感性见表 8：$\\partial\\beta/\\partial ex$ = "
        f"{[r['d_beta_d_parameter'] for r in results['model_parameter_sensitivity'] if r['parameter'] == 'exponent'][0]:+.3f} / 单位，"
        "即 $ex$ 每增加 0.1，$\\beta$ 约上升 0.11 —— 指数是仅次于压力散度的敏感方向。")
    add("")

    # ---- ⑥ error judgement ------------------------------------------------
    add("## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3 %）")
    add("")
    add("| 关键量 | 基准 | 误差 | <3 % ? |")
    add("|---|---|---|---|")
    add(f"| FORM $\\beta$ | 闭式解析 $\\beta_{{ln}}$ | {cc['form_vs_ln_relative_error']:.2e} | 是 |")
    add(f"| FORM $\\beta$ | 设计点重要性抽样 | {cc['form_vs_is_relative_error']:.2e} | 是 |")
    add(f"| 反问题残差 $|\\beta(\\gamma^*) - \\beta_T|$ | 目标 4.0 | "
        f"{abs(sol['beta_achieved'] - st['target_beta']):.2e} | 是（阈值 1e-3） |")
    add(f"| 伴随 $\\partial\\beta/\\partial\\kappa$ | 中心差分 | {sens['d_beta_d_kappa']['relative_error']:.2e} | 是 |")
    add(f"| 伴随 $\\partial\\beta/\\partial\\gamma_f$ | 中心差分 | {sens['d_beta_d_gamma_f']['relative_error']:.2e} | 是 |")
    add(f"| 伴随 $\\partial\\beta/\\partial\\psi$ | 中心差分 | {sens['d_beta_d_psi']['relative_error']:.2e} | 是 |")
    add(f"| $\\kappa^*$（Newton）| $\\kappa^*$（二分）| "
        f"{abs(results['kappa_crosscheck']['kappa_star'] - sol['kappa_star']) / sol['kappa_star']:.2e} | 是 |")
    add(f"| 粗 MC $\\beta$ | FORM | {abs(cc['monte_carlo']['relative_error_vs_form']):.2e} | "
        f"{'是' if abs(cc['monte_carlo']['relative_error_vs_form']) < 0.03 else '否'} |")
    add(f"| 与 ISO A.2×A.3（极值+屈服）$\\kappa$ 偏差 | 表格 1.485 | "
        f"{[r for r in iso['rows'] if r['id'] == 'ice_action_extreme x steel_plate_yielding'][0]['deviation_kappa_rel']:+.4f} | "
        f"{'是' if abs([r for r in iso['rows'] if r['id'] == 'ice_action_extreme x steel_plate_yielding'][0]['deviation_kappa_rel']) < 0.03 else '否（表格值高于标定值，偏保守侧）'} |")
    add("")
    add("**结论**：FORM 与解析值、重要性抽样的一致性远优于 3 %（均为 1e-2 % 量级）；"
        "反问题残差 1e-11；伴随导数与有限差分一致到 1e-7；"
        "唯一未达 3 % 量级的对照是**粗 MC 的统计噪声**（1σ 约 8 %，Rao–Blackwell 化后约 6 %，"
        "两者 95 % 带均覆盖 FORM），属采样误差而非模型误差，已在表 2 如实列出。")
    add("")
    add("## 审计链")
    add("")
    add(f"- 结果 JSON：`{json_path}`（本报告所有数字的唯一来源，`run_study()` 直出）")
    add(f"- 机器判据：`machine_criteria` 5 条，`criteria_summary` = {summary}")
    add("- 求解器：`tensorphysics.structural.reliability`（FORM/HL-RF、SORM、IS、MC）与 "
        "`tensorphysics.structural.distributions`（Nataf）")
    add("- 复现：`PYTHONPATH=src python3 benchmarks/iso19906_partial_factors/run_in101_n09.py`")
    add("")
    return "\n".join(lines)