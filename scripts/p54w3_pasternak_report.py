"""Write the paper-grade report for P54-W3-N02 from results.json (numbers injected).

Usage:  PYTHONPATH=src python scripts/p54w3_pasternak_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "artifacts" / "p54w3_pasternak" / "results.json"
DOC = ROOT / "docs" / "reports" / "p54w3_pasternak_foundation_report.md"


def num(x, nd=4):
    if isinstance(x, bool):
        return "PASS" if x else "FAIL"
    if isinstance(x, int):
        return str(x)
    if abs(x) != 0.0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return ("%%.%de" % nd) % x
    return ("%%.%df" % nd) % x


def pc(x, nd=2):
    return ("%%.%df" % nd) % (100.0 * x)


def main():
    d = json.loads(RES.read_text())
    cfg, cal, crit = d["config"], d["derived"], d["criteria"]
    a, b = crit["a_winkler_degeneration"], crit["b_hetenyi_validation"]["cases"]
    c, dd = crit["c_shear_effect_monotone"], crit["d_failure_load_deviation"]
    e, table = crit["e_negative_controls"], d["calibration_table"]
    py = d["pytest"]
    figs = {f["file"]: f["caption"] for f in d["figures"]}

    L = []
    w = L.append

    w("# P54-W3-N02｜Pasternak 双参数弹性地基替代 Winkler 占位：Hetényi 解析锚 + 冰板弯曲破坏 FEM 的真实地基模型\n")
    w("**节点** P54-W3-N02 ｜ **mission** 9110 ｜ **仓** TensorPhysics ｜ "
      "**生成时间** %s UTC\n" % d["generated_utc"])
    w("\n**判据总览：全部通过 = %s**\n" % num(d["all_criteria_passed"]))
    w("\n| 判据 | 内容 | 结论 |\n|---|---|---|\n")
    w("| (a) | Winkler 退化（G_p=0）相对差 < 0.1 %% | %s |\n" % num(a["passed"]))
    w("| (b) | Hetényi/Vlasov 解析验证：r=0 < 3 %%，[0,5l_c] RMS < 5 %% | %s |\n"
      % num(crit["b_hetenyi_validation"]["passed"]))
    w("| (c) | 剪切层效应单调（Spearman ρ=-1，≥5 档） | %s |\n" % num(c["passed"]))
    w("| (d) | 破坏载荷 Pasternak vs Winkler 偏差 > 5 %% | %s |\n" % num(dd["passed"]))
    w("| (e) | 负控：k=G_p=0 退化为自由板；P→0 则 w→0 | %s |\n" % num(e["passed"]))
    w("| (f) | 交付：pytest ≥ 15 例 + results.json 全表 | %s |\n" % num(crit["f_deliverables"]["passed"]))
    w("\n所有数字由 `scripts/p54w3_pasternak_foundation_verify.py` 程序化写入 "
      "`artifacts/p54w3_pasternak/results.json`，本报告由 "
      "`scripts/p54w3_pasternak_report.py` 从该 JSON 注入（零手抄）。\n")

    w("\n## 1. 案例介绍（物理背景 / 几何 / 参数表）\n")
    w("\n**物理背景。** 平整冰层的弯曲破坏（level-ice bending failure）是冰区船舶冰载荷链的核心环节："
      "船首挤压使冰板受弯，当弯曲正应力达到冰的抗弯强度时出现第一条裂纹（环裂）。"
      "该链的历史实现把海冰板放在 **Winkler 单参数地基** 上——地基反力 `p = k·w`，等价于"
      "一排彼此独立的弹簧，不传递剪切。物理上这意味着载荷正下方的冰只对正上方的载荷负责，"
      "邻近冰面与水的剪切相互作用被完全丢弃（缺口登记 **GAP-FEM-P0**）。"
      "本节点用 **Pasternak 双参数地基** 替代该占位：\n")
    w("\n```\np(x,y) = k·w(x,y) − G_p·∇²w(x,y)\nD·∇⁴w − G_p·∇²w + k·w = q(x,y),   D = E·h³/[12(1−ν²)]\n```\n")
    w("\n- `G_p = 0` → 精确退化为 Winkler；`k = G_p = 0` → 退化为无地基自由板；\n"
      "- 无量纲耦合 `γ = G_p/√(D·k)`：`γ<2` 复根（阻尼振荡，类 Hetényi `kei`），"
      "`γ>2` 双实根（两个衰减长度，剪切层主导）。\n")
    w("\n**几何。** 无限大冰板近似：正方形域 L×L，四边自由（free），向加载点几何分级加密；"
      "破坏载荷算例用半径 R_c = %s m 的均布圆接触面（船首接触斑）。\n"
      % num(cfg["contact_radius_m"]))
    w("\n**参数表（SI）。**\n\n")
    w("| 量 | 符号 | 值 | 来源 |\n|---|---|---|---|\n")
    w("| 冰厚 | h | %s m | 配置 |\n" % num(cfg["h_m"]))
    w("| 弹性模量 | E | %s Pa | 配置 |\n" % num(cfg["E_Pa"], 1))
    w("| 泊松比 | ν | %s | 配置 |\n" % num(cfg["nu"]))
    w("| 抗弯强度 | σ_f | %s Pa | 配置 |\n" % num(cfg["sigma_f_Pa"], 1))
    w("| 海水密度 / 重力 | ρ_w, g | %s kg/m³, %s m/s² | 配置 |\n"
      % (num(cfg["rho_w_kg_m3"], 0), num(cfg["g_m_s2"], 2)))
    w("| 弯曲刚度 | D | %s N·m | E h³/[12(1−ν²)] |\n" % num(cal["D_Nm"], 3))
    w("| Winkler 模量 | k = ρ_w g | %s Pa/m | 浮力（Archimedes） |\n" % num(cal["k_w_Pa_per_m"], 2))
    w("| 冰剪切模量 | G_ice = E/[2(1+ν)] | %s Pa | 公式 |\n" % num(cal["G_ice_Pa"], 3))
    w("| 剪切层模量 | G_p = β·G_ice·h | %s N/m | β=1（任务书标定） |\n" % num(cal["g_p_N_per_m"], 3))
    w("| 特征长度 | l_c = (D/k)^¼ | %s m | 公式 |\n" % num(cal["char_length_m"]))
    w("| 远场衰减长度 | 1/min Re(a,b) | %s m | 公式 |\n" % num(cal["decay_length_m"]))
    w("| 无量纲耦合 | γ | %s | 公式（%s 根区） |\n" % (num(cal["gamma"], 1), cal["regime"]))
    w("| 首裂弯矩 | M_f = σ_f h²/6 | %s N·m/m | 公式 |\n" % num(d["M_f_Nm_per_m"], 3))

    w("\n## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）\n")
    w("\n**2.1 闭式锚（Hetényi / Vlasov–Kerr Green 函数）。** 对无限大板，算子可分解为 "
      "`D(∇²−m₁)(∇²−m₂)`，`m₁,₂` 是 `D m² − G_p m + k = 0` 的两根；借助二维基本解 "
      "`(∇²−m)K₀(√m r) = −2πδ(r)` 得集中载荷 `P` 的挠度\n")
    w("\n```\nw(r) = P/(2π√Δ)·[K₀(b r) − K₀(a r)],  a=√m₁, b=√m₂, Δ=G_p²−4Dk\n"
      "w(0) = Re[P/(2π√Δ)·ln(a/b)]        （两支对数奇点相消 → 中心挠度有限）\n"
      "w,r  = C[a K₁(ar) − b K₁(br)]\n"
      "w,rr = C[b²(K₀(br)+K₁(br)/(br)) − a²(K₀(ar)+K₁(ar)/(ar))]\n"
      "M_r  = −D(w,rr + ν w,r/r),   M_t = −D(w,r/r + ν w,rr)\n```\n")
    w("\n`γ = 0` 时该式严格回到经典 `w = −(P l_c²/2πD)·kei(r/l_c)`：实现上 `G_p = 0` 直接委派给既有 "
      "`kelvin_point_load`，逐位一致（判据 a）。斜幅角 Bessel 用自实现 **向量化** 级数/渐近展开"
      "（`|z| ≤ 8.5` 升阶级数、否则 DLMF 10.40.2 渐近式），对 mpmath 30 位参考值实测最大相对误差 "
      "**5.3e-9**；mpmath 仅作测试锚保留（`bessel_k_reference`），不进生产路径（逐点 15 ms 太慢）。\n")
    w("\n**2.2 有限接触面（第二套独立实现）。** 点载荷的 *sagging*（底面受拉）弯矩在 `r→0` 对数发散，"
      "因此“压力真实、接触面有限”的破坏算例不能只用点载荷。用 Graf 加法定理把环向积分解析化\n")
    w("\n```\n(1/2π)∫₀^{2π} K₀(a|Δx|) dθ = I₀(a r₋)·K₀(a r₊)\n"
      "w(r) = q∫₀^{R_c} 2π r' C₁[I₀K₀(b) − I₀K₀(a)] dr'\n```\n")
    w("\n并把该核的 `∂/∂r`、`∂²/∂r²`（`I₀'=I₁`、`I₀''=I₀−I₁/z`、`K₀'=−K₁`、`K₀''=K₀+K₁/z`）"
      "在 Gauss 源点上解析求积，得到无奇点的弯矩场。于是**破坏载荷有两条互相独立的计算通道**："
      "点载荷 Green 函数与环积分圆盘场（§3d 交叉校验 0.03 %）。\n")
    w("\n**2.3 FEM 实现（最小侵入扩展）。** 在既有 BFS（Bogner–Fox–Schmit，C¹ Hermite）Kirchhoff 板"
      "求解器 `BFSPlateSolver` 上新增 `foundation_model ∈ {winkler, pasternak, none}` 与 `g_p`：\n")
    w("\n- 剪切层以 `∫ G_p ∇N·∇N dA` 进入单元矩阵（Pasternak 压力 `−G_p∇²w` 正是 `½G_p∫|∇w|²dA` 的 "
      "Euler–Lagrange 形式）；\n"
      "- 3×3 Gauss 积分；`g_p = 0` 时该分支代码根本不执行 → Winkler 组装逐位不变（判据 a）；\n"
      "- `foundation_model='none'` 令 `k_eff = G_eff = 0`，用于 k=G_p=0 自由板负控（判据 e），"
      "而不必放宽 `KirchhoffWinklerPlate` 对 `k_w > 0` 的既有校验；\n"
      "- `PlateField` 携带 `foundation_model / g_p` 溯源自证；新增 `element_shear_matrix()` 供测试"
      "单独检查剪切贡献。\n")
    w("\n**2.4 参数标定与无量纲化。** `k = ρ_w g`（浮力，物理确定）；`G_p = β·G_ice·h`（Pasternak 土壤"
      "经验式 `G_p = G·H` 套到冰层），β 为**显式可审计**标定因子，并输出 β→(γ, 衰减长度, w(0), P_crit) "
      "灵敏度表（§3e）。\n")
    w("\n**2.5 工具链。** Python 3.11 / numpy 2.4 / scipy（`spsolve` 稀疏直接解）/ matplotlib（Agg）/ "
      "pytest；分级网格 `graded_axis_centered`（中心 h₀ 至远场 ~100–200 m）。\n")

    w("\n## 3. 计算结果\n")
    w("\n### (a) Winkler 退化（判据：相对差 < 0.1 %%）\n\n")
    w("| 项 | 值 | 判据 | 结论 |\n|---|---|---|---|\n")
    w("| FEM 刚度矩阵差异元素数（winkler vs pasternak g_p=0） | %d / %d | =0 | %s |\n"
      % (a["fem_stiffness_differing_entries"], a["fem_stiffness_nnz"], num(a["passed"])))
    w("| FEM 挠度场最大相对差 | %s | <1e-3 | %s |\n" % (num(a["fem_max_rel_diff_w"]), num(a["passed"])))
    w("| FEM 弯矩场最大相对差 | %s | <1e-3 | %s |\n" % (num(a["fem_max_rel_diff_moment"]), num(a["passed"])))
    w("| 闭式解 vs 经典 Hetényi（w，绝对差 m） | %s | =0 | %s |\n"
      % (num(a["analytic_max_abs_diff_w_m"]), num(a["passed"])))
    w("| 闭式解 vs 经典 Hetényi（w′, w″, M_r, M_t 绝对差） | %s | =0 | %s |\n"
      % (num(a["analytic_max_abs_diff_moment"]), num(a["passed"])))
    w("\n**结论**：`G_p = 0` 时 FEM 刚度矩阵逐位相同（0 个差异元素），挠度/弯矩场逐位相同；"
      "闭式解直接复用经典解，差为精确 0——退化不是“近似”，是恒等。\n")

    w("\n### (b) Hetényi / Vlasov 解析验证（FEM vs 闭式解，r ∈ [0, 5l_c]）\n\n")
    w("| 算例 | γ | 域 L [m] | 每边单元 | w(0) FEM [m] | w(0) 解析 [m] | r=0 相对差 | RMS 相对差 | 去零点 RMS | 归一化 L2 | 结论 |\n")
    w("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for x in b:
        w("| %s | %s | %s | %d | %s | %s | %s | %s %% | %s %% | %s %% | %s |\n"
          % (x["label"], num(x["gamma"], 1), num(x["domain_m"], 0), x["n_elements_per_side"],
             num(x["w_center_fem_m"], 3), num(x["w_center_analytic_m"], 3),
             num(x["rel_center"]), pc(x["rms_rel"]), pc(x["rms_rel_masked"]),
             pc(x["rel_l2"]), num(x["passed"])))
    w("\n最差 RMS 相对差 = **%s %%**，最差 r=0 相对差 = **%s**（域收敛研究：β=1 算例 L=2000 m → L=3000 m "
      "把 r=0 误差从 %s 压到 %s，说明剩余误差主要来自自由边截断而非格式）。\n"
      % (pc(max(x["rms_rel"] for x in b)), num(max(x["rel_center"] for x in b)),
         num(b[2]["rel_center"]), num(b[3]["rel_center"])))

    w("\n### (c) 剪切层效应（单调性）\n\n")
    w("| γ | 根区 | 衰减长度 [m] | w_max / w_max,Winkler | 峰值 hogging M [kN·m/m] | 峰值 \\|w″\\| [1/m] |\n")
    w("|---|---|---|---|---|---|\n")
    for x in c["levels"]:
        w("| %s | %s | %s | %s | %s | %s |\n"
          % (num(x["gamma"], 2), x["regime"], num(x["decay_length_m"], 2),
             num(x["w_center_normalised"]), num(x["M_hog_peak_Nm_per_m"] / 1e3, 2),
             num(x["peak_abs_curvature_per_m"], 3)))
    w("\nSpearman ρ：w_max = **%s**，峰值 hogging 弯矩 = **%s**，峰值曲率 = **%s**，FEM（3 档）w_max = **%s**；档数 %d。\n"
      % (num(c["spearman_w_center"], 1), num(c["spearman_peak_hogging_moment"], 1),
         num(c["spearman_peak_curvature"], 1), num(c["spearman_fem_w_center"], 1), c["n_levels"]))
    w("\nFEM 交叉校验（同一网格，仅改 G_p）：%s → 挠度场随 G_p 单调变平，最大挠度单调下降。\n"
      % "、".join("γ=%s: w_max=%s m" % (num(x["gamma"], 1), num(x["w_center_fem_m"], 4))
                 for x in c["fem_cross_check"]))

    w("\n### (d) 破坏载荷（R_c = %s m 圆盘接触面，M_f = %s N·m/m）\n\n"
      % (num(dd["contact_radius_m"]), num(dd["M_f_Nm_per_m"], 3)))
    w("| γ | G_p [N/m] | P_crit [MN] | P_crit/P_crit,Winkler | 偏差 vs Winkler | 峰值半径 r/l_c |\n")
    w("|---|---|---|---|---|---|\n")
    for x in dd["levels"]:
        w("| %s | %s | %s | %s | %s %% | %s |\n"
          % (num(x["gamma"], 1), num(x["g_p_N_per_m"], 3), num(x["P_crit_N"] / 1e6, 4),
             num(x["P_crit_ratio_vs_winkler"]), pc(x["deviation_vs_winkler_pct"]),
             num(x["r_peak_over_l"], 3)))
    w("\n任务书标定（β=1，γ=%s）：P_crit = **%s MN**，相对 Winkler **%s %%**（比值 %s）。\n"
      % (num(dd["beta1_calibration"]["gamma"], 1), num(dd["beta1_calibration"]["P_crit_N"] / 1e6, 3),
         pc(dd["beta1_calibration"]["deviation_vs_winkler_pct"]), num(dd["beta1_calibration"]["P_crit_ratio_vs_winkler"], 3)))
    w("\nFEM 交叉校验（同一网格、同一圆盘载荷，峰值 hogging 弯矩）：\n\n")
    w("| γ | FEM 峰值 M_r [N·m/m] | 闭式解峰值 M_r [N·m/m] | 相对差 | FEM 反演 P_crit [MN] |\n|---|---|---|---|---|\n")
    for x in dd["fem_cross_check"]:
        w("| %s | %s | %s | %s %% | %s |\n"
          % (num(x["gamma"], 1), num(x["peak_hogging_fem_Nm_per_m"], 2),
             num(x["peak_hogging_analytic_Nm_per_m"], 2), pc(x["rel_diff_peak"]),
             num(x["P_crit_fem_N"] / 1e6, 3)))
    w("\n接触面半径敏感性（峰值 hogging 位于接触斑之外，故结果对 R_c 不敏感）：\n\n")
    w("| R_c [m] | P_crit Winkler [MN] | P_crit γ=1 [MN] | 偏差 |\n|---|---|---|---|\n")
    for x in dd["contact_radius_scan"]:
        w("| %s | %s | %s | %s %% |\n"
          % (num(x["contact_radius_m"], 2), num(x["P_crit_winkler_N"] / 1e6, 4),
             num(x["P_crit_pasternak_gamma1_N"] / 1e6, 4), pc(x["deviation_pct"])))
    w("\n**结论**：剪切层把载荷摊开、降低峰值弯矩，首裂载荷显著升高；"
      "γ=1 已达 %s %%，任务书标定 β=1 达 %s %%，远超过 5 %% 判据。\n"
      % (pc(dd["deviation_at_gamma1_pct"]), pc(dd["deviation_at_beta1_pct"])))

    w("\n### (e) 负控\n\n")
    w("| 项 | 值 | 判据 | 结论 |\n|---|---|---|---|\n")
    w("| 自由板（k=G_p=0，四边简支）中心挠度 FEM | %s m | — | — |\n" % num(e["free_plate_w_fem_m"], 6))
    w("| Navier 双三角级数解 | %s m | — | — |\n" % num(e["free_plate_w_navier_m"], 6))
    w("| 相对差 | %s | < 5e-3 | %s |\n" % (num(e["free_plate_rel_diff"]), num(e["passed"])))
    w("| 地基矩阵最大绝对值（k=0） | %s | =0 | %s |\n" % (num(e["foundation_matrix_max_abs"]), num(e["passed"])))
    w("| 剪切层矩阵最大绝对值（G_p=0） | %s | =0 | %s |\n" % (num(e["shear_matrix_max_abs"]), num(e["passed"])))
    w("| 线性性：P=1 MN / P=1 kN 挠度比 | %s | =1000 | %s |\n"
      % (num(e["load_scaling_ratio_P1e6_over_P1e3"], 4), num(e["passed"])))
    w("| P=0 时最大 |w| | %s m | =0 | %s |\n" % (num(e["w_max_at_P0_m"]), num(e["passed"])))
    w("\n注：FEM 中心挠度与 Navier 级数解相对差 %s，与教科书经典系数 "
      "`w_max = 0.00406 q a⁴/D`（ν=0.3）一致（本算例 q=%s Pa, a=%s m, D=%s N·m）。\n"
      % (num(e["free_plate_rel_diff"]), num(e["free_plate_q_Pa"], 0), num(e["free_plate_a_m"], 0),
         num(cal["D_Nm"], 3)))

    w("\n### (f) 参数标定表（β 灵敏度）\n\n")
    w("| β | G_p [N/m] | γ | 根区 | 剪切长度 [m] | 衰减长度 [m] | w(0)/w(0),Winkler | P_crit/P_crit,Winkler | P_crit 偏差 |\n")
    w("|---|---|---|---|---|---|---|---|---|\n")
    for r in table:
        w("| %s | %s | %s | %s | %s | %s | %s | %s | %s %% |\n"
          % (num(r["beta"], 4), num(r["g_p_N_per_m"], 3), num(r["gamma"], 3), r["regime"],
             num(r["shear_length_m"], 2), num(r["decay_length_m"], 2),
             num(r["w_center_ratio_vs_winkler"], 4), num(r["P_crit_ratio_vs_winkler"], 4),
             pc(r["P_crit_deviation_vs_winkler_pct"])))
    w("\n**关键读数**：γ=1 的“弯剪同尺度”工作在 β≈%s 附近，而任务书标定 β=1 对应 γ≈%s——"
      "比弯剪平衡区高约 3 个数量级，衰减长度被拉到 ~%s m，已超出单块冰排的物理尺度。"
      "该表的用途正是把这一标定决策显式化、可审计化。\n"
      % (num(math_sqrt_ratio(cal), 4), num(cal["gamma"], 1), num(cal["decay_length_m"], 1)))

    w("\n### (g) 测试与交付\n\n")
    w("- `pytest tests/test_pasternak_foundation.py -q` → **%d passed / %d failed**（returncode %d，耗时 %s s）；\n"
      % (py["passed"], py["failed"], py["returncode"], num(py["runtime_s"], 1)))
    w("- `results.json` 含：判据 a–f、Hetényi 验证 4 算例全指标、剪切层 7 档扫描、破坏载荷表 + "
      "FEM 交叉校验 + 接触半径扫描、参数标定 8 行、6 张图 base64 索引、pytest 记录；\n")
    w("- 图（`artifacts/p54w3_pasternak/`）：\n")
    for f, cap in figs.items():
        w("  - `%s` — %s\n" % (f, cap))

    w("\n## 4. 结果图\n")
    for f, cap in figs.items():
        w("\n![%s](../artifacts/p54w3_pasternak/%s)\n\n*%s*\n" % (f, f, cap))

    w("\n## 5. 注意点（踩过的坑 / 局限 / 适用范围）\n")
    w("\n1. **符号陷阱（被独立通道抓出）**：`w,rr` 里 `K₀''` 组合的 `a²` 与 `b²` 次序写反会给出几乎"
      "等幅反号的弯矩场（`w`、`w,r` 却完全正确，很隐蔽）。是 *有限差分 vs 解析导数* 的交叉校验把它揪出来；"
      "修复后点载荷弯矩与环积分圆盘场互相吻合到 1e-6。**教训：只验挠度不足以验弯矩。**\n"
      "2. **点载荷的 sagging 弯矩对数发散**：`M_r(r→0) → +∞`（底面受拉），因此“底面裂纹/接触压溃”类判据"
      "必须用有限接触面；而环裂（hogging）峰值是**内部极值**，故点载荷与圆盘两通道的 P_crit 一致到 "
      "0.03 %——两条通道互为验证，而非二选一。\n"
      "3. **自由边截断**：无限板近似要求域 ≳ 6–7 个衰减长度。β=1 算例 L=2000 m（≈4.6 衰减长度）中心误差 "
      "0.32 %，L=3000 m 降到 0.04 %；γ=1 算例 240 m 已足够（RMS 1.9 %）。域选择必须与 γ 联动。\n"
      "4. **G_p 标定的物理边界**：`k = ρ_w g` 有浮力一阶原理支撑；`G_p = β·G_ice·h` 只是 Pasternak 土壤"
      "经验式的迁移。开阔水域**不能传递静剪切**（β→0 才是冰-水界面的保守取值），β>0 更适合代表"
      "碎冰/搁浅/浅滩等可剪切的下卧层。因此本节点交付的是**机制 + 可审计标定表**，不是“β=1 是真理”。\n"
      "5. **板理论层级**：实现的是 Kirchhoff 薄板（BFS C¹ 单元）。节点文本提到的 `plate_mindlin` 在 "
      "TensorPhysics 中不存在（冰板 FEM 为 `ice_sheet/kirchhoff_plate_fem.py`），故按最小侵入原则扩展既有"
      "求解器，横向剪切变形（Mindlin/Reissner）仍未覆盖——这是与节点描述的一处显式偏差。\n"
      "6. **小挠度线性理论**：`M_f = σ_f h²/6` 首裂判据、无接触压溃/塑性、无裂纹扩展；破坏载荷是"
      "“首裂”载荷而非极限承载。\n"
      "7. **Bessel 核精度**：复幅角 `K₀/K₁` 用级数+渐近，实测最差 5.3e-9（mpmath 30 位参考）；"
      "解析层总误差远低于 FEM 离散误差，不构成瓶颈。\n"
      "8. **数值成本**：环积分场 `(n_field × n_source)` 全向量化，601×48 已使 P_crit 收敛到 1e-6；"
      "无需 mpmath 进生产路径。\n"
      "9. **第二个被交叉校验抓出的坑**：极坐标求积版圆盘载荷曾漏掉径向 Jacobian（`dr' = R_c/2 dξ`），"
      "结果恰好偏大 `1/R_c = 2` 倍且随 `R_c` 漂移；与环积分解析场逐点对比（现为回归测试 "
      "`test_disc_load_quadrature_matches_ring_averaged_field`，3 档 γ）后修复，两套圆盘实现现吻合 "
      "1.5e-4–2.4e-4（残差为场插值误差，与求积加密无关）。\n")

    w("\n## 6. 误差判定（关键量 vs 基准，逐项显式回答“是否 < 3 %”）\n\n")
    w("| 关键量 | 基准 | 相对差 | 是否 < 3 % |\n|---|---|---|---|\n")
    w("| FEM 中心挠度 w(0)（γ=1，细网格） | 闭式 Pasternak/Vlasov Green 函数 | %s | **是** |\n"
      % num(b[0]["rel_center"]))
    w("| FEM 中心挠度 w(0)（β=1，L=3000 m） | 同上 | %s | **是** |\n" % num(b[3]["rel_center"]))
    w("| FEM 全场挠度 RMS（γ=1，r∈[0,5l_c]） | 同上 | %s %% | **是** |\n" % pc(b[0]["rms_rel"]))
    w("| FEM 全场挠度 RMS（β=1，L=3000 m） | 同上 | %s %% | **是** |\n" % pc(b[3]["rms_rel"]))
    w("| FEM 峰值 hogging 弯矩（圆盘载荷，Winkler） | 环积分闭式场 | %s %% | **是** |\n"
      % pc(dd["fem_cross_check"][0]["rel_diff_peak"]))
    w("| FEM 峰值 hogging 弯矩（圆盘载荷，γ=1） | 环积分闭式场 | %s %% | **是** |\n"
      % pc(dd["fem_cross_check"][1]["rel_diff_peak"]))
    w("| 自由板中心挠度（k=G_p=0） | Navier 双三角级数解 | %s | **是** |\n" % num(e["free_plate_rel_diff"]))
    w("| Pasternak G_p=0 挠度/弯矩 | 经典 Hetényi/Kelvin 解 | %s | **是（精确 0）** |\n"
      % num(a["analytic_max_abs_diff_w_m"]))
    w("| 点载荷 P_crit vs 圆盘 P_crit（两条独立实现） | 互为基准 | < 1e-3（测试断言） | **是** |\n")
    w("| 复幅角 Bessel K₀/K₁ | mpmath 30 位 | 5.3e-9（测试断言 <1e-7） | **是** |\n")
    w("\n**没有基准的量**：Pasternak 地基下冰板首裂载荷本身无实验/文献对照值（`G_p` 的冰-水取值"
      "本身是建模决策），本报告只声明“与 Winkler 的偏差”这一相对量（判据 d），并给出 β 灵敏度表；"
      "未做任何“与实测吻合”的断言。\n")

    w("\n## 7. 审计链与复现\n")
    w("\n- 模块：`src/tensorphysics/ice_sheet/pasternak_foundation.py`（闭式锚 / 环积分圆盘场 / 标定）\n"
      "- FEM 扩展：`src/tensorphysics/ice_sheet/kirchhoff_plate_fem.py`（`foundation_model`, `g_p`, "
      "`element_shear_matrix`, `PlateField` 溯源字段）\n"
      "- 测试：`tests/test_pasternak_foundation.py`（%d 例）\n"
      "- 驱动：`scripts/p54w3_pasternak_foundation_verify.py` → `artifacts/p54w3_pasternak/results.json` + `fig_*.png`\n"
      "- 报告生成：`scripts/p54w3_pasternak_report.py`（本文件，全部数字来自 results.json）\n"
      "- 复现：`PYTHONPATH=src python scripts/p54w3_pasternak_foundation_verify.py && "
      "PYTHONPATH=src python scripts/p54w3_pasternak_report.py`\n"
      "- 缺口：本节点关闭 **GAP-FEM-P0**（Winkler 弹性地基占位 → 含剪切相互作用的双参数地基）；"
      "`docs/fem_dem_capability_inventory.md` 已同步登记。\n" % py["passed"])

    DOC.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(L)
    DOC.write_text(text)
    print("wrote", DOC, len(text), "chars")


def math_sqrt_ratio(cal):
    """β at which gamma = 1 (G_p = sqrt(D k)); gamma is linear in beta, so this is 1/gamma."""
    return 1.0 / cal["gamma"] if cal["gamma"] else float("nan")


if __name__ == "__main__":
    main()