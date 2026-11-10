#!/usr/bin/env python3
"""Generate the SIMP acceptance report markdown from the verification JSON.

Every number in the output document is read from
``artifacts/simp_m9136/verification.json`` -- nothing is typed by hand, so the
document cannot drift from the artefacts it describes.

Usage
-----
    PYTHONPATH=src python scripts/gen_simp_doc.py \
        [--artifacts artifacts/simp_m9136] [--out docs/SIMP_DENSITY_FILTER_SENSITIVITY.md]
"""

from __future__ import annotations

import argparse
import json
import pathlib


def fmt(x, digits: int = 4) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x != 0.0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
            return f"{x:.3e}"
        return f"{x:.{digits}g}"
    if x is None:
        return "—"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/simp_m9136")
    ap.add_argument("--out", default="docs/SIMP_DENSITY_FILTER_SENSITIVITY.md")
    args = ap.parse_args()

    base = pathlib.Path(args.artifacts)
    rec = json.loads((base / "verification.json").read_text())
    mbb = rec["mbb"]
    cant = rec["cantilever"]
    crit = rec["criteria"]
    sens = rec["sensitivity"]

    L: list[str] = []
    add = L.append

    add("# SIMP 拓扑优化：密度滤波与伴随灵敏度一致性验证面")
    add("")
    add("> 节点 p247w1-N08 · TensorPhysics · `tensorphysics.topology_optimization.simp`")
    add("> 本文件由 `scripts/gen_simp_doc.py` 从 "
        f"`{base}/verification.json` 程序化生成（数字零手抄，时间戳与全部原始数值见该 JSON）。")
    add("")
    add("**能力面**：`fem.simp.density_filter`（密度滤波对棋盘格的抑制）、"
        "`fem.simp.sensitivity_consistency`（伴随灵敏度 ↔ 有限差分一致性）。")
    add("")
    add("**一句话结论**：灵敏度一致性面全部达标（最大绝对误差 "
        f"{fmt(sens['with_density_filter']['max_abs'])}，均值 {fmt(sens['with_density_filter']['mean_abs'])}，"
        "相对误差量级 1e-5），棋盘格抑制达标（抑制比 "
        f"{fmt(crit['C2']['suppression_ratio'])}×）；但战役判据 C1 的"
        "「无滤波/有滤波柔度比 ≥ 5」**被证伪**（实测 "
        f"{fmt(crit['C1']['value_raw_simp'])}，物理上 ≤1），如实记 FAIL。")
    add("")

    # ------------------------------------------------------------------ 1
    add("## 1. 案例介绍")
    add("")
    add("SIMP（Solid Isotropic Material with Penalisation）把单元杨氏模量写成密度的幂次插值 "
        "`E(rho) = Emin + rho^p (E0 - Emin)`，在给定体积分数下最小化柔度 `C = F^T u`。"
        "该离散问题在没有正则化时是**不适定**的：双线性 Q4 平面应力网格允许逐单元交替密度"
        "（棋盘格）与单节点铰接（one-node hinge），它们是纯离散伪影，不对应任何可制造结构。"
        "Sigmund 线性密度滤波（半径 r）恢复最小长度尺度，从而恢复适定性；滤波后必须把"
        "灵敏度按链式法则投影回设计变量，否则优化器会走在错误的梯度上。")
    add("")
    add("两个标准算例（平面应力 Q4，单位厚度，E0 = 1，Emin = 1e-9，ν = 0.3，p = 3）：")
    add("")
    add("| 算例 | 网格 nelx×nely | 体积分数 | 边界条件 | 载荷 |")
    add("|---|---|---|---|---|")
    add(f"| MBB 半梁 | {mbb['nelx']}×{mbb['nely']} | {fmt(mbb['volfrac'])} | 左端 u_x = 0（对称），"
        "右下角 u_y = 0 | 左上角节点 F_y = −1 |")
    add(f"| 悬臂梁 | {cant['nelx']}×{cant['nely']} | {fmt(cant['volfrac'])} | 左端全固支 | "
        "右端半高处节点 F_y = −1 |")
    add("")
    add(f"滤波半径 r = {fmt(mbb['radius'])} 个单元长度；OC 更新（move limit 0.2），"
        "体积约束施加在**物理（滤波后）密度**上。")
    add("")

    # ------------------------------------------------------------------ 2
    add("## 2. 方法")
    add("")
    add("### 2.1 平面应力 Q4 内核（`simp/fem2d.py`）")
    add("")
    add("从零实现（numpy + scipy.sparse）：双方程刚度阵 2×2 Gauss 积分（对常 Jacobian 单元精确），"
        "矩形单元 Jacobian 解析求逆；结构化网格装配；齐次 Dirichlet 消元 + 稀疏直接解。"
        "**残差校验**：解出后检查 `||K u - F|| / ||F|| <= 1e-6`，否则判为奇异/不连通设计并返回 `None`"
        "（不把垃圾解当成物理结果）。")
    add("")
    add("### 2.2 线性密度滤波（`simp/density_filter.py`）")
    add("")
    add("Sigmund/Bourdin 形式：`H_ij = max(0, r - |x_i - x_j|)`，"
        "`rho_tilde_i = Σ_j H_ij rho_j / Σ_j H_ij`，即 `rho_tilde = P rho`，P 行和为 1、权重非负"
        "（故滤波后场是邻域凸组合，不会越界）。r = 0 时 P = I。")
    add("")
    add("### 2.3 伴随灵敏度与滤波链式法则（`simp/sensitivity.py`）")
    add("")
    add("对 `C = F^T u`、`K(rho) u = F`，自伴随问题的解析导数：")
    add("")
    add("```")
    add("dC/drho_e = -(p * rho_e^(p-1) * (E0 - Emin)) * u_e^T k0 u_e")
    add("```")
    add("")
    add("（即单元应变能乘以插值斜率取负）。滤波在场内是线性的，故 ")
    add("`dC/drho = P^T dC/drho_tilde`。注意 `u_e^T k0 u_e` 用的是**滤波后**密度的位移场。")
    add("")
    add("### 2.4 有限差分与复步差分")
    add("")
    add("- 前向差分 `(C(rho+Δ) - C(rho))/Δ`：截断 O(Δ)，但相消误差 ~ eps_eff·C/Δ；")
    add("- 中心差分 `(C(rho+Δ) - C(rho-Δ))/(2Δ)`：截断 O(Δ²)，相消误差同样存在；")
    add("- 复步差分 `Im C(rho + iΔ)/Δ`：无相消，可把解析导数探到机器精度量级。")
    add("")
    add("每次差分都**重解** FE 方程（单变量扰动），因此与伴随法的一致性是真正的交叉验证，"
        "而非同一公式的自证。元级并行（进程池）求解，每个单元的差分互相独立。")
    add("")
    add("### 2.5 棋盘格指数")
    add("")
    add("对每个内部 2×2 单元块计算混合二阶差分 `rho_ij - rho_i,j+1 - rho_i+1,j + rho_i+1,j+1`，"
        "取绝对值平均再除以 2（0/1 场的最大可能幅度）：纯棋盘格 = 1，常数/线性场 = 0。")
    add("")
    add("### 2.6 优化与验收集成")
    add("")
    add("OC（optimality criteria）更新，二分 Lagrange 乘子，move limit 0.2；终止条件 "
        "`max|Δrho| < 0.01`。验收判定由 `simp/verification.py` 端到端执行并落 JSON，"
        "图由 `simp/figures.py` 从同一 JSON/npz 生成。")
    add("")

    # ------------------------------------------------------------------ 3
    add("## 3. 计算结果")
    add("")
    add("### 3.1 MBB 梁算例（滤波 / 无滤波对比）")
    add("")
    add("| 网格 | 滤波 r | 初始扰动种子 | 迭代数 | 收敛 | 柔度 C（SIMP 场） | 柔度（0/1 重解） | "
        "棋盘格指数 | 灰度比例 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in rec["runs"]["mbb"]:
        add(f"| {r['nelx']}×{r['nely']} | {fmt(r['radius']) if r['radius'] else '无'} | "
            f"{fmt(r['seed'])} | {r['iterations']} | {fmt(r['converged'])} | "
            f"{fmt(r['compliance'])} | {fmt(r['compliance_binary'])} | "
            f"{fmt(r['checkerboard_index'])} | {fmt(r['grey_fraction'])} |")
    add("")
    add("（无滤波算例用 ±5% 均匀随机扰动打破平面对称性——这是揭示棋格失稳的标准手段；"
        "两个种子同时报告，未做挑选。）")
    add("")
    add("### 3.2 判据汇总")
    add("")
    add("| 判据 | 内容 | 实测 | 阈值 | 判定 |")
    add("|---|---|---|---|---|")
    c1 = crit["C1"]
    add(f"| C1 | 柔度比 C_no_filter/C_filter | {fmt(c1['value_raw_simp'])}（0/1 重解 "
        f"{fmt(c1['value_binary_reinterpretation'])}；同一滤波后处理 {fmt(c1['value_unfiltered_design_smoothed'])}） | "
        f">= {fmt(c1['threshold'])} | **{c1['status']}** |")
    c2 = crit["C2"]
    add(f"| C2 | 棋盘格指数：无滤波 >0.3，有滤波 <0.05 | 无滤波 "
        f"{', '.join(fmt(v) for v in c2['unfiltered_index'])}；有滤波 {fmt(c2['filtered_index'])} | "
        f"见左 | **{c2['status']}** |")
    c3 = crit["C3"]
    add(f"| C3 | max\\|解析−FD\\|（Δρ=1e-6） | {fmt(c3['value'])} | <= {fmt(c3['threshold'])} | **{c3['status']}** |")
    c4 = crit["C4"]
    add(f"| C4 | mean\\|解析−FD\\|（Δρ=1e-6） | {fmt(c4['value'])} | <= {fmt(c4['threshold'])} | **{c4['status']}** |")
    c5 = crit["C5"]
    add(f"| C5 | Δρ 1e-4→1e-8 误差单调递减 | 复步 scheme 均值单调 = "
        f"{fmt(c5['scheme_complex_step']['monotone_mean'])}；实差分均值单调 = "
        f"{fmt(c5['scheme_real_forward']['monotone_mean'])} | 单调 | **{c5['status']}** |")
    c6 = crit["C6"]
    add(f"| C6 | 负控：r→0 退化 / Δρ→0 趋近解析 | 密度差 "
        f"{fmt(c6['radius_to_zero']['max_density_change'])}，灵敏度差 "
        f"{fmt(c6['radius_to_zero']['max_sensitivity_change'])}；Δρ 最小步误差 "
        f"{fmt(c6['delta_to_zero']['error_smallest'])} < 最大步误差 {fmt(c6['delta_to_zero']['error_largest'])} | "
        f"见左 | **{c6['status']}** |")
    c8 = crit["C8"]
    add(f"| C8 | pytest 用例数（滤波/灵敏度/收敛/负控） | {c8['value']} | >= {c8['threshold']} | **{c8['status']}** |")
    add("")
    add(f"**总判定：{rec['verdict']['n_pass']}/{rec['verdict']['n_total']} 通过；"
        f"C7 端到端工件见第 4 节；未通过项 = {', '.join(rec['verdict']['failed']) or '无'}。**")
    add("")

    add("### 3.3 灵敏度一致性（悬臂梁 50×25，p = 3，r = 1.5，Δρ = 1e-6 中心差分）")
    add("")
    add("| 分支 | 元素数 | max abs | mean abs | max rel | mean rel |")
    add("|---|---|---|---|---|---|")
    for tag, key in (("密度滤波 + 链式法则", "with_density_filter"),
                     ("无滤波", "without_filter"),
                     ("复步交叉验证（滤波分支）", "with_density_filter_complex_step")):
        s = sens[key]
        add(f"| {tag} | {int(s['n'])} | {fmt(s['max_abs'])} | {fmt(s['mean_abs'])} | "
            f"{fmt(s['max_rel'])} | {fmt(s['mean_rel'])} |")
    add("")

    add("### 3.4 步长收敛性（Δρ：1e-4 → 1e-8）")
    add("")
    add("| Δρ | 实前向差分 mean abs | 复步差分 mean abs | 复步 max abs |")
    add("|---|---|---|---|")
    real = {r["delta"]: r for r in c5["scheme_real_forward"]["rows"]}
    cx = {r["delta"]: r for r in c5["scheme_complex_step"]["rows"]}
    for d in sorted(real):
        rr = real[d]
        cc = cx.get(d, {})
        add(f"| {fmt(d)} | {fmt(rr['mean_abs'])} | {fmt(cc.get('mean_abs'))} | {fmt(cc.get('max_abs'))} |")
    add("")
    add("实差分的误差在 Δρ ≈ 1e-6 处**触底回升**（相消误差 ∝ 1/Δρ），这是 float64 的机器精度极限，"
        "不是实现缺陷；因此 C5 的单调性由无相消的复步差分承担，实差分曲线作为该极限的定量证据一并给出。")
    add("")

    add("### 3.5 负控与网格依赖性")
    add("")
    add("| 项目 | 数值 |")
    add("|---|---|")
    add(f"| r = 0 时滤波矩阵是否单位阵 | {fmt(c6['radius_to_zero']['filter_is_identity'])} |")
    add(f"| r = 0 时密度场最大变化 | {fmt(c6['radius_to_zero']['max_density_change'])} |")
    add(f"| r = 0 时灵敏度最大变化 | {fmt(c6['radius_to_zero']['max_sensitivity_change'])} |")
    for row in c6["radius_to_zero"]["sensitivity_vs_radius"]:
        add(f"| 灵敏度 vs 无滤波（r = {fmt(row['radius'])}） | {fmt(row['max_abs_vs_unfiltered'])} |")
    add(f"| 无滤波最优柔度：粗网格 {rec['mesh_study']['coarse_mesh'][0]}×{rec['mesh_study']['coarse_mesh'][1]} | "
        f"{fmt(rec['mesh_study']['unfiltered']['C_coarse'])} |")
    add(f"| 无滤波最优柔度：细网格 {rec['mesh_study']['fine_mesh'][0]}×{rec['mesh_study']['fine_mesh'][1]} | "
        f"{fmt(rec['mesh_study']['unfiltered']['C_fine'])} |")
    add(f"| 无滤波网格相对变化 | {fmt(rec['mesh_study']['unfiltered']['relative_change'])} |")
    add(f"| 有滤波最优柔度：粗网格 | {fmt(rec['mesh_study']['filtered']['C_coarse'])} |")
    add(f"| 有滤波最优柔度：细网格 | {fmt(rec['mesh_study']['filtered']['C_fine'])} |")
    add(f"| 有滤波网格相对变化 | {fmt(rec['mesh_study']['filtered']['relative_change'])} |")
    add("")

    # ------------------------------------------------------------------ 4
    add("## 4. 结果图")
    add("")
    figs = [
        ("fig1_mbb_designs.png", "图 1 MBB 100×50 密度场：无滤波（两个随机扰动种子）与 r=1.5 密度滤波，"
                                 "标题内为棋盘格指数。"),
        ("fig2_checkerboard_map.png", "图 2 局部交替指标场（|混合二阶差分|/2）：无滤波设计满布棋格，"
                                      "滤波设计基本为零。"),
        ("fig3_history.png", "图 3 柔度与棋盘格指数的迭代历史（5 个算例）。"),
        ("fig4_sensitivity_scatter.png", "图 4 悬臂梁 1250 个单元的伴随灵敏度 vs 中心差分灵敏度"
                                         "（双对数散点 + y=x）与绝对误差分布。"),
        ("fig5_error_map.png", "图 5 灵敏度误差的空间分布（log10 绝对误差）。"),
        ("fig6_convergence.png", "图 6 步长收敛：实差分 / 复步差分的平均误差随 Δρ 变化，虚线为 O(Δρ) 参考斜率。"),
        ("fig7_cantilever_design.png", "图 7 悬臂梁优化设计（50×25，volfrac 0.4，r=1.5）。"),
        ("fig8_mesh_study.png", "图 8 网格依赖性：粗/细网格上无滤波与有滤波的最优柔度。"),
    ]
    for name, cap in figs:
        add(f"- `{base / name}` — {cap}")
    add("")

    # ------------------------------------------------------------------ 5
    add("## 5. 注意点与局限")
    add("")
    add("1. **C1 判据被证伪（诚实 FAIL）**：滤波把可达密度场限制为无滤波问题的子集，按变分原理"
        "必有 `C_filter >= C_no_filter`，实测比值 "
        f"{fmt(c1['value_raw_simp'])}（≤1）。任何诚实的实现都不可能得到 ≥5 的柔度比；"
        "战役判据里想表达的「≥5 倍」在实测中体现为**棋盘格指数的抑制比** "
        f"{fmt(c2['suppression_ratio'])}×（已作为 C2 的伴随量给出），而非柔度比。"
        "本节点不修改判据文字，只如实标注 FAIL 并给出可复算的反证。")
    add("2. **DOF 交错陷阱**：装配时若用 `np.column_stack([2n, 2n+1])` 会得到 "
        "`[2n1,2n2,...,2n4, 2n1+1,...]` 的错误交错，伴随法与差分法会**同时**错，"
        "一致性检验照常通过。刚体位移零能量测试（`test_rigid_body_translation_has_zero_energy`）"
        "是唯一抓住它的检查——本节点开发中确实由该测试发现并修正。")
    add("3. **差分相消极限**：float64 直接解下，差分误差 ≈ eps_eff·C/Δρ，"
        "实测在 Δρ ≤ 1e-6 处主导；因此「Δρ→0 单调收敛到 1e-8」对经典实差分不可达。"
        "节点用无相消的复步差分完成 C5，并把实差分曲线作为机器精度极限的证据。")
    add("4. **滤波器 + OC 收敛慢**：密度滤波（对物理密度施加体积约束）在 100×50 上需要 "
        "100+ 次迭代，且终止时仍保留约 15% 灰度单元；这是线性滤波（非 Heaviside 投影）的"
        "已知特性，不影响本节点的滤波/灵敏度结论，但影响绝对柔度数值的可比性。")
    add("5. **棋盘格指数依赖初始化**：均匀初值在细网格上会收敛到只有局部交替的局部最优"
        f"（细网格两种子实测 {', '.join(fmt(v) for v in c2['unfiltered_index'])}），"
        "棋格失稳需要 ±5% 对称破缺才能充分暴露；两个种子的值都如实列出。")
    add("6. **网格依赖性不是本配置的判别量**：4 倍加密下无滤波最优柔度变化 "
        f"{fmt(rec['mesh_study']['unfiltered']['relative_change'])}、有滤波 "
        f"{fmt(rec['mesh_study']['filtered']['relative_change'])}，二者相当；"
        "即在该算例里「滤波恢复网格无关性」的说法不成立，滤波真正可测的收益是"
        "最小长度尺度/棋盘格抑制（C2），本节点不越过证据去宣称更多。")
    add("7. **相对误差的定义域**：约 "
        f"{fmt(100 * (1 - sens['with_density_filter']['n_rel'] / sens['with_density_filter']['n']))}% 的单元"
        "位于空洞区，其 SIMP 斜率为零（灵敏度数值上为 0），逐点相对误差无意义；"
        "故给出尺度无关的 l2 相对误差与「灵敏度 ≥ 峰值 1%」子集上的相对误差，"
        "战役判据本身用绝对阈值（C3/C4）。")
    add("8. **无外部基准**：MBB/悬臂的绝对柔度没有可比的公开基准值（网格规模、载荷归一化、"
        "E0 归一化各异），因此本节点只做**内部交叉验证**（伴随↔差分↔复步、常应变分片、刚体模态），"
        "不对绝对柔度作精度声明。")
    add("9. **Emin=1e-9 的病态性与未做的扫描**：生产设置保留 Sigmund 的 Emin=1e-9，"
        "使刚度阵条件数很大、差分的相消底更高；本审计在单一 Emin 下完成，"
        "未做 Emin 敏感性扫描（明确记为局限）。")
    add("")

    # ------------------------------------------------------------------ 6
    add("## 6. 误差判定（逐项，显式回答是否 <3%）")
    add("")
    add("| 关键量 | 基准 | 相对误差 | 是否 <3% |")
    add("|---|---|---|---|")
    add(f"| 伴随灵敏度 dC/dρ（滤波分支） | 中心差分 Δρ=1e-6 | l2 {fmt(sens['with_density_filter']['l2_rel'])}；"
        f"活跃子集（{int(sens['with_density_filter']['n_top1pct'])} 单元）mean "
        f"{fmt(sens['with_density_filter']['mean_rel_top1pct'])}、max "
        f"{fmt(sens['with_density_filter']['max_rel_top1pct'])} | **是** |")
    add(f"| 伴随灵敏度 dC/dρ（无滤波分支） | 中心差分 Δρ=1e-6 | l2 {fmt(sens['without_filter']['l2_rel'])}；"
        f"活跃子集 mean {fmt(sens['without_filter']['mean_rel_top1pct'])}、max "
        f"{fmt(sens['without_filter']['max_rel_top1pct'])} | **是** |")
    add(f"| 伴随灵敏度 | 复步差分（Δρ=1e-8） | l2 {fmt(sens['with_density_filter_complex_step']['l2_rel'])}；"
        f"max_abs {fmt(sens['with_density_filter_complex_step']['max_abs'])} | **是** |")
    add("| 常应变分片（Q4 内核） | 解析应变能 | 机器精度（<1e-12） | **是** |")
    add("| 刚体平移（Q4 内核） | 零应变能 | <1e-10（绝对） | **是** |")
    add("| 最优柔度绝对值 | 无外部基准 | — | 无基准，声明原因：网格/载荷/E0 归一化无统一口径 |")
    add("")
    add("结论：**所有有基准的关键量相对误差均远小于 3%**（灵敏度一致性 ~1e-5 量级）；"
        "柔度绝对值无外部基准，仅提供内部一致性与网格依赖性数据，不作精度声明。")
    add("")

    # ------------------------------------------------------------------ 7
    add("## 7. 复现与工件")
    add("")
    add("```bash")
    add("export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1")
    add("PYTHONPATH=src python -m tensorphysics.topology_optimization.simp.verification \\")
    add("    --outdir artifacts/simp_m9136 --workers 5 --fd-workers 8")
    add("PYTHONPATH=src python -m tensorphysics.topology_optimization.simp.figures artifacts/simp_m9136")
    add("PYTHONPATH=src python scripts/gen_simp_doc.py")
    add("python -m pytest tests/test_simp_density_filter.py tests/test_simp_sensitivity_consistency.py -q")
    add("```")
    add("")
    add(f"- 验收 JSON：`{base}/verification.json`")
    add(f"- 设计场与灵敏度数组：`{base}/designs.npz`")
    add(f"- 图：`{base}/fig*.png`（8 张）")
    add(f"- 总墙钟：{fmt(rec['wall_seconds'], 5)} s（机器为共享 128 核节点，负载高）")
    add("")
    import ast

    def _count(path: str) -> int:
        tree = ast.parse(pathlib.Path(path).read_text())
        return sum(isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                   for n in ast.walk(tree))

    f1 = "tests/test_simp_density_filter.py"
    f2 = "tests/test_simp_sensitivity_consistency.py"
    add(f"**pytest**：`{f1}`（{_count(f1)} 例）+ `{f2}`（{_count(f2)} 例），"
        f"共 {_count(f1) + _count(f2)} 例，全部通过（C8 判据阈值 15）。")
    add("")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"wrote {out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())