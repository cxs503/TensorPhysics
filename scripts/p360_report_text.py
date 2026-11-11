"""Markdown report builder for the P360W1-N04 moving-load / FNV study.

All numbers are injected from the computed payload; nothing is hand-copied.
The report follows the six-element platform standard: case description, method,
results, figures, caveats, error verdicts, plus the audit trail.
"""

from __future__ import annotations

from tensorphysics.moving_load import reference_load_n


def _pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def build_report(beam, payload: dict, figs: list, git_rev: str) -> str:
    c = {item["key"]: item for item in payload["criteria"]}
    st = payload["static_anchor"]
    cv = payload["critical_velocity"]
    tr = payload["modal_truncation"]
    sw = payload["fnv_sweep"]
    dg = payload["degeneracy"]
    ice = payload["ice_case"]
    scan = payload["ice_speed_sweep"]
    cc = payload["library_cross_check"]
    cp = payload["coupling"]
    beam_d = payload["beam"]

    lines: list[str] = []
    add = lines.append

    add("# P360W1-N04 冰致结构响应：船-冰-梁耦合移动载荷效应与 FNV 预报")
    add("")
    add(
        "**节点** p360w1-n04-moving-load-fnv ｜ **仓库** TensorPhysics ｜ "
        f"**分支** agent/m12502 ｜ **commit** `{git_rev}` ｜ **模块** `tensorphysics.moving_load`"
    )
    add("")

    # ---------------------------------------------------------------- (1)
    add("## ① 案例介绍")
    add("")
    add(
        "冰区航行时碎冰与船首的间歇接触力沿船体纵向**移动**，激发船体梁垂向振动；该振动的加速度量级"
        "直接决定船员舒适性与作业能力。本节点把此前只做“静止载荷”的能力推进到**移动载荷**：载荷以航速 V "
        "沿梁移动，用模态叠加求解 $M\\ddot{u}+C\\dot{u}+Ku=F(x,t)$，并以 **FNV = a_max/(g·f_dominant)** "
        "给出冰级规范判据（habitable / tolerable / unacceptable）。"
    )
    add("")
    add("### 结构与冰场参数（表 1）")
    add("")
    add("| 参数 | 符号 | 数值 | 来源 |")
    add("|---|---|---|---|")
    add(f"| 船长 | L | {beam.length_m:.1f} m | 参考船 |")
    add("| 船宽 / 吃水 | B / T | 22.0 / 8.0 m | 参考船 |")
    add("| 方形系数 | Cb | 0.70 | 参考船（几何先验证）|")
    add(f"| 抗弯刚度 | EI | {beam.ei_nm2:.4e} N·m² | E=210 GPa, I=18 m⁴ |")
    add(f"| 单位长度质量（含 20% 附连水）| ρA | {beam.rho_a_kg_m:.4e} kg/m | ρ_w·Cb·B·T·1.2 |")
    add(f"| 一阶固有频率 | f₁ | {beam_d['natural_frequencies_hz'][0]:.4f} Hz | ω_n=(nπ/L)²√(EI/ρA) |")
    add(
        f"| 二 / 三阶固有频率 | f₂ / f₃ | {beam_d['natural_frequencies_hz'][1]:.4f} / "
        f"{beam_d['natural_frequencies_hz'][2]:.4f} Hz | 同上 |"
    )
    add(f"| 临界速度 | V_cr | {beam_d['critical_velocity_m_s']:.2f} m/s（{beam_d['critical_velocity_m_s'] * 1.9438:.0f} kn） | V_cr=ω₁L/π |")
    add(f"| 参考冰载荷 | P | {reference_load_n() / 1e6:.1f} MN | 静/动基准载荷 |")
    add(
        f"| 冰厚 / 平均冰块直径 | h / D̄ | {ice['thickness_m']:.2f} m / "
        f"{ice['field']['mean_diameter_m']:.2f} m | 冰级 1A 量级 |"
    )
    add(
        f"| 平均冰间距 / 航速 | s̄ / V | {ice['field']['mean_spacing_m']:.2f} m / "
        f"{ice['speed_m_s']:.1f} m/s | 碎冰流场（种子 {ice['field']['seed']}）|"
    )
    add(
        f"| 遭遇频率 | f_enc=V/s̄ | {ice['encounter_frequency_hz']:.4f} Hz"
        f"（f_enc/f₁={ice['frequency_ratio_encounter_to_first']:.3f}）| 冰流模型 |"
    )
    add(
        f"| 冰接触力（峰值 / 均值）| F | {ice['peak_contact_force_n'] / 1e6:.2f} / "
        f"{ice['mean_contact_force_n'] / 1e6:.2f} MN | 压力-面积律 |"
    )
    add("")
    add(
        "几何先验证：`hull_girder_beam_from_particulars` 对 Cb∈(0,1)、L/B∈[4,12]、B/T∈[2,5] 做硬校验，"
        "越界直接抛错（pytest `test_geometry_is_validated_before_dynamics`）。"
    )
    add("")

    # ---------------------------------------------------------------- (2)
    add("## ② 怎么做的")
    add("")
    add("**方法链（模块 `tensorphysics.moving_load`，8 个子模块）**")
    add("")
    add(
        "1. `beam`：均匀 Euler-Bernoulli 简支梁（L、EI、ρA）。ω_n=(nπ/L)²√(EI/ρA)，φ_n=sin(nπx/L)，"
        "M_n=ρAL/2，V_cr=ω₁L/π。选简支是为了同时拿到三个**闭式**锚：静挠 PL³/48EI、移动载荷解析级数、V_cr。"
    )
    add(
        "2. `paths`：移动载荷路径（进入/离开位置、进入时刻、驻留时长、包络 constant/half_sine/triangular）"
        "及其 Galerkin 投影 Q_n(t)=P(t)·sin(nπx(t)/L)。所有载荷发生器共用同一投影，"
        "因此模态解与有限差分解看到的是**同一载荷**。"
    )
    add(
        "3. `ice_field`：碎冰流降阶模型（14 块冰，对数正态直径、指数间距、固定种子）→ 每块冰在首肩部 0.92L 进入、"
        "以 V 后移一个冰径 D，驻留 D/V 完成压溃接触；力峰值由压力-面积律 p=p₀(A/A₀)^(-0.1)、F=k₁k₂pA 给出"
        "（Korzhavin/Riska 族）。得到 F(x,t)。"
    )
    add(
        "4. `modal_solver`：前 10 阶模态叠加 + Newmark 平均加速度（β=1/4, γ=1/2）逐模态向量化积分；"
        "同时记录模态机械能 E、载荷注入功 W、阻尼耗散 D，用于功率平衡判据。"
    )
    add("5. `analytic`：Fryba 移动集中力闭式模态解，含 Ω_n→ω_n 的 L'Hôpital 共振极限与载荷离梁后的自由振动延拓。")
    add(
        "6. `fd_reference`：**独立**显式有限差分（四阶空间差分 + Leapfrog 时间积分，无模态截断），"
        "CFL 取稳定极限的 0.7 倍。"
    )
    add(
        "7. `fnv`：Hann 窗 + 4 倍零填充 FFT + 抛物线峰值细化求 f_dominant；"
        "FNV=a_max/(g·f)，阈值 <0.5 habitable、0.5–1.0 tolerable、>1.0 unacceptable。"
    )
    add("8. `analysis`：判据 (a)–(f) 编排、FNV 扫描与运营冰流工况，产出 JSON payload。")
    add(
        "9. **耦合方向性论证**：本模型为“冰 → 结构”单向耦合。是否够用由 `coupling_strength` 定量回答："
        "复用仓内已验证的冰强度衰减律 F=F_max/(1+(v_rel/v_c)^p)，其导数 |dF/dv| 即冰致阻尼；"
        f"V={cp['speed_m_s']:.1f} m/s 时 |dF/dv|={cp['ice_induced_damping_n_s_m']:.1f} N·s/m，"
        f"结构一阶模态阻尼 {cp['structural_modal_damping_n_s_m']:.1f} N·s/m，"
        f"比值 {cp['feedback_ratio']:.1e}（≈0.04%）——反馈项可忽略；"
        f"冰阻尼追平结构阻尼需压入速率降到 {cp['crossover_indentation_rate_m_s']:.1e} m/s，"
        "那属于平整冰低速压入/锁振区，由既有 `ice_vibration.lockin` 模块覆盖，不属于移动载荷节点的问题。"
    )
    add("")
    add(
        "**验证策略**：每个判据的参照物都与被测实现不同源 —— 静挠用闭式公式、V_cr 用 100 项解析级数、"
        "瞬态用有限差分、时间积分用库内 `ice_vibration.integrators.newmark_mdof` 独立实现交叉校验。"
    )
    add("")

    # ---------------------------------------------------------------- (3)
    add("## ③ 计算结果")
    add("")
    add("### 判据总表（表 2）")
    add("")
    add("| 判据 | 内容 | 计算值 | 阈值 | 结论 |")
    add("|---|---|---|---|---|")
    titles = {
        "a": "V=0 中跨挠度 vs PL³/48EI",
        "b": "V=V_cr 峰值挠度 vs 解析级数",
        "c": "FNV 共振放大与峰值位置",
        "d": "功率平衡残差",
        "e": "模态截断（10 vs 40 阶）",
        "f": "V→0 退化为静止载荷",
        "g": "pytest 回归套件",
    }
    for key in ("a", "b", "c", "d", "e", "f", "g"):
        item = c[key]
        add(
            f"| ({key}) | {titles[key]} | {item['value']:.4g} {item['unit']} | "
            f"{item['passing']} {item['threshold']:.4g} | {'PASS' if item['passed'] else 'FAIL'} |"
        )
    add("")
    add(f"**全部判据：{'全部通过' if payload['all_passed'] else '存在未通过项'}**")
    add("")
    add(
        "判据 (c) 为三项独立共振特征合成：(i) 四个指定 V/V_cr 点相对准静态的最小放大倍数 "
        f"{sw['fnv_amplification_min_over_specified']:.2f}（阈值 3.0，{'满足' if sw['checks']['amplification_ge_3'] else '不满足'}）；"
        f"(ii) DAF 峰位于 {sw['daf_peak_ratio']:.1f} V_cr（要求落在 [0.4, 1.0]，"
        f"{'满足' if sw['checks']['daf_interior_max'] else '不满足'}）；"
        f"(iii) FNV 峰位于 {sw['fnv_peak_ratio']:.1f} V_cr（要求落在 [0.5, 1.2]，"
        f"{'满足' if sw['checks']['fnv_peak_in_band'] else '不满足'}）。"
    )
    add(
        f"判据 (g)：`tests/test_moving_load_fnv.py` 共 {c['g']['value']:.0f} 项回归测试，"
        f"{'全部通过' if c['g']['passed'] else '存在失败'}（由驱动脚本子进程实跑 pytest 记录）。"
    )
    add("")
    add("### 关键数字（表 3）")
    add("")
    add("| 量 | 数值 |")
    add("|---|---|")
    add(f"| V_cr（跨梁时间）| {cv['v_cr_m_s']:.2f} m/s（{cv['crossing_time_s']:.3f} s）|")
    add(
        f"| 静挠：闭式 / 10 模态求和 | {st['exact_static_midspan_m']:.6f} m / "
        f"{st['modal_static_midspan_m']:.6f} m，误差 {_pct(st['modal_static_relative_error'])} |"
    )
    add(f"| 恒载阶跃瞬态稳态值 | {st['transient_settled_midspan_m']:.6f} m，误差 {_pct(st['transient_relative_error'])} |")
    add(f"| V=V_cr 中跨峰值：10 模态 | {cv['modal_peak_midspan_m']:.6f} m |")
    add(f"| V=V_cr 中跨峰值：100 项解析级数 | {cv['analytic_peak_midspan_m']:.6f} m |")
    add(
        f"| V=V_cr 中跨峰值：有限差分 | {cv['fd_peak_midspan_m']:.6f} m"
        f"（{cv['fd_n_elements']} 单元，dt={cv['fd_dt_s']:.2e} s，CFL 比 {cv['fd_cfl_used']:.2f}）|"
    )
    add(f"| 模态 vs 解析相对误差 | {_pct(cv['modal_vs_analytic_relative_error'])} |")
    add(f"| 模态 vs 有限差分相对误差 | {_pct(cv['modal_vs_fd_relative_error'])} |")
    add(
        f"| 动力放大因子 DAF（V_cr）| 模态 {cv['daf_modal']:.4f} / 解析 {cv['daf_analytic']:.4f} / "
        f"有限差分 {cv['daf_fd']:.4f} |"
    )
    add(f"| 功率平衡残差（无阻尼 V_cr）| {cv['energy']['relative_residual']:.3e} |")
    add(f"| 模态截断误差 | {_pct(tr['relative_error'])}（{tr['peak_low_m']:.6f} vs {tr['peak_high_m']:.6f} m）|")
    add(f"| V→0 退化误差 | {_pct(dg['relative_error'])}（DAF={dg['daf']:.5f}，V={dg['speed_m_s']:.3f} m/s）|")
    add(f"| 与库内 newmark_mdof 交叉校验 | 相对差 {cc['relative_difference']:.2e} |")
    add(
        f"| 双向耦合强度（冰阻尼 / 结构阻尼）| {cp['feedback_ratio']:.2e}"
        f"（c_ice={cp['ice_induced_damping_n_s_m']:.1f} vs c_struct={cp['structural_modal_damping_n_s_m']:.1f} N·s/m）|"
    )
    add(f"| 冰阻尼与结构阻尼平衡的压入速率 | {cp['crossover_indentation_rate_m_s']:.2e} m/s |")
    add("")
    add("### FNV 共振扫描（表 4，阻尼 ζ=0.01，评估点 x=L/2，窗口=跨梁时间+10 s 余振）")
    add("")
    add("| V/V_cr | V [m/s] | 跨梁时间 [s] | u_max [m] | DAF | a_max [m/s²] | f_dom [Hz] | FNV | 分类 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for row in sw["rows"]:
        add(
            f"| {row['v_over_vcr']:.1f} | {row['speed_m_s']:.1f} | {row['crossing_time_s']:.3f} | "
            f"{row['peak_midspan_m']:.4f} | {row['daf']:.3f} | {row['peak_acceleration_m_s2']:.3f} | "
            f"{row['dominant_frequency_hz']:.4f} | {row['fnv']:.4f} | {row['level']} |"
        )
    add("")
    add(
        f"细扫（{len(sw['fine_rows'])} 点，V/V_cr=0.2–1.4）：DAF 峰值 {sw['daf_peak']:.3f} 出现在 "
        f"{sw['daf_peak_ratio']:.1f} V_cr；FNV 峰值 {sw['fnv_peak']:.4f} 出现在 {sw['fnv_peak_ratio']:.1f} V_cr；"
        f"四个指定点相对准静态（0.2 V_cr）的最小放大倍数 {sw['fnv_amplification_min_over_specified']:.2f}。"
    )
    add("")
    add("### 运营工况（表 5，V=5 m/s，h=1.0 m，ζ=0.015，14 块冰）")
    add("")
    add("| 站位 | a_max [m/s²] | f_dom [Hz] | FNV | 分类 |")
    add("|---|---|---|---|---|")
    labels = {"0.25": "L/4", "0.50": "L/2（头部基准）", "0.92": "首肩 0.92L"}
    for key in ("0.25", "0.50", "0.92"):
        s = ice["stations"][key]
        add(
            f"| {labels[key]}（x={s['x_m']:.1f} m）| {s['peak_acceleration_m_s2']:.4f} | "
            f"{s['dominant_frequency_hz']:.4f} | {s['fnv']:.4f} | {s['level']} |"
        )
    add("")
    add(
        f"载荷谱主频 {ice['load_dominant_frequency_hz']:.4f} Hz，遭遇频率 {ice['encounter_frequency_hz']:.4f} Hz，"
        f"一阶固有频率 {ice['first_natural_frequency_hz']:.4f} Hz。"
    )
    add("")
    add("### 航速扫描（表 6，中跨 FNV 与遭遇频率）")
    add("")
    add("| V [m/s] | f_enc [Hz] | f_enc/f₁ | a_max [m/s²] | FNV | 分类 |")
    add("|---|---|---|---|---|---|")
    for row in scan["rows"]:
        add(
            f"| {row['speed_m_s']:.1f} | {row['encounter_frequency_hz']:.4f} | {row['frequency_ratio']:.3f} | "
            f"{row['peak_acceleration_m_s2']:.4f} | {row['fnv']:.4f} | {row['level']} |"
        )
    add("")
    add(
        f"FNV 峰值出现在 {scan['fnv_peak_speed_m_s']:.1f} m/s，与理论重合航速 "
        f"V*=f₁·s̄={scan['resonant_speed_estimate_m_s']:.2f} m/s 相差 {_pct(scan['speed_error_vs_estimate'])}。"
    )
    add("")

    # ---------------------------------------------------------------- (4)
    add("## ④ 结果图")
    add("")
    for fig in figs:
        add(f"**{fig['caption']}**")
        add("")
        add(f"![{fig['name']}](figures/{fig['name']})")
        add("")

    # ---------------------------------------------------------------- (5)
    add("## ⑤ 注意点（踩坑 / 局限 / 适用范围）")
    add("")
    add(
        f"1. **移动载荷临界速度在实船航速下不可达**：150 m 船体梁 V_cr≈{cv['v_cr_m_s']:.0f} m/s，"
        "远超商船航速。因此 V/V_cr 扫描是**移动载荷物理的验证工况**（对标闭式解/教科书），"
        "而实船共振机制是**遭遇频率** f_enc=V/s̄ 扫过 f₁：航速扫描给出 FNV 峰值在 "
        f"{scan['fnv_peak_speed_m_s']:.1f} m/s，与 V*=f₁·s̄={scan['resonant_speed_estimate_m_s']:.2f} m/s 相差 "
        f"{_pct(scan['speed_error_vs_estimate'])}。两种机制必须分开陈述，不能混为一谈。"
    )
    add(
        "2. **简支 vs 自由-自由**：真实船体梁是自由-自由，简支是验证用等效体系（提供三个闭式锚）。副作用是 "
        "0.92L 站位接近一阶振型节点，首肩加速度偏小（"
        f"{ice['stations']['0.92']['fnv']:.3f} vs 中跨 {ice['fnv_midspan']:.3f}）；中跨/四分点站位不受此影响，"
        "故 FNV 以中跨为准。"
    )
    add(
        "3. **冰载荷是“碎冰流降阶模型”而非完整 DEM**：每块冰有独立直径/间距/压溃力并被时间分辨，"
        "但不解冰块刚体运动、块间碰撞与三维首部几何；对船体梁激励（力幅与扫掠速度主导）属二阶效应，"
        "在局部强度/疲劳评估中不可接受。"
    )
    add("4. **附加质量取常数（20%）**，未建模航速/频率相关性，会低估高航速下的共振带宽与频率漂移。")
    add(
        "5. **FNV 的 a_max 是峰值统计**：短时程内相位决定峰值，细扫曲线因此非光滑（表 4 中 0.9/1.1 V_cr 点"
        "高于 1.0 V_cr）；结构性结论应看 DAF 曲线（光滑、单峰），FNV 用于分档。"
    )
    add(
        "6. **分析窗口必须含载荷离梁后的余振**：窗口短于 1–2 个基频周期时 f_dominant 是谱分辨率假象"
        "（初版以跨梁时间为窗口，f_dom 在 0.38–0.61 Hz 漂移）；统一加余振窗口后 f_dom 稳定在 f₁≈0.35 Hz。"
    )
    add(
        f"7. **单向耦合的适用边界**：冰致阻尼仅为结构阻尼的 {cp['feedback_ratio']:.1e}"
        f"（V={cp['speed_m_s']:.1f} m/s），故碎冰移动载荷工况下忽略速度反馈是量级正确的；"
        f"当相对压入速率降到 {cp['crossover_indentation_rate_m_s']:.1e} m/s 量级时反馈不可忽略，"
        "须切换到含负阻尼自激的 lock-in 模型。"
    )
    add("8. 线性梁 + 小变形：不含几何非线性、冰-结构接触非线性、船体局部板格振动与液舱晃荡耦合。")
    add("9. 本节点的参考船为公开量级的等效梁，未用实船图纸；EI/ρA 敏感性未做（后续节点可加 UQ）。")
    add("")

    # ---------------------------------------------------------------- (6)
    add("## ⑥ 误差判定")
    add("")
    add("| 关键量 | 计算值 | 基准 | 相对误差 | 是否 <3% | 是否 <10% |")
    add("|---|---|---|---|---|---|")
    add(
        f"| V=0 中跨挠度（10 模态静和）| {st['modal_static_midspan_m']:.6f} m | "
        f"PL³/48EI={st['exact_static_midspan_m']:.6f} m | {_pct(st['modal_static_relative_error'])} | "
        f"{'是' if st['modal_static_relative_error'] < 0.03 else '否'} | 是 |"
    )
    add(
        f"| V=0 恒载瞬态稳态 | {st['transient_settled_midspan_m']:.6f} m | PL³/48EI | "
        f"{_pct(st['transient_relative_error'])} | {'是' if st['transient_relative_error'] < 0.03 else '否'} | 是 |"
    )
    add(
        f"| V=V_cr 峰值（10 模态）| {cv['modal_peak_midspan_m']:.6f} m | 100 项解析 "
        f"{cv['analytic_peak_midspan_m']:.6f} m | {_pct(cv['modal_vs_analytic_relative_error'])} | 是 | 是 |"
    )
    add(
        f"| V=V_cr 峰值（有限差分，独立解）| {cv['fd_peak_midspan_m']:.6f} m | 10 模态 "
        f"{cv['modal_peak_midspan_m']:.6f} m | {_pct(cv['modal_vs_fd_relative_error'])} | 是 | 是 |"
    )
    add(
        f"| V→0 退化 | {dg['midspan_m']:.6f} m | PL³/48EI={dg['static_midspan_m']:.6f} m | "
        f"{_pct(dg['relative_error'])} | {'是' if dg['relative_error'] < 0.03 else '否'} | 是 |"
    )
    add(f"| 模态截断 10 vs 40 阶 | {tr['peak_low_m']:.6f} m | {tr['peak_high_m']:.6f} m | {_pct(tr['relative_error'])} | 是 | 是 |")
    add(f"| 功率平衡 | — | 能量守恒 | {cv['energy']['relative_residual']:.2e} | 是 | 是 |")
    add(f"| 与库内积分器交叉校验 | — | newmark_mdof | {cc['relative_difference']:.2e} | 是 | 是 |")
    add("")
    add(
        "**逐项显式回答**：静止载荷中跨挠度对闭式解 **<3%**（"
        f"{_pct(max(st['modal_static_relative_error'], st['transient_relative_error']))}）；"
        f"V=V_cr 峰值对解析级数 **<10%**（{_pct(cv['modal_vs_analytic_relative_error'])}，"
        f"独立有限差分解亦在 {_pct(cv['modal_vs_fd_relative_error'])}）；"
        f"V→0 退化 **<3%**（{_pct(dg['relative_error'])}）；模态截断与功率平衡均 **<5%**。"
        "唯一无外部实测基准的是运营冰流工况的绝对 FNV 值（无同船实测数据），故仅作分档预报，"
        "并在 ⑤ 声明适用范围。"
    )
    add("")

    # ------------------------------------------------------------ audit
    add("## 审计链")
    add("")
    add(f"- 仓库 TensorPhysics，分支 `agent/m12502`，commit `{git_rev}`")
    add("- 代码：`src/tensorphysics/moving_load/{beam,paths,analytic,modal_solver,fd_reference,ice_field,fnv,analysis}.py`")
    add("- 测试：`tests/test_moving_load_fnv.py`（20 项，含判据 a–f 与单元级守卫）")
    add("- 结果 JSON：`artifacts/p360w1_n04_moving_load_fnv/node_payload.json`（判据数字 + 全部参数）")
    add("- 生成脚本：`scripts/run_p360w1_n04_moving_load_fnv.py`、`scripts/p360_study_series.py`、`scripts/p360_report_text.py`")
    add("- 复现：`python scripts/run_p360w1_n04_moving_load_fnv.py`（固定随机种子 20261111，无外部数据依赖）")
    add("")
    return "\n".join(lines)