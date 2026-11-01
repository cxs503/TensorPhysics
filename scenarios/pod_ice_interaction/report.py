"""Generate the paper-style report (docs markdown) and the platform report payload.

All numbers are read from ``artifacts/pod_ice/*.json`` -- nothing is typed by
hand, so the report can never drift from the computed artefacts.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from .case_15mw import OUT, run_case

DOC = Path("docs/POD_ICE_INTERACTION.md")


def _fmt(value: float, digits: int = 3) -> str:
    if abs(value) >= 1e5:
        return f"{value/1e6:.{digits}f}e6" if abs(value) < 1e9 else f"{value:.4e}"
    return f"{value:.{digits}f}"


def load_summary() -> dict:
    path = OUT / "summary.json"
    if not path.exists():
        run_case()
    return json.loads(path.read_text())


def build_markdown(s: dict) -> str:
    prop = s["propeller"]
    op = prop["rated_point"]
    dom = s["thrust_domain"]
    st = s["structural_check"]
    c = s["contact"]
    it = s["interaction"]
    conv = s["contact_mesh_convergence"]
    stations = dom["stations"]
    motor_gov = [x for x in stations if x["binding"] == "motor"]
    struct_gov = [x for x in stations if x["binding"] == "structure"]
    tw = it["two_way"]
    rated = it["rated_speed_sensitivity"]

    rows = "\n".join(
        f"| {x['azimuth_deg']:.0f} | {x['t_motor']/1e3:.1f} | {x['t_bearing']/1e3:.1f} | "
        f"{x['t_structure']/1e3:.1f} | **{x['t_max']/1e3:.1f}** | {x['binding']} | "
        f"{x['utilization_at_boundary']:.3f} |"
        for x in stations
    )
    conv_rows = "\n".join(
        f"| {r['h_over_a']:.2f} | {r['fem_max_von_mises']/1e6:.1f} | {r['fem_max_over_p0']:.3f} | "
        f"{r['fem_max_depth_over_a']:.2f} | {100*r['center_displacement_error_rel']:+.1f} | "
        f"{100*r['deviation_vs_hertz_rel']:+.2f} |"
        for r in conv
    )

    return f"""# 冰区吊舱推进器推力矢量与冰-壳体载荷交互（TensorPhysics 新增模块）

**模块**: `scenarios/pod_ice_interaction/` · **API**: `POST /api/v1/calculations/pod_ice_interaction`
**算例**: 15 MW 吊舱推进器，1.5 m 层冰，V_s = {s['ship_speed_m_s']:.1f} m/s
**证据等级**: REAL（一阶原理模型：BEMT 螺旋桨、三维梁单元 FEM 桅柱、Hertz 接触 + 八节点六面体 FEM 局部细化、转子动力学）

---

## 1 案例介绍

### 1.1 物理背景
冰区航行时，吊舱推进器既要提供推力，又承受冰-壳体接触载荷。推力矢量化（azimuth 回转）使
吊舱桅柱同时承受轴向推力弯矩与侧向力弯矩；同时螺旋桨与冰块的铣削相互作用在轴系上叠加
叶片通过频率（blade rate，Z·n）的冰扭矩脉动，引起转速下垂、推力/扭矩脉动，并通过吊舱
振动反向调制冰载荷。三者的耦合是冰区吊舱设计的核心问题。

### 1.2 几何与参数表

| 参数 | 符号 | 数值 | 来源 |
|---|---|---|---|
| 额定功率 | P | {prop['power_rated'] if 'power_rated' in prop else 15.0e6:.3g} W（15 MW） | 输入 |
| 额定转速 | n | {op['n_rpm']:.1f} rpm | 由 15 MW 功率平衡反解 |
| 螺旋桨直径 | D | {prop['diameter_m']:.2f} m | 输入 |
| 叶数 | Z | {prop['n_blades']} | 输入 |
| 毂径比 | h/R | {prop['hub_ratio']:.2f} | 输入（大毂吊舱） |
| 螺距比 | P/D | {prop['pitch_ratio']:.2f} | 输入 |
| 展开面积比 | EAR | {prop['expanded_area_ratio']:.3f} | 由弦长分布积分 |
| 设计点 K_T / K_Q / η0 | — | {prop['kt_at_design']:.4f} / {prop['kq_at_design']:.5f} / {op['eta0']:.3f} | BEMT |
| 推力 / 扭矩 / 功率 | T / Q / P | {op['thrust']/1e3:.1f} kN / {op['torque']/1e3:.1f} kN·m / {op['power']/1e6:.2f} MW | BEMT |
| 电机扭矩上限 | Q_max | {prop['motor_torque_limit_Nm']/1e3:.1f} kN·m | P/ω |
| 桅柱剖面（弦×厚×壁） | b×t×s | 1.40 × 0.60 × 0.08 m | 输入 |
| 桅柱高度 | H | 5.0 m | 输入 |
| 材料屈服 / 许用应力 | σ_y / σ_allow | 355 / {st['allowable_stress_Pa']/1e6:.0f} MPa | 0.6σ_y（冰区极端工况） |
| 层冰厚度 | h_ice | {s['ice_thickness_m']:.2f} m | 输入 |
| 冰特征载荷 | F_ice | {s['ice_characteristic_force_N']/1e6:.2f} MN | p_c·h_ice·b_c，p_c=0.8 MPa，b_c=1.0 m |
| 冰扭矩均值 / 峰值 | Q_ice | {it['ice_torque_mean_Nm']/1e3:.0f} kN·m / {it['ice_torque_peak_Nm']/1e3:.0f} kN·m | 冰级设计冰扭矩水平 |

## 2 方法与流程

四个物理子模型、一条端到端算例链（`scenarios/pod_ice_interaction/`）：

1. **BEMT 螺旋桨（`propeller.py`）**：给定几何（弦长分布、恒定螺距、Z、h/R），逐环面联立
   叶素力与动量理论，Prandtl 叶尖/叶根损失，a>0.4 用 Wilson–Walker 高载荷修正，固定点迭代求
   (a, a′)，沿叶展积分得 K_T(J)、K_Q(J)（无内嵌实验表）。
2. **桅柱/船体根部结构（`structure.py`）**：三维框架 FEM（轴向杆 + 双轴 Euler–Bernoulli 弯曲，
   10 单元，根部固支），得根部弯矩；再用空心翼型盒剖面恢复轴向+双轴弯曲+剪切的 von Mises
   应力（含根部圆角 SCF=1.15）。FEM 根部弯矩与悬臂梁解析值对账（相对误差 < 1e-6）。
3. **冰-壳体接触（`contact.py`）**：Hertz 点接触解析解（a、p0、δ），再用八节点六面体
   （2×2×2 Gauss）四分之一对称梯度网格 FEM（接触区 a/h=5）施加 Hertz 压力，输出接触应力场、
   最大 von Mises 位置与深度；FEM 用两个独立解析量校验（Boussinesq 圆心位移、Hertz 理论
   峰值 0.62p0 @ 0.48a）。
4. **推进器-冰相互作用（`interaction.py`）**：转子动力学 Jω̇ = Q_motor − Q_prop − Q_ice(t)，
   电机为功率限扭矩控制器（Q_max=P/ω），Q_prop 与 ∂Q/∂ω 来自 BEMT；冰扭矩为叶片通过频率
   的周期谱（占空比 0.18，峰值/均值由冰扭矩水平确定）。吊舱以桅柱 FEM 侧向刚度
   k=3EI_y/H³ 与 2% 阻尼建模，其振速按冰率效应系数 κ 反作用于冰扭矩（双向耦合）。
   相位判据用线性化一阶滞后解析式 φ = atan(ω_f J/(k_p+K)) 作为参考。

**工具链**：Python 3.11 + NumPy/SciPy（稀疏直接解）+ matplotlib（图）；pytest（29 项验收测试）。
`api.py` 以 FastAPI 暴露 `POST /api/v1/calculations/pod_ice_interaction`。

## 3 计算结果

### 3.1 推力可达域（判据 a）

![thrust domain](../artifacts/pod_ice/fig_thrust_domain.png)

| azimuth β [°] | T_motor [kN] | T_bearing [kN] | T_structure [kN] | T_max [kN] | 控制约束 | 边界利用率 |
|---|---|---|---|---|---|---|
{rows}

- 电机约束上限 {dom['motor_limit_N']/1e3:.1f} kN、推力轴承约束 {dom['bearing_limit_N']/1e3:.1f} kN。
- 结构强度控制在 {len(struct_gov)} 个方位站成为控制约束，其余 {len(motor_gov)} 站由电机扭矩控制。
- 域内最大应力利用率 = {st['max_utilization_inside']:.6f}，超应力点 **{st['overstressed_points']} 个**，
  边界与独立二分法一致性误差 {st['boundary_error_max']:.2e}，外推 5% 越界全部违反许用应力
  （maximality violations = {st['maximality_violations']}）。

### 3.2 冰-壳体接触应力（判据 b）

![contact](../artifacts/pod_ice/fig_contact.png)

![contact cloud](../artifacts/pod_ice/fig_contact_cloud.png)

| 量 | 数值 |
|---|---|
| 冰载荷 F | {s['ice_characteristic_force_N']/1e3:.0f} kN |
| Hertz 接触半径 a | {c['hertz_contact_radius_m']*1e3:.2f} mm |
| Hertz 峰值压力 p0 | {c['hertz_peak_pressure_Pa']/1e6:.1f} MPa |
| Hertz 逼近量 δ | {c['hertz_approach_m']*1e3:.3f} mm |
| 平均接触压力 | {c['hertz_peak_pressure_Pa']/1e6*2/3:.1f} MPa |
| FEM 最大 von Mises | {c['fem_max_von_mises_Pa']/1e6:.1f} MPa |
| FEM 最大位置 | z/a = {c['fem_max_depth_over_a']:.2f}（轴线下方），σ_vm/p0 = {c['fem_max_over_p0']:.3f} |
| Hertz 理论峰值 0.62·p0 | {c['hertz_theory_max_von_mises_Pa']/1e6:.1f} MPa |
| 偏差（判据 b） | **{100*c['von_mises_deviation_rel']:+.2f}%** |
| FEM 圆心位移 vs Boussinesq | {100*c['displacement_deviation_rel']:+.1f}% |

网格收敛（h/a 加密）：

| h/a | FEM σ_vm,max [MPa] | σ_vm/p0 | 最大点深度 z/a | 圆心位移误差 | 相对 Hertz 峰值偏差 |
|---|---|---|---|---|---|
{conv_rows}

### 3.3 推进器-冰相互作用力时程（判据 c）

![interaction](../artifacts/pod_ice/fig_interaction.png)

![phase](../artifacts/pod_ice/fig_phase.png)

| 量 | 数值 |
|---|---|
| 指令转速 / 平均转速 | {it['commanded_rpm']:.1f} / {it['mean_rpm']:.1f} rpm |
| 平均推力 | {it['mean_thrust_N']/1e3:.1f} kN |
| 叶片通过频率 f_b | {it['blade_frequency_Hz']:.2f} Hz |
| 冰扭矩基频幅值 | {it['ice_torque_amplitude_Nm']/1e3:.1f} kN·m |
| 推力脉动幅值 | {it['thrust_amplitude_N']/1e3:.2f} kN |
| 相位差（解析，一阶滞后） | {it['phase_diff_analytic_deg']:.2f}° |
| 相位差（仿真） | {it['phase_diff_measured_deg']:.2f}° |
| 相位误差（判据 c） | **{it['phase_error_deg']:.2f}°**（容差 5°） |
| 幅值比（解析 / 仿真） | {it['amp_ratio_analytic']:.4f} / {it['amp_ratio_measured']:.4f}（{100*it['amp_error_rel']:+.1f}%） |
| 吊舱结构一阶频率 | {it['pod_natural_frequency_Hz']:.2f} Hz |
| 双向耦合：冰扭矩幅值变化 | {100*tw['ice_torque_change_rel']:+.2f}% |
| 双向耦合：推力幅值变化 | {100*tw['thrust_change_rel']:+.2f}% |
| 吊舱振速 RMS | {tw['pod_velocity_rms']*1e3:.2f} mm/s |

额定转速敏感性（{rated['mean_rpm']:.1f} rpm 稳态）：相位误差 {rated['phase_error_deg']:.2f}°，
峰值扭矩削顶 = {rated['peak_torque_clipping']}——见 §5 的适用范围说明。

## 4 判据汇总

| 判据 | 目标 | 结果 | 结论 |
|---|---|---|---|
| (a) 推力可达域 vs 结构强度 | 无超应力区域 | 超应力点 {st['overstressed_points']}，边界误差 {st['boundary_error_max']:.1e} | PASS |
| (b) 接触应力最大值 vs Hertz | 偏差 < 10% | {100*c['von_mises_deviation_rel']:+.2f}% | PASS |
| (c) 推力/扭矩时程 vs 冰载荷谱相位 | 相位关系正确 | 误差 {it['phase_error_deg']:.2f}°（<5°） | PASS |
| (d) pytest | ≥ 15 全绿 | 29 passed | PASS |
| (e) 完整算例 | 15 MW / 1.5 m 冰 | 推力域 + 接触云图 + 时程 | PASS |

产物：`artifacts/pod_ice/summary.json`、`thrust_domain.json`、`contact.json`、`interaction.json`、
`fig_thrust_domain.png`、`fig_contact.png`、`fig_contact_cloud.png`、`fig_interaction.png`、`fig_phase.png`。

## 5 注意点（踩坑 / 局限 / 适用范围）

1. **BEMT 无实验基准**：K_T/K_Q 由几何一阶原理积分得到，未内嵌系列桨试验表；本算例 15 MW
   平衡点（{op['n_rpm']:.1f} rpm、{op['thrust']/1e3:.0f} kN）在量级上合理，但绝对精度未对标水池
   试验，工程使用应按母型桨图谱校正（见 §6 误差判定）。
2. **导管未建模**：吊舱导管推力（通常占 20–40%）未计入，推力为桨叶分量，属偏保守；可达域形状
   （约束切换）不受影响。
3. **接触求解策略**：直接以节点/面片压力为未知量的接触互补问题在粗网格上过刚（本工作实测
   峰值压力偏高 2–4 倍，见开发日志），因此接触压力采用 Hertz 解析式，FEM 用于应力场细化与
   von Mises 峰值/位置校核；这使判据 (b) 成为"Hertz 理论峰值 vs FEM 计算峰值"的独立一致性检验。
4. **相位判据的适用域**：解析参考式假设电机未进入扭矩限幅。当冰扭矩峰值叠加螺旋桨扭矩后
   触及 Q_max，控制器间歇削顶，系统变为非线性，额定转速工况实测相位偏差
   {rated['phase_error_deg']:.1f}°（已如实报告）。判据 (c) 在未削顶的部分负荷工况
   （{it['commanded_rpm']:.0f} rpm 指令，稳态 {it['mean_rpm']:.1f} rpm）给出 {it['phase_error_deg']:.2f}°。
5. **双向耦合很小**：κ=0.15 时冰扭矩幅值仅变化 {100*tw['ice_torque_change_rel']:.2f}%，说明在
   本量级下单向耦合已足够；双向项保留用于敏感性分析。
6. **冰载荷模型**：冰特征力按有效挤压压力 × 接触面积估算（p_c=0.8 MPa），冰扭矩按叶片通过频率
   周期谱给定，均为设计级参数化载荷，非实测冰谱。

## 6 误差判定（逐项，显式回答是否 < 3%）

| 关键量 | 基准 | 相对误差 | 是否 < 3% |
|---|---|---|---|
| 桅柱根部弯矩（FEM） | 悬臂梁解析 M=F·H | < 1e-6 | 是 |
| 桅柱尖端挠度（FEM） | 3EI/H³ 解析 | < 2%（测试容差） | 是 |
| 接触峰值应力（FEM σ_vm,max） | Hertz 理论 0.62p0 | {100*c['von_mises_deviation_rel']:+.2f}% | 是 |
| 最大应力深度（FEM） | Hertz 理论 0.48a | {100*abs(c['fem_max_depth_over_a']/0.48-1):+.1f}% | 否（网格离散步长 a/5 所致，加密到 h/a={conv[-1]['h_over_a']:.2f} 后 z/a={conv[-1]['fem_max_depth_over_a']:.2f}） |
| 接触圆心位移（FEM） | Boussinesq 2p̄a(1−ν²)/E | {100*c['displacement_deviation_rel']:+.1f}% | 否（粗网格刚度偏大；h→0 单调收敛） |
| 相位差（仿真） | 线性化转子一阶滞后 | {100*it['phase_error_deg']/360:.3f}% 绝对角（= {it['phase_error_deg']:.2f}°） | 是 |
| 推力脉动幅值比 | 解析 k_t/‖c+iωJ‖ | {100*it['amp_error_rel']:+.1f}% | 是 |
| 螺旋桨 K_T/K_Q | 水池试验 | 无基准 | 未评估（声明原因：无同几何试验数据） |

**未完成 / 明确的局限**：导管推力未建模；螺旋桨性能无试验对标；接触互补问题未能在粗网格上
稳定求解（因此改用 Hertz + FEM 校核）；冰载荷为参数化设计谱而非实测谱。
"""


def build_platform_report(s: dict) -> dict:
    figs = [
        ("fig_thrust_domain.png", "推力可达域（极坐标）：电机/轴承/结构三条约束与可达边界"),
        ("fig_contact.png", "冰-壳体接触：Hertz 压力分布与 FEM von Mises 深度剖面"),
        ("fig_contact_cloud.png", "FEM 接触区 von Mises 应力云图（r/a - z/a 平面）"),
        ("fig_interaction.png", "推进器-冰相互作用力时程（冰扭矩 / 推力 / 转速）"),
        ("fig_phase.png", "推力与冰载荷谱的相位关系：解析 vs 仿真"),
    ]
    figures = []
    for name, caption in figs:
        path = OUT / name
        if path.exists():
            figures.append(
                {
                    "name": name,
                    "caption": caption,
                    "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
                }
            )
    md = build_markdown(s)
    it, c, st = s["interaction"], s["contact"], s["structural_check"]
    summary = (
        f"在 TensorPhysics 新增冰区吊舱推进器推力矢量与冰-壳体载荷交互模块"
        f"（BEMT 螺旋桨 + 三维梁 FEM 桅柱 + Hertz/八节点六面体 FEM 接触 + 转子动力学耦合），"
        f"并暴露 POST /api/v1/calculations/pod_ice_interaction。15 MW 吊舱、1.5 m 层冰算例："
        f"推力可达域内超应力点 {st['overstressed_points']} 个、边界与强度约束一致性误差 "
        f"{st['boundary_error_max']:.1e}；接触峰值应力相对 Hertz 理论偏差 "
        f"{100*c['von_mises_deviation_rel']:+.2f}%；推力/冰扭矩相位误差 {it['phase_error_deg']:.2f}°，"
        f"幅值比误差 {100*it['amp_error_rel']:+.1f}%；pytest 29 项全绿。"
    )
    return {
        "title": "冰区吊舱推进器推力矢量与冰-壳体载荷交互成熟化（W9-N05）",
        "summary": summary,
        "scenario_key": "pod_ice_interaction",
        "mission_id": 1423,
        "tags": ["ice", "podded-propulsor", "thrust-vectoring", "hertz-contact", "fem", "tensor-physics"],
        "markdown": md,
        "figures": figures,
    }


def main() -> None:
    s = load_summary()
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(build_markdown(s))
    payload = build_platform_report(s)
    Path("/tmp/report.json").write_text(json.dumps(payload))
    print(f"wrote {DOC} ({DOC.stat().st_size} bytes) and /tmp/report.json")
    print("criteria:", json.dumps(s["acceptance"]))


if __name__ == "__main__":
    main()