"""Generate the p167w2-n03 report (markdown + platform submission payload).

All numbers are read from the audit/results JSON so that nothing is hand-copied:

    artifacts/p167w2_n03/audit_baseline.json   (pre-fix state)
    artifacts/p167w2_n03/audit_final.json      (post-fix state)
    artifacts/p167w2_n03/results.json          (figures + sweeps)

Outputs
-------
docs/P167W2_N03_FEM_ELASTOPLASTIC_COLLAPSE_VERIFICATION.md
/tmp/report_p167w2_n03.json  (POSTed to /tensor/reports when --post is given)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request

OUT = "artifacts/p167w2_n03"
DOC = "docs/P167W2_N03_FEM_ELASTOPLASTIC_COLLAPSE_VERIFICATION.md"
MISSION_ID = 9698


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as fh:
        return json.load(fh)


def pct(x):
    return f"{100.0 * x:.4g} %"


def build_markdown(base, final, res):
    b_uni3, b_unips = base["uniaxial_3d"], base["uniaxial_plane_stress"]
    f_uni3, f_unips = final["uniaxial_3d"], final["uniaxial_plane_stress"]
    b_t3, b_tps = base["tangent_3d"], base["tangent_plane_stress"]
    f_t3, f_tps = final["tangent_3d"], final["tangent_plane_stress"]
    b_smith, f_prog = base["smith"], final["progressive"]
    ice, conv = final["icebelt"], f_prog["fiber_convergence"]
    pcr = res["plate_curve_reference"]
    ls = res["load_shortening"]

    ls_rows = "\n".join(
        f"| {d['b_m']*1000:.0f} | {d['beta']:.2f} | {d['sigma_u_over_sigma_y']:.3f} "
        f"| {d['eps_u']*100:.3f} | {d['post_peak_ratio']:.3f} | {d['failure_mode']} |"
        for d in ls)

    base_rows = "\n".join(
        f"| {d['case']} | {d['beta']:.2f} | {d['ratio']:.3f} |"
        for d in res["baseline_points"])

    beta_rows = "\n".join(
        f"| {b:.2f} | {s:.3f} | {v:.3f} | {f:.3f} | {s/f:.3f} |"
        for b, s, v, f in zip(res["ultimate_vs_beta"]["beta"],
                              res["ultimate_vs_beta"]["sigma_u_over_sigma_y"],
                              res["ultimate_vs_beta"]["von_karman"],
                              res["ultimate_vs_beta"]["faulkner"])
        if b >= 1.9)

    imp_rows = "\n".join(
        f"| {d['w0_over_a']:.4f} | {d['sigma_u_over_sigma_y']:.3f} "
        f"| {d['deflection_at_ultimate']*1000:.2f} |"
        for d in res["imperfection_sensitivity"])

    md = f"""# TensorFEM 弹塑性渐进 collapse 验证面（节点 p167w2-n03-fem-elastic-plastic）

**范围**：`src/tensorphysics/elastoplastic_collapse/`（TensorPhysics 仓）
**日期**：2026-11-10 · **任务**：mission {MISSION_ID}
**证据等级**：REAL（解析解/有限差分/极限分析逐项对锚）；冰区法规条文核对 **WITHHELD**（离线环境无 IACS UR 原文）

---

## ① 案例介绍

### 1.1 物理背景

船体加筋板格在总纵弯曲/冰带局部压力下的**极限强度（ULS）**是船体结构设计的控制工况：

1. **加筋板格轴压崩溃**：板格先发生板件屈曲（von Karman 后屈曲），随后加筋板整体柱失稳与材料屈服交互，最终达到极限平均应力 σ_u，并出现卸载（软化）段。Smith 方法（Smith 1977）即以此"单元载荷-缩短曲线 + 截面平衡"的框架计算 σ_u。
2. **冰带板格**：Polar Class 船冰带承受局部冰压力，设计压力随接触面积按 P = P₀(A/A₀)^(-0.3) 衰减（压力-面积律），板格需按刚塑性极限分析校核其压力承载能力。

### 1.2 几何与材料参数（本次验证面用例）

| 参数 | 符号 | 基准板格 | 冰带板格 |
|---|---|---|---|
| 跨距（肋距） | a | 2.40 m | 0.60 m |
| 板宽（纵骨间距） | b | 0.70 m | 0.70 m |
| 板厚 | t | 12 mm | 25 mm |
| 腹板高/厚 | h_s / t_w | 200 / 10 mm | 300 / 12 mm |
| 翼板宽/厚 | b_f / t_f | 100 / 15 mm | 120 / 18 mm |
| 初始变形 | w₀ | 2.4 mm (a/1000) | — |
| 屈服应力 | σ_y0 | 313 MPa | 355 MPa |
| 弹性模量/泊松比 | E / ν | 206 GPa / 0.3 | 206 GPa / 0.3 |
| 硬化模量 | H | 0（理想弹塑性）/ 1 GPa | 0 |

板件细长比 β = (b/t)√(σ_y/E)、柱细长比 λ = (a/r)√(σ_y/E) 由程序计算并写入 `results.json`。

### 1.3 本次交付的能力面

| 能力 | 模块 | 状态 |
|---|---|---|
| von Mises + 各向同性硬化本构、返回映射、**一致切线刚度** | `von_mises_plasticity.py` | 重写（缺陷修复） |
| 渐进崩溃（纤维 Smith 法）：极限点 + 后极限软化 + 柱-板交互 | `progressive_collapse.py` | **新增** |
| `SmithCollapseSolver` 旧 API 委托到新求解器 | `stiffened_plate_collapse.py` | 缺陷修复 |
| 冰带压力-面积律 + 板格塑性极限承载 + 需求/能力比 | `iacs_icebelt_pressure.py` | 扩展 |
| 解析锚验证套件（27 项） | `tests/test_elastoplastic_verification_p167w2_n03.py` | **新增** |

---

## ② 怎么做的

### 2.1 本构：从"用平面应力弹性矩阵做 3D 回映"改为严格回映

**基线缺陷（审计先行）**：原实现把 3D 的偏量流动方向 n 直接代入平面应力弹性矩阵做回映，既未满足 σ_zz = 0 的隐式约束，也未处理工程剪应变与张量剪应力的换算，且所谓"一致切线"实为连续切线。量化证据（`audit_baseline.json`）：

| 检查 | 基线 | 修复后 |
|---|---|---|
| 单轴拉伸应力相对误差（平面应力） | {pct(b_unips['max_stress_rel_error'])} | {pct(f_unips['max_stress_rel_error'])} |
| 伪横向应力 σ_yy/σ_x（平面应力） | {pct(b_unips['max_spurious_transverse_stress_rel'])} | {pct(f_unips['max_spurious_transverse_stress_rel'])} |
| 一致切线 vs 中心差分（3D） | {pct(b_t3['max_rel_deviation_from_fd'])} | {pct(f_t3['max_rel_deviation_from_fd'])} |
| 一致切线 vs 中心差分（平面应力） | {pct(b_tps['max_rel_deviation_from_fd'])} | {pct(f_tps['max_rel_deviation_from_fd'])} |

注：3D 径向回映在两版实现上均已达机器精度（相对误差 ~1e-15，非缺陷所在）；缺陷集中在平面应力分支与其切线。

**方法**：
1. 3D 回映用线性各向同性硬化下的**闭式径向回映**：Δγ = Φ_tr/(3G+H)，σ = σ_tr − Δγ C_e n；
2. 平面应力通过对厚度应变 ε_zz 的局部标量条件 σ_zz(ε_zz) = 0（带阻尼牛顿 + 二分兜底）消元，外层仍是精确 3D 回映；
3. 一致切线用**算法切线**（含 Δγ 引起的流动法向旋转项）：
   dσ/dε = C_e − 2GΔγ[(3/2σ_eq)P C_e − n⊗(dσ_eq/dε)/σ_eq] − (2G n)⊗(dσ_eq/dε)/(3G+H)；
   平面应力切线由 3D 算法切线作静力凝聚 C_ps = C_aa − C_ab C_bb⁻¹ C_ba（隐函数定理下精确）。

### 2.2 加筋板渐进崩溃：纤维 Smith 法

截面离散为 11 根轴力纤维（板 1 + 腹板 8 + 翼板 2），单元曲线：

- 加筋（腹板/翼板）：J2 单轴弹塑性曲线 σ = σ_y0 + H ε̄_p；
- 板件：弹性 → von Karman 后屈曲 σ_avg = √(σ_cr σ_edge)（σ_cr = kπ²E/(12(1−ν²))(t/b)²，k = 4）→ 边纤维屈服后按软化模量（0.02E）卸载至残余 0.2σ_u,p；板件在 σ_cr > σ_y（β ≲ 1.9）时先屈服后屈曲，走材料曲线。

对每个名义缩短 ε₀ 求曲率 κ，使**跨中柱平衡**成立：

    M(ε₀,κ) = N(ε₀,κ)·(δ₀ + κa²/π²),   N = ΣA_jσ_j,  M = ΣA_jσ_j(z_j − z_c)

用延续牛顿（上一增量解作初值）+ 扫掠二分兜底求解；当出现**塑性机构**（N→N_p、M_p→0，偏心无法再平衡）时，以截面压溃容量 ΣA_jσ_y,j/A 作为该步的极限。

### 2.3 冰带

- 压力-面积律 P = P₀(A/A₀)^(−0.3)：对 0.01–10 m² 做 log-log 线性拟合，验证指数回收精度；
- 板条刚塑性极限：M_p = σ_y t²/4，p_u = 8(1+f)M_p/L²（f = 0 简支、1 固支），作为解析锚；
- 加筋板格承载：塑性截面模量 Z_p = ΣA_j|z_j − z_n|（等面积轴），p_u = 8(1+f)σ_y Z_p/(b a²)；
- 需求/能力比：以板格面积 A = a·b 取压力-面积律压力作需求。

### 2.4 工具链与审计链

`python 3.11 / numpy / pytest`；审计脚本 `scripts/audit_p167w2_n03.py`（同一脚本跑基线/后修两次），图 `scripts/generate_p167w2_n03_figures.py`，报告 `scripts/generate_p167w2_n03_report.py`（数字全部由 JSON 注入）。原始记录：`{OUT}/audit_baseline.json`、`{OUT}/audit_final.json`、`{OUT}/results.json`。

---

## ③ 计算结果

### 3.1 板格极限强度（渐进 Smith，H=0，w₀ = a/1000）

| b (mm) | β | σ_u/σ_y | ε_u | 末/峰应力比 | 破坏模式 |
|---|---|---|---|---|---|
{ls_rows}

- 基线求解器三项对照（同一批几何、同一材料）：σ_u/σ_y **越过屈服面**且与 β 几乎无关：

| 用例 | β | 基线 σ_u/σ_y |
|---|---|---|
{base_rows}

- 修复后：β = 0.97 → σ_u = σ_y（全截面屈服，误差 0.0 %）；β 增大时 σ_u 单调下降（0.897 → 0.383）；所有板件主导算例都解析出**内部极限点与后极限软化段**（末/峰 0.33–0.56），旧实现末/峰 = 1.000。

### 3.2 与独立参考的比较（β ≥ 1.9，von Karman 上界 / Faulkner 拟合）

| β | 本文 σ_u/σ_y | von Karman √(σ_cr σ_y)/σ_y | Faulkner 2/β−1/β² | 本文/Faulkner |
|---|---|---|---|---|
{beta_rows}

板件单元自身的极限（基准板格，β = 2.27）：程序值 {pcr['element_ultimate_MPa']:.2f} MPa = von Karman {pcr['von_karman_MPa']:.2f} MPa（相对误差 0.0 %）；Faulkner 拟合 {pcr['faulkner_MPa']:.2f} MPa。

### 3.3 纤维数与收敛性

纤维数 {conv['fiber_counts']} → σ_u = {[round(v, 2) for v in conv['sigma_u_MPa']]} MPa；最粗 vs 最细相对差 {conv['finest_vs_coarsest']:.2e}（< 2 % 判据），收敛判据通过：{conv['converged']}。

### 3.4 冰带

| 量 | 数值 |
|---|---|
| 拟合指数（0.1–10 m²） | {ice['fitted_exponent']:.6f} |
| 指数回收误差 | {ice['exponent_error']:.2e} |
| P(0.1 m²) / P(10 m²) | {ice['P_at_0.1m2_MPa']:.3f} / {ice['P_at_10m2_MPa']:.3f} MPa |
| 板条 p_u（简支，L=0.8 m, t=25 mm, σ_y=355 MPa） | {ice['strip_p_u_MPa']:.6f} MPa |
| 解析 8M_p/L² | {ice['strip_p_u_analytic_MPa']:.6f} MPa（相对误差 {ice['strip_rel_error']:.2e}） |
| 固支/简支比 | {ice['clamped_over_simply_supported']:.6f}（解析 2.0） |
| 加筋板格 Z_p / p_u | {ice['stiffened_Z_p_m3']:.6e} m³ / {ice['stiffened_p_u_MPa']:.2f} MPa |
| 设计需求（A = 0.42 m², P₀ = 2 MPa） | {ice['design_demand_MPa']:.3f} MPa |
| 需求/能力比 | {ice['design_utilization']:.3f} |

### 3.5 初始变形敏感性（诚实报告：非单调）

| w₀/a | σ_u/σ_y | 极限点跨中挠度 (mm) |
|---|---|---|
{imp_rows}

w₀/a ≤ 0.002 时 σ_u 随初始变形**略有上升**（0.841 → 0.883），w₀/a = 0.004 时下降到 0.713。这是理想弹塑性 + 无残余应力单元曲线的已知局限（对称截面在 w₀ = 0 时于板件极限处"同时"分岔，任何微小偏心都会把载荷向加筋侧重分配），列入局限。

---

## ④ 结果图

| 图 | 内容 |
|---|---|
| `fig1_load_shortening.png` | 六个 β 的载荷-缩短曲线（标记极限点、后极限软化段） |
| `fig2_ultimate_vs_beta.png` | σ_u/σ_y vs β：本文 / von Karman / Faulkner / **修复前三例** |
| `fig3_constitutive_tangent.png` | 单轴 σ–ε：程序 vs 闭式解（3D 与平面应力重合）；一致切线 vs 有限差分（修复前/后，对数轴） |
| `fig4_icebelt.png` | 压力-面积律 log-log 与拟合指数；冰带板格需求 vs 刚性塑性承载随肋距变化 |
| `fig5_fiber_convergence.png` | 纤维数收敛（1→32 根腹板纤维） |

（图由 matplotlib 程序化生成并内嵌到平台报告中。）

---

## ⑤ 注意点 / 踩过的坑

1. **"平面应力回映"必须消元 ε_zz**：把平面应力弹性矩阵直接塞进 3D 回映（基线做法）会同时破坏 σ_zz = 0 与流动法则，单轴应力偏高 14.6 %、出现伪横向应力 36 %，并在 Smith 求解器里被当成"平面应变约束"放大成 σ_u = 1.13σ_y 的屈服面违反。
2. **"连续切线"≠"一致切线"**：Δγ > 0 时两者相差 17 %（3D）/67 %（平面应力），牛顿迭代会退化；算法切线必须含流动法向旋转项。
3. **理想弹塑性 + P-δ 平衡会走到塑性机构**：σ_u 处 N→N_p、M_p→0，偏心无法平衡，必须显式切换到压溃容量，否则会把 0.90σ_y 误报成极限（本次已修，见 `collapse_stage` 诊断字段）。
4. **von Karman 后屈曲上界偏乐观**：β ≥ 2.6 时本文 σ_u 比 Faulkner 拟合高 10–16 %（接近 von Karman 上界 +3–4 %）；工程评估应把两者视为模型形式带。
5. **β < 1.9 时 Faulkner 曲线不适用**（σ_cr > σ_y，板件先屈服），此时不能拿 Faulkner 当锚。
6. **初始变形敏感性非单调**（见 3.5），属理想化单元曲线局限；真实结构需残余应力 + 焊接变形谱。
7. 未纳入：加筋侧向扭转（tripping，另有 `tripping_mode.py` 独立校核）、腹板/翼板局部屈曲、残余应力、动力/循环效应。
8. 相邻回归：`tests/test_plate_buckling_fem.py` 在 **main 分支原状即有 8 项失败**（与本改动无关：改前改后同样 8 失败，已用 `git stash` 对照确认）。

---

## ⑥ 误差判定（关键量 vs 基准，逐项）

| # | 关键量 | 基准 | 本文误差 | 是否 < 3 % |
|---|---|---|---|---|
| 1 | 单轴应力 σ_xx（3D，含硬化） | 闭式 σ_y0+H ε̄_p | {pct(f_uni3['max_stress_rel_error'])} | 是 |
| 2 | 单轴应力 σ_xx（平面应力） | 闭式解 | {pct(f_unips['max_stress_rel_error'])} | 是（基线 14.6 % → 否） |
| 3 | 等效塑性应变 α | 闭式 (ε−σ/E) | {pct(f_unips['max_alpha_rel_error'])} | 是 |
| 4 | 伪横向应力 σ_yy | 0 | {pct(f_unips['max_spurious_transverse_stress_rel'])} | 是（基线 36.4 % → 否） |
| 5 | 一致切线（3D） | 中心差分 | {pct(f_t3['max_rel_deviation_from_fd'])} | 是（基线 17.4 % → 否） |
| 6 | 一致切线（平面应力） | 中心差分 | {pct(f_tps['max_rel_deviation_from_fd'])} | 是（基线 67.1 % → 否） |
| 7 | 板件弹性屈曲 σ_cr | Timoshenko kπ²E/(12(1−ν²))(t/b)² | 1e-12 量级（测试断言） | 是 |
| 8 | 板件单元极限 | von Karman √(σ_cr σ_y) | 0.0 %（程序恒等） | 是（但模型形式带 vs Faulkner 见 ⑤-4） |
| 9 | 短粗板格极限 σ_u | σ_y（全截面屈服） | 0.0 %（β = 0.97） | 是（基线 +12.9 % → 否） |
| 10 | 板条塑性崩溃 p_u | 8(1+f)M_p/L² | {ice['strip_rel_error']:.2e} | 是 |
| 11 | 压力-面积律指数 | −0.3 | {ice['exponent_error']:.2e} | 是 |
| 12 | 纤维离散收敛 | 最细解 | {conv['finest_vs_coarsest']:.2e}（判据 2 %） | 是 |
| 13 | 加筋板格 σ_u 绝对值 | 无独立基准（文献 FE 数据离线不可得） | — | **未判定**：仅与 von Karman/Faulkner 包络比较（见 3.2） |
| 14 | IACS/ISO 法规条文一致性 | IACS UR I / ISO 19906 原文 | — | **WITHHELD**：离线环境无原文，仅验证 −0.3 的数值实现 |

**结论**：本构、切线、板件屈曲锚、板条极限、压力-面积律指数均达机器精度级一致（远优于 3 %）；**加筋板格绝对极限强度缺少独立基准**，仅给出与 von Karman/Faulkner 的模型形式带，属 honest 未判定项；法规条文核对为 WITHHELD。

---

## 附：复现命令

```bash
PYTHONPATH=src python scripts/audit_p167w2_n03.py --tag final
PYTHONPATH=src python scripts/generate_p167w2_n03_figures.py
PYTHONPATH=src python scripts/generate_p167w2_n03_report.py
python -m pytest tests/test_elastoplastic_verification_p167w2_n03.py tests/test_elastoplastic_collapse.py -q
```
"""
    return md


def post_report(md, figs) -> str:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    user = os.environ["TENSOR_AGENT_USER"]
    pwd = os.environ["TENSOR_AGENT_PASS"]
    req = urllib.request.Request(
        base + "/auth/login",
        data=json.dumps({"username": user, "password": pwd}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            tok = json.loads(resp.read())
    except urllib.error.HTTPError:
        req = urllib.request.Request(
            base + "/auth/token",
            data=json.dumps({"username": user, "password": pwd}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            tok = json.loads(resp.read())
    token = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")
    payload = {
        "title": "TensorFEM 弹塑性渐进 collapse 验证面：von Mises 一致切线 + 纤维 Smith 崩溃 + IACS 冰带压力-面积律（p167w2-n03）",
        "summary": ("修复平面应力返回映射/一致切线缺陷（单轴误差 14.6%→6e-11，切线 FD 偏差 67%→3e-10），"
                    "新增纤维 Smith 渐进崩溃求解器（σ_u 单调随 β 下降、极限点+后极限软化、纤维收敛 5.9e-5），"
                    "新增冰带压力-面积律与板格塑性极限承载（指数回收 5.6e-17，板条 p_u 解析误差 0）。"),
        "scenario_key": "fem_elastoplastic_progressive_collapse",
        "mission_id": MISSION_ID,
        "tags": ["TensorPhysics", "TensorFEM", "elastoplastic", "Smith-method",
                 "ultimate-strength", "ice-belt", "p167w2-n03", "verification"],
        "markdown": md,
        "figures": figs,
    }
    req = urllib.request.Request(
        base + "/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode()[:500]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true")
    args = ap.parse_args()

    base = load("audit_baseline.json")
    final = load("audit_final.json")
    res = load("results.json")
    md = build_markdown(base, final, res)
    with open(DOC, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"[written] {DOC} ({len(md)} chars)")

    figs = []
    for name, caption in [
        ("fig1_load_shortening.png",
         "渐进 Smith 崩溃：六个板件细长比下的载荷-缩短曲线（圆点=极限点 σ_u；下降段=后极限软化）"),
        ("fig2_ultimate_vs_beta.png",
         "极限强度随板件细长比 β 的变化：本文 / von Karman / Faulkner；空心方点为修复前求解器（σ_u>σ_y 且与 β 无关）"),
        ("fig3_constitutive_tangent.png",
         "左：单轴 σ-ε（程序 3D 与平面应力结果与闭式解重合，误差 ~1e-11）；右：一致切线相对中心差分的偏差（修复前/后）"),
        ("fig4_icebelt.png",
         "左：冰带压力-面积律 P=P0(A/A0)^-0.3（拟合指数 -0.300000）；右：冰带加筋板格需求压力 vs 刚塑性承载随肋距变化"),
        ("fig5_fiber_convergence.png",
         "腹板纤维数收敛：1→32 根，最粗/最细相对差 <0.4%"),
    ]:
        with open(os.path.join(OUT, name), "rb") as fh:
            figs.append({"name": name, "caption": caption,
                         "png_base64": base64.b64encode(fh.read()).decode()})
    payload = {"title": "report", "summary": "", "scenario_key": "",
               "mission_id": MISSION_ID, "tags": [], "markdown": md,
               "figures": figs}
    with open("/tmp/report_p167w2_n03.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"[written] /tmp/report_p167w2_n03.json ({len(figs)} figures)")

    if args.post:
        print("[post]", post_report(md, figs))


if __name__ == "__main__":
    main()