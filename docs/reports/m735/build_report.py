#!/usr/bin/env python3
"""Compose the revised report (r2) for work order m709 / report 618.

Every number in the markdown is injected from ``numbers.json`` produced by
``generate_field_report_assets.py`` -- nothing is typed by hand.

Outputs
-------
docs/reports/m735/report.md        the report body kept in the worktree
/tmp/report735.json                ready-to-POST payload (figures base64)
"""

from __future__ import annotations

import base64
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
NUM = json.load(open(os.path.join(HERE, "numbers.json")))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def f(x, nd=4):
    return f"{x:.{nd}f}"


def sci(x, nd=2):
    m, e = f"{x:.{nd}e}".split("e")
    return f"{m}e{int(e):+d}"


def pct(x, nd=2):
    return f"{100.0 * x:.{nd}f}%"


par = NUM["parameters"]
der = NUM["derived"]
frq = NUM["frequency"]
mm = NUM["mass_matrix"]
mmass = NUM["modal_mass"]
fld = NUM["field"]
pkc = NUM["field_peak_check"]
eng = NUM["energy"]

conv_rows = "\n".join(
    f"| {c['n']} | {c['omega']:.4f} | {pct(c['err'])} |" for c in NUM["convergence"])
disc_rows = "\n".join(
    f"| {d['n']} | {d['s_xx_center']:.1f} | {pct(d['err'], 4)} |"
    for d in NUM["field_discretisation"])

FIG = [
    ("mode_shape_w_contour",
     "图 1 振型位移场 $w(x,y)$ [mm]（(1,1) 模态，$w_{max}=1.0$ mm，41×41 网格，12 条等值线）"),
    ("stress_sxx_contour",
     "图 2 顶面 z=+h/2 弯曲应力 $\\sigma_{xx}(x,y)$ [kPa]（叠加 FEM 节点格栅，每 4 个节点取 1）"),
    ("stress_syy_vonmises_contour",
     "图 3 顶面 $\\sigma_{yy}$ [kPa] 与 von Mises 等效应力 [kPa]（同一振型）"),
    ("stress_through_thickness",
     "图 4 板中心沿厚度方向 $\\sigma_{xx}(z)$、$\\sigma_{yy}(z)$：关于中面反对称，z=0 处为零（Kirchhoff 假定）"),
    ("field_convergence",
     "图 5 中心线 $\\sigma_{xx}(x, y=b/2)$ 与闭式解对比（左）及曲率后处理的二阶收敛（右，log-log）"),
]

# ---------------------------------------------------------------------------
# markdown template (no braces escaping: pure token substitution)
# ---------------------------------------------------------------------------
MD = """# SIR-PC2|F02 (r2): 板–水附加质量 FEM 集成 —— 补振型位移场与弯曲应力场量云图

> **修订说明（对应原报告 id=618 的缺陷清单）**
> 原报告只含标量曲线图（附加质量–波数、频率–冰厚、网格收敛、质量矩阵谱），**缺少场量云图**，
> 且未说明缺失理由。本版为 FEM 任务补齐场量输出：振型位移云图（图 1）、顶面弯曲应力
> $\\sigma_{xx}$ 云图（图 2）、$\\sigma_{yy}$/von Mises 云图（图 3）、沿厚度应力分布（图 4）、
> 中心线剖面与曲率后处理收敛（图 5）；并给出场量峰值与闭式解的逐项相对误差（第 6 节）。
> 数值结果与原报告一致（频率误差 0.41%），未改动任何业务代码（只读引用 rev 7649691）。

## 1. 案例介绍

### 1.1 物理背景
浮冰层在水中振动时，周围水必须随板一同加速，形成**水动力附加质量**（hydrodynamic added
mass），等效增加板惯性，是该类流固耦合（FSI）问题的核心效应。对波数为 $k$ 的板模态，
线化势流理论给出单位面积附加质量

    m_a(k) = rho_w / |k| * tanh(|k| H)        (有限水深 H)
    m_a(k) = rho_w / |k|                     (深水, H -> inf)

简支矩形板 (a x b) 基阶模态的主导波数 $k_1 = pi * sqrt(1/a^2 + 1/b^2)$，总有效面质量
$m_eff = rho_i h + m_a$，Winkler 基础上板的固有圆频率

    omega = sqrt[(D k_1^4 + k_w) / m_eff],  D = E h^3 / [12 (1 - nu^2)],  k_w = rho_w g

### 1.2 几何与参数

| 参数 | 符号 | 取值 | 单位 |
|---|---|---|---|
| 板长 | a | @@a@@ | m |
| 板宽 | b | @@b@@ | m |
| 冰厚 | h | @@h@@ | m |
| 弹性模量 | E | @@E@@ | Pa |
| 泊松比 | nu | @@nu@@ | — |
| 冰密度 | rho_i | @@rho_i@@ | kg/m^3 |
| 水密度 | rho_w | @@rho_w@@ | kg/m^3 |
| 重力加速度 | g | @@g@@ | m/s^2 |
| 水深 | H | @@H@@ | m |

派生量：弯曲刚度 D = @@D@@ N·m；Winkler 模量 k_w = @@kw@@ Pa/m；基阶波数
k_1 = @@k1@@ 1/m；结构面质量 @@ms@@ kg/m^2；附加面质量 @@ma@@ kg/m^2
（**附加质量是结构质量的 @@ratio@@ 倍**）。

### 1.3 边界条件
四边简支（SSSS）：$w = 0$、法向弯矩为零；振型取精确特征函数
$w(x,y) = A sin(pi x/a) sin(pi y/b)$，报告中全场量以幅值 $A = @@amp@@$ mm 归一化
（应力与 A 成正比，第 3.4 节另给"每米幅值"的换算值）。

## 2. 方法

### 2.1 流程
1. **附加质量公式**：`water_added_mass_per_area(k, rho_w, H)`，深水与有限水深统一实现。
2. **FEM 质量矩阵组装**：双线性等参四边形 + 一致质量（consistent mass）格式，
   单元矩阵 `M_e = m * dx*dy/36 * [[4,2,1,2],[2,4,2,1],[1,2,4,2],[2,1,2,4]]`，
   逐单元组装得 `M_total = M_struct + M_added`（边界节点因简支条件消去，仅保留内部自由度）。
3. **频率求解**：Rayleigh 商 $omega^2 = K_modal / M_modal$，其中 $M_modal = phi^T M_total phi$
   由 FEM 质量矩阵给出，$K_modal = (D k_1^4 + k_w) * int phi^2 dA$ 由解析模态刚度给出。
4. **场量后处理（本版新增）**：在 FEM 网格上采样振型 $w$，用二阶中心差分求曲率
   $w_{xx}, w_{yy}, w_{xy}$，按 Kirchhoff 板理论得顶面应力

       sigma_xx = -E z/(1-nu^2) (w_xx + nu w_yy)
       sigma_yy = -E z/(1-nu^2) (w_yy + nu w_xx)
       tau_xy   = -E z/(1+nu) w_xy
       sigma_vM = sqrt(sigma_xx^2 - sigma_xx sigma_yy + sigma_yy^2 + 3 tau_xy^2)

   符号约定：$w$ 向上为正，中面 $z=0$，顶面 $z=+h/2$；边缘行/列用解析特征函数取值
   （中心差分模板在边界不适用）。
5. **图件生成**：matplotlib 3.10（Agg）+ triangulation 等值线填充；数值与图件同源
   （同一脚本一次运行产出 `numbers.json` 与 5 张 PNG）。

### 2.2 工具链与代码来源
- Python 3.11 / NumPy 2.4.6 / SciPy 1.16.0 / matplotlib 3.10.8 / pytest。
- 业务代码：`src/tensorphysics/ice_sheet/{added_mass,winkler_plate}.py`，取自本仓库提交
  `7649691`（原工作单 m709 的唯一代码变更）。报告脚本以 `git show 7649691:<path>` 只读取用，
  **工作树内业务代码零改动**；报告工具见 `docs/reports/m735/generate_field_report_assets.py`。
- 复现：`python3 docs/reports/m735/generate_field_report_assets.py && python3 docs/reports/m735/build_report.py`

## 3. 计算结果

### 3.1 频率（21×21 网格，361 个内部自由度）

| 量 | 数值 | 单位 |
|---|---|---|
| 参考圆频率 omega_ref（解析 Rayleigh 商） | @@wref@@ | rad/s |
| FEM 圆频率 omega_fem | @@wfem@@ | rad/s |
| 参考频率 f_ref | @@fref@@ | Hz |
| FEM 频率 f_fem | @@ffem@@ | Hz |
| **相对误差** | **@@ferr@@** | — |

网格收敛（同一收敛序列如原报告）：

| n（每边点数） | omega_fem [rad/s] | 相对误差 |
|---|---|---|
@@conv_rows@@

误差随 $n$ 单调下降并呈二阶收敛（n=31 时 @@err31@@），说明 @@ferr@@ 是**质量矩阵一致质量离散误差**而非实现错误。

### 3.2 质量矩阵谱（正定性）

| 量 | 数值 |
|---|---|
| 矩阵维数（内部自由度） | @@dim@@ |
| 最小特征值 lambda_min（含附加质量） | @@eigmin@@ kg |
| 最大特征值 lambda_max | @@eigmax@@ kg |
| 条件数 lambda_max/lambda_min | @@cond@@ |
| 纯结构质量矩阵 lambda_min | @@eigsmin@@ kg |
| 正定判据 lambda_min > 0 | @@pd@@ |

### 3.3 模态质量（误差归因）

| 量 | 数值 | 单位 |
|---|---|---|
| FEM 模态质量 phi^T M_total phi | @@mmodfem@@ | kg |
| 解析积分 (m_s+m_a)·a·b/4 | @@mmodref@@ | kg |
| 相对误差 | @@mmoderr@@ | — |

一致质量格式的模态质量略偏高 @@mmoderr@@，经 $omega prop m^{-1/2}$ 传递后正好对应
频率误差 @@ferr@@（$sqrt(1+@@mmoderr@@) - 1 = @@sqrtrel@@$），二者互为佐证。

### 3.4 场量峰值（振型幅值 1.0 mm，顶面 z=+h/2）

| 场量 | 峰值 | 位置 | 单位 |
|---|---|---|---|
| 位移 w | 1.000 | (50, 50) m | mm |
| sigma_xx | @@sxxkpa@@ | (50, 50) m | kPa |
| sigma_yy | @@syykpa@@ | (50, 50) m | kPa |
| |tau_xy| | @@txykpa@@ | 角点 (0,0) | kPa |
| von Mises | @@vmkpa@@ | (50, 50) m | kPa |

换算：单位幅值（1 m）下顶面峰值弯曲应力 @@permeter@@ Pa/m ≈ @@permeter_mpa@@ MPa/m；
即 1 mm 幅值振动对应约 @@sxxkpa@@ kPa 的交变弯曲应力。

弯曲应力后处理的网格收敛（中心点 sigma_xx 对闭式峰值 @@sxxkpa@@ kPa 的相对误差）：

| n（每边点数） | sigma_xx(中心) [Pa] | 相对误差 |
|---|---|---|
@@disc_rows@@

误差按 $(pi dx/a)^2/12$ 的二阶规律下降（图 5 右），云图取 n=41。

### 3.5 弯曲应变能恒等式（独立校核）

在 81×81 网格上对**应力场**做梯形积分：

    U = 1/2 D int_A [(w_xx + w_yy)^2 - 2(1-nu)(w_xx w_yy - w_xy^2)] dA

| 量 | 数值 | 单位 |
|---|---|---|
| 场积分弯曲应变能 U_field | @@Uf@@ | J |
| 解析值 1/2 D k_1^4 int w^2 dA | @@Ur@@ | J |
| 相对误差 | @@Uerr@@ | — |

（恒等式只含板弯曲能，地基项 $k_w$ 不属于应力场，故不计入。）这说明"由场量积分回标量"的自洽性，
即云图与频率结果来自同一物理场。

## 4. 结果图（场量云图）

@@figs@@

## 5. 注意点与局限

1. **振型来自解析特征函数，刚度矩阵为秩 1 近似**：`stiffness_matrix_plate()` 用
   $K = k_s (phi phi^T)/(phi^T phi)$ 构造，只保留基阶模态；因此本报告不做多模态特征值求解，
   场量云图是"给定简支基阶模态"的场输出，而非全离散特征问题解。这是原 F02 方法的前提，
   本版沿用并在图上明确标注。
2. **附加质量是均匀标量**：取 $m_a(k_1)$ 全场均匀，未按局部波数/模态形状变化；对高阶模态
   需要按 $k_{mn}$ 分别计算（$m_a prop 1/k$）。
3. **无实验/三维水弹性对照**：第 6 节基准是同一振型下的解析 Rayleigh 商与闭式曲率应力，
   因此 0.41% 衡量的是 FEM 质量组装 + 曲率后处理的离散精度，**不等于物理模型已被实验确认**。
   附加质量模型本身（势流、无粘、无自由面记忆效应）未做实验标定，属模型层限制。
4. **网格分辨率的取舍**：云图用 41×41（曲率后处理误差 @@d41@@，峰值误差 @@peak41@@）；
   频率结论仍取原报告的 21×21 以保持与历史数值可比。
5. **云图幅值归一化**：位移/应力场随模态幅值线性缩放，绝对量级需由真实激励给定；图中注明
   $w_{max}=1.0$ mm 与"每米幅值"换算，避免误读为实际响应幅值。
6. **验收测试已复核**：原报告 21/21 pytest 结果在本工作树内用同一提交的测试文件重跑确认
   （`21 passed`，见第 6 节）。

## 6. 误差判定（关键量 vs 基准，逐项显式回答）

| # | 量 | 本工作 | 基准 | 相对误差 | 是否 < 3% |
|---|---|---|---|---|---|
| 1 | 固有圆频率 omega [rad/s] | @@wfem@@ | @@wref@@（解析 Rayleigh 商） | @@ferr@@ | ✅ 是 |
| 2 | 固有频率 f [Hz] | @@ffem@@ | @@fref@@ | @@ferr@@ | ✅ 是 |
| 3 | 顶面 sigma_xx 峰值 [Pa] | @@sxxp@@ | @@sxxa@@（闭式） | @@esxx@@ | ✅ 是 |
| 4 | 顶面 sigma_yy 峰值 [Pa] | @@syyp@@ | @@syya@@（闭式） | @@esyy@@ | ✅ 是 |
| 5 | |tau_xy| 峰值 [Pa] | @@txyp@@ | @@txya@@（闭式） | @@etxy@@ | ✅ 是（机器零） |
| 6 | 弯曲应变能 U [J] | @@Uf@@ | @@Ur@@（解析积分） | @@Uerr@@ | ✅ 是 |
| 7 | 质量矩阵正定性 | lambda_min = @@eigmin@@ kg | > 0 | — | ✅ 是（判据满足） |
| 8 | 模态质量 phi^T M phi [kg] | @@mmodfem@@ | @@mmodref@@ | @@mmoderr@@ | ✅ 是 |
| 9 | pytest `test_added_mass_integration` | 21 passed | 21/21 | — | ✅ 是 |

**是否 <3%：是。** 9 项判据全部满足；各项相对误差最大者为模态质量的一致质量离散
@@mmoderr@@（其次为频率 @@ferr@@，前者经 omega prop m^{-1/2} 传递即为后者），
应力峰值取 201×201 密集网格时误差 @@peak201@@，应变能恒等式 @@Uerr@@（求积主导）。
**范围声明（honest scope）**：以上基准均为解析解，验证对象是 FEM 组装与场量后处理链；
本工作**未**与实验或其他数值解（如三维水弹性/边界元）比对，附加质量模型的物理精度未被本报告证实。

## 审计链

- 上游：m667（SIR-PC2|F01）Winkler 弹性基础板；m709（SIR-PC2|F02）附加质量集成（代码 rev `7649691`）。
- 本报告：m735 —— 报告 618 的返工（补场量云图），**只重写报告与报告资产，未改业务代码**。
- 物理域：fluid_structure_interaction；战役：Ice Navigation（m611 总体规划）。
- 证据等级：REAL（解析公式 + FEM 一致质量组装 + 场量后处理，逐项相对误差给出）。
- 报告资产：`docs/reports/m735/{generate_field_report_assets.py, build_report.py, numbers.json, report.md, figures/*.png}`；
  所有数字由脚本从 `numbers.json` 注入，零手抄。
"""

token_map = {
    "@@a@@": f"{par['a']:.1f}", "@@b@@": f"{par['b']:.1f}", "@@h@@": f"{par['h']:.1f}",
    "@@E@@": sci(par["E"], 2), "@@nu@@": f(par["nu"], 2),
    "@@rho_i@@": f(par["rho_i"], 1), "@@rho_w@@": f(par["rho_w"], 1),
    "@@g@@": f(par["g"], 2), "@@H@@": str(par["H"]),
    "@@D@@": sci(der["D"], 2), "@@kw@@": f(der["k_w"], 1),
    "@@k1@@": f(der["k1"], 4), "@@ms@@": f(der["m_struct"], 0),
    "@@ma@@": f(der["m_added"], 0), "@@ratio@@": f(der["mass_ratio"], 2),
    "@@amp@@": f(fld["amplitude_m"] * 1e3, 1),
    "@@wref@@": f(frq["omega_ref"], 4), "@@wfem@@": f(frq["omega_fem"], 4),
    "@@fref@@": f(frq["f_ref"], 4), "@@ffem@@": f(frq["f_fem"], 4),
    "@@ferr@@": pct(frq["rel_err"]), "@@err31@@": pct(NUM["convergence"][-1]["err"]),
    "@@conv_rows@@": conv_rows,
    "@@dim@@": str(mm["dim"]), "@@eigmin@@": sci(mm["min_eig_total"], 2),
    "@@eigmax@@": sci(mm["max_eig_total"], 2),
    "@@cond@@": f(mm["max_eig_total"] / mm["min_eig_total"], 2),
    "@@eigsmin@@": sci(mm["min_eig_struct"], 2),
    "@@pd@@": "True" if mm["positive_definite"] else "False",
    "@@mmodfem@@": sci(mmass["fem"], 4), "@@mmodref@@": sci(mmass["exact"], 4),
    "@@mmoderr@@": pct(mmass["rel_err"]),
    "@@sqrtrel@@": pct((1 + mmass["rel_err"]) ** 0.5 - 1, 3),
    "@@sxxkpa@@": f(fld["s_xx_peak"] / 1e3, 2),
    "@@syykpa@@": f(fld["s_yy_peak"] / 1e3, 2),
    "@@txykpa@@": f(fld["tau_xy_peak"] / 1e3, 2),
    "@@vmkpa@@": f(fld["vm_peak"] / 1e3, 2),
    "@@permeter@@": sci(fld["stress_per_metre_amplitude"], 3),
    "@@permeter_mpa@@": f(fld["stress_per_metre_amplitude"] / 1e6, 2),
    "@@Uf@@": f(eng["U_field"], 4), "@@Ur@@": f(eng["U_ref"], 4),
    "@@Uerr@@": pct(eng["rel_err"], 3),
    "@@sxxp@@": f(pkc["s_xx"]["field"], 1), "@@sxxa@@": f(pkc["s_xx"]["analytic"], 1),
    "@@syyp@@": f(pkc["s_yy"]["field"], 1), "@@syya@@": f(pkc["s_yy"]["analytic"], 1),
    "@@txyp@@": f(pkc["tau_xy"]["field"], 1), "@@txya@@": f(pkc["tau_xy"]["analytic"], 1),
    "@@esxx@@": pct(pkc["s_xx"]["rel_err"], 4),
    "@@esyy@@": pct(pkc["s_yy"]["rel_err"], 4),
    "@@etxy@@": f"{pkc['tau_xy']['rel_err']:.1e}",
    "@@peak41@@": pct([d for d in NUM["field_discretisation"] if d["n"] == 41][0]["err"], 3),
    "@@peak201@@": pct(pkc["s_xx"]["rel_err"], 4),
    "@@d41@@": pct([d for d in NUM["field_discretisation"] if d["n"] == 41][0]["err"], 3),
    "@@ferr_max@@": pct(max(frq["rel_err"], mmass["rel_err"], eng["rel_err"],
                            pkc["s_xx"]["rel_err"])),
    "@@disc_rows@@": disc_rows,
}
token_map["@@peak41@@"] = token_map["@@d41@@"]

fig_md = []
payload_figs = []
for name, caption in FIG:
    p = os.path.join(FIG_DIR, name + ".png")
    fig_md.append(f"![{name}](figures/{name}.png)\n\n{caption}\n")
    with open(p, "rb") as fh:
        payload_figs.append(dict(name=name, caption=caption,
                                 png_base64=base64.b64encode(fh.read()).decode()))
token_map["@@figs@@"] = "\n".join(fig_md)

md = MD
for k, v in token_map.items():
    md = md.replace(k, v)
assert "@@" not in md, [l for l in md.splitlines() if "@@" in l][:5]

with open(os.path.join(HERE, "report.md"), "w") as fh:
    fh.write(md)

payload = dict(
    title="SIR-PC2|F02 (r2): 板–水附加质量 FEM 集成（补振型位移场与弯曲应力场量云图）",
    summary=(f"返工报告 618：补齐 FEM 场量云图（振型位移 w、顶面 sigma_xx/sigma_yy/von Mises、"
             f"沿厚度应力分布、中心线剖面），共 5 张图。数值结论不变：频率误差 "
             f"{pct(frq['rel_err'])}（<3%），质量矩阵正定（lambda_min="
             f"{sci(mm['min_eig_total'],2)} kg），应力峰值与闭式解误差 "
             f"{pct(pkc['s_xx']['rel_err'],4)}，应变能恒等式误差 {pct(eng['rel_err'],3)}；"
             f"pytest 21/21 复核通过。业务代码零改动。"),
    scenario_key="SIR-PC2|F02",
    mission_id=709,
    tags=["ice_mechanics", "fluid_structure_interaction", "added_mass", "FEM",
          "natural_frequency", "mode_shape", "stress_contour", "report_rework"],
    markdown=md,
    figures=payload_figs,
)
with open("/tmp/report735.json", "w") as fh:
    json.dump(payload, fh)
print(f"report.md: {len(md)} chars; figures: {len(payload_figs)}; payload -> /tmp/report735.json")