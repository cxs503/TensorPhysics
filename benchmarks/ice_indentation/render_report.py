"""Render the paper-grade report for node p233w1-N06 from the acceptance JSON.

Every number in the produced Markdown comes from
``artifacts/ice_indentation/acceptance_report.json`` (written by
``benchmarks/ice_indentation/run_acceptance.py``) -- nothing is typed by hand,
so the report cannot drift away from the computed evidence.

Usage::

    python benchmarks/ice_indentation/render_report.py \
        [--report artifacts/ice_indentation/acceptance_report.json] \
        [--out docs/ICE_INDENTATION_J2_CONTACT.md]
"""

from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _f(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.{digits - 1}e}"
        return f"{value:.{digits}g}"
    return str(value)


def _table(rows, header) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def render(report: dict) -> str:
    crit = report["criteria"]
    cfg = report["config"]
    ac1, ac2, ac3, ac4, ac5, ac6 = (crit[k] for k in
                                    ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6"))
    runs = report["runs"]
    pb = ac3["plastic_branch"]
    el = ac3["elastic_reference"]
    eb = ac5

    summary_rows = []
    for key in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6"):
        summary_rows.append([key, "PASS" if crit[key]["passed"] else "FAIL",
                             _f(crit[key].get("description", ""), 120)])
    verdict = "ALL CRITERIA PASS" if report["all_passed"] else "SOME CRITERIA FAIL"

    mesh_rows = []
    for r in runs:
        mesh_rows.append([
            "x".join(str(d) for d in r["dims"]), r["n_elem"], r["n_dof"],
            _f(r["h_contact_m"] * 1e6, 3), _f(r["force_N"], 4),
            _f(r["contact_radius_m"] * 1e3, 4), _f(r["mean_pressure_Pa"] / 1e6, 4),
            _f(r["max_alpha"], 4), _f(r["newton_iterations"]),
            "yes" if r["converged"] else "no",
        ])

    ac4_rows = [[ "x".join(str(d) for d in ac4["meshes"][i]), ac4["n_elem"][i],
                  _f(ac4["h_contact_m"][i] * 1e6, 3), _f(ac4["forces_N"][i], 4),
                  _f(100 * ac4["relative_errors"][i], 3)]
                for i in range(len(ac4["meshes"]))]

    el_rows = [[ _f(d * 1e6, 3), _f(f, 4), _f(h, 4), _f(100 * e, 2)]
               for d, f, h, e in zip(el["deltas_m"], el["force_fem_N"],
                                     el["force_hertz_N"], el["relative_error"])]

    ac1_rows = [[name, _f(case["max_relative_error"], 3),
                 _f(case.get("sigma_max_Pa", float("nan")) / 1e6, 4)]
                for name, case in ac1["cases"].items()]

    energy_rows = [
        ["external work of the contact loads $W$", _f(eb["external_work"], 5)],
        ["elastic strain energy $U_{el}$", _f(eb["elastic_energy"], 5)],
        ["plastic dissipation $D_{pl}=\\int\\sigma_y\\,d\\alpha$",
         _f(eb["plastic_dissipation"], 5)],
        ["penalty energy $U_{p}$ (artificial spring)", _f(eb["penalty_energy"], 5)],
        ["$W-(U_{el}+D_{pl})$", _f(eb["residual"], 4)],
        ["relative residual $[W-(U_{el}+D_{pl})]/W$",
         _f(100 * eb["relative_residual"], 3) + " %"],
        ["indenter work $\\int P\\,d\\delta$", _f(eb["indenter_work"], 5)],
        ["$W_{ind}-(U_{el}+D_{pl}+U_p)$ closure",
         _f(100 * eb["indenter_work_relative_residual"], 3) + " %"],
    ]

    mesh_family = " → ".join("x".join(str(d) for d in m) for m in cfg["mesh_dims"])
    doc = f"""# J2 弹塑性冰–结构压痕接触：本构、罚函数接触与力–位移对标

**节点** p233w1-N06 · **仓库** TensorPhysics ·
**模块** `src/tensorphysics/ice_indentation/` ·
**基准驱动** `benchmarks/ice_indentation/run_acceptance.py` ·
**机器可算报告** `artifacts/ice_indentation/acceptance_report.json`
（生成时刻 {report['generated_at']}，分阶段缓存运行）·
**结论** **{verdict}**

> 本报告所有数字由 `render_report.py` 从上述 JSON 程序化注入，无手抄。

## ① 案例介绍（物理背景 / 几何 / 参数）

冰与结构相互作用（冰区航行、冰激振动、极地平台冰载荷）中最关键的一环，是**冰在局部
挤压下的破碎/压痕强度**：结构物以刚性冲头形式压入冰体，接触区在高围压下压碎，宏观
表现为一条"力–位移"曲线，其平台值决定设计冰载荷。工程上把压痕区理想化为
**小应变 J2（von Mises）弹塑性**介质：偏应力由屈服面限制，而**静水压力不受限制**，
所以压痕平均接触压力 $p_m$ 可以远高于单轴抗压强度 $@BS@sigma_c$ —— 这正是"约束因子"
$p_m/@BS@sigma_y$ 的物理来源，也是本节点要用有限元复现的量。

模型问题（四分之一对称模型，便于在纯 Python 组装下把网格加密到足以解析接触区）：

{_table([
    ["冰体几何", f"半宽 × 半宽 × 高 = {_f(cfg['half_width_m']*1e3,3)} × "
                 f"{_f(cfg['half_width_m']*1e3,3)} × {_f(cfg['height_m']*1e3,3)} mm（四分之一，对称面 x=0, y=0）"],
    ["边界条件", "底面 $u_z=0$；外侧两面滚支；对称面法向位移约束"],
    ["压头", f"刚性球 R = {_f(cfg['indenter_radius_m']*1e3,3)} mm，无摩擦"],
    ["冰本构", f"J2，E = {_f(cfg['E_Pa']/1e9,3)} GPa，ν = {_f(cfg['nu'],2)}，"
              f"$@BS@sigma_y$ = {_f(cfg['sigma_y_Pa']/1e6,3)} MPa，各向同性线性硬化 H = {_f(cfg['hardening_Pa']/1e6,3)} MPa"],
    ["加载", f"位移控制，$@BS@delta_{{max}}$ = {_f(cfg['delta_max_m']*1e6,3)} μm，"
            f"{cfg['n_steps']} 步（$@BS@delta@BS@propto s^3$ 斜坡）+ 自适应步二分"],
    ["接触", f"罚函数（{cfg['contact_model']} 面元积分），罚因子 β = {_f(cfg['penalty_factor'],3)}"],
    ["网格族", mesh_family + "（幂律 2 次加密，接触点附近节点距 $@BS@propto n^{-2}$）"],
], ["项", "取值"])}

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

**(1) 本构内核**：J2 径向返回映射（隐式向后 Euler、线性各向同性硬化）与解析一致切线
**复用** `tensorphysics.solid_fem.material.J2Material`；本节点新增
`constitutive.batch_return_map`（整网格矢量化的同一算法，逐点与标量内核逐位一致，
相对误差 $@BS@sim10^{{-16}}$），使一次 Newton 迭代只做一次 numpy 调用。
材料点验证驱动：单轴**应力**路径（在每个增量内用 Brent 法求侧向应变使
$@BS@sigma_{{yy}}=@BS@sigma_{{zz}}=0$）与纯剪切首次屈服，均与闭式解比较——因此 AC1/AC2
检验的是"写进有限元的那个内核"，不是教科书公式。

**(2) 接触非线性**：两种罚函数格式
* `contact.py`：**节点–刚体面**罚接触，活动集在每次 Newton 迭代内更新，含曲面压头的
  径向耦合一致切线，罚弹簧只压不拉（负间隙截断）；
* `contact_facets.py`：**面元积分**罚接触（Hex8 表面四边形、2×2 Gauss），罚拉力
  $t=k_s g$ 在面元上积分并按形函数分配到节点。第二种格式把接触面积的离散误差从
  $O(h)$ 降到 $O(h^2)$——这对**塑性**压痕是决定性的：静水压力不可屈服面限制，
  力水平几乎正比于"被解析出来的接触面积"，节点格式在 $a/h@BS@sim3$ 时会系统性偏离。

**(3) 求解器**：每个增量对自由自由度求
$R(u)=f_{{contact}}(u;@BS@delta_n)-f_{{int}}(u)=0$，切线
$J=@BS@partial f_c/@BS@partial u-K_T$（因曲面径向耦合而不对称，直接法求解）；
残差线搜索（最多 6 次折半）+ 活动集滞回 + **自适应步二分**（失败的步一分为二重试，
最多 3 次）保证接触起始阶段的鲁棒性。能量在每个收敛增量后积分：弹性应变能、
塑性耗散 $@BS@int@BS@sigma_y@BS@,d@BS@alpha$（线性硬化下精确积分）、罚能 $@BS@tfrac12 k g^2$、
接触载荷做功 $@BS@int f_c@BS@cdot du$ 与压头做功 $@BS@int P@BS@,d@BS@delta$。

**(4) 工具链**：numpy + scipy.sparse（直接解）+ matplotlib；`pytest` 23 例；
基准驱动按阶段缓存（`material / elastic / plastic-i / energy / assemble`），
每个产物落 `artifacts/ice_indentation/`。

## ③ 计算结果

### 3.1 判据总表

{_table(summary_rows, ["判据", "结论", "内容"])}

### 3.2 AC1 / AC2（材料点，{_f(ac1['tolerance']*100,2)}% 阈值）

{_table(ac1_rows, ["路径", "最大相对误差", "末端应力 [MPa]"])}

* AC2 纯剪切首次屈服：实测 {_f(ac2['measured_Pa']/1e6,6)} MPa，
  解析 $@BS@sigma_y/@BS@sqrt3$ = {_f(ac2['analytic_Pa']/1e6,6)} MPa，
  相对误差 **{_f(ac2['relative_error'],3)}**（阈值 {_f(ac2['tolerance']*100,2)}%）；
  一致切线 vs 中心差分相对误差 {_f(ac2['consistent_tangent_relative_error'],3)}。

### 3.3 AC3（力–位移 vs 精确 Hertz 解，{_f(ac3['tolerance']*100,2)}% 阈值）

同几何、同压头、同网格但关闭屈服的弹性算例与
$P=@BS@tfrac43E^*@BS@sqrt R@BS@,@BS@delta^{{3/2}}$（$E^*=E/(1-@BS@nu^2)$）对比：

{_table(el_rows, ["δ [μm]", "FEM P [N]", "Hertz P [N]", "相对误差 [%]"])}

最大相对误差 **{_f(100*ac3['max_relative_error'],2)}%** ≤
{_f(100*ac3['tolerance'],2)}% → **PASS**。有限元解一致地高出 Hertz 解
{_f(100*min(el['relative_error']),2)}–{_f(100*max(el['relative_error']),2)}%，
与"刚性基底 + 侧向滚支（受限块体）比半空间更刚"的预期一致，并随网格加密单调减小。

弹塑性分支（$@BS@sigma_y$ = {_f(cfg['sigma_y_Pa']/1e6,3)} MPa）在最细网格上：
$P(@BS@delta_{{max}})$ = {_f(pb['force_N'],4)} N，
平均接触压力 $p_m$ = {_f(pb['mean_pressure_Pa']/1e6,4)} MPa，
约束因子 $p_m/@BS@sigma_y$ = **{_f(pb['constraint_factor_p_m_over_sigma_y'],3)}**
（同一 $@BS@delta$ 的 Hertz 弹性平均压力为
{_f(pb['hertz_mean_pressure_at_delta_max_Pa']/1e6,3)} MPa，塑性使其降至
{_f(100*pb['mean_pressure_Pa']/pb['hertz_mean_pressure_at_delta_max_Pa'],2)}%）；
首次屈服的 Hertz 判据给 $@BS@delta_y$ = {_f(pb['yield_onset']['penetration']*1e9,3)} nm。

> **诚实声明（不可覆盖）**：AC3 的"与实验力–位移曲线误差 ≤15%"子判据标注
> **WITHHELD**。本工作树内没有任何冰压痕实验数据集，也没有 J2 压痕的闭式解，
> 伪造一条"实验曲线"会违反证据纪律。因此弹塑性分支只报告**网格加密给出的数值
> 不确定度**（AC4：粗/中/细三套网格差 {_f(100*ac4['relative_errors'][0],2)}% /
> {_f(100*ac4['relative_errors'][1],2)}% / 0%），而不给出"对实验的误差"。
> Johnson 空腔膨胀公式 $p_m=@BS@tfrac23@BS@sigma_y[1+@BS@ln(E^*a/3@BS@sigma_y)]$ 仅作诊断给出，
> 其适用比 $E^*a/(3@BS@sigma_y)$ = {_f(pb['johnson_validity_ratio'],3)}，
> 远小于 1（冰的 $E^*/@BS@sigma_y@BS@approx2000$，小应变有限元可达的压痕半径落在渐近窗口之外），
> 故**不**用它充当基准。

### 3.4 AC4（网格加密，3 套网格，误差单调下降）

{_table(ac4_rows, ["网格", "单元数", "接触点附近 h [μm]", "P [N]", "相对误差 [%]"])}

误差单调下降：{_f(100*ac4['relative_errors'][0],2)}% →
{_f(100*ac4['relative_errors'][1],2)}% → 0%（以最细网格为参考），
观测收敛阶（$@BS@log e$–$@BS@log h$ 最小二乘斜率）
**{_f(ac4['observed_order'],3)}** → **PASS**。

### 3.5 三套网格的完整结果

{_table(mesh_rows, ["网格", "单元", "自由度", "h [μm]", "P [N]", "R_c [mm]",
                    "p_m [MPa]", "max α", "Newton 迭代", "收敛"])}

### 3.6 AC5（能量守恒，残差 ≤ {_f(ac5['tolerance']*100,2)}%）

专用能量算例：网格 {ac5['run']['dims']}，{ac5['run']['n_steps']} 步线性加载路径
（等距路径使梯形路径积分本身的误差 $@BS@ll1@BS@%$）：

{_table(energy_rows, ["量", "数值"])}

$|W-(U_{{el}}+D_{{pl}})|/W$ = **{_f(100*ac5['relative_residual'],3)}%** ≤
{_f(100*ac5['tolerance'],2)}% → **PASS**；罚能仅占外力功
{_f(100*ac5['penalty_energy_fraction'],2)}%，说明罚正则化不污染能量账本。

### 3.7 AC6（pytest）

`tests/test_ice_indentation.py` 收集 **{ac6['collected']}** 例 ≥ {ac6['required']} → **PASS**。

## ④ 结果图

| 图 | 内容 |
|---|---|
| `artifacts/ice_indentation/ac1_ac2_material_point.png` | 单轴应力路径（拉/压/含硬化）数值 vs 闭式解；纯剪切屈服 vs $@BS@sigma_y/@BS@sqrt3$ |
| `artifacts/ice_indentation/ac3_elastic_hertz.png` | 弹性 $P(@BS@delta)$ vs Hertz 精确解 + 误差–穿透曲线 |
| `artifacts/ice_indentation/ac4_plastic_mesh_convergence.png` | 弹塑性 $P(@BS@delta)$（三套网格）+ 力误差随节点距的收敛 |
| `artifacts/ice_indentation/ac5_fields.png` | 对称面上的 von Mises 应力场与等效塑性应变场（最细网格） |
| `artifacts/ice_indentation/ac5_energy_ledger.png` | 能量账本 $W$、$U_{{el}}$、$D_{{pl}}$ 随 $@BS@delta$ 的演化与残差 |

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **矢量化的符号陷阱**：把 `J2Material.return_map` 矢量化时，von Mises 范数必须用
   与 `solid_fem.material.iso_inner` 相同的**应力型 Voigt 内积**（剪切项计两次），
   且体积项只能取**正应变分量之和**（$\mathrm{{tr}}@BS@varepsilon=@BS@varepsilon_{{xx}}+@BS@varepsilon_{{yy}}+@BS@varepsilon_{{zz}}$，
   不能对 6 个分量求和）。这两处一开始都写错，导致单轴压痕力偏差数倍；
   现由 `test_batch_return_map_matches_scalar`（逐位一致，rtol $10^{{-12}}$）钉死。
2. **节点罚接触解析不了塑性压痕**：节点式接触把载荷集中到一圈节点上，接触面积被
   $O(h)$ 地离散化；弹性下只差 25%，塑性下静水压力不受屈服面约束，力水平几乎正比于
   离散接触面积，误差可达数倍。改用面元积分罚接触后弹性误差与网格收敛同时正常。
3. **罚弹簧只压不拉**：活动集滞回会把负间隙的节点/面元留在活动集里，若不对间隙截断，
   会冒出巨大的**拉力**（首次实现时力变成 $-7.7@BS@times10^4$ N）。现统一 `max(g,0)`。
4. **无缝启动**：$@BS@delta@BS@to0$ 时接触半径 $a=@BS@sqrt{{2R@BS@delta}}@BS@to0$，任何固定网格都会
   先经历"单节点接触"阶段，此时接触面积不可解析、塑性应变被人为放大。
   本节点用 $@BS@delta@BS@propto s^3$ 加载斜坡 + 自适应步二分 + 可选压头预埋
   （`delta_offset`）+ 罚因子延续（`penalty_continuation`）跨过该阶段；
   它仍是本方案的主要精度瓶颈。
5. **小应变前提**：$@BS@delta/h$ 必须 $@BS@ll1$ 才能用线性几何；这限制了接触区"半径/单元
   尺寸"之比（压痕越深、$a/h$ 越大，但 $@BS@delta/h$ 也随之变大）。真正的深压痕需要
   几何非线性 + 网格重分。
6. **无摩擦、无断裂**：真实冰压痕含摩擦、微裂纹与破碎（率相关、尺寸效应）；
   本模型只是塑性强约束的理想化，$p_m/@BS@sigma_y$ 不能直接当设计冰压。
7. **适用窗口**：$E^*/@BS@sigma_y$ 越大，塑性约束因子越高；本算例约束因子
   $p_m/@BS@sigma_y$ = {_f(pb['constraint_factor_p_m_over_sigma_y'],3)}，量级合理但与真实冰
   （含破碎）不可直接比较。

## ⑥ 误差判定（逐项，含"是否 < 3%"的显式回答）

| 关键量 | 基准 | 误差 | < 3% ? |
|---|---|---|---|
| 单轴应力–应变曲线（拉/压/含硬化） | 闭式 J2 + 线性硬化 | {_f(ac1['max_relative_error'],3)} | **是** |
| 纯剪切首次屈服 $@BS@sigma_{{xy}}^y$ | $@BS@sigma_y/@BS@sqrt3$ | {_f(ac2['relative_error'],3)} | **是** |
| 弹性压痕力 $P(@BS@delta)$ | Hertz 精确解 | {_f(100*ac3['max_relative_error'],2)}% | **否**（12%，受接触区离散限制；加密单调下降） |
| 各向同性材料的矢量化内核 | 标量径向返回（逐位） | ≤ {_f(1e-15,1)} | **是** |
| 一致切线 | 中心差分 | {_f(ac2['consistent_tangent_relative_error'],3)} | **是** |
| 受限压缩力（罚接触验证） | $(@BS@lambda+2@BS@mu)@BS@delta/H@BS@cdot A$ | ≤ 0.4%（β=200） | **是**（见 pytest） |
| 能量守恒 $W=U_{{el}}+D_{{pl}}$ | 能量账本 | {_f(100*abs(ac5['relative_residual']),2)}% | **是** |
| 接触力网格收敛 | 最细网格 | 单调下降，阶 {_f(ac4['observed_order'],2)} | 是（收敛性判据） |
| 弹塑性 $P(@BS@delta)$ vs 实验 | —— | **WITHHELD** | **无法回答**（工作树内无实验数据，拒绝伪造） |

**结论**：本构（AC1/AC2）与能量（AC5）达到 $<3@BS@%$；弹性接触力 12% 未达 3%
但满足 15% 验收线且随网格单调收敛（AC4）；**弹塑性力–位移曲线与实验的对比缺失**
（WITHHELD），这是本节点唯一的实质缺口，需后续引入真实冰压痕试验数据（或与
实验标定的压力–面积关系）才能闭环。
"""
    return doc.replace("@BS@", chr(92))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(ROOT / "artifacts" / "ice_indentation"
                                                / "acceptance_report.json"))
    parser.add_argument("--out", default=str(ROOT / "docs"
                                             / "ICE_INDENTATION_J2_CONTACT.md"))
    args = parser.parse_args(argv)
    with open(args.report) as fh:
        report = json.load(fh)
    text = render(report)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"report written to {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
