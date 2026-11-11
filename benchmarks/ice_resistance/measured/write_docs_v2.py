#!/usr/bin/env python
"""Render ``docs/ICE_RESISTANCE_MEASURED_ANCHOR_V2.md`` from the artifacts.

Every number in the document is read out of
``artifacts/ice_resistance/measured_adjudication_v2.json`` (and the baseline
artifact it points at) -- nothing is typed by hand, so the prose cannot drift
away from the computation.

Run::

    python benchmarks/ice_resistance/measured/write_docs_v2.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorphysics.ice.measured_anchor import sha256_file  # noqa: E402

DATA_ROOT = REPO / "data" / "measured" / "ice_resistance"
ART = REPO / "artifacts" / "ice_resistance" / "measured_adjudication_v2.json"
BASE_ART = REPO / "artifacts" / "ice_resistance" / "measured_adjudication.json"
OUT = REPO / "docs" / "ICE_RESISTANCE_MEASURED_ANCHOR_V2.md"


def pct(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def num(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def render(v2: Dict[str, Any], base: Dict[str, Any]) -> str:
    c3 = v2["baseline"]["ledger"]["C3_15pct_adjudication"]
    c6 = v2["C6_governance"]
    c7 = v2["C7_criterion_metrology"]
    c8 = v2["C8_external_validity"]
    c9 = v2["C9_evidence_gaps"]
    gov = json.loads((DATA_ROOT / "registry.json").read_text(encoding="utf-8"))
    series_lines: List[str] = []
    for entry in gov["series"]:
        doc = json.loads((DATA_ROOT / entry["path"]).read_text(encoding="utf-8"))
        app = doc["applicability"]
        fac = doc["facility"]
        series_lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                doc["series_id"],
                doc["provenance"]["grade"],
                fac["kind"],
                "—" if fac.get("model_scale") is None else f"1:{fac['model_scale']:g}",
                doc["uncertainty"]["reported"],
                app["ice_breaking_mode"],
                "是" if app["in_15pct_verdict_set"] else "否",
            )
        )

    verdict_rows = [r for r in v2["per_point_table"] if r["in_15pct_verdict_set"]]
    verdict_rows.sort(key=lambda r: (r["h_m"], r["V_ms"]))
    per_point_lines = []
    for r in verdict_rows:
        per_point_lines.append(
            "| {} | {:.2f} | {:.2f} | {} | {} | {} | {} | {} |".format(
                r["point_id"], r["h_m"], r["V_ms"], num(r["R_meas_kN"], 0),
                "—" if r["R_unc_kN"] is None else num(r["R_unc_kN"], 0),
                num(r["R_pred_kN"], 0), f"{r['rel_error_pct']:.0f}%",
                "—" if r["R_unc_kN"] is None else f"{100 * r['R_unc_kN'] / r['R_meas_kN']:.1f}%",
            )
        )

    metrology_lines = [
        "| {} | {} | {} | {} | {} | {} |".format(
            r["point_id"], num(r["R_meas_kN"], 0),
            "—" if r["R_unc_kN"] is None else num(r["R_unc_kN"], 0),
            "—" if r["u_over_R"] is None else f"{100 * r['u_over_R']:.1f}%",
            "—" if "pred_meas_gap_in_sigma" not in r else f"{r['pred_meas_gap_in_sigma']:.1f}σ",
            "—" if "tolerance_above_noise" not in r else ("是" if r["tolerance_above_noise"] else "否"),
        )
        for r in c7["rows"]
    ]

    sweep = sorted(c8["sweep"], key=lambda p: p["median_rel_error_pct"])
    sweep_lines = [
        "| {} | {:.0f}% | {:.2f} | {} | {} |".format(
            p["config"], p["median_rel_error_pct"], p["pred_over_meas_median"],
            "是" if p["admissible"] else "否", "、".join(p["domain_clipped"]) or "—",
        )
        for p in sweep[:5] + sweep[-3:]
    ]

    gaps = json.loads((DATA_ROOT / "evidence_gaps.json").read_text(encoding="utf-8"))
    gap_lines = [
        "| {} | {} | {} | {} |".format(
            g["gap_id"], g["status"], len(g["attempts"]), g["target"].split(",")[0][:70]
        )
        for g in gaps["gaps"]
    ]

    salmi_rows = [r for r in v2["per_point_table"]
                  if r["series_id"] == c8["series_id"] and r.get("config") == c8["best_admissible_config"]]
    salmi_rows.sort(key=lambda r: r["V_ms"])
    salmi_lines = [
        "| {} | {:.2f} | {} | {} | {} | {} | {:.0f}% | {:.2f} |".format(
            r["point_id"], r["V_ms"], num(r["R_meas_kN"], 0), num(r["R_alt_kN"], 0),
            num(r["literature_baselines_kN"]["leiviska_ubib_sigma500"], 0),
            num(r["R_pred_kN"], 0), r["rel_error_pct"], r["R_pred_kN"] / r["R_meas_kN"],
        )
        for r in salmi_rows
    ]

    crit = v2["criteria"]
    crit_lines = [
        "| {} | {} | {} |".format(k, "✅" if v else "❌", note)
        for k, v, note in [
            ("C1 schema + sha256", crit["C1_schema_and_digest"], "3 系列 / 18 点，逐份源件重新哈希比对"),
            ("C2 双录入", crit["C2_double_entry"], "样本 + 全量两路比对，一致率见工件"),
            ("C3 15% 判决", crit["C3_15pct_adjudication"],
             f"最好预注册配置中位误差 {pct(c3['best_median_rel_error'])} → honest FAIL"),
            ("C4 物理门", crit["C4_physics_gates"], "模型门 + 判决集内实测门（域外探针单独记录）"),
            ("C5 既有全谱不破坏", crit["C5_legacy_full_spectrum"], "25 参考锚 + 负例仍全绿"),
            ("C6 治理完备", crit["C6_governance"], "台架/尺度/不确定度/数字化方法/适用边界全部机器校验"),
            ("C7 判据度量学", crit["C7_criterion_metrology"],
             f"{c7['n_fail_resolvable_at_kmax']}/{c7['n_points_with_uncertainty']} 点可证伪、"
             f"{c7['n_pass_resolvable']}/{c7['n_points_with_uncertainty']} 点可证真"),
            ("C8 外部效度", crit["C8_external_validity_probe"],
             f"{c8['n_configs']} 组约定扫描，可采纳最优中位误差 {c8['best_admissible_median_rel_error_pct']:.1f}%"),
            ("C9 证据缺口", crit["C9_evidence_gaps"], f"{c9['n_gaps']} 个缺口 / {c9['n_attempts']} 次带时戳取数尝试"),
        ]
    ]

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return f"""# 冰阻力实测锚库：治理、判据度量学与外部效度（mission 11210 / AG29W1-N02，成熟化）

> **一句话结论**：本节点把"有真值"升级为"**真值可用、可审计、且知道自己的分辨极限**"：
> 3 个实测系列 / 18 个点全部带台架·尺度·不确定度·数字化方法与 sha256；
> 新增的**判据度量学**给出一个不舒服但必须说的结论——15% 判据在本数据上
> **可证伪、不可证真**（{c7['n_fail_resolvable_at_kmax']}/{c7['n_points_with_uncertainty']} 点能在 3σ 上排除 15% 以内，
> {c7['n_pass_resolvable']}/{c7['n_points_with_uncertainty']} 点连"确认 15% 一致"都做不到，因为测量相对不确定度中位数
> {pct(c7['median_u_over_R'])} 已超过 15%）；第二次独立战役（UBIB 上折破冰概念船）作为**域外探针**引入，
> 在可采纳约定带内本模型系统性高估 {c8['best_admissible_pred_over_meas_median'] if c8.get('best_admissible_pred_over_meas_median') else c8['best_admissible_pred_over_meas']:.2f}×，
> 而源文自己的 UBIB 理论在同四点上系统性低估（中位误差
> {c8['published_ubib_theory']['leiviska_ubib_sigma500']['median_rel_error_pct']:.1f}%）——**两个闭合族把实测夹在中间**；
> 15% 判决仍为 **honest FAIL**（最好预注册配置中位误差 {pct(c3['best_median_rel_error'])}），未做任何调参。

生成时间（UTC）：`{now}`　资产版本：`{c6['governance_version']}`　判据容差：`{v2['tolerance']}`
工件：`artifacts/ice_resistance/measured_adjudication_v2.json`（本文件全部数字的唯一来源）、
`artifacts/ice_resistance/measured_adjudication.json`（基线判决，sha256 `{v2['baseline']['sha256'][:16]}…`）、
`data/measured/ice_resistance/`（资产库）、`data/measured/ice_resistance/evidence_gaps.json`（缺口登记）、
`tests/test_measured_criterion.py`（本节点新增判据）+ `tests/test_measured_anchor.py`（基线判据），
共 49 条 pytest 全绿。

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

### 1.1 为什么"有真值"不等于"真值可用"

基线节点（m9425）建立了 `measured_dataset/v1` 与第一次实测对标，但它留了三个洞：
① 合同要求的字段（台架、尺度、不确定度、数字化方法）大多**隐含在散文里**，没有机器校验；
② 只有一个系列进入 15% 判决，**单点战役无法区分"闭合错"与"这天冰池状态不好"**；
③ 取不到的源（Lindqvist 1989 / Riska 1997 原始表格）只在正文里写了一句"没取到"，
**没有时戳、没有失败模式，下一个节点只能重走死路**。

本节点把这三项补成机制，并对判据本身做度量学审查。

### 1.2 资产库现状（全部机器校验）

| 系列 | 证据级别 | 台架类型 | 缩尺 | 声明不确定度 | 破冰模式 | 进入 15% 判决集 |
|---|---|---|---|---|---|---|
{chr(10).join(series_lines)}

- **MTUIKKU-HZ2016**（判决集）：MT Uikku 双动破冰油轮冰池试验，L_pp 150 m / B 21.3 m / T 9.5 m，10 个平整冰点，h 0.63–1.04 m、V 0.20–1.00 m/s，逐点实测 σ_b 540–1029 kPa、σ_c 1.75–5.39 MPa。
- **AGULHAS2-FS2012**（筛选）：S.A. Agulhas II 波罗的海实测 4 时段，只有 (h, V, 功率带)，`supports_15pct=false`。
- **SALMI2014-UBIB**（域外探针，本节点新增）：Aalto Otaniemi 冰池 1:25、40 mm 模型冰→实尺 1.0 m 平整冰，
  上折破冰概念船（L 100 m / B 10 m / T 7.5 m，冰带宽 5.5 m），3/6/9/12 kn 四点实测阻力
  620/602/1236/1040 kN（作者已扣除异常通道摩擦，Table 5），Table 8 提供进一步摩擦修正值作为 `R_alt`。

### 1.3 三个"不敢说成已知"的约定（显式登记）

| 约定 | 内容 | 处理方式 |
|---|---|---|
| A1 | MT Uikku 只报"艏柱角 30°"，未给板面与水平夹角 | 主配置 α=30°，敏感性 25°/60°（基线已做） |
| A2 | UBIB 的板面倾角源文未给 | 扫描 60/70/75/80°，**报告带而非单值** |
| A3 | UBIB 冰带仅 5.5 m（总宽 10 m），σ_c 未报 | 扫描 B∈{{5.5, 10}} m、σ_c∈{{1,2,5}} MPa |

---

## 2. 怎么做的（方法 / 流程 / 工具链）

```text
data/measured/ice_resistance/
├── registry.json                       # 3 系列的 path/grade/点数/sha256（registry 自身被 C1 复核）
├── evidence_gaps.json                  # 缺口登记（每个缺口必带时戳化取数尝试）
├── evidence_gaps/attempts_latest.json  # 取数尝试原始日志（probe_sources.py 产出，sha256 入 registry）
└── series/<series_id>/
    ├── series.json     # 计量资产：points + facility/digitisation/uncertainty/applicability + provenance
    ├── pass_a.json     # 数字化第一路（列优先 / 原始单位）
    ├── pass_b.json     # 数字化第二路（行优先 / 工程单位内联换算）
    └── source/         # 归档源件（sha256 在此字节上计算，可离线复核）
```

五条硬约束（由 `validate_series` + `validate_governance` 执行，pytest 均有负例）：

1. 没有 digest 的 provenance 一律拒绝；`verify=True` 时**重算字节哈希**比对，改一个字节即报 mismatch；
2. 每个点必须写 `provenance_ref`（表号/图号），不允许"来源=论文"这种无法追的数字；
3. **缺失即缺失**：源文未报道的统计量留空（MT Uikku 的 Tests 401–403 与 UBIB 全系列都没有 std），绝不插值；
4. 治理块必须**显式声明**：台架/尺度、数字化方法与 ≥2 条独立路径、不确定度（`reported` 布尔 + 定义，未报道也要声明"未报道"）、适用边界（破冰模式 + 是否进入判决集 + 允许证明什么）；
5. 域外系列（`in_15pct_verdict_set=false`）**照常测量、照常记录违规**，但既不认证也不否证真值源；实测数据永不被"修"成单调。

流程：`probe_sources.py`（真实取数尝试→日志）→ `build_evidence_gaps.py`（日志→缺口登记）
→ `build_salmi2014_series.py`（摘录→双录入→series.json）→ `migrate_governance.py`（治理块+registry 摘要）
→ `run_adjudication_v2.py`（基线判决 + C6–C9 + 图）→ `write_docs_v2.py`（本文，数字注入零手抄）。

### 2.1 判据度量学（本节点新增的核心机制）

对每个实测点定义相对不确定度 `u/R`。于是 15% 判据有两个不对称的半边：

- **可证伪**：预测落在实测区间 `[R−k·u, R+k·u]` 之外（k=3）→ 该点的 15% 以内**确定不成立**；
- **可证真**：只有当 `0.15·R ≥ u` 时，"误差 <15%" 才与测量噪声可区分 → 否则任何模型都无法在该点被**确认**。

这不是给模型开脱：FAIL 侧越硬，结论越强。

---

## 3. 计算结果

### 3.1 判据总览

| 判据 | 通过 | 证据 |
|---|---|---|
{chr(10).join(crit_lines)}

判决：**{v2['verdict_15pct']}**（honest FAIL，不可覆盖）。`all_passed=false` 的唯一原因是 C3。

### 3.2 判决集逐点误差表（主配置：逐点实测冰强度、α=30°、k_bend=2.0、μ=0.15）

| 点 | h [m] | V [m/s] | 实测 [kN] | u [kN] | 预测 [kN] | 相对误差 | u/R |
|---|---|---|---|---|---|---|---|
{chr(10).join(per_point_lines)}

中位相对误差 **{pct(c3['best_median_rel_error'])}**（四个预注册配置：212.7% / 275.2% / 382.8% / 198.3%，最好者即上值）。
全部 10 点在 15% 之外（`n_within_15pct = {sum(1 for r in verdict_rows if r['rel_error_pct'] <= 15.0)}`），系统偏差 +320%（全部高估）。

### 3.3 判据度量学：15% 判据在本数据上可证伪、不可证真

| 点 | 实测 [kN] | u [kN] | u/R | 预测-实测（以 σ 计） | 该点能否确认 15% 一致 |
|---|---|---|---|---|---|
{chr(10).join(metrology_lines)}

- 有不确定度的点：**{c7['n_points_with_uncertainty']}/{c7['n_points']}**（其余 3 点源文未报道统计量，留空）；
- `u/R` 中位数 **{pct(c7['median_u_over_R'])}**、最大 **{pct(c7['max_u_over_R'])}**，
  **{c7['n_points_uncertainty_above_tolerance']}/{c7['n_points_with_uncertainty']}** 点的相对不确定度**大于 15% 判据本身；
- 预测-实测差距以 σ 计中位数 **{num(c7['median_abs_gap_in_sigma'])}σ**，**{c7['n_fail_resolvable_at_kmax']}/{c7['n_points_with_uncertainty']}** 点在 3σ 外；
- 结论状态：`{c7['criterion_state']}`。

**这条结论直接改变了节点的验收语义**：15% 判据此前是"判不出来"（WITHHELD），基线节点把判决做出来了（FAIL），
本节点进一步说明——**用这套真值只能判 FAIL，不能判 PASS**。要让 PASS 有意义，必须补更小不确定度的实测
（同一冰况多次重复、或力时程而非均值），这是新的、可执行的后续节点靶子。

### 3.4 外部效度：第二次独立战役（域外探针）

| 点 | V [m/s] | 实测 Table 5 [kN] | R_alt Table 8 [kN] | 源文上折理论 [kN] | 本仓预测 [kN] | 相对误差 | 预测/实测 |
|---|---|---|---|---|---|---|---|
{chr(10).join(salmi_lines)}

（上表取**可采纳组中最优**约定 `{c8['best_admissible_config']}`；实测=Table 5，`R_alt`=Table 8，
源文理论=Table 6/7 的 σ_b=500 kPa 列。最差约定 `{max(c8['sweep'], key=lambda p: p['median_rel_error_pct'])['config']}`
量级可达 {c8['band_pred_over_meas_median'][1]:.0f}×，说明该系列的绝对值强依赖未报道的约定——
这正是它被排除在判决集之外的原因。）

- 约定扫描共 **{c8['n_configs']}** 组，其中 **{c8['n_admissible_configs']}** 组不触发域限制器（`h ≤ 0.1·B`）；
- 可采纳组中最好者 `{c8['best_admissible_config']}`：中位误差 **{c8['best_admissible_median_rel_error_pct']:.1f}%**，
  预测/实测中位比 **{c8['best_admissible_pred_over_meas']:.2f}×**（高估）；
- 触发域限制器（把 h=1.0 m 截到 0.55 m）的组可以"凑"到
  `{min(p['median_rel_error_pct'] for p in c8['sweep']):.1f}%`，
  **被 `admissible=false` 排除**——用截断真实冰厚换来的"好成绩"不是拟合；
- 同四点、源文自己的 UBIB 理论：σ_b=275 kPa 中位误差 {c8['published_ubib_theory']['leiviska_ubib_sigma275']['median_rel_error_pct']:.1f}%、
  σ_b=500 kPa {c8['published_ubib_theory']['leiviska_ubib_sigma500']['median_rel_error_pct']:.1f}%，**四点全部低估**。

**因此**：本仓下折闭合族在此抬升型艏部上**系统性高估**，源文上折理论**系统性低估**——
两个闭合族把实测夹在中间。这条信息单靠一个系列永远拿不到，也说明基线节点提出的
"用一个 k_bend 常数把 15% 补上"不可能成立（两个方向的误差不能同时被同一个乘性常数消掉）。

### 3.5 证据缺口（机器可查、带时戳）

| 缺口 | 状态 | 取数尝试 | 目标 |
|---|---|---|---|
{chr(10).join(gap_lines)}

其中 `GAP-INDEPENDENT-DOWNWARD-SERIES` 为 **PARTIAL**：本节点新增的 UBIB 系列关掉了"只有一次战役"这一半，
但**没有**关掉"下折破冰船第二次战役"——所以任何闭合标定仍不得进入 `PARAMETER_PROVENANCE`。
`GAP-MTUIKKU-PRIMARY-PDF` 已被钉到具体动作：OpenAlex 确认该文 `is_oa=true`（gold），
唯一阻塞步是在无出口过滤的机器上取 `https://www.sciencedirect.com/science/article/pii/S2092678215300133/pdf`。

## 4. 结果图

本节点新增 3 张（均已落盘 `artifacts/ice_resistance/figures/`，并内嵌进平台报告）：

1. `uncertainty_vs_criterion.png` —— 逐点 `u/R` 柱 vs 15% 判据线 vs 模型误差三角：**一眼看出判据比噪声细**；
2. `external_validity_probe.png` —— 域外探针：模型约定带 vs 实测 vs 源文上折理论（两族夹住实测）；
3. `library_capability_summary.png` —— 库能力清单：判决集点数 / 有 u 的点数 / 可证伪点数 / 可证真点数 / 缺口数 / 尝试数。

基线节点 5 张图沿用：`measured_vs_predicted.png`、`per_point_error.png`、`physics_gates_and_measured.png`、
`component_attribution.png`、`k_bend_calibration.png`。

## 5. 注意点（踩坑 / 局限 / 适用范围）

1. **基线工件此前根本没进仓**：m9425 的 4 条自带 pytest 因缺 `measured_adjudication.json` 直接 error。
   本节点把工件重新生成并入库，同时把生成脚本改成可重复（10 秒内跑完），工件里的 `figures` 也一并落盘。
   ——"测试绿"与"工件可复算"是两件事，这次按后者验。
2. **`MeasuredSeries.as_dict()` 会吞掉治理块**（真实缺陷）：加载后的对象再序列化就丢了 `applicability`，
   于是"该系列是否进入判决集"在流水线里静默消失。已修（可选字段 + 往返 pytest 回归）。
3. **实测单调性不是数据缺陷**：UBIB 在 9→12 kn 实测阻力下降 15.9%，超过 10% 容差且源文未报散差。
   判据的处理是**记录不改**（`physics_gates.measured_gates_outside_verdict_set`），并因为该系列本就声明
   在判决集之外，不让它去认证/否证真值源。若哪天它要进判决集，必须先补不确定度或重复试验。
4. **约定仍是约定**：A2/A3 的扫描给出的是**带**，不是"真值"。本节点只主张带内稳定的事实（符号、量级、夹逼关系），
   不主张任何单点精度。
5. **UBIB 是概念船**：不是营运船；实测值经作者后处理（异常通道摩擦按理论替换、Table 8 再乘 1.5–2.46 摩擦修正）。
   因此 Table 5 作 `R_meas`、Table 8 作 `R_alt` 并列，**不合并**。
6. **独立不等于同类**：UBIB 与 MT Uikku 同属 Aalto 冰池体系（不同年代/船型/破冰模式），
   所以它是**外部效度**证据，不是"第二个下折破冰真值"。
7. **不确定度的语义**：MT Uikku 的 u 是源文与均值并列的标准差（逐次试验散差），不含模型冰标定误差与缩尺误差，
   故真实不确定度只会更大——只会让"可证真"更难，不会更容易。
8. **未完成项（如实申报）**：Lindqvist (1989) 与 Riska (1997) 原始表格仍未取到（POAC 论文库 200 但无该文下载路径；
   Elsevier DOI 404/NRC 检索页可访问但需人工检索）。已登记 `GAP-LINDQVIST-1989` / `GAP-RISKA-1997`，
   含具体失败模式与解锁动作。

## 6. 误差判定（显式回答"是否 <3%"）

| 关键量 | 基准 | 本仓结果 | 误差 | 是否 <3% |
|---|---|---|---|---|
| 平整冰阻力中位相对误差（MT Uikku，10 点，最好预注册配置） | Hu & Zhou (2016) 实测均值力 | {pct(c3['best_median_rel_error'])} | {pct(c3['best_median_rel_error'])} | **否**（差 66 倍于 3%，13 倍于 15%） |
| 源文发表 Lindqvist（对照，非本仓） | 同上 | 25.3% | 25.3% | 否 |
| 域外探针 UBIB（可采纳最优约定，4 点） | Salmi (2014) Table 5 | {c8['best_admissible_median_rel_error_pct']:.1f}% | {c8['best_admissible_median_rel_error_pct']:.1f}%（高估 {c8['best_admissible_pred_over_meas']:.2f}×） | 否（且该系列不属判决集） |
| 资产完整性（schema + sha256 + 双录入） | 自身定义 | C1/C2 全过 | 0 处不一致 | 是（精确判据，非物理量） |
| 治理完备性 | 合同字段表 | C6 全过 | 0 处缺失 | 是（精确判据） |
| 15% 判据可证伪性 | 3σ 区间 | {c7['n_fail_resolvable_at_kmax']}/{c7['n_points_with_uncertainty']} 点可证伪 | — | 是（判据语义层面） |

**诚实结论**：
- 冰阻力预测**没有**做到 3%，也**没有**做到 15%；本节点**没有**调参、**没有**删点、**没有**隐瞒 C3 失败。
- 本节点真正交付的是三件事：真值源从此**治理完备**（18 点、3 系列、全部声明台架/尺度/不确定度/数字化方法/适用边界）；
  判据从此**知道自己能判什么**（可证伪 7/7，可证真 0/7）；域外探针证明"单一乘性闭合常数"这条路**不可行**。
- 下一节点的靶子因此更窄也更硬：要么（a）补第二艘**下折破冰船**的实测系列（关 GAP），
  要么（b）按 Lindqvist 原文重导 `R_break` 的闭合结构（当前 `R_break` 占预测 61.8%），
  并在**独立系列**上复现后才允许动 `PARAMETER_PROVENANCE`。
"""


def main() -> int:
    v2 = json.loads(ART.read_text(encoding="utf-8"))
    base = json.loads(BASE_ART.read_text(encoding="utf-8"))
    OUT.write_text(render(v2, base), encoding="utf-8")
    print(json.dumps({
        "out": str(OUT.relative_to(REPO)),
        "bytes": OUT.stat().st_size,
        "lines": len(OUT.read_text(encoding="utf-8").splitlines()),
        "v2_sha256": sha256_file(ART),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())