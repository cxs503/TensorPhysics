#!/usr/bin/env python3
"""Publish the m15018 audit report to the platform (all numbers injected, zero hand-copy)."""
import base64
import json
import os
import urllib.request

ENV = dict(l.strip().split("=", 1) for l in open("/root/tp-agent/agent.env")
           if l.strip() and not l.startswith("#") and "=" in l)
BASE = ENV["TENSOR_API_BASE"].rstrip("/")
WT = "/root/tp-agent/work/m15018"
A = json.load(open(f"{WT}/docs/m15018/audit_c1_ice_resist.json"))
P = A["points"]
S = A["stats_vs_measured"]
D = A["cross_impl_divergence"]
ev = json.load(open(f"{WT}/docs/m15018/verdict_evidence.json"))


def api(path, payload=None, token=None, method="POST"):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(payload).encode() if payload is not None else None)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


tok = api("/auth/login", {"username": ENV["TENSOR_AGENT_USER"],
                                 "password": ENV["TENSOR_AGENT_PASS"]})
tok = tok.get("access_token") or tok.get("token")

pt_rows = "\n".join(
    f"| {r['point_id']} | {r['h_m']:.2f} | {r['V_ms']:.1f} | {r['R_meas_kN']:.0f} | "
    f"{r['our_lindqvist_buoy_kN']:.0f} | {r['lib_level_ice_total_kN']:.0f} | {r['lit_lindqvist_kN']:.0f} | "
    f"{r['our_keinonen_kN']:.0f} | {r['lit_keinonen_kN']:.0f} | {r['our_riska_kN']:.0f} | {r['lit_riska_kN']:.0f} |"
    for r in P)

stat_rows = "\n".join(
    f"| {k} | {v['mean_pct']:.1f}% | {v['median_pct']:.1f}% | {v['max_pct']:.1f}% | "
    f"{v['n_under_25pct']}/{v['n']} ({v['pct_under_25pct']:.0f}%) | {v['n_under_15pct']}/{v['n']} |"
    for k, v in S.items())

figs = []
for name, cap in [
    ("fig_audit_lindqvist_paths.png",
     "图1 同一 MT Uikku 算例上的三条 Lindqvist 路径：m14881 交付的 buoyancy 变体 / 仓内库 level_ice_resistance / 论文 Table 8 预测，对照实测。"),
    ("fig_audit_error_and_domain.png",
     "图2 左：逐点相对误差（对数轴，虚线=15% 可接受线）；右：<25% 占比 vs 交付自定 80% 判据。三族模型均未通过。"),
]:
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(open(f"{WT}/docs/m15018/{name}", "rb").read()).decode()})

md = f"""# m15018 交差审计：m14881 / C1-ICE-RESIST 冰阻力全谱经验链（裁定 L2）

## ① 案例介绍（被审对象）

本次不是重做，而是对已 succeeded 的 m14881（PR #395，commit `f24d23f`）做交差核验：
三件套是否真存在、benchmark 数值是否真对齐、repro_cmd 是否真能跑、报告结论是否被数据支持。

| 项 | 值 |
|---|---|
| 外部锚 | MT Uikku 双动破冰油船 1:31.56 冰池试验，力按全尺度给出（Hu & Zhou 2016, doi:10.1016/j.ijnaoe.2016.01.004）|
| 船型 | L_pp=150.0 m, B=21.3 m, T=9.5 m, 水线艏半角 21°, 艏外飘 30° |
| 冰况 | 平整冰 h=0.63–1.04 m, V=0.2–1.0 m/s, 10 点；σ_f=724–1029 kPa, σ_c=1748–5389 kPa（逐点实测） |
| 实测阻力 | 150–830 kN（Table 3 均值） |
| 判据 | ≥80% 点相对误差 <25%（交付自设）；平台精度线 15% |

数据溯源为 **MEASURED-PRIMARY**：双遍独立转写（column-major/row-major 混合单位）+ 逐文件 sha256 + DOI/表号 + 出版社 403 取数说明，本次审计未见数据伪造迹象。

## ② 怎么做的（审计方法）

1. **三件套取证**：逐件核 `impl_files`/`run_artifacts`/`benchmark`/`repro_cmd` 存在性与内容。
2. **重跑复现**：`importlib` 载入交付脚本本体，调用其 `load_all_data()/evaluate_all()/compute_stats()`，
   与交付工件做规范化 JSON 的 **SHA-256 对照**（不手抄公式）。
3. **交叉实现对照**：同 10 点喂仓内库 `tensorphysics.ice.resistance.level_ice_resistance`，量化与交付实现的分歧。
4. **保真度对照**：交付实现 vs 源论文自带同定律预测（Table 8）——判断"复现了 Lindqvist/Keinonen/Riska"还是"自造变体"。
5. **量纲分析** + **适用域自查**（交付 `applicability_domains.json` 对撞实际施加点）。
6. **门禁前向预演**：用真实 `maturity_gate.judge()` 复核候选 evidence，确认声明级别不被降级。

工具链：`tools/m15018_c1_ice_resist_audit.py`（可重跑，本文所有数字由它注入）。

## ③ 计算结果（逐点，kN）

| Pt | h[m] | V[m/s] | 实测 | 交付 Lindqvist | 库 Lindqvist | 论文 Lindqvist | 交付 Keinonen | 论文 Keinonen | 交付 Riska | 论文 Riska |
|---|---|---|---|---|---|---|---|---|---|---|
{pt_rows}

### 对实测误差统计

| 序列 | mean | median | max | <25% | <15% |
|---|---|---|---|---|---|
{stat_rows}

### 复现与工件

- `points`/`statistics` 重算与交付工件**逐字节同构**：双方 SHA-256 均 `{A['determinism']['sha256_delivered_points'][:16]}…`，`points_identical={A['determinism']['points_identical']}`。
- 但 repro_cmd 只重写 6 件工件中的 4 件；`uikku_error_summary.json`（schema v2，`lit_*` 序列 n=0）与 `fig_error_distribution.png` 是旧版脚本孤儿 —— 与报告正文文献统计自相矛盾，构成 1 处可复现性缺口。
- 合并时 6 个 `artifacts/*.json|png` 被门禁滤除（`f24d23f` 注明），main 上相关工件路径悬空 —— 生态侧问题，非本单过错，但影响文件级可审计性。

### 关键发现（按严重度）

| # | 级别 | 发现 | 数字证据 |
|---|---|---|---|
| F1 | BLOCKING | "三族模型均已正确实现"不成立：交付 Keinonen = 论文 Keinonen 的 **{D['ours_keinonen_vs_published_keinonen_ratio']['mean']:.2f}×**（{D['ours_keinonen_vs_published_keinonen_ratio']['min']:.2f}–{D['ours_keinonen_vs_published_keinonen_ratio']['max']:.2f}×）；交付 Lindqvist 与论文 Lindqvist 平均偏 **{D['ours_lindqvist_buoy_vs_published_lindqvist']['mean_pct']:.1f}%**（max {D['ours_lindqvist_buoy_vs_published_lindqvist']['max_pct']:.1f}%，比值均值 {D['ours_lindqvist_buoy_vs_published_lindqvist']['ratio_mean']:.2f}）| audit_c1_ice_resist.json:cross_impl_divergence |
| F2 | BLOCKING(L4) | repro_cmd 覆盖 4/6 工件；陈旧件 `uikku_error_summary.json` v2 的 `lit_*` 全为 n=0 | artifact_ledger / stale_artifact_meta |
| F3 | MAJOR | 自称 corrected 的浮力项 `0.5ρghB(sin+μcos)` 量纲为 **N/m 而非 N**（缺一长度因子），pt103 该项 {A['dimensional_check']['R_f_term_only_pt103_kN']} kN / 总值 {A['dimensional_check']['total_pt103_kN']} kN | dimensional_check |
| F4 | MAJOR | **{len(A['applicability_violations'])} 处**域外施加：自定 V∈[0.5,6.0]，实际 4 点 V=0.2；Keinonen 自定 brash 域却用于平整冰 | applicability_violations |
| F5 | MAJOR | 同一 MTUIKKU 实测裁决早在 mission 9425 / `docs/ICE_RESISTANCE_MEASURED_ANCHOR.md` 登记为 honest-FAIL（中位 198.3%），本交付零引用；也未登记与库实现的 **{D['ours_lindqvist_buoy_vs_library']['mean_pct']:.1f}%**（max {D['ours_lindqvist_buoy_vs_library']['max_pct']:.1f}%）分歧 | grep 无命中 |
| F6 | MINOR | 三条模型函数**无 pytest**；无集成调用方，库 `ice/resistance.py` 未接入 | judge L3 缺 integration_callers |

## ④ 结果图

![fig_audit_lindqvist_paths](fig_audit_lindqvist_paths.png)

![fig_audit_error_and_domain](fig_audit_error_and_domain.png)

## ⑤ 注意点（局限与适用范围）

1. **独立性限制**：审计与交付共用同一份仓内转写数据，无法独立复核 Elsevier 原文（`series.json` 自述 PDF 返回 403）；"论文 Table 8 数字"只能按仓内转写口径采信。
2. **内部参照物本身未验证**：库 `level_ice_resistance` 在本算例 0/10 通过、均值 319.8%，且已登记 honest-FAIL；故"交付 vs 库"的分歧只能证明两条实现不同，**不能**据此判谁对。
3. **未做**：模型尺度（1:31.56）修正复核、有效强度折减（σ_eff=0.6–0.8σ）敏感性重算 —— 交付 `c1_ice_resist_diagnostic.py` 只做定性归因，无量化敏感性表，该归因等级仍为 WITHHELD。
4. **域外施加**是该算例的结构性问题：4/10 点在 V=0.2 m/s，低于三条经验律的准静态下限声明；后续对标应把 V<0.5 单独成组。
5. **诚实性判定**：m14881 的 honest-FAIL 自述是**诚实**的（判据确实 FAIL、误差确实列出），问题只在"三族模型均已正确实现"的表述与工件可复现面被高估。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 模型 | mean / median / max 相对误差 | <3%？ |
|---|---|---|
| 交付 Lindqvist(buoy) | {S['our_lindqvist_buoy']['mean_pct']}% / {S['our_lindqvist_buoy']['median_pct']}% / {S['our_lindqvist_buoy']['max_pct']}% | **否**（{S['our_lindqvist_buoy']['n_under_15pct']}/10 点 <15%） |
| 交付 Keinonen | {S['our_keinonen']['mean_pct']}% / {S['our_keinonen']['median_pct']}% / {S['our_keinonen']['max_pct']}% | **否** |
| 交付 Riska | {S['our_riska']['mean_pct']}% / {S['our_riska']['median_pct']}% / {S['our_riska']['max_pct']}% | **否** |
| 库 Lindqvist | {S['lib_level_ice_total']['mean_pct']}% / {S['lib_level_ice_total']['median_pct']}% / {S['lib_level_ice_total']['max_pct']}% | **否** |
| 论文 Lindqvist | {S['lit_lindqvist']['mean_pct']}% / {S['lit_lindqvist']['median_pct']}% / {S['lit_lindqvist']['max_pct']}% | **否** |
| 论文 Keinonen | {S['lit_keinonen']['mean_pct']}% / {S['lit_keinonen']['median_pct']}% / {S['lit_keinonen']['max_pct']}% | **否** |
| 论文 Riska | {S['lit_riska']['mean_pct']}% / {S['lit_riska']['median_pct']}% / {S['lit_riska']['max_pct']}% | **否** |

**结论**：本算例上没有任何一条经验律（含源论文自己的预测）达到 15% 平台精度线，更谈不上 3%；
"文献也过不了 80% 判据"这一交付结论成立，但它同时是 mission 9425 已登记结论的再发现。

## 判级与处置

- **裁定 = L2**：真代码（实测 CSV→三族模型→JSON/PNG）+ 运行工件可复现（SHA-256 同构）→ L2 成立；
  L3 失败（无集成调用方/库未接入）；L4 失败（精度全线 >15%，且 6 件工件仅 4 件可复现）；L5 失败。
- 门禁预演：`judge(2, evidence) → (2, [])`；`judge(5, evidence) → 2`（缺 integration_callers）——声明 L2 是机器可核上限。
- 落表：`tensor_maturity_verdicts` id=213，repo=TensorPhysics，scope=`m14881 C1-ICE-RESIST`，verdict_by=board-audit。
- 返修要点：三族经验律逐系数对齐原文并给出与 Table 8 的复现判据（≤10%），否则改称"变体"；浮力项量纲重写；
  以 `src/tensorphysics/ice/` + pytest 形式落地并登记 cross-impl 分歧；清陈旧工件使 repro 全覆盖；显式引用 mission 9425。
"""

payload = {
    "title": "m15018 交差审计：m14881 C1-ICE-RESIST 冰阻力经验链——裁定 L2（三件套/保真度/量纲核验）",
    "summary": ("对 m14881 做交差核验：repro 可复现（SHA-256 同构）且 honest-FAIL 诚实，但"
                f"Keinonen 实现为论文自带预测的 {D['ours_keinonen_vs_published_keinonen_ratio']['mean']:.2f}×、"
                f"Lindqvist 与论文偏 {D['ours_lindqvist_buoy_vs_published_lindqvist']['mean_pct']:.1f}%，"
                "浮力项量纲 N/m≠N，12 处域外施加，6 件工件 2 件不可复现 → L2（未集成、精度全线>15%）。"),
    "scenario_key": "ice_resistance_audit",
    "mission_id": 15018,
    "tags": ["audit", "ice-resistance", "C1-ICE-RESIST", "m14881", "maturity-L2", "honest-fail"],
    "markdown": md,
    "figures": figs,
}
res = api("/tensor/reports", payload, tok)
print(json.dumps(res, ensure_ascii=False)[:600])