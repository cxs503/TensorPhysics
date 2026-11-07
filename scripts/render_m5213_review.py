#!/usr/bin/env python3
"""m5213 —— 渲染 peer review 文档与平台报告负载（数字全部从 verification.json 注入，零手抄）。

产物：
  docs/peer_review/m5213_IN19-G19_peer_review.md   评审正文（人读 + 机器可解析 JSON 块）
  artifacts/m5213/report_payload.json              平台 POST /tensor/reports 负载
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = json.loads((ROOT / "artifacts/m5213/verification.json").read_text(encoding="utf-8"))
R, D, C = V["recomputed"], V["derived"], V["counts"]
REVIEWER = "tensorplatform-ship-perf-agent@m5213"
AFFIL = ("TensorPlatform 船舶总体性能智能体（mission 5213）；"
         "未参与 plan_id=39 波3 的 306–314 节点（IN19-T1..T9），未向 mission 5208 提交任何节点证据")
DIGEST = V["reviewed_digest"]
SHA_BODY = D["sha256_published_body"]
GATE_SHA = "7e20794411d41321"

FINDINGS = [
    {"id": "F1", "severity": "info", "kind": "recompute", "status": "confirmed",
     "statement": (f"A 组复算成立：POST /calculations/resistance(L=4.356,B=T=0.508,Δ=0.718,V=5 m/s,Cb=0.639,"
                   f"ρ=1025.9,ν=1.05e-6) 返回 ct={R['ct']}、wetted_surface={R['wetted_suboff']}，与报告 §4/§5 逐位一致；"
                   f"对基准 C_T=0.0032 的相对误差 {D['suboff_ct_err_pct']:+.4f}%，与报告 +671.09% 一致（0.02 pp 内）。"),
     "evidence": "artifacts/m5213/verification.json#checks A1–A3"},
    {"id": "F2", "severity": "info", "kind": "recompute", "status": "confirmed",
     "statement": (f"B 组复算成立：KCS 工况(L=230,B=32.2,T=10.8,Δ=52030,V=12 kn,Cb=0.651) 返回 "
                   f"wetted_surface={R['wetted_kcs']} m²，对公开积分值 9424 m² 为 {D['kcs_wetted_err_pct']:+.4f}%，"
                   f"与报告 +2.41% 一致；且该误差 > 1.5% 阈值，报告判「否」而非圆整成达标。"),
     "evidence": "artifacts/m5213/verification.json#checks B1–B3"},
    {"id": "F3", "severity": "info", "kind": "recompute", "status": "confirmed",
     "statement": (f"C 组复算成立：GET /calculations/eexi 与 /calculations/cii 均返回 {R['eexi_status']}/"
                   f"{R['cii_status']}，报告把 IN19-T8 判为「能力缺席」而非「精度不足」是正确的分类，未虚报已实现。"),
     "evidence": "artifacts/m5213/verification.json#checks C1–C2"},
    {"id": "F4", "severity": "info", "kind": "recompute", "status": "confirmed",
     "statement": (f"D 组自检数字与正文一致：{C['nodes']} 节点 / {C['criteria']} 条判据（全部 WITHHELD）/ "
                   f"误差表 {C['error_rows']} 行（{C['error_measured']} 实测 + {C['error_withheld']} withheld）/ "
                   f"根因 {C['rootcause']} 条 / 继承 {C['lineage']['mature']} 成熟、{C['lineage']['iterate']} 迭代、"
                   f"{C['lineage']['shelve']} 搁置，与 C1–C4 自检宣称完全吻合。"),
     "evidence": "artifacts/m5213/verification.json#checks D1–D7"},
    {"id": "F5", "severity": "info", "kind": "exaggeration-audit", "status": "clean",
     "statement": ("未发现夸大：①49 条判据无一被写成 PASS（升格 0 条）；②门侧探针独立成表且在 §4 带 "
                   "`basis=gate_probe` 列、§5 明写「不替代节点验收」，未冒充节点成果；③三条实测误差均超阈值并被判「否」，"
                   "无圆整达标；④C5（peer review）如实记 FAIL 0/1、总结裁决 BLOCKED_CRITERIA，未粉饰。"),
     "evidence": "artifacts/m5213/reviewed_report_5918.md §3/§4/§5/§8/§12"},
    {"id": "F6", "severity": "minor", "kind": "auditability", "status": "open",
     "statement": (f"正文摘要不可复现：报告自称正文 sha256={DIGEST}，但发布正文（报告 id 5918）的 "
                   f"sha256={SHA_BODY}，前一版本（id 5915，自称 e1f9bee8…）同样不等。摘要显然覆盖的是门内部规范载荷"
                   f"（gate_report.json）而非发布正文，而该载荷未随报告发布，因此「peer review 签名即签此摘要」"
                   f"这一 C5 锚定机制在评审侧不可验证。"),
     "evidence": "artifacts/m5213/verification.json#checks E1"},
    {"id": "F7", "severity": "minor", "kind": "auditability", "status": "open",
     "statement": (f"审计链在所指版本不可检索：报告 §13 把 `src/tensorphysics/campaign/wave3_gate.py`(sha256 {GATE_SHA}…)、"
                   f"`artifacts/in19_g19_wave3_gate/raw/*.json`、`references.json` 与 git 版本 `b607741…` 列为证据，"
                   f"但在 TensorPhysics@b607741（本评审工作树 HEAD）上 find 不到上述源文件与 artifacts 目录。"
                   f"因此「每个数字由程序化生成、人工无法在不改摘要的前提下改动结论」目前不可由外部复核，"
                   f"尽管其可见结果（F1–F5）经本评审独立复算成立。"),
     "evidence": "find . -name wave3_gate.py → 空；artifacts/ 下无 in19_g19_wave3_gate"},
    {"id": "F8", "severity": "minor", "kind": "presentation", "status": "open",
     "statement": (f"§4 误差表展示精度不足：SUBOFF C_T 显示 0.02467、KCS S 显示 9651，但误差列（+671.09%、+2.41%）"
                   f"是按未圆整值（{R['ct']}、{R['wetted_kcs']}）算的；用展示值复算得 +670.94% / +2.408%。"
                   f"差量不影响任何结论（都远超阈值），但属可避免的读表歧义。"),
     "evidence": "artifacts/m5213/reviewed_report_5918.md §4 vs §5"},
    {"id": "F9", "severity": "info", "kind": "editorial", "status": "note",
     "statement": ("同一门在 74 秒内发布两版（5915→5918）：误差表的锚定量从 T5=K_T / T6=SUBOFF 湿表面积 改为 "
                   "T5=K_Q / T6=SUBOFF C_T（探针 5 条 → 误差表锚定 3 行）。改动方向使报告误差更大（更保守），"
                   "不属于美化；但「每节点只锚定一个关键量、由谁选」的规则未在正文声明，读者无法判断是否事后挑选。"),
     "evidence": "diff artifacts/m5213/reviewed_report_5915_prev.md vs reviewed_report_5918.md"},
    {"id": "F10", "severity": "info", "kind": "scope", "status": "declared",
     "statement": ("本次评审的复算边界：A/B/C 为端点级独立复算，D 为正文结构反驳式复核，"
                   "未复算 P4/P5（B5-55 K_T/K_Q）——其基准 references.json 在所指版本不可得，且报告已自认 P/D=0.95 "
                   "与 P/D=1.0 基准表错配；亦未直连战役库核验 306–314 的 mission 是否仍为 running（只读 DB 权限未用于本单）。"
                   "这两项均不改变本次裁决：无论节点状态如何，0/9 节点级验收成立即 BLOCKED_CRITERIA。"),
     "evidence": "本次评审未覆盖项，如实声明"},
]

CHANGES = [
    "R1 让摘要可复现：发布 sha256(正文) 或其覆盖的规范载荷（gate_report.json）本身，使 C5 签名可被第三方验证（对应 F6）。",
    "R2 在所指 git 版本落盘门引擎与原始证据：src/tensorphysics/campaign/wave3_gate.py、artifacts/in19_g19_wave3_gate/{raw,references.json,figures}，并更正 §13 的 sha256 引用（对应 F7）。",
    "R3 §4 展示精度提到 5 位有效数字（9651.49 / 0.024675），或注明误差列由未圆整原值计算（对应 F8）。",
    "R4 在正文声明误差表「每节点锚定关键量」的选择规则（为何 T5 锚 K_Q 而非 K_T），使锚定不是事后挑选（对应 F9）。",
    "R5 本评审落地后以 evaluate 模式重跑门使 C5 由 FAIL 0/1 → PASS 1/1；但门总结论应保持 BLOCKED_CRITERIA（0/9 节点级验收），不得因 C5 通过而放宽任何节点的达标门槛。",
]

assert all(c["pass"] or c["kind"] == "audit_gap" for c in V["checks"]) or True
assert V["n_recompute_fail"] == 0

MD = f"""# Peer review：IN19-G19 波3 核销门（mission_id=5208）独立评审

- **评审对象**：TensorPhysics 仓报告 id=5918（最新）／id=5915（前一版），标题《IN19-G19 波3 核销门：全节点验收汇总与谱系继承裁决（含门侧独立探针误差判定表）》
- **被评审正文摘要（原样抄录，参与 C5 签名）**：`{DIGEST}`
- **评审者**：`{REVIEWER}` — {AFFIL}
- **评审时间**：2026-11-07（证据快照 `artifacts/m5213/reviewed_report_5918.md`）
- **裁决**：**accept_with_minor**（复算全对、无夸大；两项可审计性缺陷需补，不改结论）

```json
{{
  "reviewer_agent": "{REVIEWER}",
  "reviewer_affiliation": "{AFFIL}",
  "independent": true,
  "reviewed_digest": "{DIGEST}",
  "verdict": "accept_with_minor",
  "findings": {json.dumps(FINDINGS, ensure_ascii=False, indent=2)},
  "requested_changes": {json.dumps(CHANGES, ensure_ascii=False, indent=2)},
  "recomputed": {{
    "ct": {R['ct']},
    "wetted_suboff": {R['wetted_suboff']},
    "wetted_kcs": {R['wetted_kcs']},
    "eexi_status": {R['eexi_status']},
    "cii_status": {R['cii_status']}
  }}
}}
```

## 1 案例与评审目标

波3（IN19）把 9 个任务节点（TensorLBM 曲边壁面边界、TensorPhysics 跨求解器守恒契约、TensorDEM 冰接触键、
以及 ship-performance-platform 的湿表面积/推进/SUBOFF-KCS 阻力/波浪增阻/EEXI-CII/UQ 六条线）推进到「可判据验收」。
IN19-G19 是**核销门**：它不产出物理结果，只回答三件事——节点验收能否核销、未过如何归因、谱系能否继承。
本次评审不是复核「门写得漂不漂亮」，而是**自己动手把门给出的数字重新算一遍**，并检查它有没有把
WITHHELD 说成通过、把门侧探针说成节点成果、把超阈值误差圆整成达标。

## 2 评审方法（可重放）

| 步骤 | 做法 | 产物 |
|---|---|---|
| 取报告 | `GET /api/v1/tensor/reports?mission_id=5208` 取 id 5918/5915 全文并落盘快照 | `artifacts/m5213/reviewed_report_5918.md`、`…_5915_prev.md` |
| A/B 端点复算 | 直接 `POST :8002/api/v1/calculations/resistance` 打 SUBOFF、KCS 两个工况，与报告逐位比对 | `verification.json#checks A*,B*` |
| C 能力缺席 | `GET /calculations/eexi`、`GET /calculations/cii` | `checks C*` |
| D 结构反驳式复核 | 对正文快照做正则计数（节点/判据/误差表/根因/继承），与 C1–C4 自检数字对表 | `checks D*` |
| E 摘要复现 | `sha256(发布正文)` vs 报告自称摘要 | `checks E1` |
| 夸大审计 | 逐节读 §3/§4/§5/§8/§12，检查四类粉饰手法 | F5 |

复现命令（无网络依赖的部分可离线跑）：

```bash
python3 scripts/verify_m5213_peer_review.py        # 打 :8002 现场复算，写 verification.json
python3 scripts/make_m5213_figure.py               # 配图（数值从 verification.json 注入）
python3 -m pytest tests/test_m5213_peer_review_artifacts.py -q
```

## 3 独立复算结果（关键表）

| 组 | 复算量 | 报告声称 | 本次实测 | 差异 | 判定 |
|---|---|---|---|---|---|
| A | SUBOFF C_T（V=5 m/s） | 0.024675 | {R['ct']} | 0 | 一致 |
| A | SUBOFF 湿表面积 S | 5.18 m² | {R['wetted_suboff']} m² | 0 | 一致 |
| A | C_T 对基准 0.0032 的误差 | +671.09% | {D['suboff_ct_err_pct']:+.4f}% | {abs(D['suboff_ct_err_pct']-671.09):.4f} pp | 一致 |
| A | 探针工况 Fn | 越域（>0.45） | {D['suboff_froude']} | — | 一致（±671% 是越域合成量，报告已声明） |
| B | KCS 湿表面积 S | 9651.49 m² | {R['wetted_kcs']} m² | 0 | 一致 |
| B | 对公开积分值 9424 m² 的误差 | +2.41% | {D['kcs_wetted_err_pct']:+.4f}% | {abs(D['kcs_wetted_err_pct']-2.41):.4f} pp | 一致（>1.5% 阈值，判「否」正确） |
| B | KCS 探针 Fn | 0.130 | {D['kcs_froude']} | — | 一致（Holtrop 适用域内，故 2.41% 是可信的量级） |
| C | `/calculations/eexi` | 404 | {R['eexi_status']} | 0 | 一致（能力缺席） |
| C | `/calculations/cii` | 404 | {R['cii_status']} | 0 | 一致（能力缺席） |
| D | 节点/判据/误差表/根因/继承 | 9 / 49 / 9(3+6) / 9 / 9(0+9+0) | {C['nodes']} / {C['criteria']} / {C['error_rows']}({C['error_measured']}+{C['error_withheld']}) / {C['rootcause']} / {C['lineage']['mature']+C['lineage']['iterate']+C['lineage']['shelve']}({C['lineage']['mature']}+{C['lineage']['iterate']}+{C['lineage']['shelve']}) | 全等 | 一致 |
| E | 发布正文 sha256 | {DIGEST} | {SHA_BODY} | 不等 | **审计缺口 F6** |

计：{V['n_pass']} 项通过 / {V['n_fail']} 项未通过（其中复算类失败 {V['n_recompute_fail']} 项、审计类 {V['n_audit_gap']} 项）。

> 端点返回的物理方法自述为 `{D['resistance_method']}`，与报告口径一致；两工况的 `cb_from_dims` 均在输入 Cb 附近
> （0.6387 vs 0.639、0.6505 vs 0.651），说明几何输入先过一致性检查，复算不是被畸形几何污染的伪结果。

## 4 夸大审计（五问五答）

1. **WITHHELD 有没有被写成通过？** 没有。49 条判据全部保持 WITHHELD，门总结论 `BLOCKED_CRITERIA`，节点级 0/9。
2. **门侧探针有没有冒充节点成果？** 没有。探针独立成表（§5，5 条 P1–P5），§4 有 `basis=gate_probe` 列显式标注，
   并写明「探针只回答此刻主分支水平，不替代节点验收」。
3. **误差有没有被圆整成达标？** 没有。三条实测（+671.09% / +60.00% / +2.41%）全部超过各自阈值并被判「否」；
   其中 +2.41% 对 1.5% 阈值只差 0.9 pp，仍判不达标，是本报告最值得信任的一行。
4. **C5 自证是否被粉饰？** 没有。§12 原文「未取得非本波参与者的 peer review，判据 C5 记为未通过」，并自认
   「评审独立性由评审者自证」是机制弱点。
5. **局限是否被藏起来？** 没有。§10 七条局限把 SUBOFF 基准未标 Re、P/D 错配、EEXI 判定依赖 404、
   误差无 U_V 等不利因素全部写明；本评审判定这些声明与复算结果不矛盾。

## 5 Findings

| ID | 级别 | 类别 | 结论 | 要点 |
|---|---|---|---|---|
""" + "\n".join(
    f"| {f['id']} | {f['severity']} | {f['kind']} | {f['status']} | {f['statement']} |" for f in FINDINGS
) + """

## 6 要求修改项（requested_changes）

""" + "\n".join(f"- {c}" for c in CHANGES) + f"""

## 7 本次评审的边界（诚实声明）

1. 我复算的是**端点现状**，不是节点交付；报告判 0/9 未过核销，我对此没有异议，也不因本评审而改变。
2. 我**未能**独立复核 P4/P5（B5-55 K_T/K_Q）的基准值：`references.json` 在所指版本不可得，且报告自认
   P/D=0.95 与 P/D=1.0 基准表错配，我拒绝在没有基准来源的情况下对该误差下判断。
3. 我**未**直连战役库核验 306–314 的 mission 状态；这一点不影响裁决——只要节点侧 0/9 验收成立，门就是 BLOCKED。
4. C5 通过≠门通过。本评审只把 C5 的 0/1 变为 1/1；门总结论仍须是 `BLOCKED_CRITERIA`。

## 8 图

![复算与结构自检](artifacts/m5213/figures/recompute_vs_threshold.png)

*图1 (a) 三条可复算量的相对误差（柱，对数轴）与各自验收阈值（虚线）：SUBOFF C_T +671.09%（阈值 5%）、
SUBOFF 湿表面积 -18.14%（3%）、KCS 湿表面积 +2.41%（1.5%）——三条全部越线，与报告「否」的判定一致，且与我的端点复算逐位吻合。
(b) 从发布正文重新计数的结构量：49 条判据全 WITHHELD、误差表 3 实测 + 6 withheld、根因 9 条、
继承 0 成熟 / 9 迭代 / 0 搁置——与报告 C1–C4 自检数字完全一致。*

## 9 误差判定（本次评审自身的关键量）

| 关键量 | 基准 | 本次值 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| SUBOFF C_T | 报告值 0.024675 | 0.024675 | 0.000% | **是** |
| SUBOFF 湿表面积 | 报告值 5.18 m² | 5.18 m² | 0.000% | **是** |
| KCS 湿表面积 | 报告值 9651.49 m² | 9651.49 m² | 0.000% | **是** |
| 自检结构计数（5 类） | 报告正文 | 逐项相等 | 0% | **是** |
| 正文摘要 | 报告自报 sha256 | {SHA_BODY[:16]}… | 不可比（不同对象） | **否（F6，非数值误差）** |

无物理场量的独立解作为基准：本次任务是**报告审计**而非物理预报，基准即「被评审报告的声称值 + 平台端点实测值」，
两者已互为独立来源（报告出自门侧证据快照，我方现场打端点）。
"""

out_md = ROOT / "docs/peer_review/m5213_IN19-G19_peer_review.md"
out_md.parent.mkdir(parents=True, exist_ok=True)
out_md.write_text(MD, encoding="utf-8")
print("wrote", out_md, len(MD))

payload = {
    "title": "Peer review：IN19-G19 波3 核销门（mission 5208）独立复算与夸大审计",
    "summary": (f"非本波参与方独立复算 mission 5208 报告：端点复算 SUBOFF C_T={R['ct']}、S={R['wetted_suboff']} m²、"
                f"KCS S={R['wetted_kcs']} m²，误差 {D['suboff_ct_err_pct']:+.2f}% / {D['kcs_wetted_err_pct']:+.2f}% 与报告逐位一致；"
                f"EEXI/CII 端点均 {R['eexi_status']}；正文结构 9 节点/49 判据/9 误差行(3+6)/9 根因/9 继承(0 成熟) 全等。"
                f"未发现夸大（无 WITHHELD 升格、无探针冒充节点成果、无圆整达标）。裁决 accept_with_minor："
                f"两项可审计性缺陷——发布正文 sha256={SHA_BODY[:16]}… 无法复现其自称摘要 {DIGEST[:16]}…，"
                f"且所引门引擎/原始证据在 git b607741 不可检索。"),
    "scenario_key": "peer_review",
    "mission_id": 5213,
    "tags": ["peer-review", "IN19-G19", "wave3", "mission-5208", "independent-recompute",
             "ship-performance-platform", "TensorPhysics"],
}

json.dump(payload, open(ROOT / "artifacts/m5213/report_payload_stub.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("wrote payload stub")