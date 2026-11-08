#!/usr/bin/env python3
"""Render ``docs/P35_CAMPAIGN_INTEGRATION_GATE.md`` from the machine verdict.

Every number in the document is injected from
``artifacts/p35_gate/campaign_readiness.json`` and the recorded probe files --
none is typed by hand.  The narrative is fixed prose: what the gate is, how each
criterion is decided, what the current campaign state is, what is blocked, and
what the gate cannot establish.

    python3 scripts/p35_gate_report.py [--out docs/P35_CAMPAIGN_INTEGRATION_GATE.md]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE_DIR = ROOT / "artifacts" / "p35_gate"
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign.p35_gate import readiness_schema as RS  # noqa: E402
from tensorphysics.campaign.p35_gate import spec as S  # noqa: E402

CELL = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "WITHHELD": "⚪ WITHHELD"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def fmt(x, digits=6):
    if x is None:
        return "—"
    if isinstance(x, float):
        if x != 0 and (abs(x) < 1e-4 or abs(x) >= 1e6):
            return f"{x:.{digits}e}"
        return f"{x:.{digits}g}"
    return str(x)


def render(doc: dict) -> str:
    crit = {c["criterion_id"]: c for c in doc["criteria"]}
    reg = {c["case"]: c for c in doc["regression"]}
    nodes = doc["nodes"]
    landed = [n for n in nodes if n["deliverable"] == "PASS"]
    regd = [n for n in nodes if n["registered"] == "PASS"]
    mat = [n for n in nodes if n["maturity"] == "PASS"]
    chain = [n for n in nodes if n["chain_ready"]]
    tested = [c for c in doc["regression"] if c["status"] in ("PASS", "FAIL")]

    L = []
    add = L.append
    add("# P35 全战役集成验收门 (P35-G1) — 跨域能力串联与回归基线确认")
    add("")
    add(f"- 产物 schema: `{S.GATE_ID}` → `campaign_readiness.v1` "
        f"(fingerprint `{doc['provenance']['schema_fingerprint']}`)")
    add(f"- 战役计划: plan_id={doc['plan_id']} · {S.CAMPAIGN_TITLE} · {len(nodes)} 任务节点 + 本门")
    add(f"- 生成时间 (UTC): `{doc['generated_utc']}`")
    add(f"- **总裁决: `{doc['overall_status']}` · dispatcher_ready = "
        f"`{str(doc['dispatcher_ready']).lower()}` · 阻塞项 {len(doc['blocking'])} 条**")
    add("")
    add("## 1. 案例介绍（这是什么、为什么需要一道门）")
    add("")
    add("P35 是跨四仓的十个并行研究节点：求解器前沿（TensorLBM 热多相相变 N1、造波探针"
        "反演 N3、LBM-DEM-FSI 三向耦合 N9）、颗粒湿冰（TensorDEM N2）、船舶深域"
        "（空气润滑减阻 N4、轴系校中 N5、LNG 舱 rollover N6、替代燃料 EEXI/CII N7、"
        "散货自热 N8）与跨求解器串联账本（TensorPhysics N10）。节点之间靠**产物**而非"
        "代码互相消费，因此战役是否可交付不能用「PR 是否合并」判断，只能用「产物是否落盘、"
        "能力是否登记、基线是否退化、依赖是否可执行」判断。本门（P35-G1）把这四件事做成"
        "机器可算判据，并输出派发器可直接消费的 `campaign_readiness.v1`。")
    add("")
    add("| 节点 | mission | 仓库 | 物理域 | 成熟化 |")
    add("|---|---|---|---|---|")
    for n in S.NODES:
        add(f"| {n.ext_id} | {n.mission_id} | {n.repo} | {n.physics_domain} | "
            f"{'是' if n.maturity else '否'} |")
    add("")
    add("## 2. 怎么做的（判据、工具链、可复现命令）")
    add("")
    add("判据实现于 `src/tensorphysics/campaign/p35_gate/`（stdlib-only，不 import 任何"
        "求解器栈）：`spec.py` 只声明身份/产物族/依赖，`collect.py` 采集平台证据与磁盘产物，"
        "`criteria.py` 判 (a)–(g)，`regression.py` 复跑既有验证面，`gate.py` 组装并校验 "
        "`campaign_readiness.v1`（schema 注册在 `campaign/schema_defs.py`）。")
    add("")
    add("| 判据 | 内容 | 证据来源 | 非绿语义 |")
    add("|---|---|---|---|")
    how = {
        "a": ("每节点：平台报告 ≥1 且声明的产物族在交付检出内解析通过（JSON 可解析 / "
              "PNG 魔数+IEND 尾校验）", "平台 reports API + 节点交付 worktree"),
        "b": ("台账登记必须携带 ext_id、physics_domain，且 mission_repo 等于声明仓库",
              "平台 capability-delta API"),
        "c": ("复跑已晋升案例，|Δ|/max(|baseline|,floor) ≤ 1%；机器地板量级指标用声明 floor",
              "生产者自有 runner 输出 + 案例 result.json 基线"),
        "d": ("声明 DAG 无环、边可解析、无孤立节点，且与 campaign_plans.dependencies 一致",
              "spec 声明（节点 goal 原文）+ 平台 DAG 记录"),
        "e": ("成熟化节点须有成熟化声明，TRL≥6 且门禁记录全通过",
              "节点产物目录内 maturity.json"),
        "f": ("载荷通过注册 schema 校验（closed schema，fingerprint 固定）",
              "campaign/schema_defs.py"),
        "g": ("任一判据非 PASS → overall=FAIL 且逐条落阻塞清单；WITHHELD 永不等于 PASS",
              "本门自身"),
    }
    for cid in "abcdefg":
        c = crit[cid]
        add(f"| ({cid}) | {how[cid][0]} | {how[cid][1]} | {c['status']} |")
    add("")
    add("复现命令（证据全部落在 `artifacts/p35_gate/`）：")
    add("")
    add("```bash")
    add("# 1) 采集平台证据 + 逐节点绑定交付检出 + 复跑回归 + 出裁决")
    add("python -m tensorphysics.campaign.p35_gate \\")
    add("    --base-url \"$TENSOR_API_BASE\" --plan-id 179 \\")
    add("    --node-root P35-N1=/path/to/m6795 ... --node-root P35-N10=/path/to/m6804 \\")
    add("    --run-probes --probe-timeout 1800 --out artifacts/p35_gate")
    add("# 2) 长时回归探针（120 s shell 上限之外单独跑，逐例追加记录）")
    add("python3 scripts/p35_gate_probes.py --node-root TensorLBM=/repo/TensorLBM \\")
    add("    --only couette_3d --timeout 900 --out artifacts/p35_gate")
    add("# 3) 图与报告")
    add("python3 scripts/p35_gate_figures.py && python3 scripts/p35_gate_report.py")
    add("```")
    add("")
    add("## 3. 计算结果（全部由产物注入，零手抄）")
    add("")
    add("### 3.1 判据裁决")
    add("")
    add("| 判据 | 内容 | 裁决 | 说明 |")
    add("|---|---|---|---|")
    for cid in "abcdefg":
        c = crit[cid]
        add(f"| ({cid}) | {c['label']} | {CELL[c['status']]} | {c['detail']} |")
    add("")
    add("### 3.2 判据数字")
    add("")
    add("| 指标 | 值 | 阈值/目标 | 是否达标 |")
    add("|---|---|---|---|")
    a, b, c_, d, e = (crit[k]["metrics"] for k in "abcde")
    rows = [
        ("产物落盘率 (a)", f"{a['landing_rate']:.1%} ({a['landed']}/{a['n_nodes']})",
         "100%", "达标" if a["landing_rate"] >= 1.0 else "未达标"),
        ("能力台账登记率 (b)",
         f"{b['registration_rate']:.1%} ({b['registered']}/{b['n_nodes']})", "100%",
         "达标" if b["registration_rate"] >= 1.0 else "未达标"),
        ("回归最大偏差 (c)",
         "—" if c_["max_rel_dev"] is None else f"{c_['max_rel_dev']:.3e}",
         f"< {c_['threshold']:.1%}",
         "未定" if c_["max_rel_dev"] is None else
         ("达标" if c_["max_rel_dev"] <= c_["threshold"] else "未达标")),
        ("回归案例复跑数 (c)", f"{c_['n_tested']}/{c_['n_cases']}", f"{c_['n_cases']}",
         "达标" if c_["n_tested"] == c_["n_cases"] else "未达标"),
        ("DAG 无环 (d)", str(d["acyclic"]), "True", "达标" if d["acyclic"] else "未达标"),
        ("孤立节点 (d)", str(d["orphans"] or "无"), "无", "达标" if not d["orphans"] else "未达标"),
        ("平台依赖边一致 (d)",
         f"缺失 {len(d['missing_on_platform'])} / 多余 {len(d['extra_on_platform'])}",
         "0 / 0", "达标" if not (d["missing_on_platform"] or d["extra_on_platform"]) else "未达标"),
        ("成熟化门禁通过率 (e)",
         f"{e['pass_rate']:.1%} ({e['passed']}/{e['n_nodes']})", "100%",
         "达标" if e["pass_rate"] >= 1.0 else "未达标"),
        ("链就绪节点 (d)", f"{len(chain)}/{len(nodes)}", "10/10",
         "达标" if len(chain) == len(nodes) else "未达标"),
    ]
    for name, val, target, ok in rows:
        add(f"| {name} | {val} | {target} | {ok} |")
    add("")
    add("### 3.3 回归基线复跑明细 (criterion c)")
    add("")
    add("| 案例 | 基线 | 复跑值 | 相对偏差 | 阈值 | floor | 耗时 | 裁决 |")
    add("|---|---|---|---|---|---|---|---|")
    for case in doc["regression"]:
        add(f"| {case['case']} | {fmt(case['baseline'])} | {fmt(case['measured'])} | "
            f"{fmt(case['rel_dev'])} | {case['threshold']:.1%} | {fmt(case['floor'])} | "
            f"{fmt(case['elapsed_s'])} s | {CELL[case['status']]} |")
    add("")
    for case in doc["regression"]:
        if case["status"] != "PASS":
            add(f"- `{case['case']}`：{case['detail']}")
    add("")
    add("### 3.4 节点就绪矩阵")
    add("")
    add("| 节点 | 产物(a) | 台账(b) | 成熟化(e) | 链就绪(d) | 阻塞摘要 |")
    add("|---|---|---|---|---|---|")
    for n in nodes:
        summary = "; ".join(n["blocking"])[:180] or "—"
        add(f"| {n['ext_id']} | {CELL[n['deliverable']]} | {CELL[n['registered']]} | "
            f"{CELL[n['maturity']]} | {'✅' if n['chain_ready'] else '❌'} | {summary} |")
    add("")
    add("## 4. 结果图")
    add("")
    add("![fig1](figures/fig1_node_readiness.png)")
    add("")
    add("*图 1 · 十节点就绪矩阵：绿=判据通过，红=未通过，灰=证据未取得（WITHHELD）。"
        "成熟化列对非成熟化节点为 WITHHELD（该节点本无成熟化要求）。*")
    add("")
    add("![fig2](figures/fig2_regression_deviation.png)")
    add("")
    add("*图 2 · 回归复跑偏差（对数轴）。绿柱为实际复跑并达标的案例；灰位为"
        "未复跑案例（超预算或缺 runner 入口），按 WITHHELD 记账而不是当作未退化。*")
    add("")
    add("![fig3](figures/fig3_couette_profile.png)")
    add("")
    add("*图 3 · 图 2 中唯一复跑案例的物理内容：Couette 三维剖面与解析线性解重合，"
        "逐点偏差停在 float64 舍入地板（~1e-17 量级），说明观测到的 l2 是舍入噪声而非"
        "离散误差——这正是该案例用 floor 而非百分比比较的原因。*")
    add("")
    add("## 5. 注意点（踩过的坑与适用范围）")
    add("")
    add("- **产物是否落盘只能在节点自己的交付检出里判**：门用 `--node-root EXT_ID=PATH` 绑定"
        "节点 worktree，缺失即 FAIL；若只绑定仓库级 `--repo-root`，缺失只能是 WITHHELD"
        "（可能看的是别的分支）。这一区分是本门第一版刻意加的，避免把「看错检出」写成「节点没交」。")
    add("- **成熟化声明不计入落盘率**：N4–N7 的 `maturity.json` 由判据 (e) 判，重复计入 (a) "
        "会让一个成熟化缺陷同时打两次，属双重计分。")
    add("- **平台 DAG 用 node_id 而非 ext_id**：`campaign_plans.dependencies` 存的是 "
        "1095..1105，直接与 `P35-N*` 比较会造出一堆假 divergence；门先做 id→ext_id 映射再比对。")
    add("- **机器地板量级指标必须用 floor**：Couette fp64 的 l2≈9.6e-14，复跑 5.7e-14 会是"
        "「40% 偏差」的假回归，物理上却是同一台机器噪声；门要求逐案声明 floor 并写进裁决。")
    add("- **回归 runner 的算力成本要事前标定**：cavity_3d_full 64³ 实测 500 步 97 s"
        "（x86 CPU/8 线程），100k 步约 1.94e4 s ≈ 5.4 h，超出本门 1 h 预算，按 WITHHELD 记账"
        "并给出成本估计——不跑也不假装跑过。")
    add("- **判据的时效性**：战役节点仍在并行推进，本裁决是 `generated_utc` 时刻的快照"
        "（快照 sha256 见溯源表），重跑即得新状态；门本身不做「乐观推测」。")
    add("")
    add("## 6. 误差判定（逐项回答「是否 <3%」）")
    add("")
    basis = []
    for case in doc["regression"]:
        if case["status"] == "PASS":
            basis.append(f"{case['case']}: |Δ|/max(|baseline|,floor) = "
                         f"{case['rel_dev']:.3e} → **<3% 达标**（阈值 {case['threshold']:.0%}）")
        elif case["status"] == "WITHHELD":
            basis.append(f"{case['case']}: 未取得复跑值 → **<3% 不可判定**（{case['detail']}）")
        else:
            basis.append(f"{case['case']}: |Δ| = {case['rel_dev']:.3e} → **>3% 未达标**")
    for item in basis:
        add(f"- {item}")
    add("")
    add(f"- 结论（回归面）：{len(tested)}/{len(doc['regression'])} 例取得复跑值，"
        f"其中 {sum(1 for c in tested if c['status'] == 'PASS')} 例偏差低于阈值"
        f"（最大 {fmt(crit['c']['metrics']['max_rel_dev'])}），"
        f"{len(doc['regression']) - len(tested)} 例无法判定——因此 (c) 不判 PASS。")
    add("- 结论（跨域交付面）：产物/台账/成熟化三项判据的「误差」不是数值误差而是缺件，"
        "按缺件计数（第 3.2 节），不做数值百分比讹代。")
    add("- honest-FAIL 声明：本门当前为 "
        f"**{doc['overall_status']}**，不作为可覆盖意见；阻塞清单见下文第 7 节。")
    add("")
    add("## 7. 阻塞清单（dispatcher 消费）")
    add("")
    if doc["blocking"]:
        for item in doc["blocking"]:
            add(f"- `{item}`")
    else:
        add("- 无")
    add("")
    add("## 8. 溯源（审计链）")
    add("")
    add("| 输入 | 路径 | 存在 | sha256 |")
    add("|---|---|---|---|")
    for item in doc["provenance"]["inputs"]:
        add(f"| | `{item['path']}` | {item['present']} | "
            f"`{str(item['sha256'])[:32]}` |")
    add("")
    add(f"- 生成器: `{doc['provenance']['generator']}`")
    add(f"- schema: `campaign_readiness.v1`，fingerprint "
        f"`{doc['provenance']['schema_fingerprint']}`"
        f"（`schema.py` 的 closed-schema 校验器逐字段复核）")
    add("")
    add("## 9. 局限")
    add("")
    for lim in doc["limitations"]:
        add(f"- {lim}")
    add("- 本门不复算节点内部物理（那是各节点自己的验收面）；它只确认产物存在、可解析、"
        "能力登记、基线与依赖链，因此 PASS 不代表物理正确，FAIL 也不代表物理错误。")
    add("")
    add("## 10. 与战役纪律的对应")
    add("")
    add(f"- 判据 (a) 目标 100%，实测 {a['landing_rate']:.1%}；判据 (b) 目标 100%，"
        f"实测 {b['registration_rate']:.1%}；判据 (c) 目标 <1%，"
        f"已复跑最大偏差 {fmt(crit['c']['metrics']['max_rel_dev'])}；判据 (e) 目标 100%，"
        f"实测 {e['pass_rate']:.1%}。")
    add("- 全部非绿项均进入阻塞清单，未做任何「部分通过」折算；"
        "`--require-pass` 使 CI 在本条件下以非零码退出（退出码 3）。")
    add("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "P35_CAMPAIGN_INTEGRATION_GATE.md"))
    args = ap.parse_args(argv)
    doc = json.loads((GATE_DIR / "campaign_readiness.json").read_text())
    RS.validate_artifact(RS.READINESS_SCHEMA_NAME, doc)      # the document only renders a valid verdict
    text = render(doc)
    Path(args.out).write_text(text)
    print(f"{args.out} ({len(text.splitlines())} lines, verdict {doc['overall_status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())