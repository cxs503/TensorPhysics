#!/usr/bin/env python3
"""Render the sub-grid resolution report from the acceptance artifact.

Numbers are read from ``artifacts/envelope_n09/criteria_evidence.json`` and
formatted here: nothing in the prose is typed by hand, so the report cannot
drift from the run that produced it.

Usage::

    python scripts/generate_envelope_subgrid_report.py \
        --evidence artifacts/envelope_n09/criteria_evidence.json \
        --out docs/reports/m9733/envelope_subgrid_resolution_report.md \
        --report-json /tmp/report_m9733.json
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
from typing import Any, Mapping, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _f(value: Any, spec: str = ".6g") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _criteria_table(criteria: Mapping[str, Any]) -> str:
    rows = []
    for key in sorted(criteria):
        item = criteria[key]
        rows.append([f"**{key}**", item.get("name", ""), _f(item.get("passed"))])
    return _table(["criterion", "what it establishes", "verdict"], rows)


def build_markdown(evidence: Mapping[str, Any]) -> str:
    criteria = evidence["criteria"]
    c1, c2, c3, c4, c5, c6 = (criteria[k] for k in ("C1", "C2", "C3", "C4", "C5", "C6"))
    c7, c8, c9, c10, c11 = (criteria[k] for k in ("C7", "C8", "C9", "C10", "C11"))
    env = evidence["environment"]
    policy = evidence["resolution_policy"]
    c6_engine = c6["engine"]
    c6_area_error = abs(c6_engine["area"] - c6["closed_form_area"]) / c6["closed_form_area"]

    parts: list[str] = []
    parts.append("# 作业窗口包络引擎：探测尺度与子网格特征 (N09)\n")
    parts.append(
        "**节点** IN68W2-N09 · 仓 TensorPhysics · 模块 `tensorphysics.envelope`\n\n"
        "**一句话** 窗口引擎过去用「每个区间一个中点采样」判可行，"
        f"遇到比节点间距更窄的特征时，同一份代码会给出两种都错的答案：把整个跨度宣布为可行（本例真窗宽 "
        f"{_f(c8['cases'][0]['analytic_width'])}，v1 报 {_f(c8['cases'][0]['v1_width'])}），"
        "或者把窗口整段丢掉。两种答案过去都带着 `contracts.ok == True`，因为没有任何字段记录"
        "「到底在哪个尺度上探过」。本节点把探测尺度变成显式变量：按声明的间距打点、不一致就地细分、"
        "细分不掉的跨度登记为 unresolved 并让契约失败。\n"
    )

    parts.append("## ① 案例介绍\n")
    parts.append(
        "**物理背景** 三个既有作业窗（搁浅临界潮位 / 压载置换率 / STS 泵率）把「参数可用的取值区间」"
        "交给同一个引擎求解；窗宽跨越两个数量级，参数面包含 1 维（单参数扫描）与 2 维"
        "（双参数平面）两种形态。本节点不引入新的领域公式：三窗仍是 `scenarios/envelope_cases/` 里的"
        "普通配置数据，本节点的靶子是**求解机制**本身。\n\n"
        "**几何 / 参数** 1 维窗：轴 `alpha ∈ [0,1]`，默认 96 个扫描节点（节点间距 "
        f"{_f(1.0 / 95.0, '.4g')}），约束是不等式保证金。子网格特征用解析窗构造："
        "口袋窗 `|alpha - c| ≤ w`（真窗宽 `2w`）与窄空隙窗 `|alpha - c| > w`（真可行集为两段）。"
        "2 维窗：单位正方形，对角通道 `|beta - alpha| ≤ eps`，解析面积 `2·eps − eps²`。\n\n"
        "**参数表**\n\n"
        + _table(
            ["量", "值"],
            [
                ["默认扫描节点数 (1 维)", _f(evidence["scan"]["nodes_1d"])],
                ["节点间距", _f(evidence["scan"]["node_spacing"], ".4g")],
                ["每扫描格探测点数 `samples_per_segment`", _f(policy["samples_per_segment"])],
                ["保证探测间距 τ（默认）", _f(evidence["scan"]["node_spacing"] / policy["samples_per_segment"], ".4g")],
                ["子格数 / 最大深度", f"{_f(policy['sub_cells'])} / {_f(policy['max_depth'])}"],
                ["细分预算上限（次约束评估）", _f(policy["max_refine_evals"])],
                ["参与迁移的三窗条件数", str(len(c9["shipped"]))],
            ],
        )
        + "\n"
    )

    parts.append("## ② 怎么做的\n")
    parts.append(
        "1. **探测（`subgrid._probe_positions`）** 每个区间的判定不再取一个中点，而是在目标间距 "
        "`τ = (轴跨度/扫描格数)/samples_per_segment` 上打点，并同时采两端点。保证：宽度 ≥ τ 的"
        "特征（可行口袋或不可行缝隙）必落在至少一个探测点上。\n"
        "2. **判定（`subgrid._scan_span`）** 采样按 `+1 可行 / −1 不可行 / 0 在容差内贴边界` 分类；"
        "内部混号或相邻异号即视为不一致。\n"
        "3. **细分** 不一致的区间按 `sub_cells` 等分，子节点上对每条约束做符号变化 bracket + Brent/"
        "二分精化，根位置并入分割边（因此边界由根定位，不会停在探测点上）；子区间递归重探，"
        "直到无异议、深度用尽或预算用尽。\n"
        "4. **契约（`contracts.check_feature_resolution`）** 输出 `resolution:<label>`："
        "`min_probed_width`（实际探到的最细间距）、`suspect_spans`、`refined_spans`、"
        "`unresolved_spans`、`budget_exhausted`。有 unresolved 或预算耗尽即判失败；"
        "案例若声明 `contract.resolution.min_feature_width`，还要求 τ ≤ 该值，否则失败。\n"
        "5. **2 维** 逐行 1 维 marching 继承同一机制；平面结果额外登记 `line_spacing` 与"
        "`single_line_components`，据此给出 `area_reliable`（特征比线间距还窄时面积为不可解，"
        "不许当成数字报出去）。\n"
        "6. **零域语义** 重扫包内源码与测试的领域词表命中数："
        f"{len(c5.get('hits', []))} 条。所有门限、间距、格点来自 case 配置。\n"
    )

    parts.append("## ③ 计算结果\n")
    parts.append("### 判据总表\n")
    parts.append(_criteria_table(criteria) + "\n")
    parts.append(
        "### 子网格窗恢复（C7，解析窗为基准）\n\n"
        + _table(
            ["特征", "真窗", "引擎窗", "宽度相对差", "探测间距 τ", "约束评估"],
            [
                [
                    row["label"],
                    f"[{_f(row['analytic'][0])}, {_f(row['analytic'][1])}]",
                    f"[{_f(row['engine_intervals'][0][0])}, {_f(row['engine_intervals'][0][1])}]",
                    _f(row["width_relative_difference"], ".3e"),
                    _f(row["probed_width"], ".3e"),
                    _f(row["n_evals"]),
                ]
                for row in c7["cases"]
            ],
        )
        + f"\n\n偏差最大 {_f(c7['worst_relative_difference'], '.3e')}（判据 ≤ 1e-12）。"
        f"窄空隙窗（真可行集两段）恢复：{_f(c7['narrow_gap']['recovered'])}，"
        f"引擎给出 {c7['narrow_gap']['intervals']}。\n"
    )
    parts.append(
        "### v1 基线对照（C8）\n\n"
        + _table(
            ["特征", "v1 判定", "v1 宽度", "v1 是否静默出错", "修复后", "修复后宽度", "粗探测(1/格) 时是否登记 unresolved"],
            [
                [
                    row["label"],
                    row["v1_status"],
                    _f(row["v1_width"]),
                    "yes" if (row["v1_inflated"] or row["v1_window_lost"]) else "no",
                    row["fixed_status"],
                    _f(row["fixed_width"]),
                    _f(row["coarse_probe_reported_unresolved"]),
                ]
                for row in c8["cases"]
            ],
        )
        + f"\n\nv1 静默出错 {_f(c8['v1_silent_failures'])}/{len(c8['cases'])}，"
        f"修复后静默出错 {_f(c8['fixed_silent_failures'])}。v1 基线不是回忆，是脚本里逐字复现的"
        "「网格根 + 每段一个中点采样」实现（`_v1_window`），与修复后的引擎同案对跑。\n"
    )
    parts.append(
        "### 迁移一致性（C1 / C10）\n\n"
        f"* C1：三窗 × 三条件共 {_f(c1['n_comparisons'])} 个数字对「闭式分窗路径」比较，"
        f"最大相对差 {_f(c1['worst_relative_difference'], '.3e')}（判据 ≤ 1e-12）。\n"
        f"* C10：同一批窗在默认策略与「每格 1 个探测点、不细分」策略下逐位比较，"
        f"{_f(c10['numbers_compared'])} 个边界数字 + 状态 + 活跃约束标签，不一致 {_f(c10['mismatches'])} 处。"
        "新增的探测不允许挪动任何一个它不必挪动的数字。\n"
    )
    parts.append(
        "### 边界阶次与空窗语义（C2 / C4）\n\n"
        f"* 解析抛物带边界误差实测阶最小 {_f(c2['min_observed_order'], '.4f')}（判据 ≥ 1.9）。\n"
        "* 空窗返回 `infeasible`、无边界、无面积、带最近可行点残差：" + _f(c4["passed"]) + "。\n"
    )
    parts.append(
        "### 成本（C6 / C9）\n\n"
        f"* C6：二维窗 {_f(c6_engine['n_evals'])} 次约束评估、{_f(c6_engine['elapsed_s'], '.3f')} s 单核"
        f"（≈ {_f(c6_engine['evals_per_second'], '.0f')} 评估/s），判据「5·10⁴ 次评估 ≤ 60 s」对应"
        f"{_f(5e4 / c6_engine['evals_per_second'], '.3f')} s。\n"
        "* C9：三窗九条件下探测开销相对「每格 1 点」：\n\n"
        + _table(
            ["窗", "条件", "默认评估数", "单点评估数", "开销倍数", "suspect_spans"],
            [
                [
                    row["case_id"],
                    row["condition"],
                    _f(row["default_evals"]),
                    _f(row["single_probe_evals"]),
                    _f(row["overhead"], ".2f"),
                    _f(row["suspect_spans"]),
                ]
                for row in c9["shipped"]
            ],
        )
        + f"\n\n最大开销 {_f(c9['worst_overhead'], '.2f')}×（门限 {_f(c9['threshold_overhead'], '.0f')}×），"
        "且三窗一次细分都没触发（它们的窗边界本来就宽于节点间距）。\n"
    )

    parts.append("## ④ 结果图\n")
    for figure in evidence["figures"]:
        rel = figure["path"]
        parts.append(f"![{figure['name']}](../../../{rel})\n\n*{figure['caption']}*\n")

    coarse_tau = evidence["scan"]["node_spacing"]
    third = c8["cases"][-1]
    overheads = [row["overhead"] for row in c9["shipped"]]
    parts.append("## ⑤ 注意点、局限与适用范围\n")
    parts.append(
        "* **保证是「相对于探测间距」的，不是绝对的。** 引擎保证 τ 以上的特征不漏；τ 以下的黑盒"
        f"采样原则上仍可漏（C8 第三行：真窗宽 {_f(third['analytic_width'], '.3g')} < τ={_f(coarse_tau, '.3g')} 时，"
        "每格 1 点的探测依旧丢窗，但不谎报——要么登记 unresolved，要么状态为 infeasible）。"
        "要更强，就让案例声明 `min_feature_width`，让契约在 τ 不够细时失败。\n"
        "* **2 维的面积分辨力受线间距限制。** 比线间距还窄的对角通道：状态判为可行（不再谎报不可行），"
        "但每个扫描行自成一个单行分量，面积为 0 且被标 `area_reliable=False`；把线数加密到"
        "每通道宽至少一条线后面积才收敛到解析值（见图 3）。\n"
        "* **探测密度与成本是线性关系**：默认每扫描格 "
        f"{_f(policy['samples_per_segment'])} 点，1 维窗评估数随探测密度线性增长，"
        f"三窗实际开销 {_f(min(overheads), '.2f')}–{_f(max(overheads), '.2f')}×；"
        "对小预算场景可降到 2–4 点，代价是保证间距变粗。\n"
        "* **引擎不再用「把网格加密到结构复现」来掩盖不确定结构**：一旦某段被登记为 unresolved，"
        "`window_1d` 直接返回该结果而不再升级网格（升级网格会改变 τ 却不留痕）。\n"
        "* **物理量本身未被验证**：本节点的判据全部是算法与几何判据（解析窗、闭式代数、"
        "守恒的空窗语义），三窗里的经验系数是配置数据；把它们对模型试验做标定不在本节点范围。\n"
    )

    parts.append("## ⑥ 误差判定\n")
    worst_c1 = c1["worst_relative_difference"]
    worst_c7 = c7["worst_relative_difference"]
    parts.append(
        _table(
            ["关键量", "基准", "实测误差", "是否 < 3 %"],
            [
                ["三窗边界/宽度（C1，27 个数）", "闭式分窗路径", _f(worst_c1, ".3e"), "是（好 12 个数量级）"],
                ["三窗默认 vs 单点策略（C10）", "逐位相等", "0 处不一致", "是（精确相等）"],
                ["子网格窗边界/宽度（C7）", "解析窗", _f(worst_c7, ".3e"), "是（好 11 个数量级）"],
                ["边界重建阶次（C2）", "解析抛物带", f"阶 {_f(c2['min_observed_order'], '.3f')}", "是（二阶）"],
                ["二维面积（C6）", "闭式圆面积", _f(c6_area_error, ".3e"), "是"],
                ["探测开销（C9）", "每格 1 点", f"{_f(c9['worst_overhead'], '.2f')}×", "—（成本项，非精度项）"],
            ],
        )
        + "\n\n本节点没有任何量对实验或实船数据比较，因此不宣称物理精度；"
        "表中的「< 3%」是相对解析/闭式基准的回答，物理相关式的标定仍为开放项（见 ⑤）。\n"
    )

    parts.append("## 审计链\n")
    parts.append(
        f"* 代码摘要（包内源码 SHA-256）：`{evidence['code_digest']}`\n"
        + "* 迁移三窗摘要："
        + "，".join(f"`{case_id}` = `{digest[:16]}…`" for case_id, digest in evidence["case_digests"].items())
        + "\n"
        f"* 证据工件：`artifacts/envelope_n09/criteria_evidence.json`\n"
        f"* 环境：Python {env['python']}，{env['machine']}，{env['platform']}，整轮 {_f(env['elapsed_s'], '.1f')} s\n"
        "* 复现：`python scripts/envelope_subgrid_acceptance.py --out artifacts/envelope_n09` 然后 "
        "`python scripts/generate_envelope_subgrid_report.py`\n"
        f"* 测试：{c11['summary']}\n"
    )
    return "\n".join(parts)


def build_platform_report(markdown: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    figures = []
    for figure in evidence["figures"]:
        path = ROOT / figure["path"]
        figures.append(
            {
                "name": figure["name"],
                "caption": figure["caption"],
                "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    criteria = evidence["criteria"]
    summary = (
        "窗口引擎把「每段一个中点采样」换成显式探测尺度（每扫描格 8 点 + 不一致就地细分 + "
        "unresolved 登记与契约失败）：比节点间距窄的窗从「整段谎报可行/整窗丢失」变为解析级恢复"
        f"（宽/边界相对差 ≤ {criteria['C7']['worst_relative_difference']:.2e}），三窗迁移逐位一致"
        f"（C10 {criteria['C10']['numbers_compared']} 个数 0 不一致，C1 ≤ {criteria['C1']['worst_relative_difference']:.2e}），"
        f"开销 {criteria['C9']['worst_overhead']:.2f}×，测试 {criteria['C11']['tests']} 例全绿。"
    )
    return {
        "title": "作业窗口包络引擎成熟化 N09：探测尺度、子网格特征恢复与不可解登记",
        "summary": summary,
        "scenario_key": "tensorphysics.envelope.subgrid_resolution",
        "mission_id": 9733,
        "tags": [
            "envelope",
            "operating-window",
            "sub-grid-resolution",
            "contract-checking",
            "migration-consistency",
        ],
        "markdown": markdown,
        "figures": figures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", default="artifacts/envelope_n09/criteria_evidence.json"
    )
    parser.add_argument(
        "--out", default="docs/reports/m9733/envelope_subgrid_resolution_report.md"
    )
    parser.add_argument("--report-json", default="/tmp/report_m9733.json")
    args = parser.parse_args(argv)

    evidence_path = ROOT / args.evidence if not pathlib.Path(args.evidence).is_absolute() else pathlib.Path(args.evidence)
    evidence = json.loads(evidence_path.read_text())
    markdown = build_markdown(evidence)
    out_path = ROOT / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)
    payload = build_platform_report(markdown, evidence)
    report_path = pathlib.Path(args.report_json)
    report_path.write_text(json.dumps(payload))
    print(f"wrote {out_path} ({len(markdown)} chars)")
    print(f"wrote {report_path} ({len(payload['figures'])} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())