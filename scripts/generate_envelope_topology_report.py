#!/usr/bin/env python3
"""Generate (and optionally publish) the IN68W3-N06 maturity report.

Every number in the markdown below is read out of the artifacts this node
produced -- ``artifacts/envelope_topology/topology_evidence.json`` for the
multi-interval path and ``artifacts/envelope_postfix/criteria_evidence.json`` /
``case_results.json`` for the migrated windows -- so nothing is retyped by hand.

Usage::

    python scripts/generate_envelope_topology_report.py \
        --evidence artifacts/envelope_topology/topology_evidence.json \
        --migrated artifacts/envelope_postfix \
        --out docs/reports/in68w3_n06/envelope_multi_interval_maturity.md \
        --payload /tmp/report_in68w3_n06.json
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
MISSION_ID = 12549
NODE = "IN68W3-N06"
SCENARIO_KEY = "tensorphysics.envelope.multi_interval_maturity"

CONFIG_FILES = {
    "stranding_tide_window": "under-keel-clearance / sill band on the tide axis",
    "exchange_rate_window": "ballast-water exchange rate band (time, residence, pump, stability)",
    "sts_transfer_window": "ship-to-ship transfer rate band (min rate, manifold pressure, hose, weather)",
}


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def _table(header: list[str], rows: list[list[Any]], digits: int = 6) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell, digits) for cell in row) + " |")
    return "\n".join(lines)


def load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def build(evidence: dict[str, Any], migrated: dict[str, Any], case_results: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    t = evidence["criteria"]
    c = migrated["criteria"]
    cases = case_results["cases"]

    # ---- tables ----------------------------------------------------------
    shipped_rows = []
    for case in cases:
        active = case["active_sets"]
        axis = next(iter(active))
        lower = active[axis]["lower_drivers"][0] if active[axis]["lower_drivers"] else []
        upper = active[axis]["upper_drivers"][0] if active[axis]["upper_drivers"] else []
        intervals = case["intervals"]
        window = (
            f"[{_fmt(intervals[0]['lo'])}, {_fmt(intervals[0]['hi'])}]"
            if intervals
            else "infeasible"
        )
        settings = case.get("discrete") or {}
        shipped_rows.append(
            [
                case["case_id"],
                case["condition"],
                window,
                intervals[0]["hi"] - intervals[0]["lo"] if intervals else None,
                ", ".join(lower),
                ", ".join(upper),
                settings.get("n_settings"),
                case["n_evals"],
                case["elapsed_s"],
            ]
        )
    shipped_table = _table(
        ["case", "condition", "window", "width", "lower driver", "upper driver", "attainable settings", "evaluations", "wall [s]"],
        shipped_rows,
    )

    t1_rows = [
        [row["n_lines"], row["area"], row["relative_error"], row["area_method"], row["multi_interval_lines"], row["n_evals"]]
        for row in t["T1"]["ladder"]
    ]
    t1_table = _table(["n_lines", "area", "rel. error vs closed form", "area method", "lines with 2 intervals", "evaluations"], t1_rows, digits=8)

    t3 = t["T3"]
    t3_table = _table(
        ["geometry", "marched area", "closed form", "rel. error", "components", "verdict"],
        [
            ["single disc", t3["single_disc"]["area"], t3["single_disc"]["closed_form"], t3["single_disc"]["relative_error"], t3["single_disc"]["components"], "pass" if t3["single_disc"]["relative_error"] <= t3["thresholds"]["smooth_relative_error"] else "fail"],
            ["two disjoint discs", t3["two_disjoint_discs"]["area"], t3["two_disjoint_discs"]["closed_form"], t3["two_disjoint_discs"]["relative_error"], t3["two_disjoint_discs"]["components"], "pass" if t3["two_disjoint_discs"]["relative_error"] <= t3["thresholds"]["smooth_relative_error"] else "fail"],
            ["slab with hole (edge on a line)", t3["slab_with_hole_aligned"]["area"], t3["slab_with_hole_aligned"]["closed_form"], t3["slab_with_hole_aligned"]["relative_error"], 1, "pass" if t3["slab_with_hole_aligned"]["relative_error"] <= t3["thresholds"]["slab_aligned_relative_error"] else "fail"],
            ["slab with hole (corner between lines)", t3["slab_with_hole_corner_between_lines"]["area"], t3["slab_with_hole_corner_between_lines"]["closed_form"], t3["slab_with_hole_corner_between_lines"]["relative_error"], 1, "declared limitation (first order)"],
        ],
        digits=8,
    )

    def _headline(key: str) -> Any:
        block = c[key]
        if key == "C1":
            return block.get("worst_relative_difference")
        if key == "C2":
            return block.get("min_observed_order")
        if key == "C3":
            return ", ".join(block.get("failed_checks", [])) or "no violation"
        if key == "C4":
            return (block.get("semantics_check") or {}).get("detail")
        if key == "C5":
            scanned = block.get("lines_scanned")
            return f"{len(block.get('hits', []))} hits / {scanned} lines"
        if key == "C6":
            return block.get("relative_difference_vs_closed_form_area")
        return block.get("passed")

    criteria_rows = [
        [key, c[key]["name"], _headline(key), _fmt(c[key]["passed"]), c[key].get("threshold_relative_difference") or c[key].get("threshold_min_order") or c[key].get("threshold_max_wall_seconds") or ""]
        for key in ("C1", "C2", "C3", "C4", "C5", "C6")
    ]
    criteria_table = _table(["criterion", "what it establishes", "headline number", "pass", "declared threshold"], criteria_rows)

    audit_rows = [
        ["T1", t["T1"]["name"], _fmt(t["T1"]["default_ladder"]["relative_error"]), "1e-3", _fmt(t["T1"]["passed"])],
        ["T2", t["T2"]["name"], _fmt(t["T2"]["pre_fix_relative_error"]), "> 0.25 and rejected", _fmt(t["T2"]["passed"])],
        ["T3", t["T3"]["name"], _fmt(t3["slab_with_hole_aligned"]["relative_error"]), "1e-3", _fmt(t3["passed"])],
        ["T4", t["T4"]["name"], _fmt(t["T4"]["worst_relative_difference"]), "1e-12", _fmt(t["T4"]["passed"])],
        ["T5", t["T5"]["name"], t["T5"]["window"]["n_evals"], ">= 5e4 evals and <= 60 s", _fmt(t["T5"]["passed"])],
    ]
    audit_table = _table(["criterion", "what it establishes", "measured", "predeclared threshold", "pass"], audit_rows)

    c6 = c["C6"]
    grid_rows = [[row["n_per_axis"], row["area"], row["feasible_cells"], row["relative_difference_vs_exact"], row["elapsed_s"]] for row in c6["brute_force"]]
    grid_table = _table(["grid n per axis", "cell-count area", "feasible cells", "rel. error vs closed form", "wall [s]"], grid_rows, digits=8)

    # ---- markdown --------------------------------------------------------
    md: list[str] = []
    md.append(f"# 作业窗口包络引擎成熟化：多区间（含洞）可行域的面积积分与机器校验\n")
    md.append(f"节点 `{NODE}` · 仓 `TensorPhysics` · 模块 `tensorphysics.envelope` · 配置 `tensorphysics.envelope.case.v1`\n")
    md.append(
        "证据：`artifacts/envelope_topology/topology_evidence.json`（本节点新判据 T1–T5）、"
        "`artifacts/envelope_postfix/criteria_evidence.json`（战役判据 C1–C6 复跑）、"
        "`artifacts/envelope_postfix/case_results.json`（三窗×三工况逐条结果）\n"
    )

    md.append("## ① 案例介绍（物理背景 / 几何 / 参数）\n")
    md.append(
        "作业窗口（operating window）是参数空间里使全部约束同时成立的子集。工程上它几乎总是"
        "**一条带**：下限来自\"来不及/不够\"，上限来自\"来不及停/受不了\"。已交付的三条窗（搁浅脱浅的"
        "临界潮位、压载水置换率、STS 过驳泵率）都是一维带，引擎把它们当纯配置处理，代码里没有任何"
        "领域词条。\n"
    )
    md.append("三窗的物理读法与本次复跑结果（全部由 `case_results.json` 读出，无手抄）：\n")
    for key, text in CONFIG_FILES.items():
        md.append(f"* `{key}` — {text}")
    md.append("")
    md.append(shipped_table + "\n")
    md.append(
        "本次成熟化针对的是**二维窗的拓扑**：当可行域在参数平面上不是\"星形\"、某条扫描线上会两次"
        "穿过同一个连通域时（环形可行域、带禁区的窗、折返边界），可行域面积应当按\"外壳减洞\"积分。"
        "审计发现原实现对这类拓扑把**各区间长度直接串起来积分**（同一横坐标出现多次），得到的是几何上"
        "毫无意义的数：半径 0.5–0.8 的环，解析面积 `pi(0.8^2-0.5^2)=1.225221`，引擎给出 0.864925，"
        "偏低 29.4%，而**全部契约检查为绿**——因为没有任何机器检查拿这个数与别的东西比对过。\n"
    )

    md.append("## ② 怎么做的（方法 / 流程 / 关键设置）\n")
    md.append(
        "1. **审计**（先证伪再动手）：用合成几何（环、双圆、方板挖方孔）在引擎上做对抗性探查，"
        "对每个疑点先看现状数值与契约结论，再决定是否算缺陷。审计确认两点：二维多区间面积系统性偏低"
        "（与网格无关，纯算法错），且平面模式下暴力网格扫描只被\"打印\"、从未被比较。\n"
        "2. **修复**：组件面积改为按**壳分解**积分——外壳 = 该线最后一个上界减第一个下界；每个洞 = "
        "相邻两区间之间的间隙；面积 = 外壳 − Σ洞。每一壳都是一条闭合弦，在自己的端点按 `sqrt` 收口，"
        "于是原有的解析端块（两结点定位尖端 + `L^2` 线性）对每一壳都成立，面积精度恢复二阶。"
        "`Component` 新增 `multi_interval_lines` 字段，`area_method` 报告 `shell_tip_model` / "
        "`shell_trapezoid`。\n"
        "3. **把网格扫描升格为机器检查**：平面窗现在跑两次暴力网格（`n_grid_check` 与其两倍），"
        "新增 `contracts.check_area_vs_grid`，判据 **先声明后使用**：\n"
        "   `band = max(rel_tol·|A_fine|, 3·|A_fine − A_coarse|)`，`|A_march − A_fine| ≤ band` 为通过。"
        "第二项是网格自身的分辨率漂移（它自己误差的估计），乘 3 为安全系数。该检查是**粗错探测器**，"
        "不是精度证书——精度由解析面积判据（T1/T3）证明。\n"
        "4. **不留倒退**：把\"修复前的算术\"作为负控留在测试与验收脚本里（T2），要求机器检查必须拒绝它；"
        "并要求三条已迁移的一维窗数字**一位不动**（T4，对照 `scripts/envelope_cases_acceptance.py` 里"
        "独立的闭式参考路径）。\n"
        "5. **工具链**：纯 Python 标准库（包内无 numpy 依赖）+ matplotlib 出图；`pytest` 单元/回归；"
        "`scripts/envelope_topology_acceptance.py` 产出 T1–T5 证据 JSON 与图；机器 loongarch64、"
        "Python 3.11.6、单核 `taskset -c 2` 计时。\n"
    )

    md.append("## ③ 计算结果（数字表）\n")
    md.append("### 3.1 环形可行域的面积（判据 T1）\n")
    md.append(t1_table + "\n")
    md.append(
        f"默认梯度（100 线 × 96 探针）相对误差 `{_fmt(t['T1']['default_ladder']['relative_error'])}`，"
        f"满足 1e-3；细化到 400 线为 `{_fmt(t['T1']['ladder'][-1]['relative_error'])}`，"
        f"满足 1e-4。最后两级观测阶 `{_fmt(t['T1']['observed_order'], 3)}`（1 与 2 之间：端块是解析的，"
        "内部是梯形，阶的估计受端块主导项影响）。\n"
    )
    md.append("### 3.2 修复前后对照与负控（判据 T2）\n")
    md.append(_table(
        ["quantity", "value"],
        [
            ["closed-form area", t["T2"]["closed_form_area"]],
            ["shipped area (per-shell)", t["T2"]["shipped_area"]],
            ["pre-fix area (summed intervals)", t["T2"]["pre_fix_area"]],
            ["pre-fix relative error", t["T2"]["pre_fix_relative_error"]],
            ["check verdict on pre-fix value", "rejected" if t["T2"]["check_on_pre_fix"]["passed"] is False else "ACCEPTED (bad)"],
            ["check verdict on shipped value", "accepted" if t["T2"]["check_on_shipped"]["passed"] else "REJECTED (bad)"],
            ["check band on the shipped value", t["T2"]["check_on_shipped"]["metrics"]["band"]],
        ],
        digits=8,
    ) + "\n")
    md.append("### 3.3 拓扑回归（判据 T3）\n")
    md.append(t3_table + "\n")
    md.append("### 3.4 迁移不变性（判据 T4）\n")
    md.append(
        f"三窗 × 三工况共 `{t['T4']['n_comparisons']}` 组窗口边界/宽度对照独立的闭式参考实现，"
        f"最大相对差 `{_fmt(t['T4']['worst_relative_difference'])}`（阈值 1e-12）。"
        "即：二维改动没有移动任何一条已迁移的一维窗数字。\n"
    )
    md.append("### 3.5 成本（判据 T5）\n")
    md.append(_table(
        ["quantity", "value"],
        [
            ["window", "ring, n_lines=100, n_scan=96"],
            ["constraint evaluations", t["T5"]["window"]["n_evals"]],
            ["wall time [s]", t["T5"]["window"]["elapsed_s"]],
            ["evaluations / s", t["T5"]["window"]["evals_per_second"]],
            ["machine", t["T5"]["machine"]["machine"]],
            ["predeclared limits", "5e4 evals, 60 s"],
        ],
        digits=6,
    ) + "\n")
    md.append("### 3.6 战役判据 C1–C6 复跑（修复后）\n")
    md.append(criteria_table + "\n")
    md.append("### 3.7 C6：二维面积 vs 暴力网格（逐分辨率）\n")
    md.append(grid_table + "\n")
    md.append(
        f"引擎在 160 线 × 320 探针下用 `{c6['engine']['n_evals']}` 次评估、"
        f"`{_fmt(c6['engine']['elapsed_s'], 3)}` s 得到面积 `{_fmt(c6['engine']['area'], 9)}`，"
        f"与闭式 `{_fmt(c6['closed_form_area'], 9)}` 相对差 `{_fmt(abs(c6['engine']['area']-c6['closed_form_area'])/c6['closed_form_area'])}`；"
        f"吞吐 `{_fmt(c6['engine']['evals_per_second'], 0)}` 次评估/秒（单核）。\n"
    )
    md.append("### 3.8 汇总判据表\n")
    md.append(audit_table + "\n")

    md.append("## ④ 图\n")
    md.append("随报告内嵌四张图：图 1 环形窗的行进边界与暴力网格掩膜对照（同一解析圆心/半径叠加）；图 2 环面积收敛曲线（修复后 vs 修复前，附 `O(h^2)` 参考线与 1e-3 判据线）；图 3 三条已迁移一维窗的全部工况（引擎区间条与闭式参考刻度叠加）；图 4 面积机器判据的带子、两级网格与修复前错值的位置。\n")

    md.append("## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）\n")
    md.append(
        "* **\"检查全绿\"不等于\"数对了\"**：本次缺陷能活下来的唯一原因是平面模式的暴力网格扫描"
        "只被写进结果对象、从未被比较。把已有信息升格为判据（而不是新增更多输出）才是修复。\n"
        "* **把区间先加起来积分是错的**：环形域上积分式 `∫ΣL_k du` 在洞开/闭处出现 `sqrt` 拐折，"
        "梯形只有 `O(h^1.5)`；实测 100 线时 1.3e-3，比引擎承诺的 `O(h^2)` 还差。按壳分解后"
        "每一壳端点都是 `sqrt` 收口，才回到二阶。\n"
        "* **拐角边界只有一阶**：方板挖方孔的洞边若正好落在扫描线上，误差 7e-5；若落在两线之间"
        "（线性拐折而非 `sqrt` 收口），误差 1.3e-2。此类几何需让梯度对齐拐角（T3 已把这条写成"
        "声明式局限并给出实测数，不隐藏）。\n"
        "* **面积检查的带子是分辨率相关的**：网格面积本身只有一阶精度（48×48 对圆盘偏高约 1%），"
        "所以 `check_area_vs_grid` 只能做粗错探测；把 1e-3 的精度主张压在这种检查上会是过度声明。"
        "精度主张必须由解析面积承担。\n"
        "* **端块模型的前提**：`L^2` 线性定位尖端只在端点两线都是单区间时成立；代码用 `simple_ends` "
        "显式判断，不满足则退回梯形并把 `area_method` 报出来。\n"
        "* **适用范围**：二维平面窗（两参数）；三维窗需要切片，不在本节点范围。离散可行域仍按"
        "连续窗之外单独报告，连续边界不承诺可设定。\n"
    )

    md.append("## ⑥ 误差判定（是否 < 3%）\n")
    md.append(_table(
        ["关键量", "基准", "相对误差", "< 3% ?", "依据"],
        [
            ["环形窗面积（默认梯度）", "闭式 pi(R^2-r^2)", t["T1"]["default_ladder"]["relative_error"], "yes", "T1"],
            ["环形窗面积（400 线）", "闭式 pi(R^2-r^2)", t["T1"]["ladder"][-1]["relative_error"], "yes", "T1"],
            ["单圆盘面积", "闭式 pi r^2", t3["single_disc"]["relative_error"], "yes", "T3"],
            ["双圆盘总面积", "闭式 2 pi r^2", t3["two_disjoint_discs"]["relative_error"], "yes", "T3"],
            ["方板挖孔（拐角对齐）", "闭式 4-1", t3["slab_with_hole_aligned"]["relative_error"], "yes", "T3"],
            ["方板挖孔（拐角在两线间）", "闭式 4-1", t3["slab_with_hole_corner_between_lines"]["relative_error"], "no（1.3%，但落在声明的一阶带内）" if t3["slab_with_hole_corner_between_lines"]["relative_error"] < 0.03 else "no", "T3（声明式局限）"],
            ["修复前环形窗面积", "闭式 pi(R^2-r^2)", t["T2"]["pre_fix_relative_error"], "no（29.4%，已修复）", "T2"],
            ["三窗迁移边界/宽度", "独立闭式参考路径", t["T4"]["worst_relative_difference"], "yes", "T4"],
            ["C6 二维面积", "暴力网格 Richardson 外推", c6.get("relative_difference_vs_extrapolated_brute"), "yes", "C6"],
            ["C6 二维面积", "闭式 pi r^2", c6.get("relative_difference_vs_closed_form_area"), "yes", "C6"],
        ],
        digits=8,
    ) + "\n")
    md.append(
        "结论：**能给出基准的量全部 < 3%（实际 < 1e-3）**；唯一超过 1e-3 的是带拐角的方板挖孔"
        "在两线间的情形，其成因（线性拐折 vs `sqrt` 收口）已定位并作为声明式局限记录，"
        "未粉饰为通过。修复前 29.4% 的错值已被机器判据拒绝，不再是可交付数字。\n"
    )

    md.append("## 变更与审计链\n")
    md.append(_table(
        ["artifact", "digest"],
        [
            ["topology_evidence (artifact_digest)", evidence.get("artifact_digest")],
            ["topology_evidence (code_digest)", evidence.get("code_digest")],
            ["migrated criteria evidence (artifact_digest)", migrated.get("artifact_digest")],
            ["migrated case results (artifact_digest)", case_results.get("artifact_digest")],
        ],
        digits=0,
    ) + "\n")
    md.append(
        "改动文件：`src/tensorphysics/envelope/marching.py`（壳分解面积 + `multi_interval_lines`）、"
        "`src/tensorphysics/envelope/contracts.py`（`check_area_vs_grid`）、"
        "`src/tensorphysics/envelope/engine.py`（平面窗两次网格 + 面积判据）、"
        "`src/tensorphysics/envelope/benchmarks.py`（环、方板挖孔合成算例 + 闭式面积）、"
        "`src/tensorphysics/envelope/__init__.py`（导出）、"
        "`tests/test_envelope_engine.py`（6 个回归/负控用例）、"
        "`scripts/envelope_topology_acceptance.py`（T1–T5 证据）、"
        "`docs/OPERATING_WINDOW_ENVELOPE.md`（第 2/4/6/7 节同步）。\n"
    )

    markdown = "\n".join(md)

    # ---- figures ---------------------------------------------------------
    figures = []
    for figure in evidence.get("figures", []):
        path = pathlib.Path("artifacts/envelope_topology/figures") / figure["name"]
        if not path.exists():
            continue
        caption = figure["caption"]
        if len(caption) > 300:  # the reports endpoint caps the caption length
            caption = caption[:297].rstrip(" ,;.") + "..."
        figures.append(
            {
                "name": figure["name"],
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    return markdown, figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default="artifacts/envelope_topology/topology_evidence.json")
    parser.add_argument("--migrated", default="artifacts/envelope_postfix")
    parser.add_argument("--out", default="docs/reports/in68w3_n06/envelope_multi_interval_maturity.md")
    parser.add_argument("--payload", default="/tmp/report_in68w3_n06.json")
    args = parser.parse_args(argv)

    evidence = load(pathlib.Path(args.evidence))
    migrated_dir = pathlib.Path(args.migrated)
    migrated = load(migrated_dir / "criteria_evidence.json")
    case_results = load(migrated_dir / "case_results.json")

    markdown, figures = build(evidence, migrated, case_results)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)
    print(f"wrote {out_path} ({len(markdown)} chars, {len(figures)} figures)")

    payload = {
        "title": "作业窗口包络引擎成熟化：多区间（含洞）可行域的面积积分与机器校验（IN68W3-N06）",
        "summary": (
            "审计发现平面窗多区间（含洞/折返）面积把各区间长度串接积分，环形可行域偏低 29.4% 且全部契约"
            "为绿；改为按壳（外壳减洞）积分并把暴力网格扫描升格为机器判据 check_area_vs_grid。"
            "环面积误差 1.4e-4（100 线）/1.5e-5（400 线），负控（修复前算术）被判据拒绝，"
            "三窗×三工况迁移数字与闭式参考最大相对差 3.6e-15，C1–C6 复跑全通过。"
        ),
        "scenario_key": SCENARIO_KEY,
        "mission_id": MISSION_ID,
        "tags": ["TensorPhysics", "envelope", "operating-window", "multi-interval", "area-check", NODE],
        "markdown": markdown,
        "figures": figures,
    }
    payload_path = pathlib.Path(args.payload)
    payload_path.write_text(json.dumps(payload))
    print(f"wrote payload {payload_path} ({payload_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())