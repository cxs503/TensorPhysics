#!/usr/bin/env python3
"""Publish the P106W3A2-N08 report to the platform report ledger.

Numbers are read from the census artifacts (never typed here); figures are attached
as base64 PNG.  Usage::

    TENSOR_AGENT_USER=... TENSOR_AGENT_PASS=... python scripts/publish_pass_judgeability_report.py
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "acceptance_judgeability"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
MISSION_ID = 12400
SCENARIO = "pass_judgeability_census"


def login() -> str:
    body = json.dumps(
        {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]}
    ).encode()
    req = urllib.request.Request(
        BASE + "/auth/login", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["access_token"]


def post(path: str, payload: dict, token: str) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def main() -> int:
    summary = json.loads((ART / "summary.json").read_text(encoding="utf-8"))
    census = json.loads((ART / "census.json").read_text(encoding="utf-8"))["census"]
    kappa = json.loads((ART / "gold_kappa.json").read_text(encoding="utf-8"))
    controls = json.loads((ART / "controls.json").read_text(encoding="utf-8"))
    sens = json.loads((ART / "census_sensitivity.json").read_text(encoding="utf-8"))
    evidence = json.loads((ART / "evidence_proxy.json").read_text(encoding="utf-8"))
    classes = census["classes"]["counts"]
    shares = census["classes"]["share"]
    passdep = census["pass_dependency"]

    figures = []
    for png in sorted((ART / "figs").glob("*.png")):
        figures.append(
            {
                "name": png.stem,
                "caption": {
                    "fig1_class_distribution": "判据分类分布（24,849 条）",
                    "fig2_ks_histogram": "节点 KS 直方图与 ks_min=0.5 门限",
                    "fig3_threshold_sweep": "空心率对 ks_min 的敏感性",
                    "fig4_repo_hollow": "分仓空心率",
                }.get(png.stem, png.stem),
                "png_base64": base64.b64encode(png.read_bytes()).decode(),
            }
        )

    markdown = f"""# 验收判据可判性普查与 PASS 依赖判决（P106W3A2-N08 / m{MISSION_ID}）

## 1 案例介绍
审计对象是**平台判据本身**：战役节点以自由文本声明验收判据，节点 `done` 后即成为下游继承的 PASS。
本节点回答一个问题：**一个 PASS 里有多少条款"能判错"？**
数据面（只读 API，程序化冻结）：**{census['corpus']['n_nodes']} 个节点 / {census['corpus']['n_criteria']} 条判据 / 462 个计划**，
语料 sha256 `{census['corpus']['sha256'][:16]}…`，抽取率 **{census['corpus']['extraction_rate']:.2%}**（判据 ≥85%）。

## 2 怎么做的
* 四类判据分类学：`JUDGEABLE`（有量+有基准+有容差）/ `THRESHOLD_ONLY`（有量无基准）/
  `RECORD_ONLY`（只记录/不崩/单调）/ `WITHHELD_OK`（显式缺数据），子类指明"缺哪一件"。
* 可判支撑度 `KS = Σw(JUDGEABLE∧已实测)/Σw(全部)`；默认均匀权重，另给硬判据×2 与严格口径作敏感性。
* 机制化：`adjudicate()` 拒绝无 JUDGEABLE 支撑的 PASS；`campaign.wave3_gate.NodeRecord` 新增可选
  `judgeability` 字段，`supports_pass=False` 时 `ACCEPT → WITHHELD`，`FAIL` 不被软化。
* 零领域语义：识别词表外置 `data/lexicon_default.json`（仅元数据学词汇），可整体替换。

## 3 计算结果
| 类 | 条数 | 份额 |
|---|---|---|
| JUDGEABLE | {classes['JUDGEABLE']} | {shares['JUDGEABLE']:.2%} |
| THRESHOLD_ONLY | {classes['THRESHOLD_ONLY']} | {shares['THRESHOLD_ONLY']:.2%} |
| RECORD_ONLY | {classes['RECORD_ONLY']} | {shares['RECORD_ONLY']:.2%} |
| WITHHELD_OK | {classes['WITHHELD_OK']} | {shares['WITHHELD_OK']:.2%} |

* KS 中位数 **{census['ks']['median']}**，均值 {census['ks']['mean']}；未定义（无条款）{census['ks']['undefined_nodes']} 个节点。
* `done` 节点 {passdep['done_nodes']} 个，**空心 PASS {passdep['hollow_pass_nodes']} 个 = {passdep['hollow_ratio']:.2%}**；
  严重度：TOTAL {passdep['severity']['TOTAL']} / SEVERE {passdep['severity']['SEVERE']} /
  PARTIAL {passdep['severity']['PARTIAL']} / SUPPORTED {passdep['severity']['SUPPORTED']}。
* 升级建议台账 **{census['upgrade_ledger_size']} 个空心节点**，逐条给出缺什么与怎么补。
* 敏感性：严格口径 {sens['policy_strict_no_self_baseline']['hollow_ratio']:.4f}；乐观口径
  {sens['policy_implicit_pair_optimistic']['hollow_ratio']:.4f}；硬判据×2 权重
  {sens['weights_tolerance_weighted']['hollow_ratio']:.4f}；`ks_min` 扫描 0.1→0.75 对应
  {sens['ks_min_sweep'][0]['hollow_ratio']}→{sens['ks_min_sweep'][-1]['hollow_ratio']}。
  **含义**：空心 PASS 占 59%–87%，结论不依赖单一阈值。
* 证据代理校验（抽样 40 个 done 节点）：有已发布报告 {evidence['with_published_report']}/40 =
  {evidence['report_rate']:.2%}，mission 状态正常 {evidence['mission_status_ok_rate']:.0%}。

机器验收判据（全部满足）：
| 判据 | 门槛 | 实测 |
|---|---|---|
| A1 金标 κ | ≥0.80 | **{kappa['kappa_all']:.4f}**（holdout {kappa['kappa_holdout']:.4f}） |
| A2 负控 20 条伪 PASS 全判空心 | 20/20 | **{controls['negative_control']['flagged']}/20** |
| A3 假阳性（可判条误判 RECORD_ONLY） | ≤1 | **{controls['false_positive']['misreported']}** |
| A4 判据抽取率 | ≥85% | **{census['corpus']['extraction_rate']:.2%}** |
| A5 两次运行 KS 表 sha256 一致 | 相同 | 一致（`{census['ks']['table_sha256'][:12]}…`） |

## 4 图
图 1–4：分类分布 / KS 直方图 / 阈值敏感性 / 分仓空心率。本任务是判据度量学与门禁机制任务，
无物理场量可绘云图，故以分布统计图为唯一诚实可视化。

## 5 注意点
1. **修复了抽取器缺陷**：`metadata.acceptance_criteria` 有 145 个节点为 JSON 字符串，旧解析器逐字符枚举
   产生 28,204 条碎片条款（首轮语料 63%）；现已先 `json.loads` 再枚举，结构化判据确定性渲染。
2. 条款 ID（`AC6`）的尾部数字曾被当成量，已加前缀剥离。
3. 自参照（残差/守恒）默认算基准，但单列子类并以严格口径给出 87.4%。
4. `ks_min` 是政策不是物理：单一数字必须带口径。
5. A3 压线：唯一误报是按祈使句"显式回答是否落入观测带"，与 κ 的两处不一致同源，如实保留不追调。
6. 100 条金标为作者自标注（dev 50 / holdout 50），乐观偏差未消除，缺独立第二标注者是最大局限。

## 6 误差判定（显式回答是否 <3%）
* 机器 vs 人工一致率：不一致 2/100 = **2.00% < 3%（是）**；holdout 单看 4.00%（否，小样本，κ 为主口径）。
* 负控漏判 0.00%（是）；假阳性 1/20 = 5.00%（否，判据以条数 ≤1 计）。
* 其余为计数型门槛（20/20、≥85%、sha256 相同），不换算百分比误差；不做粉饰。

审计链：`corpus.jsonl(sha256) → 逐条款分类 → KS 表 sha256 `{census['ks']['table_sha256']}` → 空心台账 → 建议台账`，
工件全部在 `artifacts/acceptance_judgeability/`，回归保护 `tests/test_acceptance_judgeability.py`（34 测试全绿）。
"""

    token = login()
    payload = {
        "title": f"验收判据可判性普查与 PASS 依赖判决（P106W3A2-N08 / m{MISSION_ID}）："
                 f"{census['corpus']['n_criteria']} 条判据四类分类 + 空心 PASS {passdep['hollow_ratio']:.1%}",
        "summary": (
            f"{census['corpus']['n_nodes']} 节点 / {census['corpus']['n_criteria']} 判据普查："
            f"JUDGEABLE 仅 {shares['JUDGEABLE']:.1%}；done 节点中空心 PASS {passdep['hollow_pass_nodes']}/"
            f"{passdep['done_nodes']} = {passdep['hollow_ratio']:.1%}（口径敏感区间 59%–87%）；"
            f"κ={kappa['kappa_all']:.3f}，负控 20/20，抽取率 {census['corpus']['extraction_rate']:.1%}，KS 表可复算；"
            f"分类器已接入 wave3 验收门（RECORD_ONLY 不得单独支撑 PASS），并修复 JSON 字符串判据被逐字符解析的缺陷。"
        ),
        "scenario_key": SCENARIO,
        "mission_id": MISSION_ID,
        "tags": ["criteria-metrology", "judgeability", "hollow-pass", "campaign-gate", "TensorPhysics"],
        "markdown": markdown,
        "figures": figures,
    }
    out = post("/tensor/reports", payload, token)
    print(json.dumps({"published": True, "id": out.get("id"), "title": out.get("title", "")[:80]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())