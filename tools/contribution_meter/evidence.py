#!/usr/bin/env python3
"""Deterministic evidence-field extraction from mission reports.

The contribution meter is *evidence-first*: every classification must be traceable to a
machine-extracted field of the report text (error-table rows against an external reference,
acceptance-gate PASS counts, evidence-level transitions, correction patterns, failure verdicts,
verification-file names, ...).  The LLM (if ever used) is only a fallback that receives these
fields, never the raw text instead of them.

Everything here is pure regex/table logic over UTF-8 text: reproducible bit-for-bit on the same
corpus (no randomness, no model calls, no network).

Key extraction detail: Chinese mission reports put the numbers in markdown tables whose *header*
carries the word 误差/error.  A line-only extractor therefore misses most error rows, so tables are
parsed with their header and a local context window (heading + neighbouring prose) to decide which
kind of reference the error is measured against.

Public API
    extract(report: dict) -> dict             # evidence field bundle
    summarise_fields(ev: dict) -> list[str]   # compact "field=value" trace for audit

CLI
    python3 tools/contribution_meter/evidence.py --corpus data/corpus_reports.jsonl --stats
    python3 tools/contribution_meter/evidence.py --corpus data/corpus_reports.jsonl --report-id 5464
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys

# --------------------------------------------------------------------------------------
# vocabularies
# --------------------------------------------------------------------------------------
REFERENCE_PATTERNS = {
    "analytic": [
        r"解析解", r"解析式", r"解析锚", r"analytic", r"analytical", r"closed[- ]form", r"闭式解",
        r"精确解", r"exact solution", r"manufactured solution", r"\bMMS\b", r"级数解", r"渐近解",
        r"Boussinesq", r"Lamb", r"Johnson", r"Inglis", r"Lame", r"Lamé", r"Stokes", r"Michell",
        r"Rayleigh", r"Leppington", r"Maidanik", r"Ekman", r"Blasius", r"Rankine-Hugoniot",
    ],
    "experiment": [
        r"实验值", r"实验数据", r"实验测量", r"试验值", r"试验数据", r"experiment",
        r"\bEFD\b", r"measured", r"实测", r"水槽试验", r"风洞", r"拖曳水池", r"试航",
    ],
    "cross_discretization": [
        r"异离散", r"跨离散", r"两套离散", r"不同离散", r"cross[- ]solver", r"cross[- ]discreti",
        r"独立实现", r"双实现", r"网格收敛", r"grid convergence", r"GCI", r"Richardson",
        r"跨求解器", r"两套求解器", r"三口径", r"互裁", r"互校",
    ],
    "literature": [
        r"文献", r"literature", r"benchmark", r"基准算例", r"经典算例", r"标准算例",
        r"Ghia", r"Schäfer", r"Turek", r"lid[- ]driven", r"published", r"对标",
        r"DIN\s*\d+", r"ITTC", r"ISO\s*\d+", r"DNV", r"IMO\s+MSC", r"MEPC", r"MKM",
        r"Longo", r"Liu\s*&\s*Huang", r"Chuang", r"Kijima", r"Lindqvist", r"Valanto",
    ],
    "internal_baseline": [
        r"基线", r"baseline", r"上一版", r"旧实现", r"回归对比", r"对照实现", r"v0",
    ],
}

#: how many prose lines above a table may contribute to the reference kind of its error rows.
#: This single hyper-parameter was chosen on the gold set (see docs/contribution_meter_report.md);
#: 0 would ignore captions, large values bleed unrelated literature mentions into ledger tables.
REF_CONTEXT_LINES = 3
ERROR_HEADER = re.compile(r"误差|error|偏差|deviation|相对差", re.IGNORECASE)
ERROR_PATTERNS = [
    r"误差\s*(?:为|是|=|:|：)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:的)?\s*(?:相对)?误差",
    r"相对误差[^\n]{0,12}?([0-9]+(?:\.[0-9]+)?)\s*%",
    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:以内|以下)",
    r"error[^\n]{0,20}?([0-9]+(?:\.[0-9]+)?)\s*%",
    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:relative\s+)?error",
    r"deviation[^\n]{0,20}?([0-9]+(?:\.[0-9]+)?)\s*%",
]
ERROR_SCI = [
    r"误差[^\n]{0,16}?([0-9]+(?:\.[0-9]+)?[eE][-+]?[0-9]+)",
    r"([0-9]+(?:\.[0-9]+)?[eE][-+]?[0-9]+)[^\n]{0,8}?(?:相对误差|relative error)",
]
PERCENT_CELL = re.compile(r"^\s*\|?\s*(?:[^\d|]{0,40}?)\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*\|?\s*$")
NUMBER_CELL = re.compile(r"([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)")

GATE_PASS = [r"\bPASS(?:ED)?\b", r"✅", r"通过", r"全绿", r"绿\b", r"passed"]
GATE_FAIL = [r"\bFAIL(?:ED)?\b", r"❌", r"未通过", r"PENDING", r"WITHHELD", r"红灯"]
ACCEPTANCE_HEADER = re.compile(r"验收|判据|接受准则|acceptance|criteria|判决|裁决|verdict|结论", re.IGNORECASE)
MD_SECTION = re.compile(r"^#{1,4}\s*(.+)$", re.MULTILINE)

#: evidence-level / capability transition (C2)
LEVEL_UP = [
    r"NOT_SUPPORTED\s*(?:→|->|=>|—>)\s*REAL",
    r"WITHHELD\s*(?:→|->|=>|—>)\s*(?:PROVISIONAL|REAL)",
    r"MOCK\s*(?:→|->|=>|—>)\s*REAL",
    r"STUB\s*(?:→|->|=>|—>)\s*REAL",
    r"(?:MOCK|STUB|NOT_SUPPORTED|WITHHELD)\s*(?:变为|成为|升级为|转正为|改判为)\s*(?:REAL|PROVISIONAL)",
    r"(?:降级|降阶|占位|桩|假算法|查表|经验表)[^\n]{0,20}(?:→|->|换成|替换为|改为)[^\n]{0,16}"
    r"(?:精确|一阶原理|REAL|真实)",
    r"(?:文档级|纸面)[^\n]{0,12}(?:→|->|落到|落地|实现)",
    r"(?:MOCK|STUB)\s*(?:升级|提升|转|改)\s*(?:为)?\s*REAL",
    r"(?:REAL|真算法)\s*(?:替换|取代)\s*(?:MOCK|STUB|桩|假)",
    r"(?:REAL|真实现|一阶原理)\s*化",
]

#: a *stated correction* of a known-wrong number (C3).
#: Pattern family A: an explicitly released/old value is replaced by a measured one.
#: Pattern family B: a defect word (勘误/错值/写错/...) must sit within 80 chars of a repair verb,
#: otherwise it is just prose about a mutation negative control and NOT a correction of the stock.
CORRECTION_OLD_NEW = [
    r"(?:旧链|旧实现|旧值|原实现|原值)[^\n]{0,200}?(?:高估|低估|错|偏差)[^\n]{0,30}?[0-9]",
    r"从[^\n]{0,16}?(?:改写为|升级为|改为|换成|转成)",
    r"(?:旧链|旧实现|旧值|原实现|原值|此前|上一版|历史记录|UNGUARDED|降级实现)[^\n]{0,60}?"
    r"(?:→|->|改为|修正为|换成|实为|实测为)[^\n]{0,40}?[0-9]",
    r"(?:高估|低估)[^\n]{0,40}?(?:→|->)[^\n]{0,30}?[0-9]+(?:\.[0-9]+)?\s*%",
    r"UNGUARDED\s*(?:→|->)\s*PINNED",
    r"errata|勘误",
]
CORRECTION_DEFECT_WORD = re.compile(
    r"错值|写错|符号错|算错|差(?:了)?一(?:个)?(?:量级|倍数)|off-by|错误数值|代数纠正|勘误|errata|"
    r"从未达到|达不到|假设[^\n]{0,12}(?:→|->|改为|实为)|从[^\n]{0,14}?(?:升级为|改为|换成)"
    , re.IGNORECASE
)
REPAIR_VERB = re.compile(
    r"修复|修正|纠正|改正|改为|更正|锁死|锁住|缺陷|根因|换掉|替换|重新算|重算|解红|退役|升级为|改写为",
    re.IGNORECASE
)
OLD_ANCHOR = re.compile(r"旧(?:链|实现|值)|原(?:实现|值)|此前|上一版|历史记录|UNGUARDED|降级实现", re.IGNORECASE)

DEFECT_MARKERS = [
    r"修复", r"\bfix(?:ed)?\b", r"\bbug\b", r"缺陷", r"错误数值", r"回归测试", r"regression",
    r"错值", r"算错", r"写错", r"符号错", r"差(?:了)?一(?:个)?(?:倍数|量级)", r"off[- ]by",
    r"根因", r"勘误", r"errata",
]

#: explicit *verdict* of failure applied to a scientific claim (C5).  Bare FAIL/不满足 tokens are
#: deliberately excluded: they occur in nearly every report (CI reds, minor honest FAILs).
FAILURE_VERDICT = [
    r"(?:判定|裁决|判决|结论|终判|verdict|verdicts?)[^\n]{0,16}?(?:FAIL|失败|WITHHELD)",
    r"honest[-_ ]?FAIL",
    r"FAIL/WITHHELD",
    r"(?:判据|准则|criterion|口径)[^\n]{0,16}?(?:被证)?(?:ill-posed|证伪|不成立)",
    r"(?:退役|撤销|撤回)[^\n]{0,14}(?:判据|结论|案例|该|此)",
    r"不可行", r"不成立", r"证伪",
    r"超出[^\n]{0,6}(?:适用|有效|方法)域",
]

NEGATIVE_MARKERS = [
    r"不可行", r"不成立", r"无效", r"反例", r"negative[- ]result", r"否定", r"证伪",
    r"没有(?:改进|收益|提升)", r"无(?:改进|收益|提升)", r"收益为负", r"不优于", r"劣于",
    r"失败结论", r"否证", r"不可操作", r"not supported", r"does not hold", r"falsif",
    r"高估", r"低估", r"根因", r"局限", r"WITHHELD",
]

#: deliverable shape that makes the *method/document* the unit (C4)
DOC_SHAPE = [
    r"教程", r"tutorial", r"讲解", r"指南", r"handbook", r"\bguide\b", r"白皮书",
    r"\bpaper\b", r"manuscript", r"成稿", r"论文稿", r"提案", r"proposal",
    r"评审", r"peer[- ]review", r"本体", r"ontology", r"分类器", r"classifier",
    r"方法论", r"方法学", r"协议", r"protocol", r"判例", r"路线图", r"roadmap",
    r"复盘", r"调查", r"质询", r"评述", r"综述", r"敏感性", r"口径", r"阈值", r"草案",
]
#: title-level shape decides precedence: a proposal/tutorial/method paper is a C4 unit even if it
#: quotes error numbers it inherited from elsewhere.
#: title-level *claim type*: does the mission headline a verification or a capability build?
VERIFY_TITLE = [
    r"验证", r"对拍", r"benchmark", r"解析", r"对标", r"误差", r"收敛阶", r"锚点", r"analytic",
    r"verification", r"精确解", r"网格收敛", r"GCI", r"复算", r"勘误",
]
CAPABILITY_TITLE = [
    r"新建", r"新增", r"链", r"引擎", r"模块", r"自动化", r"闭环", r"成熟化", r"升账", r"落地",
    r"求解点", r"实现", r"REAL\s*化", r"补齐", r"集成",
]
SHAPE_TITLE = [
    r"教程", r"tutorial", r"提案", r"proposal", r"本体", r"ontology", r"分类器", r"classifier",
    r"稿\b", r"\bpaper\b", r"manuscript", r"草案", r"敏感性", r"口径", r"复盘", r"质询",
    r"评审", r"peer[- ]review", r"判例", r"协议", r"protocol", r"指南", r"roadmap", r"路线图",
]

#: physics vocabulary: the ontology requires a *physics payload* for C1/C2, i.e. the numbers
#: must be about a physical field/problem, not about ledgers, dashboards or reports.
PHYS_TERMS = [
    "阻力", "推进", "螺旋桨", "耐波", "操纵", "稳性", "结构", "振动", "噪声", "冰载荷", "冰阻力",
    "压力", "速度", "温度", "应力", "应变", "流量", "湍流", "粘性", "黏性", "塑性", "弹性",
    "接触", "辐射", "波浪", "波高", "频率", "展弦", "载荷", "屈曲", "疲劳", "裂纹", "腐蚀",
    "传热", "多相", "液滴", "颗粒", "碰撞", "浅水", "自由面", "格林函数", "面元", "格子",
    "网格", "收敛阶", "守恒", "质量守恒", "能量", "动量", "静水", "排水量", "吃水", "浮力",
]
INFRA_MARKERS = [
    r"调度", r"daemon", r"dispatcher", r"harness", r"派发", r"沙箱", r"权限", r"审计链",
    r"\bCI\b", r"pytest", r"worktree", r"守护", r"看门狗", r"熔断", r"重试", r"幂等",
    r"报告闸门", r"门禁", r"任务机", r"会话", r"token", r"限流", r"契约", r"schema",
    r"账本", r"ledger", r"看板", r"dashboard", r"归仓", r"布局门",
]

UNIT_RE = re.compile(
    r"[0-9]+(?:\.[0-9]+)?\s*(?:Pa|kPa|MPa|GPa|N·m|N|kN|mN|kg|t\b|m/s|km/h|kn|m3/s|m\^3/s|"
    r"K\b|°C|Hz|kHz|dB|rad/s|deg|°|m\^2|m2|mm|cm|km|s\b|min|W\b|kW|MW|J\b|kJ|"
    r"N·s/m|kg/m\^3|rpm)"
)
FILE_RE = re.compile(r"(?:[\w./-]*/)?[\w.-]+\.(?:py|cpp|h|hpp|f90|jl|json|yaml|yml|md|csv|npz|png|sh|toml)")
VERIF_NAME_RE = re.compile(
    r"(?:verified|verification|validate|validation|golden|reference|baseline|analytic|"
    r"manufactured|benchmark|对拍|算例)[\w./-]*", re.IGNORECASE
)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def open_text(path: str):
    """Open a corpus file, transparently handling ``.gz`` (the repository keeps the harvested
    corpus compressed; the harvester writes plain jsonl)."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def resolve_corpus(path: str) -> str:
    """Return the first existing variant of ``path`` (``.jsonl`` or ``.jsonl.gz``)."""
    if os.path.exists(path):
        return path
    if os.path.exists(path + ".gz"):
        return path + ".gz"
    return path


def _count(patterns: list[str], text: str) -> int:
    return sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)


def _flat(patterns: list[str], text: str) -> list[str]:
    out: list[str] = []
    for p in patterns:
        try:
            out.extend(m.group(0) for m in re.finditer(p, text, re.IGNORECASE))
        except re.error:  # pragma: no cover
            continue
    return out


def _classify_reference(text: str) -> list[str]:
    kinds = []
    for kind, patterns in REFERENCE_PATTERNS.items():
        if _flat(patterns, text):
            kinds.append(kind)
    return kinds


def _sci_to_pct(value: float) -> float:
    return value * 100.0


def _section_index(lines: list[str]) -> list[str]:
    """Heading that applies to each line (crude but deterministic)."""
    heads, current = [], ""
    for line in lines:
        m = re.match(r"^#{1,4}\s*(.+)$", line.strip())
        if m:
            current = m.group(1).strip()
        heads.append(current)
    return heads


def _parse_tables(lines: list[str], cap: int = 60, headline: str = "") -> list[dict]:
    """Extract error rows from markdown tables.

    A table qualifies when its header row mentions an error column.  For every data row the
    numeric value in the first such column is taken (percent if the cell ends with '%',
    otherwise as a raw numeric relative error).  Each row inherits a *context* string
    (heading + up to 5 lines before the table + header + neighbouring prose) which is what
    decides the reference kind.
    """
    heads = _section_index(lines)
    rows: list[dict] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line.startswith("|"):
            i += 1
            continue
        block = []
        j = i
        while j < n and lines[j].strip().startswith("|"):
            block.append(j)
            j += 1
        header = lines[block[0]].strip()
        if ERROR_HEADER.search(header):
            err_cols = [k for k, cell in enumerate(header.strip("|").split("|")) if ERROR_HEADER.search(cell)]
            ctx_lo = max(0, block[0] - REF_CONTEXT_LINES)
            context = "\n".join(
                [heads[block[0]]]
                + [lines[k].strip() for k in range(ctx_lo, block[0])]
                + [header]
                + [lines[j].strip() if j < n else ""]
            )
            ref_kinds = _classify_reference(context)
            if not ref_kinds:  # fall back to the report headline (title+summary), never to far prose
                ref_kinds = _classify_reference(headline)
            for k in block[1:]:
                cells = [c.strip() for c in lines[k].strip().strip("|").split("|")]
                if len(cells) < 2:
                    continue
                for col in err_cols:
                    if col >= len(cells):
                        continue
                    cell = cells[col]
                    value, src = None, None
                    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", cell)
                    if m:
                        value, src = float(m.group(1)), "table_pct"
                    else:
                        m = re.fullmatch(r"\s*(?:≈|~)?\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*", cell)
                        if m:
                            try:
                                raw = float(m.group(1))
                            except ValueError:
                                continue
                            # unitless cells are relative errors: keep as percent only if tiny
                            if abs(raw) < 1e-1:
                                value, src = _sci_to_pct(raw), "table_frac"
                            else:
                                value, src = raw, "table_num"
                    if value is not None and math.isfinite(value) and value > 0.0:
                        rows.append({"pct": value, "source": src, "ref_kinds": ref_kinds,
                                     "line": (" | ".join(cells))[:160], "header": header[:120]})
                if len(rows) >= cap:
                    return rows[:cap]
        i = j
    return rows[:cap]


def extract_error_rows(text: str, cap: int = 60, headline: str = "") -> list[dict]:
    lines = text.splitlines()
    rows = _parse_tables(lines, cap, headline)
    if len(rows) >= cap:
        return rows[:cap]
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) < 6 or stripped.startswith("|"):
            continue
        has_err_word = bool(ERROR_HEADER.search(stripped))
        values: list[tuple[float, str]] = []
        for pat in ERROR_PATTERNS:
            for m in re.finditer(pat, stripped, re.IGNORECASE):
                values.append((float(m.group(1)), "pct"))
        for pat in ERROR_SCI:
            for m in re.finditer(pat, stripped, re.IGNORECASE):
                try:
                    values.append((_sci_to_pct(float(m.group(1))), "sci"))
                except ValueError:  # pragma: no cover
                    pass
        if not values:
            continue
        context = "\n".join(lines[max(0, idx - 5): idx + 2])
        kinds = _classify_reference(context)
        if not has_err_word and not kinds:
            continue
        for value, unit in values:
            if math.isfinite(value) and value > 0.0:
                rows.append({"pct": value, "source": unit, "ref_kinds": kinds,
                             "line": stripped[:160], "header": ""})
        if len(rows) >= cap:
            break
    return rows[:cap]


def _corrections(md: str, summary: str) -> list[str]:
    """Stated corrections of known-wrong values (C3 evidence), with the snippets kept."""
    out = _flat(CORRECTION_OLD_NEW, md)
    for m in CORRECTION_DEFECT_WORD.finditer(md):
        window = md[max(0, m.start() - 200): m.end() + 200]
        if (REPAIR_VERB.search(window) or OLD_ANCHOR.search(window)) and re.search(r"[0-9]", window):
            out.append(m.group(0))
    return out


def _gate_block(md: str) -> tuple[str, int]:
    """Text of the acceptance/judgement sections (fallback: whole document)."""
    lines = md.splitlines()
    heads = _section_index(lines)
    keep = [ln for ln, h in zip(lines, heads) if ACCEPTANCE_HEADER.search(h or "")]
    if len(keep) >= 5:
        return "\n".join(keep), len(keep)
    return md, len(lines)


def extract(report: dict) -> dict:
    md = report.get("markdown") or ""
    title = report.get("title") or ""
    summary = report.get("summary") or ""
    text = "\n".join([title, summary, md])

    err_rows = extract_error_rows(md or summary, headline=title + "\n" + summary)
    ext_rows = [r for r in err_rows if any(k != "internal_baseline" for k in r["ref_kinds"])]
    externals: list[str] = []
    for row in err_rows:
        for k in row["ref_kinds"]:
            if k not in externals:
                externals.append(k)

    gate_blob, n_gate_lines = _gate_block(md or summary)
    n_pass = _count(GATE_PASS, gate_blob)
    n_fail = _count(GATE_FAIL, gate_blob)

    sections = [s.strip() for s in MD_SECTION.findall(md)]
    files = sorted(set(FILE_RE.findall(md)))[:40]

    ev = {
        "mission_id": report.get("mission_id"),
        "report_id": report.get("report_id"),
        "title": title,
        "summary": summary[:400],
        "scenario_key": report.get("scenario_key") or "",
        "markdown_len": len(md),
        # quantitative comparison evidence
        "n_error_rows": len(err_rows),
        "error_rows": err_rows[:10],
        "min_err_pct": min((r["pct"] for r in err_rows), default=None),
        "max_err_pct": max((r["pct"] for r in err_rows), default=None),
        "reference_kinds": externals,
        "n_external_ref_kinds": len([k for k in externals if k != "internal_baseline"]),
        "n_external_error_rows": len(ext_rows),
        "min_external_err_pct": min((r["pct"] for r in ext_rows), default=None),
        # gate evidence
        "gate": {"n_acceptance_lines": n_gate_lines, "n_pass": n_pass, "n_fail": n_fail},
        # capability delta
        "level_up": _flat(LEVEL_UP, md or summary)[:6],
        "n_level_up": len(_flat(LEVEL_UP, md or summary)),
        "mentions_mock": bool(re.search(r"MOCK", text)),
        "mentions_stub": bool(re.search(r"STUB", text)),
        "mentions_real": bool(re.search(r"REAL", text)),
        # defect fix
        "defect_markers": _flat(DEFECT_MARKERS, md or summary)[:8],
        "n_defect_markers": _count(DEFECT_MARKERS, md or summary),
        "correction_pattern": _corrections(md or summary, summary)[:6],
        "n_correction_pattern": len(_corrections(md or summary, summary)),
        "has_gate_lock": bool(re.search(r"(?:锁死|回归.{0,6}(?:测试|门禁)|回归锁|PINNED|pin(?:ned)?\b)", md or "")),
        # negative result
        "failure_verdicts": _flat(FAILURE_VERDICT, md or summary)[:8],
        "n_failure_verdicts": len(_flat(FAILURE_VERDICT, md or summary)),
        "negative_markers": _flat(NEGATIVE_MARKERS, md or summary)[:8],
        "n_negative_markers": _count(NEGATIVE_MARKERS, md or summary),
        # method / doc
        "doc_shape_tokens": _flat(DOC_SHAPE, title + " " + summary)[:6],
        "n_doc_shape_tokens": len(_flat(DOC_SHAPE, title + " " + summary)),
        "title_shape": _flat(SHAPE_TITLE, title)[:4],
        "n_title_shape": len(_flat(SHAPE_TITLE, title)),
        "title_verify": _flat(VERIFY_TITLE, title)[:4],
        "n_title_verify": len(_flat(VERIFY_TITLE, title)),
        "title_capability": _flat(CAPABILITY_TITLE, title)[:4],
        "n_title_capability": len(_flat(CAPABILITY_TITLE, title)),
        "capability_language": bool(re.search(
            r"新建|新增|首次|从零|落地|实现|REAL|补齐|替换|NOT_SUPPORTED|MOCK|STUB|WITHHELD", title + " " + summary)),
        "has_method_language": bool(
            re.search(r"(?:方法|算法|流程|判据|本体|分类器|公式|推导|文档|协议)[^\n]{0,8}(?:提出|定义|设计|推导|给出|落地)", md)
        ),
        "n_formulas": len(re.findall(r"\$\$|\$[^$\n]{2,60}\$|\\frac", md)),
        "n_sections": len(sections),
        "sections": sections[:20],
        # infrastructure
        "infra_markers": _flat(INFRA_MARKERS, md or summary)[:10],
        "n_infra_markers": _count(INFRA_MARKERS, md or summary),
        "n_pytest_mentions": len(re.findall(r"pytest|测试用例|test_", md)),
        "n_units": len(UNIT_RE.findall(md)),
        "physics_terms": [t for t in PHYS_TERMS if t in text][:20],
        "n_physics_terms": len([t for t in PHYS_TERMS if t in text]),
        "n_files": len(files),
        "files": files[:15],
        "verification_names": sorted(set(VERIF_NAME_RE.findall(md)))[:20],
        "n_verification_names": len(set(VERIF_NAME_RE.findall(md))),
    }
    return ev


def summarise_fields(ev: dict, limit: int = 14) -> list[str]:
    gate = ev.get("gate", {})
    items = [
        f"n_error_rows={ev['n_error_rows']} min_err_pct={ev['min_err_pct']}",
        f"n_external_rows={ev['n_external_error_rows']} min_external_err={ev['min_external_err_pct']}",
        f"reference_kinds={','.join(ev['reference_kinds']) or '-'}",
        f"gate_pass={gate.get('n_pass')} gate_fail={gate.get('n_fail')}",
        f"level_up={ev['n_level_up']}",
        f"defect_markers={ev['n_defect_markers']} correction={ev['n_correction_pattern']} "
        f"gate_lock={ev['has_gate_lock']}",
        f"failure_verdict={ev['n_failure_verdicts']} negative={ev['n_negative_markers']}",
        f"doc_shape={ev['n_doc_shape_tokens']} sections={ev['n_sections']}",
        f"infra_markers={ev['n_infra_markers']} units={ev['n_units']}",
        f"verif_names={ev['n_verification_names']}",
        f"correction_snippet={(ev['correction_pattern'] or ['-'])[0][:28]}",
        f"verdict_snippet={(ev['failure_verdicts'] or ['-'])[0][:28]}",
    ]
    return items[:limit]


def _stats(corpus_path: str) -> dict:
    rows = [json.loads(line) for line in open_text(resolve_corpus(corpus_path))]
    evs = [extract(r) for r in rows]
    stats = {
        "n_reports": len(evs),
        "n_missions": len({e["mission_id"] for e in evs}),
        "with_error_rows": sum(1 for e in evs if e["n_error_rows"]),
        "with_external_ref": sum(1 for e in evs if e["n_external_ref_kinds"]),
        "with_gate_pass": sum(1 for e in evs if e["gate"]["n_pass"]),
        "with_level_up": sum(1 for e in evs if e["n_level_up"]),
        "with_correction": sum(1 for e in evs if e["n_correction_pattern"]),
        "with_failure_verdict": sum(1 for e in evs if e["n_failure_verdicts"]),
        "with_doc_shape": sum(1 for e in evs if e["n_doc_shape_tokens"]),
        "ref_kind_counts": {},
    }
    for e in evs:
        for k in e["reference_kinds"]:
            stats["ref_kind_counts"][k] = stats["ref_kind_counts"].get(k, 0) + 1
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/corpus_reports.jsonl")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--report-id", default=None)
    args = ap.parse_args(argv)

    if args.report_id:
        for line in open_text(resolve_corpus(args.corpus)):
            row = json.loads(line)
            if str(row.get("report_id")) == str(args.report_id):
                print(json.dumps(extract(row), ensure_ascii=False, indent=2)[:6000])
                return 0
        print("not found", file=sys.stderr)
        return 1
    if args.stats:
        print(json.dumps(_stats(args.corpus), ensure_ascii=False, indent=2))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())