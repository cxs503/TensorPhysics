"""Judging layer: node verdicts, campaign criteria C1..C4, honest FAIL semantics.

Three rules are enforced here and nowhere else:

1. **WITHHELD is never PASS.**  A criterion whose number the node did not
   deliver cannot be accepted by silence; it poisons the node verdict.
2. **A node PASSes only when everything it promised arrived**: artifact family
   parseable, declared pytest suite executed green, every declared criterion
   satisfied.
3. **The campaign gate PASSes only when C1..C4 are all PASS.**  ``overall`` is
   therefore the conjunction of the four machine criteria, and every non-green
   criterion contributes at least one remediation line to the gap analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .spec import MIN_REPORT_CHARS, Criterion


# ---------------------------------------------------------------------------
# single criterion
# ---------------------------------------------------------------------------
def _as_candidates(candidates: Any, source: str = "") -> list[dict[str, Any]]:
    """Normalise the accepted ``judge_criterion`` inputs to a candidate list."""
    if candidates is None:
        return []
    if isinstance(candidates, Mapping):
        return [dict(candidates)]
    if isinstance(candidates, (list, tuple)):
        out = []
        for item in candidates:
            if isinstance(item, Mapping) and "value" in item:
                out.append(dict(item))
            elif isinstance(item, (int, float, bool)):
                out.append({"value": item, "source": source})
        return out
    return [{"value": candidates, "source": source}]


def _worst(crit: Criterion, cands: list[Mapping[str, Any]]) -> tuple[Any, bool, str]:
    """Aggregate candidate numbers *conservatively* (worst case wins).

    * error-type criteria (``rel_dev_le`` / ``le``) take the **largest** error,
    * goodness-type criteria (``ge``) take the **smallest** value,
    * interval criteria (``in_range``) fail if **any** candidate is outside and
      report the value furthest outside,
    * boolean controls pass only if **all** candidates are true.

    This is what stops a node from passing by reporting its best case only.
    """
    if crit.comparator == "is_true":
        flags = [bool(c["value"]) for c in cands]
        return all(flags), all(flags), f"{len(flags)} boolean control(s), all={all(flags)}"
    numeric = []
    for cand in cands:
        try:
            numeric.append((float(cand["value"]), str(cand.get("source", ""))))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return None, False, "no numeric candidate"
    if crit.comparator == "rel_dev_le":
        ref = float(crit.ref if crit.ref is not None else crit.threshold)
        scored = [(abs(value - ref) / max(abs(ref), 1e-300), value, src)
                  for value, src in numeric]
        score, value, src = max(scored, key=lambda item: item[0])
        ok = score <= float(crit.threshold)
        return value, ok, (f"worst of {len(numeric)} candidate(s): "
                           f"|{value:.6g}-{ref:.6g}|/{ref:.6g} = {score:.3e} "
                           f"<= {crit.threshold} [{src}]")
    if crit.comparator == "le":
        value, src = max(numeric, key=lambda item: abs(item[0]))
        ok = abs(value) <= float(crit.threshold)
        return value, ok, f"worst |value| of {len(numeric)}: {abs(value):.6g} <= {crit.threshold} [{src}]"
    if crit.comparator == "ge":
        value, src = min(numeric, key=lambda item: item[0])
        ok = value >= float(crit.threshold)
        return value, ok, f"worst (min) of {len(numeric)}: {value:.6g} >= {crit.threshold} [{src}]"
    if crit.comparator == "in_range":
        lo, hi = crit.threshold  # type: ignore[misc]
        scored = []
        for value, src in numeric:
            distance = 0.0 if lo <= value <= hi else min(abs(value - lo), abs(value - hi))
            scored.append((distance, value, src))
        distance, value, src = max(scored, key=lambda item: item[0])
        ok = distance == 0.0
        return value, ok, (f"{len(numeric)} candidate(s), worst {value:.6g} in "
                           f"[{lo}, {hi}]: {ok} [{src}]")
    raise ValueError(f"unsupported comparator {crit.comparator}")


def judge_criterion(crit: Criterion, candidates: Any, source: str = "") -> dict[str, Any]:
    """Judge one criterion from one or more reported numbers (worst case)."""
    record = {
        "id": crit.cid,
        "statement": crit.statement,
        "status": "WITHHELD",
        "value": None,
        "threshold": crit.threshold if not isinstance(crit.threshold, tuple)
        else list(crit.threshold),
        "comparator": crit.comparator,
        "unit": crit.unit,
        "source": crit.source if not crit.external else f"{crit.source} (external)",
        "detail": "no value found in delivered artifacts",
    }
    cands = _as_candidates(candidates, source)
    if not cands:
        return record
    value, ok, detail = _worst(crit, cands)
    if value is None and crit.comparator != "is_true":
        record["detail"] = detail
        return record
    if crit.comparator == "is_true":
        record["value"] = None
    elif isinstance(value, bool):
        record["value"] = None
    else:
        record["value"] = float(value)
    record["status"] = "PASS" if ok else "FAIL"
    record["detail"] = detail
    return record


# ---------------------------------------------------------------------------
# node verdict
# ---------------------------------------------------------------------------
def node_evidence_gaps(node: Mapping[str, Any], spec: Any) -> list[str]:
    gaps: list[str] = []
    if not node.get("root_present"):
        gaps.append(f"节点工作树缺失: {node.get('root')}")
        return gaps
    json_artifacts = [a for a in node.get("artifacts", []) if a.get("kind") == "json"]
    if not json_artifacts:
        gaps.append("未落盘任何 JSON 产物 (" + ", ".join(spec.artifact_globs) + ")")
    elif not any(a.get("parseable") for a in json_artifacts):
        gaps.append("JSON 产物全部不可解析")
    pytest = node.get("pytest", {})
    if pytest.get("mode") != "executed":
        gaps.append(f"声明的 pytest 套件未执行 (mode={pytest.get('mode')}, "
                    f"declared={pytest.get('declared')})")
    elif pytest.get("status") != "PASS":
        detail = (f"pytest 未全绿: passed={pytest.get('passed')} "
                  f"failed={pytest.get('failed')} errors={pytest.get('errors')}")
        if pytest.get("failures"):
            detail += "；失败用例: " + ", ".join(pytest["failures"][:6])
        gaps.append(detail)
    for crit in node.get("criteria", []):
        if crit["status"] == "WITHHELD":
            gaps.append(f"判据 {crit['id']} 无数字证据")
        elif crit["status"] == "FAIL":
            gaps.append(f"判据 {crit['id']} 未达标: {crit.get('detail', '')}")
    return gaps


def node_verdict(node: Mapping[str, Any]) -> str:
    statuses = [c["status"] for c in node.get("criteria", [])]
    pytest_status = node.get("pytest", {}).get("status")
    artifacts = [a for a in node.get("artifacts", []) if a.get("kind") == "json"]
    if not node.get("root_present"):
        return "NOT_SUPPORTED"
    if "FAIL" in statuses or pytest_status == "FAIL":
        return "FAIL"
    if not artifacts or not any(a.get("parseable") for a in artifacts):
        return "WITHHELD"
    if any(s == "WITHHELD" for s in statuses) or pytest_status != "PASS":
        return "WITHHELD"
    return "PASS"


# ---------------------------------------------------------------------------
# campaign criteria
# ---------------------------------------------------------------------------
def evaluate_c1(nodes: Sequence[Mapping[str, Any]], evidence_dir: Path) -> dict[str, Any]:
    """C1: every node's pytest result exists as parseable JSON and was executed."""
    parseable = 0
    executed = 0
    green = 0
    gaps: list[str] = []
    for node in nodes:
        record_path = node.get("pytest", {}).get("record_path")
        ok = False
        if record_path and Path(record_path).is_file():
            try:
                json.loads(Path(record_path).read_text(encoding="utf-8"))
                ok = True
            except ValueError:
                ok = False
        if ok:
            record = json.loads(Path(record_path).read_text(encoding="utf-8"))
            if record.get("mode") in ("executed", "collect-only") and \
                    record.get("status") in ("PASS", "FAIL", "WITHHELD"):
                parseable += 1
            else:
                gaps.append(f"{node['node_key']}: pytest 记录为占位（mode="
                            f"{record.get('mode')!r}），不计为可解析证据")
        else:
            gaps.append(f"{node['node_key']}: pytest 记录缺失或不可解析")
        if node.get("pytest", {}).get("mode") == "executed":
            executed += 1
        if node.get("pytest", {}).get("status") == "PASS":
            green += 1
        else:
            gaps.append(f"{node['node_key']}: pytest 状态 {node.get('pytest', {}).get('status')}"
                        f" (mode={node.get('pytest', {}).get('mode')})")
    total = len(nodes)
    status = "PASS" if (parseable == total and executed == total and green == total) else "FAIL"
    return {
        "id": "C1",
        "statement": "所有节点 pytest 结果可解析 (JSON)",
        "status": status,
        "metrics": {
            "nodes": total,
            "pytest_json_parseable": parseable,
            "pytest_executed": executed,
            "pytest_green": green,
            "parseable_ratio": round(parseable / total, 6) if total else 0.0,
        },
        "gaps": gaps,
        "evidence": str(Path(evidence_dir)),
    }


def evaluate_c2(contracts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """C2: every required cross-domain interface contract passes bilateral checks."""
    required = [c for c in contracts if c.get("required", True)]
    gaps: list[str] = []
    passed = 0
    for contract in required:
        if contract["status"] == "PASS":
            passed += 1
        else:
            gaps.append(f"{contract['contract_id']}: {contract['status']} "
                        + ("; ".join(contract.get("gaps", [])) or ""))
    data_statuses = [c.get("data_conformance") for c in contracts]
    data_fail = [c["contract_id"] for c in contracts if c.get("data_conformance") == "FAIL"]
    status = "PASS" if (passed == len(required) and not data_fail) else "FAIL"
    return {
        "id": "C2",
        "statement": "跨域接口契约验证通过",
        "status": status,
        "metrics": {
            "contracts": len(contracts),
            "required_contracts": len(required),
            "required_passed": passed,
            "data_conformance_pass": sum(1 for s in data_statuses if s == "PASS"),
            "data_conformance_withheld": sum(1 for s in data_statuses if s == "WITHHELD"),
            "data_conformance_fail": sum(1 for s in data_statuses if s == "FAIL"),
        },
        "gaps": gaps,
        "evidence": "; ".join(c["fingerprint"] for c in contracts),
    }


def evaluate_c3(markdown_path: Path, sections: Sequence[str], figure_path: Path | None,
                min_chars: int = MIN_REPORT_CHARS) -> dict[str, Any]:
    """C3: campaign markdown report exists, long enough, six elements present."""
    gaps: list[str] = []
    text = ""
    if Path(markdown_path).is_file():
        text = Path(markdown_path).read_text(encoding="utf-8")
    else:
        gaps.append(f"报告缺失: {markdown_path}")
    chars = len(text)
    missing_sections = [s for s in sections if s not in text]
    if missing_sections:
        gaps.append(f"报告缺少章节: {missing_sections}")
    if chars < min_chars:
        gaps.append(f"报告 {chars} 字符 < 下限 {min_chars}")
    if figure_path is not None and not Path(figure_path).is_file():
        gaps.append(f"图件缺失: {figure_path}")
    status = "PASS" if (chars >= min_chars and not missing_sections
                        and (figure_path is None or Path(figure_path).is_file())) else "FAIL"
    return {
        "id": "C3",
        "statement": f"战役报告 markdown ≥ {min_chars} 字符",
        "status": status,
        "metrics": {
            "chars": chars,
            "min_chars": min_chars,
            "sections_found": len(sections) - len(missing_sections),
            "sections_required": len(sections),
            "figure_present": bool(figure_path is not None and Path(figure_path).is_file()),
        },
        "gaps": gaps,
        "evidence": str(markdown_path),
    }


def evaluate_c4(report_path: Path, schema_fingerprint_value: str,
                schema_name: str = "campaign_gate_report.v1") -> dict[str, Any]:
    """C4: the campaign verdict JSON on disk is closed-schema valid and self-sealed.

    The criterion reads the file back (it never trusts the in-memory dict), runs
    the registered closed schema over it, and re-derives the fingerprint from the
    canonical payload -- so a hand-edited or truncated JSON cannot pass.
    """
    from ..schema import payload_digest, validate
    from ..schema_defs import schema_for

    gaps: list[str] = []
    payload: dict[str, Any] | None = None
    validated = False
    fingerprint_ok = False
    if not Path(report_path).is_file():
        gaps.append(f"裁决 JSON 缺失: {report_path}")
    else:
        try:
            payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        except ValueError as exc:
            gaps.append(f"裁决 JSON 不可解析: {exc}")
        if isinstance(payload, dict):
            try:
                validate(schema_for(schema_name), payload)
                validated = True
            except Exception as exc:  # SchemaError or unsupported payload
                gaps.append(f"裁决 JSON 未通过 {schema_name} 校验: {exc}")
            stamped = payload.get("fingerprint")
            recomputed = payload_digest({k: v for k, v in payload.items() if k != "fingerprint"})
            fingerprint_ok = bool(stamped) and stamped == recomputed
            if not fingerprint_ok:
                gaps.append("裁决 JSON 指纹与内容不一致（可能被手改）")
    nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
    criteria_total = sum(len(n.get("criteria", [])) for n in nodes)
    if isinstance(payload, dict) and not nodes:
        gaps.append("裁决 JSON 无节点记录")
    if isinstance(payload, dict) and criteria_total == 0:
        gaps.append("裁决 JSON 无判据数字")
    status = "PASS" if (validated and fingerprint_ok and nodes and criteria_total > 0) else "FAIL"
    return {
        "id": "C4",
        "statement": "端到端输出战役验收 JSON（schema 校验 + 指纹落盘）",
        "status": status,
        "metrics": {
            "report_json": str(report_path),
            "schema_fingerprint": schema_fingerprint_value,
            "nodes": len(nodes),
            "criteria_total": criteria_total,
            "contracts": len(payload.get("contracts", [])) if isinstance(payload, dict) else 0,
            "schema_valid": validated,
            "fingerprint_consistent": fingerprint_ok,
        },
        "gaps": gaps,
        "evidence": str(report_path),
    }


def overall_status(criteria: Iterable[Mapping[str, Any]],
                   nodes: Sequence[Mapping[str, Any]] | None = None) -> str:
    """Campaign verdict: C1..C4 green **and** every node verdict PASS.

    The conjunction with the node verdicts is what makes ``WITHHELD`` fatal: a
    node that delivered nothing satisfies no campaign criterion, so an
    undelivered node can never be accepted by the gate's silence.
    """
    if not all(c["status"] == "PASS" for c in criteria):
        return "FAIL"
    if nodes is not None and any(n.get("verdict") != "PASS" for n in nodes):
        return "FAIL"
    return "PASS"