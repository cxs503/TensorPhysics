"""P27-W2 campaign integration gate: assemble, verify, and report.

The gate evaluates six machine-checkable criteria:

======================  =====================================================
criterion               how it is decided
======================  =====================================================
(a) DAG 收敛             拓扑排序成功 + 集成节点全部前置为 done/honest-FAIL
(b) 跨仓接口契约         RAO / 颗粒流 / GCI-UQ 三组契约 100% 覆盖且全绿
(c) 成熟化单汇总         N04/N09 成熟化节点全部有产出摘要
(d) 证据台账             每节点 result.json + 工件路径 + commit，完整率 100%
(e) 能力台账             本战役能力条目证据等级 REAL 或 WITHHELD（无 MOCK/STUB）
(f) pytest               >=10 条用例全绿
======================  =====================================================

Overall status is ``PASS`` only when (a)-(f) all pass; any hard failure
yields ``FAIL``; partial evidence yields ``WITHHELD``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import contracts as C
from . import dag as D
from . import evidence as E
from . import maturity as M

GATE_SCHEMA = "p27w2_campaign_gate_verdict.v1"
GATE_ID = "P27W2-G1"
REPORT_TITLE = "P27-W2 全战役集成验收门 (P27W2-G1)"

MIN_PYTEST_COUNT = 10


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class GateCriterion:
    criterion_id: str
    label: str
    status: str  # PASS | FAIL | WITHHELD
    detail: str
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "criterion_id": self.criterion_id,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def _aggregate(statuses: list[str]) -> str:
    if not statuses:
        return "WITHHELD"
    if "FAIL" in statuses:
        return "FAIL"
    if all(s == "PASS" for s in statuses):
        return "PASS"
    return "WITHHELD"


@dataclass
class GateVerdict:
    gate_id: str
    plan_id: int
    timestamp: str
    criteria: list[GateCriterion]
    overall: str

    def as_dict(self) -> dict:
        return {
            "schema": GATE_SCHEMA,
            "gate_id": self.gate_id,
            "plan_id": self.plan_id,
            "timestamp": self.timestamp,
            "overall": self.overall,
            "criteria": [c.as_dict() for c in self.criteria],
        }


def criterion_dag(dag: D.CampaignDAG, integration_node_id: int) -> GateCriterion:
    """Criterion (a): acyclic + all predecessors of the integration node
    in a terminal state (done or honest-FAIL)."""
    try:
        order = dag.topological_sort()
    except D.CycleError as exc:
        return GateCriterion(
            "a", "DAG 收敛", "FAIL", str(exc),
            {"acyclic": False},
        )
    result = D.verify_convergence(dag, integration_node_id)
    if result.converged:
        status, detail = "PASS", result.detail
    elif result.acyclic:
        status = "WITHHELD"
        detail = (
            f"DAG 无环但尚未收敛：{result.done_count}/{result.total_count} "
            f"节点终态，阻塞 {result.blocked_nodes}"
        )
    else:
        status, detail = "FAIL", result.detail
    return GateCriterion(
        "a", "DAG 收敛", status, detail, result.as_dict(),
    )


def criterion_contracts(results: list[C.ContractResult]) -> GateCriterion:
    """Criterion (b): all three cross-repo contracts covered and green."""
    if len(results) < 3:
        return GateCriterion(
            "b", "跨仓接口契约", "FAIL",
            f"expected 3 contracts, got {len(results)}",
            {"contracts": [r.as_dict() for r in results]},
        )
    covered = sum(1 for r in results if r.schema != "missing")
    all_ok = all(r.ok for r in results)
    if all_ok:
        status = "PASS"
        detail = "3/3 契约覆盖且全部检查通过"
    elif covered == 3:
        status = "FAIL"
        detail = "3/3 契约覆盖但存在未通过检查"
    else:
        status = "WITHHELD"
        detail = f"契约覆盖 {covered}/3，缺失载荷"
    return GateCriterion(
        "b", "跨仓接口契约", status, detail,
        {"contracts": [r.as_dict() for r in results]},
    )


def criterion_maturity(report: M.MaturityReport) -> GateCriterion:
    """Criterion (c): every maturation node has an output summary."""
    cov = M.maturity_coverage(report)
    if cov["complete"]:
        status, detail = "PASS", (
            f"成熟化节点 {cov['covered_nodes']} 全部有产出摘要"
        )
    elif cov["missing_nodes"]:
        status = "WITHHELD"
        detail = f"成熟化节点缺失产出: {cov['missing_nodes']}"
    else:
        status = "FAIL"
        detail = "成熟化节点存在但摘要为空"
    return GateCriterion("c", "成熟化单汇总", status, detail, cov)


def criterion_evidence(ledger: E.EvidenceLedger) -> GateCriterion:
    """Criterion (d): 100% evidence completeness."""
    incomplete = [
        {"node_id": e.node_id, "missing": e.missing_fields()}
        for e in ledger.incomplete()
    ]
    if ledger.complete:
        status = "PASS"
        detail = f"{ledger.completeness_rate:.0%} 完整（{len(ledger.entries)} 节点）"
    else:
        status = "WITHHELD"
        detail = (
            f"证据完整率 {ledger.completeness_rate:.0%}，"
            f"缺 {len(incomplete)} 节点"
        )
    return GateCriterion(
        "d", "证据台账", status, detail,
        {"completeness_rate": ledger.completeness_rate,
         "incomplete": incomplete},
    )


def criterion_capability(capabilities: list[dict]) -> GateCriterion:
    """Criterion (e): every campaign capability entry is REAL or WITHHELD.

    ``capabilities`` is a list of ``{"capability", "evidence_level",
    "previous_level"}`` dicts.  MOCK/STUB entries are hard failures;
    WITHHELD is acceptable but must be explicit.
    """
    if not capabilities:
        return GateCriterion(
            "e", "能力台账", "WITHHELD", "未提供能力条目",
            {"capabilities": []},
        )
    bad = [
        c for c in capabilities
        if str(c.get("evidence_level", "")).upper() not in ("REAL", "WITHHELD")
    ]
    if bad:
        return GateCriterion(
            "e", "能力台账", "FAIL",
            "存在 MOCK/STUB 未升级条目: "
            + ", ".join(f"{c.get('capability')}={c.get('evidence_level')}"
                        for c in bad),
            {"capabilities": capabilities},
        )
    real = sum(1 for c in capabilities if c.get("evidence_level") == "REAL")
    return GateCriterion(
        "e", "能力台账", "PASS",
        f"{real}/{len(capabilities)} 条 REAL，其余显式 WITHHELD",
        {"capabilities": capabilities},
    )


def criterion_pytest(test_summary: dict) -> GateCriterion:
    """Criterion (f): >=10 pytest cases, all green."""
    passed = int(test_summary.get("passed", 0))
    failed = int(test_summary.get("failed", 0))
    errors = int(test_summary.get("errors", 0))
    if passed >= MIN_PYTEST_COUNT and failed == 0 and errors == 0:
        status = "PASS"
        detail = f"pytest {passed} passed / {failed} failed / {errors} errors"
    elif failed or errors:
        status = "FAIL"
        detail = f"pytest 存在失败: {failed} failed / {errors} errors"
    else:
        status = "WITHHELD"
        detail = f"pytest 通过数 {passed} < {MIN_PYTEST_COUNT}"
    return GateCriterion(
        "f", "pytest", status, detail,
        {"passed": passed, "failed": failed, "errors": errors,
         "min_required": MIN_PYTEST_COUNT},
    )


def run_gate(
    dag: D.CampaignDAG,
    integration_node_id: int,
    contract_results: list[C.ContractResult],
    maturity_report: M.MaturityReport,
    ledger: E.EvidenceLedger,
    capabilities: list[dict],
    test_summary: dict,
    plan_id: int,
) -> GateVerdict:
    """Run all six criteria and aggregate the overall verdict."""
    criteria = [
        criterion_dag(dag, integration_node_id),
        criterion_contracts(contract_results),
        criterion_maturity(maturity_report),
        criterion_evidence(ledger),
        criterion_capability(capabilities),
        criterion_pytest(test_summary),
    ]
    overall = _aggregate([c.status for c in criteria])
    return GateVerdict(
        gate_id=GATE_ID,
        plan_id=plan_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        criteria=criteria,
        overall=overall,
    )


def write_outputs(out_dir: str | Path, verdict: GateVerdict,
                  extra: dict | None = None) -> dict[str, str]:
    """Persist the verdict JSON; returns {name: path}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = verdict.as_dict()
    if extra:
        payload["extra"] = extra
    path = out / "gate_verdict.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return {"gate_verdict": str(path)}
