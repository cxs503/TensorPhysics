"""P35 gate criteria (a)..(g), each computed from recorded evidence only.

``PASS`` is never inferred.  A criterion is green only when the evidence that
proves it is present and readable; absent evidence is ``WITHHELD`` and a
demonstrated shortfall is ``FAIL``.  The gate-level rule ((g)) is that
``overall_status`` is ``PASS`` only if every criterion is ``PASS`` -- ``WITHHELD``
forces ``FAIL``, because an undelivered node must not be accepted by silence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import spec as S

PASS, FAIL, WITHHELD = "PASS", "FAIL", "WITHHELD"
_REPORT_BAD_MISSION_STATES = ("failed", "cancelled", "cancelled_by_user")


@dataclass
class Verdict:
    criterion_id: str
    label: str
    status: str
    detail: str
    metrics: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "criterion_id": self.criterion_id,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
            "metrics": self.metrics,
            "evidence": self.evidence,
        }


@dataclass
class NodeVerdict:
    ext_id: str
    mission_id: int
    repo: str
    physics_domain: str
    deliverable: str = WITHHELD
    registered: str = WITHHELD
    maturity: str = WITHHELD
    chain_ready: bool = False
    blocking: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ext_id": self.ext_id,
            "mission_id": self.mission_id,
            "repo": self.repo,
            "physics_domain": self.physics_domain,
            "deliverable": self.deliverable,
            "registered": self.registered,
            "maturity": self.maturity,
            "chain_ready": self.chain_ready,
            "blocking": list(self.blocking),
        }


def _aggregate(statuses) -> str:
    statuses = list(statuses)
    if not statuses:
        return WITHHELD
    if FAIL in statuses:
        return FAIL
    if all(s == PASS for s in statuses):
        return PASS
    return WITHHELD


def _scan_for(scans: dict, ext_id: str) -> dict:
    return (scans or {}).get(ext_id) or {"scannable": False,
                                        "reason": "repository not bound to the gate"}
            
# --------------------------------------------------------------------------
# (a) deliverables landed
# --------------------------------------------------------------------------
def evaluate_deliverable(node_row: dict, scan: dict,
                         maturity_family: str | None = None) -> tuple[str, str, dict]:
    """Judge one node's landed deliverables: report + declared artifact families."""
    reasons: list[str] = []
    status_bits: list[str] = []
    metrics = {
        "reports": len(node_row.get("reports") or []),
        "mission_status": node_row.get("mission_status"),
        "artifact_families": len(scan.get("families") or []),
        "artifact_families_satisfied": sum(1 for f in scan.get("families") or []
                                           if f.get("satisfied")),
        "repository_scannable": bool(scan.get("scannable")),
    }
    if node_row.get("mission_status") in _REPORT_BAD_MISSION_STATES:
        reasons.append(f"mission状态={node_row.get('mission_status')}")
        status_bits.append(FAIL)
    if not node_row.get("reports"):
        reasons.append("平台未收到该节点报告 (reports=0)")
        status_bits.append(FAIL)
    authoritative = bool(scan.get("authoritative"))
    metrics["authoritative_checkout"] = authoritative
    metrics["repo_head"] = scan.get("repo_head")
    if scan.get("scannable"):
        for fam in scan.get("families") or []:
            if fam.get("satisfied"):
                continue
            if maturity_family and fam.get("pattern") == maturity_family:
                continue    # the maturity declaration is judged by criterion (e)
            reasons.append(f"产物族未落盘/无效: {fam.get('kind')} {fam.get('pattern')}")
            status_bits.append(FAIL if authoritative else WITHHELD)
    else:
        reasons.append(f"仓库不可扫描: {scan.get('reason')}")
        status_bits.append(WITHHELD)
    status = _aggregate(status_bits) if status_bits else PASS
    return status, "; ".join(reasons) or "report present and every declared artifact family resolved", metrics


def evaluate_a(snapshot: dict, scans: dict) -> tuple[Verdict, dict]:
    node_verdicts: dict[str, NodeVerdict] = {}
    evidence = []
    for node in S.NODES:
        row = _node_row(snapshot, node.ext_id)
        scan = _scan_for(scans, node.ext_id)
        status, detail, metrics = evaluate_deliverable(
            row, scan, maturity_family=S.maturity_pattern(node.ext_id) if node.maturity else None)
        nv = NodeVerdict(ext_id=node.ext_id, mission_id=node.mission_id, repo=node.repo,
                         physics_domain=node.physics_domain, deliverable=status)
        if status != PASS:
            nv.blocking.append(f"deliverable:{status}: {detail}")
        node_verdicts[node.ext_id] = nv
        evidence.append({"ext_id": node.ext_id, "status": status, "detail": detail,
                         "metrics": metrics})
    landed = sum(1 for e in evidence if e["status"] == PASS)
    rate = landed / len(evidence)
    status = _aggregate([e["status"] for e in evidence])
    return Verdict(
        "a", S.CRITERIA["a"], status,
        f"{landed}/{len(evidence)} 节点产物落盘 (落盘率 {rate:.1%})",
        metrics={"landed": landed, "n_nodes": len(evidence), "landing_rate": rate,
                 "threshold": 1.0},
        evidence={"nodes": evidence},
    ), node_verdicts


# --------------------------------------------------------------------------
# (b) capability ledger registration
# --------------------------------------------------------------------------
def evaluate_registration(node: S.NodeSpec, node_row: dict) -> tuple[str, str, dict]:
    """A registration counts only if the ledger row names the capability identity."""
    delta = node_row.get("capability_delta")
    if not delta:
        return FAIL, "能力台账无登记 (capability delta not booked)", {"booked": False}
    evidence = delta.get("evidence") or {}
    problems = []
    if evidence.get("ext_id") != node.ext_id:
        problems.append(f"ext_id 缺失/不符 (ledger={evidence.get('ext_id')!r} != {node.ext_id!r})")
    if not evidence.get("physics_domain"):
        problems.append("physics_domain 缺失")
    elif evidence.get("physics_domain") != node.physics_domain:
        problems.append(f"physics_domain 不符 (ledger={evidence.get('physics_domain')!r}"
                        f" != declared {node.physics_domain!r})")
    if delta.get("mission_repo") != node.repo:
        problems.append(f"repo 不符 (ledger={delta.get('mission_repo')!r} != {node.repo!r})")
    metrics = {"booked": True, "category": delta.get("category"),
               "ledger_ext_id": evidence.get("ext_id"),
               "ledger_physics_domain": evidence.get("physics_domain"),
               "ledger_repo": delta.get("mission_repo")}
    if problems:
        return FAIL, "; ".join(problems), metrics
    return PASS, "台账登记含 ext_id/repo/physics_domain 且与声明一致", metrics


def evaluate_b(snapshot: dict, node_verdicts: dict) -> Verdict:
    evidence = []
    for node in S.NODES:
        row = _node_row(snapshot, node.ext_id)
        status, detail, metrics = evaluate_registration(node, row)
        node_verdicts[node.ext_id].registered = status
        if status != PASS:
            node_verdicts[node.ext_id].blocking.append(f"registration:{status}: {detail}")
        evidence.append({"ext_id": node.ext_id, "status": status, "detail": detail,
                         "metrics": metrics})
    registered = sum(1 for e in evidence if e["status"] == PASS)
    rate = registered / len(evidence)
    return Verdict(
        "b", S.CRITERIA["b"], _aggregate([e["status"] for e in evidence]),
        f"{registered}/{len(evidence)} 节点能力台账登记 (登记率 {rate:.1%}); "
        f"台账契约要求 evidence 携带 ext_id 与 physics_domain",
        metrics={"registered": registered, "n_nodes": len(evidence),
                 "registration_rate": rate, "threshold": 1.0},
        evidence={"nodes": evidence},
    )


# --------------------------------------------------------------------------
# (c) regression baseline
# --------------------------------------------------------------------------
def evaluate_c(cases: list[dict]) -> Verdict:
    statuses = [c["status"] for c in cases]
    status = _aggregate(statuses) if statuses else WITHHELD
    tested = [c for c in cases if c["status"] in (PASS, FAIL)]
    devs = [c["rel_dev"] for c in tested if isinstance(c.get("rel_dev"), (int, float))]
    worst = max(devs) if devs else None
    n_pass = sum(1 for c in cases if c["status"] == PASS)
    if not cases:
        detail = "no regression case declared"
    elif tested:
        detail = (f"{n_pass}/{len(cases)} 已复跑案例 PASS；已复跑 {len(tested)} 例，"
                  f"最大偏差 {worst:.3e} (阈值 {S.REGRESSION_THRESHOLD:.1%})")
    else:
        withheld = [c for c in cases if c["status"] == WITHHELD]
        detail = (f"0/{len(cases)} 复跑成功：{len(withheld)} 例 WITHHELD "
                  f"({withheld[0]['detail'] if withheld else 'n/a'})")
    return Verdict(
        "c", S.CRITERIA["c"], status, detail,
        metrics={"n_cases": len(cases), "n_pass": n_pass,
                 "n_tested": len(tested), "max_rel_dev": worst,
                 "threshold": S.REGRESSION_THRESHOLD},
        evidence={"cases": cases},
    )


# --------------------------------------------------------------------------
# (d) dependency chain
# --------------------------------------------------------------------------
def _platform_edges(snapshot: dict) -> list[dict]:
    """Resolve ``campaign_plans.dependencies`` (node ids) onto campaign ext_ids."""
    nodes = (snapshot or {}).get("nodes") or []
    id_to_ext = {str(r.get("plan_node_id")): r.get("ext_id") for r in nodes
                 if r.get("plan_node_id") is not None}
    edges, unresolved = [], []
    for row in nodes:
        for dep in row.get("plan_dependencies") or []:
            key = str(dep)
            src = id_to_ext.get(key) or (key if key in S.NODE_BY_ID else None)
            if src is None:
                unresolved.append({"from": key, "to": row.get("ext_id")})
                continue
            edges.append({"from": src, "to": row.get("ext_id")})
    if unresolved:
        edges.append({"unresolved": unresolved})
    return [e for e in edges if "from" in e] + [e for e in edges if "unresolved" in e]


def declared_edges() -> list[dict]:
    edges = []
    for node in S.NODES:
        for pre in node.prerequisites:
            if pre.kind == "node":
                edges.append({"from": pre.ref, "to": node.ext_id})
    return edges


def _acyclic(edges: list[dict]) -> tuple[bool, list[str]]:
    graph: dict[str, list[str]] = {n.ext_id: [] for n in S.NODES}
    for e in edges:
        graph.setdefault(e["from"], []).append(e["to"])
    state: dict[str, int] = {}
    order: list[str] = []
    cycle: list[str] = []

    def visit(u: str, stack: list[str]) -> bool:
        if state.get(u) == 1:
            cycle.extend(stack[stack.index(u):] + [u] if u in stack else [u])
            return False
        if state.get(u) == 2:
            return True
        state[u] = 1
        for v in graph.get(u, ()):
            if not visit(v, stack + [u]):
                return False
        state[u] = 2
        order.append(u)
        return True

    for n in graph:
        if not visit(n, []):
            return False, cycle
    return True, list(reversed(order))


def evaluate_d(snapshot: dict, node_verdicts: dict) -> Verdict:
    declared = declared_edges()
    platform = _platform_edges(snapshot)
    unresolved = [e for e in declared if e["from"] not in S.NODE_BY_ID]
    ok, topo = _acyclic(declared)
    roots = {n.ext_id for n in S.NODES if not any(
        p.kind == "node" for p in n.prerequisites)}
    orphans = sorted(
        n.ext_id for n in S.NODES
        if not n.prerequisites and n.ext_id not in S.ROOT_NODES)
    declared_pairs = {(e["from"], e["to"]) for e in declared}
    platform_pairs = {(e["from"], e["to"]) for e in platform}
    missing_on_platform = sorted(declared_pairs - platform_pairs)
    extra_on_platform = sorted(platform_pairs - declared_pairs)

    blocked_edges = []
    for e in declared:
        upstream = node_verdicts.get(e["from"])
        if upstream is None or upstream.deliverable != PASS:
            blocked_edges.append({"from": e["from"], "to": e["to"],
                                  "reason": f"upstream deliverable {getattr(upstream, 'deliverable', 'ABSENT')}"})
    ready = {ext: False for ext in S.NODE_BY_ID}
    for ext in topo:
        direct = [e["from"] for e in declared if e["to"] == ext]
        node_ready = node_verdicts[ext].deliverable == PASS
        if direct:
            node_ready = node_ready and all(ready.get(u, False) for u in direct)
        ready[ext] = node_ready
    for ext, is_ready in ready.items():
        node_verdicts[ext].chain_ready = is_ready
        if not is_ready and node_verdicts[ext].deliverable == PASS:
            node_verdicts[ext].blocking.append("chain: upstream deliverable missing")

    problems = []
    if not ok:
        problems.append(f"依赖图成环: {topo}")
    if unresolved:
        problems.append(f"依赖指向未声明节点: {sorted({e['from'] for e in unresolved})}")
    if orphans:
        problems.append(f"孤立节点(无任何声明上游且非指定根): {orphans}")
    if missing_on_platform:
        problems.append(
            f"平台 campaign_plans.dependencies 未记录 {len(missing_on_platform)} 条声明边 "
            f"(如 {missing_on_platform[:2]})")
    if extra_on_platform:
        problems.append(f"平台记录了未声明的依赖边: {extra_on_platform}")

    structural_status = PASS if not (unresolved or orphans or not ok) else FAIL
    status = structural_status if not (missing_on_platform or extra_on_platform) else FAIL
    detail = ("; ".join(problems) if problems else
              f"声明 {len(declared)} 条 in-campaign 边，拓扑序 {len(topo)} 节点，无孤立节点；"
              f"平台依赖记录一致")
    return Verdict(
        "d", S.CRITERIA["d"], status, detail,
        metrics={"declared_edges": len(declared), "platform_edges": len(platform),
                 "acyclic": ok, "topological_order": topo, "orphans": orphans,
                 "roots": sorted(roots), "missing_on_platform": [list(p) for p in missing_on_platform],
                 "extra_on_platform": [list(p) for p in extra_on_platform],
                 "blocked_edges": blocked_edges,
                 "chain_ready_nodes": sorted(k for k, v in ready.items() if v)},
        evidence={"declared": declared, "platform": platform},
    )


# --------------------------------------------------------------------------
# (e) maturity nodes
# --------------------------------------------------------------------------
def maturity_artifact(scan: dict, ext_id: str) -> dict | None:
    """Return the maturity declaration found in the node scan, if any."""
    pattern = S.maturity_pattern(ext_id)
    for fam in scan.get("families") or []:
        if fam.get("pattern") == pattern:
            for rec in fam.get("resolved") or []:
                if rec.get("valid"):
                    return rec
    return None


def evaluate_e(snapshot: dict, scans: dict, node_verdicts: dict) -> Verdict:
    evidence = []
    for ext in S.MATURITY_NODES:
        node = S.NODE_BY_ID[ext]
        scan = _scan_for(scans, ext)
        if not scan.get("scannable"):
            status = WITHHELD
            detail = f"仓库不可扫描，成熟化声明无法核验: {scan.get('reason')}"
            metrics = {"trl": None, "gates": 0, "gates_passed": 0}
        else:
            hit = maturity_artifact(scan, ext)
            if not hit:
                status = FAIL
                detail = f"缺失成熟化声明 {S.maturity_pattern(ext)}"
                metrics = {"trl": None, "gates": 0, "gates_passed": 0}
            else:
                try:
                    payload = json.loads(
                        (Path(scan["repo_root"]) / hit["path"]).read_text())
                except (OSError, ValueError) as exc:
                    payload = None
                    detail = f"成熟化声明不可解析: {exc}"
                if payload is None:
                    status = FAIL
                    metrics = {"trl": None, "gates": 0, "gates_passed": 0}
                else:
                    trl = payload.get("trl")
                    gates = payload.get("gates") or []
                    passed = [g for g in gates if str((g or {}).get("status", "")).lower() == "pass"]
                    problems = []
                    if not isinstance(trl, int) or trl < S.MATURITY_TRL_FLOOR:
                        problems.append(f"TRL {trl!r} < {S.MATURITY_TRL_FLOOR}")
                    if not gates:
                        problems.append("门禁记录为空")
                    elif len(passed) != len(gates):
                        problems.append(f"门禁 {len(passed)}/{len(gates)} 通过")
                    status = PASS if not problems else FAIL
                    detail = ("; ".join(problems) if problems else
                              f"TRL={trl}，{len(passed)}/{len(gates)} 门禁通过")
                    metrics = {"trl": trl, "gates": len(gates), "gates_passed": len(passed),
                               "declaration": hit["path"], "declaration_sha256": hit["sha256"]}
        node_verdicts[ext].maturity = status
        if status != PASS:
            node_verdicts[ext].blocking.append(f"maturity:{status}: {detail}")
        evidence.append({"ext_id": ext, "status": status, "detail": detail, "metrics": metrics})
    passed = sum(1 for e in evidence if e["status"] == PASS)
    rate = passed / len(evidence)
    return Verdict(
        "e", S.CRITERIA["e"], _aggregate([e["status"] for e in evidence]),
        f"{passed}/{len(evidence)} 成熟化节点门禁通过 (通过率 {rate:.1%})；"
        f"要求 TRL>={S.MATURITY_TRL_FLOOR} 且成熟化声明中门禁全通过",
        metrics={"passed": passed, "n_nodes": len(evidence), "pass_rate": rate,
                 "trl_floor": S.MATURITY_TRL_FLOOR, "threshold": 1.0},
        evidence={"nodes": evidence},
    )


# --------------------------------------------------------------------------
# (g) honest verdict
# --------------------------------------------------------------------------
def evaluate_g(criteria: list[Verdict], node_verdicts: dict) -> tuple[str, list[str]]:
    blocking = [f"{c.criterion_id}:{c.status}:{c.label}" for c in criteria if c.status != PASS]
    node_blocking = []
    for ext in S.NODE_BY_ID:
        nv = node_verdicts[ext]
        for item in nv.blocking:
            node_blocking.append(f"{ext}: {item}")
    overall = PASS if all(c.status == PASS for c in criteria) else FAIL
    if overall == FAIL and not (blocking or node_blocking):
        blocking.append("g:FAIL:verdict has no blocking entry though a criterion is not PASS")
    return overall, blocking + node_blocking


def _node_row(snapshot: dict, ext_id: str) -> dict:
    for row in (snapshot or {}).get("nodes") or []:
        if row.get("ext_id") == ext_id:
            return row
    return {"ext_id": ext_id, "reports": [], "capability_delta": None,
            "mission_status": None, "plan_dependencies": []}