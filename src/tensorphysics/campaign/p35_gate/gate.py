"""P35-G1 campaign integration gate: assemble evidence, judge, publish readiness.

The gate answers the seven machine-checkable criteria of the P35 campaign:

==  ================================================================  =============
id  how it is decided                                                 non-green
==  ================================================================  =============
a   per node: platform report + every declared artifact family          FAIL/WITHHELD
    resolved on disk with JSON parse / PNG magic+trailer integrity
b   per node: ledger row booked and carrying ``ext_id``, ``repo``,      FAIL
    ``physics_domain`` equal to the declared identity
c   already-verified faces re-run against their recorded baseline       FAIL/WITHHELD
    (``|delta|/max(|baseline|, floor) <= 1%``)
d   declared dependency graph: acyclic, resolvable, no orphans, and     FAIL
    consistent with the platform's ``campaign_plans.dependencies``
e   matured nodes (N4/N5/N6/N7): maturity declaration on disk with      FAIL/WITHHELD
    ``TRL >= 6`` and every gate record passing
f   the emitted ``campaign_readiness.v1`` payload validates against      FAIL
    the registered schema (fingerprint pinned in the payload)
g   honest verdict: ``overall_status == PASS`` only when a..f are all   FAIL
    PASS; every non-green criterion and node appears in ``blocking``
==  ================================================================  =============

``WITHHELD`` (evidence the gate could not obtain) is reported as such per
criterion but still forces ``overall_status = FAIL`` and
``dispatcher_ready = false``: an undelivered node is never accepted by silence.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tensorphysics.campaign.schema import SchemaError

from . import collect as C
from . import criteria as K
from . import readiness_schema as _rs
from . import regression as R
from . import spec as S

READINESS_SCHEMA = _rs.READINESS_SCHEMA_NAME
READINESS_JSON = "campaign_readiness.json"
READINESS_MD = "campaign_readiness.md"
GENERATOR = "tensorphysics.campaign.p35_gate/gate.py"

LIMITATIONS = (
    "节点产物落盘以平台报告 + 声明产物族的磁盘解析为准：未绑定仓库根的仓库记为 WITHHELD，"
    "不当作空产物；报告存在而产物族缺失仍是 FAIL。",
    "回归判据只复跑生产者自己发布的 runner，门不重写任何求解器内核；runner 入口缺失或"
    "超出计算预算的案例记 WITHHELD 并写明原因，绝不代之以「未发现退化」。",
    "机器地板量级指标（如 fp64 Couette 的 l2≈1e-13）用声明的绝对 floor 参与比较，"
    "否则噪声会被百分比放大成伪回归。",
    "依赖链以节点 goal 的声明为准并同时记录平台 campaign_plans.dependencies；两者不一致"
    "记为阻塞项而不是择一采信。",
    "能力台账登记要求 evidence 携带 ext_id/physics_domain（平台 delta 行的第一类字段只有"
    "mission_id/repo/category），这是本门对台账契约的显式要求，属契约缺口而非门杜撰。",
)


def _read_json(path: str | Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _provenance_inputs(paths: dict, fallback_digest: str | None = None) -> list[dict]:
    out = []
    for label, path in paths.items():
        p = Path(str(path)) if path else None
        if p is not None and p.is_dir():
            # a bound checkout is an input without a single file digest; its HEAD is
            # recorded per node in criterion (a) metrics instead
            out.append({"path": f"{label}:{p}", "present": True, "sha256": None})
            continue
        present = bool(p and p.is_file())
        out.append({"path": f"{label}:{p}" if p else f"{label}:<none>",
                    "present": present,
                    "sha256": C.sha256_file(p) if present else None})
    if not out:
        # an in-memory snapshot still has an identity: its canonical digest
        out.append({"path": "snapshot:in-memory", "present": True,
                    "sha256": fallback_digest})
    return out


def build_readiness(snapshot: dict, *, scans: dict, regression_cases: list[dict],
                    plan_id: int | None = None, generated_utc: str | None = None,
                    inputs: dict | None = None, f_status: str = K.PASS,
                    f_detail: str = "", f_metrics: dict | None = None) -> dict:
    """Assemble the ``campaign_readiness.v1`` payload (criteria a..g)."""
    crit_a, node_verdicts = K.evaluate_a(snapshot, scans)
    crit_b = K.evaluate_b(snapshot, node_verdicts)
    crit_c = K.evaluate_c(regression_cases)
    crit_d = K.evaluate_d(snapshot, node_verdicts)
    crit_e = K.evaluate_e(snapshot, scans, node_verdicts)
    crit_f = K.Verdict("f", S.CRITERIA["f"], f_status,
                       f_detail or "campaign_readiness.v1 通过注册 schema 校验",
                       metrics=f_metrics or {"schema": READINESS_SCHEMA, "valid": True},
                       evidence={"schema_name": READINESS_SCHEMA,
                                 "schema_fingerprint":
                                     _rs.schema_fingerprint()})
    pre = [crit_a, crit_b, crit_c, crit_d, crit_e, crit_f]
    overall, blocking = K.evaluate_g(pre, node_verdicts)
    crit_g = K.Verdict(
        "g", S.CRITERIA["g"], overall,
        ("全部判据 PASS，dispatcher_ready=true" if overall == K.PASS else
         f"{sum(1 for c in pre if c.status != K.PASS)} 条判据未通过 → honest FAIL，"
         f"{len(blocking)} 条阻塞项落清单"),
        metrics={"overall_status": overall, "blocking_items": len(blocking),
                 "criteria_passed": sum(1 for c in pre if c.status == K.PASS)},
        evidence={"blocking": blocking[:50]},
    )
    criteria = pre + [crit_g]
    nodes = [node_verdicts[ext].as_dict() for ext in S.NODE_BY_ID]

    payload = {
        "schema_name": "campaign_readiness",
        "schema_version": 1,
        "campaign_id": S.CAMPAIGN_ID,
        "gate_id": S.GATE_ID,
        "plan_id": int(plan_id or (snapshot.get("plan") or {}).get("id") or S.PLAN_ID),
        "generated_utc": generated_utc or datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "dispatcher_ready": overall == K.PASS and not blocking,
        "criteria": [c.as_dict() for c in criteria],
        "nodes": nodes,
        "blocking": blocking,
        "regression": [
            {"case": c["case"], "status": c["status"], "baseline": c.get("baseline"),
             "measured": c.get("measured"), "rel_dev": c.get("rel_dev"),
             "threshold": c["threshold"], "floor": c["floor"], "detail": c["detail"],
             "elapsed_s": c.get("elapsed_s")}
            for c in regression_cases],
        "provenance": {
            "inputs": _provenance_inputs(
                inputs or {},
                fallback_digest=__import__("hashlib").sha256(
                    json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode()).hexdigest()),
            "generator": GENERATOR,
            "schema_fingerprint": _rs.schema_fingerprint(),
        },
        "limitations": list(LIMITATIONS),
    }
    return payload


def validate_readiness(payload: dict) -> tuple[bool, str]:
    """Criterion (f): the payload must satisfy the registered schema."""
    try:
        _rs.validate_artifact(READINESS_SCHEMA, payload)
    except SchemaError as exc:
        return False, f"schema 校验失败: {exc}"
    return True, (f"{READINESS_SCHEMA} 校验通过 "
                  f"(fingerprint {_rs.schema_fingerprint()[:16]}…)")


def run_gate(snapshot: dict, *, scans: dict | None = None, regression_records: dict | None = None,
             plan_id: int | None = None, generated_utc: str | None = None,
             inputs: dict | None = None) -> dict:
    """Evaluate all criteria and return the validated readiness payload."""
    cases, _ = R.evaluate_regression(regression_records or {})
    draft = build_readiness(snapshot, scans=scans or {}, regression_cases=cases,
                            plan_id=plan_id, generated_utc=generated_utc, inputs=inputs)
    ok, detail = validate_readiness(draft)
    if ok:
        return draft
    fallback = build_readiness(snapshot, scans=scans or {}, regression_cases=cases,
                               plan_id=plan_id, generated_utc=generated_utc, inputs=inputs,
                               f_status=K.FAIL, f_detail=detail,
                               f_metrics={"schema": READINESS_SCHEMA, "valid": False,
                                          "error": detail})
    fallback["criteria"][5]["evidence"]["first_attempt_validation"] = detail
    ok2, detail2 = validate_readiness(fallback)
    fallback["criteria"][5]["metrics"]["fallback_valid"] = ok2
    fallback["criteria"][5]["detail"] = f"{detail}; 回退载荷校验: {detail2}"
    return fallback


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------
def _cell(status: str) -> str:
    return {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "WITHHELD": "⚪ WITHHELD"}.get(status, status)


def render_markdown(payload: dict) -> str:
    crit = {c["criterion_id"]: c for c in payload["criteria"]}
    lines = [
        f"# {S.GATE_ID} 全战役集成验收门 — campaign_readiness",
        "",
        f"- schema: `{READINESS_SCHEMA}` (fingerprint "
        f"`{payload['provenance']['schema_fingerprint'][:16]}…`)",
        f"- campaign: `{payload['campaign_id']}` / plan_id={payload['plan_id']}",
        f"- 生成时间 (UTC): `{payload['generated_utc']}`",
        f"- **总裁决: `{payload['overall_status']}`；dispatcher_ready = "
        f"`{str(payload['dispatcher_ready']).lower()}`**",
        "",
        "## 1. 判据总览",
        "",
        "| 判据 | 内容 | 裁决 | 说明 |",
        "|---|---|---|---|",
    ]
    for c in payload["criteria"]:
        lines.append(f"| ({c['criterion_id']}) | {c['label']} | {_cell(c['status'])} | "
                     f"{c['detail']} |")
    lines += ["", "## 2. 节点就绪矩阵", "",
              "| 节点 | mission | 仓库 | 物理域 | 产物 | 台账 | 成熟化 | 链就绪 |",
              "|---|---|---|---|---|---|---|---|"]
    for n in payload["nodes"]:
        lines.append(
            f"| {n['ext_id']} | {n['mission_id']} | {n['repo']} | {n['physics_domain']} | "
            f"{_cell(n['deliverable'])} | {_cell(n['registered'])} | "
            f"{_cell(n['maturity'])} | {'✅' if n['chain_ready'] else '❌'} |")
    lines += ["", "## 3. 回归基线 (criterion c)", "",
              "| 案例 | 基线 | 复跑 | 相对偏差 | 阈值 | 裁决 |",
              "|---|---|---|---|---|---|"]
    for r in payload["regression"]:
        base = "—" if r["baseline"] is None else f"{r['baseline']:.6g}"
        meas = "—" if r["measured"] is None else f"{r['measured']:.6g}"
        dev = "—" if r["rel_dev"] is None else f"{r['rel_dev']:.3e}"
        lines.append(f"| {r['case']} | {base} | {meas} | {dev} | {r['threshold']:.1%} | "
                     f"{_cell(r['status'])} |")
    for r in payload["regression"]:
        if r["status"] != "PASS":
            lines.append(f"- `{r['case']}`: {r['detail']}")
    lines += ["", "## 4. 阻塞清单 (blocking)", ""]
    if payload["blocking"]:
        lines += [f"- `{b}`" for b in payload["blocking"]]
    else:
        lines.append("- 无")
    lines += ["", "## 5. 依赖链 (criterion d)", "",
              f"- 声明 in-campaign 边: {crit['d']['metrics'].get('declared_edges')}，"
              f"平台记录边: {crit['d']['metrics'].get('platform_edges')}",
              f"- 无环: {crit['d']['metrics'].get('acyclic')}；"
              f"孤立节点: {crit['d']['metrics'].get('orphans') or '无'}",
              f"- 链就绪节点: {crit['d']['metrics'].get('chain_ready_nodes') or '无'}",
              f"- 被阻塞边: {len(crit['d']['metrics'].get('blocked_edges') or [])}"]
    lines += ["", "## 6. 溯源 (provenance)", ""]
    for item in payload["provenance"]["inputs"]:
        lines.append(f"- `{item['path']}` present={item['present']} "
                     f"sha256={str(item['sha256'])[:16]}")
    lines += ["", "## 7. 局限", ""] + [f"- {lim}" for lim in payload["limitations"]]
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, out_dir: str | Path) -> dict:
    """Write ``campaign_readiness.json`` + ``.md`` and return their provenance."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    jp = d / READINESS_JSON
    mp = d / READINESS_MD
    jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    mp.write_text(render_markdown(payload))
    return {
        "readiness_json": {"path": str(jp), "sha256": C.sha256_file(jp)},
        "readiness_md": {"path": str(mp), "sha256": C.sha256_file(mp)},
    }