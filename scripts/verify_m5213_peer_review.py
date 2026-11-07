#!/usr/bin/env python3
"""m5213 —— 对 mission_id=5208（TensorPhysics 波3 核销门 IN19-G19）报告的独立复算脚本。

本脚本由**未参与 plan_id=39（306–314 节点）**的评审方编写，只读、可重放：
  1. 直接调用 ship-performance-platform（:8002）端点，复算 SUBOFF / KCS 两个探针工况；
  2. 复算 EEXI / CII 端点可达性（能力缺席判定）；
  3. 对**被评审报告正文快照**做结构计数（节点数/判据数/误差表行/根因/继承）；
  4. 复核报告自称的正文摘要 sha256 是否可由发布正文复现。

用法：
    python3 scripts/verify_m5213_peer_review.py \
        [--ship http://127.0.0.1:8002] \
        [--report artifacts/m5213/reviewed_report_5918.md] \
        [--out artifacts/m5213/verification.json]

判据全部为「报告声称值 vs 本脚本实测值」的余量检查，不依赖报告的任何中间产物。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---- 报告声称值（逐字取自被评审报告 id=5918 正文）----------------------------
CLAIM = {
    "digest": "5022e51af33a2b5bb137434a6ae4ca8d1f55d821e686e9156cedde11c2c616e5",
    "suboff_ct": 0.024675,          # §4 IN19-T6 / §5 P2-SUBOFF-CT
    "suboff_ct_ref": 0.0032,
    "suboff_ct_err_pct": 671.09,
    "suboff_wetted": 5.18,          # §5 P1-SUBOFF-S
    "suboff_wetted_ref": 6.328,
    "suboff_wetted_err_pct": -18.14,
    "kcs_wetted": 9651.49,          # §5 P3-KCS-S（§4 显示为 9651）
    "kcs_wetted_ref": 9424.0,
    "kcs_wetted_err_pct": 2.41,
    "n_nodes": 9,
    "n_criteria": 49,
    "n_error_rows": 9,
    "n_error_measured": 3,
    "n_error_withheld": 6,
    "n_rootcause": 9,
    "n_lineage": 9,
    "n_mature": 0,
    "n_iterate": 9,
    "n_shelve": 0,
}

SUBOFF_BODY = {
    "length": 4.356, "beam": 0.508, "draft": 0.508, "displacement": 0.718,
    "speed": 5.0, "speed_unit": "m/s", "cb": 0.639,
    "water_density": 1025.9, "water_viscosity": 1.05e-6,
}
KCS_BODY = {
    "length": 230.0, "beam": 32.2, "draft": 10.8, "displacement": 52030.0,
    "speed": 12.0, "speed_unit": "kn", "cb": 0.651,
}


def http(method: str, url: str, body: dict | None = None) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode(), e.code
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def pct(measured: float, reference: float) -> float:
    return (measured - reference) / reference * 100.0


def section_rows(md: str, start: str, end: str, pattern: str) -> list[str]:
    chunk = md.split(start)[1].split(end)[0]
    return [ln for ln in chunk.splitlines() if re.match(pattern, ln)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship", default="http://127.0.0.1:8002")
    ap.add_argument("--report", default="artifacts/m5213/reviewed_report_5918.md")
    ap.add_argument("--out", default="artifacts/m5213/verification.json")
    a = ap.parse_args()

    checks: list[dict] = []

    def check(cid: str, what: str, ok: bool, claimed, measured, note: str = "",
              kind: str = "recompute") -> None:
        # kind="recompute"  —— 数值/结构复算，不通过即评审拒绝；
        # kind="audit_gap"  —— 可审计性缺陷，记入 findings 但不改变复算结论。
        checks.append({
            "id": cid, "what": what, "pass": bool(ok), "kind": kind,
            "claimed": claimed, "measured": measured, "note": note,
        })
        print(f"[{'PASS' if ok else 'FAIL'}] {cid} {what}: claimed={claimed} measured={measured}")

    # ---- A. SUBOFF 探针：C_T 与湿表面积 --------------------------------------
    st, body = http("POST", f"{a.ship}/api/v1/calculations/resistance", SUBOFF_BODY)
    sub = body if isinstance(body, dict) else {}
    sub_ct, sub_s = sub.get("ct"), sub.get("wetted_surface")
    sub_err = pct(sub_ct, CLAIM["suboff_ct_ref"]) if sub_ct else None
    check("A1", "SUBOFF C_T (V=5 m/s) 复算=报告值", st == 200 and abs(sub_ct - CLAIM["suboff_ct"]) < 1e-9,
          CLAIM["suboff_ct"], sub_ct)
    check("A2", "SUBOFF 湿表面积 S 复算=报告值", abs(sub_s - CLAIM["suboff_wetted"]) < 1e-9,
          CLAIM["suboff_wetted"], sub_s)
    check("A3", "SUBOFF C_T 相对基准 0.0032 的误差", abs(sub_err - CLAIM["suboff_ct_err_pct"]) < 0.02,
          CLAIM["suboff_ct_err_pct"], round(sub_err, 4), "误差按未圆整实测值计算")
    check("A4", "SUBOFF 工况越域标记（Fn>0.45 Holtrop 适用域上限）",
          sub.get("froude_number", 0) > 0.45, "Fn 越域已由报告 §10.3 声明",
          round(sub.get("froude_number", 0), 4),
          "±671% 是越域外推合成量，报告已声明不得单独归因")

    # ---- B. KCS 探针：湿表面积 ----------------------------------------------
    st, body = http("POST", f"{a.ship}/api/v1/calculations/resistance", KCS_BODY)
    kcs = body if isinstance(body, dict) else {}
    kcs_s = kcs.get("wetted_surface")
    kcs_err = pct(kcs_s, CLAIM["kcs_wetted_ref"]) if kcs_s else None
    check("B1", "KCS 湿表面积 S 复算=报告值", st == 200 and abs(kcs_s - CLAIM["kcs_wetted"]) < 1e-6,
          CLAIM["kcs_wetted"], kcs_s)
    check("B2", "KCS S 相对公开积分值 9424 m2 的误差", abs(kcs_err - CLAIM["kcs_wetted_err_pct"]) < 0.02,
          CLAIM["kcs_wetted_err_pct"], round(kcs_err, 4))
    check("B3", "KCS 误差超阈值即判不达标（阈值 1.5%）", kcs_err > 1.5,
          "§4 判「否」", round(kcs_err, 4), "报告未把 2.41% 圆整成达标")

    # ---- C. EEXI / CII 能力缺席 ---------------------------------------------
    eexi_status, _ = http("GET", f"{a.ship}/api/v1/calculations/eexi")
    cii_status, _ = http("GET", f"{a.ship}/api/v1/calculations/cii")
    check("C1", "GET /calculations/eexi 能力缺席", eexi_status == 404, 404, eexi_status)
    check("C2", "GET /calculations/cii 能力缺席", cii_status == 404, 404, cii_status)

    # ---- D. 报告正文结构与自称数字 ------------------------------------------
    md = Path(a.report).read_text()
    matrix = section_rows(md, "## 3 验收矩阵", "## 4", r"^\| IN19-T\d")
    nodes = sorted({ln.split("|")[1].strip() for ln in matrix})
    err_rows = section_rows(md, "## 4 误差判定表", "## 5", r"^\| IN19-T\d")
    rc_rows = section_rows(md, "## 6 根因分析", "## 7", r"^\| IN19-T\d")
    lin_rows = section_rows(md, "## 7 谱系继承裁决", "## 8", r"^\| IN19-T\d")
    withheld = sum(1 for ln in err_rows if ln.rstrip().endswith("withheld |"))
    measured = len(err_rows) - withheld
    lin = collections.Counter(
        "mature" if "成熟" in ln else ("shelve" if "搁置" in ln else "iterate") for ln in lin_rows
    )
    check("D1", "节点数", len(nodes) == CLAIM["n_nodes"], CLAIM["n_nodes"], len(nodes))
    check("D2", "机器判据行数", len(matrix) == CLAIM["n_criteria"], CLAIM["n_criteria"], len(matrix))
    check("D3", "误差判定表行数", len(err_rows) == CLAIM["n_error_rows"], CLAIM["n_error_rows"], len(err_rows))
    check("D4", "误差表 有实测/withheld 行数", (measured, withheld) ==
          (CLAIM["n_error_measured"], CLAIM["n_error_withheld"]),
          [CLAIM["n_error_measured"], CLAIM["n_error_withheld"]], [measured, withheld])
    check("D5", "根因分析行数", len(rc_rows) == CLAIM["n_rootcause"], CLAIM["n_rootcause"], len(rc_rows))
    check("D6", "继承裁决行数", len(lin_rows) == CLAIM["n_lineage"], CLAIM["n_lineage"], len(lin_rows))
    check("D7", "继承分布 成熟/迭代/搁置",
          (lin["mature"], lin["iterate"], lin["shelve"]) ==
          (CLAIM["n_mature"], CLAIM["n_iterate"], CLAIM["n_shelve"]),
          [CLAIM["n_mature"], CLAIM["n_iterate"], CLAIM["n_shelve"]],
          [lin["mature"], lin["iterate"], lin["shelve"]])
    check("D8", "验收矩阵全部为 WITHHELD（无升格为 PASS）",
          all("WITHHELD" in ln for ln in matrix), "49/49 WITHHELD",
          f"{sum(1 for ln in matrix if 'WITHHELD' in ln)}/49")
    c5_row = next((ln for ln in md.splitlines() if ln.startswith("| C5 | report passed peer review")), "")
    check("D9", "门自检 C5 记 FAIL（自证不足未被粉饰）", "**FAIL**" in c5_row,
          "FAIL 0/1 reviewer", c5_row.split("|")[3].strip() if c5_row else "row not found")

    # ---- E. 正文摘要可复现性 -------------------------------------------------
    body_sha = hashlib.sha256(md.encode()).hexdigest()
    check("E1", "发布正文 sha256 == 报告自称摘要", body_sha == CLAIM["digest"],
          CLAIM["digest"], body_sha,
          "不等：报告自称摘要覆盖的是门内部规范载荷，未随正文发布，评审方无法独立复现",
          kind="audit_gap")

    out = {
        "mission_id": 5213,
        "reviewed_mission_id": 5208,
        "reviewed_report_ids": [5918, 5915],
        "reviewed_digest": CLAIM["digest"],
        "ship_api": a.ship,
        "recomputed": {
            "ct": sub_ct,
            "wetted_suboff": sub_s,
            "wetted_kcs": kcs_s,
            "eexi_status": eexi_status,
            "cii_status": cii_status,
        },
        "derived": {
            "suboff_ct_err_pct": round(sub_err, 4),
            "suboff_wetted_err_pct": round(pct(sub_s, CLAIM["suboff_wetted_ref"]), 4),
            "kcs_wetted_err_pct": round(kcs_err, 4),
            "suboff_froude": round(sub.get("froude_number", 0), 4),
            "kcs_froude": round(kcs.get("froude_number", 0), 4),
            "resistance_method": sub.get("method"),
            "sha256_published_body": body_sha,
        },
        "counts": {
            "nodes": len(nodes), "criteria": len(matrix),
            "error_rows": len(err_rows), "error_measured": measured,
            "error_withheld": withheld, "rootcause": len(rc_rows),
            "lineage": {k: int(lin.get(k, 0)) for k in ("mature", "iterate", "shelve")},
        },
        "checks": checks,
        "n_pass": sum(1 for c in checks if c["pass"]),
        "n_fail": sum(1 for c in checks if not c["pass"]),
        "n_recompute_fail": sum(1 for c in checks if not c["pass"] and c["kind"] == "recompute"),
        "n_audit_gap": sum(1 for c in checks if not c["pass"] and c["kind"] == "audit_gap"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n{out['n_pass']} pass / {out['n_fail']} fail "
          f"({out['n_recompute_fail']} recompute, {out['n_audit_gap']} audit_gap) -> {a.out}")
    return 0 if out["n_recompute_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())