"""m5213 评审产物回归测试：离线校验独立复算记录与报告快照的自洽性。

不访问网络：所有断言只针对 worktree 内的证据产物
(`artifacts/m5213/verification.json` 由 `scripts/verify_m5213_peer_review.py` 现场生成)。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = json.loads((ROOT / "artifacts/m5213/verification.json").read_text(encoding="utf-8"))
MD = (ROOT / "artifacts/m5213/reviewed_report_5918.md").read_text(encoding="utf-8")

REPORTED_DIGEST = "5022e51af33a2b5bb137434a6ae4ca8d1f55d821e686e9156cedde11c2c616e5"


def test_recomputed_core_numbers():
    r = V["recomputed"]
    assert round(r["ct"], 6) == 0.024675
    assert round(r["wetted_suboff"], 2) == 5.18
    assert round(r["wetted_kcs"], 2) == 9651.49
    assert r["eexi_status"] == 404 and r["cii_status"] == 404


def test_derived_errors_are_consistent_with_raw_values():
    d, r = V["derived"], V["recomputed"]
    assert d["suboff_ct_err_pct"] == round((r["ct"] - 0.0032) / 0.0032 * 100, 4)
    assert d["kcs_wetted_err_pct"] == round((r["wetted_kcs"] - 9424.0) / 9424.0 * 100, 4)
    assert d["suboff_wetted_err_pct"] == round((r["wetted_suboff"] - 6.328) / 6.328 * 100, 4)
    # 报告声称 +671.09% / +2.41% / -18.14%
    assert abs(d["suboff_ct_err_pct"] - 671.09) < 0.02
    assert abs(d["kcs_wetted_err_pct"] - 2.41) < 0.02
    assert abs(d["suboff_wetted_err_pct"] + 18.14) < 0.02


def test_no_recompute_disagreement_but_one_audit_gap():
    assert V["n_recompute_fail"] == 0, [c for c in V["checks"] if not c["pass"] and c["kind"] == "recompute"]
    gaps = [c["id"] for c in V["checks"] if not c["pass"] and c["kind"] == "audit_gap"]
    assert gaps == ["E1"], gaps


def test_report_body_digest_is_not_reproducible():
    sha = hashlib.sha256(MD.encode()).hexdigest()
    assert sha == V["derived"]["sha256_published_body"]
    assert sha != REPORTED_DIGEST  # 已记入 findings F6
    assert REPORTED_DIGEST in MD  # 报告确实在正文里声明了这个摘要


def test_structural_counts_reproduced_from_snapshot():
    def rows(start: str, end: str) -> list[str]:
        chunk = MD.split(start)[1].split(end)[0]
        return [ln for ln in chunk.splitlines() if re.match(r"^\| IN19-T\d", ln)]

    matrix = rows("## 3 验收矩阵", "## 4")
    errs = rows("## 4 误差判定表", "## 5")
    roots = rows("## 6 根因分析", "## 7")
    lineage = rows("## 7 谱系继承裁决", "## 8")
    assert len(matrix) == 49 and len(errs) == 9 and len(roots) == 9 and len(lineage) == 9
    assert len({ln.split("|")[1].strip() for ln in matrix}) == 9
    assert all("WITHHELD" in ln for ln in matrix)  # 无 WITHHELD 升格为 PASS
    assert sum(1 for ln in errs if ln.rstrip().endswith("withheld |")) == 6
    assert sum(1 for ln in lineage if "迭代" in ln) == 9
    assert not any("成熟" in ln or "搁置" in ln for ln in lineage)
    # §5 探针表把门侧探针与节点交付显式区分
    probe = [ln for ln in MD.split("## 5 门外独立探针")[1].split("## 6")[0].splitlines()
             if re.match(r"^\| P\d-", ln)]
    assert len(probe) == 5
    assert "不替代节点验收" in MD

def test_review_document_embeds_parseable_json_matching_evidence():
    """评审正文内嵌的机器可解析 JSON 必须与现场复算记录一致（防手抄漂移）。"""
    doc = (ROOT / "docs/peer_review/m5213_IN19-G19_peer_review.md").read_text(encoding="utf-8")
    block = re.search(r"```json\n(.*?)\n```", doc, re.S)
    assert block, "评审正文缺少 ```json 机器可解析块"
    payload = json.loads(block.group(1))
    for key in ("reviewer_agent", "reviewer_affiliation", "independent", "reviewed_digest",
                "verdict", "findings", "requested_changes", "recomputed"):
        assert key in payload, key
    assert payload["independent"] is True
    assert payload["reviewed_digest"] == REPORTED_DIGEST
    assert payload["verdict"] in {"accept", "accept_with_minor", "reject"}
    assert payload["recomputed"]["ct"] == V["recomputed"]["ct"]
    assert payload["recomputed"]["wetted_suboff"] == V["recomputed"]["wetted_suboff"]
    assert payload["recomputed"]["wetted_kcs"] == V["recomputed"]["wetted_kcs"]
    assert payload["recomputed"]["eexi_status"] == V["recomputed"]["eexi_status"] == 404
    assert payload["recomputed"]["cii_status"] == V["recomputed"]["cii_status"] == 404
    # 若复算与报告不符即必须 reject/accept_with_minor —— 这里复算全对，故给 accept_with_minor
    if V["n_recompute_fail"] != 0:
        assert payload["verdict"] in {"reject", "accept_with_minor"}
    else:
        assert payload["verdict"] in {"accept", "accept_with_minor"}
    assert "未参与 plan_id=39" in payload["reviewer_affiliation"]
