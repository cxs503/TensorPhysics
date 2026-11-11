"""Tests for the capability reuse graph (synthetic fixture + real-artefact checks).

The synthetic fixture is a tiny three-repository ecosystem written on the fly:
capability ``alpha`` owns module ``a/core.py``, capability ``beta`` imports it,
capability ``gamma`` never references anything (the orphan), and a fourth module
is imported from a file nobody owns (external reuse).  Every assertion below is
a property of the *pipeline*, not of the production snapshot, so the tests stay
fast and independent of the 1.2 GB of repository checkouts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.capability_reuse.aggregate import (build_graph, degrees,  # noqa: E402
                                                      dual_caliber_compare, pagerank)
from tensorphysics.capability_reuse.audit import run_audit, sample_edges  # noqa: E402
from tensorphysics.capability_reuse.extract import map_capabilities  # noqa: E402
from tensorphysics.capability_reuse.gold import score_extraction  # noqa: E402
from tensorphysics.capability_reuse.hints import build_hints  # noqa: E402
from tensorphysics.capability_reuse.indexer import (PathResolver, build_index,  # noqa: E402
                                                    strip_comments)
from tensorphysics.capability_reuse.model import (SNAPSHOT_SCHEMA, cluster_capabilities,  # noqa: E402
                                                  normalize_capability_name)

FIXTURE_FILES = {
    "RepoA": {
        "src/alpha/__init__.py": "",
        "src/alpha/core.py": "def solve_alpha_kernel(x):\n    return x * 2\n",
        "src/alpha/extra.py": "from .core import solve_alpha_kernel\n\nY = solve_alpha_kernel(3)\n",
        "tests/test_alpha.py": "from alpha.core import solve_alpha_kernel\n",
    },
    "RepoB": {
        "src/beta/__init__.py": "",
        "src/beta/run.py": (
            "from alpha.core import solve_alpha_kernel\n"
            "import beta.helper\n"
            "def go():\n    return solve_alpha_kernel(1)\n"
        ),
        "src/beta/helper.py": "VALUE = 3\n",
        "tests/test_beta.py": "from beta.run import go\n",
    },
    "RepoC": {
        "src/gamma/__init__.py": "",
        "src/gamma/lonely.py": "def lonely_function_here(z):\n    return z\n",
        # a comment-only reference must NOT create an edge
        "src/gamma/misleading.py": "# from alpha.core import solve_alpha_kernel\nX = 1\n",
        "scripts/glue.py": "from alpha.core import solve_alpha_kernel\n",
    },
}


def _write_repo(root: Path, name: str, files: dict) -> dict:
    repo_root = root / name
    entries = []
    for rel, text in files.items():
        p = repo_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        entries.append(rel)
    return {"root": str(repo_root), "head": "fixture00001", "dirty": False,
            "import_roots": ["", "src"], "files": sorted(entries),
            "file_count": len(entries)}


def _snapshot(tmp_path: Path) -> dict:
    repos = {name: _write_repo(tmp_path, name, files) for name, files in FIXTURE_FILES.items()}
    caps = [
        {"cid": "alpha", "node_id": 1, "name": "alpha", "repo": "RepoA", "domain": "kernel",
         "cap_class": "alpha", "mission_id": 1, "status": "done", "origin": "node",
         "done_at": "2026-01-01T00:00:00", "goal": "see src/alpha/core.py",
         "acceptance_criteria": ["pytest tests/test_alpha.py"], "dependency_hints": [],
         "evidence": [{"report_id": 11, "line": 3, "text": "实现见 src/alpha/core.py"}]},
        {"cid": "beta", "node_id": 2, "name": "beta", "repo": "RepoB", "domain": "orchestration",
         "cap_class": "beta", "mission_id": 2, "status": "done", "origin": "node",
         "done_at": "2026-02-01T00:00:00", "goal": "uses src/beta/run.py",
         "acceptance_criteria": [], "dependency_hints": ["alpha"],
         "evidence": [{"report_id": 12, "line": 5, "text": "复用 src/alpha/core.py 的核"}]},
        {"cid": "gamma", "node_id": 3, "name": "gamma", "repo": "RepoC", "domain": "side",
         "cap_class": "gamma", "mission_id": 3, "status": "done", "origin": "node",
         "done_at": "2026-03-01T00:00:00", "goal": "src/gamma/lonely.py",
         "acceptance_criteria": [], "dependency_hints": [],
         "evidence": [{"report_id": 13, "line": 1, "text": "src/gamma/misleading.py"}]},
    ]
    return {"schema": SNAPSHOT_SCHEMA, "repos": repos, "capabilities": caps,
            "provenance": {"snapshot_content_sha256": "fixture"}}


@pytest.fixture()
def fixture_snapshot(tmp_path):
    return _snapshot(tmp_path)


def test_path_resolver_prefers_own_repo_and_flags_ambiguity(tmp_path):
    snap = _snapshot(tmp_path)
    manifest = {r: set(v["files"]) for r, v in snap["repos"].items()}
    res = PathResolver(manifest)
    assert res.resolve("src/alpha/core.py", "RepoA") == ("RepoA", "src/alpha/core.py")
    assert res.resolve("src/alpha/core.py", "RepoB") == ("RepoA", "src/alpha/core.py")
    assert res.resolve("nope/missing.py", "RepoA") is None


def test_capability_mapping_and_determinism(fixture_snapshot):
    repos = build_index(fixture_snapshot)
    caps, ownership, pathmap = map_capabilities(fixture_snapshot, repos)
    assert {c.cid for c in caps} == {"alpha", "beta", "gamma"}
    assert any(p.endswith("src/alpha/core.py") for p in pathmap["alpha"]["code"])
    # the file claimed by two capabilities goes to the earlier deliverable
    assert ownership[("RepoA", "src/alpha/core.py")] == "alpha"
    g1, s1 = build_graph(fixture_snapshot, build_index(fixture_snapshot), None)
    g2, s2 = build_graph(fixture_snapshot, build_index(fixture_snapshot), None)
    assert [e.to_dict() for e in g1.edges] == [e.to_dict() for e in g2.edges]
    assert s1["counts"] == s2["counts"]


def test_edges_reuse_and_orphan_buckets(fixture_snapshot):
    repos = build_index(fixture_snapshot)
    graph, stats = build_graph(fixture_snapshot, repos, None)
    pairs = graph.edge_pairs()
    assert ("beta", "alpha") in pairs          # beta imports alpha.core
    indeg, outdeg, _ = degrees(graph)
    assert indeg["alpha"] >= 1
    assert indeg.get("gamma", 0) == 0          # gamma references nothing
    assert stats["reuse"]["reuse_rate"] == pytest.approx(1 / 3, abs=1e-5)
    # a comment-only reference does not create an edge
    assert ("gamma", "alpha") not in pairs
    # unmapped files are counted as external reuse, not as capability edges
    assert stats["extraction"]["external_refs"] >= 1
    assert stats["undetermined"] == []   # every fixture capability maps to code
    assert stats["counts"]["determined"] + stats["counts"]["undetermined"] == stats["counts"]["capabilities"]


def test_dual_caliber_reports_disagreements(fixture_snapshot):
    repos = build_index(fixture_snapshot)
    caps, ownership, pathmap = map_capabilities(fixture_snapshot, repos)
    from tensorphysics.capability_reuse.extract import extract_code_edges
    res = extract_code_edges(fixture_snapshot, repos, ownership, full_universe=False, workers=1)
    cmp = dual_caliber_compare(res.ast_edges, res.ref_edges)
    assert 0.0 <= cmp["capability_level"]["jaccard"] <= 1.0
    assert isinstance(cmp["capability_level"]["ast_only"], list)
    assert isinstance(cmp["usage_level"]["jaccard"], float)


def test_seeding_hints_schema(fixture_snapshot):
    repos = build_index(fixture_snapshot)
    graph, stats = build_graph(fixture_snapshot, repos, None)
    hints = build_hints(stats, top_k=5)
    assert hints["schema"].endswith("seeding-hints/1")
    assert hints["orphans"] and hints["orphans"][0]["recommendation"].startswith("add_")
    assert "reason" in hints["orphans"][0]
    assert isinstance(hints["summary"]["reuse_rate"], float)
    assert pagerank(graph)  # ranks every capability deterministically


def test_cluster_and_name_normalisation():
    names = {
        "a": (normalize_capability_name("p190-n01 zalesak benchmark"), "R"),
        "b": (normalize_capability_name("P190-N01: Zalesak Benchmark"), "R"),
        "c": (normalize_capability_name("完全不同 能力"), "R"),
    }
    cluster = cluster_capabilities(names, threshold=0.9)
    assert cluster["a"] == cluster["b"] != cluster["c"]
    assert normalize_capability_name("P190-N01: Zalesak Benchmark") == normalize_capability_name(
        "zalesak benchmark")


def test_strip_comments_keeps_docstrings():
    text = '"""doc mentions tensorcad.nurbs"""\n# from alpha.core import x\nimport os\n'
    out = strip_comments(text)
    assert "tensorcad.nurbs" in out
    assert "from alpha.core" not in out
    assert "import os" in out


def test_gold_scoring_and_sampling():
    gold = {"schema": "capability-reuse-gold/1", "edges": [
        {"src": "a", "dst": "b", "label": "positive", "evidence": ["f:1"]},
        {"src": "c", "dst": "d", "label": "positive", "evidence": ["g:2"]},
        {"src": "e", "dst": "f", "label": "negative"},
    ]}
    score = score_extraction(gold, [("a", "b"), ("e", "f")])
    assert score["precision"] == pytest.approx(0.5)
    assert score["recall"] == pytest.approx(0.5)
    assert score["spurious"] == [["e", "f"]]
    sample = sample_edges([("a", "b"), ("c", "d"), ("e", "f")], n=2, salt="x")
    assert sample == sample_edges([("a", "b"), ("c", "d"), ("e", "f")], n=2, salt="x")
    assert len(sample) == 2


def test_audit_marks_unknown_when_sample_not_reviewed(tmp_path):
    stats = {"counts": {"capabilities": 3, "determined": 2, "undetermined": 1, "edges": 1},
             "per_repo": {"R": {"capabilities": 3, "determined": 2, "reused": 1}},
             "calibers": {"capability_level": {"jaccard": 0.9, "ast_edges": 1, "ref_edges": 1,
                                               "intersection": 1, "union": 1, "ast_only": [],
                                               "ref_only": []}},
             "reuse": {"reuse_rate": 0.5, "reused_capabilities": 1, "orphan_capabilities": 1,
                       "undetermined_excluded": 1},
             "cap_repo": {"a": "R", "b": "R"}, "cap_class": {"a": {"cap_class": "x"},
                                                            "b": {"cap_class": "y"}}}
    criteria = json.loads((ROOT / "src/tensorphysics/capability_reuse/configs/audit_criteria.json")
                          .read_text())["criteria"]
    gold_score = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "gold_edges": 31, "positives": 25,
                  "negatives": 6, "repos_covered": 4, "repos": ["R"], "true_positives": 25,
                  "false_negatives": 0, "false_positives_on_negatives": 0, "missing": [],
                  "spurious": []}
    report = run_audit(stats=stats, gold_score=gold_score, gold={"authorship": "t"}, criteria=criteria,
                       determinism={"runs": 2, "identical": True}, sample_review=None,
                       hints={"schema": "x", "orphans": [], "hubs": [], "gaps": [],
                              "summary": {"orphans": 0, "hubs": 0, "gaps": 0, "reuse_rate": 0.5}},
                       snapshot_hash="deadbeef")
    assert report["e_edge_sample"]["pass"] is None      # not_run is NOT passed
    assert "e" in report["verdict"]["unknown"]
    assert report["a_gold_standard"]["pass"] is True
    assert report["verdict"]["status"] == "FAIL"         # (d) coverage fails on 1 repository


REAL = ROOT / "artifacts/capability_reuse/audit_report.json"


@pytest.mark.skipif(not REAL.exists(), reason="production artifacts not present")
def test_production_audit_is_internally_consistent():
    """The audit must *record* every criterion and its verdict truthfully.

    The production artefact set may honestly fail a criterion (the dual-calibre
    Jaccard and the 20-edge review are the two that did); what must never happen
    is a criterion reported as passed while its number violates the threshold,
    or a not-run check reported as passed.
    """
    report = json.loads(REAL.read_text())
    crit = report["criteria"]
    b = report["b_dual_caliber"]
    assert b["pass"] == (b["jaccard"] >= crit["dual_caliber_jaccard_min"])
    assert len(b["ast_only"]) + len(b["ref_only"]) == b["disagreements"]
    e = report["e_edge_sample"]
    if e["false_positive_rate"] is None:
        assert e["pass"] is None and "e" in report["verdict"]["unknown"]
    else:
        assert e["pass"] == (e["false_positive_rate"] <= crit["sample_false_positive_max"])
    c = report["c_determinism"]
    assert c["pass"] == (c["runs"] >= 2 and c["identical"] is True)
    a = report["a_gold_standard"]
    assert a["gold_edges"] == a["positives"] + a["negatives"]
    assert report["d_coverage"]["bucket_separation"]["orphan_denominator_is_determined_only"]
    failed = {k for k, row in report.items()
              if isinstance(row, dict) and row.get("pass") is False}
    assert report["verdict"]["status"] == ("FAIL" if failed else "PASS")


@pytest.mark.skipif(not (ROOT / "artifacts/capability_reuse/seeding_hints.json").exists(),
                    reason="production hints not present")
def test_production_hints_and_edges_are_consistent():
    hints = json.loads((ROOT / "artifacts/capability_reuse/seeding_hints.json").read_text())
    edges = json.loads((ROOT / "artifacts/capability_reuse/edges.json").read_text())
    indeg = {}
    for e in edges["edges"]:
        indeg[e["dst"]] = indeg.get(e["dst"], 0) + 1
    for row in hints["hubs"][:5]:
        assert row["in_degree"] == indeg[row["capability"]]
    for row in hints["orphans"][:5]:
        assert "reason" in row and row["capability"] not in indeg
        assert row["recommendation"] in ("add_test", "add_doc", "add_build", "advertise_reuse")