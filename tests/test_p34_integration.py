"""Tests for the P34-G1 campaign integration gate.

The suite is deliberately hermetic: no network, no node worktree, no platform DB.
It builds a synthetic "perfect campaign" in ``tmp_path`` and then breaks exactly
one thing at a time, asserting that the gate reacts with the intended verdict
(PASS / FAIL / WITHHELD) instead of with silence.  A gate that can be talked into
a PASS is worthless, so the negative cases are the point of this file.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign.p34_gate import (  # noqa: E402
    FAIL, NODES, PASS, WITHHELD, check_json, check_png, collect_node,
    evaluate_c1, evaluate_c2, evaluate_c3, evaluate_c4, evaluate_c5,
    run_gate, run_node_tests, scan_imports, scan_lexicon, scan_placeholders)
from tensorphysics.campaign.p34_gate import regression as R  # noqa: E402
from tensorphysics.campaign.p34_gate.collect import PytestRecord  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 220 + b"IEND\xaeB`\x82"


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG)
    return path


def _json(path: Path, payload: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {"ok": True}), encoding="utf-8")
    return path


def _node_ev(ext_id: str, passed: int = 20, green: bool = True,
             worktree: str | None = "/tmp/x", artifacts=None) -> dict:
    return {
        "ext_id": ext_id,
        "mission_id": 0,
        "repo": "r",
        "title": "t",
        "worktree": worktree,
        "min_tests": 10,
        "pytest": {"command": "pytest x", "exit_code": 0 if green else 1,
                   "passed": passed, "failed": 0 if green else 2, "errors": 0,
                   "collected": passed, "timed_out": False, "green": green,
                   "test_files_run": ["tests/test_x.py"], "junit": "j.xml",
                   "head_sha": "0" * 40},
        "artifacts": artifacts if artifacts is not None else [
            {"family": "a/*.json", "kind": "json", "n_matches": 1, "ok": True,
             "files": [{"path": "a/x.json", "ok": True}]},
            {"family": "a/*.png", "kind": "png", "n_matches": 1, "ok": True,
             "files": [{"path": "a/x.png", "ok": True}]},
        ],
        "gaps": [],
    }


def _all_nodes(passed: int = 20) -> list[dict]:
    return [_node_ev(n.ext_id, passed=max(passed, n.min_tests)) for n in NODES]


def _regression(red: str | None = None) -> dict:
    from tensorphysics.campaign.p34_gate.spec import REQUIRED_REGRESSION_FAMILIES
    suites = []
    for fam in REQUIRED_REGRESSION_FAMILIES:
        bad = fam == red
        suites.append({"family": fam, "target": f"{fam} suite", "repo": "r",
                       "exit_code": 1 if bad else 0, "passed": 40, "failed": 2 if bad else 0,
                       "errors": 0, "collected": 42, "skipped": 0,
                       "arch": "loongarch64", "head_sha": "0" * 40,
                       "junit": "j.xml", "duration_s": 1.0})
    return {"schema": "p34_regression_report.v1", "suites": suites}


_SIX_ELEMENTS = """# P34 战役总结报告

## 1. 案例介绍
本战役以 KVLCC2 与 Suboff 为对象，几何参数与来流条件见表 1，物理背景为跨域能力串联验收。
补充说明以凑足正文长度，避免结构检查把小节误判为空洞文本。

## 2. 怎么做的
方法部分描述证据采集、回归切片与判据重算流程，工具链为 pytest + junit 解析 + sha256 审计链。

## 3. 计算结果
计算结果的数字表由程序从裁决 JSON 注入，逐项给出判据数值与限值，不做手工抄写。

## 4. 结果图
![裁决总览](artifacts/p34_integration/fig1_criteria.png)
图注说明该图由 matplotlib 由裁决 JSON 程序化生成。

## 5. 注意点
局限：C1 只覆盖本次采集时的 pytest 快照；节点后续改分支需重跑门；C5 仅扫描代码不含散文。

## 6. 误差判定
关键量对比基准并逐项列出，最大偏差 1.8%，最小偏差 0.4%，样本量 12 个工况，
基准来自文献与解析解，逐项误差表见产物 JSON。显式判定：是否<3% —— 是，最大 1.8%。
"""


# --------------------------------------------------------------------------
# spec / format primitives
# --------------------------------------------------------------------------
def test_spec_has_seven_nodes_with_unique_missions():
    assert len(NODES) == 7
    assert len({n.ext_id for n in NODES}) == 7
    assert len({n.mission_id for n in NODES}) == 7
    for n in NODES:
        assert n.min_tests >= 1
        assert n.tests, n.ext_id
        kinds = {a.kind for a in n.artifacts}
        assert "json" in kinds, n.ext_id
        assert any(a.kind == "png" for a in n.artifacts), n.ext_id


def test_png_validation_rejects_fake_and_truncated(tmp_path):
    good = _png(tmp_path / "ok.png")
    assert check_png(good)["ok"] is True
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"not a png at all" * 40)
    assert check_png(fake)["ok"] is False
    trunc = tmp_path / "trunc.png"
    trunc.write_bytes(PNG[:60])
    assert check_png(trunc)["ok"] is False


def test_json_validation_rejects_broken_json(tmp_path):
    assert check_json(_json(tmp_path / "a.json"))["ok"] is True
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert check_json(bad)["ok"] is False


# --------------------------------------------------------------------------
# C1
# --------------------------------------------------------------------------
def test_c1_green_nodes_pass():
    v = evaluate_c1(_all_nodes())
    assert v.status == PASS
    assert v.metrics["n_pass"] == 7


def test_c1_self_declared_green_with_too_few_tests_fails():
    """A node cannot buy a pass with a green run of fewer tests than promised."""
    nodes = _all_nodes()
    nodes[0]["pytest"].update({"passed": 3, "collected": 3})
    v = evaluate_c1(nodes)
    assert v.status == FAIL
    assert "promised" in v.evidence["nodes"][0]["reason"]


def test_c1_missing_run_is_withheld_not_passed():
    nodes = _all_nodes()
    nodes[1]["pytest"] = {"command": "", "green": False}
    v = evaluate_c1(nodes)
    assert v.status == WITHHELD
    assert v.metrics["n_withheld"] == 1


def test_c1_red_and_timeout_fail():
    nodes = _all_nodes()
    nodes[2]["pytest"].update({"green": False, "failed": 2, "exit_code": 1})
    assert evaluate_c1(nodes).status == FAIL
    nodes2 = _all_nodes()
    nodes2[2]["pytest"].update({"timed_out": True, "exit_code": 124, "green": False})
    v = evaluate_c1(nodes2)
    assert v.status == FAIL and "timed out" in v.evidence["nodes"][2]["reason"]


# --------------------------------------------------------------------------
# C2
# --------------------------------------------------------------------------
def test_c2_complete_artifacts_pass():
    assert evaluate_c2(_all_nodes()).status == PASS


def test_c2_partial_artifacts_fail_and_absent_withheld():
    nodes = _all_nodes()
    nodes[0]["artifacts"][1].update({"ok": False, "n_matches": 1,
                                    "files": [{"path": "a/x.png", "ok": False,
                                               "note": "magic=False"}]})
    assert evaluate_c2(nodes).status == FAIL
    empty = _all_nodes()
    empty[0]["artifacts"] = [{"family": "a/*.json", "kind": "json", "n_matches": 0,
                              "ok": False, "files": []}]
    assert evaluate_c2(empty).status == WITHHELD


# --------------------------------------------------------------------------
# C3
# --------------------------------------------------------------------------
def test_c3_all_families_green_pass():
    v = evaluate_c3(_regression())
    assert v.status == PASS and v.metrics["n_red"] == 0


def test_c3_red_suite_fails_and_missing_family_is_withheld():
    assert evaluate_c3(_regression(red="seakeeping")).status == FAIL
    reg = _regression()
    reg["suites"] = [s for s in reg["suites"] if s["family"] != "stability"]
    v = evaluate_c3(reg)
    assert v.status == WITHHELD
    assert "stability" in v.metrics["families_missing"]
    assert evaluate_c3(None).status == WITHHELD


# --------------------------------------------------------------------------
# C4
# --------------------------------------------------------------------------
def test_c4_six_elements_pass(tmp_path):
    root = tmp_path
    _png(root / "artifacts/p34_integration/fig1_criteria.png")
    report = root / "docs/p34_campaign_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_SIX_ELEMENTS, encoding="utf-8")
    v = evaluate_c4(report, root)
    assert v.status == PASS, v.detail
    assert v.metrics["n_sections"] == 6 and v.metrics["n_figures"] == 1


def test_c4_subsections_count_toward_their_section(tmp_path):
    root = tmp_path
    _png(root / "artifacts/p34_integration/fig1_criteria.png")
    report = root / "docs/p34_campaign_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    body = _SIX_ELEMENTS.replace(
        "## 3. 计算结果\n",
        "## 3. 计算结果\n\n### 3.1 判据裁决\n逐项裁决见表。\n\n### 3.2 节点级结果\n节点表如下。\n")
    report.write_text(body, encoding="utf-8")
    v = evaluate_c4(report, root)
    assert v.status == PASS, v.detail


def test_c4_flags_missing_section_broken_figure_and_vague_error_verdict(tmp_path):
    root = tmp_path
    report = root / "docs/p34_campaign_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    text = (_SIX_ELEMENTS.replace("## 5. 注意点", "## 5. 其他")
            .replace("局限：C1 只覆盖", "说明：C1 只覆盖").replace("<3%", "很小"))
    report.write_text(text, encoding="utf-8")
    v = evaluate_c4(report, root)
    assert v.status == FAIL
    assert "notes" in v.metrics["missing"]
    assert v.metrics["broken_figures"]
    assert v.metrics["explicit_lt_3pct_verdict"] is False
    assert evaluate_c4(None, root).status == WITHHELD


# --------------------------------------------------------------------------
# C5
# --------------------------------------------------------------------------
def test_c5_gate_package_itself_is_clean():
    """The gate must satisfy its own zero-domain-semantics rule."""
    pkg = ROOT / "src/tensorphysics/campaign/p34_gate"
    v = evaluate_c5(pkg, [])
    assert v.status == PASS, v.detail


def test_c5_detects_placeholder_forbidden_import_and_lexicon(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "logic.py").write_text(
        "import numpy\n\ndef f():\n    return '阻力 TODO'\n", encoding="utf-8")
    v = evaluate_c5(pkg, [])
    assert v.status == FAIL
    assert v.metrics["n_placeholder_hits"] >= 1
    assert v.metrics["n_lexicon_hits"] >= 1
    assert "numpy" in v.evidence["forbidden_imports"]
    imp = scan_imports(pkg)
    assert imp["ok"] is False and "numpy" in imp["forbidden"]
    assert scan_lexicon([pkg / "logic.py"])["ok"] is False
    assert scan_placeholders([pkg / "logic.py"])


# --------------------------------------------------------------------------
# node worktree collection (real subprocess pytest on a synthetic repo)
# --------------------------------------------------------------------------
def _synthetic_node_repo(tmp_path: Path, passing: bool = True) -> Path:
    spec = NODES[0]
    root = tmp_path / "node"
    (root / "src/tensorlbm/wave_tank").mkdir(parents=True)
    (root / "src/tensorlbm/wave_tank/__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "benchmarks/verified/wave_tank").mkdir(parents=True)
    (root / "benchmarks/verified/wave_tank/README.md").write_text("x", encoding="utf-8")
    (root / "tests").mkdir()
    body = ("def test_ok():\n    assert 1 + 1 == 2\n" if passing else
            "def test_bad():\n    assert 1 + 1 == 3\n")
    (root / spec.tests[0]).write_text(body, encoding="utf-8")
    _json(root / "artifacts/wave_tank/summary.json", {"kh": [0.7, 1.5]})
    _png(root / "artifacts/wave_tank/waveform.png")
    return root


def test_collect_node_runs_pytest_and_validates_artifacts(tmp_path):
    spec = NODES[0]
    root = _synthetic_node_repo(tmp_path)
    rec = run_node_tests(root, spec, tmp_path / "out", timeout=120)
    assert rec.green and rec.passed == 1
    # junit evidence must land in the collector's out dir, never inside the node repo
    assert Path(rec.junit).is_absolute() and rec.junit.startswith(str(tmp_path / "out"))
    assert not (root / "artifacts/p34_integration").exists()
    ev = collect_node(spec, root, rec, report={"title": "r", "figures": 2})
    ev["min_tests"] = 1  # the synthetic node ships a single test
    assert ev["artifacts"][0]["ok"] and ev["artifacts"][1]["ok"]
    assert ev["declared"][0]["present"] is True
    assert ev["delivered"] is True, ev["gaps"]
    assert evaluate_c1([ev]).status == PASS


def test_collect_node_reports_failing_tests_and_missing_worktree(tmp_path):
    spec = NODES[0]
    root = _synthetic_node_repo(tmp_path, passing=False)
    rec = run_node_tests(root, spec, tmp_path / "out", timeout=120)
    ev = collect_node(spec, root, rec)
    assert rec.green is False and rec.failed == 1
    assert evaluate_c1([ev]).status == FAIL
    absent = collect_node(spec, None, None)
    assert absent["delivered"] is False
    assert "node worktree not found" in absent["gaps"]
    assert evaluate_c1([absent]).status == WITHHELD


# --------------------------------------------------------------------------
# end-to-end gate: PASS fixture, then one broken promise per criterion
# --------------------------------------------------------------------------
def _perfect_campaign(tmp_path: Path) -> dict:
    """Build a campaign where every criterion is genuinely satisfied."""
    root = tmp_path
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "logic.py").write_text(
        '"""workflow only"""\nimport json\n\n\ndef f(x):\n    return json.dumps(x)\n',
        encoding="utf-8")
    _png(root / "artifacts/p34_integration/fig1_criteria.png")
    report = root / "docs/p34_campaign_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_SIX_ELEMENTS, encoding="utf-8")
    reg = _regression()
    reg_path = root / "artifacts/p34_integration/regression.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(reg), encoding="utf-8")
    reg["path"] = str(reg_path)
    collection = {"utc": "2026-11-07T00:00:00Z", "arch": "loongarch64",
                  "root_template": "/root/tp-agent/work/m{mission_id}",
                  "summary": {"n_nodes": 7}, "nodes": _all_nodes()}
    return {"collection": collection, "regression": reg, "report": report,
            "pkg": pkg, "root": root}


def test_gate_passes_when_every_criterion_is_really_satisfied(tmp_path):
    f = _perfect_campaign(tmp_path)
    v = run_gate(collection=f["collection"], regression=f["regression"],
                 report_path=f["report"], pkg_dir=f["pkg"],
                 deliverables=[f["report"]], repo_root=f["root"],
                 out_dir=f["root"] / "artifacts/p34_integration")
    assert v["overall_status"] == PASS, v["criteria"]
    assert v["criterion_summary"] == {"C1": PASS, "C2": PASS, "C3": PASS,
                                      "C4": PASS, "C5": PASS}
    saved = json.loads((f["root"] / "artifacts/p34_integration/gate_verdict.json").read_text())
    assert saved["schema"] == "p34_campaign_gate_verdict.v1"
    assert saved["provenance"]["report"][0]["sha256"]
    assert "PASS" in (f["root"] / "artifacts/p34_integration/gate_verdict.md").read_text()


@pytest.mark.parametrize("break_what,blocked", [
    ("node_tests", "C1"),
    ("artifacts", "C2"),
    ("regression", "C3"),
    ("report", "C4"),
    ("placeholder", "C5"),
])
def test_gate_fails_honestly_when_one_promise_is_broken(tmp_path, break_what, blocked):
    f = _perfect_campaign(tmp_path)
    if break_what == "node_tests":
        f["collection"]["nodes"][3]["pytest"].update({"green": False, "failed": 1,
                                                      "exit_code": 1})
    elif break_what == "artifacts":
        f["collection"]["nodes"][5]["artifacts"][0].update({"ok": False, "n_matches": 0,
                                                            "files": []})
    elif break_what == "regression":
        f["regression"]["suites"][0].update({"exit_code": 1, "failed": 3})
    elif break_what == "report":
        f["report"].write_text(_SIX_ELEMENTS.replace("## 3. 计算结果", "## 3. 别的"),
                               encoding="utf-8")
    elif break_what == "placeholder":
        (f["pkg"] / "draft.py").write_text("# TODO fill in\n", encoding="utf-8")
    v = run_gate(collection=f["collection"], regression=f["regression"],
                 report_path=f["report"], pkg_dir=f["pkg"],
                 deliverables=[f["report"]], repo_root=f["root"],
                 out_dir=f["root"] / "artifacts/p34_integration")
    assert v["overall_status"] == FAIL
    assert blocked in v["blocking_criteria"], v["criterion_summary"]
    assert v["criterion_summary"][blocked] in (FAIL, WITHHELD)


# --------------------------------------------------------------------------
# regression runner (aggregation only; subprocess paths are exercised above)
# --------------------------------------------------------------------------
def test_regression_assemble_marks_uncovered_files_red(tmp_path):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    for name in ("test_a.py", "test_b.py"):
        (root / "tests" / name).write_text("def test_x():\n    assert True\n", encoding="utf-8")
    out = tmp_path / "out"
    spec = R.SuiteSpec(name="s", family="resistance", repo="ship",
                       root=root, tests=("tests/test_a.py", "tests/test_b.py"))
    (out / "s" / "junit").mkdir(parents=True)
    (out / "s" / "records.json").write_text(json.dumps([
        {"test": "tests/test_a.py", "collected": 3, "passed": 3, "failed": 0,
         "errors": 0, "skipped": 0, "exit_code": 0, "duration_s": 0.4,
         "arch": "x86_64", "head_sha": "0" * 40, "junit": "", "family": "resistance"}
    ]), encoding="utf-8")
    row = R.assemble(spec, out)
    assert row["n_covered"] == 1 and row["n_missing"] == 1
    assert row["exit_code"] == 1 and row["passed"] == 3 and row["errors"] == 1
    assert row["red_files"][0]["test"] == "tests/test_b.py"


def test_run_suite_executes_real_pytest_and_persists_records(tmp_path):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "tests/test_a.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    out = tmp_path / "out"
    spec = R.SuiteSpec(name="s", family="stability", repo="r", root=root,
                       tests=("tests/test_a.py",))
    res = R.run_suite(spec, out, budget_s=60, per_file_timeout=120)
    assert res["ran"] == ["tests/test_a.py"] and res["n_remaining"] == 0
    assert not (root / "artifacts").exists()  # no evidence written into the suite repo
    row = R.assemble(spec, out)
    assert row["exit_code"] == 0 and row["passed"] == 1
    doc = R.write_regression([spec], out, out / "regression.json")
    assert doc["n_green"] == 1