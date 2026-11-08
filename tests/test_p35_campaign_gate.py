"""P35-G1 campaign integration gate: honest-verdict and evidence-integrity tests.

The tests pin three things the campaign discipline depends on:

1. the gate can actually reach ``PASS`` -- a fully evidenced synthetic campaign
   turns every criterion green and ``dispatcher_ready`` true (a gate that can
   only fail proves nothing);
2. the gate can never reach ``PASS`` on silence -- a missing report, a missing
   ledger row, a missing maturity declaration, an unrunnable regression probe or
   a platform/view divergence each keep the verdict at ``FAIL`` with the reason
   spelled out in ``blocking``;
3. the emitted ``campaign_readiness.v1`` payload is schema-valid and carries the
   provenance digests a third party needs to re-check it.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from tensorphysics.campaign import schema as _schema
from tensorphysics.campaign.p35_gate import readiness_schema as _rs
from tensorphysics.campaign.p35_gate import collect as C
from tensorphysics.campaign.p35_gate import criteria as K
from tensorphysics.campaign.p35_gate import gate as G
from tensorphysics.campaign.p35_gate import regression as R
from tensorphysics.campaign.p35_gate import spec as S
from tensorphysics.campaign.p35_gate.__main__ import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------
def _png_bytes(width: int = 32, height: int = 32) -> bytes:
    """A real, decodable grayscale PNG (magic + IHDR + IDAT + IEND)."""
    raw = b"".join(b"\x00" + bytes([(x * 32) % 256] * width) for x in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (C.PNG_MAGIC + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, (bytes, bytearray)):
        path.write_bytes(payload)
    elif isinstance(payload, str):
        path.write_text(payload)
    else:
        path.write_text(json.dumps(payload, indent=2))
    return path


def _snapshot(*, reports=1, delta=True, deps=None, status="succeeded") -> dict:
    """Synthetic platform snapshot: every node reported, ledger booked, deps given."""
    nodes = []
    for node in S.NODES:
        node_reports = ([] if reports == 0 else
                        [{"report_id": 9000 + node.mission_id, "title": f"{node.ext_id} report",
                          "mission_id": node.mission_id, "figures": ["f1.png"],
                          "markdown_chars": 4200, "created_utc": "2026-11-08T10:00:00Z"}]
                        * max(reports, 0))
        ledger = None
        if delta:
            ledger = {"category": "new_capability", "summary": "capability booked",
                      "mission_repo": node.repo,
                      "evidence": {"ext_id": node.ext_id,
                                   "physics_domain": node.physics_domain},
                      "artifacts": [], "created_utc": "2026-11-08T10:00:00Z"}
        nodes.append({
            "ext_id": node.ext_id, "mission_id": node.mission_id,
            "declared_repo": node.repo, "declared_physics_domain": node.physics_domain,
            "plan_node_id": 1000 + node.mission_id, "plan_node_name": node.title,
            "plan_node_status": "done", "plan_dependencies": (list(deps.get(node.ext_id, [])) if isinstance(deps, dict) else list(deps or [])),
            "platform_repo": node.repo, "mission_status": status,
            "mission_result": {}, "reports": node_reports, "capability_delta": ledger,
        })
    return {
        "schema": C.SNAPSHOT_SCHEMA, "campaign_id": S.CAMPAIGN_ID, "gate_id": S.GATE_ID,
        "collected_utc": "2026-11-08T10:00:00Z",
        "plan": {"id": S.PLAN_ID, "title": S.CAMPAIGN_TITLE, "name": "P35_wave1",
                 "status": "active", "physics_scope": list(S.PLAN_PHYSICS_SCOPE),
                 "n_plan_nodes": len(S.NODES) + 1},
        "nodes": nodes, "errors": [],
    }


def _deps_view(ext_id: str) -> list[str]:
    """Platform-side dependency list for one node, matching the declaration."""
    return [e["from"] for e in K.declared_edges() if e["to"] == ext_id]


def _materialise_repo(root: Path, *, families: bool = True, maturity: bool = True) -> None:
    """Write a node-shaped checkout: artifacts per declared family + maturity rows."""
    for node in S.NODES:
        for req in node.artifacts:
            base = req.pattern.replace("/**", "")
            if base.endswith("maturity.json"):
                pattern = base
            elif "*.json" in base:
                pattern = base.replace("*.json", "result.json")
            else:
                pattern = base.replace("*.png", "figure.png")
            target = root / pattern
            if not families:
                continue
            if req.kind == "png":
                _write(target, _png_bytes())
            elif target.name == "maturity.json":
                if maturity:
                    _write(target, {"ext_id": node.ext_id, "trl": 7,
                                    "gates": [{"gate": "pytest", "status": "pass"},
                                              {"gate": "numerical", "status": "pass"}]})
            else:
                _write(target, {"case": node.ext_id, "verdict": "pass"})


def _scans(root: Path, *, authoritative: bool = False) -> dict:
    return {node.ext_id: C.scan_node_repository(root, node, authoritative=authoritative)
            for node in S.NODES}


# --------------------------------------------------------------------------
# spec integrity
# --------------------------------------------------------------------------
def test_spec_declares_ten_nodes_with_identities():
    assert len(S.NODES) == 10
    assert len({n.ext_id for n in S.NODES}) == 10
    assert len({n.mission_id for n in S.NODES}) == 10
    assert set(S.MATURITY_NODES) == {"P35-N4", "P35-N5", "P35-N6", "P35-N7"}
    for node in S.NODES:
        assert node.physics_domain in S.ALLOWED_PHYSICS_DOMAINS
        assert node.artifacts, f"{node.ext_id} declares no artifact family"
        assert node.acceptance, f"{node.ext_id} declares no acceptance criterion"


def test_declared_dependency_graph_is_acyclic_and_orphan_free():
    edges = K.declared_edges()
    ok, topo = K._acyclic(edges)
    assert ok, f"declared P35 dependency graph has a cycle: {topo}"
    assert sorted(topo) == sorted(S.NODE_BY_ID)
    orphans = [n.ext_id for n in S.NODES
               if not n.prerequisites and n.ext_id not in S.ROOT_NODES]
    assert orphans == []
    assert {e["from"] for e in edges} <= set(S.NODE_BY_ID)
    # the two three-way coupling nodes really do consume the upstream kernels
    assert ("P35-N1", "P35-N9") in {(e["from"], e["to"]) for e in edges}
    assert ("P35-N2", "P35-N9") in {(e["from"], e["to"]) for e in edges}
    assert ("P35-N1", "P35-N10") in {(e["from"], e["to"]) for e in edges}


def test_plan_physics_scope_gap_is_declared_not_hidden():
    """P35-N3 needs a physics domain the plan does not list: declared, not silent."""
    assert "wave_basin_inversion" not in S.PLAN_PHYSICS_SCOPE
    assert "wave_basin_inversion" in S.UNSCOPED_PHYSICS_DOMAINS
    assert S.NODE_BY_ID["P35-N3"].physics_domain == "wave_basin_inversion"


# --------------------------------------------------------------------------
# honest FAIL: no evidence is never a pass
# --------------------------------------------------------------------------
def test_empty_campaign_fails_every_criterion_and_lists_blocking():
    snapshot = _snapshot(reports=0, delta=False)
    payload = G.run_gate(snapshot, scans={})
    assert payload["overall_status"] == "FAIL"
    assert payload["dispatcher_ready"] is False
    status = {c["criterion_id"]: c["status"] for c in payload["criteria"]}
    assert status["a"] == "FAIL"          # no reports -> not landed
    assert status["b"] == "FAIL"          # no ledger row
    assert status["g"] == "FAIL"
    assert len(payload["blocking"]) >= 10
    assert any("deliverable:FAIL" in b for b in payload["blocking"])
    assert any("registration:FAIL" in b for b in payload["blocking"])
    assert all(n["deliverable"] == "FAIL" for n in payload["nodes"])


def test_unscannable_repository_is_withheld_not_passed():
    """A node that reported but whose checkout the gate cannot see is WITHHELD."""
    snapshot = _snapshot()
    payload = G.run_gate(snapshot, scans={})
    crit_a = next(c for c in payload["criteria"] if c["criterion_id"] == "a")
    assert crit_a["status"] == "WITHHELD"
    assert "仓库不可扫描" in crit_a["evidence"]["nodes"][0]["detail"]
    assert payload["overall_status"] == "FAIL"
    assert payload["dispatcher_ready"] is False


def test_maturity_node_without_declaration_fails():
    snapshot = _snapshot()
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    _materialise_repo(tmp, maturity=False)
    payload = G.run_gate(snapshot, scans=_scans(tmp))
    crit_e = next(c for c in payload["criteria"] if c["criterion_id"] == "e")
    assert crit_e["status"] == "FAIL"
    assert "缺失成熟化声明" in crit_e["evidence"]["nodes"][0]["detail"]


def test_ledger_row_without_identity_fields_fails():
    snapshot = _snapshot()
    for row in snapshot["nodes"]:
        row["capability_delta"]["evidence"] = {}     # booked, but identity not carried
    payload = G.run_gate(snapshot, scans={})
    crit_b = next(c for c in payload["criteria"] if c["criterion_id"] == "b")
    assert crit_b["status"] == "FAIL"
    assert "ext_id" in crit_b["evidence"]["nodes"][0]["detail"]


def test_platform_dependency_divergence_blocks():
    snapshot = _snapshot(deps=[])
    payload = G.run_gate(snapshot, scans={})
    crit_d = next(c for c in payload["criteria"] if c["criterion_id"] == "d")
    assert crit_d["status"] == "FAIL"
    assert "campaign_plans.dependencies" in crit_d["detail"]
    assert crit_d["metrics"]["missing_on_platform"]


# --------------------------------------------------------------------------
# full-green synthetic campaign: the gate can pass
# --------------------------------------------------------------------------
def test_fully_evidenced_campaign_passes_and_is_dispatcher_ready():
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    _materialise_repo(tmp)
    snapshot = _snapshot(deps={n.ext_id: _deps_view(n.ext_id) for n in S.NODES})
    records = {c.case: {"case": c.case, "repo": c.repo, "status": "PASS",
                        "detail": "measured matches baseline",
                        "baseline": 1.0, "measured": 1.004, "rel_dev": 4e-3,
                        "threshold": c.threshold, "floor": c.floor,
                        "elapsed_s": 12.0, "repo_root": str(tmp), "repo_head": "deadbeef",
                        "command": list(c.command), "stdout_tail": ""}
               for c in S.REGRESSION_CASES}
    payload = G.run_gate(snapshot, scans=_scans(tmp), regression_records=records)
    status = {c["criterion_id"]: c["status"] for c in payload["criteria"]}
    assert status == {k: "PASS" for k in "abcdefg"}, (status, payload["blocking"])
    assert payload["overall_status"] == "PASS"
    assert payload["dispatcher_ready"] is True
    assert payload["blocking"] == []
    assert all(n["chain_ready"] for n in payload["nodes"])
    _rs.validate_artifact(G.READINESS_SCHEMA, payload)


def test_chain_ready_propagates_from_undelivered_upstream():
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    _materialise_repo(tmp)
    snapshot = _snapshot(deps={n.ext_id: _deps_view(n.ext_id) for n in S.NODES})
    # P35-N1 loses its report -> N9 and N10 must not be chain ready
    for row in snapshot["nodes"]:
        if row["ext_id"] == "P35-N1":
            row["reports"] = []
    payload = G.run_gate(snapshot, scans=_scans(tmp))
    nodes = {n["ext_id"]: n for n in payload["nodes"]}
    assert nodes["P35-N1"]["deliverable"] == "FAIL"
    assert nodes["P35-N9"]["chain_ready"] is False
    assert nodes["P35-N10"]["chain_ready"] is False
    crit_d = next(c for c in payload["criteria"] if c["criterion_id"] == "d")
    blocked = {(e["from"], e["to"]) for e in crit_d["metrics"]["blocked_edges"]}
    assert {("P35-N1", "P35-N9"), ("P35-N1", "P35-N10")} <= blocked


# --------------------------------------------------------------------------
# regression machinery
# --------------------------------------------------------------------------
def test_compare_uses_declared_floor_for_machine_precision_metrics():
    assert R.compare(5.73e-14, 9.60e-14, 0.01, 0.0)[0] == R.FAIL        # noise vs noise
    assert R.compare(5.73e-14, 9.60e-14, 0.01, 1e-10)[0] == R.PASS      # declared floor
    assert R.compare(2.30, 2.243, 0.01, 0.0)[0] == R.FAIL               # +2.5% > 1%
    assert R.compare(2.250, 2.243, 0.01, 0.0)[0] == R.PASS              # +0.3% < 1%


def _fake_case_repo(tmp: Path, *, metric: float, runner: bool = True,
                    baseline: float = 1.0) -> Path:
    root = tmp / "repo"
    (root / "benchmarks/case").mkdir(parents=True, exist_ok=True)
    if runner:
        _write(root / "benchmarks/case/run.py", (
            "import json, sys\n"
            "out = sys.argv[1]\n"
            f"json.dump({{'metric': {metric!r}}}, open(out, 'w'))\n"))
    _write(root / "benchmarks/case/result.json", {"g": {"m": baseline}})
    return root


def _case() -> S.RegressionCase:
    return S.RegressionCase(case="toy", repo="Toy", command=("python3", "benchmarks/case/run.py",
                                                             "{out}/probe.json"),
                            baseline_path="benchmarks/case/result.json", metric_path="g.m",
                            metric_pattern="probe.json:metric", floor=0.0)


def test_run_probe_pass_fail_and_withheld(tmp_path):
    good = _fake_case_repo(tmp_path / "good", metric=1.004)
    res = R.run_probe(_case(), good, tmp_path / "work_good", timeout=120)
    assert (res.status, res.measured) == (R.PASS, pytest.approx(1.004))
    assert res.rel_dev == pytest.approx(4e-3)

    bad = _fake_case_repo(tmp_path / "bad", metric=1.2)
    res_bad = R.run_probe(_case(), bad, tmp_path / "work_bad", timeout=120)
    assert res_bad.status == R.FAIL and res_bad.rel_dev == pytest.approx(0.2)

    no_runner = _fake_case_repo(tmp_path / "norun", metric=1.0, runner=False)
    res_missing = R.run_probe(_case(), no_runner, tmp_path / "work_missing", timeout=120)
    assert res_missing.status == R.WITHHELD
    assert "runner entry point absent" in res_missing.detail

    empty = tmp_path / "empty"
    empty.mkdir()
    res_absent = R.run_probe(_case(), empty, tmp_path / "work_absent", timeout=120)
    assert res_absent.status == R.WITHHELD and "baseline record absent" in res_absent.detail


def test_evaluate_regression_aggregates_withheld_before_fail():
    cases, status = R.evaluate_regression({})
    assert status == R.WITHHELD and all(c["status"] == R.WITHHELD for c in cases)
    assert len(cases) == len(S.REGRESSION_CASES)


def test_regression_records_round_trip(tmp_path):
    good = _fake_case_repo(tmp_path / "good", metric=1.0)
    res = R.run_probe(_case(), good, tmp_path / "work", timeout=120)
    path = tmp_path / "records.json"
    R.append_record(path, res)
    assert R.load_records(path)["toy"]["status"] == R.PASS


# --------------------------------------------------------------------------
# readiness schema
# --------------------------------------------------------------------------
def test_readiness_schema_fingerprint_is_pinned():
    # pinch the fingerprint so a silent schema edit breaks the build on both sides
    assert _rs.schema_fingerprint("campaign_readiness.v1") == (
        "0a06491ea401e5b2423910992992c8418f14fc955518b80dda9cb5f41d804ba8")


def test_readiness_schema_rejects_missing_and_extra_fields():
    snapshot = _snapshot(reports=0, delta=False)
    payload = G.run_gate(snapshot, scans={})
    _rs.validate_artifact(G.READINESS_SCHEMA, payload)

    broken = json.loads(json.dumps(payload))
    broken.pop("dispatcher_ready")
    with pytest.raises(_schema.SchemaError):
        _rs.validate_artifact(G.READINESS_SCHEMA, broken)

    extra = json.loads(json.dumps(payload))
    extra["surprise"] = 1
    with pytest.raises(_schema.SchemaError):
        _rs.validate_artifact(G.READINESS_SCHEMA, extra)

    bad_status = json.loads(json.dumps(payload))
    bad_status["criteria"][0]["status"] = "OK"
    with pytest.raises(_schema.SchemaError):
        _rs.validate_artifact(G.READINESS_SCHEMA, bad_status)


def test_criterion_f_never_passes_on_invalid_payload():
    """If the payload cannot be validated, (f) is FAIL -- not a silent pass."""
    class _Boom:
        @staticmethod
        def validate_artifact(name, payload):
            raise _schema.SchemaError("synthetic: payload rejected")

    snapshot = _snapshot(reports=0, delta=False)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(G._rs, "validate_artifact", _Boom.validate_artifact)
    try:
        payload = G.run_gate(snapshot, scans={})
    finally:
        monkey.undo()
    crit_f = next(c for c in payload["criteria"] if c["criterion_id"] == "f")
    assert crit_f["status"] == "FAIL"
    assert "synthetic: payload rejected" in crit_f["detail"]


# --------------------------------------------------------------------------
# emit + CLI
# --------------------------------------------------------------------------
def test_write_outputs_emits_json_and_markdown(tmp_path):
    payload = G.run_gate(_snapshot(reports=0, delta=False), scans={})
    written = G.write_outputs(payload, tmp_path)
    doc = json.loads((tmp_path / G.READINESS_JSON).read_text())
    assert doc["gate_id"] == S.GATE_ID
    md = (tmp_path / G.READINESS_MD).read_text()
    assert "campaign_readiness" in md and "阻塞清单" in md
    assert written["readiness_json"]["sha256"] == C.sha256_file(tmp_path / G.READINESS_JSON)


def test_cli_renders_verdict_from_snapshot_offline(tmp_path):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(_snapshot(reports=0, delta=False)))
    out = tmp_path / "out"
    rc = cli_main(["--snapshot", str(snapshot_path), "--out", str(out)])
    assert rc == 0
    doc = json.loads((out / G.READINESS_JSON).read_text())
    assert doc["overall_status"] == "FAIL"
    rc_strict = cli_main(["--snapshot", str(snapshot_path), "--out", str(out),
                          "--require-pass"])
    assert rc_strict == 3


def test_snapshot_round_trip_provenance(tmp_path):
    snapshot = _snapshot(reports=0, delta=False)
    prov = C.write_snapshot(snapshot, tmp_path / "snap.json")
    assert C.sha256_file(prov["path"]) == prov["sha256"]
    again = C.load_snapshot(prov["path"])
    assert [n["ext_id"] for n in again["nodes"]] == [n.ext_id for n in S.NODES]


def test_package_imports_without_platform_or_torch():
    """The gate is stdlib-only: importing it must not pull solver stacks."""
    code = ("import sys;"
            "import tensorphysics.campaign.p35_gate as g;"
            "bad=[m for m in ('torch','tensorlbm','tensordem','tensorfem') if m in sys.modules];"
            "print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO_ROOT),
                         env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


def test_artifact_absence_is_fail_only_on_the_delivery_checkout():
    """A missing family is a FAIL on the node's checkout, WITHHELD on a repo guess."""
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    _materialise_repo(tmp)
    # P35-N6 half-delivered: figures gone, ledger kept
    for fig in (tmp / S.NODE_BY_ID["P35-N6"].artifact_dir).glob("*.png"):
        fig.unlink()
    snapshot = _snapshot()

    authoritative = G.run_gate(snapshot, scans={
        n.ext_id: C.scan_node_repository(tmp, n, authoritative=True) for n in S.NODES})
    n6 = {n["ext_id"]: n for n in authoritative["nodes"]}["P35-N6"]
    assert n6["deliverable"] == "FAIL"

    guessed = G.run_gate(snapshot, scans={
        n.ext_id: C.scan_node_repository(tmp, n, authoritative=False) for n in S.NODES})
    n6_guess = {n["ext_id"]: n for n in guessed["nodes"]}["P35-N6"]
    assert n6_guess["deliverable"] == "WITHHELD"
    assert guessed["overall_status"] == "FAIL"      # withheld is never a pass


def test_probe_beyond_budget_is_withheld_with_a_cost_estimate(tmp_path):
    root = _fake_case_repo(tmp_path / "repo", metric=1.0)
    priced = S.RegressionCase(
        case="expensive", repo="Toy", command=("python3", "benchmarks/case/run.py",
                                               "{out}/probe.json"),
        baseline_path="benchmarks/case/result.json", metric_path="g.m",
        metric_pattern="probe.json:metric", budget_s=60.0, estimated_cost_s=19400.0)
    res = R.run_probe(priced, root, tmp_path / "work", timeout=60)
    assert res.status == R.WITHHELD
    assert "超出本门预算" in res.detail and "19400" in res.detail


def test_maturity_declaration_path_follows_the_node_artifact_directory():
    for ext in S.MATURITY_NODES:
        node = S.NODE_BY_ID[ext]
        path = S.maturity_pattern(ext)
        assert path == f"{node.artifact_dir}/maturity.json"
        assert A_any(p.pattern == path for p in node.artifacts)


def A_any(iterable):
    return any(iterable)


def test_platform_dependencies_recorded_as_plan_node_ids_are_resolved():
    """campaign_plans stores node ids; the gate must map them before comparing."""
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    _materialise_repo(tmp)
    snapshot = _snapshot(deps={n.ext_id: _deps_view(n.ext_id) for n in S.NODES})
    id_of = {row["ext_id"]: str(row["plan_node_id"]) for row in snapshot["nodes"]}
    for row in snapshot["nodes"]:
        row["plan_dependencies"] = [id_of[d] for d in row["plan_dependencies"]]
    payload = G.run_gate(snapshot, scans=_scans(tmp))
    crit_d = next(c for c in payload["criteria"] if c["criterion_id"] == "d")
    assert crit_d["status"] == "PASS", crit_d["detail"]
    assert crit_d["metrics"]["missing_on_platform"] == []
    assert crit_d["metrics"]["platform_edges"] == crit_d["metrics"]["declared_edges"]


def test_maturity_declaration_is_scored_once_by_criterion_e():
    """Absent maturity.json fails (e) but must not be double-counted in (a)."""
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    _materialise_repo(tmp, maturity=False)
    snapshot = _snapshot()
    payload = G.run_gate(snapshot, scans=_scans(tmp, authoritative=True))
    nodes = {n["ext_id"]: n for n in payload["nodes"]}
    assert nodes["P35-N6"]["deliverable"] == "PASS"     # case.json + figures are there
    assert nodes["P35-N6"]["maturity"] == "FAIL"        # the declaration is not
    crit_e = next(c for c in payload["criteria"] if c["criterion_id"] == "e")
    assert crit_e["status"] == "FAIL"
    assert payload["overall_status"] == "FAIL"
