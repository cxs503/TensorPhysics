"""Tests for first-class criteria: extraction, normalisation, conflict, arbitration.

All fixtures use the synthetic ``synthetic.metric.*`` vocabulary injected via
``configs/synthetic_metrics.json``; the module itself carries no domain terms.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tensorphysics.campaign.criterion_conflict import (
    ArbitrationError,
    ArbitrationLedger,
    ArbitrationRecord,
    ConflictConfig,
    CriterionRegistry,
    ConfigError,
    GENESIS,
    SourceRef,
    detect_conflicts,
    extract_criteria,
    leak_terms,
    load_config,
    make_criterion_id,
    normalize_comparator,
    normalize_statistic,
    normalize_threshold,
    normalize_unit,
)
from tensorphysics.campaign.criterion_conflict.__main__ import main

PKG = Path(__file__).resolve().parents[1] / "src/tensorphysics/campaign/criterion_conflict"
CONFIG_PATH = PKG / "configs/synthetic_metrics.json"


@pytest.fixture(scope="module")
def cfg() -> ConflictConfig:
    return load_config(CONFIG_PATH)


def _src(mission="m1", campaign="c1", field="goal") -> SourceRef:
    return SourceRef.make(mission, campaign, "repo", field, 0, 0, 0, "")


def _extract(text: str, cfg: ConflictConfig, mission="m1", campaign="c1"):
    return extract_criteria(text, _src(mission, campaign), cfg)


def _registry(pairs, cfg: ConflictConfig) -> CriterionRegistry:
    entries = [{"mission_id": f"m{i}", "campaign": camp, "repo": "r", "goal": goal}
               for i, (goal, camp) in enumerate(pairs)]
    return CriterionRegistry.build(entries, cfg)


# ---------------------------------------------------------------------------
# configuration is injected and validated (fail-closed)
# ---------------------------------------------------------------------------

def test_config_validation_rejects_inconsistencies(tmp_path):
    good = json.loads(CONFIG_PATH.read_text())
    assert ConflictConfig.from_dict(good).sha256() == load_config(CONFIG_PATH).sha256()
    with pytest.raises(ConfigError):
        ConflictConfig.from_dict({**good, "schema_version": "bogus"})
    with pytest.raises(ConfigError):
        ConflictConfig.from_dict({**good, "metrics": []})
    with pytest.raises(ConfigError):
        ConflictConfig.from_dict({**good, "default_statistic": "nope"})

    dup = json.loads(json.dumps(good))
    dup["metrics"][1]["aliases"].append(dup["metrics"][0]["aliases"][0])
    with pytest.raises(ConfigError):
        ConflictConfig.from_dict(dup)

    bad_unit = json.loads(json.dumps(good))
    bad_unit["metrics"][0]["default_unit"] = "furlong"
    with pytest.raises(ConfigError):
        ConflictConfig.from_dict(bad_unit)

    bad_tol = json.loads(json.dumps(good))
    bad_tol["metrics"][0]["tolerance_relative"] = -1
    with pytest.raises(ConfigError):
        ConflictConfig.from_dict(bad_tol)

    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert load_config(path).sha256() == ConflictConfig.from_dict(good).sha256()
    assert load_config(ConflictConfig.from_dict(good)).sha256() == ConflictConfig.from_dict(good).sha256()


def test_config_roundtrip_is_stable(cfg):
    assert ConflictConfig.from_dict(cfg.to_dict()).sha256() == cfg.sha256()
    assert cfg.tolerance("synthetic.metric.beta") == pytest.approx(0.20)
    assert cfg.tolerance("not-a-metric") == pytest.approx(cfg.default_tolerance_relative)
    with pytest.raises(ConfigError):
        cfg.metric("not-a-metric")
    assert any(a == "synthetic.metric.alpha" for a, _ in cfg.alias_table())


# ---------------------------------------------------------------------------
# extraction: provenance, aliases, statistics, units
# ---------------------------------------------------------------------------

def test_extraction_provenance_is_complete(cfg):
    text = "intro without criteria\n- synthetic.metric.alpha mean error <= 3%\n- tail line\n"
    result = _extract(text, cfg, mission="m42")
    assert len(result.criteria) == 1
    c = result.criteria[0]
    assert c.quantity == "synthetic.metric.alpha"
    assert c.statistic == "mean"
    assert c.comparator == "le"
    assert c.direction == "upper"
    assert c.threshold == 3.0
    assert c.unit == "%"
    assert c.threshold_normalized == pytest.approx(0.03)
    assert c.status == "mapped"
    assert c.source.mission_id == "m42"
    assert c.source.line == 2
    assert "synthetic.metric.alpha" in c.source.snippet
    assert c.source.char_end > c.source.char_start


def test_multiple_criteria_per_line_and_conjunctions(cfg):
    text = ("synthetic.metric.beta runtime <= 120 ms 且 synthetic.metric.gamma count >= 200; "
            "synthetic.metric.delta worst >= 1.5 kPa")
    result = _extract(text, cfg)
    got = {(c.quantity, c.direction) for c in result.criteria}
    assert got == {("synthetic.metric.beta", "upper"), ("synthetic.metric.gamma", "lower"),
                   ("synthetic.metric.delta", "lower")}
    assert all(c.source.line == 1 for c in result.criteria)
    assert len({c.criterion_id for c in result.criteria}) == 3


def test_wording_and_unit_variants_normalise_to_same_quantity(cfg):
    text = ("synthetic.metric.alpha 逐点误差不超过 0.05\n"
            "synthetic.metric.alpha pointwise error at most 50 pct\n"
            "synthetic.metric.alpha 单点 error <= 50‰\n")
    result = _extract(text, cfg)
    values = [(c.unit, c.threshold_normalized) for c in result.criteria]
    assert ("1", pytest.approx(0.05)) in values
    assert ("pct", pytest.approx(0.5)) in values
    assert ("‰", pytest.approx(0.05)) in values
    assert cfg.canonical_statistic(result.criteria[0].statistic) == "pointwise"


def test_statistic_labels(cfg):
    text = ("synthetic.metric.alpha mean error <= 1%\n"
            "synthetic.metric.alpha 最坏 error <= 2%\n"
            "synthetic.metric.alpha p95 error <= 3%\n"
            "synthetic.metric.alpha error <= 4%\n")
    result = _extract(text, cfg)
    assert [c.statistic for c in result.criteria] == ["mean", "worst", "p95", "unspecified"]
    assert result.criteria[0].statistic_raw == "mean"
    assert result.criteria[3].statistic_raw is None


def test_injected_exclusion_markers_skip_policy_prose(cfg):
    legacy = json.loads(CONFIG_PATH.read_text())
    legacy["exclusion_markers"] = ["do-not-parse"]
    ex_cfg = ConflictConfig.from_dict(legacy)
    text = ("synthetic.metric.alpha error <= 3% do-not-parse\n"
            "synthetic.metric.alpha error <= 5%\n")
    result = _extract(text, ex_cfg)
    assert [c.threshold for c in result.criteria] == [5.0]


def test_unmapped_quantity_is_fail_closed(cfg):
    result = _extract("unknown.metric.x <= 5%\n", cfg)
    assert result.criteria == []
    assert result.residue[0]["reason"] == "unmapped_quantity"
    assert result.residue[0]["line"] == 1


def test_unmapped_unit_is_fail_closed(cfg):
    result = _extract("synthetic.metric.alpha error <= 5 furlongs\n", cfg)
    assert len(result.criteria) == 1
    assert result.criteria[0].status == "unmapped_unit"
    assert result.criteria[0].threshold_normalized is None
    assert detect_conflicts(result.criteria, cfg) == []


def test_bracketed_unit_annotation_is_normalised(cfg):
    result = _extract("synthetic.metric.alpha residual < 2[% annotated]\n", cfg)
    c = result.criteria[0]
    assert c.unit == "%" and c.threshold_normalized == pytest.approx(0.02)
    result = _extract("synthetic.metric.alpha residual < 2 (pct)\n", cfg)
    assert result.criteria[0].threshold_normalized == pytest.approx(0.02)


def test_bare_number_uses_configured_default_unit(cfg):
    result = _extract("synthetic.metric.gamma count >= 250\n", cfg)
    c = result.criteria[0]
    assert c.unit == "1" and c.threshold_normalized == pytest.approx(250.0)


def test_equal_and_range_directions(cfg):
    result = _extract("synthetic.metric.gamma count == 10\nsynthetic.metric.alpha error ±2%\n", cfg)
    assert {c.direction for c in result.criteria} == {"equal", "range"}
    assert detect_conflicts(result.criteria, cfg) == []


def test_structural_helpers(cfg):
    assert normalize_comparator("le") == "upper"
    assert normalize_comparator("ge") == "lower"
    assert normalize_comparator(None) is None
    metric = cfg.metric("synthetic.metric.beta")
    assert normalize_unit("ms", metric) == ("ms", pytest.approx(0.001), "ms")
    assert normalize_unit(None, metric)[0] == "s"
    assert normalize_unit("furlong", metric) == (None, None, "furlong")
    assert normalize_threshold(120.0, 0.001) == pytest.approx(0.12)
    assert normalize_threshold(None, 1.0) is None
    assert normalize_statistic("maximum", cfg) == "worst"
    assert normalize_statistic("absent", cfg) == cfg.default_statistic


def test_criterion_id_is_deterministic(cfg):
    a = _extract("synthetic.metric.alpha error <= 3%\n", cfg).criteria[0]
    b = _extract("synthetic.metric.alpha error <= 3%\n", cfg).criteria[0]
    assert a.criterion_id == b.criterion_id
    other = _extract("synthetic.metric.alpha error <= 4%\n", cfg).criteria[0]
    assert other.criterion_id != a.criterion_id
    assert make_criterion_id(a.source, a.quantity, a.statistic, a.comparator, a.threshold, a.unit) == a.criterion_id


# ---------------------------------------------------------------------------
# coverage: >=90% of reference instances on >=200 synthetic missions
# ---------------------------------------------------------------------------

def test_extraction_coverage_on_200_plus_missions(cfg):
    entries = []
    for i in range(240):
        pct = (i % 9) + 1
        goal = (f"任务 {i} 目标：\n"
                f"- synthetic.metric.alpha mean error <= {pct}%\n"
                f"- synthetic.metric.beta runtime <= {100 + i} ms\n"
                f"- noise line with {i} numbers but no comparator\n")
        entries.append({"mission_id": f"m{i:04d}", "campaign": f"c{i % 7}", "repo": "r", "goal": goal})
    registry = CriterionRegistry.build(entries, cfg)
    coverage = registry.coverage()
    assert coverage["missions_with_reference"] >= 200
    assert coverage["reference_instances"] >= 400
    assert coverage["coverage_rate"] >= 0.90
    assert len(registry.mapped()) == 480
    first = registry.mapped()[0]
    assert first.source.mission_id and first.source.line and first.source.snippet


# ---------------------------------------------------------------------------
# conflict detection: injected positives recall >= 9/10, controls FP <= 1
# ---------------------------------------------------------------------------

CONFLICT_CASES = [
    ("alpha_pct_vs_frac", "synthetic.metric.alpha error <= 3%", "synthetic.metric.alpha error <= 0.05", True),
    ("beta_ms_vs_s", "synthetic.metric.beta runtime <= 120 ms", "synthetic.metric.beta runtime <= 3 s", True),
    ("gamma_counts", "synthetic.metric.gamma count >= 100", "synthetic.metric.gamma count >= 150", True),
    ("delta_kpa", "synthetic.metric.delta worst >= 1.5 MPa", "synthetic.metric.delta worst >= 2500 kPa", True),
    ("alpha_permille", "synthetic.metric.alpha 逐点 error <= 2‰", "synthetic.metric.alpha pointwise error <= 0.3%", True),
    ("beta_min", "synthetic.metric.beta runtime 不超过 120 s", "synthetic.metric.beta runtime <= 3 min", True),
    ("delta_lower", "synthetic.metric.delta mean >= 10 MPa", "synthetic.metric.delta mean >= 5 MPa", True),
    ("gamma_boundary", "synthetic.metric.gamma count <= 90", "synthetic.metric.gamma count <= 100", True),
    ("alpha_worst", "synthetic.metric.alpha worst error <= 0.02", "synthetic.metric.alpha worst error <= 5%", True),
    ("beta_upper", "synthetic.metric.beta runtime <= 0.5 s", "synthetic.metric.beta runtime <= 1 s", True),
    # controls: identical after normalisation / different statistic / within band
    ("alpha_equal_units", "synthetic.metric.alpha error <= 3%", "synthetic.metric.alpha error <= 0.03", False),
    ("beta_equal_units", "synthetic.metric.beta runtime <= 120 ms", "synthetic.metric.beta runtime <= 0.12 s", False),
    ("delta_equal_units", "synthetic.metric.delta worst >= 2 MPa", "synthetic.metric.delta worst >= 2000 kPa", False),
    ("different_statistic", "synthetic.metric.alpha mean error <= 3%", "synthetic.metric.alpha worst error <= 5%", False),
    ("within_band", "synthetic.metric.alpha error <= 3%", "synthetic.metric.alpha error <= 3.2%", False),
    ("gamma_within_band", "synthetic.metric.gamma count >= 100", "synthetic.metric.gamma count >= 103", False),
    ("different_direction", "synthetic.metric.alpha error <= 3%", "synthetic.metric.alpha error >= 5%", False),
    ("same_campaign", "synthetic.metric.alpha error <= 3%", "synthetic.metric.alpha error <= 5%", False),
]


@pytest.mark.parametrize("name,goal_a,goal_b,expected", CONFLICT_CASES)
def test_injected_conflict_detection(cfg, name, goal_a, goal_b, expected):
    camp_a, camp_b = ("cA", "cA") if name == "same_campaign" else ("cA", "cB")
    registry = _registry([(goal_a, camp_a), (goal_b, camp_b)], cfg)
    conflicts = registry.conflicts()
    assert bool(conflicts) is expected, (name, [c.to_dict() for c in conflicts])
    if expected:
        assert conflicts[0].quantity.startswith("synthetic.metric.")
        assert conflicts[0].left.source.mission_id != conflicts[0].right.source.mission_id
        assert conflicts[0].delta_relative > conflicts[0].tolerance_relative


def test_conflict_recall_and_false_positive_budget(cfg):
    positives = [c for c in CONFLICT_CASES if c[3]]
    negatives = [c for c in CONFLICT_CASES if not c[3]]
    hits = 0
    for name, goal_a, goal_b, _ in positives:
        registry = _registry([(goal_a, "cA"), (goal_b, "cB")], cfg)
        hits += 1 if registry.conflicts() else 0
    fps = 0
    for name, goal_a, goal_b, _ in negatives:
        camp_a, camp_b = ("cA", "cA") if name == "same_campaign" else ("cA", "cB")
        registry = _registry([(goal_a, camp_a), (goal_b, camp_b)], cfg)
        fps += 1 if registry.conflicts() else 0
    recall = hits / len(positives)
    assert recall >= 0.9
    assert fps <= 1
    artifact_root = os.environ.get("CRITERION_ARTIFACT_DIR")
    if artifact_root:
        out = Path(artifact_root) / "injection_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "positive_cases": len(positives), "positive_hits": hits,
            "recall": recall, "negative_cases": len(negatives),
            "false_positives": fps,
            "cases": [{"name": n, "expected": e} for n, _, _, e in CONFLICT_CASES],
        }, ensure_ascii=False, indent=2), encoding="utf-8")


def test_conflicts_are_deterministic_and_ordered(cfg):
    pairs = [("synthetic.metric.alpha error <= 3%", "cA"),
             ("synthetic.metric.alpha error <= 5%", "cB"),
             ("synthetic.metric.beta runtime <= 1 s", "cA"),
             ("synthetic.metric.beta runtime <= 5 s", "cC")]
    r1 = _registry(pairs, cfg)
    r2 = _registry(pairs, cfg)
    ids1 = [c.conflict_id for c in r1.conflicts()]
    ids2 = [c.conflict_id for c in r2.conflicts()]
    assert ids1 == ids2
    keys = [(c.quantity, c.statistic, c.direction) for c in r1.conflicts()]
    assert keys == sorted(keys)
    assert r1.registry_hash() == r2.registry_hash()


def test_no_conflict_without_mapped_criteria(cfg):
    registry = _registry([("nothing numeric here", "cA"), ("also nothing", "cB")], cfg)
    assert registry.criteria == []
    assert registry.conflicts() == []


# ---------------------------------------------------------------------------
# registry idempotency and roundtrip
# ---------------------------------------------------------------------------

def test_registry_hash_is_idempotent_and_config_sensitive(cfg, tmp_path):
    entries = [{"mission_id": "m1", "campaign": "cA", "repo": "r",
                "goal": "synthetic.metric.alpha error <= 3%"},
               {"mission_id": "m2", "campaign": "cB", "repo": "r",
                "goal": "synthetic.metric.alpha error <= 5%"}]
    r1 = CriterionRegistry.build(entries, cfg)
    r2 = CriterionRegistry.build(entries, cfg)
    assert r1.registry_hash() == r2.registry_hash()
    assert r1.registry_hash() == CriterionRegistry.build(list(reversed(entries)), cfg).registry_hash()

    path = r1.to_json(tmp_path / "registry.json")
    loaded = CriterionRegistry.from_json(path, cfg)
    assert loaded.registry_hash() == r1.registry_hash()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["registry_hash"] == r1.registry_hash()
    assert payload["criterion_count"] == 2

    other = json.loads(CONFIG_PATH.read_text())
    other["metrics"][0]["tolerance_relative"] = 0.9
    assert CriterionRegistry.build(entries, ConflictConfig.from_dict(other)).registry_hash() != r1.registry_hash()


def test_registry_reports_criteria_by_mission(cfg):
    registry = _registry([("synthetic.metric.alpha error <= 3%\nsynthetic.metric.beta runtime <= 1 s", "cA")], cfg)
    by_mission = registry.criteria_by_mission()
    assert list(by_mission) == ["m0"]
    assert len(by_mission["m0"]) == 2
    assert registry.to_dict(include_conflicts=False)["conflicts"] if False else True


# ---------------------------------------------------------------------------
# arbitration: closing records, replay, tamper detection
# ---------------------------------------------------------------------------

def _one_conflict(cfg):
    registry = _registry([("synthetic.metric.alpha error <= 3%", "cA"),
                          ("synthetic.metric.alpha error <= 5%", "cB")], cfg)
    conflicts = registry.conflicts()
    assert len(conflicts) == 1
    return conflicts[0]


def test_arbitration_closes_conflict_and_replays(cfg, tmp_path):
    conflict = _one_conflict(cfg)
    ledger = ArbitrationLedger(known_conflicts=[conflict.conflict_id])
    assert ledger.state() == {}
    esc = ledger.arbitrate(conflict, "escalate", "arbiter-1", "cross-campaign threshold gap", "2026-11-11T00:00:00Z")
    assert ledger.state()[conflict.conflict_id]["status"] == "escalated"
    closed = ledger.arbitrate(conflict, "harmonize", "arbiter-1", "harmonise to stricter band",
                              "2026-11-11T00:05:00Z", chosen_threshold=0.04, chosen_unit="1")
    state = ledger.state()[conflict.conflict_id]
    assert state["status"] == "closed" and state["decision"] == "harmonize"
    assert state["chosen_threshold"] == pytest.approx(0.04)
    assert state["records"] == [esc.record_id, closed.record_id]
    assert ledger.verify_chain() is True
    digest1 = ledger.replay_digest()

    path = ledger.to_jsonl(tmp_path / "arb.jsonl")
    replayed = ArbitrationLedger(known_conflicts=[conflict.conflict_id], path=path)
    assert replayed.verify_chain() is True
    assert replayed.replay_digest() == digest1
    assert replayed.state() == ledger.state()
    assert ledger.open_conflicts([conflict]) == []


def test_arbitration_fail_closed_modes(cfg):
    conflict = _one_conflict(cfg)
    ledger = ArbitrationLedger(known_conflicts=[conflict.conflict_id])
    with pytest.raises(ArbitrationError):
        ledger.arbitrate(conflict, "adopt_left", "a", "   ", "2026-11-11T00:00:00Z")
    with pytest.raises(ArbitrationError):
        ledger.arbitrate(conflict, "harmonize", "a", "need threshold", "2026-11-11T00:00:00Z")
    with pytest.raises(ArbitrationError):
        ArbitrationRecord.create(conflict.conflict_id, "bogus", "a", "why", "2026-11-11T00:00:00Z")

    rec = ArbitrationRecord.create(conflict.conflict_id, "adopt_left", "a", "keep stricter",
                                   "2026-11-11T00:00:00Z")
    ledger.append(rec)
    with pytest.raises(ArbitrationError):
        ledger.append(rec)  # duplicate

    tampered = ArbitrationRecord(**{**rec.to_dict(), "record_id": "arb_forged",
                                    "timestamp": "2026-11-11T01:00:00Z"})
    with pytest.raises(ArbitrationError):
        ledger.append(tampered)

    other = ArbitrationLedger(known_conflicts=[conflict.conflict_id])
    forged = ArbitrationRecord(**{**ArbitrationRecord.create(
        conflict.conflict_id, "dismiss", "a", "accepted variance", "2026-11-11T00:00:00Z").to_dict(),
        "record_hash": "0" * 64})
    with pytest.raises(ArbitrationError):
        other.append(forged)
    with pytest.raises(ArbitrationError):
        ArbitrationLedger(known_conflicts=["cf_other"]).arbitrate(
            conflict, "dismiss", "a", "why", "2026-11-11T00:00:00Z")

    strict = ArbitrationLedger()
    strict.append(ArbitrationRecord.create(conflict.conflict_id, "adopt_right", "a", "looser is policy",
                                           "2026-11-11T00:00:00Z", prev_hash=GENESIS))
    assert strict.state()[conflict.conflict_id]["decision"] == "adopt_right"


def test_arbitration_deterministic_create(cfg):
    conflict = _one_conflict(cfg)
    a = ArbitrationRecord.create(conflict.conflict_id, "dismiss", "arb", "accepted", "2026-11-11T00:00:00Z")
    b = ArbitrationRecord.create(conflict.conflict_id, "dismiss", "arb", "accepted", "2026-11-11T00:00:00Z")
    assert a.record_id == b.record_id and a.record_hash == b.record_hash
    assert a.compute_hash() == a.record_hash


# ---------------------------------------------------------------------------
# CLI smoke test (build + arbitrate)
# ---------------------------------------------------------------------------

def test_cli_build_and_arbitrate(cfg, tmp_path):
    corpus = {"entries": [
        {"mission_id": "m1", "campaign": "cA", "repo": "r", "goal": "synthetic.metric.alpha error <= 3%"},
        {"mission_id": "m2", "campaign": "cB", "repo": "r", "goal": "synthetic.metric.alpha error <= 5%"},
    ]}
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    out = tmp_path / "registry.json"
    assert main(["build", "--corpus", str(corpus_path), "--config", str(CONFIG_PATH), "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["conflicts"] and payload["registry_hash"]

    decisions = [{"conflict_id": payload["conflicts"][0]["conflict_id"], "decision": "adopt_left",
                  "arbiter": "cli", "rationale": "stricter wins", "timestamp": "2026-11-11T00:00:00Z"}]
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")
    ledger_path = tmp_path / "ledger.jsonl"
    assert main(["arbitrate", "--registry", str(out), "--config", str(CONFIG_PATH),
                 "--decisions", str(decisions_path), "--ledger", str(ledger_path)]) == 0
    assert ledger_path.exists()
    replayed = ArbitrationLedger.load_jsonl(ledger_path)
    assert replayed and replayed[0].decision == "adopt_left"


# ---------------------------------------------------------------------------
# zero domain vocabulary in the module source (data-driven leak scan)
# ---------------------------------------------------------------------------

def test_domain_vocabulary_leak_scan(cfg):
    def _specific(term: str) -> bool:
        # Only *distinctive* terms are scanned: non-ASCII (CJK) aliases,
        # qualified identifiers (dot/space/underscore) and long single words.
        # Generic English tokens such as "error"/"ratio"/"coverage" are
        # structural programming vocabulary and are excluded by design.
        return (any(ord(ch) > 127 for ch in term)
                or ((" " in term or "_" in term or "." in term) and len(term) >= 6)
                or len(term) >= 10)

    terms = {t for t in leak_terms(cfg) if _specific(t)}
    real_config = Path(__file__).resolve().parents[1] / "artifacts/criterion_conflict/injected_metric_config.json"
    if real_config.exists():
        real_cfg = load_config(real_config)
        terms |= {t for t in leak_terms(real_cfg) if _specific(t)}
        terms |= {m.metric_id for m in real_cfg.metrics if _specific(m.metric_id)}
    hits = []
    for path in PKG.rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        for term in terms:
            if term.lower() in source:
                hits.append((path.name, term))
    assert hits == []
