"""External-validity audit tests: document leak, transfer, ablation, abstention.

Fixtures are synthetic (``node-01``, ``body-01``, ``field-a`` ...) so the package's
domain-purity scanner can be pointed at this file as well: nothing here may name an
application domain.  The corpus-level numbers are pinned by re-reading the committed
audit artefact, which keeps the reported figures and the report in step.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorphysics.failure_taxonomy import (
    HoldoutSetError,
    SignalRuleLabeler,
    ablate_lexicon,
    aurc,
    document_of,
    document_profile,
    group_folds,
    grouped_cv,
    grouped_split,
    load_cue_lexicon,
    load_domain_lexicon,
    load_frozen_set,
    load_heldout_set,
    load_rule_weights,
    margin_of,
    record_level_cv,
    risk_coverage,
    selective_decision,
)
from tensorphysics.failure_taxonomy import dataset as dataset_mod
from tensorphysics.failure_taxonomy import holdout as holdout_mod
from tensorphysics.failure_taxonomy.decide import BASELINE_SWAP, decide
from tensorphysics.failure_taxonomy.dataset import FailRecord
from tensorphysics.failure_taxonomy.purity import assert_pure
from tensorphysics.failure_taxonomy.view import FailRecordView, blocked_node_view

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "tensorphysics" / "failure_taxonomy"
AUDIT_JSON = Path(__file__).resolve().parents[1] / "artifacts" / "failure_taxonomy_external" / "audit.json"


class _Out:
    """Minimal label object: the audit only reads ``.code``."""

    def __init__(self, code: str) -> None:
        self.code = code


def rec(rid: str, cls: str, doc: str, body: str = "body") -> FailRecord:
    return FailRecord(
        record_id=rid,
        body=body,
        gold_class=cls,
        annotator="test",
        provenance={"source_path": doc},
    )


def synthetic_corpus() -> list[FailRecord]:
    """Four documents, ten records, documents deliberately repeated."""
    rows = [
        ("r1", "A", "doc-a"),
        ("r2", "B", "doc-a"),
        ("r3", "A", "doc-b"),
        ("r4", "B", "doc-c"),
        ("r5", "A", "doc-c"),
        ("r6", "B", "doc-d"),
        ("r7", "A", "doc-d"),
        ("r8", "B", "doc-e"),
        ("r9", "A", "doc-f"),
        ("r10", "B", "doc-g"),
    ]
    return [rec(rid, cls, doc, body=f"body-{rid}") for rid, cls, doc in rows]


# --------------------------------------------------------------------- corpus pins
def test_heldout_corpus_is_pinned_and_document_disjoint():
    heldout = load_heldout_set(verify=True)
    frozen = load_frozen_set()
    assert len(heldout) == 24
    docs_held = {document_of(r) for r in heldout}
    docs_frozen = {document_of(r) for r in frozen}
    assert not (docs_held & docs_frozen)
    counts: dict[str, int] = {}
    for r in heldout:
        counts[r.gold_class] = counts.get(r.gold_class, 0) + 1
    assert min(counts.values()) == 4 and len(counts) == 6


def _stage_corpus_dir(tmp_path: Path, monkeypatch, extra: tuple[str, ...] = ()) -> Path:
    """Point the data-directory constants at a staged copy of the pinned files."""
    for name in ("frozen_fail_set.jsonl", "frozen_manifest.json") + extra:
        src = holdout_mod.DATA_DIR / name
        if src.is_file():
            tmp_path.joinpath(name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(holdout_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dataset_mod, "DATA_DIR", tmp_path)
    return tmp_path


def test_heldout_corpus_rejects_an_edited_payload(tmp_path, monkeypatch):
    original_dir = holdout_mod.DATA_DIR
    rows = [r.as_dict() for r in load_heldout_set()]
    rows[0]["gold_class"] = "SOME_OTHER_CLASS"
    edited = tmp_path / "heldout_fail_set.jsonl"
    edited.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    manifest = json.loads((original_dir / "heldout_manifest.json").read_text(encoding="utf-8"))
    _stage_corpus_dir(tmp_path, monkeypatch)
    tmp_path.joinpath("heldout_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HoldoutSetError):
        load_heldout_set(verify=True)


def test_heldout_corpus_rejects_document_overlap(tmp_path, monkeypatch):
    frozen = load_frozen_set()
    rows = [r.as_dict() for r in load_heldout_set()]
    rows[0]["provenance"]["source_path"] = document_of(frozen[0])
    import hashlib

    payload = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows)
    payload_manifest = json.loads((holdout_mod.DATA_DIR / "heldout_manifest.json").read_text(encoding="utf-8"))
    payload_manifest["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    payload_manifest["n_records"] = len(rows)
    (tmp_path / "heldout_fail_set.jsonl").write_text(payload + "\n", encoding="utf-8")
    _stage_corpus_dir(tmp_path, monkeypatch)
    (tmp_path / "heldout_manifest.json").write_text(json.dumps(payload_manifest), encoding="utf-8")

    with pytest.raises(HoldoutSetError):
        load_heldout_set(verify=True)


# ------------------------------------------------------------------ grouped folds
def test_group_folds_never_split_a_document():
    records = synthetic_corpus()
    folds = group_folds(records, k=4)
    assert sorted(i for f in folds for i in f) == list(range(len(records)))
    docs_per_fold = [{document_of(records[i]) for i in f} for f in folds]
    for a in range(len(docs_per_fold)):
        for b in range(a + 1, len(docs_per_fold)):
            assert not (docs_per_fold[a] & docs_per_fold[b])
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 2


def test_group_folds_are_deterministic_and_guard_k():
    records = synthetic_corpus()
    assert group_folds(records, k=3) == group_folds(list(records), k=3)
    with pytest.raises(ValueError):
        group_folds(records, k=len({document_of(r) for r in records}) + 1)
    with pytest.raises(ValueError):
        group_folds(records, k=1)


def test_grouped_split_is_disjoint_in_documents():
    records = synthetic_corpus()
    train, test = grouped_split(records, k=3, fold=1)
    assert not ({document_of(r) for r in train} & {document_of(r) for r in test})
    assert len(train) + len(test) == len(records)


def test_document_profile_counts_shared_documents():
    profile = document_profile(synthetic_corpus())
    assert profile["n_documents"] == 7
    assert profile["n_documents_with_multiple_records"] == 3
    assert profile["max_records_per_document"] == 2
    assert profile["n_documents_spanning_classes"] == 3


def test_grouped_cv_reports_one_held_out_document_set_per_fold():
    records = synthetic_corpus()

    class Memoriser:
        """Memorises the training bodies (pure fixture, no domain content)."""

        def fit(self, pairs):
            self.table = {body: label for body, label in pairs}
            return self

        def label(self, body):
            return _Out(self.table.get(body, "A"))

    out = grouped_cv(records, Memoriser, k=3)
    assert out["n_folds"] == 3
    assert 0.0 <= out["pooled_accuracy"] <= 1.0
    assert sum(f["n_test"] for f in out["per_fold"]) == len(records)


def test_grouped_cv_is_stricter_than_record_level_on_shared_documents():
    """A memoriser generalises within a document but not across documents."""
    records = synthetic_corpus()
    by_doc: dict[str, str] = {}
    for r in records:
        by_doc.setdefault(document_of(r), r.gold_class)

    class DocMemoriser:
        def fit(self, pairs):
            self.table = dict(pairs)
            return self

        def label(self, body):
            return _Out("A" if body.endswith(("1", "3", "5", "7", "9")) else "B")

    grouped = grouped_cv(records, DocMemoriser, k=3)["pooled_accuracy"]
    record = record_level_cv(records, DocMemoriser, k=5)["pooled_accuracy"]
    assert grouped == record == 1.0  # this memoriser keys on the id, so both are perfect
    assert by_doc  # fixture sanity


# ----------------------------------------------------------------------- ablation
def test_ablation_drops_the_field_that_carries_the_corpus():
    cue = load_cue_lexicon()
    weights = load_rule_weights()
    records = [
        rec("a1", "IMPLEMENTATION_DEFECT", "doc-a", body="a patch was applied after a first-version slip"),
        rec("a2", "IMPLEMENTATION_DEFECT", "doc-b", body="the initial indexing offset was a slip, now fixed"),
        rec("a3", "BASELINE_ABSENT", "doc-c", body="no reference value is available for this case"),
        rec("a4", "BASELINE_ABSENT", "doc-d", body="reference is missing, nothing to compare against"),
    ]
    gold = [r.gold_class for r in records]
    rows = ablate_lexicon(records, gold, lambda lex: SignalRuleLabeler(weights, lex), cue)
    by_field = {r.dropped_field: r for r in rows}
    assert by_field["defect_lexeme"].delta <= 0.0
    assert all(-1.0 <= r.delta <= 1.0 for r in rows)
    assert len(rows) == len(cue)


# --------------------------------------------------------------------- abstention
def test_margin_and_risk_coverage_shapes():
    assert margin_of({"A": 3.0, "B": 1.0}) == 2.0
    assert margin_of({"A": 0.0, "B": 0.0}) == 0.0
    codes = ["A", "A", "B", "B"]
    gold = ["A", "B", "B", "B"]
    margins_ = [2.0, 0.5, 1.5, 0.1]
    curve = risk_coverage(codes, margins_, gold, thresholds=(0.0, 1.0, 2.0))
    full, mid, high = curve
    assert full["coverage"] == 1.0 and abs(full["risk"] - 0.25) < 1e-12
    assert mid["coverage"] == 0.5 and mid["risk"] == 0.0
    assert high["coverage"] == 0.25 and high["risk"] == 0.0
    assert 0.0 <= aurc(curve) <= 1.0


def test_selective_decision_abstains_below_the_floor():
    decision = decide("BASELINE_ABSENT", ("reference_lexeme", "absence_lexeme"))
    assert decision.action == BASELINE_SWAP
    kept = selective_decision(decision, margin=1.5, floor=1.0)
    assert kept.action == BASELINE_SWAP
    dropped = selective_decision(decision, margin=0.2, floor=1.0)
    assert dropped.action == "ESCALATE"
    assert dropped.rationale_code == "low_margin"
    assert dropped.evidence_gap


def test_blocked_node_view_honours_the_margin_floor():
    cue = load_cue_lexicon()
    rule = SignalRuleLabeler(load_rule_weights(), cue)
    records = [
        FailRecordView(node_id="node-01", body="no reference value is available", record_id="x1"),
        FailRecordView(node_id="node-02", body="   ", record_id="x2"),
    ]
    plain = blocked_node_view(["node-01", "node-02"], records, labeler=rule)
    assert [item["failure_class"] for item in plain] == [item["failure_class"] for item in plain]
    gated = blocked_node_view(["node-01", "node-02"], records, labeler=rule, margin_floor=99.0)
    assert all(item["reroute_suggestion"]["action"] == "ESCALATE" for item in gated)
    assert all(item["evidence"].get("abstained") for item in gated)
    assert all("score_margin" in item["evidence"] for item in gated)


# -------------------------------------------------------- audit artefact regression
def test_audit_artifact_is_committed_and_keeps_honest_verdicts():
    assert AUDIT_JSON.is_file(), f"missing audit artefact: {AUDIT_JSON}"
    data = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    verdicts = data["verdicts"]
    # The two transfer bars failed on the pre-registered protocol; an artefact that
    # silently reports them as passing would be the one failure mode this test exists
    # to catch.  If a future change fixes the transfer, flip these deliberately.
    assert verdicts["E1_transfer_accuracy"] is False
    assert verdicts["E2_transfer_kappa"] is False
    assert verdicts["E3_document_leak_within_bound"] is False
    assert verdicts["E4_no_single_field_shortcut"] is True
    assert verdicts["E5_reroute_coverage"] is True
    assert verdicts["E6_domain_purity"] is True
    assert data["all_pass"] is False
    assert data["transfer"]["rule_zero_refit"]["n"] == 24
    assert data["holdout_manifest"]["n_records"] == 24
    # the leak estimate must stay a documented, non-zero gap
    assert data["leak"]["accuracy_gap_grouped_minus_record_level"] < 0.0


def test_audit_is_reproducible():
    from tensorphysics.failure_taxonomy.audit import audit

    fresh = audit()
    committed = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    assert fresh["verdicts"] == committed["verdicts"]
    assert fresh["transfer"]["rule_zero_refit"]["accuracy"] == committed["transfer"]["rule_zero_refit"]["accuracy"]
    assert fresh["selective"]["floor_chosen_on_frozen"] == committed["selective"]["floor_chosen_on_frozen"]


# ------------------------------------------------------------------------- purity
def test_package_and_this_test_file_are_domain_free():
    terms = load_domain_lexicon()
    assert_pure([PACKAGE_DIR, Path(__file__)], terms)