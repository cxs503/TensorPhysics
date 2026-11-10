"""Criterion tests for the failure-mode taxonomy (P186-W1-N08).

Fixtures are synthetic: ``node-*`` ids and bodies built from generic method
wording, so criterion C5 (zero domain semantics) can be asserted on this file
itself by the purity scanner.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorphysics.failure_taxonomy import (
    CLASS_CODES,
    DEFAULT_TABLE,
    RANDOM_BASELINE,
    UNCLASSIFIED,
    DecisionTableError,
    FailRecordView,
    RerouteDirective,
    SeedPrototypeLabeler,
    SignalRuleLabeler,
    blocked_node_view,
    cohen_kappa,
    decide,
    extract,
    load_cue_lexicon,
    load_decoys,
    load_frozen_set,
    load_rule_weights,
    load_seeds,
    permute_labels,
    replace_bodies,
    summarise_view,
    table_coverage,
    to_ledger_entry,
    validate_decision_table,
)

REPO = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO / "src" / "tensorphysics" / "failure_taxonomy"


@pytest.fixture(scope="module")
def annotator() -> SignalRuleLabeler:
    return SignalRuleLabeler(load_rule_weights(), load_cue_lexicon())


@pytest.fixture(scope="module")
def frozen():
    return load_frozen_set()


SYNTHETIC = {
    "BASELINE_ABSENT": "no reference value or published benchmark exists for this quantity, so accuracy cannot be judged.",
    "TOLERANCE_EXCEEDED": "the reported relative error is 18 % against the reference, beyond the 10 % tolerance band.",
    "CONVERGENCE_STALL": "the iteration stalled: the residual plateaued after 500 iterations and never reached tolerance.",
    "MODEL_FIDELITY_LIMIT": "the formulation assumes reduced dimensionality, so the absolute value cannot be extrapolated.",
    "DATA_INSUFFICIENCY": "the verdict is withheld because the deliverable artefacts were never written.",
    "IMPLEMENTATION_DEFECT": "the first version ignored an index offset, a code defect that is now patched.",
}


# --------------------------------------------------------------------------- C1
def test_c1_dual_annotation_kappa(frozen, annotator):
    """C1: analyst labels vs structured-rule annotator, Cohen kappa >= 0.6."""
    gold = [r.gold_class for r in frozen]
    pred = [annotator.label(r.body).code for r in frozen]
    kappa = cohen_kappa(gold, pred)
    assert kappa >= 0.6, f"kappa={kappa:.3f} < 0.6"


def test_c1_second_annotator_pair(frozen, annotator):
    """C1 robustness: the unsupervised prototype annotator brackets the value."""
    prototype = SeedPrototypeLabeler(load_seeds())
    gold = [r.gold_class for r in frozen]
    pred_a = [annotator.label(r.body).code for r in frozen]
    pred_b = [prototype.label(r.body).code for r in frozen]
    kappa = cohen_kappa(pred_a, pred_b)
    assert kappa >= 0.5, f"structured-vs-prototype kappa={kappa:.3f} unexpectedly low"


def test_kappa_definition_on_synthetic_labels():
    assert cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]) == 1.0
    assert cohen_kappa(["A", "B", "A", "B"], ["B", "A", "B", "A"]) == -1.0


# --------------------------------------------------------------------------- C2
def test_c2_frozen_set_consistency(frozen, annotator):
    """C2: structured annotator agrees with gold on >= 70 % of the frozen set."""
    gold = [r.gold_class for r in frozen]
    pred = [annotator.label(r.body).code for r in frozen]
    acc = sum(1 for a, b in zip(gold, pred) if a == b) / len(gold)
    assert acc >= 0.70, f"accuracy={acc:.3f} < 0.70 (chance={RANDOM_BASELINE:.3f})"
    assert not annotator.label("").code is None


def test_c2_corpus_is_balanced_and_pinned(frozen):
    counts: dict[str, int] = {}
    for rec in frozen:
        counts[rec.gold_class] = counts.get(rec.gold_class, 0) + 1
    assert set(counts) == set(CLASS_CODES)
    assert min(counts.values()) >= 5, counts


# --------------------------------------------------------------------------- C3
def test_c3a_decoy_bodies_collapse(frozen, annotator):
    """C3a: same-length non-FAIL text must not reproduce the gold labels."""
    bodies = [r.body for r in frozen]
    gold = [r.gold_class for r in frozen]
    decoys = replace_bodies(bodies, load_decoys(), seed=7)
    assert all(isinstance(d, str) and d for d in decoys)
    pred = [annotator.label(b).code for b in decoys]
    acc = sum(1 for a, b in zip(gold, pred) if a == b) / len(gold)
    assert acc <= 0.35, f"decoy accuracy={acc:.3f} > 0.35: the labeler keys on wording that survives decoys"


def test_c3b_label_permutation_collapses_to_chance(frozen):
    """C3b: permuted training labels destroy the supervised arm's test accuracy."""
    from tensorphysics.failure_taxonomy import SupervisedCentroidLabeler, stratum_split

    train, test = stratum_split(frozen, mod=3, remainder=2)
    gold_test = [r.gold_class for r in test]
    true_model = SupervisedCentroidLabeler().fit((r.body, r.gold_class) for r in train)
    acc_true = sum(
        1 for r in test if true_model.label(r.body).code == r.gold_class
    ) / len(test)

    permuted = permute_labels([r.gold_class for r in train], seed=11)
    shuffled = SupervisedCentroidLabeler().fit((r.body, code) for r, code in zip(train, permuted))
    acc_perm = sum(1 for r in test if shuffled.label(r.body).code == r.gold_class) / len(test)

    assert acc_true > acc_perm + 0.2, (acc_true, acc_perm)
    assert acc_perm <= 2 * RANDOM_BASELINE + 0.1, acc_perm


# --------------------------------------------------------------------------- C4
def test_c4_every_class_has_a_triggered_recommendation():
    coverage = table_coverage()
    assert coverage["classes"] == len(CLASS_CODES)
    assert coverage["min_triggers_per_row"] >= 1
    for code in CLASS_CODES:
        present = DEFAULT_TABLE[0].trigger_fields if False else ()
        row = next(r for r in DEFAULT_TABLE if r.failure_class == code)
        decision = decide(code, row.trigger_fields, table=DEFAULT_TABLE)
        assert decision.action != "ESCALATE"
        assert decision.matched_triggers, code
        assert not decision.evidence_gap


def test_c4_schema_gate_rejects_broken_tables():
    validate_decision_table(DEFAULT_TABLE)  # baseline is valid

    missing_class = [r for r in DEFAULT_TABLE if r.failure_class != CLASS_CODES[0]]
    with pytest.raises(DecisionTableError):
        validate_decision_table(missing_class)

    no_trigger = [
        RerouteDirective(
            failure_class=r.failure_class,
            action=r.action,
            secondary_action=r.secondary_action,
            trigger_fields=(),
            rationale_code=r.rationale_code,
            expected_effect=r.expected_effect,
        )
        for r in DEFAULT_TABLE
    ]
    with pytest.raises(DecisionTableError):
        validate_decision_table(no_trigger)

    bad_field = [
        RerouteDirective(
            failure_class=r.failure_class,
            action=r.action,
            secondary_action=r.secondary_action,
            trigger_fields=("not_a_signal",),
            rationale_code=r.rationale_code,
            expected_effect=r.expected_effect,
        )
        for r in DEFAULT_TABLE
    ]
    with pytest.raises(DecisionTableError):
        validate_decision_table(bad_field)

    bad_action = [
        RerouteDirective(
            failure_class=r.failure_class,
            action="GIVE_UP",
            secondary_action=r.secondary_action,
            trigger_fields=r.trigger_fields,
            rationale_code=r.rationale_code,
            expected_effect=r.expected_effect,
        )
        for r in DEFAULT_TABLE
    ]
    with pytest.raises(DecisionTableError):
        validate_decision_table(bad_action)


def test_c4_evidence_gap_escalates_instead_of_guessing():
    decision = decide("BASELINE_ABSENT", present_fields=(), table=DEFAULT_TABLE)
    assert decision.action == "ESCALATE"
    assert decision.evidence_gap is True


# --------------------------------------------------------------------------- C5
def test_c5_no_domain_vocabulary_in_package_or_tests():
    from tensorphysics.failure_taxonomy import assert_pure, load_domain_lexicon

    assert_pure([PACKAGE_DIR, Path(__file__)], load_domain_lexicon())


def test_c5_scanner_catches_an_implanted_term(tmp_path):
    from tensorphysics.failure_taxonomy import DomainPurityError, assert_pure

    planted = tmp_path / "leaky.py"
    planted.write_text("LABEL = 'rhino-tolerance'\n", encoding="utf-8")
    with pytest.raises(DomainPurityError):
        assert_pure([planted], ["rhino"])


# ------------------------------------------------------------------ integrity
def test_frozen_manifest_pin_matches_payload():
    from tensorphysics.failure_taxonomy.dataset import digest_of, load_manifest, load_raw_frozen_set

    manifest = load_manifest()
    raw = load_raw_frozen_set()
    assert len(raw) == manifest["n_records"]
    assert digest_of(raw) == manifest["sha256"]


def test_rule_weights_cover_known_signal_keys():
    from tensorphysics.failure_taxonomy import SIGNAL_KEYS

    weights = load_rule_weights()
    assert set(weights) == set(CLASS_CODES)
    for code, row in weights.items():
        assert row, code
        assert set(row) <= set(SIGNAL_KEYS), (code, set(row) - set(SIGNAL_KEYS))


def test_stratified_split_is_deterministic_and_disjoint():
    from tensorphysics.failure_taxonomy import stratum_split

    frozen = load_frozen_set()
    train, test = stratum_split(frozen, mod=3, remainder=2)
    assert len(train) + len(test) == len(frozen)
    assert not {r.record_id for r in train} & {r.record_id for r in test}
    train_again, test_again = stratum_split(load_frozen_set(), mod=3, remainder=2)
    assert [r.record_id for r in train] == [r.record_id for r in train_again]
    assert [r.record_id for r in test] == [r.record_id for r in test_again]


def test_c6_blocked_node_view_integration(annotator):
    """C6: every blocked node yields {failure_class, evidence, reroute_suggestion}."""
    from tensorphysics.campaign.p20_gate.dag import CampaignDAG, DAGNode

    dag = CampaignDAG(
        [
            DAGNode(node_id="node-01", name="upstream", status="failed"),
            DAGNode(node_id="node-02", name="downstream", status="queued", depends_on=("node-01",)),
            DAGNode(node_id="node-03", name="orphan", status="queued", depends_on=("node-01",)),
        ]
    )
    convergence = dag.verify_convergence()
    blocked = convergence.blocked_nodes
    assert blocked, "DAG fixture must produce blocked nodes"

    records = [
        FailRecordView(node_id="node-02", record_id="S1", body=SYNTHETIC["CONVERGENCE_STALL"]),
        FailRecordView(node_id="node-03", record_id="S2", body=SYNTHETIC["DATA_INSUFFICIENCY"]),
    ]
    view = blocked_node_view(blocked, records, labeler=annotator)

    assert len(view) == len(blocked)
    for item in view:
        assert {"node_id", "failure_class", "evidence", "reroute_suggestion"} <= set(item)
        assert item["failure_class"] in CLASS_CODES or item["failure_class"] == UNCLASSIFIED
        suggestion = item["reroute_suggestion"]
        assert {"action", "matched_triggers", "rationale_code", "expected_effect"} <= set(suggestion)

    stats = summarise_view(view)
    assert stats["n_blocked"] == len(blocked)
    assert 0.0 <= stats["coverage"] <= 1.0

    classified = [item for item in view if item["failure_class"] != UNCLASSIFIED]
    assert classified and all(item["reroute_suggestion"]["action"] != "ESCALATE" for item in classified)

    # the ledger rendering consumes falsified hypotheses, it does not delete them
    entry = to_ledger_entry(classified[0], mission_id=9338)
    assert entry["category"] == "falsified_hypothesis"
    assert entry["summary"] and entry["reason"]
    json.dumps(entry)  # must be JSON-serialisable for the campaign ledger


def test_unclassified_nodes_are_reported_not_dropped(annotator):
    view = blocked_node_view(["node-99"], [], labeler=annotator)
    assert view[0]["failure_class"] == UNCLASSIFIED
    assert view[0]["reroute_suggestion"]["action"] == "ESCALATE"
    assert summarise_view(view)["unclassified"] == 1


def test_evidence_extraction_is_domain_neutral(annotator):
    evidence = extract(SYNTHETIC["TOLERANCE_EXCEEDED"], annotator.lexicon)
    assert evidence.has("quantified_gap")
    assert evidence.has("threshold_lexeme")
    assert evidence.has("reference_lexeme")
    assert evidence.numeric_tokens
