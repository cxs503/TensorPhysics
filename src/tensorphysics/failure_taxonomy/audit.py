"""External-validity audit driver for the failure taxonomy.

What this asks, beyond the in-corpus back-test
----------------------------------------------
The frozen v1 back-test grades the annotator on passages that were in front of the
analyst while the cue lexicon was curated, and several of those passages are
siblings from the same report.  Both facts make the in-corpus agreement optimistic.
This driver measures the two leaks, the transfer accuracy on documents the
classifier has never seen, and whether the annotator's own score margin is a usable
abstention signal off-corpus.

Protocol (pre-registered before the audit was run)
--------------------------------------------------
E1  transfer accuracy of the *rule* annotator on the held-out corpus (zero refit,
    documents disjoint from v1) >= 0.70
E2  Cohen's kappa between the held-out analyst labels and that annotator >= 0.60
E3  document-leak estimate: |accuracy(document-grouped CV) - accuracy(record-level
    CV)| for the supervised arm on the frozen corpus <= 0.10
E4  no single-token shortcut: for every cue field, dropping its lexemes must not
    reduce the held-out accuracy by more than 0.25
E5  reroute coverage on held-out records: share of records receiving a non-ESCALATE
    directive from the decision table >= 0.90
E6  domain purity of the package + this driver: 0 hits
E7  selective gating: with the margin floor chosen on the *frozen* corpus (the
    smallest floor whose in-corpus risk is <= 0.20, i.e. no held-out information
    used), the held-out risk must not exceed the ungated held-out risk while
    coverage stays >= 0.60.  If E7 holds, abstention is a working second line of
    defence; if it fails, the margin carries no off-corpus information and the
    reroute must be treated as advisory only.

E1/E2 are the same numeric bars as the in-corpus criteria C2/C1, deliberately: the
point is to test whether the in-corpus numbers survive the harder protocol, not to
move the goalposts.  The audit is *allowed to fail*; a failing verdict is a result
and is reported as one.

Usage::

    PYTHONPATH=src python -m tensorphysics.failure_taxonomy.audit --out artifacts/failure_taxonomy_external/audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .agreement import accuracy, cohen_kappa, permute_labels, replace_bodies
from .dataset import load_cue_lexicon, load_decoys, load_frozen_set, load_rule_weights, load_seeds
from .decide import ESCALATE, decide
from .evidence import extract
from .holdout import (
    ablate_lexicon,
    document_profile,
    evaluate,
    group_folds,
    grouped_cv,
    holdout_manifest,
    load_heldout_set,
    record_level_cv,
)
from .labelers import SeedPrototypeLabeler, SignalRuleLabeler, SupervisedCentroidLabeler
from .purity import assert_pure
from .selective import aurc, margins, risk_coverage
from .taxonomy import CLASS_CODES, RANDOM_BASELINE
from .view import FailRecordView, blocked_node_view, summarise_view

N_DECOY_DRAWS = 10
N_PERMUTATIONS = 25
GATE_RISK_TARGET = 0.20
GATE_MIN_COVERAGE = 0.60


def _labels(labeler, bodies) -> list[str]:
    return [labeler.label(b).code for b in bodies]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _domain_terms():
    from .dataset import load_domain_lexicon

    return load_domain_lexicon()


def _route_coverage(records, cue) -> dict:
    """Share of records for which the decision table fires a reroute move."""
    routed, escalated, actions = 0, 0, {}
    for rec in records:
        decision = decide(rec.gold_class, extract(rec.body, cue).present())
        if decision.action == ESCALATE:
            escalated += 1
        else:
            routed += 1
            actions[decision.action] = actions.get(decision.action, 0) + 1
    n = len(records)
    return {
        "n": n,
        "routed": routed,
        "escalated": escalated,
        "coverage": round(routed / n, 6) if n else 0.0,
        "actions": dict(sorted(actions.items())),
    }


def _choose_floor(curve: list[dict], risk_target: float, min_coverage: float) -> dict:
    """Smallest margin floor whose in-corpus risk meets ``risk_target``."""
    ok = [row for row in curve if row["risk"] <= risk_target and row["coverage"] >= min_coverage]
    if ok:
        return min(ok, key=lambda r: r["threshold"])
    return {"threshold": 0.0, "coverage": curve[0]["coverage"] if curve else 0.0, "risk": curve[0]["risk"] if curve else 1.0}


def _risk_at(curve: list[dict], floor: float) -> dict:
    return min(curve, key=lambda r: abs(r["threshold"] - floor))


def audit() -> dict[str, Any]:
    frozen = load_frozen_set()
    heldout = load_heldout_set()
    weights = load_rule_weights()
    cue = load_cue_lexicon()
    seeds = load_seeds()
    rule = SignalRuleLabeler(weights, cue)

    frozen_gold = [r.gold_class for r in frozen]
    heldout_gold = [r.gold_class for r in heldout]

    # ------------------------------------------------------------- in-corpus recap
    in_corpus = {
        "rule_vs_gold": evaluate(_labels(rule, [r.body for r in frozen]), frozen_gold),
        "prototype_vs_gold": evaluate(_labels(SeedPrototypeLabeler(seeds), [r.body for r in frozen]), frozen_gold),
        "random_baseline": RANDOM_BASELINE,
    }
    frozen_trained = SupervisedCentroidLabeler().fit((r.body, r.gold_class) for r in frozen)
    in_corpus["supervised_trained_on_frozen__in_corpus"] = evaluate(
        _labels(frozen_trained, [r.body for r in frozen]), frozen_gold,
    )
    _decoys = load_decoys()
    _frozen_pred = _labels(rule, [r.body for r in frozen])
    in_corpus["negative_controls"] = {
        "decoy_bodies_mean_accuracy": round(
            _mean([
                accuracy(_labels(rule, replace_bodies([r.body for r in frozen], _decoys, seed=300 + d)), frozen_gold)
                for d in range(N_DECOY_DRAWS)
            ]),
            6,
        ),
        "permuted_labels_mean_kappa": round(
            _mean([cohen_kappa(permute_labels(frozen_gold, seed=400 + i), _frozen_pred) for i in range(N_PERMUTATIONS)]),
            6,
        ),
    }

    # ------------------------------------------------- transfer on unseen documents
    transfer = {
        "rule_zero_refit": evaluate(_labels(rule, [r.body for r in heldout]), heldout_gold),
        "prototype_zero_refit": evaluate(_labels(SeedPrototypeLabeler(seeds), [r.body for r in heldout]), heldout_gold),
        "supervised_trained_on_frozen": evaluate(_labels(frozen_trained, [r.body for r in heldout]), heldout_gold),
        "supervised_grouped_cv_on_heldout": grouped_cv(heldout, SupervisedCentroidLabeler, k=4),
    }
    transfer["accuracy_drop_rule"] = round(
        in_corpus["rule_vs_gold"]["accuracy"] - transfer["rule_zero_refit"]["accuracy"], 6
    )
    transfer["kappa_drop_rule"] = round(in_corpus["rule_vs_gold"]["kappa"] - transfer["rule_zero_refit"]["kappa"], 6)
    transfer["per_class_recall_frozen"] = in_corpus["rule_vs_gold"]["per_class_recall"]
    transfer["per_class_recall_heldout"] = transfer["rule_zero_refit"]["per_class_recall"]
    transfer["per_class_recall_drop"] = {
        code: round(
            in_corpus["rule_vs_gold"]["per_class_recall"][code]
            - transfer["rule_zero_refit"]["per_class_recall"][code],
            6,
        )
        for code in CLASS_CODES
    }
    # The boundary confusions the class definitions themselves warn about.
    conf = transfer["rule_zero_refit"]["confusion"]
    boundary_pairs = []
    for gold_code in CLASS_CODES:
        for pred_code, n in conf[gold_code].items():
            if pred_code != gold_code and n:
                boundary_pairs.append({"gold": gold_code, "predicted": pred_code, "n": n})
    boundary_pairs.sort(key=lambda r: (-r["n"], r["gold"], r["predicted"]))
    transfer["boundary_confusions"] = boundary_pairs

    # ------------------------------------------------------- document-leak estimate
    leak = {
        "frozen_documents": document_profile(frozen),
        "heldout_documents": document_profile(heldout),
        "supervised_record_level_cv_frozen": record_level_cv(frozen, SupervisedCentroidLabeler, k=6),
        "supervised_grouped_cv_frozen": grouped_cv(frozen, SupervisedCentroidLabeler, k=5),
    }
    leak["accuracy_gap_grouped_minus_record_level"] = round(
        leak["supervised_grouped_cv_frozen"]["pooled_accuracy"]
        - leak["supervised_record_level_cv_frozen"]["pooled_accuracy"],
        6,
    )

    # ----------------------------------------------------- negative controls (held-out)
    decoys = load_decoys()
    decoy_accs = [accuracy(_labels(rule, replace_bodies([r.body for r in heldout], decoys, seed=100 + d)), heldout_gold)
                  for d in range(N_DECOY_DRAWS)]
    rule_heldout_pred = _labels(rule, [r.body for r in heldout])
    perm_kappas = [cohen_kappa(permute_labels(heldout_gold, seed=200 + i), rule_heldout_pred) for i in range(N_PERMUTATIONS)]
    controls = {
        "decoy_bodies": {
            "mean_accuracy": round(_mean(decoy_accs), 6),
            "max_accuracy": round(max(decoy_accs), 6),
            "n_draws": N_DECOY_DRAWS,
        },
        "permuted_labels_kappa_vs_rule": {
            "mean": round(_mean(perm_kappas), 6),
            "max": round(max(perm_kappas), 6),
            "n_permutations": N_PERMUTATIONS,
        },
    }

    # ----------------------------------------------------------------- cue ablation
    build = lambda lex: SignalRuleLabeler(weights, lex)  # noqa: E731
    ablation = {
        "frozen": [row.as_dict() for row in ablate_lexicon(frozen, frozen_gold, build, cue)],
        "heldout": [row.as_dict() for row in ablate_lexicon(heldout, heldout_gold, build, cue)],
    }

    # --------------------------------------------------------- reroute trigger cover
    reroute = {"frozen": _route_coverage(frozen, cue), "heldout": _route_coverage(heldout, cue)}

    # ------------------------------------------------ selective gating (E7)
    frozen_marg = margins(rule, [r.body for r in frozen], cue)
    heldout_marg = margins(rule, [r.body for r in heldout], cue)
    curve_frozen = risk_coverage([m.code for m in frozen_marg], [m.margin for m in frozen_marg], frozen_gold)
    curve_heldout = risk_coverage([m.code for m in heldout_marg], [m.margin for m in heldout_marg], heldout_gold)
    floor_row = _choose_floor(curve_frozen, GATE_RISK_TARGET, GATE_MIN_COVERAGE)
    floor = float(floor_row["threshold"])
    ungated = [row for row in curve_heldout if row["threshold"] == 0.0][0]
    gated = _risk_at(curve_heldout, floor)
    selective = {
        "risk_target_frozen": GATE_RISK_TARGET,
        "min_coverage": GATE_MIN_COVERAGE,
        "floor_chosen_on_frozen": floor,
        "frozen_curve": curve_frozen,
        "heldout_curve": curve_heldout,
        "aurc_frozen": aurc(curve_frozen),
        "aurc_heldout": aurc(curve_heldout),
        "heldout_ungated": ungated,
        "heldout_gated": gated,
        "risk_reduction_heldout": round(ungated["risk"] - gated["risk"], 6),
    }

    # ------------------------------------------------- view integration (E5 + gate)
    def _view(records, floor_value):
        views = [
            FailRecordView(node_id=r.record_id, body=r.body, record_id=r.record_id, provenance=r.provenance)
            for r in records
        ]
        return blocked_node_view(
            [r.record_id for r in records], views, labeler=rule, margin_floor=floor_value
        )

    view_plain = _view(heldout, None)
    view_gated = _view(heldout, floor)
    reroute["view_heldout"] = summarise_view(view_plain)
    reroute["view_heldout_gated"] = summarise_view(view_gated)
    reroute["view_gated_abstentions"] = sum(
        1 for item in view_gated if item["evidence"].get("abstained")
    )

    # ---------------------------------------------------------------- purity (C5/E6)
    purity_hits: list[str] = []
    try:
        assert_pure([Path(__file__).resolve().parent], _domain_terms())
    except AssertionError as exc:  # DomainPurityError
        purity_hits.append(str(exc))

    verdicts = {
        "E1_transfer_accuracy": transfer["rule_zero_refit"]["accuracy"] >= 0.70,
        "E2_transfer_kappa": transfer["rule_zero_refit"]["kappa"] >= 0.60,
        "E3_document_leak_within_bound": abs(leak["accuracy_gap_grouped_minus_record_level"]) <= 0.10,
        "E4_no_single_field_shortcut": all(row["delta_vs_full"] >= -0.25 for row in ablation["heldout"]),
        "E5_reroute_coverage": reroute["heldout"]["coverage"] >= 0.90,
        "E6_domain_purity": not purity_hits,
        "E7_selective_gating": (
            gated["risk"] <= ungated["risk"] and gated["coverage"] >= GATE_MIN_COVERAGE
        ),
    }
    return {
        "protocol_version": "failure-taxonomy-external-validity-v1",
        "preregistered_bars": {
            "E1_transfer_accuracy": 0.70,
            "E2_transfer_kappa": 0.60,
            "E3_document_leak_max_abs_gap": 0.10,
            "E4_ablation_min_delta": -0.25,
            "E5_reroute_coverage": 0.90,
            "E6_domain_purity_hits": 0,
            "E7_selective_gating": {"risk_target_frozen": GATE_RISK_TARGET, "min_coverage": GATE_MIN_COVERAGE},
        },
        "holdout_manifest": holdout_manifest(),
        "folds_heldout": [[heldout[i].record_id for i in f] for f in group_folds(heldout, k=4)],
        "in_corpus": in_corpus,
        "transfer": transfer,
        "leak": leak,
        "negative_controls": controls,
        "ablation": ablation,
        "reroute": reroute,
        "selective": selective,
        "purity_hits": purity_hits,
        "verdicts": verdicts,
        "all_pass": all(verdicts.values()),
        "classes": list(CLASS_CODES),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/failure_taxonomy_external/audit.json")
    args = ap.parse_args(argv)
    result = audit()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdicts": result["verdicts"], "all_pass": result["all_pass"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())