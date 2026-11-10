"""Replay re-adjudicator: re-judge historical criteria under both families.

Given a corpus of parsed historical criteria (see
:mod:`tensorphysics.metrology.criteria_semantics.corpus`) this module produces, for each row:

* the **baseline** verdict (``|Δ|/|y_ref| < τ`` — what the report actually did),
* the **degeneracy** class of the criterion's metrology (T1/T2/T3 or NONE),
* the **adaptive** verdict plus the *numeric value of the replacement criterion*,
* a **flip** flag: did the verdict change once the metrology was made explicit?

Aggregate outputs are the "criterion-degeneracy ledger": degeneracy rate by type, flip
counts, and the subset of rows that were silently tolerated (baseline WITHHELD/"部分" but a
defined adaptive verdict exists).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .adaptive import evaluate, evaluate_baseline, merge_multi_metric
from .degeneracy import classify
from .model import Degeneracy, Evaluation, Verdict
from .parser import CriterionRow

#: kinds of flip we distinguish
FLIP_NONE = "none"
FLIP_FAIL_TO_PASS = "fail_to_pass"
FLIP_PASS_TO_FAIL = "pass_to_fail"
FLIP_UNDECIDED_TO_JUDGED = "undecided_to_judged"      # baseline WITHHELD → adaptive PASS/FAIL
FLIP_JUDGED_TO_UNDECIDED = "judged_to_undecided"      # adaptive lost the reference → WITHHELD
#: recorded (human) verdict → adaptive verdict.  The interesting one is
#: ``softened_to_fail``: the report wrote "部分"/"是" while the explicit criterion says FAIL.
REC_SOFTENED_TO_FAIL = "softened_to_fail"
REC_WITHHELD_TO_JUDGED = "withheld_to_judged"
REC_CONSISTENT = "consistent"
REC_OTHER = "other"


@dataclass
class ReplayEntry:
    name: str
    baseline: Evaluation
    adaptive: Evaluation
    degeneracy: Degeneracy
    degeneracy_witnesses: dict[str, Any]
    baseline_verdict: Verdict
    adaptive_verdict: Verdict
    flip: str
    recorded_verdict: str
    alternative_value: float | None      # error / tau under the replacement criterion
    alternative_metric: str
    self_referential: bool = False
    recorded_flip: str = REC_OTHER
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def is_degenerate(self) -> bool:
        return self.degeneracy is not Degeneracy.NONE

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "baseline": {"verdict": self.baseline_verdict.value,
                         "metric": self.baseline.metric.value,
                         "error": self.baseline.error,
                         "tau": self.baseline.tau,
                         "reason": self.baseline.reason},
            "adaptive": {"verdict": self.adaptive_verdict.value,
                         "metric": self.adaptive.metric.value,
                         "error": self.adaptive.error,
                         "tau": self.adaptive.tau,
                         "reason": self.adaptive.reason,
                         "remedy": self.adaptive.remedy},
            "recorded_verdict": self.recorded_verdict,
            "degeneracy": self.degeneracy.value,
            "self_referential": self.self_referential,
            "degeneracy_witnesses": self.degeneracy_witnesses,
            "alternative_criterion": {"metric": self.alternative_metric,
                                      "value": self.alternative_value},
            "flip": self.flip,
            "recorded_flip": self.recorded_flip,
            "provenance": self.provenance,
        }


def _flip(base: Verdict, adapt: Verdict) -> str:
    if base is Verdict.PASS and adapt is Verdict.FAIL:
        return FLIP_PASS_TO_FAIL
    if base is Verdict.FAIL and adapt is Verdict.PASS:
        return FLIP_FAIL_TO_PASS
    if base is Verdict.WITHHELD and adapt.is_judgeable:
        return FLIP_UNDECIDED_TO_JUDGED
    if base.is_judgeable and adapt is Verdict.WITHHELD:
        return FLIP_JUDGED_TO_UNDECIDED
    return FLIP_NONE


def _recorded_flip(recorded: str, adapt: Verdict) -> str:
    """Compare the verdict a human wrote with the adaptive verdict."""
    rec = (recorded or "unknown").lower()
    if rec in ("partial", "pass") and adapt is Verdict.FAIL:
        return REC_SOFTENED_TO_FAIL
    if rec == "withheld" and adapt.is_judgeable:
        return REC_WITHHELD_TO_JUDGED
    if rec == "unknown":
        return REC_OTHER
    ok = (rec == "pass" and adapt is Verdict.PASS) or (rec == "fail" and adapt is Verdict.FAIL) \
        or (rec == "withheld" and adapt is Verdict.WITHHELD)
    return REC_CONSISTENT if ok else REC_OTHER


def replay_row(row: CriterionRow) -> ReplayEntry:
    """Re-adjudicate a single historical criterion."""
    prov = row.provenance or {}
    sign_text = " ".join(str(prov.get(k) or "") for k in
                         ("quantity_cell", "error_cell", "reference_cell"))
    dg = classify(row.criterion, ref_text=prov.get("reference_cell"),
                  verdict_text=prov.get("verdict_cell"), sign_text=sign_text,
                  error_text=prov.get("error_cell"))
    base = evaluate_baseline(row.criterion)
    adapt = evaluate(row.criterion, degeneracy=dg)
    if adapt.error is not None:
        alt_value = adapt.ratio if adapt.tau > 0 else adapt.error
    else:
        alt_value = None
    tau, metric, _rule = merge_multi_metric(row.criterion)
    return ReplayEntry(
        name=row.criterion.name,
        baseline=base, adaptive=adapt, degeneracy=dg.kind,
        degeneracy_witnesses=dg.witnesses,
        self_referential=dg.self_referential,
        baseline_verdict=base.verdict, adaptive_verdict=adapt.verdict,
        flip=_flip(base.verdict, adapt.verdict),
        recorded_verdict=row.recorded_verdict,
        alternative_value=alt_value,
        alternative_metric=adapt.metric.value,
        recorded_flip=_recorded_flip(row.recorded_verdict, adapt.verdict),
        provenance=prov,
    )


def replay_corpus(rows: Iterable[CriterionRow]) -> dict[str, Any]:
    """Replay a whole corpus; return the degeneracy ledger with its aggregates."""
    entries = [replay_row(r) for r in rows]
    n = len(entries)
    by_type: dict[str, int] = {d.value: 0 for d in Degeneracy}
    flips: dict[str, int] = {}
    for e in entries:
        by_type[e.degeneracy.value] += 1
        flips[e.flip] = flips.get(e.flip, 0) + 1
    degenerate_strict = sum(v for k, v in by_type.items() if k != Degeneracy.NONE.value)
    n_self_ref = sum(1 for e in entries if e.self_referential)
    # extended definition: a self-referential reference is a criterion-degeneracy too
    # (the "reference" is the implementation, so the criterion cannot validate anything),
    # but it is reported separately because it is not one of the spec's T1/T2/T3 classes.
    degenerate = degenerate_strict + sum(
        1 for e in entries if e.self_referential and e.degeneracy is Degeneracy.NONE)
    withheld_baseline = sum(1 for e in entries if e.baseline_verdict is Verdict.WITHHELD)
    rec_flips: dict[str, int] = {}
    for e in entries:
        rec_flips[e.recorded_flip] = rec_flips.get(e.recorded_flip, 0) + 1
    return {
        "n_criteria": n,
        "n_degenerate": degenerate,
        "n_degenerate_strict": degenerate_strict,
        "n_self_referential": n_self_ref,
        "degeneracy_rate": (degenerate / n) if n else 0.0,
        "degeneracy_rate_strict": (degenerate_strict / n) if n else 0.0,
        "by_type": by_type,
        "flips": flips,
        "n_flips": sum(v for k, v in flips.items() if k != FLIP_NONE),
        "recorded_flips": rec_flips,
        "n_recorded_softened_to_fail": rec_flips.get(REC_SOFTENED_TO_FAIL, 0),
        "baseline_withheld": withheld_baseline,
        "n_reports": len({(e.provenance or {}).get("report_id") for e in entries
                          if (e.provenance or {}).get("report_id")}),
        "entries": entries,
    }


def summarize_ledger(ledger: dict[str, Any], *, max_entries: int | None = None) -> dict[str, Any]:
    """JSON-serialisable view of a replay ledger (the 'criterion-degeneracy list')."""
    entries = ledger["entries"]
    if max_entries is not None:
        entries = entries[:max_entries]
    return {k: v for k, v in ledger.items() if k != "entries"} | {
        "entries": [e.as_dict() for e in entries]}


def find_case(entries: Iterable[ReplayEntry], mission_id: int) -> list[ReplayEntry]:
    out = []
    for e in entries:
        prov = e.provenance or {}
        if prov.get("mission_id") == mission_id:
            out.append(e)
    return out