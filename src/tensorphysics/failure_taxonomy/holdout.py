"""Document-disjoint held-out evaluation of the failure taxonomy (external validity).

The frozen v1 back-test (:mod:`tensorphysics.failure_taxonomy.__main__`) grades the
classifier on the corpus it was developed against.  In-corpus agreement answers
"can the annotator reproduce the analyst on known passages"; it cannot answer
"does the annotator transfer to passages it never saw".  Two structural leaks make
the in-corpus number optimistic:

* **document leak** -- the frozen corpus draws several records from the same
  report, so a random record-level split puts sibling passages of one document on
  both sides.  Prose templates repeat inside a document, so the fit memorises the
  document, not the mechanism;
* **wording leak** -- the analyst's labels were adjudicated while the cue lexemes
  were being curated, so a classifier that keys on those same lexemes is graded on
  its own vocabulary.

This module supplies the machinery for the two audits that measure both leaks:

``load_heldout_set``     the v2 corpus (24 verbatim segments from 17 documents that
                         the v1 corpus does not use), digest-pinned and checked for
                         document disjointness on every load;
``group_folds`` / ``grouped_cv``
                         document-grouped k-fold splitting, plus the record-level
                         split as the *baseline* it is compared against;
``document_profile``     the structural facts behind the leak (records per
                         document, documents spanning several classes);
``ablate_lexicon``       leave-one-evidence-field-out ablation: drops one field's
                         lexemes, rebuilds the rule annotator and re-grades, so a
                         classification that rests on a single token family shows up
                         as a collapse instead of hiding inside the aggregate;
``evaluate``             accuracy / macro-F1 / kappa / per-class recall / confusion.

Everything here is domain-neutral: no application wording appears in this file, and
the corpora and lexica it reads live under ``data/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from .agreement import accuracy, cohen_kappa, confusion_matrix, macro_f1, per_class_recall
from .dataset import (
    DATA_DIR,
    FailRecord,
    FrozenSetError,
    digest_of,
    load_cue_lexicon,
    load_frozen_set,
    load_jsonl,
    load_rule_weights,
)
from .taxonomy import CLASS_CODES

HOLDOUT_FILE = "heldout_fail_set.jsonl"
HOLDOUT_MANIFEST_FILE = "heldout_manifest.json"


class HoldoutSetError(FrozenSetError):
    """The held-out corpus is unpinned, edited, or not document-disjoint."""


# --------------------------------------------------------------------------- load
def _holdout_manifest() -> dict:
    import json

    return json.loads((DATA_DIR / HOLDOUT_MANIFEST_FILE).read_text(encoding="utf-8"))


def load_heldout_set(verify: bool = True) -> list[FailRecord]:
    """Load the document-disjoint held-out corpus.

    With ``verify`` the payload digest, the record count and the disjointness
    property are re-checked against the pinned manifest and against the frozen v1
    corpus, so a silent edit or a document overlap raises instead of quietly
    inflating the reported transfer accuracy.
    """
    rows = load_jsonl(HOLDOUT_FILE)
    if verify:
        manifest = _holdout_manifest()
        if len(rows) != manifest["n_records"]:
            raise HoldoutSetError(f"held-out count {len(rows)} != pinned {manifest['n_records']}")
        if digest_of(rows) != manifest["sha256"]:
            raise HoldoutSetError("held-out corpus digest mismatch: the evaluation set was edited")
        heldout_docs = {r["provenance"]["source_path"] for r in rows}
        frozen_docs = {r.provenance["source_path"] for r in load_frozen_set()}
        overlap = heldout_docs & frozen_docs
        if overlap:
            raise HoldoutSetError(f"held-out documents overlap the frozen corpus: {sorted(overlap)}")
    return [
        FailRecord(
            record_id=r["record_id"],
            body=r["body"],
            gold_class=r["gold_class"],
            annotator=r.get("annotator", "unknown"),
            provenance=r.get("provenance", {}),
        )
        for r in rows
    ]


def holdout_manifest() -> dict:
    return _holdout_manifest()


# ----------------------------------------------------------------------- grouping
def document_of(record: FailRecord) -> str:
    """Grouping key of a record: the document it was cut from."""
    return str(record.provenance.get("source_path", record.record_id))


def document_profile(records: Sequence[FailRecord]) -> dict:
    """Structural facts that determine how much a record-level split can leak."""
    per_doc: dict[str, list[str]] = {}
    for rec in records:
        per_doc.setdefault(document_of(rec), []).append(rec.gold_class)
    multi = {d: len(c) for d, c in per_doc.items() if len(c) > 1}
    spanning = {d: sorted(set(c)) for d, c in per_doc.items() if len(set(c)) > 1}
    return {
        "n_records": len(records),
        "n_documents": len(per_doc),
        "records_per_document": dict(sorted((d, len(c)) for d, c in per_doc.items())),
        "n_documents_with_multiple_records": len(multi),
        "n_records_in_shared_documents": sum(multi.values()),
        "max_records_per_document": max((len(c) for c in per_doc.values()), default=0),
        "n_documents_spanning_classes": len(spanning),
        "documents_spanning_classes": {d: c for d, c in sorted(spanning.items())},
        "mean_records_per_document": round(len(records) / len(per_doc), 3) if per_doc else 0.0,
    }


def group_folds(records: Sequence[FailRecord], k: int = 5) -> list[list[int]]:
    """Deterministic document-grouped folds, returned as index lists.

    Documents are visited largest-first (ties broken by name) and each is placed in
    the currently smallest fold, so no document is ever split across folds and the
    fold sizes stay within one record of each other.  Determinism matters: the fold
    assignment is a pure function of the corpus, not of a random seed, so the
    reported numbers reproduce byte-for-byte.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    per_doc: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        per_doc.setdefault(document_of(rec), []).append(i)
    order = sorted(per_doc.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if k > len(order):
        raise ValueError(f"k={k} exceeds the number of documents ({len(order)})")
    folds: list[list[int]] = [[] for _ in range(k)]
    for _doc, idxs in order:
        target = min(range(k), key=lambda f: (len(folds[f]), f))
        folds[target].extend(idxs)
    return [sorted(f) for f in folds]


def grouped_split(records: Sequence[FailRecord], k: int = 5, fold: int = 0) -> tuple[list[FailRecord], list[FailRecord]]:
    """Document-disjoint train/test split: ``fold`` is test, the rest is train."""
    folds = group_folds(records, k=k)
    test_ids = set(folds[fold % k])
    train = [r for i, r in enumerate(records) if i not in test_ids]
    test = [r for i, r in enumerate(records) if i in test_ids]
    return train, test


# ---------------------------------------------------------------------- evaluation
def evaluate(pred: Sequence[str], gold: Sequence[str]) -> dict:
    """Agreement block for one annotator against gold labels."""
    return {
        "n": len(gold),
        "accuracy": round(accuracy(pred, gold), 6),
        "kappa": round(cohen_kappa(gold, pred), 6),
        "macro_f1": round(macro_f1(gold, pred), 6),
        "per_class_recall": {k: round(v, 6) for k, v in per_class_recall(gold, pred, CLASS_CODES).items()},
        "confusion": {
            code: dict(zip(CLASS_CODES, confusion_matrix(gold, pred, CLASS_CODES)[i]))
            for i, code in enumerate(CLASS_CODES)
        },
    }


def _labels(labeler, bodies: Iterable[str]) -> list[str]:
    return [labeler.label(b).code for b in bodies]


def record_level_cv(records: Sequence[FailRecord], build: Callable[[], object], k: int = 6) -> dict:
    """Record-level k-fold (the *leaky* baseline the grouped audit is compared to).

    Folds are formed by taking every ``k``-th record (stratified by the file order),
    i.e. the same idea as :func:`tensorphysics.failure_taxonomy.dataset.stratum_split`
    but repeated over all folds, so both arms are scored on the same number of
    held-out records.
    """
    folds = [list(range(i, len(records), k)) for i in range(k)]
    per_fold = []
    correct = 0
    for f in folds:
        test_ids = set(f)
        train = [r for i, r in enumerate(records) if i not in test_ids]
        test = [records[i] for i in f]
        labeler = build().fit((r.body, r.gold_class) for r in train)
        pred = _labels(labeler, [r.body for r in test])
        gold = [r.gold_class for r in test]
        acc = accuracy(pred, gold)
        correct += sum(1 for a, b in zip(pred, gold) if a == b)
        per_fold.append({"n_test": len(test), "accuracy": round(acc, 6)})
    return {
        "protocol": "record_level_kfold",
        "n_folds": k,
        "pooled_accuracy": round(correct / len(records), 6),
        "per_fold": per_fold,
    }


def grouped_cv(records: Sequence[FailRecord], build: Callable[[], object], k: int = 5) -> dict:
    """Document-grouped k-fold: the honest generalisation number for a fitted arm."""
    folds = group_folds(records, k=k)
    per_fold = []
    correct = 0
    for f in folds:
        test_ids = set(f)
        train = [r for i, r in enumerate(records) if i not in test_ids]
        test = [records[i] for i in f]
        labeler = build().fit((r.body, r.gold_class) for r in train)
        pred = _labels(labeler, [r.body for r in test])
        gold = [r.gold_class for r in test]
        correct += sum(1 for a, b in zip(pred, gold) if a == b)
        per_fold.append(
            {
                "n_test": len(test),
                "n_test_documents": len({document_of(r) for r in test}),
                "accuracy": round(accuracy(pred, gold), 6),
            }
        )
    return {
        "protocol": "document_grouped_kfold",
        "n_folds": k,
        "pooled_accuracy": round(correct / len(records), 6),
        "per_fold": per_fold,
    }


# ------------------------------------------------------------------ cue ablation
@dataclass(frozen=True)
class AblationRow:
    """One leave-one-field-out arm."""

    dropped_field: str
    n_lexemes: int
    accuracy: float
    delta: float

    def as_dict(self) -> dict:
        return {
            "dropped_field": self.dropped_field,
            "n_lexemes": self.n_lexemes,
            "accuracy": round(self.accuracy, 6),
            "delta_vs_full": round(self.delta, 6),
        }


def ablate_lexicon(
    records: Sequence[FailRecord],
    gold: Sequence[str],
    build: Callable[[Mapping[str, list[str]]], object],
    lexicon: Mapping[str, list[str]] | None = None,
) -> list[AblationRow]:
    """Leave-one-evidence-field-out ablation of the cue lexicon.

    For every field, its lexemes are removed, the annotator is rebuilt through
    ``build`` and re-graded on the same records.  ``delta_vs_full`` is the accuracy
    change: a large negative delta means the classification of this corpus rests on
    that single token family, i.e. the ``|delta|`` ordering is a shortcut-detector.
    """
    lex = dict(lexicon or load_cue_lexicon())
    full = build(lex)
    base = accuracy(_labels(full, [r.body for r in records]), gold)
    rows: list[AblationRow] = []
    for field in sorted(lex):
        reduced = {k: list(v) for k, v in lex.items() if k != field}
        labeler = build(reduced)
        acc = accuracy(_labels(labeler, [r.body for r in records]), gold)
        rows.append(AblationRow(dropped_field=field, n_lexemes=len(lex[field]), accuracy=acc, delta=acc - base))
    rows.sort(key=lambda r: (r.delta, r.dropped_field))
    return rows


def baseline_rule_weights() -> dict:
    """Convenience accessor so callers need not know the data file names."""
    return load_rule_weights()


__all__ = [
    "AblationRow",
    "HOLDOUT_FILE",
    "HOLDOUT_MANIFEST_FILE",
    "HoldoutSetError",
    "ablate_lexicon",
    "baseline_rule_weights",
    "document_of",
    "document_profile",
    "evaluate",
    "group_folds",
    "grouped_cv",
    "grouped_split",
    "holdout_manifest",
    "load_heldout_set",
    "record_level_cv",
]