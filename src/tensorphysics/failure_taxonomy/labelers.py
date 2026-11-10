"""Labelers used by the dual-annotation and negative-control studies.

Three annotators with deliberately different inductive biases are provided:

``SignalRuleLabeler`` (annotator **A**, *structured*)
    Reads only the boolean signal vector produced by
    :mod:`tensorphysics.failure_taxonomy.evidence` and applies a weighted,
    externally stored rule table.  It never sees the raw prose.

``SeedPrototypeLabeler`` (annotator **B**, *distributional, unsupervised*)
    Character n-gram TF-IDF centroid over externally stored *seed exemplars*
    (a handful of generic sentences per class, written for the taxonomy, not
    lifted from the back-test corpus).  No gold label is used at any point, so
    B is an independent second opinion rather than a fitted copy of the analyst.

``SupervisedCentroidLabeler`` (learned arm)
    Fitted on the train split of the frozen corpus and evaluated on the held-out
    split; this is the arm used for the label-permutation negative control
    (criterion C3b), whose accuracy must collapse to the 1/6 chance level.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .evidence import SIGNAL_KEYS, Evidence, extract
from .taxonomy import CLASS_CODES


def _normalise(text: str) -> str:
    return " ".join((text or "").split()).lower()


def ngrams(text: str, sizes: Sequence[int] = (2, 3)) -> list[str]:
    """Character n-grams of the whitespace-collapsed, lower-cased text."""
    t = _normalise(text)
    out: list[str] = []
    for n in sizes:
        out.extend(t[i : i + n] for i in range(max(0, len(t) - n + 1)))
    return out


@dataclass
class _Tfidf:
    """Minimal tf-idf vectoriser (kept dependency-free on purpose)."""

    df: dict[str, int] = field(default_factory=dict)
    n_docs: int = 0

    def fit(self, docs: Iterable[str]) -> "_Tfidf":
        self.df = {}
        self.n_docs = 0
        for doc in docs:
            self.n_docs += 1
            for term in set(ngrams(doc)):
                self.df[term] = self.df.get(term, 0) + 1
        return self

    def weigh(self, text: str) -> dict[str, float]:
        grams = ngrams(text)
        if not grams:
            return {}
        counts: dict[str, int] = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        total = float(len(grams))
        vec: dict[str, float] = {}
        for g, c in counts.items():
            df = self.df.get(g, 0)
            idf = math.log((1.0 + self.n_docs) / (1.0 + df)) + 1.0
            vec[g] = (c / total) * idf
        return vec


def _cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    num = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0.0 or nb == 0.0 else num / (na * nb)


@dataclass(frozen=True)
class Labeled:
    """A label plus the evidence the annotator used."""

    code: str
    confidence: float = 0.0
    detail: Mapping[str, float] = field(default_factory=dict)


class SignalRuleLabeler:
    """Annotator A: weighted rule table over the boolean signal vector."""

    name = "signal_rules"

    def __init__(self, weights: Mapping[str, Mapping[str, float]], lexicon: Mapping[str, list[str]]):
        self.weights = {code: dict(w) for code, w in weights.items()}
        self.lexicon = lexicon

    def scores(self, evidence: Evidence) -> dict[str, float]:
        out: dict[str, float] = {}
        for code in CLASS_CODES:
            row = self.weights.get(code, {})
            out[code] = sum(
                w for sig, w in row.items() if sig in SIGNAL_KEYS and evidence.has(sig)
            )
        return out

    def label_from_evidence(self, evidence: Evidence) -> Labeled:
        scores = self.scores(evidence)
        ordered = sorted(CLASS_CODES, key=lambda c: (-scores[c], CLASS_CODES.index(c)))
        top, second = ordered[0], ordered[1]
        return Labeled(
            code=top,
            confidence=round(scores[top] - scores[second], 6),
            detail=scores,
        )

    def label(self, body: str) -> Labeled:
        return self.label_from_evidence(extract(body, self.lexicon))


class SeedPrototypeLabeler:
    """Annotator B: unsupervised nearest-centroid over external seed exemplars."""

    name = "seed_prototype"

    def __init__(self, seeds: Mapping[str, Sequence[str]]):
        self.seeds = {code: list(texts) for code, texts in seeds.items()}
        docs = [s for texts in self.seeds.values() for s in texts]
        self.tfidf = _Tfidf().fit(docs)
        self.centroids = {
            code: self._centroid(texts) for code, texts in self.seeds.items()
        }

    def _centroid(self, texts: Sequence[str]) -> dict[str, float]:
        acc: dict[str, float] = {}
        for text in texts:
            for g, v in self.tfidf.weigh(text).items():
                acc[g] = acc.get(g, 0.0) + v
        norm = float(len(texts)) or 1.0
        return {g: v / norm for g, v in acc.items()}

    def label(self, body: str) -> Labeled:
        vec = self.tfidf.weigh(body)
        sims = {code: _cosine(vec, c) for code, c in self.centroids.items()}
        ordered = sorted(CLASS_CODES, key=lambda c: (-sims[c], CLASS_CODES.index(c)))
        return Labeled(code=ordered[0], confidence=round(sims[ordered[0]] - sims[ordered[1]], 6), detail=sims)


class SupervisedCentroidLabeler:
    """Learned arm: tf-idf nearest centroid fitted on a labelled split."""

    name = "supervised_centroid"

    def __init__(self) -> None:
        self.tfidf = _Tfidf()
        self.centroids: dict[str, dict[str, float]] = {}

    def fit(self, pairs: Iterable[tuple[str, str]]) -> "SupervisedCentroidLabeler":
        pairs = list(pairs)
        self.tfidf = _Tfidf().fit([body for body, _ in pairs])
        buckets: dict[str, list[dict[str, float]]] = {}
        for body, code in pairs:
            buckets.setdefault(code, []).append(self.tfidf.weigh(body))
        self.centroids = {}
        for code, vecs in buckets.items():
            acc: dict[str, float] = {}
            for vec in vecs:
                for g, v in vec.items():
                    acc[g] = acc.get(g, 0.0) + v
            self.centroids[code] = {g: v / len(vecs) for g, v in acc.items()}
        return self

    def label(self, body: str) -> Labeled:
        vec = self.tfidf.weigh(body)
        known = [c for c in CLASS_CODES if c in self.centroids]
        sims = {c: _cosine(vec, self.centroids[c]) for c in known}
        if not sims:
            return Labeled(code=CLASS_CODES[0], confidence=0.0)
        ordered = sorted(sims, key=lambda c: (-sims[c], CLASS_CODES.index(c)))
        return Labeled(code=ordered[0], confidence=round(sims[ordered[0]] - sims[ordered[1]], 6), detail=sims)


__all__ = [
    "Labeled",
    "SeedPrototypeLabeler",
    "SignalRuleLabeler",
    "SupervisedCentroidLabeler",
    "ngrams",
]