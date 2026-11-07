"""Rule-based contribution classifier with LLM fallback.

Classification strategy:
1. Evidence field matching: check which required evidence fields are present
2. Keyword/signal scoring: count positive signals in title/summary/tags
3. Counter-example rejection: reject if strong counter-example signals present
4. LLM fallback: only if rule-based confidence < threshold; must output evidence fields

The classifier operates on structured items with fields:
  - mission_id: str
  - title: str
  - summary: str
  - tags: List[str]
  - evidence: Dict[str, Any]  (optional structured evidence)
  - raw_text: str  (optional full text for keyword scanning)
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .ontology import CONTRIBUTION_ONTOLOGY, ContributionCategory


@dataclass
class ClassificationResult:
    """Result of classifying a contribution item."""
    mission_id: str
    predicted_category: Optional[ContributionCategory]
    confidence: float                    # 0.0 - 1.0
    method: str                          # "rule_based" | "llm_fallback" | "uncertain"
    evidence_fields_present: List[str]   # which evidence fields were found
    evidence_fields_missing: List[str]   # which were required but not found
    positive_signals: List[str]          # matched positive signals
    counter_signals: List[str]           # matched counter-example signals
    reasoning: str                       # human-readable explanation


@dataclass
class ContributionItem:
    """Input item for classification."""
    mission_id: str
    title: str = ""
    summary: str = ""
    tags: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace for keyword matching."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def _count_signals(text: str, signals: List[str]) -> List[str]:
    """Return list of signals found in text (case-insensitive word/phrase match)."""
    found = []
    text_norm = _normalize_text(text)
    for sig in signals:
        sig_norm = _normalize_text(sig)
        # Match as word boundary or substring for multi-word phrases
        if sig_norm in text_norm:
            found.append(sig)
    return found


def _check_evidence_fields(item: ContributionItem, category: ContributionCategory) -> Tuple[List[str], List[str]]:
    """Check which evidence fields are present/missing for a category."""
    defn = CONTRIBUTION_ONTOLOGY[category]
    present = []
    missing = []
    for ef in defn.min_evidence:
        # Check in evidence dict first
        if ef in item.evidence and item.evidence[ef] is not None:
            present.append(ef)
        # Also check in metadata
        elif ef in item.metadata and item.metadata[ef] is not None:
            present.append(ef)
        # Check in tags (some evidence might be encoded as tags)
        elif any(ef.lower() in t.lower() for t in item.tags):
            present.append(ef)
        else:
            missing.append(ef)
    return present, missing


def _rule_based_classify(item: ContributionItem) -> ClassificationResult:
    """Rule-based classification using evidence fields and keyword signals."""
    all_text = " ".join([item.title, item.summary, " ".join(item.tags), item.raw_text])

    best_cat = None
    best_score = -1.0
    best_present = []
    best_missing = []
    best_signals = []
    best_counter = []

    for cat, defn in CONTRIBUTION_ONTOLOGY.items():
        # Evidence field score (0-1): fraction of required fields present
        present, missing = _check_evidence_fields(item, cat)
        evidence_score = len(present) / max(len(defn.min_evidence), 1)

        # Signal score (0-1): fraction of positive signals found
        signals = _count_signals(all_text, defn.positive_signals)
        signal_score = len(signals) / max(len(defn.positive_signals), 1)

        # Counter-example penalty
        counters = _count_signals(all_text, defn.counter_examples)
        counter_penalty = len(counters) * 0.3

        # Combined score: evidence is primary (60%), signals secondary (40%)
        score = 0.6 * evidence_score + 0.4 * signal_score - counter_penalty

        if score > best_score:
            best_score = score
            best_cat = cat
            best_present = present
            best_missing = missing
            best_signals = signals
            best_counter = counters

    # Determine confidence and method
    confidence = max(0.0, min(1.0, best_score))
    method = "rule_based"

    if confidence < 0.3:
        method = "uncertain"
    elif confidence < 0.5:
        method = "llm_fallback"

    # Build reasoning
    if best_cat:
        defn = CONTRIBUTION_ONTOLOGY[best_cat]
        reasoning = (
            f"Category {best_cat.value}: evidence {len(best_present)}/{len(defn.min_evidence)} "
            f"fields present ({', '.join(best_present) or 'none'}), "
            f"{len(best_signals)} positive signals matched, "
            f"{len(best_counter)} counter signals found."
        )
    else:
        reasoning = "No category scored above threshold."

    return ClassificationResult(
        mission_id=item.mission_id,
        predicted_category=best_cat,
        confidence=confidence,
        method=method,
        evidence_fields_present=best_present,
        evidence_fields_missing=best_missing,
        positive_signals=best_signals,
        counter_signals=best_counter,
        reasoning=reasoning,
    )


def _llm_fallback_classify(item: ContributionItem) -> ClassificationResult:
    """LLM fallback classification (simulated — in production, call actual LLM).

    This simulates what an LM would do: analyze the text and output a category
    with required evidence fields. In production, this would be an API call.

    For now, we use a more aggressive keyword matching as a proxy.
    """
    all_text = _normalize_text(" ".join([item.title, item.summary, " ".join(item.tags), item.raw_text]))

    # Simulated LLM: use expanded keyword matching with category-specific heuristics
    scores = {}
    for cat, defn in CONTRIBUTION_ONTOLOGY.items():
        # Count all signals (positive and negative)
        pos_count = len(_count_signals(all_text, defn.positive_signals))
        neg_count = len(_count_signals(all_text, defn.counter_examples))

        # Check for category-specific patterns
        cat_score = pos_count * 2 - neg_count * 3

        # Bonus for evidence fields
        present, _ = _check_evidence_fields(item, cat)
        cat_score += len(present) * 3

        scores[cat] = cat_score

    best_cat = max(scores, key=scores.get) if scores else None
    best_score_val = scores.get(best_cat, 0) if best_cat else 0

    # Normalize to 0-1 range
    max_possible = max(len(CONTRIBUTION_ONTOLOGY[cat].positive_signals) * 2 +
                       len(CONTRIBUTION_ONTOLOGY[cat].min_evidence) * 3
                       for cat in CONTRIBUTION_ONTOLOGY)
    confidence = max(0.0, min(1.0, best_score_val / max(max_possible, 1)))

    present, missing = _check_evidence_fields(item, best_cat) if best_cat else ([], [])
    signals = _count_signals(all_text, CONTRIBUTION_ONTOLOGY[best_cat].positive_signals) if best_cat else []
    counters = _count_signals(all_text, CONTRIBUTION_ONTOLOGY[best_cat].counter_examples) if best_cat else []

    reasoning = (
        f"LLM fallback for {best_cat.value if best_cat else 'unknown'}: "
        f"score={best_score_val}, evidence fields: {len(present)} present, "
        f"{len(missing)} missing."
    )

    return ClassificationResult(
        mission_id=item.mission_id,
        predicted_category=best_cat,
        confidence=confidence,
        method="llm_fallback",
        evidence_fields_present=present,
        evidence_fields_missing=missing,
        positive_signals=signals,
        counter_signals=counters,
        reasoning=reasoning,
    )


def classify_item(item: ContributionItem) -> ClassificationResult:
    """Classify a contribution item using rule-based + LLM fallback strategy.

    Strategy:
    1. Try rule-based classification
    2. If confidence >= 0.5, accept rule-based result
    3. If confidence < 0.5, try LLM fallback
    4. If LLM also low confidence, mark as uncertain
    """
    rule_result = _rule_based_classify(item)

    if rule_result.confidence >= 0.5:
        return rule_result

    # LLM fallback
    llm_result = _llm_fallback_classify(item)

    # Use LLM result if it has higher confidence
    if llm_result.confidence > rule_result.confidence:
        return llm_result

    # Otherwise return rule result but mark as uncertain
    rule_result.method = "uncertain"
    return rule_result


class ContributionClassifier:
    """Stateless classifier for contribution items."""

    def classify(self, item: ContributionItem) -> ClassificationResult:
        return classify_item(item)

    def classify_batch(self, items: List[ContributionItem]) -> List[ClassificationResult]:
        return [classify_item(item) for item in items]
