"""Drift audit: compare frozen criteria against post-execution report criteria.

Classifies changes into:
- unchanged: no difference
- tightened: threshold became stricter (e.g., 10% → 3%)
- loosened: threshold became looser (e.g., 3% → 10%)
- scope_changed: denominator/exclusion/reference changed
- undefined: cannot determine (e.g., field missing in one version)

Fail-closed: unparseable → 'unregistered', never 'pass'.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .manifest import (
    CriterionManifest,
    CriterionField,
    CRITERION_FIELDS,
    extract_criteria_from_goal,
    verify_frozen_hash,
)


@dataclass(frozen=True)
class DriftResult:
    """Result of drift classification for one criterion."""
    criterion_id: str
    classification: str  # unchanged, tightened, loosened, scope_changed, undefined, unregistered, tampered
    field_diffs: dict[str, str] = field(default_factory=dict)  # field_name → change_type
    reason: Optional[str] = None  # human-readable explanation

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "classification": self.classification,
            "field_diffs": self.field_diffs,
            "reason": self.reason,
        }


class DriftClassifier:
    """Classifies drift between frozen and current criterion manifests."""

    # Comparators that indicate "less is better" (tighter = smaller threshold)
    LESS_IS_BETTER = frozenset({"le", "lt", "eq"})
    # Comparators that indicate "more is better" (tighter = larger threshold)
    MORE_IS_BETTER = frozenset({"ge", "gt"})

    def classify_field(
        self,
        field_name: str,
        frozen_value: Any,
        current_value: Any,
        comparator: Optional[str] = None,
    ) -> str:
        """Classify drift for a single field.

        Returns:
            'unchanged', 'tightened', 'loosened', 'scope_changed', or 'undefined'
        """
        # Both None or equal
        if frozen_value == current_value:
            return "unchanged"

        # One is None
        if frozen_value is None or current_value is None:
            return "undefined"

        # Threshold drift depends on comparator direction
        if field_name == "threshold":
            if not isinstance(frozen_value, (int, float)) or not isinstance(current_value, (int, float)):
                return "undefined"

            if comparator in self.LESS_IS_BETTER:
                # Tighter = smaller value (e.g., 10% → 3%)
                if current_value < frozen_value:
                    return "tightened"
                elif current_value > frozen_value:
                    return "loosened"
            elif comparator in self.MORE_IS_BETTER:
                # Tighter = larger value (e.g., 90% → 95%)
                if current_value > frozen_value:
                    return "tightened"
                elif current_value < frozen_value:
                    return "loosened"
            else:
                # Unknown comparator direction
                return "undefined"

        # Scope fields: denominator, exclusion, reference_path
        if field_name in ("denominator", "exclusion", "reference_path"):
            return "scope_changed"

        # Unit change
        if field_name == "unit":
            return "scope_changed"

        # Comparator change
        if field_name == "comparator":
            return "scope_changed"

        return "undefined"

    def classify_criterion(
        self,
        frozen: CriterionManifest,
        current: CriterionManifest,
    ) -> DriftResult:
        """Classify drift between frozen and current versions of one criterion.

        Args:
            frozen: The frozen (pre-execution) criterion
            current: The current (post-execution) criterion

        Returns:
            DriftResult with overall classification and per-field diffs.
        """
        # Check if frozen is valid
        if frozen.status != "registered":
            return DriftResult(
                criterion_id=frozen.criterion_id,
                classification="unregistered",
                reason="frozen criterion not registered",
            )

        # Check for tampering
        if not verify_frozen_hash(frozen):
            return DriftResult(
                criterion_id=frozen.criterion_id,
                classification="tampered",
                reason="frozen hash does not match content",
            )

        # Check if current is parseable
        if current.status == "unregistered" and not current.fields:
            return DriftResult(
                criterion_id=frozen.criterion_id,
                classification="unregistered",
                reason="current criterion unparseable",
            )

        # Get comparator for threshold direction
        comparator = None
        if "comparator" in frozen.fields and frozen.fields["comparator"].present:
            comparator = frozen.fields["comparator"].value

        # Compare each field
        field_diffs = {}
        has_tightened = False
        has_loosened = False
        has_scope_changed = False
        has_undefined = False

        for fname in CRITERION_FIELDS:
            frozen_val = frozen.fields[fname].value if fname in frozen.fields and frozen.fields[fname].present else None
            current_val = current.fields[fname].value if fname in current.fields and current.fields[fname].present else None

            diff = self.classify_field(fname, frozen_val, current_val, comparator)
            if diff != "unchanged":
                field_diffs[fname] = diff

            if diff == "tightened":
                has_tightened = True
            elif diff == "loosened":
                has_loosened = True
            elif diff == "scope_changed":
                has_scope_changed = True
            elif diff == "undefined":
                has_undefined = True

        # Determine overall classification
        if not field_diffs:
            classification = "unchanged"
        elif has_scope_changed:
            classification = "scope_changed"
        elif has_loosened:
            classification = "loosened"
        elif has_tightened:
            classification = "tightened"
        elif has_undefined:
            classification = "undefined"
        else:
            classification = "unchanged"

        # Build reason
        if classification == "unchanged":
            reason = None
        else:
            changes = [f"{k}={v}" for k, v in field_diffs.items()]
            reason = f"drift detected: {', '.join(changes)}"

        return DriftResult(
            criterion_id=frozen.criterion_id,
            classification=classification,
            field_diffs=field_diffs,
            reason=reason,
        )


def classify_drift(
    frozen: CriterionManifest,
    current: CriterionManifest,
) -> DriftResult:
    """Convenience function: classify drift between two criteria."""
    classifier = DriftClassifier()
    return classifier.classify_criterion(frozen, current)


def audit_drift(
    frozen_criteria: list[CriterionManifest],
    report_text: str,
) -> list[DriftResult]:
    """Audit drift between frozen criteria and a report's extracted criteria.

    Args:
        frozen_criteria: List of frozen CriterionManifest objects
        report_text: The final report text to extract current criteria from

    Returns:
        List of DriftResult objects, one per frozen criterion.
    """
    # Extract current criteria from report
    current_criteria = extract_criteria_from_goal(report_text)

    # Build lookup by criterion_id
    current_by_id = {c.criterion_id: c for c in current_criteria}

    results = []
    classifier = DriftClassifier()

    for frozen in frozen_criteria:
        current = current_by_id.get(frozen.criterion_id)
        if current is None:
            # Criterion not found in report
            results.append(DriftResult(
                criterion_id=frozen.criterion_id,
                classification="unregistered",
                reason="criterion not found in report",
            ))
        else:
            results.append(classifier.classify_criterion(frozen, current))

    return results
