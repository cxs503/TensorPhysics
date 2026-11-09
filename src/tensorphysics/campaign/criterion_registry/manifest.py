"""Criterion manifest: extract structured criteria from mission goals.

Extracts criteria blocks from goal text → canonical JSON with six fields:
- threshold: numeric acceptance bound
- unit: measurement unit (string)
- denominator: reference basis for relative metrics
- exclusion: conditions/cases excluded from scope
- reference_path: path to reference data or baseline
- comparator: comparison operator (le, ge, eq, in_range)

Fail-closed: unparseable → status='unregistered', never 'pass'.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


# The six canonical fields (structural only, zero domain semantics)
CRITERION_FIELDS = frozenset({
    "threshold",
    "unit",
    "denominator",
    "exclusion",
    "reference_path",
    "comparator",
})


@dataclass(frozen=True)
class CriterionField:
    """A single field in a criterion manifest."""
    name: str
    value: Any
    present: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "present": self.present}


@dataclass
class CriterionManifest:
    """Structured representation of one acceptance criterion.

    Attributes:
        criterion_id: unique identifier for this criterion
        fields: dict mapping field name → CriterionField
        status: 'registered' | 'unregistered' | 'tampered'
        frozen_at: ISO timestamp of freeze moment
        frozen_by: identifier of who/what froze this
        sha256: hash of canonical JSON (for tamper detection)
    """
    criterion_id: str
    fields: dict[str, CriterionField] = field(default_factory=dict)
    status: str = "unregistered"
    frozen_at: Optional[str] = None
    frozen_by: Optional[str] = None
    sha256: Optional[str] = None

    def to_canonical_json(self) -> str:
        """Produce canonical JSON for hashing (sorted keys, no whitespace)."""
        canonical = {}
        for fname in sorted(CRITERION_FIELDS):
            if fname in self.fields and self.fields[fname].present:
                canonical[fname] = self.fields[fname].value
            else:
                canonical[fname] = None
        return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        """Full dict representation including metadata."""
        return {
            "criterion_id": self.criterion_id,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "status": self.status,
            "frozen_at": self.frozen_at,
            "frozen_by": self.frozen_by,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriterionManifest":
        """Reconstruct from dict."""
        fields = {}
        for fname, fdata in data.get("fields", {}).items():
            fields[fname] = CriterionField(
                name=fdata["name"],
                value=fdata["value"],
                present=fdata.get("present", True),
            )
        return cls(
            criterion_id=data["criterion_id"],
            fields=fields,
            status=data.get("status", "unregistered"),
            frozen_at=data.get("frozen_at"),
            frozen_by=data.get("frozen_by"),
            sha256=data.get("sha256"),
        )


def _clean_value(value: str) -> str:
    """Clean extracted value: strip whitespace and trailing punctuation."""
    if not value:
        return value
    # Strip trailing punctuation that might be from sentence structure
    value = value.strip()
    value = value.rstrip(".,;:)]}")
    value = value.lstrip("([({")
    return value.strip()


def _extract_numeric(text: str) -> Optional[float]:
    """Extract first numeric value from text (int or float)."""
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def _extract_comparator(text: str) -> Optional[str]:
    """Extract comparison operator from text."""
    text_lower = text.lower()
    # Order matters: check longer patterns first
    patterns = [
        (r"<=", "le"),
        (r">=", "ge"),
        (r"!=|not equal|≠", "ne"),
        (r"<(?!<|=)", "lt"),
        (r">(?!>|=)", "gt"),
        (r"==|equals?|equal to", "eq"),
        (r"within|in range|±", "in_range"),
    ]
    for pattern, op in patterns:
        if re.search(pattern, text_lower):
            return op
    return None


def _extract_unit(text: str) -> Optional[str]:
    """Extract unit string from text (heuristic: word after number)."""
    # Look for patterns like "3%", "5 mm", "0.1 s"
    match = re.search(r"[-+]?\d*\.?\d+\s*([a-zA-Z%°]+)", text)
    if match:
        unit = match.group(1).strip()
        if unit and len(unit) <= 10:
            return unit
    return None


def extract_criteria_from_goal(goal_text: str) -> list[CriterionManifest]:
    """Extract structured criteria from a mission goal text.

    Parses the goal text looking for criterion blocks. Each block should
    contain threshold values, units, comparators, etc.

    Args:
        goal_text: The mission goal text (may contain multiple criteria)

    Returns:
        List of CriterionManifest objects. Empty list if no criteria found.
        Each criterion has status='unregistered' until freeze_manifest() is called.
    """
    if not goal_text or not isinstance(goal_text, str):
        return []

    criteria = []

    # Split into criterion blocks by looking for numbered items or bullet points
    # Patterns: "1.", "- ", "* ", "C1:", "criterion:", etc.
    block_pattern = r"(?:^|\n)\s*(?:\d+[\.\)]|[-*•]|[Cc]\d+[:\.])\s*"
    blocks = re.split(block_pattern, goal_text)

    # If no blocks found, treat entire text as one criterion
    if len(blocks) <= 1:
        blocks = [goal_text]

    for idx, block in enumerate(blocks):
        block = block.strip()
        if not block or len(block) < 5:
            continue

        # Try to extract the six fields
        fields = {}

        # Threshold
        threshold = _extract_numeric(block)
        if threshold is not None:
            fields["threshold"] = CriterionField("threshold", threshold, True)

        # Unit
        unit = _extract_unit(block)
        if unit:
            fields["unit"] = CriterionField("unit", unit, True)

        # Comparator
        comparator = _extract_comparator(block)
        if comparator:
            fields["comparator"] = CriterionField("comparator", comparator, True)

        # Denominator (look for "of X", "relative to X", "/X")
        denom_match = re.search(r"(?:of|relative to|vs\.?|/)\s+([A-Za-z_][A-Za-z0-9_]*)", block)
        if denom_match:
            fields["denominator"] = CriterionField("denominator", _clean_value(denom_match.group(1)), True)

        # Exclusion (look for "excluding", "except", "not including")
        excl_match = re.search(r"(?:exclud(?:e|ing)|except|not including|minus)\s+([^,;.(]+)", block, re.I)
        if excl_match:
            fields["exclusion"] = CriterionField("exclusion", _clean_value(excl_match.group(1)), True)

        # Reference path (look for paths, URLs, or "ref:", "baseline:")
        ref_match = re.search(r"(?:ref(?:erence)?|baseline|source)[:\s]+([^\s,;)]+)", block, re.I)
        if ref_match:
            fields["reference_path"] = CriterionField("reference_path", _clean_value(ref_match.group(1)), True)

        # Only create criterion if we found at least one field
        if fields:
            criterion = CriterionManifest(
                criterion_id=f"criterion_{idx:03d}",
                fields=fields,
                status="unregistered",
            )
            criteria.append(criterion)

    return criteria


def freeze_manifest(
    manifest: CriterionManifest,
    frozen_by: str,
    frozen_at: Optional[datetime] = None,
) -> CriterionManifest:
    """Freeze a criterion manifest with sha256 hash.

    Args:
        manifest: The criterion to freeze
        frozen_by: Identifier of who/what is freezing (e.g., agent id, timestamp)
        frozen_at: Optional timestamp (defaults to now UTC)

    Returns:
        New CriterionManifest with status='registered', sha256 set, frozen_at/by set.
    """
    if frozen_at is None:
        frozen_at = datetime.now(timezone.utc)

    frozen_at_str = frozen_at.isoformat()

    # Compute sha256 of canonical JSON
    canonical = manifest.to_canonical_json()
    sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # Create frozen copy
    frozen = CriterionManifest(
        criterion_id=manifest.criterion_id,
        fields=dict(manifest.fields),
        status="registered",
        frozen_at=frozen_at_str,
        frozen_by=frozen_by,
        sha256=sha256,
    )
    return frozen


def verify_frozen_hash(manifest: CriterionManifest) -> bool:
    """Verify that a frozen manifest's hash matches its content.

    Args:
        manifest: A frozen CriterionManifest

    Returns:
        True if hash matches, False if tampered or not frozen.
    """
    if manifest.status != "registered" or manifest.sha256 is None:
        return False

    canonical = manifest.to_canonical_json()
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return computed == manifest.sha256
