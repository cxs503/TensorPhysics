"""
Reference Anchor Ledger - loads and manages the anchor database.

The ledger is stored as a JSON artifact (not hardcoded in source code).
This follows the domain-data-injection principle: domain data comes from
artifacts, not from code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .schema import RefAnchor


# Default path to the JSON artifact
_DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "ref_anchors.json"


class RefAnchorLedger:
    """
    Reference anchor uncertainty ledger.

    Loads anchors from JSON artifact and provides query/filter methods.
    """

    def __init__(self, anchors: List[RefAnchor], schema_version: str = "1.0.0"):
        self._anchors = anchors
        self._schema_version = schema_version
        self._index: Dict[str, RefAnchor] = {a.anchor_id: a for a in anchors}

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def anchors(self) -> List[RefAnchor]:
        """All anchors in the ledger."""
        return list(self._anchors)

    def __len__(self) -> int:
        return len(self._anchors)

    def get(self, anchor_id: str) -> Optional[RefAnchor]:
        """Get anchor by ID."""
        return self._index.get(anchor_id)

    def get_or_raise(self, anchor_id: str) -> RefAnchor:
        """Get anchor by ID, raise if not found."""
        anchor = self.get(anchor_id)
        if anchor is None:
            raise KeyError(f"Anchor '{anchor_id}' not found in ledger")
        return anchor

    def filter_by_domain(self, domain: str) -> List[RefAnchor]:
        """Filter anchors by domain."""
        return [a for a in self._anchors if a.domain == domain]

    def filter_by_category(self, category: str) -> List[RefAnchor]:
        """Filter anchors by category."""
        return [a for a in self._anchors if a.category == category]

    def filter_by_basis(self, basis: str) -> List[RefAnchor]:
        """Filter anchors by U_ref_basis type."""
        return [a for a in self._anchors if a.U_ref_basis == basis]

    def domains(self) -> List[str]:
        """List all unique domains."""
        return sorted(set(a.domain for a in self._anchors))

    def categories(self) -> List[str]:
        """List all unique categories."""
        return sorted(set(a.category for a in self._anchors))

    def coverage_summary(self) -> Dict[str, int]:
        """Summary of anchor counts by domain."""
        summary: Dict[str, int] = {}
        for a in self._anchors:
            summary[a.domain] = summary.get(a.domain, 0) + 1
        return summary

    def anchors_without_uncertainty(self) -> List[RefAnchor]:
        """Anchors with U_ref = 0 (no uncertainty defined)."""
        return [a for a in self._anchors if a.U_ref == 0]

    def to_dict(self) -> dict:
        """Export ledger to dictionary."""
        return {
            "schema_version": self._schema_version,
            "anchors": [a.to_dict() for a in self._anchors],
        }

    def to_json(self, path: Optional[Path] = None) -> str:
        """Export ledger to JSON string (and optionally write to file)."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if path is not None:
            path.write_text(json_str, encoding="utf-8")
        return json_str


def load_ledger(path: Optional[Path] = None) -> RefAnchorLedger:
    """
    Load the reference anchor ledger from JSON.

    Parameters
    ----------
    path : Path, optional
        Path to JSON file. If None, uses the default bundled artifact.

    Returns
    -------
    RefAnchorLedger
        Loaded ledger instance.
    """
    p = path or _DEFAULT_DATA_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    schema_version = data.get("schema_version", "1.0.0")
    anchors = [RefAnchor.from_dict(d) for d in data["anchors"]]
    return RefAnchorLedger(anchors, schema_version=schema_version)


# Module-level singleton (lazy-loaded)
_LEDGER_CACHE: Optional[RefAnchorLedger] = None


def get_ledger() -> RefAnchorLedger:
    """Get the default ledger (cached singleton)."""
    global _LEDGER_CACHE
    if _LEDGER_CACHE is None:
        _LEDGER_CACHE = load_ledger()
    return _LEDGER_CACHE
