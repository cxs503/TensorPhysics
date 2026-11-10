"""
Schema definitions for reference anchors.

Each anchor follows the schema:
{
    "anchor_id": str,
    "name": str,
    "value": float,
    "unit": str,
    "source": {"spec_clause": str|None, "paper": str|None, "dataset": str|None},
    "U_ref": float,  # expanded uncertainty at k=2
    "U_ref_basis": str,  # "specification_tolerance" | "experimental_scatter" | "literature_range"
    "U_ref_basis_detail": str,
    "provenance": {"file": str, "sha256": str, "locator": str},
    "domain": str,
    "category": str
}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AnchorSource:
    """Source of the reference anchor value."""
    spec_clause: Optional[str] = None
    paper: Optional[str] = None
    dataset: Optional[str] = None

    def has_any_source(self) -> bool:
        """Check if at least one source is provided."""
        return any([self.spec_clause, self.paper, self.dataset])


@dataclass(frozen=True)
class AnchorProvenance:
    """Provenance metadata for traceability."""
    file: str
    sha256: str
    locator: str  # e.g., "Table 3, p.47" or "Eq. 4.12, p.62"


@dataclass(frozen=True)
class RefAnchor:
    """
    Reference anchor with uncertainty.

    Attributes
    ----------
    anchor_id : str
        Unique identifier (e.g., "ICE-001", "RES-001")
    name : str
        Human-readable name
    value : float
        Nominal value
    unit : str
        Physical unit (or "dimensionless")
    source : AnchorSource
        Source specification
    U_ref : float
        Expanded uncertainty at k=2 (95% coverage)
    U_ref_basis : str
        Basis for uncertainty: "specification_tolerance", "experimental_scatter", "literature_range"
    U_ref_basis_detail : str
        Detailed explanation of how U_ref was determined
    provenance : AnchorProvenance
        File/sha256/locator for traceability
    domain : str
        Physics domain (e.g., "ice_navigation", "resistance", "propulsion")
    category : str
        Sub-category (e.g., "ice_resistance", "friction", "propeller")
    """
    anchor_id: str
    name: str
    value: float
    unit: str
    source: AnchorSource
    U_ref: float
    U_ref_basis: str
    U_ref_basis_detail: str
    provenance: AnchorProvenance
    domain: str
    category: str

    def __post_init__(self):
        """Validate anchor data."""
        if self.U_ref < 0:
            raise ValueError(f"U_ref must be non-negative, got {self.U_ref}")
        if not self.source.has_any_source():
            raise ValueError(f"Anchor {self.anchor_id} must have at least one source")
        if self.U_ref_basis not in ("specification_tolerance", "experimental_scatter", "literature_range"):
            raise ValueError(
                f"U_ref_basis must be one of: specification_tolerance, "
                f"experimental_scatter, literature_range; got {self.U_ref_basis}"
            )

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty U_ref / |value| (returns inf if value=0)."""
        if self.value == 0:
            return float('inf')
        return abs(self.U_ref / self.value)

    @property
    def has_uncertainty(self) -> bool:
        """Check if this anchor has a defined uncertainty."""
        return self.U_ref > 0

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "anchor_id": self.anchor_id,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": {
                "spec_clause": self.source.spec_clause,
                "paper": self.source.paper,
                "dataset": self.source.dataset,
            },
            "U_ref": self.U_ref,
            "U_ref_basis": self.U_ref_basis,
            "U_ref_basis_detail": self.U_ref_basis_detail,
            "provenance": {
                "file": self.provenance.file,
                "sha256": self.provenance.sha256,
                "locator": self.provenance.locator,
            },
            "domain": self.domain,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RefAnchor:
        """Create from dictionary representation."""
        return cls(
            anchor_id=d["anchor_id"],
            name=d["name"],
            value=d["value"],
            unit=d["unit"],
            source=AnchorSource(
                spec_clause=d["source"].get("spec_clause"),
                paper=d["source"].get("paper"),
                dataset=d["source"].get("dataset"),
            ),
            U_ref=d["U_ref"],
            U_ref_basis=d["U_ref_basis"],
            U_ref_basis_detail=d["U_ref_basis_detail"],
            provenance=AnchorProvenance(
                file=d["provenance"]["file"],
                sha256=d["provenance"]["sha256"],
                locator=d["provenance"]["locator"],
            ),
            domain=d["domain"],
            category=d["category"],
        )
