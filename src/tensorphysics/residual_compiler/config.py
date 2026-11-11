"""Configuration for the residual compiler."""
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
import os


@dataclass
class CompilerConfig:
    """Configuration for the residual compiler.
    
    Compatible with proposal_seeding.json format.
    """
    # Scanner settings
    scan_coverage_threshold: float = 0.95  # ≥95% of done nodes must be scanned
    limitation_keywords: List[str] = field(default_factory=lambda: [
        "局限", "limitation", "未做", "untested", "honest", "FAIL",
        "TODO", "后续", "gap", "missing", "deficit", "WITHHELD",
        "NOT_SUPPORTED", "MOCK", "STUB", "待改进", "需修复"
    ])
    
    # Generator settings
    generation_rate_threshold: float = 0.80  # ≥80% of gaps must generate proposals
    
    # Dedup settings
    dedup_recall_threshold: float = 0.80  # ≥80% recall
    dedup_precision_threshold: float = 0.85  # ≥85% precision
    similarity_threshold: float = 0.7  # cosine similarity threshold for dedup
    
    # Priority scoring weights
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "evidence_grade": 0.3,
        "domain_criticality": 0.3,
        "gap_severity": 0.2,
        "dependency_count": 0.2,
    })
    
    # Domain criticality mapping
    domain_criticality: Dict[str, float] = field(default_factory=lambda: {
        "resistance": 0.9,
        "propulsion": 0.85,
        "seakeeping": 0.8,
        "maneuvering": 0.8,
        "stability": 0.95,
        "structure": 0.85,
        "vibration": 0.7,
        "green_shipping": 0.75,
        "ice_navigation": 0.9,
    })
    
    # Output settings
    output_format: str = "json"
    max_proposals: int = 100
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "scan_coverage_threshold": self.scan_coverage_threshold,
            "limitation_keywords": self.limitation_keywords,
            "generation_rate_threshold": self.generation_rate_threshold,
            "dedup_recall_threshold": self.dedup_recall_threshold,
            "dedup_precision_threshold": self.dedup_precision_threshold,
            "similarity_threshold": self.similarity_threshold,
            "priority_weights": self.priority_weights,
            "domain_criticality": self.domain_criticality,
            "output_format": self.output_format,
            "max_proposals": self.max_proposals,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CompilerConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_file(cls, path: str) -> "CompilerConfig":
        """Load from JSON file (proposal_seeding.json compatible)."""
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
