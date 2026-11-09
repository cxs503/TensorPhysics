"""Water-entry (slamming) self-similarity kernels.

Modules
-------
similarity_impact
    Self-similar solution of the two-dimensional wedge/slat water-entry problem
    (constant entry velocity and constant entry acceleration) with the
    spray-root pressure singularity, the Wagner/von Karman momentum bounds and
    an explicit resolution ledger for the *non-convergent* point pressure peak.
"""

from __future__ import annotations

from .similarity_impact import (
    C_SOUND_WATER,
    RHO_WATER,
    BulkheadSDOF,
    SelfSimilarWedgeImpact,
    resolution_ledger,
)

__all__ = [
    "C_SOUND_WATER",
    "RHO_WATER",
    "BulkheadSDOF",
    "SelfSimilarWedgeImpact",
    "resolution_ledger",
]