"""General non-matching grid-to-grid conservative remapping (GAP-CROSS-MAP).

Conservative projection of cell-averaged and P1-nodal fields between two
arbitrary unstructured triangular meshes in 2D.  The overlap table is built
by exact triangle-triangle clipping (Sutherland-Hodgman), and the projection
matrix is constructed so that global conservation is an algebraic identity.

Campaign node: GAP-CROSS-MAP (conservation error < 1% hard gate).

Sub-modules
-----------
``overlap``     triangle-triangle intersection, spatial indexing (quad-tree
                bounding-box filter), and the sparse overlap table;
``remap``       conservative projection operators for cell-averaged scalars,
                P1 nodal scalars, and vector (momentum) fields, with audits
                (conservation defect, monotonicity, L2 error);
``acceptance``  end-to-end cases: conservation gate, symmetry (A->B->A
                roundtrip), and convergence-order verification.
"""
from __future__ import annotations

import importlib

__all__ = ["overlap", "remap", "acceptance"]


def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)
