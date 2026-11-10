"""Conservative mortar interpolation for multiphysics interface coupling.

Campaign node **P162W1-N06**.  Provides the *dual-basis (biorthogonal) mortar*
transfer between two non-matching interface discretisations together with the
mass / momentum / energy conservation residuals and the machine gate that
accepts or rejects a coupled run.

Public API
----------
Trace spaces (partition of unity by construction):
    :class:`~tensorphysics.multiphysics_mortar.spaces.LineTraceSpace`
        1D interface curve with an arc-length metric.
    :class:`~tensorphysics.multiphysics_mortar.spaces.TriTraceSpace`
        2D interface patch in a chart with the surface Jacobian.
Operators:
    :class:`~tensorphysics.multiphysics_mortar.assembly.MortarTransfer`
        conservative L2-projection mortar; ``apply`` / ``apply_dual`` are the two
        (algebraically identical) routes through the primal and dual bases.
    :class:`~tensorphysics.multiphysics_mortar.assembly.NearestNeighborTransfer`
        first-order donor-cell baseline used by criterion C6.
Diagnostics:
    :func:`~tensorphysics.multiphysics_mortar.conservation.conservation_residuals`
        mass / momentum / energy residuals of one exchange.
    :func:`...conservation.local_balance_defect`
        per-receiving-DOF balance defect (the residual map).
Gate:
    :class:`~tensorphysics.multiphysics_mortar.gate.ConservationGate`
        fail-closed threshold gate over the declared residuals.
Acceptance:
    :func:`~tensorphysics.multiphysics_mortar.acceptance.run_acceptance`
        C1..C7 driver returning the JSON payload.

Relationship to :mod:`tensorphysics.interface_operators`
--------------------------------------------------------
``interface_operators`` (IN19-T2 / IN50-N08) implements the *finite-volume
overlap* mortar: target cell means obtained by integrating the donor field over
the overlap areas.  This package implements the *finite-element trace* mortar:
an L2 projection onto the receiving P1 space, whose dual basis diagonalises the
coupling.  The two families are complementary -- FV overlap is diffusive but
local, FE trace is second-order and interpolation-exact on matching meshes -- and
they share one geometry kernel (polygon clipping / quadrature from
``interface_operators.polygons``), so there is no duplicated source of truth for
the interface geometry.
"""
from .assembly import (CouplingAssembly, MortarTransfer,
                       NearestNeighborTransfer, build_coupling)
from .conservation import (MaterialState, QuantityResidual, conservation_residuals,
                           integrate, local_balance_defect)
from .gate import (GATE_TOLERANCES, ConservationGate, ConservationGateError,
                   GateItem, GateReport)
from .spaces import (LineTraceSpace, TriTraceSpace, chart_metric, cosine_nodes,
                     structured_triangles)

__all__ = [
    "CouplingAssembly", "MortarTransfer", "NearestNeighborTransfer",
    "build_coupling",
    "MaterialState", "QuantityResidual", "conservation_residuals", "integrate",
    "local_balance_defect",
    "ConservationGate", "ConservationGateError", "GateItem", "GateReport",
    "GATE_TOLERANCES",
    "LineTraceSpace", "TriTraceSpace", "chart_metric", "cosine_nodes",
    "structured_triangles",
]