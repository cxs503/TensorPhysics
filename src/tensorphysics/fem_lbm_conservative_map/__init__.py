"""FEM (unstructured P1 triangles) -> LBM (uniform lattice) conservative scalar remap.

Campaign node ``p399w2-n06``.  Sub-modules

``mesh``          unstructured perturbed triangulation + P1 heat conduction solver
                  (``-div(k grad T) = f``, Dirichlet / natural BCs) + manufactured
                  analytic solutions, plus the mesh geometry gate (Cb/L/B analogue:
                  area closure, positive orientation, aspect ratio);
``lattice``       uniform Cartesian target lattice and the exact FEM/lattice
                  overlap table (Sutherland-Hodgman clipping) that carries the
                  integral operator ``M``;
``remap``         the three-stage conservative map: source cell integrals ->
                  overlap-volume weighting -> limited reconstruction with
                  bounded conservative redistribution, and its audits
                  (conservation defect, no-new-extrema margins, L2 vs analytic);
``thermal_lbm``   D2Q9 + D2Q5 thermal lattice-Boltzmann with Boussinesq buoyancy,
                  exact mass/energy bookkeeping and adiabatic/Dirichlet thermal
                  wall handling -- the consumer of the mapped field;
``acceptance``    end-to-end cases, criteria and the JSON payload of this node.

Direction of the exchange is FEM -> LBM (a scalar field: temperature), the
complement of :mod:`tensorphysics.field_exchange` which transfers *tractions*
LBM/DEM -> FEM.
"""
from __future__ import annotations

import importlib

__all__ = ["mesh", "lattice", "remap", "thermal_lbm", "acceptance", "run_acceptance"]


def __getattr__(name):
    if name == "run_acceptance":
        from .acceptance import run_acceptance

        return run_acceptance
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)