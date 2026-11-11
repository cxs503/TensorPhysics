"""LBM -> FEM partitioned fluid-structure coupling with conservative exchange.

Campaign node ``p362w2-n09`` (TensorPhysics).

Sub-modules
-----------
``lattice``    D2Q9 fluid with a moving immersed circular body, Ladd momentum
               exchange and a full discrete momentum/mass ledger
``structure``  plane-stress Q4 finite-element elastic tether (implicit Newmark),
               carrying the cylinder as a tip mass
``interface``  conservative traction -> nodal-load mapping (resultant *and*
               moment preserved by a minimum-norm partition of unity) plus the
               exchange audits
``coupling``   the partitioned two-way loop with LBM-explicit / FEM-implicit
               sub-cycling, the mass / momentum / energy ledgers and the
               Strouhal / amplitude measurement
``acceptance`` criteria (a)-(f) of the campaign node, mesh-refinement study and
               the machine-readable payload
"""

from __future__ import annotations

import importlib

__all__ = ["lattice", "structure", "interface", "coupling", "acceptance", "run_acceptance"]


def __getattr__(name):
    if name == "run_acceptance":
        from .acceptance import run_acceptance

        return run_acceptance
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)