"""Cross-solver field-exchange conservation audit (FEM -> DEM -> LBM).

Campaign node ``p270w1-N10``.  Sub-modules:

``fem_beam``      Euler-Bernoulli beam FE, bending stress field, failure faces
``cohesive_chain``1D bonded-particle (DEM) chain with cohesive hinges
``lbm_wall``      D2Q9 channel with moving wall and exact momentum bookkeeping
``multirate``     multirate (100 Hz / 1 kHz / 10 kHz) exchange error audit
``criteria``      frozen, hashed acceptance criteria (PASS / FAIL / WITHHELD)
``audit``         the sequential FEM -> DEM -> LBM chain runner
"""

from __future__ import annotations

__all__ = [
    "fem_beam",
    "cohesive_chain",
    "lbm_wall",
    "multirate",
    "criteria",
    "audit",
    "run_audit",
]


def __getattr__(name):
    # lazy so that ``import tensorphysics.cross_solver_conservation`` stays cheap
    if name == "run_audit":
        from .audit import run_audit

        return run_audit
    import importlib

    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)
