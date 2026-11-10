"""Test bootstrap: always import TensorPhysics from this working tree.

Several worktrees of this repository can be installed editable in the same
environment, in which case ``import tensorphysics`` would resolve to whichever
copy was installed last.  Putting this tree's ``src`` first on ``sys.path``
guarantees the tests exercise the code under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# BLAS thread-pool guard (performance only, no numerical effect)
# ---------------------------------------------------------------------------
# The solid-FEM kernel solves thousands of tiny (3x3) systems per assembly; on a
# many-core host the BLAS/OpenMP thread-pool overhead dominates those calls by
# more than an order of magnitude.  Pinning the pool to one thread keeps the
# finite-element test cases fast.  ``setdefault`` so an operator can override it.
import os as _os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import pytest as _pytest


@_pytest.fixture(scope="session")
def failure_handoff_run():
    """One notched three-point-bend handoff run shared by the whole session.

    The mesh is the same one the published report uses, so the numbers asserted in
    the tests are the numbers in ``docs/reports/p154w2_n03/REPORT.md``.
    """
    from tensorphysics.failure_handoff import run_handoff

    return run_handoff(dims=(32, 12, 4), target_ratio=1.4)


@_pytest.fixture(scope="session")
def beam_theory_validation():
    """Plain-beam FEM vs Euler-Bernoulli / Timoshenko (the field-certification gate)."""
    from tensorphysics.failure_handoff import BeamSpec, verify_against_beam_theory

    spec = BeamSpec(length_m=0.60, height_m=0.03, thickness_m=0.02, E_pa=30.0e9, nu=0.20)
    return verify_against_beam_theory(spec, dims=(96, 12, 3), applied_displacement_m=2.0e-5)
