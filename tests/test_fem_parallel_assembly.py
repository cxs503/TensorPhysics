"""Assembly and halo-exchange correctness against independent references."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.fem_parallel.comm import SerialComm
from tensorphysics.fem_parallel.discretisation import ElementKernel, StructuredGrid
from tensorphysics.fem_parallel.operator import (
    DistributedSystem,
    assemble_owned_rows,
    element_box,
    eliminate_dirichlet,
)
from tensorphysics.fem_parallel.partition import CartesianPartition
from tensorphysics.fem_parallel.problems import make_problem
from tensorphysics.fem_parallel.reference import assemble_reference
from tensorphysics.fem_parallel.runtime import run


def test_q1_poisson_element_matrix_matches_analytic_stencil():
    grid = StructuredGrid((5, 5))
    Ke = ElementKernel("poisson", 2, grid.h).element_matrix()
    # unit cell Q1 Laplacian: 4/6 on the diagonal, -1/6 on edges, -2/6 on corners
    assert Ke[0, 0] == pytest.approx(4.0 / 6.0)
    assert Ke[0, 1] == pytest.approx(-1.0 / 6.0)   # neighbouring corner
    assert Ke[0, 3] == pytest.approx(-2.0 / 6.0)   # diagonal corner
    assert np.allclose(Ke.sum(axis=1), 0.0, atol=1e-12)
    assert np.allclose(Ke, Ke.T)


def test_q1_elasticity_element_matrix_has_rigid_body_modes():
    grid = StructuredGrid((5, 5))
    Ke = ElementKernel("elasticity", 2, grid.h, {"E": 1.0, "nu": 0.3}).element_matrix()
    assert np.allclose(Ke, Ke.T)
    eig = np.linalg.eigvalsh(Ke)
    assert np.sum(np.abs(eig) < 1e-10) == 3   # two translations + one rotation


def test_q1_3d_stiffness_row_sum_is_zero():
    grid = StructuredGrid((4, 4, 4))
    Ke = ElementKernel("poisson", 3, grid.h).element_matrix()
    assert Ke.shape == (8, 8)
    assert np.allclose(Ke.sum(axis=1), 0.0, atol=1e-12)


@pytest.mark.parametrize("name", ["poisson_manufactured", "heat_variable_k", "elasticity_patch"])
def test_assembled_system_matches_independent_reference(name):
    pr = make_problem(name, (11, 9))
    part = CartesianPartition(pr.grid.n, 1)
    system = DistributedSystem(SerialComm(), pr.grid, part, pr.kernel)
    A, b = system.assemble(bc=True, bc_fn=pr.dirichlet)
    Aref, bref = assemble_reference(pr.grid, pr.kernel, pr.dirichlet)
    scale = max(float(abs(Aref).max()), 1e-30)
    diff = abs(A - Aref)
    assert (float(diff.max()) if diff.nnz else 0.0) <= 1e-13 * scale
    assert np.allclose(b, bref, atol=1e-12 * max(float(np.abs(bref).max()), 1e-30))
    assert abs(A - A.T).max() <= 1e-13 * scale   # symmetric after elimination
    assert A.diagonal().min() > 0.0            # SPD-ish, no empty rows


def test_dirichlet_elimination_keeps_identity_rows():
    pr = make_problem("poisson_manufactured", (9, 9))
    part = CartesianPartition(pr.grid.n, 1)
    system = DistributedSystem(SerialComm(), pr.grid, part, pr.kernel)
    A, _ = system.assemble(bc=True, bc_fn=pr.dirichlet)
    rows = np.nonzero(system.bc_dof)[0]
    assert rows.size > 0
    block = A[rows][:, :].toarray()
    assert np.allclose(block[:, rows], np.eye(rows.size))


def test_halo_plan_is_symmetric_between_peers():
    pr = make_problem("poisson_manufactured", (17, 13))

    def worker(comm):
        part = CartesianPartition(pr.grid.n, comm.size)
        system = DistributedSystem(comm, pr.grid, part, pr.kernel)
        return {
            "rank": comm.rank,
            "send": {int(k): int(v.size) for k, v in system.halo.send_dof.items()},
            "recv": {int(k): int(v.size) for k, v in system.halo.recv_dof.items()},
            "peers": list(system.halo.peers),
        }

    res = run(worker, 4, barrier_timeout=300)
    by_rank = {r["rank"]: r for r in res}
    for r in res:
        for peer in r["peers"]:
            assert by_rank[peer]["peers"].count(r["rank"]) == 1
            assert r["send"][peer] == by_rank[peer]["recv"][r["rank"]]
            assert r["recv"][peer] == by_rank[peer]["send"][r["rank"]]


@pytest.mark.parametrize("ncomp,name", [(1, "poisson_manufactured"), (2, "elasticity_patch")])
def test_halo_exchange_delivers_owner_values(ncomp, name):
    pr = make_problem(name, (15, 11))

    def worker(comm):
        part = CartesianPartition(pr.grid.n, comm.size)
        system = DistributedSystem(comm, pr.grid, part, pr.kernel)
        x = 1.0 + system.geo["gid"].astype(float).repeat(system.ncomp)
        system.halo.exchange(comm, x)
        ghost = ~system.geo["owned"]
        ref = 1.0 + system.geo["gid"][ghost].astype(float).repeat(system.ncomp)
        got = x.reshape(-1, system.ncomp)[ghost].reshape(-1)
        return {"rank": comm.rank, "err": float(np.abs(got - ref).max()) if ref.size else 0.0,
                "nghost": int(ghost.sum())}

    res = run(worker, 4, barrier_timeout=300)
    assert sum(r["nghost"] for r in res) > 0
    assert max(r["err"] for r in res) == 0.0


def test_element_box_contains_exactly_the_cells_touching_owned_nodes():
    pr = make_problem("poisson_manufactured", (13, 12))
    grid = pr.grid
    part = CartesianPartition(grid.n, 3)
    h = np.array(grid.h)
    for r in range(3):
        blk = part.block_slices(r)
        ebox = element_box(grid, part, r)
        expected = set()
        for i in range(max(blk[0].start - 1, 0), min(blk[0].stop, grid.nel[0])):
            for j in range(max(blk[1].start - 1, 0), min(blk[1].stop, grid.nel[1])):
                expected.add((i, j))
        got = set(zip(ebox["multi"][0].tolist(), ebox["multi"][1].tolist()))
        assert got == expected
        # cell centres
        expect_c = np.stack(
            [(ebox["multi"][d] + 0.5) * h[d] for d in range(2)], axis=1
        )
        assert np.allclose(ebox["centers"], expect_c)
        assert ebox["count"] == len(expected)


def test_owned_row_assembly_needs_no_off_process_accumulation():
    """Every row sum of the assembled (non-Dirichlet) operator is complete."""
    pr = make_problem("poisson_unit_source", (21, 19))
    part = CartesianPartition(pr.grid.n, 4)

    def worker(comm):
        system = DistributedSystem(comm, pr.grid, part, pr.kernel)
        A, _ = assemble_owned_rows(pr.grid, part, comm.rank, pr.kernel, system)
        rows = np.nonzero(system.geo["owned"])[0]
        rowsum = np.asarray(A[rows].sum(axis=1)).ravel()
        interior = rowsum[np.abs(rowsum) < 1e-12]
        return {"rank": comm.rank, "n_owned": int(rows.size),
                "n_zero_rowsum": int(interior.size)}

    res = run(worker, 4, barrier_timeout=300)
    # a complete owned-row assembly has zero row sums except next to the boundary
    assert sum(r["n_owned"] for r in res) == 21 * 19
    assert sum(r["n_zero_rowsum"] for r in res) > 0.5 * 21 * 19


def test_system_halo_audit_reports_zero_mismatch():
    pr = make_problem("poisson_manufactured", (17, 15))

    def worker(comm):
        part = CartesianPartition(pr.grid.n, comm.size)
        system = DistributedSystem(comm, pr.grid, part, pr.kernel)
        report = system.audit_halo()
        return {"rank": comm.rank, "err": report["max_error"],
                "n": report["n_values"]}

    res = run(worker, 4, barrier_timeout=300)
    assert max(r["err"] for r in res) == 0.0
    assert sum(r["n"] for r in res) > 0


def test_eliminate_dirichlet_handles_distinct_row_and_column_sets():
    """Ghost boundary DOFs must lose their column even if the row is not ours."""
    import scipy.sparse as sp

    n = 4
    A = sp.csr_matrix(np.ones((n, n)) - np.eye(n) * (n - 1))
    rows = np.array([False, True, False, False])   # I own DOF 1
    cols = np.array([False, True, True, False])    # DOF 2 is a ghost boundary
    E = eliminate_dirichlet(A, rows, cols).toarray()
    assert E[0, 2] == 0.0 and E[3, 2] == 0.0   # ghost column removed
    assert E[0, 0] == -2.0 and E[0, 3] == 1.0  # free rows untouched otherwise
    assert E[1, 1] == 1.0 and E[1, 2] == 0.0   # owned Dirichlet row -> identity
    assert E[2, 0] == 1.0                      # ghost row is not ours: untouched
