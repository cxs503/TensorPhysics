"""Parallel PCG solver: accuracy, preconditioner behaviour and communication accounting."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from tensorphysics.fem_parallel.comm import SerialComm
from tensorphysics.fem_parallel.operator import DistributedSystem
from tensorphysics.fem_parallel.partition import CartesianPartition
from tensorphysics.fem_parallel.pcg import ASMPreconditioner, JacobiPreconditioner, cg
from tensorphysics.fem_parallel.problems import make_problem
from tensorphysics.fem_parallel.reference import global_from_serialised, relative_l2, serialise_solution
from tensorphysics.fem_parallel.runtime import run


def _system(name, n, rank=0):
    pr = make_problem(name, n)
    part = CartesianPartition(pr.grid.n, 1)
    system = DistributedSystem(SerialComm(rank=rank, size=1), pr.grid, part, pr.kernel)
    A, b = system.assemble(bc=True, bc_fn=pr.dirichlet)
    return pr, system, A, b


def test_cg_reproduces_the_direct_solution():
    pr, system, A, b = _system("poisson_manufactured", (41, 37))
    direct = spla.spsolve(A.tocsc(), b)
    res = cg(A, b, system, tol=1e-13, maxiter=5000,
             precond=JacobiPreconditioner(A, system), fixed_dof=system.bc_dof)
    assert res.converged
    assert relative_l2(res.x, direct) < 1e-10


def test_jacobi_preconditioner_is_inverse_diagonal():
    pr, system, A, b = _system("heat_variable_k", (21, 21))
    pc = JacobiPreconditioner(A, system)
    r = np.arange(system.ndof, dtype=float) + 1.0
    z = pc.apply(r)
    assert np.allclose(z, r / A.diagonal())


def test_asm_subdomain_solve_is_exact_on_the_owned_block():
    pr, system, A, b = _system("poisson_manufactured", (21, 21))
    pc = ASMPreconditioner(A, system)
    r = np.sin(np.arange(system.ndof, dtype=float)) + 2.0
    z = pc.apply(r)
    rows = pc.rows
    assert np.allclose(A[rows][:, rows] @ z[rows], r[rows], atol=1e-8 * np.abs(r).max())


def test_preconditioning_cuts_cg_iterations_by_half():
    pr, system, A, b = _system("poisson_unit_source", (81, 79))
    plain = cg(A, b, system, tol=1e-8, maxiter=8000, precond=None, fixed_dof=system.bc_dof)
    jac = cg(A, b, system, tol=1e-8, maxiter=8000,
             precond=JacobiPreconditioner(A, system), fixed_dof=system.bc_dof)
    asm = cg(A, b, system, tol=1e-8, maxiter=8000,
             precond=ASMPreconditioner(A, system), fixed_dof=system.bc_dof)
    assert plain.converged and jac.converged and asm.converged
    assert jac.iterations <= plain.iterations  # uniform diagonal: Jacobi is a scalar scaling
    assert 1.0 - asm.iterations / plain.iterations >= 0.5


def test_manufactured_solution_error_is_second_order():
    errs, hs = [], []
    for n in ((33, 33), (65, 65)):
        pr, system, A, b = _system("poisson_manufactured", n)
        res = cg(A, b, system, tol=1e-12, maxiter=8000,
                 precond=JacobiPreconditioner(A, system), fixed_dof=system.bc_dof)
        xy = pr.grid.coords(np.arange(pr.grid.num_nodes))
        exact = np.asarray(pr.exact(xy)).reshape(-1)
        errs.append(relative_l2(res.x, exact))
        hs.append(pr.grid.h[0])
    order = np.log2(errs[0] / errs[1]) / np.log2(hs[0] / hs[1])
    assert 1.7 < order < 2.4
    assert errs[1] < errs[0] / 3.0


def test_elasticity_patch_test_is_exact():
    pr, system, A, b = _system("elasticity_patch", (25, 23))
    direct = spla.spsolve(A.tocsc(), b)
    xy = pr.grid.coords(np.arange(pr.grid.num_nodes))
    exact = np.asarray(pr.exact(xy)).reshape(-1)
    assert relative_l2(direct, exact) < 1e-10


@pytest.mark.parametrize("name", ["poisson_manufactured", "heat_variable_k", "elasticity_patch"])
def test_parallel_solution_matches_serial_solution(name):
    n = (25, 23)
    pr = make_problem(name, n)
    n_dof = pr.grid.num_nodes * pr.ncomp

    def worker(comm):
        part = CartesianPartition(pr.grid.n, comm.size)
        system = DistributedSystem(comm, pr.grid, part, pr.kernel)
        A, b = system.assemble(bc=True, bc_fn=pr.dirichlet)
        res = cg(A, b, system, tol=1e-13, maxiter=20000,
                 precond=JacobiPreconditioner(A, system), fixed_dof=system.bc_dof)
        out = serialise_solution(system, res.x)
        out["iterations"] = res.iterations
        return out

    serial = global_from_serialised([worker(SerialComm())], n_dof)
    res = run(worker, 4, barrier_timeout=600)
    parallel = global_from_serialised(res, n_dof)
    assert relative_l2(parallel, serial) < 1e-10
    assert len({r["iterations"] for r in res}) == 1  # identical on every rank


def test_communication_accounting_is_consistent():
    pr = make_problem("poisson_unit_source", (41, 41))

    def worker(comm):
        part = CartesianPartition(pr.grid.n, comm.size)
        system = DistributedSystem(comm, pr.grid, part, pr.kernel)
        A, b = system.assemble(bc=True, bc_fn=pr.dirichlet)
        system.comm.reset_stats()
        res = cg(A, b, system, tol=1e-7, maxiter=4000,
                 precond=JacobiPreconditioner(A, system), fixed_dof=system.bc_dof)
        return {
            "rank": comm.rank,
            "t_solve": res.t_total,
            "t_comm": comm.comm_time(),
            "exchanges": comm.stats["exchange_calls"],
            "allreduces": comm.stats["allreduce_calls"],
            "bytes": comm.stats["exchange_bytes"],
            "fraction": res.comm_fraction,
        }

    res = run(worker, 4, barrier_timeout=600)
    for r in res:
        assert 0.0 <= r["t_comm"] <= r["t_solve"] * 1.05
        assert 0.0 <= r["fraction"] <= 1.0
        assert r["allreduces"] >= 3      # three inner products per iteration
        assert r["exchanges"] >= 1       # one halo exchange per matvec
        assert r["bytes"] > 0            # real data was moved


def test_cg_reports_nonconvergence_honestly():
    pr, system, A, b = _system("poisson_unit_source", (65, 65))
    res = cg(A, b, system, tol=1e-14, maxiter=3, precond=None, fixed_dof=system.bc_dof)
    assert res.iterations == 3
    assert not res.converged
    assert res.residual_history[-1] > 0.0


def test_parallel_matches_serial_on_tiny_uneven_partitions():
    """Small grids where a ghost node can sit on the global Dirichlet boundary."""
    cases = [("elasticity_patch", (7, 4)), ("poisson_manufactured", (7, 4)),
             ("elasticity_patch", (9, 5))]
    for name, n in cases:
        pr = make_problem(name, n)
        n_dof = pr.grid.num_nodes * pr.ncomp

        def worker(comm, name=name, n=n):
            part = CartesianPartition(pr.grid.n, comm.size)
            system = DistributedSystem(comm, pr.grid, part, pr.kernel)
            A, b = system.assemble(bc=True, bc_fn=pr.dirichlet)
            res = cg(A, b, system, tol=1e-13, maxiter=20000,
                     precond=JacobiPreconditioner(A, system), fixed_dof=system.bc_dof)
            out = serialise_solution(system, res.x)
            out["ghost_on_boundary"] = int(np.sum((~system.geo["owned"]) & system.boundary_mask()))
            return out

        serial = global_from_serialised([worker(SerialComm())], n_dof)
        for nranks in (3, 5):
            res = run(worker, nranks, barrier_timeout=120)
            parallel = global_from_serialised(res, n_dof)
            assert relative_l2(parallel, serial) < 1e-10, (name, n, nranks)
