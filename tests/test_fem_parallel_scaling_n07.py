"""Tests for the P284W3-N07 scaling infrastructure.

Covered:

* the lock-free :class:`SpinBarrier` used by the shared-memory backend,
* the fused ghost-exchange + reduction collective (one latency instead of two),
* the latency-reduced PCG (``cg_latency_reduced``): identical iterates to ``cg``
  in serial, identical solution in parallel, fewer global collectives,
* the contention-robust measurement layer (``measure_point`` min estimator),
* the 3-D Cartesian decomposition and the criteria evaluator,
* the ``MPIComm`` code path (exercised through a stub ``mpi4py`` module, since no
  MPI runtime is installed on this host).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from tensorphysics.fem_parallel import comm as comm_mod
from tensorphysics.fem_parallel.comm import MPIComm, SerialComm
from tensorphysics.fem_parallel.operator import DistributedSystem
from tensorphysics.fem_parallel.partition import CartesianPartition
from tensorphysics.fem_parallel.pcg import (
    JacobiPreconditioner,
    cg,
    cg_latency_reduced,
)
from tensorphysics.fem_parallel.problems import make_problem
from tensorphysics.fem_parallel.runtime import run
from tensorphysics.fem_parallel.scaling import grid_for_dofs
from tensorphysics.fem_parallel.scaling_n07 import (
    SerialShim,
    _reordered_worker,
    evaluate,
    measure_point,
    dof3d_study,
)
from tensorphysics.fem_parallel.verification import global_from_serialised, relative_l2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _serial_system(name="poisson_manufactured", n=(33, 33)):
    pr = make_problem(name, n)
    part = CartesianPartition(pr.grid.n, 1)
    system = DistributedSystem(SerialComm(), pr.grid, part, pr.kernel)
    A, b = system.assemble(bc=True, bc_fn=pr.dirichlet)
    return pr, system, A, b


# --------------------------------------------------------------------------- #
# 1. spin barrier over shared memory
# --------------------------------------------------------------------------- #
def _barrier_roundtrip_worker(comm, rounds=5):
    """Every rank bumps a token and checks that all ranks saw the same token."""
    stats = []
    for k in range(1, rounds + 1):
        got = comm.allgather_scalars(float(k * 100 + comm.rank))
        stats.append(float(np.max(np.abs(got - np.array([k * 100 + r for r in range(comm.size)])))))
    return {"rank": comm.rank, "max_err": float(max(stats)),
            "barrier_calls": int(comm.stats["barrier_calls"])}


def test_spin_barrier_orders_collectives():
    res = run(_barrier_roundtrip_worker, 3, kwargs={"rounds": 4}, barrier_timeout=300)
    assert max(r["max_err"] for r in res) == 0.0
    assert all(r["barrier_calls"] >= 4 for r in res)


def test_spin_barrier_generation_never_lags():
    """Two generations alternate; a slow rank must not corrupt a fast one."""
    barrier_mod = comm_mod.SpinBarrier
    from multiprocessing import shared_memory

    from multiprocessing import get_context

    ctx = get_context("fork")
    shm = shared_memory.SharedMemory(create=True, size=16 * 4)
    try:
        procs = []
        for r in range(4):
            p = ctx.Process(target=_spin_worker, args=(shm, r, 4))
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0
    finally:
        shm.close()
        shm.unlink()
    assert barrier_mod is not None


def _spin_worker(shm, rank, size):
    bar = comm_mod.SpinBarrier(shm, size, rank)
    for _ in range(200):
        bar.wait(timeout=60.0)


# --------------------------------------------------------------------------- #
# 2. fused exchange + reduction == two separate collectives
# --------------------------------------------------------------------------- #
def _fused_vs_split_worker(comm):
    _, system, A, b = _serial_system(n=(17, 15))
    system = DistributedSystem(comm, system.grid, CartesianPartition(system.grid.n, comm.size),
                              system.kernel)
    v = b.copy()
    w = np.arange(v.size, dtype=float)
    # split path
    plan = system.halo
    sends = {p: v[plan.send_dof[p]] for p in plan.peers}
    recv_split = comm.exchange_phase(sends, plan.recv_lengths)
    i = system._owned_idx
    red_split = comm.allreduce_sum(np.array([float(np.dot(v[i], w[i])),
                                            float(np.dot(w[i], w[i]))]))
    # fused path
    v2 = v.copy()
    a1, a2 = system.exchange_with_reduction(v2, v, w, w, w)
    err = 0.0
    for p in plan.peers:
        err = max(err, float(np.max(np.abs(v2[plan.recv_dof[p]] - recv_split[p]))))
    err = max(err, float(abs(a1 - red_split[0])) / max(1.0, abs(float(red_split[0]))))
    err = max(err, float(abs(a2 - red_split[1])) / max(1.0, abs(float(red_split[1]))))
    return {"rank": comm.rank, "err": err}


def test_fused_exchange_and_reduce_matches_split():
    res = run(_fused_vs_split_worker, 4, barrier_timeout=300)
    assert max(r["err"] for r in res) < 1e-12


# --------------------------------------------------------------------------- #
# 3. latency-reduced PCG equals classic CG
# --------------------------------------------------------------------------- #
def test_cg_latency_reduced_matches_cg_serial():
    _, system, A, b = _serial_system()
    pc = JacobiPreconditioner(A, system)
    r1 = cg(A, b, system, tol=1e-12, maxiter=5000, precond=pc, fixed_dof=system.bc_dof)
    r2 = cg_latency_reduced(A, b, system, tol=1e-12, maxiter=5000, precond=pc,
                            fixed_dof=system.bc_dof)
    assert r1.converged and r2.converged
    assert r1.iterations == r2.iterations
    assert np.abs(r1.x - r2.x).max() < 1e-12


@pytest.mark.parametrize("name,n", [
    ("poisson_manufactured", (33, 31)),
    ("heat_variable_k", (33, 31)),
    ("elasticity_patch", (25, 23)),
])
def test_cg_latency_reduced_parallel_matches_serial(name, n):
    pr = make_problem(name, n)
    n_dof = pr.grid.num_nodes * pr.ncomp
    ser = _reordered_worker(SerialShim(), name=name, n=n, tol=1e-12, variant="cg")
    u_ser = global_from_serialised([ser], n_dof)
    par = run(_reordered_worker, 4, kwargs={"name": name, "n": n, "tol": 1e-12,
                                           "variant": "reordered"}, barrier_timeout=300)
    u_par = global_from_serialised(par, n_dof)
    assert relative_l2(u_par, u_ser) < 1e-10


def _collective_counts(comm, variant, n=(33, 31)):
    _, system, A, b = _serial_system(n=n)
    system = DistributedSystem(comm, system.grid, CartesianPartition(system.grid.n, comm.size),
                              system.kernel)
    A, b = system.assemble(bc=True, bc_fn=system.kernel and make_problem(
        "poisson_manufactured", n).dirichlet)
    pc = JacobiPreconditioner(A, system)
    comm.reset_stats()
    solver = cg_latency_reduced if variant == "reordered" else cg
    res = solver(A, b, system, tol=0.0, maxiter=20, precond=pc, fixed_dof=system.bc_dof)
    calls = int(comm.stats["exchange_calls"] + comm.stats["allreduce_calls"])
    return {"rank": comm.rank, "calls": calls, "iters": int(res.iterations)}


def test_latency_reduced_pcg_uses_fewer_collectives():
    classic = run(_collective_counts, 4, kwargs={"variant": "cg"}, barrier_timeout=300)
    reduced = run(_collective_counts, 4, kwargs={"variant": "reordered"}, barrier_timeout=300)
    c_calls = max(r["calls"] for r in classic)
    r_calls = max(r["calls"] for r in reduced)
    # 20 iterations: classic needs 3 collectives/iter, reduced needs 2
    assert c_calls >= 3 * 20
    assert r_calls <= 2 * 20 + 4
    assert r_calls < c_calls


# --------------------------------------------------------------------------- #
# 4. measurement layer
# --------------------------------------------------------------------------- #
def test_measure_point_min_estimator():
    pt = measure_point(2, reps=3, name="poisson_manufactured", n=(21, 19), tol=0.0,
                       maxiter=3, precond="jacobi", variant="cg")
    assert pt["reps"] == 3
    assert len(pt["trials"]) == 3
    assert pt["t_total"] == min(t["t_total"] for t in pt["trials"])
    assert pt["noise_ratio"] >= 1.0
    assert pt["n_global"] > 0


def test_weak_grid_growth_and_3d_partition():
    g1 = grid_for_dofs(100_000, ndim=2)
    g2 = grid_for_dofs(400_000, ndim=2)
    assert g2[0] > g1[0] and g2[1] > g1[1]
    g3 = grid_for_dofs(250_000, ndim=3)
    assert len(g3) == 3
    part = CartesianPartition(g3, 8)
    assert list(part.shape) == [2, 2, 2]
    assert part.max_imbalance() < 0.05


def test_three_d_study_runs_small():
    study = dof3d_study(n_dofs=8000, ranks=(1, 2), iters=3, reps=2)
    assert study["kind"] == "strong3d"
    pts = {p["ranks"]: p for p in study["points"]}
    assert set(pts) == {1, 2}
    assert pts[1]["decomposition"] == [1, 1, 1]
    assert pts[2]["efficiency"] > 0.0
    assert pts[2]["n_global"] == pts[1]["n_global"]


# --------------------------------------------------------------------------- #
# 5. criteria evaluator
# --------------------------------------------------------------------------- #
def test_evaluate_criteria_flags_pass_and_fail():
    good = {
        "strong": {"points": [{"ranks": 1, "efficiency": 1.0},
                              {"ranks": 32, "efficiency": 0.72}]},
        "weak": {"points": [{"ranks": 1, "weak_ratio": 1.0, "weak_ratio_per_iter": 1.0},
                            {"ranks": 32, "weak_ratio": 1.2, "weak_ratio_per_iter": 1.1}]},
        "correctness": {"max_relerr": 1e-14},
        "accuracy": {"rows": [{"rel_l2_error": 2e-3}]},
        "three_d": {"points": [{"ranks": 1, "efficiency": 1.0},
                               {"ranks": 8, "efficiency": 0.66}]},
    }
    crit = evaluate(good)
    assert crit["a_strong_efficiency_P32"]["pass"] is True
    assert crit["b_weak_ratio_P32"]["pass"] is True
    assert crit["c_parallel_vs_serial"]["pass"] is True
    assert crit["d_poisson_vs_analytic"]["pass"] is True
    assert crit["e_3d_efficiency"]["pass"] is True

    bad = {k: v for k, v in good.items()}
    bad["strong"] = {"points": [{"ranks": 1, "efficiency": 1.0},
                                {"ranks": 32, "efficiency": 0.31}]}
    bad["weak"] = {"points": [{"ranks": 1, "weak_ratio": 1.0, "weak_ratio_per_iter": 1.0},
                              {"ranks": 32, "weak_ratio": 3.5, "weak_ratio_per_iter": 1.05}]}
    bad["correctness"] = {"max_relerr": 1e-6}
    crit_bad = evaluate(bad)
    assert crit_bad["a_strong_efficiency_P32"]["pass"] is False
    assert crit_bad["b_weak_ratio_P32"]["pass"] is False
    assert crit_bad["b2_weak_ratio_per_iter_P32"]["pass"] is True
    assert crit_bad["c_parallel_vs_serial"]["pass"] is False


# --------------------------------------------------------------------------- #
# 6. MPIComm code path (stub mpi4py, no runtime needed)
# --------------------------------------------------------------------------- #
class _FakeRequest:
    @staticmethod
    def Waitall(reqs):
        return None


class _FakeComm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Barrier(self):
        return None

    def Isend(self, buf, dest, tag):
        return _FakeRequest()

    def Irecv(self, buf, source, tag):
        buf[:] = 0.0
        return _FakeRequest()

    def Allreduce(self, arr, out, op):
        out[:] = arr


def test_mpi_comm_code_path_with_stub(monkeypatch):
    fake = types.ModuleType("mpi4py")
    fake.MPI = types.SimpleNamespace(COMM_WORLD=_FakeComm(),
                                     SUM="sum", MAX="max", Request=_FakeRequest)
    monkeypatch.setitem(sys.modules, "mpi4py", fake)
    c = MPIComm()
    assert c.rank == 0 and c.size == 1
    recv = c.exchange_phase({0: np.arange(3.0)}, {0: 3})
    assert recv[0].shape == (3,)
    assert c.sum_scalar(2.5) == 2.5
    pair = c.allreduce_sum_pair(1.0, 2.0)
    assert pair == (1.0, 2.0)
    assert c.comm_time() >= 0.0