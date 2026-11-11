"""Parallel Krylov solvers and preconditioners for the distributed system.

* :func:`cg` -- preconditioned conjugate gradient on the row-owned operator.
  Every inner product is a global reduction (``allreduce``) and every operator
  application needs one ghost-layer exchange, which is exactly the
  communication pattern of a production MPI FE solver.
* :class:`JacobiPreconditioner` -- diagonal scaling, ``O(1)`` communication per
  application (one local reciprocal, no exchange).
* :class:`ASMPreconditioner` -- additive Schwarz / block-Jacobi with exact
  subdomain solves of the owned DOF block (``splu`` factorisation), i.e. the
  local part of a one-level Schwarz method.  Setup is amortised over all
  Krylov iterations.
* :func:`apply_local_operator` -- helper reproducing the operator rows for the
  whole box, used to build the subdomain matrix.

Timing is split into ``t_matvec`` (local sparse product), ``t_comm`` (taken from
``comm.stats`` deltas) and ``t_precond`` so the communication fraction
``T_comm / T_total`` is a measurement, not an assumption.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .operator import DistributedSystem

__all__ = [
    "SolveResult",
    "cg",
    "cg_latency_reduced",
    "JacobiPreconditioner",
    "ASMPreconditioner",
    "local_operator",
]


@dataclass
class SolveResult:
    iterations: int
    converged: bool
    residual_history: List[float] = field(default_factory=list)
    x: Optional[np.ndarray] = None
    t_total: float = 0.0
    t_matvec: float = 0.0
    t_precond: float = 0.0
    t_comm: float = 0.0
    t_setup: float = 0.0
    n_owned: int = 0

    @property
    def comm_fraction(self) -> float:
        if self.t_total <= 0.0:
            return 0.0
        return float(self.t_comm / self.t_total)


class JacobiPreconditioner:
    """Diagonal (Jacobi) preconditioner."""

    name = "jacobi"

    def __init__(self, A: sp.csr_matrix, system: DistributedSystem) -> None:
        diag = A.diagonal()
        diag = np.where(np.abs(diag) > 0.0, diag, 1.0)
        self._inv = 1.0 / diag
        self.system = system

    def apply(self, r: np.ndarray) -> np.ndarray:
        return self._inv * r


class ASMPreconditioner:
    """One-level additive Schwarz: exact solve on the owned DOF block.

    The subdomain operator is the owned block of the assembled box operator
    (all element contributions of the owned rows are complete), with the ghost
    layer acting as a Dirichlet boundary -- the standard local component of a
    one-level Schwarz preconditioner.  Factorised once with ``splu``.
    """

    name = "asm"

    def __init__(self, A: sp.csr_matrix, system: DistributedSystem, limit: int = 400_000) -> None:
        t0 = time.perf_counter()
        rows = np.nonzero(system.owned_dof)[0]
        if rows.size > limit:
            raise ValueError(
                f"ASM subdomain has {rows.size} DOFs (limit {limit}); "
                "use fewer DOFs per rank or the Jacobi preconditioner"
            )
        self.rows = rows
        A_ff = A[rows][:, rows].tocsc()
        self._lu = spla.splu(A_ff)
        self.system = system
        self.setup_time = time.perf_counter() - t0
        self.subdomain_dofs = int(rows.size)

    def apply(self, r: np.ndarray) -> np.ndarray:
        z = np.zeros_like(r)
        z[self.rows] = self._lu.solve(r[self.rows])
        return z


def local_operator(A: sp.csr_matrix, system: DistributedSystem, x: np.ndarray) -> np.ndarray:
    """Apply ``A`` after exchanging the ghost layer (no global reduction)."""
    system.halo.exchange(system.comm, x)
    return A @ x


def _comm_snapshot(system: DistributedSystem) -> float:
    return system.comm.comm_time()


def cg(
    A: sp.csr_matrix,
    b: np.ndarray,
    system: DistributedSystem,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    maxiter: int = 20000,
    precond=None,
    fixed_dof: Optional[np.ndarray] = None,
    record_every: int = 1,
) -> SolveResult:
    """Preconditioned conjugate gradient.

    ``tol`` is relative to ``||b||`` on the owned DOFs (global norm).  DOFs
    flagged in ``fixed_dof`` (Dirichlet) are held at their initial value and
    excluded from the inner products, so the iteration is confined to the free
    block.
    """
    comm = system.comm
    owned = system.owned_dof if fixed_dof is None else (system.owned_dof & ~fixed_dof)
    # timers start before the first collective so that T_comm is a strict
    # subset of T_total
    t_comm0 = _comm_snapshot(system)
    t0 = time.perf_counter()
    bnorm = comm.sum_scalar(float(np.dot(b[owned], b[owned]))) ** 0.5
    x = np.zeros_like(b) if x0 is None else x0.copy()
    if fixed_dof is not None:
        x[fixed_dof] = b[fixed_dof]

    t_mv = 0.0
    t_pc = 0.0
    Ax = _timed_matvec(A, system, x)
    t_mv += Ax[1]
    r = b - Ax[0]
    if fixed_dof is not None:
        r[fixed_dof] = 0.0
    if precond is None:
        z = r.copy()
    else:
        t1 = time.perf_counter()
        z = precond.apply(r)
        t_pc += time.perf_counter() - t1
        if fixed_dof is not None:
            z[fixed_dof] = 0.0
    p = z.copy()
    rz = system.dot(r, z)
    history = [float(np.sqrt(system.dot(r, r)))]
    converged = history[0] <= tol * bnorm
    it = 0
    for it in range(1, int(maxiter) + 1):
        Ap = _timed_matvec(A, system, p)
        t_mv += Ap[1]
        pAp = system.dot(p, Ap[0])
        if pAp <= 0.0:  # pragma: no cover - indefiniteness guard
            converged = False
            break
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap[0]
        if fixed_dof is not None:
            r[fixed_dof] = 0.0
            x[fixed_dof] = b[fixed_dof]
        if precond is None:
            z = r.copy()
        else:
            t1 = time.perf_counter()
            z = precond.apply(r)
            t_pc += time.perf_counter() - t1
            if fixed_dof is not None:
                z[fixed_dof] = 0.0
        # one collective for both inner products of this step
        rz_new, rr_new = system.dot_pair(r, z, r, r)
        rnorm = float(np.sqrt(max(rr_new, 0.0)))
        if record_every and (it % record_every == 0):
            history.append(rnorm)
        if rnorm <= tol * bnorm:
            converged = True
            break
        if rz == 0.0:  # pragma: no cover
            break
        beta = rz_new / rz
        rz = rz_new
        p = z + beta * p
        if fixed_dof is not None:
            p[fixed_dof] = 0.0

    t_total = time.perf_counter() - t0
    t_comm = _comm_snapshot(system) - t_comm0
    return SolveResult(
        iterations=it,
        converged=converged,
        residual_history=history,
        x=x,
        t_total=t_total,
        t_matvec=t_mv,
        t_precond=t_pc,
        t_comm=t_comm,
        n_owned=system.n_owned,
    )


def cg_latency_reduced(
    A: sp.csr_matrix,
    b: np.ndarray,
    system: DistributedSystem,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    maxiter: int = 20000,
    precond=None,
    fixed_dof: Optional[np.ndarray] = None,
    record_every: int = 1,
) -> SolveResult:
    """PCG with the global-synchronisation count per iteration cut from 3 to 2.

    Identical algebraic recurrences as :func:`cg` -- only the *order* of the
    local work changes so that the ghost exchange and the two inner products of
    the step can be published in the same collective:

    * classic order: ``exchange(p)`` -> ``Ap = A p`` -> ``(p, Ap)`` likely
      one reduction -> update -> ``(r, z)`` and ``(r, r)`` -> second reduction;
    * reordered: after the update the *preconditioned residual* ``z_raw`` is
      known locally, so its ghost exchange and the products ``(r, z_raw)`` and
      ``(r, r)`` share one barrier; the matvec is then applied to ``z_raw`` and
      ``A p_k = A z_raw + beta_k A p_{k-1}`` follows from linearity without an
      extra exchange.

    The iteration therefore costs one fused exchange+reduction and one scalar
    reduction, i.e. two latencies instead of three.  Round-off differs from
    :func:`cg` at the 1e-15 level only; the iterates agree to machine precision
    in serial (see ``tests/test_fem_parallel_latency_reduced.py``).
    """
    comm = system.comm
    owned = system.owned_dof if fixed_dof is None else (system.owned_dof & ~fixed_dof)
    idx = np.nonzero(owned)[0]
    t_comm0 = _comm_snapshot(system)
    t0 = time.perf_counter()

    def local_dot(u, v):
        return float(np.dot(u[idx], v[idx]))

    bnorm = comm.sum_scalar(local_dot(b, b)) ** 0.5
    x = np.zeros_like(b) if x0 is None else x0.copy()
    if fixed_dof is not None:
        x[fixed_dof] = b[fixed_dof]

    t_mv = 0.0
    t_pc = 0.0

    # first residual and preconditioned residual
    Ax = _timed_matvec(A, system, x)
    t_mv += Ax[1]
    r = b - Ax[0]
    if fixed_dof is not None:
        r[fixed_dof] = 0.0
    t1 = time.perf_counter()
    z = r.copy() if precond is None else precond.apply(r)
    t_pc += time.perf_counter() - t1
    if fixed_dof is not None:
        z[fixed_dof] = 0.0
    # exchange z once so that A z can be formed after the fused collective
    rz, rr = system.exchange_with_reduction(z, r, z, r, r)
    rnorm = float(np.sqrt(max(rr, 0.0)))
    p = z.copy()
    # ghosts of z are already valid (exchanged in the fused collective above),
    # so A z is a pure local matvec -- no second exchange
    Az = A @ z
    Ap = Az.copy()
    pAp = local_dot(p, Ap)
    pAp = comm.sum_scalar(pAp)
    history = [rnorm]
    converged = rnorm <= tol * bnorm
    it = 0
    for it in range(1, int(maxiter) + 1):
        if pAp <= 0.0:  # pragma: no cover - indefiniteness guard
            converged = False
            break
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        if fixed_dof is not None:
            r[fixed_dof] = 0.0
            x[fixed_dof] = b[fixed_dof]
        t1 = time.perf_counter()
        z_new = r.copy() if precond is None else precond.apply(r)
        t_pc += time.perf_counter() - t1
        if fixed_dof is not None:
            z_new[fixed_dof] = 0.0
        # one collective: ghost exchange of z_new + (r, z_new) + (r, r)
        rz_new, rr_new = system.exchange_with_reduction(z_new, r, z_new, r, r)
        rnorm = float(np.sqrt(max(rr_new, 0.0)))
        if record_every and (it % record_every == 0):
            history.append(rnorm)
        if rnorm <= tol * bnorm:
            converged = True
            break
        if rz == 0.0:  # pragma: no cover
            break
        beta = rz_new / rz
        p = z_new + beta * p
        t_mv0 = time.perf_counter()
        Az = A @ z_new  # ghosts already exchanged in the fused collective
        t_mv += time.perf_counter() - t_mv0
        # A p = A z_new + beta * A p_prev  (linearity: no second exchange)
        Ap = Az + beta * Ap
        if fixed_dof is not None:
            p[fixed_dof] = 0.0
            Ap[fixed_dof] = 0.0
        pAp = comm.sum_scalar(local_dot(p, Ap))
        rz = rz_new

    t_total = time.perf_counter() - t0
    t_comm = _comm_snapshot(system) - t_comm0
    return SolveResult(
        iterations=it,
        converged=converged,
        residual_history=history,
        x=x,
        t_total=t_total,
        t_matvec=t_mv,
        t_precond=t_pc,
        t_comm=t_comm,
        n_owned=system.n_owned,
    )


def _timed_matvec(A: sp.csr_matrix, system: DistributedSystem, x: np.ndarray):
    t0 = time.perf_counter()
    y = local_operator(A, system, x)
    return y, time.perf_counter() - t0
