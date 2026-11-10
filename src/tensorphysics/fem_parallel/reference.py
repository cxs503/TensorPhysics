"""Independent reference implementations used to audit the parallel code path.

The reference assembler below is deliberately written the slow, obvious way:
loop over every cell, scatter the local element matrix with explicit index
arithmetic, no halo logic, no vectorisation.  Divergence from it therefore
indicates a real defect in the fast parallel path rather than a shared bug in a
common helper.
"""

from __future__ import annotations

from itertools import product
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .discretisation import ElementKernel, StructuredGrid

__all__ = [
    "assemble_reference",
    "apply_dirichlet",
    "serialise_solution",
    "global_from_serialised",
    "relative_l2",
]


def _node_gid(grid: StructuredGrid, multi: Sequence[int]) -> int:
    gid = 0
    for d, m in enumerate(multi):
        gid += int(m) * grid.strides[d]
    return gid


def assemble_reference(
    grid: StructuredGrid, kernel: ElementKernel, bc_fn: Callable[[np.ndarray], np.ndarray]
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Assemble the *whole* problem with a plain per-element loop."""
    ncomp = kernel.ncomp
    n = grid.num_nodes * ncomp
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    b = np.zeros(n)
    h = np.array(grid.h)
    nel = np.array(grid.nel)

    for corner in product(*[range(int(v)) for v in nel]):
        centre = (np.array(corner, dtype=float) + 0.5) * h
        Ke = kernel.element_matrices(centre[None, :])[0]
        fe = kernel.element_loads(centre[None, :])[0]
        gids = [_node_gid(grid, [corner[d] + a[d] for d in range(grid.ndim)])
                for a in product((0, 1), repeat=grid.ndim)]
        for ai, gi in enumerate(gids):
            for aj, gj in enumerate(gids):
                for ca in range(ncomp):
                    for cb in range(ncomp):
                        rows.append(gi * ncomp + ca)
                        cols.append(gj * ncomp + cb)
                        vals.append(Ke[ai * ncomp + ca, aj * ncomp + cb])
            for ca in range(ncomp):
                b[gi * ncomp + ca] += fe[ai * ncomp + ca]

    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A, b = apply_dirichlet(grid, A, b, bc_fn, ncomp)
    return A, b


def apply_dirichlet(
    grid: StructuredGrid,
    A: sp.csr_matrix,
    b: np.ndarray,
    bc_fn: Callable[[np.ndarray], np.ndarray],
    ncomp: int,
    tol: float = 1e-14,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Symmetric Dirichlet elimination on the assembled global system."""
    gids = np.arange(grid.num_nodes)
    xy = grid.coords(gids)
    multi = grid.multi(gids)
    mask = np.zeros(grid.num_nodes, dtype=bool)
    for d in range(grid.ndim):
        mask |= (multi[d] == 0) | (multi[d] == grid.n[d] - 1)
    if not mask.any():
        return A, b
    g = np.zeros(grid.num_nodes * ncomp)
    v = np.asarray(bc_fn(xy[mask]), dtype=float)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    g[mask.repeat(ncomp)] = v.reshape(-1)
    rows = np.nonzero(mask.repeat(ncomp))[0]
    b = b - A @ g
    A = A.tolil(copy=True)
    A[rows, :] = 0.0
    A[:, rows] = 0.0
    A[rows, rows] = 1.0
    A = A.tocsr()
    b[rows] = g[rows]
    return A, b


def serialise_solution(system, x: np.ndarray) -> Dict[str, np.ndarray]:
    """Owned global DOF ids + values of a rank-local solution vector."""
    owned = np.nonzero(system.owned_dof)[0]
    ncomp = system.ncomp
    nodes = owned // ncomp
    comp = owned % ncomp
    gids = system.geo["gid"][nodes]
    return {"dof": gids * ncomp + comp, "values": x[owned]}


def global_from_serialised(results, n_total: int) -> np.ndarray:
    """Scatter per-rank serialised solutions into one global vector."""
    out = np.zeros(int(n_total))
    for res in results:
        out[np.asarray(res["dof"], dtype=np.int64)] = np.asarray(res["values"], dtype=float)
    return out


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    if denom == 0.0:  # pragma: no cover
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / denom)
