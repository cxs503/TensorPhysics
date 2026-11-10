"""Distributed assembly, halo exchange and parallel operator application.

The parallel FE data model is *node/DOF owned*:

* Every rank owns a contiguous block of the structured grid and works on a
  *box* = its block grown by one node in every direction (the halo layer).
* Element contributions are assembled **only for rows the rank owns**.  Because
  all elements touching an owned node lie inside the box, no off-process
  accumulation is needed and the row sums are complete -- the same result as a
  gather-scatter assembly, without the scatter phase.
* A :class:`HaloPlan` maps, for every peer, the box positions this rank sends
  (owned nodes needed by the peer) and the box positions it receives (ghost
  nodes owned by the peer).  Both sides order the chunks by global node id, so
  the exchange is metadata-free and verified by construction
  (:func:`tensorphysics.fem_parallel.operator.HaloPlan.audit`).
* :class:`DistributedSystem` exposes the operations a Krylov solver needs:
  ``matvec``, ``dot`` (global reduction over owned DOFs only), ``axpy``,
  ``exchange`` and Dirichlet elimination.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .comm import Comm
from .discretisation import ElementKernel, StructuredGrid
from .partition import CartesianPartition

__all__ = [
    "HaloPlan",
    "DistributedSystem",
    "assemble_owned_rows",
    "element_box",
    "eliminate_dirichlet",
    "dirichlet_mask",
]


@dataclass
class HaloPlan:
    """Send/receive bookkeeping for the one-node ghost layer.

    ``send_pos``/``recv_pos`` hold *node* positions in the box numbering; the
    per-component DOF index arrays (:attr:`send_dof`, :attr:`recv_dof`) are
    derived once with the physics' ``ncomp`` so that vector-valued (elasticity)
    fields are exchanged component-wise in the same order on both sides.
    """

    rank: int
    peers: Tuple[int, ...]
    send_pos: Dict[int, np.ndarray]
    recv_pos: Dict[int, np.ndarray]
    ncomp: int = 1

    def __post_init__(self) -> None:
        self.ncomp = int(self.ncomp)
        cs = np.arange(self.ncomp)
        self.send_dof = {
            p: (v[:, None] * self.ncomp + cs[None, :]).ravel() for p, v in self.send_pos.items()
        }
        self.recv_dof = {
            p: (v[:, None] * self.ncomp + cs[None, :]).ravel() for p, v in self.recv_pos.items()
        }

    @property
    def recv_lengths(self) -> Dict[int, int]:
        return {p: int(v.size) for p, v in self.recv_dof.items()}

    def exchange(self, comm: Comm, x: np.ndarray) -> None:
        """Fill the ghost entries of ``x`` with the owning rank's values."""
        if not self.peers:
            comm.exchange_phase({}, {})
            return
        sends = {p: x[self.send_dof[p]] for p in self.peers}
        recv = comm.exchange_phase(sends, self.recv_lengths)
        for p in self.peers:
            x[self.recv_dof[p]] = recv[p]

    def audit(self, comm: Comm, gid: np.ndarray) -> Dict[str, float]:
        """Self-check that every exchanged value belongs to the node it fills.

        ``gid`` is the box-wise global node id; the transported field is
        ``f(node) = 1 + gid``, which every rank can compute independently, so a
        wrong peer mapping or a mis-ordered chunk shows up as a non-zero error.
        """
        x = 1.0 + np.repeat(np.asarray(gid, dtype=float), self.ncomp)
        before = x.copy()
        self.exchange(comm, x)
        moved = np.zeros(x.size, dtype=bool)
        for idx in self.recv_dof.values():
            moved[idx] = True
        err = float(np.abs(x[moved] - before[moved]).max()) if moved.any() else 0.0
        return {"max_error": err, "n_values": float(moved.sum())}


def _box_geometry(
    grid: StructuredGrid, part: CartesianPartition, rank: int
) -> Dict[str, np.ndarray]:
    """Multi-index / linear-id bookkeeping for one rank's box."""
    box = part.box_slices(rank)
    lo = np.array([s.start for s in box], dtype=np.int64)
    hi = np.array([s.stop for s in box], dtype=np.int64)
    nbox = hi - lo
    bstrides = []
    s = 1
    for v in nbox:
        bstrides.append(s)
        s *= int(v)
    bstrides = tuple(int(v) for v in bstrides)
    nloc = int(np.prod(nbox))
    lin = np.arange(nloc, dtype=np.int64)
    multi = []
    for d in range(grid.ndim):
        multi.append((lin // bstrides[d]) % nbox[d] + lo[d])
    gid = grid.gid(multi)
    owned = np.ones(nloc, dtype=bool)
    for d in range(grid.ndim):
        sl = part.block_slices(rank)[d]
        owned &= (multi[d] >= sl.start) & (multi[d] < sl.stop)
    return {
        "lo": lo,
        "hi": hi,
        "nbox": nbox,
        "bstrides": np.array(bstrides, dtype=np.int64),
        "multi": multi,
        "gid": gid,
        "owned": owned,
        "box_slices": np.array([(s.start, s.stop) for s in box]),
    }


def build_halo_plan(
    grid: StructuredGrid, part: CartesianPartition, rank: int, ncomp: int = 1
) -> HaloPlan:
    """Construct the ghost-layer exchange plan for ``rank``.

    Receive list: every box node not owned by ``rank`` (its owner must send it).
    Send list: every owned node that lies inside a peer's box, i.e. within
    Chebyshev distance one of that peer's block -- enumerated by shifting the
    owned nodes over the ``3**ndim - 1`` non-zero offsets and asking which rank
    owns the shifted node.  Both lists are sorted by global node id, so the two
    sides of a pair agree on the chunk ordering without any metadata exchange.
    """
    geo = _box_geometry(grid, part, rank)
    multi, gid = geo["multi"], geo["gid"]
    owned = geo["owned"]
    ndim = grid.ndim

    # -- receive: ghost nodes of my box, grouped by owner -------------------
    ghost_idx = np.nonzero(~owned)[0]
    recv_pos: Dict[int, np.ndarray] = {}
    if ghost_idx.size:
        owners = part.rank_of_index([multi[d][ghost_idx] for d in range(ndim)])
        for q in np.unique(owners):
            sel = ghost_idx[owners == q]
            recv_pos[int(q)] = sel[np.argsort(gid[sel], kind="stable")]

    # -- send: owned nodes inside a peer's box -----------------------------
    own_idx = np.nonzero(owned)[0]
    buckets: Dict[int, list] = {}
    if own_idx.size:
        for off in product((-1, 0, 1), repeat=ndim):
            if not any(off):
                continue
            shifted, inside = [], np.ones(own_idx.size, dtype=bool)
            for d in range(ndim):
                pos = multi[d][own_idx] + off[d]
                inside &= (pos >= 0) & (pos <= grid.n[d] - 1)
                shifted.append(pos)
            if not inside.any():
                continue
            idx = own_idx[inside]
            s_multi = [shifted[d][inside] for d in range(ndim)]
            s_owner = part.rank_of_index(s_multi)
            for q in np.unique(s_owner):
                if int(q) == rank:
                    continue
                buckets.setdefault(int(q), []).append(idx[s_owner == q])
    send_pos: Dict[int, np.ndarray] = {}
    for q, chunks in buckets.items():
        arr = np.unique(np.concatenate(chunks))
        arr = arr[np.argsort(gid[arr], kind="stable")]
        send_pos[q] = arr

    peers = tuple(sorted(p for p in send_pos if p in recv_pos))
    send_pos = {p: send_pos[p] for p in peers}
    recv_pos = {p: recv_pos[p] for p in peers}
    return HaloPlan(rank=rank, peers=peers, send_pos=send_pos, recv_pos=recv_pos, ncomp=ncomp)


@dataclass
class DistributedSystem:
    """Distributed linear system for one physics case on one rank."""

    comm: Comm
    grid: StructuredGrid
    part: CartesianPartition
    kernel: ElementKernel
    rank: int = 0

    def __post_init__(self) -> None:
        self.rank = int(self.comm.rank)
        self.geo = _box_geometry(self.grid, self.part, self.rank)
        self.ncomp = self.kernel.ncomp
        self.nbox = int(self.geo["gid"].size)
        self.ndof = self.nbox * self.ncomp
        self.owned_node = self.geo["owned"]
        self.owned_dof = np.repeat(self.geo["owned"], self.ncomp)
        self.halo = build_halo_plan(self.grid, self.part, self.rank, self.ncomp)
        self.n_owned = int(self.owned_dof.sum())
        self.n_global = self.grid.num_nodes * self.ncomp
        self.Ke = self.kernel.element_matrix()
        self.fe = self.kernel.element_load()

    # -- coordinates / boundary conditions ---------------------------------
    def node_coords(self) -> np.ndarray:
        return self.grid.coords(self.geo["gid"])

    def boundary_mask(self) -> np.ndarray:
        """Nodes on the boundary of the *global* domain (Dirichlet)."""
        mask = np.zeros(self.nbox, dtype=bool)
        multi = self.geo["multi"]
        for d in range(self.grid.ndim):
            mask |= (multi[d] == 0) | (multi[d] == self.grid.n[d] - 1)
        return mask

    def dirichlet_values(self, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Analytic Dirichlet data ``fn(x)`` expanded to DOF layout.

        Values are filled on **every** box node that lies on the global boundary,
        ghosts included: the RHS elimination ``b - A g`` needs the known data of
        neighbour-owned Dirichlet nodes as well (a ghost node can sit on the
        global boundary for small grids / uneven splits).
        """
        x = self.node_coords()
        mask = self.boundary_mask()
        vals = np.zeros((self.nbox, self.ncomp))
        if mask.any():
            v = np.asarray(fn(x[mask]), dtype=float)
            if v.ndim == 1:
                v = v.reshape(-1, 1)
            vals[mask] = v.reshape(int(mask.sum()), self.ncomp)
        return vals.reshape(self.ndof)

    # -- assembly -----------------------------------------------------------
    def assemble(self, bc: bool = True, bc_fn=None) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Assemble the owned-row operator on the box (shape ``ndof x ndof``).

        With ``bc``/``bc_fn`` the Dirichlet rows are replaced by ``I`` and the
        known data are moved to the right-hand side (symmetric elimination,
        ``O(nnz)`` in the CSR data array), leaving an SPD system on the free
        DOFs.
        """
        A, b = assemble_owned_rows(self.grid, self.part, self.rank, self.kernel, self)
        box_bc = np.repeat(self.boundary_mask(), self.ncomp)
        self.bc_mask = self.boundary_mask() & self.owned_node
        bdof = np.repeat(self.bc_mask, self.ncomp)
        self.bc_dof = bdof
        self.bc_dof_box = box_bc
        if not (bc and bc_fn is not None):
            return A, b
        g = self.dirichlet_values(bc_fn)
        # move the known data to the RHS ...
        b = b - A @ g
        # ... then eliminate rows (owned Dirichlet DOFs) and columns (every
        # Dirichlet DOF in the box, ghosts included)
        A = eliminate_dirichlet(A, bdof, box_bc)
        rows = np.nonzero(bdof)[0]
        b[rows] = g[rows]
        return A, b

    # -- linear algebra -----------------------------------------------------
    def dot(self, a: np.ndarray, b: np.ndarray) -> float:
        local = float(np.dot(a[self.owned_dof], b[self.owned_dof]))
        return self.comm.sum_scalar(local)

    def norm(self, a: np.ndarray) -> float:
        return float(np.sqrt(max(self.dot(a, a), 0.0)))

    def matvec(self, A: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
        self.halo.exchange(self.comm, x)
        return A @ x

    def audit_halo(self) -> Dict[str, float]:
        """Exchange the analytic field ``1 + gid`` and report the mismatch."""
        return self.halo.audit(self.comm, self.geo["gid"])


def dirichlet_mask(grid: StructuredGrid, multi: Sequence[np.ndarray]) -> np.ndarray:
    mask = np.zeros(np.shape(multi[0]), dtype=bool)
    for d in range(grid.ndim):
        mask |= (multi[d] == 0) | (multi[d] == grid.n[d] - 1)
    return mask


def element_box(
    grid: StructuredGrid, part: CartesianPartition, rank: int
) -> Dict[str, np.ndarray]:
    """All cells that touch an owned node, with their centres."""
    block = part.block_slices(rank)
    lo, hi = [], []
    for d in range(grid.ndim):
        nel = grid.nel[d]
        lo.append(max(int(block[d].start) - 1, 0))
        hi.append(min(int(block[d].stop), nel))
    lo = np.array(lo, dtype=np.int64)
    hi = np.array(hi, dtype=np.int64)
    nbox = hi - lo
    strides, s = [], 1
    for v in nbox:
        strides.append(s)
        s *= int(v)
    strides = tuple(int(v) for v in strides)
    lin = np.arange(int(np.prod(nbox)), dtype=np.int64)
    multi = [(lin // strides[d]) % nbox[d] + lo[d] for d in range(grid.ndim)]
    h = np.array(grid.h, dtype=float)
    centers = np.stack([(multi[d] + 0.5) * h[d] for d in range(grid.ndim)], axis=1)
    return {
        "lo": lo,
        "nbox": nbox,
        "strides": np.array(strides, dtype=np.int64),
        "multi": multi,
        "centers": centers,
        "count": int(lin.size),
    }


def assemble_owned_rows(
    grid: StructuredGrid,
    part: CartesianPartition,
    rank: int,
    kernel: ElementKernel,
    system: Optional[DistributedSystem] = None,
    ebox: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Vectorised assembly of the rows owned by ``rank`` (box-local numbering).

    Element matrices/loads are evaluated once per cell of the element box
    (per-element coefficients such as a varying conductivity are therefore
    honoured) and then scattered into the owned rows.
    """
    geo = system.geo if system is not None else _box_geometry(grid, part, rank)
    ndim = grid.ndim
    ncomp = kernel.ncomp
    if ebox is None:
        ebox = element_box(grid, part, rank)
    if kernel.is_constant:
        Ke_all = kernel.element_matrices(ebox["centers"][:1])
        Ke_const = Ke_all[0]
    else:
        Ke_all = kernel.element_matrices(ebox["centers"])
        Ke_const = None
    fe_all = kernel.element_loads(ebox["centers"])

    bstrides = geo["bstrides"]
    lo, hi = geo["lo"], geo["hi"]
    nbox = int(geo["gid"].size)
    multi = geo["multi"]
    owned = geo["owned"]
    elo = ebox["lo"]
    estrides = ebox["strides"]
    emulti = ebox["multi"]

    owned_lin = np.nonzero(owned)[0]
    own_multi = [m[owned_lin] for m in multi]

    rows_l: List[np.ndarray] = []
    cols_l: List[np.ndarray] = []
    vals_l: List[np.ndarray] = []
    b = np.zeros(nbox * ncomp)

    for corner in product((0, 1), repeat=ndim):
        e_multi = [own_multi[d] - corner[d] for d in range(ndim)]
        inside = np.ones(owned_lin.size, dtype=bool)
        for d in range(ndim):
            inside &= (e_multi[d] >= emulti[d].min()) & (e_multi[d] <= emulti[d].max())
        if not inside.any():
            continue
        row_nodes = owned_lin[inside]
        e_flat = [m[inside] for m in e_multi]
        # element-box linear index of each element
        e_lin = np.zeros(row_nodes.size, dtype=np.int64)
        for d in range(ndim):
            e_lin = e_lin + (e_flat[d] - elo[d]) * estrides[d]
        for a in product((0, 1), repeat=ndim):
            cidx = _corner(corner)
            aidx = _corner(a)
            target = [e_flat[d] + a[d] for d in range(ndim)]
            ok = np.ones(row_nodes.size, dtype=bool)
            col_lin = np.zeros(row_nodes.size, dtype=np.int64)
            for d in range(ndim):
                ok &= (target[d] >= lo[d]) & (target[d] < hi[d])
                col_lin = col_lin + (target[d] - lo[d]) * bstrides[d]
            if not ok.any():
                continue
            rr = row_nodes[ok]
            cc = col_lin[ok]
            eo = e_lin[ok]
            if ncomp == 1:
                rows_l.append(rr)
                cols_l.append(cc)
                vals_l.append(
                    np.full(rr.size, Ke_const[cidx, aidx])
                    if Ke_const is not None
                    else Ke_all[eo, cidx, aidx]
                )
                b[rr] += fe_all[eo, aidx] if a == corner else 0.0
            else:
                for ca in range(ncomp):
                    for cb in range(ncomp):
                        rows_l.append(rr * ncomp + ca)
                        cols_l.append(cc * ncomp + cb)
                        if Ke_const is not None:
                            vals_l.append(np.full(rr.size, Ke_const[cidx * ncomp + ca, aidx * ncomp + cb]))
                        else:
                            vals_l.append(Ke_all[eo, cidx * ncomp + ca, aidx * ncomp + cb])
                if a == corner:
                    for ca in range(ncomp):
                        b[rr * ncomp + ca] += fe_all[eo, aidx * ncomp + ca]

    rows = np.concatenate(rows_l) if rows_l else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(cols_l) if cols_l else np.zeros(0, dtype=np.int64)
    vals = np.concatenate(vals_l) if vals_l else np.zeros(0)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(nbox * ncomp, nbox * ncomp)).tocsr()
    A.sum_duplicates()
    return A, b


def _corner(corner: Sequence[int]) -> int:
    """Local node number of a 0/1 corner tuple, last dimension fastest.

    Must match :func:`tensorphysics.fem_parallel.discretisation.reference_corners`
    (``itertools.product`` ordering), otherwise element matrices are scattered
    with the wrong connectivity.
    """
    idx = 0
    ndim = len(corner)
    for d, c in enumerate(corner):
        idx += int(c) << (ndim - 1 - d)
    return idx


def eliminate_dirichlet(
    A: sp.csr_matrix, bc_rows: np.ndarray, bc_cols: Optional[np.ndarray] = None
) -> sp.csr_matrix:
    """Zero the Dirichlet rows/columns and put ``1`` on the ``bc_rows`` diagonal.

    ``bc_rows`` are the DOFs this rank owns and prescribes; ``bc_cols`` (default:
    the same) are the columns to remove.  In a distributed run the two differ:
    a ghost DOF on the global boundary is *not* a row of this rank, but its
    column must still be removed -- otherwise ``A @ x`` would add the spurious
    contribution ``A[i, g] * g`` of the exchanged (non-zero) Dirichlet value and
    the Krylov iteration would converge against the wrong operator.

    ``O(nnz)`` numpy operations on the CSR arrays (no LIL round-trip), which
    matters at ``10**6`` DOFs where the LIL path is two decades slower.
    """
    bc_rows = np.asarray(bc_rows, dtype=bool)
    bc_cols = bc_rows if bc_cols is None else np.asarray(bc_cols, dtype=bool)
    if not bc_rows.any() and not bc_cols.any():
        return A
    A = A.tocsr(copy=True)
    n = A.shape[0]
    free_rows = ~bc_rows
    free_cols = ~bc_cols
    row_of = np.repeat(np.arange(n, dtype=np.int64), np.diff(A.indptr))
    keep = free_rows[row_of] & free_cols[A.indices]
    A.data[~keep] = 0.0
    A.eliminate_zeros()
    idx = np.nonzero(bc_rows)[0]
    if idx.size == 0:
        return A
    diag = sp.coo_matrix((np.ones(idx.size), (idx, idx)), shape=A.shape).tocsr()
    return (A + diag).tocsr()
