"""Domain decomposition: structured Cartesian blocking and graph partitioning.

Two partitioners are provided, matching the two mesh families of the campaign
node:

``CartesianPartition``
    Tensor-product blocking of a structured grid.  The number of ranks is
    factorised over the ``ndim`` dimensions and the factorisation minimising the
    maximum block volume (load balance) and then the interface area
    (communication volume) is selected.  Block bounds are *contiguous*
    ranges per dimension, which is the standard MPI Cartesian topology.
``graph_partition`` / ``recursive_bisection``
    Unstructured-mesh partitioning on the node graph.  ``pymetis`` is used when
    installed; otherwise a genuine recursive coordinate bisection (RCB) with a
    boundary-node Kernighan--Lin style refinement is used, so the partitioner
    works without the METIS binary.  The edge cut of a partition is computed by
    :func:`edge_cut`, so METIS-quality comparisons stay measurable either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CartesianPartition",
    "choose_factorisation",
    "partition_graph",
    "recursive_bisection",
    "refine_partition",
    "edge_cut",
    "balance_ratio",
    "metis_available",
]


def metis_available() -> bool:
    try:
        import pymetis  # noqa: F401
    except Exception:
        return False
    return True


def choose_factorisation(
    nranks: int, ndim: int, weights: Sequence[float] = None, n_per_dim: Sequence[int] = None
):
    """Factorise ``nranks`` over ``ndim`` dimensions.

    Objective: minimise the largest block volume first (load balance), then the
    total interface area (communication volume) with the block dimensions
    ``w_d / n_d``:

        area ~ sum_d prod_{e != d} (w_e / n_e)

    For a cubic domain the surface/volume heuristic prefers a shape as close to
    isotropic as possible, which is what METIS/ParMETIS also produce for a box.
    """
    weights = tuple(float(w) for w in (weights or [1.0] * ndim))
    limits = tuple(int(v) for v in n_per_dim) if n_per_dim is not None else None
    best = None
    for factors in product(range(1, nranks + 1), repeat=ndim):
        prod = 1
        for f in factors:
            prod *= f
        if prod != nranks:
            continue
        # never create a block with zero nodes: that rank would own nothing and
        # only participate in the collectives (a silent performance and
        # robustness trap, and it makes an "empty subdomain" run possible)
        if limits is not None and any(factors[d] > limits[d] for d in range(ndim)):
            continue
        block = [weights[d] / factors[d] for d in range(ndim)]
        vol = float(np.prod(block))
        # interface area in units of the (unscaled) domain measure
        area = 0.0
        for d in range(ndim):
            other = 1.0
            for e in range(ndim):
                if e != d:
                    other *= block[e]
            area += other
        key = (vol, area)
        if best is None or key < best[0]:
            best = (key, factors)
    if best is None:
        raise ValueError(
            f"cannot factorise nranks={nranks} over {ndim} dims with "
            f"n_per_dim={limits}: the grid is too coarse for that many ranks "
            "(at least one subdomain would be empty)"
        )
    return tuple(int(f) for f in best[1])


def _split_sizes(n: int, parts: int) -> np.ndarray:
    """Split ``n`` contiguous items over ``parts`` blocks as evenly as possible."""
    base, extra = divmod(int(n), int(parts))
    sizes = np.full(parts, base, dtype=np.int64)
    sizes[:extra] += 1
    return sizes


@dataclass
class CartesianPartition:
    """Contiguous block partition of a structured grid over ``nranks`` ranks."""

    n: Tuple[int, ...]
    nranks: int

    def __post_init__(self) -> None:
        self.n = tuple(int(v) for v in self.n)
        self.ndim = len(self.n)
        self.shape = choose_factorisation(self.nranks, self.ndim, n_per_dim=self.n)
        self.sizes: List[np.ndarray] = [
            _split_sizes(self.n[d], self.shape[d]) for d in range(self.ndim)
        ]
        self.bounds: List[np.ndarray] = []
        for d in range(self.ndim):
            b = np.zeros(self.shape[d] + 1, dtype=np.int64)
            b[1:] = np.cumsum(self.sizes[d])
            self.bounds.append(b)

    # -- rank <-> block ----------------------------------------------------
    def rank_of_index(self, indices) -> np.ndarray:  # list of arrays per dim
        rank = np.zeros(np.shape(indices[0]), dtype=np.int64)
        stride = 1
        for d in range(self.ndim):
            loc = np.searchsorted(self.bounds[d], indices[d], side="right") - 1
            loc = np.clip(loc, 0, self.shape[d] - 1)
            rank += loc * stride
            stride *= self.shape[d]
        return rank

    def block_rank(self, coords: Sequence[int]) -> int:
        rank, stride = 0, 1
        for d in range(self.ndim):
            rank += int(np.searchsorted(self.bounds[d], coords[d], side="right") - 1) * stride
            stride *= self.shape[d]
        return rank

    def block_slices(self, rank: int) -> Tuple[slice, ...]:
        coords = self.block_coords(rank)
        return tuple(
            slice(int(self.bounds[d][coords[d]]), int(self.bounds[d][coords[d] + 1]))
            for d in range(self.ndim)
        )

    def block_coords(self, rank: int) -> Tuple[int, ...]:
        coords = []
        for d in range(self.ndim):
            coords.append(rank % self.shape[d])
            rank //= self.shape[d]
        return tuple(coords)

    def box_slices(self, rank: int) -> Tuple[slice, ...]:
        """Block expanded by one node in every direction (the halo layer)."""
        out = []
        for d, sl in enumerate(self.block_slices(rank)):
            lo = max(int(sl.start) - 1, 0)
            hi = min(int(sl.stop) + 1, self.n[d])
            out.append(slice(lo, hi))
        return tuple(out)

    def block_volume(self, rank: int) -> int:
        v = 1
        for sl in self.block_slices(rank):
            v *= int(sl.stop) - int(sl.start)
        return v

    def owned_counts(self) -> np.ndarray:
        return np.array([self.block_volume(r) for r in range(self.nranks)], dtype=np.int64)

    def max_imbalance(self) -> float:
        counts = self.owned_counts().astype(float)
        return float(counts.max() / counts.mean() - 1.0)


def adjacency_offsets(ndim: int) -> List[Tuple[int, ...]]:
    """The ``2*ndim`` axis neighbours of a node in a tensor-product grid."""
    out = []
    for d in range(ndim):
        for s in (-1, 1):
            off = [0] * ndim
            off[d] = s
            out.append(tuple(off))
    return out


def refine_partition(
    adjacency: Sequence[Sequence[int]],
    labels: np.ndarray,
    nparts: int,
    n_sweeps: int = 6,
) -> np.ndarray:
    """Kernighan--Lin style boundary refinement of a graph partition.

    A boundary vertex moves to another part when that strictly reduces the
    number of cut edges connected to it, subject to keeping the part sizes
    within one vertex of the target.  Deterministic, ``O(sweeps * boundary)``.
    """
    labels = np.array(labels, copy=True)
    n = labels.size
    target = n / float(nparts)
    for _ in range(int(n_sweeps)):
        moved = 0
        sizes = np.bincount(labels, minlength=nparts)
        for v in range(n):
            own = labels[v]
            neigh = np.asarray(adjacency[v], dtype=np.int64)
            if neigh.size == 0:
                continue
            other = labels[neigh]
            # gain of moving v to part p: (#edges to p) - (#edges to own)
            cut_to_own = int(np.sum(other == own) - 0)  # self-loops absent
            if np.all(other == own):
                continue  # interior vertex, never move
            for p in np.unique(other):
                if p == own:
                    continue
                gain = int(np.sum(other == p)) - cut_to_own
                if gain <= 0:
                    continue
                if sizes[p] + 1 > target + 1.0 or sizes[own] - 1 < target - 1.0:
                    continue
                labels[v] = p
                sizes[own] -= 1
                sizes[p] += 1
                moved += 1
                break
        if moved == 0:
            break
    return labels


def recursive_bisection(
    coords: np.ndarray, nparts: int, adjacency: Optional[Sequence[Sequence[int]]] = None
) -> np.ndarray:
    """Recursive coordinate bisection on ``coords`` (``n x ndim``)."""
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    nparts = int(nparts)
    labels = np.zeros(n, dtype=np.int64)

    def split(idx: np.ndarray, part_lo: int, part_hi: int) -> None:
        nparts_local = part_hi - part_lo
        if nparts_local <= 1 or idx.size == 0:
            labels[idx] = part_lo
            return
        pts = coords[idx]
        spread = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(spread))
        order = np.argsort(pts[:, axis], kind="stable")
        left_parts = max(1, nparts_local // 2)
        right_parts = nparts_local - left_parts
        if right_parts <= 0:  # pragma: no cover
            labels[idx] = part_lo
            return
        # split points proportionally to the number of parts on each side
        cut = int(round(idx.size * left_parts / float(nparts_local)))
        cut = min(max(cut, 1), idx.size - 1)
        left = idx[order[:cut]]
        right = idx[order[cut:]]
        split(left, part_lo, part_lo + left_parts)
        split(right, part_lo + left_parts, part_hi)

    split(np.arange(n), 0, nparts)
    if adjacency is not None:
        labels = refine_partition(adjacency, labels, nparts)
    return labels


def _graph_adjacency(edges: np.ndarray, n: int) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    return adj


def partition_graph(
    coords: np.ndarray,
    edges: np.ndarray,
    nparts: int,
    use_metis: bool = True,
    adj_list: bool = False,
):
    """Partition the node graph of an unstructured mesh.

    Returns integer labels ``0..nparts-1`` (plus the METIS adjacency list when
    ``adj_list``).  ``pymetis`` is used when importable, otherwise RCB with KL
    refinement.  ``edges`` has shape ``(m, 2)``.
    """
    coords = np.asarray(coords, dtype=float)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n = coords.shape[0]
    adj = _graph_adjacency(edges, n)
    nparts = int(nparts)
    if use_metis and metis_available():  # pragma: no cover - optional dependency
        import pymetis  # noqa: PLC0415 - optional dependency

        _, membership = pymetis.part_graph(nparts, adjacency=adj)
        labels = np.asarray(membership, dtype=np.int64)
    else:
        labels = recursive_bisection(coords, nparts, adjacency=None)
        labels = refine_partition(adj, labels, nparts)
    if adj_list:
        return labels, adj
    return labels


def edge_cut(edges: np.ndarray, labels: np.ndarray) -> int:
    """Number of graph edges whose endpoints sit in different parts."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    labels = np.asarray(labels)
    return int(np.count_nonzero(labels[edges[:, 0]] != labels[edges[:, 1]]))


def balance_ratio(labels: np.ndarray, nparts: int) -> float:
    """``max(part size) / mean(part size)`` -- 1.0 is a perfectly balanced part."""
    sizes = np.bincount(np.asarray(labels), minlength=int(nparts)).astype(float)
    if sizes.mean() == 0:  # pragma: no cover
        return float("inf")
    return float(sizes.max() / sizes.mean())
