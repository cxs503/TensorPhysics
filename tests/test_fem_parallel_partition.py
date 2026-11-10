"""Domain-decomposition unit tests: Cartesian blocking and graph partitioning."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.fem_parallel.partition import (
    CartesianPartition,
    balance_ratio,
    choose_factorisation,
    edge_cut,
    metis_available,
    partition_graph,
    recursive_bisection,
    refine_partition,
)


def test_cartesian_partition_covers_every_node_exactly_once():
    grid_n = (13, 11)
    part = CartesianPartition(grid_n, 4)
    seen = np.zeros(int(np.prod(grid_n)), dtype=np.int64)
    for r in range(4):
        sl = part.block_slices(r)
        for i in range(sl[0].start, sl[0].stop):
            for j in range(sl[1].start, sl[1].stop):
                seen[i + grid_n[0] * j] += 1
    assert seen.min() == 1 and seen.max() == 1
    assert sum(part.block_volume(r) for r in range(4)) == int(np.prod(grid_n))


def test_cartesian_partition_is_balanced_for_even_split():
    part = CartesianPartition((100, 100), 16)
    assert part.shape == (4, 4)
    assert part.max_imbalance() == pytest.approx(0.0, abs=1e-12)


def test_cartesian_partition_handles_indivisible_domain():
    part = CartesianPartition((101, 97), 5)
    counts = part.owned_counts()
    assert counts.sum() == 101 * 97
    # at most one node of imbalance between the largest and smallest block
    assert counts.max() - counts.min() <= max(101, 97)
    assert part.max_imbalance() < 0.05


def test_box_is_block_grown_by_one_layer_and_clipped():
    part = CartesianPartition((10, 10), 4)
    for r in range(4):
        blk, box = part.block_slices(r), part.box_slices(r)
        for b, x in zip(blk, box):
            assert x.start == max(b.start - 1, 0)
            assert x.stop == min(b.stop + 1, 10)


def test_choose_factorisation_balances_volume_then_interface():
    assert choose_factorisation(16, 2) == (4, 4)
    assert choose_factorisation(32, 3) == (2, 4, 4)
    assert np.prod(choose_factorisation(6, 2)) == 6


def _polar_graph(n_side=9):
    """Unstructured point set + Delaunay-like connectivity for the partitioner."""
    xs = np.linspace(0.0, 1.0, n_side)
    coords = np.array([[x, y] for x in xs for y in xs])
    edges = []
    for i in range(n_side):
        for j in range(n_side):
            k = i * n_side + j
            if i + 1 < n_side:
                edges.append((k, (i + 1) * n_side + j))
            if j + 1 < n_side:
                edges.append((k, i * n_side + j + 1))
    return coords, np.array(edges)


def test_recursive_bisection_is_complete_and_balanced():
    coords, _ = _polar_graph()
    labels = recursive_bisection(coords, 4)
    assert labels.min() == 0 and labels.max() == 3
    assert balance_ratio(labels, 4) <= 1.15


def test_graph_partition_reduces_edge_cut_against_naive_column_split():
    coords, edges = _polar_graph(12)
    labels = partition_graph(coords, edges, 4, use_metis=metis_available())
    naive = (np.arange(coords.shape[0]) % 4)  # interleaved, worst case
    assert edge_cut(edges, labels) < edge_cut(edges, naive) / 2.0
    assert balance_ratio(labels, 4) <= 1.2


def test_refinement_never_increases_edge_cut():
    coords, edges = _polar_graph(10)
    labels = recursive_bisection(coords, 4, adjacency=None)
    before = edge_cut(edges, labels)
    from tensorphysics.fem_parallel.partition import _graph_adjacency

    adj = _graph_adjacency(edges, coords.shape[0])
    refined = refine_partition(adj, labels, 4)
    assert edge_cut(edges, refined) <= before


def test_partition_rank_lookup_matches_block_slices():
    part = CartesianPartition((17, 13), 6)
    for r in range(6):
        sl = part.block_slices(r)
        i = (sl[0].start + sl[0].stop) // 2
        j = (sl[1].start + sl[1].stop) // 2
        assert part.block_rank((i, j)) == r
        assert int(part.rank_of_index([np.array([i]), np.array([j])])[0]) == r


def test_partition_refuses_to_create_empty_subdomains():
    """A rank with no nodes would only sit in the collectives: refuse it."""
    part = CartesianPartition((7, 4), 5)
    assert part.shape == (5, 1)          # split the 7-node axis instead
    assert part.owned_counts().min() > 0
    with pytest.raises(ValueError, match="too coarse"):
        CartesianPartition((2, 2), 5)
