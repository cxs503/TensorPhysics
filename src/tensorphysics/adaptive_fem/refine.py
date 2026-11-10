"""Conforming red-green h-refinement and guarded Delaunay edge flipping."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .mesh2d import TriMesh, edge_records, signed_areas, triangle_angles

__all__ = ["refine_red_green", "refine_longest_edge", "flip_to_quality"]


def _add(parent: list, new: np.ndarray) -> None:
    if new.size:
        parent.append(new)


def refine_red_green(
    mesh: TriMesh,
    marked: Iterable[int],
    *,
    update_meta: dict | None = None,
    project=None,
) -> TriMesh:
    """Conforming red-green h-refinement of ``mesh`` on the triangles ``marked``.

    Red: a marked triangle is split 1→4 by its three edge midpoints.
    Green: an unmarked triangle that received ``k`` bisected edges is closed by
    a 1→2 split (``k == 1``, midpoint joined to the opposite vertex) or a 1→3
    split (``k == 2``); ``k == 3`` degenerates to a red split.  Midpoints are
    shared between neighbours, so the output is conforming by construction
    (no hanging nodes).  Green children are not tagged, so a later refinement
    pass simply refines them further — the scheme stays conforming.
    """
    tris = mesh.tris
    nodes = mesh.nodes
    m = len(tris)
    marked = np.asarray(sorted({int(i) for i in marked}), dtype=np.int64)
    if marked.size and (marked.min() < 0 or marked.max() >= m):
        raise ValueError("marked triangle index out of range")
    if marked.size == 0:
        out = mesh.copy()
        if update_meta:
            out.meta.update(update_meta)
        return out

    uniq, counts, inc = edge_records(tris)
    tri_marked = np.zeros(m, dtype=bool)
    tri_marked[marked] = True

    bisect_edge = np.zeros(len(uniq), dtype=bool)
    bisect_edge[inc[:, tri_marked].ravel()] = True

    mid = np.full(len(uniq), -1, dtype=np.int64)
    idx = np.flatnonzero(bisect_edge)
    n0 = len(nodes)
    mid[idx] = n0 + np.arange(len(idx))
    project = project if project is not None else mesh.meta.get("project")
    pa = nodes[uniq[idx, 0]]
    pb = nodes[uniq[idx, 1]]
    new_nodes = 0.5 * (pa + pb) if project is None else np.asarray(project(pa, pb), float)
    all_nodes = np.vstack([nodes, new_nodes]) if len(idx) else nodes.copy()

    has = bisect_edge[inc]  # (3, m) bool: local edge j of triangle i bisected?
    nbis = has.sum(axis=0)

    out: list[np.ndarray] = []

    # -- untouched ----------------------------------------------------------
    _add(out, tris[nbis == 0])

    # -- red: all three edges bisected -------------------------------------
    red = np.flatnonzero(nbis == 3)
    if red.size:
        v0, v1, v2 = tris[red, 0], tris[red, 1], tris[red, 2]
        m0 = mid[inc[0, red]]
        m1 = mid[inc[1, red]]
        m2 = mid[inc[2, red]]
        _add(out, np.stack([v0, m0, m2], axis=1))
        _add(out, np.stack([m0, v1, m1], axis=1))
        _add(out, np.stack([m2, m1, v2], axis=1))
        _add(out, np.stack([m0, m1, m2], axis=1))

    # -- green 1→2: exactly one bisected local edge ------------------------
    one = np.flatnonzero(nbis == 1)
    if one.size:
        kid = np.argmax(has[:, one], axis=0)  # bisected local edge id per triangle
        for kk in range(3):
            sel = one[kid == kk]
            if not sel.size:
                continue
            rot = tris[sel][:, [kk, (kk + 1) % 3, (kk + 2) % 3]]
            v0, v1, v2 = rot[:, 0], rot[:, 1], rot[:, 2]
            mm = mid[inc[kk, sel]]
            _add(out, np.stack([v0, mm, v2], axis=1))
            _add(out, np.stack([mm, v1, v2], axis=1))

    # -- green 1→3: two bisected local edges -------------------------------
    two = np.flatnonzero(nbis == 2)
    if two.size:
        k1 = np.argmax(has[:, two], axis=0)
        k2 = np.argmax(has[:, two] & (np.arange(3)[:, None] != k1[None, :]), axis=0)
        # vertex shared by local edges k1 and k2
        shared = np.where(k2 == (k1 + 1) % 3, k2, k1)
        for j in range(3):
            sel = two[shared == j]
            if not sel.size:
                continue
            # rotate so the shared vertex sits at local position 1 -> the two
            # bisected edges become local edges 0 = (a,b) and 1 = (b,c)
            rot = tris[sel][:, [(j + 2) % 3, j, (j + 1) % 3]]
            a, b, c = rot[:, 0], rot[:, 1], rot[:, 2]
            e0 = _local_edge_id(tris[sel], inc, sel, a, b)
            e1 = _local_edge_id(tris[sel], inc, sel, b, c)
            m_ab = mid[e0]
            m_bc = mid[e1]
            _add(out, np.stack([a, m_ab, m_bc], axis=1))
            _add(out, np.stack([m_ab, b, m_bc], axis=1))
            _add(out, np.stack([a, m_bc, c], axis=1))

    new_tris = np.vstack(out) if out else np.zeros((0, 3), dtype=np.int64)
    meta = dict(mesh.meta)
    if update_meta:
        meta.update(update_meta)
    meta["last_n_marked"] = int(marked.size)
    out_mesh = TriMesh(all_nodes, new_tris, generation=mesh.generation + 1, meta=meta)
    _assert_valid(out_mesh)
    return out_mesh


def _local_edge_id(
    tris_sel: np.ndarray, inc: np.ndarray, sel: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Unique-edge id of node pair ``(a, b)`` for triangles ``tris_sel = tris[sel]``."""
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    ids = np.empty(len(a), dtype=np.int64)
    for j in range(3):
        cand = np.sort(tris_sel[:, [j, (j + 1) % 3]], axis=1)
        hit = np.flatnonzero((cand[:, 0] == lo) & (cand[:, 1] == hi))
        ids[hit] = inc[j, sel[hit]]
    return ids


def _assert_valid(mesh: TriMesh) -> None:
    if mesh.n_tris == 0:
        return
    area = signed_areas(mesh.nodes, mesh.tris)
    if np.any(area <= 0.0):
        raise RuntimeError("refinement produced non-positive / non-CCW triangles")


def flip_to_quality(mesh: TriMesh, *, max_passes: int = 6, tol: float = 1e-10) -> TriMesh:
    """Guarded Delaunay flipping.

    An interior edge is swapped only if the swap strictly raises the minimum of
    the six involved angles, so quality is monotone non-decreasing, the domain
    is preserved, and node coordinates never change.  Used to pull green-closure
    angles away from degeneracy after h-refinement.
    """
    nodes = mesh.nodes
    tris = mesh.tris.copy()
    total_flips = 0
    for _ in range(max_passes):
        tris, n_flip = _one_flip_pass(nodes, tris, tol)
        total_flips += n_flip
        if n_flip == 0:
            break
    meta = dict(mesh.meta)
    meta["flips"] = int(total_flips)
    out = TriMesh(nodes, tris, generation=mesh.generation, meta=meta)
    _assert_valid(out)
    return out


def _signed_area(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a = p[tri[:, 0]]
    b = p[tri[:, 1]]
    c = p[tri[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def _min_angle_pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Minimum interior angle (deg) of triangles given by point arrays (k,2)."""

    def _ang(p, q, r):
        u = p - q
        w = r - q
        nu = np.linalg.norm(u, axis=1)
        nw = np.linalg.norm(w, axis=1)
        cosv = np.clip(np.sum(u * w, axis=1) / np.maximum(nu * nw, 1e-300), -1.0, 1.0)
        return np.degrees(np.arccos(cosv))

    return np.minimum(np.minimum(_ang(a, b, c), _ang(b, c, a)), _ang(c, a, b))


def _one_flip_pass(nodes: np.ndarray, tris: np.ndarray, tol: float):
    """One vectorised sweep of candidate edge swaps; returns (new_tris, n_flips)."""
    m = len(tris)
    uniq, counts, inc = edge_records(tris)
    cur_min = triangle_angles(nodes, tris).min(axis=1)
    interior = np.flatnonzero(counts == 2)
    if interior.size == 0:
        return tris, 0

    eid = inc.ravel()
    tid = np.tile(np.arange(m), 3)  # matches inc.ravel() row-major over (3, m)
    order = np.argsort(eid, kind="stable")
    eid_s, tid_s = eid[order], tid[order]
    starts = np.searchsorted(eid_s, interior)
    t1 = tid_s[starts]
    t2 = tid_s[starts + 1]

    u = uniq[interior, 0]
    v = uniq[interior, 1]
    o1 = tris[t1].sum(axis=1) - u - v
    o2 = tris[t2].sum(axis=1) - u - v

    p = nodes
    # correct CCW ordering of the two candidate triangles for the flipped quad
    newA = np.stack([u, o2, o1], axis=1)
    newB = np.stack([o2, v, o1], axis=1)
    areaA = _signed_area(p, newA)
    areaB = _signed_area(p, newB)
    # the quad must be strictly convex, otherwise the swap would create
    # overlapping triangles (a non-convex quad has one non-positive area)
    valid = (areaA > 0.0) & (areaB > 0.0)
    if not np.any(valid):
        return tris, 0

    minA = _min_angle_pts(p[newA[:, 0]], p[newA[:, 1]], p[newA[:, 2]])
    minB = _min_angle_pts(p[newB[:, 0]], p[newB[:, 1]], p[newB[:, 2]])
    best_before = np.minimum(cur_min[t1], cur_min[t2])
    cand = np.flatnonzero(valid & (np.minimum(minA, minB) > best_before + tol))
    if cand.size == 0:
        return tris, 0

    locked = np.zeros(m, dtype=bool)
    out = tris.copy()
    n_flip = 0
    for c in cand:  # small list, greedy conflict resolution
        a1, a2 = t1[c], t2[c]
        if locked[a1] or locked[a2]:
            continue
        out[a1] = newA[c]
        out[a2] = newB[c]
        locked[a1] = locked[a2] = True
        n_flip += 1
    return out, n_flip


# ---------------------------------------------------------------------------
# Rivara longest-edge bisection (quality-preserving alternative to green closure)
# ---------------------------------------------------------------------------
def refine_longest_edge(mesh: TriMesh, marked, *, max_depth: int = 400, project=None) -> TriMesh:
    """Conforming longest-edge (Rivara) bisection of the ``marked`` triangles.

    Every marked triangle bisects its own longest edge; whenever the bisected
    edge is *not* the longest edge of the neighbouring triangle, the neighbour
    is bisected first (longest-edge propagation path), so the closure never
    creates the skewed slivers that a fixed green closure can produce.  The
    minimum angle of a Rivara-refined mesh stays bounded below by half of the
    initial minimum angle, which is what the 20 deg quality criterion needs.
    """
    nodes: list[tuple[float, float]] = [(float(p[0]), float(p[1])) for p in mesh.nodes]
    tris: list[tuple[int, int, int] | None] = [None] * len(mesh.tris)
    edge2tris: dict[tuple[int, int], set[int]] = {}
    next_id = [len(mesh.tris)]

    def add_tri(tid: int, t: tuple[int, int, int]) -> None:
        if tid == len(tris):
            tris.append(t)
        else:
            tris[tid] = t
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            key = (a, b) if a < b else (b, a)
            edge2tris.setdefault(key, set()).add(tid)

    def remove_tri(tid: int) -> None:
        t = tris[tid]
        assert t is not None
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            key = (a, b) if a < b else (b, a)
            bucket = edge2tris.get(key)
            if bucket is not None:
                bucket.discard(tid)
        tris[tid] = None

    for tid, t in enumerate(np.asarray(mesh.tris, dtype=np.int64)):
        add_tri(tid, (int(t[0]), int(t[1]), int(t[2])))

    mid_cache: dict[tuple[int, int], int] = {}

    def longer_edge(t: tuple[int, int, int]) -> tuple[tuple[int, int], float]:
        best_key = None
        best_len = -1.0
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            pa, pb = nodes[a], nodes[b]
            L = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
            if L > best_len:
                best_len = L
                best_key = (a, b) if a < b else (b, a)
        assert best_key is not None
        return best_key, best_len

    proj = project if project is not None else mesh.meta.get("project")

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        mid = mid_cache.get(key)
        if mid is None:
            pa, pb = nodes[a], nodes[b]
            if proj is None:
                xy = (0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]))
            else:
                q = np.asarray(proj(np.array([pa], float), np.array([pb], float)), float)[0]
                xy = (float(q[0]), float(q[1]))
            mid = len(nodes)
            nodes.append(xy)
            mid_cache[key] = mid
        return mid

    def split_on_edge(tid: int, key: tuple[int, int]) -> None:
        t = tris[tid]
        assert t is not None
        k = -1
        for j in range(3):
            a, b = t[j], t[(j + 1) % 3]
            if (a, b) == key or (b, a) == key:
                k = j
                break
        if k < 0:
            raise RuntimeError("edge not found in triangle")
        u, v, w = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
        m = midpoint(u, v)
        remove_tri(tid)
        t1 = next_id[0]
        add_tri(t1, (u, m, w))
        t2 = next_id[0] + 1
        add_tri(t2, (m, v, w))
        next_id[0] += 2

    def refine_tri(tid: int, depth: int) -> None:
        if depth > max_depth:
            raise RuntimeError("longest-edge propagation exceeded max_depth")
        t = tris[tid]
        if t is None:
            return
        key, _ = longer_edge(t)
        others = [o for o in edge2tris.get(key, ()) if o != tid and tris[o] is not None]
        if not others:
            split_on_edge(tid, key)
            return
        o = others[0]
        key_o, _ = longer_edge(tris[o])
        if key_o != key:
            refine_tri(o, depth + 1)
            refine_tri(tid, depth + 1)
            return
        split_on_edge(tid, key)
        split_on_edge(o, key)

    for tid in sorted({int(i) for i in marked}):
        if 0 <= tid < len(tris):
            refine_tri(tid, 0)

    kept = [(tid, t) for tid, t in enumerate(tris) if t is not None]
    new_tris = np.array([t for _, t in kept], dtype=np.int64)
    meta = dict(mesh.meta)
    meta["last_n_marked"] = int(len(set(int(i) for i in marked)))
    meta["scheme"] = "rivara_longest_edge"
    out = TriMesh(np.array(nodes, dtype=float), new_tris, generation=mesh.generation + 1, meta=meta)
    _assert_valid(out)
    return out
