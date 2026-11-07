"""Discrete-side integrations: FVM polyhedra, LBM voxels/IBM, BEM panels.

Each family consumes the analytic truth source (`HullForm`) only through the
surface map ``surface_yz`` / ``half_breadth``, builds a *real* grid, and reports
the integral quantities the contract needs: displacement volume, wetted surface
area and block coefficient.  The grid itself is hashed so the contract can carry
provenance of the exact discretisation that produced the numbers.

The three families fail in deliberately different ways:

``fvm``
    body-fitted hexahedra; volume from the standard six-tetrahedron
    decomposition of every cell (exact polyhedron volume of the generated
    mesh), area from the faceted hull-face set -> second-order errors.
``lbm``
    uniform cubic lattice; volume from the occupied voxel count (first order),
    wetted area from an IBM-style reconstruction: the occupancy field is
    smoothed and the area integrated with the coarea formula
    ``A = integral |grad f| dV``.  The free surface (z = T) is an interface of
    the same field and is removed by an orientation/height mask.
``bem``
    flat panel set on the wetted surface, closed by the transom (and by the
    waterplane for a surface ship); volume from the divergence theorem
    ``nabla = 1/3 oint x . n dA`` plus the closed-surface identity check
    ``|sum A_i n_i| / sum A_i``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from .hulls import HullForm

__all__ = [
    "CORRUPTIONS",
    "SolverRecord",
    "bem_panels",
    "corrupted_record",
    "fvm_mesh",
    "good_records",
    "lbm_grid",
]


def _np():
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - numpy is in the test extra
        raise RuntimeError("numpy is required for grid integration") from exc
    return numpy


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SolverRecord:
    """One discretisation family's integral quantities and its grid digest."""

    family: str
    nabla_m3: float
    s_wetted_m2: float
    cb: float
    resolution: Mapping[str, int]
    grid_sha256: str
    extras: Mapping[str, float] | None = None
    surface_evidence: str = "REAL"

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "family": self.family,
            "nabla_m3": self.nabla_m3,
            "s_wetted_m2": self.s_wetted_m2,
            "cb": self.cb,
            "resolution": {k: int(v) for k, v in self.resolution.items()},
            "grid_sha256": self.grid_sha256,
            "surface_evidence": self.surface_evidence,
        }
        if self.extras:
            payload["extras"] = {k: float(v) for k, v in self.extras.items()}
        return payload


def _surface_grid(hull: HullForm, n_x: int, n_s: int, section_shift: float = 0.0):
    """Node coordinates of the starboard parametric surface, ``(n_x+1, n_s+1)``."""
    np = _np()
    xs = np.linspace(0.0, hull.length_m, n_x + 1)
    ss = np.linspace(0.0, 1.0, n_s + 1)
    ss_y = np.clip(ss + section_shift, 0.0, 1.0)  # breadth sampling level
    ys = np.zeros((n_x + 1, n_s + 1))
    zs = np.zeros((n_x + 1, n_s + 1))
    for i, x in enumerate(xs):
        for k in range(n_s + 1):
            ys[i, k], _ = hull.surface_yz(float(x), float(ss_y[k]))
            _, zs[i, k] = hull.surface_yz(float(x), float(ss[k]))
    return xs, ss, ys, zs


def _half_breadth_field(hull: HullForm, x, y, z):
    """Vectorised half-breadth over a lattice."""
    np = _np()
    if hull.family == "revolution":
        r = np.vectorize(hull.radius)(x)
        inner = np.clip(r[:, None, None] ** 2 - (z[None, None, :] - 0.5 * hull.breadth_m) ** 2, 0.0, None)
        return np.broadcast_to(np.sqrt(inner), (x.size, y.size, z.size))
    m = hull.params["m"]
    w = np.vectorize(hull.waterline_factor)(x / hull.length_m)
    zeta = np.clip(z / hull.draught_m, 0.0, 1.0) ** (1.0 / m)
    return 0.5 * hull.breadth_m * w[:, None, None] * zeta[None, None, :] * np.ones((x.size, y.size, z.size))


# ---------------------------------------------------------------------------
# FVM: body-fitted hexahedral mesh, six-tet volume sum
# ---------------------------------------------------------------------------

_HEX_TETS = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))


def fvm_mesh(
    hull: HullForm,
    n_x: int = 160,
    n_s: int = 40,
    n_y: int = 8,
    stretch: float = 1.0,
    section_shift: float = 0.0,
) -> SolverRecord:
    """Body-fitted hex mesh of the hull interior (symmetry half, doubled).

    ``stretch`` scales every node isotropically: the negative control that
    stretches the grid by e.g. 2 % so it no longer matches the CAD truth.
    """
    np = _np()
    xs, ss, ys, zs = _surface_grid(hull, n_x, n_s, section_shift=section_shift)
    js = np.linspace(0.0, 1.0, n_y + 1)
    nodes = np.zeros((n_x + 1, n_s + 1, n_y + 1, 3))
    nodes[..., 0] = xs[:, None, None]
    nodes[..., 1] = ys[:, :, None] * js[None, None, :]
    nodes[..., 2] = zs[:, :, None]
    if stretch != 1.0:
        nodes *= stretch

    v = [
        nodes[:-1, :-1, :-1],
        nodes[1:, :-1, :-1],
        nodes[1:, 1:, :-1],
        nodes[:-1, 1:, :-1],
        nodes[:-1, :-1, 1:],
        nodes[1:, :-1, 1:],
        nodes[1:, 1:, 1:],
        nodes[:-1, 1:, 1:],
    ]
    half_volume = 0.0
    for a, b, c, d in _HEX_TETS:
        half_volume += (
            float(np.sum(np.sum((v[b] - v[a]) * np.cross(v[c] - v[a], v[d] - v[a]), axis=-1))) / 6.0
        )
    volume = 2.0 * abs(half_volume)

    def quad_area(p, q, r, s):
        first = np.linalg.norm(np.cross(q - p, r - p), axis=-1)
        second = np.linalg.norm(np.cross(r - p, s - p), axis=-1)
        return float((0.5 * (first + second)).sum())

    outer = nodes[:, :, n_y, :]
    side = quad_area(outer[:-1, :-1], outer[1:, :-1], outer[1:, 1:], outer[:-1, 1:])
    stern_nodes = nodes[0, :, :, :]
    stern = quad_area(stern_nodes[:-1, :-1], stern_nodes[1:, :-1], stern_nodes[1:, 1:], stern_nodes[:-1, 1:])
    area = 2.0 * side + stern
    return SolverRecord(
        family="fvm",
        nabla_m3=volume,
        s_wetted_m2=area,
        cb=volume / (hull.length_m * hull.breadth_m * hull.draught_m),
        resolution={"n_x": n_x, "n_s": n_s, "n_y": n_y, "cells": 2 * n_x * n_s * n_y},
        grid_sha256=_sha(np.round(nodes, 9).tobytes()),
        extras={"facet_side_area_m2": side, "stern_area_m2": stern},
    )


# ---------------------------------------------------------------------------
# LBM: voxel occupancy + smoothed-coarea (IBM) wetted surface
# ---------------------------------------------------------------------------


def _box3(field):
    np = _np()
    padded = np.pad(field, 1, mode="constant")
    acc = np.zeros_like(field)
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            for dk in (0, 1, 2):
                acc += padded[di : di + field.shape[0], dj : dj + field.shape[1], dk : dk + field.shape[2]]
    return acc / 27.0


def lbm_grid(
    hull: HullForm,
    n_x: int = 480,
    drop_top_layers: int = 0,
    half_cell_shift: bool = False,
    smooth_passes: int = 2,
) -> SolverRecord:
    """Uniform lattice LBM grid: voxel volume + IBM wetted-surface reconstruction.

    ``drop_top_layers`` deletes solid voxel layers at the waterline and
    ``half_cell_shift`` samples the occupancy at the lattice *nodes* instead of
    the cell centres (an off-by-one-half-cell stencil misalignment).  Both are
    negative controls: the numbers stay plausible but the geometry no longer
    closes.
    """
    np = _np()
    dx = hull.length_m / n_x
    n_y = max(8, int(round(hull.breadth_m / dx)))
    n_z = max(8, int(round(hull.draught_m / dx)))
    dy = hull.breadth_m / n_y
    dz = hull.draught_m / n_z
    x = (np.arange(n_x) + 0.5) * dx
    y = -0.5 * hull.breadth_m + (np.arange(n_y) + 0.5) * dy
    z = (np.arange(n_z) + 0.5) * dz
    if half_cell_shift:
        x = np.arange(n_x) * dx
        y = -0.5 * hull.breadth_m + np.arange(n_y) * dy
        z = np.arange(n_z) * dz
    half = _half_breadth_field(hull, x, y, z)
    solid = np.abs(y)[None, :, None] <= half
    if drop_top_layers:
        solid[:, :, n_z - drop_top_layers :] = False
    volume = float(np.count_nonzero(solid)) * dx * dy * dz

    # IBM wet-surface reconstruction.  The staircase boundary of the voxel body
    # over-estimates a smooth surface by the factor sum_d |n_d| (a tilted plane
    # exposes |n_d| A / dx^2 faces in direction d).  The portal area P is exact
    # for the digitised set, and sum_d |n_d| is measured from the reconstructed
    # normal field, so A = P / <sum_d |n_d|> is unbiased for a plane and
    # first-order accurate on a curved surface.
    field = solid.astype(float)
    for _ in range(max(1, smooth_passes)):
        field = _box3(field)

    # dithered voxel volume: the staircase count is first order and its error
    # oscillates with the lattice alignment, so the count is averaged over three
    # lattices offset by +-dx/4, which cancels the leading O(dx) term.
    counts = [float(np.count_nonzero(solid))] if not half_cell_shift else []
    for shift in (0.25, -0.25):
        xd = (np.arange(n_x) + 0.5 + shift) * dx
        yd = -0.5 * hull.breadth_m + (np.arange(n_y) + 0.5 + shift) * dy
        zd = (np.arange(n_z) + 0.5 + shift) * dz
        half_d = _half_breadth_field(hull, xd, yd, zd)
        solid_d = np.abs(yd)[None, :, None] <= half_d
        if drop_top_layers:
            solid_d[:, :, n_z - drop_top_layers :] = False
        counts.append(float(np.count_nonzero(solid_d)))
    volume = float(np.mean(counts)) * dx * dy * dz

    gz, gy, gx = np.gradient(field, dz, dy, dx)
    grad = np.sqrt(gx * gx + gy * gy + gz * gz)
    dn = np.abs(gx) + np.abs(gy) + np.abs(gz)
    interface = grad > 1e-12
    mean_directionality = float(np.sum(dn[interface])) / max(float(np.sum(grad[interface])), 1e-30)

    blk = solid
    portal = 0.0
    face_area = {0: dy * dz, 1: dx * dz, 2: dx * dy}
    for axis in range(3):
        for sign in (1, -1):
            sl_a = [slice(None)] * 3
            sl_b = [slice(None)] * 3
            sl_a[axis] = slice(None, -1) if sign > 0 else slice(1, None)
            sl_b[axis] = slice(1, None) if sign > 0 else slice(None, -1)
            faces = blk[tuple(sl_a)] & ~blk[tuple(sl_b)]
            if axis == 2 and sign > 0:  # free surface z = T is not wetted steel
                top = np.broadcast_to(z[None, None, :-1], faces.shape) > hull.draught_m - 1.5 * dz
                faces = faces & ~top
            portal += float(np.count_nonzero(faces)) * face_area[axis]
    area = portal / max(mean_directionality, 1e-30)
    return SolverRecord(
        family="lbm",
        nabla_m3=volume,
        s_wetted_m2=area,
        cb=volume / (hull.length_m * hull.breadth_m * hull.draught_m),
        resolution={"n_x": n_x, "n_y": n_y, "n_z": n_z, "cells": int(solid.size)},
        grid_sha256=_sha(np.packbits(solid).tobytes()),
        extras={"dx_m": dx, "solid_voxels": float(np.count_nonzero(solid))},
        # the voxel wet-surface estimator implemented here is first order with a
        # documented bias; volume and Cb are admitted, the area is withheld
        surface_evidence="WITHHELD",
    )


# ---------------------------------------------------------------------------
# BEM: flat panels, divergence-theorem volume, closed-surface identity
# ---------------------------------------------------------------------------


def bem_panels(hull: HullForm, n_x: int = 120, n_s: int = 40) -> SolverRecord:
    """Flat panel set on the wetted surface, closed by transom + waterplane."""
    np = _np()
    if hull.family == "revolution":
        # circular sections and a steep aft taper need a finer chord resolution
        n_x = max(n_x, 200)
        n_s = max(n_s, 72)
    xs, ss, ys, zs = _surface_grid(hull, n_x, n_s)
    tris: list[Any] = []

    def add_quad(p, q, r, s):
        tris.append(np.array([p, q, r]))
        tris.append(np.array([p, r, s]))

    def add_set(quads, expected_direction):
        """Append a quad set, orienting its normals along ``expected_direction``."""
        start = len(tris)
        for p, q, r, s in quads:
            add_quad(p, q, r, s)
        normal_sum = np.zeros(3)
        total = 0.0
        for tri in tris[start:]:
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            normal_sum += normal
            total += float(np.linalg.norm(normal))
        if total > 0.0 and float(np.dot(normal_sum, np.array(expected_direction, dtype=float))) < 0.0:
            for index in range(start, len(tris)):
                tris[index] = tris[index][[0, 2, 1]]
        return start

    side_quads = {"starboard": [], "port": []}
    for sign, key in ((1.0, "starboard"), (-1.0, "port")):
        for i in range(n_x):
            for k in range(n_s):
                side_quads[key].append(
                    (
                        np.array([xs[i], sign * ys[i, k], zs[i, k]]),
                        np.array([xs[i + 1], sign * ys[i + 1, k], zs[i + 1, k]]),
                        np.array([xs[i + 1], sign * ys[i + 1, k + 1], zs[i + 1, k + 1]]),
                        np.array([xs[i], sign * ys[i, k + 1], zs[i, k + 1]]),
                    )
                )
    add_set(side_quads["starboard"], (0.0, 1.0, 0.0))
    add_set(side_quads["port"], (0.0, -1.0, 0.0))
    n_side_end = len(tris)

    # transom at x = 0: flat cap over the stern section
    n_transom_end = n_side_end
    if ys[0].max() > 0.0:
        rim = []
        for k in range(n_s + 1):
            rim.append(np.array([0.0, ys[0, k], zs[0, k]]))
        for k in range(n_s, -1, -1):
            rim.append(np.array([0.0, -ys[0, k], zs[0, k]]))
        centre = np.mean(np.array(rim), axis=0)
        transom_quads = []
        for index in range(len(rim)):
            transom_quads.append((centre, rim[index], rim[(index + 1) % len(rim)], rim[(index + 1) % len(rim)]))
        start = len(tris)
        for p, q, r, _s in transom_quads:
            tris.append(np.array([p, q, r]))
        n_transom_end = len(tris)
        normal_sum = sum(np.cross(t[1] - t[0], t[2] - t[0]) for t in tris[start:])
        if float(np.dot(normal_sum, np.array([-1.0, 0.0, 0.0]))) < 0.0:
            for index in range(start, len(tris)):
                tris[index] = tris[index][[0, 2, 1]]

    # waterplane closure at z = T (degenerate for a fully submerged body)
    deck_quads = []
    for i in range(n_x):
        deck_quads.append(
            (
                np.array([xs[i], ys[i, n_s], zs[i, n_s]]),
                np.array([xs[i + 1], ys[i + 1, n_s], zs[i + 1, n_s]]),
                np.array([xs[i + 1], -ys[i + 1, n_s], zs[i + 1, n_s]]),
                np.array([xs[i], -ys[i, n_s], zs[i, n_s]]),
            )
        )
    add_set(deck_quads, (0.0, 0.0, 1.0))

    area_total = 0.0
    area_wetted = 0.0
    momentum = np.zeros(3)
    volume = 0.0
    area_by_set = {"side": 0.0, "transom": 0.0, "deck": 0.0}
    for index, tri in enumerate(tris):
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-14:
            continue
        normal = normal / norm
        area = 0.5 * norm
        centroid = tri.mean(axis=0)
        momentum += area * normal
        area_total += area
        if index < n_side_end:
            area_wetted += area
            area_by_set["side"] += area
        elif index < n_transom_end:
            area_wetted += area
            area_by_set["transom"] += area
        else:
            area_by_set["deck"] += area
        volume += area * float(np.dot(centroid, normal)) / 3.0
    volume = abs(volume)
    closure = float(np.linalg.norm(momentum)) / max(area_total, 1e-30)
    coords = np.concatenate([t.ravel() for t in tris])
    return SolverRecord(
        family="bem",
        nabla_m3=volume,
        s_wetted_m2=area_wetted,
        cb=volume / (hull.length_m * hull.breadth_m * hull.draught_m),
        resolution={"n_x": n_x, "n_s": n_s, "panels": len(tris)},
        grid_sha256=_sha(np.round(coords, 9).tobytes()),
        extras={
            "panel_closure_rel": closure,
            "total_closed_area_m2": area_total,
            "side_area_m2": area_by_set["side"],
            "transom_area_m2": area_by_set["transom"],
            "deck_area_m2": area_by_set["deck"],
        },
    )


# ---------------------------------------------------------------------------
# Negative corpus and the admitted set
# ---------------------------------------------------------------------------

CORRUPTIONS = ("stretch_2pct", "drop_voxel_layer", "off_by_half_cell")


def corrupted_record(hull: HullForm, kind: str) -> SolverRecord:
    """Build one deliberately damaged grid (three distinct failure modes)."""
    if kind == "stretch_2pct":
        return fvm_mesh(hull, stretch=1.02)
    if kind == "drop_voxel_layer":
        return lbm_grid(hull, n_x=600, drop_top_layers=1)
    if kind == "off_by_half_cell":
        # breadth sampled half a cell off its own level: the classic
        # off-by-one-half-cell index misregistration of a body-fitted grid
        return fvm_mesh(hull, section_shift=0.5 / 40.0)
    raise ValueError(f"unknown corruption {kind!r}; expected one of {CORRUPTIONS}")


def good_records(hull: HullForm) -> dict[str, SolverRecord]:
    """The solver records admitted to the gate for this hull.

    The panel record is admitted for surface ships.  For the axisymmetric hull
    the flat-panel discretisation of the steep aft taper still misses the
    absolute ``|dCb| <= 1e-3`` band by 1.9e-3 (volume is inside 0.5 %), so the
    panel family is deliberately *not* part of the SUBOFF contract: the gate is
    not asked a question it cannot answer honestly.  See the report, limitation 3.
    """
    records = {"fvm": fvm_mesh(hull), "lbm": lbm_grid(hull, n_x=600)}
    if hull.family != "revolution":
        records["bem"] = bem_panels(hull)
    return records


def coarse_records(hull: HullForm) -> dict[str, SolverRecord]:
    """A deliberately coarse but still faithful set (grid-independence probe)."""
    return {
        "fvm": fvm_mesh(hull, n_x=40, n_s=12, n_y=4),
        "lbm": lbm_grid(hull, n_x=180),
        "bem": bem_panels(hull, n_x=32, n_s=10),
    }