"""Structured meshes for the 3-D solid element kernel.

Only structured topologies are generated (graded boxes and annular sectors);
these are exactly the topologies needed for the verification problems:

* distorted / undistorted boxes for the constant-strain patch test,
* thick-walled cylinders for the Lame benchmark and the volumetric-locking
  study.

Hex20 meshes are obtained by promoting a Hex8 grid: every edge midpoint is
created once (shared by neighbours) and snapped back onto the analytic
cylinder wall for circumferential edges, which keeps the quadratic element
edges exactly on the circular arc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from .shape import n_nodes

__all__ = [
    "SolidMesh",
    "box_mesh",
    "annulus_mesh",
    "promote_hex8_to_hex20",
    "HEX_FACES",
    "HEX20_FACES",
    "quad_shape4",
    "quad_shape8",
    "pressure_load",
]


@dataclass
class SolidMesh:
    """Homogeneous-hexahedral mesh."""

    element: str
    nodes: np.ndarray
    connectivity: np.ndarray  # (n_elem, n_nodes_per_elem)

    def __post_init__(self) -> None:
        self.element = self.element.lower()
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.connectivity = np.asarray(self.connectivity, dtype=int)
        nn = n_nodes(self.element)
        if self.connectivity.ndim != 2 or self.connectivity.shape[1] != nn:
            raise ValueError(
                f"connectivity for {self.element} must have {nn} columns, "
                f"got shape {self.connectivity.shape}"
            )
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must be (n, 3)")

    # ------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elem(self) -> int:
        return self.connectivity.shape[0]

    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes

    def element_coords(self, e: int) -> np.ndarray:
        return self.nodes[self.connectivity[e]]

    def element_dofs(self, e: int) -> np.ndarray:
        el = self.connectivity[e]
        return np.column_stack([3 * el + i for i in range(3)]).ravel()

    def nodes_where(self, predicate: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Node indices whose coordinates satisfy ``predicate(xyz_array)``."""
        mask = np.asarray(predicate(self.nodes))
        return np.nonzero(mask)[0]

    def dofs_of_nodes(self, node_ids: Iterable[int], component: int | None = None) -> np.ndarray:
        node_ids = np.asarray(list(node_ids), dtype=int)
        if component is None:
            return np.column_stack([3 * node_ids + i for i in range(3)]).ravel()
        return 3 * node_ids + int(component)


def _grid_nodes(
    dims: Sequence[int],
    origin: Sequence[float],
    lengths: Sequence[float],
    grading: Sequence[float] | None = None,
) -> np.ndarray:
    axes = []
    for d in range(3):
        n = int(dims[d])
        s = np.linspace(0.0, 1.0, n + 1)
        if grading is not None and grading[d] != 1.0:
            s = s ** float(grading[d])
        axes.append(origin[d] + lengths[d] * s)
    X, Y, Z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def _grid_connectivity(dims: Sequence[int]) -> np.ndarray:
    nx, ny, nz = (int(d) for d in dims)

    def nid(i: int, j: int, k: int) -> int:
        return (i * (ny + 1) + j) * (nz + 1) + k

    conn = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                conn.append(
                    [
                        nid(i, j, k),
                        nid(i + 1, j, k),
                        nid(i + 1, j + 1, k),
                        nid(i, j + 1, k),
                        nid(i, j, k + 1),
                        nid(i + 1, j, k + 1),
                        nid(i + 1, j + 1, k + 1),
                        nid(i, j + 1, k + 1),
                    ]
                )
    return np.array(conn, dtype=int)


def box_mesh(
    dims: Sequence[int] = (2, 2, 2),
    lengths: Sequence[float] = (1.0, 1.0, 1.0),
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    element: str = "hex8",
    distort: float = 0.0,
    seed: int = 0,
) -> SolidMesh:
    """Structured box mesh.

    ``distort`` perturbs interior nodes by up to ``distort`` times the local
    element size (used for the distorted patch test).
    """
    nodes = _grid_nodes(dims, origin, lengths)
    dims_arr = np.array(dims, dtype=int)
    conn = _grid_connectivity(dims_arr)
    if distort > 0.0:
        rng = np.random.default_rng(seed)
        nx, ny, nz = dims_arr
        h = np.array([lengths[0] / nx, lengths[1] / ny, lengths[2] / nz])
        interior = np.ones(len(nodes), dtype=bool)
        for d in range(3):
            interior &= nodes[:, d] > origin[d] + 1e-12
            interior &= nodes[:, d] < origin[d] + lengths[d] - 1e-12
        nodes[interior] += distort * h * rng.uniform(-1.0, 1.0, size=(interior.sum(), 3))
    mesh = SolidMesh(element="hex8", nodes=nodes, connectivity=conn)
    if element.lower() == "hex20":
        mesh = promote_hex8_to_hex20(mesh)
    return mesh


def promote_hex8_to_hex20(
    mesh: SolidMesh,
    project: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> SolidMesh:
    """Build a Hex20 mesh from a Hex8 grid, sharing edge mid-side nodes.

    ``project(midpoint, p_a, p_b)`` may move the mid-side node away from the
    chord midpoint (e.g. onto a circular arc).
    """
    if mesh.element != "hex8":
        raise ValueError("promote_hex8_to_hex20 expects a hex8 mesh")
    nodes = [p.copy() for p in mesh.nodes]
    mid: dict[tuple[int, int], int] = {}
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    def midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key in mid:
            return mid[key]
        pa, pb = nodes[key[0]], nodes[key[1]]
        p = 0.5 * (pa + pb)
        if project is not None:
            p = np.asarray(project(p, pa, pb), dtype=float).ravel()
        nodes.append(p)
        mid[key] = len(nodes) - 1
        return mid[key]

    conn20 = []
    for el in mesh.connectivity:
        row = [int(c) for c in el]
        row.extend(midpoint(int(el[a]), int(el[b])) for a, b in edges)
        conn20.append(row)
    return SolidMesh(element="hex20", nodes=np.array(nodes), connectivity=np.array(conn20))


def _arc_midpoint(mid: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Snap circumferential edge mid-side nodes onto the exact circular arc."""
    ra = float(np.hypot(pa[0], pa[1]))
    rb = float(np.hypot(pb[0], pb[1]))
    same_radius = abs(ra - rb) <= 1e-10 * max(ra, rb, 1.0)
    same_z = abs(pa[2] - pb[2]) <= 1e-12 * max(1.0, abs(pa[2]))
    if not (same_radius and same_z) or ra == 0.0:
        return np.asarray(mid, dtype=float)
    direction = mid[:2] / np.linalg.norm(mid[:2])
    return np.array([ra * direction[0], ra * direction[1], 0.5 * (pa[2] + pb[2])])


def annulus_mesh(
    a: float = 1.0,
    b: float = 2.0,
    length: float = 1.0,
    n_radial: int = 4,
    n_circ: int = 24,
    n_axial: int = 1,
    element: str = "hex8",
    theta_span: float = 2.0 * np.pi,
) -> SolidMesh:
    """Thick-walled cylinder (annular sector) extruded along ``z``."""
    if not (0.0 < a < b):
        raise ValueError("require 0 < a < b")
    rs = np.linspace(a, b, n_radial + 1)
    ts = np.linspace(0.0, theta_span, n_circ + 1)
    zs = np.linspace(0.0, length, n_axial + 1)
    nodes = []
    for k in range(n_axial + 1):
        for j in range(n_circ + 1):
            for i in range(n_radial + 1):
                nodes.append([rs[i] * np.cos(ts[j]), rs[i] * np.sin(ts[j]), zs[k]])
    nodes = np.array(nodes)

    def nid(i: int, j: int, k: int) -> int:
        return (k * (n_circ + 1) + j) * (n_radial + 1) + i

    conn = []
    for k in range(n_axial):
        for j in range(n_circ):
            for i in range(n_radial):
                conn.append(
                    [
                        nid(i, j, k),
                        nid(i + 1, j, k),
                        nid(i + 1, j + 1, k),
                        nid(i, j + 1, k),
                        nid(i, j, k + 1),
                        nid(i + 1, j, k + 1),
                        nid(i + 1, j + 1, k + 1),
                        nid(i, j + 1, k + 1),
                    ]
                )
    mesh = SolidMesh(element="hex8", nodes=nodes, connectivity=np.array(conn))
    if element.lower() == "hex20":
        mesh = promote_hex8_to_hex20(mesh, project=_arc_midpoint)
    return mesh


# ---------------------------------------------------------------------------
# Faces and pressure loads
# ---------------------------------------------------------------------------
# Face corner ordering follows quad_shape4: (-1,-1), (+1,-1), (+1,+1), (-1,+1)
# in the face parametric coordinates; Hex20 faces append the four mid-side
# nodes in the order (0,-1), (+1,0), (0,+1), (-1,0).

HEX_FACES = {
    "bottom": [0, 1, 2, 3],
    "top": [4, 5, 6, 7],
    "front": [0, 1, 5, 4],   # eta = -1
    "right": [1, 2, 6, 5],   # xi = +1
    "back": [3, 2, 6, 7],    # eta = +1
    "left": [0, 3, 7, 4],    # xi = -1
}

HEX20_FACES = {
    "bottom": [0, 1, 2, 3, 8, 9, 10, 11],
    "top": [4, 5, 6, 7, 12, 13, 14, 15],
    "front": [0, 1, 5, 4, 8, 17, 12, 16],
    "right": [1, 2, 6, 5, 9, 18, 13, 17],
    "back": [3, 2, 6, 7, 10, 18, 14, 19],
    "left": [0, 3, 7, 4, 11, 19, 15, 16],
}


def quad_shape4(t: float, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear 4-node face shape functions and derivatives (2x4)."""
    N = 0.25 * np.array(
        [(1 - t) * (1 - s), (1 + t) * (1 - s), (1 + t) * (1 + s), (1 - t) * (1 + s)]
    )
    dNdt = 0.25 * np.array([-(1 - s), (1 - s), (1 + s), -(1 + s)])
    dNds = 0.25 * np.array([-(1 - t), -(1 + t), (1 + t), (1 - t)])
    return N, np.vstack([dNdt, dNds])


def quad_shape8(t: float, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic serendipity 8-node face shape functions and derivatives."""
    N = np.zeros(8)
    dN = np.zeros((2, 8))
    for a, (ta, sa) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        N[a] = 0.25 * (1 + ta * t) * (1 + sa * s) * (ta * t + sa * s - 1)
        dN[0, a] = 0.25 * ta * (1 + sa * s) * (2 * ta * t + sa * s)
        dN[1, a] = 0.25 * sa * (1 + ta * t) * (ta * t + 2 * sa * s)
    N[4] = 0.5 * (1 - t * t) * (1 - s)
    dN[0, 4] = -t * (1 - s)
    dN[1, 4] = -0.5 * (1 - t * t)
    N[5] = 0.5 * (1 + t) * (1 - s * s)
    dN[0, 5] = 0.5 * (1 - s * s)
    dN[1, 5] = -s * (1 + t)
    N[6] = 0.5 * (1 - t * t) * (1 + s)
    dN[0, 6] = -t * (1 + s)
    dN[1, 6] = 0.5 * (1 - t * t)
    N[7] = 0.5 * (1 - t) * (1 - s * s)
    dN[0, 7] = -0.5 * (1 - s * s)
    dN[1, 7] = -s * (1 - t)
    return N, dN


def _quad_rule2() -> tuple[list[tuple[float, float]], np.ndarray]:
    g = 1.0 / np.sqrt(3.0)
    return [(-g, -g), (g, -g), (g, g), (-g, g)], np.ones(4)


def _quad_rule3() -> tuple[list[tuple[float, float]], np.ndarray]:
    p = (-np.sqrt(0.6), 0.0, np.sqrt(0.6))
    w = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
    pts = [(t, s) for t in p for s in p]
    wts = np.array([wt * ws for wt in w for ws in w])
    return pts, wts


def pressure_load(
    mesh: SolidMesh,
    pressure: float,
    face_selector: Callable[[np.ndarray, np.ndarray, np.ndarray], bool],
) -> np.ndarray:
    """Consistent nodal loads from a uniform pressure on selected faces.

    ``face_selector(centroid, outward_normal, corner_coords)`` decides whether
    a face carries the pressure; the traction is ``-pressure * outward_normal``
    (i.e. the pressure pushes *into* the solid).  The outward normal is defined
    with respect to the element centroid, so the result does not depend on the
    local face ordering.
    """
    faces = HEX_FACES if mesh.element == "hex8" else HEX20_FACES
    f = np.zeros(mesh.n_dof)
    for e in range(mesh.n_elem):
        coords = mesh.element_coords(e)
        elem_centroid = coords[:8].mean(axis=0) if mesh.element == "hex20" else coords.mean(axis=0)
        for face in faces.values():
            face_corner_coords = coords[face[:4]]
            centroid = face_corner_coords.mean(axis=0)
            n0 = np.cross(
                face_corner_coords[1] - face_corner_coords[0],
                face_corner_coords[3] - face_corner_coords[0],
            )
            norm = np.linalg.norm(n0)
            if norm <= 0.0:
                continue
            n0 = n0 / norm
            if n0 @ (centroid - elem_centroid) < 0.0:
                n0 = -n0
            if not face_selector(centroid, n0, face_corner_coords):
                continue
            shape_fn = quad_shape4 if len(face) == 4 else quad_shape8
            pts, wts = _quad_rule2() if len(face) == 4 else _quad_rule3()
            el = mesh.connectivity[e][face]
            for (t, s), w in zip(pts, wts):
                N, dN = shape_fn(t, s)
                dXdt = dN[0] @ face_corner_coords if len(face) == 4 else dN[0] @ coords[face]
                dXds = dN[1] @ face_corner_coords if len(face) == 4 else dN[1] @ coords[face]
                n_vec = np.cross(dXdt, dXds)
                jac = np.linalg.norm(n_vec)
                if jac <= 0.0:
                    continue
                n_unit = n_vec / jac
                if n_unit @ n0 < 0.0:
                    n_unit = -n_unit
                trac = -pressure * n_unit
                for a_local, a_global in enumerate(el):
                    f[3 * a_global: 3 * a_global + 3] += N[a_local] * trac * w * jac
    return f