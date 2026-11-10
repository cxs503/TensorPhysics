"""Real 2-D plane-stress finite-element kernel for detail (hot-spot) fatigue.

Why this module exists
----------------------
The pre-existing ``tensorphysics.fatigue.detail_fatigue.fem_stress_field_*``
functions are **analytic surrogates** (exponential decay profiles explicitly
tuned so that the hot-spot extrapolation recovers the analytic Kt).  They
therefore cannot provide an independent verification of either Kt or of the
extrapolation procedure -- the check would be circular.  Here a genuine
displacement-based FEM is implemented:

* 6-node quadratic plane-stress triangles (T6), obtained by splitting structured
  quadrilateral grids into two T6 triangles per quad;
* straight-sided T6 element, 3-point Gauss quadrature (exact for the quadratic
  stiffness integrand);
* consistent nodal loads for uniform tractions on straight edges and 3-point
  Gauss quadrature over curved (quadratic) edges -- needed for the circular outer
  boundary of the Kirsch annulus;
* sparse direct solution (``scipy.sparse.linalg.spsolve``);
* linear element stress recovery at element nodes + nodal averaging.

Model problems (each with an independent analytical anchor)
------------------------------------------------------------
``solve_plate_with_hole``
    Quarter annulus around a circular hole, outer boundary loaded by the traction
    of a uniform far-field uniaxial stress.  The uniform field satisfies
    equilibrium and the outer traction boundary condition exactly, so the exact
    solution of this BVP *is* the Kirsch field for every outer radius ``R/a``
    (no finite-width error): hole-edge stress ``3 sigma_0`` at theta = 90 deg.
    Anchor: Kirsch (1898), ``Kt = 3``.

``solve_plate_with_edge_notch``
    Upper-half rectangular plate with a circular-arc edge notch (depth ``h``,
    root radius ``rho``) under uniaxial tension; ``h == rho`` is the semicircular
    notch.  Idealises the toe of a transverse butt weld (``h`` = reinforcement
    height, ``rho`` = toe radius).  Anchor: Inglis (1913) elliptical-notch
    solution ``Kt = 1 + 2 sqrt(h/rho)`` -- the classical basis of the Peterson
    stress-concentration charts.

``solve_plate_with_thickness_step``
    Plate with a surface thickness step ``t1 -> t2`` closed by a fillet of radius
    ``rho`` (structural discontinuity of the type found at a butt weld between
    plates of different thickness).  Unlike the notch models this generates a
    *structural* surface-stress gradient with decay length O(t), which is what the
    0.4t/0.9t/1.4t hot-spot extrapolation is designed for; used for the
    extrapolation-accuracy and mesh-convergence checks.

Units: any consistent set (mm, MPa, N) is used throughout the fatigue package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class PlaneStressModel:
    """A 2-D plane-stress FE model.

    Attributes
    ----------
    nodes : (nn, 2) ndarray
        Node coordinates.
    elements : (ne, 6) ndarray
        T6 connectivity (3 corners, then midside of edges 1-2, 2-3, 3-1).
    E, nu : float
        Isotropic elastic constants (MPa, -).
    thickness : float
        Plate thickness; multiplies the plane-stress stiffness and all nodal
        loads consistently (unit thickness is used with tractions normalised per
        unit thickness).
    """

    nodes: np.ndarray
    elements: np.ndarray
    E: float = 210000.0     # MPa (steel)
    nu: float = 0.3
    thickness: float = 1.0
    meta: Dict[str, float] = field(default_factory=dict)

    @property
    def ndof(self) -> int:
        return 2 * len(self.nodes)


@dataclass
class FESolution:
    """Result of a plane-stress solve."""

    displacement: np.ndarray       # (nn, 2)
    nodal_stress: np.ndarray       # (nn, 3) -> [sxx, syy, sxy]
    model: PlaneStressModel

    def von_mises(self) -> np.ndarray:
        sxx, syy, sxy = self.nodal_stress.T
        return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)


# ============================================================================
# Element level: T6 shape functions, stiffness and stress recovery
# ============================================================================

# 3-point Gauss rule in area coordinates (exact for polynomials of degree 2)
_GP3 = np.array([[1.0 / 6.0, 1.0 / 6.0],
                 [2.0 / 3.0, 1.0 / 6.0],
                 [1.0 / 6.0, 2.0 / 3.0]])


def tri6_shape(L1: float, L2: float, L3: float) -> np.ndarray:
    """T6 shape functions at area coordinates (L1, L2, L3 = 1-L1-L2)."""
    return np.array([
        L1 * (2.0 * L1 - 1.0),
        L2 * (2.0 * L2 - 1.0),
        L3 * (2.0 * L3 - 1.0),
        4.0 * L1 * L2,
        4.0 * L2 * L3,
        4.0 * L3 * L1,
    ])


def tri6_shape_derivatives(L1: float, L2: float, L3: float) -> np.ndarray:
    """dN/dLi for T6 (rows = basis function, cols = L1, L2, L3)."""
    dN = np.zeros((6, 3))
    dN[0, 0] = 4.0 * L1 - 1.0
    dN[1, 1] = 4.0 * L2 - 1.0
    dN[2, 2] = 4.0 * L3 - 1.0
    dN[3, 0] = 4.0 * L2
    dN[3, 1] = 4.0 * L1
    dN[4, 1] = 4.0 * L3
    dN[4, 2] = 4.0 * L2
    dN[5, 0] = 4.0 * L3
    dN[5, 2] = 4.0 * L1
    return dN


def _tri6_jacobian(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (dL/dx, dL/dy, 2A) for a straight-sided T6 triangle."""
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]
    det2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)   # = 2A
    b = np.array([y2 - y3, y3 - y1, y1 - y2]) / det2        # dLi/dx
    c = np.array([x3 - x2, x1 - x3, x2 - x1]) / det2        # dLi/dy
    return b, c, det2


def tri6_B_matrix(xy: np.ndarray, L1: float, L2: float, L3: float) -> np.ndarray:
    """Strain-displacement matrix B (3 x 12) for a T6 triangle."""
    dN = tri6_shape_derivatives(L1, L2, L3)
    b, c, _ = _tri6_jacobian(xy)
    dNdx = dN @ b
    dNdy = dN @ c
    B = np.zeros((3, 12))
    B[0, 0::2] = dNdx
    B[1, 1::2] = dNdy
    B[2, 0::2] = dNdy
    B[2, 1::2] = dNdx
    return B


def plane_stress_D(E: float, nu: float) -> np.ndarray:
    """Isotropic plane-stress elasticity matrix."""
    f = E / (1.0 - nu ** 2)
    return f * np.array([[1.0, nu, 0.0],
                         [nu, 1.0, 0.0],
                         [0.0, 0.0, (1.0 - nu) / 2.0]])


def tri6_stiffness(xy: np.ndarray, E: float, nu: float, thickness: float) -> np.ndarray:
    """12x12 element stiffness matrix of a straight-sided T6 triangle."""
    D = plane_stress_D(E, nu)
    _, _, det2 = _tri6_jacobian(xy)
    area = 0.5 * det2
    ke = np.zeros((12, 12))
    for L1, L2 in _GP3:
        L3 = 1.0 - L1 - L2
        B = tri6_B_matrix(xy, L1, L2, L3)
        ke += B.T @ D @ B * (area / 3.0)
    return ke * thickness


def tri6_stress(xy: np.ndarray, ue: np.ndarray, E: float, nu: float,
                L1: float, L2: float, L3: float) -> np.ndarray:
    """Stress vector [sxx, syy, sxy] at an area coordinate of a T6 element."""
    D = plane_stress_D(E, nu)
    B = tri6_B_matrix(xy, L1, L2, L3)
    return D @ (B @ ue)


# ============================================================================
# Quad-grid -> conforming T6 mesh (multi-block, coordinate-deduplicated)
# ============================================================================

class T6MeshBuilder:
    """Assemble a conforming T6 mesh from one or more structured quad grids.

    Node identity is established by rounded coordinates, so grids sharing an
    interface line are welded automatically.
    """

    def __init__(self, tol: float = 1e-7):
        self.tol = tol
        self.nodes: List[Tuple[float, float]] = []
        self.elements: List[List[int]] = []
        self._key: Dict[Tuple[int, int], int] = {}

    def _node(self, x: float, y: float) -> int:
        k = (int(round(x / self.tol)), int(round(y / self.tol)))
        idx = self._key.get(k)
        if idx is None:
            idx = len(self.nodes)
            self.nodes.append((float(x), float(y)))
            self._key[k] = idx
        return idx

    def _mid(self, a: int, b: int) -> Tuple[float, float]:
        pa, pb = self.nodes[a], self.nodes[b]
        return 0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])

    def add_grid(self, nodes_ij: np.ndarray, closed_j: bool = False) -> np.ndarray:
        """Add a structured quad grid; returns the (i, j) -> node index array."""
        nodes_ij = np.asarray(nodes_ij, dtype=float)
        ni, nj = nodes_ij.shape[0], nodes_ij.shape[1]
        gi = np.zeros((ni, nj), dtype=int)
        for i in range(ni):
            for j in range(nj):
                gi[i, j] = self._node(nodes_ij[i, j, 0], nodes_ij[i, j, 1])

        n_jquads = nj if closed_j else nj - 1
        for i in range(ni - 1):
            for j in range(n_jquads):
                jn = (j + 1) % nj
                n00, n10 = gi[i, j], gi[i + 1, j]
                n11, n01 = gi[i + 1, jn], gi[i, jn]
                m_ab = self._node(*self._mid(n00, n10))
                m_bc = self._node(*self._mid(n10, n11))
                m_cd = self._node(*self._mid(n01, n11))
                m_da = self._node(*self._mid(n00, n01))
                diag = self._node(*self._mid(n00, n11))
                # T1: corners (n00, n10, n11); midsides (n00-n10, n10-n11, n11-n00)
                self.elements.append([n00, n10, n11, m_ab, m_bc, diag])
                # T2: corners (n11, n01, n00); midsides (n11-n01, n01-n00, n00-n11)
                self.elements.append([n11, n01, n00, m_cd, m_da, diag])
        return gi

    def build(self, E: float = 210000.0, nu: float = 0.3,
              thickness: float = 1.0, **meta) -> PlaneStressModel:
        if not self.elements:
            raise RuntimeError("no elements were added")
        return PlaneStressModel(
            nodes=np.asarray(self.nodes, dtype=float),
            elements=np.asarray(self.elements, dtype=int),
            E=E, nu=nu, thickness=thickness, meta=dict(meta))


def quad_grid_to_tri6(nodes_ij: np.ndarray, closed_j: bool = False):
    """Single-grid wrapper around :class:`T6MeshBuilder` (legacy signature)."""
    b = T6MeshBuilder()
    gi = b.add_grid(nodes_ij, closed_j=closed_j)
    nodes, elements = np.asarray(b.nodes, float), np.asarray(b.elements, int)
    return nodes, elements, lambda i, j: int(gi[i, j % gi.shape[1]])


# ============================================================================
# Loads and boundary conditions
# ============================================================================

def consistent_edge_load(model: PlaneStressModel, tri_nodes: Sequence[int],
                         traction_fn: Callable[[np.ndarray], np.ndarray],
                         n_gauss: int = 3) -> np.ndarray:
    """Consistent nodal loads for a curved quadratic edge.

    ``tri_nodes`` = (end, mid, end) nodes of the quadratic edge; ``traction_fn``
    returns the traction vector (force per unit area).  Returned order is
    ``[fx0, fy0, fx1, fy1, fx2, fy2]`` matching the three supplied nodes.
    """
    a, m, b = (np.asarray(model.nodes[n], dtype=float) for n in tri_nodes)
    xi, w = np.polynomial.legendre.leggauss(n_gauss)
    fx = np.zeros(3)
    fy = np.zeros(3)
    for xg, wg in zip(xi, w):
        N1 = 0.5 * xg * (xg - 1.0)
        N2 = 1.0 - xg * xg
        N3 = 0.5 * xg * (xg + 1.0)
        pt = N1 * a + N2 * m + N3 * b
        dNds = np.array([xg - 0.5, -2.0 * xg, xg + 0.5])
        dpt = dNds[0] * a + dNds[1] * m + dNds[2] * b
        jac = float(np.linalg.norm(dpt))
        tr = np.asarray(traction_fn(pt), dtype=float)
        nvec = np.array([N1, N2, N3])
        fx += wg * jac * nvec * tr[0]
        fy += wg * jac * nvec * tr[1]
    out = np.zeros(6)
    out[0::2] = fx
    out[1::2] = fy
    return out



def _coord_lookup(model: PlaneStressModel, tol: float = 1e-6) -> Dict[Tuple[int, int], int]:
    """Map rounded coordinates to node ids (for interface/boundary queries)."""
    lut: Dict[Tuple[int, int], int] = {}
    for nid, (x, y) in enumerate(model.nodes):
        lut[(int(round(x / tol)), int(round(y / tol)))] = nid
    return lut


def boundary_edge_loads(model: PlaneStressModel,
                        ordered_corner_ids: Sequence[int],
                        traction_fn: Callable[[np.ndarray], np.ndarray]) -> List[Tuple[List[int], np.ndarray]]:
    """Consistent edge loads on a boundary polyline given its corner nodes in order.

    The T6 edge of every boundary quad carries a midside node located at the
    chord midpoint; it is recovered by coordinate lookup, so the load is applied
    with the correct quadratic geometry (important for curved boundaries).
    """
    lut = _coord_lookup(model)
    ids = [int(n) for n in ordered_corner_ids]
    loads: List[Tuple[List[int], np.ndarray]] = []
    for a, b in zip(ids[:-1], ids[1:]):
        pa, pb = model.nodes[a], model.nodes[b]
        mid_xy = 0.5 * (pa + pb)
        key = (int(round(mid_xy[0] / 1e-6)), int(round(mid_xy[1] / 1e-6)))
        m = lut.get(key)
        if m is None:
            raise KeyError("midside node of boundary edge not found in mesh")
        trio = [a, m, b]
        loads.append((trio, consistent_edge_load(model, trio, traction_fn)))
    return loads


def uniform_traction(traction_vector: Sequence[float]) -> Callable[[np.ndarray], np.ndarray]:
    """Return a constant traction function."""
    tv = np.asarray(traction_vector, dtype=float)
    return lambda _pt: tv


def assemble_loads(model: PlaneStressModel,
                   edge_loads: List[Tuple[Sequence[int], np.ndarray]]) -> np.ndarray:
    """Assemble a global load vector from ``(node_ids, load6)`` pairs."""
    f = np.zeros(model.ndof)
    for node_ids, load6 in edge_loads:
        dofs = np.array([[2 * int(n), 2 * int(n) + 1] for n in node_ids]).reshape(-1)
        f[dofs] += np.asarray(load6, dtype=float) * model.thickness
    return f


def solve_plane_stress(model: PlaneStressModel,
                       load: np.ndarray,
                       fixed: Dict[int, Tuple[Optional[float], Optional[float]]]) -> FESolution:
    """Solve the plane-stress boundary-value problem.

    Parameters
    ----------
    model : PlaneStressModel
    load : (ndof,) ndarray
        Global consistent load vector.
    fixed : dict
        ``{node_id: (ux_or_None, uy_or_None)}``; ``None`` leaves that component
        free, a value prescribes the displacement.
    """
    n = len(model.nodes)
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    for el in model.elements:
        xy = model.nodes[el[:3]]
        ke = tri6_stiffness(xy, model.E, model.nu, model.thickness)
        dofs = np.array([[2 * int(nn), 2 * int(nn) + 1] for nn in el]).reshape(-1)
        ii, jj = np.meshgrid(dofs, dofs, indexing="ij")
        rows.append(ii.ravel())
        cols.append(jj.ravel())
        vals.append(ke.ravel())
    K = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(model.ndof, model.ndof)).tocsr()

    prescribed: Dict[int, float] = {}
    for node, (ux, uy) in fixed.items():
        if ux is not None:
            prescribed[2 * int(node)] = float(ux)
        if uy is not None:
            prescribed[2 * int(node) + 1] = float(uy)

    free = np.ones(model.ndof, dtype=bool)
    free[list(prescribed.keys())] = False
    free_idx = np.where(free)[0]
    fixed_idx = np.array(sorted(prescribed.keys()), dtype=int)
    u = np.zeros(model.ndof)
    if fixed_idx.size:
        u[fixed_idx] = np.array([prescribed[int(d)] for d in fixed_idx])

    rhs = load.copy()
    if fixed_idx.size:
        rhs -= K[:, fixed_idx] @ u[fixed_idx]
    u[free_idx] = spsolve(K[free_idx][:, free_idx].tocsc(), rhs[free_idx])

    # ---- stress recovery: linear T6 stress evaluated at the 6 element nodes ----
    nodal_L = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.5, 0.5, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5]])
    nstr = np.zeros((n, 3))
    cnt = np.zeros(n)
    for el in model.elements:
        xy = model.nodes[el[:3]]
        dofs = np.array([[2 * int(nn), 2 * int(nn) + 1] for nn in el]).reshape(-1)
        ue = u[dofs]
        for local, (L1, L2, L3) in enumerate(nodal_L):
            nstr[int(el[local])] += tri6_stress(xy, ue, model.E, model.nu, L1, L2, L3)
            cnt[int(el[local])] += 1.0
    cnt[cnt == 0] = 1.0
    nstr /= cnt[:, None]

    return FESolution(displacement=u.reshape(-1, 2), nodal_stress=nstr, model=model)


# ============================================================================
# Grading helpers
# ============================================================================

def graded_nodes(first_size: float, total: float, growth: float,
                 max_layers: int = 5000) -> np.ndarray:
    """Node coordinates clustered near 0 spanning ``[0, total]``."""
    if first_size <= 0 or total <= 0:
        raise ValueError("sizes must be positive")
    sizes: List[float] = []
    acc = 0.0
    h = first_size
    while acc < total - 1e-15 * total and len(sizes) < max_layers:
        h = min(h, total - acc)
        sizes.append(h)
        acc += h
        h *= growth
    return np.concatenate([[0.0], np.cumsum(sizes)])


def _eta_profile(first_layer: float, height: float, growth: float = 1.2) -> np.ndarray:
    """Normalised through-thickness coordinates, finest at the top surface.

    Returns ``eta`` in ``[0, 1]`` with ``eta[-1] = 1`` (top) and the first layer
    from the top equal to ``first_layer / height``.
    """
    frac = max(min(first_layer / height, 0.5), 1e-3)
    layers: List[float] = []
    acc = 0.0
    h = frac
    while acc < 1.0 - 1e-12:
        h = min(h, 1.0 - acc)
        layers.append(h)
        acc += h
        h *= growth
    eta = 1.0 - np.concatenate([[0.0], np.cumsum(layers)])
    eta[0], eta[-1] = 1.0, 0.0
    return np.sort(np.clip(eta, 0.0, 1.0))


def _column_mesh(x_nodes: np.ndarray, top_y: Callable[[float], float],
                 eta: np.ndarray) -> np.ndarray:
    """Structured grid built from vertical columns reaching the top surface."""
    nodes_ij = np.zeros((len(x_nodes), len(eta), 2))
    for i, x in enumerate(x_nodes):
        nodes_ij[i, :, 1] = eta * top_y(x)
        nodes_ij[i, :, 0] = x
    return nodes_ij


# ============================================================================
# Mesh generators
# ============================================================================

def mesh_plate_with_hole(a: float = 10.0,
                         R_factor: float = 12.0,
                         n_theta: int = 48,
                         first_layer_frac: float = 0.02,
                         growth: float = 1.15) -> PlaneStressModel:
    """Quarter annulus mesh around a circular hole (Kirsch model)."""
    R = a * R_factor
    r_nodes = a + graded_nodes(first_layer_frac * a, R - a, growth)
    r_nodes[-1] = R
    th = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)
    nodes_ij = np.zeros((len(r_nodes), len(th), 2))
    for i, rr in enumerate(r_nodes):
        nodes_ij[i, :, 0] = rr * np.cos(th)
        nodes_ij[i, :, 1] = rr * np.sin(th)
    builder = T6MeshBuilder()
    builder.add_grid(nodes_ij)
    return builder.build(a=a, R=R, n_theta=n_theta,
                         n_r=int(len(r_nodes) - 1), geometry="hole")


def mesh_plate_with_edge_notch(L: float = 120.0, W: float = 40.0,
                               h: float = 1.0, rho: float = 1.0,
                               n_arc: int = 16, first_layer: Optional[float] = None,
                               growth_x: float = 1.25, growth_y: float = 1.2,
                               notch_x: Optional[float] = None) -> PlaneStressModel:
    """Upper-half plate with a circular-arc edge notch (butt-weld-toe idealisation).

    Parameters
    ----------
    L, W : float
        Plate length and full width.
    h, rho : float
        Notch depth and root radius (``0 < h <= rho`` for the circular-arc family;
        ``h == rho`` is the semicircular notch).
    n_arc : int
        Elements along one half of the notch arc.
    first_layer : float, optional
        Size of the first element layer at the notch surface (default ``rho/4``).
    notch_x : float, optional
        Notch centre (default ``L/2``).
    """
    if not (0.0 < h <= rho + 1e-12):
        raise ValueError("circular-arc notch requires 0 < h <= rho")
    notch_x = L / 2.0 if notch_x is None else notch_x
    first_layer = rho / 4.0 if first_layer is None else first_layer
    y_half = 0.5 * W

    # --- x nodes: refined along the arc, graded into the far field ---
    k = np.arange(n_arc + 1)
    phi = 0.5 * np.pi * (k / n_arc) ** 1.5
    dx_arc = rho * np.sin(phi)                       # 0 .. rho
    right = notch_x + dx_arc
    left = notch_x - dx_arc
    far = graded_nodes(first_layer, 0.5 * L - rho, growth_x) + rho
    x_nodes = np.unique(np.concatenate([
        notch_x - far, left, right, notch_x + far,
    ]))
    x_nodes = x_nodes[(x_nodes >= 0.0) & (x_nodes <= L)]

    def top_y(x: float) -> float:
        if abs(x - notch_x) <= rho:
            return y_half - (np.sqrt(rho ** 2 - (x - notch_x) ** 2) - (rho - h))
        return y_half

    eta = _eta_profile(first_layer, y_half, growth_y)
    nodes_ij = _column_mesh(x_nodes, top_y, eta)
    builder = T6MeshBuilder()
    builder.add_grid(nodes_ij)
    return builder.build(L=L, W=W, h=h, rho=rho, notch_x=notch_x,
                         n_arc=n_arc, geometry="edge_notch")


def mesh_plate_with_thickness_step(L: float = 120.0, t1: float = 12.0,
                                   t2: float = 16.0, rho: float = 8.0,
                                   first_layer: Optional[float] = None,
                                   growth_x: float = 1.22,
                                   growth_y: float = 1.2,
                                   step_x: Optional[float] = None) -> PlaneStressModel:
    """Upper-half plate with a surface thickness step closed by a fillet.

    The top surface is ``y = t1`` for ``x <= step_x - w``, a circular arc of
    radius ``rho`` (tangent to both surfaces) for the transition, and ``y = t2``
    for ``x >= step_x``, with ``w = sqrt(2 rho h - h**2)`` and ``h = t2 - t1``
    (requires ``h <= 2 rho``).  The toe of the transition is at
    ``x_toe = step_x - w``.
    """
    h = t2 - t1
    if h <= 0:
        raise ValueError("t2 must exceed t1")
    if h > 2.0 * rho:
        raise ValueError("fillet radius too small: need h <= 2*rho")
    step_x = L / 2.0 if step_x is None else step_x
    first_layer = t1 / 16.0 if first_layer is None else first_layer
    w = float(np.sqrt(2.0 * rho * h - h ** 2))
    x_toe = step_x - w

    def top_y(x: float) -> float:
        if x <= x_toe:
            return t1
        if x >= step_x:
            return t2
        return t2 - rho + np.sqrt(max(rho ** 2 - (x - step_x) ** 2, 0.0))

    # x nodes: fine at the toe and at the step, graded into both far fields
    arc = np.linspace(0.0, 1.0, 17)
    x_arc = x_toe + w * (1.0 - np.cos(0.5 * np.pi * arc))     # clustered at the toe
    left = x_toe - graded_nodes(first_layer, x_toe, growth_x)
    right = step_x + graded_nodes(first_layer, L - step_x, growth_x)
    x_nodes = np.unique(np.concatenate([left, x_arc, right, [x_toe, step_x]]))
    x_nodes = x_nodes[(x_nodes >= 0.0) & (x_nodes <= L)]

    eta = _eta_profile(first_layer, t2, growth_y)
    nodes_ij = _column_mesh(x_nodes, top_y, eta)
    builder = T6MeshBuilder()
    builder.add_grid(nodes_ij)
    return builder.build(L=L, t1=t1, t2=t2, rho=rho, step_x=step_x,
                         x_toe=x_toe, w=w, first_layer=first_layer,
                         geometry="thickness_step")


# ============================================================================
# Boundary helpers and canonical solves
# ============================================================================


def _corner_only(model: PlaneStressModel, ids: Sequence[int],
                 tol: float = 1e-6) -> List[int]:
    """Return the grid-corner nodes of an ordered straight boundary node list.

    Along a straight boundary the quad edges are equal segments, so the ordered
    node list alternates corner, chord-midpoint, corner, ... ; the alternation is
    verified explicitly before thinning.
    """
    ids = [int(n) for n in ids]
    if len(ids) < 3:
        return ids
    pts = model.nodes[ids]
    for i in range(0, len(ids) - 2, 2):
        mid = 0.5 * (pts[i] + pts[i + 2])
        err = np.linalg.norm(pts[i + 1] - mid)
        scale = max(np.linalg.norm(pts[i + 2] - pts[i]), 1e-30)
        if err > tol * scale:
            raise ValueError("boundary node list is not an alternating corner/midpoint chain")
    corners = ids[::2]
    if (len(ids) - 1) % 2:
        raise ValueError("odd number of boundary edges")
    return corners


def nodes_on_line(model: PlaneStressModel, axis: int, value: float,
                  tol: Optional[float] = None) -> np.ndarray:
    """Indices of nodes lying on ``coord[axis] == value`` (within tolerance)."""
    if tol is None:
        span = float(np.ptp(model.nodes[:, axis]))
        tol = 1e-9 * max(span, 1.0) + 1e-7
    return np.where(np.abs(model.nodes[:, axis] - value) <= tol)[0]


def _edge_triples_along(model: PlaneStressModel, ids: np.ndarray,
                        along: int = 1) -> List[List[int]]:
    """Group boundary nodes (sorted along a coordinate) into quadratic triples."""
    ids = np.asarray(ids)
    order = ids[np.argsort(model.nodes[ids, along])]
    triples: List[List[int]] = []
    k = 0
    while k + 2 < len(order):
        triples.append([int(order[k]), int(order[k + 1]), int(order[k + 2])])
        k += 2
    return triples


def solve_plate_with_hole(sigma_0: float = 1.0, a: float = 10.0,
                          R_factor: float = 12.0, n_theta: int = 48,
                          first_layer_frac: float = 0.02,
                          E: float = 210000.0, nu: float = 0.3) -> FESolution:
    """Solve the Kirsch quarter-annulus problem (uniform far-field traction)."""
    model = mesh_plate_with_hole(a=a, R_factor=R_factor, n_theta=n_theta,
                                 first_layer_frac=first_layer_frac)
    model.E, model.nu = E, nu
    R = a * R_factor

    fixed: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    for nid in nodes_on_line(model, 1, 0.0):
        fixed[int(nid)] = (None, 0.0)          # theta = 0 symmetry line
    for nid in nodes_on_line(model, 0, 0.0):
        ux, _ = fixed.get(int(nid), (None, None))
        fixed[int(nid)] = (0.0, fixed.get(int(nid), (None, None))[1])

    def traction(pt: np.ndarray) -> np.ndarray:
        rr = np.hypot(pt[0], pt[1])
        return np.array([sigma_0 * pt[0] / rr, 0.0])

    r = np.hypot(model.nodes[:, 0], model.nodes[:, 1])
    ids = np.where(np.abs(r - R) <= 1e-6 * R + 1e-9)[0]
    ang = np.arctan2(model.nodes[ids, 1], model.nodes[ids, 0])
    ids = ids[np.argsort(ang)]
    loads = boundary_edge_loads(model, ids, traction)
    return solve_plane_stress(model, assemble_loads(model, loads), fixed)


def solve_plate_with_edge_notch(sigma_0: float = 1.0, L: float = 120.0,
                                W: float = 40.0, h: float = 1.0, rho: float = 1.0,
                                n_arc: int = 16, first_layer: Optional[float] = None,
                                E: float = 210000.0, nu: float = 0.3) -> FESolution:
    """Solve the edge-notched plate (upper-half model) under uniaxial tension."""
    model = mesh_plate_with_edge_notch(L=L, W=W, h=h, rho=rho, n_arc=n_arc,
                                       first_layer=first_layer)
    model.E, model.nu = E, nu
    fixed: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    for nid in nodes_on_line(model, 1, 0.0):
        fixed[int(nid)] = (None, 0.0)                 # symmetry
    loads = []
    for x0, sign in ((0.0, -1.0), (L, 1.0)):
        ids = nodes_on_line(model, 0, x0)
        ids = ids[np.argsort(model.nodes[ids, 1])]
        # outward traction: -sigma_0 e_x at x=0, +sigma_0 e_x at x=L
        loads.extend(boundary_edge_loads(model, _corner_only(model, ids),
                                         uniform_traction([sign * sigma_0, 0.0])))
    corner_ids = nodes_on_line(model, 0, 0.0)
    corner = int(corner_ids[np.argmin(model.nodes[corner_ids, 1])])
    ux, uy = fixed.get(corner, (None, None))
    fixed[corner] = (0.0, uy)
    return solve_plane_stress(model, assemble_loads(model, loads), fixed)


def solve_plate_with_thickness_step(sigma_0: float = 1.0, L: float = 120.0,
                                    t1: float = 12.0, t2: float = 16.0,
                                    rho: float = 8.0,
                                    first_layer: Optional[float] = None,
                                    E: float = 210000.0, nu: float = 0.3) -> FESolution:
    """Solve the thickness-step plate (upper-half model) under uniaxial tension."""
    model = mesh_plate_with_thickness_step(L=L, t1=t1, t2=t2, rho=rho,
                                           first_layer=first_layer)
    model.E, model.nu = E, nu
    fixed: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    for nid in nodes_on_line(model, 1, 0.0):
        fixed[int(nid)] = (None, 0.0)
    # equal membrane force at both ends: N = sigma_ref * t1 (thin side reference)
    membrane = sigma_0 * t1
    loads = []
    for x0, sig in ((0.0, -membrane / t1), (L, membrane / t2)):
        ids = nodes_on_line(model, 0, x0)
        ids = ids[np.argsort(model.nodes[ids, 1])]
        loads.extend(boundary_edge_loads(model, _corner_only(model, ids),
                                         uniform_traction([sig, 0.0])))
    corner_ids = nodes_on_line(model, 0, 0.0)
    corner = int(corner_ids[np.argmin(model.nodes[corner_ids, 1])])
    ux, uy = fixed.get(corner, (None, None))
    fixed[corner] = (0.0, uy)
    return solve_plane_stress(model, assemble_loads(model, loads), fixed)


# ============================================================================
# Post-processing for hot-spot fatigue
# ============================================================================

def surface_stress_profile(solution: FESolution, sigma_axis: int = 0,
                           x_range: Optional[Tuple[float, float]] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(x, sigma)`` of the nodes on the upper free surface.

    For every distinct x the highest node is taken; the result is sorted by x and
    optionally restricted to ``x_range``.
    """
    model = solution.model
    x = model.nodes[:, 0]
    y = model.nodes[:, 1]
    best: Dict[float, int] = {}
    for nid in range(len(x)):
        key = round(float(x[nid]), 8)
        j = best.get(key)
        if j is None or y[nid] > y[j]:
            best[key] = nid
    keys = np.array(sorted(best.keys()))
    sel = np.array([best[k] for k in keys])
    order = np.argsort(x[sel])
    xs, sig = x[sel][order], solution.nodal_stress[sel[order], sigma_axis]
    if x_range is not None:
        m = (xs >= x_range[0]) & (xs <= x_range[1])
        xs, sig = xs[m], sig[m]
    return xs, sig


def peak_surface_stress(solution: FESolution, x_center: float,
                        half_width: float, sigma_axis: int = 0) -> Tuple[float, float]:
    """Maximum upper-surface stress within ``|x - x_center| <= half_width``.

    Returns ``(peak_stress, x_at_peak)``.
    """
    xs, sig = surface_stress_profile(solution, sigma_axis=sigma_axis)
    mask = np.abs(xs - x_center) <= half_width
    if not mask.any():
        raise ValueError("no surface nodes inside the requested window")
    i = int(np.argmax(sig[mask]))
    return float(sig[mask][i]), float(xs[mask][i])


def mesh_t_joint(L: float = 200.0, t: float = 20.0, a: float = 60.0,
                 t_a: float = 10.0, n_attach: int = 8,
                 first_layer: Optional[float] = None, growth_x: float = 1.18,
                 x_attach: Optional[float] = None) -> PlaneStressModel:
    """Half-thickness T-joint (transverse attachment on one side of a plate).

    Geometry: main plate ``y in [0, t/2]`` (symmetry at ``y = 0``), transverse
    attachment ``x in [x_a, x_a + t_a]``, ``y in [t/2, t/2 + a]`` welded to the
    plate surface.  The weld toe is modelled as a *sharp re-entrant corner*
    (standard hot-spot FE practice: the notch radius is deliberately not
    resolved, the structural hot-spot stress is obtained by extrapolation).

    Parameters
    ----------
    L : float
        Plate length.
    t : float
        Main plate thickness (``t/2`` is the modelled half-width).
    a, t_a : float
        Attachment height and thickness.
    n_attach : int
        Elements across the attachment thickness.
    first_layer : float, optional
        First element size at the toe (default ``t/16``).
    """
    y_half = 0.5 * t
    x_a = 0.5 * (L - t_a) if x_attach is None else x_attach
    first_layer = t / 16.0 if first_layer is None else first_layer

    left = x_a - graded_nodes(first_layer, x_a, growth_x)
    mid = np.linspace(x_a, x_a + t_a, n_attach + 1)
    right = x_a + t_a + graded_nodes(first_layer, L - (x_a + t_a), growth_x)
    x_all = np.unique(np.concatenate([left, mid, right]))
    x_all = x_all[(x_all >= 0.0) & (x_all <= L)]

    builder = T6MeshBuilder()
    # block A: main plate half-thickness
    eta_a = _eta_profile(first_layer, y_half, 1.2)
    grid_a = np.zeros((len(x_all), len(eta_a), 2))
    grid_a[..., 0] = x_all[:, None]
    grid_a[..., 1] = (eta_a * y_half)[None, :]
    builder.add_grid(grid_a)
    # block B: attachment (shares the interface nodes at y = y_half)
    x_att = x_all[(x_all >= x_a - 1e-12) & (x_all <= x_a + t_a + 1e-12)]
    eta_b = _eta_profile(first_layer, a, 1.2)
    grid_b = np.zeros((len(x_att), len(eta_b), 2))
    grid_b[..., 0] = x_att[:, None]
    grid_b[..., 1] = (y_half + eta_b * a)[None, :]
    builder.add_grid(grid_b)
    return builder.build(L=L, t=t, a=a, t_a=t_a, x_attach=x_a, geometry="t_joint")


def solve_t_joint(sigma_0: float = 1.0, L: float = 200.0, t: float = 20.0,
                  a: float = 60.0, t_a: float = 10.0, n_attach: int = 8,
                  first_layer: Optional[float] = None,
                  E: float = 210000.0, nu: float = 0.3) -> FESolution:
    """Solve the T-joint model under uniaxial main-plate tension."""
    model = mesh_t_joint(L=L, t=t, a=a, t_a=t_a, n_attach=n_attach,
                         first_layer=first_layer)
    model.E, model.nu = E, nu
    fixed: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    for nid in nodes_on_line(model, 1, 0.0):
        fixed[int(nid)] = (None, 0.0)                # symmetry plane
    loads = []
    for x0, sign in ((0.0, -1.0), (L, 1.0)):
        ids = nodes_on_line(model, 0, x0)
        ids = ids[np.argsort(model.nodes[ids, 1])]
        loads.extend(boundary_edge_loads(model, _corner_only(model, ids),
                                         uniform_traction([sign * sigma_0, 0.0])))
    corner_ids = nodes_on_line(model, 0, 0.0)
    corner = int(corner_ids[np.argmin(model.nodes[corner_ids, 1])])
    fixed[corner] = (0.0, 0.0)
    return solve_plane_stress(model, assemble_loads(model, loads), fixed)
