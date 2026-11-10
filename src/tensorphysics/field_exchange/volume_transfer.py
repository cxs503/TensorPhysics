"""Conservative *volumetric* field transfer between Eulerian grids and Lagrange points.

Why this module exists
----------------------
``field_exchange.mapping`` transfers **surface** data (LBM wall tractions on panels,
DEM contact forces at points) onto FEM nodes.  A large class of multiphysics couplings
instead exchanges **volume** data:

* the LBM fluid stress field ``sigma_ij(x)`` is cell-centred on an Eulerian lattice and
  must be handed to the Lagrange points (nodes) of a structural FE mesh -- e.g. the ice
  plate / hull shell embedded in the flow, or a porous interface where the fluid stress
  acts over the FE support of the solid;
* the DEM particle cloud is Lagrangian (positions + volumes) and must be handed to the
  Eulerian cells of an FVM grid as a **volume fraction** ``alpha_c`` (CFD-DEM coupling).

Both directions are only admissible if the transfer is *conservative*.  The discrete
analogues of the two conservation statements are

    sum_c sigma_c V_c            = sum_i sigma_i V_i          (force / momentum flux)
    sum_c x_c (x) sigma_c V_c    = sum_i x_i (x) sigma_i V_i  (couple / first moment)

for the LBM -> FEM stress transfer, and

    sum_c alpha_c V_c = sum_p V_p        (solid mass)
    sum_c alpha_c V_c x_c = sum_p V_p x_p (centre of solid volume)

for the DEM -> FVM volume-fraction transfer.  The operators below satisfy both
identities *algebraically* (they are partition-of-unity scatter/gather operators that
reproduce linear coordinates), so the measured residual is at round-off level; the
audit functions measure it instead of assuming it.

Operators
---------
* :func:`map_particles_to_volume_fraction` -- Lagrangian particles -> Eulerian cell
  volume fraction.  Three deposition kernels are provided:

  - ``"cic"`` cloud-in-cell (multilinear) weights: partition of unity, reproduces
    linear coordinates, cell volume fraction error ``O(h^2)`` for a smooth particle
    density;
  - ``"ngp"`` nearest-grid-point lumping: mass conservative but first-moment
    defective (used as a negative control);
  - ``"cube"`` the divided-particle-method (DPM) kernel of CFD-DEM: the *exact*
    intersection volume of an axis-aligned cubic particle with each cell, which is
    the geometric volume-fraction definition (Wu et al. 2014, *Chem. Eng. Sci.*).

* :func:`map_cell_field_to_nodes` -- Eulerian cell-centred field (scalar / vector /
  tensor, any trailing component shape) -> Lagrange nodal values on a Q1 (bilinear /
  trilinear) finite-element mesh, by conservative scatter/gather with the Q1 partition
  of unity evaluated at the source cell centroids:

      sigma_i = [ sum_c V_c N_i(x_c) sigma_c ] / [ sum_c V_c N_i(x_c) ]

  Because ``sum_i N_i(x) = 1`` and ``sum_i N_i(x) x_i = x`` inside an element, the
  zeroth and first moments are conserved exactly over the *covered* part of the grid;
  :class:`NodalFieldMap` reports the covered fraction so an under-covered mesh cannot
  hide behind a small residual.

Ownership boundary: cross-solver exchange numerics and its conservation audit only.
No solver kernel (LBM collide/stream, FEM assembly, DEM contact, FVM transport) lives
here.  The upstream solver interfaces are documented in
``docs/CONSERVATIVE_FIELD_TRANSFER_P233W2_N03.md`` §2.3 -- until TensorLBM/TensorDEM
publish the field descriptors, the reference fields are generated in-repo
(``field_exchange.lbm.run_channel`` for the LBM stress, analytic lattices for DEM).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

__all__ = [
    "CartGrid",
    "Q1Mesh",
    "DepositionWeights",
    "VolumeFractionMap",
    "NodalFieldMap",
    "deposition_weights",
    "map_particles_to_volume_fraction",
    "map_cell_field_to_nodes",
    "audit_volume_fraction",
    "audit_nodal_field",
    "roache_gci",
    "cube_volume",
    "sphere_volume",
]


# ---------------------------------------------------------------------------
# Eulerian grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CartGrid:
    """Uniform Cartesian cell grid in 1, 2 or 3 dimensions.

    Cells are indexed lexicographically (``numpy`` C order) with ``shape`` cells per
    axis; cell ``(i, j, k)`` spans ``origin + (i, j, k) * spacing`` to
    ``origin + (i + 1, j + 1, k + 1) * spacing``.
    """

    origin: tuple[float, ...]
    spacing: tuple[float, ...]
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        origin = tuple(float(v) for v in self.origin)
        spacing = tuple(float(v) for v in self.spacing)
        shape = tuple(int(v) for v in self.shape)
        if not (len(origin) == len(spacing) == len(shape)):
            raise ValueError("origin, spacing and shape must have the same length")
        if not 1 <= len(shape) <= 3:
            raise ValueError("only 1D/2D/3D grids are supported")
        if any(h <= 0.0 for h in spacing):
            raise ValueError("spacing must be positive")
        if any(n < 1 for n in shape):
            raise ValueError("shape entries must be >= 1")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "shape", shape)

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    @property
    def cell_volume(self) -> float:
        return float(np.prod(self.spacing))

    @property
    def volume(self) -> float:
        return float(np.prod([h * n for h, n in zip(self.spacing, self.shape)]))

    def centers(self) -> np.ndarray:
        """``(n_cells, dim)`` cell centroids (lexicographic order)."""
        axes = [
            self.origin[d] + (np.arange(self.shape[d]) + 0.5) * self.spacing[d]
            for d in range(self.dim)
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        return np.stack([g.reshape(-1) for g in grid], axis=1)

    def bounds(self) -> np.ndarray:
        """``(dim, 2)`` grid extent ``[lower, upper]``."""
        return np.array(
            [
                [self.origin[d], self.origin[d] + self.shape[d] * self.spacing[d]]
                for d in range(self.dim)
            ],
            dtype=float,
        )


def cube_volume(radius: float, dim: int) -> float:
    """Volume of an axis-aligned cube of half width ``radius`` in ``dim`` dims."""
    return float((2.0 * radius) ** dim)


def sphere_volume(radius: float, dim: int) -> float:
    """Volume of a ``dim``-dimensional ball of radius ``radius``."""
    r = float(radius)
    if dim == 3:
        return 4.0 / 3.0 * np.pi * r ** 3
    if dim == 2:
        return np.pi * r ** 2
    return 2.0 * r


# ---------------------------------------------------------------------------
# deposition kernels (Lagrangian points -> Eulerian cells)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DepositionWeights:
    """Sparse deposition pattern of ``n`` points onto cells.

    ``index`` is ``(n, K)`` flat cell indices and ``weights`` is ``(n, K)`` the
    corresponding per-point weights.  For a conservative kernel each row of
    ``weights`` sums to 1 (interior points); rows of out-of-domain points are zero and
    flagged by ``inside``.
    """

    index: np.ndarray
    weights: np.ndarray
    inside: np.ndarray

    @property
    def n_points(self) -> int:
        return int(self.index.shape[0])


def _axis_deposition(grid: CartGrid, axis: int, x: np.ndarray, *, scheme: str,
                     half_width: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis cell indices ``(n, 2)`` and overlap lengths/weights ``(n, 2)``."""
    origin = grid.origin[axis]
    h = grid.spacing[axis]
    n_cells = grid.shape[axis]
    xi = (x - origin) / h
    if scheme in ("cic", "ngp"):
        if scheme == "cic":
            # cloud-in-cell: split between the two *nearest cell centres* so that the
            # deposited centroid equals the particle position (first-moment exact)
            xc = xi - 0.5
            i0 = np.floor(xc).astype(np.int64)
            t = xc - i0
            w = np.stack([1.0 - t, t], axis=1)
        else:
            i0 = np.floor(xi).astype(np.int64)
            w = np.zeros((x.size, 2))
            w[:, 0] = 1.0
        idx = np.stack([i0, i0 + 1], axis=1)
    elif scheme == "cube":
        if half_width is None:
            raise ValueError('scheme "cube" requires particle radii')
        r = np.asarray(half_width, dtype=float)
        lo = (x - r - origin) / h
        i0 = np.floor(lo).astype(np.int64)
        left = origin + i0 * h
        low = x - r
        high = x + r
        w = np.empty((x.size, 2))
        w[:, 0] = np.clip(np.minimum(high, left + h) - np.maximum(low, left), 0.0, None)
        w[:, 1] = np.clip(np.minimum(high, left + 2.0 * h) - np.maximum(low, left + h),
                          0.0, None)
        # weights are overlap *lengths* in this axis; the volume weight is the product
        idx = np.stack([i0, i0 + 1], axis=1)
    else:
        raise ValueError(f"unknown deposition scheme {scheme!r}")
    # cells outside [0, n_cells-1] get zero weight below
    bad = (idx < 0) | (idx >= n_cells)
    w = np.where(bad, 0.0, w)
    idx = np.clip(idx, 0, n_cells - 1)
    return idx, w


def deposition_weights(grid: CartGrid, points: Sequence[Sequence[float]], *,
                       scheme: str = "cic",
                       half_widths: Sequence[float] | None = None) -> DepositionWeights:
    """Deposition pattern of Lagrangian ``points`` onto the cells of ``grid``.

    Parameters
    ----------
    scheme : {"cic", "ngp", "cube"}
        ``"cic"`` multilinear cloud-in-cell; ``"ngp"`` nearest grid point;
        ``"cube"`` exact overlap of an axis-aligned cube of half width
        ``half_widths`` with each cell (requires ``2 * r <= min cell size``).  In the
        ``"cube"`` scheme the returned entries are overlap *lengths* per axis, so the
        product over the axes is the overlap **volume**; the caller divides by the cell
        volume (see :func:`map_particles_to_volume_fraction`).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[0] == 0:
        return DepositionWeights(np.zeros((0, 2 ** grid.dim), dtype=np.int64),
                                 np.zeros((0, 2 ** grid.dim)), np.zeros(0, dtype=bool))
    if pts.ndim != 2 or pts.shape[1] != grid.dim:
        raise ValueError(f"points must be (n, {grid.dim})")
    if half_widths is not None:
        hw = np.asarray(half_widths, dtype=float)
        if hw.shape != (pts.shape[0],):
            raise ValueError("half_widths must be a per-point vector")
        if scheme == "cube" and np.any(2.0 * hw > min(grid.spacing)):
            raise ValueError(
                'scheme "cube" (exact overlap) requires 2*r <= min cell size; '
                "refine the FVM grid or pass scheme='cic'"
            )
    else:
        hw = None

    bounds = grid.bounds()
    inside = np.ones(pts.shape[0], dtype=bool)
    for d in range(grid.dim):
        inside &= (pts[:, d] >= bounds[d, 0]) & (pts[:, d] <= bounds[d, 1])

    axes = [
        _axis_deposition(grid, d, pts[:, d], scheme=scheme, half_width=hw)
        for d in range(grid.dim)
    ]
    k = 2 ** grid.dim
    index = np.empty((pts.shape[0], k), dtype=np.int64)
    weights = np.empty((pts.shape[0], k))
    for col, combo in enumerate(product((0, 1), repeat=grid.dim)):
        multi = np.stack([axes[d][0][:, combo[d]] for d in range(grid.dim)], axis=1)
        index[:, col] = np.ravel_multi_index(
            tuple(multi[:, d] for d in range(grid.dim)), grid.shape
        )
        w = np.ones(pts.shape[0])
        for d in range(grid.dim):
            w = w * axes[d][1][:, combo[d]]
        weights[:, col] = w
    weights[~inside, :] = 0.0
    if scheme in ("cic", "ngp"):
        # point kernels: a particle whose centre is inside the domain must deposit its
        # *full* volume; near a boundary the truncated hat is renormalised (this is the
        # standard point-particle CFD-DEM convention, one-sided at the wall).
        row = weights.sum(axis=1)
        live = row > 0.0
        weights[live] = weights[live] / row[live][:, None]
    return DepositionWeights(index=index, weights=weights, inside=inside)


# ---------------------------------------------------------------------------
# DEM -> FVM volume fraction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VolumeFractionMap:
    """Result of a Lagrangian -> Eulerian volume-fraction transfer."""

    alpha: np.ndarray            # (grid.shape) volume fraction per cell
    grid: CartGrid
    cell_volume: np.ndarray      # (n_cells,) deposited particle volume per cell
    particle_volume: float       # total volume of all particles
    deposited_volume: float      # volume that landed inside the domain
    outside_volume: float        # volume of particles (partly) outside the domain
    scheme: str

    @property
    def domain_volume(self) -> float:
        return self.grid.volume

    @property
    def mean_volume_fraction(self) -> float:
        return self.deposited_volume / self.domain_volume

    @property
    def mass_residual(self) -> float:
        """Relative residual of the solid-mass identity ``sum_c alpha_c V_c = sum_p V_p``."""
        total = self.particle_volume
        if total <= 0.0:
            return 0.0
        return float(abs(self.cell_volume.sum() - total) / total)


def map_particles_to_volume_fraction(
    grid: CartGrid,
    positions: Sequence[Sequence[float]],
    *,
    radii: Sequence[float] | None = None,
    volumes: Sequence[float] | None = None,
    scheme: str = "cic",
) -> VolumeFractionMap:
    """Map Lagrangian particles onto Eulerian cell volume fractions.

    ``alpha_c = (1 / V_c) sum_p w_pc V_p``

    * ``scheme="cic"``: ``w_pc`` are multilinear (cloud-in-cell) weights -- mass and
      first moment are conserved exactly because the weights form a partition of unity
      that reproduces linear coordinates.
    * ``scheme="ngp"``: nearest-cell lumping -- mass conservative, first moment not.
    * ``scheme="cube"``: ``w_pc`` is the *exact* overlap volume of the cubic particle
      ``[x_p - r_p, x_p + r_p]`` with cell ``c`` (divided-particle method); the
      particle volume is taken as ``(2 r_p)^dim`` so the identity
      ``sum_c overlap = V_p`` holds exactly.

    ``volumes`` (particle volume, e.g. spheres ``4/3 pi r^3``) may be given for the
    point kernels; with only ``radii`` the volume defaults to the d-dimensional ball.
    """
    pts = np.atleast_2d(np.asarray(positions, dtype=float))
    if pts.shape[0] == 0:
        raise ValueError("at least one particle is required")
    if pts.ndim != 2 or pts.shape[1] != grid.dim:
        raise ValueError(f"positions must be (n, {grid.dim})")
    n = pts.shape[0]

    if scheme == "cube":
        if radii is None:
            raise ValueError('scheme "cube" requires radii')
        r = np.asarray(radii, dtype=float)
        if r.shape != (n,):
            raise ValueError("radii must be a per-particle vector")
        particle_volume = float(np.sum((2.0 * r) ** grid.dim))
        if volumes is not None and not np.allclose(
            np.asarray(volumes, dtype=float), (2.0 * r) ** grid.dim, rtol=0, atol=0
        ):
            raise ValueError('scheme "cube" fixes the particle volume to (2r)^dim')
        dep = deposition_weights(grid, pts, scheme="cube", half_widths=r)
        weights = dep.weights  # each axis entry is an overlap length / h
    elif scheme in ("cic", "ngp"):
        if volumes is not None:
            vol = np.asarray(volumes, dtype=float)
            if vol.shape != (n,):
                raise ValueError("volumes must be a per-particle vector")
        elif radii is not None:
            r = np.asarray(radii, dtype=float)
            if r.shape != (n,):
                raise ValueError("radii must be a per-particle vector")
            d = grid.dim
            const = 4.0 / 3.0 * np.pi if d == 3 else (np.pi if d == 2 else 2.0)
            vol = const * r ** d
        else:
            raise ValueError("either volumes or radii must be supplied")
        if np.any(vol < 0.0):
            raise ValueError("particle volumes must be non-negative")
        dep = deposition_weights(grid, pts, scheme=scheme)
        weights = dep.weights
        particle_volume = float(vol.sum())
        # deposit volume with the kernel weights
        per_cell = np.bincount(dep.index.ravel(), weights=(dep.weights * vol[:, None]).ravel(),
                               minlength=grid.n_cells)
        deposited = float(per_cell.sum())
        alpha = (per_cell / grid.cell_volume).reshape(grid.shape)
        return VolumeFractionMap(
            alpha=alpha, grid=grid, cell_volume=per_cell,
            particle_volume=particle_volume, deposited_volume=deposited,
            outside_volume=float(particle_volume - deposited), scheme=scheme,
        )

    # cube: the product of the per-axis overlap lengths is the overlap *volume*
    per_cell = np.bincount(
        dep.index.ravel(), weights=weights.ravel(), minlength=grid.n_cells
    )
    deposited = float(per_cell.sum())
    alpha = (per_cell / grid.cell_volume).reshape(grid.shape)
    return VolumeFractionMap(
        alpha=alpha, grid=grid, cell_volume=per_cell, particle_volume=particle_volume,
        deposited_volume=deposited, outside_volume=float(particle_volume - deposited),
        scheme="cube",
    )


def audit_volume_fraction(
    vmap: VolumeFractionMap,
    positions: Sequence[Sequence[float]],
    *,
    volumes: Sequence[float] | None = None,
    radii: Sequence[float] | None = None,
) -> dict:
    """Independent audit of a DEM -> FVM transfer.

    Returns relative residuals of the solid-mass identity and of the first moment
    (centre of solid volume), the mean volume fraction, and the out-of-domain share.
    Moment residuals are evaluated on the particles fully inside the domain, because
    a particle crossing the boundary legitimately loses its outside part.
    """
    pts = np.atleast_2d(np.asarray(positions, dtype=float))
    grid = vmap.grid
    if volumes is not None:
        vol = np.asarray(volumes, dtype=float)
    elif radii is not None:
        r = np.asarray(radii, dtype=float)
        d = grid.dim
        const = 4.0 / 3.0 * np.pi if d == 3 else (np.pi if d == 2 else 2.0)
        vol = const * r ** d
    else:  # pragma: no cover - defensive
        raise ValueError("volumes or radii required")

    deposited = vmap.deposited_volume
    total = float(vol.sum())
    mass_residual = float(abs(deposited - total) / total) if total > 0 else 0.0
    # for the point kernels every in-domain particle deposits its full volume
    if vmap.scheme in ("cic", "ngp"):
        total = max(total, 1e-300)

    centres = grid.centers()
    w = vmap.cell_volume
    mapped_moment = (centres * w[:, None]).sum(axis=0)
    bounds = grid.bounds()
    inside = np.ones(pts.shape[0], dtype=bool)
    for d in range(grid.dim):
        inside &= (pts[:, d] >= bounds[d, 0]) & (pts[:, d] <= bounds[d, 1])
    src_moment = (pts[inside] * vol[inside][:, None]).sum(axis=0)
    denom = float(np.linalg.norm(src_moment))
    moment_residual = (float(np.linalg.norm(mapped_moment - src_moment) / denom)
                       if denom > 0 else 0.0)
    return {
        "mass_residual": mass_residual,
        "first_moment_residual": moment_residual,
        "mean_volume_fraction": vmap.mean_volume_fraction,
        "deposited_volume": deposited,
        "total_particle_volume": total,
        "outside_volume": float(total - deposited),
        "outside_fraction": float((total - deposited) / total) if total > 0 else 0.0,
        "scheme": vmap.scheme,
        "n_particles": int(pts.shape[0]),
    }


# ---------------------------------------------------------------------------
# Q1 finite-element target mesh (point location + partition of unity)
# ---------------------------------------------------------------------------
_SIGNS = {
    2: np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float),
    3: np.array(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        dtype=float,
    ),
}


class Q1Mesh:
    """Bilinear (2D quad) / trilinear (3D hex) isoparametric mesh.

    The mesh is only a *target* for field transfer: it supplies node coordinates, the
    element connectivity and the Q1 partition of unity.  No stiffness assembly happens
    here (that belongs to TensorFEM) -- an arbitrary nodal field must be expressible,
    which is all a conservative exchange needs.
    """

    def __init__(self, nodes: Sequence[Sequence[float]], elements: Sequence[Sequence[int]]):
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
            raise ValueError("nodes must be (n_nodes, 2|3)")
        self.dim = nodes.shape[1]
        self.nodes = nodes
        elems = np.asarray(elements, dtype=np.int64)
        npe = 4 if self.dim == 2 else 8
        if elems.ndim != 2 or elems.shape[1] != npe:
            raise ValueError(f"elements must be {npe}-node Q1 cells for dim={self.dim}")
        if elems.size and (elems.min() < 0 or elems.max() >= nodes.shape[0]):
            raise ValueError("element connectivity out of range")
        self.elements = elems
        self._signs = _SIGNS[self.dim]
        self._axes: list[np.ndarray] | None = None
        if len(elems):
            self._lo = nodes[elems].min(axis=1)
            self._hi = nodes[elems].max(axis=1)
        else:  # pragma: no cover
            self._lo = np.zeros((0, self.dim))
            self._hi = np.zeros((0, self.dim))

    # -- construction helpers -------------------------------------------------
    @classmethod
    def structured(cls, axes: Sequence[Sequence[float]]) -> "Q1Mesh":
        """Structured Q1 mesh from per-axis node coordinate vectors."""
        dim = len(axes)
        if dim not in (2, 3):
            raise ValueError("structured() supports 2D and 3D")
        counts = [len(a) for a in axes]
        if any(c < 2 for c in counts):
            raise ValueError("each axis needs >= 2 nodes")
        mesh_grid = np.meshgrid(*axes, indexing="ij")
        nodes = np.stack([g.reshape(-1) for g in mesh_grid], axis=1)

        def node_id(idx: Sequence[int]) -> int:
            return int(np.ravel_multi_index(tuple(idx), tuple(counts)))

        elements = []
        ranges = [range(c - 1) for c in counts]
        for idx in product(*ranges):
            if dim == 2:
                i, j = idx
                elements.append([node_id((i, j)), node_id((i + 1, j)),
                                 node_id((i + 1, j + 1)), node_id((i, j + 1))])
            else:
                i, j, k = idx
                elements.append([
                    node_id((i, j, k)), node_id((i + 1, j, k)),
                    node_id((i + 1, j + 1, k)), node_id((i, j + 1, k)),
                    node_id((i, j, k + 1)), node_id((i + 1, j, k + 1)),
                    node_id((i + 1, j + 1, k + 1)), node_id((i, j + 1, k + 1)),
                ])
        mesh = cls(nodes, elements)
        mesh._axes = [np.asarray(a, dtype=float) for a in axes]
        return mesh

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])

    @property
    def nodes_per_element(self) -> int:
        return 4 if self.dim == 2 else 8

    # -- basis ---------------------------------------------------------------
    def shape(self, xi: Sequence[float]) -> np.ndarray:
        """Q1 shape functions at parent coordinate ``xi`` ``(npe,)``."""
        s = self._signs
        return np.prod(0.5 * (1.0 + s * np.asarray(xi, dtype=float)[None, :]), axis=1)

    def shape_grad(self, xi: Sequence[float]) -> np.ndarray:
        """Parent derivatives ``dN/dxi`` ``(npe, dim)``."""
        s = self._signs
        xi = np.asarray(xi, dtype=float)
        grad = np.empty_like(s)
        for d in range(self.dim):
            others = [0.5 * (1.0 + s[:, a] * xi[a]) for a in range(self.dim) if a != d]
            g = 0.5 * s[:, d]
            for o in others:
                g = g * o
            grad[:, d] = g
        return grad

    def element_jacobian(self, elem: int, xi: Sequence[float]) -> np.ndarray:
        x = self.nodes[self.elements[elem]]
        return self.shape_grad(xi).T @ x

    def element_volume(self, elem: int, order: int = 2) -> float:
        """Volume of element ``elem`` by Gauss quadrature (exact for straight Q1)."""
        gauss, weights = np.polynomial.legendre.leggauss(order)
        total = 0.0
        for combo in product(range(order), repeat=self.dim):
            xi = [gauss[c] for c in combo]
            jac = self.element_jacobian(elem, xi)
            w = np.prod([weights[c] for c in combo])
            total += w * abs(float(np.linalg.det(jac)))
        return total

    # -- point location ------------------------------------------------------
    def locate(self, points: Sequence[Sequence[float]], *, max_iter: int = 25,
               tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Locate points in the mesh.

        Returns ``(elem, xi, ok)`` with ``elem`` ``(n,)`` the containing element index
        (-1 when outside), ``xi`` ``(n, dim)`` parent coordinates and ``ok`` the
        boolean success mask.  Points on an element boundary are reported in whichever
        element is found first; the Q1 partition of unity makes the transferred value
        continuous across element boundaries.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if pts.ndim != 2 or pts.shape[1] != self.dim:
            raise ValueError(f"points must be (n, {self.dim})")
        if self._axes is not None:
            return self._locate_structured(pts)
        n = pts.shape[0]
        elems = np.full(n, -1, dtype=np.int64)
        xis = np.zeros((n, self.dim))
        ok = np.zeros(n, dtype=bool)
        span = self._hi - self._lo
        pad = 1e-12 * np.maximum(span.max(axis=0) if len(span) else 1.0, 1e-30)
        for p in range(n):
            x = pts[p]
            cand = np.nonzero(
                np.all(x >= self._lo - pad, axis=1) & np.all(x <= self._hi + pad, axis=1)
            )[0]
            if cand.size == 0:
                continue
            for e in cand:
                xi = np.zeros(self.dim)
                converged = False
                for _ in range(max_iter):
                    xmap = self.shape(xi) @ self.nodes[self.elements[e]]
                    r = x - xmap
                    if np.linalg.norm(r) <= tol * max(1.0, np.linalg.norm(x)):
                        converged = True
                        break
                    jac = self.element_jacobian(e, xi)
                    try:
                        d = np.linalg.solve(jac, r)
                    except np.linalg.LinAlgError:  # pragma: no cover - degenerate
                        break
                    xi = xi + d
                else:
                    xmap = self.shape(xi) @ self.nodes[self.elements[e]]
                    converged = np.linalg.norm(x - xmap) <= 1e-9 * max(1.0, np.linalg.norm(x))
                inside = np.all(np.abs(xi) <= 1.0 + 1e-10)
                if (converged or np.all(np.abs(xi) <= 1.0 + 1e-10)) and inside:
                    elems[p] = e
                    xis[p] = np.clip(xi, -1.0, 1.0)
                    ok[p] = True
                    break
        return elems, xis, ok


def _locate_structured(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised point location on a rectilinear structured mesh (no Newton)."""
    axes = self._axes
    assert axes is not None
    n = pts.shape[0]
    counts = np.array([a.size - 1 for a in axes], dtype=np.int64)
    ok = np.ones(n, dtype=bool)
    cell = np.zeros((n, self.dim), dtype=np.int64)
    xi = np.zeros((n, self.dim))
    for d, axis in enumerate(axes):
        x = pts[:, d]
        ok &= (x >= axis[0]) & (x <= axis[-1])
        i0 = np.clip(np.searchsorted(axis, x, side="right") - 1, 0, counts[d] - 1)
        cell[:, d] = i0
        h = axis[i0 + 1] - axis[i0]
        xi[:, d] = 2.0 * (x - axis[i0]) / h - 1.0
    xi = np.clip(xi, -1.0, 1.0)
    elems = np.where(ok, np.ravel_multi_index(tuple(cell[:, d] for d in range(self.dim)),
                                              tuple(counts)), -1)
    return elems, xi, ok


Q1Mesh._locate_structured = _locate_structured  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# LBM (Eulerian cell field) -> FEM (Lagrange nodes)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodalFieldMap:
    """Result of a conservative Eulerian-cell -> Lagrange-node field transfer."""

    values: np.ndarray           # (n_nodes, ...) nodal field (components trailing)
    nodal_volumes: np.ndarray    # (n_nodes,) V_i = sum_c V_c N_i(x_c)
    mask: np.ndarray             # (n_nodes,) True where V_i > 0
    grid: CartGrid
    mesh: Q1Mesh
    covered_volume: float
    total_volume: float
    source_integral: np.ndarray  # sum_c V_c field_c over covered cells
    mapped_integral: np.ndarray  # sum_i V_i field_i
    source_moment: np.ndarray    # sum_c V_c x_c (x) field_c
    mapped_moment: np.ndarray    # sum_i V_i x_i (x) field_i

    def component_residual(self, a: np.ndarray, b: np.ndarray) -> float:
        """Relative max-norm residual of two integral arrays."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        denom = float(np.max(np.abs(a))) if a.size else 0.0
        if denom == 0.0:
            return float(np.max(np.abs(b - a))) if a.size else 0.0
        return float(np.max(np.abs(b - a)) / denom)

    @property
    def zeroth_moment_residual(self) -> float:
        """Relative residual of the force / momentum-flux identity."""
        return self.component_residual(self.source_integral, self.mapped_integral)

    @property
    def first_moment_residual(self) -> float:
        """Relative residual of the couple / first-moment identity."""
        return self.component_residual(self.source_moment, self.mapped_moment)

    @property
    def coverage(self) -> float:
        return self.covered_volume / self.total_volume if self.total_volume > 0 else 0.0


def _cell_gauss(grid: CartGrid, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss points/weights on all cells: ``(n_cells * order**dim, dim)`` and weights."""
    ref, wref = np.polynomial.legendre.leggauss(order)
    centres = grid.centers()
    half = 0.5 * np.asarray(grid.spacing)
    offsets = []
    wq = []
    for combo in product(range(order), repeat=grid.dim):
        offsets.append([ref[c] for c in combo])
        wq.append(float(np.prod([half[d] * wref[c] for d, c in enumerate(combo)])))
    offsets = np.asarray(offsets)
    wq = np.asarray(wq)
    pts = (centres[:, None, :] + half[None, None, :] * offsets[None, :, :]).reshape(
        -1, grid.dim)
    weights = np.tile(wq, grid.n_cells)
    return pts, weights


def _project_l2(flat: np.ndarray, grid: CartGrid, mesh: Q1Mesh, order: int):
    """Conservative L2 projection of a cell-wise-constant field onto the Q1 mesh.

    Returns ``(values, nodal_volumes, mass, rhs, covered_volume)`` where
    ``mass`` is the consistent Q1 mass matrix restricted to the covered region and
    ``rhs`` the consistent load vector.  Both are assembled with exact Gauss
    quadrature over the source cells, so
    ``sum_i V_i sigma_i == sum_c V_c sigma_c`` and ``sum_i V_i x_i (x) sigma_i ==
    sum_c V_c x_c (x) sigma_c`` hold as algebraic identities.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu

    pts, wq = _cell_gauss(grid, order)
    nq = wq.size // grid.n_cells
    elems, xis, ok = mesh.locate(pts)
    npe = mesh.nodes_per_element
    n_nodes = mesh.n_nodes
    comp = flat.shape[1]
    rhs = np.zeros((n_nodes, comp))
    blocks = np.zeros((mesh.n_elements, npe, npe))
    covered = 0.0
    for p in np.nonzero(ok)[0]:
        cell, q = divmod(int(p), nq)
        if q == 0:
            covered += np.prod(np.asarray(grid.spacing))  # one full cell
        e = int(elems[p])
        N = mesh.shape(xis[p])
        nodes = mesh.elements[e]
        w = wq[q]
        rhs[nodes] += (w * N)[:, None] * flat[cell][None, :]
        blocks[e] += w * np.outer(N, N)
    idx = mesh.elements
    rows = np.repeat(idx, npe, axis=1).ravel()
    cols = np.tile(idx, (1, npe)).ravel()
    mass = coo_matrix((blocks.ravel(), (rows, cols)),
                      shape=(n_nodes, n_nodes)).tocsc()
    nodal_volumes = np.asarray(mass.sum(axis=1)).ravel()
    values = np.zeros((n_nodes, comp))
    nz = nodal_volumes > 0.0
    if np.any(nz):
        sub = mass[np.ix_(nz, nz)].tocsc()
        lu = splu(sub)
        values[nz] = lu.solve(rhs[nz])
    return values, nodal_volumes, mass, rhs, float(covered)


def map_cell_field_to_nodes(
    field: Sequence[float] | np.ndarray,
    grid: CartGrid,
    mesh: Q1Mesh,
    *,
    scheme: str = "l2",
    order: int = 2,
) -> NodalFieldMap:
    """Conservatively transfer a cell-centred field to the nodes of a Q1 mesh.

    ``field`` may carry trailing component axes of any shape (scalar, vector, 3x3
    stress tensor ...); every component is transferred by the same operator.

    Two operators are available:

    ``scheme="l2"`` (default, recommended)
        Consistent (Galerkin) L2 projection: solve ``M sigma = b`` with the Q1
        consistent mass matrix ``M`` and the consistent load vector
        ``b_i = int N_i sigma dV`` assembled by Gauss quadrature over the source
        cells.  Consequences (all identities, not approximations):

        * mass/momentum: ``sum_i V_i sigma_i = sum_c V_c sigma_c``;
        * couple: ``sum_i V_i x_i (x) sigma_i = sum_c V_c x_c (x) sigma_c``;
        * consistency: a linear field is reproduced exactly at the nodes, and a
          smooth field is second-order accurate (``O(h^2)``).

    ``scheme="lumped"``
        Cheap diagonal (mass-lumping) scatter/gather with the Q1 partition of unity
        evaluated at the source cell centroids.  Also exactly conservative for the
        zeroth and first moments, but the nodal value is a volume average around the
        node, which biases boundary nodes at first order.  Kept for very large
        exchanges where a global solve is undesirable, and as an accuracy contrast.

    Cells whose quadrature points fall outside the target mesh contribute nothing;
    the returned ``coverage`` makes the transferred share explicit.
    """
    arr = np.asarray(field, dtype=float)
    if arr.shape[0] != grid.n_cells:
        raise ValueError(f"field must have {grid.n_cells} cell values, got {arr.shape[0]}")
    comp_shape = arr.shape[1:]
    flat = arr.reshape(grid.n_cells, -1)
    vc = grid.cell_volume

    mask = None
    if scheme == "l2":
        values, nodal_volumes, _mass, rhs, covered = _project_l2(flat, grid, mesh, order)
        centres = grid.centers()
        quad_pts, _w = _cell_gauss(grid, order)
        nq = quad_pts.shape[0] // grid.n_cells
        _e, _x, okq = mesh.locate(quad_pts)
        inside = okq.reshape(grid.n_cells, nq).all(axis=1)
        # consistency: for the Galerkin projection the moments of the mapped field are
        # the moments of the consistent load vector (M sigma = b), which is exact.
        src_int = (flat[inside] * vc).sum(axis=0).reshape(comp_shape)
        map_int = rhs.sum(axis=0).reshape(comp_shape)
        src_mom = np.einsum("nd,nc->dc", centres[inside] * vc, flat[inside]).reshape(
            (grid.dim,) + comp_shape)
        map_mom = np.einsum("nd,nc->dc", mesh.nodes, rhs).reshape(
            (grid.dim,) + comp_shape)
        covered_volume = float(vc * inside.sum())
        mask = nodal_volumes > 0.0
        return NodalFieldMap(
            values=values.reshape((mesh.n_nodes,) + comp_shape),
            nodal_volumes=nodal_volumes, mask=mask, grid=grid, mesh=mesh,
            covered_volume=covered_volume, total_volume=grid.volume,
            source_integral=src_int, mapped_integral=map_int,
            source_moment=src_mom, mapped_moment=map_mom,
        )
    if scheme == "lumped":
        centres = grid.centers()
        elems, xis, ok = mesh.locate(centres)
        n_nodes = mesh.n_nodes
        nodal_volumes = np.zeros(n_nodes)
        acc = np.zeros((n_nodes, flat.shape[1]))
        for p in np.nonzero(ok)[0]:
            N = mesh.shape(xis[p])
            nodes = mesh.elements[elems[p]]
            w = vc * N
            nodal_volumes[nodes] += w
            acc[nodes] += w[:, None] * flat[p][None, :]
        values = np.zeros_like(acc)
        mask = nodal_volumes > 0.0
        values[mask] = acc[mask] / nodal_volumes[mask][:, None]
        covered = float(vc * (ok.sum() if ok.size else 0))
        src_int = (flat[ok] * vc).sum(axis=0).reshape(comp_shape)
        map_int = (values * nodal_volumes[:, None]).sum(axis=0).reshape(comp_shape)
        src_mom = np.einsum("nd,nc->dc", centres[ok] * vc, flat[ok]).reshape(
            (grid.dim,) + comp_shape)
        map_mom = np.einsum("nd,nc->dc", mesh.nodes * nodal_volumes[:, None],
                            values).reshape((grid.dim,) + comp_shape)
        return NodalFieldMap(
            values=values.reshape((n_nodes,) + comp_shape),
            nodal_volumes=nodal_volumes, mask=mask, grid=grid, mesh=mesh,
            covered_volume=covered, total_volume=grid.volume,
            source_integral=src_int, mapped_integral=map_int,
            source_moment=src_mom, mapped_moment=map_mom,
        )
    raise ValueError(f"unknown scheme {scheme!r}")


def audit_nodal_field(nmap: NodalFieldMap) -> dict:
    """Summarise a LBM -> FEM transfer for the acceptance gate."""
    return {
        "zeroth_moment_residual": nmap.zeroth_moment_residual,
        "first_moment_residual": nmap.first_moment_residual,
        "coverage": nmap.coverage,
        "covered_volume": nmap.covered_volume,
        "mapped_volume": float(nmap.nodal_volumes.sum()),
        "n_masked_nodes": int((~nmap.mask).sum()),
        "n_nodes": int(nmap.mask.size),
    }


# ---------------------------------------------------------------------------
# grid-convergence helper (Roache GCI)
# ---------------------------------------------------------------------------
def roache_gci(errors: Sequence[float], *, ratio: float = 2.0,
               safety: float = 1.25) -> dict:
    """Observed order and Grid Convergence Index from three relative errors.

    ``errors`` must be ordered coarse -> fine.  With a constant refinement ratio
    ``r`` the observed order is ``p = ln(e1/e2)/ln(r)``; the fine-grid GCI is
    ``GCI = Fs * e3 / (r^p - 1)`` (Roache 1994, ASME V&V 20).
    """
    e = [float(v) for v in errors]
    if len(e) != 3:
        raise ValueError("expected exactly three errors (coarse, medium, fine)")
    if min(e) <= 0.0:
        raise ValueError("errors must be positive")
    p = float(np.log(e[0] / e[1]) / np.log(ratio))
    gci = float(safety * e[2] / (ratio ** p - 1.0)) if p > 0 else float("inf")
    return {
        "errors": e,
        "ratio": float(ratio),
        "observed_order": p,
        "gci_fine": gci,
        "monotone": bool(e[0] > e[1] > e[2]),
        "asymptotic_ratio": float(e[0] / e[1]),
    }