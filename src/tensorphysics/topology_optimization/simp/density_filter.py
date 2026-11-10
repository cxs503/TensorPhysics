"""Linear (hat) density filter for SIMP topology optimisation.

Following Sigmund (1994) / Bourdin (2001), the physical (*filtered*) density of
element ``e`` is the normalised weighted average of the design densities in a
neighbourhood of radius ``r``::

    H_ij = max(0, r - |x_i - x_j|),        rho_tilde_i = sum_j H_ij rho_j / sum_j H_ij

Distances are measured in element lengths, so ``r`` is expressed in elements
(``r = 1.5`` means 1.5 element sizes).  The filter regularises the problem:
the minimum length scale is ``~ r`` and checkerboard modes cannot be represented.

Because ``rho_tilde = P rho`` is linear, sensitivities project with the
transpose::

    dC/drho = P^T dC/drho_tilde

which is what :func:`filter_sensitivity` implements.

The :func:`checkerboard_index` diagnostic quantifies element-wise alternation:
for every interior 2x2 block it measures the discrete mixed second difference
``rho_ij - rho_i,j+1 - rho_i+1,j + rho_i+1,j+1`` and normalises by its maximum
possible magnitude (2 for a 0/1 field), so a perfect checkerboard scores 1 and
a smooth field scores ~0.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = [
    "DensityFilter",
    "checkerboard_index",
    "filter_sensitivity",
]


class DensityFilter:
    """Sparse linear density filter on an ``(nely, nelx)`` element grid."""

    def __init__(self, nelx: int, nely: int, radius: float) -> None:
        if radius < 0.0:
            raise ValueError("filter radius must be non-negative")
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.radius = float(radius)
        self._H = self._build_weights()
        self._row_sum = np.asarray(self._H.sum(axis=1)).ravel()

    # -- construction -------------------------------------------------------
    def _build_weights(self) -> sp.csr_matrix:
        nelx, nely, r = self.nelx, self.nely, self.radius
        n = nelx * nely
        if r <= 0.0:
            return sp.identity(n, format="csr")
        k = int(np.ceil(r))
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for ey in range(nely):
            for ex in range(nelx):
                i = ey * nelx + ex
                for dy in range(-k, k + 1):
                    ny = ey + dy
                    if ny < 0 or ny >= nely:
                        continue
                    for dx in range(-k, k + 1):
                        nx = ex + dx
                        if nx < 0 or nx >= nelx:
                            continue
                        dist = np.hypot(dx, dy)
                        w = r - dist
                        if w > 0.0:
                            rows.append(i)
                            cols.append(ny * nelx + nx)
                            vals.append(float(w))
        return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    # -- public API ---------------------------------------------------------
    @property
    def n_elems(self) -> int:
        return self.nelx * self.nely

    @property
    def matrix(self) -> sp.csr_matrix:
        """Normalised operator ``P`` such that ``rho_tilde = P @ rho``."""
        return sp.diags(1.0 / self._row_sum) @ self._H

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply the filter to a flat or ``(nely, nelx)`` density field."""
        flat = np.asarray(rho, dtype=float).ravel()
        if flat.size != self.n_elems:
            raise ValueError(f"expected {self.n_elems} densities, got {flat.size}")
        return self.matrix @ flat

    def weight_sum(self) -> np.ndarray:
        """Per-element neighbourhood weight sums ``sum_j H_ij``."""
        return self._row_sum.copy()

    def is_identity(self, tol: float = 1e-12) -> bool:
        P = self.matrix
        eye = sp.identity(self.n_elems, format="csr")
        diff = abs(P - eye)
        return diff.max() <= tol if diff.nnz else True


def filter_sensitivity(dC_drho_tilde: np.ndarray, dens_filter: DensityFilter | None) -> np.ndarray:
    """Project ``dC/drho_tilde`` to ``dC/drho`` by the transpose chain rule."""
    g = np.asarray(dC_drho_tilde, dtype=float).ravel()
    if dens_filter is None:
        return g.copy()
    return dens_filter.matrix.T @ g


def checkerboard_index(rho: np.ndarray) -> float:
    """Normalised element-wise alternation index in ``[0, 1]``.

    ``mean |mixed second difference| / 2`` over all interior 2x2 blocks.  A
    0/1 checkerboard gives 1.0, a constant field gives 0.0.
    """
    r = np.asarray(rho, dtype=float)
    if r.ndim == 1:
        raise ValueError("checkerboard_index expects an (nely, nelx) array")
    if r.shape[0] < 3 or r.shape[1] < 3:
        raise ValueError("checkerboard index needs at least a 3x3 element grid")
    d = r[:-1, :-1] - r[:-1, 1:] - r[1:, :-1] + r[1:, 1:]
    return float(np.mean(np.abs(d)) / 2.0)