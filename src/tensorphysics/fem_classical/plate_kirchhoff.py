"""Kirchhoff simply-supported square plate solver.

Uses the bicubic Hermite (Bogner-Fox-Schmit) plate-bending element from the
existing ``plate_buckling`` module, but provides a clean standalone interface
for the SSSS verification case with Navier-series reference.

Each node has 4 DOFs: (w, θx, θy, w_xy) where
    θx = ∂w/∂y,  θy = −∂w/∂x,  w_xy = ∂²w/∂x∂y.

For a simply-supported square plate under uniform load q, the analytical
center deflection is given by the Navier double Fourier series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tensorphysics.plate_buckling import (
    PlateGeometry,
    PlateModel,
    navier_simply_supported_deflection,
)

__all__ = ["KirchhoffPlateSSSS"]


@dataclass(frozen=True)
class KirchhoffPlateSSSS:
    """Simply-supported square plate under uniform pressure.

    Parameters
    ----------
    a : float
        Plate side length [m] (square, so b = a).
    t : float
        Plate thickness [m].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    n_elem : int
        Number of elements per side (total mesh = n_elem × n_elem).
    """

    a: float
    t: float
    E: float
    nu: float
    n_elem: int

    @property
    def D(self) -> float:
        """Flexural rigidity."""
        return self.E * self.t ** 3 / (12.0 * (1.0 - self.nu ** 2))

    def _make_geom(self) -> PlateGeometry:
        return PlateGeometry(
            a=self.a, b=self.a, t=self.t,
            E=self.E, nu=self.nu,
            sigma_y=1e12,  # effectively elastic (no yielding)
            nx=self.n_elem, ny=self.n_elem,
        )

    def solve(self, q: float) -> Tuple[np.ndarray, PlateModel]:
        """Solve the linear plate problem under uniform pressure q.

        Parameters
        ----------
        q : float
            Uniform pressure [Pa] (positive downward).

        Returns
        -------
        u : ndarray
            Full displacement vector (6 DOFs per node).
        model : PlateModel
            The assembled model (for post-processing).
        """
        geom = self._make_geom()
        model = PlateModel(geom, bc="SSSS", validate=False)
        u = model.solve_linear(model.pressure_load(q))
        return u, model

    def center_deflection(self, q: float) -> float:
        """Compute center deflection under uniform load q.

        Returns
        -------
        float
            Center deflection [m] (positive downward).
        """
        u, model = self.solve(q)
        geom = model.geom
        # Center node index for odd n_elem: middle of the grid
        # For even n_elem: average of 4 center nodes
        nx, ny = geom.nx, geom.ny
        nid = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)

        if nx % 2 == 1 and ny % 2 == 1:
            # Odd: exact center node exists
            ci, cj = nx // 2, ny // 2
            center_node = nid[cj, ci]
            return float(u[6 * center_node + 2])  # w DOF
        else:
            # Even: average of 4 center nodes
            ci = nx // 2
            cj = ny // 2
            nodes = [nid[cj, ci], nid[cj, ci + 1], nid[cj + 1, ci], nid[cj + 1, ci + 1]]
            w_vals = [u[6 * n + 2] for n in nodes]
            return float(np.mean(w_vals))

    def navier_reference(self, q: float, n_terms: int = 25) -> float:
        """Navier-series exact center deflection."""
        return navier_simply_supported_deflection(
            self._make_geom(), q, n_terms=n_terms
        )

    def condition_number_no_bc(self, n_elem_small: int = 4) -> float:
        """Condition number of stiffness matrix WITHOUT BCs (negative control).

        Uses a small mesh for efficiency.
        """
        geom = PlateGeometry(
            a=self.a, b=self.a, t=self.t,
            E=self.E, nu=self.nu,
            sigma_y=1e12,
            nx=n_elem_small, ny=n_elem_small,
        )
        model = PlateModel(geom, bc="SSSS", validate=False)
        # Get the elastic stiffness without applying BCs
        K = model.elastic_stiffness()
        # Convert to dense for condition number (small mesh)
        K_dense = K.toarray() if sp.issparse(K) else K
        s = np.linalg.svd(K_dense, compute_uv=False)
        return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
