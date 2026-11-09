"""2-D Kirchhoff plate for the bow-flare panel (mass-spring lattice).

The panel is a simply supported square plate of side ``L`` and thickness
``t``.  It is discretised as an ``N x N`` node lattice with a single
out-of-plane DOF ``w`` per node.  The bending operator is the discrete
biharmonic ``nabla^4 = (nabla^2)^2`` built from the 5-point Laplacian
applied twice (a 9-point stencil), which is the standard discrete Kirchhoff
plate.  The boundary nodes are simply supported (``w = 0``).

The lattice is used for two purposes:

* a **static** circular-load solve that yields the effective center stiffness
  ``k_eff = F / w_center`` (the panel's share of the contact stiffness);
* a **quasi-static** solve at the peak load that yields the 2-D pressure /
  deflection / stress field ``p(x, y)`` for the report.

Mesh convergence
----------------
``k_eff`` is a functional of the mesh; refining ``N`` drives it to the
continuum value.  The acceptance harness checks that the resulting ``P_max``
moves by < 3 % across three refinement levels.

Reference
---------
* Timoshenko & Woinowsky, *Theory of Plates and Shells* (1959): simply
  supported square plate, central point load, center deflection
  ``delta = 0.0056 F L^2 / D`` (used as the analytic cross-check).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["PlateFEM", "PlateFEMResult"]


@dataclass
class PlateFEMResult:
    """Result of a static circular-load solve."""

    n: int
    k_eff: float                 # N/m, F / w_center
    w_center: float              # m
    total_force: float           # N
    w_field: np.ndarray          # (N, N) deflection field [m]
    p_field: np.ndarray          # (N, N) applied pressure field [Pa]
    sigma_field: np.ndarray      # (N, N) top-surface bending stress [Pa]
    strain_energy: float         # J, 0.5 * p^T w


class PlateFEM:
    """Discrete Kirchhoff plate (simply supported) with a circular load."""

    def __init__(self, side_m: float, thickness_m: float, E: float,
                 nu: float, rho: float, n: int = 16):
        if n < 4:
            raise ValueError("n must be >= 4")
        self.L = float(side_m)
        self.t = float(thickness_m)
        self.E = float(E)
        self.nu = float(nu)
        self.rho = float(rho)
        self.n = int(n)
        self.h = self.L / (self.n - 1)
        self.D = self.E * self.t**3 / (12.0 * (1.0 - self.nu**2))
        self._build()

    # -- assembly ---------------------------------------------------------
    def _build(self) -> None:
        n = self.n
        # interior node index map (boundary is simply supported, w = 0)
        self._idx = {}
        k = 0
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                self._idx[(i, j)] = k
                k += 1
        self.n_int = k
        # 5-point Laplacian (dimensionless), applied twice for nabla^4.
        lap = {
            (0, 0): -4.0,
            (1, 0): 1.0, (-1, 0): 1.0,
            (0, 1): 1.0, (0, -1): 1.0,
        }
        # biharmonic stencil = lap * lap
        bio = {}
        for (di, dj), a in lap.items():
            for (di2, dj2), b in lap.items():
                bio[(di + di2, dj + dj2)] = bio.get((di + di2, dj + dj2), 0.0) + a * b
        rows, cols, vals = [], [], []
        for (i, j), row in self._idx.items():
            for (di, dj), c in bio.items():
                ni, nj = i + di, j + dj
                if 1 <= ni <= n - 2 and 1 <= nj <= n - 2:
                    col = self._idx[(ni, nj)]
                else:
                    # simply supported: boundary w = 0 -> no contribution to K
                    # (the reaction is carried by the support, not the interior)
                    continue
                rows.append(row)
                cols.append(col)
                vals.append(c)
        # store as dense for the small systems we solve here
        K = np.zeros((self.n_int, self.n_int))
        for r, c, v in zip(rows, cols, vals):
            K[r, c] += v
        self.K = self.D / self.h**4 * K
        # mass per node (lumped)
        self.m_node = self.rho * self.t * self.h**2
        # node coordinates
        self.x = np.linspace(0.0, self.L, n)
        self._ij = np.array(sorted(self._idx.keys()))

    # -- load -------------------------------------------------------------
    def _circular_load(self, radius_m: float, pressure_pa: float) -> np.ndarray:
        """Uniform pressure over a central circle of radius ``radius_m``."""
        f = np.zeros(self.n_int)
        a2 = radius_m**2
        for (i, j), row in self._idx.items():
            x = self.x[i] - self.L / 2.0
            y = self.x[j] - self.L / 2.0
            if x * x + y * y <= a2:
                f[row] = pressure_pa * self.h**2
        return f

    def _center_row(self) -> int:
        n = self.n
        c = n // 2
        # nearest interior node to the center
        best, bestd = None, None
        for (i, j), row in self._idx.items():
            d = (i - c) ** 2 + (j - c) ** 2
            if bestd is None or d < bestd:
                best, bestd = row, d
        return best

    # -- solves -----------------------------------------------------------
    def static_circular(self, radius_m: float, pressure_pa: float) -> PlateFEMResult:
        """Static solve for a uniform circular pressure; returns k_eff etc."""
        f = self._circular_load(radius_m, pressure_pa)
        w = np.linalg.solve(self.K, f)
        wc = w[self._center_row()]
        total = float(f.sum())
        k_eff = total / wc if wc > 0 else float("inf")
        # top-surface bending stress from the discrete curvature
        # sigma = -E t / (2 (1-nu^2)) * (w_xx + w_yy)  (mid-surface, top face)
        wfull = np.zeros((self.n, self.n))
        for (i, j), row in self._idx.items():
            wfull[i, j] = w[row]
        wx = np.zeros_like(wfull)
        wy = np.zeros_like(wfull)
        wx[1:-1, 1:-1] = (wfull[2:, 1:-1] - 2 * wfull[1:-1, 1:-1] + wfull[:-2, 1:-1]) / self.h**2
        wy[1:-1, 1:-1] = (wfull[1:-1, 2:] - 2 * wfull[1:-1, 1:-1] + wfull[1:-1, :-2]) / self.h**2
        sigma = -self.E * self.t / (2.0 * (1.0 - self.nu**2)) * (wx + wy)
        p_field = np.zeros((self.n, self.n))
        a2 = radius_m**2
        for (i, j) in self._idx:
            x = self.x[i] - self.L / 2.0
            y = self.x[j] - self.L / 2.0
            if x * x + y * y <= a2:
                p_field[i, j] = pressure_pa
        se = 0.5 * float(w @ (self.K @ w))
        return PlateFEMResult(
            n=self.n, k_eff=float(k_eff), w_center=float(wc),
            total_force=total, w_field=wfull, p_field=p_field,
            sigma_field=sigma, strain_energy=se,
        )

    def static_field(self, radius_m: float, pressure_pa: float) -> PlateFEMResult:
        """Alias of :meth:`static_circular` (peak-load field for the report)."""
        return self.static_circular(radius_m, pressure_pa)

    def analytic_center_deflection(self, force_n: float) -> float:
        """Analytic center deflection of a simply supported square plate.

        ``delta = 0.0056 F L^2 / D`` (Timoshenko & Woinowsky, central point
        load on a simply supported square plate, ``L`` = side).
        """
        return 0.0056 * force_n * self.L**2 / self.D
