"""Cylindrical shell axial-compression buckling solver.

Provides two complementary approaches for thin cylindrical shell buckling
under uniform axial compression:

1. **Analytical Donnell** — closed-form buckling load for each (m, n) mode
   from the Donnell shell equations; the minimum over all modes recovers
   the classical formula σ_cr = E/√(3(1−ν²))·t/R.

2. **Semi-analytical FEM** — axial direction discretised with Hermite beam
   elements, circumferential direction via Fourier modes.  Converges to the
   analytical solution as the mesh is refined.

The classical formula (Timoshenko & Gere, §11.2) is:

    σ_cr = E / √(3(1 − ν²)) · t / R
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.linalg as sla

__all__ = ["CylindricalShellBuckling"]


@dataclass(frozen=True)
class CylindricalShellBuckling:
    """Thin cylindrical shell under axial compression.

    Parameters
    ----------
    R : float
        Mid-surface radius [m].
    L : float
        Shell length [m].
    t : float
        Wall thickness [m].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    n_axial : int
        Number of axial elements (for FEM).
    n_circ : int
        Maximum circumferential wave number to scan.
    """

    R: float
    L: float
    t: float
    E: float
    nu: float
    n_axial: int
    n_circ: int

    @property
    def D(self) -> float:
        """Flexural rigidity per unit width."""
        return self.E * self.t ** 3 / (12.0 * (1.0 - self.nu ** 2))

    @property
    def C(self) -> float:
        """Extensional stiffness per unit width."""
        return self.E * self.t / (1.0 - self.nu ** 2)

    @property
    def le(self) -> float:
        """Axial element length."""
        return self.L / self.n_axial

    def classical_buckling_stress(self) -> float:
        """Classical critical buckling stress (Timoshenko & Gere)."""
        from .analytical import classical_cylindrical_shell_buckling_stress
        return classical_cylindrical_shell_buckling_stress(
            self.E, self.nu, self.t, self.R
        )

    def analytical_Ncr_for_mode(self, m: int, n: int) -> float:
        """Analytical buckling load (force/width) for mode (m, n).

        From Donnell theory, for a simply-supported shell:

            N_cr = D·(α² + β²)²/α² + C·α²/[R²(α² + β²)²]

        where α = mπ/L, β = n/R.

        The critical stress is σ_cr = N_cr / t.
        """
        alpha = m * math.pi / self.L
        beta = n / self.R
        D_val = self.D
        C_val = self.C
        return D_val * (alpha ** 2 + beta ** 2) ** 2 / alpha ** 2 + \
               C_val * alpha ** 2 / (self.R ** 2 * (alpha ** 2 + beta ** 2) ** 2)

    def analytical_min_buckling(self, m_max: int = 30) -> Dict[str, float]:
        """Find minimum analytical buckling load over m and n.

        Parameters
        ----------
        m_max : int
            Maximum axial half-wave number to scan.

        Returns
        -------
        dict
            N_cr: minimum buckling load [N/m]
            sigma_cr: critical stress [Pa]
            m_crit: critical axial mode
            n_crit: critical circumferential mode
        """
        best_N = float("inf")
        best_m, best_n = -1, -1

        for m in range(1, m_max + 1):
            for n in range(1, self.n_circ + 1):
                N_cr = self.analytical_Ncr_for_mode(m, n)
                if N_cr < best_N:
                    best_N = N_cr
                    best_m, best_n = m, n

        return {
            "N_cr": float(best_N),
            "sigma_cr": float(best_N / self.t),
            "m_crit": int(best_m),
            "n_crit": int(best_n),
        }

    def _donnell_stiffness_matrices(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build Donnell-type K and K_G for circumferential wave number n.

        Uses Hermite beam elements for w (4 DOFs/element) and linear elements
        for Φ (2 DOFs/element).  The coupled system captures the Donnell
        shell buckling physics.

        Returns
        -------
        K : ndarray
            Linear elastic stiffness.
        KG : ndarray
            Geometric stiffness from unit axial compression.
        """
        R, L, t = self.R, self.L, self.t
        E, nu = self.E, self.nu
        le = self.le
        na = self.n_axial

        D_val = self.D
        C_val = self.C
        beta = n / R

        # DOFs per node: [w, w', Φ]  (3 per node)
        n_nodes = na + 1
        ndof = 3 * n_nodes

        K = np.zeros((ndof, ndof))
        KG = np.zeros((ndof, ndof))

        for e in range(na):
            w_dofs = [3 * e, 3 * e + 1, 3 * (e + 1), 3 * (e + 1) + 1]
            phi_dofs = [3 * e + 2, 3 * (e + 1) + 2]

            # Bending: D ∫ w''ψ'' dx
            kb = D_val * np.array([
                [12 / le ** 3, 6 / le ** 2, -12 / le ** 3, 6 / le ** 2],
                [6 / le ** 2, 4 / le, -6 / le ** 2, 2 / le],
                [-12 / le ** 3, -6 / le ** 2, 12 / le ** 3, -6 / le ** 2],
                [6 / le ** 2, 2 / le, -6 / le ** 2, 4 / le],
            ])
            # Circumferential bending: D·β⁴ ∫ wψ dx
            kb_circ = D_val * beta ** 4 * le * np.array([
                [1 / 3, 0, 1 / 6, 0],
                [0, le ** 2 / 30, 0, le ** 2 / 60],
                [1 / 6, 0, 1 / 3, 0],
                [0, le ** 2 / 60, 0, le ** 2 / 30],
            ])
            # Poisson coupling: 2Dνβ² ∫ w'ψ' dx
            kb_nu = 2 * D_val * nu * beta ** 2 * np.array([
                [1 / le, 0.5, -1 / le, 0.5],
                [0.5, le / 3, -0.5, le / 6],
                [-1 / le, -0.5, 1 / le, -0.5],
                [0.5, le / 6, -0.5, le / 3],
            ])

            for i in range(4):
                for j in range(4):
                    K[w_dofs[i], w_dofs[j]] += kb[i, j] + kb_circ[i, j] + kb_nu[i, j]

            # Φ stiffness: (1/C) ∫ [Φ''ψ'' + 2β²Φ'ψ' + β⁴Φψ] dx
            # For linear Φ, Φ'' = 0, so only β² and β⁴ terms
            kphi = (1.0 / C_val) * (
                2 * beta ** 2 * le * np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]]) +
                beta ** 4 * le * np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])
            )
            for i in range(2):
                for j in range(2):
                    K[phi_dofs[i], phi_dofs[j]] += kphi[i, j]

            # Coupling: (1/R) ∫ Φ'ψ' dx and -(1/R) ∫ w'φ' dx
            kw_phi = -(1.0 / R) * np.array([
                [-1, 1], [-le / 2, le / 2], [1, -1], [-le / 2, le / 2]
            ])
            kphi_w = (1.0 / R) * np.array([
                [-1, -le / 2, 1, -le / 2],
                [1, le / 2, -1, le / 2],
            ])
            for i in range(4):
                for j in range(2):
                    K[w_dofs[i], phi_dofs[j]] += kw_phi[i, j]
                    K[phi_dofs[j], w_dofs[i]] += kphi_w[j, i]

            # Geometric stiffness: ∫ w'ψ' dx + β² ∫ wψ dx
            kg_w = np.array([
                [6 / (5 * le), le / 10, -6 / (5 * le), le / 10],
                [le / 10, 2 * le / 15, -le / 10, -le / 30],
                [-6 / (5 * le), -le / 10, 6 / (5 * le), -le / 10],
                [le / 10, -le / 30, -le / 10, 2 * le / 15],
            ])
            kg_circ = beta ** 2 * le * np.array([
                [1 / 3, 0, 1 / 6, 0],
                [0, le ** 2 / 30, 0, le ** 2 / 60],
                [1 / 6, 0, 1 / 3, 0],
                [0, le ** 2 / 60, 0, le ** 2 / 30],
            ])
            for i in range(4):
                for j in range(4):
                    KG[w_dofs[i], w_dofs[j]] += kg_w[i, j] + kg_circ[i, j]

        return K, KG

    def buckling_analysis(self) -> Dict[str, float]:
        """Solve eigenvalue buckling for the most critical circumferential mode.

        Returns
        -------
        dict
            sigma_cr: critical buckling stress [Pa]
            n_crit: critical circumferential wave number
            lam_cr: critical load factor (N/m)
        """
        n_min = max(1, self.n_circ // 4)
        n_max = self.n_circ

        results = []
        for n in range(n_min, n_max + 1):
            K, KG = self._donnell_stiffness_matrices(n)
            fixed = [0, 3 * self.n_axial]
            free = np.setdiff1d(np.arange(K.shape[0]), fixed)
            Kff = K[np.ix_(free, free)]
            KGff = KG[np.ix_(free, free)]

            try:
                eigvals = sla.eig(Kff, KGff, left=False, right=False)
                eigvals = np.real(eigvals)
                pos = eigvals[eigvals > 1e-10]
                if len(pos) > 0:
                    lam_cr = float(np.min(pos))
                    results.append((n, lam_cr))
            except Exception:
                continue

        if not results:
            return {"sigma_cr": float("nan"), "n_crit": -1, "lam_cr": float("nan")}

        n_crit, lam_cr = min(results, key=lambda x: x[1])
        sigma_cr = lam_cr / self.t

        return {
            "sigma_cr": float(sigma_cr),
            "n_crit": int(n_crit),
            "lam_cr": float(lam_cr),
        }

    def condition_number_no_bc(self) -> float:
        """Condition number of K without BCs (negative control)."""
        K, _ = self._donnell_stiffness_matrices(2)
        s = np.linalg.svd(K, compute_uv=False)
        return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
