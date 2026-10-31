"""
7-DOF thin-walled beam element for Vlasov constrained torsion.

Each node has 7 DOFs:
  u, v, w, θx, θy, θz, θx'  (axial, transverse, twist, rotations, warping)

The element stiffness matrix includes:
- Axial stiffness (EA/L)
- Bending stiffness (EI_y, EI_z)
- Saint-Venant torsion (GJ/L)
- Warping torsion (E*I_ω terms)
- Coupling between twist and warping

Reference: Vlasov, V.Z. "Thin-Walled Elastic Beams", 1961.
"""

import numpy as np
from typing import Tuple, Optional


class ThinWalledBeam7DOF:
    """
    7-DOF thin-walled beam element with constrained torsion.

    Nodal DOFs: [u1, v1, w1, θx1, θy1, θz1, θx'1,
                 u2, v2, w2, θx2, θy2, θz2, θx'2]

    where θx' = dθx/dx is the warping DOF.
    """

    def __init__(
        self,
        L: float,
        E: float,
        G: float,
        A: float,
        Iy: float,
        Iz: float,
        J: float,
        Iw: float,
    ):
        """
        Parameters
        ----------
        L : float
            Element length [m].
        E : float
            Young's modulus [Pa].
        G : float
            Shear modulus [Pa].
        A : float
            Cross-sectional area [m²].
        Iy : float
            Second moment of area about y-axis [m⁴].
        Iz : float
            Second moment of area about z-axis [m⁴].
        J : float
            Saint-Venant torsion constant [m⁴].
        Iw : float
            Warping constant [m⁶].
        """
        self.L = L
        self.E = E
        self.G = G
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.Iw = Iw

        self._K = None
        self._build_stiffness()

    def _build_stiffness(self):
        """Build the 14x14 element stiffness matrix."""
        L = self.L
        E = self.E
        G = self.G
        A = self.A
        Iy = self.Iy
        Iz = self.Iz
        J = self.J
        Iw = self.Iw

        K = np.zeros((14, 14))

        # Axial stiffness (DOFs 0, 7)
        k_axial = E * A / L
        K[0, 0] = k_axial
        K[0, 7] = -k_axial
        K[7, 0] = -k_axial
        K[7, 7] = k_axial

        # Bending about y-axis (DOFs 2, 4, 9, 11) - w, θy
        k_bend_y = E * Iy / L ** 3
        K[2, 2] = 12 * k_bend_y
        K[2, 4] = 6 * k_bend_y * L
        K[2, 9] = -12 * k_bend_y
        K[2, 11] = 6 * k_bend_y * L
        K[4, 2] = K[2, 4]
        K[4, 4] = 4 * k_bend_y * L ** 2
        K[4, 9] = -6 * k_bend_y * L
        K[4, 11] = 2 * k_bend_y * L ** 2
        K[9, 2] = K[2, 9]
        K[9, 4] = K[4, 9]
        K[9, 9] = 12 * k_bend_y
        K[9, 11] = -6 * k_bend_y * L
        K[11, 2] = K[2, 11]
        K[11, 4] = K[4, 11]
        K[11, 9] = K[9, 11]
        K[11, 11] = 4 * k_bend_y * L ** 2

        # Bending about z-axis (DOFs 1, 5, 8, 10) - v, θz
        k_bend_z = E * Iz / L ** 3
        K[1, 1] = 12 * k_bend_z
        K[1, 5] = -6 * k_bend_z * L
        K[1, 8] = -12 * k_bend_z
        K[1, 10] = -6 * k_bend_z * L
        K[5, 1] = K[1, 5]
        K[5, 5] = 4 * k_bend_z * L ** 2
        K[5, 8] = 6 * k_bend_z * L
        K[5, 10] = 2 * k_bend_z * L ** 2
        K[8, 1] = K[1, 8]
        K[8, 5] = K[5, 8]
        K[8, 8] = 12 * k_bend_z
        K[8, 10] = 6 * k_bend_z * L
        K[10, 1] = K[1, 10]
        K[10, 5] = K[5, 10]
        K[10, 8] = K[8, 10]
        K[10, 10] = 4 * k_bend_z * L ** 2

        # Saint-Venant torsion (DOFs 3, 10) - θx
        k_torsion = G * J / L
        K[3, 3] = k_torsion
        K[3, 10] = -k_torsion
        K[10, 3] = -k_torsion
        K[10, 10] = k_torsion

        # Warping torsion (DOFs 3, 6, 10, 13) - θx, θx'
        if Iw > 0:
            lam_sq = G * J / (E * Iw)
            lam = np.sqrt(lam_sq)
            lam_L = lam * L

            sinh = np.sinh(lam_L)
            cosh = np.cosh(lam_L)
            denom = sinh ** 2 - lam_L ** 2

            if abs(denom) < 1e-15:
                # Fallback for small lam_L
                k_warp = E * Iw / L ** 3
                K[6, 6] = 4 * k_warp * L ** 2
                K[6, 13] = 2 * k_warp * L ** 2
                K[13, 6] = K[6, 13]
                K[13, 13] = 4 * k_warp * L ** 2
            else:
                # Vlasov warping stiffness coefficients
                c1 = E * Iw * lam ** 3 / denom
                c2 = E * Iw * lam ** 2 / denom

                # θx-θx' coupling
                K[3, 3] += c1 * (sinh * cosh - lam_L)
                K[3, 6] += c2 * (1 - cosh)
                K[3, 10] += c1 * (lam_L - sinh * cosh)
                K[3, 13] += c2 * (1 - cosh)

                K[6, 3] = K[3, 6]
                K[6, 6] += c1 * L * (sinh - lam_L * cosh) / lam
                K[6, 10] = K[3, 13]
                K[6, 13] += c1 * L * (lam_L * cosh - sinh) / lam

                K[10, 3] = K[3, 10]
                K[10, 6] = K[6, 10]
                K[10, 10] += K[3, 3] - c1 * (sinh * cosh - lam_L)
                K[10, 13] = K[3, 6]

                K[13, 3] = K[3, 13]
                K[13, 6] = K[6, 13]
                K[13, 10] = K[10, 13]
                K[13, 13] += K[6, 6]
        else:
            # No warping: θx' DOFs are decoupled
            K[6, 6] = 1e-10  # Small penalty to avoid singularity
            K[13, 13] = 1e-10

        self._K = K

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Return the 14x14 element stiffness matrix."""
        return self._K.copy()

    def compute_warping_stress(
        self,
        nodal_displacements: np.ndarray,
        omega_bar: np.ndarray
    ) -> np.ndarray:
        """
        Compute warping stress from nodal displacements.

        Parameters
        ----------
        nodal_displacements : ndarray
            14-element displacement vector.
        omega_bar : ndarray
            Normalized sectorial coordinate at evaluation points.

        Returns
        -------
        sigma_omega : ndarray
            Warping stress at evaluation points [Pa].
        """
        # Extract θx' at both nodes
        theta_prime_1 = nodal_displacements[6]
        theta_prime_2 = nodal_displacements[13]

        # Bimoment at node 1: B = E * Iw * θ''
        # For linear interpolation: θ'' ≈ (θx'2 - θx'1) / L
        theta_double_prime = (theta_prime_2 - theta_prime_1) / self.L
        bimoment = self.E * self.Iw * theta_double_prime

        return warping_stress_from_bimoment(bimoment, omega_bar, self.Iw)

    def rigid_body_check(self, theta_x: float) -> float:
        """
        Check rigid body objectivity: pure rotation should produce zero warping stress.

        Parameters
        ----------
        theta_x : float
            Applied rigid body rotation [rad].

        Returns
        -------
        max_warping_stress : float
            Maximum warping stress (should be ~0 for rigid body motion).
        """
        # For rigid body rotation: θx1 = θx2 = theta_x, θx'1 = θx'2 = 0
        u_rb = np.zeros(14)
        u_rb[3] = theta_x
        u_rb[10] = theta_x

        # Warping DOFs are zero for rigid body motion
        # So bimoment should be zero
        omega_bar_test = np.array([1.0, -1.0, 0.5, -0.5])
        stresses = self.compute_warping_stress(u_rb, omega_bar_test)

        return np.max(np.abs(stresses))


def warping_stress_from_bimoment(
    bimoment: float,
    omega_bar: np.ndarray,
    Iw: float
) -> np.ndarray:
    """Compute warping stress from bimoment and sectorial coordinate."""
    if Iw <= 0:
        return np.zeros_like(omega_bar)
    return bimoment * omega_bar / Iw
