"""Von Karman circular plate: large deflection under center point load.

Implements:
- Reference solution from Camp (1949) and Berger (1955) for the nonlinear
  load-deflection relationship of a clamped circular plate under center load.
- FEM solution using axisymmetric von Karman plate theory with Newton-Raphson.

Physics
-------
For a thin circular plate (radius a, thickness h, Young's modulus E, 
Poisson's ratio ν) clamped at the edge and loaded by a concentrated 
force P at the center, the von Karman large-deflection equations couple
the transverse deflection w(r) and the stress function φ(r):

    D ∇⁴w = h [φ, w]/r + P δ(r) / (2πr)
    ∇⁴φ = -E/(2r) [w, w]

where D = Eh³/(12(1-ν²)) is the flexural rigidity and [f,g] denotes the
Monge bracket. For axisymmetric deformation:

    D(d²w/dr² + 1/r dw/dr) = h/(r) * dφ/dr * dw/dr + P/(2πr)
    d²φ/dr² + 1/r dφ/dr - φ/r² = -E/2 * (dw/dr)² / r

The key result is the nonlinear load-deflection curve:
    P = C₁ * D * w₀/a² + C₂ * E*h * w₀³/a²  (for clamped edge)

where C₁ = 16/(1+ν)·(5+ν)/(3(1-ν²)) ... (exact coefficients from the series).

For small deflections (w₀/h << 1): P ≈ C₁ D w₀/a² (linear)
For large deflections (w₀/h > 1): cubic term dominates (membrane stiffening)

Verification target: P(w₀) curve vs Camp/Berger literature ≤ 5%.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize
from dataclasses import dataclass
from typing import Tuple, Optional

__all__ = ["VonKarmanPlateExact", "VonKarmanPlateFEM", "solve_von_karman_plate",
           "plate_cubic_coefficient", "load_path_work"]


#: Coefficient of the cubic (membrane-stiffening) term of the clamped circular
#: plate load-deflection law.  Value for nu = 0.3 taken from the classical
#: series solutions (Chien 1948; Yamaki 1958; Camp 1949) as quoted by
#: Timoshenko & Woinowsky-Krieger (1959) and Berger (1955).  The weak Poisson
#: dependence is a linear interpolation of that data.
_CUBIC_COEFF_NU030 = 0.376


def plate_cubic_coefficient(nu: float) -> float:
    """Cubic-term coefficient c(nu) of the clamped-plate law

        P a^2 / (16 pi D h) = (w0/h) + c(nu) (w0/h)^3 .

    Single source of truth for this constant (also used by
    ``verification.camp_berger_reference``).  Valid for 0.2 <= nu <= 0.4.
    """
    if not 0.15 <= nu <= 0.45:
        raise ValueError(f"plate_cubic_coefficient: nu={nu} outside validated range [0.15, 0.45]")
    return _CUBIC_COEFF_NU030 + 0.01 * (nu - 0.3)



@dataclass
class PlateSolution:
    """Container for plate solution."""
    P: float              # Center load [N]
    w0: float             # Center deflection [m]
    w0_over_h: float      # Normalized deflection
    a: float              # Plate radius [m]
    h: float              # Plate thickness [m]
    D: float              # Flexural rigidity [N·m]
    r_array: np.ndarray   # Radial coordinates
    w_array: np.ndarray   # Deflection profile
    phi_array: np.ndarray  # Stress function


class VonKarmanPlateExact:
    """Reference solution for von Karman circular plate.
    
    Uses the series/numerical solution of the coupled ODEs.
    For validation, we use the well-known result:
    
    P/(Eh³) = α * (w₀/h) + β * (w₀/h)³
    
    where for a clamped circular plate with ν=0.3:
    α ≈ 5.74 (from linear plate theory, Timoshenko)
    β ≈ 1.38 (from large deflection theory, Camp 1949)
    
    More precisely, the coefficients from the literature:
    - Linear: P = (16D/a²) * w₀ * (5+ν)/(3(1+ν)) ... for clamped
    - Actually: P_linear = 4πD/(a²) * C(ν) * w₀ where C depends on BC
    
    The standard result (Timoshenko & Woinowsky-Krieger, §153):
    For clamped circular plate under center load P:
    w₀ = P*a² / (16πD)  (linear, ν=0.3)
    
    With large deflection (von Karman):
    P*a²/(D) = 16π * [w₀/h + 0.394*(1-ν²)*(w₀/h)³]  (approx, Berger 1955)
    
    More accurate coefficients from numerical integration of the ODEs.
    """
    
    def __init__(self, a: float = 1.0, h: float = 0.01, 
                 E: float = 210e9, nu: float = 0.3):
        self.a = a
        self.h = h
        self.E = E
        self.nu = nu
        self.D = E * h**3 / (12 * (1 - nu**2))
    
    def linear_solution(self, P: float) -> float:
        """Linear (small deflection) center deflection.
        
        For clamped circular plate under center point load:
        w₀ = P*a² / (16πD)
        
        Reference: Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, §97.
        """
        return P * self.a**2 / (16 * np.pi * self.D)
    
    def compute_load_deflection_curve(
        self, w0_h_range: Optional[np.ndarray] = None, n_points: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Closed-form approximation of the nonlinear load-deflection curve.

        Clamped circular plate, concentric point load, dimensionless form

            P a^2 / (16 pi D h) = (w0/h) + c(nu) (w0/h)^3 ,

        i.e. the linear Timoshenko result (w0 = P a^2 / (16 pi D)) plus the
        cubic membrane-stiffening term with c(nu) from
        :func:`plate_cubic_coefficient` (0.376 at nu = 0.3; Chien 1948,
        Yamaki 1958, Camp 1949, as summarised by Timoshenko &
        Woinowsky-Krieger 1959 and Berger 1955).

        Accuracy (this module, verified): against the mesh-converged
        von Karman finite element solution of :class:`VonKarmanPlateFEM`
        the implied centre deflection deviates by at most 2.3% for
        0.5 <= w0/h <= 3 (nu = 0.3, n_elem = 40); the effective cubic
        coefficient fitted to the FE curve drifts from 0.44 at w0/h = 0.5 to
        0.35 at w0/h = 3, which is the intrinsic limitation of a
        constant-coefficient cubic representation -- see
        docs/m8870_r1_audit.md (artifacts/m8870_r1/).

        Parameters
        ----------
        w0_h_range : ndarray, optional
            Normalised centre deflections w0/h (default: 0.1 ... 6).
        n_points : int
            Number of points when ``w0_h_range`` is None.

        Returns
        -------
        w0_h, P_normalized, P_values
            Normalised deflection, P a^2 / (16 pi D h), and the dimensional
            load [N].
        """
        if w0_h_range is None:
            w0_h_range = np.linspace(0.1, 6.0, n_points)

        c = plate_cubic_coefficient(self.nu)

        w0_h = np.asarray(w0_h_range, dtype=float)
        P_normalized = w0_h + c * w0_h**3
        P_values = P_normalized * 16 * np.pi * self.D * self.h / self.a**2

        return w0_h, P_normalized, P_values

    def compute_deflection_profile(self, P: float, n_pts: int = 50) -> PlateSolution:
        """Compute the deflection profile for a given load.
        
        For the approximate solution, the shape is close to the linear one:
        w(r) = w₀ * (1 - (r/a)²)²  for clamped plate (approximate)
        
        More accurately, the shape changes slightly with load level.
        """
        w0_h, P_norm, P_vals = self.compute_load_deflection_curve(n_points=100)
        
        # Find w0 for given P
        P_target_norm = P * self.a**2 / (16 * np.pi * self.D * self.h)
        idx = np.searchsorted(P_vals, P)
        if idx >= len(P_vals):
            w0 = w0_h[-1] * self.h
        else:
            # Linear interpolation
            w0 = np.interp(P, P_vals, w0_h * self.h)
        
        # Approximate deflection shape (clamped plate)
        r = np.linspace(0, self.a, n_pts)
        rho = r / self.a
        # For clamped plate, the linear shape is:
        # w(r) = w₀ * (1 - ρ²)² * (1 + correction)
        # The correction for large deflection makes it slightly fuller
        w = w0 * (1 - rho**2)**2
        
        # Stress function (approximate)
        # φ(r) = N_r * a² / 2 where N_r is the radial membrane force
        # For the Berger approximation:
        phi = self.E * self.h * w0**2 / (4 * self.a**2) * (1 - rho**2)
        
        return PlateSolution(
            P=P, w0=w0, w0_over_h=w0/self.h,
            a=self.a, h=self.h, D=self.D,
            r_array=r, w_array=w, phi_array=phi,
        )


class VonKarmanPlateFEM:
    """Axisymmetric FEM for von Karman circular plate.
    
    Uses a 1D mesh in the radial direction with:
    - Cubic Hermite elements for w (transverse deflection)
    - Linear elements for the membrane displacement u_r
    
    The von Karman strain-displacement relations for axisymmetric plate:
    ε_r = du_r/dr + 1/2*(dw/dr)²
    ε_θ = u_r/r
    κ_r = -d²w/dr²
    κ_θ = -1/r * dw/dr
    
    Constitutive (plane stress):
    N_r = C*(ε_r + ν*ε_θ)
    N_θ = C*(ε_θ + ν*ε_r)
    M_r = D*(κ_r + ν*κ_θ)
    M_θ = D*(κ_θ + ν*κ_r)
    
    where C = Eh/(1-ν²) is the membrane stiffness.
    """
    
    def __init__(self, a: float, h: float, E: float, nu: float, n_elem: int = 20):
        self.a = a
        self.h = h
        self.E = E
        self.nu = nu
        self.n_elem = n_elem
        self.n_nodes = n_elem + 1
        self.D = E * h**3 / (12 * (1 - nu**2))
        self.C = E * h / (1 - nu**2)
        self.dr = a / n_elem
        
        # DOFs per node: w, dw/dr, u_r  (3 DOFs)
        self.ndof = 3 * self.n_nodes
    
    def _hermite_funcs(self, s: float, he: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cubic Hermite shape functions and derivatives.
        
        s: local coordinate in [0, 1]
        he: element length
        
        Returns N, dN_ds, dN_dx (dN/dx = dN_ds / he)
        """
        N = np.array([
            1 - 3*s**2 + 2*s**3,
            he * (s - 2*s**2 + s**3),
            3*s**2 - 2*s**3,
            he * (-s**2 + s**3),
        ])
        dN_ds = np.array([
            -6*s + 6*s**2,
            he * (1 - 4*s + 3*s**2),
            6*s - 6*s**2,
            he * (-2*s + 3*s**2),
        ])
        dN_dx = dN_ds / he
        
        # Second derivatives
        ddN_ds2 = np.array([
            -6 + 12*s,
            he * (-4 + 6*s),
            6 - 12*s,
            he * (-2 + 6*s),
        ])
        ddN_dx2 = ddN_ds2 / he**2
        
        return N, dN_dx, ddN_dx2
    
    def assemble_internal_force(self, u: np.ndarray) -> np.ndarray:
        """Assemble global internal force vector."""
        ndof = self.ndof
        f_int = np.zeros(ndof)
        dr = self.dr
        nu = self.nu
        
        # 2-point Gauss quadrature
        xi_gps = np.array([(-1/np.sqrt(3) + 1)/2, (1/np.sqrt(3) + 1)/2])
        w_gps = np.array([0.5, 0.5])
        
        for e in range(self.n_elem):
            # Node indices
            n1 = e
            n2 = e + 1
            
            # DOF indices for this element
            # w1, theta1, u1, w2, theta2, u2
            dofs = np.array([3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2])
            u_e = u[dofs]
            
            w1, t1, ur1, w2, t2, ur2 = u_e
            
            for xi, wt in zip(xi_gps, w_gps):
                s = xi  # s ∈ [0, 1]
                r = (e + s) * dr  # radial coordinate
                
                # Hermite shape functions for w
                N, dN_dx, ddN_dx2 = self._hermite_funcs(s, dr)
                
                # Linear shape functions for u_r
                N1_u = 1 - s
                N2_u = s
                dN1_u_dx = -1/dr
                dN2_u_dx = 1/dr
                
                # Displacements and derivatives
                w = N[0]*w1 + N[1]*t1 + N[2]*w2 + N[3]*t2
                dw_dx = dN_dx[0]*w1 + dN_dx[1]*t1 + dN_dx[2]*w2 + dN_dx[3]*t2
                d2w_dx2 = ddN_dx2[0]*w1 + ddN_dx2[1]*t1 + ddN_dx2[2]*w2 + ddN_dx2[3]*t2
                u_r = N1_u * ur1 + N2_u * ur2
                dur_dx = dN1_u_dx * ur1 + dN2_u_dx * ur2
                
                # Von Karman strains
                eps_r = dur_dx + 0.5 * dw_dx**2
                eps_theta = u_r / max(r, 1e-10)
                kappa_r = d2w_dx2
                kappa_theta = dw_dx / max(r, 1e-10)
                
                # Stress resultants
                N_r = self.C * (eps_r + nu * eps_theta)
                N_theta = self.C * (eps_theta + nu * eps_r)
                M_r = self.D * (kappa_r + nu * kappa_theta)
                M_theta = self.D * (kappa_theta + nu * kappa_r)
                
                # B matrices for strain-displacement
                # eps_r = dur_dx + 0.5*dw_dx^2
                # deps_r/d(ur1) = dN1_u_dx, deps_r/d(ur2) = dN2_u_dx
                # deps_r/d(w1) = dw_dx * dN_dx[0], etc.
                
                r_safe = max(r, dr * 0.01)
                wt_r = wt * r_safe * dr  # integration weight * r * dr (axisymmetric)
                
                # Internal force from membrane
                # f_mem = ∫ B_eps^T * N * r dr dθ = 2π ∫ B_eps^T * N * r dr
                # We drop 2π as it cancels
                
                # Membrane contributions
                B_eps_r = np.array([
                    dw_dx * dN_dx[0],  # d(eps_r)/d(w1)
                    dw_dx * dN_dx[1],  # d(eps_r)/d(theta1)
                    dN1_u_dx,           # d(eps_r)/d(ur1)
                    dw_dx * dN_dx[2],  # d(eps_r)/d(w2)
                    dw_dx * dN_dx[3],  # d(eps_r)/d(theta2)
                    dN2_u_dx,           # d(eps_r)/d(ur2)
                ])
                
                # eps_theta = u_r / r
                B_eps_theta = np.array([
                    0, 0, N1_u/r_safe,
                    0, 0, N2_u/r_safe,
                ])
                
                # Total membrane internal force
                f_mem = (N_r * (B_eps_r + nu * B_eps_theta) + 
                         N_theta * (B_eps_theta + nu * B_eps_r))
                # Actually, more correctly:
                # δU_mem = ∫ (N_r * δε_r + N_θ * δε_θ) r dr
                f_mem = N_r * B_eps_r + N_theta * B_eps_theta
                
                f_int[dofs] += f_mem * wt_r
                
                # Bending contributions
                B_kappa_r = np.array([ddN_dx2[0], ddN_dx2[1], 0, ddN_dx2[2], ddN_dx2[3], 0])
                B_kappa_theta = np.array([dN_dx[0]/r_safe, dN_dx[1]/r_safe, 0,
                                          dN_dx[2]/r_safe, dN_dx[3]/r_safe, 0])
                
                f_bend = M_r * B_kappa_r + M_theta * B_kappa_theta
                f_int[dofs] += f_bend * wt_r
        
        return f_int
    
    def assemble_tangent_stiffness(self, u: np.ndarray) -> np.ndarray:
        """Assemble tangent stiffness matrix (numerical differentiation)."""
        ndof = self.ndof
        K = np.zeros((ndof, ndof))
        eps = 1e-7
        
        f0 = self.assemble_internal_force(u)
        
        for j in range(ndof):
            u_pert = u.copy()
            h_j = max(eps * abs(u[j]), eps)
            u_pert[j] += h_j
            f1 = self.assemble_internal_force(u_pert)
            K[:, j] = (f1 - f0) / h_j
        
        return K
    
    def energy(self, u: np.ndarray) -> dict:
        """Strain energy of the discrete von Karman plate model [J].

        Integrates

            U = 1/2 * 2 pi * int_0^a (N_r eps_r + N_theta eps_theta
                                      + M_r kappa_r + M_theta kappa_theta) r dr

        with the same Gauss rule and the same Green-Lagrange kinematics as
        :meth:`assemble_internal_force`, and splits it into the membrane
        (stretching) and bending contributions.  Because ``energy`` and
        ``assemble_internal_force`` are independent loops, checking
        ``dU/du == f_int`` (see tests/test_geometric_nonlinear_accuracy.py)
        is a genuine consistency test of the two.

        The 2 pi factor of the axisymmetric integration is included, so the
        result can be compared directly with the dead-load work
        ``int P dw0`` (:func:`load_path_work`).
        """
        u = np.asarray(u, dtype=float)
        dr = self.dr
        nu = self.nu
        n = self.n_nodes

        xi_gps = np.array([(-1 / np.sqrt(3) + 1) / 2, (1 / np.sqrt(3) + 1) / 2])
        w_gps = np.array([0.5, 0.5])

        u_mem = 0.0
        u_bend = 0.0

        for e in range(self.n_elem):
            dofs = np.array([3 * e, 3 * e + 1, 3 * e + 2,
                             3 * (e + 1), 3 * (e + 1) + 1, 3 * (e + 1) + 2])
            w1, t1, ur1, w2, t2, ur2 = u[dofs]

            for s, wt in zip(xi_gps, w_gps):
                r = (e + s) * dr
                r_safe = max(r, dr * 0.01)

                N, dN_dx, ddN_dx2 = self._hermite_funcs(s, dr)

                w = N[0] * w1 + N[1] * t1 + N[2] * w2 + N[3] * t2
                dw_dx = dN_dx[0] * w1 + dN_dx[1] * t1 + dN_dx[2] * w2 + dN_dx[3] * t2
                d2w_dx2 = (ddN_dx2[0] * w1 + ddN_dx2[1] * t1
                           + ddN_dx2[2] * w2 + ddN_dx2[3] * t2)

                u_r = (1 - s) * ur1 + s * ur2
                dur_dx = (-ur1 + ur2) / dr

                eps_r = dur_dx + 0.5 * dw_dx**2
                eps_theta = u_r / r_safe
                kappa_r = d2w_dx2
                kappa_theta = dw_dx / r_safe

                N_r = self.C * (eps_r + nu * eps_theta)
                N_theta = self.C * (eps_theta + nu * eps_r)
                M_r = self.D * (kappa_r + nu * kappa_theta)
                M_theta = self.D * (kappa_theta + nu * kappa_r)

                wgt = wt * r_safe * dr
                u_mem += 0.5 * (N_r * eps_r + N_theta * eps_theta) * wgt
                u_bend += 0.5 * (M_r * kappa_r + M_theta * kappa_theta) * wgt

        two_pi = 2.0 * np.pi
        u_mem *= two_pi
        u_bend *= two_pi
        return {"membrane": u_mem, "bending": u_bend, "total": u_mem + u_bend}

    def strain_energy(self, u: np.ndarray) -> float:
        """Total strain energy [J] (see :meth:`energy`)."""
        return self.energy(u)["total"]

    def solve(self, P: float, n_steps: int = 15, max_iter: int = 30,
              tol: float = 1e-8) -> dict:
        """Solve for center point load P using Newton-Raphson.
        
        The center load is applied as a concentrated force at r=0.
        In the FEM, this is applied as a point force at node 0 (DOF: w at r=0).
        """
        ndof = self.ndof
        u = np.zeros(ndof)
        
        # BCs: 
        # At r=0 (node 0): dw/dr = 0 (symmetry) → DOF 1 = 0
        # At r=a (last node): w = 0, dw/dr = 0, u_r = 0 (clamped) → DOFs of last node
        fixed_dofs = [1]  # symmetry at center: theta = 0
        last_node = self.n_nodes - 1
        fixed_dofs += [3*last_node, 3*last_node+1, 3*last_node+2]  # clamped edge
        # Also fix u_r at center to prevent rigid body
        fixed_dofs.append(2)  # u_r = 0 at center
        fixed_dofs = np.unique(np.array(fixed_dofs, dtype=int))
        free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
        
        # Load vector: center point load at w DOF of node 0
        w0_dof = 0  # w at r=0
        
        history = []
        converged_all = True
        
        load_factors = np.linspace(1.0/n_steps, 1.0, n_steps)
        
        for lf in load_factors:
            P_current = lf * P
            f_ext = np.zeros(ndof)
            f_ext[w0_dof] = P_current / (2 * np.pi)  # distributed over circumference
            
            for iteration in range(max_iter):
                f_int = self.assemble_internal_force(u)
                residual = f_ext - f_int
                res_free = residual[free_dofs]
                res_norm = np.linalg.norm(res_free)
                
                if res_norm < tol * max(1.0, np.linalg.norm(f_ext[free_dofs])):
                    break
                
                K = self.assemble_tangent_stiffness(u)
                K_ff = K[np.ix_(free_dofs, free_dofs)]
                
                try:
                    delta_u = np.zeros(ndof)
                    delta_u[free_dofs] = np.linalg.solve(K_ff, res_free)
                except np.linalg.LinAlgError:
                    converged_all = False
                    break
                
                u += delta_u
            
            if iteration == max_iter - 1:
                converged_all = False
            
            history.append({
                'load_factor': lf,
                'P': P_current,
                'w0': u[w0_dof],
                'w0_over_h': u[w0_dof] / self.h,
                'iterations': iteration + 1,
            })
        
        return {
            'u': u,
            'converged': converged_all,
            'history': history,
            'w0': u[w0_dof],
        }


def load_path_work(history: list) -> float:
    """Dead-load work ``int_0^{w0} P dw0`` along a monotonic load path [J].

    ``history`` is the list of load steps returned by
    :meth:`VonKarmanPlateFEM.solve` (each entry has ``P`` and ``w0``).  The
    path starts at (P, w0) = (0, 0); the integral is evaluated with the
    trapezoidal rule, so its accuracy is limited by the load-step size
    (O(dP^2) for the smooth, slightly convex P-w0 curve).
    """
    P = np.concatenate(([0.0], np.array([h["P"] for h in history], dtype=float)))
    w0 = np.concatenate(([0.0], np.array([h["w0"] for h in history], dtype=float)))
    return float(np.trapezoid(P, w0))


def solve_von_karman_plate(
    a: float = 1.0, h: float = 0.01, E: float = 210e9, nu: float = 0.3,
    w0_h_range: Optional[np.ndarray] = None,
    n_elem: int = 20,
) -> dict:
    """Solve the von Karman plate problem, comparing reference and FEM.
    
    Returns
    -------
    dict with comparison results.
    """
    ref = VonKarmanPlateExact(a=a, h=h, E=E, nu=nu)
    
    if w0_h_range is None:
        w0_h_range = np.linspace(0.5, 5.0, 15)
    
    # Reference solution
    w0_h_ref, P_norm_ref, P_ref = ref.compute_load_deflection_curve(w0_h_range)
    
    # FEM solution - for each target w0, find the load
    fem = VonKarmanPlateFEM(a=a, h=h, E=E, nu=nu, n_elem=n_elem)
    
    # Use the FEM to solve for increasing loads and extract w0
    P_max = P_ref[-1]
    n_steps = 20
    P_test = np.linspace(P_max * 0.05, P_max, n_steps)
    
    w0_fem = []
    for P_val in P_test:
        result = fem.solve(P_val, n_steps=5, max_iter=30, tol=1e-6)
        w0_fem.append(result['w0'])
    w0_fem = np.array(w0_fem)
    
    return {
        'w0_h_ref': w0_h_ref,
        'P_norm_ref': P_norm_ref,
        'P_ref': P_ref,
        'P_test': P_test,
        'w0_fem': w0_fem,
        'w0_h_fem': w0_fem / h,
    }
