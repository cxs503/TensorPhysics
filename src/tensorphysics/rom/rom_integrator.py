"""
ROM time integrator.

Integrates the POD-Galerkin-DEIM ROM system:
    da/dt = -N(a) + A*a + f

using the same time integration scheme as the FOM (RK4) for fair comparison.
"""

import numpy as np
import time


class ROMIntegrator:
    """Time integrator for POD-Galerkin ROM.
    
    Parameters
    ----------
    projector : GalerkinProjector
        Galerkin projector with precomputed operators
    deim : DEIMInterpolator, optional
        DEIM interpolator for nonlinear term (if None, use full evaluation)
    """
    
    def __init__(self, projector, deim=None):
        self.projector = projector
        self.deim = deim
        self.n_rhs_evals = 0
        self.total_time = 0.0
    
    def rhs(self, a):
        """Compute ROM RHS.
        
        Parameters
        ----------
        a : ndarray, shape (2*k,)
            ROM coefficients [a_u; a_v]
            
        If DEIM is available, use it for nonlinear term.
        Otherwise, evaluate full nonlinear term.
        """
        self.n_rhs_evals += 1
        
        k = self.projector.n_modes
        a_u = a[:k]
        a_v = a[k:]
        
        if self.deim is not None:
            # DEIM approximation
            def N_full_func(a_combined):
                a_u_local = a_combined[:k]
                N_u, N_v = self.projector.compute_nonlinear(a_u_local, use_skew_symmetric=True)
                return np.concatenate([N_u, N_v])
            
            N_approx = self.deim.evaluate(N_full_func, a)
            
            # Linear term
            L_u = self.projector.A_uu @ a_u
            L_v = self.projector.A_vv @ a_v
            
            da_u = -N_approx[:k] + L_u + self.projector.f_proj_u
            da_v = -N_approx[k:] + L_v + self.projector.f_proj_v
            
            return np.concatenate([da_u, da_v])
        else:
            # Full evaluation
            N_u, N_v = self.projector.compute_nonlinear(a_u, use_skew_symmetric=True)
            
            # Linear term
            L_u = self.projector.A_uu @ a_u
            L_v = self.projector.A_vv @ a_v
            
            da_u = -N_u + L_u + self.projector.f_proj_u
            da_v = -N_v + L_v + self.projector.f_proj_v
            
            return np.concatenate([da_u, da_v])
    
    def step_rk4(self, a, dt):
        """Advance one timestep using RK4.
        
        Parameters
        ----------
        a : ndarray
            Current ROM coefficients
        dt : float
            Time step
            
        Returns
        -------
        a_new : ndarray
            Updated ROM coefficients
        """
        k1 = self.rhs(a)
        k2 = self.rhs(a + 0.5 * dt * k1)
        k3 = self.rhs(a + 0.5 * dt * k2)
        k4 = self.rhs(a + dt * k3)
        
        return a + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def run(self, a0, t_end, dt, save_interval=1, verbose=False):
        """Integrate ROM system.
        
        Parameters
        ----------
        a0 : ndarray
            Initial ROM coefficients
        t_end : float
            End time
        dt : float
            Time step
        save_interval : int
            Save solution every save_interval steps
        verbose : bool
            Print progress
            
        Returns
        -------
        solutions : list of ndarray
            ROM coefficient snapshots
        times : list of float
            Snapshot times
        """
        n_steps = int(t_end / dt)
        solutions = []
        times = []
        
        a = a0.copy()
        self.n_rhs_evals = 0
        t_start = time.time()
        
        for n in range(n_steps + 1):
            if n % save_interval == 0:
                solutions.append(a.copy())
                times.append(n * dt)
                if verbose and n % (save_interval * 10) == 0:
                    print(f"ROM: t={n*dt:.3f}, ‖a‖={np.linalg.norm(a):.6f}")
            
            if n < n_steps:
                a = self.step_rk4(a, dt)
        
        self.total_time = time.time() - t_start
        
        return solutions, times
    
    def reconstruct_fields(self, solutions):
        """Reconstruct physical fields from ROM solutions.
        
        Parameters
        ----------
        solutions : list of ndarray
            ROM coefficient snapshots
            
        Returns
        -------
        fields_u : list of ndarray
            x-velocity fields
        fields_v : list of ndarray
            y-velocity fields
        """
        nx = len(self.projector.kx)
        ny = len(self.projector.ky)
        
        fields_u = []
        fields_v = []
        
        for a in solutions:
            k = self.projector.n_modes
            a_u = a[:k]
            a_v = a[k:]
            
            u = self.projector.modes_u @ a_u + self.projector.mean_u.flatten()
            v = self.projector.modes_v @ a_v + self.projector.mean_v.flatten()
            
            fields_u.append(u.reshape(nx, ny))
            fields_v.append(v.reshape(nx, ny))
        
        return fields_u, fields_v
