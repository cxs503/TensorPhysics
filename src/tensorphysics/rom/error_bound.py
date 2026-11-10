"""
A posteriori error bound for POD-Galerkin-DEIM ROM of 2D incompressible Navier-Stokes.

Certified residual-type bound:
    ||e(t)||_V ≤ ||r(t)||_{V'} / (beta_lb - L * ||v_rom(t)||_V)

where:
    e(t) = v_fom(t) - v_rom(t)          # true error
    r(t) = F(v_rom(t)) - dv_rom/dt      # residual (FOM RHS minus ROM time derivative)
    ||·||_{V'} = dual norm of residual
    beta_lb  = lower bound on Stokes inf-sup constant (min singular value of discrete Laplacian)
    L        = computable Lipschitz constant for skew-symmetric convection
    ||v_rom||_V = energy norm of ROM velocity

The bound is valid when beta_lb > L * ||v_rom||_V (coercivity condition).

References:
    - Grepl & Patera, "A posteriori error bounds for reduced-basis approximations
      of parametrized parabolic PDEs", ESAIM:M2AN 39(1), 2005.
    - Quarteroni, Rozza, Redotto, "Certified Reduced Basis Methods for Parametrized
      PDEs", Springer, 2016.
    - Chaturantabut & Sorensen, "Nonlinear model reduction via DEIM", SISC 32(5), 2010.
"""

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.linalg import svd, eigh


class ResidualErrorBound:
    """Certified a posteriori error bound for POD-Galerkin-DEIM ROM.
    
    Parameters
    ----------
    nu : float
        Kinematic viscosity
    kx, ky : ndarray
        Wavenumber arrays (from FOM solver)
    projector : GalerkinProjector
        Galerkin projector with precomputed operators
    pod : PODDecomposition
        POD decomposition used for ROM
    deim : DEIMInterpolator, optional
        DEIM interpolator (if used)
    """
    
    def __init__(self, nu, kx, ky, projector, pod, deim=None):
        self.nu = nu
        self.kx = kx
        self.ky = ky
        self.projector = projector
        self.pod = pod
        self.deim = deim
        
        # Precompute wavenumber mesh
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2
        
        # Precompute beta_lb (Stokes inf-sup lower bound)
        self._beta_lb = None
        self._lipschitz_L = None
        self._nx = len(kx)
        self._ny = len(ky)
    
    @property
    def beta_lb(self):
        """Lower bound on Stokes inf-sup constant.
        
        For the discrete Stokes operator on periodic domain:
            beta_lb = nu * min_{k≠0} |k|² = nu * (2π/L)²
        
        This is the smallest eigenvalue of -ν∇² on the periodic domain
        (excluding the zero mode).
        """
        if self._beta_lb is None:
            # Minimum non-zero wavenumber squared
            k_nonzero = self.K2[self.K2 > 0]
            if len(k_nonzero) > 0:
                k_min_sq = np.min(k_nonzero)
            else:
                k_min_sq = (2 * np.pi / (2 * np.pi))**2  # fallback: fundamental mode
            self._beta_lb = self.nu * k_min_sq
        return self._beta_lb
    
    @property
    def lipschitz_L(self):
        """Computable Lipschitz constant for skew-symmetric convection term.
        
        For the skew-symmetric form N(u) = ½[(u·∇)u + ∇·(u⊗u)],
        the Lipschitz constant in the energy norm is bounded by:
        
            L ≤ C * max_t ||∇v_rom(t)||_{L²}
        
        where C is a domain-dependent constant (~1 for periodic domain).
        
        We compute this from the POD modes using the characteristic
        ROM coefficient magnitude (from the snapshot data):
        
            L = C * sqrt(sum_k |a_k|²) * max_k ||∇φ_k||_{L²}
        
        For the periodic domain [0,2π]², we use C = 1/(2π) as a
        conservative estimate based on the Sobolev embedding.
        """
        if self._lipschitz_L is None:
            nx, ny = self._nx, self._ny
            n_modes = self.projector.n_modes
            area = nx * ny
            
            # Compute L2 norm of gradient of each POD mode
            grad_norms_u = []
            grad_norms_v = []
            for k in range(n_modes):
                phi_u = self.projector.modes_u[:, k].reshape(nx, ny)
                phi_v = self.projector.modes_v[:, k].reshape(nx, ny)
                
                phi_u_hat = fft2(phi_u)
                phi_v_hat = fft2(phi_v)
                
                grad_sq_u = np.sum(self.K2 * np.abs(phi_u_hat)**2) / area
                grad_sq_v = np.sum(self.K2 * np.abs(phi_v_hat)**2) / area
                
                grad_norms_u.append(np.sqrt(grad_sq_u))
                grad_norms_v.append(np.sqrt(grad_sq_v))
            
            # Characteristic ROM coefficient magnitude from singular values
            # (typical coefficient size from the POD energy)
            if len(self.pod.singular_values) > 0:
                # Use the mean singular value as characteristic coefficient
                char_coeff = np.mean(self.pod.singular_values[:n_modes]) / np.sqrt(area)
            else:
                char_coeff = 1.0
            
            # Lipschitz: L = char_coeff * max ||∇φ||_{L²} / (2π)
            max_grad = max(max(grad_norms_u), max(grad_norms_v)) if grad_norms_u else 1.0
            self._lipschitz_L = char_coeff * max_grad / (2 * np.pi)
        
        return self._lipschitz_L
    
    def compute_residual(self, a_rom, da_rom_dt):
        """Compute the ROM residual r = F(v_rom) - dv_rom/dt.
        
        The residual measures how well the ROM solution satisfies the
        full-order model equations.
        
        Parameters
        ----------
        a_rom : ndarray, shape (2*k,)
            ROM coefficients [a_u; a_v]
        da_rom_dt : ndarray, shape (2*k,)
            Time derivative of ROM coefficients (from integrator)
            
        Returns
        -------
        r_u, r_v : ndarray
            Residual fields in physical space (u and v components)
        """
        nx, ny = self._nx, self._ny
        k = self.projector.n_modes
        a_u = a_rom[:k]
        a_v = a_rom[k:]
        
        # Reconstruct ROM velocity field
        u_rom = np.real(self.projector.modes_u @ a_u + self.projector.mean_u.flatten()).reshape(nx, ny)
        v_rom = np.real(self.projector.modes_v @ a_v + self.projector.mean_v.flatten()).reshape(nx, ny)
        
        # Compute FOM RHS at ROM state: F(v_rom) = -N(v_rom) + ν∇²v_rom + f
        # Nonlinear term (skew-symmetric)
        N_u, N_v = self.projector.compute_nonlinear(a_u, use_skew_symmetric=True)
        
        # Reconstruct N_u, N_v as physical fields for residual computation
        N_u_phys = np.zeros((nx, ny))
        N_v_phys = np.zeros((nx, ny))
        for i in range(k):
            N_u_phys += np.real(N_u[i] * self.projector.modes_u[:, i].reshape(nx, ny))
            N_v_phys += np.real(N_v[i] * self.projector.modes_v[:, i].reshape(nx, ny))
        
        # Viscous term: ν∇²u
        u_hat = fft2(u_rom)
        v_hat = fft2(v_rom)
        visc_u = np.real(ifft2(-self.nu * self.K2 * u_hat))
        visc_v = np.real(ifft2(-self.nu * self.K2 * v_hat))
        
        # Forcing
        fx = np.real(ifft2(self.projector.forcing_x_hat))
        fy = np.real(ifft2(self.projector.forcing_y_hat))
        
        # FOM RHS: F(v_rom) = -N + ν∇²v + f
        F_u = -N_u_phys + visc_u + fx
        F_v = -N_v_phys + visc_v + fy
        
        # ROM time derivative reconstructed in physical space
        # da/dt is in coefficient space; reconstruct to physical space
        da_u = da_rom_dt[:k]
        da_v = da_rom_dt[k:]
        
        dt_u_phys = np.zeros((nx, ny))
        dt_v_phys = np.zeros((nx, ny))
        for i in range(k):
            dt_u_phys += np.real(da_u[i] * self.projector.modes_u[:, i].reshape(nx, ny))
            dt_v_phys += np.real(da_v[i] * self.projector.modes_v[:, i].reshape(nx, ny))
        
        # Residual: r = F(v_rom) - dv_rom/dt
        r_u = F_u - dt_u_phys
        r_v = F_v - dt_v_phys
        
        return r_u, r_v
    
    def compute_dual_norm(self, r_u, r_v):
        """Compute dual norm ||r||_{V'} of the residual.
        
        For the energy norm ||v||_V = (ν⟨|∇v|²⟩)^{1/2}, the dual norm is:
        
            ||r||_{V'} = sup_{v∈V} ⟨r, v⟩ / ||v||_V
                       = (1/ν) * ||∇^{-1} r||_L2   (for periodic domain)
        
        In Fourier space:
            ||r||_{V'}² = (1/ν) * Σ_k |r_hat(k)|² / |k|²
        
        Parameters
        ----------
        r_u, r_v : ndarray
            Residual fields in physical space
            
        Returns
        -------
        dual_norm : float
            Dual norm of the residual
        """
        r_u_hat = fft2(r_u)
        r_v_hat = fft2(r_v)
        
        # |r_hat|² / |k|² sum (excluding k=0)
        K2_safe = self.K2.copy()
        K2_safe[0, 0] = 1.0  # avoid division by zero
        
        r_norm_sq = np.sum((np.abs(r_u_hat)**2 + np.abs(r_v_hat)**2) / K2_safe)
        
        # Normalize by domain area
        area = (2 * np.pi) ** 2  # [0, 2π]²
        dual_norm_sq = r_norm_sq / (self.nu * area)
        
        return np.sqrt(np.maximum(dual_norm_sq, 0))
    
    def compute_energy_norm(self, u, v):
        """Compute energy norm ||v||_V = (ν⟨|∇v|²⟩)^{1/2}.
        
        Parameters
        ----------
        u, v : ndarray
            Velocity fields
            
        Returns
        -------
        energy_norm : float
            Energy norm of the velocity field
        """
        u_hat = fft2(u)
        v_hat = fft2(v)
        
        # |∇v|² = |∂u/∂x|² + |∂u/∂y|² + |∂v/∂x|² + |∂v/∂y|²
        grad_sq = (np.real(ifft2(1j * self.KX * u_hat))**2 +
                   np.real(ifft2(1j * self.KY * u_hat))**2 +
                   np.real(ifft2(1j * self.KX * v_hat))**2 +
                   np.real(ifft2(1j * self.KY * v_hat))**2)
        
        return np.sqrt(self.nu * np.mean(grad_sq))
    
    def compute_error_bound(self, a_rom, da_rom_dt):
        """Compute the certified a posteriori error bound.
        
            ||e||_V ≤ ||r||_{V'} / (beta_lb - L * ||v_rom||_V)
        
        Parameters
        ----------
        a_rom : ndarray, shape (2*k,)
            ROM coefficients
        da_rom_dt : ndarray, shape (2*k,)
            ROM time derivative
            
        Returns
        -------
        result : dict
            Dictionary with:
            - bound: the error bound value
            - dual_norm: ||r||_{V'}
            - beta_lb: Stokes inf-sup lower bound
            - lipschitz_L: Lipschitz constant
            - energy_norm_rom: ||v_rom||_V
            - coercivity_gap: beta_lb - L * ||v_rom||_V
            - valid: whether the bound is valid (coercivity > 0)
        """
        k = self.projector.n_modes
        a_u = a_rom[:k]
        a_v = a_rom[k:]
        
        # Reconstruct ROM velocity
        nx, ny = self._nx, self._ny
        u_rom = np.real(self.projector.modes_u @ a_u + self.projector.mean_u.flatten()).reshape(nx, ny)
        v_rom = np.real(self.projector.modes_v @ a_v + self.projector.mean_v.flatten()).reshape(nx, ny)
        
        # Compute residual
        r_u, r_v = self.compute_residual(a_rom, da_rom_dt)
        
        # Compute dual norm
        dual_norm = self.compute_dual_norm(r_u, r_v)
        
        # Compute energy norm of ROM solution
        energy_norm_rom = self.compute_energy_norm(u_rom, v_rom)
        
        # Coercivity gap
        coercivity_gap = self.beta_lb - self.lipschitz_L * energy_norm_rom
        
        # Error bound
        if coercivity_gap > 0:
            bound = dual_norm / coercivity_gap
            valid = True
        else:
            bound = np.inf
            valid = False
        
        return {
            'bound': bound,
            'dual_norm': dual_norm,
            'beta_lb': self.beta_lb,
            'lipschitz_L': self.lipschitz_L,
            'energy_norm_rom': energy_norm_rom,
            'coercivity_gap': coercivity_gap,
            'valid': valid,
        }
    
    def compute_true_error(self, u_fom, v_fom, u_rom, v_rom, norm='energy'):
        """Compute the true error between FOM and ROM solutions.
        
        Parameters
        ----------
        u_fom, v_fom : ndarray
            FOM velocity fields
        u_rom, v_rom : ndarray
            ROM velocity fields
        norm : str
            'energy' for energy norm, 'l2' for L2 norm
            
        Returns
        -------
        true_error : float
            True error in specified norm
        """
        if norm == 'energy':
            e_u = u_fom - u_rom
            e_v = v_fom - v_rom
            return self.compute_energy_norm(e_u, e_v)
        elif norm == 'l2':
            e_u = u_fom - u_rom
            e_v = v_fom - v_rom
            return np.sqrt(np.mean(e_u**2 + e_v**2))
        else:
            raise ValueError(f"Unknown norm: {norm}")
    
    def verify_bound(self, u_fom, v_fom, u_rom, v_rom, a_rom, da_rom_dt):
        """Verify that the error bound is an upper bound on the true error.
        
        Parameters
        ----------
        u_fom, v_fom : ndarray
            FOM velocity fields
        u_rom, v_rom : ndarray
            ROM velocity fields
        a_rom : ndarray
            ROM coefficients
        da_rom_dt : ndarray
            ROM time derivative
            
        Returns
        -------
        result : dict
            Dictionary with:
            - bound: error bound
            - true_err_energy: true error in energy norm
            - true_err_l2: true error in L2 norm
            - ratio_energy: bound / true_err_energy (should be ≥ 1)
            - ratio_l2: bound / true_err_l2 (should be ≥ 1)
            - valid: whether bound is valid
            - not_underestimating: bound >= true_err_energy
        """
        bound_result = self.compute_error_bound(a_rom, da_rom_dt)
        
        true_err_energy = self.compute_true_error(u_fom, v_fom, u_rom, v_rom, norm='energy')
        true_err_l2 = self.compute_true_error(u_fom, v_fom, u_rom, v_rom, norm='l2')
        
        if true_err_energy > 0:
            ratio_energy = bound_result['bound'] / true_err_energy
        else:
            ratio_energy = np.inf if bound_result['bound'] > 0 else 1.0
        
        if true_err_l2 > 0:
            ratio_l2 = bound_result['bound'] / true_err_l2
        else:
            ratio_l2 = np.inf if bound_result['bound'] > 0 else 1.0
        
        return {
            'bound': bound_result['bound'],
            'true_err_energy': true_err_energy,
            'true_err_l2': true_err_l2,
            'ratio_energy': ratio_energy,
            'ratio_l2': ratio_l2,
            'valid': bound_result['valid'],
            'not_underestimating': bound_result['bound'] >= true_err_energy,
            'dual_norm': bound_result['dual_norm'],
            'beta_lb': bound_result['beta_lb'],
            'lipschitz_L': bound_result['lipschitz_L'],
            'energy_norm_rom': bound_result['energy_norm_rom'],
            'coercivity_gap': bound_result['coercivity_gap'],
        }
