"""
Galerkin projection for POD-ROM.

Given FOM: du/dt = F(u) = N(u) + L(u) + f
where N = nonlinear (convection), L = linear (diffusion), f = forcing

POD-Galerkin ROM: da/dt = Φ^T F(Φa + ū)
where Φ = POD modes, a = ROM coefficients, ū = mean field

Key property: skew-symmetric convection ensures
    a^T (Φ^T N(Φa)) = 0  (energy conservation of nonlinear term)
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


class GalerkinProjector:
    """Galerkin projection of NS operators onto POD basis.
    
    The projected system:
        da/dt = Φ^T [N(Φa + ū) + L(Φa + ū) + f]
    
    where:
        N(u) = skew-symmetric convection
        L(u) = ν∇²u (diffusion)
        f = external forcing
    
    For efficiency, we precompute:
        - Linear operator: A = Φ^T L Φ (k×k matrix)
        - Forcing projection: f_proj = Φ^T f (k-vector)
    
    The nonlinear term is evaluated via DEIM or directly.
    """
    
    def __init__(self, pod, nu, kx, ky, forcing_x_hat=None, forcing_y_hat=None):
        """
        Parameters
        ----------
        pod : PODDecomposition
            Fitted POD decomposition
        nu : float
            Kinematic viscosity
        kx, ky : ndarray
            Wavenumber arrays
        forcing_x_hat, forcing_y_hat : ndarray
            Forcing in Fourier space (optional)
        """
        self.pod = pod
        self.nu = nu
        self.kx = kx
        self.ky = ky
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2
        self.forcing_x_hat = forcing_x_hat
        self.forcing_y_hat = forcing_y_hat
        
        # Precomputed operators (set by precompute())
        self.A_uu = None  # Linear operator for u
        self.A_vv = None  # Linear operator for v
        self.f_proj_u = None  # Forcing projection for u
        self.f_proj_v = None  # Forcing projection for v
    
    def precompute(self, n_modes=None):
        """Precompute linear operators.
        
        Parameters
        ----------
        n_modes : int, optional
            Number of modes to use
        """
        if n_modes is None:
            n_modes = self.pod.modes.shape[1]
        
        self.n_modes = n_modes
        nx = len(self.kx)
        ny = len(self.ky)
        
        # Split POD modes into u and v components
        # modes shape: (nx*ny*2, k) if combined, or we handle separately
        # For our implementation, we store u and v modes separately
        # modes_u: (nx*ny, k), modes_v: (nx*ny, k)
        
        # Actually, let's handle the combined state
        # State vector: [u_flat; v_flat] of length 2*nx*ny
        # modes: (2*nx*ny, k)
        
        n_dof_half = nx * ny
        self.n_dof_half = n_dof_half
        
        # Extract u and v modes
        self.modes_u = self.pod.modes[:n_dof_half, :n_modes]  # (n_dof_half, k)
        self.modes_v = self.pod.modes[n_dof_half:, :n_modes]  # (n_dof_half, k)
        
        # Mean field split
        self.mean_u = self.pod.mean_field[:n_dof_half].reshape(nx, ny)
        self.mean_v = self.pod.mean_field[n_dof_half:].reshape(nx, ny)
        
        # Linear operator: L(u) = ν∇²u
        # In Fourier space: L_hat = -ν k² u_hat
        # Galerkin: A_ij = <φ_i, L(φ_j)> = -ν <φ_i, k² φ_j>
        
        # For each mode, compute ∇² in Fourier space
        A_uu = np.zeros((n_modes, n_modes))
        A_vv = np.zeros((n_modes, n_modes))
        
        for j in range(n_modes):
            # Mode j in physical space
            phi_u_j = self.modes_u[:, j].reshape(nx, ny)
            phi_v_j = self.modes_v[:, j].reshape(nx, ny)
            
            # Laplacian in Fourier space
            lap_u_j = np.real(ifft2(-self.K2 * fft2(phi_u_j)))
            lap_v_j = np.real(ifft2(-self.K2 * fft2(phi_v_j)))
            
            for i in range(n_modes):
                phi_u_i = self.modes_u[:, i].reshape(nx, ny)
                phi_v_i = self.modes_v[:, i].reshape(nx, ny)
                
                A_uu[i, j] = self.nu * np.sum(phi_u_i * lap_u_j) / (nx * ny)
                A_vv[i, j] = self.nu * np.sum(phi_v_i * lap_v_j) / (nx * ny)
        
        self.A_uu = A_uu
        self.A_vv = A_vv
        
        # Forcing projection
        if self.forcing_x_hat is not None:
            fx_phys = np.real(ifft2(self.forcing_x_hat))
            fy_phys = np.real(ifft2(self.forcing_y_hat))
            self.f_proj_u = np.array([np.sum(self.modes_u[:, i].reshape(nx, ny) * fx_phys) / (nx * ny) 
                                       for i in range(n_modes)])
            self.f_proj_v = np.array([np.sum(self.modes_v[:, i].reshape(nx, ny) * fy_phys) / (nx * ny) 
                                       for i in range(n_modes)])
        else:
            self.f_proj_u = np.zeros(n_modes)
            self.f_proj_v = np.zeros(n_modes)
    
    def compute_nonlinear(self, a_u, use_skew_symmetric=True):
        """Compute nonlinear term projection: Φ^T N(Φa + ū).
        
        Parameters
        ----------
        a_u : ndarray, shape (k,)
            ROM coefficients for u (we use same for v in this simplified version)
        use_skew_symmetric : bool
            Use skew-symmetric form for convection
            
        Returns
        -------
        N_proj_u, N_proj_v : ndarray
            Projected nonlinear terms
        """
        nx = len(self.kx)
        ny = len(self.ky)
        k = len(a_u)
        
        # Reconstruct full field (using same coefficients for u and v)
        u = self.modes_u @ a_u + self.mean_u.flatten()
        v = self.modes_v @ a_u + self.mean_v.flatten()
        
        u_phys = u.reshape(nx, ny)
        v_phys = v.reshape(nx, ny)
        
        # Compute nonlinear term N(u) = (u·∇)u in skew-symmetric form
        u_hat = fft2(u_phys)
        v_hat = fft2(v_phys)
        
        # Products
        uu = u_phys * u_phys
        uv = u_phys * v_phys
        vv = v_phys * v_phys
        
        uu_hat = fft2(uu)
        uv_hat = fft2(uv)
        vv_hat = fft2(vv)
        
        # Convective derivatives
        dux_dx = ifft2(1j * self.KX * u_hat)
        dux_dy = ifft2(1j * self.KY * u_hat)
        dvx_dx = ifft2(1j * self.KX * v_hat)
        dvx_dy = ifft2(1j * self.KY * v_hat)
        
        conv_u = np.real(u_phys * dux_dx + v_phys * dux_dy)
        conv_v = np.real(u_phys * dvx_dx + v_phys * dvx_dy)
        
        if use_skew_symmetric:
            # Add divergence form
            div_u = np.real(ifft2(1j * self.KX * uu_hat + 1j * self.KY * uv_hat))
            div_v = np.real(ifft2(1j * self.KX * uv_hat + 1j * self.KY * vv_hat))
            
            N_u = 0.5 * (conv_u + div_u)
            N_v = 0.5 * (conv_v + div_v)
        else:
            N_u = conv_u
            N_v = conv_v
        
        # Project onto POD modes
        N_proj_u = np.array([np.sum(self.modes_u[:, i].reshape(nx, ny) * N_u) / (nx * ny) 
                             for i in range(k)])
        N_proj_v = np.array([np.sum(self.modes_v[:, i].reshape(nx, ny) * N_v) / (nx * ny) 
                             for i in range(k)])
        
        return N_proj_u, N_proj_v
    
    def rhs(self, a):
        """Compute ROM RHS: da/dt = -N(a) + A*a + f_proj.
        
        Parameters
        ----------
        a : ndarray, shape (2*k,)
            ROM coefficients [a_u; a_v]
            
        Returns
        -------
        da_dt : ndarray, shape (2*k,)
            Time derivative of ROM coefficients
        """
        k = self.n_modes
        a_u = a[:k]
        a_v = a[k:]
        
        # Nonlinear term
        N_u, N_v = self.compute_nonlinear(a_u, use_skew_symmetric=True)
        
        # Linear term
        L_u = self.A_uu @ a_u
        L_v = self.A_vv @ a_v
        
        # Combine
        da_u = -N_u + L_u + self.f_proj_u
        da_v = -N_v + L_v + self.f_proj_v
        
        return np.concatenate([da_u, da_v])
    
    def check_skew_symmetry(self, a_u, tol=1e-10):
        """Check that nonlinear term is skew-symmetric: a^T N(a) = 0.
        
        Parameters
        ----------
        a_u : ndarray
            ROM coefficients for u
        tol : float
            Tolerance for skew-symmetry check
            
        Returns
        -------
        is_skew : bool
            Whether skew-symmetry holds
        residual : float
            |a^T N(a)|
        """
        N_u, N_v = self.compute_nonlinear(a_u, use_skew_symmetric=True)
        
        # a^T N(a) should be 0 for skew-symmetric N
        residual = np.abs(np.dot(a_u, N_u) + np.dot(a_u, N_v))
        return residual < tol, residual
