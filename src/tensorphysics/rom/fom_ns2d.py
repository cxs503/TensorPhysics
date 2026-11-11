"""
2D Incompressible Navier-Stokes solver using pseudo-spectral method.

Domain: [0, 2π]² with periodic boundary conditions.
Forcing: Kolmogorov-type forcing F = γ sin(kx x) cos(ky y) in x-direction.
Method: Fourier pseudo-spectral in space, RK4 in time.
Incompressibility enforced via projection in Fourier space.

This solver produces rich vortex dynamics suitable for ROM testing.
The periodic domain ensures clean energy identities:
    d/dt ‖u‖² = -2ν‖∇u‖² + 2⟨f, u⟩
For unforced decay: d/dt ‖u‖² = -2ν‖∇u‖²
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


class NS2DSolver:
    """2D Navier-Stokes solver using pseudo-spectral method.
    
    Solves:
        ∂u/∂t + (u·∇)u = -∇p + ν∇²u + f
        ∇·u = 0
    
    on [0, Lx] × [0, Ly] with periodic BCs.
    
    Uses skew-symmetric form for convection:
        (u·∇)u → ½[(u·∇)u + ∇·(u⊗u)]
    which ensures exact energy conservation of the nonlinear term.
    
    Parameters
    ----------
    nx, ny : int
        Number of grid points in x, y
    Lx, Ly : float
        Domain size
    nu : float
        Kinematic viscosity
    forcing_amp : float
        Amplitude of Kolmogorov forcing
    forcing_kx : int
        Wavenumber of forcing in x
    dealias : bool
        Use 2/3 dealiasing
    """
    
    def __init__(self, nx=64, ny=64, Lx=2*np.pi, Ly=2*np.pi,
                 nu=0.01, forcing_amp=1.0, forcing_kx=4, dealias=True):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.nu = nu
        self.forcing_amp = forcing_amp
        self.forcing_kx = forcing_kx
        self.dealias = dealias
        
        # Grid spacing
        self.dx = Lx / nx
        self.dy = Ly / ny
        
        # Wavenumbers
        self.kx = 2 * np.pi * fftfreq(nx, d=Lx/nx)
        self.ky = 2 * np.pi * fftfreq(ny, d=Ly/ny)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2
        
        # Dealiasing mask (2/3 rule)
        if dealias:
            kx_max = np.max(np.abs(self.kx))
            ky_max = np.max(np.abs(self.ky))
            self.dealias_mask = ((np.abs(self.KX) < 2/3 * kx_max) & 
                                 (np.abs(self.KY) < 2/3 * ky_max)).astype(float)
        else:
            self.dealias_mask = np.ones_like(self.K2)
        
        # Forcing in Fourier space (Kolmogorov forcing)
        # f_x = γ * sin(kx_forcing * x) * cos(ky_forcing * y)
        # In Fourier space: f_x_hat = γ * (1/4i) * [δ(k-k_f) - ...]
        # Simpler: compute in physical space and transform
        x = np.linspace(0, Lx, nx, endpoint=False)
        y = np.linspace(0, Ly, ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        self.forcing_x_phys = forcing_amp * np.sin(forcing_kx * X) * np.cos(2 * Y)
        self.forcing_y_phys = -forcing_amp * np.cos(forcing_kx * X) * np.sin(2 * Y)
        self.forcing_x_hat = fft2(self.forcing_x_phys)
        self.forcing_y_hat = fft2(self.forcing_y_phys)
        
        # State: velocity fields in physical space
        self.u = None  # x-velocity
        self.v = None  # y-velocity
        
    def initialize(self, u0=None, v0=None, seed=42):
        """Initialize velocity field.
        
        If u0, v0 provided, use them. Otherwise generate random divergence-free field.
        """
        if u0 is not None and v0 is not None:
            self.u = u0.copy()
            self.v = v0.copy()
        else:
            # Generate random divergence-free initial condition
            rng = np.random.RandomState(seed)
            # Random stream function
            psi_hat = (rng.randn(self.nx, self.ny) + 1j * rng.randn(self.nx, self.ny))
            psi_hat *= self.dealias_mask
            psi_hat[0, 0] = 0  # zero mean
            
            # u = dψ/dy, v = -dψ/dx (in Fourier space)
            u_hat = 1j * self.KY * psi_hat
            v_hat = -1j * self.KX * psi_hat
            
            # Scale to desired energy
            u = np.real(ifft2(u_hat))
            v = np.real(ifft2(v_hat))
            energy = 0.5 * (np.mean(u**2) + np.mean(v**2))
            if energy > 0:
                scale = 1.0 / np.sqrt(energy)
                u *= scale
                v *= scale
            
            self.u = u
            self.v = v
    
    def _compute_rhs(self, u, v):
        """Compute RHS of NS equations using skew-symmetric form.
        
        Skew-symmetric convection: N(u) = ½[(u·∇)u + ∇·(u⊗u)]
        This ensures ⟨N(u), u⟩ = 0 exactly (in continuous limit).
        """
        # Transform to Fourier space
        u_hat = fft2(u)
        v_hat = fft2(v)
        
        # Compute products in physical space (for pseudo-spectral)
        uu = u * u
        uv = u * v
        vv = v * v
        
        # Transform products to Fourier space
        uu_hat = fft2(uu)
        uv_hat = fft2(uv)
        vv_hat = fft2(vv)
        
        # Apply dealiasing
        if self.dealias:
            uu_hat *= self.dealias_mask
            uv_hat *= self.dealias_mask
            vv_hat *= self.dealias_mask
        
        # Skew-symmetric form: ½[∂(uu)/∂x + ∂(uv)/∂y + u·∇u]
        # = ½[∂(uu)/∂x + ∂(uv)/∂y] + ½[u·∇u]
        # In Fourier space:
        # ∂(uu)/∂x → ikx * uu_hat
        # ∂(uv)/∂y → iky * uv_hat
        # u·∇u = u*∂u/∂x + v*∂u/∂y
        
        # Compute ∇u in Fourier space
        dux_dx = ifft2(1j * self.KX * u_hat)
        dux_dy = ifft2(1j * self.KY * u_hat)
        dvx_dx = ifft2(1j * self.KX * v_hat)
        dvx_dy = ifft2(1j * self.KY * v_hat)
        
        # Convective terms in physical space
        conv_u = u * dux_dx + v * dux_dy
        conv_v = u * dvx_dx + v * dvx_dy
        
        # Transform to Fourier
        conv_u_hat = fft2(conv_u)
        conv_v_hat = fft2(conv_v)
        
        # Divergence form
        div_u_hat = 1j * self.KX * uu_hat + 1j * self.KY * uv_hat
        div_v_hat = 1j * self.KX * uv_hat + 1j * self.KY * vv_hat
        
        # Skew-symmetric: ½(conv + div_form)
        Nu_hat = 0.5 * (conv_u_hat + div_u_hat)
        Nv_hat = 0.5 * (conv_v_hat + div_v_hat)
        
        # Viscous term: -ν k² u_hat
        visc_u_hat = -self.nu * self.K2 * u_hat
        visc_v_hat = -self.nu * self.K2 * v_hat
        
        # Total RHS (without pressure)
        rhs_u_hat = -Nu_hat + visc_u_hat + self.forcing_x_hat
        rhs_v_hat = -Nv_hat + visc_v_hat + self.forcing_y_hat
        
        # Pressure projection: remove divergent part
        # p_hat = (ikx * rhs_u + iky * rhs_v) / k²
        div_rhs = 1j * self.KX * rhs_u_hat + 1j * self.KY * rhs_v_hat
        
        # Avoid division by zero at k=0
        K2_safe = self.K2.copy()
        K2_safe[0, 0] = 1.0
        
        # Pressure Poisson solve: Lap(p) = div(rhs)  =>  p_hat = -div_rhs/k^2
        # (legacy code used p_hat = +div_rhs/k^2, which *doubled* the
        # divergence at every step instead of removing it -- measured
        # discrete divergence 0.64 after 100 steps at 16x16.)
        p_hat = -div_rhs / K2_safe
        p_hat[0, 0] = 0.0
        
        # Projected RHS
        rhs_u_hat -= 1j * self.KX * p_hat
        rhs_v_hat -= 1j * self.KY * p_hat
        
        # Zero mean velocity (optional, keeps things clean)
        rhs_u_hat[0, 0] = 0.0
        rhs_v_hat[0, 0] = 0.0
        
        return rhs_u_hat, rhs_v_hat
    
    def step_rk4_real(self, dt):
        """RK4 step that keeps the state real (Hermitian-safe variant).

        The legacy :meth:`step_rk4` adds complex increments directly; because
        the pseudo-spectral operator ``i*k`` does not preserve Hermitian
        symmetry at the Nyquist wavenumber, a spurious imaginary component
        accumulates (measured: |Im u| ~ 5e-5 after one step, discrete
        divergence 0.6 after 100 steps at 16x16).  This variant takes the real
        part of every RK4 increment; it does not modify the legacy method.
        """
        u, v = self.u, self.v
        ru1, rv1 = self.rhs_physical(u, v)
        ru2, rv2 = self.rhs_physical(u + 0.5 * dt * ru1, v + 0.5 * dt * rv1)
        ru3, rv3 = self.rhs_physical(u + 0.5 * dt * ru2, v + 0.5 * dt * rv2)
        ru4, rv4 = self.rhs_physical(u + dt * ru3, v + dt * rv3)
        self.u = u + (dt / 6.0) * (ru1 + 2 * ru2 + 2 * ru3 + ru4)
        self.v = v + (dt / 6.0) * (rv1 + 2 * rv2 + 2 * rv3 + rv4)
        return self.u, self.v

    def run_real(self, t_end, dt, save_interval=1):
        """Same as :meth:`run` but using :meth:`step_rk4_real`."""
        n_steps = int(t_end / dt)
        su, sv, times, energies = [], [], [], []
        for n in range(n_steps + 1):
            if n % save_interval == 0:
                su.append(np.real(self.u).copy())
                sv.append(np.real(self.v).copy())
                times.append(n * dt)
                energies.append(float(np.real(self.compute_energy())))
            if n < n_steps:
                self.step_rk4_real(dt)
        return su, sv, times, energies

    def rhs_physical(self, u, v):
        """FOM right-hand side in physical space at an arbitrary state.

        Additive public wrapper around :meth:`_compute_rhs`; used by the
        a-posteriori residual evaluator.  Returns ``(F_u, F_v)``.
        """
        rhs_u_hat, rhs_v_hat = self._compute_rhs(u, v)
        return np.real(ifft2(rhs_u_hat)), np.real(ifft2(rhs_v_hat))

    def step_rk4(self, dt):
        """Advance one timestep using RK4.
        
        Returns
        -------
        u, v : ndarray
            Updated velocity fields
        """
        u, v = self.u, self.v
        
        # k1
        rhs_u1, rhs_v1 = self._compute_rhs(u, v)
        ru1 = ifft2(rhs_u1)
        rv1 = ifft2(rhs_v1)
        
        # k2
        rhs_u2, rhs_v2 = self._compute_rhs(u + 0.5*dt*ru1, v + 0.5*dt*rv1)
        ru2 = ifft2(rhs_u2)
        rv2 = ifft2(rhs_v2)
        
        # k3
        rhs_u3, rhs_v3 = self._compute_rhs(u + 0.5*dt*ru2, v + 0.5*dt*rv2)
        ru3 = ifft2(rhs_u3)
        rv3 = ifft2(rhs_v3)
        
        # k4
        rhs_u4, rhs_v4 = self._compute_rhs(u + dt*ru3, v + dt*rv3)
        ru4 = ifft2(rhs_u4)
        rv4 = ifft2(rhs_v4)
        
        # Update
        self.u = u + (dt/6.0) * (ru1 + 2*ru2 + 2*ru3 + ru4)
        self.v = v + (dt/6.0) * (rv1 + 2*rv2 + 2*rv3 + rv4)
        
        return self.u, self.v
    
    def compute_energy(self, u=None, v=None):
        """Compute kinetic energy E = ½∫|u|² dA / Area."""
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        return 0.5 * np.mean(u**2 + v**2)
    
    def compute_enstrophy(self, u=None, v=None):
        """Compute enstrophy Z = ½∫|ω|² dA / Area, ω = ∂v/∂x - ∂u/∂y."""
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        u_hat = fft2(u)
        v_hat = fft2(v)
        # ω = ∂v/∂x - ∂u/∂y
        dv_dx = ifft2(1j * self.KX * v_hat)
        du_dy = ifft2(1j * self.KY * u_hat)
        omega = np.real(dv_dx - du_dy)
        return 0.5 * np.mean(omega**2)
    
    def compute_dissipation(self, u=None, v=None):
        """Compute viscous dissipation ε = ν∫|∇u|² dA / Area."""
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        u_hat = fft2(u)
        v_hat = fft2(v)
        # |∇u|² = |∂u/∂x|² + |∂u/∂y|² + |∂v/∂x|² + |∂v/∂y|²
        grad_u2 = np.mean(np.abs(ifft2(1j * self.KX * u_hat))**2 + 
                         np.abs(ifft2(1j * self.KY * u_hat))**2 +
                         np.abs(ifft2(1j * self.KX * v_hat))**2 + 
                         np.abs(ifft2(1j * self.KY * v_hat))**2)
        return self.nu * grad_u2
    
    def compute_divergence(self, u=None, v=None):
        """Compute max divergence (should be ~0 for incompressible flow)."""
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        u_hat = fft2(u)
        v_hat = fft2(v)
        div_hat = 1j * self.KX * u_hat + 1j * self.KY * v_hat
        div_phys = ifft2(div_hat)
        return np.max(np.abs(np.real(div_phys)))
    
    def run(self, t_end, dt, save_interval=1, verbose=False):
        """Run simulation and collect snapshots.
        
        Parameters
        ----------
        t_end : float
            End time
        dt : float
            Time step
        save_interval : int
            Save snapshot every save_interval steps
        verbose : bool
            Print progress
            
        Returns
        -------
        snapshots_u : list of ndarray
            x-velocity snapshots
        snapshots_v : list of ndarray
            y-velocity snapshots
        times : list of float
            Snapshot times
        energies : list of float
            Kinetic energy at each snapshot
        """
        n_steps = int(t_end / dt)
        snapshots_u = []
        snapshots_v = []
        times = []
        energies = []
        
        for n in range(n_steps + 1):
            if n % save_interval == 0:
                snapshots_u.append(self.u.copy())
                snapshots_v.append(self.v.copy())
                times.append(n * dt)
                energies.append(self.compute_energy())
                if verbose and n % (save_interval * 10) == 0:
                    print(f"t={n*dt:.3f}, E={energies[-1]:.6f}")
            
            if n < n_steps:
                self.step_rk4(dt)
        
        return snapshots_u, snapshots_v, times, energies
