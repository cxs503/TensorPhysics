"""
Energy stability verification for ROM.

Verifies:
C1: Skew-symmetric energy identity
    FOM: d/dt ‖u‖² = -2ν‖∇u‖² (for unforced periodic flow)
    ROM: same identity should hold after projection (up to truncation error)

C2: POD cumulative energy ≥ 99% at optimal rank k*

C5: Negative control - without DEIM, energy injection may occur
    (DEIM approximation can break skew-symmetry)

The energy identity is the fundamental stability certificate for
incompressible flow ROMs.
"""

import numpy as np
from scipy.fft import fft2, ifft2


class EnergyVerifier:
    """Energy stability verification utilities.
    
    Parameters
    ----------
    nu : float
        Kinematic viscosity
    kx, ky : ndarray
        Wavenumber arrays
    """
    
    def __init__(self, nu, kx, ky):
        self.nu = nu
        self.kx = kx
        self.ky = ky
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2
    
    def compute_kinetic_energy(self, u, v):
        """Compute kinetic energy E = ½⟨u² + v²⟩."""
        return 0.5 * np.mean(u**2 + v**2)
    
    def compute_dissipation(self, u, v):
        """Compute viscous dissipation ε = ν⟨|∇u|²⟩."""
        u_hat = fft2(u)
        v_hat = fft2(v)
        
        # |∇u|² = |∂u/∂x|² + |∂u/∂y|² + |∂v/∂x|² + |∂v/∂y|²
        dudx = np.real(ifft2(1j * self.KX * u_hat))
        dudy = np.real(ifft2(1j * self.KY * u_hat))
        dvdx = np.real(ifft2(1j * self.KX * v_hat))
        dvdy = np.real(ifft2(1j * self.KY * v_hat))
        
        grad_sq = dudx**2 + dudy**2 + dvdx**2 + dvdy**2
        return self.nu * np.mean(grad_sq)
    
    def compute_energy_rate(self, energies, times):
        """Compute dE/dt from energy time series.
        
        Parameters
        ----------
        energies : array_like
            Kinetic energy at each time
        times : array_like
            Time values
            
        Returns
        -------
        dEdt : ndarray
            Energy rate (central differences)
        """
        energies = np.array(energies)
        times = np.array(times)
        dEdt = np.gradient(energies, times)
        return dEdt
    
    def verify_energy_identity_fom(self, u_snapshots, v_snapshots, times, 
                                    forcing_x_phys=None, forcing_y_phys=None):
        """Verify energy identity for FOM.
        
        For unforced flow: dE/dt = -ε
        For forced flow: dE/dt = -ε + ⟨f, u⟩
        
        Parameters
        ----------
        u_snapshots, v_snapshots : list of ndarray
            Velocity snapshots
        times : list of float
            Snapshot times
        forcing_x_phys, forcing_y_phys : ndarray, optional
            Forcing fields (if present)
            
        Returns
        -------
        results : dict
            Verification results including residual
        """
        n = len(times)
        energies = np.array([self.compute_kinetic_energy(u_snapshots[i], v_snapshots[i]) 
                            for i in range(n)])
        dissipations = np.array([self.compute_dissipation(u_snapshots[i], v_snapshots[i]) 
                                for i in range(n)])
        
        dEdt = self.compute_energy_rate(energies, times)
        
        # For unforced: dE/dt + ε = 0
        # For forced: dE/dt + ε - ⟨f, u⟩ = 0
        if forcing_x_phys is not None:
            forcing_work = np.array([
                np.mean(forcing_x_phys * u_snapshots[i] + forcing_y_phys * v_snapshots[i])
                for i in range(n)
            ])
            residual = dEdt + dissipations - forcing_work
        else:
            residual = dEdt + dissipations
        
        # Compute residual norm (excluding boundaries)
        residual_norm = np.sqrt(np.mean(residual[1:-1]**2))
        
        return {
            'energies': energies,
            'dissipations': dissipations,
            'dEdt': dEdt,
            'residual': residual,
            'residual_norm': residual_norm,
            'times': np.array(times),
        }
    
    def verify_energy_identity_rom(self, rom_solutions, times, projector):
        """Verify energy identity for ROM.
        
        Parameters
        ----------
        rom_solutions : list of ndarray
            ROM coefficient snapshots
        times : list of float
            Snapshot times
        projector : GalerkinProjector
            Galerkin projector
            
        Returns
        -------
        results : dict
            Verification results
        """
        nx = len(projector.kx)
        ny = len(projector.ky)
        n = len(times)
        
        energies = []
        dissipations = []
        
        for a in rom_solutions:
            k = projector.n_modes
            a_u = a[:k]
            a_v = a[k:]
            
            u = (projector.modes_u @ a_u + projector.mean_u.flatten()).reshape(nx, ny)
            v = (projector.modes_v @ a_v + projector.mean_v.flatten()).reshape(nx, ny)
            
            energies.append(self.compute_kinetic_energy(u, v))
            dissipations.append(self.compute_dissipation(u, v))
        
        energies = np.array(energies)
        dissipations = np.array(dissipations)
        dEdt = self.compute_energy_rate(energies, times)
        
        residual = dEdt + dissipations
        residual_norm = np.sqrt(np.mean(residual[1:-1]**2))
        
        return {
            'energies': energies,
            'dissipations': dissipations,
            'dEdt': dEdt,
            'residual': residual,
            'residual_norm': residual_norm,
            'times': np.array(times),
        }
    
    def check_skew_symmetry_discrete(self, u, v, tol=1e-10):
        """Check discrete skew-symmetry of convection.
        
        For skew-symmetric form: ⟨N(u), u⟩ = 0
        
        Parameters
        ----------
        u, v : ndarray
            Velocity fields
        tol : float
            Tolerance
            
        Returns
        -------
        is_skew : bool
            Whether skew-symmetry holds
        residual : float
            |⟨N(u), u⟩|
        """
        nx, ny = u.shape
        KX, KY = self.KX, self.KY
        
        u_hat = fft2(u)
        v_hat = fft2(v)
        
        # Products
        uu = u * u
        uv = u * v
        vv = v * v
        
        uu_hat = fft2(uu)
        uv_hat = fft2(uv)
        vv_hat = fft2(vv)
        
        # Convective form
        dux_dx = np.real(ifft2(1j * KX * u_hat))
        dux_dy = np.real(ifft2(1j * KY * u_hat))
        dvx_dx = np.real(ifft2(1j * KX * v_hat))
        dvx_dy = np.real(ifft2(1j * KY * v_hat))
        
        conv_u = u * dux_dx + v * dux_dy
        conv_v = u * dvx_dx + v * dvx_dy
        
        # Divergence form
        div_u = np.real(ifft2(1j * KX * uu_hat + 1j * KY * uv_hat))
        div_v = np.real(ifft2(1j * KX * uv_hat + 1j * KY * vv_hat))
        
        # Skew-symmetric form
        N_u = 0.5 * (conv_u + div_u)
        N_v = 0.5 * (conv_v + div_v)
        
        # Check ⟨N(u), u⟩ = 0
        residual = np.abs(np.mean(N_u * u + N_v * v))
        
        return residual < tol, residual
