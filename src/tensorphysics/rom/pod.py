"""
Proper Orthogonal Decomposition (POD) via SVD.

Given snapshot matrix X = [x1, x2, ..., xN] (each xi is a flattened field),
compute:
    X = U Σ V^T
where:
    U = POD modes (spatial bases)
    Σ = singular values (√energy)
    V = temporal coefficients

Energy spectrum: E_k = σ_k² / Σ σ_j²
Cumulative energy: C_k = Σ_{j=1}^k E_j
Optimal rank k*: smallest k such that C_k ≥ threshold (e.g., 99%)
"""

import numpy as np
from scipy.linalg import svd


class PODDecomposition:
    """POD decomposition of snapshot matrix.
    
    Parameters
    ----------
    method : str
        'svd' for full SVD, 'method_of_snapshots' for efficient small-sample SVD
    
    Attributes
    ----------
    modes : ndarray, shape (n_dof, n_modes)
        POD modes (spatial bases), column-normalized
    singular_values : ndarray
        Singular values σ_k
    temporal_coeffs : ndarray, shape (n_modes, n_snapshots)
        Temporal coefficients V^T * Σ
    energy_spectrum : ndarray
        Normalized energy per mode: σ_k² / Σ σ_j²
    cumulative_energy : ndarray
        Cumulative energy fraction
    mean_field : ndarray
        Mean of snapshots (subtracted before POD)
    """
    
    def __init__(self, method='svd'):
        self.method = method
        self.modes = None
        self.singular_values = None
        self.temporal_coeffs = None
        self.energy_spectrum = None
        self.cumulative_energy = None
        self.mean_field = None
        self.n_snapshots = None
        self.n_dof = None
    
    def fit(self, snapshots, subtract_mean=True):
        """Compute POD from snapshots.
        
        Parameters
        ----------
        snapshots : list of ndarray or ndarray
            If list: each element is a snapshot (any shape)
            If ndarray: shape (n_snapshots, ...) 
        subtract_mean : bool
            Whether to subtract mean field before POD
            
        Returns
        -------
        self
        """
        # Stack snapshots into matrix
        if isinstance(snapshots, list):
            snap_array = np.array([s.flatten() for s in snapshots])  # (N, n_dof)
        else:
            orig_shape = snapshots.shape
            snap_array = snapshots.reshape(len(snapshots), -1)
        
        self.n_snapshots, self.n_dof = snap_array.shape
        self.snapshot_shape = snapshots[0].shape if isinstance(snapshots, list) else snapshots.shape[1:]
        
        # Subtract mean
        if subtract_mean:
            self.mean_field = np.mean(snap_array, axis=0)
            X = snap_array - self.mean_field
        else:
            self.mean_field = np.zeros(self.n_dof)
            X = snap_array
        
        # SVD
        if self.method == 'svd' or self.n_snapshots <= self.n_dof:
            # Direct SVD of X^T (n_dof × N)
            # X = U Σ V^T where X is (N, n_dof)
            # We want modes as columns of U (n_dof × N)
            U, s, Vt = svd(X.T, full_matrices=False)
        else:
            # Method of snapshots: SVD of X X^T (N × N)
            C = X @ X.T / (self.n_snapshots - 1)
            eigvals, eigvecs = np.linalg.eigh(C)
            # Sort descending
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            # Singular values
            s = np.sqrt(np.maximum(eigvals * (self.n_snapshots - 1), 0))
            # Modes
            U = X.T @ eigvecs / s[np.newaxis, :]
            Vt = eigvecs.T
        
        self.modes = U  # (n_dof, N)
        self.singular_values = s
        self.temporal_coeffs = np.diag(s) @ Vt  # (N, N)
        
        # Energy spectrum
        total_energy = np.sum(s**2)
        if total_energy > 0:
            self.energy_spectrum = s**2 / total_energy
        else:
            self.energy_spectrum = np.zeros_like(s)
        self.cumulative_energy = np.cumsum(self.energy_spectrum)
        
        return self
    
    def get_optimal_rank(self, threshold=0.99):
        """Get optimal rank k* for given energy threshold.
        
        Parameters
        ----------
        threshold : float
            Cumulative energy threshold (e.g., 0.99 for 99%)
            
        Returns
        -------
        k_star : int
            Smallest k such that cumulative_energy[k] >= threshold
        """
        idx = np.searchsorted(self.cumulative_energy, threshold)
        return min(idx + 1, len(self.cumulative_energy))
    
    def project(self, field, n_modes=None):
        """Project field onto POD basis.
        
        Parameters
        ----------
        field : ndarray
            Field to project (same shape as snapshots)
        n_modes : int, optional
            Number of modes to use (default: all)
            
        Returns
        -------
        coeffs : ndarray
            Projection coefficients a_k = <φ_k, field - mean>
        """
        flat = field.flatten() - self.mean_field
        if n_modes is None:
            n_modes = self.modes.shape[1]
        return self.modes[:, :n_modes].T @ flat
    
    def reconstruct(self, coeffs):
        """Reconstruct field from POD coefficients.
        
        Parameters
        ----------
        coeffs : ndarray
            POD coefficients a_k
            
        Returns
        -------
        field : ndarray
            Reconstructed field (original shape)
        """
        k = len(coeffs)
        flat = self.modes[:, :k] @ coeffs + self.mean_field
        return flat.reshape(self.snapshot_shape)
    
    def projection_error(self, field, n_modes=None):
        """Compute L2 projection error.
        
        Parameters
        ----------
        field : ndarray
            Original field
        n_modes : int, optional
            Number of modes
            
        Returns
        -------
        error : float
            Relative L2 error ‖field - reconstructed‖ / ‖field‖
        """
        coeffs = self.project(field, n_modes)
        recon = self.reconstruct(coeffs)
        return np.linalg.norm(field - recon) / np.linalg.norm(field)
