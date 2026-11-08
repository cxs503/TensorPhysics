"""
Discrete Empirical Interpolation Method (DEIM).

DEIM approximates the nonlinear term N(a) efficiently:
    N(a) ≈ U_m (P^T U_m)^{-1} P^T N(a)

where:
    U_m = POD modes of nonlinear term snapshots (m columns)
    P = interpolation point selection matrix (n × m)
    P^T N(a) = evaluation of N at m selected points

This reduces the complexity from O(n) to O(m) per evaluation,
where m << n is the number of DEIM points.

The interpolation points are selected via a greedy algorithm
that maximizes the information content at each step.
"""

import numpy as np
from scipy.linalg import svd, solve


class DEIMInterpolator:
    """DEIM approximation of nonlinear terms.
    
    Parameters
    ----------
    n_points : int
        Number of DEIM interpolation points (m)
    
    Attributes
    ----------
    U_deim : ndarray, shape (n_dof, m)
        POD modes of nonlinear term snapshots
    points : ndarray, shape (m,)
        Selected interpolation point indices
    M_inv : ndarray, shape (m, m)
        Precomputed (P^T U_m)^{-1}
    """
    
    def __init__(self, n_points=None):
        self.n_points = n_points
        self.U_deim = None
        self.points = None
        self.M_inv = None
        self.n_snapshots_used = None
    
    def fit(self, nonlinear_snapshots, n_points=None):
        """Fit DEIM from nonlinear term snapshots.
        
        Parameters
        ----------
        nonlinear_snapshots : list of ndarray or ndarray
            Snapshots of nonlinear term N(u(t_k))
            Each snapshot is a vector of length n_dof
        n_points : int, optional
            Number of DEIM points (default: number of snapshots)
            
        Returns
        -------
        self
        """
        if isinstance(nonlinear_snapshots, list):
            N_snap = np.array([n.flatten() for n in nonlinear_snapshots]).T  # (n_dof, N)
        else:
            N_snap = nonlinear_snapshots
        
        n_dof, n_snap = N_snap.shape
        self.n_snapshots_used = n_snap
        
        if n_points is None:
            n_points = min(n_snap, n_dof // 4)  # heuristic
        self.n_points = n_points
        
        # SVD of nonlinear snapshots
        U, s, Vt = svd(N_snap, full_matrices=False)
        
        # Take first m modes
        m = min(n_points, U.shape[1])
        self.U_deim = U[:, :m]
        
        # Greedy point selection (SORENSENS algorithm)
        self.points = self._select_points(self.U_deim, m)
        
        # Precompute (P^T U_m)^{-1}
        P_U = self.U_deim[self.points, :]  # (m, m)
        self.M_inv = np.linalg.inv(P_U)
        
        return self
    
    def _select_points(self, U, m):
        """Select interpolation points via greedy algorithm.
        
        Algorithm (from Chaturantabut & Sorensen 2010):
        1. Find point of max abs value in first column
        2. For each subsequent column, find point that maximizes
           the minimum singular value of the selected submatrix
        
        Parameters
        ----------
        U : ndarray, shape (n, m)
            Basis vectors
        m : int
            Number of points to select
            
        Returns
        -------
        points : ndarray, shape (m,)
            Selected point indices
        """
        n, m_max = U.shape
        m = min(m, m_max)
        
        points = np.zeros(m, dtype=int)
        
        # First point: max abs value in first column
        points[0] = np.argmax(np.abs(U[:, 0]))
        
        # Subsequent points
        for j in range(1, m):
            # Consider submatrix of selected points so far
            U_prev = U[points[:j], :j]  # (j, j)
            
            # For each candidate point, compute the "information gain"
            best_point = -1
            best_score = -np.inf
            
            for i in range(n):
                if i in points[:j]:
                    continue
                
                # Form candidate submatrix
                U_cand = U[np.append(points[:j], i), :j+1]
                
                # Score: minimum singular value (maximize information)
                try:
                    s = np.linalg.svd(U_cand, compute_uv=False)
                    score = np.min(s)
                except:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_point = i
            
            if best_point < 0:
                # Fallback: pick any unselected point
                for i in range(n):
                    if i not in points[:j]:
                        best_point = i
                        break
            
            points[j] = best_point
        
        return points
    
    def interpolate(self, N_at_points):
        """Reconstruct full nonlinear term from values at DEIM points.
        
        Parameters
        ----------
        N_at_points : ndarray, shape (m,)
            Values of nonlinear term at DEIM points
            
        Returns
        -------
        N_full : ndarray, shape (n_dof,)
            Approximated full nonlinear term
        """
        # N ≈ U_m (P^T U_m)^{-1} P^T N = U_m @ M_inv @ N_at_points
        coeffs = self.M_inv @ N_at_points
        return self.U_deim @ coeffs
    
    def evaluate(self, N_full_func, a):
        """Evaluate DEIM approximation of nonlinear term.
        
        Parameters
        ----------
        N_full_func : callable
            Function that computes full nonlinear term: N_full_func(a) -> ndarray
        a : ndarray
            ROM coefficients
            
        Returns
        -------
        N_approx : ndarray
            DEIM-approximated nonlinear term
        """
        # Evaluate at DEIM points only
        N_full = N_full_func(a)
        N_at_points = N_full[self.points]
        
        # Interpolate
        return self.interpolate(N_at_points)
    
    def approximation_error(self, N_full, N_at_points):
        """Compute DEIM approximation error.
        
        Parameters
        ----------
        N_full : ndarray
            True full nonlinear term
        N_at_points : ndarray
            Values at DEIM points
            
        Returns
        -------
        error : float
            Relative L2 error
        """
        N_approx = self.interpolate(N_at_points)
        return np.linalg.norm(N_full - N_approx) / np.linalg.norm(N_full)
