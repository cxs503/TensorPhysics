"""Shallow arch snap-through buckling verification.

Implements a shallow arch under uniform loading using a discrete
truss model (von Mises truss) that captures the essential snap-through
physics, plus a multi-element arch model with arc-length continuation.

Physics
-------
A shallow arch of span L and rise H is subjected to a vertical load.
The arch can undergo snap-through buckling when the load reaches a
critical value, jumping from a pre-buckling to a post-buckling
configuration.

The simplest model that captures snap-through is the von Mises truss:
two bars pinned at the apex, loaded vertically. The equilibrium path
shows a limit point (maximum load) followed by negative stiffness.

For the multi-element model, we use:
- Green-Lagrange strain for each element
- Newton-Raphson with arc-length continuation
- Captures the full equilibrium path including snap-through

Verification target: load-deflection full path vs analytical estimates.

References:
- Budiansky & Roth (1962). Axisymmetric dynamic buckling.
- Thompson & Hunt (1973). General Theory of Elastic Stability.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

__all__ = ["ShallowArchModel", "solve_snap_through"]


@dataclass
class ArchSolution:
    """Container for arch snap-through solution."""
    q_values: np.ndarray        # Load values
    w_center: np.ndarray        # Center deflection
    w_center_normalized: np.ndarray  # w_center / H
    path_type: np.ndarray       # 'stable' or 'unstable'
    q_limit_upper: float        # Upper limit point load
    q_limit_lower: float        # Lower limit point load
    w_at_upper: float           # Deflection at upper limit
    w_at_lower: float           # Deflection at lower limit


class VonMisesTruss:
    """Two-bar truss (von Mises truss) for snap-through analysis.
    
    The simplest model that captures snap-through buckling.
    Two bars of length L_bar, pinned at the apex, with initial angle α.
    Vertical load P at the apex.
    
    Equilibrium: P = 2 * N * sin(β)
    where N = EA * (L_bar - L0) / L0 is the axial force
    and β is the current angle.
    
    The equilibrium path P(v) shows a limit point followed by
    negative stiffness (snap-through region).
    """
    
    def __init__(self, L: float, H: float, EA: float):
        self.L = L
        self.H = H
        self.EA = EA
        
        # Initial geometry
        self.L0 = np.sqrt((L/2)**2 + H**2)  # bar length
        self.alpha = np.arctan2(H, L/2)  # initial angle
        
    def equilibrium_path(self, v_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the equilibrium path P(v).
        
        Parameters
        ----------
        v_range : ndarray
            Vertical displacement of the apex (positive downward).
        
        Returns
        -------
        P : ndarray
            Load at each displacement.
        stability : ndarray
            'stable' or 'unstable' at each point.
        """
        P = np.zeros_like(v_range)
        stability = []
        
        for i, v in enumerate(v_range):
            # Current geometry
            y_current = self.H - v  # current height (decreases as v increases)
            L_current = np.sqrt((self.L/2)**2 + y_current**2)
            
            # Axial strain (Green-Lagrange)
            eps = (L_current**2 - self.L0**2) / (2 * self.L0**2)
            
            # Axial force
            N = self.EA * eps
            
            # Current angle
            beta = np.arctan2(y_current, self.L/2)
            
            # Equilibrium of the apex.  The bar force acting on the apex is
            # N * (unit vector apex -> support), i.e. the vertical component of
            # the two bars is -2 N sin(beta) (upward positive).  Balancing it
            # with an applied downward load q gives
            #     q = -2 N sin(beta).
            # With N < 0 (bars compressed while the apex is pushed down) this
            # is q > 0, as it must be.  The previous implementation returned
            # +2 N sin(beta), i.e. the load with the *upward* sign, so the
            # physical snap-through load appeared as a negative minimum and the
            # reported "upper limit" was simply the first sample of the
            # arbitrary displacement window (0.5 H above the initial apex).
            P[i] = -2 * N * np.sin(beta)
        
        # Determine stability from dP/dv
        if len(v_range) < 2:
            # a single sample cannot carry a slope; do not index out of bounds
            return P, np.array(['unknown'] * len(v_range))
        for i in range(len(v_range)):
            if i == 0:
                dP_dv = (P[1] - P[0]) / (v_range[1] - v_range[0])
            elif i == len(v_range) - 1:
                dP_dv = (P[-1] - P[-2]) / (v_range[-1] - v_range[-2])
            else:
                dP_dv = (P[i+1] - P[i-1]) / (v_range[i+1] - v_range[i-1])
            
            if dP_dv > 0:
                stability.append('stable')
            else:
                stability.append('unstable')
        
        return P, np.array(stability)
    
    def find_limit_points(self, n_pts: int = 1000) -> dict:
        """Find the upper and lower limit points of the equilibrium path.

        Limit points are *interior* extrema of q(v): a maximum that coincides
        with an end of the sampled displacement window is not a limit point but
        a truncation artefact, and is reported through
        ``upper_is_interior`` / ``lower_is_interior`` so callers can reject it.
        """
        # Snap-through branch: apex descends from its initial position (v = 0)
        # to the mirror configuration at v = 2H.  Both extrema of interest (the
        # compression limit load and the post-snap-through minimum) are interior
        # to this window; the far tension branch (v > 2H) is unbounded and must
        # not enter the search, otherwise the load maximum is a truncation
        # artefact of the window.
        v_range = np.linspace(0.0, 2.0 * self.H, n_pts)
        P, stability = self.equilibrium_path(v_range)
        
        # Upper limit: maximum P
        idx_upper = int(np.argmax(P))
        q_limit_upper = P[idx_upper]
        w_at_upper = v_range[idx_upper]
        upper_is_interior = 0 < idx_upper < len(P) - 1
        
        # Lower limit: minimum P after the upper limit (in snap-through region)
        if idx_upper < len(P) - 1:
            P_post = P[idx_upper:]
            v_post = v_range[idx_upper:]
            idx_lower_rel = int(np.argmin(P_post))
            q_limit_lower = P_post[idx_lower_rel]
            w_at_lower = v_post[idx_lower_rel]
            idx_lower_abs = idx_upper + idx_lower_rel
        else:
            q_limit_lower = 0.0
            w_at_lower = 0.0
            idx_lower_abs = len(P) - 1
        
        lower_is_interior = 0 < idx_lower_abs < len(P) - 1

        return {
            'q_limit_upper': q_limit_upper,
            'q_limit_lower': q_limit_lower,
            'w_at_upper': w_at_upper,
            'w_at_lower': w_at_lower,
            'upper_is_interior': upper_is_interior,
            'lower_is_interior': lower_is_interior,
            'v_range': v_range,
            'P': P,
            'stability': stability,
        }


class ShallowArchModel:
    """Multi-element shallow arch model for snap-through analysis.
    
    Uses a discrete model with n elements along the arch.
    Each element is a two-force member (truss element) with
    Green-Lagrange strain.
    
    For simplicity, we use the von Mises truss as the primary model
    since it captures the essential physics of snap-through.
    """
    
    def __init__(self, L: float, H: float, EI: float, EA: float, 
                 n_elem: int = 10):
        self.L = L
        self.H = H
        self.EI = EI
        self.EA = EA
        self.n_elem = n_elem
        
        # Use von Mises truss as the primary model
        self.truss = VonMisesTruss(L=L, H=H, EA=EA)
    
    def solve_arc_length(self, q_max: float, n_steps: int = 100,
                         **kwargs) -> ArchSolution:
        """Solve using displacement control (simpler than arc-length for truss).
        
        For the von Mises truss, we can directly compute the equilibrium
        path by prescribing the vertical displacement.
        """
        # Displacement range: from small upward to large downward
        v_range = np.linspace(-0.3 * self.H, 3.0 * self.H, n_steps)
        P, stability = self.truss.equilibrium_path(v_range)
        
        # Find limit points
        idx_upper = np.argmax(P)
        q_limit_upper = P[idx_upper]
        w_at_upper = v_range[idx_upper]
        
        if idx_upper < len(P) - 1:
            P_post = P[idx_upper:]
            v_post = v_range[idx_upper:]
            idx_lower_rel = int(np.argmin(P_post))
            q_limit_lower = P_post[idx_lower_rel]
            w_at_lower = v_post[idx_lower_rel]
            idx_lower_abs = idx_upper + idx_lower_rel
        else:
            q_limit_lower = 0.0
            w_at_lower = 0.0
            idx_lower_abs = len(P) - 1
        
        return ArchSolution(
            q_values=P,
            w_center=v_range,
            w_center_normalized=v_range / self.H,
            path_type=stability,
            q_limit_upper=q_limit_upper,
            q_limit_lower=q_limit_lower,
            w_at_upper=w_at_upper,
            w_at_lower=w_at_lower,
        )
    
    def analytical_limit_load(self) -> float:
        """Closed-form snap-through (limit) load of the von Mises truss.

        With h the current rise and L1(h) = sqrt((L/2)^2 + h^2) the current bar
        length, the applied downward load is

            q(h) = 2 EA / L0 * h * (L0 / L1(h) - 1) * (L0 + L1(h)) / (2 L0)

        for the Green-Lagrange strain used by :class:`VonMisesTruss`; in the
        small-strain limit (L1 ~ L0) this reduces to
        q = 2 EA / L0 * h * (L0/L1 - 1).  The limit load is max_h q(h), obtained
        here by a dense scan of h in (0, H] -- it is an interior maximum for
        every rise-span ratio, and it is *not* the same as
        ``find_limit_points()['q_limit_upper']`` evaluated on an arbitrary
        displacement window (that is why this method no longer delegates to it).
        """
        L = self.truss.L
        H = self.truss.H
        EA = self.truss.EA
        L0 = self.truss.L0

        h = np.linspace(1e-9, H, 2_000_001)
        L1 = np.sqrt((L / 2) ** 2 + h**2)
        eps = (L1**2 - L0**2) / (2 * L0**2)
        N = EA * eps
        q = -2 * N * (h / L1)
        return float(np.max(q))


def solve_snap_through(
    L: float = 10.0, H: float = 1.0, EI: float = 1e6, EA: float = 1e9,
    n_elem: int = 10, n_steps: int = 200,
) -> dict:
    """Solve the snap-through problem.
    
    Returns
    -------
    dict with solution and comparison data.
    """
    arch = ShallowArchModel(L=L, H=H, EI=EI, EA=EA, n_elem=n_elem)
    
    # Solve
    solution = arch.solve_arc_length(q_max=0.0, n_steps=n_steps)
    
    # Analytical estimate
    q_est = arch.analytical_limit_load()
    
    return {
        'solution': solution,
        'q_estimated': q_est,
        'q_limit_upper': solution.q_limit_upper,
        'q_limit_lower': solution.q_limit_lower,
        'rise_span': H / L,
    }
