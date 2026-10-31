"""Hetényi (1946) series solution for simply-supported plate on Winkler foundation.

Reference
---------
Hetényi, M. (1946). *Beams on Elastic Foundation*. University of Michigan Press.
  – Chapter X: "Rectangular Plates on Elastic Foundations"
  – The Navier double-sine series for simply-supported edges is the classical
    closed-form benchmark against which FEM ice-plate solvers are validated.

Governing equation
------------------
    D ∇⁴w + k_w w = q(x,y)

where
    D  = E h³ / [12(1 − ν²)]   plate flexural rigidity  [N·m]
    k_w                         Winkler foundation modulus [Pa/m]
    q                           transverse load intensity  [Pa]

For a simply-supported rectangular plate (a × b) under uniform load q,
the exact Navier–Hetényi series is:

    w(x,y) = Σ_m Σ_n  W_mn  sin(mπx/a) sin(nπy/b)

with  m, n = 1, 3, 5, …  (odd only for uniform load on full plate)

    W_mn = q_mn / [D π⁴ (m²/a² + n²/b²)² + k_w]
    q_mn = 16 q / (m n π²)

Center deflection (x = a/2, y = b/2):

    w_c = Σ_{m,n odd} W_mn · (−1)^{(m−1)/2} · (−1)^{(n−1)/2}

This module provides:
    1. Center deflection via truncated series (any n_terms).
    2. Full deflection field on a grid.
    3. Convergence study: partial sums vs. reference (high n_terms).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HetenyiPlateParams:
    """Parameters for a simply-supported rectangular plate on Winkler foundation.

    Attributes
    ----------
    a : float
        Plate length in x-direction [m].
    b : float
        Plate width in y-direction [m].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio [-].
    h : float
        Plate thickness [m].
    k_w : float
        Winkler foundation modulus [Pa/m].
    q : float
        Uniform transverse load [Pa].  Positive = downward.
    """
    a: float
    b: float
    E: float
    nu: float
    h: float
    k_w: float
    q: float

    @property
    def D(self) -> float:
        """Flexural rigidity D = E h³ / [12(1 − ν²)]  [N·m]."""
        return self.E * self.h ** 3 / (12.0 * (1.0 - self.nu ** 2))

    @property
    def char_length(self) -> float:
        """Characteristic length l = (D / k_w)^{1/4}  [m]."""
        if self.k_w <= 0:
            return float("inf")
        return (self.D / self.k_w) ** 0.25

    @property
    def relative_stiffness(self) -> float:
        """Dimensionless relative stiffness λ = a / l."""
        l = self.char_length
        if l == float("inf"):
            return 0.0
        return self.a / l


# ---------------------------------------------------------------------------
# Core series evaluation
# ---------------------------------------------------------------------------

def _single_term_weight(m: int, n: int, params: HetenyiPlateParams) -> float:
    """Compute W_mn for one (m, n) pair (both odd).

    W_mn = [16 q / (m n π²)] / [D π⁴ (m²/a² + n²/b²)² + k_w]
    """
    a, b = params.a, params.b
    D, k_w, q = params.D, params.k_w, params.q
    pi = math.pi
    q_mn = 16.0 * q / (m * n * pi ** 2)
    denom = D * pi ** 4 * (m ** 2 / a ** 2 + n ** 2 / b ** 2) ** 2 + k_w
    return q_mn / denom


def hetenyi_center_deflection(params: HetenyiPlateParams, n_terms: int = 51) -> float:
    """Center deflection w(a/2, b/2) via Hetényi/Navier double Fourier series.

    Parameters
    ----------
    params : HetenyiPlateParams
    n_terms : int
        Number of odd harmonics per direction.
        Harmonics used: m, n = 1, 3, 5, …, 2·n_terms − 1.

    Returns
    -------
    float
        Center deflection [m].  Positive = downward (same sign as q).
    """
    a, b = params.a, params.b
    w_c = 0.0
    for mi in range(n_terms):
        m = 2 * mi + 1
        sin_m_half = (-1.0) ** mi  # sin(m π / 2) for odd m
        for ni in range(n_terms):
            n = 2 * ni + 1
            sin_n_half = (-1.0) ** ni
            W_mn = _single_term_weight(m, n, params)
            w_c += W_mn * sin_m_half * sin_n_half
    return w_c


def hetenyi_deflection_field(
    params: HetenyiPlateParams,
    nx: int = 51,
    ny: int = 51,
    n_terms: int = 51,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full deflection field w(x, y) via Hetényi/Navier series.

    Parameters
    ----------
    params : HetenyiPlateParams
    nx, ny : int
        Number of grid points in each direction (including boundaries).
    n_terms : int
        Number of odd harmonics per direction.

    Returns
    -------
    x : ndarray, shape (nx,)
    y : ndarray, shape (ny,)
    w : ndarray, shape (nx, ny)
        Deflection [m].
    """
    a, b = params.a, params.b
    pi = math.pi
    x = np.linspace(0.0, a, nx)
    y = np.linspace(0.0, b, ny)
    w = np.zeros((nx, ny))

    for mi in range(n_terms):
        m = 2 * mi + 1
        sx = np.sin(m * pi * x / a)
        for ni in range(n_terms):
            n = 2 * ni + 1
            sy = np.sin(n * pi * y / b)
            W_mn = _single_term_weight(m, n, params)
            w += W_mn * np.outer(sx, sy)

    return x, y, w


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def hetenyi_convergence_study(
    params: HetenyiPlateParams,
    n_terms_list: List[int] | None = None,
) -> dict:
    """Study convergence of center deflection with series truncation.

    The double Fourier series for a SS plate does NOT have monotonically
    increasing partial sums (cross-terms with mixed signs exist).  Instead,
    we check that the *absolute error* |w_n − w_ref| decreases monotonically
    as n_terms increases — this is the physically meaningful convergence
    criterion for alternating-sign series.

    Parameters
    ----------
    params : HetenyiPlateParams
    n_terms_list : list of int, optional
        Truncation levels to evaluate.  Default: [1, 2, 3, 5, 10, 20, 50, 100, 200].

    Returns
    -------
    dict with keys:
        n_terms_list : list[int]
        w_center : list[float]   – center deflection at each truncation
        relative_to_reference : list[float]  – |w_n − w_ref| / |w_ref|
        reference_n : int  – the highest n_terms used as reference
        reference_w : float
        is_monotone : bool  – True if relative error decreases monotonically
    """
    if n_terms_list is None:
        n_terms_list = [1, 2, 3, 5, 10, 20, 50, 100, 200]

    n_terms_list = sorted(n_terms_list)
    w_values = [hetenyi_center_deflection(params, n) for n in n_terms_list]

    # Reference = highest truncation
    ref_idx = len(n_terms_list) - 1
    w_ref = w_values[ref_idx]
    n_ref = n_terms_list[ref_idx]

    rel_errors = []
    for w_n in w_values:
        if abs(w_ref) > 0:
            rel_errors.append(abs(w_n - w_ref) / abs(w_ref))
        else:
            rel_errors.append(0.0)

    # Check monotone convergence of ERROR (not partial sums).
    # For a double Fourier series with alternating signs, the partial sums
    # oscillate, but the error |w_n - w_ref| should decrease monotonically
    # once we're past the first few terms.  We check from n_terms >= 2 onward
    # to avoid the very coarse truncation regime.
    #
    # Actually, for the Hetenyi series the dominant (1,1) term is always positive,
    # and higher terms add smaller corrections.  The error should decrease
    # monotonically for all n >= 1 because each new "ring" of terms adds
    # contributions that are smaller in magnitude than the previous ring.
    is_monotone = all(
        rel_errors[i] >= rel_errors[i + 1] - 1e-15
        for i in range(len(rel_errors) - 1)
    )

    return {
        "n_terms_list": n_terms_list,
        "w_center": w_values,
        "relative_to_reference": rel_errors,
        "reference_n": n_ref,
        "reference_w": w_ref,
        "is_monotone": is_monotone,
    }


# ---------------------------------------------------------------------------
# Ice-specific convenience
# ---------------------------------------------------------------------------

def ice_plate_default_params(
    a: float = 10.0,
    h: float = 0.5,
    E: float = 5.0e9,
    nu: float = 0.3,
    q: float = 1000.0,
    rho_w: float = 1025.0,
    g: float = 9.81,
) -> HetenyiPlateParams:
    """Create HetenyiPlateParams with typical Arctic ice-sheet values.

    Default: 10 m square plate, 0.5 m thick, E = 5 GPa, uniform load 1 kPa,
    seawater buoyancy foundation k_w = ρ_w g ≈ 10055 Pa/m.
    """
    k_w = rho_w * g
    return HetenyiPlateParams(a=a, b=a, E=E, nu=nu, h=h, k_w=k_w, q=q)
