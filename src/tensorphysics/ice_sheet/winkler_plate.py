"""Winkler elastic foundation plate model for ice sheet mechanics.

Implements the Winkler foundation model for floating ice sheets:
  - k_w parameterization based on buoyancy restoring force (Archimedes)
  - Navier double-Fourier-series analytical solution for simply supported
    rectangular plates on Winkler foundation (Hetényi-type)
  - Finite-difference numerical solver via coupled Laplacian splitting
  - Quantitative comparison with analytical reference

Governing equation
------------------
    D ∇⁴w + k_w w = q

where D = Eh³/[12(1-ν²)] is flexural rigidity, k_w is the Winkler
foundation modulus, q is the transverse load, and w is the deflection.

For a floating ice sheet the Winkler modulus arises from buoyancy:
    k_w = ρ_w · g

References
----------
- Hetényi, M. (1946). Beams on Elastic Foundation. Univ. Michigan Press.
- Squire, V.A., Dugan, J.P., Wadhams, P., Rottier, P.J., & Liu, A.K.
  (1996). Of Ocean Waves and Sea Ice. Annu. Rev. Fluid Mech., 28, 115-168.
- Kerr, A.D. (1975). The Bearing Capacity of Floating Ice Plates Subjected
  to Static or Quasi-Static Loads. CRREL Report 75-1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class WinklerPlateParams:
    """Parameters for a Winkler elastic foundation plate (ice sheet model).

    All SI units.

    Attributes
    ----------
    a : float
        Plate length in x-direction [m].
    b : float
        Plate width in y-direction [m].
    h : float
        Plate (ice) thickness [m].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio [-].
    k_w : float
        Winkler foundation modulus [Pa/m].
    q : float
        Uniform transverse load [Pa].
    """

    a: float
    b: float
    h: float
    E: float
    nu: float
    k_w: float
    q: float

    @property
    def D(self) -> float:
        """Flexural rigidity D = Eh³ / [12(1-ν²)] [N·m]."""
        return self.E * self.h ** 3 / (12.0 * (1.0 - self.nu ** 2))

    @property
    def char_length(self) -> float:
        """Characteristic length l = (D / k_w)^{1/4} [m]."""
        if self.k_w <= 0:
            return float("inf")
        return (self.D / self.k_w) ** 0.25


def winkler_kw_parametric(rho_w: float = 1025.0, g: float = 9.81) -> float:
    """Compute Winkler foundation modulus for a floating ice sheet.

    The buoyancy restoring force per unit deflection per unit area:
        k_w = ρ_w · g

    A unit downward deflection *w* of the ice sheet displaces additional
    water volume, generating an upward pressure ρ_w · g · w (Archimedes).

    Parameters
    ----------
    rho_w : float
        Water density [kg/m³]. Default 1025 (seawater).
    g : float
        Gravitational acceleration [m/s²]. Default 9.81.

    Returns
    -------
    float
        Winkler foundation modulus k_w [Pa/m].
    """
    return rho_w * g


def navier_center_deflection(
    params: WinklerPlateParams, n_terms: int = 51
) -> float:
    """Analytical center deflection via Navier double Fourier series.

    For a simply supported rectangular plate on Winkler foundation under
    uniform load *q*, the deflection is

        w(x,y) = Σ_m Σ_n W_mn sin(mπx/a) sin(nπy/b)

    with
        W_mn = q_mn / [D π⁴ (m²/a² + n²/b²)² + k_w]
        q_mn = 16 q / (m n π²)     for m, n odd;  0 otherwise.

    Center deflection at (a/2, b/2):

        w_c = Σ_{m,n odd} W_mn · (-1)^{(m-1)/2} · (-1)^{(n-1)/2}

    Parameters
    ----------
    params : WinklerPlateParams
    n_terms : int
        Number of odd harmonics per direction
        (m, n = 1, 3, …, 2·n_terms − 1).

    Returns
    -------
    float
        Center deflection w_c [m].
    """
    a, b = params.a, params.b
    D, k_w, q = params.D, params.k_w, params.q
    pi = math.pi
    w_c = 0.0
    for mi in range(n_terms):
        m = 2 * mi + 1
        sin_m = (-1) ** mi  # sin(m π/2) for odd m
        for ni in range(n_terms):
            n = 2 * ni + 1
            sin_n = (-1) ** ni  # sin(n π/2) for odd n
            q_mn = 16.0 * q / (m * n * pi ** 2)
            denom = D * pi ** 4 * (m ** 2 / a ** 2 + n ** 2 / b ** 2) ** 2 + k_w
            W_mn = q_mn / denom
            w_c += W_mn * sin_m * sin_n
    return w_c


def navier_deflection_field(
    params: WinklerPlateParams,
    nx: int = 51,
    ny: int = 51,
    n_terms: int = 51,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full deflection field via Navier double Fourier series.

    Parameters
    ----------
    params : WinklerPlateParams
    nx, ny : int
        Grid points (including boundary).
    n_terms : int
        Number of odd Fourier harmonics per direction.

    Returns
    -------
    x : ndarray, shape (nx,)
    y : ndarray, shape (ny,)
    w : ndarray, shape (nx, ny)
        Deflection field [m].
    """
    a, b = params.a, params.b
    D, k_w, q = params.D, params.k_w, params.q
    pi = math.pi
    x = np.linspace(0, a, nx)
    y = np.linspace(0, b, ny)
    w = np.zeros((nx, ny))
    for mi in range(n_terms):
        m = 2 * mi + 1
        sx = np.sin(m * pi * x / a)
        for ni in range(n_terms):
            n = 2 * ni + 1
            sy = np.sin(n * pi * y / b)
            q_mn = 16.0 * q / (m * n * pi ** 2)
            denom = D * pi ** 4 * (m ** 2 / a ** 2 + n ** 2 / b ** 2) ** 2 + k_w
            W_mn = q_mn / denom
            w += W_mn * np.outer(sx, sy)
    return x, y, w


def solve_winkler_plate_fd(
    params: WinklerPlateParams,
    nx: int = 51,
    ny: int = 51,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve Winkler foundation plate by finite differences.

    Solves  D ∇⁴w + k_w w = q  using the coupled Laplacian splitting:

        ∇²w  = φ
        ∇²φ  = (q − k_w w) / D

    with simply supported BCs:  w = 0,  ∇²w = 0  on boundary.

    Parameters
    ----------
    params : WinklerPlateParams
    nx, ny : int
        Grid points including boundary.

    Returns
    -------
    x : ndarray (nx,)
    y : ndarray (ny,)
    w : ndarray (nx, ny)
    """
    a, b = params.a, params.b
    D, k_w, q_val = params.D, params.k_w, params.q
    hx = a / (nx - 1)
    hy = b / (ny - 1)

    nix = nx - 2  # interior points in x
    niy = ny - 2  # interior points in y
    n_int = nix * niy
    N = 2 * n_int  # total DOFs: w and φ at each interior point

    # Laplacian stencil coefficients
    cx = 1.0 / hx ** 2
    cy = 1.0 / hy ** 2
    cc = -2.0 * (cx + cy)

    # Build sparse-like dense matrix (direct solve for accuracy)
    A = np.zeros((N, N))
    rhs = np.zeros(N)

    for i in range(nix):
        for j in range(niy):
            iw = i * niy + j          # w-DOF index
            ip = n_int + i * niy + j  # φ-DOF index

            # --- Equation 1: ∇²w = φ  →  ∇²w − φ = 0 ---
            A[iw, iw] = cc
            if i > 0:
                A[iw, (i - 1) * niy + j] = cx
            if i < nix - 1:
                A[iw, (i + 1) * niy + j] = cx
            if j > 0:
                A[iw, i * niy + (j - 1)] = cy
            if j < niy - 1:
                A[iw, i * niy + (j + 1)] = cy
            A[iw, ip] = -1.0
            rhs[iw] = 0.0

            # --- Equation 2: ∇²φ = (q − k_w w)/D  →  ∇²φ + (k_w/D)w = q/D ---
            A[ip, ip] = cc
            if i > 0:
                A[ip, n_int + (i - 1) * niy + j] = cx
            if i < nix - 1:
                A[ip, n_int + (i + 1) * niy + j] = cx
            if j > 0:
                A[ip, n_int + i * niy + (j - 1)] = cy
            if j < niy - 1:
                A[ip, n_int + i * niy + (j + 1)] = cy
            A[ip, iw] = k_w / D
            rhs[ip] = q_val / D

    sol = np.linalg.solve(A, rhs)

    # Assemble full field (boundary = 0)
    x = np.linspace(0, a, nx)
    y = np.linspace(0, b, ny)
    w = np.zeros((nx, ny))
    for i in range(nix):
        for j in range(niy):
            w[i + 1, j + 1] = sol[i * niy + j]

    return x, y, w
