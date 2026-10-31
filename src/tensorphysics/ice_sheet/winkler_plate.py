"""Winkler elastic foundation plate model for floating ice sheets.

Implements the biharmonic plate equation on a Winkler foundation:
    D ∇⁴w + k_w w = q

where D = Eh³/[12(1−ν²)] is the flexural rigidity and k_w = ρ_w·g
is the buoyancy-derived Winkler modulus.

Upstream: SIR-PC2|F01 (m667)
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class WinklerPlateParams:
    """Parameters for a rectangular plate on Winkler elastic foundation.

    Attributes
    ----------
    a, b : float
        Plate dimensions [m].
    h : float
        Plate thickness [m].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
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
    q: float = 0.0

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
    """Analytical center deflection via Navier double Fourier series."""
    a, b = params.a, params.b
    D, k_w, q = params.D, params.k_w, params.q
    pi = math.pi
    w_c = 0.0
    for mi in range(n_terms):
        m = 2 * mi + 1
        sin_m = (-1) ** mi
        for ni in range(n_terms):
            n = 2 * ni + 1
            sin_n = (-1) ** ni
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
    """Full deflection field via Navier double Fourier series."""
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
    """Solve Winkler foundation plate by finite differences."""
    a, b = params.a, params.b
    D, k_w, q_val = params.D, params.k_w, params.q
    hx = a / (nx - 1)
    hy = b / (ny - 1)

    nix = nx - 2
    niy = ny - 2
    n_int = nix * niy
    N = 2 * n_int

    cx = 1.0 / hx ** 2
    cy = 1.0 / hy ** 2
    cc = -2.0 * (cx + cy)

    A = np.zeros((N, N))
    rhs = np.zeros(N)

    for i in range(nix):
        for j in range(niy):
            iw = i * niy + j
            ip = n_int + i * niy + j

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

    x = np.linspace(0, a, nx)
    y = np.linspace(0, b, ny)
    w = np.zeros((nx, ny))
    for i in range(nix):
        for j in range(niy):
            w[i + 1, j + 1] = sol[i * niy + j]

    return x, y, w
