"""
Thin-walled torsion constants: Saint-Venant J for various cross-sections.

References:
- Roark's Formulas for Stress and Strain, 7th Ed., Table 10.1
- Bredt's formula for single-cell closed sections: J = 4*A_m^2 / ∮(ds/t)
- Vlasov thin-walled beam theory
"""

import numpy as np
from typing import Tuple, List, Optional


def saint_venant_J_rectangular(b: float, h: float, n_terms: int = 20) -> float:
    """
    Saint-Venant torsion constant for a solid rectangular section.

    Uses the series expansion (Roark 7th Ed., Table 10.1):
        J = beta * b * h^3
    where b >= h and:
        beta = (1/3) * [1 - (192/pi^5) * (h/b) * sum_{n odd} tanh(n*pi*b/(2*h)) / n^5]

    Parameters
    ----------
    b : float
        Width of rectangle (larger dimension).
    h : float
        Height of rectangle (smaller dimension).
    n_terms : int
        Number of terms in the series (default 20, gives <0.1% error).

    Returns
    -------
    J : float
        Saint-Venant torsion constant [m^4].
    """
    # Ensure b >= h for series convergence
    if h > b:
        b, h = h, b

    if h <= 0 or b <= 0:
        raise ValueError("Dimensions must be positive")

    ratio = b / h  # >= 1

    # Series: sum_{n odd} tanh(n*pi*b/(2*h)) / n^5
    sum_series = 0.0
    for n in range(1, 2 * n_terms, 2):  # odd terms only
        sum_series += np.tanh(n * np.pi * b / (2 * h)) / (n ** 5)

    beta = (1.0 / 3.0) * (1.0 - (192.0 / (np.pi ** 5)) * (h / b) * sum_series)

    # For very wide sections (b/h > 10), use thin-strip approximation
    if ratio > 10:
        beta_thin = 1.0 / 3.0 * (1.0 - 0.63 * h / b)
        J_series = beta * b * h ** 3
        J_thin = beta_thin * b * h ** 3
        return max(J_series, J_thin)

    J = beta * b * h ** 3
    return J


def roark_J_rectangular(b: float, h: float, n_terms: int = 20) -> float:
    """
    Alias for saint_venant_J_rectangular — Roark's formula for solid rectangular J.

    Parameters
    ----------
    b : float
        Width (larger dimension).
    h : float
        Height (smaller dimension).
    n_terms : int
        Number of series terms.

    Returns
    -------
    J : float
        Torsion constant [m^4].
    """
    return saint_venant_J_rectangular(b, h, n_terms)


def bredt_J(A_m: float, perimeter_integral: float) -> float:
    """
    Bredt's formula for single-cell closed thin-walled section.

    J = 4 * A_m^2 / ∮(ds/t)

    where A_m is the area enclosed by the median line,
    and ∮(ds/t) is the line integral of ds/t around the perimeter.

    Parameters
    ----------
    A_m : float
        Area enclosed by the median line [m^2].
    perimeter_integral : float
        ∮(ds/t) around the closed section [m].

    Returns
    -------
    J : float
        Saint-Venant torsion constant [m^4].
    """
    if A_m <= 0:
        raise ValueError("Enclosed area must be positive")
    if perimeter_integral <= 0:
        raise ValueError("Perimeter integral must be positive")

    return 4.0 * A_m ** 2 / perimeter_integral


def saint_venant_J_closed_box(
    B: float, H: float, t_w: float, t_f: float, n_cells: int = 1
) -> float:
    """
    Saint-Venant torsion constant for a single-cell rectangular box section
    using Bredt's formula.

    For a box of outer dimensions B×H with wall thicknesses t_w (webs) and t_f (flanges):
    - Median line dimensions: b_m = B - t_w, h_m = H - t_f
    - Enclosed area: A_m = b_m * h_m
    - Perimeter integral: ∮ds/t = 2*b_m/t_f + 2*h_m/t_w

    Parameters
    ----------
    B : float
        Outer width [m].
    H : float
        Outer height [m].
    t_w : float
        Web (vertical wall) thickness [m].
    t_f : float
        Flange (horizontal wall) thickness [m].
    n_cells : int
        Number of cells (currently only 1 supported).

    Returns
    -------
    J : float
        Torsion constant [m^4].
    """
    if n_cells != 1:
        raise NotImplementedError("Only single-cell boxes supported")

    # Median line dimensions
    b_m = B - t_w
    h_m = H - t_f

    if b_m <= 0 or h_m <= 0:
        raise ValueError("Wall thicknesses too large for given dimensions")

    A_m = b_m * h_m

    # Perimeter integral: top/bottom flanges + side webs
    perimeter_integral = 2.0 * b_m / t_f + 2.0 * h_m / t_w

    return bredt_J(A_m, perimeter_integral)


def warping_constant_Iw_open(
    segments: List[Tuple[float, float, float, float]]
) -> float:
    """
    Warping constant I_ω for an open thin-walled section.

    For open sections composed of straight segments:
    I_ω = (1/3) * Σ t_i * ∫ ω̄(s)^2 ds

    where ω̄ is the normalized sectorial coordinate.

    Parameters
    ----------
    segments : list of (x1, y1, x2, y2)
        List of segment endpoints defining the open cross-section.

    Returns
    -------
    Iw : float
        Warping constant [m^6].
    """
    # Compute centroid
    total_length = 0.0
    cx, cy = 0.0, 0.0
    for x1, y1, x2, y2 in segments:
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        cx += L * (x1 + x2) / 2
        cy += L * (y1 + y2) / 2
        total_length += L

    if total_length == 0:
        return 0.0

    cx /= total_length
    cy /= total_length

    # Compute shear center (simplified for symmetric sections)
    sc_x, sc_y = cx, cy

    # Compute sectorial coordinates
    omega_values = []
    s_cumulative = [0.0]

    for x1, y1, x2, y2 in segments:
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if L == 0:
            continue
        r = abs((x2 - x1) * (y1 - sc_y) - (y2 - y1) * (x1 - sc_x)) / L
        omega_values.append((r, L))
        s_cumulative.append(s_cumulative[-1] + L)

    Iw = 0.0
    for (r, L), s_start, s_end in zip(omega_values, s_cumulative[:-1], s_cumulative[1:]):
        omega_start = r * s_start
        omega_end = r * s_end
        Iw += (omega_start ** 2 + omega_start * omega_end + omega_end ** 2) * L / 3.0

    return Iw


def warping_constant_Iw_box(B: float, H: float, t_w: float, t_f: float) -> float:
    """
    Warping constant I_ω for a single-cell rectangular box section.

    Parameters
    ----------
    B : float
        Outer width [m].
    H : float
        Outer height [m].
    t_w : float
        Web thickness [m].
    t_f : float
        Flange thickness [m].

    Returns
    -------
    Iw : float
        Warping constant [m^6].
    """
    b_m = B - t_w
    h_m = H - t_f
    A_m = b_m * h_m

    perimeter_integral = 2.0 * b_m / t_f + 2.0 * h_m / t_w
    J = bredt_J(A_m, perimeter_integral)

    # For a rectangular box, a reasonable approximation:
    Iw = J * (b_m ** 2 + h_m ** 2) / 12.0

    return Iw


def characteristic_length(E: float, Iw: float, G: float, J: float) -> float:
    """
    Characteristic length for Vlasov constrained torsion.

    λ = √(E*I_ω / (G*J))

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    Iw : float
        Warping constant [m^6].
    G : float
        Shear modulus [Pa].
    J : float
        Saint-Venant torsion constant [m^4].

    Returns
    -------
    lambda_ : float
        Characteristic length [m].
    """
    if G <= 0 or J <= 0:
        raise ValueError("G and J must be positive")
    if E <= 0 or Iw < 0:
        raise ValueError("E must be positive, Iw must be non-negative")

    return np.sqrt(E * Iw / (G * J))
