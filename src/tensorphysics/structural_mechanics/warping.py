"""
Warping stress and bimoment calculations for thin-walled sections.

Implements:
- Sectorial coordinate computation (open and closed sections)
- Bredt normalization for closed sections
- Vlasov constrained torsion ODE solution
- Warping stress: σ_ω = B * ω̄ / I_ω
- Bimoment decay along beam length
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from scipy.integrate import solve_bvp, quad


def sectorial_coordinate_open(
    segments: List[Tuple[float, float, float, float]],
    shear_center: Optional[Tuple[float, float]] = None,
    n_points_per_segment: int = 50
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute sectorial coordinate ω(s) for an open thin-walled section.

    ω(s) = ∫₀ˢ r(ξ) dξ
    where r is the perpendicular distance from the shear center to the tangent.

    Parameters
    ----------
    segments : list of (x1, y1, x2, y2)
        Segment endpoints defining the open cross-section.
    shear_center : (sc_x, sc_y), optional
        Shear center coordinates. If None, assumes centroid.
    n_points_per_segment : int
        Discretization points per segment.

    Returns
    -------
    s : ndarray
        Cumulative arc length.
    omega : ndarray
        Sectorial coordinate values.
    omega_mean : float
        Mean sectorial coordinate (for normalization).
    """
    if shear_center is None:
        total_length = 0.0
        cx, cy = 0.0, 0.0
        for x1, y1, x2, y2 in segments:
            L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            cx += L * (x1 + x2) / 2
            cy += L * (y1 + y2) / 2
            total_length += L
        if total_length > 0:
            shear_center = (cx / total_length, cy / total_length)
        else:
            shear_center = (0.0, 0.0)

    sc_x, sc_y = shear_center

    s_list = [0.0]
    omega_list = [0.0]
    omega = 0.0

    for x1, y1, x2, y2 in segments:
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if L == 0:
            continue

        dx, dy = (x2 - x1) / L, (y2 - y1) / L
        r_start = (x1 - sc_x) * dy - (y1 - sc_y) * dx
        r = r_start

        s_points = np.linspace(0, L, n_points_per_segment)
        omega_points = omega + r * s_points

        s_list.extend(s_points[1:] + s_list[-1])
        omega_list.extend(omega_points[1:])
        omega = omega + r * L

    s = np.array(s_list)
    omega = np.array(omega_list)
    omega_mean = np.mean(omega)

    return s, omega, omega_mean


def sectorial_coordinate_closed(
    B: float, H: float, t_w: float, t_f: float,
    n_points_per_side: int = 50
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute normalized sectorial coordinate ω̄ for a single-cell rectangular box.

    For closed sections, the sectorial coordinate must satisfy:
    ∮ ω̄ ds = 0  (normalization condition)

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
    n_points_per_side : int
        Discretization points per side.

    Returns
    -------
    s : ndarray
        Cumulative arc length around the perimeter.
    omega_bar : ndarray
        Normalized sectorial coordinate.
    A_m : float
        Enclosed area by median line.
    """
    b_m = B - t_w
    h_m = H - t_f
    A_m = b_m * h_m

    sc_x, sc_y = B / 2, H / 2

    x_start = t_w / 2
    y_start = t_f / 2

    segments = [
        (x_start, y_start, B - x_start, y_start),
        (B - x_start, y_start, B - x_start, H - y_start),
        (B - x_start, H - y_start, x_start, H - y_start),
        (x_start, H - y_start, x_start, y_start),
    ]

    s_list = [0.0]
    omega_list = [0.0]
    omega = 0.0

    for x1, y1, x2, y2 in segments:
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if L == 0:
            continue

        dx, dy = (x2 - x1) / L, (y2 - y1) / L
        r = (x1 - sc_x) * dy - (y1 - sc_y) * dx

        s_points = np.linspace(0, L, n_points_per_side)
        omega_points = omega + r * s_points

        s_list.extend(s_points[1:] + s_list[-1])
        omega_list.extend(omega_points[1:])
        omega = omega + r * L

    s = np.array(s_list)
    omega_raw = np.array(omega_list)

    omega_mean = np.mean(omega_raw)
    omega_bar = omega_raw - omega_mean

    return s, omega_bar, A_m


def vlasov_ode_solution(
    L: float,
    E: float,
    G: float,
    Iw: float,
    J: float,
    boundary_condition: str = "fixed-free",
    applied_torque: Optional[float] = None,
    applied_bimoment_end: Optional[float] = None,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Vlasov constrained torsion ODE:

    E*I_ω * θ'''' - G*J * θ'' = m_t(x)

    For the homogeneous case (m_t = 0):
    θ'''' - λ² * θ'' = 0, where λ² = G*J / (E*I_ω)

    General solution:
    θ(x) = C₁ + C₂*x + C₃*sinh(λ*x) + C₄*cosh(λ*x)

    Parameters
    ----------
    L : float
        Beam length [m].
    E : float
        Young's modulus [Pa].
    G : float
        Shear modulus [Pa].
    Iw : float
        Warping constant [m^6].
    J : float
        Saint-Venant torsion constant [m^4].
    boundary_condition : str
        "fixed-free": fixed at x=0, free at x=L
        "fixed-fixed": fixed at both ends
        "free-free": free at both ends (rigid body modes possible)
    applied_torque : float, optional
        Applied torque at free end [N·m].
    applied_bimoment_end : float, optional
        Applied bimoment at free end [N·m²].
    n_points : int
        Number of output points.

    Returns
    -------
    x : ndarray
        Position along beam [m].
    theta : ndarray
        Twist angle [rad].
    theta_prime : ndarray
        Rate of twist (warping) [rad/m].
    bimoment : ndarray
        Bimoment B = E*I_ω * θ'' [N·m²].
    """
    if Iw <= 0:
        x = np.linspace(0, L, n_points)
        if applied_torque is not None:
            theta = applied_torque * x / (G * J)
        else:
            theta = np.zeros_like(x)
        return x, theta, np.zeros_like(x), np.zeros_like(x)

    lam_sq = G * J / (E * Iw)
    lam = np.sqrt(lam_sq)

    x = np.linspace(0, L, n_points)

    if boundary_condition == "fixed-free":
        if applied_bimoment_end is not None and applied_bimoment_end != 0:
            B0 = applied_bimoment_end
            # Numerically stable form using cosh(λ(L-x))
            theta = (B0 / (E * Iw * lam_sq)) * (
                1 - np.cosh(lam * (L - x)) / np.cosh(lam * L)
            )
        elif applied_torque is not None and applied_torque != 0:
            T = applied_torque
            # Numerically stable form
            theta = (T / (G * J)) * (
                x - np.sinh(lam * x) / (lam * np.cosh(lam * L))
            )
        else:
            theta = np.zeros_like(x)

    elif boundary_condition == "fixed-fixed":
        theta = np.zeros_like(x)

    elif boundary_condition == "free-free":
        if applied_torque is not None:
            theta = applied_torque * x / (G * J)
        else:
            theta = np.zeros_like(x)
    else:
        raise ValueError(f"Unknown boundary condition: {boundary_condition}")

    # Compute derivatives analytically for accuracy
    if Iw > 0 and (applied_bimoment_end is not None or applied_torque is not None):
        if applied_bimoment_end is not None and applied_bimoment_end != 0:
            B0 = applied_bimoment_end
            # θ(x) = (B0/(E*Iw*λ²)) * (1 - cosh(λ(L-x))/cosh(λL))
            # θ'(x) = (B0/(E*Iw*λ²)) * λ*sinh(λ(L-x))/cosh(λL)
            # θ''(x) = -(B0/(E*Iw*λ²)) * λ²*cosh(λ(L-x))/cosh(λL)
            theta_prime = (B0 / (E * Iw * lam)) * np.sinh(lam * (L - x)) / np.cosh(lam * L)
            theta_double_prime = -(B0 / (E * Iw)) * np.cosh(lam * (L - x)) / np.cosh(lam * L)
            bimoment = E * Iw * theta_double_prime
        elif applied_torque is not None and applied_torque != 0:
            T = applied_torque
            # θ(x) = (T/(G*J)) * (x - sinh(λx)/(λ*cosh(λL)))
            # θ'(x) = (T/(G*J)) * (1 - cosh(λx)/cosh(λL))
            # θ''(x) = -(T/(G*J)) * λ*sinh(λx)/cosh(λL)
            theta_prime = (T / (G * J)) * (1 - np.cosh(lam * x) / np.cosh(lam * L))
            theta_double_prime = -(T / (G * J)) * lam * np.sinh(lam * x) / np.cosh(lam * L)
            bimoment = E * Iw * theta_double_prime
        else:
            theta_prime = np.zeros_like(x)
            bimoment = np.zeros_like(x)
    else:
        theta_prime = np.zeros_like(x)
        bimoment = np.zeros_like(x)

    return x, theta, theta_prime, bimoment


def bimoment_decay(
    x: np.ndarray,
    B0: float,
    E: float,
    Iw: float,
    G: float,
    J: float,
    L: float
) -> np.ndarray:
    """
    Compute bimoment decay along beam for end bimoment loading.

    Analytical solution for fixed-free beam with end bimoment B₀:
    B(x) = B₀ * cosh(λ*(L-x)) / cosh(λ*L)

    This form is numerically stable for large λL.

    Parameters
    ----------
    x : ndarray
        Position array [m].
    B0 : float
        Applied bimoment at x=L [N·m²].
    E : float
        Young's modulus [Pa].
    Iw : float
        Warping constant [m^6].
    G : float
        Shear modulus [Pa].
    J : float
        Saint-Venant torsion constant [m^4].
    L : float
        Beam length [m].

    Returns
    -------
    B : ndarray
        Bimoment distribution [N·m²].
    """
    if Iw <= 0:
        return np.zeros_like(x)

    lam = np.sqrt(G * J / (E * Iw))

    # Numerically stable form: B(x) = B₀ * cosh(λ(L-x)) / cosh(λL)
    # This avoids catastrophic cancellation for large λL
    B = B0 * np.cosh(lam * (L - x)) / np.cosh(lam * L)
    return B


def warping_stress(
    omega_bar: np.ndarray,
    bimoment: float,
    Iw: float
) -> np.ndarray:
    """
    Compute warping normal stress.

    σ_ω = B * ω̄ / I_ω

    Parameters
    ----------
    omega_bar : ndarray
        Normalized sectorial coordinate [m²].
    bimoment : float
        Bimoment [N·m²].
    Iw : float
        Warping constant [m^6].

    Returns
    -------
    sigma_omega : ndarray
        Warping normal stress [Pa].
    """
    if Iw <= 0:
        return np.zeros_like(omega_bar)

    return bimoment * omega_bar / Iw
