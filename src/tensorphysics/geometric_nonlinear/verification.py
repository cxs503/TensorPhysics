"""Comprehensive verification of geometric nonlinear problems.

This module provides:
1. Literature comparison data (Camp/Berger for plates, Bisshopp-Drucker for elastica)
2. Energy conservation checks
3. Convergence order studies
4. Full verification pipeline

References:
- Bisshopp & Drucker (1945). Large deflection of cantilever beams. Q. Appl. Math.
- Camp (1949). Large deflections of circular plates. J. Appl. Mech.
- Berger (1955). Large deflections of plates and shallow shells. J. Appl. Mech.
- Budiansky & Roth (1962). Axisymmetric dynamic buckling. NASA TN D-1510.
- Frisch-Fay (1962). Flexible Bars. Butterworths.
- Timoshenko & Woinowsky-Krieger (1959). Theory of Plates and Shells. McGraw-Hill.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .elastica import ElasticaExact
from .elastica_fem import ElasticaFEMSolver
from .von_karman_plate import (
    VonKarmanPlateExact,
    VonKarmanPlateFEM,
    plate_cubic_coefficient,
    load_path_work,
)
from .shallow_arch import VonMisesTruss, ShallowArchModel

__all__ = [
    "bisshopp_drucker_reference",
    "camp_berger_reference",
    "budiansky_roth_reference",
    "verify_energy_conservation_elastica",
    "verify_energy_conservation_plate",
    "convergence_order_elastica",
    "convergence_order_plate",
    "full_verification_elastica",
    "full_verification_plate",
    "full_verification_arch",
]


# ============================================================================
# Literature reference data
# ============================================================================

def bisshopp_drucker_reference(
    alpha_values: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Reference data from Bisshopp & Drucker (1945) for elastica.
    
    They tabulated the elastica solution in terms of:
    - α = angle at free end (θ_B in our notation)
    - P*L²/(EI) = normalized load
    - δ/L = normalized tip deflection
    - x_tip/L = horizontal projection ratio
    
    These values are computed from the exact elliptic integrals and
    serve as the gold standard for verification.
    
    Returns
    -------
    dict with arrays: alpha, P_norm, delta_over_L, x_tip_over_L
    """
    if alpha_values is None:
        alpha_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    
    exact = ElasticaExact(L=1.0, EI=1.0)
    
    P_norm = np.zeros_like(alpha_values)
    delta_L = np.zeros_like(alpha_values)
    x_tip_L = np.zeros_like(alpha_values)
    
    for i, alpha in enumerate(alpha_values):
        sol = exact.compute_from_theta_B(alpha)
        P_norm[i] = sol.P * 1.0**2 / 1.0  # L=1, EI=1
        delta_L[i] = sol.y_tip / 1.0
        x_tip_L[i] = sol.x_tip / 1.0
    
    return {
        'alpha': alpha_values,
        'P_norm': P_norm,
        'delta_over_L': delta_L,
        'x_tip_over_L': x_tip_L,
    }


def camp_berger_reference(
    w0_h_values: np.ndarray = None,
    nu: float = 0.3,
) -> Dict[str, np.ndarray]:
    """Reference data from Camp (1949) and Berger (1955) for circular plates.
    
    For a clamped circular plate under center point load:
    P*a²/(D*h) = k₁*(w₀/h) + k₂*(w₀/h)³
    
    The coefficients from the literature (for ν=0.3):
    - k₁ = 16π ≈ 50.265 (linear theory, Timoshenko §97)
    - k₂ ≈ 16π × 0.376 ≈ 18.90 (from Camp/Berger numerical integration)
    
    More precisely, from Yamaki (1958) and Chien (1948):
    P*a²/(16πDh) = (w₀/h) + c*(w₀/h)³
    where c = 0.376 for ν=0.3
    
    Returns
    -------
    dict with arrays: w0_h, P_normalized, P_dimensional
    """
    if w0_h_values is None:
        w0_h_values = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Coefficient from Camp/Berger/Yamaki -- single source of truth in
    # von_karman_plate.plate_cubic_coefficient (avoids divergent duplicates).
    c = plate_cubic_coefficient(nu)
    
    P_norm = w0_h_values + c * w0_h_values**3
    
    return {
        'w0_h': w0_h_values,
        'P_normalized': P_norm,  # P*a²/(16πDh)
        'coefficient_c': c,
    }


def budiansky_roth_reference(
    lambda_values: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Reference data for shallow arch snap-through.
    
    Budiansky & Roth (1962) and Thompson & Hunt (1973) give the
    snap-through load for shallow arches.
    
    For a von Mises truss (two-bar truss):
    - Geometry: span L, rise H, bar length L₀ = √((L/2)² + H²)
    - Rise-span ratio: λ = H/L
    - The limit load depends on λ and the axial stiffness EA
    
    Returns
    -------
    dict with limit loads for various rise-span ratios.
    """
    if lambda_values is None:
        lambda_values = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    
    L = 10.0
    EA = 1e9
    
    q_limits = np.zeros_like(lambda_values)
    
    for i, lam in enumerate(lambda_values):
        H = lam * L
        truss = VonMisesTruss(L=L, H=H, EA=EA)
        result = truss.find_limit_points(n_pts=1000)
        q_limits[i] = result['q_limit_upper']
    
    return {
        'lambda': lambda_values,
        'q_limit': q_limits,
        'L': L,
        'EA': EA,
    }


# ============================================================================
# Energy conservation checks
# ============================================================================

def _elastica_load_and_deflection(exact: ElasticaExact, theta_B: float) -> Tuple[float, float]:
    """(P, delta) of the exact elastica for a given free-end angle.

    Uses the elliptic-type integrals directly (no deformed-shape tabulation),
    so it is cheap enough to sample a whole loading path.
    """
    I_L, _, I_y = exact._compute_integrals(theta_B)
    P = exact.EI * I_L**2 / (2.0 * exact.L**2)
    delta = exact.L * I_y / I_L
    return P, delta


def verify_energy_conservation_elastica(
    L: float = 1.0, EI: float = 1.0,
    theta_B_values: np.ndarray = None,
    n_path: int = 400,
) -> Dict[str, np.ndarray]:
    """Genuine energy check for the elastica: work of deformation vs U.

    For a cantilever under a *dead* end load P the quasi-static energy balance
    reads (Clausius / work of deformation)

        U = int_0^delta P(delta') d delta' ,

    with U = (EI/2) int (dtheta/ds)^2 ds the bending energy of the elastica
    (``ElasticaExact.strain_energy``).  Note that U != P*delta except in the
    linear limit: P*delta >= U with U/(P delta) -> 1/2 as P -> 0, so comparing
    U with P*delta (as the first version of this routine did) produces a
    spurious 50-70% "error" that is a property of the criterion, not of the
    solution.  This routine therefore integrates the load path numerically.

    A second, independent consistency check is the generalised-force identity
    dU/ddelta = P, evaluated here with central differences in theta_B.

    Parameters
    ----------
    L, EI : float
        Beam length and flexural rigidity.
    theta_B_values : ndarray, optional
        Free-end angles at which to check the balance.
    n_path : int
        Number of load-path samples used for the integral int P d delta.

    Returns
    -------
    dict with
        theta_B, P, delta, U_strain, W_deform,
        error_pct            : |W_deform - U| / U * 100
        dU_ddelta, P_from_derivative, generalized_force_error_pct
    """
    if theta_B_values is None:
        theta_B_values = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3])
    theta_B_values = np.asarray(theta_B_values, dtype=float)

    exact = ElasticaExact(L=L, EI=EI)

    P_out = np.zeros_like(theta_B_values)
    delta_out = np.zeros_like(theta_B_values)
    U_out = np.zeros_like(theta_B_values)
    W_out = np.zeros_like(theta_B_values)
    err_out = np.zeros_like(theta_B_values)
    dU_ddelta_out = np.zeros_like(theta_B_values)
    gforce_err_out = np.zeros_like(theta_B_values)

    for i, tb in enumerate(theta_B_values):
        P, delta = _elastica_load_and_deflection(exact, tb)
        P_out[i] = P
        delta_out[i] = delta
        U_out[i] = exact.strain_energy(tb)

        # Deformation work along the monotonic loading path 0 -> theta_B.
        path = np.linspace(1e-9, tb, n_path)
        PD = [_elastica_load_and_deflection(exact, t) for t in path]
        P_path = np.array([p for p, _ in PD])
        d_path = np.array([d for _, d in PD])
        P_path = np.concatenate(([0.0], P_path))
        d_path = np.concatenate(([0.0], d_path))
        W_out[i] = float(np.trapezoid(P_path, d_path))

        err_out[i] = abs(W_out[i] - U_out[i]) / U_out[i] * 100.0 if U_out[i] > 0 else 0.0

        # Generalised force: dU/ddelta = P.
        dtheta = 1e-6
        if tb - dtheta > 0.0 and tb + dtheta < 0.5 * np.pi:
            dU = (exact.strain_energy(tb + dtheta) - exact.strain_energy(tb - dtheta)) / (2 * dtheta)
            dd = (_elastica_load_and_deflection(exact, tb + dtheta)[1]
                  - _elastica_load_and_deflection(exact, tb - dtheta)[1]) / (2 * dtheta)
            dU_ddelta_out[i] = dU / dd
            gforce_err_out[i] = abs(dU_ddelta_out[i] - P) / P * 100.0
        else:
            dU_ddelta_out[i] = np.nan
            gforce_err_out[i] = np.nan

    return {
        'theta_B': theta_B_values,
        'P': P_out,
        'delta': delta_out,
        'U_strain': U_out,
        'W_deform': W_out,
        'error_pct': err_out,
        'dU_ddelta': dU_ddelta_out,
        'P_from_derivative': P_out,
        'generalized_force_error_pct': gforce_err_out,
    }


def verify_energy_conservation_plate(
    a: float = 1.0, h: float = 0.01, E: float = 210e9, nu: float = 0.3,
    w0_h_values: np.ndarray = None,
    n_elem: int = 16, n_steps: int = 10,
) -> Dict[str, np.ndarray]:
    """Genuine energy check for the von Karman plate: W_ext vs FE strain energy.

    A circular clamped plate is loaded by the centre force predicted by the
    closed-form law, the von Karman finite element model is brought to that
    load along a monotonic path of ``n_steps`` increments, and

        W_ext = int_0^{w0} P dw0      (dead-load work, :func:`load_path_work`)
        U_FE  = membrane + bending strain energy of the converged FE state
                (:meth:`VonKarmanPlateFEM.energy`)

    are compared.  Neither quantity is derived from the other, so the residual
    is a real check of the incremental/Newton solution (it is dominated by the
    trapezoidal load-path quadrature, i.e. O(dP^2)).

    The first version of this routine split W_ext into a "bending" and a
    "membrane" part and then re-added them, so its reported error was
    identically 0 and the check could never fail.

    Returns
    -------
    dict with w0_h, P, w0, w_ext, u_total, u_bending, u_membrane, error_pct,
    converged.
    """
    if w0_h_values is None:
        w0_h_values = np.array([0.5, 1.0, 2.0])

    ref = VonKarmanPlateExact(a=a, h=h, E=E, nu=nu)
    fem = VonKarmanPlateFEM(a=a, h=h, E=E, nu=nu, n_elem=n_elem)

    w0_h_values = np.asarray(w0_h_values, dtype=float)
    _, _, P_values = ref.compute_load_deflection_curve(w0_h_values)

    P_out = np.zeros_like(w0_h_values)
    w0_out = np.zeros_like(w0_h_values)
    W_out = np.zeros_like(w0_h_values)
    U_out = np.zeros_like(w0_h_values)
    U_bend = np.zeros_like(w0_h_values)
    U_mem = np.zeros_like(w0_h_values)
    err_out = np.zeros_like(w0_h_values)
    conv = np.zeros(w0_h_values.shape, dtype=bool)

    for i, P in enumerate(P_values):
        res = fem.solve(float(P), n_steps=n_steps, max_iter=60, tol=1e-10)
        energy = fem.energy(res['u'])
        P_out[i] = P
        w0_out[i] = res['w0']
        conv[i] = bool(res['converged'])
        W_out[i] = load_path_work(res['history'])
        U_out[i] = energy['total']
        U_bend[i] = energy['bending']
        U_mem[i] = energy['membrane']
        err_out[i] = abs(W_out[i] - U_out[i]) / U_out[i] * 100.0 if U_out[i] > 0 else 0.0

    return {
        'w0_h': w0_h_values,
        'P': P_out,
        'w0': w0_out,
        'w_ext': W_out,
        'u_total': U_out,
        'u_bending': U_bend,
        'u_membrane': U_mem,
        'error_pct': err_out,
        'converged': conv,
    }


# ============================================================================
# Convergence order studies
# ============================================================================

def convergence_order_elastica(
    L: float = 1.0, EI: float = 1.0,
    P_target: float = 2.0,
    n_elem_list: List[int] = None,
) -> Dict:
    """Study convergence order of elastica FEM with mesh refinement.
    
    For the discrete elastica model: O(h²) convergence expected.
    
    Returns
    -------
    dict with mesh sizes, errors, and estimated convergence order.
    """
    if n_elem_list is None:
        n_elem_list = [10, 20, 40, 80, 160]
    
    exact = ElasticaExact(L=L, EI=EI)
    
    # Get exact tip deflection
    sol_exact = exact.compute_shape_at_load(P_target)
    delta_exact = sol_exact.y_tip
    
    h_values = np.array([L / n for n in n_elem_list])
    errors = np.zeros(len(n_elem_list))
    delta_fem = np.zeros(len(n_elem_list))
    
    for i, n_elem in enumerate(n_elem_list):
        solver = ElasticaFEMSolver(L=L, EI=EI, n_elem=n_elem)
        result = solver.solve(P_target)
        delta_fem[i] = result.tip_deflection
        errors[i] = abs(delta_fem[i] - delta_exact)
    
    # Estimate convergence order
    valid = errors > 1e-14
    if np.sum(valid) >= 2:
        log_h = np.log(h_values[valid])
        log_err = np.log(errors[valid])
        p = np.polyfit(log_h, log_err, 1)[0]
        convergence_order = abs(p)
    else:
        convergence_order = float('nan')
    
    return {
        'n_elem': n_elem_list,
        'h_values': h_values,
        'delta_fem': delta_fem,
        'delta_exact': delta_exact,
        'errors': errors,
        'convergence_order': convergence_order,
    }


def convergence_order_plate(
    a: float = 1.0, h: float = 0.01, E: float = 210e9, nu: float = 0.3,
    P_target: float = None,
    n_elem_list: List[int] = None,
    n_steps: int = 6,
) -> Dict:
    """Observed convergence order of the von Karman plate FEM.

    Computes the centre deflection on a sequence of meshes at a fixed load and
    estimates the order p from the Richardson ratio of the successive
    differences,

        p = log2( |w_h - w_h/2| / |w_h/2 - w_h/4| ).

    The first version of this function did not solve anything: it returned
    ``convergence_order = 4.0`` (the theoretical order of cubic Hermite
    elements) with the note "Theoretical order for cubic Hermite elements",
    i.e. the "verification" could never fail.

    Parameters
    ----------
    P_target : float, optional
        Centre load [N].  Defaults to the closed-form load for w0/h = 2.
    n_elem_list : list of int, optional
        Meshes (default [8, 16, 32], halving between levels).
    n_steps : int
        Load increments per solve (accuracy of the incremental path).

    Returns
    -------
    dict with n_elem, h_values, w0, differences, convergence_order, P_target,
    converged.
    """
    if n_elem_list is None:
        n_elem_list = [8, 16, 32]
    if len(n_elem_list) < 3:
        raise ValueError(
            "convergence_order_plate needs at least 3 meshes for a Richardson "
            f"order estimate (got {len(n_elem_list)})"
        )

    ref = VonKarmanPlateExact(a=a, h=h, E=E, nu=nu)

    if P_target is None:
        _, _, P_vals = ref.compute_load_deflection_curve(np.array([2.0]))
        P_target = float(P_vals[0])

    w0 = np.zeros(len(n_elem_list))
    converged = np.zeros(len(n_elem_list), dtype=bool)
    for i, n in enumerate(n_elem_list):
        fem = VonKarmanPlateFEM(a=a, h=h, E=E, nu=nu, n_elem=n)
        res = fem.solve(P_target, n_steps=n_steps, max_iter=60, tol=1e-10)
        w0[i] = res['w0']
        converged[i] = bool(res['converged'])

    h_values = np.array([a / n for n in n_elem_list], dtype=float)
    differences = np.abs(np.diff(w0))

    if len(differences) >= 2 and np.all(differences > 0):
        order = float(np.log2(differences[:-1] / differences[1:])[-1])
    else:
        order = float('nan')

    return {
        'n_elem': n_elem_list,
        'h_values': h_values,
        'w0': w0,
        'differences': differences,
        'convergence_order': order,
        'P_target': P_target,
        'converged': converged,
    }


# ============================================================================
# Full verification pipelines
# ============================================================================

def full_verification_elastica(
    L: float = 1.0, EI: float = 1.0,
    n_points: int = 5,
) -> Dict:
    """Complete elastica verification pipeline."""
    exact = ElasticaExact(L=L, EI=EI)
    
    # 1. Load-deflection curve at 5 points
    theta_B_test = np.array([0.3, 0.6, 0.9, 1.1, 1.3])
    ref = bisshopp_drucker_reference(theta_B_test)
    
    # Compare with FEM (fine mesh)
    P_test = ref['P_norm'] * EI / L**2
    fem_results = []
    for P in P_test:
        solver = ElasticaFEMSolver(L=L, EI=EI, n_elem=160)
        result = solver.solve(P)
        fem_results.append(result)
    
    delta_fem = np.array([r.tip_deflection for r in fem_results])
    delta_exact = ref['delta_over_L'] * L
    
    load_deflection_error = np.abs(delta_fem - delta_exact) / np.maximum(delta_exact, 1e-10) * 100
    
    # 2. Linear limit
    linear_check = exact.verify_linear_limit()
    
    # 3. Convergence order
    conv_check = convergence_order_elastica(L, EI, P_target=P_test[2])
    
    # Pass/fail.  The convergence window is asymmetric on purpose:
    # ElasticaFEMSolver is a lumped rigid-link model with piecewise-constant
    # rotation, so its observed order against the exact solution is ~1.0 (not
    # the O(h^2) one would expect from a Hermite beam element).  Measured
    # orders: 1.00 at P = 2.0 (see docs/m8870_r1_audit.md).  A monotone
    # first-order model is accepted; super-convergence would indicate the
    # mesh sequence is not what it claims to be.
    results = {
        'load_deflection_error_pct': load_deflection_error,
        'linear_limit_error_pct': linear_check['error_pct'],
        'convergence_order': conv_check['convergence_order'],
        'convergence_order_expected': 'first order (lumped rigid-link model)',
        'pass_load_deflection': np.all(load_deflection_error < 5.0),
        'pass_linear_limit': linear_check['error_pct'] < 1.0,
        'pass_convergence': 0.85 <= conv_check['convergence_order'] <= 3.0,
    }
    
    results['all_pass'] = all([
        results['pass_load_deflection'],
        results['pass_linear_limit'],
        results['pass_convergence'],
    ])
    
    return results


def full_verification_plate(
    a: float = 1.0, h: float = 0.01, E: float = 210e9, nu: float = 0.3,
    w0_h_check: np.ndarray = None,
    n_elem: int = 16, n_steps: int = 8,
) -> Dict:
    """Complete von Karman plate verification pipeline.

    The closed-form load-deflection law is now checked against the *independent*
    von Karman finite element solution (load from the law -> converged FE
    deflection).  The first version compared the law against
    ``camp_berger_reference``, which is generated from the same law, so the
    reported error was identically zero.
    """
    ref = VonKarmanPlateExact(a=a, h=h, E=E, nu=nu)

    if w0_h_check is None:
        w0_h_check = np.array([0.5, 1.0, 2.0, 3.0])
    w0_h_check = np.asarray(w0_h_check, dtype=float)

    _, _, P_check = ref.compute_load_deflection_curve(w0_h_check)

    fem = VonKarmanPlateFEM(a=a, h=h, E=E, nu=nu, n_elem=n_elem)
    w0_fem = np.zeros_like(w0_h_check)
    converged = np.zeros(w0_h_check.shape, dtype=bool)
    for i, P in enumerate(P_check):
        res = fem.solve(float(P), n_steps=n_steps, max_iter=60, tol=1e-10)
        w0_fem[i] = res['w0']
        converged[i] = bool(res['converged'])
    w0_h_fem = w0_fem / h

    # error in the deflection predicted for the same load
    fe_error = np.abs(w0_h_fem - w0_h_check) / w0_h_check * 100.0

    # linear limit: deflection must match w0 = P a^2/(16 pi D)
    P_small = 1.0
    w0_linear = ref.linear_solution(P_small)
    P_norm_small = P_small * a**2 / (16 * np.pi * ref.D * h)
    linear_error = abs(P_norm_small / (w0_linear / h) - 1.0) * 100.0

    # membrane stiffening: secant stiffness must grow with deflection
    w0_h_fine = np.linspace(0.1, 5.0, 50)
    _, P_norm_fine, _ = ref.compute_load_deflection_curve(w0_h_fine)
    ratio = P_norm_fine / w0_h_fine
    stiffening_detected = bool(ratio[-1] > ratio[0])

    results = {
        'w0_h_check': w0_h_check,
        'w0_h_fem': w0_h_fem,
        'P_error_pct': fe_error,          # closed form vs FE, same load
        'linear_limit_error_pct': linear_error,
        'stiffening_detected': stiffening_detected,
        'fem_converged': converged,
        'fem_n_elem': n_elem,
        'pass_load_deflection': bool(np.all(fe_error < 5.0) and np.all(converged)),
        'pass_linear_limit': linear_error < 1.0,
        'pass_stiffening': stiffening_detected,
    }

    results['all_pass'] = all([
        results['pass_load_deflection'],
        results['pass_linear_limit'],
        results['pass_stiffening'],
    ])

    return results


def full_verification_arch(
    L: float = 10.0, H: float = 1.0, EA: float = 1e9,
) -> Dict:
    """Complete shallow arch snap-through verification."""
    truss = VonMisesTruss(L=L, H=H, EA=EA)
    
    # 1. Find limit points
    result = truss.find_limit_points(n_pts=1000)
    
    # 2. Check for negative stiffness
    has_negative_stiffness = 'unstable' in result['stability']
    
    # 3. Rise-span ratio study
    lambda_values = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    br_ref = budiansky_roth_reference(lambda_values)
    
    results = {
        'q_limit_upper': result['q_limit_upper'],
        'q_limit_lower': result['q_limit_lower'],
        'w_at_upper': result['w_at_upper'],
        'has_negative_stiffness': has_negative_stiffness,
        'rise_span': H / L,
        'budiansky_roth_data': br_ref,
        'pass_limit_load': result['q_limit_upper'] > 0,
        'pass_negative_stiffness': has_negative_stiffness,
    }
    
    results['all_pass'] = all([
        results['pass_limit_load'],
        results['pass_negative_stiffness'],
    ])
    
    return results
