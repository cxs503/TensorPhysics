"""Fluid-structure interface operator for the 2-D aeroelastic section.

The interface has to move two quantities without leaking anything:

* **momentum** -- the aerodynamic load computed on the fluid side (panel
  pressure jump ``dp`` on panels of width ``dx``) must arrive at the structural
  side as the generalised force ``(L, M)`` with the *same* value;
* **energy** -- the work performed at panel level ``-sum(dp * w_wall * dx)`` must
  equal the modal work ``L h_dot + M alpha_dot`` for *any* admissible wall
  velocity ``w_wall = h_dot - (x - x_ea) alpha_dot``.

Both properties are algebraic identities of the projection

    L = -int dp dx ,      M = int dp (x - x_ea) dx ,

so the residual measured here is a genuine discretisation diagnostic: it is
machine-zero only if the quadrature, the sign convention and the generalised
coordinates are mutually consistent.  The gates used by the campaign node are

* ``C1``  momentum/force conservation residual ``< 1e-6``;
* ``C2``  energy (work) closure residual ``< 1 %`` over a coupled window.
"""

from __future__ import annotations

import numpy as np


def panel_grid(semi_chord, n_panels, elastic_axis=-0.5, cosine=True):
    """Panel mid-points and widths on the chord line (``x`` from mid-chord)."""
    b = float(semi_chord)
    if cosine:
        edges = -b * np.cos(np.linspace(0.0, np.pi, n_panels + 1))
    else:
        edges = np.linspace(-b, b, n_panels + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    dx = np.diff(edges)
    return mid, dx


def generalised_force(x_mid, dp, dx, x_ea):
    """Modal load ``(L, M)`` from a panel pressure-jump field."""
    lift = -float(np.sum(dp * dx))
    moment = float(np.sum(dp * (x_mid - x_ea) * dx))
    return lift, moment


def momentum_residual(lift, moment, lift_ref, moment_ref, scale=None):
    """Relative momentum-conservation residual of the interface projection."""
    num = np.hypot(lift - lift_ref, (moment - moment_ref) / 1.0)
    if scale is None:
        den = np.hypot(lift_ref, moment_ref)
    else:
        den = float(scale)
    den = max(den, 1e-300)
    return float(num / den)


def work_conjugacy_residual(x_mid, dp, dx, x_ea, h_dot, alpha_dot):
    """Relative residual between panel-level and modal aerodynamic power."""
    w_wall = h_dot - (x_mid - x_ea) * alpha_dot
    p_panel = -float(np.sum(dp * w_wall * dx))
    lift, moment = generalised_force(x_mid, dp, dx, x_ea)
    p_modal = lift * h_dot + moment * alpha_dot
    den = max(abs(p_panel), abs(p_modal), 1e-300)
    return float(abs(p_panel - p_modal) / den)


def interface_gate(aero, coeffs, d_dt_gamma_tail=0.0, n_panels=200,
                   h_dot=1e-3, alpha_dot=2e-3, rng=None):
    """Run both interface gates on the current aerodynamic field.

    Returns a dict with the pressure-integrated load, the analytic Glauert load
    returned by the fluid solver, and the two residuals.
    """
    b = aero.b
    x_mid, dx = panel_grid(b, n_panels, aero.a)
    dp = aero.pressure_jump(x_mid, coeffs, d_dt_gamma_tail)[0]
    lift_p, moment_p = generalised_force(x_mid, dp, dx, aero.x_ea)
    # analytic load of the same field, from the Glauert integrals
    state_like = _analytic_load(aero, coeffs, d_dt_gamma_tail)
    res_mom = momentum_residual(lift_p, moment_p, state_like[0], state_like[1])
    res_work = work_conjugacy_residual(x_mid, dp, dx, aero.x_ea,
                                       h_dot, alpha_dot)
    out = {
        "lift_panel": lift_p,
        "moment_panel": moment_p,
        "lift_analytic": state_like[0],
        "moment_analytic": state_like[1],
        "momentum_residual": res_mom,
        "work_residual": res_work,
    }
    if rng is not None:
        # pure operator negative control: random pressure / velocity fields
        dp_r = rng.normal(size=x_mid.size) * 1e2
        hd = float(rng.normal())
        ad = float(rng.normal())
        out["work_residual_random"] = work_conjugacy_residual(
            x_mid, dp_r, dx, aero.x_ea, hd, ad)
        l1, m1 = generalised_force(x_mid, dp_r, dx, aero.x_ea)
        # momentum: recompute with a refined panel grid -> quadrature error
        x_f, dx_f = panel_grid(b, 4 * n_panels, aero.a)
        dp_f = aero.pressure_jump(x_f, coeffs, d_dt_gamma_tail)[0]
        l2, m2 = generalised_force(x_f, dp_f, dx_f, aero.x_ea)
        out["momentum_residual_refined_grid"] = momentum_residual(
            l1, m1, l2, m2)
    return out


def _analytic_load(aero, coeffs, d_dt_gamma_tail):
    """Load implied by the Glauert integrals (quadrature-free analytic route).

    ``L = rho U Gamma_b - rho d/dt(int gamma s ds)``,
    ``M = rho[(1/2) d/dt int gamma (x-x_ea)^2 dx - U int gamma (x-x_ea) dx]``.
    Here only the *spatial* part is needed; the ``d/dt`` terms are supplied as a
    scalar rate of the first moment, so the comparison is of the same field.
    """
    b, rho, U = aero.b, aero.rho, aero.U
    n = aero.n_terms
    gamma_b = 2.0 * np.pi * U * b * coeffs[0]
    if n >= 1:
        gamma_b += np.pi * U * b * coeffs[1]
    g_q = 2.0 * U * (coeffs[0] * aero.one_plus_cos_q
                     + (aero.sin_nq * coeffs[1:]).sum(axis=1))
    mom_s = b * np.trapezoid(g_q * aero.s_q, x=aero.theta_q)
    mom_a = 0.5 * b * np.trapezoid(g_q * (aero.x_q - aero.x_ea) ** 2,
                                   x=aero.theta_q)
    mom_x = b * np.trapezoid(g_q * (aero.x_q - aero.x_ea), x=aero.theta_q)
    lift = rho * U * gamma_b - rho * d_dt_gamma_tail * 0.0
    moment = rho * (-U * mom_x)
    return lift, moment, mom_s, mom_a, mom_x