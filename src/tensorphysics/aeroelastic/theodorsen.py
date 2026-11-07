"""Theodorsen frequency-domain reference for the 2-D typical section.

Conventions (identical to :mod:`tensorphysics.aeroelastic.uvlm`, pinned by the
first-principles checks in ``tests/test_aeroelastic_uvlm.py``):

* ``h`` plunge up, ``alpha`` pitch nose-up about the elastic axis at ``a*b``
  from mid-chord;
* ``L`` lift positive up (N/m), ``M`` moment positive nose-up (N);
* flat-plate apparent mass ``M_a = pi rho b^2 [[1, -ba], [-ba, b^2(1/8+a^2)]]``
  enters as ``F = -M_a q_ddot`` (a force opposing acceleration).

Circulatory loading uses Theodorsen's function ``C(k)`` with the effective
downwash ``D = -h_dot + U alpha + b(1/2 - a) alpha_dot``:  the *minus* sign on
``h_dot`` is the physical statement that heaving up reduces the incidence.  The
non-circulatory convective pair proportional to ``U alpha_dot`` is kept with the
sign flag ``u_alpha_sign`` which is *determined numerically* by comparison with
the time-domain solver (:func:`calibrate_convention`).
"""

from __future__ import annotations

import numpy as np

try:  # scipy is an optional dependency of the package
    from scipy.special import hankel2
except Exception:  # pragma: no cover - exercised only without scipy
    hankel2 = None


def theodorsen_c(k):
    """Theodorsen's lift-deficiency function ``C(k) = H1/H1+iH0`` (2nd kind)."""
    k = np.asarray(k, dtype=float)
    if hankel2 is None:  # pragma: no cover
        raise RuntimeError("scipy is required for the Theodorsen function")
    h0 = hankel2(0, k)
    h1 = hankel2(1, k)
    return h1 / (h1 + 1j * h0)


def apparent_mass(b, a, rho):
    return np.pi * rho * b * b * np.array([[1.0, -b * a],
                                           [-b * a, b * b * (0.125 + a * a)]])


def aero_force_harmonic(k, U, b, a, rho, h_amp, alpha_amp, u_alpha_sign=1.0):
    """Complex amplitudes of ``(L, M)`` for ``h = h0 e^{i w t}``, ``alpha = a0 e^{i w t}``.

    ``w = k U / b``.  Returns the complex generalised force vector.
    """
    w = k * U / b
    q = np.array([h_amp, alpha_amp], dtype=complex)
    qd = 1j * w * q
    qdd = -(w * w) * q
    ck = theodorsen_c(k)
    force = -apparent_mass(b, a, rho) @ qdd
    down = -qd[0] + U * q[1] + b * (0.5 - a) * qd[1]
    g = np.array([1.0, b * (a + 0.5)])
    force = force + 2.0 * np.pi * rho * U * b * ck * g * down
    force[0] += u_alpha_sign * np.pi * rho * b * b * U * qd[1]
    force[1] += -u_alpha_sign * np.pi * rho * b * b * U * b * (0.5 - a) * qd[1]
    return force


def aero_operators(k, U, b, a, rho, u_alpha_sign=1.0):
    """Matrices ``(A_extra, B_extra, K_extra)`` such that
    ``F = A q_ddot + B q_dot + K q`` for the Theodorsen model."""
    ck = theodorsen_c(k)
    am = apparent_mass(b, a, rho)
    g = np.array([1.0, b * (a + 0.5)])
    d = np.array([-1.0, b * (0.5 - a)])
    e = np.array([0.0, 1.0])
    A = -am
    B = 2.0 * np.pi * rho * U * b * ck * np.outer(g, d)
    B[0, 1] += u_alpha_sign * np.pi * rho * b * b * U
    B[1, 1] += -u_alpha_sign * np.pi * rho * b * b * U * b * (0.5 - a)
    K = 2.0 * np.pi * rho * U * U * b * ck * np.outer(g, e)
    return A, B, K


def pk_eigenvalues(section, U, rho, k_guess=None, n_iter=60, u_alpha_sign=1.0):
    """p-k eigenvalues at speed ``U`` for the Theodorsen model.

    Returns ``(p, k)``: the four complex eigenvalues with ``k = Im(p) b / U``
    consistent with the frequency of the eigenvalue itself.
    """
    if U <= 0.0:
        return np.linalg.eigvals(np.linalg.solve(section.M, -section.K)), 0.0
    b = section.b
    if k_guess is None:
        k_guess = 0.1
    k = float(k_guess)
    p = None
    for _ in range(n_iter):
        A, B, K = aero_operators(k, U, b, section.a, rho, u_alpha_sign)
        M = section.M - A
        C = section.C - B
        Kk = section.K - K
        Mi = np.linalg.inv(M)
        comp = np.zeros((4, 4), dtype=complex)
        comp[:2, 2:] = np.eye(2)
        comp[2:, :2] = -Mi @ Kk
        comp[2:, 2:] = -Mi @ C
        p = np.linalg.eigvals(comp)
        idx = int(np.argmax(p.real))
        p_max = p[idx]
        k_new = abs(p_max.imag) * b / U
        if k_new <= 0.0:
            break
        if abs(k_new - k) < 1e-10:
            k = k_new
            break
        k = 0.5 * (k + k_new) if False else k_new
    return p, k


def pk_flutter(section, rho, U_lo=1e-3, U_hi=None, uom=1.0, u_alpha_sign=1.0,
               n_scan=300, tol=1e-10, oscillatory_only=True):
    """Critical flutter speed and frequency from the p-k method.

    Scans ``U`` until the maximum real part of the p-k eigenvalues changes sign,
    then brackets it to ``tol``.  Returns a dict with ``U_f``, ``omega_f``,
    ``k_f`` and the damping trace used for the bracket.
    """
    b = section.b
    if U_hi is None:
        U_hi = 40.0 * b * max(section.omega_h, section.omega_alpha)

    w_ref = max(section.omega_h, section.omega_alpha)

    def damping(U):
        p, _ = pk_eigenvalues(section, U, rho, u_alpha_sign=u_alpha_sign)
        if oscillatory_only:
            osc = np.abs(p.imag) > 0.02 * w_ref
            if not np.any(osc):
                return -np.inf, 0.0
            p = p[osc]
        idx = int(np.argmax(p.real))
        return p[idx].real, p[idx].imag

    grid = np.geomspace(max(U_lo, 1e-6), U_hi, n_scan)
    prev = None
    lo = hi = None
    for U in grid:
        g, w = damping(U)
        if prev is not None and prev[0] < 0.0 <= g:
            lo, hi = prev[2], U
            break
        prev = (g, w, U)
    if lo is None:
        return {"U_f": None, "omega_f": None, "k_f": None,
                "converged": False, "reason": "no flutter crossing found"}

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        g, w = damping(mid)
        if g >= 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol * max(1.0, hi):
            break
    U_f = 0.5 * (lo + hi)
    p, k = pk_eigenvalues(section, U_f, rho, u_alpha_sign=u_alpha_sign)
    idx = int(np.argmax(p.real))
    omega_f = abs(p[idx].imag)
    return {"U_f": float(U_f), "omega_f": float(omega_f),
            "k_f": float(omega_f * b / U_f), "damping": float(p[idx].real),
            "converged": True}


def calibrate_convention(section, rho, U, k, h_amp=1e-4, alpha_amp=1e-4):
    """Helper used offline to pin ``u_alpha_sign`` against the time-domain solver."""
    return {
        "+1": aero_force_harmonic(k, U, section.b, section.a, rho,
                                  h_amp, alpha_amp, 1.0),
        "-1": aero_force_harmonic(k, U, section.b, section.a, rho,
                                  h_amp, alpha_amp, -1.0),
    }

def flutter_determinant_matrix(section, U, omega, rho, u_alpha_sign=1.0):
    """Complex matrix of the neutral (harmonic) Theodorsen problem."""
    b, a = section.b, section.a
    k = omega * b / U if U > 0 else 0.0
    ck = theodorsen_c(k)
    A = section.M + apparent_mass(b, a, rho)
    g = np.array([1.0, b * (a + 0.5)])
    d = np.array([-1.0, b * (0.5 - a)])
    e = np.array([0.0, 1.0])
    mat = -(omega ** 2) * A + section.K
    mat = mat - 2.0 * np.pi * rho * U * b * ck * np.outer(g, 1j * omega * d
                                                         + U * e)
    bnc = np.zeros((2, 2), dtype=complex)
    bnc[0, 1] = u_alpha_sign * np.pi * rho * b * b * U
    bnc[1, 1] = -u_alpha_sign * np.pi * rho * b * b * U * b * (0.5 - a)
    mat = mat - 1j * omega * bnc
    return mat


def flutter_determinant(section, rho, U0=None, omega0=None, u_alpha_sign=1.0):
    """Exact neutral flutter point of the Theodorsen model by root finding.

    Solves ``det(-w^2 (M_s+M_a) + K_s - aerodynamics) = 0`` for the two real
    unknowns ``(U, w)`` -- the classical Theodorsen flutter determinant, with no
    p-k iteration error.
    """
    from scipy.optimize import root
    b = section.b
    if omega0 is None:
        omega0 = float(np.sqrt(np.max(np.linalg.solve(
            section.M, section.K))))
    if U0 is None:
        U0 = 5.0 * b * omega0

    def fun(x):
        U, w = x
        w = abs(w)
        mat = flutter_determinant_matrix(section, U, w, rho, u_alpha_sign)
        det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
        scale = abs(mat[0, 0] * mat[1, 1]) + abs(mat[0, 1] * mat[1, 0])
        if scale == 0:
            scale = 1.0
        return [det.real / scale, det.imag / scale]

    sol = root(fun, [U0, omega0], method="hybr", options={"maxfev": 400})
    U_f = float(abs(sol.x[0]))
    w_f = float(abs(sol.x[1]))
    return {"U_f": U_f, "omega_f": w_f, "k_f": w_f * b / U_f,
            "residual": float(np.max(np.abs(fun([U_f, w_f])))),
            "success": bool(sol.success), "u_alpha_sign": u_alpha_sign}


def divergence_speed(section, rho, u_alpha_sign=1.0):
    """Static divergence speed: smallest U with det(K_eff(k->0)) = 0."""
    b, a = section.b, section.a
    coef = 2.0 * np.pi * rho * b * b * (a + 0.5)
    kalpha = section.K[1, 1]
    if coef <= 0:
        return None
    if kalpha / coef <= 0:
        return None
    return float(np.sqrt(kalpha / coef))


def pk_eigen_tracked(section, rho, U, p_guess=None, u_alpha_sign=1.0,
                     n_iter=80):
    """Mode-tracked p-k eigenvalues: every mode carries its own reduced frequency.

    The textbook p-k iteration shares one ``k`` between all modes, which
    mispredicts the static divergence branch (``k -> 0``) whenever an
    oscillatory mode with ``k = O(1)`` is present.  Here each of the four
    eigenvalues is iterated with ``k_i = Im(p_i) b / U`` and tracked by
    continuity in the complex plane.
    """
    b = section.b
    A = section.M + apparent_mass(b, section.a, rho)
    Mi = np.linalg.inv(A)
    if p_guess is None:
        p_guess = np.array([1j * section.omega_h, -1j * section.omega_h,
                            1j * section.omega_alpha, -1j * section.omega_alpha])
    p_cur = np.asarray(p_guess, dtype=complex).copy()
    for _ in range(n_iter):
        new = p_cur.copy()
        for i in range(p_cur.size):
            ki = abs(p_cur[i].imag) * b / U if U > 0 else 0.0
            _, Bx, Kx = aero_operators(ki, U, b, section.a, rho, u_alpha_sign)
            Cx = section.C - Bx
            Kk = section.K - Kx
            comp = np.zeros((4, 4), dtype=complex)
            comp[:2, 2:] = np.eye(2)
            comp[2:, :2] = -Mi @ Kk
            comp[2:, 2:] = -Mi @ Cx
            ev = np.linalg.eigvals(comp)
            new[i] = ev[int(np.argmin(np.abs(ev - p_cur[i])))]
        if np.max(np.abs(new - p_cur)) < 1e-12:
            p_cur = new
            break
        p_cur = new
    ks = np.abs(p_cur.imag) * b / U if U > 0 else np.zeros_like(p_cur.imag)
    return p_cur, ks


def flutter_boundary_tracked(section, rho, U_lo=None, U_hi=None, n_scan=120,
                             u_alpha_sign=1.0, tol=1e-9):
    """First instability (flutter or divergence) along ``U`` with mode tracking."""
    b = section.b
    w_ref = max(section.omega_h, section.omega_alpha)
    if U_hi is None:
        U_hi = 40.0 * b * w_ref
    if U_lo is None:
        U_lo = 0.02 * b * w_ref
    grid = np.geomspace(U_lo, U_hi, n_scan)
    guess = None
    prev = None
    out = {"U_crit": None, "omega_crit": 0.0, "k_crit": 0.0, "kind": None,
           "converged": False, "trace": []}
    for U in grid:
        p, ks = pk_eigen_tracked(section, rho, U, p_guess=guess,
                                 u_alpha_sign=u_alpha_sign)
        guess = p
        i = int(np.argmax(p.real))
        g, w, k = p[i].real, abs(p[i].imag), float(ks[i])
        out["trace"].append((float(U), float(g), float(w), float(k)))
        if prev is not None and prev[1] < 0.0 <= g:
            lo, hi = prev[0], U
            for _ in range(120):
                mid = 0.5 * (lo + hi)
                pm, km = pk_eigen_tracked(section, rho, mid, p_guess=guess,
                                          u_alpha_sign=u_alpha_sign)
                im = int(np.argmax(pm.real))
                if pm[im].real >= 0.0:
                    hi = mid
                else:
                    lo = mid
                if hi - lo < tol * max(1.0, hi):
                    break
            pm, km = pk_eigen_tracked(section, rho, hi, p_guess=guess,
                                      u_alpha_sign=u_alpha_sign)
            im = int(np.argmax(pm.real))
            out.update(U_crit=float(hi), omega_crit=float(abs(pm[im].imag)),
                       k_crit=float(km[im]),
                       kind="flutter" if abs(pm[im].imag) > 0.02 * w_ref
                       else "divergence", converged=True)
            return out
        prev = (U, g)
    return out
