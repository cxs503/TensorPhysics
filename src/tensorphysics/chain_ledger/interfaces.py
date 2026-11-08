"""The three hand-off faces of the wave -> 6-DOF -> whipping chain.

Every face is written as an **identity that a correct hand-off must satisfy**,
evaluated by two independent numerical routes so that the residual measures the
hand-off error and not the physics model.

Face A  fluid pressure field  ->  rigid-body resultant force / moment
        route 1: piecewise-constant boundary-node hold (what an LBM boundary
                 momentum-exchange summation produces: one pressure per boundary
                 link, one area, one normal)
        route 2: Gauss quadrature per panel
        reference: dense quadrature on a much finer panel set
        gate: ``|F_h - F_ref| / |F_ref| < 2 %`` with observed order >= 1.

Face B  rigid-body motion  ->  moving/immersed boundary of the flow solver
        checks: (i) closed-surface zero-flux identity ``oint u.n dA = 0`` for a
        rigid motion (normal-rotation consistency), (ii) immersed-volume rate
        ``dV/dt = -int_{S_w} u.n dA`` against an exact volume reference,
        (iii) momentum/power reciprocity: the power computed from the *delivered*
        surface velocity must equal the power from the analytic surface velocity.

Face C  slamming pressure field  ->  beam modal generalised force + Duhamel
        route 1: full 2-D field virtual work ``Q_n = int int p phi_n dxi dx``
        route 2: strip line load ``f(x) = int p dxi`` projected on the beam grid
        route 3: the *sampled* time history handed over by the coupling, then
                 Duhamel convolution
        reference: RK4 integration of the modal ODE driven by route 1
        gate: ``< 3 %`` on the generalised force and on the Duhamel response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .body import BoxBody, RigidPose, Panels, force_moment, rigid_velocity
from .waves import StokesWave

__all__ = [
    "gauss_legendre",
    "face_a_force",
    "face_a_convergence",
    "face_b_closed_flux",
    "face_b_volume_rate",
    "face_b_power_reciprocity",
    "face_c_generalised_force",
    "duhamel_response",
    "rock4_modal",
]

# --------------------------------------------------------------------------
# quadrature helpers
# --------------------------------------------------------------------------
def gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights on ``[-1, 1]`` (``order`` >= 1)."""
    if order < 1:
        raise ValueError("order must be >= 1")
    return np.polynomial.legendre.leggauss(int(order))


def _panel_samples(panels: Panels, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(points, weights, panel_index)`` for per-panel Gauss quadrature.

    Fully vectorised: ``points`` has shape ``(n_panels * order**2, 3)`` and the
    weights of each panel sum to its area, so ``order=1`` reproduces the
    boundary-node hold exactly.
    """
    xg, wg = gauss_legendre(order)
    a = xg[:, None]
    b = xg[None, :]
    local = (
        a[..., None] * panels.tangent1[:, None, None, :]
        + b[..., None] * panels.tangent2[:, None, None, :]
    )
    pts = panels.centroid[:, None, None, :] + local
    w = (wg[:, None] * wg[None, :])[None, :, :] * 0.25 * panels.area[:, None, None]
    idx = np.repeat(np.arange(len(panels)), order * order)
    return pts.reshape(-1, 3), w.reshape(-1), idx


def face_a_force(
    pressure_fn: Callable[[np.ndarray], np.ndarray],
    panels: Panels,
    origin: np.ndarray | None = None,
    method: str = "gauss",
    order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Resultant force/moment of a pressure field over ``panels``.

    ``method="hold"`` reproduces the boundary-node momentum-exchange summation
    of a lattice-Boltzmann solver (one pressure per boundary panel, piecewise
    constant).  ``method="gauss"`` uses per-panel Gauss quadrature of ``order``
    points per panel direction.

    Force of the fluid on the solid: ``F = -int p n dA`` (``n`` outward of the
    solid, hence the minus sign).
    """
    origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float).reshape(3)
    if method == "hold":
        p = np.asarray(pressure_fn(panels.centroid), dtype=float)
        return force_moment(panels, p, origin)
    if method == "gauss":
        pts, wts, idx = _panel_samples(panels, order)
        p = np.asarray(pressure_fn(pts), dtype=float)
        f_panel = -(p * wts)[:, None] * panels.normal[idx]
        F = f_panel.sum(axis=0)
        arm = pts - origin[None, :]
        M = np.cross(arm, f_panel).sum(axis=0)
        return F, M
    raise ValueError(f"unknown method {method!r}")


def face_a_convergence(
    wave: StokesWave,
    body: BoxBody,
    t: float,
    n_edges: Sequence[int] = (2, 4, 8, 16, 32, 64),
    order: int = 2,
    reference: tuple[int, int] = (128, 8),
) -> dict:
    """Mesh-convergence study of the pressure-to-force hand-off (face A).

    The reference force is computed on a much finer panel set with a higher
    quadrature order.  The observed convergence order is the least-squares slope
    of ``log(error)`` versus ``log(h)`` with ``h ~ 1 / n_edge``.
    """
    origin = np.zeros(3)
    pose = RigidPose()

    def pressure_at(points: np.ndarray) -> np.ndarray:
        return wave.pressure(points[:, 0], points[:, 2], t)

    n_ref, q_ref = reference
    pref = body.panels(pose, n_edge=n_ref)
    F_ref, M_ref = face_a_force(pressure_at, pref, origin, method="gauss", order=q_ref)
    # reference self-consistency: one level finer must agree to well below the
    # 2 % gate, otherwise the reference itself is not converged.
    p_ref2 = body.panels(pose, n_edge=2 * n_ref)
    F_ref2, _ = face_a_force(pressure_at, p_ref2, origin, method="gauss", order=q_ref)
    ref_rel = float(np.linalg.norm(F_ref2 - F_ref) / max(np.linalg.norm(F_ref), 1e-30))

    rows = []
    for ne in n_edges:
        panels = body.panels(pose, n_edge=ne)
        F, M = face_a_force(pressure_at, panels, origin, method="gauss", order=order)
        Fh, Mh = face_a_force(pressure_at, panels, origin, method="hold")
        rows.append(
            {
                "n_edge": int(ne),
                "n_panels": int(len(panels)),
                "quadrature_rel_err": float(np.linalg.norm(F - F_ref) / max(np.linalg.norm(F_ref), 1e-30)),
                "hold_rel_err": float(np.linalg.norm(Fh - F_ref) / max(np.linalg.norm(F_ref), 1e-30)),
                "moment_quadrature_rel_err": float(
                    np.linalg.norm(M - M_ref) / max(np.linalg.norm(M_ref), 1e-30)
                ),
                "force_z_hold": float(Fh[2]),
                "force_z_gauss": float(F[2]),
            }
        )
    h = np.array([1.0 / r["n_edge"] for r in rows])
    out = {"t_s": float(t), "reference": {"n_edge": int(n_ref), "order": int(q_ref), "self_consistency": ref_rel},
           "rows": rows, "F_ref": F_ref.tolist(), "M_ref": M_ref.tolist()}
    for key in ("hold_rel_err", "quadrature_rel_err"):
        e = np.array([r[key] for r in rows])
        ok = (e > 1e-14)
        if ok.sum() >= 3:
            slope = float(np.polyfit(np.log(h[ok]), np.log(e[ok]), 1)[0])
        else:  # pragma: no cover - degenerate study
            slope = float("nan")
        out[f"{key}_order"] = slope
    # the finest level is the one the gate reads
    out["rel_err"] = float(rows[-1]["hold_rel_err"])
    out["order"] = float(out["hold_rel_err_order"])
    out["rel_err_gauss"] = float(rows[-1]["quadrature_rel_err"])
    return out


# --------------------------------------------------------------------------
# Face B: kinematics, volume/mass conservation, power reciprocity
# --------------------------------------------------------------------------
def face_b_closed_flux(
    body: BoxBody,
    pose: RigidPose,
    v: np.ndarray,
    omega: np.ndarray,
    n_edges: Sequence[int] = (2, 4, 8, 16),
    rotate_normals: bool = True,
) -> dict:
    """Closed-surface zero-flux identity ``oint u.n dA = 0`` for a rigid motion.

    This is the sharpest available test of a moving/immersed boundary
    implementation: any inconsistency between the delivered surface velocity and
    the delivered normal (for example rotating the positions but forgetting the
    normals, a real defect class in moving-boundary codes) breaks the identity.

    ``rotate_normals=False`` reproduces that defect on purpose - it is the
    negative control for this face.
    """
    rows = []
    for ne in n_edges:
        panels = body.panels(pose, n_edge=ne, wetted=False)
        u = rigid_velocity(pose, v, omega, panels.centroid)
        normals = panels.normal if rotate_normals else body.panels(RigidPose(), n_edge=ne, wetted=False).normal
        if not rotate_normals:
            # the un-rotated panel normals must be paired with the *same* panel
            # layout; both surfaces have identical ordering by construction.
            if normals.shape != panels.normal.shape:
                raise RuntimeError("panel layout mismatch for the negative control")
        flux = float(np.sum(np.einsum("ij,ij->i", u, normals) * panels.area))
        scale = float(np.linalg.norm(np.asarray(v, dtype=float)) * body.panels(pose, ne, wetted=False).total_area
                      + np.linalg.norm(np.asarray(omega, dtype=float)) * body.half_beam
                      * body.panels(pose, ne, wetted=False).total_area)
        rows.append({"n_edge": int(ne), "flux_m3_s": flux, "rel_flux": float(abs(flux) / max(scale, 1e-30))})
    return {
        "rows": rows,
        "rel_flux": float(rows[-1]["rel_flux"]),
        "rotate_normals": bool(rotate_normals),
    }


def face_b_volume_rate(
    body: BoxBody,
    heave: float,
    heave_rate: float,
    n_edges: Sequence[int] = (2, 4, 8, 16),
) -> dict:
    """Immersed-volume identity ``dV/dt = -int_{S_w} u.n dA`` in heave.

    The exact reference is ``V(zeta) = A_bottom (d + zeta)``, so
    ``dV/dt = A_bottom * zeta_dot`` needs no faceted-volume approximation.
    """
    pose = RigidPose(r=np.array([0.0, 0.0, float(heave)]))
    v = np.array([0.0, 0.0, float(heave_rate)])
    a_bottom = 2.0 * body.half_beam * body.width
    dVdt = a_bottom * float(heave_rate)
    rows = []
    for ne in n_edges:
        panels = body.panels(pose, n_edge=ne)
        u = rigid_velocity(pose, v, np.zeros(3), panels.centroid)
        flux = float(np.sum(np.einsum("ij,ij->i", u, panels.normal) * panels.area))
        rows.append(
            {
                "n_edge": int(ne),
                "flux_m3_s": flux,
                "dVdt_m3_s": -flux,
                "dVdt_exact": dVdt,
                "rel_err": float(abs(-flux - dVdt) / max(abs(dVdt), 1e-30)),
            }
        )
    return {"heave_m": float(heave), "heave_rate_m_s": float(heave_rate), "rows": rows,
            "rel_err": float(rows[-1]["rel_err"])}


def face_b_power_reciprocity(
    panels: Panels,
    force_panels: np.ndarray,
    u_analytic: np.ndarray,
    u_delivered: np.ndarray,
) -> dict:
    """Power reciprocity across face B: ``sum F . u_delivered`` vs ``sum F . u_analytic``.

    ``u_delivered`` is the velocity the flow solver actually receives (after the
    coupling's temporal interpolation/offset); ``u_analytic`` is the rigid-body
    velocity at the panel centroid.  Their ratio must be 1 for the hand-off to
    be power-neutral.
    """
    f = np.asarray(force_panels, dtype=float)
    p_a = float(np.sum(np.einsum("ij,ij->i", f, np.asarray(u_analytic, dtype=float))))
    p_d = float(np.sum(np.einsum("ij,ij->i", f, np.asarray(u_delivered, dtype=float))))
    return {
        "power_analytic_W": p_a,
        "power_delivered_W": p_d,
        "rel_err": float(abs(p_d - p_a) / max(abs(p_a), 1e-30)),
    }


# --------------------------------------------------------------------------
# Face C: slamming pressure -> modal generalised force -> Duhamel response
# --------------------------------------------------------------------------
def face_c_generalised_force(
    field_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    xi_half_width: float,
    shapes: np.ndarray,
    n_xi: int = 241,
    n_x_field: int | None = None,
) -> dict:
    """Modal generalised force from a pressure field ``p(x, xi)``.

    route 1 (field): ``Q_n = int int p(x, xi) phi_n(x) dxi dx`` on a field grid
    route 2 (strip): ``f(x) = int p dxi`` then ``Q_n = int f(x) phi_n(x) dx`` on
    the beam grid

    Returns both, their relative difference, and the intermediate line load.
    """
    x = np.asarray(x, dtype=float)
    shapes = np.atleast_2d(np.asarray(shapes, dtype=float))
    xi = np.linspace(-xi_half_width, xi_half_width, int(n_xi))
    nx_field = int(n_x_field or max(4 * x.size, 801))
    x_field = np.linspace(x[0], x[-1], nx_field)

    p = np.asarray(field_fn(x_field, xi), dtype=float)
    if p.shape != (x_field.size, xi.size):
        raise ValueError("field_fn must return an array of shape (x.size, xi.size)")

    # route 1: two-dimensional quadrature on the field grid
    phi_field = np.array([np.interp(x_field, x, phi) for phi in shapes])
    q_field = np.trapezoid(np.trapezoid(p[None, :, :], xi, axis=2) * phi_field, x_field, axis=1)
    # (the inner xi integral is the line load; route 1 keeps the field grid)

    # route 2: strip line load projected on the *beam* grid
    f_line = np.trapezoid(p, xi, axis=1)
    f_beam = np.interp(x, x_field, f_line)
    q_strip = np.trapezoid(shapes * f_beam[None, :], x, axis=1)

    rel = np.abs(q_strip - q_field) / np.maximum(np.abs(q_field), 1e-30)
    # beam-grid resolution error of the same route (audit of route 2 itself)
    q_strip_fine = np.array(
        [np.trapezoid(np.interp(x_field, x, phi) * f_line, x_field) for phi in shapes]
    )
    rel_beam = np.abs(q_strip_fine - q_field) / np.maximum(np.abs(q_field), 1e-30)
    return {
        "Q_field_N": q_field.tolist(),
        "Q_strip_N": q_strip.tolist(),
        "rel_err": rel.tolist(),
        "max_rel_err": float(np.max(rel)),
        "rel_err_beam_resolution": rel_beam.tolist(),
        "line_load_N_m": f_beam.tolist(),
        "x_m": x.tolist(),
    }


def duhamel_response(
    t: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:
    """Modal response by Duhamel convolution (mass-normalised modes).

    ``q_n(t) = int_0^t Q_n(tau) h_n(t - tau) dtau``,
    ``h_n(tau) = exp(-zeta omega tau) sin(omega_d tau) / omega_d``.
    """
    from scipy.signal import fftconvolve
    from scipy.fft import next_fast_len

    t = np.asarray(t, dtype=float)
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    dt = float(t[1] - t[0])
    out = np.zeros_like(Q)
    n_fft = next_fast_len(2 * t.size)
    for i, (w, z) in enumerate(zip(np.atleast_1d(omega), np.atleast_1d(zeta))):
        wd = w * np.sqrt(max(1.0 - z * z, 1e-12))
        h = np.exp(-z * w * t) * np.sin(wd * t) / wd
        out[i] = fftconvolve(Q[i], h, mode="full")[: t.size] * dt
    return out


def rock4_modal(
    t: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:
    """Modal response by RK4 marching of ``q'' + 2 zeta omega q' + omega^2 q = Q``.

    Independent (time-domain) reference for :func:`duhamel_response`.
    """
    t = np.asarray(t, dtype=float)
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    zeta = np.atleast_1d(np.asarray(zeta, dtype=float))
    n = omega.size
    dt = float(t[1] - t[0])
    q = np.zeros(n)
    qd = np.zeros(n)
    out = np.zeros_like(Q)
    out[:, 0] = q
    for k in range(t.size - 1):
        def acc(qv, qdv, fv):
            return fv - 2.0 * zeta * omega * qdv - omega ** 2 * qv

        qa, qb = Q[:, k], Q[:, k + 1]
        qm = 0.5 * (qa + qb)
        f1_q, f1_v = qd, acc(q, qd, qa)
        f2_q, f2_v = qd + 0.5 * dt * f1_v, acc(q + 0.5 * dt * f1_q, qd + 0.5 * dt * f1_v, qm)
        f3_q, f3_v = qd + 0.5 * dt * f2_v, acc(q + 0.5 * dt * f2_q, qd + 0.5 * dt * f2_v, qm)
        f4_q, f4_v = qd + dt * f3_v, acc(q + dt * f3_q, qd + dt * f3_v, qb)
        q = q + dt / 6.0 * (f1_q + 2 * f2_q + 2 * f3_q + f4_q)
        qd = qd + dt / 6.0 * (f1_v + 2 * f2_v + 2 * f3_v + f4_v)
        out[:, k + 1] = q
    return out