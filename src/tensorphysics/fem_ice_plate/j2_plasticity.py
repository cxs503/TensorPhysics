"""J2 flow theory with isotropic hardening -- the *numerical* (FEM) face.

This module is the numerical counterpart of
:mod:`tensorphysics.plasticity.j2_flow` (which is analytic).  It provides

* a small-strain von Mises (J2) return-mapping algorithm with isotropic
  hardening (linear / Voce / Swift / perfect) storing the plastic strain in
  the Gauss-point state,
* a numerically consistent tangent ``d sigma / d eps`` (central differences
  of the return map at fixed previous state), validated against the classical
  elastoplastic modulus,
* a uniaxial *stress*-controlled driver that exercises the full 3-D return
  map (no 1-D reduction shortcuts) and a closed-form uniaxial reference,
* a genuine finite element solution of the plane-strain **thick-walled
  cylinder under internal pressure** -- radial mesh, incremental
  displacement-driven Newton iterations, radial return mapping at Gauss
  points,
* closed-form reference fields (Lame elastic, von Mises perfectly plastic
  autofrettage) used as verification anchors, and
* energy bookkeeping (external work, stored strain energy, plastic
  dissipation).

Conventions
-----------
Stress/strain are Voigt 6-vectors ``[xx, yy, zz, xy, yz, zx]`` with
**engineering** shear strains (``gamma = 2 eps_xy``).  Plane strain in the
axial direction: ``eps_zz = 0``.

All quantities are in SI (Pa, m, J).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "J2Material",
    "J2State",
    "voigt_dev_norm",
    "equivalent_stress",
    "return_map",
    "consistent_tangent_numerical",
    "plane_stress_response",
    "uniaxial_tension_curve",
    "uniaxial_reference",
    "thick_cylinder_fe",
    "thick_cylinder_elastic",
    "thick_cylinder_plastic_reference",
    "lamé_compliance",
    "uniaxial_elastoplastic_modulus",
]

_I6 = np.eye(6)
#: ``1 (x) 1`` in Voigt notation: only the normal-normal block is non-zero
#: (the shear rows/columns of the tensor pair product vanish).
_J1 = np.zeros((6, 6))
_J1[:3, :3] = 1.0
_I_DEV = _I6 - _J1 / 3.0
#: slots of the engineering-shear components in the Voigt vector
_SHEAR = np.array([3, 4, 5])


def _elastic_matrix(E: float, nu: float) -> np.ndarray:
    """Isotropic stiffness for Voigt vectors with *engineering* shear.

    The normal-normal block carries ``lambda + 2 mu`` on the diagonal and
    ``lambda`` off-diagonal; the shear diagonal carries ``2 mu`` only.
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam * _J1 + 2.0 * mu * _I6


def voigt_dev_norm(sigma: np.ndarray) -> float:
    """``sqrt(s:s)`` of the deviator of a Voigt stress vector."""
    sigma = np.asarray(sigma, dtype=float)
    p = (sigma[0] + sigma[1] + sigma[2]) / 3.0
    s = sigma.copy()
    s[:3] -= p
    sq = s[0] ** 2 + s[1] ** 2 + s[2] ** 2 + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
    return float(np.sqrt(max(sq, 0.0)))


def equivalent_stress(sigma: np.ndarray) -> float:
    """von Mises equivalent stress ``sqrt(3/2 s:s)``."""
    return float(np.sqrt(1.5) * voigt_dev_norm(sigma))


# ---------------------------------------------------------------------------
# material
# ---------------------------------------------------------------------------
@dataclass
class J2Material:
    """Isotropic-hardening von Mises material.

    Parameters
    ----------
    E, nu : float
        Young's modulus [Pa], Poisson ratio [-].
    sigma_y0 : float
        Initial yield stress [Pa].
    hardening : {"linear", "voce", "swift", "perfect"}
    H : float
        Linear hardening modulus [Pa].
    Q, b : float
        Voce: ``sigma_y = sigma_y0 + Q (1 - exp(-b ep))``.
    K, n, eps0 : float
        Swift: ``sigma_y = K (eps0 + ep)**n`` (used when ``K > 0``).
    """

    E: float
    nu: float = 0.3
    sigma_y0: float = 250.0e6
    hardening: str = "linear"
    H: float = 0.0
    Q: float = 0.0
    b: float = 0.0
    K: float = 0.0
    n: float = 0.0
    eps0: float = 0.0

    def __post_init__(self) -> None:
        if self.E <= 0.0:
            raise ValueError("E must be positive")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError("nu must lie in (-1, 0.5)")
        if self.hardening not in ("linear", "voce", "swift", "perfect"):
            raise ValueError(f"unknown hardening family {self.hardening!r}")
        if self.hardening == "swift":
            if self.K <= 0.0:
                raise ValueError("Swift hardening needs K > 0")
        elif self.sigma_y0 <= 0.0:
            raise ValueError("sigma_y0 must be positive")

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def kappa(self) -> float:
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    def elastic_matrix(self) -> np.ndarray:
        return _elastic_matrix(self.E, self.nu)

    def yield_stress(self, ep: float) -> float:
        ep = max(0.0, float(ep))
        if self.hardening == "perfect":
            return self.sigma_y0
        if self.hardening == "linear":
            return self.sigma_y0 + self.H * ep
        if self.hardening == "voce":
            return self.sigma_y0 + self.Q * (1.0 - np.exp(-self.b * ep))
        if self.hardening == "swift":
            return self.K * (self.eps0 + ep) ** self.n
        raise ValueError(self.hardening)  # pragma: no cover

    def hardening_modulus(self, ep: float) -> float:
        ep = max(0.0, float(ep))
        if self.hardening == "perfect":
            return 0.0
        if self.hardening == "linear":
            return self.H
        if self.hardening == "voce":
            return self.Q * self.b * np.exp(-self.b * ep)
        if self.hardening == "swift":
            return self.K * self.n * (self.eps0 + ep) ** (self.n - 1.0)
        raise ValueError(self.hardening)  # pragma: no cover


@dataclass
class J2State:
    """Gauss-point state: stress [Pa], plastic strain, equivalent plastic strain."""

    sigma: np.ndarray
    eps_p: np.ndarray
    ep: float = 0.0
    plastic_work: float = 0.0  # plastic dissipation density [J/m^3]

    @staticmethod
    def zero() -> "J2State":
        return J2State(np.zeros(6), np.zeros(6), 0.0, 0.0)

    def copy(self) -> "J2State":
        return J2State(self.sigma.copy(), self.eps_p.copy(), self.ep, self.plastic_work)


# ---------------------------------------------------------------------------
# radial return mapping
# ---------------------------------------------------------------------------
def return_map(
    mat: J2Material,
    eps: np.ndarray,
    state: Optional[J2State] = None,
    tol: float = 1e-12,
    max_iter: int = 60,
) -> Tuple[np.ndarray, J2State]:
    """Implicit radial return mapping (small strain, isotropic hardening).

    Parameters
    ----------
    mat : J2Material
    eps : array_like, shape (6,)
        Total strain (engineering shear), relative to the undeformed state.
    state : J2State, optional
        Previous converged state (defaults to virgin material).

    Returns
    -------
    sigma : ndarray, shape (6,)
    state_new : J2State
    """
    if state is None:
        state = J2State.zero()
    eps = np.asarray(eps, dtype=float)
    D_e = mat.elastic_matrix()
    sigma_tr = D_e @ (eps - state.eps_p)
    q_tr = equivalent_stress(sigma_tr)
    sy_old = mat.yield_stress(state.ep)

    if q_tr - sy_old <= 1e-12 * max(sy_old, 1.0):
        return sigma_tr, J2State(sigma_tr, state.eps_p.copy(), state.ep, state.plastic_work)

    # unit deviator of the trial stress
    p_tr = (sigma_tr[0] + sigma_tr[1] + sigma_tr[2]) / 3.0
    s_tr = sigma_tr.copy()
    s_tr[:3] -= p_tr
    s_norm = voigt_dev_norm(sigma_tr)
    u_dir = s_tr / s_norm

    G = mat.mu
    dg = 0.0
    for _ in range(max_iter):
        sy = mat.yield_stress(state.ep + dg)
        Hp = mat.hardening_modulus(state.ep + dg)
        res = q_tr - 3.0 * G * dg - sy
        step = res / (3.0 * G + Hp)
        dg = max(dg + step, 0.0)
        if abs(res) <= tol * max(sy_old, 1.0) or abs(step) <= tol * 1e-3:
            break

    # sigma = sigma_tr - 2 G sqrt(1.5) dg u   (q drops by exactly 3 G dg)
    kfac = 2.0 * G * np.sqrt(1.5) * dg
    sigma = sigma_tr - kfac * u_dir
    # plastic strain increment: d eps_p = sqrt(1.5) dg u  (tensor components);
    # the Voigt storage uses engineering shear -> the shear slots are doubled.
    d_eps_p = np.sqrt(1.5) * dg * u_dir
    d_eps_p[_SHEAR] *= 2.0
    eps_p = state.eps_p + d_eps_p
    ep_new = state.ep + dg
    sy_mid = 0.5 * (sy_old + mat.yield_stress(ep_new))
    new = J2State(sigma, eps_p, ep_new, state.plastic_work + sy_mid * dg)
    return sigma, new


def consistent_tangent_numerical(
    mat: J2Material,
    eps: np.ndarray,
    state: Optional[J2State] = None,
    rel_step: float = 1e-7,
) -> np.ndarray:
    """Numerically consistent tangent ``d sigma / d eps`` (central differences).

    The perturbation is applied to the total strain while the *previous*
    state is held fixed, which is exactly the algorithmic (consistent)
    tangent used by the Newton iteration of the FE solver.
    """
    eps = np.asarray(eps, dtype=float)
    scale = max(float(np.max(np.abs(eps))), 1e-14)
    # floor the perturbation so that it stays meaningful in the elastic range
    h = rel_step * scale + 1e-14
    D = np.zeros((6, 6))
    for j in range(6):
        ep = eps.copy()
        em = eps.copy()
        ep[j] += h
        em[j] -= h
        sp, _ = return_map(mat, ep, state)
        sm, _ = return_map(mat, em, state)
        D[:, j] = (sp - sm) / (2.0 * h)
    return D


def plane_stress_response(
    mat: J2Material,
    eps_inplane: Sequence[float],
    state: Optional[J2State] = None,
    tol: float = 1e-11,
    max_iter: int = 60,
) -> Tuple[np.ndarray, J2State]:
    """Plane-stress response ``(eps_rr, eps_tt) -> (sigma, state)`` with
    ``sigma_zz = 0``.

    The out-of-plane strain ``eps_zz`` is an internal unknown, solved from
    ``sigma_zz = 0`` around the 3-D return mapping (Newton with a numeric
    derivative).  This makes the plane-stress thick disk comparison against
    the exact closed form meaningful.
    """
    er, et = float(eps_inplane[0]), float(eps_inplane[1])
    nu = mat.nu
    ez = -nu / max(1.0 - nu, 1e-6) * (er + et)  # elastic plane-stress guess
    if state is not None:
        ez += state.eps_p[2]
    h = 1e-8 * max(abs(er) + abs(et), 1e-6)
    sigma = np.zeros(6)
    for _ in range(max_iter):
        sigma, _ = return_map(mat, np.array([er, et, ez, 0.0, 0.0, 0.0]), state)
        f = sigma[2]
        if abs(f) <= tol * max(mat.yield_stress(0.0 if state is None else state.ep), 1.0):
            break
        sp, _ = return_map(mat, np.array([er, et, ez + h, 0.0, 0.0, 0.0]), state)
        sm, _ = return_map(mat, np.array([er, et, ez - h, 0.0, 0.0, 0.0]), state)
        d = (sp[2] - sm[2]) / (2.0 * h)
        if abs(d) < 1e-12:
            break
        step = -f / d
        ez += float(np.clip(step, -0.5 * h * 1e6, 0.5 * h * 1e6))
    _, state_new = return_map(mat, np.array([er, et, ez, 0.0, 0.0, 0.0]), state)
    return sigma, state_new


def uniaxial_elastoplastic_modulus(mat: J2Material) -> float:
    """Classical uniaxial elastoplastic modulus ``E H' / (E + H')`` at ep=0."""
    Hp = mat.hardening_modulus(0.0)
    return mat.E * Hp / (mat.E + Hp)


# ---------------------------------------------------------------------------
# uniaxial tension: numerical (3-D return map) vs closed form
# ---------------------------------------------------------------------------
def uniaxial_tension_curve(
    mat: J2Material,
    eps_max: float = 0.02,
    n_steps: int = 200,
    tol: float = 1e-10,
    max_iter: int = 40,
) -> Dict[str, np.ndarray]:
    """Uniaxial tensile curve driven through the full 3-D return map.

    The axial strain ``eps_1`` is prescribed and the lateral strain
    ``eps_2 = eps_3`` is solved from the plane-stress condition
    ``sigma_2 = sigma_3 = 0`` by a Newton iteration **around the return
    mapping itself** (no incompressibility shortcut, so the constitutive
    algorithm is exercised end-to-end).

    Returns
    -------
    dict with keys ``eps``, ``sigma``, ``ep``, ``eps_p_axial``,
    ``plastic_work``.
    """
    eps_hist = np.linspace(0.0, eps_max, n_steps + 1)
    sigma_hist = np.zeros_like(eps_hist)
    ep_hist = np.zeros_like(eps_hist)
    epax_hist = np.zeros_like(eps_hist)
    work_hist = np.zeros_like(eps_hist)
    state = J2State.zero()
    E, nu = mat.E, mat.nu

    def sig_at(eps1: float, eps2: float, st: J2State) -> np.ndarray:
        sigma, _ = return_map(mat, np.array([eps1, eps2, eps2, 0.0, 0.0, 0.0]), st)
        return sigma

    for k, eps1 in enumerate(eps_hist):
        if k == 0:
            continue
        eps2 = -nu * eps1
        sigma = sig_at(eps1, eps2, state)
        h = 1e-7 * max(abs(eps2), 1e-8)
        for _ in range(max_iter):
            s0 = sig_at(eps1, eps2, state)
            f = s0[1]
            if abs(f) <= tol * max(mat.yield_stress(state.ep), 1.0):
                break
            dplus = sig_at(eps1, eps2 + h, state)[1]
            dminus = sig_at(eps1, eps2 - h, state)[1]
            dsd = (dplus - dminus) / (2.0 * h)
            if abs(dsd) < 1e-12:
                break
            step = -f / dsd
            # damp the update to keep the plane-stress iteration stable
            eps2 += float(np.clip(step, -0.2 * max(abs(eps1), 1e-6), 0.2 * max(abs(eps1), 1e-6)))
        _, state = return_map(mat, np.array([eps1, eps2, eps2, 0.0, 0.0, 0.0]), state)
        sigma_hist[k] = state.sigma[0]
        ep_hist[k] = state.ep
        epax_hist[k] = state.eps_p[0]
        work_hist[k] = state.plastic_work

    return {
        "eps": eps_hist,
        "sigma": sigma_hist,
        "ep": ep_hist,
        "eps_p_axial": epax_hist,
        "plastic_work": work_hist,
    }


def uniaxial_reference(mat: J2Material, eps: Sequence[float]) -> Dict[str, np.ndarray]:
    """Closed-form uniaxial curve: ``eps = sigma/E + ep`` with ``sigma = sy(ep)``.

    Solves the scalar consistency equation by bisection -- an independent
    numerical route from the incremental return mapping.
    """
    eps = np.asarray(eps, dtype=float)
    sigma = np.zeros_like(eps)
    ep = np.zeros_like(eps)
    for i, e in enumerate(eps):
        if e <= 0.0:
            continue

        def g(x: float) -> float:
            return mat.yield_stress(x) / mat.E + x - e

        if g(0.0) >= 0.0:  # eps below the initial yield strain -> elastic
            sigma[i] = mat.E * e
            ep[i] = 0.0
            continue
        hi = max(e, 1e-12)
        while g(hi) < 0.0:
            hi *= 2.0
            if hi > 10.0:
                raise RuntimeError("uniaxial_reference: ep bracket failed")
        lo = 0.0
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            if g(mid) > 0.0:
                hi = mid
            else:
                lo = mid
            if abs(hi - lo) < 1e-16:
                break
        ep[i] = 0.5 * (lo + hi)
        sigma[i] = mat.yield_stress(ep[i])
    return {"eps": eps, "sigma": sigma, "ep": ep}


# ---------------------------------------------------------------------------
# thick-walled cylinder: closed-form references
# ---------------------------------------------------------------------------
def thick_cylinder_elastic(
    a: float, b: float, p: float, E: float, nu: float, r: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Lame plane-strain elastic field of a pressurised cylinder.

    ``sigma_z = nu (sigma_r + sigma_theta)`` follows from ``eps_z = 0``.
    """
    if r is None:
        r = np.linspace(a, b, 201)
    r = np.asarray(r, dtype=float)
    c = p * a**2 / (b**2 - a**2)
    # internal pressure: sigma_r(a) = -p (compression), sigma_theta > 0 (tension)
    sig_r = c * (1.0 - b**2 / r**2)
    sig_t = c * (1.0 + b**2 / r**2)
    sig_z = nu * (sig_r + sig_t)
    u = r / E * ((1.0 - nu**2) * sig_t - nu * (1.0 + nu) * sig_r)
    return {"r": r, "sigma_r": sig_r, "sigma_theta": sig_t, "sigma_z": sig_z, "u": u}


def lamé_compliance(a: float, b: float, E: float, nu: float) -> float:
    """Elastic plane-strain compliance ``u(a)/p`` of a pressurised cylinder."""
    ratio = (b**2 + a**2) / (b**2 - a**2)
    return a / E * ((1.0 - nu**2) * ratio + nu * (1.0 + nu))


def thick_cylinder_plastic_reference(
    a: float,
    b: float,
    p: float,
    sigma_y: float,
    nu: float = 0.3,
    E: Optional[float] = None,
) -> Dict[str, object]:
    """von Mises perfectly-plastic (autofrettage) plane-strain reference.

    Plastic zone (``a <= r <= c``)::

        sigma_theta - sigma_r = 2 k,      k = sigma_y / sqrt(3)
        sigma_r = 2 k ln(r/a) - p
        sigma_theta = sigma_r + 2 k
        sigma_z = sigma_r + k            (plane strain + von Mises)

    Elastic zone (``c <= r <= b``)::

        sigma_r = k c^2 (1/b^2 - 1/r^2),  sigma_theta = k c^2 (1/b^2 + 1/r^2)
        sigma_z = nu (sigma_r + sigma_theta)

    The plastic radius follows from ``p = 2k ln(c/a) + k (1 - c^2/b^2)``
    (solved here by bisection).  Returns ``None``-valued fields when the
    cylinder stays elastic.
    """
    k = sigma_y / np.sqrt(3.0)
    p_yield = k * (b**2 - a**2) / b**2
    out: Dict[str, object] = {
        "p_yield": p_yield,
        "k": k,
        "plastic": p > p_yield,
    }
    if p <= p_yield:
        def elastic_fields(r):
            f = thick_cylinder_elastic(a, b, p, E or 200.0e9, nu, r=r)
            return {
                "r": f["r"],
                "sigma_r": f["sigma_r"],
                "sigma_theta": f["sigma_theta"],
                "sigma_z": f["sigma_z"],
            }

        out.update({"c": a, "p": p, "fields": elastic_fields})
        return out

    def f(c: float) -> float:
        return 2.0 * k * np.log(c / a) + k * (1.0 - c**2 / b**2) - p

    if f(b) < 0.0:  # fully plastic -- outside the validity range
        c = b
    else:
        lo, hi = a, b
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        c = 0.5 * (lo + hi)

    def fields(r):
        r = np.asarray(r, dtype=float)
        sr = np.where(
            r <= c,
            2.0 * k * np.log(np.maximum(r, a) / a) - p,
            k * c**2 * (1.0 / b**2 - 1.0 / r**2),
        )
        st = np.where(r <= c, sr + 2.0 * k, k * c**2 * (1.0 / b**2 + 1.0 / r**2))
        sz = np.where(r <= c, sr + k, nu * (sr + st))
        return {"r": r, "sigma_r": sr, "sigma_theta": st, "sigma_z": sz}

    out.update({"c": c, "p": p, "fields": fields})
    # residual (after unloading) hoop stress at the bore -- used by C5 checks
    if E is not None:
        r_a = np.array([a])
        fld = fields(r_a)
        el = thick_cylinder_elastic(a, b, p, E, nu, r=r_a)
        sr_res = fld["sigma_r"][0] - el["sigma_r"][0]
        st_res = fld["sigma_theta"][0] - el["sigma_theta"][0]
        u_res = a / E * ((1.0 - nu**2) * st_res - nu * (1.0 + nu) * sr_res)
        out["u_residual_elastic_part"] = float(u_res)
        out["sigma_theta_residual_bore"] = float(st_res)
        out["sigma_r_residual_bore"] = float(sr_res)
    return out


def thick_cylinder_plane_stress_reference(
    a: float,
    b: float,
    p: float,
    sigma_y: float,
    E: Optional[float] = None,
    nu: float = 0.3,
) -> Dict[str, object]:
    """Exact plane-stress von Mises thick-disk reference (``sigma_z = 0``).

    Plastic zone (``a <= r <= c``)::

        sigma_theta - sigma_r = sigma_y
        sigma_r = sigma_y ln(r/a) - p
        sigma_theta = sigma_r + sigma_y

    Elastic zone: ``sigma_r = A - B/r^2``, ``sigma_theta = A + B/r^2`` with
    ``A = B/b^2``, ``B = sigma_y c^2 / 2``, and the plastic radius from

        p = sigma_y ln(c/a) + (sigma_y/2) (1 - c^2/b^2).

    Unlike the classical plane-strain autofrettage formula, this solution is
    *exact* (``sigma_z = 0`` everywhere, no axial compatibility assumption),
    which makes it a clean verification anchor for the FE solver.
    """
    p_yield = 0.5 * sigma_y * (b**2 - a**2) / b**2
    p_full = sigma_y * np.log(b / a)
    out: Dict[str, object] = {"p_yield": p_yield, "p_full_plastic": p_full, "plastic": p > p_yield}
    if p <= p_yield:
        c = a
    elif p >= p_full:
        c = b
    else:
        def f(cc: float) -> float:
            return sigma_y * np.log(cc / a) + 0.5 * sigma_y * (1.0 - cc**2 / b**2) - p

        lo, hi = a, b
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        c = 0.5 * (lo + hi)

    B = 0.5 * sigma_y * c**2

    def fields(r):
        r = np.asarray(r, dtype=float)
        sr = np.where(r <= c, sigma_y * np.log(np.maximum(r, a) / a) - p,
                      B * (1.0 / b**2 - 1.0 / r**2))
        st = np.where(r <= c, sr + sigma_y, B * (1.0 / b**2 + 1.0 / r**2))
        return {"r": r, "sigma_r": sr, "sigma_theta": st, "sigma_z": np.zeros_like(r)}

    out.update({"c": c, "p": p, "fields": fields})
    if E is not None:
        r_a = np.array([a])
        fld = fields(r_a)
        el = thick_cylinder_elastic(a, b, p, E, nu, r=r_a)  # plane strain Lamé
        # plane-stress elastic bore displacement
        sig_r, sig_t = fld["sigma_r"][0], fld["sigma_theta"][0]
        u_total = a / E * (sig_t - nu * sig_r)
        u_el = a / E * (el["sigma_theta"][0] - nu * el["sigma_r"][0])
        out["u_bore"] = float(u_total)
        out["u_residual_bore"] = float(u_total - u_el)
        out["sigma_theta_residual_bore"] = float(sig_t - el["sigma_theta"][0])
    return out


# ---------------------------------------------------------------------------
# thick-walled cylinder: incremental FE solution
# ---------------------------------------------------------------------------
@dataclass
class CylinderFEResult:
    """Result container of :func:`thick_cylinder_fe`."""

    a: float
    b: float
    p_target: float
    r_nodes: np.ndarray
    u_nodes: np.ndarray
    r_gauss: np.ndarray
    sigma_gauss: np.ndarray  # (n_el*2, 6) Voigt
    eps_gauss: np.ndarray  # total strain (Voigt, engineering shear)
    eps_p_gauss: np.ndarray  # plastic strain (Voigt, engineering shear)
    ep_gauss: np.ndarray
    plastic_work: np.ndarray
    le: np.ndarray
    p_history: np.ndarray
    u_bore_history: np.ndarray
    r_plastic: float
    converged: bool
    n_newton: int

    def field(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        r = self.r_gauss
        idx = {"radial": 0, "hoop": 1, "axial": 2}[name]
        return r, self.sigma_gauss[:, idx]

    def energies(self) -> Dict[str, float]:
        """External work, stored elastic energy and plastic dissipation."""
        return _cylinder_energies(self)


def thick_cylinder_fe(
    mat: J2Material,
    a: float,
    b: float,
    p_target: float,
    n_el: int = 80,
    n_inc: int = 20,
    max_newton: int = 25,
    tol: float = 1e-9,
    mode: str = "plane_strain",
    p_path: Optional[Sequence[float]] = None,
) -> CylinderFEResult:
    """Plane-strain thick cylinder under internal pressure -- radial FE.

    * 2-node linear radial elements, 2-point Gauss quadrature, axisymmetric
      measure ``2*pi*r dr``;
    * incremental internal pressure with displacement-driven Newton
      iterations (the pressure is applied as a consistent nodal load
      ``2*pi*a*p`` at the bore);
    * radial return mapping at each Gauss point, numerically consistent
      tangent for the element stiffness.

    ``mode`` selects the axial constraint: ``"plane_strain"`` (``eps_zz = 0``,
    closed/thick body) or ``"plane_stress"`` (``sigma_zz = 0``, free ends,
    solved through :func:`plane_stress_response`).
    """
    if mode not in ("plane_strain", "plane_stress"):
        raise ValueError(mode)
    r_nodes = np.linspace(a, b, n_el + 1)
    le = np.diff(r_nodes)
    gp_xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    gp_w = np.array([1.0, 1.0])

    r_gp = np.zeros((n_el, 2))
    for e in range(n_el):
        r_gp[e] = r_nodes[e] + 0.5 * le[e] * (gp_xi + 1.0)

    states = [J2State.zero() for _ in range(n_el * 2)]
    u = np.zeros(n_el + 1)
    p_hist, u_hist = [0.0], [0.0]
    total_newton = 0

    def strain_at(e: int, gp: int, u_vec: np.ndarray) -> np.ndarray:
        N = np.array([(1.0 - gp_xi[gp]) / 2.0, (1.0 + gp_xi[gp]) / 2.0])
        dN = np.array([-1.0 / le[e], 1.0 / le[e]])
        ue = u_vec[e : e + 2]
        eps_r = float(dN @ ue)
        eps_t = float(N @ ue) / r_gp[e, gp]
        return np.array([eps_r, eps_t, 0.0, 0.0, 0.0, 0.0])

    def response(eps6: np.ndarray, st: J2State):
        if mode == "plane_strain":
            return return_map(mat, eps6, st)
        return plane_stress_response(mat, (eps6[0], eps6[1]), st)

    def tangent(eps6: np.ndarray, st: J2State) -> np.ndarray:
        """Algorithmic tangent for the two in-plane strain components.

        Only ``eps_rr`` and ``eps_tt`` are element degrees of freedom here, so
        only those columns are needed; the plane-stress condensation is
        included automatically because ``response`` re-solves ``eps_zz``.
        """
        D = np.zeros((6, 6))
        h = 1e-7 * max(abs(eps6[0]), abs(eps6[1]), 1e-12)
        for j in (0, 1):
            ep = eps6.copy()
            em = eps6.copy()
            ep[j] += h
            em[j] -= h
            sp = response(ep, st)[0]
            sm = response(em, st)[0]
            D[:, j] = (sp - sm) / (2.0 * h)
        return D

    def b_matrix(e: int, gp: int) -> np.ndarray:
        N = np.array([(1.0 - gp_xi[gp]) / 2.0, (1.0 + gp_xi[gp]) / 2.0])
        B = np.zeros((6, 2))
        B[0, 0] = -1.0 / le[e]
        B[0, 1] = 1.0 / le[e]
        B[1, 0] = N[0] / r_gp[e, gp]
        B[1, 1] = N[1] / r_gp[e, gp]
        return B

    if p_path is not None:
        pressure_steps = [float(x) for x in p_path]
        p_target = pressure_steps[-1]
    else:
        pressure_steps = [p_target * inc / n_inc for inc in range(1, n_inc + 1)]

    converged = True
    for p_now in pressure_steps:
        f_ext = np.zeros(n_el + 1)
        f_ext[0] = 2.0 * np.pi * a * p_now
        it = 0
        for it in range(max_newton):
            K = np.zeros((n_el + 1, n_el + 1))
            f_int = np.zeros(n_el + 1)
            for e in range(n_el):
                for gp in range(2):
                    idx = e * 2 + gp
                    r = r_gp[e, gp]
                    w = gp_w[gp] * 0.5 * le[e] * 2.0 * np.pi * r
                    eps = strain_at(e, gp, u)
                    sigma, _ = response(eps, states[idx])
                    B = b_matrix(e, gp)
                    f_int[e : e + 2] += w * (B.T @ sigma)
                    D = tangent(eps, states[idx])
                    K[e : e + 2, e : e + 2] += w * (B.T @ D @ B)
            res = f_ext - f_int
            # a full annulus has no rigid-body mode in the radial direction:
            # every radial DOF is free
            free = np.arange(0, n_el + 1)
            K_ff = K[np.ix_(free, free)]
            res_f = res[free]
            scale = max(np.max(np.abs(f_ext)), 1.0)
            if np.max(np.abs(res_f)) <= tol * scale:
                total_newton += it
                break
            du = np.linalg.solve(K_ff, res_f)
            # simple backtracking line search on the residual norm
            alpha = 1.0
            u_trial = u.copy()
            u_trial[free] += alpha * du
            for _ in range(8):
                r_new = _residual_norm(response, u_trial, states, n_el, gp_w, le,
                                       r_gp, f_ext, strain_at, b_matrix)
                if r_new <= np.max(np.abs(res_f)) or alpha < 1e-3:
                    break
                alpha *= 0.5
                u_trial = u.copy()
                u_trial[free] += alpha * du
            u = u_trial
        else:
            converged = False

        # commit the Gauss-point states for the converged load level
        for e in range(n_el):
            for gp in range(2):
                idx = e * 2 + gp
                eps = strain_at(e, gp, u)
                _, states[idx] = response(eps, states[idx])
        p_hist.append(p_now)
        u_hist.append(u[0])

    # collect fields
    sig = np.zeros((n_el * 2, 6))
    epsv = np.zeros((n_el * 2, 6))
    epspv = np.zeros((n_el * 2, 6))
    epv = np.zeros(n_el * 2)
    pw = np.zeros(n_el * 2)
    for e in range(n_el):
        for gp in range(2):
            idx = e * 2 + gp
            sig[idx] = states[idx].sigma
            epsv[idx] = strain_at(e, gp, u)
            epspv[idx] = states[idx].eps_p
            epv[idx] = states[idx].ep
            pw[idx] = states[idx].plastic_work
    r_flat = r_gp.reshape(-1)
    order = np.argsort(r_flat)
    # plastic radius: outermost Gauss point with ep > 1e-10 (contiguous from bore)
    mask = epv[order] > 1e-10
    n_pl = int(np.sum(np.cumprod(mask))) if mask.size else 0
    r_plastic = float(r_flat[order][n_pl - 1]) if n_pl > 0 else float(a)

    res = CylinderFEResult(
        a=a,
        b=b,
        p_target=p_target,
        r_nodes=r_nodes,
        u_nodes=u,
        r_gauss=r_flat[order],
        sigma_gauss=sig[order],
        eps_gauss=epsv[order],
        eps_p_gauss=epspv[order],
        ep_gauss=epv[order],
        plastic_work=pw[order],
        le=le,
        p_history=np.array(p_hist),
        u_bore_history=np.array(u_hist),
        r_plastic=r_plastic,
        converged=converged,
        n_newton=total_newton,
    )
    return res


def _residual_norm(response, u, states, n_el, gp_w, le, r_gp, f_ext, strain_at, b_matrix):
    f_int = np.zeros(n_el + 1)
    for e in range(n_el):
        for gp in range(2):
            idx = e * 2 + gp
            r = r_gp[e, gp]
            w = gp_w[gp] * 0.5 * le[e] * 2.0 * np.pi * r
            eps = strain_at(e, gp, u)
            sigma, _ = response(eps, states[idx])
            B = b_matrix(e, gp)
            f_int[e : e + 2] += w * (B.T @ sigma)
    return float(np.max(np.abs(f_ext - f_int)))


def _cylinder_energies(res: CylinderFEResult) -> Dict[str, float]:
    """Energy audit of the cylinder FE solution (per unit axial length).

    ``W_ext`` -- external work ``int 2 pi a p du_bore`` (trapezoidal over the
    load history);
    ``W_elastic`` -- stored elastic energy ``sum w * (1/2) sigma : eps_e``;
    ``W_plastic`` -- plastic dissipation ``sum w * int sigma_y dep``.
    """
    W_ext = float(np.trapezoid(2.0 * np.pi * res.a * res.p_history, res.u_bore_history))
    W_p = 0.0
    W_e = 0.0
    n_el = len(res.le)
    for e in range(n_el):
        for gp in range(2):
            idx = e * 2 + gp
            w = 0.5 * res.le[e] * 2.0 * np.pi * res.r_gauss[idx]
            W_p += w * res.plastic_work[idx]
            eps_e = res.eps_gauss[idx] - res.eps_p_gauss[idx]
            W_e += w * 0.5 * float(res.sigma_gauss[idx] @ eps_e)
    return {"W_ext": W_ext, "W_plastic": float(W_p), "W_elastic": float(W_e)}
