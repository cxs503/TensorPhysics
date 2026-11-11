"""Elastoplastic progressive collapse of stiffened panels (P167W2B-N04).

Verification surface added on top of the existing ``elastoplastic_collapse``
kernel:

1. **von Mises + linear isotropic hardening** (``sigma_y = sigma_y0 + H ebar_p``):
   exact 1-D closed form, an independent incremental radial-return integrator,
   and the *algorithmic (consistent) tangent* of the 3-D return mapping
   (Simo & Taylor 1986), so an implicit FE driver gets a tangent matrix.

2. **Stiffened panel axial compression to collapse (Smith method with
   elastoplastic element curves)**: iso-strain parallel assembly of a plating
   element (elastic plate buckling -> von Karman post-buckling branch with the
   *current* yield stress, so hardening raises the plateau -> Faulkner (1975)
   ultimate -> declared post-ultimate shedding) and a stiffener element
   (Perry-Robertson elastoplastic column = buckling-yield interaction).  The
   peak of the load-shortening curve is the ultimate strength ``sigma_u``.

3. **Paik & Thayamballi (2003) simplified-method reference**: element-wise
   rigid-plastic ultimate summation with published element curves (Faulkner
   plating + Johnson-Ostenfeld elastoplastic column), implemented independently
   of the fiber solver and used as the published-method benchmark.

4. **Simply supported plate elastic buckling vs Timoshenko**: closed form
   ``k(m) = (m b/a + a/(m b))^2``, ``sigma_cr = k pi^2 D /(b^2 t)`` compared with
   an independent finite-difference eigenvalue solution (13-point biharmonic
   stencil, central-difference geometric term) plus Richardson extrapolation.

5. **IACS ice belt pressure-area law**: ``P = P0 (A/A0)^-0.3`` and the inverse
   problem (log-log least squares exponent recovery) with a noise-robustness
   study of the exponent's identifiability.

Evidence grading
----------------
REAL: every numerical result of this module (closed forms, radial return,
consistent tangent, FD eigenvalues, log-log inversion, fiber integration).
WITHHELD: the regulatory text of IACS UR I2.1 / ISO 19906 is not available
offline; the -0.3 exponent is therefore a *given model parameter* and only the
numerical inversion and its identifiability are claimed, never a regulatory
compliance statement.

References
----------
- Smith, C.S. (1977) Trans. SNAME 85, 1-18.
- Faulkner, D. (1975) J. Ship Research 19(1), 1-17.
- Paik, J.K. & Thayamballi, A.K. (2003) *Ultimate Limit State Design of
  Steel-Plated Structures*, Wiley (Ch. 5 simplified method and column curves).
- Johnson, J.B. & Ostenfeld, A. (1943) *Ingeniørvidenskabelige Skrifter* 4(2).
- Timoshenko, S.P. & Gere, J.M. (1961) *Theory of Elastic Stability*, 2nd ed.
- Simo, J.C. & Taylor, R.L. (1986) CMAME 58, 1-30.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "IsotropicHardeningMaterial",
    "uniaxial_closed_form",
    "uniaxial_incremental",
    "strain_for_uniaxial_stress",
    "j2_return_mapping",
    "j2_consistent_tangent",
    "elastic_plate_buckling_timoshenko",
    "fd_plate_buckling_stress",
    "iacs_pressure",
    "fit_pressure_area_exponent",
    "pressure_area_identifiability",
    "faulkner_plate_ultimate",
    "stiffener_section_properties",
    "euler_column_stress",
    "johnson_ostenfeld",
    "perry_robertson",
    "StiffenedPanelScantlings",
    "smith_panel_axial_collapse",
    "paik_thayamballi_simplified_ultimate",
]


# ---------------------------------------------------------------------------
# 1. material model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IsotropicHardeningMaterial:
    """Linear isotropic hardening: ``sigma_y = sigma_y0 + H * ebar_p``."""

    E: float = 206.0e9
    nu: float = 0.3
    sigma_y0: float = 313.0e6
    H: float = 1.0e9

    def __post_init__(self) -> None:
        if self.E <= 0.0:
            raise ValueError("E must be positive")
        if not (0.0 <= self.nu < 0.5):
            raise ValueError("nu must lie in [0, 0.5)")
        if self.sigma_y0 <= 0.0:
            raise ValueError("sigma_y0 must be positive")
        if self.H < 0.0:
            raise ValueError("H must be non-negative")

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def K_bulk(self) -> float:
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    @property
    def tangent_modulus(self) -> float:
        """1-D elastoplastic tangent ``E_t = E H /(E + H)``."""
        return self.E * self.H / (self.E + self.H)

    @property
    def eps_yield(self) -> float:
        return self.sigma_y0 / self.E


def uniaxial_closed_form(eps: float, mat: IsotropicHardeningMaterial) -> Tuple[float, float, float]:
    """Exact monotone uniaxial response; returns ``(sigma, E_t, ebar_p)``."""
    a = abs(float(eps))
    if a <= mat.eps_yield:
        return mat.E * float(eps), mat.E, 0.0
    eps_p = (a - mat.eps_yield) / (1.0 + mat.H / mat.E)
    sigma = mat.sigma_y0 + mat.H * eps_p
    return math.copysign(sigma, float(eps)), mat.tangent_modulus, eps_p


def strain_for_uniaxial_stress(sigma: float, mat: IsotropicHardeningMaterial) -> float:
    """Inverse of :func:`uniaxial_closed_form` (monotone branch)."""
    s = max(float(sigma), 0.0)
    if s <= mat.sigma_y0:
        return s / mat.E
    ebar = (s - mat.sigma_y0) / mat.H
    return s / mat.E + ebar


def uniaxial_incremental(eps_total: Sequence[float],
                         mat: IsotropicHardeningMaterial) -> Dict[str, object]:
    """Independent incremental integration (1-D radial return, monotone path)."""
    eps = np.asarray(eps_total, dtype=float)
    sig = np.zeros_like(eps)
    ebar_p = 0.0
    sig_old, eps_old = 0.0, 0.0
    for i, e in enumerate(eps):
        sig_trial = sig_old + mat.E * (float(e) - eps_old)
        f_trial = abs(sig_trial) - (mat.sigma_y0 + mat.H * ebar_p)
        if f_trial <= 0.0:
            sig_new = sig_trial
        else:
            d_gamma = f_trial / (mat.E + mat.H)
            sig_new = sig_trial - math.copysign(mat.E * d_gamma, sig_trial)
            ebar_p += d_gamma
        sig[i] = sig_new
        sig_old, eps_old = sig_new, float(e)
    return {"eps": eps, "sigma": sig, "ebar_p": ebar_p}


def _dev(S: np.ndarray) -> np.ndarray:
    return S - np.trace(S) / 3.0 * np.eye(3)


def _voigt_to_tensor(v: Sequence[float]) -> np.ndarray:
    vxx, vyy, vzz, vxy, vyz, vzx = [float(x) for x in v]
    return np.array([[vxx, vxy, vzx], [vxy, vyy, vyz], [vzx, vyz, vzz]], dtype=float)


def _tensor_to_voigt(T: np.ndarray) -> np.ndarray:
    return np.array([T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[1, 2], T[2, 0]], dtype=float)


def j2_return_mapping(strain: Sequence[float], eps_p: Sequence[float], ebar_p: float,
                      mat: IsotropicHardeningMaterial) -> Dict[str, object]:
    """Radial return for 3-D J2 plasticity with linear isotropic hardening.

    ``strain`` / ``eps_p`` use *tensor* shear components (no factor 2).
    """
    eps = _voigt_to_tensor(strain)
    ep = _voigt_to_tensor(eps_p)
    K, mu = mat.K_bulk, mat.mu
    e_el = eps - ep
    theta = float(np.trace(e_el))
    s_tr = 2.0 * mu * _dev(e_el)
    xi = float(np.sqrt(np.sum(s_tr * s_tr)))
    sy_old = mat.sigma_y0 + mat.H * ebar_p
    f_tr = xi - math.sqrt(2.0 / 3.0) * sy_old
    if f_tr <= 0.0:
        sigma = K * theta * np.eye(3) + s_tr
        return {
            "stress": _tensor_to_voigt(sigma),
            "eps_p": np.array(eps_p, dtype=float),
            "ebar_p": float(ebar_p),
            "dgamma": 0.0,
            "yielded": False,
            "xi": xi,
            "f_trial": f_tr,
        }
    d_gamma = f_tr / (2.0 * mu + mat.H)
    n = s_tr / xi
    s = s_tr - 2.0 * mu * d_gamma * n
    sigma = K * theta * np.eye(3) + s
    dep = 1.5 * d_gamma * n
    ebar_new = ebar_p + math.sqrt(2.0 / 3.0) * float(np.sqrt(np.sum(dep * dep)))
    return {
        "stress": _tensor_to_voigt(sigma),
        "eps_p": _tensor_to_voigt(ep + dep),
        "ebar_p": float(ebar_new),
        "dgamma": float(d_gamma),
        "yielded": True,
        "xi": xi,
        "f_trial": f_tr,
    }


def _fourth_order_identity() -> np.ndarray:
    I4 = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    I4[i, j, k, l] = 0.5 * (float(i == k) * float(j == l)
                                            + float(i == l) * float(j == k))
    return I4


def j2_consistent_tangent(strain: Sequence[float], eps_p: Sequence[float], ebar_p: float,
                          mat: IsotropicHardeningMaterial) -> np.ndarray:
    """Algorithmic tangent of :func:`j2_return_mapping` (Simo & Taylor 1986).

    ``C = K 1(x)1 + 2 mu beta I_dev - 2 mu gamma n(x)n`` with
    ``beta = 1 - 2 mu dgamma/xi``, ``gamma = 1/(1 + H/(3 mu)) - (1 - beta)``.
    Returned as a 6x6 matrix in the tensor-shear basis of the return mapping;
    verified against a central finite difference in the test suite.
    """
    eps = _voigt_to_tensor(strain)
    ep = _voigt_to_tensor(eps_p)
    K, mu = mat.K_bulk, mat.mu
    e_el = eps - ep
    s_tr = 2.0 * mu * _dev(e_el)
    xi = float(np.sqrt(np.sum(s_tr * s_tr)))
    sy_old = mat.sigma_y0 + mat.H * ebar_p
    f_tr = xi - math.sqrt(2.0 / 3.0) * sy_old
    one = np.eye(3)
    I4 = _fourth_order_identity()
    Ifour_dev = I4 - np.einsum("ij,kl->ijkl", one, one) / 3.0
    if f_tr <= 0.0:
        C = K * np.einsum("ij,kl->ijkl", one, one) + 2.0 * mu * Ifour_dev
    else:
        d_gamma = f_tr / (2.0 * mu + mat.H)
        n = s_tr / xi
        beta = 1.0 - 2.0 * mu * d_gamma / xi
        gamma = 2.0 * mu / (2.0 * mu + mat.H) - (1.0 - beta)
        C = (K * np.einsum("ij,kl->ijkl", one, one)
             + 2.0 * mu * beta * Ifour_dev
             - 2.0 * mu * gamma * np.einsum("ij,kl->ijkl", n, n))
    idx = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]
    C6 = np.zeros((6, 6))
    for p, (i, j) in enumerate(idx):
        for q, (k, l) in enumerate(idx):
            # a perturbation of a tensor shear component enters both (k,l) and
            # (l,k), so the Jacobian column picks up a factor 2
            C6[p, q] = C[i, j, k, l] * (2.0 if k != l else 1.0)
    return C6


# ---------------------------------------------------------------------------
# 2. plate elastic buckling
# ---------------------------------------------------------------------------
def elastic_plate_buckling_timoshenko(a: float, b: float, t: float,
                                      mat: IsotropicHardeningMaterial,
                                      m_max: int = 8) -> Dict[str, float]:
    """Simply supported plate under uniaxial compression (Timoshenko & Gere)."""
    D = mat.E * t ** 3 / (12.0 * (1.0 - mat.nu ** 2))
    best_m, best_k = 1, float("inf")
    for m in range(1, m_max + 1):
        k = (m * b / a + a / (m * b)) ** 2
        if k < best_k:
            best_m, best_k = m, k
    sigma_cr = best_k * math.pi ** 2 * D / (b ** 2 * t)
    return {"m": float(best_m), "k": best_k, "D": D, "sigma_cr": sigma_cr}


def _fd_buckling_matrix(nx: int, ny: int, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """(biharmonic, -d2/dx2) matrices on the interior grid.

    Simply supported edges are enforced with ``w = 0`` on the boundary and the
    **odd extension** in the ghost rows (``w_{-1} = -w_1``), which is what makes
    the 13-point biharmonic stencil consistent (zero bending moment at the
    edge).  Dropping the ghost rows instead biases the eigenvalues upward.
    """

    def idx(i: int, j: int) -> int:
        return (i - 1) + (j - 1) * (nx - 1)

    def reflect(i: int, j: int) -> Tuple[int, int, float]:
        """Odd extension: returns (reflected i, reflected j, sign); sign 0 on the
        supported edge (``w = 0``)."""
        sign = 1.0
        ii, jj = i, j
        while ii < 0 or ii > nx:
            ii = -ii if ii < 0 else 2 * nx - ii
            sign = -sign
        while jj < 0 or jj > ny:
            jj = -jj if jj < 0 else 2 * ny - jj
            sign = -sign
        if ii == 0 or ii == nx or jj == 0 or jj == ny:
            return ii, jj, 0.0
        return ii, jj, sign

    offsets = ([((0, 0), 0, 20.0)]
               + [((di, dj), 0, -8.0) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))]
               + [((di, dj), 0, 2.0) for di, dj in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
               + [((di, dj), 0, 1.0) for di, dj in ((2, 0), (-2, 0), (0, 2), (0, -2))])
    n = (nx - 1) * (ny - 1)
    A = np.zeros((n, n))
    M = np.zeros((n, n))
    for j in range(1, ny):
        for i in range(1, nx):
            p = idx(i, j)
            for (di, dj), _tag, w in offsets:
                ii, jj, c = reflect(i + di, j + dj)
                if c == 0.0:
                    continue
                A[p, idx(ii, jj)] += w * c
            M[p, p] += 2.0
            for di in (1, -1):
                ii, jj, c = reflect(i + di, j)
                if c != 0.0:
                    M[p, idx(ii, jj)] -= c
    A /= h ** 4
    M /= h ** 2
    return A, M


def _generalized_eigh_min(A: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Generalised symmetric eigenvalues via Cholesky whitening + numpy eigh.

    ``scipy.linalg.eigh(a, b)`` (LAPACK driver ``?sygv``) crashes the BLAS in
    this environment, so the standard reduction ``M = L L^T``,
    ``B = L^-1 A L^-T`` is done explicitly with numpy only.
    """
    L = np.linalg.cholesky(M)
    Linv = np.linalg.inv(L)
    B = Linv @ A @ Linv.T
    B = 0.5 * (B + B.T)
    return np.linalg.eigvalsh(B)


def fd_plate_buckling_stress(a: float, b: float, t: float,
                             mat: IsotropicHardeningMaterial,
                             h: Optional[float] = None,
                             refinements: int = 2) -> Dict[str, object]:
    """Independent FD eigenvalue solution of the plate buckling problem."""
    D = mat.E * t ** 3 / (12.0 * (1.0 - mat.nu ** 2))
    h0 = b / 12.0 if h is None else float(h)
    seq: List[Dict[str, float]] = []
    for level in range(refinements + 1):
        hh = h0 / (2 ** level)
        nx = int(round(a / hh))
        ny = int(round(b / hh))
        A, M = _fd_buckling_matrix(nx, ny, hh)
        vals = _generalized_eigh_min(D * A, M)
        pos = vals[vals > 0.0]
        N_cr = float(np.min(pos))
        seq.append({"h": hh, "nx": float(nx), "ny": float(ny),
                    "N_cr": N_cr, "sigma_cr": N_cr / t})
    sigma = [s["sigma_cr"] for s in seq]
    rich = None
    obs_rate = None
    if len(sigma) >= 3:
        s0, s1, s2 = sigma[-3], sigma[-2], sigma[-1]
        rich = s2 + (s2 - s1) / 3.0
        if s2 != s1 and s1 != s0:
            r = (s1 - s0) / (s2 - s1)
            if r > 0:
                obs_rate = math.log(r) / math.log(2.0)
    return {"sequence": seq, "sigma_cr": sigma[-1], "richardson": rich,
            "observed_order": obs_rate, "D": D}


# ---------------------------------------------------------------------------
# 3. IACS pressure-area law
# ---------------------------------------------------------------------------
def iacs_pressure(area: float, P0: float, A0: float, exponent: float = -0.3) -> float:
    """``P = P0 (A/A0)^exponent`` for ``A >= A0`` (clamped below ``A0``)."""
    if area <= 0.0 or A0 <= 0.0:
        raise ValueError("areas must be positive")
    ratio = max(area, A0) / A0
    return float(P0 * ratio ** exponent)


def fit_pressure_area_exponent(areas: Sequence[float], pressures: Sequence[float],
                               A0: Optional[float] = None) -> Dict[str, float]:
    """Log-log least squares fit of ``P = P0 (A/A0)^n``."""
    A = np.asarray(areas, dtype=float)
    P = np.asarray(pressures, dtype=float)
    if A.shape != P.shape or A.size < 3:
        raise ValueError("need at least three (area, pressure) pairs of equal shape")
    if np.any(A <= 0.0) or np.any(P <= 0.0):
        raise ValueError("areas and pressures must be positive")
    ref = float(A0) if A0 is not None else float(np.min(A))
    x = np.log(A / ref)
    y = np.log(P)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    n = int(x.size)
    se = math.sqrt(ss_res / max(n - 2, 1) / float(np.sum((x - np.mean(x)) ** 2)))
    return {"exponent": float(slope), "P0": float(math.exp(intercept)),
            "r2": r2, "exponent_se": se, "A0": ref, "n": n}


def pressure_area_identifiability(P0: float, A0: float, exponent: float = -0.3,
                                  area_ratios: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0),
                                  noise_cov: float = 0.05,
                                  seeds: Sequence[int] = (7, 11, 2026)) -> Dict[str, object]:
    """Exponent recovery from sampled pressures, clean and noise-contaminated."""
    ratios = np.asarray(area_ratios, dtype=float)
    areas = ratios * A0
    clean = np.array([iacs_pressure(a_, P0, A0, exponent) for a_ in areas])
    out: Dict[str, object] = {"clean": fit_pressure_area_exponent(areas, clean, A0=A0),
                              "noisy": []}
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        noisy = clean * np.exp(rng.normal(0.0, noise_cov, size=clean.size))
        fit = fit_pressure_area_exponent(areas, noisy, A0=A0)
        fit["seed"] = int(seed)
        out["noisy"].append(fit)
    return out


# ---------------------------------------------------------------------------
# 4. element curves and panel collapse
# ---------------------------------------------------------------------------
def faulkner_plate_ultimate(b: float, t: float, mat: IsotropicHardeningMaterial,
                            residual_stress_ratio: float = 0.15,
                            sigma_y_current: Optional[float] = None) -> Dict[str, float]:
    """Faulkner (1975) plate ultimate ``Phi = 2/beta - 1/beta^2``."""
    beta = (b / t) * math.sqrt(mat.sigma_y0 / mat.E)
    # Faulkner curve is fitted for beta >= 1; for stocky plates it must be
    # clipped at 1 (the raw expression turns negative below beta ~ 0.5)
    phi = 1.0 if beta <= 1.0 else 2.0 / beta - 1.0 / beta ** 2
    sy = mat.sigma_y0 if sigma_y_current is None else float(sigma_y_current)
    return {"beta": beta, "phi": phi,
            "sigma_u": min(phi, 1.0) * sy * (1.0 - residual_stress_ratio)}


def stiffener_section_properties(h_s: float, t_w: float, b_f: float, t_f: float) -> Dict[str, float]:
    """T-section properties about its strong (bending) axis."""
    A = h_s * t_w + b_f * t_f
    y_bar = (h_s * t_w * (h_s / 2.0) + b_f * t_f * (h_s + t_f / 2.0)) / A
    I = (t_w * h_s ** 3 / 12.0 + h_s * t_w * (h_s / 2.0 - y_bar) ** 2
         + b_f * t_f ** 3 / 12.0 + b_f * t_f * (h_s + t_f / 2.0 - y_bar) ** 2)
    c = max(y_bar, h_s + t_f - y_bar)
    return {"A": A, "y_bar": y_bar, "I": I, "c": c, "r2": I / A, "r": math.sqrt(I / A)}


def euler_column_stress(mat: IsotropicHardeningMaterial, r2: float, L: float) -> float:
    if L <= 0.0:
        raise ValueError("column length must be positive")
    return math.pi ** 2 * mat.E * r2 / L ** 2


def johnson_ostenfeld(sigma_y: float, sigma_E: float) -> float:
    """Elastoplastic column curve (smooth buckling-yield interaction)."""
    if sigma_E >= 0.5 * sigma_y:
        return sigma_y * (1.0 - sigma_y / (4.0 * sigma_E))
    return sigma_E


def perry_robertson(sigma_y: float, sigma_E: float, eta: float = 0.0) -> float:
    """Perry-Robertson column strength with imperfection parameter ``eta``."""
    b = sigma_y + sigma_E * (1.0 + eta)
    disc = max(b * b - 4.0 * sigma_y * sigma_E, 0.0)
    return 0.5 * (b - math.sqrt(disc))


@dataclass(frozen=True)
class StiffenedPanelScantlings:
    """Single-bay, single-stiffener panel (metres)."""

    a: float = 2.4
    b: float = 0.7
    t: float = 0.012
    h_s: float = 0.200
    t_w: float = 0.010
    b_f: float = 0.100
    t_f: float = 0.015

    def __post_init__(self) -> None:
        for name in ("a", "b", "t", "h_s", "t_w", "b_f", "t_f"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    @property
    def plate_area(self) -> float:
        return self.b * self.t

    @property
    def stiffener_area(self) -> float:
        return self.h_s * self.t_w + self.b_f * self.t_f

    @property
    def total_area(self) -> float:
        return self.plate_area + self.stiffener_area


def _plate_cap(mat: IsotropicHardeningMaterial, phi: float, sigma_cr: float,
               residual: float) -> Tuple[float, float]:
    """Fixed point for (strain at cap, cap) with hardening in the plateau."""
    ebar = 0.0
    e_cap, cap = 0.0, 0.0
    for _ in range(12):
        sy_cur = mat.sigma_y0 + mat.H * ebar
        cap = min(phi, 1.0) * sy_cur * (1.0 - residual)
        sig_edge_target = cap ** 2 / sigma_cr if sigma_cr < mat.sigma_y0 else cap
        e_cap = strain_for_uniaxial_stress(sig_edge_target, mat)
        _, _, ebar_new = uniaxial_closed_form(e_cap, mat)
        if abs(ebar_new - ebar) < 1e-15:
            break
        ebar = ebar_new
    return e_cap, cap


def smith_panel_axial_collapse(panel: StiffenedPanelScantlings,
                               mat: IsotropicHardeningMaterial,
                               n_steps: int = 600,
                               eps_max: float = 0.012,
                               residual_stress_ratio: float = 0.15,
                               softening_ratio: float = 0.05,
                               imperfection_ratio: float = 1.0 / 1000.0,
                               imperfection_factor_eta: Optional[float] = None) -> Dict[str, object]:
    """Smith progressive collapse under axial compression (iso-strain assembly)."""
    sec = stiffener_section_properties(panel.h_s, panel.t_w, panel.b_f, panel.t_f)
    sigma_E = euler_column_stress(mat, sec["r2"], panel.a)
    delta0 = imperfection_ratio * panel.a
    eta = (delta0 * sec["c"] / sec["r2"]) if imperfection_factor_eta is None else float(imperfection_factor_eta)
    sigma_u_stiff = perry_robertson(mat.sigma_y0, sigma_E, eta)
    plate = faulkner_plate_ultimate(panel.b, panel.t, mat, residual_stress_ratio)
    sigma_cr_plate = elastic_plate_buckling_timoshenko(panel.a, panel.b, panel.t, mat)["sigma_cr"]
    e_cap_plate, cap_plate = _plate_cap(mat, plate["phi"], sigma_cr_plate, residual_stress_ratio)
    e_cap_stiff = strain_for_uniaxial_stress(sigma_u_stiff, mat)
    soft = softening_ratio * mat.E

    eps = np.linspace(0.0, eps_max, int(n_steps) + 1)
    sig_plate = np.zeros_like(eps)
    sig_stiff = np.zeros_like(eps)
    for i, e in enumerate(eps):
        e = float(e)
        sig_edge, _, ebar_p = uniaxial_closed_form(e, mat)
        sy_cur = mat.sigma_y0 + mat.H * ebar_p
        cap_p = min(plate["phi"], 1.0) * sy_cur * (1.0 - residual_stress_ratio)
        if sigma_cr_plate >= mat.sigma_y0:
            raw = sig_edge
        elif mat.E * e <= sigma_cr_plate:
            raw = mat.E * e
        else:
            raw = math.sqrt(sigma_cr_plate * max(sig_edge, 0.0))
        sig_plate[i] = min(raw, cap_p) if e <= e_cap_plate else max(cap_p - soft * (e - e_cap_plate), 0.0)
        sig_stiff[i] = min(mat.E * e, sigma_u_stiff) if e <= e_cap_stiff else \
            max(sigma_u_stiff - soft * (e - e_cap_stiff), 0.0)

    sigma_avg = (panel.plate_area * sig_plate + panel.stiffener_area * sig_stiff) / panel.total_area
    k = int(np.argmax(sigma_avg))
    sigma_u = float(sigma_avg[k])
    tangent = np.gradient(sigma_avg, eps)
    return {
        "eps": eps,
        "sigma_panel": sigma_avg,
        "sigma_plate": sig_plate,
        "sigma_stiffener": sig_stiff,
        "sigma_u": sigma_u,
        "eps_u": float(eps[k]),
        "sigma_u_over_sigma_y": sigma_u / mat.sigma_y0,
        "post_peak_ratio": float(sigma_avg[-1] / sigma_u) if sigma_u > 0 else float("nan"),
        "elastic_slope": float(tangent[1]),
        "sigma_E_stiffener": sigma_E,
        "eta_imperfection": eta,
        "perry_robertson_stiffener": sigma_u_stiff,
        "beta_plate": plate["beta"],
        "phi_faulkner": plate["phi"],
        "sigma_cr_plate": sigma_cr_plate,
        "eps_cap_plate": e_cap_plate,
        "cap_plate": cap_plate,
        "n_steps": int(n_steps),
    }


def paik_thayamballi_simplified_ultimate(panel: StiffenedPanelScantlings,
                                         mat: IsotropicHardeningMaterial,
                                         residual_stress_ratio: float = 0.15) -> Dict[str, float]:
    """Paik & Thayamballi (2003) Ch.5 simplified method (rigid-plastic summation).

    Element ultimates: Faulkner plating (residual-stress factor included) and
    Johnson-Ostenfeld elastoplastic stiffener column.  Implemented from the
    published element curves only — no dependency on the fiber solver, so it is
    an independent method-level benchmark.
    """
    plate = faulkner_plate_ultimate(panel.b, panel.t, mat, residual_stress_ratio)
    sec = stiffener_section_properties(panel.h_s, panel.t_w, panel.b_f, panel.t_f)
    sigma_E = euler_column_stress(mat, sec["r2"], panel.a)
    sig_stiff = johnson_ostenfeld(mat.sigma_y0, sigma_E)
    sigma_u = (panel.plate_area * plate["sigma_u"] + panel.stiffener_area * sig_stiff) / panel.total_area
    return {
        "sigma_u": sigma_u,
        "sigma_u_plate": plate["sigma_u"],
        "sigma_u_stiffener": sig_stiff,
        "sigma_E_stiffener": sigma_E,
        "sigma_u_over_sigma_y": sigma_u / mat.sigma_y0,
    }