"""Elastoplastic beam verification surface: M-kappa, plastic hinges, springback.

This module builds the **section/beam level** counterpart of the J2 constitutive
work in :mod:`tensorphysics.plasticity.j2_flow` (which stops at the material
point).  Three verification layers are provided, each with an independent
closed-form or classical reference:

Layer 1 -- moment-curvature with fiber sections
    ``FiberSection`` discretises a cross-section into fibers (Gauss quadrature
    with the exact width function) and exposes both the quadrature values and
    the closed-form elastic modulus ``W_el`` / plastic modulus ``Z`` for three
    geometries (rectangle, doubly symmetric I-beam, solid circle).  Uniaxial J2
    flow theory with linear isotropic hardening is integrated fiber-by-fiber
    through ``section_response`` (radial return mapping with sub-increments),
    giving ``M(kappa)``, the algorithmic tangent ``EI_t``, the recoverable
    energy density and the accumulated plastic dissipation.

    References: the elastic slope must reproduce ``E*I`` (closed form), the
    first-yield moment ``sigma_y0*W_el`` and the fully-plastic moment
    ``sigma_y0*Z`` (closed form; shape factors 1.5 / ~1.14 / 1.698).

Layer 2 -- plastic hinge formation and location
    ``solve_beam`` is a displacement-based Euler-Bernoulli FE (Hermite cubics,
    Gauss integration along the span) with *distributed* fiber plasticity: the
    curvature at each span quadrature point drives its own fiber-section return
    mapping.  Two benchmark structures are exercised:

    * simply supported beam, eccentric point load (statically determinate);
    * propped cantilever, uniform load (one redundant, so collapse needs the
      limit-analysis mechanism with hinges at the fixed end *and* at
      ``x/L = sqrt(2)/(1+sqrt(2)) = 0.585786`` and load factor
      ``q_c L^2 / M_p = 4/t + 2/(1-t) = 11.6569``).

Layer 3 -- unloading / springback and energy accounting
    ``unload_from`` reverses the curvature path through the same return mapping
    (reverse yielding included) until the moment vanishes, giving the residual
    curvature ``kappa_res``.  ``section_energy_audit`` and the beam solver audit
    ``W_ext = U_el + W_pl`` with per-step trapezoidal external work.

Sign conventions
    ``kappa`` and ``M`` share a sign; ``M > 0`` for ``kappa > 0``.  Strain in a
    fiber is ``eps = kappa * y`` (``y`` upward from the section centroid).  Beam
    deflections are reported with the sign of the applied load.

References
----------
- Chen & Han, *Plasticity for Structural Engineers*, Springer, 1988 (fiber
  sections, shape factors, springback).
- Simo & Hughes, *Computational Inelasticity*, Springer, 1998 (return mapping).
- Neal, *The Plastic Methods of Structural Analysis*, Chapman & Hall, 1977
  (limit analysis of the propped cantilever).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "FiberSection",
    "J2UniaxialMaterial",
    "FiberState",
    "SectionResponse",
    "MomentCurvatureCurve",
    "section_response",
    "moment_curvature",
    "unload_from",
    "section_energy_audit",
    "rectangle_ideal_plastic_moment",
    "propped_cantilever_limit",
    "SolveResult",
    "solve_beam",
    "simply_supported_point_load",
    "propped_cantilever_udl",
]

# ---------------------------------------------------------------------------
# Section geometry
# ---------------------------------------------------------------------------


def _gauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(int(n))
    return x, w


@dataclass(frozen=True)
class FiberSection:
    """Doubly symmetric cross-section described by half-height intervals.

    ``dims`` is the shape specific parameter tuple; ``kind`` selects the closed
    form used by :meth:`analytic`.
    """

    kind: str
    dims: tuple[float, ...]
    n_gauss: int = 24

    # -- constructors ----------------------------------------------------
    @classmethod
    def rectangle(cls, b: float, h: float, n_gauss: int = 16) -> "FiberSection":
        return cls("rectangle", (float(b), float(h)), n_gauss)

    @classmethod
    def ibeam(
        cls, h: float, bf: float, tf: float, tw: float, n_gauss: int = 12
    ) -> "FiberSection":
        if 2.0 * tf >= h:
            raise ValueError("flange thickness exceeds half the section height")
        return cls("ibeam", (float(h), float(bf), float(tf), float(tw)), n_gauss)

    @classmethod
    def circle(cls, R: float, n_gauss: int = 40) -> "FiberSection":
        return cls("circle", (float(R),), n_gauss)

    # -- geometry --------------------------------------------------------
    @property
    def label(self) -> str:
        if self.kind == "rectangle":
            b, h = self.dims
            return f"rectangle b={b:g} h={h:g}"
        if self.kind == "ibeam":
            h, bf, tf, tw = self.dims
            return f"ibeam h={h:g} bf={bf:g} tf={tf:g} tw={tw:g}"
        (r,) = self.dims
        return f"circle R={r:g}"

    @property
    def outer_height(self) -> float:
        if self.kind == "rectangle":
            return self.dims[1]
        if self.kind == "circle":
            return 2.0 * self.dims[0]
        return self.dims[0]

    @property
    def half_height(self) -> float:
        return 0.5 * self.outer_height

    def _half_intervals(self) -> list[tuple[float, float, Any]]:
        """Intervals ``[y_lo, y_hi]`` on ``[0, c]`` with a width or callable."""
        if self.kind == "rectangle":
            b, h = self.dims
            return [(0.0, 0.5 * h, float(b))]
        if self.kind == "ibeam":
            h, bf, tf, tw = self.dims
            segments: list[tuple[float, float, Any]] = []
            if 0.5 * h - tf > 1e-12:
                segments.append((0.0, 0.5 * h - tf, float(tw)))
            segments.append((0.5 * h - tf, 0.5 * h, float(bf)))
            return segments
        (r,) = self.dims
        return [
            (0.0, r, lambda y, r=r: 2.0 * np.sqrt(np.maximum(r * r - y * y, 0.0)))
        ]

    @functools.lru_cache(maxsize=64)
    def fibers(self) -> tuple[np.ndarray, np.ndarray]:
        """Fiber ordinates ``y`` and areas ``A`` (mirror symmetric)."""
        ys: list[np.ndarray] = []
        areas: list[np.ndarray] = []
        xg, wg = _gauss(self.n_gauss)
        for y_lo, y_hi, width in self._half_intervals():
            half = 0.5 * (y_hi - y_lo)
            mid = 0.5 * (y_hi + y_lo)
            y = mid + half * xg
            wq = half * wg
            wid = width(y) if callable(width) else np.full_like(y, float(width))
            ys.append(y)
            areas.append(wid * wq)
        y_half = np.concatenate(ys)
        a_half = np.concatenate(areas)
        y = np.concatenate([-y_half[::-1], y_half])
        a = np.concatenate([a_half[::-1], a_half])
        return y, a

    def fiber_properties(self) -> dict[str, float]:
        """``A``, ``I``, ``W_el``, ``Z`` from the fiber quadrature."""
        y, a = self.fibers()
        A = float(a.sum())
        I = float((a * y * y).sum())
        c = self.half_height
        return {
            "A": A,
            "I": I,
            "W_el": I / c,
            "Z": float((a * np.abs(y)).sum()),
        }

    def analytic(self) -> dict[str, float]:
        """Closed-form ``A``, ``I``, ``W_el``, ``Z`` and shape factor."""
        if self.kind == "rectangle":
            b, h = self.dims
            A = b * h
            I = b * h**3 / 12.0
            W = b * h**2 / 6.0
            Z = b * h**2 / 4.0
        elif self.kind == "ibeam":
            h, bf, tf, tw = self.dims
            A = 2.0 * bf * tf + (h - 2.0 * tf) * tw
            I = (bf * h**3 - (bf - tw) * (h - 2.0 * tf) ** 3) / 12.0
            W = I / (0.5 * h)
            # first moment of area of the half section, doubled
            Z_flange = bf * tf * (0.5 * (h - tf))
            Z_web = tw * (0.5 * h - tf) * (0.5 * (0.5 * h - tf))
            Z = 2.0 * (Z_flange + Z_web)
        elif self.kind == "circle":
            (r,) = self.dims
            A = math.pi * r * r
            I = math.pi * r**4 / 4.0
            W = math.pi * r**3 / 4.0
            Z = 4.0 * r**3 / 3.0
        else:  # pragma: no cover - guarded by constructors
            raise ValueError(f"unknown section kind {self.kind!r}")
        return {"A": A, "I": I, "W_el": W, "Z": Z, "shape_factor": Z / W}


# ---------------------------------------------------------------------------
# Material and fiber state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class J2UniaxialMaterial:
    """Uniaxial J2 flow theory with linear isotropic hardening.

    ``sigma_y = sigma_y0 + H * eps_p`` with ``H = hardening_ratio * E``.
    """

    E: float
    sigma_y0: float
    hardening_ratio: float = 0.0

    @property
    def hardening_modulus(self) -> float:
        return self.hardening_ratio * self.E

    def yield_stress(self, eps_p: Any) -> Any:
        return self.sigma_y0 + self.hardening_modulus * eps_p

    def to_dict(self) -> dict[str, float]:
        return {
            "E": self.E,
            "sigma_y0": self.sigma_y0,
            "hardening_ratio": self.hardening_ratio,
            "hardening_modulus": self.hardening_modulus,
        }


@dataclass
class FiberState:
    """Committed constitutive state of a section (per unit length)."""

    kappa: float
    eps_p: np.ndarray
    dissipation: float

    def copy(self) -> "FiberState":
        return FiberState(self.kappa, self.eps_p.copy(), self.dissipation)

    @classmethod
    def virgin(cls, n_fibers: int) -> "FiberState":
        return cls(0.0, np.zeros(n_fibers), 0.0)


@dataclass
class SectionResponse:
    moment: float
    tangent: float
    state: FiberState
    stress: np.ndarray
    elastic_energy: float
    dissipation_increment: float


def section_response(
    section: FiberSection,
    material: J2UniaxialMaterial,
    kappa: float,
    state: FiberState,
    n_sub: int = 8,
) -> SectionResponse:
    """Integrate the section from ``state.kappa`` to ``kappa`` (radial return)."""
    y, a = section.fibers()
    E = material.E
    H = material.hardening_modulus
    E_ep = E * H / (E + H) if H > 0.0 else 0.0

    # Fast path: a trial state inside the yield surface cannot flow (1-D
    # proportional loading), so the response is purely elastic.
    sig_trial = E * (kappa * y - state.eps_p)
    if not np.any(np.abs(sig_trial) - material.yield_stress(state.eps_p) > 0.0):
        moment = float(np.sum(sig_trial * a * y))
        EI = float(np.sum(E * a * y * y))
        return SectionResponse(
            moment=moment,
            tangent=EI,
            state=FiberState(kappa, state.eps_p.copy(), state.dissipation),
            stress=sig_trial,
            elastic_energy=float(np.sum(0.5 * sig_trial * sig_trial / E * a)),
            dissipation_increment=0.0,
        )

    eps_p = state.eps_p.copy()
    diss = state.dissipation
    dk = (kappa - state.kappa) / max(int(n_sub), 1)
    for i in range(int(n_sub)):
        k_cur = state.kappa + dk * (i + 1)
        eps = k_cur * y
        sig_tr = E * (eps - eps_p)
        f = np.abs(sig_tr) - material.yield_stress(eps_p)
        active = f > 0.0
        if np.any(active):
            sgn = np.sign(sig_tr[active])
            d_ep = f[active] / (E + H) * sgn
            eps_p[active] += d_ep
            sig_active = sgn * material.yield_stress(eps_p[active])
            diss += float(np.sum(np.abs(sig_active * d_ep * a[active])))

    eps = kappa * y
    sig = E * (eps - eps_p)
    # Enforce the consistency condition on the yield surface.
    f_trial = np.abs(sig) - material.yield_stress(eps_p)
    on_surface = f_trial > 1e-10 * material.sigma_y0
    if np.any(on_surface):
        sig = np.where(
            on_surface, np.sign(sig) * material.yield_stress(eps_p), sig
        )

    moment = float(np.sum(sig * a * y))
    yielded = np.abs(sig) >= material.yield_stress(eps_p) * (1.0 - 1e-12)
    EI = float(np.sum(np.where(yielded, E_ep, E) * a * y * y))
    if EI <= 0.0:  # fully plastic: keep a floor for the Newton tangent
        EI = 1e-9 * float(np.sum(E * a * y * y))
    energy = float(np.sum(0.5 * sig * sig / E * a))
    return SectionResponse(
        moment=moment,
        tangent=EI,
        state=FiberState(kappa, eps_p, diss),
        stress=sig,
        elastic_energy=energy,
        dissipation_increment=diss - state.dissipation,
    )


def rectangle_ideal_plastic_moment(
    kappa: float, b: float, h: float, E: float, sigma_y0: float
) -> float:
    """Closed-form ``M(kappa)`` of a rectangular section, ideal plasticity."""
    c = 0.5 * h
    kappa_y = sigma_y0 / (E * c)
    k = abs(float(kappa))
    if k <= kappa_y:
        return E * b * h**3 / 12.0 * kappa
    y_e = min(c, kappa_y / k * c)
    k_sign = 1.0 if kappa >= 0 else -1.0
    M = 2.0 * b * (E * k * y_e**3 / 3.0 + sigma_y0 * (c * c - y_e * y_e) / 2.0)
    return k_sign * M

# ---------------------------------------------------------------------------
# Layer 1: moment-curvature curve
# ---------------------------------------------------------------------------


@dataclass
class MomentCurvatureCurve:
    """Sampled moment-curvature path with committed constitutive states."""

    section_label: str
    kappa: np.ndarray
    moment: np.ndarray
    tangent: np.ndarray
    dissipation: np.ndarray
    elastic_energy: np.ndarray
    states: list[FiberState]

    def slope(self, kappa_max: float | None = None) -> float:
        """Least-squares secant/regression slope of the elastic branch."""
        if kappa_max is None:
            kappa_max = 0.5 * self.kappa[1] if self.kappa.size > 1 else 0.0
        mask = np.abs(self.kappa) <= kappa_max
        if mask.sum() < 2:
            return float(self.moment[1] / self.kappa[1])
        k = self.kappa[mask]
        m = self.moment[mask]
        return float(np.dot(k, m) / np.dot(k, k))

    def tail_moment(self, fraction: float = 0.05) -> float:
        n = max(1, int(round(fraction * self.kappa.size)))
        return float(np.mean(np.abs(self.moment[-n:])))

    def as_dict(self, limit: int = 400) -> dict[str, Any]:
        step = max(1, self.kappa.size // limit)
        return {
            "section": self.section_label,
            "kappa": [float(v) for v in self.kappa[::step]],
            "moment": [float(v) for v in self.moment[::step]],
        }


def moment_curvature(
    section: FiberSection,
    material: J2UniaxialMaterial,
    kappa_max: float | None = None,
    n_steps: int = 200,
    n_sub: int = 6,
) -> MomentCurvatureCurve:
    """Monotone curvature ramp from ``0`` to ``kappa_max`` (multiple of kappa_y)."""
    y, _ = section.fibers()
    kappa_y = material.sigma_y0 / (material.E * section.half_height)
    if kappa_max is None:
        kappa_max = 20.0 * kappa_y
    kappas = np.linspace(0.0, kappa_max, n_steps + 1)
    states: list[FiberState] = []
    state = FiberState.virgin(y.size)
    mom = np.zeros(kappas.size)
    tan = np.zeros(kappas.size)
    diss = np.zeros(kappas.size)
    energy = np.zeros(kappas.size)
    states.append(state.copy())
    tan[0] = float(np.sum(material.E * (section.fiber_properties()["I"])))
    for i, k in enumerate(kappas[1:], start=1):
        resp = section_response(section, material, float(k), state, n_sub=n_sub)
        state = resp.state
        mom[i] = resp.moment
        tan[i] = resp.tangent
        diss[i] = state.dissipation
        energy[i] = resp.elastic_energy
        states.append(state.copy())
    return MomentCurvatureCurve(
        section_label=section.label,
        kappa=kappas,
        moment=mom,
        tangent=tan,
        dissipation=diss,
        elastic_energy=energy,
        states=states,
    )


# ---------------------------------------------------------------------------
# Layer 3: unloading / springback
# ---------------------------------------------------------------------------


@dataclass
class SpringbackResult:
    kappa_peak: float
    moment_peak: float
    kappa_residual: float
    ratio_residual: float
    initial_unload_slope: float
    reverse_yielded: bool
    path_kappa: np.ndarray
    path_moment: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "kappa_peak": self.kappa_peak,
            "moment_peak": self.moment_peak,
            "kappa_residual": self.kappa_residual,
            "kappa_residual_over_peak": self.ratio_residual,
            "initial_unload_slope": self.initial_unload_slope,
            "reverse_yielded": self.reverse_yielded,
        }


def unload_from(
    curve: MomentCurvatureCurve,
    section: FiberSection,
    material: J2UniaxialMaterial,
    peak_index: int = -1,
    n_steps: int = 100,
    n_sub: int = 6,
    overshoot: float = 1.5,
) -> SpringbackResult:
    """Reverse the curvature path from ``peak_index`` until the moment vanishes.

    The unloading increment is sized with the *elastic* section stiffness
    ``E*I`` (the physically correct unloading slope, not the vanishing
    elastoplastic tangent), times ``overshoot``, so the path always crosses
    ``M = 0``.  Reverse yielding is detected from the tangent dropping below
    the elastic stiffness while the moment keeps its sign.
    """
    peak_index = peak_index % curve.kappa.size
    state = curve.states[peak_index].copy()
    kappa_peak = float(curve.kappa[peak_index])
    moment_peak = float(curve.moment[peak_index])
    slope0 = float(curve.tangent[peak_index])
    EI_el = material.E * section.fiber_properties()["I"]
    d_kappa = -moment_peak / EI_el * overshoot
    path_k = [kappa_peak]
    path_m = [moment_peak]
    kappa_res = kappa_peak
    reverse_yielded = False
    prev_m = moment_peak
    for i in range(1, n_steps + 1):
        target = kappa_peak + d_kappa * i / n_steps
        resp = section_response(section, material, target, state, n_sub=n_sub)
        state = resp.state
        path_k.append(target)
        path_m.append(resp.moment)
        if prev_m != 0.0 and resp.moment * prev_m > 0.0 and resp.tangent < 0.9 * EI_el:
            reverse_yielded = True
        if prev_m > 0.0 >= resp.moment:
            frac = prev_m / (prev_m - resp.moment)
            kappa_res = float(path_k[-2] + frac * (target - path_k[-2]))
            break
        if resp.moment > 0.0 >= prev_m:
            frac = prev_m / (prev_m - resp.moment)
            kappa_res = float(path_k[-2] + frac * (target - path_k[-2]))
            break
        prev_m = resp.moment
    return SpringbackResult(
        kappa_peak=kappa_peak,
        moment_peak=moment_peak,
        kappa_residual=kappa_res,
        ratio_residual=kappa_res / kappa_peak if kappa_peak else 0.0,
        initial_unload_slope=slope0,
        reverse_yielded=reverse_yielded,
        path_kappa=np.asarray(path_k),
        path_moment=np.asarray(path_m),
    )


def section_energy_audit(curve: MomentCurvatureCurve) -> dict[str, float]:
    """``W_ext = int M dkappa`` versus stored + dissipated energy."""
    w_ext = float(np.trapezoid(curve.moment, curve.kappa))
    u_el = float(curve.elastic_energy[-1])
    w_pl = float(curve.dissipation[-1])
    resid = abs(w_ext - u_el - w_pl)
    return {
        "external_work": w_ext,
        "elastic_energy": u_el,
        "plastic_dissipation": w_pl,
        "residual": resid,
        "relative_residual": resid / max(abs(w_ext), 1e-30),
    }


# ---------------------------------------------------------------------------
# Layer 2: limit analysis reference for the propped cantilever
# ---------------------------------------------------------------------------


def propped_cantilever_limit(M_p: float, length: float) -> dict[str, float]:
    """Collapse of a propped cantilever under uniform load (limit analysis).

    Mechanism: plastic hinge at the fixed end (rotation ``theta``) and at the
    sagging hinge ``x_1`` (rotation ``theta + phi``); ``phi = theta x1/(L-x1)``.
    Internal work ``M_p(2 theta + phi)``, external work ``q theta x1 L / 2``,
    giving ``q L^2 / M_p = 4/t + 2/(1-t)`` with ``t = x1/L`` and the minimum at
    ``t = sqrt(2)/(1+sqrt(2)) = 0.585786``.
    """
    t = math.sqrt(2.0) / (1.0 + math.sqrt(2.0))
    factor = 4.0 / t + 2.0 / (1.0 - t)
    return {
        "hinge_position_ratio": t,
        "hinge_position": t * length,
        "load_factor": factor,
        "q_collapse": factor * M_p / length**2,
        # statics: R_A/(q L) = 1/2 + M_p/(q L^2) = 1/2 + 1/factor = t
        "support_reaction_ratio": t,
    }


# ---------------------------------------------------------------------------
# Layer 2: displacement-based beam FE with distributed fiber plasticity
# ---------------------------------------------------------------------------

_SUPPORT_DOFS: dict[str, tuple[tuple[int, int], ...]] = {
    # (node index, dof index) with 0 = w, 1 = theta; negative node = last node
    "simply_supported": ((0, 0), (-1, 0)),
    "propped_cantilever": ((0, 0), (0, 1), (-1, 0)),
    "cantilever": ((0, 0), (0, 1)),
}


@dataclass
class SolveResult:
    """Converged state of an incremental elastoplastic beam analysis."""

    support: str
    load_factor: float
    load_factors: np.ndarray
    control_displacement: np.ndarray
    x_gauss: np.ndarray
    curvature: np.ndarray
    moment: np.ndarray
    plastic_curvature: np.ndarray
    x_nodes: np.ndarray
    deflection: np.ndarray
    energy: dict[str, float]
    hinge: dict[str, Any]
    converged: bool
    n_newton_last: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "support": self.support,
            "load_factor": self.load_factor,
            "load_factors": [float(v) for v in self.load_factors],
            "control_displacement": [float(v) for v in self.control_displacement],
            "energy": self.energy,
            "hinge": self.hinge,
            "converged_through_final_step": self.converged,
            "n_newton_last_step": self.n_newton_last,
            "max_deflection": float(np.max(np.abs(self.deflection))),
        }


def _hermite_b(xi: np.ndarray, le: float) -> np.ndarray:
    """Second derivatives of the Hermite cubics (curvature-displacement matrix)."""
    xi = np.atleast_1d(xi)
    inv = 1.0 / le
    b = np.empty((xi.size, 4))
    b[:, 0] = (-6.0 + 12.0 * xi) * inv**2
    b[:, 1] = (6.0 * xi - 4.0) * inv
    b[:, 2] = (6.0 - 12.0 * xi) * inv**2
    b[:, 3] = (6.0 * xi - 2.0) * inv
    return b


def _hermite_n(xi: float, le: float) -> np.ndarray:
    xi2 = xi * xi
    xi3 = xi2 * xi
    return np.array(
        [
            1.0 - 3.0 * xi2 + 2.0 * xi3,
            le * (xi - 2.0 * xi2 + xi3),
            3.0 * xi2 - 2.0 * xi3,
            le * (-xi2 + xi3),
        ]
    )


def _section_response_batch(
    section: FiberSection,
    material: J2UniaxialMaterial,
    kappa: np.ndarray,
    kappa_prev: np.ndarray,
    eps_p_in: np.ndarray,
    diss_in: np.ndarray,
    n_sub: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched fiber return mapping (rows = span quadrature points).

    Same constitutive update as :func:`section_response`, vectorised over the
    whole beam so that the Newton assembly costs a single pass over the fibers.
    """
    y, a = section.fibers()
    E = material.E
    H = material.hardening_modulus
    E_ep = E * H / (E + H) if H > 0.0 else 0.0
    eps_p = eps_p_in.copy()
    diss = diss_in.copy()
    dk = (kappa - kappa_prev) / max(int(n_sub), 1)
    for i in range(int(n_sub)):
        k = kappa_prev + dk * (i + 1)
        sig_tr = E * (k[:, None] * y[None, :] - eps_p)
        sy = material.yield_stress(eps_p)
        f = np.abs(sig_tr) - sy
        active = f > 0.0
        d_ep = np.where(active, f / (E + H), 0.0) * np.sign(sig_tr)
        eps_p = eps_p + d_ep
        sig_pl = np.sign(sig_tr) * material.yield_stress(eps_p)
        diss += np.sum(np.abs(np.where(active, sig_pl, 0.0) * d_ep) * a, axis=1)
    sig = E * (kappa[:, None] * y[None, :] - eps_p)
    sy = material.yield_stress(eps_p)
    surface = np.abs(sig) - sy > 1e-10 * material.sigma_y0
    sig = np.where(surface, np.sign(sig) * sy, sig)
    moment = np.sum(sig * a * y, axis=1)
    yielded = np.abs(sig) >= sy * (1.0 - 1e-12)
    EI = np.sum(np.where(yielded, E_ep, E) * a * y * y, axis=1)
    EI = np.where(EI > 0.0, EI, 1e-9 * E * section.fiber_properties()["I"])
    energy = np.sum(0.5 * sig * sig / E * a, axis=1)
    return moment, EI, eps_p, diss, energy


def interior_hinge_centroid(
    x: np.ndarray,
    plastic_curvature: np.ndarray,
    length: float,
    window: int = 3,
    threshold: float = 0.2,
    edge_fraction: float = 0.02,
) -> float:
    """Locate the interior plastic hinge from the plastic-curvature field.

    The curvature peak of a distributed-plasticity beam is only sampled at the
    span quadrature points, so the raw argmax is mesh biased.  The centroid of
    the plastic curvature in a window around the interior peak is a
    super-convergent estimator (see the mesh study in the verification report).
    """
    x = np.asarray(x, dtype=float)
    a = np.abs(np.asarray(plastic_curvature, dtype=float))
    interior = (x > edge_fraction * length) & (x < (1.0 - edge_fraction) * length)
    if not np.any(interior) or a.size == 0:
        return float(x[int(np.argmax(a))]) if a.size else 0.0
    masked = np.where(interior, a, -1.0)
    i = int(np.argmax(masked))
    if a[i] <= 0.0:
        return float(x[i])
    lo = max(0, i - window)
    hi = min(x.size, i + window + 1)
    w = np.clip(a[lo:hi] - threshold * a[i], 0.0, None)
    total = float(w.sum())
    if total <= 0.0:  # pragma: no cover - defensive
        return float(x[i])
    return float((x[lo:hi] * w).sum() / total)


def _solve_beam_system(
    K: np.ndarray,
    rhs: np.ndarray,
    dof_scale: np.ndarray,
    fixed: Sequence[int],
    bandwidth: int = 4,
) -> np.ndarray:
    """Solve the scaled, constrained beam system.

    The Hermite beam stiffness couples only adjacent nodes, so the matrix is
    banded with a half bandwidth of 3 (padded to ``bandwidth``).  A banded
    Cholesky (LAPACK ``dpbsv`` through :func:`scipy.linalg.solveh_banded`) is
    ~150x faster than a dense solve on the production meshes; a dense fallback
    is used when SciPy is unavailable.
    """
    n = K.shape[0]
    Ks = dof_scale[:, None] * K * dof_scale[None, :]
    rhs_s = dof_scale * rhs
    if fixed:
        idx = list(fixed)
        Ks[idx, :] = 0.0
        Ks[:, idx] = 0.0
        for i in idx:
            Ks[i, i] = 1.0
        rhs_s = rhs_s.copy()
        rhs_s[idx] = 0.0
    try:
        from scipy.linalg import solveh_banded
    except Exception:  # pragma: no cover - scipy is an optional extra
        return np.linalg.solve(Ks, rhs_s)
    bw = min(bandwidth, n - 1)
    ab = np.zeros((bw + 1, n))
    for j in range(n):
        lo = max(0, j - bw)
        ab[bw + lo - j : bw + 1, j] = Ks[lo : j + 1, j]
    return solveh_banded(ab, rhs_s, lower=False)


def _assemble_beam(
    d_vec: np.ndarray,
    section: FiberSection,
    material: J2UniaxialMaterial,
    B_all: np.ndarray,
    dof_map: np.ndarray,
    dofs_qp: np.ndarray,
    w_qp: np.ndarray,
    f_ref: np.ndarray,
    lam: float,
    kappa_prev: np.ndarray,
    eps_p: np.ndarray,
    diss: np.ndarray,
    n_dof: int,
    n_gauss: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, tuple]:
    """Assemble residual, tangent and section results for one beam state."""
    d_el = d_vec[dof_map]  # (n_el, 4)
    kappa = np.einsum("qk,ek->eq", B_all, d_el).ravel()
    moment, EI, eps_p_n, diss_n, energy = _section_response_batch(
        section, material, kappa, kappa_prev, eps_p, diss, n_sub=4
    )
    r = -lam * f_ref.copy()
    Beff = (w_qp * moment)[:, None] * np.tile(B_all, (dof_map.shape[0], 1))
    np.add.at(r, dofs_qp.ravel(), Beff.ravel())
    K = np.zeros((n_dof, n_dof))
    n_el = dof_map.shape[0]
    Bq = np.tile(B_all, (n_el, 1))  # (n_qp, 4), one row per quadrature point
    wEI = w_qp * EI
    for i in range(4):
        for j in range(4):
            np.add.at(
                K,
                (dofs_qp[:, i], dofs_qp[:, j]),
                wEI * Bq[:, i] * Bq[:, j],
            )
    return (
        r,
        K,
        moment.reshape(n_el, n_gauss),
        kappa.reshape(n_el, n_gauss),
        float(np.sum(w_qp * energy)),
        float(np.sum(w_qp * diss_n)),
        (kappa, eps_p_n, diss_n),
    )


def solve_beam(
    section: FiberSection,
    material: J2UniaxialMaterial,
    length: float,
    n_elements: int,
    support: str,
    uniform_load: float = 0.0,
    nodal_loads: dict[int, float] | None = None,
    n_steps: int = 20,
    lambda_max: float = 1.0,
    n_gauss: int = 3,
    tol: float = 1e-7,
    max_iter: int = 40,
    control_node: int | None = None,
    verbose: bool = False,
) -> SolveResult:
    """Incremental elastoplastic analysis of an Euler-Bernoulli beam.

    ``uniform_load`` (per unit length) and ``nodal_loads`` describe the
    reference load at ``lambda = 1``; the analysis ramps ``lambda`` from 0 to
    ``lambda_max`` in ``n_steps`` equal increments with Newton iterations.
    Failing increments are bisected up to four times, which keeps the analysis
    stable as the structure approaches a mechanism.
    """
    if support not in _SUPPORT_DOFS:
        raise ValueError(f"unknown support {support!r}")
    nodal_loads = dict(nodal_loads or {})
    n_el = int(n_elements)
    le = length / n_el
    n_node = n_el + 1
    n_dof = 2 * n_node
    xg, wg = _gauss(n_gauss)
    xi = 0.5 * (xg + 1.0)
    wx = 0.5 * wg * le
    B_all = _hermite_b(xi, le)  # (n_gauss, 4)
    x_gauss = (np.arange(n_el)[:, None] + xi[None, :]) * le
    n_qp = n_el * n_gauss

    fixed: list[int] = []
    for node, dof in _SUPPORT_DOFS[support]:
        idx = n_node - 1 if node < 0 else node
        fixed.append(2 * idx + dof)
    fixed = sorted(set(fixed))
    free_set = set(fixed)
    free = np.array([i for i in range(n_dof) if i not in free_set], dtype=int)

    f_ref = np.zeros(n_dof)
    if uniform_load:
        ql = uniform_load * le
        f_el = np.array([0.5 * ql, ql * le / 12.0, 0.5 * ql, -ql * le / 12.0])
        for e in range(n_el):
            f_ref[2 * e : 2 * e + 4] += f_el
    for node, p in nodal_loads.items():
        f_ref[2 * int(node)] += float(p)

    y, _ = section.fibers()
    n_fib = y.size
    EI_el = material.E * section.fiber_properties()["I"]
    kappa_prev = np.zeros(n_qp)
    eps_p = np.zeros((n_qp, n_fib))
    diss = np.zeros(n_qp)

    # DOF scaling (rotations by the element length) keeps the condition number
    # of the free-free block at O(1e4) instead of O(1e10).
    dof_scale = np.ones(n_dof)
    dof_scale[1::2] = le
    f_ref_s = dof_scale * f_ref
    # element DOF maps and quadrature weights flattened over all quadrature pts
    dof_map = np.arange(n_el)[:, None] * 2 + np.array([0, 1, 2, 3])[None, :]
    dofs_qp = np.repeat(dof_map, n_gauss, axis=0)  # one row per quadrature point
    w_qp = np.tile(wx, n_el)

    def assemble(d_vec: np.ndarray, lam: float):
        return _assemble_beam(
            d_vec,
            section,
            material,
            B_all,
            dof_map,
            dofs_qp,
            w_qp,
            f_ref,
            lam,
            kappa_prev,
            eps_p,
            diss,
            n_dof,
            n_gauss,
        )

    d = np.zeros(n_dof)
    lams = [0.0]
    ctrl: list[float] = [0.0]
    w_ext = 0.0
    if control_node is None:
        control_node = n_node // 2
    control_dof = 2 * int(control_node)

    def newton(lam_target: float, d0: np.ndarray):
        d_loc = d0.copy()
        n_it = 0
        history: list[float] = []
        for it in range(1, max_iter + 1):
            n_it = it
            r, K, moments, curv, u_el, w_pl, trial = assemble(d_loc, lam_target)
            r_f = (dof_scale * r)[free]
            res_norm = float(np.linalg.norm(r_f))
            scale = max(float(np.linalg.norm(lam_target * f_ref_s[free])), 1e-30)
            if verbose:
                print(
                    f"    lam={lam_target:.5f} it={it} res={res_norm / scale:.3e}"
                    f" |r|={res_norm:.3e} scale={scale:.3e} ndof={free.size}"
                    f" |d|={np.abs(d_loc).max():.4e}",
                    flush=True,
                )
            history.append(res_norm / scale)
            stalled = (
                len(history) >= 5
                and history[-1] < 1e-4
                and history[-1] > 0.98 * min(history[:-1])
            )
            if res_norm / scale < tol or stalled:
                return True, d_loc, (moments, curv, u_el, w_pl, trial), n_it
            try:
                du = _solve_beam_system(K, -r, dof_scale, fixed)
            except Exception:  # pragma: no cover - numerical fallback
                K_ff = (dof_scale[:, None] * K * dof_scale[None, :])[
                    np.ix_(free, free)
                ]
                du_f = np.linalg.lstsq(K_ff, -r_f, rcond=None)[0]
                du = np.zeros(n_dof)
                du[free] = du_f
            alpha = 1.0
            accepted = False
            for _ in range(5):
                d_try = d_loc.copy()
                d_try[free] += alpha * dof_scale[free] * du[free]
                r_try = (dof_scale * assemble(d_try, lam_target)[0])[free]
                if np.linalg.norm(r_try) <= res_norm:
                    d_loc = d_try
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                d_loc[free] += 0.1 * dof_scale[free] * du[free]
        return False, d_loc, None, n_it

    base_dlam = lambda_max / n_steps
    converged_all = True
    n_iter_last = 0
    lam = 0.0
    guard = 0
    final = None
    while lam < lambda_max - 1e-12 and guard < 12 * n_steps:
        guard += 1
        dlam = min(base_dlam, lambda_max - lam)
        d_prev = d.copy()
        attempts = 0
        while True:
            ok, d_try, info, n_it = newton(lam + dlam, d)
            if ok or attempts >= 4:
                break
            dlam *= 0.5
            attempts += 1
        n_iter_last = n_it
        if not ok:
            converged_all = False
            break
        d = d_try
        moments, curv, u_el, w_pl, trial = info
        kappa_prev, eps_p, diss = trial
        lam_new = lam + dlam
        # trapezoidal work of the ramp: mid-step load level times the increment
        w_ext += 0.5 * (lam + lam_new) * float(f_ref @ (d - d_prev))
        lam = lam_new
        lams.append(lam)
        ctrl.append(float(d[control_dof]))

    if lams[-1] > 0.0:
        r, K, moments, curv, u_el, w_pl, _ = assemble(d, lams[-1])
    else:  # no converged increment: report the (elastic) first trial
        r, K, moments, curv, u_el, w_pl, _ = assemble(d, 0.0)

    kappa_p = curv - moments / EI_el
    energy = {
        "external_work": w_ext,
        "elastic_energy": u_el,
        "plastic_dissipation": w_pl,
        "residual": abs(w_ext - u_el - w_pl),
        "relative_residual": abs(w_ext - u_el - w_pl) / max(abs(w_ext), 1e-30),
    }

    xf = x_gauss.ravel()
    kapf = kappa_p.ravel()
    momf = moments.ravel()
    order = np.argsort(xf)
    xf, kapf, momf = xf[order], kapf[order], momf[order]
    abs_kp = np.abs(kapf)
    i_max = int(np.argmax(abs_kp))
    hinge_pos = float(xf[i_max])
    if 0 < i_max < xf.size - 1:
        xs = xf[i_max - 1 : i_max + 2]
        ys = abs_kp[i_max - 1 : i_max + 2]
        try:
            coef = np.polyfit(xs, ys, 2)
            if coef[0] < 0:
                x_vertex = float(-coef[1] / (2.0 * coef[0]))
                if xs[0] <= x_vertex <= xs[-1]:
                    hinge_pos = x_vertex
        except np.linalg.LinAlgError:  # pragma: no cover
            pass
    peak = float(abs_kp.max()) if abs_kp.size else 0.0
    kappa_y = material.sigma_y0 / (material.E * section.half_height)
    floor = max(1e-3 * kappa_y, 1e-6 * peak)
    peaks: list[tuple[float, float]] = []
    for i in range(1, xf.size - 1):
        if (
            abs_kp[i] >= abs_kp[i - 1]
            and abs_kp[i] >= abs_kp[i + 1]
            and abs_kp[i] > floor
        ):
            peaks.append((float(xf[i]), float(abs_kp[i])))
    if abs_kp.size and abs_kp[0] > floor:
        peaks.insert(0, (float(xf[0]), float(abs_kp[0])))
    interior = [p for p in peaks if 1e-3 * length < p[0] < (1.0 - 1e-3) * length]
    x_centroid = interior_hinge_centroid(xf, kapf, length)
    hinge = {
        "position": hinge_pos,
        "ratio": hinge_pos / length,
        "plastic_curvature_peak": peak,
        "moment_at_peak": float(abs(momf[i_max])),
        "peaks": peaks,
        "count": len(peaks),
        "interior_peaks": interior,
        "interior_count": len(interior),
        "interior_centroid_position": x_centroid,
        "interior_centroid_ratio": x_centroid / length,
    }

    deflection = np.array([d[2 * i] for i in range(n_node)])
    return SolveResult(
        support=support,
        load_factor=float(lams[-1]),
        load_factors=np.asarray(lams),
        control_displacement=np.asarray(ctrl),
        x_gauss=xf,
        curvature=curv.ravel()[order],
        moment=momf,
        plastic_curvature=kapf,
        x_nodes=np.arange(n_node) * le,
        deflection=deflection,
        energy=energy,
        hinge=hinge,
        converged=converged_all,
        n_newton_last=n_iter_last,
    )


# ---------------------------------------------------------------------------
# Benchmark wrappers
# ---------------------------------------------------------------------------


def simply_supported_point_load(
    section: FiberSection,
    material: J2UniaxialMaterial,
    length: float,
    load_ratio: float = 0.3,
    n_elements: int = 100,
    n_steps: int = 12,
    load_fraction: float = 0.98,
) -> SolveResult:
    """Simply supported beam, single eccentric point load (determinate).

    Reference: first-yield / plastic collapse at
    ``P_c = M_p L / (a b)`` with the hinge at ``x = a``.
    """
    n_left = int(round(load_ratio * n_elements))
    n_left = max(1, min(n_elements - 1, n_left))
    a = length * n_left / n_elements
    b = length - a
    M_p = material.sigma_y0 * section.analytic()["Z"]
    P_ref = M_p * length / (a * b)
    return solve_beam(
        section,
        material,
        length,
        n_elements,
        "simply_supported",
        nodal_loads={n_left: P_ref},
        n_steps=n_steps,
        lambda_max=load_fraction,
        control_node=n_left,
    )


def propped_cantilever_udl(
    section: FiberSection,
    material: J2UniaxialMaterial,
    length: float,
    n_elements: int = 120,
    n_steps: int = 24,
    load_fraction: float = 0.97,
) -> SolveResult:
    """Propped cantilever under uniform load (one redundant -> limit analysis)."""
    M_p = material.sigma_y0 * section.analytic()["Z"]
    limit = propped_cantilever_limit(M_p, length)
    return solve_beam(
        section,
        material,
        length,
        n_elements,
        "propped_cantilever",
        uniform_load=limit["q_collapse"],
        n_steps=n_steps,
        lambda_max=load_fraction,
    )


def elastic_deflection_point_load(
    P: float, a: float, b: float, length: float, E: float, I: float
) -> float:
    """Elastic deflection under the load of a simply supported beam."""
    return P * a * a * b * b / (3.0 * E * I * length)
