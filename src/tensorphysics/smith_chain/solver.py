"""Incremental Smith solver for sections in bending and panels in compression.

Two loading modes share one element-law machinery:

``loading="bending"``
    Classical incremental Smith method.  A curvature ``kappa`` is imposed, the
    strain field is linear across the depth, ``eps(z) = eps0 + kappa z``
    (compression positive, ``z`` from the section bottom), the instantaneous
    neutral axis follows from axial equilibrium ``N(eps0) = 0`` (vectorised
    bisection) and the bending moment is ``M = sum sigma_i A_i (z_i - z_NA)``.
    ``M_u = max_kappa |M|``, refined to the local maximum by golden-section
    search so that ``M_u`` does not depend on the curvature sampling density.

``loading="axial"``
    Panel-level Smith analysis: a stiffened panel under overall shortening is
    iso-strain between plating and stiffeners (full shear connection), hence
    ``sigma_avg(eps) = sum A_i sigma_i(eps) / sum A_i`` and
    ``sigma_u = max_eps sigma_avg``.  This is the form in which the Smith method
    is validated against closed-form panel strength formulae.

Discretisation
--------------
The only refinement parameter is the fibre count ``n_sub`` per element in the
through-thickness direction (see :func:`~.idealisation.fibre_arrays`).  All
state is recomputed from the fibre set, so a grid study is a pure ``n_sub``
refinement and can be Richardson-extrapolated (:mod:`.convergence`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..elastoplastic_collapse import VonMisesMaterial
from .idealisation import GirderSection, fibre_arrays
from .laws import (
    PlateLaw,
    StrutLaw,
    effective_plate_buckling_stress,
    plate_slenderness,
    plate_strength,
    stiffener_strength,
)

__all__ = ["ElementModel", "SmithResult", "SmithSolver", "smith_analysis"]

_EPS_MIN = 1e-16


@dataclass
class ElementModel:
    """Per-element strength model attached to the section idealisation."""

    id: str
    kind: str                      # "plate" | "stiffener"
    z: float                       # representative height (centroid) [m]
    area: float                    # [m^2]
    beta: Optional[float]          # plate slenderness (plates only)
    sigma_u: float                 # element capacity in compression [Pa]
    law: object                    # PlateLaw | StrutLaw
    mechanism: List[str]           # candidate failure mechanisms
    diagnostics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SmithResult:
    """Result of a Smith analysis."""

    loading: str
    bending: str
    M_u: float                              # ultimate moment magnitude [N m] (0 for axial)
    kappa_u: float                          # curvature at ultimate [-]
    eps0_u: float
    z_NA_u: float
    sigma_u_avg: float = 0.0                # axial mode: ultimate average stress [Pa]
    eps_u: float = 0.0                      # axial mode: strain at ultimate [-]
    curve: List[Tuple[float, float]] = field(default_factory=list)
    converged: bool = True
    n_fibers: int = 0
    elements: Dict[str, Dict[str, object]] = field(default_factory=dict)
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def M_u_MNm(self) -> float:
        return float(self.M_u / 1e6)

    def moment_curvature(self) -> List[Tuple[float, float]]:
        return list(self.curve)


class SmithSolver:
    """Incremental Smith solver on a :class:`GirderSection`."""

    def __init__(
        self,
        section: GirderSection,
        n_sub: int = 4,
        n_steps: int = 120,
        kappa_max: Optional[float] = None,
        eps_max: Optional[float] = None,
        bending: str = "sagging",
        loading: str = "bending",
        softening_modulus_ratio: float = 0.02,
        residual_stress_ratio: float = 0.2,
        peak_refine_iters: int = 60,
    ) -> None:
        if loading not in ("bending", "axial"):
            raise ValueError("loading must be 'bending' or 'axial'")
        if bending not in ("sagging", "hogging"):
            raise ValueError("bending must be 'sagging' or 'hogging'")
        if n_sub < 1 or n_steps < 8:
            raise ValueError("n_sub >= 1 and n_steps >= 8 required")
        self.section = section
        self.mat = section.material
        self.n_sub = int(n_sub)
        self.n_steps = int(n_steps)
        self.bending = bending
        self.loading = loading
        self.soft = float(softening_modulus_ratio)
        self.resid = float(residual_stress_ratio)
        self.peak_iters = int(peak_refine_iters)
        eps_y = self.mat.sigma_y0 / self.mat.E
        self.kappa_max = float(kappa_max) if kappa_max is not None else 20.0 * eps_y / section.depth
        self.eps_max = float(eps_max) if eps_max is not None else 20.0 * eps_y
        self.models: Dict[str, ElementModel] = self._build_models()
        self._build_fibers()

    # -- model construction ------------------------------------------------
    def _build_models(self) -> Dict[str, ElementModel]:
        mat = self.mat
        models: Dict[str, ElementModel] = {}
        plate_beta: Dict[str, float] = {}
        for p in self.section.plates:
            beta = plate_slenderness(p.spacing_b, p.thickness, mat)
            sigma_u = plate_strength(beta, mat)
            sigma_cr = effective_plate_buckling_stress(
                p.spacing_b, p.thickness, p.span_a, mat, p.k_factor
            )
            plate_beta[p.id] = beta
            mechanism = ["plate_buckling"] if beta > 1.0 else ["yielding"]
            models[p.id] = ElementModel(
                id=p.id, kind="plate", z=p.z, area=p.area, beta=beta,
                sigma_u=sigma_u,
                law=PlateLaw(mat, sigma_cr, sigma_u, self.soft, self.resid),
                mechanism=mechanism,
                diagnostics={
                    "beta": beta,
                    "phi": sigma_u / mat.sigma_y0,
                    "sigma_cr_eff_MPa": sigma_cr / 1e6,
                    "sigma_u_MPa": sigma_u / 1e6,
                    "width_m": p.width,
                    "thickness_mm": p.thickness * 1e3,
                    "spacing_b_m": p.spacing_b,
                    "span_a_m": p.span_a,
                },
            )
        for s in self.section.stiffeners:
            beta = plate_beta[s.plate_id]
            strengths = stiffener_strength(
                h=s.h, t_w=s.t_w, b_f=s.b_f, t_f=s.t_f, span=s.span_a,
                spacing_b=s.spacing_b, plate_t=s.plate_t or self._plate_t(s.plate_id),
                beta=beta, material=mat,
            )
            cap = strengths["sigma_cap"]
            mechanism: List[str] = []
            if strengths["sigma_col"] <= cap + 1e-9:
                mechanism.append("stiffener_column_buckling")
            if strengths["sigma_trip"] <= cap + 1e-9:
                mechanism.append("stiffener_tripping")
            if cap >= mat.sigma_y0 - 1e-9:
                mechanism.append("yielding")
            z_rep = s.z_attach + 0.5 * s.h
            models[s.id] = ElementModel(
                id=s.id, kind="stiffener", z=z_rep, area=s.area, beta=beta,
                sigma_u=cap, law=StrutLaw(mat, cap), mechanism=mechanism,
                diagnostics={
                    "beta": beta,
                    "sigma_cap_MPa": cap / 1e6,
                    "sigma_col_MPa": strengths["sigma_col"] / 1e6,
                    "sigma_trip_MPa": strengths["sigma_trip"] / 1e6,
                    "lambda_column": strengths["lambda_column"],
                    "h_m": s.h,
                    "n_long": float(s.n),
                },
            )
        return models

    def _plate_t(self, plate_id: str) -> float:
        for p in self.section.plates:
            if p.id == plate_id:
                return p.thickness
        raise KeyError(plate_id)

    # -- fibre bookkeeping -------------------------------------------------
    def _build_fibers(self) -> None:
        z, a = fibre_arrays(self.section, n_sub=self.n_sub)
        # map every fibre to its element by re-walking the idealisation
        elem_of_fibre: List[str] = []
        for p in self.section.plates:
            elem_of_fibre += [p.id] * self.n_sub
        for s in self.section.stiffeners:
            elem_of_fibre += [s.id] * self.n_sub
            if s.is_tee:
                elem_of_fibre += [s.id]
        if len(elem_of_fibre) != len(z):
            raise RuntimeError("fibre/element bookkeeping mismatch")
        order = sorted(range(len(z)), key=lambda i: elem_of_fibre[i])
        self.z = z[order]
        self.area = a[order]
        labels = [elem_of_fibre[i] for i in order]
        self.slices: List[Tuple[str, int, int]] = []
        start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[start]:
                self.slices.append((labels[start], start, i))
                start = i
        self.n_fibers = len(self.z)
        self.A_total = float(np.sum(self.area))

    # -- equilibrium -------------------------------------------------------
    def _axial_force(self, eps0: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        eps = eps0[:, None] + kappa[:, None] * self.z[None, :]
        N = np.zeros(eps.shape[0])
        for elem_id, i0, i1 in self.slices:
            sigma = self.models[elem_id].law.stress(eps[:, i0:i1])
            N += np.asarray(sigma) @ self.area[i0:i1]
        return N

    def _moment(self, eps0: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        eps = eps0[:, None] + kappa[:, None] * self.z[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            z_na = np.where(np.abs(kappa) > 0.0, -eps0 / np.where(kappa == 0.0, 1.0, kappa), 0.0)
        lever = self.z[None, :] - z_na[:, None]
        M = np.zeros(eps.shape[0])
        for elem_id, i0, i1 in self.slices:
            sigma = np.asarray(self.models[elem_id].law.stress(eps[:, i0:i1]))
            M += np.sum(sigma * self.area[None, i0:i1] * lever[:, i0:i1], axis=1)
        return M

    def solve_neutral_axis(self, kappa: float, tol: float = 1e-14) -> float:
        """Root of ``N(eps0) = 0`` at curvature ``kappa`` (vectorised bisection)."""
        eps_y = self.mat.sigma_y0 / self.mat.E
        lo = np.array([-40.0 * eps_y])
        hi = np.array([40.0 * eps_y])
        k = np.array([kappa], dtype=float)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            N = self._axial_force(mid, k)
            pos = N > 0.0
            hi = np.where(pos, mid, hi)
            lo = np.where(pos, lo, mid)
            if np.max(np.abs(hi - lo)) < tol:
                break
        return float(0.5 * (lo[0] + hi[0]))

    def moment_at(self, kappa: float) -> Tuple[float, float, float]:
        """``(M, eps0, z_NA)`` at a prescribed curvature."""
        eps0 = self.solve_neutral_axis(kappa)
        M = float(self._moment(np.array([eps0]), np.array([kappa]))[0])
        return M, eps0, float(-eps0 / kappa if kappa != 0.0 else 0.0)

    # -- sweeps ------------------------------------------------------------
    def _curvatures(self) -> np.ndarray:
        sign = 1.0 if self.bending == "sagging" else -1.0
        return sign * np.linspace(self.kappa_max / self.n_steps, self.kappa_max, self.n_steps)

    def _sweep(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        kappas = self._curvatures()
        eps_y = self.mat.sigma_y0 / self.mat.E
        lo = np.full(kappas.shape, -40.0 * eps_y)
        hi = np.full(kappas.shape, 40.0 * eps_y)
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            N = self._axial_force(mid, kappas)
            pos = N > 0.0
            hi = np.where(pos, mid, hi)
            lo = np.where(pos, lo, mid)
            if np.max(np.abs(hi - lo)) < 1e-14:
                break
        eps0 = 0.5 * (lo + hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_na = -eps0 / kappas
        M = np.zeros(kappas.shape)
        eps = eps0[:, None] + kappas[:, None] * self.z[None, :]
        lever = self.z[None, :] - z_na[:, None]
        for elem_id, i0, i1 in self.slices:
            sigma = np.asarray(self.models[elem_id].law.stress(eps[:, i0:i1]))
            M += np.sum(sigma * self.area[None, i0:i1] * lever[:, i0:i1], axis=1)
        return kappas, eps0, z_na, M

    def _refine_peak(self, k_lo: float, k_hi: float) -> Tuple[float, float, float, float]:
        """Golden-section maximisation of ``|M(kappa)|`` in ``[k_lo, k_hi]``."""
        g = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = float(k_lo), float(k_hi)
        c = b - g * (b - a)
        d = a + g * (b - a)
        Mc, _, _ = self.moment_at(c)
        Md, _, _ = self.moment_at(d)
        for _ in range(self.peak_iters):
            if abs(Mc) > abs(Md):
                b, d, Md = d, c, Mc
                c = b - g * (b - a)
                Mc, _, _ = self.moment_at(c)
            else:
                a, c, Mc = c, d, Md
                d = a + g * (b - a)
                Md, _, _ = self.moment_at(d)
        k = 0.5 * (a + b)
        M, eps0, z_na = self.moment_at(k)
        return M, k, eps0, z_na

    def _element_response(self, eps0: float, kappa: float, z_na: float) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for elem_id, i0, i1 in self.slices:
            model = self.models[elem_id]
            zz = self.z[i0:i1]
            aa = self.area[i0:i1]
            eps = eps0 + kappa * zz
            sigma = np.asarray(model.law.stress(eps))
            z_rep = float(np.sum(aa * zz) / np.sum(aa))
            eps_rep = eps0 + kappa * z_rep
            sigma_rep = float(np.asarray(model.law.stress(np.array([eps_rep])))[0])
            contrib = float(np.sum(sigma * aa * (zz - z_na)))
            # utilisation is measured against the capacity that governs the sign
            # of the stress: compression capacity (plate/strut law) in
            # compression, tensile yield in tension.
            cap_here = model.sigma_u if eps_rep > 0.0 else self.mat.sigma_y0
            out[elem_id] = {
                "kind": model.kind,
                "z_m": z_rep,
                "area_m2": float(np.sum(aa)),
                "strain": float(eps_rep),
                "state": "compression" if eps_rep > 0.0 else "tension",
                "stress_MPa": sigma_rep / 1e6,
                "capacity_MPa": cap_here / 1e6,
                "compression_capacity_MPa": model.sigma_u / 1e6,
                "utilization": float(abs(sigma_rep) / cap_here) if cap_here > 0 else 0.0,
                "moment_contribution_MNm": contrib / 1e6,
                "beta": model.beta,
                "mechanisms": list(model.mechanism),
                "element_diagnostics": dict(model.diagnostics),
            }
        return out

    # -- public entry points ----------------------------------------------
    def solve(self) -> SmithResult:
        if self.loading == "axial":
            return self._solve_axial()
        kappas, eps0s, z_nas, Ms = self._sweep()
        mag = np.abs(Ms)
        i = int(np.argmax(mag))
        if self.peak_iters > 0:
            lo = kappas[max(i - 1, 0)]
            hi = kappas[min(i + 1, len(kappas) - 1)]
            M_u, kappa_u, eps0_u, z_na_u = self._refine_peak(lo, hi)
        else:
            # pure grid maximum: used for grid-convergence studies, where the
            # curvature sampling must stay identical across refinement levels so
            # that the differences measure the *element subdivision* error only.
            M_u, kappa_u = float(Ms[i]), float(kappas[i])
            eps0_u, z_na_u = float(eps0s[i]), float(z_nas[i])
        elems = self._element_response(eps0_u, kappa_u, z_na_u)
        curve = list(zip(kappas.tolist(), Ms.tolist()))
        converged = bool(np.all(np.isfinite(Ms)))
        return SmithResult(
            loading="bending",
            bending=self.bending,
            M_u=float(abs(M_u)),
            kappa_u=float(kappa_u),
            eps0_u=float(eps0_u),
            z_NA_u=float(z_na_u),
            curve=curve,
            converged=converged,
            n_fibers=self.n_fibers,
            elements=elems,
            diagnostics={
                "n_sub": self.n_sub,
                "n_steps": self.n_steps,
                "kappa_max": self.kappa_max,
                "z_NA_elastic_m": self.section.z_neutral_elastic,
                "z_NA_shift_m": float(z_na_u - self.section.z_neutral_elastic),
                "eps_flange_ultimate": float(
                    eps0_u + kappa_u * (self.section.depth if self.bending == "sagging" else 0.0)
                ),
                "M_first_yield_MNm": self.section.first_yield_moment() / 1e6,
                "M_first_yield_compression_side_MNm": float(
                    self.mat.sigma_y0
                    * self.section.section_modulus(
                        "deck" if self.bending == "sagging" else "bottom")
                    / 1e6
                ),
                "M_plastic_MNm": self.section.plastic_moment() / 1e6,
                "M_u_over_M_yield": float(
                    abs(M_u) / max(self.section.first_yield_moment(), _EPS_MIN)
                ),
                "M_u_over_M_yield_compression_side": float(
                    abs(M_u)
                    / max(
                        self.mat.sigma_y0
                        * self.section.section_modulus(
                            "deck" if self.bending == "sagging" else "bottom"),
                        _EPS_MIN,
                    )
                ),
                "M_u_over_M_plastic": float(
                    abs(M_u) / max(self.section.plastic_moment(), _EPS_MIN)
                ),
                "A_total_m2": self.A_total,
                "section_kind": self.section.kind,
            },
        )

    def _solve_axial(self) -> SmithResult:
        eps_y = self.mat.sigma_y0 / self.mat.E
        grid = np.linspace(0.0, self.eps_max, self.n_steps)
        avg = np.zeros(grid.shape)
        eps_mat = np.repeat(grid[:, None], len(self.z), axis=1)
        for elem_id, i0, i1 in self.slices:
            sigma = np.asarray(self.models[elem_id].law.stress(eps_mat[:, i0:i1]))
            avg += sigma @ self.area[i0:i1]
        avg /= self.A_total
        i = int(np.argmax(avg))
        lo = grid[max(i - 1, 0)]
        hi = grid[min(i + 1, len(grid) - 1)]
        g = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = float(lo), float(hi)
        c = b - g * (b - a)
        d = a + g * (b - a)
        fc, fd = self._avg_stress(c), self._avg_stress(d)
        for _ in range(self.peak_iters):
            if fc > fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = self._avg_stress(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = self._avg_stress(d)
        eps_u = 0.5 * (a + b)
        sigma_u = self._avg_stress(eps_u)
        elems = {}
        for elem_id, i0, i1 in self.slices:
            model = self.models[elem_id]
            zz = self.z[i0:i1]
            aa = self.area[i0:i1]
            sigma = np.asarray(model.law.stress(np.full(zz.shape, eps_u)))
            sigma_rep = float(np.asarray(model.law.stress(np.array([eps_u])))[0])
            elems[elem_id] = {
                "kind": model.kind,
                "z_m": float(np.sum(aa * zz) / np.sum(aa)),
                "area_m2": float(np.sum(aa)),
                "strain": float(eps_u),
                "state": "compression",
                "stress_MPa": sigma_rep / 1e6,
                "capacity_MPa": model.sigma_u / 1e6,
                "compression_capacity_MPa": model.sigma_u / 1e6,
                "utilization": float(abs(sigma_rep) / model.sigma_u) if model.sigma_u > 0 else 0.0,
                "mean_stress_MPa": float(np.sum(sigma * aa) / np.sum(aa)) / 1e6,
                "beta": model.beta,
                "mechanisms": list(model.mechanism),
                "element_diagnostics": dict(model.diagnostics),
            }
        return SmithResult(
            loading="axial",
            bending=self.bending,
            M_u=0.0,
            kappa_u=0.0,
            eps0_u=float(eps_u),
            z_NA_u=0.0,
            sigma_u_avg=float(sigma_u),
            eps_u=float(eps_u),
            curve=list(zip(grid.tolist(), avg.tolist())),
            converged=bool(np.all(np.isfinite(avg))),
            n_fibers=self.n_fibers,
            elements=elems,
            diagnostics={
                "n_sub": self.n_sub,
                "n_steps": self.n_steps,
                "eps_y": eps_y,
                "sigma_u_MPa": sigma_u / 1e6,
                "sigma_u_over_sigma_y": float(sigma_u / self.mat.sigma_y0),
                "sigma_peak_strain_ratio": float(eps_u / eps_y),
                "A_total_m2": self.A_total,
                "section_kind": self.section.kind,
            },
        )

    def _avg_stress(self, eps: float) -> float:
        total = 0.0
        for elem_id, i0, i1 in self.slices:
            sigma = np.asarray(self.models[elem_id].law.stress(np.full(i1 - i0, eps)))
            total += float(sigma @ self.area[i0:i1])
        return total / self.A_total


def smith_analysis(section: GirderSection, **kwargs) -> SmithResult:
    """Convenience wrapper: run the Smith solver with keyword defaults."""
    return SmithSolver(section, **kwargs).solve()