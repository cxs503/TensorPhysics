"""Section-level incremental Smith method for the residual ultimate moment.

Algorithm (classical Smith progressive-collapse, fiber form)
------------------------------------------------------------
1. The damaged section is discretised into fibers: one fiber per plate strip
   (thin plate, placed at its centroid) and ``n_web_fibers`` fibers per
   stiffener group, each carrying ``A/n``.
2. A curvature :math:`\\kappa` is imposed (sagging positive, compression at the
   deck).  The strain field is linear, ``eps(z) = eps0 + kappa z`` with
   compression positive.
3. The neutral axis is found from axial equilibrium ``N(eps0) = 0`` by bisection
   (``N`` is monotone increasing in ``eps0``), giving the instantaneous neutral
   axis ``z_NA = -eps0 / kappa``.
4. The bending moment follows from the fiber stresses,
   ``M(kappa) = sum sigma_i A_i (z_i - z_NA)``.
5. ``M_u = max_kappa M(kappa)`` - the peak of the moment-curvature curve, which
   is where the compressive flange (deck plating + longitudinals) has shed
   enough load through plate buckling / stiffener tripping / yielding.

Element laws are the ones of :mod:`tensorphysics.hull_redundancy.elements`,
which re-use the ``elastoplastic_collapse`` plate and J2 curves, so this solver
is an *extension* of the verified panel-level kernel rather than a re-write.

Verification anchors
--------------------
- Linear-elastic limit: for small curvature the solver reproduces
  ``M = E I_eff kappa`` with ``I_eff`` from the elastic section properties.
- First-yield check: the deck stress at yield must coincide with
  ``M = sigma_y S_deck`` within the element discretisation error.
- Monotone: ``M_u`` must not increase when a component is degraded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..elastoplastic_collapse import VonMisesMaterial
from .damage import DamageSpec, DamagedSection, apply_damage
from .elements import plate_element_stress, strut_element_stress
from .section import HullSection

__all__ = [
    "Fiber",
    "SmithResult",
    "SmithSectionSolver",
    "ultimate_moment",
]


@dataclass(frozen=True)
class Fiber:
    z: float
    area: float
    kind: str            # "plate" | "stiffener"
    sigma_cr: Optional[float] = None
    cap: Optional[float] = None


@dataclass
class SmithResult:
    M_u: float
    kappa_u: float
    eps_u_compression_flange: float
    epso_u: float
    z_NA_u: float
    moment_curvature: List[Tuple[float, float]]
    converged: bool
    n_fibers: int
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def M_u_MNm(self) -> float:
        return float(self.M_u / 1e6)


class SmithSectionSolver:
    """Incremental-curvature Smith solver for a damaged hull-girder section."""

    def __init__(
        self,
        damaged: DamagedSection,
        n_steps: int = 80,
        kappa_max: float = 6.0e-4,
        n_web_fibers: int = 3,
        softening_modulus_ratio: float = 0.02,
        residual_stress_ratio: float = 0.2,
        kappa_min: Optional[float] = None,
        bending: str = "sagging",
    ) -> None:
        if n_steps < 4:
            raise ValueError("n_steps must be >= 4")
        if kappa_max <= 0.0:
            raise ValueError("kappa_max must be positive")
        if n_web_fibers < 1:
            raise ValueError("n_web_fibers must be >= 1")
        self.damaged = damaged
        self.section = damaged.section
        self.mat: VonMisesMaterial = damaged.material
        self.n_steps = int(n_steps)
        self.kappa_max = float(kappa_max)
        self.kappa_min = float(kappa_min if kappa_min is not None else kappa_max / n_steps)
        if bending not in ("sagging", "hogging"):
            raise ValueError("bending must be 'sagging' or 'hogging'")
        self.bending = bending
        self.sign = 1.0 if bending == "sagging" else -1.0
        self.n_web = int(n_web_fibers)
        self.soft = float(softening_modulus_ratio)
        self.resid = float(residual_stress_ratio)
        self.fibers = self._build_fibers()

    # -- discretisation ----------------------------------------------------
    def _build_fibers(self) -> Tuple[Fiber, ...]:
        fibers: List[Fiber] = []
        for p in self.damaged.plates:
            if p.removed or p.area <= 0.0:
                continue
            fibers.append(Fiber(z=p.z, area=p.area, kind="plate", sigma_cr=p.sigma_cr))
        for s in self.damaged.stiffeners:
            if s.removed or s.area <= 0.0:
                continue
            n = self.n_web
            dz = s.h / n
            for i in range(n):
                z = s.z - s.h / 2.0 + (i + 0.5) * dz
                fibers.append(
                    Fiber(z=z, area=s.area / n, kind="stiffener", cap=s.cap)
                )
        if not fibers:
            raise ValueError("damaged section has no load-carrying element left")
        return tuple(fibers)

    @property
    def area_total(self) -> float:
        return float(sum(f.area for f in self.fibers))

    # -- element / section response ---------------------------------------
    def fiber_stress(self, f: Fiber, eps: float) -> float:
        if f.kind == "plate":
            return plate_element_stress(
                eps, self.mat, f.sigma_cr,
                softening_modulus_ratio=self.soft,
                residual_stress_ratio=self.resid,
            )
        return strut_element_stress(eps, self.mat, cap=f.cap)

    def axial_force(self, eps0: float, kappa: float) -> float:
        N = 0.0
        for f in self.fibers:
            N += self.fiber_stress(f, eps0 + kappa * f.z) * f.area
        return N

    def moment(self, eps0: float, kappa: float) -> Tuple[float, float]:
        """Moment about the instantaneous neutral axis and that axis position."""
        z_na = -eps0 / kappa
        M = 0.0
        for f in self.fibers:
            sigma = self.fiber_stress(f, eps0 + kappa * f.z)
            M += sigma * f.area * (f.z - z_na)
        return M, z_na

    # -- equilibrium -------------------------------------------------------
    def solve_neutral_axis(
        self, kappa: float, eps_lo: float = -0.05, eps_hi: float = 0.05,
        tol: float = 1e-12, max_iter: int = 60
    ) -> Optional[float]:
        """Root of ``N(eps0) = 0`` for a prescribed curvature (bisection)."""
        f_lo = self.axial_force(eps_lo, kappa)
        f_hi = self.axial_force(eps_hi, kappa)
        if f_lo > 0.0 or f_hi < 0.0:
            return None
        for _ in range(max_iter):
            mid = 0.5 * (eps_lo + eps_hi)
            f_mid = self.axial_force(mid, kappa)
            if abs(f_hi - f_lo) < tol or abs(f_mid) < 1e-9:
                return float(mid)
            if f_mid > 0.0:
                eps_hi, f_hi = mid, f_mid
            else:
                eps_lo, f_lo = mid, f_mid
        return float(0.5 * (eps_lo + eps_hi))

    # -- main sweep --------------------------------------------------------
    def solve(self) -> SmithResult:
        kappas = self.sign * np.linspace(
            self.kappa_min, self.kappa_max, self.n_steps)
        curve: List[Tuple[float, float]] = []
        M_u = -1.0
        kappa_u = 0.0
        eps0_u = 0.0
        z_na_u = 0.0
        converged = True
        for kappa in kappas:
            eps0 = self.solve_neutral_axis(float(kappa))
            if eps0 is None:
                converged = False
                break
            M, z_na = self.moment(eps0, float(kappa))
            M_eff = self.sign * M          # positive magnitude at the loaded flange
            curve.append((float(kappa), float(M)))
            if M_eff > M_u:
                M_u = float(M_eff)
                kappa_u = float(kappa)
                eps0_u = float(eps0)
                z_na_u = float(z_na)
        if not curve:
            raise RuntimeError("Smith sweep produced no equilibrium curvature")
        # strain at the loaded (compression) extreme fibre
        z_load = self.section.depth if self.sign > 0 else 0.0
        eps_flange = eps0_u + kappa_u * z_load
        return SmithResult(
            M_u=M_u,
            kappa_u=kappa_u,
            eps_u_compression_flange=float(eps_flange),
            epso_u=eps0_u,
            z_NA_u=z_na_u,
            moment_curvature=curve,
            converged=converged,
            n_fibers=len(self.fibers),
            diagnostics={
                "component_id": self.damaged.spec.component_id,
                "mode": self.damaged.spec.mode,
                "area_total_m2": self.area_total,
                "n_steps": self.n_steps,
                "kappa_max": self.kappa_max,
                "n_web_fibers": self.n_web,
                "bending": self.bending,
                "z_NA_at_ultimate_m": z_na_u,
                "kappa_u_signed": kappa_u,
                "eps_compression_flange_at_ultimate": float(eps_flange),
                "sigma_y_MPa": self.mat.sigma_y0 / 1e6,
            },
        )

    # -- elastic consistency ----------------------------------------------
    def elastic_slope(self, kappa: float = 1e-9) -> float:
        """Numerical ``dM/dkappa`` at zero curvature (should equal E*I_eff)."""
        h = self.sign * max(kappa, 1e-9)
        eps0 = self.solve_neutral_axis(h)
        if eps0 is None:
            raise RuntimeError("elastic regime not found")
        M, _ = self.moment(eps0, h)
        return float(self.sign * M / h)


def ultimate_moment(
    section: HullSection,
    spec: Optional[DamageSpec] = None,
    **solver_kwargs,
) -> Tuple[float, SmithResult]:
    """Convenience wrapper: ultimate moment of ``section`` under ``spec``."""
    dmg = apply_damage(section, spec) if spec is not None else apply_damage(
        section, DamageSpec(section.component_ids()[0], "none")
    )
    solver = SmithSectionSolver(dmg, **solver_kwargs)
    res = solver.solve()
    return res.M_u, res