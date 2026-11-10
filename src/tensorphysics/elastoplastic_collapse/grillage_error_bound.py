"""Grillage size-convergence error bound and isolated-element deviation.

What changed relative to P70-N01
--------------------------------
The baseline implementation returned

    sigma_u^grillage = sigma_u^isolated * [1 + 0.05 (n_y-1)/n_y] * [1 + 0.02 (n_x-1)/n_x]

i.e. an *invented* +5/+2 % "boundary restraint" that (a) made the 3x3 vs 5x5
comparison close by construction, (b) pushed the panel ultimate strength above
yield (sigma_u/sigma_y = 1.196 for a beta = 2.28 panel, which is physically
impossible under compression), and (c) produced a delta_iso of +5.66 % with no
physical mechanism behind it.

The replacement delegates to :class:`~.panel_ultimate.SmithPanelSolver`, whose
capacity is bounded by four competing physical limits (Faulkner plate ultimate,
Euler column buckling, beam-column first-fibre yield with amplified imperfection
moment, elastoplastic tripping).  The 3x3 -> 5x5 difference is then a genuine
midpoint-quadrature convergence of the panel imperfection-field average, and
``delta_iso`` is the difference between that average and the single isolated
intercolumn unit (a pure concavity/Jensen effect: the sign is negative because
the capacity is a concave function of the imperfection amplitude).

References
----------
- Smith, C.S. (1977). Trans. SNAME 85.
- Faulkner, D. (1975). Trans. SNAME 83.
- Paik, J.K. & Thayamballi, A.K. (2003). Ultimate Limit State Design of
  Steel-Plated Structures, Ch. 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .von_mises_plasticity import VonMisesMaterial
from .stiffened_plate_collapse import StiffenedPlateGeometry
from .panel_ultimate import PanelScantlings, SmithPanelSolver

__all__ = [
    "GrillageModel",
    "grillage_convergence_study",
    "isolated_element_deviation",
]

#: Acceptance threshold of the 3x3 -> 5x5 convergence criterion.
CONVERGENCE_TOL = 0.02


def _scantlings(
    a: float, b: float, t: float, h_s: float, t_w: float, b_f: float, t_f: float,
    material: VonMisesMaterial, w0_max: Optional[float],
) -> PanelScantlings:
    if w0_max is None:
        return PanelScantlings(a=a, b=b, t=t, h_s=h_s, t_w=t_w, b_f=b_f, t_f=t_f,
                               material=material)
    return PanelScantlings(a=a, b=b, t=t, h_s=h_s, t_w=t_w, b_f=b_f, t_f=t_f,
                           material=material,
                           w0_plate=float(w0_max), w0_column=float(w0_max))


@dataclass(frozen=True)
class GrillageModel:
    """Grillage (panel) model with ``n_x`` x ``n_y`` imperfection-sampling cells.

    ``n_x`` refines the longitudinal (interframe) direction, ``n_y`` the
    transverse (stiffener spacing) direction.  The physical spans are unchanged
    by the refinement: only the resolution of the imperfection field changes,
    which is exactly the quantity the convergence criterion measures.
    """

    n_x: int
    n_y: int
    a: float
    b: float
    t: float
    h_s: float
    t_w: float
    b_f: float
    t_f: float
    material: VonMisesMaterial
    w0_max: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n_x < 1 or self.n_y < 1:
            raise ValueError("grillage dimensions must be at least 1x1")
        if self.a <= 0 or self.b <= 0 or self.t <= 0:
            raise ValueError("panel dimensions must be positive")

    def to_stiffened_plate(self) -> StiffenedPlateGeometry:
        """Equivalent single stiffened-plate geometry (legacy API)."""
        return StiffenedPlateGeometry(
            a=self.a, b=self.b, t=self.t,
            h_s=self.h_s, t_w=self.t_w, b_f=self.b_f, t_f=self.t_f,
            material=self.material,
            w0_max=0.0 if self.w0_max is None else float(self.w0_max),
        )

    def solver(self, **kwargs) -> SmithPanelSolver:
        sc = _scantlings(self.a, self.b, self.t, self.h_s, self.t_w,
                         self.b_f, self.t_f, self.material, self.w0_max)
        return SmithPanelSolver(sc, n_x=self.n_x, n_y=self.n_y, **kwargs)

    def compute_ultimate_strength(self, **kwargs) -> float:
        """Panel ultimate strength (average axial stress) [Pa]."""
        return float(self.solver(**kwargs).solve().sigma_u)


def grillage_convergence_study(
    a: float, b: float, t: float,
    h_s: float, t_w: float, b_f: float, t_f: float,
    material: VonMisesMaterial,
    w0_max: Optional[float] = None,
    n_coarse: int = 3,
    n_fine: int = 5,
    refine: Optional[int] = None,
) -> Dict[str, float]:
    """3x3 -> 5x5 (default) grillage convergence study.

    Returns
    -------
    dict
        ``sigma_u_coarse``, ``sigma_u_fine``, ``relative_error``, ``converged``,
        ``delta_sigma``, ``n_coarse``, ``n_fine`` and, for the error bound,
        ``error_bound`` = |sigma_u^fine - sigma_u^coarse| / sigma_u^coarse and
        ``richardson_extrapolated`` (O(h^2) extrapolation of the fine value),
        ``richardson_relative_change``.
    """
    coarse = GrillageModel(n_coarse, n_coarse, a, b, t, h_s, t_w, b_f, t_f,
                           material, w0_max)
    fine = GrillageModel(n_fine, n_fine, a, b, t, h_s, t_w, b_f, t_f,
                         material, w0_max)
    s_coarse = coarse.compute_ultimate_strength()
    s_fine = fine.compute_ultimate_strength()

    rel = abs(s_fine - s_coarse) / s_coarse if s_coarse > 0 else float("inf")
    out: Dict[str, float] = {
        "sigma_u_coarse": s_coarse,
        "sigma_u_fine": s_fine,
        "relative_error": rel,
        "converged": bool(rel < CONVERGENCE_TOL),
        "delta_sigma": s_fine - s_coarse,
        "n_coarse": n_coarse,
        "n_fine": n_fine,
        "error_bound": rel,
        "error_bound_tolerance": CONVERGENCE_TOL,
    }

    # Richardson extrapolation for an O(h^2) quadrature (midpoint rule)
    h_c, h_f = 1.0 / n_coarse, 1.0 / n_fine
    if abs(s_fine - s_coarse) > 0.0 and h_f < h_c:
        ratio = (h_c / h_f) ** 2
        s_exact = s_fine + (s_fine - s_coarse) / (ratio - 1.0)
        out["richardson_extrapolated"] = float(s_exact)
        out["richardson_relative_change"] = float(abs(s_exact - s_fine) / s_fine)

    if refine is not None and refine >= n_fine:
        seq = {}
        for n in (max(1, n_coarse), n_fine, int(refine)):
            seq[int(n)] = GrillageModel(n, n, a, b, t, h_s, t_w, b_f, t_f,
                                        material, w0_max).compute_ultimate_strength()
        out["refinement_sequence"] = seq
    return out


def isolated_element_deviation(
    a: float, b: float, t: float,
    h_s: float, t_w: float, b_f: float, t_f: float,
    material: VonMisesMaterial,
    w0_max: Optional[float] = None,
    n_x: int = 5,
    n_y: int = 5,
    imperfection_field: str = "sine",
) -> Dict[str, float]:
    """Isolated-element deviation ``delta_iso`` and its error-bound status.

    ``delta_iso = (sigma_u^panel - sigma_u^isolated) / sigma_u^isolated`` where
    the isolated unit is the single intercolumn unit evaluated at the *mean*
    cell imperfection of the panel, so that the deviation isolates the
    concavity (Jensen) effect of averaging over the imperfection field.
    """
    if w0_max is not None:
        sc = _scantlings(a, b, t, h_s, t_w, b_f, t_f, material, w0_max)
    else:
        sc = _scantlings(a, b, t, h_s, t_w, b_f, t_f, material, None)
    solver = SmithPanelSolver(sc, n_x=n_x, n_y=n_y,
                              imperfection_field=imperfection_field)
    res = solver.solve()
    sigma_iso = solver.isolated_capacity()
    delta = (res.sigma_u - sigma_iso) / sigma_iso if sigma_iso > 0 else 0.0
    # second reference: the isolated unit at the *nominal design* imperfection
    # (w0p + w0c, i.e. what an engineer gets from a single-element check)
    sigma_iso_nominal = float(solver.unit_capacity(sc.w0p + sc.w0c)[0])
    delta_nominal = ((res.sigma_u - sigma_iso_nominal) / sigma_iso_nominal
                     if sigma_iso_nominal > 0 else 0.0)
    return {
        "sigma_u_isolated": float(sigma_iso),
        "sigma_u_grillage": float(res.sigma_u),
        "delta_iso": float(delta),
        "delta_iso_sign": "positive" if delta >= 0 else "negative",
        "delta_iso_magnitude": float(abs(delta)),
        "sigma_u_over_sigma_y": float(res.sigma_u_over_sigma_y),
        "mode": res.mode,
        "sigma_u_isolated_nominal": sigma_iso_nominal,
        "delta_iso_nominal": float(delta_nominal),
        "delta_iso_nominal_sign": "positive" if delta_nominal >= 0 else "negative",
        "interpretation": (
            "sign of delta_iso = curvature of the capacity-vs-imperfection "
            "relation at the panel mean imperfection: negative = concave "
            "(Jensen loss), positive = convex (isolated element conservative). "
            "Magnitude is the isolated-element error bound of the panel average."
        ),
        "n_x": n_x,
        "n_y": n_y,
    }
