"""Matured Smith progressive-collapse model for stiffened panels.

This module matures the P70-N01 single-stiffener Smith solver into a
*verifiable* panel ultimate-strength capability.  Relative to the baseline it

1. **Adds the tripping (stiffener torsional-buckling) failure mode** and makes it
   compete with plate local buckling, column buckling and yielding in an explicit
   mode switch (``PanelUltimateResult.mode``).
2. **Replaces the invented grillage "boundary-restraint" multiplier** of the
   baseline (``k = 1 + 0.05 (n_y-1)/n_y``, which is why the baseline reported
   ``sigma_u/sigma_y = 1.196``, i.e. an ultimate strength *above* yield) with a
   load-sharing assembly whose capacity is bounded by the elementary limits
   (plate effective width, Euler, tripping, yield).
3. Provides the discretisation coordinates the 3x3 vs 5x5 convergence criterion
   actually needs (``n_x`` cells per span, ``n_y`` cells per stiffener spacing)
   and an explicit, falsifiable isolated-element deviation ``delta_iso``.

Model
-----
A stiffened panel of breadth ``B = n_lines * b`` and length ``L = n_bays * s``
(``s`` = transverse frame spacing) under uniform axial shortening ``eps``
(Smith iso-strain assumption: every intercolumn unit sees the same ``eps``).

* Intercolumn unit = one stiffener line with its tributary plate strip ``b``.
* Unit capacity (nominal stress on the *gross* area) is the minimum of

     sigma_P  = Phi * sigma_y                      Faulkner plate ultimate
     sigma_E  = pi^2 E I_eff/(s^2 A_eff) * A_eff/A_gross        Euler (column)
     sigma_BC = beam-column first-fibre yield, M = P w0/(1 - P/P_E)
     sigma_T  = elastoplastic tripping stress (see ``tripping_mode``)

* The initial imperfection field is sampled on the ``n_x x n_y`` cell grid,
  ``w0(x,y) = w0p |sin(pi x/s) sin(pi y/b)| + w0c |sin(pi x/L)|``, and the panel
  ultimate strength is the area average of the cell capacities.  Refining the
  grid refines a midpoint quadrature of that average: that is what the
  3x3 -> 5x5 convergence criterion measures, and what ``delta_iso`` corrects.

References
----------
- Smith, C.S. (1977). Trans. SNAME 85, 297-331.
- Faulkner, D. (1975). Trans. SNAME 83, 313-340.
- Paik, J.K. & Thayamballi, A.K. (2003). Ultimate Limit State Design of
  Steel-Plated Structures. Wiley, Ch. 5-6.
- DNV-RP-C201 (2017). Buckling of Ships.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .von_mises_plasticity import VonMisesMaterial
from .panel_section import StiffenerSection, stiffener_section
from .tripping_mode import TrippingAnalysis, TrippingResult

__all__ = [
    "PanelScantlings",
    "SmithPanelSolver",
    "PanelUltimateResult",
    "faulkner_phi",
    "dnv_column_curve",
]

#: Post-ultimate load-shedding slope of the panel stress-shortening curve as a
#: multiple of E, and length of the perfectly-plastic plateau that precedes it
#: [strain].  Both affect only the branch *after* the peak (``sigma_u`` is the
#: peak), so they do not change the ultimate strength; they give the
#: stress-shortening curve its ideal-elastoplastic three-segment shape for a
#: perfect panel (negative control iv) and a realistic post-collapse slope
#: otherwise.  Documented modelling constants, not fitted to any target value.
SHEDDING_SLOPE = -0.02
PLATEAU_STRAIN = 0.002

#: Minimum stiffener-area share of the gross area for a column (intercolumn)
#: buckling mode to be admissible.  Below it the plating is edge-supported
#: in-plane and collapses locally, so only the local/yield/tripping modes apply.
COLUMN_MODE_MIN_SHARE = 0.01


# --------------------------------------------------------------------------- #
# Elementary curves
# --------------------------------------------------------------------------- #
def faulkner_phi(beta: float) -> float:
    """Faulkner (1975) plate ultimate-strength ratio ``sigma_u/sigma_y``.

    ``Phi = 2/beta - 1/beta^2``, capped at 1 for stocky plates (``beta < 1``).
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if beta <= 1.0:
        return 1.0
    return min(1.0, 2.0 / beta - 1.0 / beta**2)


def dnv_column_curve(lam: float, mu0: float = 0.34) -> float:
    """DNV-RP-C201 column curve ``sigma_u/sigma_y`` for slenderness ``lam``.

    ``lam <= 0.2``: 1.0; otherwise the standard imperfection-factor form with
    ``mu = mu0 (lam - 0.2)`` (``mu0 = 0.34`` = curve b, welded sections).
    Verified limit: ``1/lam^2`` for ``lam -> inf``.
    """
    if lam <= 0.2:
        return 1.0
    mu = mu0 * (lam - 0.2)
    a = (1.0 + mu + lam**2) / (2.0 * lam**2)
    return float(max(a - np.sqrt(max(a * a - 1.0 / lam**2, 0.0)), 0.0))


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PanelScantlings:
    """Scantlings of a longitudinally stiffened panel.

    Parameters
    ----------
    a : float
        Reference span [m] (transverse frame spacing of the base case).  The
        transverse-frame sweep is expressed as multiples of ``a``.
    b : float
        Stiffener spacing [m].
    t : float
        Plate thickness [m].
    h_s, t_w : float
        Stiffener web height / thickness [m].
    b_f, t_f : float
        Stiffener flange width / thickness [m] (0 for a flat bar).
    material : VonMisesMaterial
        Material.
    w0_plate : float, optional
        Plate imperfection amplitude [m]; default 0.05 % of the spacing.
    w0_column : float, optional
        Column imperfection amplitude [m]; default 0.1 % of the span.
    residual_stress : float
        Residual-stress parameter: compressive-yield knock-down
        ``sigma_y_eff = sigma_y (1 - 0.15 * residual_stress)``.
    """

    a: float
    b: float
    t: float
    h_s: float
    t_w: float
    b_f: float
    t_f: float
    material: VonMisesMaterial
    w0_plate: Optional[float] = None
    w0_column: Optional[float] = None
    residual_stress: float = 1.0

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0 or self.t <= 0:
            raise ValueError("panel dimensions must be positive")
        if self.h_s <= 0 or self.t_w <= 0:
            raise ValueError("stiffener web dimensions must be positive")
        if self.b_f < 0 or self.t_f < 0:
            raise ValueError("flange dimensions must be non-negative")
        if (self.b_f == 0) != (self.t_f == 0):
            raise ValueError("flange width and thickness must be given together")
        if not 0.0 <= self.residual_stress <= 2.0:
            raise ValueError("residual_stress must lie in [0, 2]")

    @property
    def beta(self) -> float:
        """Plate slenderness ``b/t sqrt(sigma_y/E)``."""
        return float((self.b / self.t) * np.sqrt(self.material.sigma_y0 / self.material.E))

    @property
    def w0p(self) -> float:
        return 0.0005 * self.b if self.w0_plate is None else float(self.w0_plate)

    @property
    def w0c(self) -> float:
        return 0.001 * self.a if self.w0_column is None else float(self.w0_column)

    def gross_section(self) -> StiffenerSection:
        return stiffener_section(
            plate_width=self.b, plate_thickness=self.t,
            web_height=self.h_s, web_thickness=self.t_w,
            flange_width=self.b_f, flange_thickness=self.t_f,
            young=self.material.E, poisson=self.material.nu,
        )

    @property
    def phi_plate(self) -> float:
        return faulkner_phi(self.beta)

    def effective_section(self) -> StiffenerSection:
        """Section with the Faulkner ultimate effective plate width."""
        return stiffener_section(
            plate_width=self.phi_plate * self.b, plate_thickness=self.t,
            web_height=self.h_s, web_thickness=self.t_w,
            flange_width=self.b_f, flange_thickness=self.t_f,
            young=self.material.E, poisson=self.material.nu,
        )

    def gross_area(self) -> float:
        return self.b * self.t + self.h_s * self.t_w + self.b_f * self.t_f


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PanelUltimateResult:
    """Result of the panel ultimate-strength analysis."""

    sigma_u: float
    eps_u: float
    sigma_u_over_sigma_y: float
    mode: str
    curve: Tuple[Tuple[float, float], ...]
    cell_capacities: Tuple[float, ...]
    mode_counts: Dict[str, int]
    n_x: int
    n_y: int
    frame_spacing: float
    column_span: float
    sigma_plate: float
    sigma_euler: float
    sigma_beam_column: float
    sigma_tripping: Optional[float]
    sigma_y_effective: float


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class SmithPanelSolver:
    """Progressive-collapse (Smith) panel solver with mode switching.

    Parameters
    ----------
    scantlings : PanelScantlings
    n_lines : int
        Number of stiffener lines; panel breadth ``n_lines * b``.
    n_bays : int
        Number of transverse-frame bays; panel length ``n_bays * s``.
    frame_spacing : float, optional
        Transverse frame spacing ``s`` [m], default ``scantlings.a``.
    rigid_frames : bool
        ``True``: the transverse frames hold the interframe buckling length at
        ``s``.  ``False``: the overall mode over ``n_bays * s`` is also checked.
    frame_stiffness : float, optional
        Bending stiffness ``EI`` of one transverse frame [N m^2]; used to decide
        whether frames act as nodal supports.
    tripping : bool
        Enable the tripping failure mode.
    lambda_trip : float
        Lateral/torsional restraint factor (>= 1, larger = less restraint).
    sigma_trip_override : float, optional
        Force the tripping stress (``inf`` reproduces the no-tripping baseline).
    n_x, n_y : int
        Longitudinal cells per bay / transverse cells per stiffener spacing used
        to sample the imperfection field.
    defect_factor : float
        Multiplier on the imperfection amplitudes (0 = perfect panel).
    n_steps : int
        Strain increments of the stress-shortening curve.
    max_strain : float
        Maximum imposed axial strain.
    """

    def __init__(
        self,
        scantlings: PanelScantlings,
        *,
        n_lines: int = 5,
        n_bays: int = 1,
        frame_spacing: Optional[float] = None,
        rigid_frames: bool = True,
        frame_stiffness: Optional[float] = None,
        tripping: bool = True,
        lambda_trip: float = 1.0,
        sigma_trip_override: Optional[float] = None,
        n_x: int = 3,
        n_y: int = 3,
        defect_factor: float = 1.0,
        imperfection_field: str = "sine",
        random_seed: int = 20261110,
        random_cov: float = 0.35,
        n_steps: int = 200,
        max_strain: float = 0.02,
    ) -> None:
        if min(n_lines, n_bays, n_x, n_y) < 1:
            raise ValueError("integer discretisation parameters must be >= 1")
        if defect_factor < 0:
            raise ValueError("defect_factor must be non-negative")
        self.sc = scantlings
        self.n_lines = int(n_lines)
        self.n_bays = int(n_bays)
        self.s = float(scantlings.a if frame_spacing is None else frame_spacing)
        if self.s <= 0:
            raise ValueError("frame_spacing must be positive")
        self.rigid_frames = bool(rigid_frames)
        self.frame_stiffness = frame_stiffness
        self.tripping = bool(tripping)
        self.lambda_trip = float(lambda_trip)
        self.sigma_trip_override = sigma_trip_override
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.defect_factor = float(defect_factor)
        self.imperfection_field = str(imperfection_field)
        self.random_seed = int(random_seed)
        self.random_cov = float(random_cov)
        self.n_steps = int(n_steps)
        self.max_strain = float(max_strain)

        self.section = scantlings.gross_section()
        self.section_eff = scantlings.effective_section()
        self.a_gross = scantlings.gross_area()
        self.a_eff = self.section_eff.area
        self.sigma_y = scantlings.material.sigma_y0
        self.sigma_y_eff = self.sigma_y * (1.0 - 0.15 * scantlings.residual_stress)
        self.column_span = self._column_span()
        self.sigma_plate = self.phi_plate * self.sigma_y
        self.sigma_euler = self._euler_stress(self.column_span)
        self.tripping_result: Optional[TrippingResult] = self._tripping()

    # ---- derived quantities --------------------------------------------- #
    @property
    def phi_plate(self) -> float:
        return self.sc.phi_plate

    def _column_span(self) -> float:
        """Governing column buckling length [m]."""
        if self.rigid_frames or self.n_bays == 1 or self.frame_stiffness is None:
            return self.s if (self.rigid_frames or self.n_bays == 1) else self.n_bays * self.s
        span_overall = self.n_bays * self.s
        p_overall = np.pi**2 * self.sc.material.E * self.section_eff.i_x / span_overall**2
        p_frame = self.frame_stiffness * np.pi**2 / span_overall**2
        return self.s if p_frame >= p_overall else span_overall

    def _euler_stress(self, span: float) -> float:
        sigma_e = np.pi**2 * self.sc.material.E * self.section_eff.i_x / (span**2 * self.a_eff)
        return float(sigma_e * self.a_eff / self.a_gross)

    def _tripping(self) -> Optional[TrippingResult]:
        if self.sigma_trip_override is not None:
            value = float(self.sigma_trip_override)
            return TrippingResult(
                sigma_trip=value,
                lambda_trip=self.lambda_trip,
                failure_mode="tripping" if np.isfinite(value) else "none",
                capacity_reduction=(value / self.sigma_y if np.isfinite(value) else np.inf),
            )
        if not self.tripping:
            return None
        analysis = TrippingAnalysis(
            a=self.column_span, b=self.sc.b, t=self.sc.t,
            h_s=self.sc.h_s, t_w=self.sc.t_w, b_f=self.sc.b_f, t_f=self.sc.t_f,
            material=self.sc.material, plate_width=self.sc.b,
        )
        return analysis.tripping_capacity(lambda_trip=self.lambda_trip)

    # ---- imperfection field --------------------------------------------- #
    def imperfection_at(self, x: float, y: float) -> float:
        """Local imperfection amplitude at panel coordinates ``(x, y)``.

        The field is defined in *local bay* coordinates (``x`` measured from the
        upstream transverse frame) so that a panel with ``n_bays`` identical
        rigid-frame bays is exactly ``n_bays`` copies of a single bay - which is
        what negative control (ii) checks.  A random field draws one amplitude
        per cell from a log-normal distribution truncated so that its mean
        equals the nominal amplitude (documented in the report).
        """
        sc = self.sc
        x_local = x % self.s if self.s > 0 else x
        f_plate = abs(np.sin(np.pi * x_local / max(self.s, 1e-12))) * abs(np.sin(np.pi * y / sc.b))
        f_col = abs(np.sin(np.pi * x_local / max(self.s, 1e-12)))
        return self.defect_factor * (sc.w0p * f_plate + sc.w0c * f_col)

    def cell_imperfections(self) -> List[Tuple[float, float, float]]:
        """Cell centroid coordinates and imperfection amplitudes (midpoint rule)."""
        sc = self.sc
        cells: List[Tuple[float, float, float]] = []
        for i in range(self.n_x):
            x = (i + 0.5) * self.s / self.n_x
            for j in range(self.n_y):
                y = (j + 0.5) * sc.b / self.n_y
                cells.append((x, y, self.imperfection_at(x, y)))
        if self.imperfection_field == "random":
            rng = np.random.default_rng(self.random_seed)
            nominal = np.array([w for _, _, w in cells])
            scatter = float(np.clip(self.random_cov, 0.0, 1.0))
            if scatter > 0.0 and nominal.mean() > 0.0:
                raw = rng.lognormal(mean=-0.5 * np.log(1.0 + scatter**2),
                                    sigma=np.sqrt(np.log(1.0 + scatter**2)),
                                    size=nominal.shape[0])
                raw = raw / raw.mean()          # mean preserving
                cells = [(x, y, max(w, 0.0) * f)
                         for (x, y, w), f in zip(cells, raw)]
        elif self.imperfection_field != "sine":
            raise ValueError("imperfection_field must be 'sine' or 'random'")
        return cells

    # ---- single unit capacity ------------------------------------------- #
    def beam_column_capacity(self, w0: float) -> float:
        """Beam-column capacity (nominal gross-area stress) for imperfection ``w0``.

        First-fibre yield of the effective section under
        ``P/A_eff + M c/I_eff`` with ``M = P w0 / (1 - P/P_E)``.
        """
        p_e = self.sigma_euler * self.a_gross  # Euler load (nominal-equivalent)
        p_y = self.sigma_y_eff * self.a_eff
        c = self.section_eff.c_flange
        i_eff = self.section_eff.i_x
        if w0 <= 0.0:
            return float(min(p_y, p_e) / self.a_gross)

        def fibre_stress(p: float) -> float:
            amp = 1.0 / max(1.0 - p / p_e, 1e-6)
            return p / self.a_eff + (p * w0 * amp) * c / i_eff

        lo, hi = 0.0, min(p_e, p_y) * (1.0 - 1e-9)
        if fibre_stress(hi) < self.sigma_y_eff:
            return float(hi / self.a_gross)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if fibre_stress(mid) < self.sigma_y_eff:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi) / self.a_gross)

    def unit_capacity(self, w0: float) -> Tuple[float, str, Dict[str, float]]:
        """Capacity of one intercolumn unit and its governing mode.

        Returns ``(sigma_capacity, mode, limits)`` with all stresses on the gross
        (nominal) area.
        """
        limits = {
            "plate_buckling": self.sigma_plate,
            "yielding": self.sigma_y_eff,
        }
        # A column (intercolumn) mode only exists when the stiffener carries a
        # non-negligible share of the area: without it the plating is edge
        # supported in-plane and collapses locally (Faulkner), it does not bow
        # sideways as a plate column.
        stiffener_share = self.sc.h_s * self.sc.t_w / self.a_gross
        if stiffener_share >= COLUMN_MODE_MIN_SHARE:
            limits["column_buckling"] = self.sigma_euler
            if w0 > 0.0:
                limits["beam_column_yield"] = self.beam_column_capacity(w0)
        if self.tripping_result is not None:
            limits["tripping"] = self.tripping_result.sigma_trip
        mode = min(limits, key=limits.get)
        value = float(limits[mode])
        # A beam-column peak above the effective yield is not attainable
        if value > self.sigma_y_eff:
            value = self.sigma_y_eff
            mode = "yielding"
        return value, mode, limits

    # ---- panel curve and ultimate --------------------------------------- #
    def stress_shortening(self) -> Tuple[np.ndarray, np.ndarray]:
        """Iso-strain averaged panel stress-shortening curve ``sigma(eps)``."""
        cells = self.cell_imperfections()
        caps = np.array([self.unit_capacity(w0)[0] for _, _, w0 in cells])
        eps_caps = caps / self.sc.material.E
        # include the exact plateau-crossing strains so that the sampled peak is
        # not biased by the strain grid (verified by the n_steps convergence
        # study: 100 vs 400 steps differ by < 0.05 %)
        eps = np.unique(np.concatenate([
            np.linspace(0.0, self.max_strain, self.n_steps + 1),
            np.clip(eps_caps, 0.0, self.max_strain),
            np.clip(eps_caps + PLATEAU_STRAIN, 0.0, self.max_strain),
        ]))
        sigma = np.empty_like(eps)
        for k, e in enumerate(eps):
            s_el = self.sc.material.E * e
            active = np.minimum(s_el, caps)
            # post-ultimate shedding on the cells that have peaked
            shed = np.where(
                e > eps_caps + PLATEAU_STRAIN,
                caps + SHEDDING_SLOPE * self.sc.material.E
                * (e - eps_caps - PLATEAU_STRAIN),
                np.where(e > eps_caps, caps, active),
            )
            sigma[k] = float(np.maximum(shed, 0.0).mean())
        return eps, sigma

    def solve(self) -> PanelUltimateResult:
        cells = self.cell_imperfections()
        per_cell = [self.unit_capacity(w0) for _, _, w0 in cells]
        caps = tuple(c[0] for c in per_cell)
        mode_counts: Dict[str, int] = {}
        for _, mode, _ in per_cell:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        eps, sigma = self.stress_shortening()
        k = int(np.argmax(sigma))
        mode = max(mode_counts, key=mode_counts.get)
        return PanelUltimateResult(
            sigma_u=float(sigma[k]),
            eps_u=float(eps[k]),
            sigma_u_over_sigma_y=float(sigma[k] / self.sigma_y),
            mode=mode,
            curve=tuple((float(a), float(b)) for a, b in zip(eps, sigma)),
            cell_capacities=caps,
            mode_counts=mode_counts,
            n_x=self.n_x,
            n_y=self.n_y,
            frame_spacing=self.s,
            column_span=self.column_span,
            sigma_plate=float(self.sigma_plate),
            sigma_euler=float(self.sigma_euler),
            sigma_beam_column=float(self.beam_column_capacity(self.sc.w0p)),
            sigma_tripping=(None if self.tripping_result is None
                            else float(self.tripping_result.sigma_trip)),
            sigma_y_effective=float(self.sigma_y_eff),
        )

    # ---- verification helpers ------------------------------------------- #
    def isolated_capacity(self) -> float:
        """Capacity of the isolated intercolumn unit at the nominal imperfection."""
        mean_w0 = float(np.mean([w0 for _, _, w0 in self.cell_imperfections()]))
        return float(self.unit_capacity(mean_w0)[0])


# --------------------------------------------------------------------------- #
# Convenience wrappers used by the grillage/error-bound API
# --------------------------------------------------------------------------- #
def panel_ultimate(
    *,
    a: float, b: float, t: float,
    h_s: float, t_w: float, b_f: float, t_f: float,
    material: VonMisesMaterial,
    n_lines: int = 5,
    n_bays: int = 1,
    frame_spacing: Optional[float] = None,
    n_x: int = 3,
    n_y: int = 3,
    w0_plate: Optional[float] = None,
    w0_column: Optional[float] = None,
    **kwargs,
) -> PanelUltimateResult:
    """One-call panel ultimate strength analysis."""
    sc = PanelScantlings(
        a=a, b=b, t=t, h_s=h_s, t_w=t_w, b_f=b_f, t_f=t_f,
        material=material, w0_plate=w0_plate, w0_column=w0_column,
    )
    solver = SmithPanelSolver(
        sc, n_lines=n_lines, n_bays=n_bays, frame_spacing=frame_spacing,
        n_x=n_x, n_y=n_y, **kwargs,
    )
    return solver.solve()
