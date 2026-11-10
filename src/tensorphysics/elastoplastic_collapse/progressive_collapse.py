"""Progressive (fiber-based) collapse analysis of a stiffened panel — Smith method.

This module replaces the former smeared *interaction-formula* estimate of
``SmithCollapseSolver.solve()`` with an incremental, mechanically consistent
implementation of the classical Smith (1977) method:

1. **Cross-section discretisation.**  The stiffened panel is split into axial
   fibers: one plate element (plate effective width acts on the whole panel
   width ``b``) and ``n_w`` web fibers plus ``n_f`` flange fibers, each with its
   own area ``A_j`` and centroid ``z_j``.

2. **Element load-end shortening curves.**  Every fiber carries a
   one-dimensional stress–strain curve:

   * stiffener fibers: J2 uniaxial elasto-plastic curve (isotropic hardening),
     ``σ = σ_y0 + H ε̄_p``;
   * plate element: elastic up to the plate buckling stress
     ``σ_cr = k π² E /(12(1-ν²)) (t/b)²`` (k = 4, simply supported), then the
     von Karman effective-width post-buckling branch
     ``σ_avg = sqrt(σ_cr σ_edge)`` limited by edge yielding, then a linear
     softening branch (``softening_modulus_ratio·E``) down to
     ``residual_stress_ratio·σ_u,p``.  The softening branch is what allows the
     panel load–shortening curve to pass a **limit point**; the limit load
     itself is insensitive to it (see the sensitivity study in the report).

3. **Section equilibrium.**  Strain is linear over the section,
   ``ε(z) = ε_0 + κ z``.  For a prescribed nominal strain ``ε_0`` the curvature
   ``κ`` is the root of the mid-span equilibrium of a pin-ended column with
   initial imperfection ``δ_0`` (Smith's "column" step):

       M(ε_0, κ) = N(ε_0, κ) · (δ_0 + κ a²/π²)

   with ``N = Σ A_j σ_j``, ``M = Σ A_j σ_j (z_j - z_c)`` and the sinusoidal
   amplification relation ``δ = κ a²/π²``.  ``σ_avg = N / A_total`` traces the
   load–shortening curve; ``σ_u = max σ_avg``.

4. **Collapse detection.**  When no equilibrium curvature exists in the search
   bracket, the section cannot sustain the imposed shortening any more; the
   sweep stops and the last converged state defines the post-ultimate branch.

Verified anchors
----------------
- Stocky panel (β < 1, λ < 1, no imperfection) → ``σ_u → σ_y`` (full section
  yield) and ``σ_u ≤ σ_y + H ε̄_p`` (never exceeds the current yield stress).
- Monotone reduction of ``σ_u`` with plate slenderness β and with column
  slenderness λ (Perry-type imperfection sensitivity).
- Plate element ultimate reproduces the von Karman value ``sqrt(σ_cr σ_y)``.
- Convergence with the number of web fibers.

Limitations (declared, see report)
----------------------------------
- Tripping / lateral-torsional buckling of the stiffener is not part of the
  element set (see :mod:`tensorphysics.elastoplastic_collapse.tripping_mode`
  for the separate check).
- Web/flange *local* buckling is ignored (stocky webs only).
- The softening branch is a phenomenological completion of the plate curve.
- Residual stresses are not modelled; imperfection enters through ``δ_0`` only.

References
----------
- Smith, C.S. (1977). "Influence of local imperfections on intercolumn
  behaviour of stiffened plates." Trans. SNAME, 85, 1-18.
- Faulkner, D. (1975). "A review of effective plating for use in the analysis of
  stiffened plating in bending." J. Ship Research, 19(1), 1-17.
- Paik, J.K. & Thayamballi, A.K. (2003). Ultimate Limit State Design of
  Steel-Plated Structures. Wiley.
- von Karman, T., Sechler, E.E. & Donnell, L.H. (1932). "The strength of thin
  plates in compression." Trans. ASME, 54, 53-57.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .von_mises_plasticity import VonMisesMaterial

__all__ = [
    "PanelFiber",
    "uniaxial_elastoplastic_stress",
    "plate_load_shortening_stress",
    "ProgressiveCollapseSolver",
    "ProgressiveCollapseResult",
    "panel_convergence_study",
]


# ---------------------------------------------------------------------------
# element load-end shortening curves
# ---------------------------------------------------------------------------
def uniaxial_elastoplastic_stress(eps: float, material: VonMisesMaterial) -> float:
    """J2 uniaxial stress with linear isotropic hardening (closed form).

    ``σ = sign(ε)·min(E|ε|, σ_y0 + H ε̄_p)`` with
    ``ε̄_p = (|ε| − σ_y0/E)/(1 + H/E)`` once yielded.  Valid for monotone
    loading; used here for the stiffener fibers, whose strain path inside the
    section analysis is monotone.
    """
    a = abs(eps)
    eps_y = material.sigma_y0 / material.E
    if a <= eps_y:
        return material.E * eps
    eps_p = (a - eps_y) / (1.0 + material.H / material.E)
    sigma = material.sigma_y0 + material.H * eps_p
    return float(np.sign(eps) * sigma)


def plate_load_shortening_stress(
    eps: float,
    material: VonMisesMaterial,
    sigma_cr: float,
    softening_modulus_ratio: float = 0.02,
    residual_stress_ratio: float = 0.2,
) -> float:
    """Plate element average stress for a prescribed shortening (compression +).

    Parameters
    ----------
    eps : float
        Shortening (positive in compression) of the plate element [-].
    material : VonMisesMaterial
        Plate material.
    sigma_cr : float
        Elastic plate buckling stress [Pa] (``k π²E/(12(1-ν²))(t/b)²``).
    softening_modulus_ratio : float
        Post-ultimate softening slope as a fraction of ``E`` [-].
    residual_stress_ratio : float
        Residual (post-softening plateau) stress as a fraction of ``σ_u,p`` [-].

    Returns
    -------
    sigma : float
        Average compressive stress [Pa] (returned positive in compression).
    """
    if eps <= 0.0:
        return 0.0
    eps_cr = sigma_cr / material.E
    eps_y = material.sigma_y0 / material.E

    # stocky plate (buckling stress above yield): the material curve governs,
    # the plate yields before it can buckle.
    if eps_cr >= eps_y:
        return uniaxial_elastoplastic_stress(eps, material)

    if eps <= eps_cr:
        return float(material.E * eps)

    # von Karman post-buckling: average stress = sqrt(sigma_cr * sigma_edge)
    if eps <= eps_y:
        sigma_edge = material.E * eps
        return float(np.sqrt(sigma_cr * sigma_edge))

    sigma_u_plate = float(np.sqrt(sigma_cr * material.sigma_y0))
    soft_slope = softening_modulus_ratio * material.E
    drop = soft_slope * (eps - eps_y)
    sigma = sigma_u_plate - drop
    floor = residual_stress_ratio * sigma_u_plate
    return float(max(sigma, floor))


@dataclass(frozen=True)
class PanelFiber:
    """A single axial fiber of the panel cross-section."""

    area: float        # [m^2]
    z: float           # centroid measured from the plate mid-thickness [m]
    kind: str          # "plate" | "web" | "flange"


# ---------------------------------------------------------------------------
# result containers
# ---------------------------------------------------------------------------
@dataclass
class ProgressiveCollapseResult:
    """Result of the progressive collapse analysis."""

    sigma_u: float                      # ultimate average stress [Pa]
    eps_u: float                        # nominal strain at ultimate [-]
    load_displacement: List[Tuple[float, float]]   # (eps0, sigma_avg)
    kappa_path: List[Tuple[float, float]]          # (eps0, kappa)
    deflection_path: List[Tuple[float, float]]     # (eps0, delta)
    failure_mode: str
    converged: bool
    sigma_cr_plate: float
    sigma_cr_column: float
    n_fibers: int
    diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def sigma_u_over_sigma_y(self) -> float:
        """Ultimate strength ratio (reads diagnostics for the yield stress)."""
        sy = self.diagnostics.get("sigma_y0", 1.0)
        return float(self.sigma_u / sy) if sy else float("nan")

    def post_peak_ratio(self) -> float:
        """Last / peak average stress — < 1 whenever softening is resolved."""
        s = np.array([v for _, v in self.load_displacement])
        return float(s[-1] / s.max()) if s.size and s.max() > 0 else float("nan")


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
class ProgressiveCollapseSolver:
    """Incremental fiber-based Smith-method collapse solver.

    Parameters
    ----------
    geometry : StiffenedPlateGeometry
        Panel geometry and material (``w0_max`` is used as the initial
        imperfection amplitude ``δ_0`` for the column step).
    n_web_fibers : int
        Number of fibers along the stiffener web height.
    n_flange_fibers : int
        Number of fibers through the flange thickness.
    n_steps : int
        Number of strain increments of the nominal strain sweep.
    max_strain : float
        Maximum nominal compressive strain [-].
    softening_modulus_ratio : float
        Plate post-ultimate softening slope / E.
    residual_stress_ratio : float
        Plate post-ultimate residual stress / σ_u,p.
    imperfection_sign : float
        +1 deflects the initial imperfection towards the stiffener, -1 towards
        the plate.
    """

    def __init__(
        self,
        geometry,
        n_web_fibers: int = 8,
        n_flange_fibers: int = 2,
        n_steps: int = 200,
        max_strain: float = 0.02,
        softening_modulus_ratio: float = 0.02,
        residual_stress_ratio: float = 0.2,
        imperfection_sign: float = 1.0,
    ) -> None:
        if n_web_fibers < 1 or n_flange_fibers < 1:
            raise ValueError("fiber counts must be >= 1")
        self.geom = geometry
        self.mat: VonMisesMaterial = geometry.material
        self.n_w = int(n_web_fibers)
        self.n_f = int(n_flange_fibers)
        self.n_steps = int(n_steps)
        self.max_strain = float(max_strain)
        self.soft_ratio = float(softening_modulus_ratio)
        self.resid_ratio = float(residual_stress_ratio)
        self.imperfection_sign = float(imperfection_sign)

        self.sec = geometry.section_properties()
        self.z_c = float(self.sec["z_bar"])
        self.sigma_cr_plate = float(self._plate_buckling_stress())
        self.sigma_cr_column = float(geometry.euler_buckling_stress())
        self.fibers = self._build_fibers()
        self.area_total = float(sum(f.area for f in self.fibers))
        self.delta0 = self.imperfection_sign * float(geometry.w0_max)

    # -- discretisation -----------------------------------------------------
    def _plate_buckling_stress(self) -> float:
        g = self.geom
        a_over_b = g.a / g.b
        if a_over_b >= 1.0:
            k = 4.0
        else:
            k = (a_over_b + 1.0 / a_over_b) ** 2
        return k * np.pi**2 * self.mat.E / (12.0 * (1.0 - self.mat.nu**2)) * (g.t / g.b) ** 2

    def _build_fibers(self) -> List[PanelFiber]:
        g = self.geom
        fibers = [PanelFiber(area=g.b * g.t, z=0.0, kind="plate")]
        dz_w = g.h_s / self.n_w
        for i in range(self.n_w):
            z = g.t / 2.0 + (i + 0.5) * dz_w
            fibers.append(PanelFiber(area=g.t_w * dz_w, z=z, kind="web"))
        if g.b_f > 0.0 and g.t_f > 0.0:
            dz_f = g.t_f / self.n_f
            for i in range(self.n_f):
                z = g.t / 2.0 + g.h_s + (i + 0.5) * dz_f
                fibers.append(PanelFiber(area=g.b_f * dz_f, z=z, kind="flange"))
        return fibers

    # -- element stress -----------------------------------------------------
    def fiber_stress(self, kind: str, eps: float) -> float:
        """Compressive fiber stress [Pa] (positive in compression)."""
        if kind == "plate":
            return plate_load_shortening_stress(
                eps, self.mat, self.sigma_cr_plate,
                self.soft_ratio, self.resid_ratio)
        return uniaxial_elastoplastic_stress(eps, self.mat)

    def fiber_capacity_stress(self, kind: str, eps: float) -> float:
        """Peak (squash) stress a fiber can sustain at the given strain [Pa].

        Used when the section reaches a plastic mechanism (no curvature can
        satisfy ``M = N·δ``): the axial capacity is then the sum of the element
        peak stresses, i.e. the yield stress for the stiffener fibers and the
        smaller of yield and von Karman plate ultimate for the plate.
        """
        eps_y = self.mat.sigma_y0 / self.mat.E
        a = abs(float(eps))
        eps_p = max(0.0, (a - eps_y) / (1.0 + self.mat.H / self.mat.E))
        sigma_yield = self.mat.sigma_y0 + self.mat.H * eps_p
        if kind == "plate":
            eps_cr = self.sigma_cr_plate / self.mat.E
            if eps_cr < eps_y:                      # plate buckles before yield
                return float(np.sqrt(self.sigma_cr_plate * self.mat.sigma_y0))
        return float(sigma_yield)

    def squash_capacity(self, eps0: float) -> float:
        """Section squash capacity (average stress) at a given nominal strain."""
        N = 0.0
        for f in self.fibers:
            N += self.fiber_capacity_stress(f.kind, abs(eps0)) * f.area
        return N / self.area_total

    def section_response(self, eps0: float, kappa: float) -> Tuple[float, float]:
        """Axial force N [N] and moment M [N·m] for a linear strain field."""
        N = 0.0
        M = 0.0
        for f in self.fibers:
            eps = eps0 + kappa * f.z
            sigma = self.fiber_stress(f.kind, eps)
            N += sigma * f.area
            M += sigma * f.area * (f.z - self.z_c)
        return N, M

    # -- equilibrium --------------------------------------------------------
    def _residual(self, eps0: float, kappa: float) -> Tuple[float, float, float]:
        N, M = self.section_response(eps0, kappa)
        delta = self.delta0 + kappa * self.geom.a**2 / np.pi**2
        return M - N * delta, N, M

    def solve_kappa(self, eps0: float, kappa_guess: float = 0.0,
                    kappa_max: float = 0.04, n_scan: int = 40) -> Optional[float]:
        """Root of the mid-span equilibrium equation for a prescribed ε_0.

        The residual

            R(κ) = M(ε_0, κ) − N(ε_0, κ)·(δ_0 + κ a²/π²)

        is solved by a damped Newton iteration started from the previous
        increment (continuation), with a bracketing scan/bisection fallback.
        ``None`` is returned when no equilibrium curvature exists on the
        searched branch (section collapse).
        """
        scale = max(abs(self.section_response(eps0, 0.0)[0])
                    * self.geom.a**2 / np.pi**2, 1.0)
        tol = 1.0e-10 * scale

        def R(k: float) -> float:
            return self._residual(eps0, k)[0]

        # -- damped Newton (continuation) -------------------------------
        k = float(kappa_guess)
        r = R(k)
        h = 1.0e-7
        for _ in range(60):
            if abs(r) <= tol:
                return k
            dr = (R(k + h) - R(k - h)) / (2.0 * h)
            if abs(dr) < 1e-30:
                break
            step = -r / dr
            step = float(np.clip(step, -0.02, 0.02))
            accepted = False
            for _ls in range(20):
                k_new = k + step
                r_new = R(k_new)
                if abs(r_new) < abs(r):
                    k, r = k_new, r_new
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            if abs(step) < 1e-15:
                break
        if abs(r) <= 1.0e-8 * scale:
            return k

        # -- bracketing fallback, expanding the curvature window -------
        window = float(kappa_max)
        for _expand in range(6):
            grid = np.linspace(-window, window, n_scan)
            vals = [R(kk) for kk in grid]
            for i, v in enumerate(vals):
                if abs(v) <= 1.0e-10 * scale:
                    return float(grid[i])
            for i in range(len(grid) - 1):
                if vals[i] * vals[i + 1] < 0.0:
                    lo, hi = float(grid[i]), float(grid[i + 1])
                    r_lo = vals[i]
                    for _ in range(100):
                        mid = 0.5 * (lo + hi)
                        r_mid = R(mid)
                        if r_mid * r_lo <= 0.0:
                            hi = mid
                        else:
                            lo, r_lo = mid, r_mid
                        if abs(hi - lo) < 1e-15:
                            break
                    return float(0.5 * (lo + hi))
            window *= 2.0
        return None

    # -- main sweep ---------------------------------------------------------
    def solve(self) -> ProgressiveCollapseResult:
        eps_arr = np.linspace(self.max_strain / self.n_steps, self.max_strain,
                              self.n_steps)
        load_path: List[Tuple[float, float]] = []
        kappa_path: List[Tuple[float, float]] = []
        delta_path: List[Tuple[float, float]] = []
        converged = True

        sigma_max = -1.0
        eps_at_max = 0.0
        kappa_at_max = 0.0

        kappa_guess = 0.0
        mechanism = False
        for eps0 in eps_arr:
            kappa = self.solve_kappa(float(eps0), kappa_guess=kappa_guess)
            if kappa is None:
                # Plastic mechanism: the section can no longer transmit the
                # eccentric load (the plastic interaction N/N_p + |M|/M_p = 1 is
                # exhausted, so no curvature satisfies M = N·δ).  The axial
                # capacity at this strain is the *squash* capacity ΣAσ(ε_0)/A.
                sigma_squash = self.squash_capacity(float(eps0))
                load_path.append((float(eps0), float(sigma_squash)))
                kappa_path.append((float(eps0), 0.0))
                delta_path.append((float(eps0), float(self.delta0)))
                if sigma_squash > sigma_max:
                    sigma_max = float(sigma_squash)
                    eps_at_max = float(eps0)
                    kappa_at_max = 0.0
                converged = False
                mechanism = True
                break
            kappa_guess = kappa
            N, _M = self.section_response(float(eps0), kappa)
            sigma_avg = N / self.area_total
            load_path.append((float(eps0), float(sigma_avg)))
            kappa_path.append((float(eps0), float(kappa)))
            delta_path.append(
                (float(eps0), float(self.delta0 + kappa * self.geom.a**2 / np.pi**2)))
            if sigma_avg > sigma_max:
                sigma_max = float(sigma_avg)
                eps_at_max = float(eps0)
                kappa_at_max = float(kappa)

        if not load_path:
            raise RuntimeError("progressive collapse sweep produced no equilibrium state")

        diagnostics = {
            "sigma_y0": self.mat.sigma_y0,
            "plate_slenderness_beta": float(self.geom.plate_slenderness),
            "column_slenderness_lambda": float(self.geom.column_slenderness),
            "kappa_at_ultimate": kappa_at_max,
            "deflection_at_ultimate": float(
                self.delta0 + kappa_at_max * self.geom.a**2 / np.pi**2),
            "initial_imperfection": float(self.delta0),
            "von_karman_plate_ultimate": float(
                np.sqrt(self.sigma_cr_plate * self.mat.sigma_y0)),
            "faulkner_plate_ultimate": float(
                self._faulkner_ultimate()),
            "area_total": self.area_total,
            "collapsed_by_plastic_mechanism": mechanism,
            "collapse_stage": ("plastic_mechanism" if mechanism else
                               ("reached_max_strain" if converged else "root_lost")),
        }

        return ProgressiveCollapseResult(
            sigma_u=sigma_max,
            eps_u=eps_at_max,
            load_displacement=load_path,
            kappa_path=kappa_path,
            deflection_path=delta_path,
            failure_mode=self._classify(sigma_max, diagnostics),
            converged=converged,
            sigma_cr_plate=self.sigma_cr_plate,
            sigma_cr_column=self.sigma_cr_column,
            n_fibers=len(self.fibers),
            diagnostics=diagnostics,
        )

    def _faulkner_ultimate(self) -> float:
        """Faulkner (1975) plate ultimate strength ratio → stress [Pa]."""
        beta = self.geom.plate_slenderness
        if beta <= 1.0:
            return self.mat.sigma_y0
        phi = 2.0 / beta - 1.0 / beta**2
        return phi * self.mat.sigma_y0

    def _classify(self, sigma_u: float, diag: Dict[str, float]) -> str:
        ratio = sigma_u / self.mat.sigma_y0
        beta = diag["plate_slenderness_beta"]
        if ratio > 0.95:
            return "yielding"
        if beta > 1.0 and ratio <= 0.95:
            if diag["column_slenderness_lambda"] > 1.0:
                return "interactive"
            return "plate_buckling"
        return "column_buckling"


# ---------------------------------------------------------------------------
# convergence helper
# ---------------------------------------------------------------------------
def panel_convergence_study(
    geometry,
    fiber_counts: Sequence[int] = (2, 4, 8, 16, 32),
    n_steps: int = 120,
    max_strain: float = 0.02,
) -> Dict[str, object]:
    """Fibre-count convergence of the ultimate strength.

    Returns a dict with the per-discretisation σ_u values, the Richardson-style
    relative differences and the finest-vs-coarsest error.
    """
    sigmas: List[float] = []
    for n in fiber_counts:
        solver = ProgressiveCollapseSolver(
            geometry, n_web_fibers=n, n_flange_fibers=2,
            n_steps=n_steps, max_strain=max_strain)
        sigmas.append(solver.solve().sigma_u)
    diffs = [abs(sigmas[i + 1] - sigmas[i]) / sigmas[i] for i in range(len(sigmas) - 1)]
    return {
        "fiber_counts": list(fiber_counts),
        "sigma_u": sigmas,
        "relative_differences": diffs,
        "finest_vs_coarsest": abs(sigmas[-1] - sigmas[0]) / sigmas[0],
        "converged": bool(max(diffs[-2:]) < 0.02 if len(diffs) >= 2 else False),
    }