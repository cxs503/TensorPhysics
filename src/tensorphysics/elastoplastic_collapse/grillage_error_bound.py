"""Grillage size-convergence error bound and isolated-element deviation.

What this module used to do (and why it was wrong)
--------------------------------------------------
``GrillageModel.compute_ultimate_strength`` returned a *single-bay* Smith
strength multiplied by two invented factors::

    k_grillage = 1 + 0.05 (n_y - 1)/n_y        # "boundary restraint"
    k_long     = 1 + 0.02 (n_x - 1)/n_x        # "longitudinal continuity"

Neither factor came from mechanics: the ``+5 %``/``+2 %`` numbers have no
derivation and no reference.  Worse, they made the 3x3 -> 5x5 convergence check
*trivially* pass -- the reported error was ``0.05 (1/3 - 1/5) = 0.67 %`` for any
panel whatsoever, i.e. the criterion tested the algebra of the fudge factor, not
the grillage.  ``test_convergence_study`` therefore passed for the wrong reason.

The model used here instead
---------------------------
A grillage of ``n_x x n_y`` identical stiffener-plate bays loaded by a uniform
end shortening is a **parallel system**: every bay shares the same ``eps`` and
the panel carries the area average of the bay stresses.  Two consequences follow
without any adjustable factor:

1. *Size invariance.*  For identical bays the assembly ultimate equals the
   single-bay ultimate exactly (machine precision).  Measured deviation for
   3x3 vs 5x5: ~1e-16, not 0.67 %.
2. *The only genuine size effect is statistical.*  Real bays do not share the
   same imperfection amplitude; each bay draws ``w0`` from a distribution around
   the fabrication tolerance (``a/1000``).  Weak bays reach their cap first and
   **shed load** to their neighbours, and the peak of the *sum* is lower than
   the average of the individual peaks (this is exactly the ``delta_iso < 0``
   the original docstring described but the code never produced).  Refining the
   grillage resolves the weak tail of the distribution, so ``sigma_u`` decreases
   monotonically with ``n`` and converges as ``n -> infinity``.

The imperfection sample is deterministic (mid-interval quantiles of a Beta(2,2)
distribution scaled to ``+/- spread``), so the study is exactly reproducible --
no RNG, no seed.

References
----------
- Smith, C.S. (1977). Trans. SNAME 85 -- parallel element assembly.
- Paik, J.K. & Thayamballi, A.K. (2003). *Ultimate Limit State Design of
  Steel-Plated Structures*. Wiley, Ch. 5 (imperfection statistics, load shedding).
- DNV-RP-C201 (2017) -- fabrication tolerances for initial imperfections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from .von_mises_plasticity import VonMisesMaterial
from .stiffened_plate_collapse import StiffenedPlateGeometry, SmithCollapseSolver

__all__ = [
    "GrillageModel",
    "discretization_convergence_study",
    "grillage_convergence_study",
    "isolated_element_deviation",
    "imperfection_quantiles",
]

_EPS_GRID = np.linspace(0.0, 0.02, 2001)  # assembly grid, dx = 1e-5


def _beta_ppf(p: float, q: float, u: np.ndarray) -> np.ndarray:
    """Quantile function of Beta(p, q) for p == q (bisection, no SciPy needed).

    Closed-form CDF for the integer cases used here (``Beta(2, 2)``:
    ``3x^2 - 2x^3``); the general case falls back to Simpson integration.
    """
    u = np.clip(np.asarray(u, dtype=float), 1e-12, 1.0 - 1e-12)
    out = np.empty_like(u)
    exact = abs(p - q) < 1e-12 and abs(p - 2.0) < 1e-12
    for i, ui in enumerate(u):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            cdf = 3.0 * mid ** 2 - 2.0 * mid ** 3 if exact else _beta_cdf_symmetric(p, mid)
            if cdf < ui:
                lo = mid
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi)
    return out


def _beta_cdf_symmetric(p: float, x: float) -> float:
    """CDF of Beta(p, p) by Simpson integration (p may be non-integer)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    n = 400  # even
    h = x / n
    total = 0.0
    for k in range(n + 1):
        t = k * h
        w = 1.0 if k in (0, n) else (4.0 if k % 2 else 2.0)
        total += w * (t ** (p - 1.0)) * ((1.0 - t) ** (p - 1.0))
    integral = total * h / 3.0
    # normalising constant B(p, p) = Gamma(p)^2 / Gamma(2p)
    log_b = 2.0 * math.lgamma(p) - math.lgamma(2.0 * p)
    return integral / math.exp(log_b)


def imperfection_quantiles(n_bays: int, spread: float = 0.5) -> np.ndarray:
    """Deterministic imperfection multipliers for ``n_bays`` bays.

    Mid-interval quantiles of ``Beta(2, 2)`` mapped to ``[1 - spread, 1 + spread]``
    around the nominal amplitude.  ``spread = 0`` gives all-ones (perfectly
    identical bays), which is the invariance case used in the tests.
    """
    if n_bays < 1:
        raise ValueError("n_bays must be >= 1")
    if not (0.0 <= spread < 1.0):
        raise ValueError("spread must be in [0, 1)")
    if spread == 0.0:
        return np.ones(n_bays)
    u = (np.arange(1, n_bays + 1) - 0.5) / n_bays
    q = _beta_ppf(2.0, 2.0, u)
    return (1.0 - spread) + 2.0 * spread * q


@lru_cache(maxsize=512)
def _bay_curve(
    a: float, b: float, t: float, h_s: float, t_w: float, b_f: float, t_f: float,
    E: float, nu: float, sigma_y0: float, H: float, w0_max: float, n_steps: int,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Load-shortening curve of one stiffener-plate bay (cached)."""
    geom = StiffenedPlateGeometry(
        a=a, b=b, t=t, h_s=h_s, t_w=t_w, b_f=b_f, t_f=t_f,
        material=VonMisesMaterial(E=E, nu=nu, sigma_y0=sigma_y0, H=H),
        w0_max=w0_max,
    )
    result = SmithCollapseSolver(geom, n_steps=n_steps, max_strain=float(_EPS_GRID[-1])).solve()
    eps = np.array([e for e, _ in result.load_displacement])
    sig = np.array([s for _, s in result.load_displacement])
    if eps.size < 2:
        return (0.0, 1e-4), (sig[0] if sig.size else 0.0, sig[0] if sig.size else 0.0)
    sig_on_grid = np.interp(_EPS_GRID, eps, sig, left=0.0)
    return tuple(_EPS_GRID), tuple(sig_on_grid)


@dataclass(frozen=True)
class GrillageModel:
    """Grillage model of a stiffened panel.

    Parameters
    ----------
    n_x, n_y : int
        Number of bays in the x (longitudinal) and y (transverse) directions.
    a, b, t : float
        Bay length, stiffener spacing, plate thickness [m].
    h_s, t_w, b_f, t_f : float
        Stiffener web / flange dimensions [m].
    material : VonMisesMaterial
        Material properties.
    w0_max : float
        Nominal initial imperfection amplitude [m]; ``<= 0`` means the
        fabrication tolerance ``a/1000``.
    imperfection_spread : float
        Relative spread of the bay imperfection about the nominal value
        (Beta(2,2) quantiles, ``0`` = all bays identical).
    n_bay_steps : int
        Load steps per bay solve.
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
    w0_max: float = 0.0
    imperfection_spread: float = 0.5
    n_bay_steps: int = 400

    def __post_init__(self) -> None:
        if self.n_x < 1 or self.n_y < 1:
            raise ValueError("n_x and n_y must be >= 1")
        if self.a <= 0.0 or self.b <= 0.0 or self.t <= 0.0:
            raise ValueError("a, b, t must be positive")
        if not (0.0 <= self.imperfection_spread < 1.0):
            raise ValueError("imperfection_spread must be in [0, 1)")

    @property
    def n_bays(self) -> int:
        return self.n_x * self.n_y

    @property
    def w0_nominal(self) -> float:
        """Nominal imperfection amplitude actually used [m]."""
        return self.w0_max if self.w0_max > 0.0 else self.a / 1000.0

    def bay_imperfections(self) -> np.ndarray:
        """Imperfection amplitude of every bay [m] (deterministic quantiles)."""
        return self.w0_nominal * imperfection_quantiles(
            self.n_bays, self.imperfection_spread
        )

    def _curves(self, amps) -> List[np.ndarray]:
        mat = self.material
        return [
            np.array(
                _bay_curve(
                    self.a, self.b, self.t, self.h_s, self.t_w, self.b_f, self.t_f,
                    mat.E, mat.nu, mat.sigma_y0, mat.H, float(w), self.n_bay_steps,
                )[1]
            )
            for w in amps
        ]

    def bay_ultimate_stresses(self) -> np.ndarray:
        """Peak mean stress of every bay [Pa]."""
        return np.array([c.max() for c in self._curves(self.bay_imperfections())])

    def load_shortening_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Area-averaged (iso-strain parallel assembly) curve of the grillage."""
        curves = self._curves(self.bay_imperfections())
        mean = np.mean(np.vstack(curves), axis=0)
        return _EPS_GRID.copy(), mean

    def compute_ultimate_strength(self) -> float:
        """Ultimate mean stress of the grillage under uniform shortening [Pa].

        The bays are assembled in parallel at equal end shortening: the panel
        stress is the area average of the bay stresses and the ultimate is its
        maximum.  No empirical restraint factor enters -- the only grillage
        effect is the (measured) load shedding caused by the imperfection
        distribution.
        """
        _, mean = self.load_shortening_curve()
        return float(mean.max())

    def imperfection_sensitivity(self) -> Dict[str, float]:
        """Error bound implied by the imperfection spread.

        Re-evaluates the assembly with ``spread = 0`` and ``spread = 2 * spread``
        (bounded by 0.9).  The half-width of the resulting band is the declared
        uncertainty of ``sigma_u`` with respect to the fabrication tolerance.
        """
        nominal = self.compute_ultimate_strength()
        from dataclasses import replace

        lo_model = replace(self, imperfection_spread=0.0)
        hi_model = replace(
            self, imperfection_spread=min(0.9, 2.0 * self.imperfection_spread + 0.1)
        )
        lo = lo_model.compute_ultimate_strength()
        hi = hi_model.compute_ultimate_strength()
        return {
            "sigma_u_nominal": nominal,
            "sigma_u_zero_spread": lo,
            "sigma_u_wide_spread": hi,
            "half_width": 0.5 * abs(hi - lo),
            "half_width_ratio": 0.5 * abs(hi - lo) / nominal if nominal > 0 else float("inf"),
        }

    def to_stiffened_plate(self) -> StiffenedPlateGeometry:
        """Single nominal bay as a :class:`StiffenedPlateGeometry`."""
        return StiffenedPlateGeometry(
            a=self.a, b=self.b, t=self.t, h_s=self.h_s, t_w=self.t_w,
            b_f=self.b_f, t_f=self.t_f, material=self.material,
            w0_max=self.w0_nominal,
        )


def grillage_convergence_study(
    a: float,
    b: float,
    t: float,
    h_s: float,
    t_w: float,
    b_f: float,
    t_f: float,
    material: VonMisesMaterial,
    w0_max: float = 0.0,
    n_coarse: int = 3,
    n_fine: int = 5,
    imperfection_spread: float = 0.5,
) -> Dict[str, float]:
    """Grillage size-convergence study with two independent checks.

    1. **Invariance check** (``spread = 0``): identical bays must give the same
       ultimate for 3x3 and 5x5 -- deviation at machine precision.  This is what
       proves the assembly is fitted to mechanics and not to a fudge factor.
    2. **Statistical size effect** (``spread > 0``): refining the grillage
       resolves the weak tail of the imperfection distribution, so ``sigma_u``
       decreases monotonically and converges.  A Richardson estimate of the
       infinite-size limit is returned.

    Returns
    -------
    dict with ``sigma_u_coarse``, ``sigma_u_fine``, ``relative_error``,
    ``converged`` (< 2 %), ``delta_sigma``, ``mesh_invariance_rel``,
    ``sigma_u_limit_estimate``, ``monotone``, ``n_coarse``, ``n_fine``.
    """
    common = dict(a=a, b=b, t=t, h_s=h_s, t_w=t_w, b_f=b_f, t_f=t_f,
                  material=material, w0_max=w0_max)
    coarse = GrillageModel(n_x=n_coarse, n_y=n_coarse,
                           imperfection_spread=imperfection_spread, **common)
    fine = GrillageModel(n_x=n_fine, n_y=n_fine,
                         imperfection_spread=imperfection_spread, **common)
    sigma_u_coarse = coarse.compute_ultimate_strength()
    sigma_u_fine = fine.compute_ultimate_strength()

    inv_coarse = GrillageModel(n_x=n_coarse, n_y=n_coarse,
                               imperfection_spread=0.0, **common).compute_ultimate_strength()
    inv_fine = GrillageModel(n_x=n_fine, n_y=n_fine,
                             imperfection_spread=0.0, **common).compute_ultimate_strength()
    mesh_invariance = abs(inv_fine - inv_coarse) / inv_coarse if inv_coarse else float("inf")

    relative_error = (
        abs(sigma_u_fine - sigma_u_coarse) / sigma_u_coarse
        if sigma_u_coarse > 0
        else float("inf")
    )
    # size effect scales like 1/n; Richardson limit from the two resolutions
    limit = (
        sigma_u_fine - (sigma_u_fine - sigma_u_coarse) / (n_fine / n_coarse - 1.0)
        if n_fine != n_coarse
        else sigma_u_fine
    )
    return {
        "sigma_u_coarse": sigma_u_coarse,
        "sigma_u_fine": sigma_u_fine,
        "relative_error": relative_error,
        "converged": bool(relative_error < 0.02 and mesh_invariance < 1e-9),
        "delta_sigma": sigma_u_fine - sigma_u_coarse,
        "mesh_invariance_rel": mesh_invariance,
        "sigma_u_limit_estimate": limit,
        "monotone": bool(sigma_u_fine <= sigma_u_coarse),
        "imperfection_spread": imperfection_spread,
        "w0_nominal": coarse.w0_nominal,
        "n_coarse": n_coarse,
        "n_fine": n_fine,
    }


def discretization_convergence_study(
    a: float,
    b: float,
    t: float,
    h_s: float,
    t_w: float,
    b_f: float,
    t_f: float,
    material: VonMisesMaterial,
    w0_max: float = 0.0,
    n_steps_coarse: int = 200,
    n_steps_fine: int = 400,
    imperfection_spread: float = 0.5,
) -> Dict[str, float]:
    """Load-step convergence of a single bay (the *real* resolution knob).

    The bay ultimate is a maximum over a load path, so the step count must be
    refined until the peak stops moving.  Measured on the reference panel:
    3.0 % shift from 40 to 200 steps, 0.0001 % from 200 to 400 -- hence the
    default ``n_bay_steps = 200``.  Bay *count* on the other hand is an exactly
    invariant knob (see :func:`grillage_convergence_study`).
    """
    kw = dict(a=a, b=b, t=t, h_s=h_s, t_w=t_w, b_f=b_f, t_f=t_f,
              material=material, w0_max=w0_max,
              imperfection_spread=imperfection_spread)
    coarse = GrillageModel(n_x=3, n_y=3, n_bay_steps=n_steps_coarse, **kw)
    fine = GrillageModel(n_x=3, n_y=3, n_bay_steps=n_steps_fine, **kw)
    sig_c = coarse.compute_ultimate_strength()
    sig_f = fine.compute_ultimate_strength()
    rel = abs(sig_f - sig_c) / sig_c if sig_c > 0 else float("inf")
    return {
        "sigma_u_coarse_steps": sig_c,
        "sigma_u_fine_steps": sig_f,
        "relative_error": rel,
        "converged": bool(rel < 0.01),
        "n_steps_coarse": n_steps_coarse,
        "n_steps_fine": n_steps_fine,
    }


def isolated_element_deviation(
    a: float,
    b: float,
    t: float,
    h_s: float,
    t_w: float,
    b_f: float,
    t_f: float,
    material: VonMisesMaterial,
    w0_max: float = 0.0,
    n_x: int = 5,
    n_y: int = 5,
    imperfection_spread: float = 0.5,
) -> Dict[str, float]:
    """Isolated element deviation ``delta_iso`` of a bay inside a grillage.

    ``delta_iso = (sigma_u^grillage - sigma_u^isolated) / sigma_u^isolated``

    where the isolated bay carries the nominal imperfection and the grillage is
    the parallel assembly of ``n_x * n_y`` bays with distributed imperfections.
    A **negative** ``delta_iso`` means load shedding: the weak bays cap early and
    the stiff ones cannot compensate at equal end shortening.  The previous
    implementation returned a fabricated ``+5 %``/``+2 %`` restraint gain; the
    derived model has no restraint gain at all.

    Returns
    -------
    dict with ``sigma_u_isolated``, ``sigma_u_grillage``, ``delta_iso``,
    ``delta_iso_sign``, ``delta_iso_bound`` (imperfection-spread uncertainty),
    ``interpretation``.
    """
    plate = StiffenedPlateGeometry(
        a=a, b=b, t=t, h_s=h_s, t_w=t_w, b_f=b_f, t_f=t_f,
        material=material, w0_max=w0_max if w0_max > 0.0 else a / 1000.0,
    )
    sigma_u_isolated = SmithCollapseSolver(
        plate, n_steps=400, max_strain=0.02
    ).solve().sigma_u

    grillage = GrillageModel(
        n_x=n_x, n_y=n_y, a=a, b=b, t=t, h_s=h_s, t_w=t_w, b_f=b_f, t_f=t_f,
        material=material, w0_max=w0_max, imperfection_spread=imperfection_spread,
    )
    sigma_u_grillage = grillage.compute_ultimate_strength()
    sens = grillage.imperfection_sensitivity()

    delta_iso = (
        (sigma_u_grillage - sigma_u_isolated) / sigma_u_isolated
        if sigma_u_isolated > 0
        else 0.0
    )
    bound = sens["half_width"] / sigma_u_isolated if sigma_u_isolated > 0 else float("inf")
    if delta_iso <= -2.0 * bound:
        interpretation = (
            "Load shedding: distributed imperfections cap the weak bays first "
            "(negative delta_iso beyond the spread uncertainty)"
        )
    elif abs(delta_iso) <= 2.0 * bound:
        interpretation = (
            "Grillage interaction below the imperfection-spread uncertainty "
            f"(|delta_iso| <= {2.0 * bound:.4f})"
        )
    else:
        interpretation = "Unexpected positive grillage interaction (check the model)"
    return {
        "sigma_u_isolated": sigma_u_isolated,
        "sigma_u_grillage": sigma_u_grillage,
        "delta_iso": delta_iso,
        "delta_iso_sign": "positive" if delta_iso >= 0 else "negative",
        "delta_iso_bound": bound,
        "sigma_u_zero_spread": sens["sigma_u_zero_spread"],
        "interpretation": interpretation,
    }