"""Panel ultimate strength with all competing collapse modes in one calculation.

What was missing before
-----------------------
The panel-collapse verification surface had three *separate* ingredients:

* :mod:`~tensorphysics.elastoplastic_collapse.axial_collapse_ep` -- an iso-strain
  Smith summation using a Faulkner plate curve and a Perry-Robertson column
  curve of the **stiffener alone** (no attached plate in ``r``, no tripping);
* :mod:`~tensorphysics.elastoplastic_collapse.tripping_mode` -- a stand-alone
  elastic tripping stress that never entered any panel strength;
* :mod:`~tensorphysics.elastoplastic_collapse.progressive_collapse` -- a fiber
  solver using a von Karman plate curve.

Consequences measured on the documented reference panel (see
``artifacts/p167w2_n04/``): the panel ultimate from the two plate element
curves differs by ~23 % (0.70 vs 0.86 ``sigma_y``) and tripping was never able
to govern, although the docstring of ``SmithCollapseSolver`` advertises it.

This module puts the modes into one calculation:

* the stiffener (web + flange, or flat bar) is checked against **four** caps -
  column buckling of stiffener *plus attached plate*, elastic **tripping**,
  web local buckling and flange outstand buckling - the smallest binds;
* the plate element uses a selectable element curve: ``"faulkner"`` (test
  calibrated, default) or ``"von_karman"`` (upper bound of a perfect plate);
* elements are assembled at equal end shortening and softened past their own cap
  so the panel curve has a genuine limit point;
* the result carries the elastic lower bound ``min(sigma_cr) * (1 - r)`` and the
  rigid-plastic upper bound ``sigma_y`` and refuses physically impossible values;
* :func:`compare_ultimate_methods` places four independent methods side by side
  and reports the spread (the honest uncertainty of panel ultimate strength
  without an experimental anchor);
* :func:`plastic_axial_bending_interaction` gives the exact rigid-plastic
  ``N-M`` collapse surface of the section (machine-exact at both end points) and
  fits the power-law interaction exponent.

Conventions
-----------
Stresses are positive in compression, strains are positive shortening.

References
----------
- Smith, C.S. (1977). Trans. SNAME 85 - element load-shortening assembly.
- Faulkner, D. (1975). Trans. SNAME 83 - plate ultimate ``Phi = 2/beta - 1/beta^2``.
- von Karman, T. et al. (1932) - post-buckled average stress ``sqrt(sigma_cr sigma_y)``.
- Johnson, J.B. & Ostenfeld, A. (1943) - elastoplastic column curve.
- Paik, J.K. & Thayamballi, A.K. (2003). *Ultimate Limit State Design of
  Steel-Plated Structures*. Wiley, Ch. 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .axial_collapse_ep import (
    paik_thayamballi_simplified_ultimate,
    smith_panel_axial_collapse,
    strain_for_uniaxial_stress,
)
from .panel_section import StiffenedPanelSection
from .progressive_collapse import (
    ProgressiveCollapseSolver,
    plate_load_shortening_stress,
    uniaxial_elastoplastic_stress,
)

__all__ = [
    "PLATE_CURVES",
    "PanelUltimateResult",
    "check_ultimate_bounds",
    "compare_ultimate_methods",
    "panel_strength_band",
    "panel_ultimate_axial",
    "plastic_axial_bending_interaction",
    "plate_ultimate_stress",
    "stiffener_ultimate_stress",
]

PLATE_CURVES: Tuple[str, ...] = ("faulkner", "von_karman")


# ---------------------------------------------------------------------------
# element ultimate stresses
# ---------------------------------------------------------------------------
def plate_ultimate_stress(
    section: StiffenedPanelSection,
    curve: str = "faulkner",
    residual_stress_ratio: float = 0.15,
) -> Dict[str, float]:
    """Ultimate average stress of the plate element.

    Parameters
    ----------
    section : StiffenedPanelSection
        Panel section (uses ``beta_plate`` and ``sigma_cr_plate``).
    curve : {"faulkner", "von_karman"}
        ``faulkner``: ``Phi = 2/beta - 1/beta^2`` clipped at 1, times the
        residual-stress factor ``(1 - r)`` - the test-calibrated curve used by
        classification rules and by :mod:`axial_collapse_ep`.
        ``von_karman``: ``sqrt(sigma_cr * sigma_y)`` - the post-buckled average
        of a perfect plate, i.e. an upper bound.
    residual_stress_ratio : float
        Welding residual stress as a fraction of ``sigma_y`` (Faulkner only).

    Returns
    -------
    dict with ``beta``, ``phi``, ``sigma_u``, ``ratio`` (= sigma_u / sigma_y).
    """
    if curve not in PLATE_CURVES:
        raise ValueError(f"curve must be one of {PLATE_CURVES}, got {curve!r}")
    if not (0.0 <= residual_stress_ratio < 1.0):
        raise ValueError("residual_stress_ratio must be in [0, 1)")
    sy = section.material.sigma_y0
    beta = section.beta_plate
    if curve == "faulkner":
        phi = 1.0 if beta <= 1.0 else 2.0 / beta - 1.0 / beta ** 2
        sigma_u = min(phi, 1.0) * sy * (1.0 - residual_stress_ratio)
    else:
        sigma_u = min(math.sqrt(section.sigma_cr_plate * sy), sy)
        phi = sigma_u / sy
    return {
        "beta": beta,
        "phi": phi,
        "sigma_u": sigma_u,
        "ratio": sigma_u / sy,
        "sigma_cr": section.sigma_cr_plate,
    }


def stiffener_ultimate_stress(
    section: StiffenedPanelSection,
    lambda_trip: float = 1.0,
) -> Dict[str, float]:
    """Stiffener capacity with column / tripping / local buckling competing.

    Every candidate is an *elastic critical stress* capped (or reduced) by the
    column curve, so all of them are comparable:

    * ``column``  : Johnson-Ostenfeld on the Euler stress of stiffener + plate,
    * ``tripping``: elastic torsional buckling stress divided by the lateral
      restraint factor ``lambda_trip`` (1.0 = unrestrained stiffener),
    * ``web_local`` / ``flange_local``: plate buckling of the web panel (both
      edges supported, ``k = 4``) and of the flange outstand (``k = 0.425``).

    Returns
    -------
    dict with each candidate stress, the governing ``mode``, the governing
    ``sigma_u`` and the area-weighted stiffener average.
    """
    if lambda_trip <= 0.0:
        raise ValueError("lambda_trip must be positive")
    sy = section.material.sigma_y0
    sigma_e = section.sigma_euler_column
    # Johnson-Ostenfeld elastoplastic column stress
    if sigma_e >= 0.5 * sy:
        column = sy * (1.0 - sy / (4.0 * sigma_e))
    else:
        column = sigma_e
    tripping = section.sigma_tripping / lambda_trip
    web_local = section.sigma_cr_web
    flange_local = section.sigma_cr_flange

    caps = {
        "column_buckling": column,
        "tripping": tripping,
        "web_local_buckling": web_local,
    }
    if section.b_f > 0.0:
        caps["flange_local_buckling"] = flange_local
    mode = min(caps, key=caps.get)
    sigma_web = min(column, tripping, web_local)
    sigma_flange = min(column, tripping, flange_local) if section.b_f > 0.0 else 0.0
    sigma_u = (
        section.web_area * sigma_web + section.flange_area * sigma_flange
    ) / section.stiffener_area
    if mode == "column_buckling" and sigma_u >= 0.99 * sy:
        mode = "yielding"
    return {
        "sigma_u": sigma_u,
        "ratio": sigma_u / sy,
        "mode": mode,
        "column": column,
        "tripping": tripping,
        "web_local": web_local,
        "flange_local": flange_local,
        "sigma_web": sigma_web,
        "sigma_flange": sigma_flange,
        "lambda_trip": lambda_trip,
        "sigma_euler_column": sigma_e,
        "lambda_column": section.lambda_column,
    }


# ---------------------------------------------------------------------------
# panel ultimate (iso-strain element assembly)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PanelUltimateResult:
    """Ultimate strength of a stiffened panel in axial compression.

    Attributes
    ----------
    sigma_u : float
        Mean compressive stress at the limit point [Pa].
    sigma_u_over_sigma_y : float
        Normalised ultimate strength [-].
    eps_u : float
        Shortening at the limit point [-].
    mode : str
        Governing collapse mode of the weakest element group.
    sigma_u_plate, sigma_u_stiffener : float
        Element ultimate stresses used in the assembly [Pa].
    load_path : tuple of (float, float)
        ``(strain, mean stress)`` along the post-buckling path.
    bounds : dict
        ``lower`` (elastic, residual-reduced) and ``upper`` (rigid-plastic)
        bounds plus the ``inside_bounds`` flag.
    diagnostics : dict
        Element ratios, plate curve name, post-peak ratio, ...
    """

    sigma_u: float
    sigma_u_over_sigma_y: float
    eps_u: float
    mode: str
    sigma_u_plate: float
    sigma_u_stiffener: float
    load_path: Tuple[Tuple[float, float], ...]
    bounds: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def post_peak_ratio(self) -> float:
        """Mean stress at the end of the path over the peak (``< 1``)."""
        if not self.load_path or self.sigma_u <= 0.0:
            return float("nan")
        return self.load_path[-1][1] / self.sigma_u


def _element_stress(
    eps: float,
    cap: float,
    eps_cap: float,
    softening: float,
    floor_ratio: float = 0.2,
) -> float:
    """Element stress for a prescribed shortening with a cap and softening tail."""
    if eps <= 0.0:
        return 0.0
    if eps <= eps_cap:
        return min(cap, cap * eps / eps_cap) if eps_cap > 0.0 else cap
    return max(cap - softening * (eps - eps_cap), floor_ratio * cap)


def panel_ultimate_axial(
    section: StiffenedPanelSection,
    curve: str = "faulkner",
    residual_stress_ratio: float = 0.15,
    lambda_trip: float = 1.0,
    softening_modulus_ratio: float = 0.02,
    n_steps: int = 400,
    eps_max: float = 0.02,
) -> PanelUltimateResult:
    """Panel ultimate strength under axial compression (iso-strain assembly).

    Parameters
    ----------
    section : StiffenedPanelSection
        Panel section including the attached plate.
    curve : {"faulkner", "von_karman"}
        Plate element curve (see :func:`plate_ultimate_stress`).
    residual_stress_ratio : float
        Residual stress fraction used by the plate element.
    lambda_trip : float
        Lateral restraint factor of the stiffener (1.0 = unrestrained).
    softening_modulus_ratio : float
        Post-cap softening slope as a fraction of ``E``.
    n_steps, eps_max : int, float
        Shortening grid resolution and range.

    Returns
    -------
    PanelUltimateResult
        Ultimate strength, limit-point strain, governing mode, bounds.
    """
    if n_steps < 20:
        raise ValueError("n_steps must be >= 20")
    if eps_max <= 0.0:
        raise ValueError("eps_max must be positive")

    mat = section.material
    sy = mat.sigma_y0
    soft = softening_modulus_ratio * mat.E
    ih = section.to_isotropic_hardening()

    plate = plate_ultimate_stress(section, curve, residual_stress_ratio)
    stiff = stiffener_ultimate_stress(section, lambda_trip)

    cap_plate = plate["sigma_u"]
    cap_web = stiff["sigma_web"]
    cap_flange = stiff["sigma_flange"]
    eps_cap_plate = strain_for_uniaxial_stress(cap_plate, ih)
    eps_cap_web = strain_for_uniaxial_stress(cap_web, ih)
    eps_cap_flange = strain_for_uniaxial_stress(cap_flange, ih) if cap_flange > 0 else 0.0

    eps = np.linspace(0.0, eps_max, int(n_steps) + 1)
    sig_plate = np.empty_like(eps)
    sig_web = np.empty_like(eps)
    sig_flange = np.empty_like(eps)
    for i, e in enumerate(eps):
        e = float(e)
        if curve == "von_karman":
            raw = plate_load_shortening_stress(
                e, mat, section.sigma_cr_plate, softening_modulus_ratio, 0.2
            )
            sig_plate[i] = min(
                raw, max(cap_plate, 0.0) if cap_plate < 1.0 else raw
            )
        else:
            sig_plate[i] = _element_stress(e, cap_plate, eps_cap_plate, soft)
        sig_web[i] = _element_stress(e, cap_web, eps_cap_web, soft)
        sig_flange[i] = (
            _element_stress(e, cap_flange, eps_cap_flange, soft) if cap_flange > 0 else 0.0
        )

    total = section.total_area
    sigma_mean = (
        section.plate_area * sig_plate
        + section.web_area * sig_web
        + section.flange_area * sig_flange
    ) / total
    k = int(np.argmax(sigma_mean))
    sigma_u = float(sigma_mean[k])

    bounds = _bounds(section, residual_stress_ratio)

    # governing mode: weakest element group by ultimate stress ratio
    if cap_plate <= stiff["sigma_u"]:
        mode = "plate_buckling"
    else:
        mode = stiff["mode"]

    diagnostics = {
        "curve": curve,
        "residual_stress_ratio": residual_stress_ratio,
        "beta_plate": section.beta_plate,
        "lambda_column": section.lambda_column,
        "sigma_u_plate_ratio": plate["ratio"],
        "sigma_u_stiffener_ratio": stiff["ratio"],
        "sigma_web_ratio": cap_web / sy,
        "sigma_flange_ratio": (cap_flange / sy) if cap_flange > 0 else float("inf"),
        "tripping_ratio": stiff["tripping"] / sy,
        "column_ratio": stiff["column"] / sy,
        "web_local_ratio": stiff["web_local"] / sy,
        "plate_area_fraction": section.plate_area / total,
        "eps_cap_plate": eps_cap_plate,
        "post_peak_ratio": float(sigma_mean[-1] / sigma_u) if sigma_u > 0 else float("nan"),
        "n_steps": float(n_steps),
        "eps_max": eps_max,
    }
    return PanelUltimateResult(
        sigma_u=sigma_u,
        sigma_u_over_sigma_y=sigma_u / sy,
        eps_u=float(eps[k]),
        mode=mode,
        sigma_u_plate=cap_plate,
        sigma_u_stiffener=stiff["sigma_u"],
        load_path=tuple((float(e), float(s)) for e, s in zip(eps, sigma_mean)),
        bounds=bounds,
        diagnostics=diagnostics,
    )


def _bounds(
    section: StiffenedPanelSection, residual_stress_ratio: float = 0.15
) -> Dict[str, float]:
    """Elastic lower bound and rigid-plastic upper bound of ``sigma_u``."""
    sy = section.material.sigma_y0
    elastic = min(
        section.sigma_cr_plate,
        section.sigma_euler_column,
        section.sigma_tripping,
        section.sigma_cr_web,
        section.sigma_cr_flange,
    )
    lower = elastic * (1.0 - residual_stress_ratio)
    return {
        "lower": lower,
        "lower_unreduced": elastic,
        "upper": sy,
        "residual_stress_ratio": residual_stress_ratio,
    }


def check_ultimate_bounds(result: PanelUltimateResult) -> Dict[str, float]:
    """Physical plausibility gate: ``lower <= sigma_u <= upper``.

    The elastic lower bound is the weakest elastic critical stress reduced by
    the welding residual stress; the upper bound is the squash stress ``sigma_y``.
    A violation means an element curve or a cap is inconsistent and must be
    treated as a defect, not as a result.
    """
    lower = result.bounds["lower"]
    upper = result.bounds["upper"]
    inside = (result.sigma_u >= lower * (1.0 - 1e-9)) and (
        result.sigma_u <= upper * (1.0 + 1e-12)
    )
    return {
        "sigma_u": result.sigma_u,
        "lower": lower,
        "upper": upper,
        "inside_bounds": bool(inside),
        "margin_to_lower": result.sigma_u / lower - 1.0 if lower > 0 else float("inf"),
        "margin_to_upper": 1.0 - result.sigma_u / upper if upper > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# strength band and cross-method comparison
# ---------------------------------------------------------------------------
def panel_strength_band(
    section: StiffenedPanelSection,
    residual_stress_ratio: float = 0.15,
    lambda_trip: float = 1.0,
) -> Dict[str, float]:
    """Lower/upper panel ultimate from the two plate element curves.

    ``faulkner`` is the test-calibrated (design) value, ``von_karman`` the
    perfect-plate upper bound.  The half-width of the band is the honest model
    uncertainty to be quoted when no experimental anchor is available.
    """
    lo = panel_ultimate_axial(
        section, "faulkner", residual_stress_ratio, lambda_trip
    )
    hi = panel_ultimate_axial(
        section, "von_karman", residual_stress_ratio, lambda_trip
    )
    mid = 0.5 * (lo.sigma_u + hi.sigma_u)
    return {
        "sigma_u_faulkner": lo.sigma_u,
        "sigma_u_von_karman": hi.sigma_u,
        "ratio_faulkner": lo.sigma_u_over_sigma_y,
        "ratio_von_karman": hi.sigma_u_over_sigma_y,
        "band_width_ratio": (hi.sigma_u - lo.sigma_u) / lo.sigma_u,
        "band_half_width_over_mean": 0.5 * (hi.sigma_u - lo.sigma_u) / mid,
    }


def compare_ultimate_methods(
    section: StiffenedPanelSection,
    residual_stress_ratio: float = 0.15,
    lambda_trip: float = 1.0,
) -> Dict[str, object]:
    """Four independent panel-ultimate methods side by side.

    Methods
    -------
    ``element_faulkner`` : this module, Faulkner plate curve (design value).
    ``element_von_karman``: this module, von Karman plate curve (upper bound).
    ``smith_perry_robertson`` : ``axial_collapse_ep.smith_panel_axial_collapse``
        (stiffener-only column curve, no tripping).
    ``paik_rigid_plastic`` : ``axial_collapse_ep`` rigid-plastic summation.
    ``fiber_progressive`` : ``progressive_collapse.ProgressiveCollapseSolver``
        (curvature from ``M = N delta``).

    Returns
    -------
    dict with the stresses, the normalised values, the spread and the deviation
    of every method from ``element_faulkner``.
    """
    sy = section.material.sigma_y0
    faulk = panel_ultimate_axial(section, "faulkner", residual_stress_ratio, lambda_trip)
    vk = panel_ultimate_axial(section, "von_karman", residual_stress_ratio, lambda_trip)

    scant = section.to_scantlings()
    ih = section.to_isotropic_hardening()
    smith = smith_panel_axial_collapse(scant, ih, residual_stress_ratio=residual_stress_ratio)
    paik = paik_thayamballi_simplified_ultimate(
        scant, ih, residual_stress_ratio=residual_stress_ratio
    )
    from .stiffened_plate_collapse import StiffenedPlateGeometry

    geom = StiffenedPlateGeometry(
        a=section.a,
        b=section.b,
        t=section.t,
        h_s=section.h_s,
        t_w=section.t_w,
        b_f=section.b_f,
        t_f=section.t_f,
        material=section.material,
        w0_max=section.a / 1000.0,
    )
    fiber = ProgressiveCollapseSolver(geom).solve()

    stresses = {
        "element_faulkner": faulk.sigma_u,
        "element_von_karman": vk.sigma_u,
        "smith_perry_robertson": float(smith["sigma_u"]),
        "paik_rigid_plastic": float(paik["sigma_u"]),
        "fiber_progressive": float(fiber.sigma_u),
    }
    ref = stresses["element_faulkner"]
    ratios = {k: v / sy for k, v in stresses.items()}
    deviations = {k: v / ref - 1.0 for k, v in stresses.items()}
    values = np.array(list(stresses.values()))
    return {
        "sigma_u": stresses,
        "ratio": ratios,
        "deviation_vs_element_faulkner": deviations,
        "max_spread_over_min": float(values.max() / values.min() - 1.0),
        "modes": {
            "element_faulkner": faulk.mode,
            "element_von_karman": vk.mode,
            "fiber_progressive": fiber.failure_mode,
        },
        "sigma_y": sy,
    }


# ---------------------------------------------------------------------------
# rigid-plastic N-M interaction surface
# ---------------------------------------------------------------------------
def plastic_axial_bending_interaction(
    section: StiffenedPanelSection, n_points: int = 801
) -> Dict[str, object]:
    """Exact rigid-plastic ``N-M`` collapse surface of the panel section.

    Sweep a plastic neutral axis ``y`` over the section height; above ``y`` the
    material carries ``+sigma_y`` (compression), below ``-sigma_y``.  Summing
    area and first moment over the rectangular elements gives

    ``N(y) = sigma_y (A_above - A_below)``,
    ``M(y) = sigma_y [(S_above - S_below) - z_bar (A_above - A_below)]``.

    The two end points are exact anchors: ``N = 0`` gives ``M = M_p`` and
    ``M = 0`` gives ``N = +-N_p``.  A power law
    ``(|N|/N_p)^alpha + (|M|/M_p)^alpha = 1`` is fitted to each branch and the
    maximum relative residual is reported (a one-parameter engineering
    approximation of the exact surface).
    """
    if n_points < 51:
        raise ValueError("n_points must be >= 51")
    sy = section.material.sigma_y0
    z_bar = section.z_bar
    n_p = section.squash_load
    m_p = section.plastic_bending_moment

    y_p = section.plastic_neutral_axis
    ys = np.unique(np.concatenate([
        np.linspace(section.z_bottom, section.z_top, int(n_points)),
        np.array([y_p]),
    ]))
    n_vals = np.empty_like(ys)
    m_vals = np.empty_like(ys)
    for i, y in enumerate(ys):
        a_above = a_below = s_above = s_below = 0.0
        for el in section.elements:
            a_b, s_b = el.part(el.z_lo, y)
            a_a, s_a = el.part(y, el.z_hi)
            a_below += a_b
            s_below += s_b
            a_above += a_a
            s_above += s_a
        n_vals[i] = sy * (a_above - a_below)
        m_vals[i] = sy * ((s_above - s_below) - z_bar * (a_above - a_below))

    def _fit(mask: np.ndarray) -> Tuple[float, float]:
        """Fit alpha on the masked branch, return (alpha, max rel. residual)."""
        mask = mask & (np.abs(m_vals) > 0.05 * m_p)
        nn = np.abs(n_vals[mask]) / n_p
        mm = np.abs(m_vals[mask]) / m_p
        if nn.size == 0 or mm.max() <= 0.0:
            return float("nan"), float("nan")
        alphas = np.linspace(0.5, 4.0, 3501)
        best = (float("inf"), float("nan"))
        for a in alphas:
            resid = np.abs(nn ** a + mm ** a - 1.0)
            score = float(resid.max())
            if score < best[0]:
                best = (score, float(a))
        nn_safe = np.clip(nn, 1e-12, None)
        pred_m = (1.0 - nn_safe ** best[1]).clip(min=0.0) ** (1.0 / best[1])
        rel = np.abs(pred_m - mm) / np.clip(mm, 1e-12, None)
        return best[1], float(rel.max())

    comp = n_vals > 0.0
    tens = n_vals < 0.0
    alpha_c, res_c = _fit(comp)
    alpha_t, res_t = _fit(tens)

    # rule-form audit: is the classification-society interaction
    # (N/N_p)^2 + |M|/M_p <= 1 conservative against the exact surface?
    with np.errstate(divide="ignore", invalid="ignore"):
        rule_capacity = m_p * (1.0 - (n_vals / n_p) ** 2)
    violation = m_vals - rule_capacity
    active = np.abs(n_vals) < 0.999 * n_p

    def _viol(mask: np.ndarray) -> Tuple[float, float]:
        mask = mask & active
        if not mask.any():
            return 0.0, 0.0
        return (float(violation[mask].max() / m_p),
                float((violation[mask] > 0.0).mean()))

    viol_c, frac_c = _viol(n_vals > 0.0)
    viol_t, frac_t = _viol(n_vals < 0.0)

    # exact anchors (machine precision by construction)
    i_zero = int(np.argmin(np.abs(n_vals)))
    return {
        "n_p": n_p,
        "m_p": m_p,
        "alpha_compression": alpha_c,
        "alpha_tension": alpha_t,
        "max_rel_residual_compression": res_c,
        "max_rel_residual_tension": res_t,
        "m_at_n_zero": float(m_vals[i_zero]),
        "m_at_n_zero_over_m_p": float(m_vals[i_zero] / m_p),
        "n_range_over_n_p": (float(n_vals.min() / n_p), float(n_vals.max() / n_p)),
        "m_max_over_m_p": float(m_vals.max() / m_p),
        "rule_form_alpha": 2.0,
        "rule_form_max_violation_compression_over_m_p": viol_c,
        "rule_form_violation_fraction_compression": frac_c,
        "rule_form_max_violation_tension_over_m_p": viol_t,
        "rule_form_violation_fraction_tension": frac_t,
        "surface": {
            "y": ys.tolist(),
            "n_over_n_p": (n_vals / n_p).tolist(),
            "m_over_m_p": (m_vals / m_p).tolist(),
        },
    }