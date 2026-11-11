"""Benchmark chain for the incremental Smith method: panel -> box -> hull girder.

Three rungs, each with a *pre-declared* reference that is computed
independently of the Smith solver:

Rung 1 - ``paik_panel`` (stiffened panel, beta-series)
    Plate slenderness ``beta`` is swept by scaling the plating thickness at
    fixed stiffener (FB 300x20 @ 800 mm, span 4.0 m) and fixed material.  The
    reference ultimate strength is the classical effective-width panel strength
    ``phi(beta) sigma_y`` (Faulkner 1975, re-used from the verified kernel) with
    the CSR-style min-of-modes closed form as a second, independent check.
    Purpose: validates the *element laws* (plate buckling + stiffener capacity)
    at the smallest scale, where no neutral-axis integration is involved.

Rung 2 - ``box_girder`` (unstiffened thin-walled box)
    Two variants:
    ``compact`` (beta < 1 everywhere) - the exact rigid-plastic limit load of a
    rectangular-plate section is an *analytic* reference
    (:func:`exact_box_plastic_modulus`), so the solver must reproduce
    ``M_u = sigma_y Z_p`` to within a couple of percent and must identify a
    pure yielding mode;
    ``slender`` (beta ~ 3) - reference is the effective-width rigid-plastic
    moment (:func:`effective_width_ultimate_moment`) and the requirement
    ``M_y <= M_u <= M_p``.
    Purpose: validates *sectional integration and neutral-axis migration*.

Rung 3 - ``issc2009_double_bottom`` (double-bottom hull-girder box)
    A 20 m x 2.4 m double bottom (AH32, bottom 18 mm + FB 350x20, inner bottom
    15 mm + FB 300x17, hopper sides and three bottom girders), analysed in
    sagging and hogging.  References: effective-width moment, elastic
    first-yield moment, and the idealisation-independent plastic moment.
    Purpose: validates the *cross-scale scale-up* plus grid convergence and
    failure-mode attribution at hull-girder scale.

Provenance and honesty statement
--------------------------------
The ISSC 2009 double-bottom benchmark coordinates and the Dow (1991) box-girder
test ultimate moments are **not shipped with this offline worktree**, hence
they are not quoted: quoting them from memory would be fabrication.  The two
geometries used here are therefore *reconstructed* from the published
structural classes (VLCC double bottom scantlings; thin-walled box girder test
girders) and every numeric comparison in the acceptance gates is made against a
reference that this module computes and documents (analytic plastic limit load,
Faulkner effective width, published empirical panel formula).  The
``WITHHELD`` items are listed in ``docs/smith_chain/report.md``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, Iterable, List, Sequence, Tuple

from ..elastoplastic_collapse import VonMisesMaterial
from .idealisation import GirderSection, PlatePanel, StiffenerGroup, fibre_arrays
from .laws import (
    column_strength,
    faulkner_plate_strength_ratio,
    plate_slenderness,
    plate_strength,
    stiffener_strength,
)

__all__ = [
    "AH32",
    "MILD_STEEL",
    "paik_panel",
    "paik_panel_series",
    "box_girder",
    "issc2009_double_bottom",
    "paik_thayamballi_panel_strength",
    "csr_min_of_modes_panel_strength",
    "effective_width_ultimate_moment",
    "exact_box_plastic_modulus",
]

AH32 = VonMisesMaterial(E=2.06e11, nu=0.3, sigma_y0=315.0e6, H=0.0)
MILD_STEEL = VonMisesMaterial(E=2.06e11, nu=0.3, sigma_y0=250.0e6, H=0.0)


# ---------------------------------------------------------------------------
# generic web idealisation helper
# ---------------------------------------------------------------------------
def _web_panels(
    prefix: str,
    z_bottom: float,
    height: float,
    n_strips: int,
    total_thickness: float,
    span_a: float,
    buckling_width: float | None = None,
    label: str = "",
) -> List[PlatePanel]:
    """Idealise ``n_strips`` transverse webs as axial strips along the depth.

    ``total_thickness`` is the summed thickness of all webs represented by this
    group; the area of every strip is ``total_thickness * height / n_strips``.
    The buckling width defaults to the full unsupported depth ``height`` (one
    panel), and ``span_a`` is the frame spacing in the load direction.
    """
    if n_strips < 1:
        raise ValueError("n_strips must be >= 1")
    dz = height / n_strips
    b = float(buckling_width if buckling_width is not None else height)
    out: List[PlatePanel] = []
    for i in range(n_strips):
        out.append(
            PlatePanel(
                id=f"{prefix}_{i}",
                z=z_bottom + (i + 0.5) * dz,
                width=dz,
                thickness=total_thickness,
                spacing_b=b,
                span_a=span_a,
                label=label,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Rung 1 - stiffened panel, beta series
# ---------------------------------------------------------------------------
def paik_panel(
    beta: float = 2.0,
    span_a: float = 4.0,
    spacing_b: float = 0.8,
    material: VonMisesMaterial = AH32,
    h: float = 0.30,
    t_w: float = 0.020,
) -> GirderSection:
    """Single stiffener-plate unit with prescribed plate slenderness ``beta``.

    The plating thickness follows from ``beta = (b/t) sqrt(sigma_y/E)`` so the
    series is a pure plate-slenderness sweep at fixed stiffener.  The section
    depth is the profile depth; the panel is analysed with
    ``loading="axial"`` (uniform shortening), not in bending.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    t = (spacing_b / beta) * math.sqrt(material.sigma_y0 / material.E)
    plate = PlatePanel(
        id="plate", z=0.5 * t, width=spacing_b, thickness=t,
        spacing_b=spacing_b, span_a=span_a, label="panel plating",
    )
    stiff = StiffenerGroup(
        id="stiffener", z_attach=0.5 * t, n=1, h=h, t_w=t_w,
        spacing_b=spacing_b, span_a=span_a, plate_id="plate",
        plate_t=t, label="FB %.0fx%.0f" % (h * 1e3, t_w * 1e3),
    )
    return GirderSection(
        name=f"paik_panel_beta{beta:.2f}", depth=h, plates=(plate,),
        stiffeners=(stiff,), material=material,
        label="stiffened panel beta-series, axial compression",
    )


def paik_panel_series(
    betas: Sequence[float] = (1.5, 2.0, 2.5, 3.0), **kwargs
) -> Dict[float, GirderSection]:
    return {float(b): paik_panel(float(b), **kwargs) for b in betas}


def paik_thayamballi_panel_strength(beta: float, lam: float) -> float:
    """Paik & Thayamballi (2003) empirical stiffened-panel strength ratio.

    ``sigma_u/sigma_y = 1 / (0.995 + 0.936 lam^2 + 0.17 beta^2)`` with ``lam``
    the column slenderness and ``beta`` the plate slenderness.

    .. warning::
       The coefficients are transcribed from the published empirical fit; the
       source table could not be re-read in this offline worktree, so this
       formula is reported **for reference only** and is *not* part of any
       acceptance gate (see ``docs/smith_chain/report.md``).
    """
    if beta <= 0.0 or lam <= 0.0:
        raise ValueError("beta and lam must be positive")
    return float(1.0 / (0.995 + 0.936 * lam**2 + 0.17 * beta**2))


def csr_min_of_modes_panel_strength(
    beta: float,
    h: float,
    t_w: float,
    span: float,
    spacing_b: float,
    plate_t: float,
    material: VonMisesMaterial = AH32,
) -> Dict[str, float]:
    """CSR-style min-of-modes closed form for a stiffened panel.

    The governing stress is the minimum of the three design modes computed from
    *the same verified strength laws used by the solver's element laws*:

    ``plate``   ``phi(beta) sigma_y`` (Falkner/Johnstone-Faulkner),
    ``column``  Johnson-Ostenfeld Euler stress of the profile with attached
                plating of effective breadth ``phi b``,
    ``trip``    Johnson-Ostenfeld flat-bar tripping stress.

    It is a *method-level* companion (min-of-modes vs progressive averaging by
    the Smith solver), documented as such; CSR tabulated coefficients are not
    transcribed here.
    """
    strengths = stiffener_strength(
        h=h, t_w=t_w, b_f=0.0, t_f=0.0, span=span, spacing_b=spacing_b,
        plate_t=plate_t, beta=beta, material=material,
    )
    phi = faulkner_plate_strength_ratio(beta)
    ratio = min(phi, strengths["sigma_col"] / material.sigma_y0,
                strengths["sigma_trip"] / material.sigma_y0)
    return {
        "sigma_u_over_sigma_y": float(ratio),
        "plate_ratio": float(phi),
        "column_ratio": float(strengths["sigma_col"] / material.sigma_y0),
        "tripping_ratio": float(strengths["sigma_trip"] / material.sigma_y0),
        "governing_mode": min(
            {"plate": phi,
             "column": strengths["sigma_col"] / material.sigma_y0,
             "trip": strengths["sigma_trip"] / material.sigma_y0}.items(),
            key=lambda kv: kv[1],
        )[0],
        "lambda_column": float(strengths["lambda_column"]),
    }


# ---------------------------------------------------------------------------
# Rung 2 - unstiffened box girder
# ---------------------------------------------------------------------------
def box_girder(
    variant: str = "slender",
    n_web_strips: int = 4,
    material: VonMisesMaterial | None = None,
) -> GirderSection:
    """Unstiffened single-cell box girder (thin-walled, rectangular).

    ``compact``  B = 0.50 m, H = 0.40 m, t = 20 mm  -> beta ~ 0.87 (no buckling)
    ``slender``  B = 1.20 m, H = 0.90 m, t = 12 mm  -> beta ~ 3.5  (plate buckling)

    Diaphragm spacing (panel length in the load direction) 4.0 m.  The webs are
    idealised as ``n_web_strips`` axial strips distributed over the depth, which
    is the refinement parameter of the grid study.
    """
    if variant not in ("compact", "slender"):
        raise ValueError("variant must be 'compact' or 'slender'")
    mat = material or MILD_STEEL
    if variant == "compact":
        B, H, t = 0.50, 0.40, 0.020
    else:
        B, H, t = 1.20, 0.90, 0.012
    L = 4.0
    plates: List[PlatePanel] = [
        PlatePanel(id="bottom_flange", z=0.5 * t, width=B, thickness=t,
                   spacing_b=B, span_a=L, label="compression/tension flange"),
        PlatePanel(id="top_flange", z=H - 0.5 * t, width=B, thickness=t,
                   spacing_b=B, span_a=L, label="compression/tension flange"),
    ]
    plates += _web_panels(
        prefix="webs", z_bottom=t, height=H - 2.0 * t, n_strips=n_web_strips,
        total_thickness=2.0 * t, span_a=L, buckling_width=H - 2.0 * t,
        label="two side webs (summed thickness)",
    )
    return GirderSection(
        name=f"box_girder_{variant}_{B:.2f}x{H:.2f}x{t*1e3:.0f}mm",
        depth=H, plates=tuple(plates), stiffeners=(), material=mat,
        label="unstiffened single-cell box girder in bending",
    )


def _rect_plastic_integral(z1: float, z2: float, t: float, z_p: float) -> Tuple[float, float]:
    """``(area, integral of t|z-z_p| dz)`` for a rectangle ``[z1, z2] x t``."""
    area = t * (z2 - z1)
    if z_p <= z1:
        return area, t * (0.5 * (z2**2 - z1**2) - z_p * (z2 - z1))
    if z_p >= z2:
        return area, t * (z_p * (z2 - z1) - 0.5 * (z2**2 - z1**2))
    a1, a2 = z_p - z1, z2 - z_p
    return area, t * (0.5 * a1**2 + 0.5 * a2**2)


def exact_box_plastic_modulus(B: float, H: float, t: float, n_iter: int = 200) -> float:
    """Exact plastic modulus ``Z_p`` of a rectangular single-cell box [m^3].

    Analytic integration over the true rectangles (bottom flange ``[0, t]``, top
    flange ``[H-t, H]``, two webs ``[t, H-t]`` of combined thickness ``2t``),
    with the equal-area plastic neutral axis found by bisection.  This is
    independent of the fibre discretisation used by the Smith solver.
    """
    rects = [(0.0, t, B), (H - t, H, B), (t, H - t, 2.0 * t)]
    lo, hi = 0.0, H
    for _ in range(n_iter):
        zp = 0.5 * (lo + hi)
        A_above = sum(t_ * max(0.0, z2 - max(z1, zp)) for z1, z2, t_ in rects)
        A_below = sum(t_ * max(0.0, min(z2, zp) - z1) for z1, z2, t_ in rects)
        if A_above > A_below:
            lo = zp
        else:
            hi = zp
    zp = 0.5 * (lo + hi)
    return float(sum(_rect_plastic_integral(z1, z2, t_, zp)[1] for z1, z2, t_ in rects))


# ---------------------------------------------------------------------------
# Rung 3 - double bottom hull girder
# ---------------------------------------------------------------------------
def issc2009_double_bottom(
    n_web_strips: int = 4,
    material: VonMisesMaterial = AH32,
    breadth: float = 20.0,
    height: float = 2.4,
    frame_spacing: float = 4.8,
) -> GirderSection:
    """Idealised double-bottom box section (ISSC2009-type hull-girder benchmark).

    ======================  ==========  =========  ==========  =================
    component               thickness   spacing    z (mid)     stiffening
    ======================  ==========  =========  ==========  =================
    bottom shell            18 mm       0.8 m      0.009 m     FB 350x20 @ 0.8 m
    inner bottom            15 mm       0.8 m      2.3925 m    FB 300x17 @ 0.8 m
    hopper sides (2)        14 mm       --         distributed --
    bottom girders (3)      12 mm       --         distributed --
    ======================  ==========  =========  ==========  =================

    Material AH32 (sigma_y = 315 MPa).  The geometry is *reconstructed* from the
    published VLCC double-bottom structural class; the original ISSC 2009
    coordinates are not shipped in this worktree (see module docstring).
    """
    t_bot, t_ib, t_side, t_gird = 0.018, 0.015, 0.014, 0.012
    n_long_bot = int(round(breadth / 0.8))
    plates: List[PlatePanel] = [
        PlatePanel(id="bottom_shell", z=0.5 * t_bot, width=breadth, thickness=t_bot,
                   spacing_b=0.8, span_a=frame_spacing, label="bottom shell plating"),
        PlatePanel(id="inner_bottom", z=height - 0.5 * t_ib, width=breadth,
                   thickness=t_ib, spacing_b=0.8, span_a=frame_spacing,
                   label="inner bottom plating"),
    ]
    plates += _web_panels(
        prefix="hopper", z_bottom=t_bot, height=height - t_bot - t_ib,
        n_strips=n_web_strips, total_thickness=2.0 * t_side, span_a=frame_spacing,
        buckling_width=height - t_bot - t_ib, label="double-hull side webs",
    )
    plates += _web_panels(
        prefix="girder", z_bottom=t_bot, height=height - t_bot - t_ib,
        n_strips=n_web_strips, total_thickness=3.0 * t_gird, span_a=frame_spacing,
        buckling_width=height - t_bot - t_ib, label="bottom girders (3)",
    )
    stiffeners = (
        StiffenerGroup(
            id="bottom_longitudinal", z_attach=0.5 * t_bot, n=n_long_bot, h=0.35,
            t_w=0.020, spacing_b=0.8, span_a=frame_spacing, plate_id="bottom_shell",
            plate_t=t_bot, label="FB 350x20 bottom longitudinals",
        ),
        StiffenerGroup(
            id="inner_bottom_longitudinal", z_attach=height - 0.5 * t_ib,
            n=n_long_bot, h=0.30, t_w=0.017, spacing_b=0.8, span_a=frame_spacing,
            plate_id="inner_bottom", plate_t=t_ib,
            label="FB 300x17 inner-bottom longitudinals",
        ),
    )
    return GirderSection(
        name="issc2009_double_bottom", depth=height, plates=tuple(plates),
        stiffeners=stiffeners, material=material,
        label="ISSC2009-type double-bottom box section (reconstructed scantlings)",
    )


# ---------------------------------------------------------------------------
# independent closed-form references
# ---------------------------------------------------------------------------
def effective_width_ultimate_moment(
    section: GirderSection, bending: str = "sagging"
) -> Dict[str, object]:
    """Rigid-plastic effective-width ultimate moment [N m].

    Compression-side plating is reduced by the Faulkner strength ``phi(beta)``,
    compression-side stiffeners are capped at their column/tripping capacity,
    the tension side is taken at yield.  The equal-force "plastic" neutral axis
    of the reduced section is found by bisection and

    ``M_EW = sum_e sigma_e A_e |z_e - z_NA|``.

    This is the classical closed-form companion of the Smith analysis (same
    element strengths, single-step collapse instead of progressive averaging),
    so a Smith/effective-width comparison is a *method-level* verification.
    """
    if bending not in ("sagging", "hogging"):
        raise ValueError("bending must be 'sagging' or 'hogging'")
    mat = section.material
    sy = mat.sigma_y0
    items: List[Dict[str, object]] = []
    for p in section.plates:
        beta = plate_slenderness(p.spacing_b, p.thickness, mat)
        items.append({"id": p.id, "z": p.z, "area": p.area, "kind": "plate",
                      "beta": beta, "sigma_c": plate_strength(beta, mat)})
    for s in section.stiffeners:
        plate_t = next(p.thickness for p in section.plates if p.id == s.plate_id)
        beta = plate_slenderness(s.spacing_b, plate_t, mat)
        st = stiffener_strength(h=s.h, t_w=s.t_w, b_f=s.b_f, t_f=s.t_f, span=s.span_a,
                                spacing_b=s.spacing_b, plate_t=plate_t, beta=beta,
                                material=mat)
        items.append({"id": s.id, "z": s.z_attach + 0.5 * s.h, "area": s.area,
                      "kind": "stiffener", "beta": beta, "sigma_c": st["sigma_cap"]})

    def side(z: float, z_na: float) -> str:
        comp_high = bending == "sagging"
        if z >= z_na:
            return "compression" if comp_high else "tension"
        return "tension" if comp_high else "compression"

    def force(z_na: float) -> float:
        """Signed axial force: compression positive, tension negative."""
        tot = 0.0
        for it in items:
            s = side(float(it["z"]), z_na)
            sig = float(it["sigma_c"]) if s == "compression" else sy
            tot += sig * float(it["area"]) * (1.0 if s == "compression" else -1.0)
        return tot

    lo = min(float(it["z"]) for it in items)
    hi = max(float(it["z"]) for it in items)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if force(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    z_na = 0.5 * (lo + hi)
    M = 0.0
    detail: List[Dict[str, object]] = []
    for it in items:
        s = side(float(it["z"]), z_na)
        sig = float(it["sigma_c"]) if s == "compression" else sy
        contrib = sig * float(it["area"]) * abs(float(it["z"]) - z_na)
        M += contrib
        detail.append({**it, "side": s, "sigma_MPa": sig / 1e6,
                       "lever_m": abs(float(it["z"]) - z_na),
                       "moment_MNm": contrib / 1e6})
    return {
        "method": "rigid-plastic effective-width (Faulkner phi on compression plating, "
                  "Johnson-Ostenfeld column/tripping cap on stiffeners)",
        "bending": bending,
        "z_NA_m": z_na,
        "M_u_Nm": float(M),
        "M_u_MNm": float(M / 1e6),
        "M_u_over_M_yield": float(M / max(section.first_yield_moment(), 1e-9)),
        "M_u_over_M_plastic": float(M / max(section.plastic_moment(), 1e-9)),
        "elements": detail,
    }


def effective_width_column_ratio(
    beta: float, lam: float
) -> float:
    """Ratio of the column mode to the plate mode (diagnostic helper)."""
    return float(column_strength(
        h=0.1, t_w=0.01, b_f=0.0, t_f=0.0, span=1.0,
        attached_width=faulkner_plate_strength_ratio(beta) * 1.0,
        plate_t=0.01, material=AH32)[0] / AH32.sigma_y0) if lam > 0 else float("nan")