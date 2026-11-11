"""Acceptance harness for the J2 elastoplastic beam/plate verification surface.

Every criterion is a comparison between the *model* (fiber-section integration
or the incremental beam FE) and an *independent* reference (closed form, limit
analysis, or an exact identity), never a self-assessment.  Thresholds are the
task acceptance criteria.

Criteria
--------
(a) Moment-curvature, three sections (rectangle / I-beam / circle):
    elastic slope vs ``E*I`` <= 2 %, first-yield moment vs ``sigma_y*W_el``
    <= 3 %, fully plastic moment vs ``sigma_y*Z`` <= 5 %.
(b) Plastic hinge location vs limit analysis <= 1 % of the span
    (propped cantilever UDL -> ``sqrt(2)/(1+sqrt(2)) = 0.585786``; simply
    supported beam with an eccentric point load -> ``x = 0.3 L``).
(c) Unloading springback ``kappa_res`` vs the elastic-unloading identity
    <= 5 % (rectangle: vs the closed-form ideal-plastic ``M(kappa)``).
(d) Energy balance ``W_ext = U_el + W_pl`` <= 1 % (section level and beam
    level, with and without plastic flow).
(e) Two negative controls that must FAIL: a section-height transcription error
    (+10 % height -> +21 % ``M_y``) and a fabricated hinge position (``L/2``
    instead of the limit-analysis value).

Returns JSON-serialisable data only, so that every number in the report is
machine-read from the artefact.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .beam_hinge import (
    FiberSection,
    J2UniaxialMaterial,
    elastic_deflection_point_load,
    moment_curvature,
    propped_cantilever_limit,
    rectangle_ideal_plastic_moment,
    section_energy_audit,
    simply_supported_point_load,
    propped_cantilever_udl,
    unload_from,
)
from .plate_limit import (
    circular_plate_mechanism,
    plastic_moment_per_width,
    rectangular_plate_mechanism,
    square_plate_limit,
    strip_plate_mechanism,
)

E_STEEL = 210.0e3  # MPa
SIGMA_Y = 235.0  # MPa
HARDENING_FE = 1.0e-3  # fraction of E, keeps the FE tangent regular

TOL_EI = 0.02
TOL_MY = 0.03
TOL_MP = 0.05
TOL_HINGE = 0.01
TOL_SPRINGBACK = 0.05
TOL_ENERGY = 0.01
TOL_PLATE = 0.001


def _rel_err(model: float, reference: float) -> float:
    if reference == 0.0:  # pragma: no cover - defensive
        return abs(model)
    return abs(model - reference) / abs(reference)


def _criterion(
    name: str,
    description: str,
    model: float,
    reference: float,
    error: float,
    tolerance: float,
    note: str = "",
    scale: str = "relative",
) -> dict:
    return {
        "name": name,
        "description": description,
        "model": float(model),
        "reference": float(reference),
        "error": float(error),
        "tolerance": float(tolerance),
        "scale": scale,
        "passed": bool(error <= tolerance * (1.0 + 1e-9) + 1e-15),
        "note": note,
    }


def bench_sections() -> dict[str, FiberSection]:
    """The three verification sections (mm)."""
    return {
        "rectangle": FiberSection.rectangle(100.0, 200.0),
        "ibeam": FiberSection.ibeam(300.0, 150.0, 15.0, 10.0),
        "circle": FiberSection.circle(100.0),
    }


def ideal_material() -> J2UniaxialMaterial:
    return J2UniaxialMaterial(E=E_STEEL, sigma_y0=SIGMA_Y, hardening_ratio=0.0)


def fe_material() -> J2UniaxialMaterial:
    return J2UniaxialMaterial(
        E=E_STEEL, sigma_y0=SIGMA_Y, hardening_ratio=HARDENING_FE
    )


# ---------------------------------------------------------------------------
# (a) moment-curvature
# ---------------------------------------------------------------------------


def verify_moment_curvature(sections: dict[str, FiberSection] | None = None) -> dict:
    sections = sections or bench_sections()
    material = ideal_material()
    criteria: list[dict] = []
    rows: dict[str, dict] = {}
    for key, section in sections.items():
        analytic = section.analytic()
        kappa_y = material.sigma_y0 / (material.E * section.half_height)
        curve = moment_curvature(
            section, material, kappa_max=20.0 * kappa_y, n_steps=240, n_sub=6
        )
        EI_model = curve.slope(kappa_max=0.5 * kappa_y)
        EI_ref = material.E * analytic["I"]
        idx_y = int(np.searchsorted(curve.kappa, kappa_y))
        M_y_model = float(curve.moment[idx_y])
        M_y_ref = material.sigma_y0 * analytic["W_el"]
        M_p_model = curve.tail_moment(0.02)
        M_p_ref = material.sigma_y0 * analytic["Z"]
        fiber = section.fiber_properties()
        rows[key] = {
            "label": section.label,
            "shape_factor_model": fiber["Z"] / fiber["W_el"],
            "shape_factor_reference": analytic["shape_factor"],
            "kappa_y": kappa_y,
            "EI_model": EI_model,
            "EI_reference": EI_ref,
            "M_y_model": M_y_model,
            "M_y_reference": M_y_ref,
            "M_p_model": M_p_model,
            "M_p_reference": M_p_ref,
            "kappa": [float(v) for v in curve.kappa[::6]],
            "moment": [float(v) for v in curve.moment[::6]],
            "moment_normalised": [
                float(v / M_p_ref) for v in curve.moment[::6]
            ],
            "kappa_normalised": [float(v / kappa_y) for v in curve.kappa[::6]],
        }
        criteria.append(
            _criterion(
                f"a_elastic_EI_{key}",
                f"elastic M-kappa slope vs E*I, {section.label}",
                EI_model,
                EI_ref,
                _rel_err(EI_model, EI_ref),
                TOL_EI,
            )
        )
        criteria.append(
            _criterion(
                f"a_yield_M_{key}",
                f"first-yield moment vs sigma_y*W_el, {section.label}",
                M_y_model,
                M_y_ref,
                _rel_err(M_y_model, M_y_ref),
                TOL_MY,
            )
        )
        criteria.append(
            _criterion(
                f"a_plastic_M_{key}",
                f"fully plastic moment vs sigma_y*Z, {section.label}",
                M_p_model,
                M_p_ref,
                _rel_err(M_p_model, M_p_ref),
                TOL_MP,
            )
        )
    return {"rows": rows, "criteria": criteria}


# ---------------------------------------------------------------------------
# (b) plastic hinge formation
# ---------------------------------------------------------------------------


def verify_hinge_location(fast: bool = False) -> dict:
    material = fe_material()
    sections = bench_sections()
    criteria: list[dict] = []
    cases: dict[str, dict] = {}

    # --- propped cantilever, uniform load (statically indeterminate) --------
    ibeam = sections["ibeam"]
    length = 6000.0
    M_p = material.sigma_y0 * ibeam.analytic()["Z"]
    limit = propped_cantilever_limit(M_p, length)
    n_el = 40 if fast else 60
    n_steps = 24 if fast else 40
    pc = propped_cantilever_udl(
        ibeam,
        material,
        length,
        n_elements=n_el,
        n_steps=n_steps,
        load_fraction=0.99,
    )
    hinge_model = float(pc.hinge["interior_centroid_position"])
    ratio_model = hinge_model / length
    ratio_ref = limit["hinge_position_ratio"]
    cases["propped_cantilever_udl"] = {
        "span": length,
        "n_elements": n_el,
        "n_steps": n_steps,
        "load_factor_final": pc.load_factor,
        "load_fraction_of_limit": pc.load_factor,
        "converged": pc.converged,
        "hinge_position_model": hinge_model,
        "hinge_ratio_model": ratio_model,
        "hinge_ratio_argmax_interior": [
            [float(p[0] / length), float(p[1])] for p in pc.hinge["interior_peaks"]
        ],
        "hinge_ratio_reference": ratio_ref,
        "limit_load_factor_reference": limit["load_factor"],
        "all_peaks_ratio": [float(p[0] / length) for p in pc.hinge["peaks"]],
        "x_over_L": [float(v / length) for v in pc.x_gauss],
        "plastic_curvature_normalised": [
            float(v / (material.sigma_y0 / (material.E * ibeam.half_height)))
            for v in pc.plastic_curvature
        ],
        "moment_normalised": [float(v / M_p) for v in pc.moment],
        "deflection": [float(v) for v in pc.deflection],
        "energy": pc.energy,
    }
    criteria.append(
        _criterion(
            "b_hinge_propped_cantilever",
            "interior plastic hinge position vs limit analysis sqrt(2)/(1+sqrt(2))",
            ratio_model,
            ratio_ref,
            abs(ratio_model - ratio_ref),
            TOL_HINGE,
            note="error measured as a fraction of the span",
            scale="span_fraction",
        )
    )
    criteria.append(
        _criterion(
            "b_converged_propped_cantilever",
            "incremental FE reached lambda = 0.99 of the limit load",
            1.0 if pc.converged and pc.load_factor > 0.98 else 0.0,
            1.0,
            0.0 if pc.converged and pc.load_factor > 0.98 else 1.0,
            0.0,
        )
    )

    # --- simply supported, eccentric point load (determinate) --------------
    n_el_ss = 60 if fast else 80
    ss = simply_supported_point_load(
        ibeam,
        material,
        length,
        load_ratio=0.3,
        n_elements=n_el_ss,
        n_steps=16 if fast else 20,
        load_fraction=0.99,
    )
    a = 0.3 * length
    P_c = M_p * length / (a * (length - a))
    analytic_defl = elastic_deflection_point_load(
        0.99 * P_c, a, length - a, length, material.E, ibeam.fiber_properties()["I"]
    )
    cases["simply_supported_point_load"] = {
        "span": length,
        "load_position_ratio": 0.3,
        "n_elements": n_el_ss,
        "hinge_ratio_model": float(ss.hinge["interior_centroid_ratio"]),
        "hinge_ratio_argmax": float(ss.hinge["ratio"]),
        "hinge_ratio_reference": 0.3,
        "converged": ss.converged,
        "load_factor_final": ss.load_factor,
        "deflection_at_load_model": float(ss.deflection[: n_el_ss + 1][
            int(round(0.3 * n_el_ss))
        ]),
        "deflection_elastic_reference": analytic_defl,
        "energy": ss.energy,
        "x_over_L": [float(v / length) for v in ss.x_gauss],
        "moment_normalised": [float(v / M_p) for v in ss.moment],
        "plastic_curvature_normalised": [
            float(v / (material.sigma_y0 / (material.E * ibeam.half_height)))
            for v in ss.plastic_curvature
        ],
    }
    criteria.append(
        _criterion(
            "b_hinge_simply_supported",
            "hinge position vs x = 0.3 L, simply supported with eccentric load",
            float(ss.hinge["interior_centroid_ratio"]),
            0.3,
            abs(float(ss.hinge["interior_centroid_ratio"]) - 0.3),
            TOL_HINGE,
            note="error measured as a fraction of the span",
            scale="span_fraction",
        )
    )
    return {"cases": cases, "criteria": criteria}


# ---------------------------------------------------------------------------
# (c) unloading / springback
# ---------------------------------------------------------------------------


def verify_springback(sections: dict[str, FiberSection] | None = None) -> dict:
    sections = sections or bench_sections()
    material = ideal_material()
    criteria: list[dict] = []
    rows: list[dict] = []
    for key, section in sections.items():
        analytic = section.analytic()
        kappa_y = material.sigma_y0 / (material.E * section.half_height)
        curve = moment_curvature(
            section, material, kappa_max=6.0 * kappa_y, n_steps=180, n_sub=6
        )
        EI = material.E * analytic["I"]
        for factor in (2.0, 4.0, 6.0):
            kappa_peak = factor * kappa_y
            idx = int(np.argmin(np.abs(curve.kappa - kappa_peak)))
            sb = unload_from(curve, section, material, peak_index=idx)
            if key == "rectangle":
                b, h = section.dims
                M_peak_ref = rectangle_ideal_plastic_moment(
                    kappa_peak, b, h, material.E, material.sigma_y0
                )
                note = "closed-form ideal-plastic rectangle"
            else:
                M_peak_ref = float(curve.moment[idx])
                note = "elastic-unloading identity with the fiber moment"
            kappa_res_ref = kappa_peak - M_peak_ref / EI
            # linearity of the unloading path about the elastic slope
            dk = sb.path_kappa - sb.path_kappa[0]
            dm = sb.path_moment - sb.path_moment[0]
            if abs(dk[-1]) > 0:
                slope_fit = float(np.dot(dk, dm) / max(np.dot(dk, dk), 1e-300))
            else:  # pragma: no cover - defensive
                slope_fit = EI
            rows.append(
                {
                    "section": section.label,
                    "kappa_peak_over_kappa_y": factor,
                    "M_peak": float(curve.moment[idx]),
                    "kappa_residual_model": sb.kappa_residual,
                    "kappa_residual_reference": kappa_res_ref,
                    "unload_slope_fit": slope_fit,
                    "EI_reference": EI,
                    "reverse_yielded": sb.reverse_yielded,
                    "path_kappa_normalised": [
                        float(v / kappa_y) for v in sb.path_kappa[::6]
                    ],
                    "path_moment_normalised": [
                        float(v / (material.sigma_y0 * analytic["W_el"]))
                        for v in sb.path_moment[::6]
                    ],
                }
            )
            criteria.append(
                _criterion(
                    f"c_springback_{key}_{factor:g}kappay",
                    f"residual curvature after unloading, {section.label}, "
                    f"kappa_peak = {factor:g} kappa_y ({note})",
                    sb.kappa_residual,
                    kappa_res_ref,
                    _rel_err(sb.kappa_residual, kappa_res_ref),
                    TOL_SPRINGBACK,
                )
            )
            criteria.append(
                _criterion(
                    f"c_unload_slope_{key}_{factor:g}kappay",
                    f"unloading slope vs E*I, {section.label}, "
                    f"kappa_peak = {factor:g} kappa_y",
                    slope_fit,
                    EI,
                    _rel_err(slope_fit, EI),
                    0.01,
                )
            )
    return {"rows": rows, "criteria": criteria}


# ---------------------------------------------------------------------------
# (d) energy balance
# ---------------------------------------------------------------------------


def verify_energy(fast: bool = False) -> dict:
    sections = bench_sections()
    material = ideal_material()
    criteria: list[dict] = []
    rows: list[dict] = []
    for key, section in sections.items():
        kappa_y = material.sigma_y0 / (material.E * section.half_height)
        curve = moment_curvature(
            section, material, kappa_max=12.0 * kappa_y, n_steps=200, n_sub=6
        )
        audit = section_energy_audit(curve)
        rows.append({"level": f"section:{section.label}", **audit})
        criteria.append(
            _criterion(
                f"d_energy_section_{key}",
                f"W_ext = U_el + W_pl over the M-kappa path, {section.label}",
                audit["relative_residual"],
                0.0,
                audit["relative_residual"],
                TOL_ENERGY,
            )
        )
    # beam level, with plastic flow at the fixed end
    ibeam = sections["ibeam"]
    fe = fe_material()
    length = 6000.0
    M_p = fe.sigma_y0 * ibeam.analytic()["Z"]
    q_c = propped_cantilever_limit(M_p, length)["q_collapse"]
    pc = propped_cantilever_udl(
        ibeam,
        fe,
        length,
        n_elements=40 if fast else 60,
        n_steps=24 if fast else 40,
        load_fraction=0.7,
    )
    rows.append({"level": "beam:propped_cantilever_udl_0.7", **pc.energy})
    criteria.append(
        _criterion(
            "d_energy_beam",
            "beam-level W_ext = U_el + W_pl, propped cantilever at 0.7 q_c",
            pc.energy["relative_residual"],
            0.0,
            pc.energy["relative_residual"],
            TOL_ENERGY,
        )
    )
    return {"rows": rows, "criteria": criteria}


# ---------------------------------------------------------------------------
# plate limit loads (classical anchors)
# ---------------------------------------------------------------------------


def verify_plate_limits() -> dict:
    M_p = plastic_moment_per_width(SIGMA_Y, 10.0)
    b = 1000.0
    square = square_plate_limit(b, M_p)
    circ = circular_plate_mechanism(b, M_p)
    strip = strip_plate_mechanism(4.0 * b, b, M_p)
    rect = rectangular_plate_mechanism(2.0 * b, b, M_p)
    rows = {
        "M_p_per_width": M_p,
        "square": {
            "pressure": square,
            "p_u_a2_over_Mp": square * b**2 / M_p,
            "reference": 24.0,
        },
        "circular": {
            "pressure": circ.pressure,
            "p_u_R2_over_Mp": circ.pressure * b**2 / M_p,
            "reference": 6.0,
        },
        "strip": {
            "pressure": strip.pressure,
            "p_u_b2_over_Mp": strip.pressure * b**2 / M_p,
            "reference": 8.0,
        },
        "rectangular_a_over_b_2": {
            "pressure": rect.pressure,
            "p_u_b2_over_Mp": rect.pressure * b**2 / M_p,
            "hinges": len(rect.hinges),
        },
    }
    criteria = [
        _criterion(
            "p_square_plate",
            "yield-line pyramid, simply supported square plate vs 24 M_p/a^2",
            rows["square"]["p_u_a2_over_Mp"],
            24.0,
            _rel_err(rows["square"]["p_u_a2_over_Mp"], 24.0),
            TOL_PLATE,
        ),
        _criterion(
            "p_circular_plate",
            "conical mechanism, simply supported circular plate vs 6 M_p/R^2",
            rows["circular"]["p_u_R2_over_Mp"],
            6.0,
            _rel_err(rows["circular"]["p_u_R2_over_Mp"], 6.0),
            TOL_PLATE,
        ),
        _criterion(
            "p_strip_plate",
            "mid-line hinge mechanism, long plate vs 8 M_p/b^2",
            rows["strip"]["p_u_b2_over_Mp"],
            8.0,
            _rel_err(rows["strip"]["p_u_b2_over_Mp"], 8.0),
            TOL_PLATE,
        ),
    ]
    return {"rows": rows, "criteria": criteria}


# ---------------------------------------------------------------------------
# (e) negative controls
# ---------------------------------------------------------------------------


def negative_controls() -> dict:
    """Controls that must FAIL -- they prove the thresholds are discriminating."""
    material = ideal_material()
    criteria: list[dict] = []

    # NC1: section-height transcription error (+10 % height vs the reference)
    h = 200.0
    b = 100.0
    wrong = FiberSection.rectangle(b, 1.1 * h)
    kappa_y = material.sigma_y0 / (material.E * wrong.half_height)
    curve = moment_curvature(
        wrong, material, kappa_max=2.0 * kappa_y, n_steps=40, n_sub=2
    )
    M_y_model = float(curve.moment[int(np.searchsorted(curve.kappa, kappa_y))])
    M_y_ref = material.sigma_y0 * b * h**2 / 6.0
    err = _rel_err(M_y_model, M_y_ref)
    crit = _criterion(
        "e_nc1_geometry_transcription",
        "NEGATIVE CONTROL: M_y of a +10 % section height vs the nominal reference "
        "(must fail the 3 % gate)",
        M_y_model,
        M_y_ref,
        err,
        TOL_MY,
    )
    crit["expected"] = "fail"
    crit["control_ok"] = not crit["passed"]
    criteria.append(crit)

    # NC2: fabricated hinge position (mid-span instead of the limit-analysis value)
    ratio_ref = propped_cantilever_limit(1.0, 1.0)["hinge_position_ratio"]
    fabricated = 0.5
    crit2 = _criterion(
        "e_nc2_fabricated_hinge",
        "NEGATIVE CONTROL: hinge reported at mid-span vs limit analysis "
        "(must fail the 1 % span gate)",
        fabricated,
        ratio_ref,
        abs(fabricated - ratio_ref),
        TOL_HINGE,
        note="error measured as a fraction of the span",
        scale="span_fraction",
    )
    crit2["expected"] = "fail"
    crit2["control_ok"] = not crit2["passed"]
    criteria.append(crit2)

    # NC3: elastic-only springback (no plastic residual) must fail the 5 % gate
    section = FiberSection.rectangle(b, h)
    analytic = section.analytic()
    kappa_y = material.sigma_y0 / (material.E * section.half_height)
    kappa_peak = 4.0 * kappa_y
    M_peak = rectangle_ideal_plastic_moment(
        kappa_peak, b, h, material.E, material.sigma_y0
    )
    kappa_res_ref = kappa_peak - M_peak / (material.E * analytic["I"])
    crit3 = _criterion(
        "e_nc3_elastic_springback",
        "NEGATIVE CONTROL: residual curvature assumed zero (purely elastic "
        "springback claim) must fail the 5 % gate",
        0.0,
        kappa_res_ref,
        _rel_err(0.0, kappa_res_ref),
        TOL_SPRINGBACK,
    )
    crit3["expected"] = "fail"
    crit3["control_ok"] = not crit3["passed"]
    criteria.append(crit3)
    return {"criteria": criteria}


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def run_all_criteria(fast: bool = False) -> dict[str, Any]:
    sections = verify_moment_curvature()
    hinge = verify_hinge_location(fast=fast)
    spring = verify_springback()
    energy = verify_energy(fast=fast)
    plate = verify_plate_limits()
    nc = negative_controls()

    criteria: list[dict] = []
    criteria.extend(sections["criteria"])
    criteria.extend(hinge["criteria"])
    criteria.extend(spring["criteria"])
    criteria.extend(energy["criteria"])
    criteria.extend(plate["criteria"])
    negative = nc["criteria"]
    report: dict[str, Any] = {
        "task": "p387w2-n06 J2 elastoplastic beam/plate verification surface",
        "material": {
            "E_MPa": E_STEEL,
            "sigma_y0_MPa": SIGMA_Y,
            "hardening_ratio_FE": HARDENING_FE,
            "hardening_ratio_ideal": 0.0,
        },
        "tolerances": {
            "EI": TOL_EI,
            "M_y": TOL_MY,
            "M_p": TOL_MP,
            "hinge_span_fraction": TOL_HINGE,
            "springback": TOL_SPRINGBACK,
            "energy": TOL_ENERGY,
            "plate": TOL_PLATE,
        },
        "momcurv": sections["rows"],
        "hinge": hinge["cases"],
        "springback": spring["rows"],
        "energy": energy["rows"],
        "plate": plate["rows"],
        "criteria": criteria,
        "negative_controls": negative,
        "passed": [c["name"] for c in criteria if c["passed"]],
        "failed": [c["name"] for c in criteria if not c["passed"]],
        "negative_controls_ok": all(c["control_ok"] for c in negative),
    }
    report["all_passed"] = len(report["failed"]) == 0 and report["negative_controls_ok"]
    return report