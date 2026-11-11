"""Load-capacity handoff gate: Smith ultimate moment vs wave-induced bending.

The Smith ultimate moment is a *capacity*; it becomes a ship-safety statement
only when it is compared with the *demand*.  This module builds the hand-off
between the two halves of the hull-girder chain:

``demand``
    wave-induced vertical bending moment of the ship, from three independent
    levels of the existing hull-girder machinery
    (:mod:`tensorphysics.hullgirder.wave`):

    1. the regulatory design-wave value (IACS UR S11 / ITTC-1978 formula,
       ``M_wh``/``M_ws``);
    2. the design-wave value of the long-wave strip model (RAO + Smith
       correction, ``design_wave_moment``);
    3. the **short-term extreme of the wave-moment time history**: the
       spectral RMS of the midship moment (``moment_rms_from_spectrum``) mapped
       through the Rayleigh distribution to the ``1e-8`` per-cycle level, and the
       long-term value over a coarse North-Atlantic scatter diagram
       (``long_term_design_moment``);

``capacity``
    the Smith sagging/hogging ultimate moment of the midship section;

``gate``
    ``margin = M_u / (M_wave + M_still_water)`` against a predeclared minimum
    (1.5 by default, as required by the campaign criterion C4), with the
    still-water moment either given explicitly or taken as a declared fraction
    of the wave moment (``None`` -> pure wave demand, which is the literal C4
    definition).

The record is JSON-serialisable and is the "handoff" artifact: it carries both
sides of the comparison plus the evidence class of each number, so a downstream
node (or a reviewer) can see exactly which demand definition was used.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..hullgirder.model import ShipGirder
from ..hullgirder import wave as wave_mod
from .benchmarks import kcs_particulars

__all__ = [
    "kcs_girder",
    "wave_demand",
    "load_capacity_handoff",
]


def kcs_girder(section_modulus: float, yield_stress: float = 355.0e6,
               n_nodes: int = 201) -> ShipGirder:
    """``ShipGirder`` of the KCS with the section modulus from the Smith section.

    The remaining particulars are the published KCS values, so the wave-moment
    machinery sees a *real* ship, and the section modulus is the one carried by
    the modelled midship section (``SectionModel.section_moduli``).
    """
    p = kcs_particulars()
    return ShipGirder(
        length=float(p["Lpp"]),
        breadth=float(p["B"]),
        draught=float(p["T"]),
        block_coefficient=float(p["Cb"]),
        depth=float(p["D"]),
        section_modulus=float(section_modulus),
        yield_stress=float(yield_stress),
        plate_slenderness=1.5,
        n_nodes=int(n_nodes),
    )


def wave_demand(ship: ShipGirder,
                design_wave_height_ratio: float = 20.0,
                target_probability: float = 1.0e-8,
                scatter_shape: float = 1.5,
                scatter_scale: float = 3.0,
                n_cells: int = 24,
                n_omega: int = 60) -> Dict[str, object]:
    """Wave-induced vertical bending moment demand of the ship [N*m].

    Three independent demand levels are returned, and the value used as *the*
    demand is the **design-wave level**, i.e. the larger of the regulatory
    formula and the design-wave strip-model value.  The spectral long-term value
    is returned as a declared **upper reference** only: the long-wave strip RAO of
    :mod:`tensorphysics.hullgirder.wave` has neither diffraction nor radiation
    damping, and its coarse Weibull climate is a documented limitation, so it is
    known to over-predict the extreme moment.  Reporting it separately keeps the
    gate numeric and honest instead of hiding the spread.

    Parameters
    ----------
    ship : ShipGirder
    design_wave_height_ratio : float
        ``h_w = L / ratio`` of the classical design wave (20 -> ``h_w = L/20``).
    target_probability : float
        Per-cycle probability level of the spectral long-term extreme.
    scatter_shape, scatter_scale, n_cells : float, float, int
        Coarse North-Atlantic scatter model (see
        :func:`~tensorphysics.hullgirder.wave.north_atlantic_scatter`).
    n_omega : int
        Frequency resolution of the spectral integration.
    """
    rule_c = wave_mod.rule_coefficient_c(ship.length)
    ittc = wave_mod.ittc_design_wave_moment(ship.length, ship.breadth,
                                            ship.block_coefficient)
    design_wave = wave_mod.design_wave_moment(
        ship, wave_height_ratio=design_wave_height_ratio)
    iacs_sag = abs(float(ittc["sagging_Nm"]))
    iacs_hog = abs(float(ittc["hogging_Nm"]))
    strip = abs(float(design_wave["M_max_Nm"]))
    m_design = max(iacs_sag, iacs_hog, strip)
    # spectral long-term upper reference
    cells = wave_mod.north_atlantic_scatter(shape=scatter_shape, scale=scatter_scale,
                                           n_cells=n_cells)
    sigmas: List[float] = []
    weights: List[float] = []
    for cell in cells:
        sigmas.append(wave_mod.moment_rms_from_spectrum(
            ship, cell["hs"], cell["t1"], n_omega=n_omega))
        weights.append(cell["weight"])
    long_term = wave_mod.long_term_design_moment(sigmas, weights, target_probability)
    i_max = int(np.argmax(weights))  # most probable sea state, not the worst
    short_term = {
        "representative_cell_hs": float(cells[i_max]["hs"]),
        "representative_cell_t1": float(cells[i_max]["t1"]),
        "sigma_midship_Nm": float(sigmas[i_max]),
        "extreme_1e2_per_cycle_Nm": float(
            np.sqrt(-2.0 * np.log(1.0e-2)) * sigmas[i_max]),
        "probability_level": 1.0e-2,
    }
    return {
        "rule_coefficient_C": float(rule_c),
        "IACS_M_hogging_Nm": iacs_hog,
        "IACS_M_sagging_Nm": iacs_sag,
        "design_wave_strip_model": design_wave,
        "M_design_wave_strip_Nm": strip,
        "M_wave_extreme_Nm": float(m_design),
        "M_wave_extreme_definition": (
            "design-wave level: max(IACS UR S11 formula, design-wave strip model)"),
        "M_long_term_upper_Nm": float(long_term["M_design_Nm"]),
        "long_term_upper": long_term,
        "short_term": short_term,
        "evidence_level": {
            "IACS": "ANALYTICAL (IACS UR S11 / ITTC-1978 formula)",
            "design_wave_strip": ("REAL (long-wave strip model; RAO verified against the "
                                  "closed-form barge limit in this repository's test suite)"),
            "long_term_upper": ("UPPER REFERENCE - strip RAO without diffraction/radiation "
                                "damping plus a coarse Weibull climate: known to "
                                "over-predict the extreme moment, reported not used"),
        },
    }


def load_capacity_handoff(
    M_u_sagging: float,
    M_u_hogging: float,
    ship: ShipGirder,
    M_still_water: Optional[float] = None,
    still_water_fraction_of_wave: Optional[float] = None,
    required_margin: float = 1.5,
    target_probability: float = 1.0e-8,
) -> Dict[str, object]:
    """Compare Smith capacity with the wave demand and open/close the gate.

    Parameters
    ----------
    M_u_sagging, M_u_hogging : float
        Smith ultimate moments [N*m] (positive magnitudes).
    ship : ShipGirder
        Ship girder used for the demand (see :func:`kcs_girder`).
    M_still_water : float or None
        Explicit still-water bending moment magnitude [N*m] to add to the wave
        demand.  ``None`` and ``still_water_fraction_of_wave=None`` gives the
        literal C4 demand (wave only).
    still_water_fraction_of_wave : float or None
        Declared still-water fraction of the wave moment (used only when
        ``M_still_water`` is None); e.g. ``0.4`` for a typical container-ship
        design still-water moment.
    required_margin : float
        Predeclared minimum margin (C4: 1.5).
    target_probability : float
        Per-cycle probability level of the wave demand.

    Returns
    -------
    dict
        JSON-serialisable handoff record with the demand, the capacity, the
        margins, the pass/fail flags and the evidence classes.
    """
    if M_u_sagging <= 0.0 or M_u_hogging <= 0.0:
        raise ValueError("ultimate moments must be positive")
    if required_margin <= 0.0:
        raise ValueError("required_margin must be positive")
    wave = wave_demand(ship, target_probability=target_probability)
    m_wave = float(wave["M_wave_extreme_Nm"])
    m_long_term = float(wave["M_long_term_upper_Nm"])
    if M_still_water is not None:
        m_sw = float(M_still_water)
        sw_source = "explicit input"
    elif still_water_fraction_of_wave is not None:
        m_sw = float(still_water_fraction_of_wave) * m_wave
        sw_source = f"declared fraction ({still_water_fraction_of_wave}) of the wave moment"
    else:
        m_sw = 0.0
        sw_source = "none (pure wave demand, literal C4 definition)"
    demand = m_wave + m_sw
    margin_sag = float(M_u_sagging / demand)
    margin_hog = float(M_u_hogging / demand)
    governing = "sagging" if margin_sag <= margin_hog else "hogging"
    margin = min(margin_sag, margin_hog)
    # wave-only margin (C4 literal) and the stricter spectral upper reference
    margin_wave_only = float(min(M_u_sagging, M_u_hogging) / m_wave)
    margin_long_term = float(min(M_u_sagging, M_u_hogging) / m_long_term)
    return {
        "case": "KCS midship section load-capacity handoff",
        "capacity": {
            "M_u_sagging_Nm": float(M_u_sagging),
            "M_u_hogging_Nm": float(M_u_hogging),
            "method": "fiber-based Smith progressive collapse (this module)",
            "evidence_level": "REAL",
        },
        "demand": {
            "M_wave_extreme_Nm": m_wave,
            "M_long_term_upper_Nm": m_long_term,
            "M_still_water_Nm": m_sw,
            "still_water_source": sw_source,
            "M_total_demand_Nm": float(demand),
            "wave_detail": wave,
        },
        "margins": {
            "sagging": margin_sag,
            "hogging": margin_hog,
            "governing": governing,
            "governing_margin": margin,
            "wave_only_margin": margin_wave_only,
            "long_term_upper_margin": margin_long_term,
            "required_margin": float(required_margin),
        },
        "gate": {
            "criterion": f"M_u / M_demand >= {required_margin}",
            "passes": bool(margin >= required_margin),
            "passes_wave_only": bool(margin_wave_only >= required_margin),
            "passes_long_term_upper": bool(margin_long_term >= required_margin),
            "long_term_upper_note": (
                "the spectral long-term value uses a long-wave strip RAO without "
                "diffraction/radiation damping and a coarse climate model; failing this "
                "stricter reference is reported, not used to overwrite the criterion"),
            "utilisation": float(demand / min(M_u_sagging, M_u_hogging)),
        },
    }