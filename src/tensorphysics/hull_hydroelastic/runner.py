"""End-to-end springing / whipping study for one hull case (gates included).

``run_case`` is the script-level entry point: it builds the non-uniform
Timoshenko beam, the strip added-mass distribution, the wet modes, the springing
stress spectrum, the whipping transient, the rigid/elastic coupling table and the
utilisation-ledger gate, and returns one JSON-serialisable dictionary.
All acceptance-gate numbers are computed here (never typed in by hand).
"""

from __future__ import annotations

import numpy as np

from ..hullgirder.model import GRAVITY
from .presets import HullCase
from .response import (
    coupling_table,
    springing_fatigue_rate,
    springing_stress,
    utilisation_gate,
    whipping_fatigue,
    whipping_response,
)
from .sections import added_mass_matrix, circle_section, superellipse_section
from .timoshenko import (
    TimoshenkoBeam,
    euler_bernoulli_free_free_roots,
    uniform_beam,
)
from .wet import added_mass_distribution, ship_beam_properties, wet_modes

__all__ = [
    "dry_mode_analytic_gate",
    "added_mass_gate",
    "convergence_gate",
    "sign_gates",
    "run_case",
]

RHO_WATER = 1025.0


def dry_mode_analytic_gate(n_elements: int = 120) -> dict:
    """Gate (1): Euler-Bernoulli analytic roots + Timoshenko softening sign."""
    length = 100.0
    roots = euler_bernoulli_free_free_roots(4)
    eb_beam = uniform_beam(length=length, height=1.0, section="solid_rect", n_elements=n_elements)
    eb = TimoshenkoBeam(eb_beam, shear_rigid=True).dry_modes(4)
    analytic = (roots ** 2) / (2.0 * np.pi * length ** 2) * np.sqrt(eb_beam.EI[0] / eb_beam.m[0])

    slender = {}
    for h, label in ((5.0, 20.0), (1.0, 100.0)):
        props = uniform_beam(length=length, height=h, section="solid_rect", n_elements=n_elements)
        t = TimoshenkoBeam(props).dry_modes(3)
        e = TimoshenkoBeam(props, shear_rigid=True).dry_modes(3)
        slender[str(int(label))] = {
            "L_over_h": float(label),
            "T_Hz": t.frequencies_hz.tolist(),
            "EB_Hz": e.frequencies_hz.tolist(),
            "drop_pct": [float(100.0 * (1.0 - a / b)) for a, b in zip(t.frequencies_hz, e.frequencies_hz)],
        }
    # thin-walled box (hull-girder-like I/A = h^2/4) at the same slenderness
    box_props = uniform_beam(length=length, height=5.0, section="box", n_elements=n_elements)
    box_t = TimoshenkoBeam(box_props).dry_modes(3)
    box_e = TimoshenkoBeam(box_props, shear_rigid=True).dry_modes(3)
    box = {
        "L_over_h": 20.0,
        "T_Hz": box_t.frequencies_hz.tolist(),
        "EB_Hz": box_e.frequencies_hz.tolist(),
        "drop_pct": [float(100.0 * (1.0 - a / b)) for a, b in zip(box_t.frequencies_hz, box_e.frequencies_hz)],
    }
    rel = np.abs(eb.frequencies_hz[:3] / analytic[:3] - 1.0)
    return {
        "analytic_roots_betaL": roots[:3].tolist(),
        "EB_FE_Hz": eb.frequencies_hz[:3].tolist(),
        "EB_analytic_Hz": analytic[:3].tolist(),
        "relative_error_pct": [float(100.0 * v) for v in rel],
        "max_relative_error_pct": float(100.0 * rel.max()),
        "tolerance_pct": 3.0,
        "pass_analytic": bool(rel.max() <= 0.03),
        "slender": slender,
        "box_girder": box,
        "shear_softening_sign": bool(
            all(d > 0.0 for d in slender["20"]["drop_pct"]) and all(d > 0.0 for d in slender["100"]["drop_pct"])
        ),
        "pass_slender_20_mode1_under_1pct": bool(slender["20"]["drop_pct"][0] < 1.0),
        "n_elements": int(n_elements),
    }


def added_mass_gate(n_panels: tuple[int, ...] = (80, 160, 320)) -> dict:
    """Gate (2): 2-D infinite-frequency added mass vs ``rho pi a^2``."""
    radius = 1.0
    reference = RHO_WATER * np.pi * radius ** 2
    rows = []
    for n in n_panels:
        sec = circle_section(radius, centre_z=-10.0)
        m = added_mass_matrix(sec, modes=("heave", "sway"), n_panels=n, free_surface=False)["matrix"]
        rows.append(
            {
                "n_panels": int(n),
                "m33_kg_per_m": float(m[0, 0]),
                "m22_kg_per_m": float(m[1, 1]),
                "error_heave_pct": float(100.0 * (m[0, 0] / reference - 1.0)),
                "error_sway_pct": float(100.0 * (m[1, 1] / reference - 1.0)),
            }
        )
    # second analytic case: ellipse with semi-axes (2, 1) -> m_heave = rho pi a_y^2
    ell = superellipse_section(2.0, 1.0, exponent=2.0)
    from .sections import Section2D

    th = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    ell_full = Section2D(y=2.0 * np.cos(th), z=-10.0 + np.sin(th), name="ellipse_full")
    m_ell = added_mass_matrix(ell_full, modes=("heave", "sway"), n_panels=320)["matrix"]
    ellipse = {
        "semi_axis_y_m": 2.0,
        "semi_axis_z_m": 1.0,
        "m33_expected_kg_per_m": float(RHO_WATER * np.pi * 2.0 ** 2),
        "m33_kg_per_m": float(m_ell[0, 0]),
        "error_heave_pct": float(100.0 * (m_ell[0, 0] / (RHO_WATER * np.pi * 4.0) - 1.0)),
        "error_sway_pct": float(100.0 * (m_ell[1, 1] / (RHO_WATER * np.pi * 1.0) - 1.0)),
    }
    worst = max(abs(r["error_heave_pct"]) for r in rows)
    return {
        "reference_rho_pi_a2": float(reference),
        "circle_rows": rows,
        "ellipse": ellipse,
        "worst_circle_error_pct": float(worst),
        "tolerance_pct": 5.0,
        "pass": bool(worst <= 5.0 and abs(ellipse["error_heave_pct"]) <= 5.0),
        "note": "unbounded fluid; the floating-section coefficient (phi=0 on z=0) "
                "has no closed form and is checked by panel convergence only",
    }


def convergence_gate(
    ship, beam, a33, counts=(5, 10, 20), hs=6.0, t1=9.0, speed=0.0, zeta_structural=0.02
) -> dict:
    b_wp = ship.waterplane_width(beam.props.x)
    """Gate (3): wet frequencies and springing stress RMS vs modal truncation."""
    rows = []
    for n in counts:
        wet = wet_modes(beam, a33, n_modes=n, zeta_structural=zeta_structural, b_waterplane=b_wp)
        spr = springing_stress(ship, wet, hs, t1, speed=speed)
        rows.append(
            {
                "n_modes": int(n),
                "wet_hz": wet.wet_frequencies_hz[:3].tolist(),
                "sigma_rms_MPa": spr["sigma_rms_MPa"],
            }
        )
    base = rows[-1]
    changes = []
    for r in rows[:-1]:
        df = max(
            abs(r["wet_hz"][i] / base["wet_hz"][i] - 1.0) for i in range(3)
        )
        ds = abs(r["sigma_rms_MPa"] / base["sigma_rms_MPa"] - 1.0)
        changes.append(
            {
                "n_modes": r["n_modes"],
                "max_wet_freq_change_pct": float(100.0 * df),
                "stress_rms_change_pct": float(100.0 * ds),
            }
        )
    worst = max(max(c["max_wet_freq_change_pct"], c["stress_rms_change_pct"]) for c in changes)
    return {
        "rows": rows,
        "changes_vs_20": changes,
        "tolerance_pct": 1.0,
        "worst_change_pct": float(worst),
        "pass": bool(worst <= 1.0),
    }


def sign_gates(wet) -> dict:
    """Gate (4): wet < dry frequency, damping ratio > 0."""
    wet_f = wet.wet_frequencies_hz
    dry_f = wet.dry_omega / (2.0 * np.pi)
    strictly_below = bool(np.all(wet_f < dry_f))
    return {
        "dry_hz": dry_f.tolist(),
        "wet_hz": wet_f.tolist(),
        "downshift_pct": [float(100.0 * (w / d - 1.0)) for w, d in zip(wet_f, dry_f)],
        "all_wet_below_dry": strictly_below,
        "min_zeta": float(np.min(wet.zeta)),
        "all_zeta_positive": bool(np.all(wet.zeta > 0.0)),
        "pass": bool(strictly_below and np.all(wet.zeta > 0.0)),
    }


def run_case(
    case: HullCase,
    hs: float = 6.0,
    t1: float = 9.0,
    n_modes: int = 10,
    n_stations: int = 9,
    n_panels: int = 80,
    slamming_force: float = 5.0e6,
    patch_fraction: float = 0.10,
    zeta_structural: float = 0.02,
    zeta_band: tuple[float, ...] = (0.005, 0.02, 0.05),
    slamming_band: tuple[float, ...] = (2.5e6, 5.0e6, 10.0e6),
    duration_band: tuple[float, ...] = (0.12, 0.25),
) -> dict:
    ship = case.girder()
    beam = TimoshenkoBeam(ship_beam_properties(ship, n_elements=case.n_elements))
    am = added_mass_distribution(
        ship, n_stations=n_stations, section_exponent=case.section_exponent, n_panels=n_panels, x=beam.props.x
    )
    wet = wet_modes(
        beam,
        am["a33"],
        n_modes=n_modes,
        zeta_structural=zeta_structural,
        b_waterplane=ship.waterplane_width(beam.props.x),
    )

    spring = springing_stress(ship, wet, hs, t1, speed=case.design_speed_kn * 0.514444)
    whip = whipping_response(
        ship, wet, peak_force=slamming_force, patch_fraction=patch_fraction, duration=0.25
    )
    froude = tuple(case.froude_number * f for f in (0.55, 0.75, 1.0))
    coupling = coupling_table(
        ship,
        wet,
        hs,
        t1,
        froude_numbers=froude,
        slamming_force_at_ref=slamming_force,
        ref_froude=case.froude_number,
        patch_fraction=patch_fraction,
    )
    last = coupling["rows"][-1]
    ledger = utilisation_gate(
        ship,
        m_coupled_extreme=last["M_coupled_extreme_MNm"] * 1e6,
        m_decoupled_extreme=last["M_decoupled_extreme_MNm"] * 1e6,
    )
    fat_whip = whipping_fatigue(
        whip["peak_stress_Pa"], wet.wet_omega[0], zeta_structural, duration_factor=1.0
    )
    # wave-process zero-crossing rate of the ITTC spectrum: T_z = 0.71 T1
    omega_z = 2.0 * np.pi / (0.71 * t1)
    fat_wave = springing_fatigue_rate(
        sigma_rms=float(np.hypot(spring["sigma_rms_Pa"], spring["rigid_stress_rms_Pa"])),
        mean_zero_crossing=omega_z,
    )
    # slamming event rate: one impact per ``events_per_hour`` in the design sea
    events_per_hour = 60.0
    whipping_contribution_pct = float(
        100.0
        * (fat_whip["damage_per_event"] * events_per_hour)
        / (fat_whip["damage_per_event"] * events_per_hour + fat_wave["damage_rate_per_s"] * 3600.0)
    )
    convergence = convergence_gate(ship, beam, am["a33"], hs=hs, t1=t1,
                                   speed=case.design_speed_kn * 0.514444,
                                   zeta_structural=zeta_structural)
    # damping sensitivity of the resonant (springing) level
    zeta_sensitivity = []
    for z in zeta_band:
        spr_z = springing_stress(ship, wet, hs, t1, zeta=z, speed=case.design_speed_kn * 0.514444)
        zeta_sensitivity.append(
            {
                "zeta": float(z),
                "sigma_rms_MPa": spr_z["sigma_rms_MPa"],
                "elastic_share_pct": float(
                    100.0
                    * (
                        np.hypot(spr_z["sigma_rms_Pa"], spr_z["rigid_stress_rms_Pa"])
                        / spr_z["rigid_stress_rms_Pa"]
                        - 1.0
                    )
                ),
            }
        )
    # slamming severity / pulse duration sensitivity of the whipping peak
    whip_sensitivity = []
    for f in slamming_band:
        for d in duration_band:
            wv = whipping_response(ship, wet, peak_force=f, duration=d, patch_fraction=patch_fraction)
            whip_sensitivity.append(
                {
                    "peak_force_MN": f / 1e6,
                    "duration_s": float(d),
                    "impulse_MNs": wv["impulse_Ns"] / 1e6,
                    "peak_stress_MPa": wv["peak_stress_MPa"],
                    "two_node_stress_MPa": wv["modal_peak_stress_Pa"][0] / 1e6,
                    "modes_kept": wv["modes_kept"],
                }
            )
    return {
        "hull": case.as_dict(),
        "sea_state": {"hs_m": hs, "t1_s": t1},
        "beam": {
            "n_elements": int(case.n_elements),
            "EI_mid_Nm2": float(beam.props.EI[beam.props.n_nodes // 2]),
            "GAs_mid_N": float(beam.props.GAs[beam.props.n_nodes // 2]),
            "m_mid_kg_per_m": float(beam.props.m[beam.props.n_nodes // 2]),
            "rotary_inertia_kgm": float(beam.props.rI[beam.props.n_nodes // 2]),
        },
        "dry_modes": [
            {
                "mode": i + 1,
                "frequency_hz": float(wet.dry_omega[i] / (2.0 * np.pi)),
                "curvature_max_per_m": float(np.max(np.abs(wet.dry.curvature[i]))),
            }
            for i in range(wet.n_modes)
        ],
        "added_mass": {
            "x_stations_m": am["x_stations"].tolist(),
            "a33_stations_kg_per_m": am["a33_stations"].tolist(),
            "a33_mid_kg_per_m": float(am["a33"][am["a33"].size // 2]),
            "half_beam_stations_m": am["half_beam_stations"].tolist(),
            "section_exponent": am["section_exponent"],
        },
        "wet_modes": wet.shift_table(),
        "sign_gates": sign_gates(wet),
        "springing": {
            "sigma_rms_MPa": spring["sigma_rms_MPa"],
            "rigid_stress_rms_MPa": spring["rigid_stress_rms_Pa"] / 1e6,
            "hot_spot_x_m": spring["hot_spot_x_m"],
            "dominant_mode": spring["dominant_mode"],
            "per_mode_rms_MPa": [v / 1e6 for v in spring["per_mode_rms_Pa"]],
            "spectrum": {
                "omega": spring["omega"],
                "S_zeta": spring["spectrum_m2s"],
                "S_sigma_MPa2s": [v / 1e12 for v in spring["stress_spectrum_Pa2s"]],
            },
        },
        "whipping": {
            "peak_stress_MPa": whip["peak_stress_MPa"],
            "t_peak_s": whip["t_peak_s"],
            "peak_force_MN": whip["peak_force_N"] / 1e6,
            "impulse_MNs": whip["impulse_Ns"] / 1e6,
            "patch_length_m": whip["patch_length_m"],
            "patch_area_m2": whip["patch_area_m2"],
            "dominant_mode": whip["dominant_mode"],
            "modes_kept": whip["modes_kept"],
            "min_wavelength_over_depth": whip["min_wavelength_over_depth"],
            "peak_pressure_MPa": whip["peak_pressure_MPa"],
            "modal_peak_stress_MPa": [v / 1e6 for v in whip["modal_peak_stress_Pa"]],
            "time_history": {
                "t_s": whip["t_s"][::10],
                "stress_MPa": [v / 1e6 for v in whip["stress_Pa"][::10]],
            },
        },
        "coupling": coupling,
        "utilisation_ledger": ledger,
        "fatigue": {
            "wave_springing_damage_rate_per_s": fat_wave["damage_rate_per_s"],
            "whipping_damage_per_event": fat_whip["damage_per_event"],
            "events_per_hour": events_per_hour,
            "whipping_contribution_pct": whipping_contribution_pct,
            "whipping_peak_stress_MPa": whip["peak_stress_MPa"],
            "wave_stress_rms_MPa": fat_wave["stress_rms_MPa"],
            "sn_curve": "DNV-RP-C203 curve D in air, m=3, log10 a=12.164",
        },
        "convergence": convergence,
        "damping_sensitivity": zeta_sensitivity,
        "whipping_sensitivity": whip_sensitivity,
    }