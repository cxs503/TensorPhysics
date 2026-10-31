#!/usr/bin/env python3
"""W3-N02 benchmark: adfreeze bond mechanics and beset-vessel breakout criterion.

Produces (all numbers programmatically, nothing hand-copied):

* ``artifacts/adfreeze_bond/w3n02_results.json``  - validation table, ship table,
  degradation table, criteria verdicts, parameter provenance.
* ``artifacts/adfreeze_bond/figures/*.png``        - four result figures.

Run from the repository root::

    python benchmarks/icebreaking/adfreeze_breakout.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.ice import (  # noqa: E402
    FREDERKING_ADFREEZE,
    PARAMETER_PROVENANCE,
    SALINE_TO_FRESH_RATIO,
    SHIPS,
    adfreeze_strength,
    adfreeze_strength_state,
    bollard_thrust_mn,
    brine_volume_fraction,
    calibrate_reference_strength,
    contact_time_to_saturation,
    critical_slip,
    empirical_contact_area,
    evaluate_breakout,
    frederking_strengths,
    frederking_temperatures,
    frozen_area_fraction,
    hull_ice_contact_area,
    peel_softening,
    reference_strength_mpa,
    required_debonded_area,
    salinity_degradation_factor,
    salinity_factor,
    seawater_freezing_point,
    thermal_degradation_factor,
)

OUT = ROOT / "artifacts" / "adfreeze_bond"
FIG = OUT / "figures"


# ---------------------------------------------------------------------------
# 1. Criterion (a): strength vs Frederking data at four temperatures
# ---------------------------------------------------------------------------
def strength_validation() -> dict:
    fitted = calibrate_reference_strength(frederking_temperatures(), frederking_strengths())
    rows = []
    for rec in FREDERKING_ADFREEZE:
        t = rec["T_C"]
        ref = reference_strength_mpa(-t)  # smooth, saturated, freshwater
        hull = adfreeze_strength(t, contact_time_s=1.0e5, salinity_gkg=0.0, roughness_um=2.5)
        rows.append(
            {
                "T_C": t,
                "measured_MPa": rec["tau_MPa"],
                "measured_band_MPa": rec["band_MPa"],
                "model_smooth_saturated_MPa": round(ref, 5),
                "model_err_pct": round(100.0 * (ref / rec["tau_MPa"] - 1.0), 3),
                "model_hull_Ra2p5_MPa": round(hull, 5),
                "model_hull_err_pct": round(100.0 * (hull / rec["tau_MPa"] - 1.0), 3),
                "inside_measured_band": bool(abs(hull - rec["tau_MPa"]) <= rec["band_MPa"]),
                "source": rec["source"],
                "provenance": rec["provenance"],
                "exact": rec["exact"],
            }
        )
    max_err = max(abs(r["model_err_pct"]) for r in rows)
    max_err_hull = max(abs(r["model_hull_err_pct"]) for r in rows)
    return {
        "calibration": {
            "tau_sat_MPa": round(fitted[0], 4),
            "d_half_K": round(fitted[1], 4),
            "rms_relative_error": round(fitted[2], 5),
            "n_points": len(rows),
            "range_C": [min(r["T_C"] for r in rows), max(r["T_C"] for r in rows)],
        },
        "rows": rows,
        "max_abs_error_pct_smooth": round(max_err, 3),
        "max_abs_error_pct_hull": round(max_err_hull, 3),
        "criterion_pct": 15.0,
        "pass": bool(max_err < 15.0 and max_err_hull < 15.0),
    }


# ---------------------------------------------------------------------------
# 2. Criterion (b): breakout force vs empirical closed-form contact estimate
# ---------------------------------------------------------------------------
def breakout_validation() -> dict:
    rows = []
    for ship in SHIPS:
        area = hull_ice_contact_area(ship)
        area_emp = empirical_contact_area(ship)
        res = evaluate_breakout(ship, -10.0, contact_time_s=1.0e5)
        force_emp = (res.tau_adfreeze_mpa + 0.12 * 3.0e4 / 1e6) * area_emp
        peel = required_debonded_area(
            res.thrust_minus_friction_mn, res.tau_adfreeze_mpa, res.contact_area_m2
        )
        rows.append(
            {
                "ship": ship.name,
                "L_pp_m": ship.length_pp_m,
                "beam_m": ship.beam_m,
                "ice_thickness_m": ship.ice_thickness_m,
                "shaft_power_kW": ship.shaft_power_kw,
                "model_contact_area_m2": round(area, 2),
                "empirical_contact_area_m2": round(area_emp, 2),
                "area_dev_pct": round(100.0 * (area / area_emp - 1.0), 3),
                "tau_adfreeze_MPa": round(res.tau_adfreeze_mpa, 5),
                "adhesion_force_MN": round(res.adhesion_force_mn, 3),
                "friction_force_MN": round(res.friction_force_mn, 3),
                "breakout_force_MN": round(res.breakout_force_mn, 3),
                "breakout_force_empirical_MN": round(force_emp, 3),
                "force_dev_pct": round(100.0 * (res.breakout_force_mn / force_emp - 1.0), 3),
                "bollard_thrust_MN": round(res.thrust_mn, 4),
                "thrust_over_breakout": round(res.margin, 5),
                "critical_area_fraction": round(res.critical_area_fraction, 5),
                "required_debonded_area_m2": round(peel, 2),
            }
        )
    max_area_dev = max(abs(r["area_dev_pct"]) for r in rows)
    max_force_dev = max(abs(r["force_dev_pct"]) for r in rows)
    return {
        "rows": rows,
        "max_abs_area_dev_pct": round(max_area_dev, 3),
        "max_abs_force_dev_pct": round(max_force_dev, 3),
        "criterion_pct": 10.0,
        "pass": bool(max_area_dev < 10.0 and max_force_dev < 10.0),
        "independence": (
            "cross-method consistency (discretised hull-form integration vs closed-form "
            "rectangular engineering estimate); NOT independent experimental validation"
        ),
    }


# ---------------------------------------------------------------------------
# 3. Kinetics, degradation and environment sensitivity tables
# ---------------------------------------------------------------------------
def kinetics_table() -> dict:
    rows = []
    for t in (-2.0, -5.0, -10.0, -20.0):
        rows.append(
            {
                "T_C": t,
                "frozen_fraction_60s": round(frozen_area_fraction(-t, 60.0), 4),
                "frozen_fraction_600s": round(frozen_area_fraction(-t, 600.0), 4),
                "frozen_fraction_1h": round(frozen_area_fraction(-t, 3600.0), 4),
                "t_to_95pct_s": round(contact_time_to_saturation(t, 0.0, 0.95), 1),
                "tau_saturated_MPa": round(adfreeze_strength(t, 1.0e6, 0.0, 2.5), 5),
                "tau_after_60s_MPa": round(adfreeze_strength(t, 60.0, 0.0, 2.5), 5),
            }
        )
    return {"rows": rows}


def degradation_table() -> dict:
    tau = adfreeze_strength(-10.0, 1.0e5, 0.0, 2.5)
    dc = critical_slip(tau)
    peel_rows = [
        {
            "slip_um": s,
            "residual_fraction": round(peel_softening(tau, s * 1e-6), 4),
        }
        for s in (0.0, 1.0, 2.0, 5.0, 10.0, 13.5, 20.0)
    ]
    thermal_rows = [
        {
            "T_old_C": -10.0,
            "T_new_C": t,
            "factor": round(thermal_degradation_factor(-10.0, t), 4),
        }
        for t in (-20.0, -10.0, -5.0, -2.0, -1.0, 0.0)
    ]
    saline_rows = [
        {
            "S_gkg": s,
            "salinity_factor_at_-10C": round(salinity_factor(-10.0, s), 4),
            "brine_volume_fraction": round(brine_volume_fraction(-10.0, s), 4),
        }
        for s in (0.0, 5.0, 10.0, 20.0, 32.0, 50.0)
    ]
    return {
        "tau_peak_MPa": round(tau, 5),
        "critical_slip_um": round(dc * 1e6, 3),
        "interface_Gc_J_m2": 5.0,
        "peel_rows": peel_rows,
        "thermal_rows": thermal_rows,
        "salinity_rows": saline_rows,
        "saline_datum": SALINE_TO_FRESH_RATIO,
        "salinity_factor_ratio_at_datum": round(
            salinity_factor(-10.0, 32.0) / salinity_factor(-10.0, 0.0), 4
        ),
        "fresh_salinity_degradation_32_to_0": round(salinity_degradation_factor(-10.0, 32.0, 0.0), 4),
    }


def sensitivity_table() -> dict:
    """Breakout force sensitivity to the two closure parameters."""
    ship = SHIPS[1]
    base = evaluate_breakout(ship, -10.0, contact_time_s=1.0e5)
    clearance_rows = []
    for cl in (0.05, 0.10, 0.15, 0.20, 0.30):
        r = evaluate_breakout(ship, -10.0, contact_time_s=1.0e5, clearance_m=cl)
        clearance_rows.append(
            {"clearance_m": cl, "contact_area_m2": round(r.contact_area_m2, 2),
             "breakout_force_MN": round(r.breakout_force_mn, 3)}
        )
    friction_rows = []
    for mu in (0.05, 0.12, 0.20, 0.30):
        r = evaluate_breakout(ship, -10.0, contact_time_s=1.0e5, mu=mu)
        friction_rows.append(
            {"mu": mu, "friction_force_MN": round(r.friction_force_mn, 3),
             "breakout_force_MN": round(r.breakout_force_mn, 3)}
        )
    return {
        "base_ship": ship.name,
        "base_breakout_force_MN": round(base.breakout_force_mn, 3),
        "clearance_rows": clearance_rows,
        "friction_rows": friction_rows,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_strength(validation: dict) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    temps = np.linspace(-30.0, -0.2, 400)
    ax.plot(temps, [adfreeze_strength(t, 1e6, 0.0, 2.5) for t in temps], "-",
            label="model, hull Ra=2.5 um (saturated)")
    ax.plot(temps, [reference_strength_mpa(-t) for t in temps], "--",
            label="reference closure (smooth steel, freshwater)")
    ax.plot(temps, [adfreeze_strength(t, 1e6, 32.0, 2.5) for t in temps], "-.",
            label="model, seawater joint S=32 g/kg")
    for rec in FREDERKING_ADFREEZE:
        ax.errorbar(rec["T_C"], rec["tau_MPa"], yerr=rec["band_MPa"], fmt="o",
                    color="k", capsize=4, label="Frederking (1984) data" if rec["T_C"] == -2 else None)
    ax.set_xlabel("joint temperature [degC]")
    ax.set_ylabel("adfreeze shear strength [MPa]")
    ax.set_title("Adfreeze strength: model vs Frederking (1984) at four temperatures")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "fig1_adfreeze_strength.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_kinetics() -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    t = np.logspace(-1, 5.5, 400)
    for temp in (-2.0, -5.0, -10.0, -20.0):
        ax.semilogx(t, [frozen_area_fraction(-temp, tt) for tt in t], label=f"T = {temp:.0f} degC")
    ax.set_xlabel("contact time [s]")
    ax.set_ylabel("frozen area fraction of the joint [-]")
    ax.set_title("Adfreeze formation kinetics (Stefan freezing of the joint film)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "fig2_contact_time_kinetics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_degradation(degrad: dict) -> pathlib.Path:
    tau = degrad["tau_peak_MPa"]
    dc = degrad["critical_slip_um"] * 1e-6
    slips = np.linspace(0.0, 2.2 * dc, 300)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    axes[0].plot(slips * 1e6, [tau * peel_softening(tau, s) for s in slips], "-")
    axes[0].set_xlabel("interface slip [um]")
    axes[0].set_ylabel("shear traction [MPa]")
    axes[0].set_title("(a) mechanical peel, Gc = 5 J/m2")
    axes[0].grid(alpha=0.3)

    temps = np.linspace(-25.0, -0.05, 300)
    for s_old, label in ((0.0, "freshwater"), (32.0, "seawater S=32")):
        axes[1].plot(temps, [thermal_degradation_factor(-10.0, t, s_old) for t in temps],
                     label=label)
    axes[1].set_xlabel("new joint temperature [degC]")
    axes[1].set_ylabel("thermal degradation factor [-]")
    axes[1].set_title("(b) thermal degradation from T = -10 degC")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    sals = np.linspace(0.0, 60.0, 300)
    axes[2].plot(sals, [salinity_factor(-10.0, s) for s in sals], "-", label="T = -10 degC")
    axes[2].plot(sals, [salinity_factor(-5.0, s) for s in sals], "--", label="T = -5 degC")
    axes[2].set_xlabel("joint salinity [g/kg]")
    axes[2].set_ylabel("salinity factor [-]")
    axes[2].set_title("(c) brine penalty")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "fig3_degradation_mechanisms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_breakout(breakout: dict) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    areas = np.linspace(0.0, 300.0, 200)
    tau = breakout["rows"][0]["tau_adfreeze_MPa"]
    f_per_area = tau + 0.12 * 3.0e4 / 1e6
    ax.plot(areas, f_per_area * areas, "-", color="C0", label="breakout force (tau + mu*p_n) A")
    for i, row in enumerate(breakout["rows"]):
        ax.axhline(row["bollard_thrust_MN"], ls=":", color=f"C{i+1}",
                   label=f"thrust: {row['ship']} = {row['bollard_thrust_MN']:.2f} MN")
        ax.plot(row["model_contact_area_m2"], row["breakout_force_MN"], "o", color=f"C{i+1}")
        ax.annotate(f"{row['critical_area_fraction']*100:.2f}% of A",
                    (row["model_contact_area_m2"], row["breakout_force_MN"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=7)
    ax.set_xlabel("ice/hull contact area [m$^2$]")
    ax.set_ylabel("force [MN]")
    ax.set_title("Breakout criterion at T = -10 degC, freshwater ice (markers = model area)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    path = FIG / "fig4_breakout_vs_thrust.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    strength = strength_validation()
    breakout = breakout_validation()
    kinetics = kinetics_table()
    degradation = degradation_table()
    sensitivity = sensitivity_table()

    figures = {
        "fig1_adfreeze_strength": fig_strength(strength),
        "fig2_contact_time_kinetics": fig_kinetics(),
        "fig3_degradation_mechanisms": fig_degradation(degradation),
        "fig4_breakout_vs_thrust": fig_breakout(breakout),
    }

    results = {
        "node": "W3-N02",
        "title": "Ice-structure adfreeze bond mechanics and beset-vessel breakout criterion",
        "module": "src/tensorphysics/ice",
        "criterion_a": strength,
        "criterion_b": breakout,
        "kinetics": kinetics,
        "degradation": degradation,
        "sensitivity": sensitivity,
        "parameter_provenance": PARAMETER_PROVENANCE,
        "seawater_freezing_point_32gkg_C": round(seawater_freezing_point(32.0), 4),
        "figures": {k: str(v.relative_to(ROOT)) for k, v in figures.items()},
        "verdicts": {
            "a_strength_vs_frederking_lt15pct": strength["pass"],
            "b_breakout_vs_closed_form_lt10pct": breakout["pass"],
        },
    }
    out_json = OUT / "w3n02_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"wrote {out_json}")
    print("criterion (a) max |err| =", strength["max_abs_error_pct_smooth"], "% (smooth),",
          strength["max_abs_error_pct_hull"], "% (Ra=2.5 um) -> pass:", strength["pass"])
    print("criterion (b) max area dev =", breakout["max_abs_area_dev_pct"], "%,",
          "max force dev =", breakout["max_abs_force_dev_pct"], "% -> pass:", breakout["pass"])
    for row in breakout["rows"]:
        print(f"  {row['ship']}: A={row['model_contact_area_m2']} m2, "
              f"F_bo={row['breakout_force_MN']} MN, T={row['bollard_thrust_MN']} MN, "
              f"critical area fraction={row['critical_area_fraction']*100:.2f}%")
    print("figures:", [k for k in figures])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())