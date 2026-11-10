"""IN13-W3-N05 study driver: ice-ridge consolidation thermodynamics.

Runs the full parameter sweep of the campaign node and writes every number and
figure that the report quotes into ``runs/in13_w3n05/``:

    results.json          all machine-read numbers (no hand-typed values)
    sweep_consolidation.csv   phi x T_air x h_keel x t -> h_c, degree, sigma_c
    figures/*.png         the report figures

Usage
-----
    python benchmarks/ice_ridge_consolidation/run_ridge_consolidation.py \
        [--out runs/in13_w3n05]

The script is deterministic (closed-form physics, no random sampling); the
environment block records the git revision and library versions so a rerun can be
proven to reproduce the artefact.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import platform
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.ice.resistance import ShipGeometry, ridge_resistance  # noqa: E402
from tensorphysics.ice_mechanics.ridge_consolidation import (  # noqa: E402
    consolidation_state,
    consolidation_time_to_thickness,
    consolidated_strength,
    keel_properties,
    ridge_total_resistance,
    solid_ice_strength,
    stefan_constant,
    stefan_number,
)
from tensorphysics.ice_mechanics.ridge_verification import (  # noqa: E402
    DAY,
    criterion_c_enthalpy_reference,
    run_all_criteria,
)

PHI_SWEEP = (0.30, 0.35, 0.40, 0.45, 0.50)
T_SWEEP = (-10.0, -25.0, -40.0)
H_SWEEP = (5.0, 10.0, 20.0)
T_DAYS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 30.0, 45.0, 60.0)
SPEED = 1.0  # m/s ramming speed for the resistance coupling
SHIP = ShipGeometry()


# ── helpers ────────────────────────────────────────────────────────────


def _env() -> dict:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # pragma: no cover
        rev = "unknown"
    return {
        "git_revision": rev,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "deterministic": True,
        "random_seed": None,
    }


def _sweep() -> list[dict]:
    rows: list[dict] = []
    for phi in PHI_SWEEP:
        for t_air in T_SWEEP:
            for h_k in H_SWEEP:
                for days in T_DAYS:
                    st = consolidation_state(
                        days * DAY, keel_depth=h_k, air_temperature=t_air, porosity=phi
                    )
                    sigma = consolidated_strength(st.degree, phi, st.mean_temperature)
                    rows.append(
                        {
                            "porosity": phi,
                            "air_temperature_C": t_air,
                            "keel_depth_m": h_k,
                            "t_days": days,
                            "h_consolidated_m": st.h_consolidated,
                            "degree": st.degree,
                            "mean_temperature_C": st.mean_temperature,
                            "sigma_c_Pa": sigma,
                            "sigma_c_MPa": sigma / 1e6,
                            "stefan_number": st.ste_upper,
                            "kc": st.kc_upper,
                            "kappa_eff_m2s": st.props.kappa_eff,
                            "ste_lower": st.ste_lower,
                            "h_lower_m": st.h_lower,
                        }
                    )
    return rows


def _time_scales() -> dict:
    targets = (0.3, 0.5, 0.8, 1.2, 2.0)
    table = []
    for phi in (0.30, 0.40, 0.50):
        for t_air in T_SWEEP:
            row = {"porosity": phi, "air_temperature_C": t_air}
            for h in targets:
                row[f"t_days_h{str(h).replace('.', 'p')}"] = (
                    consolidation_time_to_thickness(h, air_temperature=t_air, porosity=phi) / DAY
                )
            table.append(row)
    degree_table = []
    for phi in (0.30, 0.40, 0.50):
        for h_k in H_SWEEP:
            for t_air in T_SWEEP:
                row = {"porosity": phi, "keel_depth_m": h_k, "air_temperature_C": t_air}
                for target in (0.1, 0.3, 0.5, 0.9):
                    t = consolidation_time_to_thickness(
                        target * h_k, air_temperature=t_air, porosity=phi
                    )
                    row[f"t_days_C{str(target).replace('.', 'p')}"] = t / DAY
                degree_table.append(row)
    return {
        "thickness_targets_m": list(targets),
        "time_to_thickness": table,
        "time_to_degree": degree_table,
        "snow_delay_0p3m": [
            {
                "air_temperature_C": t_air,
                "bare_days_h0p5": consolidation_time_to_thickness(
                    0.5, air_temperature=t_air
                )
                / DAY,
                "snow_days_h0p5": consolidation_time_to_thickness(
                    0.5, air_temperature=t_air, snow_thickness=0.3
                )
                / DAY,
            }
            for t_air in T_SWEEP
        ],
    }


def _resistance_case() -> dict:
    """Case study: 8 m keel, 1.6 m sail, -25 degC, first winter."""
    h_k, t_air, phi = 8.0, -25.0, 0.40
    days = T_DAYS
    rows = []
    for d in days:
        st = consolidation_state(d * DAY, keel_depth=h_k, air_temperature=t_air, porosity=phi)
        out = ridge_total_resistance(
            h_k,
            SPEED,
            SHIP,
            degree=st.degree,
            porosity=phi,
            mean_temperature=st.mean_temperature,
        )
        rows.append(
            {
                "t_days": d,
                "h_consolidated_m": st.h_consolidated,
                "degree": st.degree,
                "sigma_c_MPa": out["sigma_c_MPa"],
                "R_sail_kN": out["R_sail_kN"],
                "R_keel_kN": out["R_keel_kN"],
                "R_plate_kN": out["keel"]["R_plate_N"] / 1e3,
                "R_rubble_kN": (
                    out["keel"]["R_rubble_buoyancy_N"] + out["keel"]["R_rubble_momentum_N"]
                )
                / 1e3,
                "R_ridge_kN": out["R_ridge_kN"],
                "flags": out["flags"],
            }
        )
    # reference limits: the existing ridge law (truncated at its own validity
    # limit) and the C = 0 limit of the coupled model at the *requested* depth
    rubble = ridge_resistance(h_k, SPEED, SHIP, porosity=phi, strict=False)
    from tensorphysics.ice_mechanics.ridge_consolidation import keel_resistance_coupled

    extrapolated = keel_resistance_coupled(
        h_k, SPEED, SHIP, degree=0.0, sigma_c=1.0, porosity=phi
    )
    credit = {
        "keel_depth_m": h_k,
        "sail_height_m": 0.2 * h_k,
        "speed_ms": SPEED,
        "porosity": phi,
        "air_temperature_C": t_air,
        "ship": SHIP.as_dict(),
        "R_rubble_law_kN": rubble["R_total_N"] / 1e3,
        "R_rubble_law_flags": rubble["flags"],
        "R_rubble_law_effective_depth_m": rubble["h_keel_m"],
        "R_rubble_extrapolated_kN": extrapolated["R_keel_N"] / 1e3,
        "R_rubble_extrapolated_flags": extrapolated["flags"],
        "note": "the 8 m keel is outside the Keinonen law validity "
        "(h_k <= min(0.15 B, 4 m)), so the unconsolidated baseline is an "
        "extrapolation of the same coefficient set, flagged as such",
    }
    return {"rows": rows, "credit": credit}


def _degree_sensitivity() -> dict:
    """Sensitivity of the coupling to the two SEMI-EMPIRICAL closures."""
    h_k, t_air = 8.0, -25.0
    st = consolidation_state(30 * DAY, keel_depth=h_k, air_temperature=t_air, porosity=0.40)
    by_exponent = []
    for q in (0.5, 1.0, 2.0):
        sigma = consolidated_strength(st.degree, 0.40, st.mean_temperature, exponent=q)
        out = ridge_total_resistance(
            h_k, SPEED, SHIP, degree=st.degree, sigma_c=sigma, porosity=0.40
        )
        by_exponent.append(
            {
                "q": q,
                "sigma_c_MPa": sigma / 1e6,
                "R_keel_kN": out["R_keel_kN"],
                "R_ridge_kN": out["R_ridge_kN"],
            }
        )
    by_conductivity = []
    for mix, k_eff in (
        ("geometric", keel_properties(0.40).k_eff),
        ("voigt_parallel", keel_properties(0.40).k_voigt),
        ("reuss_series", keel_properties(0.40).k_reuss),
        ("maxwell_eucken", keel_properties(0.40).k_maxwell),
    ):
        props = keel_properties(0.40)
        # rescale kappa by the alternative mixing law, everything else unchanged
        kappa = k_eff / props.cvol
        h_c = st.kc_upper * (kappa * st.t) ** 0.5
        degree = min(h_c / h_k, 1.0)
        sigma = consolidated_strength(degree, 0.40, st.mean_temperature)
        out = ridge_total_resistance(
            h_k, SPEED, SHIP, degree=degree, sigma_c=sigma, porosity=0.40
        )
        by_conductivity.append(
            {
                "mixing_law": mix,
                "k_eff": k_eff,
                "kappa_eff": kappa,
                "h_consolidated_m": h_c,
                "degree": degree,
                "R_ridge_kN": out["R_ridge_kN"],
            }
        )
    return {
        "base": {
            "t_days": 30.0,
            "degree": st.degree,
            "h_consolidated_m": st.h_consolidated,
            "sigma_c_MPa": consolidated_strength(st.degree, 0.40, st.mean_temperature) / 1e6,
        },
        "strength_exponent": by_exponent,
        "conductivity_mixing": by_conductivity,
    }


# ── figures ────────────────────────────────────────────────────────────


def _fig_front_growth(path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    times = np.linspace(0.0, 60.0, 400)
    for t_air, colour in zip(T_SWEEP, ("#1f77b4", "#d62728", "#2ca02c")):
        for phi, style in zip((0.30, 0.40, 0.50), ("-", "--", ":")):
            h = [
                consolidation_state(t * DAY, keel_depth=20.0, air_temperature=t_air, porosity=phi)
                .h_consolidated
                for t in times
            ]
            ax.plot(times, h, style, color=colour, lw=1.6,
                    label=f"T_air={t_air:g} C, phi={phi:.2f}")
    ax.set_xlabel("time [days]")
    ax.set_ylabel("consolidated thickness $h_c$ [m]")
    ax.set_title("Modified Stefan growth $h_c = 2\\lambda(Ste)\\sqrt{\\kappa_{eff} t}$ (20 m keel, upper front)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fig_degree(path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    times = np.linspace(0.0, 60.0, 300)
    for h_k, colour in zip(H_SWEEP, ("#444444", "#ff7f0e", "#9467bd")):
        for t_air, style in zip(T_SWEEP, ("-", "--", ":")):
            d = [
                consolidation_state(t * DAY, keel_depth=h_k, air_temperature=t_air, porosity=0.40)
                .degree
                for t in times
            ]
            ax.plot(times, np.array(d) * 100.0, style, color=colour, lw=1.6,
                    label=f"h_k={h_k:g} m, T_air={t_air:g} C")
    ax.set_xlabel("time [days]")
    ax.set_ylabel("consolidation degree C [%]")
    ax.set_title("Consolidation degree of the keel (phi=0.40, lower front stagnant)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fig_strength_resistance(path: pathlib.Path, case: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    rows = case["rows"]
    t = [r["t_days"] for r in rows]
    axes[0].plot(t, [r["sigma_c_MPa"] for r in rows], "o-", color="#d62728")
    axes[0].set_xlabel("time [days]")
    axes[0].set_ylabel("$\\sigma_c$ [MPa]")
    axes[0].set_title("Consolidated-layer strength, $\\sigma_c=\\sigma_{ice}(1-\\phi)C^q$")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, [r["R_plate_kN"] for r in rows], "s-", label="Lindqvist plate $h_c$", lw=1.5)
    axes[1].plot(t, [r["R_rubble_kN"] for r in rows], "^-", label="rubble (buoyancy+momentum)", lw=1.5)
    axes[1].plot(t, [r["R_sail_kN"] for r in rows], "d-", label="sail", lw=1.5)
    axes[1].plot(t, [r["R_ridge_kN"] for r in rows], "k-", label="total $R_{ridge}$", lw=2.2)
    axes[1].axhline(case["credit"]["R_rubble_extrapolated_kN"], color="grey", ls=":", lw=1.4,
                    label="unconsolidated rubble, C=0 (extrapolated to 8 m)")
    axes[1].set_xlabel("time [days]")
    axes[1].set_ylabel("resistance [kN]")
    axes[1].set_title("Consolidation-resistance coupling, 8 m keel / 1.6 m sail / 1 m s$^{-1}$")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fig_mixing(path: pathlib.Path) -> None:
    phi = np.linspace(0.25, 0.55, 200)
    k_eff, k_v, k_r, k_m = [], [], [], []
    from tensorphysics.ice_mechanics.ridge_consolidation import maxwell_eucken_conductivity

    for p in phi:
        solid = 1.0 - p
        k_eff.append(2.1**solid * 0.55**p)
        k_v.append(solid * 2.1 + p * 0.55)
        k_r.append(1.0 / (solid / 2.1 + p / 0.55))
        k_m.append(maxwell_eucken_conductivity(float(p)))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(phi, k_v, label="Voigt (parallel) bound", color="#1f77b4")
    ax.plot(phi, k_m, "--", label="Maxwell-Eucken", color="#2ca02c")
    ax.plot(phi, k_eff, "-", lw=2.4, label="geometric mean (model)", color="#000000")
    ax.plot(phi, k_r, label="Reuss (series) bound", color="#d62728")
    ax.axvspan(0.30, 0.50, color="grey", alpha=0.12, label="keel porosity range")
    ax.set_xlabel("keel porosity $\\phi$")
    ax.set_ylabel("$k_{eff}$ [W m$^{-1}$ K$^{-1}$]")
    ax.set_title("Effective conductivity of the brine-saturated rubble keel")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fig_literature(path: pathlib.Path) -> None:
    """Model output against published observed ranges (anchoring, not fitting)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    depths = np.linspace(5.0, 20.0, 60)
    for t_air, colour in zip(T_SWEEP, ("#1f77b4", "#d62728", "#2ca02c")):
        h = [
            consolidation_state(100 * DAY, keel_depth=float(d), air_temperature=t_air)
            .h_consolidated
            for d in depths
        ]
        axes[0].plot(depths, h, color=colour, lw=2,
                     label=f"model, T_air={t_air:g} C, 100 d, phi=0.40")
    axes[0].axhspan(0.2, 1.0, color="grey", alpha=0.25,
                    label="observed consolidated layer 0.2-1.0 m (Timco & Weeks 2010)")
    axes[0].axhline(3.6, color="k", ls=":", lw=1.0, label="ridge-law validity limit 3.6 m")
    axes[0].set_xlabel("keel depth $h_k$ [m]")
    axes[0].set_ylabel("$h_c$ after one winter [m]")
    axes[0].set_title("Consolidated thickness vs observed range")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    c = np.linspace(0.0, 1.0, 200)
    for phi, style in zip((0.30, 0.40, 0.50), ("-", "--", ":")):
        ratio = [(1.0 - phi) * float(cc) for cc in c]
        axes[1].plot(c, ratio, style, lw=1.8, label=f"model, phi={phi:.2f}")
    axes[1].axhspan(0.2, 0.5, color="orange", alpha=0.2,
                    label="partially consolidated keel / level ice 0.2-0.5")
    axes[1].set_xlabel("consolidation degree C")
    axes[1].set_ylabel("$\\sigma_c/\\sigma_{ice}$")
    axes[1].set_title("Strength ratio closure $\\sigma_c=\\sigma_{ice}(1-\\phi)C^q$")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fig_verification(path: pathlib.Path, enthalpy: dict) -> None:
    rows = enthalpy["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    t = [r["t_days"] for r in rows]
    axes[0].plot(t, [r["front_analytic_m"] for r in rows], "k-", lw=2.2, label="similarity 2$\\lambda\\sqrt{\\kappa t}$")
    axes[0].plot(t, [r["front_enthalpy_fine_m"] for r in rows], "o", ms=7, mfc="none",
                 label="enthalpy solver, 800 cells")
    axes[0].plot(t, [r["front_enthalpy_coarse_m"] for r in rows], "s", ms=5, mfc="none",
                 label="enthalpy solver, 200 cells")
    axes[0].set_xlabel("time [days]")
    axes[0].set_ylabel("consolidated thickness [m]")
    axes[0].set_title("Front position: model vs independent numerical solution")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    ste = np.linspace(0.01, 1.0, 200)
    exact = np.array([stefan_constant(float(s)) for s in ste])
    qs = np.sqrt(2.0 * ste)
    axes[1].plot(ste, np.abs(qs - exact) / exact * 100.0, color="#d62728", lw=2)
    axes[1].axhline(3.0, color="grey", ls="--", lw=1.2, label="3 % reference")
    axes[1].axvspan(0.19, 0.88, color="orange", alpha=0.12, label="Ste range of the sweep")
    axes[1].set_xlabel("Stefan number Ste")
    axes[1].set_ylabel("quasi-steady error [%]")
    axes[1].set_title("Why the model uses the exact $2\\lambda(Ste)$, not $\\sqrt{2Ste}$")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/in13_w3n05")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    (out / "figures").mkdir(parents=True, exist_ok=True)

    sweep = _sweep()
    with (out / "sweep_consolidation.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)

    criteria = run_all_criteria()
    enthalpy = criterion_c_enthalpy_reference()
    case = _resistance_case()
    sensitivity = _degree_sensitivity()
    time_scales = _time_scales()

    _fig_front_growth(out / "figures" / "fig1_front_growth.png")
    _fig_degree(out / "figures" / "fig2_consolidation_degree.png")
    _fig_strength_resistance(out / "figures" / "fig3_strength_resistance.png", case)
    _fig_mixing(out / "figures" / "fig4_conductivity_mixing.png")
    _fig_verification(out / "figures" / "fig5_verification.png", enthalpy)
    _fig_literature(out / "figures" / "fig6_literature_anchor.png")

    props = keel_properties(0.40)
    summary = {
        "node": "IN13-W3-N05",
        "title": "Ice-ridge consolidation thermodynamics: keel macro-pore refreezing, "
        "consolidated-layer strength evolution and consolidation-resistance closure",
        "module": "src/tensorphysics/ice_mechanics/ridge_consolidation.py",
        "harness": "src/tensorphysics/ice_mechanics/ridge_verification.py",
        "tests": "tests/test_ridge_consolidation.py",
        "environment": _env(),
        "parameters": {
            "phi_keel": list(PHI_SWEEP),
            "T_air_C": list(T_SWEEP),
            "h_keel_m": list(H_SWEEP),
            "t_days": list(T_DAYS),
            "speed_ms": SPEED,
            "reference_properties_phi0p40": props.as_dict(),
        },
        "acceptance": {
            "criteria_passed": criteria["all_passed"],
            "n_criteria": criteria["n_criteria"],
            "criteria": criteria["criteria"],
        },
        "enthalpy_reference": enthalpy,
        "time_scales": time_scales,
        "resistance_case": case,
        "sensitivity": sensitivity,
        "sweep_rows": len(sweep),
        "headline": {
            "h_c_30d_25C_phi0p40_m": consolidation_state(
                30 * DAY, keel_depth=8.0, air_temperature=-25.0
            ).h_consolidated,
            "degree_30d_25C_phi0p40": consolidation_state(
                30 * DAY, keel_depth=8.0, air_temperature=-25.0
            ).degree,
            "sigma_ice_MPa_at_minus13p4C": solid_ice_strength(-13.364) / 1e6,
            "R_ridge_kN_30d_8m_keel": case["rows"][8]["R_ridge_kN"],
            "R_rubble_law_kN_C0_truncated": case["credit"]["R_rubble_law_kN"],
            "R_rubble_extrapolated_kN_C0": case["credit"]["R_rubble_extrapolated_kN"],
            "consolidation_uplift_factor": case["rows"][8]["R_ridge_kN"]
            / case["credit"]["R_rubble_extrapolated_kN"],
        },
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["headline"], indent=2))
    print(f"criteria passed: {criteria['all_passed']} ({criteria['n_criteria']})")
    print(f"artefacts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())