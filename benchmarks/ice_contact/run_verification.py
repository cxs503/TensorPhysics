#!/usr/bin/env python3
"""Verification and figure runner for the ice/structure contact phase-change model.

Produces, under ``artifacts/ice_contact_m1212/``:

* ``results.json``  -- every acceptance-criterion number, machine generated;
* ``pytest.log``    -- the test-suite run (criterion (f));
* ``figures/*.png`` -- the figures embedded in the report.

Nothing in the report is typed by hand: the report renderer reads only these
artefacts.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorphysics.ice_contact import verification as V  # noqa: E402
from tensorphysics.ice_contact.adhesion import IceAdhesionModel  # noqa: E402
from tensorphysics.ice_contact.competition import (  # noqa: E402
    melting_onset_velocity,
    scan_mechanism_map,
)
from tensorphysics.ice_contact.film import ContactState, FilmModel  # noqa: E402
from tensorphysics.ice_contact.pressure_melting import (  # noqa: E402
    clapeyron_slope,
    melt_temperature,
    melt_temperature_linear,
    pressure_melting_threshold,
)
from tensorphysics.ice_contact.properties import IceWaterProperties  # noqa: E402
from tensorphysics.ice_contact.stefan import (  # noqa: E402
    neumann_front,
    neumann_lambda,
    solve_stefan_1d,
)
from tensorphysics.ice_contact.tribology import (  # noqa: E402
    StribeckParams,
    friction_coefficient,
    lubrication_regime,
)

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "ice_contact_m1212"
FIGURE_DIR = ARTIFACT_DIR / "figures"
PROPS = IceWaterProperties()

MECHANISM_COLORS = {
    "friction_melting": "#c1440e",
    "pressure_melting": "#1f6f8b",
    "refreezing": "#4f6d7a",
}


def run_pytest() -> dict:
    """Run the ice-contact test suite and parse the summary line (criterion f)."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_ice_contact_pressure_melting.py",
        "tests/test_ice_contact_stefan.py",
        "tests/test_ice_contact_film.py",
        "tests/test_ice_contact_adhesion.py",
        "tests/test_ice_contact_tribology.py",
        "tests/test_ice_contact_competition.py",
        "tests/test_ice_contact_end_to_end.py",
        "-q",
        "--no-header",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}", "PATH": "/usr/bin:/bin"},
    )
    output = completed.stdout + completed.stderr
    (ARTIFACT_DIR / "pytest.log").write_text(output)
    match = re.search(r"(\d+) passed", output)
    passed = int(match.group(1)) if match else 0
    failed_match = re.search(r"(\d+) failed", output)
    failed = int(failed_match.group(1)) if failed_match else 0
    return {
        "criterion": "f_pytest",
        "passed": bool(passed >= 18 and failed == 0),
        "tests_passed": passed,
        "tests_failed": failed,
        "threshold": 18,
        "command": " ".join(command),
    }


def figure_pressure_melting() -> str:
    pressures = np.linspace(0.0, 2.0e8, 200)
    exact = np.array([melt_temperature(p) for p in pressures]) - PROPS.melting_temperature
    linear = np.array([melt_temperature_linear(p) for p in pressures]) - PROPS.melting_temperature
    threshold = pressure_melting_threshold(263.15)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(pressures / 1e6, exact, label="exact Clapeyron integral", color="#1f6f8b", lw=2)
    ax.plot(pressures / 1e6, linear, "--", label="tangent line (dT/dp = -0.0742 K/MPa)", color="#c1440e")
    ax.axhline(-10.0, color="0.4", lw=1, ls=":")
    ax.axvline(threshold / 1e6, color="0.4", lw=1, ls=":")
    ax.annotate(
        f"ice at -10 $^\\circ$C melts\nabove {threshold/1e6:.1f} MPa",
        xy=(threshold / 1e6, -10.0),
        xytext=(30, -6.0),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    ax.set_xlabel("contact pressure [MPa]")
    ax.set_ylabel("melting point depression [K]")
    ax.set_title("Pressure melting: Clausius-Clapeyron melting line of ice Ih")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    path = FIGURE_DIR / "fig1_pressure_melting.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ARTIFACT_DIR))


def figure_stefan() -> str:
    delta_t = 10.0
    stefan_number = PROPS.c_water * delta_t / PROPS.latent_heat_fusion
    lam = neumann_lambda(stefan_number)
    t_end = 1.0e-3
    sample_times = [t_end * f for f in (0.2, 0.4, 0.6, 0.8, 1.0)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    times = np.linspace(0.0, t_end, 100)
    ax.plot(
        times * 1e3,
        [neumann_front(t, PROPS.alpha_water, lam) * 1e6 for t in times],
        color="#1f6f8b",
        lw=2,
        label="Neumann similarity solution",
    )
    for cells, marker in ((150, "o"), (300, "s"), (600, "^")):
        solution = solve_stefan_1d(
            length=4.0e-5,
            n_cells=cells,
            t_end=t_end,
            melting_temperature=PROPS.melting_temperature,
            wall_temperature=PROPS.melting_temperature - delta_t,
            initial_temperature=PROPS.melting_temperature,
            rho=PROPS.rho_water,
            c=PROPS.c_water,
            k=PROPS.k_water,
            latent_heat=PROPS.latent_heat_fusion,
            mode="freezing",
            far_boundary="adiabatic",
            sample_times=sample_times,
        )
        ax.plot(
            np.array(solution["times"]) * 1e3,
            np.array(solution["front"]) * 1e6,
            marker=marker,
            ls="none",
            ms=6,
            label=f"enthalpy solver, {cells} cells",
        )
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("refrozen thickness [µm]")
    ax.set_title(
        f"Refreezing (Stefan) verification: $\\Delta T$ = {delta_t:.0f} K, "
        f"Ste = {stefan_number:.3f}, $\\lambda$ = {lam:.4f}"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    path = FIGURE_DIR / "fig2_stefan_refreezing.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ARTIFACT_DIR))


def figure_steady_map() -> str:
    model = FilmModel()
    velocities = np.geomspace(0.05, 10.0, 25)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for pressure, color in zip((0.5e6, 1.0e6, 2.0e6, 4.0e6), ("#8aa1b1", "#1f6f8b", "#3a7d44", "#c1440e")):
        thicknesses, mus = [], []
        for velocity in velocities:
            t_melt = melt_temperature(pressure)
            state = ContactState(
                pressure=pressure,
                velocity=float(velocity),
                ice_temperature=263.15,
                structure_temperature=t_melt,
            )
            steady = model.steady_state(state)
            thicknesses.append(steady["film_thickness"] if steady["exists"] else np.nan)
            mus.append(steady["friction_coefficient"])
        ax.plot(velocities, np.array(thicknesses) * 1e6, "o-", ms=3, color=color,
                label=f"p = {pressure/1e6:.1f} MPa")
        ax2.plot(velocities, mus, "o-", ms=3, color=color, label=f"p = {pressure/1e6:.1f} MPa")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sliding speed [m/s]")
    ax.set_ylabel("steady film thickness [µm]")
    ax.set_title("Steady melt film at -10 $^\\circ$C (equilibrated slider)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("sliding speed [m/s]")
    ax2.set_ylabel("friction coefficient [-]")
    ax2.set_title("Friction along the extended Stribeck curve")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    path = FIGURE_DIR / "fig3_steady_film_map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ARTIFACT_DIR))


def figure_case_histories(cases: list[dict]) -> str:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for case, color in zip(cases, ("#c1440e", "#1f6f8b", "#3a7d44", "#4f6d7a")):
        history = case["history"]
        times = np.array(history["time"]) * 1e3
        ax.plot(times, np.maximum(np.array(history["film_thickness"]), 1e-10) * 1e6,
                color=color, lw=1.8, label=case["name"])
        ax2.plot(times, history["friction_coefficient"], color=color, lw=1.8, label=case["name"])
    ax.set_yscale("log")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("film thickness [µm]")
    ax.set_title("Melt film time histories")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax2.set_xlabel("time [ms]")
    ax2.set_ylabel("friction coefficient [-]")
    ax2.set_title("Friction coefficient time histories")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    path = FIGURE_DIR / "fig4_case_histories.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ARTIFACT_DIR))


def figure_mechanism_map() -> str:
    model = FilmModel(adhesion=IceAdhesionModel())
    pressures = np.geomspace(2.0e5, 8.0e7, 14)
    velocities = np.geomspace(1.0e-3, 10.0, 14)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, temperature in zip(axes, (263.15, 272.5)):
        rows = scan_mechanism_map(
            model,
            pressures=[float(p) for p in pressures],
            velocities=[float(v) for v in velocities],
            temperatures=[temperature],
            equilibrated_structure=True,
        )
        for row in rows:
            ax.scatter(
                row["pressure"] / 1e6,
                row["velocity"],
                s=18,
                color=MECHANISM_COLORS[row["dominant_mechanism"]],
                alpha=0.85,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("contact pressure [MPa]")
        ax.set_title(f"dominant mechanism, ice at {temperature - PROPS.melting_temperature:+.2f} $^\\circ$C")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("sliding speed [m/s]")
    handles = [
        plt.Line2D([], [], marker="o", ls="none", color=color, label=label.replace("_", " "))
        for label, color in MECHANISM_COLORS.items()
    ]
    axes[1].legend(handles=handles, fontsize=8, loc="lower right")
    path = FIGURE_DIR / "fig5_mechanism_map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ARTIFACT_DIR))


def figure_adhesion() -> str:
    model = IceAdhesionModel()
    temperatures = PROPS.melting_temperature - np.linspace(0.2, 45.0, 200)
    strength = np.array([model.adhesion_strength(t) for t in temperatures])
    qll = np.array([model.qll_thickness(t) for t in temperatures]) * 1e9
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(temperatures - PROPS.melting_temperature, strength / 1e6, color="#c1440e", lw=2,
            label="adhesion strength (model)")
    ax.set_xlabel("ice temperature [°C]")
    ax.set_ylabel("interfacial shear strength [MPa]", color="#c1440e")
    ax.tick_params(axis="y", labelcolor="#c1440e")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(temperatures - PROPS.melting_temperature, qll, color="#1f6f8b", ls="--", lw=1.5,
             label="quasi-liquid layer thickness")
    ax2.set_ylabel("QLL thickness [nm]", color="#1f6f8b")
    ax2.tick_params(axis="y", labelcolor="#1f6f8b")
    ax.set_title("Ice adhesion on steel: peak at -10 $^\\circ$C, premelting collapse at 0 $^\\circ$C")
    path = FIGURE_DIR / "fig6_adhesion.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ARTIFACT_DIR))


def figure_stribeck() -> str:
    params = StribeckParams()
    thicknesses = np.geomspace(1e-9, 1e-4, 200)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for velocity, color in zip((0.01, 0.1, 1.0, 10.0), ("#8aa1b1", "#1f6f8b", "#3a7d44", "#c1440e")):
        mus = [
            friction_coefficient(h, 1.0e6, velocity, 263.15, params, PROPS) for h in thicknesses
        ]
        ax.plot(thicknesses * 1e6, mus, color=color, lw=1.8, label=f"V = {velocity} m/s")
    for limit, label in ((params.roughness * params.boundary_limit, "boundary|mixed"),
                         (params.roughness * params.hydrodynamic_limit, "mixed|hydrodynamic")):
        ax.axvline(limit * 1e6, color="0.5", ls=":", lw=1)
        ax.annotate(label, xy=(limit * 1e6, 0.005), rotation=90, fontsize=7, color="0.4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("film thickness [µm]")
    ax.set_ylabel("friction coefficient [-]")
    ax.set_title("Extended Stribeck curve vs melt-film thickness (p = 1 MPa, -10 $^\\circ$C)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    path = FIGURE_DIR / "fig7_stribeck.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ARTIFACT_DIR))


def main() -> dict:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "runs").mkdir(exist_ok=True)

    results = V.run_all_criteria()
    results["f_pytest"] = run_pytest()

    cases = V.case_studies()
    results["figures"] = {
        "fig1_pressure_melting": figure_pressure_melting(),
        "fig2_stefan_refreezing": figure_stefan(),
        "fig3_steady_film_map": figure_steady_map(),
        "fig4_case_histories": figure_case_histories(cases),
        "fig5_mechanism_map": figure_mechanism_map(),
        "fig6_adhesion": figure_adhesion(),
        "fig7_stribeck": figure_stribeck(),
    }

    model = FilmModel()
    onset_samples = []
    for pressure in (0.5e6, 1.0e6, 2.0e6, 4.0e6):
        state = ContactState(
            pressure=pressure,
            velocity=1.0,
            ice_temperature=263.15,
            structure_temperature=melt_temperature(pressure),
        )
        onset_samples.append(
            {
                "pressure": pressure,
                "onset_velocity": melting_onset_velocity(model, state),
                "onset_velocity_analytic": V.melting_onset_velocity_analytic(model, state),
            }
        )
    results["onset_samples"] = onset_samples
    results["clapeyron_slope_K_per_MPa"] = clapeyron_slope(PROPS) * 1.0e6
    results["provenance"] = {
        "module": "tensorphysics.ice_contact",
        "properties": {
            "rho_ice": PROPS.rho_ice,
            "rho_water": PROPS.rho_water,
            "latent_heat_fusion": PROPS.latent_heat_fusion,
            "k_ice": PROPS.k_ice,
            "k_water": PROPS.k_water,
            "melting_temperature": PROPS.melting_temperature,
        },
        "figure_dir": str(FIGURE_DIR.relative_to(REPO_ROOT)),
    }
    payload = json.dumps(results, indent=2, sort_keys=True)
    (ARTIFACT_DIR / "results.json").write_text(payload)
    summary = results["summary"]["criteria"]
    print(json.dumps({"criteria": summary, "pytest": results["f_pytest"]}, indent=2))
    return results


if __name__ == "__main__":
    main()