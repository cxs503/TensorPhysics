#!/usr/bin/env python3
"""Coupled ice-induced vibration / underwater radiated noise campaign run.

Produces the evidence bundle for campaign node P210W2-N06:

* ``report.json``   full chain payload for the design case + parameter studies
* ``metrics.json``  acceptance criteria C1-C5 plus headline quantities
* ``figures/*.png`` publication figures
* ``study.csv``     drift-speed / damping sweep table

Usage
-----
    PYTHONPATH=src python3 scenarios/ice_vib_noise_coupling/run_chain.py [--out-dir DIR]

The script is deterministic (fixed seeds, no wall-clock in the physics), writes
only inside ``--out-dir`` and prints a compact summary.  Every number quoted in
the documentation is read back from ``report.json`` — nothing is transcribed by
hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tensorphysics.ice_vibration.modal import HullParameters
from tensorphysics.ice_vib_noise import (
    CoupledCase,
    IceExcitationCase,
    HullGirderModes,
    RadiatorSurface,
    iso19906_informed_envelope,
    one_third_octave_bands,
    run_with_idempotency,
)
from tensorphysics.ice_vib_noise.chain import _run_core
from tensorphysics.ice_vib_noise.bands import G

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "ice_vib_noise_coupling"

HULL = HullParameters(
    length_m=130.0,
    breadth_m=28.0,
    draught_m=8.5,
    block_coefficient=0.725,
    displacement_kg=2.3e7,
    name="P210W2-N06 ice-class 130 m reference hull",
)
BENDING_STIFFNESS = 2.0e12

DESIGN_EXCITATION = IceExcitationCase(
    ice_thickness_m=0.8,
    contact_pressure_mpa=2.0,
    contact_width_m=3.0,
    drift_speed_m_s=1.0,
    flaking_length_m=0.5,
    duration_s=60.0,
    dt_s=0.005,
    bow_contact_length_m=6.0,
    seed=20261110,
)


def design_case() -> CoupledCase:
    return CoupledCase(
        hull=HULL,
        bending_stiffness_nm2=BENDING_STIFFNESS,
        n_modes=3,
        structural_damping=(0.015, 0.015, 0.02),
        excitation=DESIGN_EXCITATION,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 10,
            "figure.autolayout": True,
        }
    )
    return plt


def fig_load(core: Dict[str, Any], modes: HullGirderModes, path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0))
    t = core["t"]
    force = core["force"]
    ax = axes[0]
    ax.plot(t[t <= 12.0], force[t <= 12.0] / 1e6, lw=0.9, color="#1f4e79")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("$F_{ice}$ [MN]")
    ax.set_title("Ice crushing load history (first 12 s of the 60 s record)")
    ax = axes[1]
    freqs = core["freqs"]
    force_sel = force
    from tensorphysics.ice_vib_noise.chain import _amplitude_spectrum, _psd_from_amplitude

    ff, fa = _amplitude_spectrum(force_sel, core["dt"])
    psd = _psd_from_amplitude(ff, fa)
    ax.loglog(ff[1:], np.maximum(psd[1:], 1e-30), lw=0.9, color="#c00000", label="load PSD")
    for i, f in enumerate(modes.frequencies_hz):
        ax.axvline(f, ls="--", lw=0.8, color="#404040")
        ax.text(f * 1.05, np.max(psd[1:]) * 0.4, f"$f_{i + 1}$={f:.2f} Hz", fontsize=7, rotation=90)
    ax.set_xlim(0.1, 100.0)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("$S_{FF}$ [N$^2$/Hz]")
    ax.set_title("Excitation spectrum and hull-girder natural frequencies")
    ax.legend(fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def fig_vibration(core: Dict[str, Any], result: Dict[str, Any], path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4))
    t = core["t"]
    q = core["q"]
    ax = axes[0]
    for i in range(q.shape[0]):
        ax.plot(t, q[i] * 1e3, lw=0.9, label=f"mode {i + 1} (q$_1^{{max}}$={np.max(np.abs(q[i])) * 1e3:.1f} mm)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("modal displacement q$_i$ [mm]")
    ax.set_title("Modal superposition response (exact piecewise-linear integration)")
    ax.legend(fontsize=7, ncol=3)

    ax = axes[1]
    bands = result["bands"]
    fc = np.array(bands["centre_hz"])
    env = np.array(bands["acceleration_envelope_m_s2"])
    width = np.diff(np.log10(np.array([fc[0] / G ** 0.5] + list(fc)))) * 0.9
    for idx, (key, colour, label) in enumerate(
        (("midship", "#1f4e79", "midship"), ("bow", "#c00000", "bow"))
    ):
        acc = np.array(bands["acceleration_rms_m_s2"][key])
        ax.bar(np.log10(fc) + (idx - 0.5) * 0.05, acc, width=width, color=colour, alpha=0.75, label=label)
    ax.plot(np.log10(fc), env, "k--", lw=1.4, label="ISO 19906-informed envelope")
    ax.set_yscale("log")
    ax.set_xticks(np.log10(fc))
    ax.set_xticklabels([f"{f:.2g}" for f in fc], rotation=60, fontsize=7)
    ax.set_xlabel("1/3-octave band centre [Hz]")
    ax.set_ylabel("band RMS acceleration [m/s$^2$]")
    ax.set_title("1/3-octave acceleration versus habitability envelope")
    ax.legend(fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def fig_radiation(core: Dict[str, Any], result: Dict[str, Any], path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    rad = result["radiation"]
    f_grid = np.array(rad["resistance_grid"]["frequencies_hz"])
    ax = axes[0]
    for i, row in enumerate(rad["resistance_grid"]["per_mode_n_s_m"]):
        ax.loglog(f_grid, np.maximum(row, 1e-30), lw=1.0, label=f"mode {i + 1}")
        ax.plot(
            result["modes"]["frequencies_hz"][i],
            rad["modal_radiation_resistance_n_s_m"][i],
            "o",
            ms=4,
            color=f"C{i}",
        )
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("modal radiation resistance R$_i$ [N s/m]")
    ax.set_title("Rayleigh radiation resistance (markers at $f_{n,i}$)")
    ax.legend(fontsize=7)

    ax = axes[1]
    bands = result["bands"]
    fc = np.array(bands["centre_hz"])
    sl = np.array(bands["source_level_directional_max_db_1m"])
    sl_omni = np.array(bands["equivalent_omni_source_level_db"])
    ax.plot(fc, sl_omni, "o-", ms=3, lw=0.9, label="omni-directional equivalent (used in propagation)")
    ax.plot(fc, sl, "s--", ms=3, lw=1.0, label="directional maximum (baffle upper bound)")
    ax.set_xscale("log")
    ax.set_xlabel("1/3-octave band centre [Hz]")
    ax.set_ylabel("source level [dB re 1 µPa·m]")
    ax.set_title("Source spectrum at 1 m")
    ax.legend(fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def fig_noise(result: Dict[str, Any], path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bands = result["bands"]
    fc = np.array(bands["centre_hz"])
    sl_env = np.array(bands["equivalent_spl_envelope_db"])
    ax.plot(fc, sl_env, "k--", lw=1.2, label="equivalent SPL envelope (from the vibration envelope)")
    colours = {100.0: "#0b5394", 1000.0: "#38761d", 10000.0: "#b45f06"}
    for key, payload in result["propagation"]["by_range"].items():
        r = payload["range_m"]
        ax.plot(fc, payload["spl_bands_db"], "o-", ms=3, lw=1.0,
                color=colours.get(r, "grey"), label=f"RL at {r:g} m")
    thr = result["propagation"]["marine_mammal_thresholds"]["continuous_pts_db"]
    ax.axhline(thr, color="#c00000", ls=":", lw=1.2,
               label=f"NMFS(2018) continuous PTS {thr:.0f} dB")
    ax.set_xscale("log")
    ax.set_xlabel("1/3-octave band centre [Hz]")
    ax.set_ylabel("band SPL [dB re 1 µPa]")
    ax.set_title("Propagated 1/3-octave SPL versus criteria")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path)
    plt.close(fig)


def fig_energy_study(result: Dict[str, Any], study: List[Dict[str, Any]], path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    ax = axes[0]
    energy = result["energy"]
    labels = ["$E_{crush}$", "$W_{in}$", "$E_{struct}$", "$E_{rad}$", "residual"]
    residual = abs(
        energy["work_in_j"]
        - (
            energy["ledger"]["diss_struct"]
            + energy["ledger"]["strain"]
            + energy["ledger"]["kinetic"]
            + energy["radiated_acoustic_j_spectral"]
        )
    )
    values = [
        energy["crushing_energy_j"],
        energy["work_in_j"],
        energy["ledger"]["diss_struct"],
        energy["radiated_acoustic_j_spectral"],
        max(residual, 1e-12),
    ]
    ax.bar(labels, values, color=["#999999", "#1f4e79", "#38761d", "#c00000", "#e69138"])
    ax.set_yscale("log")
    ax.set_ylabel("energy over the record [J]")
    ax.set_title("Energy budget of the coupled chain")

    ax = axes[1]
    ratios = [row["frequency_ratio"] for row in study]
    acc_ratio = [row["midship_acceleration_max_ratio"] for row in study]
    sl = [row["max_source_level_db"] for row in study]
    ax.plot(ratios, acc_ratio, "o-", color="#1f4e79", label="max accel / envelope")
    ax.set_xlabel("$f_{crush}/f_1$")
    ax.set_ylabel("max band acceleration / envelope", color="#1f4e79")
    ax2 = ax.twinx()
    ax2.plot(ratios, sl, "s--", color="#c00000", label="max SL")
    ax2.set_ylabel("max SL [dB re 1 µPa·m]", color="#c00000")
    ax2.grid(False)
    ax.set_title("Drift-speed sweep")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_study() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base = design_case()
    for speed in (0.2, 0.5, 1.0, 1.5, 2.0):
        exc = replace(DESIGN_EXCITATION, drift_speed_m_s=speed)
        res = run_chain_light(replace(base, excitation=exc))
        rows.append(_study_row(res, f"drift_speed_{speed:g}", speed))
    for zeta in (0.0075, 0.03):
        res = run_chain_light(
            replace(base, structural_damping=(zeta, zeta, zeta * 1.33))
        )
        rows.append(_study_row(res, f"structural_damping_{zeta:g}", base.excitation.drift_speed_m_s,
                               extra={"structural_damping": zeta}))
    return rows


def run_chain_light(case: CoupledCase) -> Dict[str, Any]:
    """Chain evaluation without the two extra core runs of the linearity check."""

    from tensorphysics.ice_vib_noise.envelope import band_margins, iso19906_informed_envelope

    core = _run_core(case)
    env = iso19906_informed_envelope(case.envelope_plateau_rms_m_s2)
    a_lim = env.limit(core["centres"])
    margins = band_margins(core["centres"], core["accel_bands"]["midship"], a_lim)
    return {
        "core": core,
        "margins": margins,
        "max_sl": float(np.max(core["sl_omni"])),
        "energy_efficiency": core["e_rad_spectral"] / max(core["e_crush"], 1e-300),
        "work_in": core["ledger"]["work_in"],
    }


def _study_row(res: Dict[str, Any], label: str, drift_speed: float, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    core = res["core"]
    f1 = core["modes"].frequencies_hz[0]
    row = {
        "label": label,
        "drift_speed_m_s": drift_speed,
        "crushing_frequency_hz": drift_speed / 0.5,
        "frequency_ratio": (drift_speed / 0.5) / f1,
        "midship_acceleration_max_ratio": float(res["margins"]["max_ratio"]),
        "governing_band_hz": float(res["margins"]["governing_band_hz"]),
        "max_source_level_db": res["max_sl"],
        "acoustic_conversion_efficiency": res["energy_efficiency"],
        "work_in_j": res["work_in"],
    }
    row.update(extra or {})
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--with-pytest", action="store_true", help="run the node test module")
    args = parser.parse_args()
    out_dir = args.out_dir
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    case = design_case()
    result, idem = run_with_idempotency(case)
    result["run"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module": "tensorphysics.ice_vib_noise",
        "node": "P210W2-N06",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    core = _run_core(case)

    fig_load(core, result["modes"] and HullGirderModes(
        hull=case.hull,
        bending_stiffness_nm2=case.bending_stiffness_nm2,
        n_modes=case.n_modes,
        damping_ratios=case.structural_damping,
    ), out_dir / "figures" / "fig1_ice_load_and_spectrum.png")

    fig_vibration(core, result, out_dir / "figures" / "fig2_vibration_and_envelope.png")
    fig_radiation(core, result, out_dir / "figures" / "fig3_radiation_resistance_source.png")
    fig_noise(result, out_dir / "figures" / "fig4_propagated_spl.png")

    study = run_study()
    fig_energy_study(result, study, out_dir / "figures" / "fig5_energy_and_sweep.png")

    fieldnames = sorted({key for row in study for key in row})
    with (out_dir / "study.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(study)

    metrics = {
        "node": "P210W2-N06",
        "criteria": {
            "C1_zero_load_control": {
                "relative_residual": result["criteria"]["C1_zero_load_control"]["relative_residual"],
                "threshold": 1e-9,
                "pass": result["criteria"]["C1_zero_load_control"]["pass"],
            },
            "C2_modal_orthogonality": {
                "max_offdiag_mass_ratio": result["criteria"]["C2_modal_orthogonality"]["max_offdiag_mass_ratio"],
                "threshold": 1e-6,
                "pass": result["criteria"]["C2_modal_orthogonality"]["pass"],
            },
            "C3_energy_conservation": {
                "residual_over_work": result["criteria"]["C3_energy_conservation"]["residual_over_work"],
                "threshold": 0.01,
                "parseval_band_closure_rel_error": result["criteria"]["C3_energy_conservation"][
                    "parseval_band_closure_rel_error"
                ],
                "pass": result["criteria"]["C3_energy_conservation"]["pass"]
                and result["criteria"]["C3_energy_conservation"]["parseval_pass"],
            },
            "C4_envelope_consistency": {
                "max_gain_deviation_db": result["criteria"]["C4_envelope_consistency"]["max_gain_deviation_db"],
                "threshold_db": 1e-9,
                "pass": result["criteria"]["C4_envelope_consistency"]["pass"],
            },
            "C5_idempotency": {
                "digest_run_1": idem["digest_run_1"],
                "digest_run_2": idem["digest_run_2"],
                "pass": idem["pass"],
            },
        },
        "headline": {
            "hull_length_m": case.hull.length_m,
            "crushing_frequency_hz": result["load"]["crushing_frequency_hz"],
            "natural_frequencies_hz": result["modes"]["frequencies_hz"],
            "peak_ice_force_n": result["load"]["peak_force_n"],
            "crushing_energy_j": result["energy"]["crushing_energy_j"],
            "work_in_j": result["energy"]["work_in_j"],
            "radiated_acoustic_energy_j": result["energy"]["radiated_acoustic_j_spectral"],
            "acoustic_conversion_efficiency": result["energy"]["acoustic_conversion_efficiency"],
            "modal_radiation_damping_ratio": result["radiation"]["modal_radiation_damping_ratio"],
            "max_source_level_db": float(np.max(result["bands"]["source_level_db_1m"])),
            "max_directional_source_level_db": float(
                np.max(result["bands"]["source_level_directional_max_db_1m"])
            ),
            "max_directivity_index_db": float(np.max(result["radiation"]["directivity_index_db"])),
            "midship_acceleration_max_ratio": result["criteria"]["C4_envelope_consistency"][
                "physical_finding_C4b"
            ]["midship_acceleration_max_ratio"],
            "bow_acceleration_max_ratio": result["criteria"]["C4_envelope_consistency"][
                "physical_finding_C4b"
            ]["bow_acceleration_max_ratio"],
            "range_to_continuous_threshold_m": result["propagation"]["marine_mammal_thresholds"][
                "range_to_continuous_threshold_m"
            ],
            "modal_radiation_interference_max_rel": result["modal_radiation_interference"][
                "max_relative_deviation"
            ],
        },
        "all_pass": all(v["pass"] for v in [
            {
                "pass": result["criteria"]["C1_zero_load_control"]["pass"],
            },
            {"pass": result["criteria"]["C2_modal_orthogonality"]["pass"]},
            {
                "pass": result["criteria"]["C3_energy_conservation"]["pass"]
                and result["criteria"]["C3_energy_conservation"]["parseval_pass"]
            },
            {"pass": result["criteria"]["C4_envelope_consistency"]["pass"]},
            {"pass": idem["pass"]},
        ]),
    }

    with (out_dir / "report.json").open("w") as fh:
        json.dump(result, fh, indent=1)
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=1)
    with (out_dir / "study.json").open("w") as fh:
        json.dump(study, fh, indent=1)

    if args.with_pytest:
        import subprocess

        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_ice_vib_noise_chain.py", "-q"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        (out_dir / "pytest_summary.json").write_text(
            json.dumps(
                {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-2000:]},
                indent=1,
            )
        )

    print(json.dumps(metrics["headline"], indent=1))
    print("criteria:", {k: v["pass"] for k, v in metrics["criteria"].items()})
    print("out dir:", out_dir)
    return 0 if metrics["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())