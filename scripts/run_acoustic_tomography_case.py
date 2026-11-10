"""Run the under-ice acoustic tomography case and write all artifacts.

Usage
-----
    PYTHONPATH=src python scripts/run_acoustic_tomography_case.py [--fast]

``--fast`` runs a heavily reduced configuration (used by the smoke tests); the
default is the campaign case.  Outputs

* ``runs/m9890_acoustic_tomography/results.json``  -- every criterion number
  (machine-checkable, no hand-copied values),
* ``artifacts/m9890_acoustic_tomography/*.png``    -- report figures,
* ``artifacts/m9890_acoustic_tomography/ray_paths.npz`` -- ray geometry used
  by the inversion (provenance for the coverage figure).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tensorphysics.acoustic_tomography.case_under_ice import (  # noqa: E402
    CaseConfig,
    background_sound_speed,
    background_state,
    ice_station_ranges,
    ice_thickness_truth,
    noise_sensitivity,
    node_grid,
    run_case,
    summarise,
    truth_state,
    truth_sound_speed,
)
from tensorphysics.acoustic_tomography.sound_speed import (  # noqa: E402
    invert_temperature_from_sound_speed,
)

RUNS = _ROOT / "runs" / "m9890_acoustic_tomography"
FIGDIR = _ROOT / "artifacts" / "m9890_acoustic_tomography"


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def make_figures(result: dict, cfg: CaseConfig) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(parents=True, exist_ok=True)
    nodes = result["nodes"]
    fields = result["fields"]
    figures = []

    r_grid, z_grid = nodes.mesh()
    c_truth_node = truth_sound_speed(r_grid, z_grid, cfg)
    c_bg_node = background_sound_speed(z_grid)
    c_ret_node = fields["c_retrieved_grid"]

    # --- figure 1: anomaly truth vs retrieval, profiles and error ----------
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    ax = axes[0, 0]
    vmax = float(np.max(np.abs(c_truth_node - c_bg_node))) or 1.0
    cs = ax.contourf(r_grid / 1e3, z_grid, c_truth_node - c_bg_node, 21, cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax)
    ax.invert_yaxis()
    ax.set_xlabel("range r [km]")
    ax.set_ylabel("depth z [m]")
    ax.set_title("(a) truth sound-speed anomaly $c-c_{bg}$ [m/s]")
    fig.colorbar(cs, ax=ax)

    ax = axes[0, 1]
    cs = ax.contourf(r_grid / 1e3, z_grid, c_ret_node - c_bg_node, 21, cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax)
    ax.invert_yaxis()
    ax.set_xlabel("range r [km]")
    ax.set_ylabel("depth z [m]")
    ax.set_title("(b) SIRT retrieval (node grid, 1 ms timing noise)")
    fig.colorbar(cs, ax=ax)

    ax = axes[1, 0]
    for r_prof, style in ((500.0, "-"), (2200.0, "--"), (3500.0, "-.")):
        zz = np.linspace(0.0, cfg.z_max, 80)
        ax.plot(truth_sound_speed(np.full_like(zz, r_prof), zz, cfg), zz, style, color="0.2",
                label=f"truth r={r_prof/1e3:.1f} km")
        ax.plot(
            np.interp(zz, nodes.z_nodes, c_ret_node[np.argmin(np.abs(nodes.r_nodes - r_prof)), :]),
            zz,
            style,
            color="tab:red",
            alpha=0.8,
            label=f"retrieved r={r_prof/1e3:.1f} km",
        )
    ax.invert_yaxis()
    ax.set_xlabel("c [m/s]")
    ax.set_ylabel("depth z [m]")
    ax.set_title("(c) profiles: truth (grey) vs retrieved (red)")
    ax.legend(fontsize=6, loc="lower right")

    ax = axes[1, 1]
    ax.plot(fields["c_truth_samples"], fields["c_retrieved_samples"], "o", ms=4)
    lim = [
        float(np.min(fields["c_truth_samples"])) - 2,
        float(np.max(fields["c_truth_samples"])) + 2,
    ]
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.plot(lim, [lim[0] + 5, lim[-1] + 5], "k--", lw=0.5)
    ax.plot(lim, [lim[0] - 5, lim[-1] - 5], "k--", lw=0.5)
    ax.set_xlabel("c truth (CTD) [m/s]")
    ax.set_ylabel("c retrieved [m/s]")
    crit = result["criteria"]["a_sound_speed"]["value"]
    ax.set_title(f"(d) CTD comparison, rms={crit:.2f} m/s (dashed: $\\pm$5 m/s)")
    fig.suptitle("Under-ice acoustic tomography: sound-speed field reconstruction", fontsize=12)
    fig.tight_layout()
    path = FIGDIR / "fig1_sound_speed_reconstruction.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(
        {
            "name": path.name,
            "caption": (
                "Figure 1. (a) true sound-speed anomaly relative to the background "
                "climatology (warm Atlantic intrusion + melt-water lens); (b) SIRT "
                "retrieval from 1 ms travel times; (c) vertical profiles at three "
                "ranges; (d) retrieved vs true sound speed at the 35 CTD comparison "
                "points with the +/-5 m/s criterion band."
            ),
        }
    )

    # --- figure 2: ducted ray coverage -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    ax = axes[0]
    cs = ax.contourf(r_grid / 1e3, z_grid, c_truth_node, 21, cmap="viridis")
    for path in result["observations"]["paths"]:
        ax.plot(path[:, 0] / 1e3, path[:, 1], "-", lw=0.4, color="w", alpha=0.55)
    ax.invert_yaxis()
    ax.set_xlabel("range r [km]")
    ax.set_ylabel("depth z [m]")
    ax.set_title("(a) inversion ray paths over the true field")
    fig.colorbar(cs, ax=ax, label="c [m/s]")

    ax = axes[1]
    times = [np.asarray(t, dtype=float) for t in result["rays_truth"]["times"]]
    src_rcv = result["src_rcv"]
    for i, tt in enumerate(times):
        sep = abs(src_rcv[i, 2] - src_rcv[i, 0]) / 1e3
        ax.plot(np.full(tt.size, sep), tt, "o", ms=2.5, color="tab:blue", alpha=0.6)
    ax.set_xlabel("pair separation [km]")
    ax.set_ylabel("arrival time [s]")
    ax.set_title("(b) multipath arrivals: all ice-ducted eigenrays")
    fig.tight_layout()
    path = FIGDIR / "fig2_ray_coverage_multipath.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(
        {
            "name": path.name,
            "caption": (
                "Figure 2. (a) eigenray geometry used by the inversion: the sound-speed "
                "minimum at the ice traps the rays in the under-ice surface duct, so the "
                "array is connected by ice-underside reflections; (b) all traced arrivals "
                "versus pair separation (multi-bounce structure)."
            ),
        }
    )

    # --- figure 3: data fit + ice thickness --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    obs = result["observations"]
    ax.plot(obs["t_model_background"], obs["t_obs"], "o", ms=4, label="arrivals")
    lim = [float(np.min(obs["t_obs"])) - 0.05, float(np.max(obs["t_obs"])) + 0.05]
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.set_xlabel("traced arrival time in background model [s]")
    ax.set_ylabel("observed arrival time [s]")
    ax.set_title(f"(a) arrival identification: {obs['t_obs'].size} of "
                 f"{result['diagnostics']['n_arrivals_truth_model']} arrivals")
    ax.legend(fontsize=8)

    ax = axes[1]
    r_ice = fields["r_ice"]
    ax.plot(r_ice / 1e3, fields["h_truth"], "k-", lw=1.4, label="truth thickness")
    ax.plot(r_ice / 1e3, fields["h_retrieved"], "o", ms=4, color="tab:red",
            label="echo inversion")
    ax.fill_between(
        r_ice / 1e3,
        fields["h_retrieved"] - 2 * fields["sigma_h"],
        fields["h_retrieved"] + 2 * fields["sigma_h"],
        color="tab:red",
        alpha=0.25,
        label="+/-2 sigma",
    )
    ax.set_xlabel("range along track [km]")
    ax.set_ylabel("ice thickness h [m]")
    ax.set_title(f"(b) ice-thickness inversion, rms={result['criteria']['b_ice_thickness']['value']:.3f} m")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGDIR / "fig3_data_fit_and_ice_thickness.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(
        {
            "name": path.name,
            "caption": (
                "Figure 3. (a) identified arrivals: observed (fully traced truth field) "
                "against the background-model prediction; (b) ice thickness from the "
                "draft-mode echo inversion with the +/-2 sigma band from the propagated "
                "timing, sound-speed and density-ratio uncertainties."
            ),
        }
    )

    # --- figure 4: temperature inversion -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    ax.plot(fields["t_truth_samples"], fields["t_retrieved_samples"], "o", ms=4)
    lim = [
        float(np.nanmin(fields["t_truth_samples"])) - 0.3,
        float(np.nanmax(fields["t_truth_samples"])) + 0.3,
    ]
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.plot(lim, [lim[0] + 0.5, lim[-1] + 0.5], "k--", lw=0.5)
    ax.plot(lim, [lim[0] - 0.5, lim[-1] - 0.5], "k--", lw=0.5)
    ax.set_xlabel("T truth (CTD) [$^\\circ$C]")
    ax.set_ylabel("T retrieved from c [$^\\circ$C]")
    ax.set_title(f"(a) temperature inversion, rms={result['criteria']['c_temperature']['value']:.3f} $^\\circ$C")

    ax = axes[1]
    zz = np.linspace(0.0, cfg.z_max, 60)
    t_bg, _ = background_state(zz)
    t_tr, _ = truth_state(np.full_like(zz, 2200.0), zz, cfg)
    ax.plot(t_tr, zz, "k-", label="truth (r=2.2 km)")
    ax.plot(t_bg, zz, "0.6", ls="--", label="background prior")
    ax.plot(fields["t_retrieved_samples"].reshape(len(cfg.ctd_r), len(cfg.ctd_z))[3],
            np.asarray(cfg.ctd_z), "o-", color="tab:red", label="retrieved (r=2.0 km)")
    ax.set_xlabel("T [$^\\circ$C]")
    ax.set_ylabel("depth z [m]")
    ax.invert_yaxis()
    ax.set_title("(b) temperature profiles: truth / prior / retrieved")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGDIR / "fig4_temperature_inversion.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(
        {
            "name": path.name,
            "caption": (
                "Figure 4. (a) temperature recovered by inverting the retrieved sound "
                "speed through the campaign c(T,S,P) relation (salinity taken from the "
                "CTD climatology) against the truth; (b) profile comparison at the "
                "anomaly centre."
            ),
        }
    )

    # --- figure 5: uncertainty and sensitivity -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    sc = ax.scatter(
        fields["r_s"] / 1e3, fields["z_s"], c=fields["sigma_c_samples"], s=55,
        cmap="magma_r", edgecolor="k", linewidth=0.3,
    )
    ax.invert_yaxis()
    ax.set_xlabel("range r [km]")
    ax.set_ylabel("depth z [m]")
    ax.set_title("(a) ensemble uncertainty $\\sigma_c$ [m/s] (Monte-Carlo, 1 ms)")
    fig.colorbar(sc, ax=ax)

    ax = axes[1]
    scan = noise_sensitivity(result, cfg)
    noises = np.asarray(scan["sigma_time_ms"], dtype=float) * 1e-3
    rms_c = scan["rms_sound_speed_m_s"]
    rms_t = scan["rms_temperature_degC"]
    ax.loglog(noises * 1e3, rms_c, "o-", label="rms c error")
    ax.loglog(noises * 1e3, rms_t, "s-", label="rms T error [$^\\circ$C]")
    ax.axhline(5.0, color="tab:blue", ls=":", lw=0.8)
    ax.axhline(0.5, color="tab:orange", ls=":", lw=0.8)
    ax.axvline(1.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("timing noise $\\sigma_t$ [ms]")
    ax.set_ylabel("retrieval error")
    ax.set_title("(b) sensitivity to timing noise (dotted: criteria)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGDIR / "fig5_uncertainty_sensitivity.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(
        {
            "name": path.name,
            "caption": (
                "Figure 5. (a) ensemble (Monte-Carlo) standard deviation of the "
                "retrieved sound speed at the comparison points; (b) retrieval error "
                "versus timing noise, with the campaign criterion levels (5 m/s and "
                "0.5 degC) as dotted lines and the nominal 1 ms level marked."
            ),
            "noise_scan": scan,
        }
    )
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="reduced smoke configuration")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    if args.fast:
        cfg = CaseConfig(
            instrument_ranges=(0.0, 2000.0, 4000.0),
            instrument_depths=(40.0, 160.0),
            n_scan=61,
            theta_max_deg=12.0,
            n_refine=10,
            step_len=50.0,
            n_r_nodes=5,
            n_z_nodes=4,
            n_iter=200,
            n_mc=5,
            n_bending=0,
            n_reciprocity_pairs=2,
            ice_stations=20,
            ctd_r=(400.0, 1600.0, 2800.0, 3600.0),
            ctd_z=(30.0, 90.0, 150.0),
        )
    else:
        cfg = CaseConfig()

    result = run_case(cfg, verbose=True)
    summary = summarise(result)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "results.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )

    figures = []
    if not args.no_figures:
        figures = make_figures(result, cfg)
        for fig in figures:
            if "noise_scan" in fig:
                summary["noise_sensitivity"] = fig["noise_scan"]
        (RUNS / "results.json").write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
        )
        (RUNS / "figures.json").write_text(
            json.dumps(
                [{"name": f["name"], "caption": f["caption"]} for f in figures],
                indent=2,
            ),
            encoding="utf-8",
        )

    np.savez_compressed(
        RUNS / "case_data.npz",
        r_s=result["fields"]["r_s"],
        z_s=result["fields"]["z_s"],
        c_truth=result["fields"]["c_truth_samples"],
        c_retrieved=result["fields"]["c_retrieved_samples"],
        t_truth=result["fields"]["t_truth_samples"],
        t_retrieved=result["fields"]["t_retrieved_samples"],
        sigma_c=result["fields"]["sigma_c_samples"],
        r_ice=result["fields"]["r_ice"],
        h_truth=result["fields"]["h_truth"],
        h_retrieved=result["fields"]["h_retrieved"],
        sigma_h=result["fields"]["sigma_h"],
        t_obs=result["observations"]["t_obs"],
        t_model=result["observations"]["t_model_background"],
    )
    paths_npz = {f"path_{k}": p for k, p in enumerate(result["observations"]["paths"])}
    np.savez_compressed(FIGDIR / "ray_paths.npz", **paths_npz)

    print(json.dumps(summary["criteria"], indent=2, default=_json_default))
    print(f"results: {RUNS / 'results.json'}")
    print(f"figures: {FIGDIR} ({len(figures)} files)")
    return summary


if __name__ == "__main__":
    main()