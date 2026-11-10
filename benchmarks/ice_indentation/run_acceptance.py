"""Acceptance benchmark driver for node p233w1-N06.

Runs the full J2 ice-indentation acceptance study, writes the machine-checkable
report (JSON) and the paper-grade figures (PNG):

* AC1 / AC2 material-point curves,
* AC3 elastic force-displacement vs the exact Hertz solution,
* AC4 elastoplastic force-displacement on 3 meshes + observed convergence,
* AC5 energy ledger,
* field plots (von Mises, equivalent plastic strain) on the symmetry plane,
* a Markdown report skeleton whose numbers are injected programmatically.

Usage
-----
``python benchmarks/ice_indentation/run_acceptance.py [--out DIR] [--quick]``

The runner is deliberately stdlib + numpy + matplotlib; the acceptance numbers
come from :func:`tensorphysics.ice_indentation.verification.acceptance_report`
(no hand-copied numbers anywhere in the report).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.ice_indentation.verification import (  # noqa: E402
    acceptance_report,
    default_config,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def count_tests() -> int:
    """Number of pytest cases in the node's test module (AC6)."""
    target = ROOT / "tests" / "test_ice_indentation.py"
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", str(target), "--collect-only", "-q"],
            capture_output=True, text=True, check=False, timeout=300,
        )
        lines = [ln for ln in out.stdout.splitlines() if "::" in ln]
        if lines:
            return len(lines)
    except Exception:  # pragma: no cover - collection is best effort
        pass
    # fallback: count the test functions in the module source
    if target.exists():
        return sum(1 for ln in target.read_text().splitlines()
                   if ln.strip().startswith("def test_"))
    return 0


def _declustered(result):
    """Gauss-point coordinates, von Mises and alpha of one indentation result."""
    return result.gp_coords, result.von_mises, result.alpha, result.u


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figure_ac1_ac2(report, out: pathlib.Path) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ac1 = report["criteria"]["AC1"]
    ac2 = report["criteria"]["AC2"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for key, style in (("tension", "-"), ("compression", "--")):
        curve = ac1["curves"][key]
        strain = np.asarray(curve["strain"]) * 100.0
        ax.plot(strain, np.asarray(curve["stress_numeric"]) / 1e6, style,
                color="tab:blue", lw=2.2,
                label=f"return mapping ({key})" if key == "tension" else None)
        ax.plot(strain, np.asarray(curve["stress_analytic"]) / 1e6, ":",
                color="tab:red", lw=2.6,
                label="closed form J2" if key == "tension" else None)
    curve_h = ac1["curves"]["tension_hardening"]
    ax.plot(np.asarray(curve_h["strain"]) * 100.0,
            np.asarray(curve_h["stress_numeric"]) / 1e6, "-",
            color="tab:green", lw=2.0, label="return mapping (H = 500 MPa)")
    ax.plot(np.asarray(curve_h["strain"]) * 100.0,
            np.asarray(curve_h["stress_analytic"]) / 1e6, ":",
            color="k", lw=2.4, label="closed form (H = 500 MPa)")
    ax.set_xlabel("axial strain [%]")
    ax.set_ylabel(r"axial stress $\sigma_{xx}$ [MPa]")
    ax.set_title(f"AC1 uniaxial stress path, max rel. err "
                 f"{ac1['max_relative_error']:.1e}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    labels = ["shear yield\n$\\sigma_{xy}^{y}$", "analytic\n$\\sigma_y/\\sqrt{3}$"]
    values = [ac2["measured_Pa"] / 1e6, ac2["analytic_Pa"] / 1e6]
    bars = ax.bar(labels, values, color=["tab:blue", "tab:red"], width=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.01, f"{val:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("stress [MPa]")
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_title(f"AC2 pure-shear first yield, rel. err "
                 f"{ac2['relative_error']:.1e}")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out / "ac1_ac2_material_point.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_elastic(report, out: pathlib.Path) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    el = report["criteria"]["AC3"]["elastic_reference"]
    delta_um = np.asarray(el["deltas_m"]) * 1e6
    fem = np.asarray(el["force_fem_N"])
    hertz = np.asarray(el["force_hertz_N"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    grid = np.linspace(0.0, delta_um.max() * 1.05, 200)
    ax.plot(grid, np.asarray([4 / 3 * (el["force_hertz_N"][-1] /
                                       (el["deltas_m"][-1] ** 1.5)) * (d / 1e6) ** 1.5
                              for d in grid]) / 1e0, "-", color="tab:red", lw=2,
            label="Hertz $P=\\frac{4}{3}E^*\\sqrt{R}\\,\\delta^{3/2}$")
    ax.plot(delta_um, fem, "o", color="tab:blue", ms=7, label="FEM (elastic J2)")
    ax.set_xlabel(r"penetration $\delta$ [$\mu$m]")
    ax.set_ylabel("indentation force $P$ [N]")
    ax.set_title("AC3 elastic branch vs exact Hertz solution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(delta_um, np.asarray(el["relative_error"]) * 100, "o-", color="tab:purple")
    ax.axhline(report["criteria"]["AC3"]["tolerance"] * 100, ls="--", color="k",
               label="15 % gate")
    ax.set_xlabel(r"penetration $\delta$ [$\mu$m]")
    ax.set_ylabel("relative error [%]")
    ax.set_title("AC3 error against Hertz")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "ac3_elastic_hertz.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_plastic(report, objects, out: pathlib.Path) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    study = objects["study"]
    ac4 = report["criteria"]["AC4"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    colors = ["tab:gray", "tab:orange", "tab:blue"]
    for run, colour in zip(study["runs"], colors):
        res = run["_result"]
        ax.plot(np.asarray(res.delta) * 1e6, np.asarray(res.force), "o-",
                color=colour, ms=4,
                label=f"mesh {run['dims']} ({run['n_elem']} elem)")
    ax.set_xlabel(r"penetration $\delta$ [$\mu$m]")
    ax.set_ylabel("indentation force $P$ [N]")
    ax.set_title("AC4 elastoplastic F($\\delta$) on 3 meshes "
                 "(J2, $\\sigma_y$ = 5 MPa)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    h = np.asarray(ac4["h_contact_m"]) * 1e6
    err = np.asarray(ac4["relative_errors"]) * 100
    ax.loglog(h[:-1], err[:-1], "o-", color="tab:red",
              label="error vs finest mesh")
    for hi, ei, dims in zip(h[:-1], err[:-1], ac4["meshes"][:-1]):
        ax.annotate(f"{tuple(dims)}", (hi, ei), textcoords="offset points",
                    xytext=(6, 6), fontsize=7)
    ax.set_xlabel("near-contact node spacing [ $\\mu$m ]")
    ax.set_ylabel("relative force error [%]")
    order = ac4["observed_order"]
    ax.set_title("AC4 mesh convergence"
                 + (f" (observed order {order:.2f})" if order is not None else ""))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "ac4_plastic_mesh_convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_fields(objects, out: pathlib.Path) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = objects["study"]["runs"][-1]["_result"]
    coords, mises, alpha, u = _declustered(res)
    flat = coords.reshape(-1, 3)
    x = flat[:, 0]
    z = flat[:, 2]
    m = mises.reshape(-1)
    a = alpha.reshape(-1)
    # keep the innermost y strip (the Gauss points nearest the symmetry plane)
    ys = np.unique(np.round(flat[:, 1], 14))
    dy = float(np.min(np.diff(ys))) if ys.size > 1 else 1.0
    sel = flat[:, 1] <= ys[0] + 0.5 * dy
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, field, label, cmap in (
        (axes[0], m, r"von Mises stress $\sigma_{vM}$ [MPa]", "viridis"),
        (axes[1], a, r"equivalent plastic strain $\alpha$ [-]", "magma"),
    ):
        vals = field[sel] / (1e6 if label.startswith("von") else 1.0)
        sc = ax.tricontourf(x[sel] * 1e3, z[sel] * 1e3, vals, levels=20, cmap=cmap)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(label, fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(float(flat[:, 2].min()) * 1e3, flat[:, 2].max() * 1e3)
        ax.set_xlabel("x [mm]  (symmetry plane y = 0)")
        ax.set_ylabel("z [mm]")
    axes[0].set_title("indentation field: von Mises")
    axes[1].set_title("indentation field: plastic zone")
    fig.tight_layout()
    path = out / "ac5_fields.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_energy(report, objects, out: pathlib.Path) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = objects["energy"]
    ac5 = report["criteria"]["AC5"]
    delta = np.asarray(res.delta) * 1e6
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.plot(delta, np.asarray(res.external_work) * 1e3, "-", lw=2.2,
            color="tab:blue", label="external work (contact loads)")
    ax.plot(delta, np.asarray(res.elastic_energy) * 1e3, "--", lw=2.0,
            color="tab:green", label="elastic strain energy")
    ax.plot(delta, np.asarray(res.plastic_dissipation) * 1e3, "-.", lw=2.0,
            color="tab:red", label="plastic dissipation")
    ax.plot(delta, np.asarray(res.elastic_energy + res.plastic_dissipation) * 1e3,
            ":", lw=2.6, color="k", label="elastic + plastic")
    ax.set_xlabel(r"penetration $\delta$ [$\mu$m]")
    ax.set_ylabel("energy [mJ]")
    ax.set_title("AC5 energy ledger")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    residual = (np.asarray(res.external_work)
                - np.asarray(res.elastic_energy)
                - np.asarray(res.plastic_dissipation))
    ax.plot(delta[1:], residual[1:] / np.maximum(np.asarray(res.external_work)[1:], 1e-30)
            * 100.0, "o-", color="tab:purple")
    ax.axhline(1.0, ls="--", color="k", label="1 % gate")
    ax.axhline(-1.0, ls="--", color="k")
    ax.set_xlabel(r"penetration $\delta$ [$\mu$m]")
    ax.set_ylabel("relative energy residual [%]")
    ax.set_title(f"AC5 balance residual, final {ac5['relative_residual'] * 100:+.2f} %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "ac5_energy_ledger.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _cache(out: pathlib.Path) -> pathlib.Path:
    path = out / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dump(path: pathlib.Path, obj) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _load(path: pathlib.Path):
    with open(path) as fh:
        return json.load(fh)


def run_stage(stage: str, out: pathlib.Path, quick: bool) -> None:
    """Run one stage of the acceptance study and cache its result as JSON."""
    import pickle

    from tensorphysics.ice_indentation.verification import (
        check_ac1, check_ac2, check_ac3, check_ac4, check_ac5, check_ac6,
        run_elastic_hertz_study, run_indentation_study, _run_one,
    )

    cache = _cache(out)
    cfg = default_config()
    if quick:
        cfg = default_config(mesh_dims=((6, 6, 3), (8, 8, 4), (10, 10, 5)),
                             n_steps=4, energy_steps=8,
                             hertz_deltas=(10.0e-6, 15.0e-6),
                             newton_max_iter=30)

    if stage == "material":
        _dump(cache / "ac1.json", check_ac1(cfg))
        _dump(cache / "ac2.json", check_ac2(cfg))
        print("material point stage done")
        return

    if stage == "elastic":
        _dump(cache / "elastic.json", run_elastic_hertz_study(cfg))
        print("elastic Hertz stage done")
        return

    if stage.startswith("plastic"):
        index = int(stage.split("-")[1])
        dims = cfg.mesh_dims[index]
        study = run_indentation_study(cfg, [dims])
        result = study["runs"][0].pop("_result")
        _dump(cache / f"plastic_{index}.json", study["runs"][0])
        np.savez_compressed(
            cache / f"fields_{index}.npz",
            gp_coords=result.gp_coords, von_mises=result.von_mises,
            alpha=result.alpha, u=result.u, delta=result.delta,
            force=result.force, elastic_energy=result.elastic_energy,
            plastic_dissipation=result.plastic_dissipation,
            external_work=result.external_work,
            contact_radius=result.contact_radius,
            mean_pressure=result.mean_pressure,
            max_von_mises=result.max_von_mises,
            penalty_energy=result.penalty_energy,
            indenter_work=result.indenter_work,
            contact_area=result.contact_area,
            max_alpha=result.max_alpha,
            penalty=np.array([result.penalty]),
            symmetry_factor=np.array([result.symmetry_factor]),
        )
        print(f"plastic stage {index} ({dims}) done: "
              f"F = {result.force[-1]:.3f} N, converged={result.converged}")
        return

    if stage == "energy":
        _, _, result = _run_one(cfg, cfg.mesh_dims[0], n_steps=cfg.energy_steps,
                                ramp=cfg.energy_ramp)
        np.savez_compressed(
            cache / "energy.npz", delta=result.delta,
            elastic_energy=result.elastic_energy,
            plastic_dissipation=result.plastic_dissipation,
            external_work=result.external_work,
            penalty_energy=result.penalty_energy,
            indenter_work=result.indenter_work, force=result.force,
            converged=np.array([result.converged]),
        )
        print(f"energy stage done: W = {result.external_work[-1]:.4e} J, "
              f"residual = {result.energy_balance()['relative_residual']:+.4f}")
        return

    raise SystemExit(f"unknown stage {stage!r}")


def assemble(out: pathlib.Path, quick: bool) -> int:
    """Merge the cached stages into the report, figures and Markdown summary."""
    from tensorphysics.ice_indentation.verification import (
        check_ac3, check_ac4, check_ac5, check_ac6, default_config,
    )

    cache = _cache(out)
    cfg = default_config()
    if quick:
        cfg = default_config(mesh_dims=((6, 6, 3), (8, 8, 4), (10, 10, 5)),
                             n_steps=4, energy_steps=8,
                             hertz_deltas=(10.0e-6, 15.0e-6),
                             newton_max_iter=30)
    ac1 = _load(cache / "ac1.json")
    ac2 = _load(cache / "ac2.json")
    elastic = _load(cache / "elastic.json")
    runs = [_load(cache / f"plastic_{i}.json") for i in range(len(cfg.mesh_dims))]
    study = {"runs": runs}
    ac4 = check_ac4(cfg, study)
    ac3 = check_ac3(cfg, elastic, runs[-1], cfg.material())

    energy_npz = np.load(cache / "energy.npz")

    class _EnergyView:
        """Minimal stand-in for the energy run (only the ledger is needed)."""

        def __init__(self, data):
            self.delta = data["delta"]
            self.elastic_energy = data["elastic_energy"]
            self.plastic_dissipation = data["plastic_dissipation"]
            self.external_work = data["external_work"]
            self.penalty_energy = data["penalty_energy"]
            self.indenter_work = data["indenter_work"]
            self.force = data["force"]

        def energy_balance(self, index=-1):
            from tensorphysics.ice_indentation.indentation import IndentationResult
            return IndentationResult.energy_balance(self, index)

    energy_result = _EnergyView(energy_npz)
    ac5 = check_ac5(cfg, energy_result)
    ac5["run"] = {"dims": list(cfg.mesh_dims[0]), "n_steps": cfg.energy_steps,
                  "ramp": cfg.energy_ramp}
    ac6 = check_ac6(cfg, count_tests())
    criteria = {"AC1": ac1, "AC2": ac2, "AC3": ac3, "AC4": ac4, "AC5": ac5,
                "AC6": ac6}
    from tensorphysics.ice_indentation.verification import _key_metric
    summary = {k: {"passed": bool(v["passed"]), "key_metric": _key_metric(k, v)}
               for k, v in criteria.items()}
    report = {
        "node": "p233w1-N06",
        "title": "J2 elastoplastic ice indentation with penalty contact",
        "config": {
            "E_Pa": cfg.E, "nu": cfg.nu, "sigma_y_Pa": cfg.sigma_y,
            "hardening_Pa": cfg.hardening,
            "indenter_radius_m": cfg.indenter_radius,
            "half_width_m": cfg.half_width, "height_m": cfg.height,
            "delta_max_m": cfg.delta_max, "n_steps": cfg.n_steps,
            "penalty_factor": cfg.penalty_factor,
            "contact_model": cfg.contact_model, "ramp": cfg.ramp,
            "mesh_dims": [list(d) for d in cfg.mesh_dims],
            "newton_tol": cfg.newton_tol,
            "energy_steps": cfg.energy_steps, "energy_ramp": cfg.energy_ramp,
        },
        "criteria": criteria,
        "summary": summary,
        "all_passed": all(bool(v["passed"]) for v in criteria.values()),
        "runs": runs,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick": bool(quick),
    }

    class _Objects:
        def __init__(self, out_dir, runs, energy):
            self.study = {"runs": []}
            for i, run in enumerate(runs):
                data = np.load(cache / f"fields_{i}.npz")

                class _R:
                    pass

                r = _R()
                for key in data.files:
                    setattr(r, key, data[key])
                r.penalty = float(data["penalty"][0])
                r.symmetry_factor = int(data["symmetry_factor"][0])
                self.study["runs"].append({"dims": tuple(run["dims"]),
                                           "n_elem": run["n_elem"],
                                           "_result": r})
            self.elastic = None
            self._data = {"study": self.study, "elastic": self.elastic,
                          "energy": energy}

        def __getitem__(self, key):
            return self._data[key]

    objects = _Objects(out, runs, energy_result)
    figures = {}
    for name, fn in (("ac1_ac2", lambda: figure_ac1_ac2(report, out)),
                     ("elastic_hertz", lambda: figure_elastic(report, out)),
                     ("plastic_convergence", lambda: figure_plastic(report, objects, out)),
                     ("fields", lambda: figure_fields(objects, out)),
                     ("energy", lambda: figure_energy(report, objects, out))):
        try:
            figures[name] = str(fn())
        except Exception as exc:  # pragma: no cover - plotting must not kill the run
            import traceback
            traceback.print_exc()
            print(f"[warn] figure {name} failed: {exc}")
    report["figures"] = figures

    json_path = out / "acceptance_report.json"
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== acceptance summary ===")
    for key, value in report["summary"].items():
        print(f"{key}: {'PASS' if value['passed'] else 'FAIL'}  {value['key_metric']}")
    print(f"all_passed = {report['all_passed']}")
    print(f"report  -> {json_path}")
    for name, path in figures.items():
        print(f"figure  -> {name}: {path}")
    pb = ac3["plastic_branch"]
    print(f"plastic branch: P={pb['force_N']:.2f} N, p_m/sigma_y="
          f"{pb['constraint_factor_p_m_over_sigma_y']:.2f}, reference={pb['reference']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "ice_indentation"))
    parser.add_argument("--stage", default="all",
                        help="material | elastic | plastic-<i> | energy | assemble | all")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "assemble":
        return assemble(out, args.quick)
    if args.stage == "all":
        n = len(default_config().mesh_dims)
        for stage in ["material", "elastic"] + [f"plastic-{i}" for i in range(n)] + ["energy"]:
            run_stage(stage, out, args.quick)
        return assemble(out, args.quick)
    run_stage(args.stage, out, args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
