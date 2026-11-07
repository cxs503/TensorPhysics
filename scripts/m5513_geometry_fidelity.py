"""m5513 campaign driver: contracts, negative corpus, figures, acceptance JSON.

Run with ``PYTHONPATH=src python3 scripts/m5513_geometry_fidelity.py``.  Every
number that ends up in the report and in ``acceptance.json`` is produced here --
nothing is transcribed by hand.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorphysics.geometry_fidelity import (  # noqa: E402
    GATE_PASS,
    GATE_REJECT,
    MeshNotAdmitted,
    assert_admitted,
    build_contract,
    canonical_json,
    contract_hulls,
    corrupted_record,
    evaluate_gate,
    good_records,
    suboff_area_envelope,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "geometry_fidelity" / "m5513"
FIG = OUT / "figures"
TOL = {
    "cad_volume_rel": 1e-3,
    "cad_surface_rel": 3e-3,
    "discrete_volume_rel": 5e-3,
    "discrete_surface_rel": 1e-2,
    "discrete_cb_abs": 1e-3,
}
REFERENCES = {
    "suboff": {"nabla_m3": 0.699211, "s_wetted_m2": 5.998},
    "kcs": {"cb": 0.651},
    "kvlcc2": {"cb": 0.8098},
}


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)


def build_all(hulls):
    contracts = {}
    idempotent = {}
    for name, hull in hulls.items():
        contracts[name] = build_contract(hull, good_records(hull))
        (OUT / f"contract_{name}.json").write_text(canonical_json(contracts[name]) + "\n")
        # idempotency check on the same input (kept outside the contract document,
        # which has additionalProperties = false)
        again = canonical_json(build_contract(hull, good_records(hull)))
        idempotent[name] = again == canonical_json(contracts[name])
    return contracts, idempotent


def solver_table(contracts):
    rows = []
    for name, contract in contracts.items():
        cad = contract["cad"]
        for family, record in contract["grid"]["solvers"].items():
            rows.append(
                {
                    "hull": name,
                    "family": family,
                    "nabla_cad_m3": cad["nabla_m3"],
                    "nabla_solver_m3": record["nabla_m3"],
                    "nabla_rel": (record["nabla_m3"] - cad["nabla_m3"]) / cad["nabla_m3"],
                    "s_cad_m2": cad["s_wetted_m2"],
                    "s_solver_m2": record["s_wetted_m2"],
                    "s_rel": (record["s_wetted_m2"] - cad["s_wetted_m2"]) / cad["s_wetted_m2"],
                    "cb_cad": cad["cb"],
                    "cb_solver": record["cb"],
                    "cb_abs": abs(record["cb"] - cad["cb"]),
                    "surface_evidence": record.get("surface_evidence", "REAL"),
                    "resolution": record["resolution"],
                    "grid_sha256": record["grid_sha256"],
                }
            )
    return rows


def cad_table(hulls, contracts):
    rows = []
    for name, hull in hulls.items():
        cad = contracts[name]["cad"]
        rows.append(
            {
                "hull": name,
                "L_m": hull.length_m,
                "B_m": hull.breadth_m,
                "T_m": hull.draught_m,
                "nabla_m3": cad["nabla_m3"],
                "nabla_slice_m3": cad["nabla_slice_m3"],
                "nabla_mc_m3": cad["nabla_monte_carlo_m3"],
                "nabla_rel_residual": abs(cad["nabla_slice_m3"] - cad["nabla_monte_carlo_m3"]) / cad["nabla_m3"],
                "s_wetted_m2": cad["s_wetted_m2"],
                "s_quadrature_m2": cad["s_quadrature_m2"],
                "s_mc_m2": cad["s_monte_carlo_m2"],
                "s_rel_residual": abs(cad["s_quadrature_m2"] - cad["s_monte_carlo_m2"]) / cad["s_wetted_m2"],
                "cb": cad["cb"],
                "cwp": cad["cwp"],
                "cad_sha256": contracts[name]["provenance"]["cad_sha256"],
            }
        )
    return rows


def negative_table(hulls, contracts):
    rows = []
    for name in ("kcs", "suboff"):
        hull = hulls[name]
        sound = good_records(hull)
        cad = contracts[name]["cad"]
        for kind in ("stretch_2pct", "drop_voxel_layer", "off_by_half_cell"):
            record = corrupted_record(hull, kind)
            polluted = dict(sound)
            polluted[record.family] = record
            contract = build_contract(hull, polluted)
            rows.append(
                {
                    "hull": name,
                    "corruption": kind,
                    "family": record.family,
                    "nabla_rel": (record.nabla_m3 - cad["nabla_m3"]) / cad["nabla_m3"],
                    "s_rel": (record.s_wetted_m2 - cad["s_wetted_m2"]) / cad["s_wetted_m2"],
                    "cb_abs": abs(record.cb - cad["cb"]),
                    "verdict": contract["gate"]["verdict"],
                    "failures": sorted({entry["id"] for entry in contract["gate"]["failures"]}),
                    "rejected": contract["gate"]["verdict"] == GATE_REJECT,
                }
            )
    return rows


def convergence_table(hull):
    rows = []
    for n_x, n_s in ((120, 80), (240, 160), (480, 320), (960, 640)):
        rows.append(
            {
                "n_x": n_x,
                "n_s": n_s,
                "nabla_m3": hull.cad_volume_slice(n_x, n_s),
                "s_wetted_m2": hull.cad_surface_area(n_x, n_s),
            }
        )
    return rows


def m1_figure(rows, path):
    hulls = sorted({row["hull"] for row in rows})
    families = ["fvm", "lbm", "bem"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, metric, limit, title in zip(
        axes,
        ("nabla_rel", "s_rel", "cb_abs"),
        (TOL["discrete_volume_rel"], TOL["discrete_surface_rel"], TOL["discrete_cb_abs"]),
        ("displacement volume", "wetted surface area", "block coefficient"),
    ):
        width = 0.25
        for index, family in enumerate(families):
            values = []
            for hull in hulls:
                match = [r for r in rows if r["hull"] == hull and r["family"] == family]
                value = abs(match[0][metric]) if match else 0.0
                if match and match[0]["surface_evidence"] == "WITHHELD" and metric == "s_rel":
                    value = np.nan
                values.append(value)
            positions = np.arange(len(hulls)) + (index - 1) * width
            ax.bar(positions, values, width, label=family)
        ax.axhline(limit, color="crimson", linestyle="--", linewidth=1.2, label="gate limit")
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(hulls)), [h.upper() for h in hulls])
        ax.set_title(f"{title}: |discrete - CAD|")
        ax.set_ylabel("relative (abs for Cb)")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("m5513 C2: discrete solver families vs CAD truth source (log scale, NaN = withheld area)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def m2_figure(cad_rows, conv, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    nx = [row["n_x"] for row in conv]
    vol = [row["nabla_m3"] for row in conv]
    area = [row["s_wetted_m2"] for row in conv]
    axes[0].plot(nx, vol, "o-", label="slice quadrature")
    axes[0].axhline(cad_rows["nabla_mc_m3"], color="k", linestyle="--", label="Monte-Carlo N=1e6")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("streamwise panels n_x")
    axes[0].set_ylabel(r"$\nabla$ [m$^3$]")
    axes[0].set_title("CAD volume: quadrature vs Monte-Carlo")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(nx, area, "s-", label="parametric quadrature")
    axes[1].axhline(cad_rows["s_mc_m2"], color="k", linestyle="--", label="Monte-Carlo N=1e6")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("streamwise panels n_x")
    axes[1].set_ylabel(r"$S$ [m$^2$]")
    axes[1].set_title("CAD wetted area: quadrature vs Monte-Carlo")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"m5513 C1: two independent CAD integrations converge on {cad_rows['hull'].upper()}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def m3_figure(neg, path):
    labels = [f"{row['hull']}\n{row['corruption']}" for row in neg]
    vol = [abs(row["nabla_rel"]) for row in neg]
    cb = [row["cb_abs"] for row in neg]
    s = [abs(row["s_rel"]) for row in neg]
    x = np.arange(len(neg))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - 0.2, vol, 0.4, label=r"$|\Delta \nabla|/\nabla$")
    ax.bar(x + 0.2, cb, 0.4, label=r"$|\Delta C_b|$")
    ax.plot(x, s, "k^", label=r"$|\Delta S|/S$")
    ax.axhline(TOL["discrete_volume_rel"], color="crimson", linestyle="--", linewidth=1.1, label="nabla limit 0.5 %")
    ax.axhline(TOL["discrete_cb_abs"], color="darkorange", linestyle=":", linewidth=1.1, label="Cb limit 1e-3")
    ax.set_yscale("log")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("deviation from CAD (log)")
    ax.set_title("m5513 C4: three damaged grids, all rejected by GATE_REJECT")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def m4_figure(envelope, path):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    exps = [row["fairing_exponent"] for row in envelope]
    areas = [row["s_wetted_m2"] for row in envelope]
    ax.plot(exps, areas, "o", label="displacement-closed fairings")
    ax.axhline(5.998, color="crimson", linestyle="--", label="delivered reference S = 5.998 m$^2$")
    ax.set_xlabel("fore/aft fairing exponent $b$ in $r=R(1-u^2)^b$")
    ax.set_ylabel(r"$S$ [m$^2$]")
    ax.set_title("m5513 C3: SUBOFF wetted area over the closed-shape family")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def m5_figure(hulls, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (name, hull) in zip(axes, hulls.items()):
        stations = np.linspace(0.05, 0.95, 7)
        for xi in stations:
            s = np.linspace(0.0, 1.0, 80)
            ys = []
            zs = []
            for sv in s:
                y, z = hull.surface_yz(float(xi) * hull.length_m, float(sv))
                ys.append(y)
                zs.append(z)
            ax.plot(np.array(ys) / hull.breadth_m, np.array(zs) / hull.draught_m, linewidth=1.0)
            ax.plot(-np.array(ys) / hull.breadth_m, np.array(zs) / hull.draught_m, linewidth=1.0)
        ax.set_title(f"{name.upper()}: L={hull.length_m:.3g} m, B={hull.breadth_m:.3g} m")
        ax.set_xlabel("y / B")
        ax.set_ylabel("z / T")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
    fig.suptitle("m5513 body plan of the CAD truth sources (sections at 7 stations)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    hulls = contract_hulls()
    contracts, idempotent = build_all(hulls)
    cad_rows = cad_table(hulls, contracts)
    solver_rows = solver_table(contracts)
    neg = negative_table(hulls, contracts)
    env = suboff_area_envelope()
    conv = convergence_table(hulls["kcs"])

    reference = {}
    for name, refs in REFERENCES.items():
        cad = contracts[name]["cad"]
        for key, target in refs.items():
            reference[f"{name}_{key}"] = {
                "reference": target,
                "computed": cad[key],
                "rel": (cad[key] - target) / target,
                "within_1e_3": abs(cad[key] - target) / target <= 1e-3,
            }

    admission = {}
    for name, contract in contracts.items():
        assert_admitted(contract)
        polluted = dict(good_records(hulls[name]))
        polluted["fvm"] = corrupted_record(hulls[name], "stretch_2pct")
        bad = build_contract(hulls[name], polluted)
        try:
            assert_admitted(bad)
            admission[name] = "ADMITTED (unexpected)"
        except MeshNotAdmitted as exc:
            admission[name] = f"REFUSED: {exc}"

    acceptance = {
        "mission_id": 5513,
        "module": "tensorphysics.geometry_fidelity",
        "c1": {
            "cad_volume_rel_max": max(row["nabla_rel_residual"] for row in cad_rows),
            "cad_surface_rel_max": max(row["s_rel_residual"] for row in cad_rows),
            "limits": [TOL["cad_volume_rel"], TOL["cad_surface_rel"]],
            "pass": all(
                row["nabla_rel_residual"] <= TOL["cad_volume_rel"] and row["s_rel_residual"] <= TOL["cad_surface_rel"]
                for row in cad_rows
            ),
        },
        "c2": {
            "nabla_rel_max": max(abs(row["nabla_rel"]) for row in solver_rows),
            "s_rel_max": max(
                abs(row["s_rel"]) for row in solver_rows if row["surface_evidence"] == "REAL"
            ),
            "cb_abs_max": max(row["cb_abs"] for row in solver_rows),
            "gate_verdicts": {name: contract["gate"]["verdict"] for name, contract in contracts.items()},
            "pass": all(contract["gate"]["verdict"] == GATE_PASS for contract in contracts.values()),
            "withheld": [
                {"hull": row["hull"], "family": row["family"], "s_rel": row["s_rel"]}
                for row in solver_rows
                if row["surface_evidence"] == "WITHHELD"
            ],
        },
        "c3": {
            "references": reference,
            "suboff_area_envelope_m2": [min(r["s_wetted_m2"] for r in env), max(r["s_wetted_m2"] for r in env)],
            "all_within_1e_3": all(item["within_1e_3"] for item in reference.values()),
        },
        "c4": {
            "negatives_rejected": sum(1 for row in neg if row["rejected"]),
            "negatives_total": len(neg),
            "false_positives": sum(1 for contract in contracts.values() if contract["gate"]["verdict"] != GATE_PASS),
            "pass": sum(1 for row in neg if row["rejected"]) == len(neg)
            and all(contract["gate"]["verdict"] == GATE_PASS for contract in contracts.values()),
        },
        "c5": {
            "schema": contracts["kcs"]["schema"],
            "digests": {name: contract["provenance"] for name, contract in contracts.items()},
            "pass": all(
                len(contract["provenance"]["cad_sha256"]) == 64 and len(contract["provenance"]["grid_sha256"]) == 64
                for contract in contracts.values()
            ),
        },
        "c6": {"idempotent": idempotent},
        "c7": {"admission": admission},
    }
    acceptance["c6"]["pass"] = all(acceptance["c6"]["idempotent"].values())
    acceptance["c7"]["pass"] = all(value.startswith("REFUSED") for value in admission.values())

    m1_figure(solver_rows, FIG / "fig1_discrete_vs_cad.png")
    m2_figure(cad_rows[0], conv, FIG / "fig2_cad_integrator_convergence.png")
    m3_figure(neg, FIG / "fig3_negative_controls.png")
    m4_figure(env, FIG / "fig4_suboff_area_envelope.png")
    m5_figure(hulls, FIG / "fig5_body_plan.png")

    (OUT / "cad_integrals.json").write_text(json.dumps(cad_rows, indent=2, sort_keys=True) + "\n")
    (OUT / "solver_deviations.json").write_text(json.dumps(solver_rows, indent=2, sort_keys=True) + "\n")
    (OUT / "negative_controls.json").write_text(json.dumps(neg, indent=2, sort_keys=True) + "\n")
    (OUT / "suboff_area_envelope.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    (OUT / "cad_convergence.json").write_text(json.dumps(conv, indent=2, sort_keys=True) + "\n")
    (OUT / "acceptance.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: acceptance[k].get("pass") for k in ("c1", "c2", "c4", "c5", "c6", "c7")}, indent=1))
    print("c3 all_within_1e_3:", acceptance["c3"]["all_within_1e_3"])
    print("c2 withheld:", acceptance["c2"]["withheld"])
    return 0


if __name__ == "__main__":
    sys.exit(main())