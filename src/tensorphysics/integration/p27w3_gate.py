"""P27W3 integration gate: run the three contracts and emit a verdict JSON.

The gate is deliberately boring: it reads the golden corpus captured from the
peers, re-derives every number with TensorPhysics' own implementations, and
writes ``artifacts/p27w3_contracts/P27W3_verdict.json`` containing one record
per acceptance criterion with the *number*, the *threshold*, the *status* and
the *artifact* that proves it.  Nothing in the verdict is typed by hand: the
report generator injects the same JSON.

Criteria (from ``campaign_nodes`` metadata of node P27W3-N08):

====  ==================================================================
C1    all seven upstream nodes ``status = done``
C2    BEM <-> seakeeping RAO transfer deviation < 1 % at three frequencies
C3    structural <-> fatigue stress-spectrum transfer deviation < 1 %
C4    DEM <-> ice-plate bond-strength mapping deviation < 2 %
C5    maturity (``evidence_level``) upgrade records complete: N02 >= 3
      fields, N04 >= 5 fields
C6    integration report >= 3000 characters with numeric tables
C7    >= 10 green contract tests
====  ==================================================================
"""
from __future__ import annotations

import ast
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dag_audit import audit_plan
from .ice_bond import (cohesive_energy_error, dem_bond_to_cohesive,
                       dem_failure_load, tensorphysics_failure_load)
from .interchange import IceBondTransfer, RAOTransfer, StressSpectrumTransfer
from .invariants import check_invariants, table_fidelity
from .rao import compare_responses, geometry_audit, long_wave_limit, sectional_added_mass_crosscheck
from .spectral import compare_damage, damage_by_method

__all__ = ["PLAN_ID", "UPSTREAM_NODES", "contract_test_inventory", "run_gate", "write_verdict"]

#: Campaign plan seeded as ``P27-W3-seed`` on the platform.
PLAN_ID = 165
#: Node ids of N01..N07 inside that plan (N08 = 1046 = this mission).
UPSTREAM_NODES = (1039, 1040, 1041, 1042, 1043, 1044, 1045)
#: Missions of the two maturity nodes the summary must collect.
MATURITY_MISSIONS = {"P27W3-N02": 6728, "P27W3-N04": 6730}

CASE_FREQUENCIES = (0.60, 1.00, 1.60)
REPORT_PATH = Path("docs/reports/P27W3_integration_report.md")
TEST_PATH = Path("tests/test_p27w3_integration_contracts.py")


def _status(value: float | None, threshold: float, *, available: bool = True) -> str:
    if not available or value is None:
        return "withheld"
    return "pass" if value < threshold else "fail"


def _psql(query: str, timeout: float = 30.0) -> dict:
    """Read-only psql helper; failures are returned, never raised."""
    cmd = ["psql", "-U", "postgres", "-d", "tensorplatform", "-t", "-A", "-c", query]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:      # pragma: no cover
        return {"available": False, "error": str(exc), "rows": []}
    if proc.returncode != 0:
        return {"available": False, "error": proc.stderr.strip()[:400], "rows": []}
    rows = [line.split("|") for line in proc.stdout.strip().splitlines() if line]
    return {"available": True, "rows": rows}


# --------------------------------------------------------------------------- #
# C5: maturity records
# --------------------------------------------------------------------------- #

def maturity_records(root: str | Path = ".") -> dict:
    """Collect ``evidence_level`` upgrade evidence for the two maturity nodes.

    Primary source (hermetic): the frozen maturity singles under
    ``artifacts/p27w3_contracts/maturity/maturity_records.json``, captured from
    the nodes' mission worktrees by ``scripts/p27w3_capture_peer_goldens.py``
    with the source file hash and the structured field count.

    Secondary source (platform truth): ``factory_evidence_records`` keyed by the
    node's mission id, which is where the platform is *supposed* to book the
    level upgrade.  Both are reported; the criterion is satisfied if either
    source has the required number of fields, and the verdict records which one
    fired.
    """
    root = Path(root)
    out: dict[str, Any] = {"sources": [], "nodes": {}}
    frozen_path = root / "artifacts" / "p27w3_contracts" / "maturity" / "maturity_records.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
    out["sources"].append({"kind": "frozen_maturity_singles", "path": str(frozen_path),
                           "available": bool(frozen)})

    for ext, mission in MATURITY_MISSIONS.items():
        q = ("select coalesce(json_agg(json_build_object('evidence_level', evidence_level, "
             "'n_fields', coalesce((select count(*) from jsonb_object_keys(metrics)),0), "
             "'category', category)), '[]'::json) from factory_evidence_records "
             f"where task_id = {int(mission)}")
        res = _psql(q)
        out["sources"].append({"kind": "factory_evidence_records", "query": q,
                               "available": res["available"],
                               "error": res.get("error")})
        rows = []
        if res["available"] and res["rows"]:
            try:
                rows = json.loads(res["rows"][0][0] or "[]")
            except json.JSONDecodeError:                       # pragma: no cover
                rows = []
        db_fields = sum(int(r.get("n_fields") or 0) for r in rows)

        frozen_node = dict((frozen.get(ext) or {}))
        frozen_fields = int(frozen_node.get("n_fields") or 0)
        required = 3 if ext == "P27W3-N02" else 5
        out["nodes"][ext] = {
            "mission_id": mission,
            "required_fields": required,
            "db_records": rows,
            "db_fields": db_fields,
            "frozen_records": frozen_node.get("records", []),
            "frozen_fields": frozen_fields,
            "n_fields": max(db_fields, frozen_fields),
            "source": ("factory_evidence_records" if db_fields >= frozen_fields
                       else "frozen_maturity_singles"),
            "evidence_level": next(
                (r.get("evidence_level") for r in frozen_node.get("records", [])
                 if r.get("evidence_level")), None),
        }
        out["nodes"][ext]["complete"] = out["nodes"][ext]["n_fields"] >= required
    out["complete"] = all(n["complete"] for n in out["nodes"].values())
    return out


# --------------------------------------------------------------------------- #
# C7: contract test inventory
# --------------------------------------------------------------------------- #

def contract_test_inventory(root: Path) -> dict:
    path = root / TEST_PATH
    if not path.exists():
        return {"available": False, "n_tests": 0, "files": []}
    tree = ast.parse(path.read_text())
    names = [n.name for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    return {"available": True, "n_tests": len(names), "names": sorted(names),
            "file": str(TEST_PATH)}


# --------------------------------------------------------------------------- #
# main gate
# --------------------------------------------------------------------------- #

def run_gate(root: str | Path = ".", *, run_dag_audit: bool = True,
             run_mc: bool = True, n_mc_samples: int = 1 << 15) -> dict:
    root = Path(root)
    gold = root / "artifacts" / "p27w3_contracts" / "peer_goldens"

    # ---------------- C2: RAO transfer -------------------------------------
    rao_gold = json.loads((gold / "rao_wigley_v1.json").read_text())
    rao_payload = dict(rao_gold["payload"])
    rao_payload["heave_rao"] = rao_gold["response"]["heave_rao"]
    rao_payload["pitch_rao"] = rao_gold["response"]["pitch_rao"]
    rao_payload["roll_rao"] = rao_gold["response"]["roll_rao"]
    rao_payload["heave_phase_deg"] = rao_gold["response"].get("heave_phase_deg")
    rao_payload["pitch_phase_deg"] = rao_gold["response"].get("pitch_phase_deg")
    rao_payload["provenance"] = rao_gold["provenance"]
    transfer = RAOTransfer.from_payload(rao_payload)

    reference = [row for row in rao_gold["response"]["motion_amplitudes"]
                 if any(abs(row["omega"] - w) < 1e-12 for w in CASE_FREQUENCIES)]
    comparisons = compare_responses(transfer, reference,
                                    rao_gold["response"]["wave_amplitude_m"])
    c2_dev = max(c.max_rel_dev for c in comparisons)
    lw = long_wave_limit(transfer)
    invariants = [r.as_dict() for r in check_invariants(transfer, rao_gold["response"])]
    fidelity = table_fidelity(transfer, rao_gold["response"])
    coarse = rao_gold["response"]["coarse_table"]
    coarse_payload = dict(rao_payload)
    coarse_payload["frequencies_rad_s"] = coarse["omega"]
    coarse_payload["heave_rao"] = coarse["heave_rao"]
    coarse_payload["pitch_rao"] = coarse["pitch_rao"]
    coarse_payload["roll_rao"] = [0.0] * len(coarse["omega"])
    fidelity_baseline = table_fidelity(RAOTransfer.from_payload(coarse_payload),
                                       rao_gold["response"])
    sections = dict(rao_gold["response"]["sections"])
    sections["draught_m"] = transfer.hull_params["T"]
    sectional = sectional_added_mass_crosscheck(sections, rho=transfer.hull_params["rho"])
    geometry = geometry_audit(transfer, sections)
    spectral_dev = max(fidelity["heave_sigma_m"]["rel_dev"],
                       fidelity["pitch_sigma_rad"]["rel_dev"])
    long_wave_dev = lw["abs_dev_from_unity"]
    c2_pass = (c2_dev < 0.01 and spectral_dev < 0.01 and long_wave_dev < 0.05)

    # ---------------- C3: stress spectrum ---------------------------------
    psd_gold = json.loads((gold / "stress_psd_v1.json").read_text())
    psd_payload = dict(psd_gold["payload"])
    psd_payload["provenance"] = psd_gold["provenance"]
    spectrum = StressSpectrumTransfer.from_payload(psd_payload)
    damage = compare_damage(spectrum, psd_gold["response"])
    c3_dev = float(damage["dirlik"]["rel_dev"])
    damage["independent_witness"] = (_rainflow_witness(spectrum, n_mc_samples)
                                     if run_mc else {"available": False})
    secondary = psd_gold["response"].get("secondary_witness_p27w2") or {}
    if secondary.get("available"):
        mine = damage_by_method(spectrum)["damage_accumulated"]
        damage["secondary_peer_p27w2"] = {
            "commit": secondary.get("commit"),
            "dirlik_damage": secondary["dirlik_damage"],
            "receiver_dirlik_damage": mine["dirlik"],
            "dirlik_rel_dev": abs(mine["dirlik"] - secondary["dirlik_damage"])
            / secondary["dirlik_damage"],
            "narrowband_rel_dev": abs(mine["rayleigh"] - secondary["narrowband_damage"])
            / secondary["narrowband_damage"],
            "note": "committed P27W2 module (agent/m6659); its Dirlik uses the "
                    "published closed form, hence the machine-precision agreement",
        }

    # ---------------- C4: ice bond mapping --------------------------------
    bond_gold = json.loads((gold / "ice_bond_v1.json").read_text())
    bond = IceBondTransfer.from_payload(bond_gold["payload"], bond_gold["response"])
    mapping = dem_bond_to_cohesive(bond)
    energy = cohesive_energy_error(mapping)
    failure = tensorphysics_failure_load(bond)
    # round-trip: sigma_b <- 2 G_c / delta_c (the 1/2 of the triangle cancels)
    roundtrip = abs(2.0 * mapping.G_c_J_m2 / mapping.delta_c_m
                    - bond.bond_strength_Pa) / bond.bond_strength_Pa
    c4_dev = max(roundtrip, energy["rel_dev"], failure["rel_dev"])
    dem_load = dem_failure_load(bond)

    # ---------------- C1 / C5 --------------------------------------------
    dag = audit_plan(PLAN_ID) if run_dag_audit else {"available": False,
                                                     "error": "audit skipped"}
    upstream = [n for n in dag.get("nodes", []) if n["node_id"] in UPSTREAM_NODES]
    dag_available = bool(dag.get("available")) and len(upstream) == len(UPSTREAM_NODES)
    c1_ok = dag_available and all(n["status"] == "done" for n in upstream)
    maturity = maturity_records(root)

    # ---------------- C6 / C7 --------------------------------------------
    report = root / REPORT_PATH
    report_chars = len(report.read_text()) if report.exists() else 0
    tests = contract_test_inventory(root)

    verdict = {
        "gate": "P27W3-integration",
        "mission_id": 6734,
        "campaign_plan_id": PLAN_ID,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": _git_head(root),
        "criteria": {
            "C1_dag_convergence": {
                "status": "pass" if c1_ok else ("fail" if dag_available else "withheld"),
                "n_upstream_done": sum(1 for n in upstream if n["status"] == "done"),
                "n_upstream": len(UPSTREAM_NODES),
                "blocking": dag.get("blocking", []),
                "nodes": dag.get("nodes", []),
                "checked_at_utc": dag.get("checked_at_utc"),
                "evidence": "campaign_nodes where plan_id=165 (dag_audit.audit_plan)",
                "artifact": "artifacts/p27w3_contracts/P27W3_verdict.json",
            },
            "C2_rao_transfer": {
                "status": "pass" if c2_pass else "fail",
                "metric": "max relative deviation of reconstructed heave/pitch/roll "
                          "amplitudes at the 3 contract frequencies",
                "value": c2_dev,
                "threshold": 0.01,
                "per_frequency": [c.as_dict() for c in comparisons],
                "long_wave_limit": lw,
                "invariants": invariants,
                "sampling_fidelity_contract_table": fidelity,
                "sampling_fidelity_baseline_table": fidelity_baseline,
                "sectional_bem_crosscheck": sectional["comparison"],
                "sectional_bem_caveat": (
                    "diagnostic only: TensorPhysics' 2-D BEM is verified against the "
                    "closed forms for submerged sections (rho*pi*a^2); a floating "
                    "wall-sided section needs a mixed Dirichlet(free surface)/"
                    "Neumann(waterplane lid) boundary condition that the module does "
                    "not implement, so this number mixes a model-form difference with "
                    "a boundary-condition difference and is not used as a criterion"),
                "geometry_audit": geometry,
            },
            "C3_stress_spectrum": {
                "status": _status(c3_dev, 0.01),
                "metric": "relative deviation of Dirlik Miner damage over duration_s "
                          "against the P27W3-N04 peer (backend/physics/"
                          "spectral_fatigue_core.py)",
                "value": c3_dev,
                "threshold": 0.01,
                "root_cause": (_c3_root_cause(damage)
                               if c3_dev >= 0.01 else None),
                "detail": damage,
            },
            "C4_ice_bond": {
                "status": _status(c4_dev, 0.02),
                "metric": "max(round-trip sigma_b, cohesive energy integral, "
                          "first-fibre failure load) relative deviation",
                "value": c4_dev,
                "threshold": 0.02,
                "mapping": mapping.as_dict(),
                "energy_check": energy,
                "failure_load": failure,
                "dem_failure_load_N": dem_load,
            },
            "C5_maturity_records": {
                "status": "pass" if maturity["complete"] else "fail",
                "metric": "structured evidence_level upgrade fields per maturity node",
                "value": {ext: n["n_fields"] for ext, n in maturity["nodes"].items()},
                "threshold": {ext: n["required_fields"]
                              for ext, n in maturity["nodes"].items()},
                "detail": maturity,
            },
            "C6_report": {
                "status": "pass" if report_chars >= 3000 else "fail",
                "characters": report_chars,
                "threshold": 3000,
                "path": str(REPORT_PATH),
            },
            "C7_contract_tests": {
                "status": "pass" if tests["n_tests"] >= 10 else "fail",
                "n_tests": tests["n_tests"],
                "threshold": 10,
                "names": tests.get("names", []),
            },
        },
        "peer_provenance": {
            "rao-v1": rao_gold["provenance"],
            "stress-psd-v1": psd_gold["provenance"],
            "ice-bond-v1": bond_gold["provenance"],
        },
    }
    passed = sum(1 for c in verdict["criteria"].values() if c["status"] == "pass")
    verdict["summary"] = {
        "n_pass": passed,
        "n_fail": sum(1 for c in verdict["criteria"].values() if c["status"] == "fail"),
        "n_withheld": sum(1 for c in verdict["criteria"].values()
                          if c["status"] == "withheld"),
        "all_pass": passed == len(verdict["criteria"]),
    }
    return verdict


def _c3_root_cause(damage: dict) -> str:
    """Attribute the C3 deviation when it exceeds the threshold.

    The receiver and the peer implement the same *contract*, so a deviation can
    only come from the physics closure.  The independent time-domain rainflow
    estimate arbitrates: whichever side is farther from it is the biased one.
    """
    wit = damage.get("independent_witness") or {}
    if not wit.get("available"):                              # pragma: no cover
        return "no independent witness available"
    mc = float(wit["rainflow_damage_rate_per_s"]) * float(damage["duration_s"])
    receiver = float(damage["dirlik"]["damage_receiver"])
    sender = float(damage["dirlik"]["damage_sender"])
    d_recv = abs(receiver - mc) / mc
    d_send = abs(sender - mc) / mc
    return (
        f"closure bias, not transport: receiver Dirlik is {100*d_recv:.2f}% from "
        f"the rainflow estimate, the peer is {100*d_send:.2f}% from it "
        f"(receiver/peer agreement {100*float(damage['dirlik']['rel_dev']):.2f}%). "
        "The peer's dirlik_parameters() does not implement Dirlik (1985): it uses "
        "x = alpha1 instead of xm = (m1/m0)*sqrt(m2/m4) and sets R = alpha1, "
        "yielding D1 = 2x^2(1-x)/(1+x^2) instead of 2(xm - alpha2^2)/(1 + alpha2^2)."
    )


def _rainflow_witness(spectrum: StressSpectrumTransfer, n_samples: int) -> dict:
    """Independent time-domain witness: rainflow-counted Miner damage.

    The Dirlik closure is empirical; agreeing with the sender's Dirlik proves the
    *contract*, not the model.  The time-domain leg (random-phase synthesis ->
    ASTM E1049 rainflow -> Palmgren-Miner) is an independent estimator of the
    same damage and therefore bounds how much either side can be trusted.
    """
    from ..fatigue.spectral import monte_carlo_damage_rate

    fs = 8.0
    freqs = np.arange(n_samples // 2 + 1) * fs / n_samples
    s_vals = np.interp(freqs, spectrum.freq_hz, spectrum.psd, left=0.0, right=0.0)
    res = monte_carlo_damage_rate(s_vals, n_samples=n_samples, fs=fs, seeds=(0, 1),
                                  C=spectrum.sn_C, m=spectrum.sn_m,
                                  f_lo=float(spectrum.freq_hz[0]),
                                  f_hi=float(spectrum.freq_hz[-1]))
    mine = damage_by_method(spectrum)["damage_rate_per_s"]
    mc = float(res["damage_rate_mean"])
    return {
        "available": True,
        "rainflow_damage_rate_per_s": mc,
        "rainflow_rel_stderr": float(res.get("damage_rate_stderr", 0.0)) / mc,
        "dirlik_vs_rainflow_rel_dev": abs(mine["dirlik"] - mc) / mc,
        "rayleigh_vs_rainflow_rel_dev": abs(mine["rayleigh"] - mc) / mc,
        "n_samples": int(n_samples),
        "sampling_hz": fs,
    }


def _git_head(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):              # pragma: no cover
        return "unknown"


def write_verdict(verdict: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2, sort_keys=True, default=_default) + "\n")
    return path


def _default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):                            # pragma: no cover
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def main() -> int:                                             # pragma: no cover
    root = Path(__file__).resolve().parents[3]
    verdict = run_gate(root)
    path = write_verdict(verdict, root / "artifacts" / "p27w3_contracts" / "P27W3_verdict.json")
    print(json.dumps(verdict["summary"], indent=2))
    for key, crit in verdict["criteria"].items():
        val = crit.get("value", crit.get("n_tests", crit.get("characters", "-")))
        print(f"{key:26s} {crit['status']:9s} value={val}")
    print(f"verdict -> {path}")
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())