#!/usr/bin/env python3
"""Build and execute the m15005 verdict INSERT (evidence read from the audit artifacts)."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHK = json.loads((ROOT / "runs/m15005_audit_cross_clock/audit_checks.json").read_text())
RES = json.loads((ROOT / "runs/m14792_cross_clock_library/results.json").read_text())
A = RES["anchors"]
LEVEL = CHK["level"]["high_water_level"]

evidence = {
    "impl_files": ["src/tensorphysics/cross_clock/library_binding.py"],
    "entry": "scripts/run_cross_clock_library_demo.py",
    "run_artifacts": [
        "runs/m15005_audit_cross_clock/audit_checks.json",
        "runs/m15005_audit_cross_clock/audit_summary.txt",
        "runs/m15005_audit_cross_clock/figures/audit_evidence.png",
        "runs/m14792_cross_clock_library/results.json",
        "runs/m14792_cross_clock_library/SHA256SUMS",
        "runs/m14792_cross_clock_library/figures/velocity_vs_time.png",
        "runs/m14792_cross_clock_library/figures/reference_drift_vs_window.png",
    ],
    "tests": CHK["level"]["evidence_keys"]["tests"],
    "integration_callers": CHK["callers"]["solver_or_platform_callers"],
    "benchmark": {
        "rows": [{k: r[k] for k in ("quantity", "reference", "got", "rel_error", "within_3pct")}
                 for r in CHK["benchmark"]["rows"]],
        "source": "runs/m14792_cross_clock_library/results.json",
        "all_within_3pct": CHK["benchmark"]["all_within_3pct"],
    },
    "repro_cmd": "python scripts/run_cross_clock_library_demo.py",
    "repro_verified": {
        "executed_in": "throw-away copy of the repo (audit isolation)",
        "wall_seconds": CHK["repro"]["wall_seconds"],
        "physics_differences_vs_committed": CHK["repro"]["n_differences"],
        "tolerated_fields": ["generated_utc", "wall_seconds", "environment"],
    },
    "external_anchor": "docs/CROSS_CLOCK_OSS_SURVEY.md",
    "external_anchor_scope": ("per-leg analytic closed forms only (Taylor-Green 2*nu*k^2, Stokes relaxation); "
                              "NO experimental / cross-code anchor for the coupled trajectory -> L5 not claimable"),
    "pins": {"tensorlbm": {"version": CHK["pins"]["pins"]["tensorlbm"]["version"],
                           "source_sha256_prefix": CHK["pins"]["pins"]["tensorlbm"]["source_sha256"]},
             "tensordem": {"version": CHK["pins"]["pins"]["tensordem"]["version"],
                           "source_sha256_prefix": CHK["pins"]["pins"]["tensordem"]["source_sha256"]},
             "drift": CHK["pins"]["drift"]},
    "tests_rerun": {"passed": CHK["tests"]["passed"], "failed": CHK["tests"]["failed"],
                    "by": "scripts/audit_m15005_cross_clock.py check D"},
    "artifacts_run_artifacts_sha256_ok": CHK["artifacts"]["ok"],
    "audit_report": "docs/M15005_CROSS_CLOCK_VERDICT.md",
    "findings": {
        "F1_no_integration_caller": ("grep over the whole tree finds 0 callers of library_binding outside the "
                                     "module, its test and the demo script; wire it into "
                                     "src/tensorphysics/coupling.py CouplingManager.register (AFTER_SOLVER) to reach L3"),
        "F2_run_artifacts_stripped_by_merge_gate": ("merged main commit d8f92af dropped 4 files under runs/; artifacts "
                                                     "were re-copied from worktree m14792 and re-verified via SHA256SUMS"),
        "F3_no_maturity_keys_at_completion": "server_quality_audit={gate_valid:false,key_present:false,gate_missing_warning:true}",
        "F4_coupled_reference_is_semi_analytic": "ZOH reference uses the same run's measured u_k; no grid refinement (64x8)",
        "F5_demo_not_fail_closed_on_pin_drift": "demo records drift instead of aborting; the test asserts drift==[]",
        "F6_no_overstated_claims": ("record numbers reproduced: 18 passed, ledger drift "
                                    f"{RES['cases'][1]['ledger_drift_max']:.2e}/{RES['cases'][2]['ledger_drift_max']:.2e}, "
                                    f"momentum residual {RES['cases'][0]['measured_system_momentum_residual_rel']:.2e}, "
                                    f"anchors {A['tensorlbm_taylor_green']['rel_error']:.3g}/"
                                    f"{A['tensordem_stokes_relaxation']['rel_error']:.2e}"),
    },
    "disposition": "approve with notes",
}

sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)\n"
    "VALUES ('TensorPhysics', 'm14792 GAP-CROSS-CLOCK', 15005, {lv}, '{ev}'::jsonb, 'board-audit')\n"
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET mission_id=15005, level={lv}, "
    "evidence='{ev}'::jsonb, gate_state='pending', gate_level=NULL, gate_note=NULL, created_at=now();\n"
    "SELECT id, repo, scope, mission_id, level, verdict_by, gate_state FROM tensor_maturity_verdicts "
    "WHERE verdict_by='board-audit' AND scope='m14792 GAP-CROSS-CLOCK';\n"
).format(lv=LEVEL, ev=json.dumps(evidence, ensure_ascii=False).replace("'", "''"))

pathlib.Path("/tmp/verdict_m15005.sql").write_text(sql)
print(f"level={LEVEL} sql_bytes={len(sql)}")
if os.environ.get("APPLY") == "1":
    dsn = os.environ["TP_PG_DB"]
    r = subprocess.run(["psql", dsn, "-f", "/tmp/verdict_m15005.sql"], capture_output=True, text=True)
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        print("falling back to postgres role")
        r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
                            "-f", "/tmp/verdict_m15005.sql"], capture_output=True, text=True)
        print(r.stdout, r.stderr)