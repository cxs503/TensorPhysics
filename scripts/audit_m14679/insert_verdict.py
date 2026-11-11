#!/usr/bin/env python3
"""Insert the m15004 board-audit verdict for m14679 (GAP-CROSS-MAP) into
tensor_maturity_verdicts.

Evidence keys follow the maturity_gate JUDGE_SCHEME contract
(impl_files / entry / run_artifacts / tests / integration_callers /
benchmark / repro_cmd / external_anchor); every value is a real pointer inside
the audit work tree (/root/tp-agent/work/m15004, which holds the TensorPhysics
checkout) or the delivery work tree /root/tp-agent/work/m14679.

Level = 2 (independent functional implementation).  L3 (system integration) is
NOT claimed: no production caller exists (grep-verified), so the gate ladder
cannot reach L4 even though benchmark-grade material is present.

Usage: python scripts/audit_m14679/insert_verdict.py
"""
from __future__ import annotations

import json
import os
import re
import sys

ENV = "/root/tensor-ecosystem/TensorPlatform/.env"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CHECKS = os.path.join(ROOT, "runs", "m15004_audit_m14679", "checks.json")


def dsn() -> dict:
    url = None
    for line in open(ENV):
        if line.startswith("DATABASE_URL="):
            url = line.split("=", 1)[1].strip()
    assert url, "DATABASE_URL missing"
    m = re.match(r"postgresql\+\w+://([^:]+):([^@]+)@([^/]+)/(\S+)", url)
    user, pw, host, db = m.groups()
    db = db.split("?")[0]
    host, _, port = host.partition(":")
    return dict(host=host or "127.0.0.1", port=int(port or 5432), user=user,
                password=pw, dbname=db)


def build_evidence() -> dict:
    c = json.load(open(CHECKS))
    gf = c["C_convergence"]["global_fit"]
    shipped = c["C_convergence"]["shipped_claim"]
    return {
        "impl_files": [
            "src/tensorphysics/grid_to_grid_remap/overlap.py",     # clip 132, table 251
            "src/tensorphysics/grid_to_grid_remap/remap.py",       # avg 63, BJ 193, repair 213, bounded 255
            "src/tensorphysics/grid_to_grid_remap/acceptance.py",  # cases 151/241/286, run 353
            "src/tensorphysics/grid_to_grid_remap/__init__.py",
        ],
        "entry": "src/tensorphysics/grid_to_grid_remap/__init__.py",
        "run_artifacts": [
            "runs/m14679/benchmark.json",
            "runs/m14679/conservation.json",
            "runs/m14679/convergence.json",
            "runs/m14679/symmetry.json",
            "artifacts/m14679/conservation_bar.png",
            "artifacts/m14679/convergence_plot.png",
            "artifacts/m14679/symmetry_field.png",
            "runs/m15004_audit_m14679/checks.json",
            "artifacts/m15004/m14679_convergence_audit.png",
        ],
        "tests": [
            "tests/test_grid_to_grid_remap.py::test_conservation_smooth_field",
            "tests/test_grid_to_grid_remap.py::test_conservation_hot_spot",
            "tests/test_grid_to_grid_remap.py::test_conservation_thermal_front",
            "tests/test_grid_to_grid_remap.py::test_bounded_mode_conservation",
            "tests/test_grid_to_grid_remap.py::test_no_new_extrema_bounded",
            "tests/test_grid_to_grid_remap.py::test_symmetry_roundtrip_integral",
            "tests/test_grid_to_grid_remap.py::test_convergence_order",
            "tests/test_grid_to_grid_remap.py::test_vector_remap_conservation",
            "tests/test_grid_to_grid_remap.py::test_acceptance_all_pass",
            "audit re-run: pytest tests/test_grid_to_grid_remap.py -q -> 14 passed rc=0",
        ],
        "integration_callers": "",
        "integration_callers_note": (
            "none: repo-wide grep for 'grid_to_grid_remap' returns only "
            "tests/test_grid_to_grid_remap.py and generate_m14679_artifacts.py "
            "(no solver / coupling / platform caller) -> L3 not claimable"
        ),
        "benchmark": {
            "source": "runs/m14679/benchmark.json (regenerated bit-identical)",
            "conservation_rel_defect": {"smooth_sin": [4.11e-16, 0.0],
                                        "hot_spot": [7.71e-15, 1.57e-16],
                                        "thermal_front": [1.00e-15, 7.78e-16]},
            "convergence_ref_vs_got": {
                "shipped_global_fit": {"avg": shipped["avg_order"],
                                       "bnd": shipped["bnd_order"]},
                "audit_refit_4levels_8_32": gf["four_levels_8_32_pointref"],
                "audit_local_orders_bnd": [round(x, 2) for x in
                                           c["C_convergence"]["local_orders"]["bnd"]],
                "reference_pollution_l2": c["C_convergence"][
                    "reference_point_eval_l2_vs_exact_cell_average"],
            },
            "symmetry_roundtrip_l2_rel": 0.026408351336941575,
            "independent_checks": {
                "clip_max_rel_vs_raster_800": c["A_clip"]["max_rel_discrepancy_vs_raster"],
                "partition_row_col_rel_err": [c["B_partition"]["max_rel_err_row_sum_vs_target_area"],
                                              c["B_partition"]["max_rel_err_col_sum_vs_source_area"]],
            },
        },
        "repro_cmd": ("cd /root/tp-agent/work/m15004 && python generate_m14679_artifacts.py && "
                      "python -m pytest tests/test_grid_to_grid_remap.py -q && "
                      "python scripts/audit_m14679/audit_m14679.py"),
        "external_anchor": "src/tensorphysics/grid_to_grid_remap/acceptance.py",
        "external_anchor_note": (
            "analytic only: acceptance.py:167 uses the closed-form integral "
            "int_0^1 int_0^1 sin(pi x) sin(pi y) dx dy = 4/pi^2; no literature or "
            "third-party-solver comparison (supermesh CSR etc.) is provided"
        ),
        "audit_doc": "docs/AUDIT_m14679_GAP_CROSS_MAP.md",
        "audited_identity": {"merge_commit": "3568988", "content_commit": "8b6807a",
                             "pr": 391, "repo": "TensorPhysics"},
        "severity_findings": {
            "B1_convergence_order_not_asymptotic": {
                "shipped": shipped["bnd_order"],
                "four_level_fit": gf["four_levels_8_32_pointref"]["bnd"],
                "local_orders": [round(x, 2) for x in c["C_convergence"]["local_orders"]["bnd"]],
            },
            "B2_reference_is_centroid_point_eval_not_cell_average": {
                "doc_claim": "acceptance.py:296 'exact cell average of the analytic field'",
                "code": "acceptance.py:303 exact_fn(tgt_centroids)",
                "pollution_l2": c["C_convergence"]["reference_point_eval_l2_vs_exact_cell_average"],
                "ratio_to_finest_bounded_error": c["C_convergence"][
                    "pollution_over_finest_bounded_error_at_48"],
            },
            "M1_artifacts_not_in_merged_commit": (
                "git ls-files runs/m14679 artifacts/m14679 -> empty; commit 8b6807a "
                "'7 files filtered by merge gate'; reproducible bit-identically"),
            "M2_no_integration": "see integration_callers_note",
            "N1_no_literature_anchor": "analytic integral only",
            "N2_criterion_scope": (
                "'mapping error <1%' holds for the conservation functional (<=1.2e-14); "
                "field accuracy is not <1%: average-mode L2 rel 1.17e-2 at n=48, "
                "A->B->A roundtrip L2 rel 2.64e-2"),
        },
        "level_rationale": (
            "L2 confirmed: real clip/limit/repair code with entry + reproducible run "
            "artifacts + invariant-asserting tests. L3 denied: zero production callers. "
            "L4 material half-present (analytic benchmark + repro_cmd) but gate ladder "
            "needs L3, and the order/reference claims are overstated (B1/B2)."),
        "checks_not_run": [
            "comparison against literature/third-party remappers (supermesh/CSR/ESMF)",
            "3D, curved/spherical, non-convex or holed domains",
            "performance/memory scaling on 1e6-element meshes",
            "p1_recon (unlimited) overshoot quantification",
            "end-to-end conservation through a real TensorLBM/TensorFEM coupling (no caller)",
        ],
        "disposition": "approve with notes / L2 confirmed",
    }


def main() -> int:
    import psycopg
    ev = build_evidence()
    with psycopg.connect(**dsn(), autocommit=True) as conn:
        cur = conn.execute(
            "INSERT INTO tensor_maturity_verdicts"
            " (repo, scope, mission_id, level, evidence, verdict_by)"
            " VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (repo, scope, verdict_by) DO UPDATE"
            " SET level = EXCLUDED.level, evidence = EXCLUDED.evidence,"
            "     mission_id = EXCLUDED.mission_id"
            " RETURNING id, repo, scope, mission_id, level, verdict_by, gate_state",
            ("TensorPhysics", "m14679 GAP-CROSS-MAP", 15004, 2,
             json.dumps(ev, ensure_ascii=False), "board-audit"))
        print(json.dumps(cur.fetchone(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())