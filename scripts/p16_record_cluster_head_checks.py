#!/usr/bin/env python3
"""Record fresh-head (GitHub main) probe observations from the compute cluster.

The local P16 regression run executes the P15 probes against local checkouts.
Some checkouts are stale (their HEAD predates the P15 merge), in which case a
missing probe file only means "not visible here".  To separate a *stale mirror*
from a *genuinely dropped verification case* the same probe targets were resolved
on the 外协方 compute cluster, which checks out GitHub ``main`` fresh.

This script embeds those observed resolutions verbatim (job ids included) into
``probe_results.json`` under ``fresh_head_absence``.  Nothing is inferred: every
entry below is a literal ``MISS <target>`` line from the cluster job log.

Usage::

    python3 scripts/p16_record_cluster_head_checks.py \
        --probe-results artifacts/p16_gate/probe_results.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

#: Observed ``run_cluster`` resolutions at GitHub main.  ``MISS`` lines are
#: copied from the job logs; ``collection_errors`` are pytest error counts.
CLUSTER_OBSERVATIONS = [
    {
        "repo": "TensorLBM",
        "job_id": "c1794033269_256",
        "node": "n315",
        "head": "866438ac",
        "miss": ["tests/test_stokes_sphere_g12.py",
                 "tests/test_force_control_surface.py",
                 "tests/test_d3q27_enumeration.py",
                 "tests/test_lid_driven_cavity_3d.py"],
        "resolved": ["tests/test_public_api_lazy_exports.py", "tests/test_d3q27.py"],
        "collection_errors": 1,
        "log_excerpt": ("MISS tests/test_stokes_sphere_g12.py -> none; MISS "
                        "tests/test_force_control_surface.py -> none; MISS "
                        "tests/test_d3q27_enumeration.py -> none; MISS "
                        "tests/test_lid_driven_cavity_3d.py -> none; "
                        "ERROR tests/test_d3q27.py; 1 error during collection"),
        "ext_ids": ["IN14-D7", "IN14-D8", "IN14-D9"],
    },
    {
        "repo": "TensorDEM",
        "job_id": "c1794033291_199",
        "node": "n315",
        "head": "03e857d",
        "miss": ["tests/test_hertz_mindlin.py",
                 "tests/test_neighbor_list_vectorized.py",
                 "tests/test_clump_multisphere.py"],
        "resolved": ["tests/test_dem3d_hertz_contact.py", "tests/test_contact_tangential.py",
                     "tests/test_rudder_ice_load.py", "tests/test_dem.py", "tests/test_dem3d.py"],
        "collection_errors": 5,
        "log_excerpt": ("MISS tests/test_hertz_mindlin.py -> none; MISS "
                        "tests/test_neighbor_list_vectorized.py -> none; MISS "
                        "tests/test_clump_multisphere.py -> none; "
                        "ERROR tests/test_contact_tangential.py (x4); 5 errors during collection "
                        "(node environment lacks the repo's runtime deps)"),
        "ext_ids": ["IN14-D1", "IN14-D2", "IN14-D3"],
    },
    {
        "repo": "TensorFVM",
        "job_id": "c1794033623_83",
        "node": "n330",
        "head": None,
        "miss": [],
        "resolved": [],
        "collection_errors": None,
        "log_excerpt": "repo fetch failed on the node: github.com:443 unreachable after 135 s",
        "ext_ids": ["IN14-D10"],
    },
    {
        "repo": "TensorBEM",
        "job_id": "c1794033760_144",
        "node": "n330",
        "head": None,
        "miss": [],
        "resolved": [],
        "collection_errors": None,
        "log_excerpt": "repo fetch failed on the node: github.com:443 unreachable after 134 s",
        "ext_ids": ["IN14-D11"],
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-results", default="artifacts/p16_gate/probe_results.json")
    args = ap.parse_args()
    path = pathlib.Path(args.probe_results)
    data = json.loads(path.read_text())

    absent: dict[str, dict] = {}
    for obs in CLUSTER_OBSERVATIONS:
        for ext in obs["ext_ids"]:
            entry = absent.setdefault(ext, {"repo": obs["repo"], "head": obs["head"],
                                            "targets_absent_at_fresh_head": []})
            entry["targets_absent_at_fresh_head"] = obs["miss"]
            entry["job_id"] = obs["job_id"]

    data["cluster_head_checks"] = {"observations": CLUSTER_OBSERVATIONS,
                                   "source": "run_cluster(run, repo=<repo>, cmd=pytest resolve loop)"}
    data["fresh_head_absence"] = absent
    data["note"] = (
        "Local probe re-execution on /root/tensor-ecosystem checkouts plus fresh-head "
        "(GitHub main) resolution on the 702 compute cluster. 'unresolved_detail' classifies "
        "each missing P15 probe file; 'fresh_head_absence' lists targets confirmed absent at "
        "a fresh head, which is a dropped verification case rather than a stale checkout.")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(json.dumps({"fresh_head_absence": absent, "repos": [o["repo"] for o in CLUSTER_OBSERVATIONS]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())