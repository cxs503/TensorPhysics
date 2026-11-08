"""CLI: ``python -m tensorphysics.campaign.p35_gate``.

Typical gate run (live platform + bound repositories + regression probes)::

    python -m tensorphysics.campaign.p35_gate \
        --base-url "$TENSOR_API_BASE" --plan-id 179 \
        --repo-root TensorLBM=/path/to/TensorLBM \
        --repo-root TensorPhysics=. \
        --run-probes --probe-timeout 3600 \
        --out artifacts/p35_gate

Re-rendering a verdict from a recorded snapshot (offline audit, no network)::

    python -m tensorphysics.campaign.p35_gate --snapshot artifacts/p35_gate/snapshot.json \
        --records artifacts/p35_gate/regression_records.json --out artifacts/p35_gate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import collect as C
from . import gate as G
from . import regression as R
from . import spec as S

DEFAULT_OUT = "artifacts/p35_gate"
SNAPSHOT_NAME = "platform_snapshot.json"
RECORDS_NAME = "regression_records.json"


def login(base_url: str, username: str, password: str, *, timeout: float = 20.0) -> str:
    """Exchange agent credentials for a bearer token (``/auth/login``)."""
    url = base_url.rstrip("/") + "/auth/login"
    body = json.dumps({"username": username, "password": password}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    token = payload.get("access_token") or payload.get("token")
    if not token:
        raise RuntimeError("login succeeded but no access token in response")
    return token


def resolve_token(args) -> str:
    if args.token:
        return args.token
    env_token = os.environ.get("TENSOR_TOKEN")
    if env_token:
        return env_token
    user, password = os.environ.get("TENSOR_AGENT_USER"), os.environ.get("TENSOR_AGENT_PASS")
    if user and password:
        return login(args.base_url, user, password)
    raise SystemExit("no token: pass --token, set TENSOR_TOKEN, or TENSOR_AGENT_USER/_PASS")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tensorphysics.campaign.p35_gate",
                                 description="P35-G1 campaign integration acceptance gate")
    ap.add_argument("--base-url", default=os.environ.get("TENSOR_API_BASE", ""))
    ap.add_argument("--token", default=None)
    ap.add_argument("--plan-id", type=int, default=S.PLAN_ID)
    ap.add_argument("--snapshot", default=None, help="reuse a recorded platform snapshot")
    ap.add_argument("--repo-root", action="append", default=[], metavar="NAME=PATH",
                    help="repository-level binding: artifacts found there count as evidence, "
                         "absence there is inconclusive (WITHHELD)")
    ap.add_argument("--node-root", action="append", default=[], metavar="EXT_ID=PATH",
                    help="authoritative binding of one node's delivery worktree: a missing "
                         "declared artifact family there is a FAIL")
    ap.add_argument("--records", default=None, help="regression record file to read/append")
    ap.add_argument("--run-probes", action="store_true",
                    help="execute the declared regression probes before judging")
    ap.add_argument("--probe-timeout", type=float, default=3600.0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--require-pass", action="store_true",
                    help="exit non-zero when the gate does not reach PASS")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    inputs: dict = {}

    if args.snapshot:
        snapshot = C.load_snapshot(args.snapshot)
        inputs["snapshot"] = args.snapshot
    else:
        if not args.base_url:
            raise SystemExit("no --base-url (or TENSOR_API_BASE) and no --snapshot")
        client = C.PlatformClient(args.base_url, resolve_token(args))
        snapshot = C.build_snapshot(client, plan_id=args.plan_id)
        prov = C.write_snapshot(snapshot, out_dir / SNAPSHOT_NAME)
        inputs["snapshot"] = prov["path"]
        print(f"snapshot written: {prov['path']} sha256={prov['sha256'][:16]}…", flush=True)

    roots = C.parse_repo_roots(args.repo_root)
    node_roots = C.parse_repo_roots(args.node_root)
    scans = {}
    for node in S.NODES:
        node_root = node_roots.get(node.ext_id) or node_roots.get(node.ext_id.lower())
        root = node_root or roots.get(node.repo) or roots.get(node.repo.lower())
        if root:
            scans[node.ext_id] = C.scan_node_repository(root, node,
                                                        authoritative=bool(node_root))
    bound = {ext: ("node" if (node_roots.get(ext) or node_roots.get(ext.lower())) else "repo")
             for ext, scan in scans.items()}
    print(f"bound checkouts: {len(bound)}/10 "
          f"(authoritative node_root: {sum(1 for v in bound.values() if v == 'node')}, "
          f"repo_root: {sum(1 for v in bound.values() if v == 'repo')})", flush=True)

    records_path = args.records or str(out_dir / RECORDS_NAME)
    if args.run_probes:
        for case in S.REGRESSION_CASES:
            root = roots.get(case.repo) or roots.get(case.repo.lower())
            if not root:
                print(f"[skip] {case.case}: repo {case.repo} not bound", flush=True)
                continue
            result = R.run_probe(case, root, out_dir / "probes",
                                 timeout=args.probe_timeout)
            R.append_record(records_path, result)
            print(f"[probe] {case.case}: {result.status} {result.detail}", flush=True)
    records = R.load_records(records_path)
    if records:
        inputs["regression_records"] = records_path
    for ext, scan in scans.items():
        inputs[f"checkout:{ext}"] = scan.get("repo_root")

    readiness = G.run_gate(snapshot, scans=scans, regression_records=records,
                           plan_id=args.plan_id, inputs=inputs)
    written = G.write_outputs(readiness, out_dir)
    print(f"campaign_readiness: {written['readiness_json']['path']} "
          f"sha256={written['readiness_json']['sha256'][:16]}…", flush=True)
    print(f"overall_status={readiness['overall_status']} "
          f"dispatcher_ready={readiness['dispatcher_ready']} "
          f"blocking={len(readiness['blocking'])}", flush=True)
    for line in readiness["blocking"][:20]:
        print(f"  - {line}", flush=True)
    if args.require_pass and readiness["overall_status"] != "PASS":
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())