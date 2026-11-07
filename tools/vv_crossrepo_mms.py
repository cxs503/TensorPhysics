"""Cross-repository V&V run #2: TensorFVM manufactured solution -> convergence gate.

This is the manufactured-solution (MMS) leg of criterion (c): take a case that
*ships with TensorFVM* -- ``tensorfvm.verification.native_order3d`` -- and push
its three-grid refinement study of the native viscous operator through the
:mod:`tensorphysics.vv` machine gate.

The case (unchanged, nothing inside TensorFVM is modified or monkey-patched):

``manufactured_solution_study(n)``
    solves ``div(tau(u)) = source`` on a ``n x n x n`` hexahedral box (``h = 1/n``)
    with the analytic divergence of the stress of the manufactured field
    ``u* = [sin(pi x) cos(pi y/2) e^z, cos(pi x/2) sin(pi y) e^-z, sin(pi x) sin(pi y) z]``
    imposed as the cell source, so the *exact discrete solution* is ``u*`` and the
    residual against it is a genuine discretisation error.  Three error norms are
    returned (``l2_interior`` excludes one layer of wall-adjacent cells, whose
    Dirichlet closure is a separate approximation; ``l2_global`` includes them;
    ``linf_global`` is the max-norm).

The design order of a linear-reconstruction viscous operator is 2, so the gate is
asked with ``expected_order = 2.0``; the observed order is whatever the code
produces.

Zero intrusion: the module is imported read-only (``TENSORFVM_SRC``) and the
digests of the files that produced the numbers are recorded in the raw artifact.
Heavy levels (``n`` >= 24) were executed on the cluster; use ``--results`` to
ingest those records instead of recomputing, and ``--run`` to regenerate locally.

Usage::

    python tools/vv_crossrepo_mms.py --levels 4 8 16 --emit-dir artifacts/vv_gate
    python tools/vv_crossrepo_mms.py --results artifacts/vv_gate/tensorfvm_mms_cluster_runs.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.vv import GatePolicy, Observation, evaluate  # noqa: E402

DEFAULT_SRC = os.environ.get("TENSORFVM_SRC", "/tmp/tfvm/src")
OBSERVABLES = ("l2_interior", "l2_global", "linf_global")
EXPECTED_ORDER = 2.0
DIGEST_FILES = (
    "tensorfvm/verification/native_order3d.py",
    "tensorfvm/native_viscous3d.py",
    "tensorfvm/native_mesh3d.py",
)


def _digest(src: Path, rel: str) -> str | None:
    path = src / rel
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(src: Path) -> dict:
    version = None
    for candidate in (src.parent / "pyproject.toml", src.parent.parent / "pyproject.toml"):
        if candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                if line.startswith("version"):
                    version = line.split("=", 1)[1].strip().strip('"')
                    break
            break
    return {
        "repo": "TensorFVM",
        "package_version": version,
        "source_root": str(src),
        "module_sha256": {rel: _digest(src, rel) for rel in DIGEST_FILES},
        "intrusion": "read-only import; no file of TensorFVM is modified",
    }


def run_levels(src: Path, levels: list[int]) -> list[dict]:
    sys.path.insert(0, str(src))
    from tensorfvm.verification.native_order3d import manufactured_solution_study  # noqa: PLC0415

    rows = []
    for n in levels:
        t0 = time.time()
        r = manufactured_solution_study(n)
        rows.append({
            "n": int(r["n"]),
            "h": float(r["h"]),
            "warp": float(r["warp"]),
            "l2_interior": float(r["l2_interior"]),
            "l2_global": float(r["l2_global"]),
            "linf_global": float(r["linf_global"]),
            "field_scale": float(r["field_scale"]),
            "residual": float(r["residual"]),
            "elapsed_s": round(time.time() - t0, 1),
            "executed_on": "local",
        })
        print(f"n={n:3d} h={r['h']:.5g} l2_int={r['l2_interior']:.6e} "
              f"l2_glob={r['l2_global']:.6e} t={rows[-1]['elapsed_s']}s", flush=True)
    return rows


def _rows_from_results(path: Path) -> tuple[list[dict], dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    meta = ({k: v for k, v in payload.items()
             if k not in ("rows", "supplementary_rows")} if isinstance(payload, dict) else {})
    extra = list(payload.get("supplementary_rows", [])) if isinstance(payload, dict) else []
    out = []
    for row in rows:
        rec = {k: row[k] for k in ("n", "h", "l2_interior", "l2_global", "linf_global")}
        for key in ("warp", "field_scale", "residual", "elapsed_s"):
            if key in row:
                rec[key] = row[key]
        rec["executed_on"] = row.get("executed_on", "cluster")
        out.append(rec)
    out.sort(key=lambda r: -r["h"])  # coarse -> fine
    extra_records = []
    for row in extra:
        rec = {k: row[k] for k in ("n", "h", "l2_interior", "l2_global", "linf_global")}
        rec["executed_on"] = row.get("executed_on", "cluster")
        extra_records.append(rec)
    return out, meta, extra_records


def build_raw(rows: list[dict], prov: dict, meta: dict, note: str | None = None) -> dict:
    return {
        "schema": "tensorphysics.vv.crossrepo/1",
        "repo": "TensorFVM",
        "case": "verification/native_order3d.manufactured_solution_study",
        "operator": "tensorfvm.native_viscous3d.NativeViscous3D (linear reconstruction)",
        "manufactured_field": "u* = [sin(pi x)cos(pi y/2)e^z, cos(pi x/2)sin(pi y)e^-z, sin(pi x)sin(pi y)z]",
        "source_term": "analytic divergence of the stress of u* imposed as cell source",
        "expected_order": EXPECTED_ORDER,
        "boundary_condition": "Dirichlet u* on all six box faces, hex_box(n, warp=0)",
        "levels": rows,
        "provenance": prov,
        "execution": meta,
        "study_note": note,
    }


def emit(emit_dir: Path, raw: dict, *, prefix: str = "tensorfvm_mms") -> dict:
    emit_dir.mkdir(parents=True, exist_ok=True)
    raw_path = emit_dir / f"{prefix}_raw.json"
    raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    gates: dict[str, dict] = {}
    for observable in OBSERVABLES:
        payload = {
            "schema": "tensorphysics.vv.observations/1",
            "quantity": observable,
            "expected_order": EXPECTED_ORDER,
            "source": {"repo": "TensorFVM",
                       "path": f"artifacts/vv_gate/{prefix}_raw.json",
                       "study": "native_order3d/manufactured",
                       "metadata": {"module_sha256": raw["provenance"]["module_sha256"]}},
            "levels": [{"h": row["h"], "f": row[observable], "label": f"n={row['n']}"}
                       for row in raw["levels"]],
        }
        obs_path = emit_dir / f"obs_{prefix}_{observable}.json"
        obs_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        verdict = evaluate(
            [Observation(h=row["h"], f=row[observable], label=f"n={row['n']}")
             for row in raw["levels"]],
            quantity=observable,
            policy=GatePolicy(expected_order=EXPECTED_ORDER),
        )
        record = verdict.as_dict()
        record["adapter"] = {"repo": "TensorFVM", "dialect": "observations/1",
                             "source": str(obs_path.relative_to(ROOT)),
                             "raw": str(raw_path.relative_to(ROOT))}
        record["expected_order"] = EXPECTED_ORDER
        gate_path = emit_dir / f"gate_{prefix}_{observable}.json"
        gate_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        gates[observable] = record
        study = record["study"]
        order = study["order_p"]
        print(f"{observable:<13} p={order:.4f} |p-2|={abs(order - EXPECTED_ORDER):.4f} "
              f"GCI={study['gci_fine']:.4e} -> {record['status']}", flush=True)
    return gates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--src", default=DEFAULT_SRC, help="TensorFVM src directory")
    parser.add_argument("--results", default=None,
                        help="ingest a JSON record of precomputed levels (cluster run)")
    parser.add_argument("--emit-dir", default=str(ROOT / "artifacts" / "vv_gate"))
    args = parser.parse_args(argv)

    src = Path(args.src)
    prov = provenance(src)
    extra: list[dict] = []
    if args.results:
        rows, meta, extra = _rows_from_results(Path(args.results))
    else:
        if not src.is_dir():
            print(f"error: TensorFVM source not found at {src}", file=sys.stderr)
            return 64
        rows, meta = run_levels(src, args.levels), {"executed_on": "local"}
    emit_dir = Path(args.emit_dir)
    emit(emit_dir, build_raw(rows, prov, meta))
    if extra:
        merged = sorted(rows + extra, key=lambda r: -r["h"])
        emit(emit_dir,
             build_raw(merged, prov, meta,
                       note="supplementary non-uniform study: the r=2 triple is "
                            "extended by one level, so the general (least-squares) "
                            "path of the gate is exercised on real data"),
             prefix="tensorfvm_mms_4level")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())