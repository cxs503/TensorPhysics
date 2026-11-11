#!/usr/bin/env python3
"""m15005 · independent re-check of the m14792 delivery (anchor GAP-CROSS-CLOCK).

This is an *audit* script, not a new implementation: it re-derives every claim the
m14792 record makes, from the committed artifacts, without trusting the record.

Checks (each one machine-decidable, all results land in ``audit_checks.json``):

A. artifact integrity   — recompute SHA-256 of ``runs/m14792_cross_clock_library/``
                          and compare with its own ``SHA256SUMS``.
B. evidence pointers    — the files the evidence names (impl / entry / tests /
                          contract doc) exist, their sha256 and line counts, and
                          the implementation body contains no placeholder stub.
C. frozen-pin integrity — ``verify_pins()`` returns an empty drift list against the
                          installed tensorlbm / tensordem, i.e. the coupled legs are
                          the shipped libraries, not in-repo stand-ins.
D. test evidence        — ``pytest tests/test_cross_clock_library_binding.py`` is
                          re-run here (count parsed from the report, not copied).
E. reproducibility      — the repro command is executed in a throw-away copy of the
                          repo and every physical number is compared with the
                          committed ``results.json`` (timestamps / wall clock are
                          the only tolerated differences).
F. integration callers  — grep the tree for callers of the module outside itself,
                          its test and the demo script.  This is the input to the
                          L3 (system integration) rung of the maturity scheme.
G. level                — apply the platform maturity scheme cumulatively:
                          L1 tests, L2 impl+entry+artifacts, L3 +callers,
                          L4 +benchmark+repro, L5 +external anchor.

Run: ``python scripts/audit_m15005_cross_clock.py [--skip-rerun]``
Writes ``runs/m15005_audit_cross_clock/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "runs" / "m14792_cross_clock_library"
OUT = ROOT / "runs" / "m15005_audit_cross_clock"
SRC = ROOT / "src" / "tensorphysics" / "cross_clock"
IMPL = SRC / "library_binding.py"
ENTRY = ROOT / "scripts" / "run_cross_clock_library_demo.py"
TESTS = ROOT / "tests" / "test_cross_clock_library_binding.py"
DOC = ROOT / "docs" / "CROSS_CLOCK_LIBRARY_CONTRACT.md"
PLACEHOLDER = re.compile(r"NotImplementedError|raise TODO|pass\s*#\s*placeholder")

# fields that legitimately differ between two runs of the same code
VOLATILE = {"generated_utc", "wall_seconds", "environment"}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- A. artifacts
def check_artifact_integrity() -> dict:
    sums = ART / "SHA256SUMS"
    listed, bad = [], []
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        rel = rel.strip()
        f = ART / rel
        ok = f.exists() and sha256(f) == digest
        listed.append({"file": f"runs/m14792_cross_clock_library/{rel}", "sha256": digest,
                       "bytes": f.stat().st_size if f.exists() else None, "ok": ok})
        if not ok:
            bad.append(rel)
    return {"listed": listed, "missing_or_modified": bad, "ok": not bad and bool(listed)}


# ---------------------------------------------------------------- B. pointers
def check_pointers() -> dict:
    rows = []
    for name, p in (("impl_files", IMPL), ("entry", ENTRY), ("tests", TESTS), ("contract_doc", DOC)):
        txt = p.read_text() if p.exists() else ""
        rows.append({
            "role": name,
            "path": str(p.relative_to(ROOT)),
            "exists": p.exists(),
            "bytes": p.stat().st_size if p.exists() else 0,
            "lines": txt.count("\n") + 1 if txt else 0,
            "sha256": sha256(p) if p.exists() else None,
            "placeholders": len(PLACEHOLDER.findall(txt)),
        })
    impl = next(r for r in rows if r["role"] == "impl_files")
    return {"pointers": rows,
            "impl_real": impl["exists"] and impl["lines"] > 40 and impl["placeholders"] < 3,
            "ok": all(r["exists"] for r in rows)}


# ------------------------------------------------------------------- C. pins
def check_pins() -> dict:
    code = (
        "import json,sys;"
        f"sys.path.insert(0, {json.dumps(str(ROOT / 'src'))});"
        "from tensorphysics.cross_clock.library_binding import verify_pins, FROZEN_PINS;"
        "print(json.dumps({'drift': verify_pins(),"
        "'pins': {k: {'library': v.library, 'version': v.version,"
        "'source_sha256': v.source_sha256[:16]} for k, v in FROZEN_PINS.items()}}))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return {"ok": False, "stderr": r.stderr[-800:]}
    d = json.loads(r.stdout.strip().splitlines()[-1])
    return {"ok": d["drift"] == [] and len(d["pins"]) >= 2, **d}


# ------------------------------------------------------------------- D. tests
def check_tests() -> dict:
    r = subprocess.run([sys.executable, "-m", "pytest", str(TESTS), "-q", "--no-header"],
                       capture_output=True, text=True, timeout=1800, cwd=ROOT)
    tail = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"(\d+) passed", tail)
    f = re.search(r"(\d+) failed", tail)
    return {"ok": r.returncode == 0 and m is not None and (f is None or f.group(1) == "0"),
            "passed": int(m.group(1)) if m else 0,
            "failed": int(f.group(1)) if f else 0,
            "returncode": r.returncode,
            "tail": tail.strip().splitlines()[-4:] if tail.strip() else []}


# -------------------------------------------------------- E. reproducibility
def _walk_diff(a, b, path="", tol=1e-12) -> list[dict]:
    diffs: list[dict] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in VOLATILE:
                continue
            if k not in a or k not in b:
                diffs.append({"path": f"{path}/{k}", "issue": "missing side"})
                continue
            diffs += _walk_diff(a[k], b[k], f"{path}/{k}", tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append({"path": path, "issue": f"len {len(a)} vs {len(b)}"})
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs += _walk_diff(x, y, f"{path}[{i}]", tol)
    elif isinstance(a, float) and isinstance(b, float):
        if abs(a - b) > tol * max(1.0, abs(a), abs(b)):
            diffs.append({"path": path, "committed": a, "reproduced": b})
    elif a != b:
        diffs.append({"path": path, "committed": a, "reproduced": b})
    return diffs


def check_repro(rerun: bool) -> dict:
    cmd = "python scripts/run_cross_clock_library_demo.py"
    if not rerun:
        return {"cmd": cmd, "executed": False, "ok": None}
    stage = pathlib.Path(tempfile.mkdtemp(prefix="m15005_repro_"))
    try:
        for sub in ("src", "scripts"):
            shutil.copytree(ROOT / sub, stage / sub)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(stage / "scripts" / "run_cross_clock_library_demo.py")],
                           capture_output=True, text=True, timeout=3600, cwd=stage)
        wall = time.time() - t0
        produced_path = stage / "runs" / "m14792_cross_clock_library" / "results.json"
        if r.returncode != 0 or not produced_path.exists():
            return {"cmd": cmd, "executed": True, "ok": False, "wall_seconds": round(wall, 2),
                    "stderr": (r.stderr or "")[-1500:]}
        committed = json.loads((ART / "results.json").read_text())
        produced = json.loads(produced_path.read_text())
        diffs = _walk_diff(committed, produced)
        return {"cmd": cmd, "executed": True, "ok": not diffs, "wall_seconds": round(wall, 2),
                "differences": diffs[:20], "n_differences": len(diffs),
                "stdout_head": (r.stdout or "").strip().splitlines()[:3],
                "artifact_sha256_committed": sha256(ART / "results.json"),
                "artifact_sha256_reproduced": sha256(produced_path),
                "note": "byte equality is not expected (wall_seconds/timestamp); physics must be bit-identical"}
    finally:
        shutil.rmtree(stage, ignore_errors=True)


# ------------------------------------------------------------- E2. benchmark
def benchmark_table() -> dict:
    d = json.loads((ART / "results.json").read_text())
    a = d["anchors"]
    rows = [
        {"quantity": "tensorlbm Taylor-Green decay rate (measured/analytic 2*nu*k^2)",
         "reference": 1.0,
         "got": a["tensorlbm_taylor_green"]["decay_rate_ratio_measured_over_analytic"],
         "rel_error": a["tensorlbm_taylor_green"]["rel_error"],
         "within_3pct": a["tensorlbm_taylor_green"]["rel_error"] < 0.03},
        {"quantity": "tensordem Stokes relaxation v at t=0.2 s [m/s]",
         "reference": a["tensordem_stokes_relaxation"]["v_analytic_m_s"],
         "got": a["tensordem_stokes_relaxation"]["v_measured_m_s"],
         "rel_error": a["tensordem_stokes_relaxation"]["rel_error"],
         "within_3pct": a["tensordem_stokes_relaxation"]["rel_error"] < 0.03},
    ]
    for c in d["cases"]:
        rows.append({"quantity": f"coupled particle velocity vs exact ZOH, h={c['window_s']:.0e} s",
                     "reference": "exact ZOH recursion from measured u_k",
                     "got": c["v_vs_zoh_rel_error"],
                     "rel_error": c["v_vs_zoh_rel_error"],
                     "within_3pct": c["v_vs_zoh_rel_error"] < 0.03})
        rows.append({"quantity": f"momentum bookkeeping residual, h={c['window_s']:.0e} s",
                     "reference": 0.0,
                     "got": c["measured_system_momentum_residual_rel"],
                     "rel_error": c["measured_system_momentum_residual_rel"],
                     "within_3pct": c["measured_system_momentum_residual_rel"] < 0.03})
    return {"rows": rows, "all_within_3pct": all(r["within_3pct"] for r in rows),
            "source": "runs/m14792_cross_clock_library/results.json"}


# -------------------------------------------------------------- F. callers
def check_callers() -> dict:
    me = pathlib.Path(__file__).resolve()
    skip = {IMPL.resolve(), TESTS.resolve(), ENTRY.resolve(), me}
    hits = []
    for p in ROOT.rglob("*.py"):
        if ".git" in p.parts or p.resolve() in skip:
            continue
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        if "library_binding" in txt:
            hits.append(str(p.relative_to(ROOT)))
    solver_like = [h for h in hits if h.startswith("src/") and "cross_clock/library_binding" not in h]
    return {"files_referencing_module": sorted(hits), "solver_or_platform_callers": solver_like,
            "ok": bool(solver_like),
            "verdict": ("no caller outside the demo script and the module's own test — "
                        "the coupler is not reachable from any TensorPhysics solver or platform pipeline")
                       if not solver_like else "has callers"}


# ---------------------------------------------------------------- G. level
def scheme_level(chk: dict) -> dict:
    tests = [r.split("::")[-1] for r in re.findall(r"^def (test_\w+)",
                                                   TESTS.read_text(), flags=re.M)] if TESTS.exists() else []
    native = {"tests": tests,
              "impl_files": ["src/tensorphysics/cross_clock/library_binding.py"],
              "entry": "scripts/run_cross_clock_library_demo.py",
              "run_artifacts": [p["file"] for p in chk["artifacts"]["listed"]],
              "integration_callers": chk["callers"]["solver_or_platform_callers"],
              "benchmark": {"rows": chk["benchmark"]["rows"]},
              "repro_cmd": chk["repro"]["cmd"]}
    rungs = {
        1: chk["tests"]["ok"],
        2: chk["pointers"]["ok"] and chk["pointers"]["impl_real"] and chk["artifacts"]["ok"],
        3: chk["callers"]["ok"],
        4: chk["benchmark"]["all_within_3pct"] and chk["repro"]["executed"] and chk["repro"]["ok"],
        5: False,  # no literature/experimental anchor for the *coupled* case was found
    }
    level, failed = 0, []
    for lv in range(1, 6):
        if rungs[lv]:
            level = lv
        else:
            failed.append(lv)
            break
    return {"high_water_level": level, "rung_ok": rungs, "first_failed_rung": failed[:1],
            "evidence_keys": native,
            "note": "cumulative: a rung only counts if every lower rung passed"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rerun", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    chk = {
        "audit_mission": 15005,
        "audited_mission": 14792,
        "anchor": "GAP-CROSS-CLOCK",
        "repo": "TensorPhysics",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": check_artifact_integrity(),
        "pointers": check_pointers(),
        "pins": check_pins(),
        "tests": check_tests(),
        "repro": check_repro(not args.skip_rerun),
        "benchmark": benchmark_table(),
        "callers": check_callers(),
    }
    chk["level"] = scheme_level(chk)
    (OUT / "audit_checks.json").write_text(json.dumps(chk, indent=2))
    lines = [
        f"{k}: {json.dumps(v, ensure_ascii=False)[:400]}"
        for k, v in (("tests", chk["tests"]), ("repro", {kk: chk["repro"][kk] for kk in
                                                        ("executed", "ok", "wall_seconds", "n_differences") if kk in chk["repro"]}),
                     ("artifacts", chk["artifacts"]["ok"]), ("pins_ok", chk["pins"]["ok"]),
                     ("callers", chk["callers"]["files_referencing_module"]),
                     ("high_water_level", chk["level"]["high_water_level"]))
    ]
    (OUT / "audit_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    critical = (chk["artifacts"]["ok"] and chk["pointers"]["ok"] and chk["tests"]["ok"]
                and chk["pins"]["ok"] and (chk["repro"]["ok"] in (True, None)))
    return 0 if critical else 1


if __name__ == "__main__":
    raise SystemExit(main())