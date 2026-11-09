"""Assemble and POST the m8614 paper-level progress report."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request
import urllib.error

ROOT = pathlib.Path("/root/tp-agent/work/m8614")
RUN = ROOT / "runs" / "m8614_cross_solver"
nums = json.loads((RUN / "numbers.json").read_text())
ca, cb = nums["case_A"], nums["case_B"]

figs = []
for name, cap in (
    (
        "fig1_consensus.png",
        "Four solvers (LBM/FVM/FEM/BEM) on a drag coefficient: each point is the solver value with its 1-sigma bar; the red dashed line is the inverse-variance weighted consensus and the shaded band its uncertainty. Widest pair LBM/FEM, relative spread 0.83% -> PASS.",
    ),
    (
        "fig2_gate_sweep.png",
        "Gate verdict as one solver (BEM) is offset from three anchors at 1.00. The outlier check (3-sigma) trips the FAIL at ~4% offset while the relative spread band is still only WARN, showing the two checks are complementary and the gate is conservative.",
    ),
):
    figs.append(
        {
            "name": name,
            "caption": cap,
            "png_base64": base64.b64encode((RUN / name).read_bytes()).decode(),
        }
    )

sol = ca["solvers"]
md = f"""# TensorPhysics#161 (reopened) · Round 1 — cross-solver agreement gate (`vv.cross_solver`)

Reopened mission m8614. The original PR#161 was a 12-file big-bang refactor of the
`vv` package and was closed by the owner (2026-11-05). Per the "small unit step"
delivery discipline (<=10 files, single functional point), Round 1 delivers **one
genuinely-new, self-contained capability missing from `main`**: the cross-solver
agreement gate — the mutual-consistency half of the V&V machine gate.

## 1. Case introduction (background / parameters)

The V&V gate already judges (a) one solver's own grid convergence
(`vv.gci`) and (b) one solver against a reference (`vv.validation`). The missing
third question has **no reference at all**: *N solvers each report a value for the
same observable on the same problem — do they agree, and what consensus value with
an uncertainty may be published?*

The gate consumes one `SolverSample` per solver (name, value `f`, optional 1-sigma
`u`, optional grid `h`) and returns a consensus, an agreement band and a
PASS/WARN/FAIL verdict. Policy thresholds (defaults, ASME-style):

| parameter | default | meaning |
|---|---|---|
| `warn_spread` | 0.02 | relative band above which the verdict is WARN |
| `fail_spread` | 0.05 | relative band above which the verdict is FAIL |
| `outlier_sigma` | 3.0 | a solver > this many of its own sigmas from consensus is an outlier (FAIL) |
| `relative` | True | band measured as a fraction of \\|consensus\\| (absolute fallback at zero consensus) |

## 2. How it was done (method / flow / toolchain)

* **Consensus** — inverse-variance weighted mean `f_c = sum w_i f_i / sum w_i`,
  `w_i = 1/u_i^2`, with `u_c = 1/sqrt(sum w_i)`. With no declared uncertainties it
  degrades to the plain mean (`u_c = None`). Mixed declared/undeclared
  uncertainties are rejected (the caller must be explicit).
* **Agreement band** — largest pairwise disagreement `spread = max|f_i - f_j|`,
  `spread_rel = spread/|f_c|` (absolute fallback when `f_c = 0`).
* **Outlier test** — a solver is an outlier when `|f_i - f_c| > outlier_sigma * u_i`.
* **Verdict** — FAIL if the band exceeds `fail_spread` **or** any outlier; WARN if
  the band exceeds `warn_spread`; else PASS. A FAIL is never overwritten.

Toolchain: pure-stdlib Python (frozen dataclasses, `as_dict`/`from_mapping`),
pytest with hand-recomputed literals (independent check, not a snapshot),
matplotlib for the figures. Files: `src/tensorphysics/vv/cross_solver.py` (new),
`tests/test_vv_cross_solver_m8614.py` (new), `src/tensorphysics/vv/__init__.py`
(additive exports only).

## 3. Computation results (programmatic, zero hand-copy)

All numbers below are produced by importing the module and calling its API
(`scripts/m8614_cross_solver_figures.py` -> `runs/m8614_cross_solver/numbers.json`).

**Case A — four solvers on a drag coefficient `Cd`:**

| solver | f | u |
|---|---|---|
| TensorLBM | {sol['TensorLBM']['f']:.4f} | {sol['TensorLBM']['u']:.4f} |
| TensorFVM | {sol['TensorFVM']['f']:.4f} | {sol['TensorFVM']['u']:.4f} |
| TensorFEM | {sol['TensorFEM']['f']:.4f} | {sol['TensorFEM']['u']:.4f} |
| TensorBEM | {sol['TensorBEM']['f']:.4f} | {sol['TensorBEM']['u']:.4f} |

| quantity | value |
|---|---|
| consensus `f_c` | {ca['consensus']:.6e} |
| consensus `u_c` | {ca['consensus_u']:.6e} |
| spread (abs) | {ca['spread']:.6e} |
| spread (rel) | {ca['spread_rel']:.6e} |
| widest pair | {ca['widest_pair'][0]} / {ca['widest_pair'][1]} |
| status | **{ca['status']}** |
| outliers | {ca['outliers'] or 'none'} |

**Case B — one solver offset from three anchors at 1.00 (81-point sweep):**

| quantity | value |
|---|---|
| first non-PASS offset | {cb['first_non_pass_delta']:.3f} ({cb['first_non_pass_delta']*100:.1f}%) |
| first FAIL offset | {cb['first_fail_delta']:.3f} ({cb['first_fail_delta']*100:.1f}%) |
| sweep points | {cb['n_sweeps']} |

The FAIL at {cb['first_fail_delta']*100:.0f}% is triggered by the **outlier** check
(BEM at 3-sigma) while the spread band is still WARN — the two checks are
complementary, and the gate fails on the earlier of the two.

## 4. Result figures

This is a **verification-infrastructure** task: there is no physical field to
contour, so the figures show the gate's numerical behaviour instead (stated per
the no-figure rule for backend tasks).

* Figure 1 — consensus and its uncertainty band for the four-solver `Cd` case.
* Figure 2 — gate verdict vs. one solver's offset, with the warn/fail bands.

## 5. Notes (pitfalls / limitations / scope)

* **Float boundary at exactly 3-sigma**: a deviation of exactly `outlier_sigma`
  can land a few ulp over in floating point and trip the outlier check. The gate
  is deliberately conservative (a marginal case fails rather than passes); tests
  use offsets clear of the boundary.
* **All-or-nothing uncertainty**: a study mixing declared and undeclared
  uncertainties is rejected, forcing the caller to choose the weighted or the
  plain-mean regime explicitly.
* **Outlier test uses the full consensus**, which an extreme outlier pulls toward
  itself, slightly weakening the test for a single gross outlier. A leave-one-out
  (or median) reference is a natural Round-2 hardening.
* **Scope**: mutual consistency of an ensemble with no reference. It does not
  replace `vv.validation` (vs. reference) or `vv.gci` (single-solver convergence);
  it complements them.

## 6. Error judgement (key quantities vs. baseline, is it <3%?)

Baseline = the hand-recomputed analytic references used as literals in
`tests/test_vv_cross_solver_m8614.py` (independent of the implementation).

| key quantity | implementation | hand reference | rel. error | <3%? |
|---|---|---|---|---|
| weighted consensus (A:1.0/0.01, B:1.02/0.02) | 1.004 | 1.004 | ~1e-14 | **yes** |
| consensus uncertainty | 1/sqrt(12500) | 1/sqrt(12500) | ~1e-14 | **yes** |
| spread (A:1.00, B:1.03) | 0.03 | 0.03 | ~1e-14 | **yes** |
| spread_rel | 0.03/1.015 | 0.03/1.015 | ~1e-14 | **yes** |
| gate verdicts (8-case hand table) | 8/8 match | 8/8 | 0 | **yes** |

Every key quantity matches its hand-computed reference to machine precision
(~1e-14), i.e. **well below the 3% bar**. The 21/21 new tests pass; the full
`vv` suite is 175 passed / 0 failed. The 24 failed + 29 errors in the full repo
suite are **pre-existing on pristine `origin/main`** (reproduced by stashing this
change and re-running the same files: identical 24 failed + 29 errors) and none of
those files import `tensorphysics.vv`.
"""

payload = {
    "title": "TensorPhysics#161 R1: cross-solver agreement gate (vv.cross_solver)",
    "summary": (
        "Added vv.cross_solver: inverse-variance weighted consensus, largest-pairwise "
        "agreement band and a conservative PASS/WARN/FAIL cross-solver agreement gate, "
        "with 21 hand-referenced tests. Full vv suite green; the 24F+29E in the repo "
        "suite are pre-existing on pristine main (verified by stash baseline)."
    ),
    "scenario_key": "tensorphysics_vv_cross_solver_agreement",
    "mission_id": 8614,
    "tags": ["tensorphysics", "vv", "verification", "cross-solver", "asme-vv20", "m8614"],
    "markdown": md,
    "figures": figs,
}
(ROOT / "runs" / "m8614_cross_solver" / "report.json").write_text(json.dumps(payload))

base = os.environ["TENSOR_API_BASE"]
tok = open("/tmp/m8614_token.txt").read().strip()
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(payload).encode(), method="POST")
req.add_header("Content-Type", "application/json")
req.add_header("Authorization", "Bearer " + tok)
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print("report status", r.status)
        print(r.read().decode()[:400])
except urllib.error.HTTPError as e:
    print("report HTTP", e.code)
    print(e.read().decode()[:600])
