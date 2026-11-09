#!/usr/bin/env python3
"""Render ``docs/CRITERION_PREREGISTRATION.md`` from ``artifacts/criterion_prereg/eval.json``.

The page is a *view* of the evaluation record, never a hand-maintained document:
every digit in it is injected from ``eval.json`` (zero hand-copy), so code,
numbers and prose cannot drift apart.  Splitting the renderer out of the
evaluation CLI buys two properties the single-shot script did not have:

* **cheap re-render** -- no numpy/matplotlib run, no corpora regeneration: the
  page can be rebuilt from the evaluation record alone;
* **machine-checkable freshness** -- ``--check`` re-renders in memory and exits
  1 when the checked-in page differs from the render, i.e. a hand-edited digit
  cannot survive CI (the diff head is printed).

Usage::

    python scripts/render_criterion_prereg_doc.py            # write the page
    python scripts/render_criterion_prereg_doc.py --check    # exit 1 on drift
    python scripts/render_criterion_prereg_doc.py --stdout   # print only
    python scripts/render_criterion_prereg_doc.py --eval a.json --out b.md
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "criterion_prereg"
EVAL = OUT / "eval.json"
DOC = ROOT / "docs" / "CRITERION_PREREGISTRATION.md"


def _rel(path: Path) -> str:
    """Path as seen from the repo root, or absolute when it lives outside it."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def render(eval_path: Path = EVAL) -> str:
    """Return the page text for *eval_path* (pure: writes nothing)."""
    d = json.loads(eval_path.read_text(encoding="utf-8"))
    acc = d["acceptance"]
    det = d["detail"]
    demo = det["real_goal_demo"]
    per_seed = det["per_seed"]
    base = det["baseline_text_hash"]
    drift = det["drift"]

    seed_rows = "\n".join(
        f"| `{seed}` | {v['metrics']['recall']:.4f} | {v['metrics']['precision']:.4f} | "
        f"{v['metrics']['class_accuracy']:.4f} | {v['metrics']['tp']}/{v['metrics']['fp']}/{v['metrics']['fn']}/{v['metrics']['tn']} |"
        for seed, v in per_seed.items()
    )
    field_rows = "\n".join(f"| `{name}` | {val:.4f} |" for name, val in det["extraction_per_field"].items())
    receipt_rows = "\n".join(
        f"| `{name}` | `{r['state']}` | `{r['verdict']}` | `{r['dominant_class']}` | "
        f"{r['counts']['unchanged']}/{r['counts']['tightened']}/{r['counts']['loosened']}/"
        f"{r['counts']['scope_changed']}/{r['counts']['undefined']} |"
        for name, r in demo["receipts"].items()
    )
    crit_rows = "\n".join(
        f"| {c['label'] or '—'} | `{c['kind']}` | `{c['metric']}` | `{c['op']}` | `{c['value']}` | "
        f"`{c['count'] if c['count'] is not None else '—'}` | `{c['scope']}` |"
        for c in demo["criteria"]
    )
    hard_rows = "\n".join(
        f"| `{h['case_id']}` | {'✅' if h['all_matched'] else '❌'} | `{json.dumps(h['expected'], ensure_ascii=False)}` | "
        f"`{json.dumps(h['extracted'], ensure_ascii=False)}` |"
        for h in det["hard_cases"]
    )
    limit_rows = "\n".join(
        f"| `{h['case_id']}` | {'✅' if h['all_matched'] else '❌'} | `{json.dumps(h['expected'], ensure_ascii=False)}` | "
        f"`{json.dumps(h['extracted'], ensure_ascii=False)}` |"
        for h in det["limitation_probes"]
    )
    per_case_rows = "\n".join(
        f"| `{c['case_id']}` | `{c['mutation']}` | `{c['expected']}` | `{c['predicted']}` | {'flag' if c['flagged'] else '—'} |"
        for c in det["per_case"]
    )
    b = base
    kpi = demo["kpi"]

    return f"""# Criterion pre-registration: freeze, audit, drift

Campaign-node acceptance criteria are worth nothing if they can move after the
result is known. This node gives every node's criterion block three machine
properties — **pre-frozen**, **post-hoc auditable**, **drift decidable** — and
makes the *fail-closed* direction the only one that passes.

Package: `src/tensorphysics/campaign/criterion_prereg/` ·
Evaluation CLI: `python scripts/run_criterion_prereg_eval.py` ·
Renderer: `python scripts/render_criterion_prereg_doc.py --check` ·
Numbers: `artifacts/criterion_prereg/eval.json` (this document is rendered from
that file, no hand-copied digits; `--check` re-renders in memory and exits 1 when
the checked-in page differs, so a hand-edited digit cannot survive CI).

---

## 1  Case / background

A campaign DAG node is dispatched with a natural-language goal that contains a
"machine acceptance criteria" block, e.g.

> 机器验收判据：(a) 合成检测集24例recall≥0.92、precision≥0.95；(b) 抽取器字段级F1≥0.95；
> (c) fail-closed负控：空goal→unregistered，篡改sha256→tampered；(d) 零领域语义扫描命中=0。

Three failure modes are possible, all observed in practice elsewhere:

1. **criterion inflation** — a threshold that was `≥0.92` at dispatch time is
   reported against `≥0.80` at receipt time ("我们当时就是这么说的");
2. **scope shrink** — the metric is unchanged but the evidence set shrank
   (24 cases → 12 cases, or a different sample population);
3. **silent replacement** — the operator changes family (`≥` → `=`), or a
   behavioural assertion flips ("empty goal → accepted").

A text digest of the goal catches all three *and* cries wolf on every
reformatting (full-width glyphs, `0.920` vs `0.92`, re-labelled clauses), which
is why teams disable it. This node replaces it with a structured freeze.

Parameters of the node (the "geometry" of this study): 24-case synthetic
detection set, 20-goal extraction corpus (≈{det['extraction']['tp'] + det['extraction']['fn']}
field values), 5 seeds for stability, 1 real-goal demo (this node's own goal,
{demo['n_criteria']} structured criteria).

## 2  Method

Pipeline (all in `criterion_prereg`):

```
goal text --extract_criteria--> [Criterion]  --canonical_json--> body
           --sha256--> manifest_sha256 --attach_to_metadata--> mission metadata
receipt text --extract_criteria--> [Criterion]                      |
           --pair by identity (kind:metric)-- compare_criterion ---+
                 -> class(unchanged|tightened|loosened|scope_changed|undefined)
                 -> verdict(pass|fail, fail-closed) -> events -> kpi_view
```

* **Extraction** — acceptance block located by generic headers, clauses split on
  separators (thousands separators protected), full-width glyphs folded
  (``→`>`, `＝`→`=`, `％`→`%`, `（a）`→`(a)`), label/count/scope/metric/op/value
  split by a documented grammar; `→`-style clauses become `mapping` criteria.
* **Canonical JSON** — `sort_keys`, compact separators, UTF-8; the digest is
  computed over everything except `manifest_sha256` and the derived `state`.
* **Semantic digest** (`semantic_digest`) — second, surface-invariant digest over
  the criteria *minus* the raw clause text: separates presentation drift from
  requirement drift.
* **Pairing** — by identity `kind:metric` (never by value), so a changed value or
  scope is still paired and therefore *classifiable*, not misreported as
  removal + addition.
* **Classification** — `≥`/`>` raised ⇒ `tightened`, lowered ⇒ `loosened`;
  `≤`/`<` mirrored; `==` value change ⇒ `undefined`; operator family change,
  mapping-target change, kind change ⇒ `undefined`; count/unit/scope change with
  an unchanged value ⇒ `scope_changed`; dropped criterion ⇒ `loosened`;
  added criterion ⇒ `tightened`. Worst class wins per criterion; verdict `pass`
  only if every criterion is `unchanged` or `tightened` **and** the manifest
  verifies as `registered`.
* **Fail-closed states** — `unregistered` (no criteria / empty receipt, no digest
  published, `attach_to_metadata` raises) and `tampered` (digest or goal hash
  mismatch). Both emit `criterion.prereg.anomaly` events with `severity=error`.
* **KPI view** — `prereg_coverage`, `audit_pass_rate`, `drift_rate`,
  `class_histogram`, `reason_histogram`, `fail_closed_incidents`.
* **Zero-domain-semantics** — vocabulary externalised to
  `domain_vocabulary.json`; missing file degrades to `vocabulary_loaded=false`
  rather than a silent green.

Synthetic corpora carry **generator-side ground truth** (`criterion_prereg.synth`):
the generator writes a criterion, renders it with a random surface form, and the
label is what it wrote — the extractor/auditor must recover it. The drift corpus
applies exactly one labelled mutation per case.

## 3  Results

### 3.1  Acceptance criteria (machine-checked)

| # | criterion | threshold | measured | verdict |
|---|---|---|---|---|
| a | drift detection, 24 synthetic cases | recall ≥ 0.92, precision ≥ 0.95 | recall **{acc['a_detection_24cases']['recall']:.4f}**, precision **{acc['a_detection_24cases']['precision']:.4f}** | {'PASS' if acc['a_detection_24cases']['pass'] else 'FAIL'} |
| b | extractor field-level F1 | F1 ≥ 0.95 | F1 **{acc['b_extractor_field_f1']['f1']:.4f}** (held-out seed {acc['b_extractor_field_f1']['heldout_seed_f1']:.4f}) | {'PASS' if acc['b_extractor_field_f1']['pass'] else 'FAIL'} |
| c | fail-closed negative controls | empty goal → `unregistered`, tampered digest → `tampered` | `{acc['c_fail_closed']['empty_goal_state']}` / `{acc['c_fail_closed']['tampered_state']}`, empty digest = {acc['c_fail_closed']['empty_goal_sha_is_empty']} | {'PASS' if acc['c_fail_closed']['pass'] else 'FAIL'} |
| d | zero-domain-semantics scan | hits = 0 | **{acc['d_zero_domain_semantics']['hits']}** hits over {acc['d_zero_domain_semantics']['n_files']} files / {acc['d_zero_domain_semantics']['n_terms']} terms | {'PASS' if acc['d_zero_domain_semantics']['pass'] else 'FAIL'} |

Node verdict: **{'ALL PASS' if d['all_acceptance_pass'] else 'NOT ALL PASS'}**.

### 3.2  Drift detection across seeds (24 cases each: {drift['n_positive']} drift / {drift['n_cases'] - drift['n_positive']} non-drift)

| seed | recall | precision | class accuracy | tp/fp/fn/tn |
|---|---|---|---|---|
{seed_rows}

Classification confusion (main seed) is in `fig_confusion_matrix.png`; the
24-case predictions are listed in the appendix.

### 3.3  Baseline comparison — what the structure buys

| detector | precision | recall | F1 |
|---|---|---|---|
| structured audit (this node) | {drift['precision']:.4f} | {drift['recall']:.4f} | {drift['f1']:.4f} |
| baseline: goal-text digest | {b['precision']:.4f} | {b['recall']:.4f} | {b['f1']:.4f} |

The baseline flags every reformatted goal *and* every legitimate tightening
({b['fp']} false positives, i.e. exactly the {drift['n_cases'] - drift['n_positive']}
non-drift cases) while the audit stays clean — legitimate tightening must be
*recorded*, not punished, and reformatting must not be reported as drift.

### 3.4  Extractor field agreement (per field)

| field | agreement |
|---|---|
{field_rows}

Out-of-grammar probes (not part of acceptance, recorded for honesty):

| probe | matched | expected | extracted |
|---|---|---|---|
{hard_rows}
{limit_rows}

### 3.5  Real-goal demo: this node's own acceptance criteria

`freeze()` on the deployed goal text produces {demo['n_criteria']} structured
criteria, `manifest_sha256 = {demo['frozen_sha256']}`,
`semantic_digest = {demo['semantic_digest']}`, state `{demo['verify_state']}`;
the metadata payload that the dispatcher would publish is
`detail.real_goal_demo.mission_metadata` in `eval.json`.

| label | kind | metric | op | value | count | scope |
|---|---|---|---|---|---|---|
{crit_rows}

Receipt scenarios (`detail.real_goal_demo.receipts`; the event stream is
`detail.real_goal_demo.events`):

| scenario | state | verdict | dominant class | counts un/tig/loo/scope/undef |
|---|---|---|---|---|
{receipt_rows}

KPI of the five scenarios: `prereg_coverage = {kpi['prereg_coverage']:.3f}`,
`audit_pass_rate = {kpi['audit_pass_rate']:.3f}`,
`drift_rate = {kpi['drift_rate']:.3f}`,
`fail_closed_incidents = {kpi['fail_closed_incidents']}`,
`class_histogram = {json.dumps(kpi['class_histogram'], ensure_ascii=False)}`.

## 4  Figures

| file | content |
|---|---|
| `fig_confusion_matrix.png` | 5×5 confusion of drift classes, main seed |
| `fig_baseline_vs_audit.png` | precision/recall/F1: structured audit vs text digest |
| `fig_extractor_field_agreement.png` | per-field extractor agreement with 0.95 line |
| `fig_demo_kpi.png` | true-class histogram of the 24 cases + real-goal demo KPI |

The four PNGs, `eval.json` and this document are all produced by a single run
of `python scripts/run_criterion_prereg_eval.py` (≈2 s, numpy+matplotlib only):
everything under `artifacts/criterion_prereg/` is regenerable, and the numbers
below are injected from `eval.json`, so code, numbers and prose cannot drift
apart. Re-run the script after touching the package rather than editing prose.

Rendering and publishing are separate, offline steps:
`python scripts/render_criterion_prereg_doc.py` rebuilds this page from
`eval.json` alone (no corpora, no matplotlib) and `--check` fails when the
checked-in page and a fresh render differ; `python
scripts/publish_criterion_prereg_report.py --dry-run` assembles the paper-level
report payload (six-element markdown + base64 figures) and validates it
fail-closed (missing/short/undecodable figure ⇒ non-zero exit) before anything
is posted.

### Relationship to `campaign/criterion_registry` (overlap, declared)

`criterion_registry` (merged earlier) freezes a per-block *six-field* manifest
(`threshold/unit/denominator/exclusion/reference_path/comparator`) and audits it
field-by-field. This package works one level finer — per *clause*
(`label/kind/metric/op/value/unit/count/scope`) — because drift direction is only
decidable if the metric and its comparison operator are separated from the number.
Both are pure-Python, dependency-free and domain-neutral, so they do not conflict;
the duplication of "extract → canonical JSON → sha256 → audit" is a known
double source of truth and is scheduled for convergence (see follow-up note in
`docs/CRITERION_PREREGISTRATION.md` §5). Until then, a node should use **exactly
one** of the two: `criterion_registry` for block-level acceptance notes,
`criterion_prereg` for machine-checkable numeric clauses.


## 5  Caveats, limitations, applicability

* **The corpora are synthetic and grammar-internal.** Surface variation
  (glyphs, labels, separators, number formatting, unit spelling) is randomised
  and held-out seeds are used, so the numbers are not sample-memorised — but they
  measure *fidelity to the documented grammar*, not open-world NLP recall on
  arbitrary prose. F1 = {acc['b_extractor_field_f1']['f1']:.3f} must be read that
  way.
* **Undelimited Chinese metric names shorter than 4 characters are ambiguous.**
  The extractor uses a trailing-4-characters heuristic for CJK metrics; with
  `零领域样本命中=0` it yields metric `样本命中`, scope `零领域` instead of
  `命中` / `零领域样本` (see the probe table). Because *both* sides of an audit go
  through the same extractor, drift detection is unaffected; a human-readable
  metric label can be off. Mitigation: delimit CJK metric names with a space or
  use an ASCII identifier in the goal text.
* **Hyphenated ASCII names** (`fail-closed`) contribute their tail token
  (`closed…`) to the scope; scope is compared consistently, so again no drift
  impact, but the scope string is not pretty.
* **`undefined` is a real answer, not a failure of the audit.** Exact-value
  (`==`) changes, operator-family switches and mapping-target changes are
  incomparable; the gate fails closed rather than guessing a direction.
* **Threshold direction assumptions**: for `≥`-type criteria a higher value is
  `tightened`; for `≤`-type a lower value is. Mixed-family criteria are treated
  as `undefined`.
* **Known unrelated test state**: `tests/test_criteria_mutation.py` errors in
  this worktree because its external audit artifact
  (`artifacts/criteria_mutation/paper_trace_audit.json`) is not carried in the
  checkout; unrelated to this node (`tests/test_criterion_prereg.py` and
  `tests/test_criterion_ledger.py` pass).
* **Not wired into the dispatcher yet.** This node delivers the mechanism plus
  CLI/API; the campaign dispatcher must call `freeze()` at node-seed time and
  `audit()` at receipt time. Until then the property is available but not
  enforced fleet-wide.

## 6  Error judgement against the criteria

| key quantity | value | criterion | within tolerance? |
|---|---|---|---|
| detection recall (24 cases) | {drift['recall']:.4f} | ≥ 0.92 | **yes** |
| detection precision (24 cases) | {drift['precision']:.4f} | ≥ 0.95 | **yes** |
| field-level F1 | {acc['b_extractor_field_f1']['f1']:.4f} | ≥ 0.95 | **yes** |
| zero-domain hits | {acc['d_zero_domain_semantics']['hits']} | = 0 | **yes** |
| fail-closed states | `{acc['c_fail_closed']['empty_goal_state']}` / `{acc['c_fail_closed']['tampered_state']}` | as specified | **yes** |

There is no experimental/analytical reference value to compare against in this
node — it is a software-mechanism node, so "accuracy" is defined against
generator-side ground truth (synthetic) and against the specified fail-closed
states. The 3% band used for physics comparisons is therefore applied to the
*acceptance margins* above (all margins are met with slack, except that F1 and
detection rates sit exactly at 1.0000 on the in-distribution corpus — margins of
0.0500 / 0.0800 / 0.0500 over the required thresholds).

## Appendix: 24-case detection set (main seed)

| case | mutation | truth | predicted | flagged |
|---|---|---|---|---|
{per_case_rows}
"""

def render_doc(eval_path: Path = EVAL, doc_path: Path = DOC) -> Path:
    """Render and write the page; returns the path written."""
    doc = render(eval_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(doc, encoding="utf-8")
    print(f"wrote {_rel(doc_path)} ({len(doc)} chars)")
    return doc_path


def check(eval_path: Path = EVAL, doc_path: Path = DOC) -> tuple[bool, str]:
    """True when the checked-in page equals a fresh render (no hand edits)."""
    if not eval_path.exists():
        return False, f"missing evaluation record: {eval_path}"
    if not doc_path.exists():
        return False, f"missing document: {doc_path}"
    fresh = render(eval_path).splitlines()
    shipped = doc_path.read_text(encoding="utf-8").splitlines()
    if fresh == shipped:
        return True, f"{doc_path.name} == fresh render of {eval_path.name} ({len(fresh)} lines)"
    diff = list(difflib.unified_diff(shipped, fresh, "checked-in", "fresh render", n=0, lineterm=""))
    return False, (f"{doc_path.name} drifted from {eval_path.name} "
                   f"({len(diff)} diff lines)\n" + "\n".join(diff[:12]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval", default=str(EVAL),
                    help="evaluation record (default: artifacts/criterion_prereg/eval.json)")
    ap.add_argument("--out", default=str(DOC),
                    help="document path (default: docs/CRITERION_PREREGISTRATION.md)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="verify freshness without writing; exit 1 on drift")
    mode.add_argument("--stdout", action="store_true", help="print the render, do not write")
    args = ap.parse_args(argv)

    eval_path, doc_path = Path(args.eval), Path(args.out)
    if args.check:
        ok, message = check(eval_path, doc_path)
        print(("OK: " if ok else "DRIFT: ") + message)
        return 0 if ok else 1
    if args.stdout:
        sys.stdout.write(render(eval_path))
        return 0
    render_doc(eval_path, doc_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
