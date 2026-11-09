# Criterion pre-registration: freeze, audit, drift

Campaign-node acceptance criteria are worth nothing if they can move after the
result is known. This node gives every node's criterion block three machine
properties — **pre-frozen**, **post-hoc auditable**, **drift decidable** — and
makes the *fail-closed* direction the only one that passes.

Package: `src/tensorphysics/campaign/criterion_prereg/` ·
Evaluation CLI: `python scripts/run_criterion_prereg_eval.py` ·
Numbers: `artifacts/criterion_prereg/eval.json` (this document is rendered from
that file, no hand-copied digits).

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
detection set, 20-goal extraction corpus (≈456
field values), 5 seeds for stability, 1 real-goal demo (this node's own goal,
6 structured criteria).

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
| a | drift detection, 24 synthetic cases | recall ≥ 0.92, precision ≥ 0.95 | recall **1.0000**, precision **1.0000** | PASS |
| b | extractor field-level F1 | F1 ≥ 0.95 | F1 **1.0000** (held-out seed 1.0000) | PASS |
| c | fail-closed negative controls | empty goal → `unregistered`, tampered digest → `tampered` | `unregistered` / `tampered`, empty digest = True | PASS |
| d | zero-domain-semantics scan | hits = 0 | **0** hits over 6 files / 36 terms | PASS |

Node verdict: **ALL PASS**.

### 3.2  Drift detection across seeds (24 cases each: 13 drift / 11 non-drift)

| seed | recall | precision | class accuracy | tp/fp/fn/tn |
|---|---|---|---|---|
| `20261109` | 1.0000 | 1.0000 | 1.0000 | 13/0/0/11 |
| `7` | 1.0000 | 1.0000 | 1.0000 | 13/0/0/11 |
| `42` | 1.0000 | 1.0000 | 1.0000 | 13/0/0/11 |
| `999` | 1.0000 | 1.0000 | 1.0000 | 13/0/0/11 |
| `1` | 1.0000 | 1.0000 | 1.0000 | 13/0/0/11 |

Classification confusion (main seed) is in `fig_confusion_matrix.png`; the
24-case predictions are listed in the appendix.

### 3.3  Baseline comparison — what the structure buys

| detector | precision | recall | F1 |
|---|---|---|---|
| structured audit (this node) | 1.0000 | 1.0000 | 1.0000 |
| baseline: goal-text digest | 0.5417 | 1.0000 | 0.7027 |

The baseline flags every reformatted goal *and* every legitimate tightening
(11 false positives, i.e. exactly the 11
non-drift cases) while the audit stays clean — legitimate tightening must be
*recorded*, not punished, and reformatting must not be reported as drift.

### 3.4  Extractor field agreement (per field)

| field | agreement |
|---|---|
| `label` | 1.0000 |
| `kind` | 1.0000 |
| `metric` | 1.0000 |
| `op` | 1.0000 |
| `value` | 1.0000 |
| `unit` | 1.0000 |
| `count` | 1.0000 |
| `scope` | 1.0000 |

Out-of-grammar probes (not part of acceptance, recorded for honesty):

| probe | matched | expected | extracted |
|---|---|---|---|
| `hard-fullwidth` | ✅ | `{"metric": "recall", "op": ">=", "value": "0.92", "count": 24, "scope": "合成检测集", "label": "a"}` | `{"metric": "recall", "op": ">=", "value": "0.92", "count": 24, "scope": "合成检测集", "label": "a"}` |
| `hard-percent-unit` | ✅ | `{"metric": "f1", "op": ">=", "value": "95", "unit": "%", "count": 48, "scope": "全量用例"}` | `{"metric": "f1", "op": ">=", "value": "95", "unit": "%", "count": 48, "scope": "全量用例"}` |
| `limit-undelimited-cjk-2char` | ❌ | `{"metric": "命中", "op": "==", "value": "0", "scope": "零领域样本"}` | `{"metric": "样本命中", "op": "==", "value": "0", "scope": "零领域"}` |

### 3.5  Real-goal demo: this node's own acceptance criteria

`freeze()` on the deployed goal text produces 6 structured
criteria, `manifest_sha256 = 597a6aeb702c94d7db17472950aca95f1a9234506114c3c3f6e296d4ae59a144`,
`semantic_digest = 699a01ad6c64ff6bdd282ba96146477ab1253230cbb363cefc368e556f0e5dfe`, state `registered`;
the metadata payload that the dispatcher would publish is
`detail.real_goal_demo.mission_metadata` in `eval.json`.

| label | kind | metric | op | value | count | scope |
|---|---|---|---|---|---|---|
| a | `threshold` | `recall` | `>=` | `0.92` | `24` | `合成检测集` |
| — | `threshold` | `precision` | `>=` | `0.95` | `None` | `` |
| b | `threshold` | `F1` | `>=` | `0.95` | `None` | `抽取器字段级` |
| c | `mapping` | `goal` | `->` | `unregistered` | `None` | `closed负控:空` |
| — | `mapping` | `sha256` | `->` | `tampered` | `None` | `篡改` |
| d | `threshold` | `扫描命中` | `==` | `0` | `None` | `零领域语义` |

Receipt scenarios (`detail.real_goal_demo.receipts`; the event stream is
`detail.real_goal_demo.events`):

| scenario | state | verdict | dominant class | counts un/tig/loo/scope/undef |
|---|---|---|---|---|
| `unchanged_reformatted` | `registered` | `pass` | `unchanged` | 6/0/0/0/0 |
| `loosened_recall` | `registered` | `fail` | `loosened` | 5/0/1/0/0 |
| `narrowed_scope` | `registered` | `fail` | `scope_changed` | 5/0/0/1/0 |
| `dropped_zero_domain` | `registered` | `fail` | `loosened` | 5/0/1/0/0 |
| `tampered_manifest` | `tampered` | `fail` | `unchanged` | 0/0/0/0/0 |

KPI of the five scenarios: `prereg_coverage = 0.800`,
`audit_pass_rate = 0.200`,
`drift_rate = 0.125`,
`fail_closed_incidents = 1`,
`class_histogram = {"loosened": 2, "scope_changed": 1, "tightened": 0, "unchanged": 21, "undefined": 0}`.

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
  arbitrary prose. F1 = 1.000 must be read that
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
| detection recall (24 cases) | 1.0000 | ≥ 0.92 | **yes** |
| detection precision (24 cases) | 1.0000 | ≥ 0.95 | **yes** |
| field-level F1 | 1.0000 | ≥ 0.95 | **yes** |
| zero-domain hits | 0 | = 0 | **yes** |
| fail-closed states | `unregistered` / `tampered` | as specified | **yes** |

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
| `drift-00` | `tightened_raise` | `tightened` | `tightened` | — |
| `drift-01` | `op_family` | `undefined` | `undefined` | flag |
| `drift-02` | `criterion_dropped` | `loosened` | `loosened` | flag |
| `drift-03` | `scope_desc` | `scope_changed` | `scope_changed` | flag |
| `drift-04` | `unchanged_reformatted` | `unchanged` | `unchanged` | — |
| `drift-05` | `scope_count` | `scope_changed` | `scope_changed` | flag |
| `drift-06` | `tightened_raise` | `tightened` | `tightened` | — |
| `drift-07` | `op_family` | `undefined` | `undefined` | flag |
| `drift-08` | `criterion_added` | `tightened` | `tightened` | — |
| `drift-09` | `loosened_lower` | `loosened` | `loosened` | flag |
| `drift-10` | `unchanged_reformatted` | `unchanged` | `unchanged` | — |
| `drift-11` | `exact_value` | `undefined` | `undefined` | flag |
| `drift-12` | `tightened_raise` | `tightened` | `tightened` | — |
| `drift-13` | `scope_count` | `scope_changed` | `scope_changed` | flag |
| `drift-14` | `unchanged_reformatted` | `unchanged` | `unchanged` | — |
| `drift-15` | `mapping_target` | `undefined` | `undefined` | flag |
| `drift-16` | `loosened_lower` | `loosened` | `loosened` | flag |
| `drift-17` | `loosened_lower` | `loosened` | `loosened` | flag |
| `drift-18` | `unchanged_reformatted` | `unchanged` | `unchanged` | — |
| `drift-19` | `scope_desc` | `scope_changed` | `scope_changed` | flag |
| `drift-20` | `criterion_added` | `tightened` | `tightened` | — |
| `drift-21` | `criterion_dropped` | `loosened` | `loosened` | flag |
| `drift-22` | `unchanged_reformatted` | `unchanged` | `unchanged` | — |
| `drift-23` | `unchanged_reformatted` | `unchanged` | `unchanged` | — |
