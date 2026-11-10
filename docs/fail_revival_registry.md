# Honest-FAIL revival registry and blocker-graph scheduler (m9954 / p337w1-n04)

## 1. What this is

An honest FAIL in this ecosystem is a *debt with a fingerprint*: a criterion that was pre-registered and then missed, with the deviation archived in a campaign node report. Until now the only thing that kept such a debt alive was prose -- a paragraph in the next wave saying it inherits the parent's FAILs. Nothing prevented a FAIL from being silently re-run, silently re-thresholded, or silently declared PASS.

This delivery turns the debt into a machine-checkable loop: **extract** the FAIL records verbatim from the acceptance artifacts, **declare** the obstruction that blocks each one, **schedule** a revival task the moment the obstruction lifts (digests only, no editable threshold), **adjudicate** the revival three ways (PASS / STILL_FAIL / NOT_COMPARABLE), and **refuse** any verdict whose criterion digest is not the registered one. Every accepted verdict writes one lineage edge that a single SQL query answers.

The physical target is not a new numerical result but an *accounting* capability: the campaign can now prove that no honest FAIL is quietly forgotten between waves.

## 2. Corpus and method

The registry never invents evidence. `tools/sync_campaign_nodes.py` snapshots, verbatim, the node goal (which carries the pre-registered machine criteria) and the acceptance report markdown of every plan-337 node plus three further real FAIL nodes, stamping each file with the report's sha256. The extraction run re-verifies that digest before lifting a single record (a tampered artifact yields `status=tampered` and zero records).

A table row becomes a FAIL record when one of its cells carries a FAIL marker. Marker precision is tiered: unambiguous markers (`FAIL`, `未达成`, `未收敛`, `拒绝（诚实` ...) fire anywhere in a row, while ambiguous ones (`否` / `no`) fire only inside a recognised verdict column and never in a column with pass semantics (`实测被拒`: there `no` means the rejection did *not* happen, i.e. a passing negative control). A tick always wins. The verbatim row text is the criterion string and its sha256 the criterion digest -- that digest is what the gate, the scheduler key and the lineage edge are all pinned to.

| mission | report | status | FAIL rows | WITHHELD rows | records | pre-registered criteria |
|---|---|---|---|---|---|---|
| 7813 | 9211 | ok | 1 | 0 | 1 | 6 |
| 7814 | 9195 | ok | 8 | 3 | 8 | 5 |
| 7815 | 9178 | ok | 4 | 1 | 4 | 6 |
| 7816 | None | no_report | 0 | 0 | 0 | 5 |
| 7817 | 9162 | ok | 0 | 0 | 0 | 7 |
| 7818 | None | no_report | 0 | 0 | 0 | 6 |
| 7819 | 9238 | ok | 0 | 0 | 0 | 5 |
| 7820 | None | no_report | 0 | 0 | 0 | 6 |
| 7821 | 9207 | ok | 0 | 0 | 0 | 6 |
| 7822 | None | no_report | 0 | 0 | 0 | 6 |
| 7823 | None | no_report | 0 | 0 | 0 | 6 |
| 7827 | 9181 | ok | 4 | 3 | 4 | 5 |
| 9801 | 11723 | ok | 2 | 0 | 2 | 1 |
| 9824 | 11727 | ok | 4 | 1 | 4 | 4 |
| 9840 | 11692 | ok | 8 | 0 | 8 | 0 |
| 9860 | 11718 | ok | 1 | 0 | 1 | 3 |
| 9871 | 11799 | ok | 5 | 0 | 5 | 0 |

Records per mission: m7813=1, m7814=8, m7815=4, m7827=4, m9801=2, m9824=4, m9840=8, m9860=1, m9871=5 (total 37 records from 17 artifacts, 12 with a report).

## 3. Blocker graph

Each FAIL declares exactly one blocker, of three kinds: a capability slug, an upstream mission, or a numerical-caliber change. Declarations are data (`tensor_reports/campaign_nodes/revival_blockers.json`) and carry their provenance: the 9 curated declarations (the plan-337 fixture) cite the report section that states the obstruction; the remaining 28 are rule-derived from the record's attribution label and say so. FAILs with no declaration keep an `undeclared` blocker that can never lift automatically, so the debt stays visible instead of disappearing.

The graph is acyclic (`acyclic=True`) and carries the campaign DAG on top of the declarations, so an upstream mission landing is visible to the scheduler even if nobody edited a declaration.

| blocker | kind | done in final scenario |
|---|---|---|
| caliber:derived-7814 | numerical_caliber_change | False |
| caliber:derived-9824 | numerical_caliber_change | False |
| caliber:derived-9860 | numerical_caliber_change | False |
| caliber:derived-9871 | numerical_caliber_change | False |
| caliber:fs-discretisation-sign | numerical_caliber_change | True |
| caliber:response-bandwidth | numerical_caliber_change | True |
| capability:derived-gap-7814 | capability_slug | False |
| capability:derived-model-form-7814 | capability_slug | False |
| capability:derived-model-form-9801 | capability_slug | False |
| capability:derived-model-form-9840 | capability_slug | False |
| capability:maneuvering-derivative-uncertainty | capability_slug | True |
| capability:reference-anchor | capability_slug | False |
| mission:6008 | upstream_mission | True |
| undeclared:FR-749b29919965330f | undeclared | False |
| undeclared:FR-75bbc1925c76180c | undeclared | False |
| undeclared:FR-9f929a5e7afb705b | undeclared | False |
| undeclared:FR-f32971afaa991a73 | undeclared | False |

## 4. Results

### 4.1 Machine acceptance

| criterion | measured | required | verdict |
|---|---|---|---|
| C1 extraction rate | 1.0000 | = 1.000 | PASS |
| C1 criterion/report digest match | 1.0000 / 1.0000 | = 1.000 (n>=20) | PASS |
| C2 rewrite attempts refused | 10/10 | 10/10 with correct code | PASS |
| C3 false triggers (nothing landed) | 0 | = 0 | PASS |
| C3 max trigger latency [cycles] | 1 | <= 1 | PASS |
| C4 identical repeat runs | True | True | PASS |
| C4 duplicate tasks on re-tick | 0 | = 0 | PASS |
| C5 forged PASS refused | True | True | PASS |
| C5 OVERCLAIM ledger rows | 1 | = 1 | PASS |
| C5 NOT_COMPARABLE kept in family | True | True | PASS |
| C6 domain-vocabulary hits | 0 | = 0 | PASS |

All six criteria pass: yes.

### 4.2 Revival tasks and verdicts on the real fixture

| mission | fail key | blocker kind | verdict | registered tolerance | measured | in FAIL family |
|---|---|---|---|---|---|---|
| 7815 | FR-e18b9172d4d79c35 | upstream_mission | STILL_FAIL | <3% | -53.6 | True |
| 7827 | FR-3583aef3d0fed859 | numerical_caliber_change | NOT_COMPARABLE | None | None | True |
| 7827 | FR-84d6e999d7fb01e5 | numerical_caliber_change | NOT_COMPARABLE | None | None | True |
| 7827 | FR-f682748ea4f02635 | numerical_caliber_change | NOT_COMPARABLE | <3% | None | True |
| 7815 | FR-8c6bb4d80c5fde26 | upstream_mission | STILL_FAIL | <3% | 73.6 | True |
| 7815 | FR-20237c8983a52e30 | capability_slug | STILL_FAIL | <3% | -61.2 | True |
| 7815 | FR-8e0eb67d5829100f | capability_slug | STILL_FAIL | <3% | None | True |
| 7813 | FR-12ab2c5f2336a4fb | numerical_caliber_change | NOT_COMPARABLE | <3% | None | True |
| 7827 | FR-90bd2a6b95d93d77 | numerical_caliber_change | NOT_COMPARABLE | None | None | True |

### 4.3 Zero-rewrite gate

Ten synthetic rewrite attempts against the registered criterion digest are refused, each with its own reason code, while the unmodified request control is admitted:

| case | expected reason | actual reason | outcome |
|---|---|---|---|
| relaxed_tolerance | EDITABLE_FIELD_FORBIDDEN | EDITABLE_FIELD_FORBIDDEN | refused |
| edited_threshold | EDITABLE_FIELD_FORBIDDEN | EDITABLE_FIELD_FORBIDDEN | refused |
| rewritten_metric | EDITABLE_FIELD_FORBIDDEN | EDITABLE_FIELD_FORBIDDEN | refused |
| verdict_injection | VERDICT_INJECTION_FORBIDDEN | VERDICT_INJECTION_FORBIDDEN | refused |
| criterion_text_edit | EDITABLE_FIELD_FORBIDDEN | EDITABLE_FIELD_FORBIDDEN | refused |
| truncated_digest | HASH_FORMAT_INVALID | HASH_FORMAT_INVALID | refused |
| unregistered_fail | UNREGISTERED_FAIL | UNREGISTERED_FAIL | refused |
| missing_input | INPUT_HASH_MISSING | INPUT_HASH_MISSING | refused |
| substituted_criterion | CRITERION_HASH_MISMATCH | CRITERION_HASH_MISMATCH | refused |
| registry_digest_claim | CRITERION_HASH_MISMATCH | CRITERION_HASH_MISMATCH | refused |

### 4.4 Figures

![blocker graph](figures/fig1_blocker_graph.png)

*Figure 1 -- blocker graph of the plan-337 revival fixture.*

![acceptance](figures/fig2_acceptance_criteria.png)

*Figure 2 -- machine acceptance dashboard (C1..C6).*

![lineage](figures/fig3_verdicts_lineage.png)

*Figure 3 -- three-way verdicts and the anti-whitewash ledger.*

## 5. Pitfalls, scope and limitations

1. **Marker ambiguity is the real risk.** A naive `否`/`no` scan lifts 13 false FAILs out of the m7814 report (negative controls that were correctly *not* rejected). The tiered marker model plus pass-semantics columns is what keeps the completeness denominator and the extracted set identical; both are re-counted independently (`scan_fail_marks` does no column resolution).
2. **WITHHELD is not FAIL.** Eight WITHHELD rows are counted and explicitly excluded from the FAIL denominator instead of being folded into it.
3. **Attribution is a label, not a measurement.** It is assigned by a rule table and records its basis (`row:<token>` or `report_summary:<token>`); of the records, some stay `unclassified`. No gate decision depends on it.
4. **Baseline input digests are run-context digests.** The artifacts do not contain the raw inputs, so the baseline digest pins (mission, report digest, criterion digest) -- the archived evidence a revival must re-open, not a fingerprint of the original mesh or data file. Declarations may override it; a change is then reported as a NOT_COMPARABLE change.
5. **Caliber revivals can never produce PASS.** By construction a revival that adopts a revised numerical caliber reports NOT_COMPARABLE and stays in the FAIL family; only a comparable re-run with a reproduction control can close a FAIL. That is a feature, and it is why the PASS path is demonstrated on a synthetic control rather than claimed on real data.
6. **Scope of the corpus.** 17 artifacts / 37 FAIL records is a real but small corpus; the extraction covers markdown tables, not prose-only FAIL declarations. Rows whose verdict is carried by a bare number (no marker at all) are deliberately not extracted.

## 6. Error judgement

No new physical quantity is predicted here, so there is no experimental benchmark to compare against and the usual `<3%` question does not apply to a numerical field: the deliverable is an accounting mechanism. Its own error budget is measured instead, and each item is exact rather than approximate:

* extraction completeness = 37/37 FAIL marks = 1.0000 (required 1.0000);
* criterion-string and report-digest agreement on a 20-record cross-check re-read from disk = 1.0000 / 1.0000 (required 1.0000);
* gate rejection = 10/10 (required 10/10) with 6 distinct reason codes;
* trigger latency = 1 cycle(s) (required <= 1), false triggers = 0, missed = 0;
* idempotency: identical repeat run = True, new tasks on re-tick = 0;
* anti-whitewash: forged PASS refused = True, OVERCLAIM rows = 1, forged PASS lineage edges = 0;
* domain-vocabulary hits in code + fixtures = 0 across 18 files.

**Is the error < 3%?** Not applicable as a physical error and answered exactly instead: all six criteria are met at the level of exact integers/booleans, so the mechanism's own error is 0 by construction -- with the honest caveat that this says nothing about whether the *re-measured physics* is right; that is the job of the revival runs the scheduler will trigger.

## 7. Reproduce

```bash
python3 tools/sync_campaign_nodes.py --plan 337 --missions 9801 9824 9871 9840 9860
python3 tools/make_revival_blockers.py
PYTHONPATH=src python3 -m tensorphysics.fail_revival acceptance
PYTHONPATH=src python3 -m pytest tests/test_fail_revival_extract.py tests/test_fail_revival_gate.py \
    tests/test_fail_revival_scheduler.py tests/test_fail_revival_lineage.py \
    tests/test_fail_revival_scan.py -q
python3 tools/make_fail_revival_report.py
```

Artifacts: `artifacts/fail_revival/acceptance.json`, `artifacts/fail_revival/lineage.sqlite3`, `artifacts/fail_revival/figures/*.png`, `docs/fail_revival_registry.md`.

