# Failure-taxonomy external validity: does the in-corpus agreement survive unseen documents?

Node **P187-W1-N10** · repository `TensorPhysics` · module `tensorphysics.failure_taxonomy`
· audit protocol `failure-taxonomy-external-validity-v1` · generated from `artifacts/failure_taxonomy_external/audit.json`
(no number below is typed by hand).

## 1. Case and question

The parent work delivered a six-class, method-level failure taxonomy for blocked campaign
nodes, a reroute decision table, and a frozen 48-record back-test that cleared the
pre-registered bars (dual-annotation kappa 0.750, agreement
0.792, decoy control 0.146,
permuted-label control 0.003,
decision-table schema coverage 6/6 — all four re-measured inside this audit, not quoted from
the parent run). Those bars are
*in-corpus*: the corpus passages were in front of the analyst while the cue lexicon was
curated, and 28 of the
48 records are siblings of another record from the
same document (12 documents
carry more than one record, up to
5 each).

So the question this node answers is not "is the classifier accurate on the corpus it was
built on" but **"does it transfer to passages from documents it has never seen, and does a
record-level cross-validation split flatter it?"** Two failure modes are being tested, and
both are allowed to win:

* *document leak* — prose templates repeat inside a report, so a record-level split lets a
  fitted arm memorise the document rather than the mechanism;
* *wording leak* — a classifier keyed on the analyst's own cue words is graded on its own
  vocabulary.

### Held-out corpus (new, document-disjoint, pinned)

| property | value |
|---|---|
| records | 24 |
| source documents | 17 |
| max records per document | 3 |
| records per class | BASELINE_ABSENT=4, TOLERANCE_EXCEEDED=4, CONVERGENCE_STALL=4, MODEL_FIDELITY_LIMIT=4, DATA_INSUFFICIENCY=4, IMPLEMENTATION_DEFECT=4 |
| payload sha256 | 611ae758ababe5309ea52d9ae0fa0dc0… |
| document overlap with the frozen corpus | none (verified on every load) |

The corpus is built by `scripts/build_failure_taxonomy_holdout.py` from a curation sheet
that names, per record, one document, one marker substring occurring exactly once, and the
analyst-assigned class. The builder refuses (a) a document already used by the frozen set,
(b) a marker that is not unique, (c) a body sharing any 40-character shard with a frozen
body, and it re-verifies the class coverage. The digest pin makes later edits raise instead
of silently changing the numbers.

## 2. Method

1. **In-corpus recap** — rule annotator, unsupervised prototype annotator, and a supervised
   centroid arm fitted on the frozen corpus (48 records).
2. **Transfer** — the same annotators, *zero refit*, on the 24 held-out records from 17
   documents never used by the frozen corpus.
3. **Document leak** — the supervised arm under two splits of the frozen corpus:
   record-level 6-fold (the leaky protocol) and document-grouped 5-fold (no document on
   both sides), same pooled accuracy definition.
4. **Cue ablation** — leave-one-evidence-field-out: drop one cue family, rebuild the rule
   annotator, re-grade. A classification resting on a single token family collapses here.
5. **Negative controls carried to the held-out set** — same-length non-cue decoy bodies
   (10 draws) and permuted held-out labels (25 permutations).
6. **Selective rerouting (E7)** — the annotator's own score margin
   (top1 − top2) as an abstention signal. The margin floor is chosen **on the frozen
   corpus only** (smallest floor whose in-corpus risk ≤ 0.20
   at coverage ≥ 0.60), then applied unchanged to the
   held-out documents. An abstained node receives `ESCALATE` with reason code `low_margin`.

Pre-registered bars are recorded verbatim in the artefact under `preregistered_bars`.

## 3. Results

| criterion | measured | verdict |
|---|---|---|
| E1 transfer accuracy >= 0.70 | 0.625 | **FAIL** |
| E2 transfer kappa >= 0.60 | 0.550 | **FAIL** |
| E3 document-leak |gap| <= 0.10 | 0.188 | **FAIL** |
| E4 ablation min delta >= -0.25 | -0.125 | PASS |
| E5 reroute coverage >= 0.90 | 0.958 | PASS |
| E6 domain-purity hits == 0 | 0 | PASS |
| E7 selective gating | risk 0.375 -> 0.318 @ coverage 0.917 | PASS |

**Score: 4/7 bars pass. The three that fail are the three that decide whether the earlier
in-corpus numbers can be quoted as a capability.**

### 3.1 Transfer per class

| class | in-corpus recall | held-out recall | drop |
|---|---|---|---|
| BASELINE_ABSENT | 0.750 | 0.250 | +0.500 |
| TOLERANCE_EXCEEDED | 0.625 | 0.250 | +0.375 |
| CONVERGENCE_STALL | 1.000 | 1.000 | +0.000 |
| MODEL_FIDELITY_LIMIT | 0.625 | 0.750 | -0.125 |
| DATA_INSUFFICIENCY | 1.000 | 0.750 | +0.250 |
| IMPLEMENTATION_DEFECT | 0.750 | 0.750 | +0.000 |

Overall: accuracy **0.792 → 0.625**
(Δ +0.167), kappa **0.750 → 0.550**
(Δ +0.200), against a chance accuracy of 0.167. `CONVERGENCE_STALL` and
`IMPLEMENTATION_DEFECT` transfer without loss; the loss is concentrated in
`BASELINE_ABSENT` (+0.500) and
`TOLERANCE_EXCEEDED` (+0.375).

### 3.2 Where the errors go (held-out)

| gold -> predicted | count |
|---|---|
| BASELINE_ABSENT -> DATA_INSUFFICIENCY | 2 |
| BASELINE_ABSENT -> MODEL_FIDELITY_LIMIT | 1 |
| DATA_INSUFFICIENCY -> MODEL_FIDELITY_LIMIT | 1 |
| IMPLEMENTATION_DEFECT -> CONVERGENCE_STALL | 1 |
| MODEL_FIDELITY_LIMIT -> TOLERANCE_EXCEEDED | 1 |
| TOLERANCE_EXCEEDED -> BASELINE_ABSENT | 1 |
| TOLERANCE_EXCEEDED -> CONVERGENCE_STALL | 1 |
| TOLERANCE_EXCEEDED -> DATA_INSUFFICIENCY | 1 |

The dominant confusion is `BASELINE_ABSENT -> DATA_INSUFFICIENCY`, which is exactly the
boundary the class definitions warn about: *the deliverable exists and is quantified, only
the yardstick is missing* versus *nothing was delivered at all*. In unseen prose the
"no external reference is readable" wording is phrased with the same absence/withheld
vocabulary that the additive scorer reads as a missing artefact.

### 3.3 Document leak

* record-level 6-fold, supervised arm: **0.604**
* document-grouped 5-fold, same arm: **0.417**
* gap (grouped − record-level): **-0.188**

A record-level split overstates this arm by ~19
percentage points. The per-fold spread of the grouped arm is
0.111–0.600,
i.e. the estimate is also unstable, not just optimistic.

### 3.4 Negative controls (held-out)

| control | value |
|---|---|
| decoy bodies, mean accuracy | 0.217 |
| decoy bodies, max over draws | 0.333 |
| permuted labels, mean kappa | 0.010 |
| permuted labels, max kappa | 0.200 |

Both controls collapse on the held-out set too, so the transfer number above is a real
signal (not luck of a degenerate labeler): the classifier does read something in the prose.

### 3.5 Cue ablation (held-out)

| dropped cue field | lexemes | accuracy | delta |
|---|---|---|---|
| defect_lexeme | 14 | 0.500 | -0.125 |
| assumption_lexeme | 17 | 0.542 | -0.083 |
| convergence_lexeme | 18 | 0.542 | -0.083 |
| reference_lexeme | 18 | 0.583 | -0.042 |
| envelope_lexeme | 10 | 0.625 | +0.000 |
| threshold_lexeme | 13 | 0.625 | +0.000 |
| absence_lexeme | 17 | 0.667 | +0.042 |
| artifact_lexeme | 11 | 0.667 | +0.042 |
| withheld_lexeme | 8 | 0.667 | +0.042 |

No single family carries the corpus on its own (E4 passes): the worst single-field removal
costs 0.125. The evidence is
diffuse — which is also why the classifier degrades gracefully but transfers poorly: it
aggregates many weak cues, and the aggregate shifts when the wording of a different
document set shifts.

### 3.6 Selective rerouting

* floor chosen on the frozen corpus: **0.20**
* held-out ungated: coverage 1.000, risk 0.375
* held-out gated: coverage 0.917, risk 0.318
  (Δrisk +0.057)
* AURC: in-corpus 0.089 vs held-out 0.201

The gate moves in the right direction on unseen documents but only marginally, and the
calibration degrades by a factor 2.2. Treat the
margin as a weak abstention signal, not as a calibrated probability.

### 3.7 Reroute decision table on held-out records

| corpus | records | routed | escalated | coverage | actions |
|---|---|---|---|---|---|
| frozen | 48 | 45 | 3 | 0.938 | {"BASELINE_SWAP": 8, "METHOD_SWAP": 23, "SCOPE_REDUCTION": 14} |
| held-out | 24 | 23 | 1 | 0.958 | {"BASELINE_SWAP": 3, "METHOD_SWAP": 12, "SCOPE_REDUCTION": 8} |

Blocked-node view integration (`blocked_node_view(..., margin_floor=...)`) emits
`{failure_class, evidence, reroute_suggestion}` per node; with the frozen-chosen floor it
abstains on 2 of 24 held-out nodes
(`by_action = {"BASELINE_SWAP": 1, "METHOD_SWAP": 13, "SCOPE_REDUCTION": 8, "ESCALATE": 2}`).

## 4. Figures

![per-class transfer](fig1_transfer_recall.png)

*Fig. 1 — per-class recall, in-corpus vs document-disjoint held-out; chance line at 1/6.*

![risk-coverage](fig2_risk_coverage.png)

*Fig. 2 — risk-coverage of the margin gate. The floor is chosen on the blue curve and
applied to the red one.*

![leak arms](fig3_leak_arms.png)

*Fig. 3 — the five accuracy arms; the pair of grey bars is the document-leak measurement.*

## 5. What this changes operationally

1. **The in-corpus numbers are a ceiling, not an estimate.** Quote
   0.62 ± document-level uncertainty for unseen documents,
   not 0.79. The 0.70 bar of the parent node is met only on
   the corpus that the lexicon was curated against.
2. **`BASELINE_ABSENT` and `TOLERANCE_EXCEEDED` must not be auto-routed.** Their transfer
   recall is 0.25 and
   0.25; a wrong class
   produces a wrong *move* (baseline swap vs scope reduction), which is worse than an
   abstention. Routing the four stable classes and escalating these two keeps
   67% of the
   class space automated.
3. **A document-level split must be the default** for any supervised arm fitted on failure
   records in this repository; the record-level number is not conservative.

## 6. Limitations and honest statement

* The held-out corpus was annotated by a single analyst (this agent). The kappa in E2 is
  therefore an annotator-vs-classifier agreement, **not** a human-human kappa; no second
  human labeler was available in this environment. Label noise is a floor on the reported
  transfer accuracy, and it is not separable from classifier error with n = 24.
* n = 24 records, 4 per class. One record is 4.2 percentage points of accuracy; the
  per-class numbers are indicative, not precise. The corpus is pinned and reusable, so a
  larger version is an additive change, not a rewrite.
* The held-out documents are all markdown reports from this repository; ASR/ASCII-only
  reports and other repositories' prose are unrepresented, so nothing is claimed about
  them.
* The abstention gate was calibrated on the frozen corpus, and although its floor was not
  tuned on held-out data, the *decision to use a margin gate at all* was taken after
  observing the transfer failure. E7 is therefore a weaker claim than E1/E2.
* The error analysis suggests (but does not prove) that a boundary-aware adjudicator —
  testing "is a quantity delivered?" before "is a yardstick available?" — would recover part
  of the `BASELINE_ABSENT` loss. This is left as an explicitly *unvalidated* hypothesis; it
  was not fitted here precisely because fitting it on the only held-out set available would
  have consumed the test set.
* Domain purity is machine-checked (E6) over the package and the test file, but the data
  files necessarily carry domain wording; the separation is code-vs-data, not
  corpus-vs-everything.

## 7. Reproduction

```bash
PYTHONPATH=src python scripts/build_failure_taxonomy_holdout.py          # rebuild + pin the held-out corpus
PYTHONPATH=src python -m tensorphysics.failure_taxonomy.audit --out artifacts/failure_taxonomy_external/audit.json
PYTHONPATH=src python scripts/render_failure_taxonomy_external_audit.py  # this report + figures
PYTHONPATH=src python -m pytest tests/test_failure_taxonomy_criteria.py tests/test_failure_taxonomy_external_validity.py -q
```
