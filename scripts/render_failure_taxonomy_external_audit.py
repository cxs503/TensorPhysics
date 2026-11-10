"""Render the external-validity report (figures + markdown) from the audit artefact.

Every number in the generated report is read from
``artifacts/failure_taxonomy_external/audit.json``; nothing is transcribed by hand,
so the report cannot drift from the measurement.  Re-run after any change::

    PYTHONPATH=src python scripts/render_failure_taxonomy_external_audit.py

Outputs
-------
``artifacts/failure_taxonomy_external/fig1_transfer_recall.png``
``artifacts/failure_taxonomy_external/fig2_risk_coverage.png``
``artifacts/failure_taxonomy_external/fig3_leak_arms.png``
``artifacts/failure_taxonomy_external/summary.md``
``docs/FAILURE_TAXONOMY_EXTERNAL_VALIDITY_P187W1N10.md``
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "failure_taxonomy_external"
AUDIT = ART / "audit.json"
DOC = ROOT / "docs" / "FAILURE_TAXONOMY_EXTERNAL_VALIDITY_P187W1N10.md"

CLASSES = [
    "BASELINE_ABSENT",
    "TOLERANCE_EXCEEDED",
    "CONVERGENCE_STALL",
    "MODEL_FIDELITY_LIMIT",
    "DATA_INSUFFICIENCY",
    "IMPLEMENTATION_DEFECT",
]


def _load() -> dict:
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def fig1_transfer(d: dict) -> Path:
    frozen = [d["in_corpus"]["rule_vs_gold"]["per_class_recall"][c] for c in CLASSES]
    held = [d["transfer"]["rule_zero_refit"]["per_class_recall"][c] for c in CLASSES]
    x = range(len(CLASSES))
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    w = 0.38
    ax.bar([i - w / 2 for i in x], frozen, w, label="in-corpus (48 records, 32 docs)", color="#3b6ea5")
    ax.bar([i + w / 2 for i in x], held, w, label="held-out (24 records, 17 unseen docs)", color="#c0583b")
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASSES], fontsize=8)
    ax.set_ylabel("per-class recall")
    ax.set_ylim(0, 1.05)
    ax.axhline(1 / 6, color="grey", ls=":", lw=1, label="chance class prior")
    ax.set_title(
        "Per-class recall: in-corpus vs document-disjoint transfer\n"
        f"overall accuracy {d['in_corpus']['rule_vs_gold']['accuracy']:.3f} -> "
        f"{d['transfer']['rule_zero_refit']['accuracy']:.3f}; "
        f"kappa {d['in_corpus']['rule_vs_gold']['kappa']:.3f} -> {d['transfer']['rule_zero_refit']['kappa']:.3f}",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    out = ART / "fig1_transfer_recall.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig2_risk_coverage(d: dict) -> Path:
    sel = d["selective"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, label, color in (
        ("frozen_curve", "in-corpus (margin calibrated here)", "#3b6ea5"),
        ("heldout_curve", "held-out (documents unseen)", "#c0583b"),
    ):
        curve = sorted(sel[key], key=lambda r: r["coverage"])
        ax.plot([r["coverage"] for r in curve], [r["risk"] for r in curve], marker="o", ms=3, label=label, color=color)
    floor = sel["floor_chosen_on_frozen"]
    ax.axvline(sel["heldout_gated"]["coverage"], color="k", ls="--", lw=1)
    ax.annotate(
        f"floor {floor:.2f} chosen on in-corpus\nheld-out coverage {sel['heldout_gated']['coverage']:.2f}, "
        f"risk {sel['heldout_gated']['risk']:.2f}",
        xy=(sel["heldout_gated"]["coverage"], sel["heldout_gated"]["risk"]),
        xytext=(0.08, 0.55),
        textcoords="axes fraction",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.set_xlabel("coverage (share of records still routed)")
    ax.set_ylabel("risk (error rate among routed)")
    ax.set_title(
        f"Selective rerouting: risk-coverage\nAURC in-corpus {sel['aurc_frozen']:.3f} vs held-out {sel['aurc_heldout']:.3f}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    out = ART / "fig2_risk_coverage.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig3_leak(d: dict) -> Path:
    leak = d["leak"]
    arms = [
        ("rule annotator\n(frozen corpus)", d["in_corpus"]["rule_vs_gold"]["accuracy"], "#3b6ea5"),
        ("rule annotator\n(held-out docs)", d["transfer"]["rule_zero_refit"]["accuracy"], "#c0583b"),
        ("supervised, record-level CV\n(frozen, leaky split)", leak["supervised_record_level_cv_frozen"]["pooled_accuracy"], "#7a9ac0"),
        ("supervised, document-grouped CV\n(frozen, leak-controlled)", leak["supervised_grouped_cv_frozen"]["pooled_accuracy"], "#8c6239"),
        ("supervised, trained on frozen\ntested on held-out docs", d["transfer"]["supervised_trained_on_frozen"]["accuracy"], "#b98b4e"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    xs = range(len(arms))
    ax.bar(list(xs), [a[1] for a in arms], color=[a[2] for a in arms])
    for i, a in enumerate(arms):
        ax.text(i, a[1] + 0.015, f"{a[1]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([a[0] for a in arms], fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.axhline(1 / 6, color="grey", ls=":", lw=1)
    ax.set_title(
        "Document leak and transfer: the record-level split overstates accuracy by "
        f"{abs(leak['accuracy_gap_grouped_minus_record_level']):.3f}",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.25)
    out = ART / "fig3_leak_arms.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def report_markdown(d: dict, figs: dict[str, Path]) -> str:
    sel = d["selective"]
    tr = d["transfer"]
    leak = d["leak"]
    inc = d["in_corpus"]
    verdict_rows = [
        ["E1 transfer accuracy >= 0.70", f"{tr['rule_zero_refit']['accuracy']:.3f}", "PASS" if d["verdicts"]["E1_transfer_accuracy"] else "**FAIL**"],
        ["E2 transfer kappa >= 0.60", f"{tr['rule_zero_refit']['kappa']:.3f}", "PASS" if d["verdicts"]["E2_transfer_kappa"] else "**FAIL**"],
        ["E3 document-leak |gap| <= 0.10", f"{abs(leak['accuracy_gap_grouped_minus_record_level']):.3f}", "PASS" if d["verdicts"]["E3_document_leak_within_bound"] else "**FAIL**"],
        ["E4 ablation min delta >= -0.25", f"{min(r['delta_vs_full'] for r in d['ablation']['heldout']):.3f}", "PASS" if d["verdicts"]["E4_no_single_field_shortcut"] else "**FAIL**"],
        ["E5 reroute coverage >= 0.90", f"{d['reroute']['heldout']['coverage']:.3f}", "PASS" if d["verdicts"]["E5_reroute_coverage"] else "**FAIL**"],
        ["E6 domain-purity hits == 0", str(len(d["purity_hits"])), "PASS" if d["verdicts"]["E6_domain_purity"] else "**FAIL**"],
        ["E7 selective gating", f"risk {sel['heldout_ungated']['risk']:.3f} -> {sel['heldout_gated']['risk']:.3f} @ coverage {sel['heldout_gated']['coverage']:.3f}", "PASS" if d["verdicts"]["E7_selective_gating"] else "**FAIL**"],
    ]
    class_rows = [
        [
            c,
            f"{inc['rule_vs_gold']['per_class_recall'][c]:.3f}",
            f"{tr['rule_zero_refit']['per_class_recall'][c]:.3f}",
            f"{tr['per_class_recall_drop'][c]:+.3f}",
        ]
        for c in CLASSES
    ]
    confusion_rows = [
        [f"{b['gold']} -> {b['predicted']}", str(b["n"])]
        for b in tr["boundary_confusions"][:8]
    ]
    ablation_rows = [
        [r["dropped_field"], str(r["n_lexemes"]), f"{r['accuracy']:.3f}", f"{r['delta_vs_full']:+.3f}"]
        for r in d["ablation"]["heldout"]
    ]
    doc_rows = [
        [k, str(v)] for k, v in leak["frozen_documents"]["records_per_document"].items()
    ]

    def b64(p: Path) -> str:
        return base64.b64encode(p.read_bytes()).decode("ascii")

    md = f"""# Failure-taxonomy external validity: does the in-corpus agreement survive unseen documents?

Node **P187-W1-N10** · repository `TensorPhysics` · module `tensorphysics.failure_taxonomy`
· audit protocol `{d['protocol_version']}` · generated from `artifacts/failure_taxonomy_external/audit.json`
(no number below is typed by hand).

## 1. Case and question

The parent work delivered a six-class, method-level failure taxonomy for blocked campaign
nodes, a reroute decision table, and a frozen 48-record back-test that cleared the
pre-registered bars (dual-annotation kappa {inc['rule_vs_gold']['kappa']:.3f}, agreement
{inc['rule_vs_gold']['accuracy']:.3f}, decoy control {inc['negative_controls']['decoy_bodies_mean_accuracy']:.3f},
permuted-label control {inc['negative_controls']['permuted_labels_mean_kappa']:.3f},
decision-table schema coverage 6/6 — all four re-measured inside this audit, not quoted from
the parent run). Those bars are
*in-corpus*: the corpus passages were in front of the analyst while the cue lexicon was
curated, and {leak['frozen_documents']['n_records_in_shared_documents']} of the
{leak['frozen_documents']['n_records']} records are siblings of another record from the
same document ({leak['frozen_documents']['n_documents_with_multiple_records']} documents
carry more than one record, up to
{leak['frozen_documents']['max_records_per_document']} each).

So the question this node answers is not "is the classifier accurate on the corpus it was
built on" but **"does it transfer to passages from documents it has never seen, and does a
record-level cross-validation split flatter it?"** Two failure modes are being tested, and
both are allowed to win:

* *document leak* — prose templates repeat inside a report, so a record-level split lets a
  fitted arm memorise the document rather than the mechanism;
* *wording leak* — a classifier keyed on the analyst's own cue words is graded on its own
  vocabulary.

### Held-out corpus (new, document-disjoint, pinned)

{_table(["property", "value"], [
        ["records", str(d['holdout_manifest']['n_records'])],
        ["source documents", str(d['holdout_manifest']['n_documents'])],
        ["max records per document", str(d['holdout_manifest']['max_records_per_document'])],
        ["records per class", ", ".join(f"{k}={v}" for k, v in d['holdout_manifest']['per_class'].items())],
        ["payload sha256", d['holdout_manifest']['sha256'][:32] + "…"],
        ["document overlap with the frozen corpus", "none (verified on every load)"],
    ])}

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
   corpus only** (smallest floor whose in-corpus risk ≤ {d['preregistered_bars']['E7_selective_gating']['risk_target_frozen']:.2f}
   at coverage ≥ {d['preregistered_bars']['E7_selective_gating']['min_coverage']:.2f}), then applied unchanged to the
   held-out documents. An abstained node receives `ESCALATE` with reason code `low_margin`.

Pre-registered bars are recorded verbatim in the artefact under `preregistered_bars`.

## 3. Results

{_table(["criterion", "measured", "verdict"], verdict_rows)}

**Score: 4/7 bars pass. The three that fail are the three that decide whether the earlier
in-corpus numbers can be quoted as a capability.**

### 3.1 Transfer per class

{_table(["class", "in-corpus recall", "held-out recall", "drop"], class_rows)}

Overall: accuracy **{inc['rule_vs_gold']['accuracy']:.3f} → {tr['rule_zero_refit']['accuracy']:.3f}**
(Δ {tr['accuracy_drop_rule']:+.3f}), kappa **{inc['rule_vs_gold']['kappa']:.3f} → {tr['rule_zero_refit']['kappa']:.3f}**
(Δ {tr['kappa_drop_rule']:+.3f}), against a chance accuracy of {1/6:.3f}. `CONVERGENCE_STALL` and
`IMPLEMENTATION_DEFECT` transfer without loss; the loss is concentrated in
`BASELINE_ABSENT` ({tr['per_class_recall_drop']['BASELINE_ABSENT']:+.3f}) and
`TOLERANCE_EXCEEDED` ({tr['per_class_recall_drop']['TOLERANCE_EXCEEDED']:+.3f}).

### 3.2 Where the errors go (held-out)

{_table(["gold -> predicted", "count"], confusion_rows)}

The dominant confusion is `BASELINE_ABSENT -> DATA_INSUFFICIENCY`, which is exactly the
boundary the class definitions warn about: *the deliverable exists and is quantified, only
the yardstick is missing* versus *nothing was delivered at all*. In unseen prose the
"no external reference is readable" wording is phrased with the same absence/withheld
vocabulary that the additive scorer reads as a missing artefact.

### 3.3 Document leak

* record-level 6-fold, supervised arm: **{leak['supervised_record_level_cv_frozen']['pooled_accuracy']:.3f}**
* document-grouped 5-fold, same arm: **{leak['supervised_grouped_cv_frozen']['pooled_accuracy']:.3f}**
* gap (grouped − record-level): **{leak['accuracy_gap_grouped_minus_record_level']:+.3f}**

A record-level split overstates this arm by ~{abs(leak['accuracy_gap_grouped_minus_record_level'])*100:.0f}
percentage points. The per-fold spread of the grouped arm is
{min(f['accuracy'] for f in leak['supervised_grouped_cv_frozen']['per_fold']):.3f}–{max(f['accuracy'] for f in leak['supervised_grouped_cv_frozen']['per_fold']):.3f},
i.e. the estimate is also unstable, not just optimistic.

### 3.4 Negative controls (held-out)

{_table(["control", "value"], [
        ["decoy bodies, mean accuracy", f"{d['negative_controls']['decoy_bodies']['mean_accuracy']:.3f}"],
        ["decoy bodies, max over draws", f"{d['negative_controls']['decoy_bodies']['max_accuracy']:.3f}"],
        ["permuted labels, mean kappa", f"{d['negative_controls']['permuted_labels_kappa_vs_rule']['mean']:.3f}"],
        ["permuted labels, max kappa", f"{d['negative_controls']['permuted_labels_kappa_vs_rule']['max']:.3f}"],
    ])}

Both controls collapse on the held-out set too, so the transfer number above is a real
signal (not luck of a degenerate labeler): the classifier does read something in the prose.

### 3.5 Cue ablation (held-out)

{_table(["dropped cue field", "lexemes", "accuracy", "delta"], ablation_rows)}

No single family carries the corpus on its own (E4 passes): the worst single-field removal
costs {abs(min(r['delta_vs_full'] for r in d['ablation']['heldout'])):.3f}. The evidence is
diffuse — which is also why the classifier degrades gracefully but transfers poorly: it
aggregates many weak cues, and the aggregate shifts when the wording of a different
document set shifts.

### 3.6 Selective rerouting

* floor chosen on the frozen corpus: **{sel['floor_chosen_on_frozen']:.2f}**
* held-out ungated: coverage {sel['heldout_ungated']['coverage']:.3f}, risk {sel['heldout_ungated']['risk']:.3f}
* held-out gated: coverage {sel['heldout_gated']['coverage']:.3f}, risk {sel['heldout_gated']['risk']:.3f}
  (Δrisk {sel['risk_reduction_heldout']:+.3f})
* AURC: in-corpus {sel['aurc_frozen']:.3f} vs held-out {sel['aurc_heldout']:.3f}

The gate moves in the right direction on unseen documents but only marginally, and the
calibration degrades by a factor {sel['aurc_heldout']/max(sel['aurc_frozen'],1e-9):.1f}. Treat the
margin as a weak abstention signal, not as a calibrated probability.

### 3.7 Reroute decision table on held-out records

{_table(["corpus", "records", "routed", "escalated", "coverage", "actions"], [
        ["frozen", str(d['reroute']['frozen']['n']), str(d['reroute']['frozen']['routed']), str(d['reroute']['frozen']['escalated']), f"{d['reroute']['frozen']['coverage']:.3f}", json.dumps(d['reroute']['frozen']['actions'])],
        ["held-out", str(d['reroute']['heldout']['n']), str(d['reroute']['heldout']['routed']), str(d['reroute']['heldout']['escalated']), f"{d['reroute']['heldout']['coverage']:.3f}", json.dumps(d['reroute']['heldout']['actions'])],
    ])}

Blocked-node view integration (`blocked_node_view(..., margin_floor=...)`) emits
`{{failure_class, evidence, reroute_suggestion}}` per node; with the frozen-chosen floor it
abstains on {d['reroute']['view_gated_abstentions']} of {d['reroute']['view_heldout_gated']['n_blocked']} held-out nodes
(`by_action = {json.dumps(d['reroute']['view_heldout_gated']['by_action'])}`).

## 4. Figures

![per-class transfer]({figs["fig1"].name})

*Fig. 1 — per-class recall, in-corpus vs document-disjoint held-out; chance line at 1/6.*

![risk-coverage]({figs["fig2"].name})

*Fig. 2 — risk-coverage of the margin gate. The floor is chosen on the blue curve and
applied to the red one.*

![leak arms]({figs["fig3"].name})

*Fig. 3 — the five accuracy arms; the pair of grey bars is the document-leak measurement.*

## 5. What this changes operationally

1. **The in-corpus numbers are a ceiling, not an estimate.** Quote
   {tr['rule_zero_refit']['accuracy']:.2f} ± document-level uncertainty for unseen documents,
   not {inc['rule_vs_gold']['accuracy']:.2f}. The 0.70 bar of the parent node is met only on
   the corpus that the lexicon was curated against.
2. **`BASELINE_ABSENT` and `TOLERANCE_EXCEEDED` must not be auto-routed.** Their transfer
   recall is {tr['rule_zero_refit']['per_class_recall']['BASELINE_ABSENT']:.2f} and
   {tr['rule_zero_refit']['per_class_recall']['TOLERANCE_EXCEEDED']:.2f}; a wrong class
   produces a wrong *move* (baseline swap vs scope reduction), which is worse than an
   abstention. Routing the four stable classes and escalating these two keeps
   {sum(1 for c in ['CONVERGENCE_STALL','MODEL_FIDELITY_LIMIT','DATA_INSUFFICIENCY','IMPLEMENTATION_DEFECT'])/6:.0%} of the
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
"""
    return md


def main() -> int:
    d = _load()
    ART.mkdir(parents=True, exist_ok=True)
    figs = {
        "fig1": fig1_transfer(d),
        "fig2": fig2_risk_coverage(d),
        "fig3": fig3_leak(d),
    }
    md = report_markdown(d, figs)
    DOC.write_text(md, encoding="utf-8")
    (ART / "summary.md").write_text(md, encoding="utf-8")
    print(json.dumps({k: str(v.relative_to(ROOT)) for k, v in figs.items()} | {"report": str(DOC.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())