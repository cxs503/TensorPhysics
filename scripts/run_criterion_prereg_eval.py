"""Acceptance evaluation for the criterion pre-registration gate.

Deliverable of mission 8598 (port of the closed PR #164 / mission 7803 node
``p92w2b-n07``); the node identifiers below are the current ones.

Produces, programmatically (no hand-copied numbers):

* ``artifacts/criterion_prereg/eval.json``   -- every acceptance number,
  per-case predictions and the fail-closed negative controls;
* ``artifacts/criterion_prereg/fig_*.png``   -- confusion matrix, baseline vs
  audit precision/recall, extractor field F1, KPI of the real-goal demo;
* the freeze/audit records of this node's own acceptance criteria (the "real
  goal" case) travel *inside* ``eval.json`` (``detail.real_goal_demo``) so the
  record stays a single, digest-verifiable file instead of a set of loose dumps.

and re-renders ``docs/CRITERION_PREREGISTRATION.md`` from that JSON (so no
number in the document is hand-copied, and the report/doc pair can never drift
apart), and emits the mission's own acceptance record into
``docs/m8598/outputs/acceptance_criteria.json`` from the same payload.

Run: ``python scripts/run_criterion_prereg_eval.py`` (writes into the repo);
``--out-dir`` / ``--doc`` redirect both outputs so CI can re-derive the record
into a scratch directory and byte-compare it with the checked-in copy -- the
eval record is code output, not a hand-saved snapshot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorphysics.campaign.criterion_prereg import (  # noqa: E402
    audit,
    attach_to_metadata,
    extract_criteria,
    freeze,
    kpi_view,
    semantic_digest,
    sha256_obj,
    verify_manifest,
)
from tensorphysics.campaign.criterion_prereg.scan import scan_paths  # noqa: E402
from tensorphysics.campaign.criterion_prereg.synth import (  # noqa: E402
    baseline_text_hash,
    detection_metrics,
    field_scores,
    gen_drift_cases,
    gen_extraction_corpus,
)

OUT = ROOT / "artifacts" / "criterion_prereg"
EVAL = OUT / "eval.json"
DOC = ROOT / "docs" / "CRITERION_PREREGISTRATION.md"
DELIVERABLE = ROOT / "docs" / "m8598" / "outputs" / "acceptance_criteria.json"
STAMP = "2026-11-09T00:00:00+00:00"

#: The acceptance criteria of this node, verbatim from the campaign DAG node.
NODE_GOAL = (
    "机器验收判据：(a) 合成检测集24例recall≥0.92、precision≥0.95；"
    "(b) 抽取器字段级F1≥0.95；"
    "(c) fail-closed负控：空goal→unregistered，篡改sha256→tampered；"
    "(d) 零领域语义扫描命中=0。"
)

#: Held-out surface perturbations (out of the generator's default grammar) used
#: to characterise -- not to tune -- the extractor.  Each entry is
#: (case_id, text, expected_fields_for_first_criterion).
HARD_CASES = [
    (
        "hard-fullwidth",
        "机器验收判据：（a）合成检测集24例 recall ＞＝ 0.92；（B）precision ＜ 0.5。",
        {"metric": "recall", "op": ">=", "value": "0.92", "count": 24, "scope": "合成检测集", "label": "a"},
    ),
    (
        "hard-percent-unit",
        "机器验收判据：(a) 全量用例48cases f1 ≥ 95%。",
        {"metric": "f1", "op": ">=", "value": "95", "unit": "%", "count": 48, "scope": "全量用例"},
    ),
]

#: Probes that are *expected to fail*: they document the known limitation
#: (undelimited Chinese metric names shorter than 4 characters cannot be
#: boundary-detected without a lexicon).  Recorded, never scored as acceptance.
LIMITATION_PROBES = [
    (
        "limit-undelimited-cjk-2char",
        "机器验收判据：(a) 零领域样本命中=0。",
        {"metric": "命中", "op": "==", "value": "0", "scope": "零领域样本"},
    ),
]


def _drift_run(seed: int) -> tuple[dict, list[dict], list[dict]]:
    cases = gen_drift_cases(seed=seed)
    rows, base_rows = [], []
    for case in cases:
        fz = freeze(case.frozen_goal, frozen_at=STAMP)
        report = audit(fz.manifest, receipt_goal=case.receipt_goal)
        rows.append((report.failed(), report.dominant_class))
        base = baseline_text_hash(case.frozen_goal, case.receipt_goal)
        base_rows.append((base, "drift" if base else "unchanged"))
        case.predicted = report.dominant_class
        case.predicted_drift = report.failed()
    metrics = detection_metrics(rows, cases)
    base_metrics = detection_metrics(base_rows, cases)
    per_case = [
        {
            "case_id": c.case_id,
            "mutation": c.mutation,
            "expected": c.expected_class,
            "predicted": c.predicted,
            "is_drift": c.is_drift,
            "flagged": c.predicted_drift,
        }
        for c in cases
    ]
    return metrics, base_metrics, per_case


def _hard_case_scores(cases=HARD_CASES) -> list[dict]:
    rows = []
    for case_id, text, expected in cases:
        crit = extract_criteria(text).criteria
        got = crit[0].to_dict() if crit else {}
        matched = {k: (got.get(k, "") == v) for k, v in expected.items()}
        rows.append(
            {
                "case_id": case_id,
                "text": text,
                "expected": expected,
                "extracted": {k: got.get(k) for k in expected},
                "matched": matched,
                "all_matched": all(matched.values()),
            }
        )
    return rows


def _demo_real_goal() -> dict:
    """Freeze this node's own criteria, then audit five receipts.

    Four are behavioural re-statements of the same goal (reformat / loosened
    threshold / narrowed scope / dropped criterion) and one is a manifest-level
    tamper; the empty goal is audited separately as the fail-closed control.
    """
    fz = freeze(NODE_GOAL, mission_id=8598, node_id="criterion-preregistration-gate", frozen_by="m8598", frozen_at=STAMP)
    metadata = attach_to_metadata({"mission_id": 8598, "owner": "autonomy"}, fz)
    receipts = {
        "unchanged_reformatted": NODE_GOAL.replace("≥", ">=").replace("、", "；"),
        "loosened_recall": NODE_GOAL.replace("recall≥0.92", "recall≥0.80"),
        "narrowed_scope": NODE_GOAL.replace("24例", "12例"),
        "dropped_zero_domain": NODE_GOAL.replace("；(d) 零领域语义扫描命中=0", ""),
        "tampered_manifest": None,
    }
    reports, events = {}, []
    for name, goal in receipts.items():
        if name == "tampered_manifest":
            tampered = json.loads(json.dumps(fz.manifest))
            tampered["criteria"][0]["value"] = "0.80"
            report = audit(tampered, receipt_goal=NODE_GOAL, audited_at=STAMP)
        else:
            report = audit(fz.manifest, receipt_goal=goal, audited_at=STAMP)
        reports[name] = report.to_dict()
        events.extend(report.events())
    empty = freeze("", mission_id=8598, node_id="criterion-preregistration-gate", frozen_at=STAMP)
    # records travel inside eval.json (detail.real_goal_demo) -- no free-floating dumps
    return {
        "frozen_sha256": fz.sha256,
        "semantic_digest": semantic_digest(fz.manifest),
        "n_criteria": fz.manifest["n_criteria"],
        "mission_metadata": metadata["criterion_manifest"],
        "frozen_manifest": fz.manifest,
        "frozen_of": {"mission_id": 8598, "node_id": "criterion-preregistration-gate", "frozen_by": "m8598", "frozen_at": STAMP},
        "criteria": fz.manifest["criteria"],
        "verify_state": verify_manifest(fz.manifest, goal=NODE_GOAL),
        "empty_goal_state": empty.state,
        "empty_goal_sha": empty.sha256,
        "receipts": {k: {"state": v["state"], "verdict": v["verdict"], "dominant_class": v["dominant_class"], "counts": v["counts"], "notes": v["notes"]} for k, v in reports.items()},
        "events": events,
        "n_events": len(events),
        "error_events": sum(1 for e in events if e["severity"] == "error"),
        "kpi": kpi_view(list(reports.values())),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_confusion(metrics: dict, path: Path) -> Path:
    classes = ["unchanged", "tightened", "loosened", "scope_changed", "undefined"]
    mat = np.zeros((len(classes), len(classes)), dtype=float)
    for i, expected in enumerate(classes):
        row = metrics["confusion"].get(expected, {})
        for j, predicted in enumerate(classes):
            mat[i, j] = row.get(predicted, 0)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(1.0, mat.max()))
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("generator truth")
    ax.set_title(f"drift classification, 24 synthetic cases\nrecall={metrics['recall']:.3f} precision={metrics['precision']:.3f}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if mat[i, j]:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_baseline(drift: dict, base: dict, path: Path) -> Path:
    labels = ["precision", "recall", "f1"]
    x = np.arange(len(labels))
    audit_vals = [drift[k] for k in labels]
    base_vals = [base[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.bar(x - 0.2, audit_vals, 0.4, label="structured audit", color="#2c7fb8")
    ax.bar(x + 0.2, base_vals, 0.4, label="baseline: goal-text digest", color="#bdbdbd")
    for xi, v in zip(x - 0.2, audit_vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.2, base_vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    ax.axhline(0.95, ls="--", lw=1, color="#d62728")
    ax.text(2.35, 0.96, "criterion 0.95", color="#d62728", fontsize=8, ha="right")
    ax.set_title("drift detection: structured audit vs text-digest baseline")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def field_agreement(cases) -> dict[str, float]:
    """Per-field agreement of the extractor (correct / total field values)."""
    from tensorphysics.campaign.criterion_prereg.synth import iter_field_triples

    fields = ("label", "kind", "metric", "op", "value", "unit", "count", "scope")
    per_field = {name: [0, 0] for name in fields}
    for case in cases:
        gold = {t for t in iter_field_triples(case.gold, fields)}
        pred = {t for t in iter_field_triples(extract_criteria(case.text).criteria, fields)}
        for idx, name, value in gold:
            per_field[name][1] += 1
            if (idx, name, value) in pred:
                per_field[name][0] += 1
    return {n: (per_field[n][0] / per_field[n][1] if per_field[n][1] else 0.0) for n in fields}


def fig_field_f1(field: dict, path: Path) -> Path:
    """Per-field agreement of the extractor (from the same corpus run)."""
    from tensorphysics.campaign.criterion_prereg.synth import iter_field_triples

    cases = gen_extraction_corpus(20, seed=20261109)
    fields = ("label", "kind", "metric", "op", "value", "unit", "count", "scope")
    per_field = {name: [0, 0] for name in fields}  # [correct, total]
    for case in cases:
        gold = {t for t in iter_field_triples(case.gold, fields)}
        pred = {t for t in iter_field_triples(extract_criteria(case.text).criteria, fields)}
        for idx, name, value in gold:
            per_field[name][1] += 1
            if (idx, name, value) in pred:
                per_field[name][0] += 1
    names = list(fields)
    acc = [per_field[n][0] / per_field[n][1] if per_field[n][1] else 0.0 for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(names, acc, color=["#31a354" if a >= 0.95 else "#de2d26" for a in acc])
    for i, a in enumerate(acc):
        ax.text(i, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("field agreement")
    ax.set_title(f"extractor field-level agreement (F1={field['f1']:.3f}, n={field['tp'] + field['fn']} fields)")
    ax.axhline(0.95, ls="--", lw=1, color="#d62728")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_demo_kpi(demo: dict, drift: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax = axes[0]
    classes = ["unchanged", "tightened", "loosened", "scope_changed", "undefined"]
    counts = [drift["confusion"].get(c, {}).get(c, 0) for c in classes]
    flags = ["#31a354" if c in ("unchanged", "tightened") else "#de2d26" for c in classes]
    ax.bar(classes, counts, color=flags)
    for i, v in enumerate(counts):
        ax.text(i, v + 0.05, str(v), ha="center", fontsize=9)
    ax.set_title("24-case detection set: cases per true class")
    ax.set_ylabel("cases")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    kpi = demo["kpi"]
    names = ["prereg_coverage", "audit_pass_rate", "drift_rate", "fail_closed_rate"]
    vals = [kpi[n] for n in names]
    ax.bar(names, vals, color="#6baed6")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("real-goal demo KPI (5 receipt scenarios)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path



# --------------------------------------------------------------------------- #
# documentation -- rendered from eval.json by a standalone, dependency-free
# renderer (`scripts/render_criterion_prereg_doc.py`), so the page can be
# re-rendered/verified without re-running the corpora (`--check` guard).
# --------------------------------------------------------------------------- #
sys.path.insert(0, str(ROOT / "scripts"))
from render_criterion_prereg_doc import render_doc  # noqa: E402


def _artifact_label(path: Path) -> str:
    """Repo-relative label of an artifact, independent of ``--out-dir``."""
    return str(Path("artifacts") / "criterion_prereg" / path.name)


#: Key holding the deliverable's self-digest (never part of its own digest).
DIGEST_KEY = "payload_sha256"


def deliverable_digest(record: dict) -> str:
    """Digest of every field of ``record`` except the digest itself.

    Canonical JSON (sorted keys, no whitespace) so the digest survives
    re-indentation, but *any* change to an id, a measured value, a threshold or
    a ``pass`` flag flips it.
    """
    body = {k: v for k, v in record.items() if k != DIGEST_KEY}
    return sha256_obj(body)


def verify_deliverable(record: dict) -> tuple[bool, str]:
    """Offline self-check of a sealed deliverable record.

    Returns ``(ok, reason)``.  This needs no artefact cache and no re-run of the
    corpora: a stale or hand-edited copy (the failure mode that once shipped a
    *different mission's* record into ``docs/m8598/outputs/``) refutes itself.
    """
    declared = record.get(DIGEST_KEY)
    if not isinstance(declared, str) or not declared:
        return False, f"missing {DIGEST_KEY} (unsealed record)"
    actual = deliverable_digest(record)
    if actual != declared:
        return False, f"{DIGEST_KEY} mismatch: declared {declared[:12]}, content {actual[:12]}"
    return True, "ok"


def deliverable_record(payload: dict, eval_label: str) -> dict:
    """The mission's own acceptance record (``docs/m8598/outputs/...``).

    Generated from the evaluation payload, never typed: the criteria and their
    measured values exist exactly once, in ``eval.json``.  The file used to be a
    copy of another mission's record (it carried shaft-torsion metrics and a
    ``pytest_test_shaft_torsion`` flag this node never runs), i.e. the very
    "second source" this gate exists to prevent.

    The record is sealed with ``payload_sha256`` over its own canonical body so
    the checked-in copy can be validated in any checkout -- including one where
    the artefact cache was stripped and the corpora are not re-run.
    """
    record = {
        "schema": "criterion-prereg-deliverable/1",
        "mission_id": payload["mission_id"],
        "node_id": payload["node"],
        "generated_by": "scripts/run_criterion_prereg_eval.py",
        "source": eval_label,
        "page": "docs/CRITERION_PREREGISTRATION.md",
        "all_acceptance_pass": payload["all_acceptance_pass"],
        "acceptance": payload["acceptance"],
    }
    record[DIGEST_KEY] = deliverable_digest(record)
    return record


def run(out_dir: Path = OUT, doc_path: Path = DOC, eval_path: Path | None = None,
        deliverable_path: Path | None = None, verbose: bool = True) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = Path(eval_path) if eval_path is not None else out_dir / "eval.json"
    deliverable_path = Path(deliverable_path) if deliverable_path is not None else DELIVERABLE
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)

    field = field_scores(gen_extraction_corpus(20, seed=20261109))
    heldout = field_scores(gen_extraction_corpus(20, seed=42))
    ood = field_scores(gen_extraction_corpus(20, seed=20261109, undelimited_cjk=True))
    hard = _hard_case_scores()
    limits = _hard_case_scores(LIMITATION_PROBES)

    seeds = [20261109, 7, 42, 999, 1]
    per_seed = {}
    for seed in seeds:
        metrics, base, per_case = _drift_run(seed)
        per_seed[str(seed)] = {"metrics": metrics, "baseline": base, "per_case": per_case}
    main_seed = per_seed[str(seeds[0])]
    demo = _demo_real_goal()
    scan = scan_paths([ROOT / "src" / "tensorphysics" / "campaign" / "criterion_prereg"])

    figs = {
        "confusion": fig_confusion(main_seed["metrics"], out_dir / "fig_confusion_matrix.png"),
        "baseline": fig_baseline(main_seed["metrics"], main_seed["baseline"], out_dir / "fig_baseline_vs_audit.png"),
        "field_f1": fig_field_f1(field, out_dir / "fig_extractor_field_agreement.png"),
        "demo_kpi": fig_demo_kpi(demo, main_seed["metrics"], out_dir / "fig_demo_kpi.png"),
    }

    all_seeds_ok = all(v["metrics"]["recall"] >= 0.92 and v["metrics"]["precision"] >= 0.95 for v in per_seed.values())
    payload = {
        "node": "criterion-preregistration-gate",
        "lineage": {"origin_mission": 7803, "origin_node": "p92w2b-n07-criterion-preregistration", "reopened_as_mission": 8598},
        "mission_id": 8598,
        "schema": "criterion-prereg-eval/1",
        "acceptance": {
            "a_detection_24cases": {
                "recall": main_seed["metrics"]["recall"],
                "precision": main_seed["metrics"]["precision"],
                "thresholds": {"recall": 0.92, "precision": 0.95},
                "pass": main_seed["metrics"]["recall"] >= 0.92 and main_seed["metrics"]["precision"] >= 0.95,
                "class_accuracy": main_seed["metrics"]["class_accuracy"],
                "n_cases": main_seed["metrics"]["n_cases"],
                "n_positive": main_seed["metrics"]["n_positive"],
                "across_seeds_pass": all_seeds_ok,
            },
            "b_extractor_field_f1": {
                "f1": field["f1"],
                "precision": field["precision"],
                "recall": field["recall"],
                "thresholds": {"f1": 0.95},
                "pass": field["f1"] >= 0.95,
                "heldout_seed_f1": heldout["f1"],
                "heldout_pass": heldout["f1"] >= 0.95,
                "ood_undelimited_cjk_f1": ood["f1"],
                "hard_case_matched": sum(1 for h in hard if h["all_matched"]),
                "hard_case_total": len(hard),
                "limitation_probes_matched": sum(1 for h in limits if h["all_matched"]),
                "limitation_probes_total": len(limits),
            },
            "c_fail_closed": {
                "empty_goal_state": demo["empty_goal_state"],
                "empty_goal_sha_is_empty": demo["empty_goal_sha"] == "",
                "tampered_state": demo["receipts"]["tampered_manifest"]["state"],
                "tampered_verdict": demo["receipts"]["tampered_manifest"]["verdict"],
                "pass": demo["empty_goal_state"] == "unregistered"
                and demo["empty_goal_sha"] == ""
                and demo["receipts"]["tampered_manifest"]["state"] == "tampered"
                and demo["receipts"]["tampered_manifest"]["verdict"] == "fail",
            },
            "d_zero_domain_semantics": {
                "hits": scan.n_hits,
                "n_files": scan.n_files,
                "n_terms": scan.n_terms,
                "vocabulary_loaded": scan.vocabulary_loaded,
                "pass": scan.clean and scan.vocabulary_loaded,
            },
        },
        "detail": {
            "extraction": {k: v for k, v in field.items() if k != "error_cases"},
            "extraction_heldout": {k: v for k, v in heldout.items() if k != "error_cases"},
            "extraction_ood_undelimited": {k: v for k, v in ood.items() if k != "error_cases"},
            "extraction_per_field": field_agreement(gen_extraction_corpus(20, seed=20261109)),
            "hard_cases": hard,
            "limitation_probes": limits,
            "drift": {k: v for k, v in main_seed["metrics"].items()},
            "baseline_text_hash": main_seed["baseline"],
            "per_case": main_seed["per_case"],
            "per_seed": {k: {"metrics": {kk: vv for kk, vv in v["metrics"].items() if kk != "mismatches"}, "baseline": v["baseline"]} for k, v in per_seed.items()},
            "real_goal_demo": demo,
            "scan": scan.to_dict(),
            "figures": {k: _artifact_label(p) for k, p in figs.items()},
        },
    }
    payload["all_acceptance_pass"] = all(
        v["pass"] for v in payload["acceptance"].values() if isinstance(v, dict) and "pass" in v
    ) and all_seeds_ok
    eval_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_doc(eval_path, doc_path)
    deliverable = deliverable_record(payload, _artifact_label(eval_path))
    deliverable_path.write_text(json.dumps(deliverable, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    if verbose:
        print(json.dumps(payload["acceptance"], ensure_ascii=False, indent=2))
        print("all_acceptance_pass:", payload["all_acceptance_pass"])
        print("doc:", doc_path)
        print("deliverable:", deliverable_path, "sealed", deliverable[DIGEST_KEY][:12])
        print("artifacts:", ", ".join(_artifact_label(p) for p in sorted(out_dir.glob("*"))))
    return 0 if payload["all_acceptance_pass"] else 1


def check(doc_path: Path = DOC, deliverable_path: Path = DELIVERABLE) -> int:
    """Re-derive everything into a scratch dir and byte-compare what is committed.

    A merge-time gate: the checked-in page and mission record are code output, so
    "stale copy in git" (the exact failure this mission was reopened for: the
    committed record was another mission's) must be a non-zero exit.  Writes
    nothing into the repository.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        code = run(out_dir=tmp / "artifacts", doc_path=tmp / "page.md",
                   eval_path=tmp / "artifacts" / "eval.json",
                   deliverable_path=tmp / "acceptance_criteria.json", verbose=False)
        pairs = (
            (Path(doc_path), tmp / "page.md"),
            (Path(deliverable_path), tmp / "acceptance_criteria.json"),
        )
        problems = []
        for committed, fresh in pairs:
            label = committed.relative_to(ROOT) if committed.is_absolute() and ROOT in committed.parents else committed
            if not committed.exists():
                problems.append(f"{label}: missing")
            elif committed.read_bytes() != fresh.read_bytes():
                problems.append(f"{label}: stale vs derivation")
        # the committed copy must also pass its own offline seal check
        if Path(deliverable_path).exists():
            ok, reason = verify_deliverable(json.loads(Path(deliverable_path).read_text(encoding="utf-8")))
            if not ok:
                problems.append(f"{deliverable_path.name}: {reason}")
    if problems:
        print("criterion-prereg check: FAIL")
        for p in problems:
            print(" -", p)
        print("fix: python scripts/run_criterion_prereg_eval.py")
        return 1
    print("criterion-prereg check: OK (page + mission record equal the derivation, seal valid)")
    return 0 if code == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(OUT),
                    help="artifact directory (default: artifacts/criterion_prereg)")
    ap.add_argument("--doc", default=str(DOC),
                    help="page to render (default: docs/CRITERION_PREREGISTRATION.md)")
    ap.add_argument("--eval", default=None,
                    help="evaluation record path (default: <out-dir>/eval.json)")
    ap.add_argument("--deliverable", default=str(DELIVERABLE),
                    help="mission acceptance record to emit "
                         "(default: docs/m8598/outputs/acceptance_criteria.json)")
    ap.add_argument("--check", action="store_true",
                    help="derive into a scratch dir and byte-compare the committed "
                         "page/record (exit 1 on drift); writes nothing")
    args = ap.parse_args(argv)
    if args.check:
        return check(Path(args.doc), Path(args.deliverable))
    return run(Path(args.out_dir), Path(args.doc), Path(args.eval) if args.eval else None,
               Path(args.deliverable))


if __name__ == "__main__":
    sys.exit(main())