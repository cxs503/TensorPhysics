#!/usr/bin/env python3
"""Build the hand-annotated golden set (60 real mission reports, one per mission).

Annotation protocol (documented because the gold set is the measuring stick):
  * Sampling: stratified over the classifier's *preliminary* pass (used only for stratification,
    never for labelling) plus a targeted adversarial pool (error-table-bearing infrastructure
    reports) and a targeted C3/C5 pool (correction / negative-result vocabulary) because those
    two classes are rare in the window.
  * Each item was labelled by reading a bounded digest (title + summary + the lines carrying
    contribution evidence: 误差 / PASS / 基线 / 解析解 / MOCK->REAL / 修复 / 不可行 / 反例).
  * Rationale and the evidence field(s) relied upon are stored per item; the classifier is
    evaluated afterwards, so labels are not derived from the classifier output.
  * Single annotator; no inter-annotator agreement is claimed.

Usage
    python3 tools/contribution_meter/build_golden_set.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# (mission_id, label, rationale, adversarial)
ANNOTATIONS = [
    # ---------------- C1 verified-case ----------------
    (4721, "C1", "New independent analytic benchmark (truncated multipole series, N=24, boundary residual 3.7e-11) "
                 "for the two-cylinder added-mass matrix vs BEM on disjoint rings: max rel err 2.20e-3 -> 5.51e-4 as "
                 "panels refine. External reference, quantified, reproducible script.", False),
    (4886, "C1", "D3Q19 BGK four analytic cases (shear wave / Couette / Poiseuille / wall friction) worst 0.338% against "
                 "the 1% criterion, with criterion-level negative controls (9.877% / 95.238%) and a provenance script "
                 "that acts as the gate.", False),
    (4951, "C1", "J2 elastoplasticity material point: elastic slope err 1.5e-16, plastic slope EH/(E+H) err 5.8e-15, "
                 "yield surface err 2.0e-16 against closed-form anchors plus six post-hoc residual audits.", False),
    (4973, "C1", "Analytic verification ladder 4/4 under the 1% criterion (max 0.337%); negative controls upgraded to "
                 "criterion level (mis-ordered backfill -> Couette 4.762%, Poiseuille 17.17%) and a drift gate pins "
                 "metrics -> table -> document.", False),
    (4990, "C1", "New Step E in the LBM ladder: 10x10 square-duct Poiseuille vs the Boussinesq series, section-wise max "
                 "error 0.4159% (<1%) and monotone decay 0.827/0.416/0.235% over L=6/10/14 -> discretisation error, "
                 "not non-convergence.", False),
    (5068, "C1", "First analytic anchor for DEM-LBM two-phase coupling: ice-floe drift in uniform current, 9-case "
                 "parameter sweep all within 0.01% of the analytic solution.", False),
    (5072, "C1", "Ice-steel friction law calibrated to 17 compiled literature anchors (RMS 1.06%), holdout set max error "
                 "6.67% (<10% gate), 5-fold CV 4.33%; DEM implementation recovery <=1.17%.", False),
    (5043, "C1", "DG Burgers shock capturing: smooth-region observed orders 1.970/2.929/3.935 (>= p+0.9), Rankine-Hugoniot "
                 "shock position inverted to 1e-15, Gibbs metrics bounded - design order and RH are external "
                 "references.", False),
    (5083, "C1", "Device icing chain: radar icing rate 0.904 mm/h inside the 0.5-2.0 literature band, gain attenuation "
                 "0.390 dB/mm within 0.3-0.5 (<3% deviation), 5/6 criteria pass with one honest FAIL recorded.", False),
    (5082, "C1", "Weld-toe notch FE sub-model vs the Inglis exact solution (0.7-4.3%) giving Kt=2.33/2.40 inside the IIW "
                 "1.5-2.5 band, plus DNV-RP-C203 D low-temperature correction.", False),
    (5205, "C1", "Holtrop-normalised C_AW + ITTC STAWAVE-1 spectral integral reproduced to 9.3e-05% against the closed "
                 "form, geometry gate 0.12%; the RAO resonance failure (73.0% peak-frequency error, no diffraction) is "
                 "recorded as FAIL - the unit is still the verified closed-form case.", False),
    (5197, "C1", "Armstrong-Frederick cyclic plasticity with consistent tangent: four closed-form anchors aligned "
                 "point-by-point - stable hysteresis peak 0.15%, dissipation per cycle 0.26%, Bauschinger drop 0.68%, "
                 "ratcheting rate 2.6% (Richardson).", False),
    (5262, "C1", "Damaged-stability chain from first principles: single-compartment equilibrium level error 2.3e-12%, "
                 "heel angle vs the first-order formula 0.115%, 20/20 machine criteria; three independent integration "
                 "routes for the waterplane.", False),
    (5290, "C1", "Baffled simply-supported plate radiation efficiency pinned by three independent anchors "
                 "(Rayleigh integral, Leppington high-frequency asymptote, Maidanik low-frequency asymptote) plus a "
                 "power-balance check, 19 tests.", False),
    (5255, "C1", "Michell wave-resistance verification face: Wigley / ellipsoid / tandem multi-body, 37 machine "
                 "criteria; ellipsoid R_w three-point error <=0.0110% (GCI <=0.014%), tandem interference <=1.49%.", False),
    (4747, "C1", "Hertz contact duration compared with the Johnson analytic solution: TensorDEM time-domain verification "
                 "debt cleared with an external closed-form reference.", False),
    # ---------------- C2 capability-delta ----------------
    (4858, "C2", "Young-Laplace closure adjudication matrix: evidence level WITHHELD -> PROVISIONAL by machine "
                 "adjudication of 19 read-outs against Adami et al. (2012) admissibility; 9 tests gate the rules. "
                 "Evidence-level upgrade, no new external error table.", False),
    (4909, "C2", "PC6 full-scale ice-load chain implemented on top of an existing module (pressure-area law, bow-impact "
                 "zone coefficients, global bending/shear); evidence graded formula-structure REAL / coefficient table "
                 "RECONSTRUCTED / design thickness ASSUMPTION - acceptance by L2 anchor bands, not by an external "
                 "error table.", False),
    (5044, "C2", "Coupled clock semantics brought from document level into src (rational-ratio GCD lattice, LCM exchange "
                 "cadence, causality gate) and cross-checked against an analytic modal solution; 16 pytest cases.", False),
    (5076, "C2", "Ice self-propulsion module built where none existed: ITTC-78 extrapolation with round-trip "
                 "5.31e-16 self-consistency, B-series open water, ice thrust deduction, 15 machine criteria. "
                 "Acceptance is self-consistency, so this is a capability delta rather than a verified case.", False),
    (5071, "C2", "Mooring / ice station-keeping module created from absent: ISO 19906 ice force, catenary statics + "
                 "lumped-mass dynamics, 3-DOF quasi-static coupling, 14/14 criteria, LHS envelope.", False),
    (5103, "C2", "Frequency-domain Rankine panel method seakeeping chain: A/B/F_exc from first principles, Cb gate "
                 "1e-6%, body/free-surface residuals 4e-12/2e-13; capability delivered without an external error table "
                 "against an experimental or cross-code reference.", False),
    (5186, "C2", "Clump-clump contact explicitly moved NOT_SUPPORTED -> REAL: contact-spot merging with a geometric "
                 "junction criterion, Hertz-Mindlin force, third-law/angular-momentum structural conservation "
                 "(degenerate consistency 0, independent geometry estimate 0.40%).", False),
    (5243, "C2", "Degraded damaged-stability implementation replaced by an exact polygon-geometry solver "
                 "(single-compartment heel vs added-weight analytic 1.85% < 2%, GM/draught 0.000%): the level change "
                 "degraded -> exact is the delta; acceptance is analytic-anchored rather than a new case family.", False),
    # ---------------- C3 defect-fix ----------------
    (4760, "C3", "Errata-2 on the ice-wave evidence base: recomputation with three non-overlapping formulas shows the "
                 "group-velocity table row at k=0.005 1/m disagrees with its own dispersion relation by 5.68% while "
                 "the other four rows reproduce to 4.0e-08 -> the wrong value must be corrected and the criterion "
                 "re-graded from two-path self-consistency to three-formula reproduction (1.6e-04).", False),
    (4804, "C3", "Main CI red repaired at the root: seven ledger negative controls rewritten from 'fail on purpose' into "
                 "rejection predicates, with the tampered residual equal to the analytic value (rel 1e-9) and a "
                 "quantified sensitivity margin >=6.6x tol; the gate is locked by tests.", False),
    (5277, "C3", "Known wrong constant corrected: wall origin assumed 0.5 is measured to be 0.645 cells from laminar "
                 "Poiseuille (historical 0.1 also wrong), the hardcoded offset in the analysis pipeline is fixed "
                 "chain-wide, and the u_tau three-way closure (17.06% spread) is locked as a gate; the mission is an "
                 "honest FAIL on the MKM-1999 benchmark, which does not undo the correction unit.", False),
    (4854, "C3", "wss_from_fneq_3d regression criterion upgraded from 'non-zero' to the analytic magnitude "
                 "(rel 1e-5) plus a tau-prefactor ratio criterion, with a 1/(2tau) mutation negative control "
                 "(2 failed / 2 passed) - the previous un-guarded (effectively wrong) expectation is locked out.", False),
    (5204, "C3", "SUBOFF resistance precision repaired: the old chain (full-submerged hull treated as a surface ship) "
                 "overestimated by +62%..+140% on Liu & Huang (1998) Table 14; the corrected fully-submerged chain "
                 "reaches <=3.61%, and KVLCC2 Fr=0.142 closes the regression. Old wrong numbers -> new numbers + gate.", False),
    (4811, "C3", "TensorSPH main CI cleared without relaxing physics: the lamb-frequency guard asserted a 30% accuracy "
                 "that the code never reached (measured -61.69%) and derived figures had gone missing; assertions "
                 "corrected, xfail(strict) pins the open debt.", False),
    (4687, "C3", "Hertz stability criterion algebraically corrected and the DEM3D energy-ledger sign fixed - both were "
                 "wrong numbers inside gate logic, now nailed by the PR gates.", False),
    (4917, "C3", "Stokes second problem: five real defects in run.py fixed (result.json write crash, grid comparison "
                 "evaluated in non-similar coordinates -> a wrong error number), then the pre-registered gate passes at "
                 "2.160%/1.012% - the wrong comparison number is corrected and locked.", False),
    # ---------------- C4 method-doc ----------------
    (4867, "C4", "Method contribution: closed-form fault-detectability threshold |R(z)| = TOL*S(z) for ledger gate "
                 "channels (back-substitution 3.94e-16), quantifying a 20x inter-channel sensitivity spread and "
                 "proving external_work_J is a multiplicative idle channel. The unit is the method, not a case.", False),
    (4925, "C4", "Formal 8-section manuscript draft for the ice-navigation operational chain aggregating 13 reports, "
                 "43 figures, 98 criteria rows - a paper-shaped deliverable that adds no new case.", False),
    (4978, "C4", "Campaign proposal document: five never-done frontier tasks (thermal-mechanical ice contact, ice-induced "
                 "slamming/vibration, icing risk, ice-thickness inversion, weld fatigue) with 82 acceptance criteria; "
                 "planning artefact.", False),
    (5196, "C4", "Proposal set derived mechanically from plan-25 artefacts (48-cell criterion extraction: PASS 35 / "
                 "honest FAIL 12) mapping seven self-stated limits into six new tasks - document-shaped output.", False),
    (5265, "C4", "Metrology method: six-class contribution ontology with evidence gates, a rule+field classifier and a "
                 "60-item gold set (the previous, partial-FAIL attempt at this very problem). The unit is the reusable "
                 "procedure; no new physics case is established.", False),
    (5317, "C4", "Five proposals plus a 307-item pool analysis derived from ten terminal nodes of plan 22 and their "
                 "honest FAILs / artificial thresholds.", False),
    (4827, "C4", "Runnable tutorial ('first verified run') with an executable extraction gate, four mutation-negative "
                 "controls and 10 closed-form benchmarks <3%. The deliverable is documentation that re-uses existing "
                 "verification, so C4 rather than C1.", False),
    (4819, "C4", "Beginner tutorial for a minimal ice-lattice DEM run with an executable assertion gate (Cauchy-Born "
                 "hand-derived tensor 5e-18, bond energy inversion 1.9e-7) - doc-shaped deliverable.", False),
    # ---------------- C5 negative-result ----------------
    (4972, "C5", "Adversarial peer review judging the ice-navigation chain FAIL: two of thirteen 'chain evidences' are "
                 "off-topic, L1/L2 links mis-assigned, 4/8 provenance anchors unverifiable, and five self-assessed "
                 "'no' items (27.9%, 41.9%, ...) contradict the 'all below 3%' summary - a falsified claim with a "
                 "consequence.", False),
    (4996, "C5", "Series-60 batch (330 cases): geometric Cb closure 0.066% is fine, but the Cr table is MOCK and Ct "
                 "differs from the Holtrop baseline by a median 229%, so the platform's resistance chain does not meet "
                 "the <3% bar - the validity limit of the capability is the result.", False),
    (5075, "C5", "High-Re wall-function ladder: overall verdict FAIL (channel_turb_3d, blasius_flat_plate, "
                 "suboff_highre_resistance all unmet) while delivering GCI / wall-law tooling; the negative verdict on "
                 "the model's high-Re validity is the contribution.", False),
    (4997, "C5", "DTMB 5415 honest-FAIL: 5/5 points exceed the 10% hard gate, max deviation 165.2%, mean +129.2%, "
                 "R^2 = -16.6, root cause the MOCK piecewise-linear Cr proxy overestimating slender hulls by 3-40x.", False),
    (4848, "C5", "SUBOFF root-cause analysis: |dCt| up to 688% against DTRC towing-tank measurements, attributed "
                 "term-by-term (wave term must vanish for a deeply submerged body, etc.) with an exactly closing "
                 "six-term decomposition - a quantified statement of where Holtrop stops being valid.", False),
    (4751, "C5", "Gauge audit of a registered -43.5% frequency FAIL: the trajectory had blown up (rms radius 3531 R, "
                 "KE 4.0e8 sigmaR), and the stable window gives -2.16%, so the claimed CSF accuracy defect is retired "
                 "as a numerical artefact.", False),
    (5291, "C5", "KVLCC2 MMG 3-DOF honest-FAIL: structure REAL and equivalent derivatives inverted by weighted least "
                 "squares, but absolute accuracy stays WITHHELD against free-running trials; the bounded claim is the "
                 "unit.", False),
    (4924, "C5", "The pre-registered 5% masked relative-error criterion is proven ill-posed (denominator pinned at "
                 "1.01-1.10x the threshold, platform 4.99/5.16/4.86% across a 4x resolution span) and retired; the "
                 "case is re-judged PASS under the corrected reading.", False),
    # ---------------- C6 infra ----------------
    (4672, "C6", "CI triage: 14/14 check-runs cancelled with zero steps -> account-level runner eviction, not a code "
                 "failure; local pytest subset 43 passed. Infra diagnosis, no physics.", True),
    (4697, "C6", "Pipeline smoke marker relocated into tests/ to satisfy the repository-layout gate (19 passed); the "
                 "report prints baseline commits and gate counts (looks quantitative) but no scientific quantity.", True),
    (4727, "C6", "Backend registry contract hardening: 18 tests / 32 subtests over registration, resolution, invalid-name "
                 "errors and factory fallback, plus mutation-driven negative cases 11/11 red.", True),
    (4849, "C6", "SDAA artefact ledger v2: 3190 artefacts across 193 worktrees normalised, evidence-graded and "
                 "re-adjudicated (550 conflicts) - the tables of numbers are ledger bookkeeping, not verification.", True),
    (4934, "C6", "Campaign orchestration contracts: M12 clock lattice, dual mass/energy/impulse ledgers reconciled to "
                 "machine precision, geometry contract registration. Machine-precision residuals between producers and "
                 "consumers are self-consistency, not external verification.", True),
    (5023, "C6", "Four-centre minimal dashboards: 44/44 metrics bit-equal to an independent SQL query, zero fake data. "
                 "Reconciliation against the database that produced the numbers is the archetypal 'looks like C1'.", True),
    (5039, "C6", "Ice-wave spectrum data contract v1.0.0: schema frozen, 4/4 producer samples accepted, 24/24 breakers "
                 "rejected with expected codes, spectral-moment error 0.135%. Contract work with error numbers, no "
                 "physics case.", True),
    (5176, "C6", "SxE closure acceptance gate: five criteria met, coverage 100%, ledger 548 chains ok, 44/44 numbers "
                 "recomputed - process accounting.", True),
    (5251, "C6", "Capability-audit CLI: bounded worktree scan, relevance/topic gates against false positives, numeric "
                 "tier-A/B criteria (error N% <= 3%) applied to *other* missions' claims - an audit tool is infra.", True),
    (5274, "C6", "Evidence grading of 21 platform capabilities (REAL 12 / MOCK 3 / WITHHELD 5 / NOT_SUPPORTED 1) with "
                 "worst quantitative benchmark 4.77% < 15% and geometry closure 2.50% < 3% - the numbers are re-graded "
                 "from existing evidence, so no new case is added.", True),
    (5112, "C6", "ship-platform API health fix: six API defects (potential-flow negative Re 500, NameError in wave "
                 "functions, schema mismatch, over-strict required fields) repaired with 85 tests. Availability defects "
                 "without a wrong scientific number.", False),
    (5303, "C6", "Council constitutionalisation: written constitution with clause-level provenance, red-line triage and "
                 "a 4-family supermajority vote, replayed on three historical disputes (2 concordant, 1 defensible "
                 "divergence). Governance infrastructure.", False),
]


def main() -> int:
    corpus = [json.loads(line) for line in
              open_text(resolve_corpus(os.path.join(ROOT, "data/corpus_reports.jsonl")))]
    best: dict[str, dict] = {}
    for r in corpus:
        mid = str(r["mission_id"])
        if mid not in best or (r.get("markdown_len") or 0) > (best[mid].get("markdown_len") or 0):
            best[mid] = r

    items, missing = [], []
    for mid, label, rationale, adv in ANNOTATIONS:
        rep = best.get(str(mid))
        if rep is None:
            missing.append(mid)
            continue
        items.append(
            {
                "mission_id": int(mid),
                "report_id": str(rep["report_id"]),
                "label": label,
                "title": (rep.get("title") or "")[:120],
                "repo_signal": rep.get("scenario_key") or "",
                "adversarial_looks_like_C1": bool(adv),
                "rationale": rationale,
                "evidence_basis": "report title+summary+evidence lines (误差/PASS/基线/解析解/MOCK→REAL/修复/不可行)",
                "annotator": "agent-single",
                "labeled_from": "bounded digest, classifier output not consulted",
            }
        )
    counts: dict[str, int] = {}
    for it in items:
        counts[it["label"]] = counts.get(it["label"], 0) + 1
    payload = {
        "version": "1.0.0",
        "n_items": len(items),
        "class_counts": counts,
        "n_adversarial": sum(1 for it in items if it["adversarial_looks_like_C1"]),
        "sampling": "stratified on the preliminary rule pass + targeted C3/C5 vocabulary pool + adversarial "
                    "error-table-bearing infra pool; one report per mission (largest markdown)",
        "corpus": "data/corpus_reports.jsonl (real platform reports harvested over a 96 h window)",
        "annotation_limits": [
            "single annotator, no inter-annotator agreement",
            "labels are read off bounded digests, not full reports",
            "C1/C2 and C1/C3 boundaries involve judgement; rationale is stored per item",
        ],
        "missing_mission_ids": missing,
        "items": items,
    }
    out = os.path.join(ROOT, "data/golden_set_contributions.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"n": len(items), "counts": counts,
                      "adversarial": payload["n_adversarial"], "missing": missing},
                     ensure_ascii=False))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())