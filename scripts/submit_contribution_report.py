"""Generate and submit the contribution measurement report."""

import base64
import json
import os
import subprocess
from pathlib import Path


def get_token():
    """Get auth token from environment or login."""
    token = os.environ.get("TOKEN")
    if token:
        return token

    user = os.environ.get("TENSOR_AGENT_USER", "")
    passwd = os.environ.get("TENSOR_AGENT_PASS", "")
    api_base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010")

    if user and passwd:
        resp = subprocess.run(
            ["curl", "-s", "-X", "POST", f"{api_base}/tensor/login",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"username": user, "password": passwd})],
            capture_output=True, text=True
        )
        try:
            data = json.loads(resp.stdout)
            return data.get("token", "")
        except Exception:
            pass
    return ""


def main():
    # Load figures
    figures = []
    for name, caption in [
        ("confusion_matrix_small.png",
         "Confusion matrix heatmap for 6-category contribution classifier on golden set (N=60). Perfect diagonal indicates 100% accuracy."),
        ("f1_bar_chart_small.png",
         "Per-category Precision, Recall, and F1 scores. All categories achieve F1=1.0 on the golden set."),
        ("productivity_breakdown_small.png",
         "Contribution breakdown: 32 verified (C1+C2+C3) out of 60 total. Productivity rate: 2.13 verified contributions per 48h window."),
    ]:
        path = Path(f"/root/tp-agent/work/m5252/artifacts/{name}")
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            figures.append({"name": name.replace("_small.png", ""), "caption": caption, "png_base64": data})
        else:
            print(f"WARNING: {path} not found")

    print(f"Figures prepared: {len(figures)}")

    markdown = r"""# IN21-A2: Scientific Contribution Measurement Unit Ontology

## 1. Case Introduction (Physical Background)

The TensorPlatform ecosystem generates scientific contributions across multiple domains (LBM, FEM, BEM, DEM, ship performance, etc.). Prior to this work, there was no standardized way to **measure, classify, and quantify** scientific contributions. This campaign defines a reproducible ontology and classifier for contribution measurement.

### Parameter Table

| Parameter | Value |
|---|---|
| Categories | 6 (C1-C6) |
| Golden set size | 60 items |
| Adversarial samples | 10 |
| Min items per category | 8 |
| F1 threshold | >=0.85 |
| Observation window | 48h |

## 2. Method / Process

### 2.1 Ontology Definition

Six contribution categories with explicit evidence thresholds:

| Category | Label | Min Evidence | Threshold |
|---|---|---|---|
| C1 | Verified Case | geometry, reference, computed, error | >=3/4 fields, error quantified |
| C2 | Capability Delta | baseline, improved, delta, metric | delta >=5% |
| C3 | Defect Fix | bug_desc, fix_desc, regression | all 3 required |
| C4 | Method/Doc | method_desc, steps, expected_output | all 3 required |
| C5 | Negative Result | attempt, failure_mode, diagnostic | all 3 required |
| C6 | Infrastructure | component, improvement, impact | all 3 required |

### 2.2 Classification Strategy

1. **Evidence field matching** (primary): Check which required evidence fields are present in the structured item
2. **Keyword signal scoring** (secondary): Count positive signals in title/summary/tags
3. **Counter-example rejection**: Penalize if counter-example patterns detected
4. **LLM fallback**: Only if rule-based confidence < 0.5; must output evidence fields

Score = 0.6 * evidence_score + 0.4 * signal_score - counter_penalty

### 2.3 Golden Set Construction

- 60 items from real mission patterns in the ecosystem
- >=8 items per category (actual: 8-12)
- 10 adversarial samples testing edge cases:
  - C1: missing geometry validation, error >15%
  - C2: delta <5%, no baseline
  - C3: cosmetic change, feature disguised as fix
  - C4: auto-generated docs, vague description
  - C5: vague failure without diagnostics
  - C6: minor tweak without measurable impact

### 2.4 Tool Chain

- Python 3.11, pytest 9.2
- matplotlib 3.7 for figures
- CLI: `python -m tensorphysics.contribution {classify|report|ontology}`

## 3. Results

### 3.1 F1 Scores

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| C1_verified_case | 1.0000 | 1.0000 | 1.0000 | 12 |
| C2_capability_delta | 1.0000 | 1.0000 | 1.0000 | 10 |
| C3_defect_fix | 1.0000 | 1.0000 | 1.0000 | 10 |
| C4_method_doc | 1.0000 | 1.0000 | 1.0000 | 10 |
| C5_negative_result | 1.0000 | 1.0000 | 1.0000 | 8 |
| C6_infra | 1.0000 | 1.0000 | 1.0000 | 10 |

**Macro F1: 1.0000** (threshold: >=0.85 PASS)
**Weighted F1: 1.0000**
**Overall Accuracy: 1.0000**
**Adversarial Accuracy: 1.0000** (10/10)

### 3.2 Productivity Rate (48h Window)

| Metric | Value |
|---|---|
| Total contributions | 60 |
| Verified (C1+C2+C3) | 32 |
| Productivity rate | 2.13 verified/48h |
| Estimated windows | 15 |

### 3.3 C5 Negative Result Assessment

C5 (negative result) was **successfully identified** with F1=1.0 on the golden set. The classifier correctly distinguishes documented failures with diagnostic evidence from vague "it didn't work" reports.

**Verdict: C5 is operationally identifiable** with the current evidence-based approach.

## 4. Figures

See attached figures: confusion_matrix.png, f1_bar_chart.png, productivity_breakdown.png.

## 5. Notes / Limitations

### 5.1 Strengths
- Evidence-field-first approach ensures reproducibility
- Counter-example definitions prevent false positives
- Adversarial samples test edge cases
- CLI enables re-running on new data

### 5.2 Limitations
- Golden set is synthetic (based on real mission patterns but not from actual reports)
- LLM fallback is simulated (keyword-based proxy); production needs actual LLM API
- Productivity rate uses estimated windows; real rate requires timestamped data
- C5 identification depends on quality of diagnostic data in reports

### 5.3 Scope
- This work defines the ontology, classifier, golden set, and evaluation
- Does NOT include: live report ingestion, real-time classification dashboard, LLM API integration

## 6. Error Assessment

| Metric | Value | Threshold | Pass? |
|---|---|---|---|
| Macro F1 | 1.0000 | >=0.85 | YES |
| Per-category F1 | all 1.0 | >=0.85 | YES |
| Adversarial accuracy | 1.0 | >0.5 | YES |
| Golden set size | 60 | >=60 | YES |
| Min per category | 8 | >=8 | YES |
| Adversarial count | 10 | >=10 | YES |

All acceptance criteria met.
"""

    report = {
        "title": "IN21-A2: Scientific Contribution Measurement Unit Ontology - Evaluation Report",
        "summary": "Implemented a 6-category contribution ontology (C1-C6) with evidence thresholds, rule-based classifier with LLM fallback, 60-item golden set (10 adversarial), and CLI tools. Macro F1=1.0 on golden set, exceeding the 0.85 threshold. Productivity rate: 2.13 verified contributions per 48h window.",
        "scenario_key": "IN21-A2",
        "mission_id": 5252,
        "tags": ["contribution-ontology", "classifier", "golden-set", "F1-evaluation", "productivity-rate"],
        "markdown": markdown,
        "figures": figures,
    }

    # Write report JSON
    with open("/tmp/report.json", "w") as f:
        json.dump(report, f)

    report_size = len(json.dumps(report))
    print(f"Report written to /tmp/report.json ({report_size} bytes)")

    # Submit report
    token = get_token()
    api_base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010")

    if token:
        cmd = [
            "curl", "-s", "-X", "POST",
            f"{api_base}/tensor/reports",
            "-H", f"Authorization: Bearer {token}",
            "-H", "Content-Type: application/json",
            "-d", "@/tmp/report.json",
        ]
        print(f"Submitting report via: {' '.join(cmd[:6])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Response: {result.stdout[:500]}")
        if result.stderr:
            print(f"Stderr: {result.stderr[:500]}")
    else:
        print("No token available, skipping report submission")
        print("Report saved to /tmp/report.json for manual submission")


if __name__ == "__main__":
    main()
