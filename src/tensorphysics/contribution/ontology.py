"""Contribution category ontology: 6 classes with evidence thresholds and counter-examples.

Categories:
  C1 verified-case    — A verified computational/experimental case with reference comparison
  C2 capability-delta — Measurable capability improvement (performance/accuracy/scope)
  C3 defect-fix       — Bug fix with regression test or before/after evidence
  C4 method/doc       — New method documentation, tutorial, or workflow description
  C5 negative-result  — Documented failure/negative result with diagnostic evidence
  C6 infra            — Infrastructure/tooling improvement (CI, harness, deployment)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ContributionCategory(str, Enum):
    C1_VERIFIED_CASE = "C1_verified_case"
    C2_CAPABILITY_DELTA = "C2_capability_delta"
    C3_DEFECT_FIX = "C3_defect_fix"
    C4_METHOD_DOC = "C4_method_doc"
    C5_NEGATIVE_RESULT = "C5_negative_result"
    C6_INFRA = "C6_infra"


@dataclass
class CategoryDefinition:
    """Definition of a contribution category."""
    name: str
    label: str
    description: str
    min_evidence: List[str]          # minimum evidence fields required
    positive_signals: List[str]       # keywords/patterns that suggest this category
    counter_examples: List[str]       # what does NOT count (anti-patterns)
    evidence_threshold: str           # textual description of the bar


CONTRIBUTION_ONTOLOGY: Dict[ContributionCategory, CategoryDefinition] = {
    ContributionCategory.C1_VERIFIED_CASE: CategoryDefinition(
        name="C1_verified_case",
        label="Verified Case",
        description=(
            "A computational or experimental case study that has been verified against "
            "a reference solution (analytical, experimental, or established benchmark). "
            "Must include geometry validation (Cb/L/B or equivalent), parameter table, "
            "and quantitative error analysis vs reference."
        ),
        min_evidence=[
            "geometry_parameters",       # Cb, L, B or equivalent
            "reference_value",           # analytical/experimental/benchmark value
            "computed_value",            # result from the case
            "error_percent",             # |computed - reference| / reference * 100
        ],
        positive_signals=[
            "verified", "benchmark", "validation", "reference solution",
            "analytical solution", "experimental data", "error analysis",
            "Cb=", "L/B", "comparison with", "against experiment",
        ],
        counter_examples=[
            "Running a simulation without any reference comparison",
            "Reporting results without geometry validation",
            "Claiming 'good agreement' without numerical error quantification",
            "Pure code execution logs with no physics content",
        ],
        evidence_threshold=(
            "Must have ≥3 of 4 evidence fields present; error must be quantified; "
            "geometry must be validated (Cb/L/B or equivalent dimensional check)."
        ),
    ),
    ContributionCategory.C2_CAPABILITY_DELTA: CategoryDefinition(
        name="C2_capability_delta",
        label="Capability Delta",
        description=(
            "A measurable improvement in capability: performance (speed, memory), "
            "accuracy (error reduction), or scope (new physics, new regime). "
            "Must include before/after metrics with quantified delta."
        ),
        min_evidence=[
            "baseline_metric",           # metric before improvement
            "improved_metric",           # metric after improvement
            "delta_percent",             # (improved - baseline) / baseline * 100
            "metric_name",               # what is being measured
        ],
        positive_signals=[
            "improved", "speedup", "acceleration", "reduced error",
            "extended", "new capability", "performance gain", "delta",
            "before/after", "baseline vs", "x faster", "x% improvement",
        ],
        counter_examples=[
            "Claiming improvement without baseline numbers",
            "Subjective 'feels faster' without measurement",
            "Adding a feature without quantifying its impact",
            "Refactoring with no measurable change in output",
        ],
        evidence_threshold=(
            "Must have baseline and improved metrics with quantified delta; "
            "delta must be ≥5% to count as meaningful capability change."
        ),
    ),
    ContributionCategory.C3_DEFECT_FIX: CategoryDefinition(
        name="C3_defect_fix",
        label="Defect Fix",
        description=(
            "A bug fix that includes evidence of the defect and its resolution. "
            "Must include: description of the bug, the fix applied, and either "
            "a regression test or before/after output comparison."
        ),
        min_evidence=[
            "bug_description",           # what was wrong
            "fix_description",           # what was changed
            "regression_test",           # test that catches the bug (or before/after)
        ],
        positive_signals=[
            "fix", "bug", "defect", "regression", "corrected",
            "issue #", "patch", "wrong result", "incorrect",
            "was producing", "now produces", "fixed",
        ],
        counter_examples=[
            "Cosmetic changes (formatting, comments) labeled as fixes",
            "Feature additions disguised as bug fixes",
            "Fixes without any test or verification",
            "Reverting a commit without explaining the defect",
        ],
        evidence_threshold=(
            "Must have bug description + fix description + regression test "
            "or before/after output evidence."
        ),
    ),
    ContributionCategory.C4_METHOD_DOC: CategoryDefinition(
        name="C4_method_doc",
        label="Method / Documentation",
        description=(
            "New method documentation, tutorial, workflow description, or "
            "educational material that enables reproducibility. Must include "
            "actionable steps, parameter guidance, and expected outputs."
        ),
        min_evidence=[
            "method_description",        # what the method does
            "steps_or_workflow",         # how to use it
            "expected_output",           # what the user should see
        ],
        positive_signals=[
            "tutorial", "guide", "how to", "workflow", "method",
            "documentation", "README", "example", "walkthrough",
            "step-by-step", "procedure", "protocol",
        ],
        counter_examples=[
            "Auto-generated API docs without explanatory content",
            "Code comments that are not user-facing documentation",
            "Vague descriptions without actionable steps",
            "Duplicate documentation with no new information",
        ],
        evidence_threshold=(
            "Must have method description + actionable steps/workflow + "
            "expected output description."
        ),
    ),
    ContributionCategory.C5_NEGATIVE_RESULT: CategoryDefinition(
        name="C5_negative_result",
        label="Negative Result",
        description=(
            "A documented failure or negative result with diagnostic evidence. "
            "Must include: what was attempted, why it failed, diagnostic data, "
            "and lessons learned. Negative results are valuable scientific contributions."
        ),
        min_evidence=[
            "attempt_description",       # what was tried
            "failure_mode",              # how/why it failed
            "diagnostic_data",           # logs, error messages, intermediate results
        ],
        positive_signals=[
            "failed", "did not converge", "negative result", "unable to",
            "diverged", "unstable", "not reproducible", "limitation",
            "does not work", "infeasible", "counter-example",
        ],
        counter_examples=[
            "Vague 'it didn't work' without diagnostic detail",
            "Failures due to user error (wrong parameters, missing files)",
            "Known limitations already documented elsewhere",
            "Transient infrastructure failures (network, disk full)",
        ],
        evidence_threshold=(
            "Must have attempt description + failure mode + diagnostic data; "
            "NOTE: C5 is inherently harder to identify stably — if classification "
            "confidence is low, mark as NOT_OPERATIONAL."
        ),
    ),
    ContributionCategory.C6_INFRA: CategoryDefinition(
        name="C6_infra",
        label="Infrastructure",
        description=(
            "Infrastructure or tooling improvement: CI/CD, test harness, "
            "deployment automation, monitoring, or developer tooling. "
            "Must include what was improved and measurable impact."
        ),
        min_evidence=[
            "infra_component",           # what part of infra was changed
            "improvement_description",   # what was improved
            "impact_metric",             # how to measure the improvement
        ],
        positive_signals=[
            "CI", "pipeline", "deploy", "infrastructure", "tooling",
            "automation", "harness", "monitoring", "build",
            "test framework", "docker", "container", "workflow",
        ],
        counter_examples=[
            "Minor script tweaks without measurable impact",
            "Configuration changes that don't improve anything",
            "Infrastructure changes that break existing functionality",
            "Documentation-only changes to infra files",
        ],
        evidence_threshold=(
            "Must have infra component + improvement description + impact metric; "
            "impact must be measurable (time saved, coverage increased, etc.)."
        ),
    ),
}

# Convenience: counter-examples lookup by category
COUNTER_EXAMPLES: Dict[str, List[str]] = {
    cat.value: defn.counter_examples
    for cat, defn in CONTRIBUTION_ONTOLOGY.items()
}

# Convenience: evidence thresholds lookup
EVIDENCE_THRESHOLDS: Dict[str, str] = {
    cat.value: defn.evidence_threshold
    for cat, defn in CONTRIBUTION_ONTOLOGY.items()
}


def get_category(cat_str: str) -> Optional[ContributionCategory]:
    """Look up a category by string value."""
    for cat in ContributionCategory:
        if cat.value == cat_str or cat.name == cat_str:
            return cat
    return None


def list_categories() -> List[Dict]:
    """Return a list of category summaries for CLI/display."""
    return [
        {
            "name": defn.name,
            "label": defn.label,
            "description": defn.description,
            "min_evidence": defn.min_evidence,
            "evidence_threshold": defn.evidence_threshold,
        }
        for defn in CONTRIBUTION_ONTOLOGY.values()
    ]
