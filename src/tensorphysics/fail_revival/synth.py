"""Synthetic fixtures with generator-side ground truth (no domain vocabulary).

Everything here is deliberately abstract -- ``alpha``, ``beta``, ``sigma_1`` -- so
that a passing test cannot be explained by an engineering prior.  Each builder
returns both the fixture and the expected outcome, so a test compares two
independently produced objects instead of a literal the implementation could
have been tuned to.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .model import canonical_json, sha256_text

SYNTH_ARTIFACT_SCHEMA = "campaign_node_artifact.v1"


def _artifact(mission_id: int, markdown: str, *, goal: str, summary: str = "",
              plan_id: int = 900) -> dict[str, Any]:
    return {
        "schema": SYNTH_ARTIFACT_SCHEMA,
        "mission_id": mission_id,
        "plan_id": plan_id,
        "node_ext_id": "synth-%d" % mission_id,
        "goal": goal,
        "report": {
            "id": mission_id + 500000,
            "title": "synthetic node %d" % mission_id,
            "summary": summary,
            "markdown": markdown,
            "sha256": sha256_text(markdown),
        },
    }


def criterion_goal(items: tuple[str, ...]) -> str:
    body = "\n".join("%d) %s" % (i + 1, t) for i, t in enumerate(items))
    return "Synthetic mission.\n\n[key]\n机器判据：\n" + body + "\n\n实现路径：\n(a) nothing."


def corpus() -> list[dict[str, Any]]:
    """Three synthetic artifacts: one clean FAIL, one pass-only, one no report."""
    md_fail = "\n".join([
        "## table",
        "| quantity | reference | computed | error | <3%? | verdict |",
        "|---|---|---|---|---|---|",
        "| sigma_1 | 1.000 | 1.500 | +50.0% | 否 | FAIL |",
        "| sigma_2 | 2.000 | 2.010 | +0.5% | 是 | PASS |",
        "| sigma_3 | 3.000 | 3.000 | 0.0% | 是 | PASS |",
        "",
        "## control table (rejection column: 'no' means NOT rejected -> pass)",
        "| quantity | expected | rejected | note |",
        "|---|---|---|---|",
        "| control_a | reject | 否 | not rejected as required |",
        "| control_b | pass   | 是 | rejected |",
    ])
    a = _artifact(9101, md_fail,
                  goal=criterion_goal(("sigma_1 error <= 3%", "no hand-typed numbers")),
                  summary="synthetic: numerical convergence residual not met")
    md_pass = "\n".join([
        "| quantity | reference | computed | error | <3%? |",
        "|---|---|---|---|---|",
        "| tau_1 | 5.0 | 5.01 | 0.2% | 是 |",
    ])
    b = _artifact(9102, md_pass, goal=criterion_goal(("tau_1 error <= 3%",)))
    c = _artifact(9103, "", goal=criterion_goal(("omega_1 error <= 1%",)))
    return [a, b, c]


def expected_fail_rows() -> list[dict[str, Any]]:
    """Ground truth for :func:`corpus`: exactly one FAIL row, at line 4."""
    return [{"mission_id": 9101, "line_no": 4, "metric": "sigma_1",
             "error": "+50.0%", "allowed": "<3%"}]


# --------------------------------------------------------------------------- #
# tampered revival requests (C2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TamperCase:
    name: str
    payload: dict[str, Any]
    expected_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "payload": self.payload, "expected_reason": self.expected_reason}


def tamper_cases(fail_key: str, criterion_sha256: str, input_sha256: str) -> list[TamperCase]:
    """Ten rewrite attempts, each with the reason it must be refused for.

    The list is built from the *registered* digests, so a case that mutates
    nothing would be admitted and the test would fail loudly.
    """
    core = {
        "fail_key": fail_key,
        "criterion_sha256": criterion_sha256,
        "input_sha256": input_sha256,
    }
    cases = [
        TamperCase("relaxed_tolerance", dict(core, tolerance=0.2), "EDITABLE_FIELD_FORBIDDEN"),
        TamperCase("edited_threshold", dict(core, threshold="<10%"), "EDITABLE_FIELD_FORBIDDEN"),
        TamperCase("rewritten_metric", dict(core, metric="sigma_renamed"), "EDITABLE_FIELD_FORBIDDEN"),
        TamperCase("verdict_injection", dict(core, verdict="PASS"), "VERDICT_INJECTION_FORBIDDEN"),
        TamperCase("criterion_text_edit", dict(core, criterion_text="sigma_1 error <= 30%"),
                   "EDITABLE_FIELD_FORBIDDEN"),
        TamperCase("truncated_digest", dict(core, criterion_sha256=criterion_sha256[:16]),
                   "HASH_FORMAT_INVALID"),
        TamperCase("unregistered_fail", dict(core, fail_key="FR-does-not-exist"),
                   "UNREGISTERED_FAIL"),
        TamperCase("missing_input", {k: v for k, v in core.items() if k != "input_sha256"},
                   "INPUT_HASH_MISSING"),
        TamperCase("substituted_criterion", dict(core, criterion_sha256="0" * 64),
                   "CRITERION_HASH_MISMATCH"),
        TamperCase("registry_digest_claim", dict(core, registry_criterion_sha256="1" * 64),
                   "CRITERION_HASH_MISMATCH"),
    ]
    return cases


def admissible_case(fail_key: str, criterion_sha256: str, input_sha256: str) -> TamperCase:
    """The one request that must be admitted (control for the ten rejections)."""
    return TamperCase(
        "admissible_core",
        {"fail_key": fail_key, "criterion_sha256": criterion_sha256, "input_sha256": input_sha256},
        "ACCEPTED",
    )


def expected_schedule() -> dict[str, Any]:
    """Ground truth for the synthetic three-blocker scenario."""
    return {
        "cycle1": [],
        "cycle2": ["fail_cap"],
        "cycle3": ["fail_caliber", "fail_mission"],
        "latency": {"fail_cap": 1, "fail_mission": 1, "fail_caliber": 1},
    }


def canonical(obj: Any) -> str:
    return canonical_json(obj)