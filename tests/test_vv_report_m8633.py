"""Tests for the machine-generated V&V report (mission m8633).

The report renderer fills ``docs/vv_report_template.md`` from a
:class:`~tensorphysics.vv.gci.GateVerdict`.  The tests below check the three
properties that make such a report trustworthy:

1. **zero hand-copy** -- every number in the markdown equals the number the gate
   computed (compared through ``section.data``, not by re-parsing the prose);
2. **no silent blanks** -- a section the machine cannot fill is ``WITHHELD`` and
   the report status can then never be better than ``WITHHELD``;
3. **provenance** -- the report carries a digest of the verdict payload and of
   its own body, so a report is tied to the artefacts it was rendered from.
"""

from __future__ import annotations

import hashlib

import pytest

from tensorphysics.vv.gci import Observation, evaluate
from tensorphysics.vv.report import (
    DEFAULT_ERROR_ACCEPTANCE,
    PLACEHOLDER_TOKENS,
    STATUS_ORDER,
    build_report,
    render_markdown,
)


# ---------------------------------------------------------------------------
# fixtures: a manufactured third-order sequence and its gate verdict
# ---------------------------------------------------------------------------


def _levels(p: float = 1.8, f_inf: float = 2.0, c: float = 3.0, hs=(0.4, 0.2, 0.1)):
    return [Observation(h=h, f=f_inf + c * h**p, label=f"h={h:g}") for h in hs]


@pytest.fixture()
def convergence_verdict():
    """Converging study, no reference value (validation stage not run)."""
    return evaluate(_levels(), quantity="Cd")


@pytest.fixture()
def validated_verdict():
    """Converging study plus a reference value inside the GCI band."""
    return evaluate(_levels(), quantity="Cd", f_exp=2.01, u_input=0.02)


@pytest.fixture()
def case_metadata():
    return {
        "case_name": "synthetic manufactured sequence",
        "reference_source": "manufactured solution f(h) = 2 + 3 h^1.8",
        "parameters": [
            {"name": "f_inf", "value": 2.0, "unit": "-", "uncertainty": 0.0},
            {"name": "refinement", "value": 2.0, "unit": "-", "uncertainty": 0.0},
        ],
    }


# ---------------------------------------------------------------------------
# 1. zero hand-copy: the report numbers are the gate numbers
# ---------------------------------------------------------------------------


def test_convergence_numbers_match_verdict_exactly(convergence_verdict):
    verdict = convergence_verdict
    report = build_report(verdict, title="Cd convergence")
    data = report.section("grid-convergence").data

    assert verdict.study is not None
    assert data["order_p"] == verdict.study.order_p
    assert data["f_extrapolated"] == verdict.study.f_extrapolated
    assert data["f_fine"] == verdict.study.f_fine
    assert data["gci_fine"] == verdict.study.gci_fine
    assert data["convergence_ratio"] == verdict.study.convergence_ratio
    assert data["n_levels"] == 3


def test_manufactured_order_is_recovered_and_rendered(convergence_verdict):
    """The gate recovers the manufactured order and the report shows that value."""
    verdict = convergence_verdict
    report = build_report(verdict, title="Cd convergence")
    assert verdict.study is not None and verdict.study.order_p is not None
    assert verdict.study.order_p == pytest.approx(1.8, abs=1e-6)
    # the exact formatted value the gate produced must appear in the markdown
    assert f"{verdict.study.order_p:.6g}" in report.markdown
    assert f"{verdict.study.f_extrapolated:.6g}" in report.markdown


def test_validation_numbers_match_verdict_exactly(validated_verdict):
    verdict = validated_verdict
    assert verdict.validation is not None
    report = build_report(verdict, title="Cd validation")
    data = report.section("asme-vv20").data

    for key in ("f_sim", "f_exp", "u_num", "u_input", "u_val", "error", "ratio", "passed"):
        assert data[key] == getattr(verdict.validation, key), key
    assert f"{verdict.validation.u_val:.6g}" in report.markdown


def test_error_judgement_states_the_acceptance_verdict(validated_verdict):
    verdict = validated_verdict
    report = build_report(verdict, title="Cd validation")
    data = report.section("error-judgement").data

    expected_rel = abs(verdict.validation.f_sim - verdict.validation.f_exp) / abs(verdict.validation.f_exp)
    assert data["relative_error"] == pytest.approx(expected_rel, rel=0, abs=0)
    assert data["acceptance"] == DEFAULT_ERROR_ACCEPTANCE
    assert data["within_acceptance"] is (expected_rel < DEFAULT_ERROR_ACCEPTANCE)
    assert "error criterion" in report.section("error-judgement").markdown
    # the number is printed and the answer to "is it < 3 %?" is explicit
    assert f"{expected_rel * 100:.4g} %" in report.section("error-judgement").markdown


# ---------------------------------------------------------------------------
# 2. no silent blanks / no placeholders
# ---------------------------------------------------------------------------


def test_no_template_placeholder_survives(validated_verdict, case_metadata):
    report = build_report(validated_verdict, title="Cd validation", metadata=case_metadata)
    for token in PLACEHOLDER_TOKENS:
        assert token not in report.markdown, token


def test_missing_reference_withholds_validation_and_report(convergence_verdict, case_metadata):
    verdict = convergence_verdict
    report = build_report(verdict, title="Cd convergence", metadata=case_metadata)

    assert report.section("asme-vv20").status == "WITHHELD"
    assert report.section("error-judgement").status == "WITHHELD"
    assert "WITHHELD" in report.section("asme-vv20").markdown
    # a report with a withheld required section can never claim PASS
    assert report.status != "PASS"
    assert STATUS_ORDER[report.status] >= STATUS_ORDER["WITHHELD"]
    assert set(report.coverage["required_withheld"]) == {"asme-vv20", "error-judgement"}


def test_missing_case_metadata_withholds_case_section(validated_verdict):
    report = build_report(validated_verdict, title="Cd validation")
    assert report.section("case").status == "WITHHELD"
    assert "cannot invent" in report.section("case").markdown
    assert "case" in report.coverage["required_withheld"]


def test_complete_metadata_fills_every_required_section(validated_verdict, case_metadata):
    report = build_report(validated_verdict, title="Cd validation", metadata=case_metadata)
    assert report.coverage["required_withheld"] == []
    assert report.coverage["ratio"] == 1.0
    assert report.status == report.verdict_status


def test_failed_validation_is_not_laundered(convergence_verdict):
    """|E| far outside U_val must stay FAIL through the report."""
    verdict = evaluate(_levels(), quantity="Cd", f_exp=5.0, u_input=0.01)
    assert verdict.status == "FAIL"
    report = build_report(
        verdict,
        title="Cd validation",
        metadata={"parameters": [{"name": "f_inf", "value": 2.0, "unit": "-"}]},
    )
    assert report.status == "FAIL"
    assert report.verdict_status == "FAIL"
    assert "FAIL" in report.section("conclusion").markdown
    # and the error judgement must say the criterion is NOT met
    assert report.section("error-judgement").data["within_acceptance"] is False


def test_two_level_study_still_renders_with_unobservable_order():
    verdict = evaluate(_levels(hs=(0.4, 0.2)), quantity="Cd")
    report = build_report(verdict, title="two grids")
    data = report.section("grid-convergence").data
    assert data["n_levels"] == 2
    assert data["order_p"] is None
    assert "n/a" in report.markdown
    assert report.section("grid-convergence").status == "FILLED"


def test_optional_sections_default_to_not_requested(validated_verdict, case_metadata):
    report = build_report(validated_verdict, title="Cd validation", metadata=case_metadata)
    assert report.section("resolvability").status == "FILLED"
    assert report.section("resolvability").required is False
    assert "not requested" in report.section("resolvability").markdown
    assert "not requested" in report.section("cross-solver").markdown


def test_cross_solver_metadata_is_rendered():
    verdict = evaluate(_levels(), quantity="Cd")
    report = build_report(
        verdict,
        title="cross solver",
        metadata={
            "parameters": [{"name": "f_inf", "value": 2.0, "unit": "-"}],
            "cross_solver": {
                "consensus": 2.0,
                "spread": 0.002,
                "status": "PASS",
                "samples": [
                    {"solver": "TensorFVM", "f": 2.0, "sigma": 0.001},
                    {"solver": "TensorFEM", "f": 2.002, "sigma": 0.001},
                ],
            },
        },
    )
    body = report.section("cross-solver").markdown
    assert "TensorFVM" in body and "TensorFEM" in body
    assert "consensus = 2" in body


# ---------------------------------------------------------------------------
# 3. provenance
# ---------------------------------------------------------------------------


def test_provenance_digests_are_deterministic(validated_verdict, case_metadata):
    a = build_report(validated_verdict, title="Cd validation", metadata=case_metadata)
    b = build_report(validated_verdict, title="Cd validation", metadata=case_metadata)
    assert a.markdown == b.markdown
    assert a.provenance["source_digest"] == b.provenance["source_digest"]
    assert a.provenance["body_digest"] == b.provenance["body_digest"]


def test_body_digest_matches_the_body(validated_verdict, case_metadata):
    report = build_report(validated_verdict, title="Cd validation", metadata=case_metadata)
    body = "\n".join(s.markdown for s in report.sections)
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == report.provenance["body_digest"]


def test_source_digest_tracks_the_verdict(convergence_verdict):
    other = evaluate(_levels(c=3.05), quantity="Cd")
    a = build_report(convergence_verdict, title="t")
    b = build_report(other, title="t")
    assert a.provenance["source_digest"] != b.provenance["source_digest"]


def test_digests_appear_in_the_markdown(validated_verdict):
    report = build_report(validated_verdict, title="Cd validation")
    assert report.provenance["source_digest"] in report.markdown
    assert report.provenance["body_digest"] in report.markdown


# ---------------------------------------------------------------------------
# guards: hand-filling and bad input are rejected, not rendered
# ---------------------------------------------------------------------------


def test_hand_filled_placeholder_in_metadata_is_rejected(validated_verdict):
    with pytest.raises(ValueError, match="placeholder"):
        build_report(
            validated_verdict,
            title="Cd validation",
            metadata={"parameters": [{"name": "f_inf", "value": "[value]", "unit": "-"}]},
        )


def test_hand_filled_placeholder_in_reference_source_is_rejected(validated_verdict):
    with pytest.raises(ValueError, match="placeholder"):
        build_report(
            validated_verdict,
            title="Cd validation",
            metadata={"reference_source": "[Reference]", "parameters": [{"name": "x", "value": 1}]},
        )


def test_non_verdict_input_is_rejected():
    with pytest.raises(TypeError):
        build_report({"status": "PASS"}, title="t")  # type: ignore[arg-type]


def test_empty_title_is_rejected(validated_verdict):
    with pytest.raises(ValueError):
        build_report(validated_verdict, title="   ")


def test_non_positive_acceptance_is_rejected(validated_verdict):
    with pytest.raises(ValueError):
        build_report(validated_verdict, title="t", error_acceptance=0.0)


def test_render_markdown_is_pure_and_ordered(validated_verdict, case_metadata):
    report = build_report(validated_verdict, title="Cd validation", metadata=case_metadata)
    md = render_markdown("Cd validation", report.sections, report.provenance)
    assert md.startswith("# Cd validation\n")
    order = [md.index(s.markdown.splitlines()[0]) for s in report.sections]
    assert order == sorted(order)


def test_as_dict_roundtrips_json(validated_verdict, case_metadata):
    import json

    report = build_report(validated_verdict, title="Cd validation", metadata=case_metadata)
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["schema"] == "tensorphysics.vv.report/1"
    assert payload["status"] == report.status
    assert payload["provenance"]["source_digest"] == report.provenance["source_digest"]
    assert [s["key"] for s in payload["sections"]][0] == "case"