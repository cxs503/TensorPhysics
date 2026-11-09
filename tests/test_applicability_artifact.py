"""Tests of the consumer side of the envelope mechanism: artifact round trip.

The artefact written by ``python -m tensorphysics.applicability --out`` is only
useful if a later process can reload it and gate a grade *without* re-probing.
These tests pin that contract:

* the reloaded envelopes are byte-identical to the probed ones (so the stored
  suite digest still verifies);
* classification is the same code, therefore the same verdicts, without the
  evaluators being importable;
* a modified artifact is rejected (tamper evidence, fail closed);
* the offline gate keeps the ``OUT -> OUT_OF_DOMAIN`` / ``UNKNOWN -> WITHHELD``
  rule of the live path.

Capability ids, axes and spans are injected by the synthetic benchmark registry:
nothing here is domain vocabulary.
"""
from __future__ import annotations

import json
import pathlib
import time

import pytest

from tensorphysics.applicability import (ArtifactError, DomainStatus, GradeWriteError, assign_grade,
                                         gate, gate_queries, load_envelopes, probe_registry,
                                         suite_digest)
from tensorphysics.applicability.__main__ import main
from tensorphysics.applicability.benchmarks import synthetic_registry


def _artifact_dict() -> dict:
    """Probe the synthetic registry and serialize it exactly like the CLI does."""
    suite = probe_registry(synthetic_registry(), rel_tol=1e-3)
    return suite.to_dict()


@pytest.fixture(scope="module")
def artifact() -> dict:
    return _artifact_dict()


@pytest.fixture(scope="module")
def artifact_path(tmp_path_factory, artifact) -> pathlib.Path:
    path = tmp_path_factory.mktemp("artifact") / "envelopes.json"
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------

def test_round_trip_is_exact_and_keeps_the_suite_digest(artifact, artifact_path):
    loaded = load_envelopes(artifact_path)
    original = {item["capability_id"]: item["envelope"] for item in artifact["capabilities"]}
    assert set(loaded) == set(original)
    for capability_id, envelope in loaded.items():
        assert envelope.to_dict() == original[capability_id]
    assert suite_digest(loaded) == artifact["suite_digest"]


def test_integer_serialized_numbers_are_not_floatified():
    """A snapped boundary can serialize as ``int``; reloading must keep the type.

    ``1`` and ``1.0`` compare equal but serialize differently, so a float-cast in
    the loader would silently change the suite digest of a valid artifact.
    """
    from tensorphysics.applicability.envelope import Interval
    interval = Interval.from_dict({"low": 2, "high": 10, "low_open": True,
                                   "high_open": False})
    assert isinstance(interval.low, int) and isinstance(interval.high, int)
    loaded = load_envelopes(_artifact_dict())
    assert suite_digest(loaded) == _artifact_dict()["suite_digest"]


def test_numeric_types_survive_the_round_trip(artifact):
    loaded = load_envelopes(artifact)
    original = {item["capability_id"]: item["envelope"] for item in artifact["capabilities"]}
    for capability_id, envelope in loaded.items():
        for name, axis in envelope.axes.items():
            source = original[capability_id]["axes"][name]
            for got, want in zip([b.value for b in axis.boundaries],
                                 [b["value"] for b in source["boundaries"]]):
                assert type(got) is type(want)
            for got, want in zip([iv.high for iv in axis.intervals],
                                 [iv["high"] for iv in source["intervals"]]):
                assert type(got) is type(want)
    assert suite_digest(loaded) == artifact["suite_digest"]


def test_suite_digest_is_reproducible_across_runs():
    """The digest identifies content, not the wall clock.

    Two runs one second apart have different ``generated_utc`` but must produce
    the same suite digest -- otherwise the digest printed in a report could never
    be reproduced, and a reloaded artifact could never be verified.
    """
    first = _artifact_dict()
    time.sleep(1.1)
    second = _artifact_dict()
    assert first["generated_utc"] != second["generated_utc"]
    assert first["suite_digest"] == second["suite_digest"]
    assert {item["envelope"]["generated_utc"] for item in first["capabilities"]} != \
           {item["envelope"]["generated_utc"] for item in second["capabilities"]}


def test_round_trip_classification_matches_the_live_envelope(artifact):
    suite = probe_registry(synthetic_registry(), rel_tol=1e-3)
    live = suite.envelopes()
    loaded = load_envelopes(artifact)
    checked = 0
    for capability_id, live_envelope in live.items():
        for name, axis in live_envelope.axes.items():
            span = axis.span_high - axis.span_low
            probes = [axis.span_low, axis.span_low + 0.01 * span,
                      axis.span_low + 0.25 * span, axis.span_low + 0.5 * span,
                      axis.span_low + 0.75 * span, axis.span_high,
                      axis.span_high + 0.5 * span]           # last one: outside the span
            for value in probes:
                point = dict(live_envelope.nominal)
                point[name] = float(value)
                live_verdict = live_envelope.classify(point, verify=False)
                loaded_verdict = loaded[capability_id].classify(point, verify=False)
                assert loaded_verdict.domain_status is live_verdict.domain_status, (
                    capability_id, name, value)
                checked += 1
    assert checked >= 50


def test_multi_axis_capability_round_trips_every_axis(artifact):
    loaded = load_envelopes(artifact)
    envelope = loaded["syn-two-axis-product"]
    assert set(envelope.axes) == {"x", "y"}
    mixed = envelope.classify({"x": envelope.axes["x"].span_low + 1e-9, "y": 2.0}, verify=False)
    assert mixed.domain_status in (DomainStatus.IN, DomainStatus.OUT)


# ---------------------------------------------------------------------------
# fail closed on a damaged artifact
# ---------------------------------------------------------------------------

def test_modified_envelope_is_rejected(artifact):
    tampered = json.loads(json.dumps(artifact))
    target = next(item for item in tampered["capabilities"]
                  if item["capability_id"] == "syn-square-window")
    interval = target["envelope"]["axes"]["x"]["intervals"][0]
    interval["high"] = float(interval["high"]) * 10.0
    with pytest.raises(ArtifactError, match="digest mismatch"):
        load_envelopes(tampered)
    # the escape hatch is explicit, and the loaded envelope really is wider
    relaxed = load_envelopes(tampered, verify_digest=False)
    assert relaxed["syn-square-window"].axes["x"].intervals[0].high == interval["high"]


def test_unexpected_schema_is_rejected(artifact):
    wrong = dict(artifact)
    wrong["schema"] = "tensorphysics.applicability.envelopes.v0"
    with pytest.raises(ArtifactError, match="unexpected artifact schema"):
        load_envelopes(wrong)


def test_artifact_without_digest_is_rejected(artifact):
    stripped = dict(artifact)
    stripped.pop("suite_digest")
    with pytest.raises(ArtifactError, match="no suite_digest"):
        load_envelopes(stripped)


def test_malformed_envelope_is_rejected(artifact):
    broken = json.loads(json.dumps(artifact))
    broken["capabilities"][0]["envelope"].pop("axes")
    with pytest.raises(ArtifactError, match="missing required field"):
        load_envelopes(broken, verify_digest=False)


def test_missing_artifact_path_is_reported(tmp_path):
    with pytest.raises(ArtifactError, match="artifact not found"):
        load_envelopes(tmp_path / "does-not-exist.json")


# ---------------------------------------------------------------------------
# the offline gate
# ---------------------------------------------------------------------------

def test_offline_gate_is_fail_closed(artifact):
    loaded = load_envelopes(artifact)
    envelope = loaded["syn-square-window"]
    span = envelope.axes["x"]
    interval = span.intervals[0]
    inside = 0.5 * (interval.low + interval.high)

    accepted = gate(loaded, "syn-square-window", {"x": inside}, "REAL")
    assert accepted.accepted and accepted.effective == "REAL"

    beyond = span.span_high + 1.0                        # beyond the searched span
    withheld = gate(loaded, "syn-square-window", {"x": beyond}, "REAL")
    assert withheld.effective == "WITHHELD" and withheld.accepted is False

    outside_interval = span.span_high - 1e-9             # inside span, outside interval
    refused = gate(loaded, "syn-square-window", {"x": outside_interval}, "REAL", strict=False)
    assert refused.effective == "OUT_OF_DOMAIN" and refused.accepted is False

    empty = loaded["syn-empty-domain"]
    out_point = {"x": 0.5 * (empty.axes["x"].span_low + empty.axes["x"].span_high)}
    with pytest.raises(GradeWriteError):
        gate(loaded, "syn-empty-domain", out_point, "REAL")
    lenient = gate(loaded, "syn-empty-domain", out_point, "REAL", strict=False)
    assert lenient.effective == "OUT_OF_DOMAIN" and lenient.accepted is False


def test_loaded_verdict_matches_live_assign_grade(artifact):
    suite = probe_registry(synthetic_registry(), rel_tol=1e-3)
    loaded = load_envelopes(artifact)
    live_envelope = suite.envelopes()["syn-square-window"]
    point = {"x": live_envelope.axes["x"].span_high + 1.0}
    live_record = assign_grade(live_envelope.classify(point, verify=False), "REAL")
    offline_record = gate(loaded, "syn-square-window", point, "REAL")
    assert live_record.to_dict() == offline_record.to_dict()


def test_batch_gate_reports_errors_instead_of_aborting(artifact):
    loaded = load_envelopes(artifact)
    answers = gate_queries(loaded, [
        {"label": "ok", "capability_id": "syn-square-window", "point": {"x": 0.5}},
        {"label": "unknown-capability", "capability_id": "no-such-capability",
         "point": {"x": 0.5}},
        {"label": "missing-axis", "capability_id": "syn-square-window", "point": {}},
    ])
    assert [a["label"] for a in answers] == ["ok", "unknown-capability", "missing-axis"]
    assert "error" in answers[1] and "not in artifact" in answers[1]["error"]
    assert "missing axes: x" in answers[2]["error"]


# ---------------------------------------------------------------------------
# the offline CLI path
# ---------------------------------------------------------------------------

def test_offline_cli_answers_from_the_artifact_without_probing(artifact_path, tmp_path,
                                                               monkeypatch):
    """No probe, and no dependence on the recorded config/evaluator location."""
    import tensorphysics.applicability.__main__ as cli

    def _boom(*args, **kwargs):                      # pragma: no cover - must not run
        raise AssertionError("the offline path must not re-run the probe")

    monkeypatch.setattr(cli, "probe_registry", _boom)
    out = tmp_path / "verdicts.json"
    code = cli.main(["--envelope", str(artifact_path), "--quiet", "--out", str(out),
                     "--query", json.dumps({"syn-square-window": {"x": 0.5}})])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "tensorphysics.applicability.verdicts.v1"
    assert payload["artifact_digest"]
    assert payload["query_summary"]["total"] == 1
    assert payload["queries"][0]["grade"]["accepted"] is True


def test_offline_cli_requires_a_query(artifact_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["--envelope", str(artifact_path)])
    assert excinfo.value.code == 2