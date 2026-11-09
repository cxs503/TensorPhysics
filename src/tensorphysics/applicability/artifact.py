"""Consumer side of the envelope machinery: reload a persisted artifact.

Stage 2-4 of the machinery *writes* an envelope artifact (``python -m
tensorphysics.applicability --out ...``).  Until this module existed that
artifact was write-only: gating a later result against the mechanism required
re-running the probe, i.e. importing every evaluator and paying the scan again
inside the publishing process.  :func:`load_envelopes` closes that loop:

* the artifact is reconstructed into the same frozen dataclasses the probe
  produced (:meth:`~.envelope.ApplicabilityEnvelope.from_dict`), so the
  classification logic is literally the same code, not a copy;
* the stored ``suite_digest`` is recomputed from the reloaded envelopes and
  compared (:func:`suite_digest`); a hand-edited envelope is rejected instead
  of silently certified -- tamper evidence, fail closed;
* :func:`gate` applies the unchanged out-of-domain rule
  (:func:`~.envelope.assign_grade`) to an offline query.

Nothing here is capability specific: capability ids, axes and thresholds come
from the artifact that configuration produced.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Mapping

from .envelope import ApplicabilityEnvelope, EnvelopeFormatError, GradeRecord, Verdict, assign_grade
from .probe import _digest, envelope_content_material

__all__ = ["ARTIFACT_SCHEMA", "ArtifactError", "load_envelopes", "suite_digest", "gate",
           "gate_queries"]

ARTIFACT_SCHEMA = "tensorphysics.applicability.envelopes.v1"


class ArtifactError(RuntimeError):
    """A persisted envelope artifact cannot be trusted or consumed."""


def suite_digest(envelopes: Mapping[str, ApplicabilityEnvelope]) -> str:
    """Digest of a set of envelopes, identical to ``EnvelopeSuite.digest``.

    The formula lives in :mod:`.probe` (one definition, no drift): the digest of
    ``{capability_id: content(envelope)}`` with sorted keys, where the volatile
    ``generated_utc`` is excluded so the same envelopes always hash the same.
    """
    return _digest({k: envelope_content_material(v.to_dict())
                    for k, v in envelopes.items()})


def _read_payload(source: Any) -> tuple[dict[str, Any], str]:
    if isinstance(source, Mapping):
        return dict(source), "<mapping>"
    if isinstance(source, (str, pathlib.Path)):
        text = str(source)
        path = pathlib.Path(text)
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"{path}: not valid JSON ({exc})") from None
            return payload, str(path)
        if text.lstrip().startswith("{"):
            try:
                return json.loads(text), "<inline>"
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"inline artifact is not valid JSON ({exc})") from None
        raise ArtifactError(f"artifact not found: {path}")
    raise ArtifactError(f"unsupported artifact source type: {type(source).__name__}")


def load_envelopes(source: Any, *, verify_digest: bool = True) -> dict[str, ApplicabilityEnvelope]:
    """Load an envelope artifact into ``{capability_id: ApplicabilityEnvelope}``.

    ``source`` may be a path, a JSON string, or an already-parsed artifact
    mapping.  The artifact must carry the expected ``schema``; with
    ``verify_digest`` (default) the recomputed suite digest must equal the stored
    one, otherwise :class:`ArtifactError` is raised -- a modified envelope is
    never certified.
    """
    payload, label = _read_payload(source)
    schema = payload.get("schema")
    if schema != ARTIFACT_SCHEMA:
        raise ArtifactError(f"{label}: unexpected artifact schema {schema!r} "
                            f"(expected {ARTIFACT_SCHEMA!r})")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        raise ArtifactError(f"{label}: artifact contains no capabilities")

    envelopes: dict[str, ApplicabilityEnvelope] = {}
    for index, item in enumerate(capabilities):
        if not isinstance(item, Mapping) or not isinstance(item.get("envelope"), Mapping):
            raise ArtifactError(f"{label}: capabilities[{index}] has no 'envelope' mapping")
        try:
            envelope = ApplicabilityEnvelope.from_dict(item["envelope"])
        except EnvelopeFormatError as exc:
            raise ArtifactError(f"{label}: capabilities[{index}]: {exc}") from None
        if envelope.capability_id in envelopes:
            raise ArtifactError(f"{label}: duplicate capability {envelope.capability_id!r}")
        envelopes[envelope.capability_id] = envelope

    if verify_digest:
        stored = payload.get("suite_digest")
        if not stored:
            raise ArtifactError(f"{label}: artifact carries no suite_digest; cannot verify "
                                f"the envelopes (pass verify_digest=False to override)")
        actual = suite_digest(envelopes)
        if actual != stored:
            raise ArtifactError(f"{label}: suite digest mismatch ({actual} != {stored}): the "
                                f"artifact was modified after it was written")
    return envelopes


def gate(envelopes: Mapping[str, ApplicabilityEnvelope], capability_id: str,
         point: Mapping[str, float], requested: str = "REAL", *, strict: bool = True
         ) -> GradeRecord:
    """Classify ``point`` from a loaded envelope and apply the grading rule.

    No evaluator is imported and no scan is repeated: the verdict is decided by
    the persisted intervals plus (fail-closed) the searched spans of the
    artifact.  ``IN`` -> ``requested`` passes; ``UNKNOWN`` -> ``WITHHELD``;
    ``OUT`` -> ``OUT_OF_DOMAIN`` and ``REAL`` refused (raises when ``strict``).
    """
    try:
        envelope = envelopes[capability_id]
    except KeyError:
        known = ", ".join(sorted(envelopes)) or "<none>"
        raise ArtifactError(f"capability {capability_id!r} is not in the artifact "
                            f"(available: {known})") from None
    return assign_grade(envelope.classify(point, verify=False), requested, strict=strict)


def gate_queries(envelopes: Mapping[str, ApplicabilityEnvelope], queries: Any, *,
                 requested: str = "REAL", strict: bool = False
                 ) -> list[dict[str, Any]]:
    """Answer a list of queries offline; one record per query, never raising.

    Accepted forms: ``{capability_id: {axis: value}}`` or a list of
    ``{"label", "capability_id", "point"}`` records (the same two forms the CLI
    accepts).  ``strict`` is ``False`` by default so a batch report can record a
    refusal instead of aborting on the first out-of-domain point; the refusal is
    still ``accepted=False`` with grade ``OUT_OF_DOMAIN``.
    """
    if queries is None:
        return []
    if isinstance(queries, Mapping):
        records = [{"label": str(k), "capability_id": str(k), "point": dict(v)}
                   for k, v in queries.items()]
    else:
        records = [dict(q) for q in queries]
    out: list[dict[str, Any]] = []
    for record in records:
        capability_id = str(record["capability_id"])
        point = {str(a): float(v) for a, v in dict(record["point"]).items()}
        label = str(record.get("label", capability_id))
        envelope = envelopes.get(capability_id)
        if envelope is None:
            out.append({"label": label, "capability_id": capability_id, "point": point,
                        "error": f"capability not in artifact (available: "
                                 f"{', '.join(sorted(envelopes)) or '<none>'})"})
            continue
        missing = sorted(set(envelope.axes) - set(point))
        if missing:
            out.append({"label": label, "capability_id": capability_id, "point": point,
                        "error": f"query is missing axes: {', '.join(missing)}"})
            continue
        verdict: Verdict = envelope.classify(point, verify=False)
        grade = assign_grade(verdict, requested, strict=strict)
        out.append({"label": label, "capability_id": capability_id, "point": point,
                    "verdict": verdict.to_dict(), "grade": grade.to_dict()})
    return out