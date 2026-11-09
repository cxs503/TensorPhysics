"""Regenerate the criterion drift-audit artifacts (reproducibility contract).

The maturity tests in ``tests/test_criterion_drift_hardening.py`` quote numbers
from two JSON artifacts and refuse to run when they are missing:

``artifacts/criterion_drift_audit/bench_v2_hardened.json``
    the shipped (hardened) extractor measured on the frozen benchmark corpus;
``artifacts/criterion_drift_audit/bench_v1_legacy.json``
    the *pre-hardening* extractor on the same corpus — the baseline that makes
    the reported delta meaningful.

Both were filtered out of the original pull request, so a clean checkout could
not reproduce the quoted numbers.  This script re-creates them deterministically:

    python benchmarks/criterion_drift_audit/generate.py

Determinism: the corpus fingerprint (``corpus_sha256``) and every metric are
pure functions of the corpus in ``criterion_registry.bench``; there is no clock,
no randomness and no network.  Re-running on an unchanged tree reproduces the
artifacts byte for byte (sha256 is printed and recorded below).

The v1 baseline is *not* written here: it is the frozen snapshot
``legacy_manifest_v1.py``, a byte-identical copy of the extractor as introduced
by commit 15326d3 (m7397).  Its whole-file sha256 is pinned below and checked
before use, so the baseline cannot be quietly retuned to flatter the hardened
code.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign.criterion_registry import bench  # noqa: E402

LEGACY_PATH = HERE / "legacy_manifest_v1.py"
#: sha256 of the frozen v1 snapshot (provenance: commit 15326d3).
EXPECTED_SHA256 = "82fe953188c730f0bcddac95429b75b2afec3dac1cebe7924c1084fe847d610e"

ARTIFACT_DIR = ROOT / "artifacts" / "criterion_drift_audit"
V2_ARTIFACT = ARTIFACT_DIR / "bench_v2_hardened.json"
V1_ARTIFACT = ARTIFACT_DIR / "bench_v1_legacy.json"


class LegacySnapshotTampered(RuntimeError):
    """The frozen v1 snapshot no longer matches its pinned digest."""


def _load_legacy():
    """Import the frozen v1 extractor, verifying it is untouched."""
    digest = hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise LegacySnapshotTampered(
            f"{LEGACY_PATH.name} sha256={digest} != pinned {EXPECTED_SHA256}"
        )
    sys.path.insert(0, str(HERE))
    import legacy_manifest_v1  # noqa: PLC0415 - local, verified import

    return legacy_manifest_v1


def legacy_extractor(goal_text: str) -> dict:
    """v1 extraction exposed through the benchmark's six-field contract.

    The field projection and value normalisation are *identical* to
    ``bench.extracted_fields`` (same field order, same ``_normalize``); only the
    extraction logic differs.  That keeps the comparison about the parser and
    not about whoever wrote the adapter.
    """
    legacy = _load_legacy()
    criteria = legacy.extract_criteria_from_goal(goal_text)
    if not criteria:
        return {name: None for name in bench.FIELD_ORDER}
    crit = criteria[0]
    out: dict = {}
    for name in bench.FIELD_ORDER:
        fld = crit.fields.get(name)
        out[name] = bench._normalize(fld.value) if fld is not None and fld.present else None
    return out


def _dump(record: dict) -> str:
    """Same serialisation as ``criterion_registry.bench.main``."""
    return json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False)


def main(argv=None) -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    hardened = bench.run_benchmark(label="v2_hardened")
    baseline = bench.run_benchmark(label="v1_legacy", extractor=legacy_extractor)

    # Gates on the *hardened* record; the baseline must be strictly worse on
    # the headline metric and must not pass the suite.
    assert hardened["all_gates_pass"], hardened["gates"]
    assert baseline["corpus_sha256"] == hardened["corpus_sha256"], "corpus drift"
    assert baseline["extraction"]["f1"] < hardened["extraction"]["f1"], "baseline not worse"
    assert baseline["detection"]["recall"] <= hardened["detection"]["recall"]
    assert not baseline["all_gates_pass"], "legacy baseline unexpectedly passes"

    for path, record in ((V2_ARTIFACT, hardened), (V1_ARTIFACT, baseline)):
        blob = _dump(record)
        path.write_text(blob, encoding="utf-8")
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        print(f"wrote {path.relative_to(ROOT)}  sha256={digest}")

    print(
        "delta  extraction.f1 "
        f"{baseline['extraction']['f1']:.4f} -> {hardened['extraction']['f1']:.4f} | "
        "detection.recall "
        f"{baseline['detection']['recall']:.4f} -> {hardened['detection']['recall']:.4f}"
    )
    print(f"corpus_sha256 {hardened['corpus_sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())