"""Reproducibility contract for the criterion drift-audit artifacts.

``tests/test_criterion_drift_hardening.py`` consumes the two committed JSON
artifacts; this module guards the other direction — that those artifacts can be
*regenerated* from the tree, and that the v1 baseline is the frozen pre-hardening
code rather than something authored to look bad.

Kept deliberately small: provenance, reproducibility, and the direction of the
claimed delta.  The metrics themselves are asserted in the hardening suite.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign.criterion_registry import bench  # noqa: E402

BENCH_DIR = ROOT / "benchmarks" / "criterion_drift_audit"
ARTIFACT_DIR = ROOT / "artifacts" / "criterion_drift_audit"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "criterion_drift_audit_generate", BENCH_DIR / "generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_generator()


class TestBaselineProvenance:
    def test_snapshot_matches_pinned_digest(self, gen):
        """The v1 snapshot must still be the byte-identical frozen copy."""
        digest = hashlib.sha256(gen.LEGACY_PATH.read_bytes()).hexdigest()
        assert digest == gen.EXPECTED_SHA256

    def test_snapshot_is_the_pre_hardening_extractor(self):
        """v1 is structurally weaker: it lacks the hardened field parsers."""
        v1 = (BENCH_DIR / "legacy_manifest_v1.py").read_text(encoding="utf-8")
        current = (ROOT / "src" / "tensorphysics" / "campaign"
                   / "criterion_registry" / "manifest.py").read_text(encoding="utf-8")
        for helper in ("_extract_level", "_extract_denominator", "_extract_exclusion"):
            assert helper not in v1, f"{helper} belongs to the hardened extractor"
            assert helper in current

    def test_digest_guard_fails_closed(self, gen, tmp_path, monkeypatch):
        """A modified snapshot must be rejected, not silently used."""
        tampered = tmp_path / "legacy_manifest_v1.py"
        tampered.write_text(gen.LEGACY_PATH.read_text(encoding="utf-8") + "\n# x\n",
                            encoding="utf-8")
        monkeypatch.setattr(gen, "LEGACY_PATH", tampered)
        with pytest.raises(gen.LegacySnapshotTampered):
            gen._load_legacy()


class TestArtifactsAreReproducible:
    def test_baseline_record_is_reproduced_bit_for_bit(self, gen):
        """Re-running the v1 path reproduces the committed baseline exactly."""
        record = bench.run_benchmark(label="v1_legacy", extractor=gen.legacy_extractor)
        committed = (ARTIFACT_DIR / "bench_v1_legacy.json").read_text(encoding="utf-8")
        assert gen._dump(record) == committed

    def test_hardened_record_is_reproduced_bit_for_bit(self, gen):
        record = bench.run_benchmark(label="v2_hardened")
        committed = (ARTIFACT_DIR / "bench_v2_hardened.json").read_text(encoding="utf-8")
        assert gen._dump(record) == committed

    @pytest.mark.parametrize("name", ["bench_v2_hardened.json", "bench_v1_legacy.json"])
    def test_artifacts_are_pinned_to_the_current_corpus(self, name):
        """Numbers quoted by the tests must come from today's corpus."""
        record = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
        assert record["corpus_sha256"] == bench.corpus_fingerprint()

    def test_baseline_is_weaker_by_a_wide_margin(self, gen):
        """Guard the direction of the claimed improvement (not a coin flip)."""
        hardened = json.loads((ARTIFACT_DIR / "bench_v2_hardened.json").read_text())
        legacy = json.loads((ARTIFACT_DIR / "bench_v1_legacy.json").read_text())
        assert hardened["extraction"]["f1"] - legacy["extraction"]["f1"] > 0.10
        assert legacy["all_gates_pass"] is False
        assert hardened["all_gates_pass"] is True