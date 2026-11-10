"""Zero-domain-semantics scan and whole-acceptance tests (C6, C1..C6)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorphysics.fail_revival import acceptance, scan

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "tensorphysics" / "fail_revival"
TESTS = sorted((ROOT / "tests").glob("test_fail_revival*.py"))


def test_package_and_tests_carry_no_domain_term():
    result = scan.scan_paths([PACKAGE, *TESTS])
    assert result["vocabulary_loaded"] is True
    assert result["files_scanned"] >= len(TESTS) + 5
    assert result["hits"] == [], json.dumps(result["hits"], ensure_ascii=False)[:600]
    assert result["clean"] is True


def test_scanner_detects_a_planted_term(tmp_path):
    terms, loaded = scan.load_vocabulary()
    assert loaded and terms
    probe = tmp_path / "probe.py"
    probe.write_text("# planted: %s\nx = 1\n" % terms[0], encoding="utf-8")
    hits = scan.scan_paths([probe], vocabulary=terms)
    assert hits["hit_count"] >= 1
    assert hits["clean"] is False
    assert hits["hits"][0]["term"] == terms[0]


def test_missing_vocabulary_is_not_a_silent_pass(tmp_path):
    result = scan.load_vocabulary(tmp_path / "absent.json")
    assert result == ((), False)
    out = scan.scan_paths([PACKAGE], vocabulary=())
    assert out["vocabulary_loaded"] is False
    assert out["clean"] is False


def test_layout_file_holds_the_lexicon_not_the_code():
    lay = json.loads(Path("src/tensorphysics/fail_revival/layout.json").read_text(encoding="utf-8"))
    for key in ("criterion_headings", "verdict_classes", "column_keywords", "attribution_rules"):
        assert key in lay


def test_acceptance_all_criteria_pass(tmp_path):
    result = acceptance.run(out_dir=tmp_path / "fail_revival", sample=20)
    summary = result["summary"]
    assert summary["all_pass"], json.dumps(summary, ensure_ascii=False)
    assert summary["C1_extraction_rate"] == 1.0
    assert summary["C1_criterion_match_rate"] == 1.0
    assert summary["C2_rejected"] == "10/10"
    assert summary["C3_false_triggers"] == 0
    assert summary["C3_max_latency"] <= 1
    assert summary["C4_identical"] is True
    assert summary["C5_overclaim_rows"] == 1
    assert summary["C6_hits"] == 0
    path = Path(result["acceptance_path"])
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == "fail_revival_acceptance.v1"