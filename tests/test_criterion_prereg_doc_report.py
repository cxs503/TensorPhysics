"""Tests for the standalone document renderer and the report pipeline.

Two deliverables are gated here:

* ``scripts/render_criterion_prereg_doc.py`` -- the checked-in page
  ``docs/CRITERION_PREREGISTRATION.md`` must equal a fresh render of the
  checked-in evaluation record (``artifacts/criterion_prereg/eval.json``), so a
  hand-edited digit is a test failure, and ``--check`` must reproduce that
  verdict through the CLI (exit 1 on drift, exit 0 when fresh);
* ``scripts/publish_criterion_prereg_report.py`` -- the paper-level payload must
  carry the six report elements, four inline PNG figures, and must fail closed
  on a missing/short/undecodable figure or an unrendered ``{placeholder}`` /
  ``None`` / non-finite number.

The PNG fixtures are written by a ~20-line dependency-free encoder, so these
tests do not need matplotlib: the report pipeline is checked offline.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import publish_criterion_prereg_report as pub  # noqa: E402
import render_criterion_prereg_doc as rend  # noqa: E402

EVAL = ROOT / "artifacts" / "criterion_prereg" / "eval.json"
DOC = ROOT / "docs" / "CRITERION_PREREGISTRATION.md"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _png(width: int = 60, height: int = 60, seed: int = 7) -> bytes:
    """A valid, incompressible-enough RGB PNG (no third-party dependencies)."""
    state = seed
    rows = bytearray()
    for _ in range(height):
        rows.append(0)  # filter type 0
        for _ in range(width * 3):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            rows.append(state & 0xFF)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(rows), 0)) + chunk(b"IEND", b""))


@pytest.fixture(scope="module")
def eval_record() -> dict:
    assert EVAL.exists(), f"the evaluation record must be checked in: {EVAL}"
    return json.loads(EVAL.read_text(encoding="utf-8"))


@pytest.fixture()
def art_dir(tmp_path: Path, eval_record: dict) -> Path:
    """An artifacts directory with four valid PNGs named as eval.json claims."""
    out = tmp_path / "artifacts"
    out.mkdir()
    for i, rel in enumerate(eval_record["detail"]["figures"].values()):
        (out / Path(rel).name).write_bytes(_png(seed=7 + i))
    return out


# --------------------------------------------------------------------------- #
# (a) renderer: purity, determinism, freshness guard
# --------------------------------------------------------------------------- #
def test_render_is_deterministic_and_pure(eval_record):
    first = rend.render(EVAL)
    second = rend.render(EVAL)
    assert first == second
    assert first.startswith("# Criterion pre-registration")
    # rendering must not write the checked-in page
    before = DOC.read_bytes()
    rend.render(EVAL)
    assert DOC.read_bytes() == before


def test_checked_in_doc_equals_fresh_render():
    ok, message = rend.check(EVAL, DOC)
    assert ok, message


def test_check_fails_closed_on_missing_inputs(tmp_path):
    ok, message = rend.check(tmp_path / "nope.json", DOC)
    assert not ok and "missing evaluation record" in message
    ok, message = rend.check(EVAL, tmp_path / "nope.md")
    assert not ok and "missing document" in message


def test_hand_edited_digit_is_detected(tmp_path):
    """A single hand-edited digit must break freshness, and say where."""
    shipped = DOC.read_text(encoding="utf-8")
    assert "| `20261109` | 1.0000 |" in shipped
    edited = tmp_path / "edited.md"
    edited.write_text(shipped.replace("| `20261109` | 1.0000 |", "| `20261109` | 0.9999 |", 1),
                      encoding="utf-8")
    ok, message = rend.check(EVAL, edited)
    assert not ok
    assert "drifted" in message and "20261109" in message


def test_cli_check_exit_codes(tmp_path):
    fresh = tmp_path / "fresh.md"
    fresh.write_text(rend.render(EVAL), encoding="utf-8")
    good = subprocess.run([sys.executable, str(SCRIPTS / "render_criterion_prereg_doc.py"),
                           "--check", "--eval", str(EVAL), "--out", str(fresh)],
                          capture_output=True, text=True)
    assert good.returncode == 0, good.stdout + good.stderr
    assert "OK:" in good.stdout

    bad = tmp_path / "bad.md"
    bad.write_text(rend.render(EVAL).replace("F1", "F2", 1), encoding="utf-8")
    drifted = subprocess.run([sys.executable, str(SCRIPTS / "render_criterion_prereg_doc.py"),
                              "--check", "--eval", str(EVAL), "--out", str(bad)],
                             capture_output=True, text=True)
    assert drifted.returncode == 1
    assert "DRIFT:" in drifted.stdout


def test_cli_stdout_is_the_written_page():
    out = subprocess.run([sys.executable, str(SCRIPTS / "render_criterion_prereg_doc.py"),
                          "--stdout", "--eval", str(EVAL)],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert out.stdout == DOC.read_text(encoding="utf-8")


def test_eval_record_is_reproducible_from_scratch(tmp_path):
    """The checked-in record is code output, not a hand-saved snapshot.

    Re-running the evaluation into a scratch directory must reproduce both
    ``eval.json`` and the page byte for byte; if the corpora, the extractor or
    a metric changes without the record being regenerated, this fails.
    """
    doc = tmp_path / "page.md"
    proc = subprocess.run([sys.executable, str(SCRIPTS / "run_criterion_prereg_eval.py"),
                           "--out-dir", str(tmp_path / "artifacts"), "--doc", str(doc)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    fresh = (tmp_path / "artifacts" / "eval.json").read_text(encoding="utf-8")
    assert fresh == EVAL.read_text(encoding="utf-8")
    assert doc.read_text(encoding="utf-8") == DOC.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# (b) report payload: six elements, inline figures, numeric provenance
# --------------------------------------------------------------------------- #
def test_payload_passes_validation_and_has_six_elements(eval_record, art_dir, tmp_path):
    payload = pub.build_payload(eval_record, art_dir)
    assert pub.validate(payload) == []
    md = payload["markdown"]
    for marker in pub.SIX_MARKERS:
        assert marker in md
    assert payload["mission_id"] == pub.MISSION_ID
    assert len(payload["figures"]) == len(eval_record["detail"]["figures"]) == 4
    for fig in payload["figures"]:
        raw = base64.b64decode(fig["png_base64"])
        assert raw.startswith(pub.PNG_SIGNATURE)
        assert fig["caption"], fig["name"]


def test_payload_numbers_come_from_the_evaluation_record(eval_record, art_dir):
    """Every headline number in the report is a format of a field of eval.json."""
    payload = pub.build_payload(eval_record, art_dir)
    md = payload["markdown"]
    acc, det = eval_record["acceptance"], eval_record["detail"]
    expected = [
        f"{acc['a_detection_24cases']['recall']:.4f}",
        f"{acc['a_detection_24cases']['precision']:.4f}",
        f"{acc['b_extractor_field_f1']['f1']:.4f}",
        f"{acc['b_extractor_field_f1']['heldout_seed_f1']:.4f}",
        f"{acc['b_extractor_field_f1']['ood_undelimited_cjk_f1']:.4f}",
        f"{det['baseline_text_hash']['precision']:.4f}",
        f"{det['real_goal_demo']['kpi']['prereg_coverage']:.3f}",
    ]
    for value in expected:
        assert value in md, value
    # margins are computed, not typed: recompute and look them up
    margin = acc["a_detection_24cases"]["recall"] - acc["a_detection_24cases"]["thresholds"]["recall"]
    assert f"{margin:+.4f}" in md


def test_validate_rejects_missing_figure(eval_record, art_dir):
    payload = pub.build_payload(eval_record, art_dir)
    payload["figures"] = payload["figures"][:3]
    problems = pub.validate(payload)
    assert any("figures:" in p for p in problems)


def test_validate_rejects_undecodable_and_short_png(eval_record, art_dir):
    payload = pub.build_payload(eval_record, art_dir)
    payload["figures"][0]["png_base64"] = "not-base64!!"
    payload["figures"][1]["png_base64"] = base64.b64encode(b"\x89PNG\r\n\x1a\nshort").decode()
    problems = pub.validate(payload)
    assert any("decode failed" in p for p in problems)
    assert any("suspiciously small" in p for p in problems)


def test_validate_rejects_unrendered_tokens(eval_record, art_dir):
    payload = pub.build_payload(eval_record, art_dir)
    payload["markdown"] += "\n| x | {det['f1']} | None | nan |\n"
    problems = pub.validate(payload)
    joined = " | ".join(problems)
    assert "unrendered placeholder" in joined
    assert "python None" in joined
    assert "non-finite number" in joined


def test_validate_rejects_wrong_mission(eval_record, art_dir):
    payload = pub.build_payload(eval_record, art_dir)
    payload["mission_id"] = 7803
    assert any("mission_id" in p for p in pub.validate(payload))


def test_publisher_dry_run_cli(tmp_path, art_dir):
    out = tmp_path / "payload.json"
    proc = subprocess.run([sys.executable, str(SCRIPTS / "publish_criterion_prereg_report.py"),
                           "--dry-run", "--eval", str(EVAL), "--art-dir", str(art_dir),
                           "--out", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "payload OK" in proc.stdout
    written = json.loads(out.read_text(encoding="utf-8"))
    for key in pub.REQUIRED_KEYS:
        assert key in written
    # the on-disk copy must be inspectable without dragging the base64 blobs along
    assert all(isinstance(f["png_base64"], str) and f["png_base64"].startswith("<")
               for f in written["figures"])


def test_publisher_fails_closed_when_a_figure_is_missing(tmp_path, art_dir):
    (art_dir / "fig_demo_kpi.png").unlink()
    proc = subprocess.run([sys.executable, str(SCRIPTS / "publish_criterion_prereg_report.py"),
                           "--dry-run", "--eval", str(EVAL), "--art-dir", str(art_dir),
                           "--out", str(tmp_path / "payload.json")],
                          capture_output=True, text=True)
    assert proc.returncode == 1
    assert "FAIL:" in proc.stdout


# --------------------------------------------------------------------------- #
# (c) the published artefacts; the record is checked in, so the figures are too
# --------------------------------------------------------------------------- #
ART = ROOT / "artifacts" / "criterion_prereg"


def test_checked_in_figures_exist_and_are_valid_png(eval_record):
    """``eval.json`` is checked in, therefore the figures it names must be too.

    The page and the report are both rendered from the (record, figures) pair,
    so a artefact filter that drops the binary figures while keeping the JSON
    leaves an unverifiable deliverable.  The only test that touched the real
    figures used to be ``skipif``-guarded, i.e. it went green on a checkout
    where they were missing; the skip is now a hard requirement.
    """
    labels = eval_record["detail"]["figures"]
    assert labels, "eval.json must list its figures"
    for key, rel in labels.items():
        path = ROOT / rel
        assert path.exists(), f"figure {key!r} missing from the checkout: {rel}"
        raw = path.read_bytes()
        assert raw.startswith(pub.PNG_SIGNATURE), f"{rel} is not a PNG"
        assert len(raw) > 1000, f"{rel} is suspiciously small ({len(raw)} bytes)"


def test_real_artifacts_produce_a_publishable_payload(eval_record):
    payload = pub.build_payload(eval_record, ART)
    assert pub.validate(payload) == []
    for fig in payload["figures"]:
        assert base64.b64decode(fig["png_base64"]).startswith(pub.PNG_SIGNATURE)
    assert pub.validate(payload) == []
