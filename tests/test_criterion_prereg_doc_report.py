"""Tests for the standalone document renderer and the report pipeline.

Two deliverables are gated here:

* ``scripts/render_criterion_prereg_doc.py`` -- the checked-in page
  ``docs/CRITERION_PREREGISTRATION.md`` must equal a fresh render of the
  evaluation record *as re-derived from code*: the ``derived`` session fixture
  runs ``scripts/run_criterion_prereg_eval.py`` once into a scratch directory
  and everything below asserts against that output, so a hand-edited digit is a
  test failure and ``--check`` must reproduce the verdict through the CLI
  (exit 1 on drift, exit 0 when fresh);
* ``scripts/publish_criterion_prereg_report.py`` -- the paper-level payload must
  carry the six report elements, four inline PNG figures, and must fail closed
  on a missing/short/undecodable figure or an unrendered ``{placeholder}`` /
  ``None`` / non-finite number.

The evaluation record, its four figures and the mission's acceptance record
(``docs/m8598/outputs/acceptance_criteria.json``) are *also* committed so a
report can cite stable paths.
That copy is treated as a **cache**, not as the source of truth: it is verified
byte-exact against the derivation, and when a merge gate strips it the rest of
the suite still runs -- against freshly derived artefacts -- instead of going
red (or, worse, quietly green behind a ``skipif``).  The derivation is executed,
never skipped, so no number can rot unnoticed.

The PNG fixtures of the payload tests are written by a ~20-line
dependency-free encoder, so those tests do not need matplotlib: the report
pipeline is checked offline.
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


#: The four figures the record names, in the order it lists them.
FIGURES = ("fig_confusion_matrix.png", "fig_baseline_vs_audit.png",
           "fig_extractor_field_agreement.png", "fig_demo_kpi.png")


@pytest.fixture(scope="session")
def derived(tmp_path_factory) -> dict:
    """Re-derive record + figures + page from the evaluation script, once.

    ``run_criterion_prereg_eval.py`` is deterministic (fixed corpora seeds and a
    frozen ``STAMP``, figure labels are ``--out-dir`` independent), so the record
    it writes into the scratch directory is byte-identical to the committed
    cache -- and the page it renders is byte-identical to the checked-in page.
    """
    out = tmp_path_factory.mktemp("derived")
    art = out / "artifacts"
    page = out / "page.md"
    mission = out / "acceptance_criteria.json"
    proc = subprocess.run([sys.executable, str(SCRIPTS / "run_criterion_prereg_eval.py"),
                           "--out-dir", str(art), "--doc", str(page),
                           "--deliverable", str(mission)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ev = art / "eval.json"
    assert ev.exists(), f"the evaluation script wrote no record:\n{proc.stdout}"
    return {"dir": art, "eval": ev, "page": page, "mission": mission,
            "record": json.loads(ev.read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def eval_record(derived) -> dict:
    """The record every assertion below uses: code output, not a saved snapshot."""
    return derived["record"]


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
def test_render_is_deterministic_and_pure(derived):
    first = rend.render(derived["eval"])
    second = rend.render(derived["eval"])
    assert first == second
    assert first.startswith("# Criterion pre-registration")
    # rendering must not write the checked-in page
    before = DOC.read_bytes()
    rend.render(derived["eval"])
    assert DOC.read_bytes() == before


def test_checked_in_doc_equals_fresh_render(derived):
    """The page must be the render of *code output*, in this very checkout."""
    ok, message = rend.check(derived["eval"], DOC)
    assert ok, message
    assert derived["page"].read_text(encoding="utf-8") == DOC.read_text(encoding="utf-8")


def test_check_fails_closed_on_missing_inputs(tmp_path, derived):
    ok, message = rend.check(tmp_path / "nope.json", DOC)
    assert not ok and "missing evaluation record" in message
    ok, message = rend.check(derived["eval"], tmp_path / "nope.md")
    assert not ok and "missing document" in message


def test_hand_edited_digit_is_detected(tmp_path, derived):
    """A single hand-edited digit must break freshness, and say where."""
    shipped = DOC.read_text(encoding="utf-8")
    assert "| `20261109` | 1.0000 |" in shipped
    edited = tmp_path / "edited.md"
    edited.write_text(shipped.replace("| `20261109` | 1.0000 |", "| `20261109` | 0.9999 |", 1),
                      encoding="utf-8")
    ok, message = rend.check(derived["eval"], edited)
    assert not ok
    assert "drifted" in message and "20261109" in message


def test_cli_check_exit_codes(tmp_path, derived):
    fresh = tmp_path / "fresh.md"
    fresh.write_text(rend.render(derived["eval"]), encoding="utf-8")
    good = subprocess.run([sys.executable, str(SCRIPTS / "render_criterion_prereg_doc.py"),
                           "--check", "--eval", str(derived["eval"]), "--out", str(fresh)],
                          capture_output=True, text=True)
    assert good.returncode == 0, good.stdout + good.stderr
    assert "OK:" in good.stdout

    bad = tmp_path / "bad.md"
    bad.write_text(rend.render(derived["eval"]).replace("F1", "F2", 1), encoding="utf-8")
    drifted = subprocess.run([sys.executable, str(SCRIPTS / "render_criterion_prereg_doc.py"),
                              "--check", "--eval", str(derived["eval"]), "--out", str(bad)],
                             capture_output=True, text=True)
    assert drifted.returncode == 1
    assert "DRIFT:" in drifted.stdout


def test_cli_stdout_is_the_written_page(derived):
    out = subprocess.run([sys.executable, str(SCRIPTS / "render_criterion_prereg_doc.py"),
                          "--stdout", "--eval", str(derived["eval"])],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert out.stdout == DOC.read_text(encoding="utf-8")


def test_eval_record_is_reproducible_from_scratch(derived):
    """The record is code output, not a hand-saved snapshot.

    A second, independent run of the evaluation (here: into a scratch directory)
    must reproduce ``eval.json`` and the page byte for byte; if the corpora, the
    extractor or a metric changes without the page being regenerated, this
    fails.  Everything under test therefore rests on the derivation, and no
    number can drift away from the code that produced it.
    """
    assert derived["record"]["all_acceptance_pass"] is True
    assert derived["page"].read_text(encoding="utf-8") == DOC.read_text(encoding="utf-8")
    for name in FIGURES:
        assert (derived["dir"] / name).exists(), f"the evaluation script wrote no {name}"
    assert derived["mission"].exists(), "the evaluation script wrote no mission record"


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


def test_publisher_dry_run_cli(tmp_path, art_dir, derived):
    out = tmp_path / "payload.json"
    proc = subprocess.run([sys.executable, str(SCRIPTS / "publish_criterion_prereg_report.py"),
                           "--dry-run", "--eval", str(derived["eval"]),
                           "--art-dir", str(art_dir), "--out", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "payload OK" in proc.stdout
    written = json.loads(out.read_text(encoding="utf-8"))
    for key in pub.REQUIRED_KEYS:
        assert key in written
    # the on-disk copy must be inspectable without dragging the base64 blobs along
    assert all(isinstance(f["png_base64"], str) and f["png_base64"].startswith("<")
               for f in written["figures"])


def test_publisher_fails_closed_when_a_figure_is_missing(tmp_path, art_dir, derived):
    (art_dir / "fig_demo_kpi.png").unlink()
    proc = subprocess.run([sys.executable, str(SCRIPTS / "publish_criterion_prereg_report.py"),
                           "--dry-run", "--eval", str(derived["eval"]),
                           "--art-dir", str(art_dir), "--out", str(tmp_path / "payload.json")],
                          capture_output=True, text=True)
    assert proc.returncode == 1
    assert "FAIL:" in proc.stdout


# --------------------------------------------------------------------------- #
# (c) the artefacts: derived pairs are hard requirements, committed copies are
#     verified as a byte-exact cache (so a stripped artifact directory cannot
#     turn this suite red -- nor, worse, silently green)
# --------------------------------------------------------------------------- #
ART = ROOT / "artifacts" / "criterion_prereg"


def _assert_valid_png(path: Path, what: str) -> None:
    raw = path.read_bytes()
    assert raw.startswith(pub.PNG_SIGNATURE), f"{what} is not a PNG"
    assert len(raw) > 1000, f"{what} is suspiciously small ({len(raw)} bytes)"


def test_derived_figures_are_complete_and_are_valid_png(derived):
    """The record names four figures and the evaluation script writes all four.

    The page and the report are both rendered from the (record, figures) pair,
    so a deliverable with a record and no figures is unverifiable.  This holds
    for the derivation, which every checkout can produce, whatever an artifact
    filter may have dropped on the way into git.
    """
    labels = derived["record"]["detail"]["figures"]
    assert labels, "the record must list its figures"
    assert sorted(Path(rel).name for rel in labels.values()) == sorted(FIGURES)
    for key, rel in labels.items():
        path = derived["dir"] / Path(rel).name
        assert path.exists(), f"figure {key!r} not derived: {rel}"
        _assert_valid_png(path, rel)


def test_committed_record_is_a_byte_exact_cache(derived):
    """The committed record must be the derivation, byte for byte.

    ``artifacts/criterion_prereg/eval.json`` is checked in so the published
    report can cite a stable path; if it ever stops matching the code it is a
    stale snapshot and must be re-derived with
    ``python scripts/run_criterion_prereg_eval.py``.
    """
    if not EVAL.exists():
        pytest.skip("record cache absent from this checkout (filtered artifact); "
                    "the derivation above is authoritative and was executed")
    assert EVAL.read_text(encoding="utf-8") == derived["eval"].read_text(encoding="utf-8"), (
        "artifacts/criterion_prereg/eval.json is stale; re-run "
        "scripts/run_criterion_prereg_eval.py"
    )


DELIVERABLE = ROOT / "docs" / "m8598" / "outputs" / "acceptance_criteria.json"


def test_mission_record_is_emitted_from_the_same_payload(derived):
    """``docs/m8598/outputs/acceptance_criteria.json`` is code output, not a copy.

    It used to be a byte-for-byte copy of another mission's record (shaft
    torsion metrics plus a ``pytest_test_shaft_torsion`` flag this node never
    runs), i.e. exactly the second source this gate exists to prevent.  It is now
    emitted by the evaluation script from the same payload as the page, so: the
    committed copy must equal the derivation, and its acceptance block must be
    the record's acceptance block -- no re-typed threshold.
    """
    assert DELIVERABLE.exists(), f"the mission record is missing: {DELIVERABLE}"
    shipped = json.loads(DELIVERABLE.read_text(encoding="utf-8"))
    assert shipped["acceptance"] == derived["record"]["acceptance"]
    assert shipped["mission_id"] == derived["record"]["mission_id"] == 8598
    assert shipped["node_id"] == derived["record"]["node"]
    # the record *declares* its provenance; the artefact cache itself is optional
    # in git (a merge gate strips artifacts/), so its presence is not required --
    # the byte-equality with the derivation below is the authoritative check.
    assert shipped["source"] == "artifacts/criterion_prereg/eval.json"
    assert (ROOT / shipped["page"]).exists()
    # threshold values must not appear twice: the numbers live in the record only
    for block in shipped["acceptance"].values():
        assert isinstance(block, dict) and "pass" in block
    assert DELIVERABLE.read_text(encoding="utf-8") == derived["mission"].read_text(encoding="utf-8"), (
        "docs/m8598/outputs/acceptance_criteria.json is stale; re-run "
        "scripts/run_criterion_prereg_eval.py"
    )


def _sealed_shipped() -> dict:
    import run_criterion_prereg_eval as ev  # noqa: E402  (scripts/ is on sys.path)

    return ev, json.loads(DELIVERABLE.read_text(encoding="utf-8"))


def test_deliverable_is_sealed_and_verifiable_without_the_cache(derived):
    """The committed record refutes a stale copy on its own, offline.

    Regression guard for the failure this mission was reopened for: the
    checked-in record was another mission's file (shaft-torsion metrics), and
    detecting that needed a full re-derivation.  ``payload_sha256`` makes the
    copy self-verifying: no artefact cache, no corpora re-run, just the bytes.
    """
    ev, shipped = _sealed_shipped()
    assert isinstance(shipped.get(ev.DIGEST_KEY), str) and len(shipped[ev.DIGEST_KEY]) == 64
    ok, reason = ev.verify_deliverable(shipped)
    assert ok, reason
    # the shipped seal is the derivation's seal: same body -> same digest
    fresh = ev.deliverable_record(derived["record"], "artifacts/criterion_prereg/eval.json")
    assert ev.verify_deliverable(fresh) == (True, "ok")
    assert shipped[ev.DIGEST_KEY] == fresh[ev.DIGEST_KEY]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["acceptance"]["a_detection_24cases"].__setitem__("recall", 0.5),
        lambda r: r["acceptance"]["b_extractor_field_f1"].__setitem__("pass", False),
        lambda r: r["acceptance"]["c_fail_closed"].__setitem__("tampered_verdict", "pass"),
        lambda r: r.__setitem__("mission_id", 7803),
        lambda r: r.__setitem__("all_acceptance_pass", False),
    ],
    ids=["threshold-edited", "pass-flag-flipped", "negative-control-weakened",
         "wrong-mission", "overall-verdict-flipped"],
)
def test_seal_detects_tampering(mutate):
    """Negative controls: every edit the old record suffered must flip the seal."""
    ev, shipped = _sealed_shipped()
    tampered = json.loads(json.dumps(shipped))
    mutate(tampered)
    ok, reason = ev.verify_deliverable(tampered)
    assert not ok and ev.DIGEST_KEY in reason, reason


def test_seal_is_absent_for_unsealed_records():
    """An unsealed record (the shape that shipped stale) fails closed, not open."""
    ev, shipped = _sealed_shipped()
    del shipped[ev.DIGEST_KEY]
    ok, reason = ev.verify_deliverable(shipped)
    assert not ok and "unsealed" in reason


def test_check_mode_gates_committed_copies(tmp_path):
    """``run_criterion_prereg_eval.py --check`` is a usable merge gate.

    Exit 0 on the committed pair; exit 1 when the record was edited by hand or
    when the page lags the record -- both without touching the repository.
    """
    script = SCRIPTS / "run_criterion_prereg_eval.py"
    good = subprocess.run([sys.executable, str(script), "--check"],
                          capture_output=True, text=True)
    assert good.returncode == 0, good.stdout + good.stderr
    assert "check: OK" in good.stdout

    stale_record = tmp_path / "acceptance_criteria.json"
    record = json.loads(DELIVERABLE.read_text(encoding="utf-8"))
    record["acceptance"]["a_detection_24cases"]["recall"] = 0.5
    stale_record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bad = subprocess.run([sys.executable, str(script), "--check",
                          "--deliverable", str(stale_record)],
                         capture_output=True, text=True)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "stale vs derivation" in bad.stdout and "mismatch" in bad.stdout

    stale_page = tmp_path / "page.md"
    stale_page.write_text(DOC.read_text(encoding="utf-8") + "\nedited\n", encoding="utf-8")
    bad_page = subprocess.run([sys.executable, str(script), "--check", "--doc", str(stale_page)],
                              capture_output=True, text=True)
    assert bad_page.returncode == 1, bad_page.stdout + bad_page.stderr
    assert "stale vs derivation" in bad_page.stdout


def test_committed_figures_when_present_are_valid_png(derived):
    """Committed figures are optional cache, but never an invalid PNG when there."""
    for name in FIGURES:
        path = ART / name
        if not path.exists():
            continue
        _assert_valid_png(path, f"artifacts/criterion_prereg/{name}")


def test_real_artifacts_produce_a_publishable_payload(derived):
    """The report pipeline accepts the derived (record, figures) pair as-is."""
    payload = pub.build_payload(derived["record"], derived["dir"])
    assert pub.validate(payload) == []
    assert len(payload["figures"]) == len(FIGURES)
    for fig in payload["figures"]:
        assert base64.b64decode(fig["png_base64"]).startswith(pub.PNG_SIGNATURE)
