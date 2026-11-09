"""Cross-repository benchmark contract and the ship-performance adapter.

Acceptance of this node:

* the shipped JSON schema and :func:`tensorphysics.vv.schema.check_benchmark`
  describe the same contract (regenerating the schema file is a no-op);
* one document per solver dialect (TensorLBM, TensorFEM, ship-performance) passes
  the contract, is adapted to a gate study and reaches a verdict -- the gate is
  consumable from outside this repository without changing the emitters;
* a document that violates the contract is reported with a reason, not a
  traceback (an unreadable cross-repo document is evidence in itself);
* the CLI exposes the new stages (``--feature-scale``, ``--u-ref``) and returns
  the documented exit codes, including the ``WITHHELD``/``UNRESOLVED`` ones.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tensorphysics.vv import Observation, adapters, evaluate
from tensorphysics.vv.schema import (
    VV_BENCHMARK_SCHEMA,
    check_benchmark,
    load_benchmark,
    schema_path,
    write_schema,
)

_DATA = Path(__file__).resolve().parent / "data" / "vv_benchmark"
_ROOT = Path(__file__).resolve().parents[1]


def _payload(name: str):
    return json.loads((_DATA / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------


def test_shipped_schema_matches_the_code():
    shipped = json.loads(schema_path().read_text(encoding="utf-8"))
    assert shipped == VV_BENCHMARK_SCHEMA
    assert shipped["$schema"].endswith("2020-12/schema")


def test_write_schema_is_idempotent(tmp_path):
    first = write_schema(tmp_path / "a.json")
    second = write_schema(tmp_path / "b.json")
    assert first.read_text() == second.read_text()


@pytest.mark.parametrize(
    "name, repo",
    [
        ("ship_resistance_study.json", "ship"),
        ("tensorlbm_channel_study.json", "tensorlbm"),
        ("tensorfem_bending_study.json", "tensorfem"),
    ],
)
def test_each_dialect_conforms_and_adapts(name, repo):
    payload = _payload(name)
    assert check_benchmark(payload) == []
    series = adapters.extract_series(payload)
    assert series.repo == repo
    assert len(series.levels) == len(payload["levels"])
    # spacings are normalised coarse -> fine
    spacings = [lvl.h for lvl in series.levels]
    assert all(a > b for a, b in zip(spacings, spacings[1:]))


def test_ship_counts_are_inverted_to_spacings():
    series = adapters.extract_series(_payload("ship_resistance_study.json"))
    assert series.levels[-1].h == pytest.approx(1.0 / 4800)
    assert series.levels[-1].f == pytest.approx(402.24)


def test_invalid_ordering_is_reported_with_a_reason():
    payload, problems = load_benchmark(_DATA / "invalid_no_ordering.json")
    assert problems, "a coarse->fine ordering violation must be reported"
    assert any("strictly decrease" in p for p in problems)


def test_contract_rejects_non_finite_and_missing_fields():
    problems = check_benchmark({"schema": "x", "quantity": "y", "levels": [{"h": 1.0, "value": float("nan")}, {"h": 0.5}]})
    joined = " ".join(problems)
    assert "not finite" in joined
    assert "no observable" in joined
    assert check_benchmark({"quantity": "q", "levels": []}) != []
    assert check_benchmark([1, 2, 3]) == ["benchmark document must be a JSON object, got list"]


# ---------------------------------------------------------------------------
# end-to-end: ship document -> gate verdict
# ---------------------------------------------------------------------------


def test_ship_study_reaches_a_gate_verdict():
    payload = _payload("ship_resistance_study.json")
    series = adapters.extract_series(payload)
    features = payload["features"]
    verdict = evaluate(
        series.levels,
        quantity=series.quantity,
        features=features,
        dx=series.levels[-1].h,
        f_exp=payload["validation"]["f_exp"],
        u_input=payload["validation"]["u_input"],
        u_ref=payload["validation"]["u_ref"],
    )
    # 4800 panels -> dx = 1/4800 = 2.08e-4 m, feature 4.2 mm -> R = 20.2 (resolved)
    assert verdict.resolvability.status == "RESOLVED"
    assert verdict.resolvability.ratio == pytest.approx(0.0042 / (1.0 / 4800))
    # the ledger is built from the panel-refinement GCI: U_num = 2.29 kN,
    # U_val = sqrt(U_num^2 + U_input^2) = 2.74 kN, U_ref = 2.00 kN -> the
    # reference is still sharper than U_val (ratio 0.73), so the ASME test can
    # decide, and it decides FAIL: |E| = |402.24 - 398.0| = 4.24 kN > 2.74 kN
    assert verdict.uncertainty is not None
    assert verdict.uncertainty.u_num == pytest.approx(2.2887, rel=1e-3)
    assert verdict.uncertainty.u_val == pytest.approx(2.73648, rel=1e-4)
    assert verdict.uncertainty.reference_ratio == pytest.approx(2.0 / verdict.uncertainty.u_val)
    assert verdict.uncertainty.discriminative()
    assert verdict.status == "FAIL"
    record = verdict.as_dict()
    assert record["resolvability"]["law"] == "R = delta / dx"
    assert record["uncertainty"]["dominant_term"] == "u_num"


def test_ship_study_with_a_blurry_reference_is_withheld():
    payload = _payload("ship_resistance_study.json")
    series = adapters.extract_series(payload)
    verdict = evaluate(
        series.levels,
        quantity=series.quantity,
        f_exp=payload["validation"]["f_exp"],
        u_input=payload["validation"]["u_input"],
        u_ref=10.0,
    )
    assert not verdict.uncertainty.discriminative()
    assert verdict.status == "WITHHELD"
    check = next(c for c in verdict.checks if c["check"] == "validation/V&V20-ledger")
    assert "cannot discriminate" in check["detail"]


def test_ship_study_with_a_sharp_reference_validates():
    payload = _payload("ship_resistance_study.json")
    series = adapters.extract_series(payload)
    verdict = evaluate(
        series.levels,
        quantity=series.quantity,
        features=payload["features"],
        f_exp=402.0,
        u_input=0.5,
        u_ref=0.2,
    )
    assert verdict.uncertainty.discriminative()
    assert verdict.status in {"PASS", "WARN"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(args, **kwargs):
    """Run the CLI against *this* working tree (not an installed copy)."""
    import os

    cmd = [sys.executable, "-m", "tensorphysics.vv", *args]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True, env=env, **kwargs)


def test_cli_ship_document_json_includes_new_stages():
    doc = _DATA / "ship_resistance_study.json"
    result = _run_cli(
        ["--benchmark", str(doc), "--format", "json", "--u-ref", "2.0", "--u-input", "1.5"]
    )
    assert result.returncode == 2, result.stderr  # |E| = 4.24 kN > U_val -> honest FAIL
    record = json.loads(result.stdout)
    assert record["status"] == "FAIL"
    assert record["resolvability"]["status"] == "RESOLVED"
    assert record["resolvability"]["ratio"] == pytest.approx(0.0042 * 4800)
    assert record["uncertainty"]["u_ref"] == pytest.approx(2.0)
    assert record["validation"]["status"] == "FAIL"
    assert any(c["check"] == "validation/V&V20-ledger" for c in record["checks"])


def test_cli_subgrid_feature_returns_unresolved_exit_code():
    doc = _DATA / "tensorlbm_channel_study.json"
    result = _run_cli(
        ["--benchmark", str(doc), "--format", "json", "--feature-scale", "wake_shear_layer=1e-6"]
    )
    assert result.returncode == 4, result.stderr  # EXIT_CODES[UNRESOLVED]
    record = json.loads(result.stdout)
    assert record["status"] == "UNRESOLVED"


def test_cli_require_resolvability_without_declaration_withholds():
    doc = _DATA / "tensorlbm_channel_study.json"
    result = _run_cli(["--benchmark", str(doc), "--format", "json", "--require-resolvability"])
    assert result.returncode == 3, result.stderr  # EXIT_CODES[WITHHELD]
    record = json.loads(result.stdout)
    assert record["status"] == "WITHHELD"


def test_cli_rejects_a_malformed_feature_flag():
    doc = _DATA / "tensorlbm_channel_study.json"
    result = _run_cli(["--benchmark", str(doc), "--feature-scale", "boundary_layer"])
    assert result.returncode == 64
    assert "NAME=DELTA" in result.stderr