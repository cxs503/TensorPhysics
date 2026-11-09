"""ASME V&V 20 uncertainty ledger: ``U_num``, ``U_input``, ``U_val``, ``U_ref``.

Acceptance of this node:

* ``U_val = sqrt(U_num^2 + U_input^2)`` (ASME V&V 20-2009) exactly, including
  the zero-uncertainty edge case;
* the GCI of the finest grid is converted into an *absolute* ``U_num`` whether
  the study reports a relative or an absolute band, with the provenance string
  naming the conversion;
* a reference that is not sharper than ``U_val`` yields ``WITHHELD`` -- a PASS
  whose reference could not have separated a good model from a bad one is not
  allowed to exist;
* the ledger is self-hashing so a published number can be traced to its terms;
* the gate consumes the ledger and *keeps its historical decision* when
  ``U_ref`` is not declared (the regression lock lives in
  ``test_vv_in30_regression.py``).
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from tensorphysics.vv import (
    Observation,
    UncertaintyLedger,
    compare,
    evaluate,
    grid_convergence_study,
    ledger_from_study,
    u_num_from_study,
)
from tensorphysics.vv.uncertainty import evaluate_ledger_table
from tensorphysics.vv.gci import FAIL, WITHHELD


def _study(gci_relative=True):
    hs = (0.05, 0.025, 0.0125)
    levels = [Observation(h=h, f=1.0 + 1.6 * h**2) for h in hs]
    return grid_convergence_study(levels)


# ---------------------------------------------------------------------------
# the ASME arithmetic
# ---------------------------------------------------------------------------


def test_u_val_is_the_asme_quadrature():
    ledger = UncertaintyLedger(u_num=0.3, u_input=0.4)
    assert ledger.u_val == pytest.approx(0.5)
    assert ledger.u_val_total == pytest.approx(0.5)  # u_ref defaults to 0
    assert ledger.u_num_fraction == pytest.approx(0.6)
    assert ledger.dominant_term() == "u_input"


def test_terms_must_be_finite_and_non_negative():
    with pytest.raises(ValueError):
        UncertaintyLedger(u_num=-1.0)
    with pytest.raises(ValueError):
        UncertaintyLedger(u_num=0.1, u_ref=float("nan"))
    with pytest.raises(ValueError):
        UncertaintyLedger(u_num=True)  # bool is not a number here


def test_zero_uncertainty_needs_an_exact_match():
    ledger = UncertaintyLedger(u_num=0.0)
    ok = compare(1.0, 1.0, ledger)
    assert ok.status == "PASS" and ok.ratio == 0.0
    bad = compare(1.0, 1.0000001, ledger)
    assert bad.status == "FAIL" and math.isinf(bad.ratio)


# ---------------------------------------------------------------------------
# U_num from the GCI
# ---------------------------------------------------------------------------


def test_relative_gci_is_converted_to_an_absolute_u_num():
    study = _study()
    assert study.gci_is_relative is True
    u_num, source = u_num_from_study(study)
    assert u_num == pytest.approx(study.gci_fine * abs(study.f_fine))
    assert "relative" in source and "GCI_fine" in source


def test_absolute_gci_passes_through():
    """An absolute band (error-norm study, --gci-absolute) must not be rescaled."""
    levels = [Observation(h=h, f=0.4 * h**2) for h in (0.04, 0.02, 0.01)]
    study = grid_convergence_study(levels)
    absolute = replace(study, gci_is_relative=False)
    u_num, source = u_num_from_study(absolute)
    assert u_num == pytest.approx(abs(absolute.gci_fine))
    assert "absolute" in source
    # the same GCI interpreted as a relative band is scaled by |f_fine|
    relative, source_rel = u_num_from_study(study)
    assert relative == pytest.approx(abs(study.gci_fine) * abs(study.f_fine))
    assert "relative" in source_rel


def test_study_without_gci_refuses_to_invent_u_num():
    with pytest.raises(ValueError):
        u_num_from_study(None)
    stagnant = grid_convergence_study([Observation(h=h, f=0.5) for h in (0.04, 0.02, 0.01)])
    assert stagnant.gci_fine is None
    with pytest.raises(ValueError):
        u_num_from_study(stagnant)


# ---------------------------------------------------------------------------
# discriminability of the reference
# ---------------------------------------------------------------------------


def test_sharp_reference_gives_an_asme_pass_or_fail():
    ledger = UncertaintyLedger(u_num=0.010, u_input=0.005, u_ref=0.001)
    assert ledger.reference_ratio == pytest.approx(0.001 / math.sqrt(0.010**2 + 0.005**2))
    assert ledger.discriminative() is True

    inside = compare(1.010, 1.000, ledger)
    assert inside.status == "PASS"
    assert inside.ratio == pytest.approx(0.010 / ledger.u_val)
    assert inside.ratio < 1.0

    outside = compare(1.030, 1.000, ledger)
    assert outside.status == "FAIL"


def test_blurry_reference_withholds_a_would_be_pass():
    """U_ref >= U_val: the benchmark cannot discriminate, so no grade is issued."""
    ledger = UncertaintyLedger(u_num=0.001, u_input=0.0, u_ref=0.050)
    assert ledger.discriminative() is False
    verdict = compare(1.0002, 1.000, ledger)
    assert verdict.passed is True  # numerically inside U_val ...
    assert verdict.status == WITHHELD  # ... but the comparison says nothing
    assert verdict.ratio < 1.0
    assert "cannot discriminate" in verdict.detail


def test_blurry_reference_also_withholds_a_would_be_fail():
    ledger = UncertaintyLedger(u_num=0.001, u_ref=0.050)
    verdict = compare(1.010, 1.000, ledger)
    assert verdict.passed is False
    assert verdict.status == WITHHELD


def test_discrimination_factor_is_configurable():
    ledger = UncertaintyLedger(u_num=0.010, u_ref=0.005)
    assert ledger.discriminative(1.0) is True  # 0.5 < 1
    assert ledger.discriminative(0.4) is False  # 0.5 >= 0.4


# ---------------------------------------------------------------------------
# audit chain and table driver
# ---------------------------------------------------------------------------


def test_ledger_digest_is_stable_and_content_addressed():
    a = UncertaintyLedger(u_num=0.01, u_input=0.002, u_ref=0.003).as_dict()
    b = UncertaintyLedger(u_num=0.01, u_input=0.002, u_ref=0.003).as_dict()
    c = UncertaintyLedger(u_num=0.01, u_input=0.002, u_ref=0.004).as_dict()
    assert a["digest"] == b["digest"]
    assert a["digest"] != c["digest"]
    assert len(a["digest"]) == 64
    assert a["schema"] == "tensorphysics.vv.uncertainty-ledger/1"


def test_ledger_from_study_records_provenance():
    study = _study()
    ledger = ledger_from_study(study, u_input=0.002, u_ref=0.001, u_ref_source="NMRI model test 2019")
    assert ledger.u_num == pytest.approx(study.gci_fine * study.f_fine)
    assert ledger.u_ref_source == "NMRI model test 2019"
    assert ledger.u_val == pytest.approx(math.hypot(ledger.u_num, 0.002))


def test_table_driver():
    rows = [
        {"label": "sharp", "f_sim": 1.001, "f_ref": 1.000, "u_num": 0.01, "u_ref": 0.001},
        {"label": "blurry", "f_sim": 1.001, "f_ref": 1.000, "u_num": 0.001, "u_ref": 0.05},
    ]
    verdicts = evaluate_ledger_table(rows)
    assert [v.status for v in verdicts] == ["PASS", "WITHHELD"]
    with pytest.raises(ValueError):
        evaluate_ledger_table([{"f_sim": 1.0}])


# ---------------------------------------------------------------------------
# gate wiring
# ---------------------------------------------------------------------------


def test_gate_reports_the_ledger_even_without_u_ref():
    levels = [Observation(h=h, f=1.0 + 1.6 * h**2) for h in (0.05, 0.025, 0.0125)]
    verdict = evaluate(levels, f_exp=1.0003, u_input=0.0004)
    assert verdict.uncertainty is not None
    assert verdict.uncertainty.u_ref == 0.0
    # historical decision kept: the plain |E| <= U_val check
    assert verdict.validation is not None and verdict.validation.passed
    assert not any(c["check"] == "validation/V&V20-ledger" for c in verdict.checks)


def test_gate_withholds_when_the_reference_cannot_discriminate():
    levels = [Observation(h=h, f=1.0 + 1.6 * h**2) for h in (0.05, 0.025, 0.0125)]
    verdict = evaluate(levels, f_exp=1.0003, u_input=0.0004, u_ref=0.5)
    assert verdict.status == WITHHELD
    check = next(c for c in verdict.checks if c["check"] == "validation/V&V20-ledger")
    assert check["status"] == WITHHELD
    assert verdict.uncertainty.reference_ratio > 1.0


def test_gate_fails_when_a_sharp_reference_rejects_the_model():
    levels = [Observation(h=h, f=1.0 + 1.6 * h**2) for h in (0.05, 0.025, 0.0125)]
    verdict = evaluate(levels, f_exp=1.002, u_input=0.0001, u_ref=0.0001)
    assert verdict.status == FAIL
    check = next(c for c in verdict.checks if c["check"] == "validation/V&V20-ledger")
    assert check["status"] == FAIL
    assert verdict.uncertainty.discriminative()