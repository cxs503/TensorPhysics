"""Tests for dual mass / energy / impulse ledger reconciliation.

Acceptance criterion of the campaign node: a *constructed known input* (constant
fluxes over clocked exchange windows) must reconcile to machine precision, i.e.
the relative residual must sit inside a few units in the last place, and the
same batch must reproduce bit-for-bit (``diff = 0``).
"""
from __future__ import annotations

import math

import pytest

from tensorphysics.campaign import (
    CouplingClockSpec,
    LedgerEntry,
    LedgerError,
    QuantityKind,
    ReferenceFluxes,
    SideLedger,
    build_reference_dual_ledgers,
    canonical_json,
    ledger_totals,
    reconcile,
    reconcile_books,
    verify_same_batch,
)
from tensorphysics.campaign.ledger import DEFAULT_ATOL_ULPS, EPS

DECIMAL_FLUXES = ReferenceFluxes(
    mass_flow_kg_s=12.3, power_w=4200.0, force_n=(1.2e5, 0.0, -3.4e4)
)
BINARY_FLUXES = ReferenceFluxes(
    mass_flow_kg_s=2.0, power_w=4096.0, force_n=(8.0, 0.0, -4.0)
)


def decimal_spec() -> CouplingClockSpec:
    """Decimal clocks: 0.004 s master, 0.0004 s sub, exchange every 5 ticks."""
    return CouplingClockSpec("rigid_body", "lbm_flow", 0.004, 0.0004, 5)


def binary_spec() -> CouplingClockSpec:
    """Binary clocks: exact power-of-two steps, so books must agree exactly."""
    return CouplingClockSpec(
        "rigid_body", "lbm_flow", 1.0 / 256.0, 1.0 / 4096.0, exchange_every=4
    )


def test_reference_ledger_matches_analytic_totals():
    spec = decimal_spec()
    books = build_reference_dual_ledgers(spec, 50, DECIMAL_FLUXES)
    totals = ledger_totals(books)
    horizon = 50 * spec.master_dt_s
    assert totals["mass"]["windows"] == 10
    assert totals["mass"]["producer_total"][0] == pytest.approx(
        DECIMAL_FLUXES.mass_flow_kg_s * horizon, rel=1e-15
    )
    assert totals["energy"]["producer_total"][0] == pytest.approx(
        DECIMAL_FLUXES.power_w * horizon, rel=1e-15
    )
    for index in range(3):
        assert totals["impulse"]["producer_total"][index] == pytest.approx(
            DECIMAL_FLUXES.force_n[index] * horizon, rel=1e-15, abs=1e-9
        )


def test_decimal_books_reconcile_at_machine_precision():
    books = build_reference_dual_ledgers(decimal_spec(), 50, DECIMAL_FLUXES)
    artifact = reconcile_books(books, scenario="decimal_clock")
    assert artifact["status"] == "pass"
    assert artifact["max_relative_residual"] <= DEFAULT_ATOL_ULPS * EPS
    grades = [
        residual["grade"]
        for book in artifact["quantity_books"]
        for residual in book["residuals"]
    ]
    assert "out_of_tolerance" not in grades
    assert "machine_precision" in grades, "different summation orders must leave a residual"
    for book in artifact["quantity_books"]:
        for residual in book["residuals"]:
            assert residual["abs_residual"] <= residual["tolerance"]
            assert residual["tolerance"] == pytest.approx(
                DEFAULT_ATOL_ULPS * EPS * residual["scale"], rel=1e-15
            )


def test_binary_books_reconcile_exactly():
    books = build_reference_dual_ledgers(binary_spec(), 16, BINARY_FLUXES)
    artifact = reconcile_books(books, scenario="binary_clock")
    assert artifact["status"] == "pass"
    assert artifact["max_relative_residual"] == 0.0
    for book in artifact["quantity_books"]:
        for residual in book["residuals"]:
            assert residual["grade"] == "exact"
            assert residual["residual"] == 0.0


def test_impulse_book_is_three_dimensional_and_mass_energy_are_scalars():
    books = build_reference_dual_ledgers(decimal_spec(), 50, DECIMAL_FLUXES)
    assert books[QuantityKind.MASS][0].dimension == 1
    assert books[QuantityKind.ENERGY][0].dimension == 1
    assert books[QuantityKind.IMPULSE][0].dimension == 3
    artifact = reconcile_books(books, scenario="decimal_clock")
    dims = {book["quantity"]: book["component_count"] for book in artifact["quantity_books"]}
    assert dims == {"mass": 1, "energy": 1, "impulse": 3}
    assert {book["quantity"]: book["units"] for book in artifact["quantity_books"]} == {
        "mass": "kg",
        "energy": "J",
        "impulse": "N*s",
    }


def test_out_of_tolerance_books_are_flagged():
    books = build_reference_dual_ledgers(decimal_spec(), 50, DECIMAL_FLUXES)
    producer, consumer = books[QuantityKind.MASS]
    tampered_entries = list(consumer.entries)
    first = tampered_entries[0]
    tampered_entries[0] = LedgerEntry(
        label=first.label,
        step=first.step,
        components=(first.components[0] * (1.0 + 1e-6),),
    )
    tampered = SideLedger(
        quantity=QuantityKind.MASS, side="consumer", entries=tuple(tampered_entries)
    )
    verdict = reconcile(producer, tampered)
    assert verdict.status == "fail"
    assert verdict.residuals[0].grade == "out_of_tolerance"
    assert verdict.max_relative_residual > 5e-8  # 1e-6 of one window out of ten


def test_reconcile_rejects_inconsistent_books():
    books = build_reference_dual_ledgers(decimal_spec(), 50, DECIMAL_FLUXES)
    producer, consumer = books[QuantityKind.ENERGY]
    with pytest.raises(LedgerError, match="different quantities"):
        reconcile(producer, books[QuantityKind.MASS][1])
    with pytest.raises(LedgerError, match="argument order"):
        reconcile(consumer, producer)
    shorter = SideLedger(
        quantity=QuantityKind.ENERGY, side="consumer", entries=consumer.entries[:-1]
    )
    with pytest.raises(LedgerError, match="same transfer windows"):
        reconcile(producer, shorter)
    with pytest.raises(LedgerError):
        reconcile(producer, consumer, atol_ulps=0.0)


def test_side_ledger_detects_tampered_recorded_total():
    books = build_reference_dual_ledgers(decimal_spec(), 50, DECIMAL_FLUXES)
    payload = books[QuantityKind.ENERGY][0].to_dict()
    payload["total"] = [payload["total"][0] * 1.000001]
    with pytest.raises(LedgerError, match="recorded total"):
        SideLedger.from_dict(payload)


def test_side_ledger_schema_rejects_vector_for_scalar_quantity():
    books = build_reference_dual_ledgers(decimal_spec(), 50, DECIMAL_FLUXES)
    payload = books[QuantityKind.MASS][0].to_dict()
    payload["entries"][0]["components"] = [1.0, 2.0]
    with pytest.raises(LedgerError, match="needs 1 component"):
        SideLedger.from_dict(payload)


def test_same_batch_reproduces_bit_for_bit():
    first = reconcile_books(
        build_reference_dual_ledgers(decimal_spec(), 50, DECIMAL_FLUXES),
        scenario="decimal_clock",
    )
    second = reconcile_books(
        build_reference_dual_ledgers(decimal_spec(), 50, DECIMAL_FLUXES),
        scenario="decimal_clock",
    )
    assert canonical_json(first) == canonical_json(second)
    assert first["same_batch_digest"] == second["same_batch_digest"]
    assert verify_same_batch(first) == "pass"
    first["same_batch_digest"] = "0" * 64
    assert verify_same_batch(first) == "fail"


def test_reconciliation_artifact_schema_is_closed():
    artifact = reconcile_books(
        build_reference_dual_ledgers(decimal_spec(), 50, DECIMAL_FLUXES),
        scenario="decimal_clock",
    )
    artifact["quantity_books"][0]["residuals"][0]["tolerance"] = math.nan
    with pytest.raises(ValueError):
        reconcile_books(
            build_reference_dual_ledgers(decimal_spec(), 50, DECIMAL_FLUXES),
            scenario="",
        )
    from tensorphysics.campaign import validate_artifact

    with pytest.raises(Exception):
        validate_artifact("ledger_reconciliation.v1", artifact)