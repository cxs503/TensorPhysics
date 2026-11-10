"""Tests of the exchange-operator algebra and the interval attribution ledger."""
from __future__ import annotations

import math

import pytest

from tensorphysics.coupling_energy.exchange_operators import (
    SCHEME_TABLE,
    CenteredExchange,
    ExchangeScheme,
    IntervalExchangeLedger,
    LagExchange,
    PredictorCorrectorExchange,
    StampConvention,
    StampRule,
    operator_for,
)
from tensorphysics.coupling_energy.ledger import CouplingLedgerError


# ---------------------------------------------------------------------------
# algebra
# ---------------------------------------------------------------------------
def test_three_schemes_share_the_interface():
    orders = {scheme: spec.declared_order for scheme, spec in SCHEME_TABLE.items()}
    assert orders == {ExchangeScheme.LAG: 1, ExchangeScheme.CENTERED: 2,
                      ExchangeScheme.PREDICTOR_CORRECTOR: 2}
    for scheme in SCHEME_TABLE:
        op = operator_for(scheme)
        transfer = op.provisional_transfer(0, 0.0, 0.5, q_lo=2.0, q_hi=4.0, q_prev=1.0)
        assert transfer.scheme is scheme
        assert math.isfinite(transfer.rate)


def test_lag_holds_the_newest_closed_sample():
    op = LagExchange()
    transfer = op.provisional_transfer(3, 0.0, 0.5, q_lo=7.0, q_hi=9.0, q_prev=5.0)
    assert transfer.rate == 7.0
    assert transfer.integral == pytest.approx(3.5)


def test_centered_is_the_trapezoid_and_needs_the_endpoint():
    op = CenteredExchange()
    transfer = op.provisional_transfer(0, 0.0, 0.5, q_lo=2.0, q_hi=4.0, q_prev=None)
    assert transfer.rate == pytest.approx(3.0)
    with pytest.raises(CouplingLedgerError):
        op.provisional_transfer(0, 0.0, 0.5, q_lo=2.0, q_hi=None, q_prev=None)


def test_predictor_corrector_matches_centered_after_correction():
    pc = PredictorCorrectorExchange()
    ce = CenteredExchange()
    prov = pc.provisional_transfer(0, 0.0, 0.5, q_lo=2.0, q_hi=4.0, q_prev=1.0)
    assert prov.predicted and prov.provisional
    # linear extrapolation: q_hat = 2*2 - 1 = 3, so the provisional rate is 2.5
    assert prov.rate == pytest.approx(2.5)
    corrected = pc.corrected_transfer(prov, 4.0)
    assert corrected.rate == pytest.approx(ce.provisional_transfer(
        0, 0.0, 0.5, q_lo=2.0, q_hi=4.0, q_prev=None).rate)


def test_predictor_corrector_degrades_to_lag_on_the_first_interval():
    pc = PredictorCorrectorExchange()
    prov = pc.provisional_transfer(0, 0.0, 0.5, q_lo=2.0, q_hi=4.0, q_prev=None)
    assert not prov.predicted
    assert prov.rate == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# stamp conventions
# ---------------------------------------------------------------------------
def test_endpoint_convention_rejects_an_offset_and_recentring():
    with pytest.raises(CouplingLedgerError):
        StampRule(StampConvention.ENDPOINT, offset=0.5)
    with pytest.raises(CouplingLedgerError):
        StampRule(StampConvention.ENDPOINT, recenter=True)


def test_destamp_is_exact_on_a_linear_stream():
    stamp = StampRule(StampConvention.WINDOW_MIDPOINT, offset=0.25, recenter=True)
    q = lambda t: 3.0 + 2.0 * t
    lo, hi = stamp.endpoints(q(1.0 - 0.25), q(2.0 - 0.25), q(0.0 - 0.25), 1.0)
    assert lo == pytest.approx(q(1.0))
    assert hi == pytest.approx(q(2.0))


def test_raw_window_convention_keeps_the_phase_offset():
    stamp = StampRule(StampConvention.WINDOW_MIDPOINT, offset=0.25, recenter=False)
    lo, _ = stamp.endpoints(1.0, 2.0, 0.5, 1.0)
    assert lo == 1.0  # published value consumed as an endpoint value


def test_centered_operator_with_destamp_uses_the_endpoint_values():
    stamp = StampRule(StampConvention.WINDOW_MIDPOINT, offset=0.25, recenter=True)
    op = CenteredExchange(stamp)
    # published stream m(t) = q(t - 0.25) for q(t) = t: m(0) = -0.25, m(1) = 0.75,
    # m(-1) = -1.25
    prov = op.provisional_transfer(0, 0.0, 1.0, q_lo=-0.25, q_hi=0.75, q_prev=-1.25)
    # de-stamped endpoints are 0.0 and 1.0 -> trapezoid 0.5, exactly the mean of
    # q(t) = t over [0, 1]; the raw trapezoid would be 0.25, the O(dt) bias the
    # contract reports
    assert prov.rate == pytest.approx(0.5)
    assert prov.recentered


# ---------------------------------------------------------------------------
# attribution ledger
# ---------------------------------------------------------------------------
def _booked_ledger(n: int = 3) -> IntervalExchangeLedger:
    ledger = IntervalExchangeLedger("test_iface", n)
    for i in range(n):
        ledger.declare_interval(i, float(i), float(i + 1))
        ledger.book_produced(i, 2.0)
        ledger.book_release(i, 2.0)
        ledger.book_applied(i, 2.0)
        ledger.close_interval(i)
    return ledger


def test_complete_booking_reports_zero_unattributed_and_zero_deficit():
    summary = _booked_ledger().summary()
    assert summary.complete
    assert summary.unattributed_share == 0.0
    assert summary.deficit_share == 0.0
    assert summary.open_intervals == ()
    assert _booked_ledger().assert_complete() is not None


def test_tail_booking_leaves_most_of_the_quantity_unattributed():
    ledger = IntervalExchangeLedger("tail", 2)
    ledger.declare_interval(0, 0.0, 1.0)
    ledger.declare_interval(1, 1.0, 2.0)
    for i in range(2):
        for _ in range(8):
            ledger.book_produced(i, 1.0)      # eight sub-window shares
        ledger.book_release(i, 1.0)           # only the tail one is released
        ledger.book_applied(i, 1.0)
        ledger.close_interval(i)
    summary = ledger.summary()
    assert summary.unattributed_share == pytest.approx(7.0 / 8.0)
    with pytest.raises(CouplingLedgerError):
        ledger.assert_complete()


def test_double_booking_is_refused():
    ledger = IntervalExchangeLedger("double", 1)
    ledger.declare_interval(0, 0.0, 1.0)
    ledger.book_applied(0, 1.0)
    with pytest.raises(CouplingLedgerError):
        ledger.book_applied(0, 1.0)


def test_double_close_is_refused():
    ledger = IntervalExchangeLedger("close", 1)
    ledger.declare_interval(0, 0.0, 1.0)
    ledger.close_interval(0)
    with pytest.raises(CouplingLedgerError):
        ledger.close_interval(0)
    with pytest.raises(CouplingLedgerError):
        ledger.book_produced(0, 1.0)


def test_missing_interval_is_detected():
    ledger = IntervalExchangeLedger("missing", 2)
    ledger.declare_interval(0, 0.0, 1.0)
    ledger.book_produced(0, 1.0)
    ledger.book_release(0, 1.0)
    ledger.book_applied(0, 1.0)
    ledger.close_interval(0)
    summary = ledger.summary()
    assert summary.intervals == 1 and summary.expected_intervals == 2
    assert not summary.complete


def test_correction_is_booked_on_the_same_interval():
    ledger = IntervalExchangeLedger("correction", 1)
    ledger.declare_interval(0, 0.0, 1.0)
    ledger.book_produced(0, 1.0)
    ledger.book_release(0, 1.0)
    ledger.book_applied(0, 1.0)
    ledger.book_correction(0, 0.25)
    ledger.close_interval(0)
    summary = ledger.summary()
    assert summary.applied_total == 1.0
    assert summary.correction_total == 0.25
    assert summary.deficit_share == pytest.approx(0.25 / 1.0)


def test_non_finite_bookings_are_refused():
    ledger = IntervalExchangeLedger("finite", 1)
    ledger.declare_interval(0, 0.0, 1.0)
    for fn in (ledger.book_produced, ledger.book_release, ledger.book_applied):
        with pytest.raises(CouplingLedgerError):
            fn(0, float("nan"))