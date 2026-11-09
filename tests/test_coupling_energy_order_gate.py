"""Tests for the P40W3-N09 sub-cycle interpolation order gate.

The suite has three jobs:

1. unit-check the order estimator and the staggered reconstruction algebra;
2. prove that the *exact* (double-entry) booking is an identity, i.e. that it
   cannot see the interpolation order at all — and that the *partitioned*
   booking does see it, with the declared order (the falsification arm);
3. gate the real runtime studies (ratio trend, long-time boundedness) and make
   sure the gate itself can fail on a wrong record.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.coupling_energy import order_gate as G
from tensorphysics.coupling_energy.exchange import (
    ExchangeSample,
    TemporalScheme,
    reconstruction_poly,
)
from tensorphysics.coupling_energy.ledger import CouplingLedgerError


# ---------------------------------------------------------------------------
# order estimator
# ---------------------------------------------------------------------------
def test_fit_order_recovers_declared_slopes():
    hs = [1.0, 0.5, 0.25, 0.125]
    for p in (1, 2, 3):
        errs = [2.5e-3 * h ** p for h in hs]
        slope, r2 = G.fit_order(hs, errs)
        assert slope == pytest.approx(p, abs=1e-9)
        assert r2 == pytest.approx(1.0, abs=1e-12)


def test_fit_order_r2_flags_a_flushed_sequence():
    hs = [1.0, 0.5, 0.25, 0.125]
    errs = [1e-4, 2e-5, 1.9e-5, 1.85e-5]   # hits a floor on the fine levels
    slope, r2 = G.fit_order(hs, errs)
    assert slope < 1.0
    assert r2 < 0.98


def test_fit_order_rejects_length_mismatch():
    with pytest.raises(CouplingLedgerError):
        G.fit_order([1.0, 0.5], [1e-3])


def test_fit_order_rejects_non_positive_errors():
    with pytest.raises(CouplingLedgerError):
        G.fit_order([1.0, 0.5], [1e-3, 0.0])


def test_fit_order_rejects_degenerate_steps():
    with pytest.raises(CouplingLedgerError):
        G.fit_order([1.0, 1.0], [1e-3, 2e-3])


# ---------------------------------------------------------------------------
# staggered reconstruction algebra
# ---------------------------------------------------------------------------
def _rate_history(signal, h, n):
    """Interval-average rate samples stamped at their interval midpoints."""
    out = []
    for k in range(n):
        t0, t1 = k * h, (k + 1) * h
        exact = (math.sin(t1) - math.sin(t0)) / h      # integral of cos
        out.append(ExchangeSample(t0 + 0.5 * h, exact))
    return out


def _exact_interval_integral(signal, t0, h):
    # Simpson on a smooth signal: enough for the tolerance used below
    return (h / 6.0) * (signal(t0) + 4.0 * signal(t0 + 0.5 * h) + signal(t0 + h))


def test_linear_scheme_is_exact_on_a_linear_rate():
    # constant rate: every scheme reproduces the interval integral exactly
    h = 0.25
    hist = [ExchangeSample(0.5 * h, 3.0), ExchangeSample(1.5 * h, 3.0)]
    assert G._reconstructed(TemporalScheme.ZOH, hist, h) == pytest.approx(3.0 * h)
    assert G._reconstructed(TemporalScheme.LINEAR, hist, h) == pytest.approx(3.0 * h)
    hist3 = hist + [ExchangeSample(2.5 * h, 3.0)]
    assert G._reconstructed(TemporalScheme.QUADRATIC, hist3, h) == pytest.approx(3.0 * h)


def test_quadratic_scheme_is_exact_up_to_the_interval_average_bias():
    """q(t) = t^2, sampled as *interval averages*.

    An interval average of a parabola is ``q(midpoint) + h^2/12``: the bias is
    common to every sample, so the 3-point extrapolation reproduces it exactly
    (the weights sum to one) instead of cancelling it.  The prediction is thus
    the true interval integral **plus** ``h^3/12``.  This is the intrinsic
    ceiling of any reconstruction fed with interval averages: it puts an O(h^2)
    floor on the handover error and is why the second-order scheme measures
    ~2.8 rather than 3.0 in the asymptotic window.
    """
    h = 0.2
    hist = []
    for k in range(3):
        t0, t1 = k * h, (k + 1) * h
        hist.append(ExchangeSample(t0 + 0.5 * h,
                                   (t1 ** 3 - t0 ** 3) / (3.0 * h)))
    t0 = 3 * h
    exact = (t0 + h) ** 3 / 3.0 - t0 ** 3 / 3.0
    assert G._reconstructed(TemporalScheme.QUADRATIC, hist, h) == pytest.approx(
        exact + h ** 3 / 12.0, rel=1e-10)


def test_interval_average_bias_caps_the_attainable_order():
    """Pure reconstruction on a cosine: order 1/2 for zoh/linear, <= 3 quadratic."""
    h0, period = 6.25, 2000.0
    w = 2.0 * math.pi / period

    def average(t0, t1):
        return (math.sin(w * t1) - math.sin(w * t0)) / (w * (t1 - t0))

    orders = {}
    for scheme in G.ORDER_SCHEMES:
        hs, errs = [], []
        for r in (1, 2, 4, 8, 16):
            h = h0 * r
            n = int(round(2 * period / h))
            hist, err, ref = [], 0.0, 0.0
            for k in range(n):
                t0 = k * h
                a = average(t0, t0 + h)
                if k >= 3:
                    err += abs(G._reconstructed(scheme, hist, h) - a * h)
                    ref += abs(a * h)
                hist.append(ExchangeSample(t0 + 0.5 * h, a))
            hs.append(h)
            errs.append(err / ref)
        orders[scheme.value] = G.fit_order(hs, errs)[0]
    assert orders["zoh"] == pytest.approx(1.0, abs=0.1)
    assert orders["linear"] == pytest.approx(2.0, abs=0.15)
    assert 2.5 < orders["quadratic"] <= 3.0


def test_zero_order_hold_has_a_half_interval_lag():
    h = 0.1
    hist = [ExchangeSample(0.5 * h, 1.0)]              # last closed interval: q = 1
    got = G._reconstructed(TemporalScheme.ZOH, hist, h)
    assert got == pytest.approx(1.0 * h)               # holds the old average
    # the coming interval of a rising ramp receives h*q' more than the hold
    assert got < _exact_interval_integral(lambda t: 1.0 + 10.0 * (t - 0.05), h, h)


def test_linear_scheme_reproduces_a_linear_ramp_interval():
    h = 0.1
    hist = _rate_history(math.cos, h, 2)
    # a linear ramp has a linear interval average, so first order is exact
    ramp = [(s.t, 2.0 + 5.0 * s.t) for s in hist]
    hist = [ExchangeSample(t, v) for t, v in ramp]
    t0 = 2 * h
    got = G._reconstructed(TemporalScheme.LINEAR, hist, h)
    exact = 2.0 * h + 5.0 * ((t0 + h) ** 2 - t0 ** 2) / 2.0
    assert got == pytest.approx(exact, rel=1e-12)


def test_reconstruction_polynomial_matches_the_integral_helper():
    h = 0.3
    hist = [ExchangeSample(0.5 * h, 1.0), ExchangeSample(1.5 * h, 1.7),
            ExchangeSample(2.5 * h, 2.1)]
    f0, a1, a2 = reconstruction_poly(TemporalScheme.QUADRATIC, hist[-2], hist[-1],
                                     hist[-3])
    lo, hi = 0.5 * h, 1.5 * h
    manual = f0 * h + 0.5 * a1 * (hi ** 2 - lo ** 2) + a2 / 3.0 * (hi ** 3 - lo ** 3)
    assert G._reconstructed(TemporalScheme.QUADRATIC, hist, h) == pytest.approx(manual)


# ---------------------------------------------------------------------------
# the two bookings
# ---------------------------------------------------------------------------
def test_exact_booking_is_a_round_off_identity():
    kw = dict(period_s=1000.0, n_periods=1.0, fluid_nx=10, fluid_ny=5,
              solid_nx=10, solid_ny=5)
    for ratio in (1, 8):
        run = G.run_energy_exchange(ratio=ratio, scheme=TemporalScheme.LINEAR,
                                    booking="exact", **kw)
        assert run.residual_max < 1e-12
        assert run.mismatch_l1 == 0.0


def test_partitioned_booking_carries_a_physical_error():
    kw = dict(period_s=1000.0, n_periods=1.0, fluid_nx=10, fluid_ny=5,
              solid_nx=10, solid_ny=5)
    exact = G.run_energy_exchange(ratio=8, scheme=TemporalScheme.LINEAR,
                                  booking="exact", **kw)
    recon = G.run_energy_exchange(ratio=8, scheme=TemporalScheme.LINEAR,
                                  booking="reconstructed", **kw)
    assert recon.mismatch_l1 > 1e-3
    assert recon.mismatch_l1 > 1e6 * max(exact.mismatch_l1, 1e-300)


def test_momentum_exact_booking_satisfies_newton_third_law():
    for ratio in (1, 8):
        run = G.run_momentum_exchange(ratio=ratio, scheme=TemporalScheme.LINEAR,
                                      booking="exact", n_periods=1.0)
        assert run.residual_max < 1e-6
        assert run.mismatch_l1 == 0.0


def test_momentum_partitioned_booking_carries_a_physical_error():
    exact = G.run_momentum_exchange(ratio=8, scheme=TemporalScheme.LINEAR,
                                    booking="exact", n_periods=1.0)
    recon = G.run_momentum_exchange(ratio=8, scheme=TemporalScheme.LINEAR,
                                    booking="reconstructed", n_periods=1.0)
    assert recon.mismatch_l1 > 1e-3
    assert recon.mismatch_l1 > 1e6 * max(exact.mismatch_l1, 1e-300)


def test_unknown_booking_is_rejected():
    with pytest.raises(CouplingLedgerError):
        G.run_energy_exchange(ratio=1, booking="telepathy")
    with pytest.raises(CouplingLedgerError):
        G.run_momentum_exchange(ratio=1, booking="telepathy")


def test_independent_scheme_is_rejected_by_the_gate():
    with pytest.raises(CouplingLedgerError):
        G.run_energy_exchange(ratio=1, scheme=TemporalScheme.INDEPENDENT)


# ---------------------------------------------------------------------------
# runtime order gates
# ---------------------------------------------------------------------------
#: Small but asymptotic-enough configuration: 1000-1500 s physical period,
#: 12 x 6 lattice, 12 x 6 solid mesh.  The cheaper 10 x 5 / 1000 s variant was
#: rejected because its zero-order fit leaves the asymptotic range (R^2 0.93).
ENERGY_FAST = dict(period_s=1500.0, n_periods=1.0, fluid_nx=12, fluid_ny=6,
                   solid_nx=12, solid_ny=6)


def _orders(name, **kwargs):
    study = G.order_study(name=name, booking="reconstructed",
                          ratios=(1, 2, 4, 8), **kwargs)
    return study, {s: study["fits"][s]["mismatch_l1"]["order"]
                   for s in study["schemes"]}


def test_energy_handover_order_tracks_the_declared_order():
    study, orders = _orders("energy", **ENERGY_FAST)
    assert orders["zoh"] >= 0.8
    assert orders["linear"] >= 1.6
    assert orders["quadratic"] >= 2.2
    assert orders["zoh"] < orders["linear"] < orders["quadratic"]


def test_momentum_handover_order_tracks_the_declared_order():
    study, orders = _orders("momentum", n_periods=1.0)
    assert orders["zoh"] >= 0.8
    assert orders["linear"] >= 1.5
    assert orders["quadratic"] >= 2.2
    assert orders["zoh"] < orders["linear"] < orders["quadratic"]


def test_ratio_trend_grows_with_the_sub_cycling_ratio():
    study, _ = _orders("energy", **ENERGY_FAST)
    for scheme in study["schemes"]:
        runs = sorted([r for r in study["runs"] if r["scheme"] == scheme],
                      key=lambda r: r["ratio"])
        errs = [r["mismatch_l1"] for r in runs]
        assert all(b >= a for a, b in zip(errs, errs[1:])), (scheme, errs)
        for r, e in zip([x["ratio"] for x in runs], errs):
            assert e == pytest.approx(e)          # finite
    ratios = [r["ratio"] for r in study["runs"]]
    assert sorted(set(ratios)) == [1, 2, 4, 8]


def test_gate_study_passes_the_real_energy_study():
    study = G.order_study(name="energy", booking="reconstructed",
                          ratios=(1, 2, 4, 8), **ENERGY_FAST)
    gates = G.gate_study(study, energy=True)
    assert gates["overall_pass"] is True
    assert gates["criteria"]["energy_order_monotone"]["passed"] is True


def test_gate_study_fails_a_wrong_record():
    study = G.order_study(name="momentum", booking="reconstructed",
                          ratios=(1, 2, 4, 8), n_periods=1.0)
    # flip the declared orders: the gate must notice that the measured orders
    # do not follow them any more
    study["declared_orders"] = {"zoh": 3, "linear": 2, "quadratic": 1}
    study["fits"]["zoh"], study["fits"]["quadratic"] = (study["fits"]["quadratic"],
                                                        study["fits"]["zoh"])
    gates = G.gate_study(study, energy=False)
    assert gates["overall_pass"] is False
    assert gates["criteria"]["momentum_order_monotone"]["passed"] is False


def test_gate_study_ratio_trend_fails_when_the_trend_is_reversed():
    study = G.order_study(name="momentum", booking="reconstructed",
                          ratios=(1, 2, 4, 8), n_periods=1.0)
    for run in study["runs"]:
        run["mismatch_l1"] = 1.0 / run["ratio"]          # invent a decreasing trend
    gates = G.gate_study(study, energy=False)
    assert gates["criteria"]["momentum_ratio_monotone"]["passed"] is False


# ---------------------------------------------------------------------------
# long-time stability and the blindness diagnostic
# ---------------------------------------------------------------------------
def test_long_time_stability_is_bounded():
    st = G.stability_study(name="energy", n_intervals=150, **ENERGY_FAST)
    assert st["all_finite"] is True
    assert st["divergent"] is False
    assert st["growth_ratio"] < 4.0
    assert st["n_intervals"] == 150


def test_identity_diagnostic_is_not_blind():
    diag = G.identity_diagnostic(energy_kw=ENERGY_FAST,
                                 momentum_kw=dict(n_periods=1.0))
    for name in ("energy", "momentum"):
        assert diag[name]["exact_residual"] < 1e-10
        assert diag[name]["reconstructed_mismatch"] > 1e-4
        assert diag[name]["blind"] is False


def test_study_record_round_trips_through_json():
    import json
    study = G.order_study(name="momentum", booking="reconstructed",
                          ratios=(1, 4), schemes=(TemporalScheme.ZOH,),
                          n_periods=0.5)
    payload = json.dumps(study)
    back = json.loads(payload)
    assert back["booking"] == "reconstructed"
    assert len(back["runs"]) == 2
    assert np.isfinite(back["fits"]["zoh"]["mismatch_l1"]["order"])