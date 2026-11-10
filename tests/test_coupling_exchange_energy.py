"""Tests of the interval-complete *energy* ledger of the coupling exchange.

The instrument under test is
:mod:`tensorphysics.coupling_energy.exchange_energy`.  Every test states the
invariant it protects; the numbers are deliberately small so the suite stays
fast, and the full acceptance ladder is run by
``scripts/run_p223w3_n05_energy_ledger.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.coupling_energy.exchange_energy import (
    ENERGY_ARMS,
    _affine_stepper,
    _sub_trajectory,
    energy_account_for,
    gate_energy,
    run_energy_ladder,
    run_energy_level,
    run_energy_study,
)
from tensorphysics.coupling_energy.ledger import CouplingLedgerError
from tensorphysics.coupling_energy.time_exchange import build_chain

SMALL = {"viv": dict(t_end=0.5, dt_fine=1.0e-3),
         "weld": dict(t_end=0.6, dt_fine=1.0e-3),
         "settling": dict(t_end=0.3, dt_fine=5.0e-4)}


def small_chain(name: str):
    return build_chain(name, **SMALL[name])


@pytest.mark.parametrize("name", ["viv", "weld", "settling"])
def test_energy_identity_matches_rhs(name):
    """dE/dt of the energy table must equal ``q g - dis`` of the chain rhs."""
    chain = build_chain(name)
    acct = energy_account_for(chain)
    y = np.asarray(chain.y0, dtype=float) + 1.0e-3
    t0 = 0.37
    dy = np.asarray(chain.rhs(t0, y), dtype=float)
    eps = 1.0e-7
    dE = (acct.stored_of(y + eps * dy) - acct.stored_of(y - eps * dy)) / (2.0 * eps)
    predicted = chain.mms.flux(t0) * acct.conjugate_of(y) - acct.dissipation_of(y)
    assert abs(dE - predicted) / max(abs(predicted), 1e-300) <= 1e-6


@pytest.mark.parametrize("name", ["viv", "weld", "settling"])
def test_affine_stepper_reproduces_chain_advance(name):
    """The sub-grid must be the chain's own trajectory, not an approximation."""
    chain = build_chain(name)
    y = np.asarray(chain.y0, dtype=float) + 1.0e-2
    h, rate, substeps = 1.0e-3, 2.5, 8
    m, b, c = _affine_stepper(chain, h / substeps)
    ys = [y]
    for _ in range(substeps):
        ys.append(m @ ys[-1] + b * rate + c)
    direct = [np.asarray(chain.advance(y, rate, (j + 1) * h / substeps), dtype=float)
              for j in range(substeps)]
    for composed, exact in zip(ys[1:], direct):
        scale = max(float(np.max(np.abs(exact))), 1e-30)
        assert float(np.max(np.abs(composed - exact))) / scale <= 1e-9


@pytest.mark.parametrize("name", ["viv", "weld", "settling"])
def test_affine_stepper_refuses_odd_substeps(name):
    chain = build_chain(name)
    with pytest.raises(CouplingLedgerError):
        _sub_trajectory(chain, np.asarray(chain.y0, dtype=float), 1.0, 1e-3, 3)


@pytest.mark.parametrize("name", ["viv", "weld"])
def test_energy_defect_shows_declared_order(name):
    """lag is first order in interface energy, the CN midpoint second order."""
    chain = small_chain(name)
    ladder = {"lag": run_energy_ladder(chain, "lag", dt_fractions=(2.0, 1.0, 0.5)),
              "centered": run_energy_ladder(chain, "centered",
                                            dt_fractions=(2.0, 1.0, 0.5))}
    lag_order = ladder["lag"].as_dict()["energy_defect_order"]
    cn_order = ladder["centered"].as_dict()["energy_defect_order"]
    assert 0.8 <= lag_order <= 1.2, lag_order
    assert 1.8 <= cn_order <= 2.2, cn_order


@pytest.mark.parametrize("name", ["viv", "weld", "settling"])
def test_no_delay_energy_residual_below_1e_5(name):
    """Criterion (d): a time-centred exchange keeps the energy residual < 1e-5."""
    chain = small_chain(name)
    for scheme in ("centered", "predictor_corrector"):
        ladder = run_energy_ladder(chain, scheme, dt_fractions=(1.0, 0.5, 0.25))
        finest = ladder.levels[0]
        assert finest.energy_defect <= 1.0e-5, (name, scheme, finest.energy_defect)
        assert finest.reference_energy_error <= 1.0e-5, (
            name, scheme, finest.reference_energy_error)
    lag = run_energy_ladder(chain, "lag", dt_fractions=(1.0, 0.5, 0.25)).levels[0]
    centred = run_energy_ladder(chain, "centered",
                                dt_fractions=(1.0, 0.5, 0.25)).levels[0]
    assert lag.energy_defect >= 10.0 * centred.energy_defect


def test_predictor_corrector_reproduces_centered_energy():
    """The causal operator must deliver the same interface energy as CN."""
    chain = small_chain("weld")
    cn = run_energy_level(chain, "centered")
    pc = run_energy_level(chain, "predictor_corrector")
    assert abs(pc.interface_energy - cn.interface_energy) / max(
        abs(cn.interface_energy), 1e-300) <= 1e-12


@pytest.mark.parametrize("name", ["viv", "weld", "settling"])
def test_energy_balance_closes(name):
    """E(T) - E(0) - W_applied + D must close: the instrument cross-checks itself."""
    chain = small_chain(name)
    level = run_energy_level(chain, "centered", substeps=8)
    assert level.energy_balance_residual <= 1e-9, level.energy_balance_residual


def test_balance_residual_is_quadrature_limited():
    """Refining the sub-grid reduces the closure error (it is not a leak)."""
    chain = small_chain("settling")
    residuals = [run_energy_level(chain, "centered", substeps=k).energy_balance_residual
                 for k in (4, 8, 16)]
    assert residuals[0] / residuals[1] >= 6.0
    assert residuals[1] / residuals[2] >= 6.0


def test_attribution_complete_leaves_no_unattributed_energy():
    """With every sub-window booked, the booking loss vanishes.

    The remaining ``deficit`` is the *operator* channel (released but not
    delivered), so with a complete rule it must coincide with the measured
    interface energy defect: the two loss channels of the ledger separate
    cleanly instead of hiding each other.
    """
    chain = small_chain("weld")
    for ratio in (2, 4, 8):
        level = run_energy_level(chain, "centered", ratio=ratio)
        assert level.unattributed_share <= 1e-12
        assert level.attribution_identity_residual <= 1e-12
        assert level.closed_intervals == level.intervals
        assert level.deficit_share == pytest.approx(level.net_energy_error,
                                                    rel=1e-6, abs=1e-15)
        assert level.max_interval_energy_gap == pytest.approx(0.0, abs=1e-15)


def test_tail_booking_loses_the_reported_energy_band():
    """Booking only the last sub-window loses (r-1)/r of the exchanged energy."""
    chain = small_chain("weld")
    for ratio in (8, 16):
        level = run_energy_level(chain, "centered", ratio=ratio, tail_booking=True)
        predicted = (ratio - 1.0) / ratio
        assert abs(level.unattributed_share - predicted) <= 0.03, (
            ratio, level.unattributed_share, predicted)
    assert 0.85 <= run_energy_level(chain, "centered", ratio=8,
                                    tail_booking=True).unattributed_share <= 0.90
    assert 0.92 <= run_energy_level(chain, "centered", ratio=16,
                                    tail_booking=True).unattributed_share <= 0.95


def test_destamp_restores_second_order_in_energy():
    chain = small_chain("weld")
    raw = run_energy_ladder(chain, "centered", stamp_offset_ratio=0.5,
                            dt_fractions=(2.0, 1.0, 0.5))
    fixed = run_energy_ladder(chain, "centered", stamp_offset_ratio=0.5, recenter=True,
                              dt_fractions=(2.0, 1.0, 0.5))
    raw_order = raw.as_dict()["energy_defect_order"]
    fixed_order = fixed.as_dict()["energy_defect_order"]
    assert 0.85 <= raw_order <= 1.2, raw_order
    assert 1.8 <= fixed_order <= 2.2, fixed_order
    gain = raw.levels[0].energy_defect / fixed.levels[0].energy_defect
    assert gain >= 10.0, gain


def test_gate_energy_passes_on_a_single_chain_study():
    """The gate must agree with the study it is handed (weld-only, fast ladder)."""
    study = run_energy_study(chains=("weld",), dt_fractions=(4.0, 2.0, 1.0, 0.5),
                             tail_ratios=(2, 8, 16))
    verdict = gate_energy(study)
    failed = [c["key"] for c in verdict["criteria"] if c["verdict"] != "pass"]
    assert verdict["verdict"] == "pass", failed
    assert verdict["passed"] == verdict["total"]
    assert set(ENERGY_ARMS) <= set(study["chains"]["weld"]["arms"])