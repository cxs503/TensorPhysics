"""Contract and conservative-remap tests for the cross-clock coupling package."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.cross_clock import (
    ContractError,
    CouplingContract,
    IntervalSample,
    SolverClock,
    WindowLedger,
    check_tiling,
    conservative_regrid,
    drift_report,
    first_moment,
    overlap_fractions,
    plm_slopes,
    van_leer_limiter,
)


# --------------------------------------------------------------------------- #
# clocks and samples
# --------------------------------------------------------------------------- #
def test_clock_ratio_accepts_only_integer_multiples():
    clock = SolverClock("lbm", dt=2.0e-6)
    assert clock.ratio(2.0e-5) == 10
    assert clock.steps(2.0e-6) == 1
    with pytest.raises(ContractError):
        clock.ratio(2.1e-5)
    with pytest.raises(ContractError):
        SolverClock("bad", dt=0.0)


def test_interval_sample_total_split_and_centroid():
    s = IntervalSample(0.0, 2.0, {"momentum": 3.0, "mass": -1.0})
    assert s.duration == 2.0
    assert s.total("momentum") == pytest.approx(6.0)
    assert s.total("energy") == 0.0
    assert s.centroid() == pytest.approx(1.0)
    parts = s.split(4)
    assert len(parts) == 4
    assert sum(p.total("momentum") for p in parts) == pytest.approx(s.total("momentum"))
    assert parts[1].t0 == pytest.approx(0.5)
    with pytest.raises(ContractError):
        IntervalSample(1.0, 1.0, {"momentum": 1.0})
    with pytest.raises(ContractError):
        IntervalSample(0.0, 1.0, {"entropy": 1.0})


def test_check_tiling_detects_gap_overlap_and_is_clean_otherwise():
    clean = [IntervalSample(i, i + 1.0, {"momentum": 1.0}) for i in range(3)]
    assert check_tiling(clean) == []
    gap = [IntervalSample(0.0, 1.0, {"momentum": 1.0}), IntervalSample(1.5, 2.5, {"momentum": 1.0})]
    assert any("gap" in p for p in check_tiling(gap))
    overlap = [IntervalSample(0.0, 1.0, {"momentum": 1.0}), IntervalSample(0.5, 2.5, {"momentum": 1.0})]
    assert any("overlap" in p for p in check_tiling(overlap))


def test_contract_rejects_bad_quantity_order_and_tolerance():
    with pytest.raises(ContractError):
        CouplingContract("lbm", "fem", ("momentum", "mass"), 1e-2)
    with pytest.raises(ContractError):
        CouplingContract("lbm", "fem", ("mass",), 0.0)
    with pytest.raises(ContractError):
        CouplingContract("lbm", "fem", ("mass",), 1e-2, mode="magic")
    ok = CouplingContract("lbm", "fem", ("mass", "momentum"), 1e-2)
    assert ok.admits(10) and not ok.admits(0)
    assert ok.as_dict()["quantities"] == ["mass", "momentum"]


# --------------------------------------------------------------------------- #
# ledger and drift definitions
# --------------------------------------------------------------------------- #
def test_window_ledger_separates_bookkeeping_from_reference_drift():
    w = WindowLedger(0, 0.0, 1.0, produced={"momentum": 100.0},
                     applied={"momentum": 100.0}, reference={"momentum": 101.0})
    assert w.ledger_drift("momentum", 1.0) == pytest.approx(0.0)
    assert w.reference_drift("momentum", 1.0) == pytest.approx(1.0 / 101.0)
    w.applied["momentum"] = 99.0
    # the normaliser is the largest magnitude in play (still the reference, 101)
    assert w.ledger_drift("momentum", 1.0) == pytest.approx(1.0 / 101.0)


def test_drift_report_floors_the_normaliser():
    windows = [
        WindowLedger(0, 0.0, 1.0, produced={"momentum": 1e-12}, applied={"momentum": 2e-12}),
        WindowLedger(1, 1.0, 2.0, produced={"momentum": 5.0}, applied={"momentum": 5.0}),
    ]
    rep = drift_report(windows, scale=1.0)
    # without the floor the first window would report a 100 % drift
    assert rep["momentum"]["ledger_drift_max"] == pytest.approx(1e-12)


# --------------------------------------------------------------------------- #
# conservative regridding
# --------------------------------------------------------------------------- #
def test_overlap_matrix_tiles_both_partitions():
    src = [0.0, 0.5, 1.0, 2.0]
    dst = [0.0, 0.25, 1.0, 2.0]
    O = overlap_fractions(src, dst)
    assert np.allclose(O.sum(axis=1), np.diff(dst))
    assert np.allclose(O.sum(axis=0), np.diff(src))
    with pytest.raises(ContractError):
        overlap_fractions(src, [0.5, 1.0, 2.0])  # destination not covered


@pytest.mark.parametrize("scheme", ["donor", "plm"])
def test_regrid_conserves_the_total(scheme):
    src = np.linspace(0.0, 1.0, 21)
    q = np.sin(2.0 * np.pi * src[:-1]) + 2.0
    dst = np.linspace(0.0, 1.0, 8)
    res = conservative_regrid(src, q, dst, scheme)
    assert res.total_dst == pytest.approx(res.total_src, rel=0, abs=1e-13)


def test_plm_regrid_is_second_order_in_the_reconstructed_density():
    """Rebinning a smooth signal: donor-cell error is O(h), slope-limited PLM O(h^2).

    The target grid is *fine* (inside the source cells), so the comparison is the
    reconstructed density against the analytic cell average: that is exactly the
    information a clock-mismatched partner is missing.
    """
    a = 0.5

    def q_exact(lo, hi):
        return (np.exp(a * hi) - np.exp(a * lo)) / a

    dense = np.linspace(0.0, 1.0, 513)
    dw = np.diff(dense)
    qh = np.array([q_exact(lo, hi) for lo, hi in zip(dense[:-1], dense[1:])]) / dw
    err = {"donor": [], "plm": []}
    for n in (4, 8, 16, 32):
        edges = np.linspace(0.0, 1.0, n + 1)
        totals = np.array([q_exact(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])])
        assert totals.sum() == pytest.approx(q_exact(0.0, 1.0))
        for scheme in ("donor", "plm"):
            res = conservative_regrid(edges, totals, dense, scheme)
            dens = res.dst_totals / dw
            err[scheme].append(float(np.sum(np.abs(dens - qh) * dw)))
    for scheme in ("donor", "plm"):
        e = np.asarray(err[scheme])
        order = math.log2(e[0] / e[-1]) / math.log2(8.0)
        assert order > (0.7 if scheme == "donor" else 1.7), f"{scheme} measured order {order:.2f}"
    assert err["plm"][-1] < err["donor"][-1]


def test_plm_limiter_does_not_create_extrema():
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    q = np.array([1.0, 1.0, 1.0, 5.0, 1.0])
    slopes = plm_slopes(q, edges)
    assert np.all(np.isfinite(slopes))
    assert np.allclose(van_leer_limiter(np.array([1.0]), np.array([-1.0])), [0.0])
    dense = np.linspace(0.0, 5.0, 51)
    res = conservative_regrid(edges, q, dense, "plm")
    assert res.dst_totals.min() >= -1e-12
    assert res.dst_totals.max() <= 5.0 / 1.0 + 1e-12


def test_regrid_report_exposes_both_schemes():
    from tensorphysics.cross_clock import regrid_report

    rep = regrid_report([0.0, 1.0, 2.0], [1.0, 2.0], [0.0, 0.5, 2.0])
    assert set(rep) == {"donor", "plm"}
    assert all(abs(v["relative_drift"]) < 1e-13 for v in rep.values())