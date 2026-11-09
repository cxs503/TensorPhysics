"""Acceptance tests for the P40W3B-N09 multi-solver energy-conservation contract.

The tests exercise the four layers of ``tensorphysics.coupling_energy``:

* reference-solver conservation (LBM scalar, FEM lumped capacity, DEM contact),
* interface reconstruction order (zero / first / second),
* the three coupled scenarios, including their *falsification arm* (the
  independent-flux control must lose the contract, otherwise the ledger would be
  blind),
* the acceptance numbers recorded in ``artifacts/p40w3b_n09/verification.json``.

Configurations are deliberately small so that the whole module runs in a few
tens of seconds; the production numbers are produced by
``scripts/run_coupling_energy_contract.py``.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from tensorphysics.coupling_energy import scenarios as S
from tensorphysics.coupling_energy.exchange import (
    SCHEME_ORDER,
    ExchangeSample,
    MultiRateSchedule,
    TemporalScheme,
    quadrature_error,
    reconstruction_order_scan,
    reconstruction_poly,
)
from tensorphysics.coupling_energy.ledger import (
    CouplingLedgerError,
    EnergyLedger,
    MomentumLedger,
)
from tensorphysics.coupling_energy.solvers import (
    DEM2D,
    ContactLaw,
    Q4Mesh,
    Q4PlaneStressFEM,
    ThermalLBM2D,
)


# ---------------------------------------------------------------------------
# reference solvers
# ---------------------------------------------------------------------------
def test_lbm_is_exactly_conservative_with_insulated_walls():
    lbm = ThermalLBM2D(10, 12, tau=1.25, T_init=3.0)
    for _ in range(200):
        lbm.step()
    assert lbm.total() == pytest.approx(3.0 * 120, rel=1e-13)


def test_lbm_wall_injection_equals_measured_scalar_change():
    lbm = ThermalLBM2D(12, 8, tau=1.15, T_init=1.0)
    for k in range(40):
        before = lbm.total()
        injected = lbm.step(np.full(12, 20.0 * math.sin(0.2 * k)))
        assert lbm.total() - before == pytest.approx(injected, abs=1e-11)


def test_lbm_dirichlet_wall_reaches_the_wall_temperature():
    lbm = ThermalLBM2D(16, 10, tau=1.25)
    for _ in range(1500):
        lbm.step(np.full(16, 50.0))
    # the anti-bounce-back wall value sits half a cell outside the first node, so
    # the node value converges to Tw + O(dx) -- a documented first-order
    # artefact of the one-node Dirichlet wall (measured +6.3 % here).
    assert np.all(lbm.T[0, 2:-2] == pytest.approx(50.0, rel=0.08))
    assert np.all(lbm.T[-1, :] == pytest.approx(50.0, rel=0.08))
    assert lbm.T.mean() == pytest.approx(50.0, rel=0.02)


def test_dem_contact_force_is_reciprocal():
    law = ContactLaw(stiffness=1.0e6)
    dem = DEM2D(np.array([[0.55, 0.0]]), np.array([[-1.0, 0.0]]),
                np.array([0.1]), density=1000.0, law=law)
    f_particles, f_nodes, overlaps, _ = dem.contact_forces(
        np.array([0.5, 0.5]), np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([0]))
    assert overlaps[0] == pytest.approx(0.05)
    assert f_particles[0, 0] == pytest.approx(-f_nodes.sum(), rel=1e-14)
    assert f_particles[0, 0] > 0.0     # compression pushes the particle away
    assert f_nodes[0, 0] < 0.0         # ... and the wall in the other direction


def test_dem_free_flight_conserves_energy_and_momentum():
    dem = DEM2D(np.array([[0.0, 0.0]]), np.array([[2.0, 0.5]]), np.array([0.01]),
                density=900.0)
    e0, p0 = dem.kinetic_energy, dem.momentum()
    for _ in range(500):
        dem.step(1e-5, np.zeros((1, 2)))
    assert dem.kinetic_energy == pytest.approx(e0, rel=1e-12)
    assert dem.momentum() == pytest.approx(p0, rel=1e-12)


def test_plate_first_frequency_matches_euler_bernoulli_axial_mode():
    out = S.plate_first_frequency(nx=16, ny=2, n_steps=2500)
    assert out["rel_error"] < 0.03


# ---------------------------------------------------------------------------
# interface reconstruction
# ---------------------------------------------------------------------------
def _signal(t: float) -> float:
    return math.sin(2.0 * math.pi * t) + 0.3 * math.cos(6.0 * math.pi * t)


def test_declared_scheme_orders_are_measured():
    scan = reconstruction_order_scan(_signal, 0.2, 2.0)
    assert 0.6 < scan["zoh"].order < 1.5
    assert 1.5 < scan["linear"].order < 2.5
    assert 2.4 < scan["quadratic"].order < 3.6
    assert SCHEME_ORDER[TemporalScheme.QUADRATIC] > SCHEME_ORDER[TemporalScheme.LINEAR]


def test_quadratic_reconstruction_is_exact_on_a_quadratic():
    # samples of f(t) = 1 + 2 t^2 at t = 0, 1, 2 fix that parabola exactly
    samples = [ExchangeSample(float(t), 1.0 + 2.0 * t * t) for t in (0.0, 1.0, 2.0)]
    f0, a1, a2 = reconstruction_poly(TemporalScheme.QUADRATIC, samples[1], samples[2],
                                     samples[0])
    for s in (0.25, 1.0, 2.0):
        value = f0 + a1 * s + a2 * s * s
        exact = 1.0 + 2.0 * (2.0 + s) ** 2
        assert value == pytest.approx(exact, rel=1e-12)


def test_independent_scheme_refuses_shared_history():
    with pytest.raises(CouplingLedgerError):
        reconstruction_poly(TemporalScheme.INDEPENDENT, None, ExchangeSample(0.0, 1.0))


def test_subcycling_schedule():
    sched = MultiRateSchedule(dt_fine=0.25, ratio=4)
    assert sched.dt_coarse == pytest.approx(1.0)
    assert sched.exchange_instants(3) == [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(CouplingLedgerError):
        MultiRateSchedule(dt_fine=0.25, ratio=0)


# ---------------------------------------------------------------------------
# ledgers
# ---------------------------------------------------------------------------
def test_energy_ledger_double_entry_preserves_total():
    ledger = EnergyLedger(sides=("fluid", "solid"))
    ledger.set("solid", 10.0)
    ledger.sample(0.0)
    ledger.exchange("solid", "fluid", 2.5)
    ledger.sample(1.0)
    assert ledger.total() == pytest.approx(10.0)
    assert ledger.residual() == pytest.approx(0.0)
    assert ledger.interface == pytest.approx(2.5)


def test_energy_ledger_detects_non_reciprocal_transfer():
    ledger = EnergyLedger(sides=("fluid", "solid"))
    ledger.set("solid", 10.0)
    ledger.sample(0.0)
    ledger.add("fluid", 2.5)
    ledger.add("solid", -2.0)      # the donor gave less than the receiver got
    ledger.sample(1.0)
    assert ledger.residual() == pytest.approx(0.05)


def test_momentum_ledger_rejects_same_side_exchange():
    ledger = MomentumLedger(sides=("solid", "particles"))
    with pytest.raises(CouplingLedgerError):
        ledger.exchange("solid", "solid", 1.0, 0.0)


# ---------------------------------------------------------------------------
# (a) thermal coupling
# ---------------------------------------------------------------------------
def test_thermal_coupling_conserves_energy_and_matches_the_exact_solution():
    res = S.run_thermal_lbm_fem(ratio=1, scheme=TemporalScheme.LINEAR, t_end=1000.0,
                                relax=0.2, sample_times=[250.0, 500.0, 1000.0])
    assert res.residual_max < 1e-12
    assert res.metrics["profile_rel_l2_error"] < 0.03


def test_thermal_coupling_error_grows_with_the_subcycling_ratio():
    coarse = S.run_thermal_lbm_fem(ratio=1, scheme=TemporalScheme.LINEAR, t_end=1000.0,
                                   relax=0.2, sample_times=[500.0, 1000.0])
    fine_step = S.run_thermal_lbm_fem(ratio=8, scheme=TemporalScheme.LINEAR, t_end=1000.0,
                                      relax=0.2, sample_times=[500.0, 1000.0])
    assert fine_step.n_steps < coarse.n_steps
    assert fine_step.metrics["profile_rel_l2_error"] > coarse.metrics["profile_rel_l2_error"]
    assert fine_step.residual_max < 1e-12


def test_thermal_independent_flux_control_breaks_conservation():
    shared = S.run_thermal_lbm_fem(ratio=2, scheme=TemporalScheme.LINEAR, t_end=1000.0,
                                   relax=0.2)
    control = S.run_thermal_lbm_fem(ratio=2, scheme=TemporalScheme.LINEAR, t_end=1000.0,
                                    relax=0.2, exchange_mode="independent")
    assert shared.residual_max < 1e-12
    assert control.residual_max > 1e-3
    assert control.residual_max > 1e3 * shared.residual_max


# ---------------------------------------------------------------------------
# (b) mechanical coupling
# ---------------------------------------------------------------------------
def test_mechanical_coupling_obeys_newtons_third_law():
    res = S.run_mechanical_fem_dem(ratio=4, scheme=TemporalScheme.LINEAR, n_macro=300)
    assert max(res.series["momentum_residual"]) < 1e-10
    assert res.metrics["impulse_total_Ns"] != 0.0        # contact really happened
    assert res.metrics["max_overlap_m"] > 0.0


def test_mechanical_independent_reaction_control_breaks_momentum():
    control = S.run_mechanical_fem_dem(ratio=4, scheme=TemporalScheme.LINEAR, n_macro=300,
                                       exchange_mode="independent")
    assert max(control.series["momentum_residual"]) > 1e-3


def test_mechanical_energy_error_decreases_with_the_subcycling_ratio():
    coarse = S.run_mechanical_fem_dem(ratio=1, scheme=TemporalScheme.LINEAR, n_macro=250)
    fine = S.run_mechanical_fem_dem(ratio=8, scheme=TemporalScheme.LINEAR, n_macro=250)
    assert fine.residual_max < coarse.residual_max


# ---------------------------------------------------------------------------
# (c) three-field chain
# ---------------------------------------------------------------------------
def test_three_field_energy_chain_is_conserved():
    res = S.run_three_field(ratio=2, scheme=TemporalScheme.LINEAR, n_macro=80)
    assert res.residual_max < 1e-3
    assert res.metrics["thermal_residual_max"] < 1e-10
    # both physics really advanced: thermal field charged and contact occurred
    assert res.series["q_external"][-1] > 0.0
    assert max(res.series["e_contact"]) > 0.0


def test_three_field_independent_control_breaks_the_chain():
    control = S.run_three_field(ratio=2, scheme=TemporalScheme.LINEAR, n_macro=80,
                                exchange_mode="independent")
    assert control.residual_max > 1e-2


# ---------------------------------------------------------------------------
# quick end-to-end suite (the same code path as the artifact script)
# ---------------------------------------------------------------------------
def test_quick_suite_produces_the_ratio_scheme_matrix():
    out = S.run_suite(ratios=(1, 2), schemes=(TemporalScheme.ZOH, TemporalScheme.LINEAR),
                      include_control=True,
                      thermal_kw=dict(t_end=300.0, relax=0.2),
                      mechanical_kw=dict(n_macro=80),
                      three_field_kw=dict(n_macro=30))
    for name in ("thermal", "mechanical", "three_field"):
        assert len(out[name]) == 2 * 2 + 2      # ratios x schemes + controls
        assert all(run["residual_max"] == run["residual_max"] for run in out[name])
    assert out["thermal"][0]["scheme"] == "zoh"
    assert any(r["exchange_mode"] == "independent" for r in out["thermal"])

# ---------------------------------------------------------------------------
# acceptance gate over the same suite (frozen criteria, three-valued outcomes)
# ---------------------------------------------------------------------------
def test_quick_suite_is_gated_by_the_frozen_criteria():
    """The contract test now ends where the campaign ends: at a verdict.

    A partial suite cannot be certified, so the verdict is ``WITHHELD`` -- what
    matters here is that the gate is *fed* by real runs and that every criterion
    it can evaluate agrees with the direct assertions above.
    """
    from tensorphysics.coupling_energy import acceptance as A

    out = S.run_suite(ratios=(1, 2),
                      schemes=(TemporalScheme.ZOH, TemporalScheme.LINEAR),
                      include_control=True,
                      thermal_kw=dict(t_end=300.0, relax=0.2,
                                      sample_times=[150.0, 300.0]),
                      mechanical_kw=dict(n_macro=200),
                      three_field_kw=dict(n_macro=30))
    report = A.evaluate(out, {"order_scan": {"zoh": 1.0, "linear": 2.0,
                                             "quadratic": 2.8}})
    assert report.digest == A.CRITERIA_DIGEST
    assert report.verdict == "WITHHELD"                     # 12 of 48 runs
    for cid in ("thermal_energy_conservation", "thermal_falsification_ratio",
                "mechanical_momentum_conservation", "mechanical_contact_engaged",
                "three_field_thermal_subchain", "three_field_falsification_ratio"):
        assert report.outcomes[cid].status == "PASS", (cid, report.outcomes[cid])
    assert report.outcomes["matrix_completeness"].status == "WITHHELD"
    # a withheld record is not silently accepted
    with pytest.raises(A.AcceptanceError):
        A.assert_accepted(report)

# ---------------------------------------------------------------------------
# the committed acceptance record: the manuscript's numbers must be traceable
# ---------------------------------------------------------------------------
ARTIFACT = (pathlib.Path(__file__).resolve().parents[1]
            / "artifacts" / "p40w3b_n09" / "verification.json")


def test_committed_verification_record_regates_to_real():
    """The shipped ``verification.json`` must re-gate to REAL on its own.

    The manuscript quotes the acceptance table; this guards that the quoted
    numbers are backed by a committed, machine-readable record rather than
    hand-copied prose, and that the record stays small enough to ship (the raw
    plotting series are pruned by ``_compact_suite``).
    """
    import json

    from tensorphysics.coupling_energy import acceptance as A

    assert ARTIFACT.is_file(), f"missing committed acceptance record: {ARTIFACT}"
    assert ARTIFACT.stat().st_size < 512_000, "verification.json is not compact"

    record = json.loads(ARTIFACT.read_text())
    assert record["overall_pass"] is True
    assert record["verdict"] == "REAL"
    assert record["mission_id"] == 7597

    report = A.evaluate_record(record)
    assert report.verdict == "REAL"
    assert report.digest == record["criteria_digest"] == A.CRITERIA_DIGEST
    assert not report.failures and not report.withheld
    # the record carries no plotting-only series, only the gated ones
    kept_series = {k for group in record["runs"].values()
                   for run in group for k in (run.get("series") or {})}
    assert kept_series <= {"momentum_residual", "overlap"}
