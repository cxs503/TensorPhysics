"""C5 - stability limits, coordination strategies and interface conservation."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.torsional_chain.acceptance import c5_time_step_coordination
from tensorphysics.torsional_chain.timestep import (AdaptiveTimeStep, SolverRate,
                                                    TimeStepCoordinator, cfl_step,
                                                    circuit_step, exchange_work_residual,
                                                    mesh_step, structural_accuracy_step,
                                                    structural_explicit_step, upwind_step)


@pytest.fixture(scope="module")
def c5():
    return c5_time_step_coordination()


def _rates():
    return [
        SolverRate("flow", lambda s: cfl_step(dx=5.0e-3, speed=s.get("u_max", 1.0), cfl=0.5)),
        SolverRate("structure", lambda s: structural_accuracy_step(s.get("f_max", 28.7), 20)),
        SolverRate("circuit", lambda s: circuit_step(5.0e-3, 0.05, 0.2)),
    ]


def test_cfl_step_formula_and_guard():
    assert cfl_step(dx=0.01, speed=2.0, cfl=0.5) == pytest.approx(2.5e-3)
    with pytest.raises(ValueError):
        cfl_step(dx=0.0, speed=1.0)
    with pytest.raises(ValueError):
        cfl_step(dx=0.01, speed=1.0, cfl=1.5)
    # zero velocity does not produce an infinite step
    assert math.isfinite(cfl_step(dx=0.01, speed=0.0))


def test_structural_limits():
    f = 40.0
    assert structural_explicit_step(f, factor=0.5) == pytest.approx(
        0.5 * 2.0 / (2.0 * math.pi * f))
    assert structural_accuracy_step(f, 20) == pytest.approx(1.0 / (20.0 * f))
    assert structural_accuracy_step(f, 20) < structural_explicit_step(f, 1.0)


def test_mesh_and_circuit_limits():
    assert mesh_step(21, 10.0, 20) == pytest.approx(2.0 * math.pi / (21 * 10.0 * 20))
    assert circuit_step(5.0e-3, 0.05, 0.2) == pytest.approx(0.02)
    with pytest.raises(ValueError):
        mesh_step(2, 10.0)
    with pytest.raises(ValueError):
        circuit_step(0.0, 1.0)


def test_synchronized_plan_respects_every_limit():
    rates = _rates()
    plan = TimeStepCoordinator(rates, strategy="synchronized", safety=0.9).plan({})
    plan.check()
    strict = min(plan.limits.values())
    assert plan.dt_macro == pytest.approx(0.9 * strict)
    assert all(n == 1 for n in plan.substeps.values())


def test_subcycling_plan_is_integral_and_stable():
    rates = _rates()
    plan = TimeStepCoordinator(rates, strategy="subcycling", safety=0.9).plan({})
    plan.check()
    for name, limit in plan.limits.items():
        assert plan.dt_solver[name] <= limit
        assert plan.dt_macro / plan.substeps[name] == pytest.approx(plan.dt_solver[name])
    assert plan.max_ratio >= 1


def test_supercycling_updates_slow_solvers_less_often():
    rates = _rates()
    plan = TimeStepCoordinator(rates, strategy="supercycling", safety=0.9).plan({})
    plan.check()
    fastest = min(plan.limits, key=plan.limits.get)
    assert plan.dt_macro == pytest.approx(0.9 * plan.limits[fastest])
    assert plan.supersteps[fastest] == 1
    assert max(plan.supersteps.values()) >= 1
    for name, limit in plan.limits.items():
        assert plan.dt_solver[name] <= limit


def test_max_ratio_guard_shrinks_macro_step():
    rates = [SolverRate("slow", lambda s: 1.0), SolverRate("fast", lambda s: 1.0e-4)]
    plan = TimeStepCoordinator(rates, strategy="subcycling", safety=1.0, max_ratio=8).plan({})
    assert plan.substeps["fast"] <= 8
    plan.check()
    with pytest.raises(ValueError):
        TimeStepCoordinator([], strategy="subcycling")


def test_coordinator_rejects_bad_configuration():
    with pytest.raises(ValueError):
        TimeStepCoordinator(_rates(), strategy="magic")
    with pytest.raises(ValueError):
        TimeStepCoordinator(_rates(), safety=0.0)
    with pytest.raises(ValueError):
        TimeStepCoordinator(_rates() + [_rates()[0]])


def test_adaptive_controller_grows_gradually_and_stays_stable():
    adaptive = AdaptiveTimeStep(TimeStepCoordinator(_rates(), "synchronized"), growth=1.5)
    limits = {r.name: r.dt_max({}) for r in _rates()}
    dt = adaptive.update({}, previous_dt=1.0e-8)
    sequence = [dt]
    for _ in range(20):
        dt = adaptive.update({}, previous_dt=dt)
        sequence.append(dt)
        assert dt <= min(limits.values()) * (1.0 + 1e-12)
    assert all(b <= 1.5 * a + 1e-18 for a, b in zip(sequence[:-1], sequence[1:]))


def test_adaptive_controller_shrinks_on_stability_alert():
    adaptive = AdaptiveTimeStep(TimeStepCoordinator(_rates(), "synchronized"), shrink=0.5)
    previous = adaptive.update({}, previous_dt=None)
    shrunk = adaptive.update({}, previous_dt=previous, stability_alert=True)
    assert shrunk == pytest.approx(0.5 * previous)


def test_exchange_work_residual_small_for_linear_reconstruction():
    def torque(t):
        return 3.0e4 * math.sin(2.0 * math.pi * 7.0 * t)

    def omega(t):
        return 12.0 + 0.5 * math.sin(2.0 * math.pi * 7.0 * t + 0.3)

    for n_sub in (2, 4, 8):
        out = exchange_work_residual(torque, omega, dt_macro=4.0e-3, n_sub=n_sub, mode="linear")
        assert out["work_residual"] <= 0.02
        assert out["impulse_residual"] <= 0.02


def test_exchange_hold_reconstruction_is_worse_than_linear():
    def torque(t):
        return 3.0e4 * math.sin(2.0 * math.pi * 15.0 * t)

    def omega(t):
        return 12.0 * np.ones_like(np.asarray(t, dtype=float))

    linear = exchange_work_residual(torque, omega, dt_macro=4.0e-3, n_sub=8, mode="linear")
    hold = exchange_work_residual(torque, omega, dt_macro=4.0e-3, n_sub=8, mode="hold")
    assert hold["work_residual"] > linear["work_residual"]


def test_upwind_cfl_stability_boundary():
    dx, velocity = 0.01, 1.0
    x = np.linspace(0.0, 1.0, 100, endpoint=False)
    field = np.sin(2.0 * math.pi * x)

    stable = field.copy()
    for _ in range(300):
        stable = upwind_step(stable, velocity, dx, 0.5 * dx / velocity)
    assert np.all(np.isfinite(stable))
    assert np.max(np.abs(stable)) < 1.5

    unstable = field.copy()
    for _ in range(300):
        unstable = upwind_step(unstable, velocity, dx, 1.5 * dx / velocity)
    assert not np.all(np.isfinite(unstable)) or np.max(np.abs(unstable)) > 10.0


def test_c5_acceptance_block(c5):
    assert c5["criterion"] == "C5"
    assert c5["pass"] is True
    assert c5["worst_work_residual"] <= 0.02
    assert [row["subcycle_ratio"] for row in c5["cases"]] == [2, 4, 8]
    for plan in c5["plans"].values():
        for name, dt in plan["dt_solver"].items():
            assert dt <= plan["limits"][name]