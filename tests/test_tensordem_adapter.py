import pytest

from tensorphysics import Simulation, SimulationConfig, SimulationError, SimulationState
from tensorphysics.adapters import TensorDEMAdapter


class FakeIceDEM:
    def __init__(self, dt=0.025):
        self.dt = dt
        self.time = 0.0
        self.step_count = 0
        self.positions = type("Positions", (), {"shape": (3, 2)})()
        self.calls = []

    def step(self, external_forces=None, *, load_time=None):
        assert load_time == pytest.approx(self.time)
        self.calls.append((self.time, external_forces))
        self.time += self.dt
        self.step_count += 1


def test_tensordem_adapter_subcycles_and_requests_loads_at_each_substep():
    model = FakeIceDEM(dt=0.025)
    load_times = []

    def load_provider(context, dem):
        load_times.append(context.time)
        return ("load", context.step, context.metadata["substep_index"])

    adapter = TensorDEMAdapter(model, load_provider=load_provider)
    simulation = Simulation(SimulationConfig(dt=0.1, end_time=0.2), [adapter])
    try:
        simulation.run()
        assert simulation.state is SimulationState.FINISHED
        assert model.step_count == 8
        assert load_times == pytest.approx([0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175])
        assert [call[1][2] for call in model.calls] == [0, 1, 2, 3] * 2
        snapshot = simulation.snapshots()["TensorDEM"]
        assert snapshot["native_step_count"] == 8
        assert snapshot["particle_count"] == 3
        assert snapshot["restartable"] is False
    finally:
        simulation.close()


def test_tensordem_adapter_rejects_nonintegral_global_step_ratio():
    model = FakeIceDEM(dt=0.03)
    adapter = TensorDEMAdapter(model)
    simulation = Simulation(SimulationConfig(dt=0.1, end_time=0.1), [adapter])
    with pytest.raises(SimulationError, match="integer multiple"):
        simulation.run()
    assert simulation.state is SimulationState.FAILED
    assert model.step_count == 0
    simulation.close()


def test_tensordem_adapter_rejects_native_clock_drift():
    model = FakeIceDEM(dt=0.025)
    model.time = 0.01
    adapter = TensorDEMAdapter(model)
    simulation = Simulation(SimulationConfig(dt=0.1, end_time=0.1), [adapter])
    with pytest.raises(SimulationError, match="does not match simulation time"):
        simulation.run()
    assert simulation.state is SimulationState.FAILED
    simulation.close()
