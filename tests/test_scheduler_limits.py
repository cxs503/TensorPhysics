from dataclasses import dataclass, field
import pytest
from tensorphysics import Simulation, SimulationConfig, SimulationError, SimulationState


@dataclass
class CountingSolver:
    name: str = "counter"
    calls: int = 0
    finalized: bool = False

    def initialize(self, context):
        pass

    def advance(self, context):
        self.calls += 1

    def finalize(self, context):
        self.finalized = True

    def snapshot(self, context):
        return {"calls": self.calls}


def test_max_steps_limit_is_not_reported_as_physical_completion():
    solver = CountingSolver()
    sim = Simulation(SimulationConfig(dt=0.1, end_time=10.0, max_steps=3), [solver])
    sim.run()
    assert solver.calls == 3
    assert sim.time == pytest.approx(0.3)
    assert sim.state is SimulationState.LIMIT_REACHED
    with pytest.raises(SimulationError, match="limit_reached"):
        sim.run()


def test_zero_step_budget_and_zero_end_time_have_distinct_states():
    limited = Simulation(SimulationConfig(dt=0.1, end_time=1.0, max_steps=0))
    limited.run()
    assert limited.step == 0
    assert limited.state is SimulationState.LIMIT_REACHED

    completed = Simulation(SimulationConfig(dt=0.1, end_time=0.0))
    completed.run()
    assert completed.step == 0
    assert completed.state is SimulationState.FINISHED


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_invalid_run_step_count_is_rejected(steps):
    sim = Simulation(SimulationConfig(dt=0.1, end_time=1.0))
    with pytest.raises((TypeError, ValueError)):
        sim.run(steps=steps)
