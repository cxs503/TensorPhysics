from dataclasses import dataclass, field
import pytest
from tensorphysics import Simulation, SimulationConfig, SimulationError, SimulationState, SolverState

@dataclass
class CounterSolver:
    name: str
    calls: list = field(default_factory=list)
    initialized: bool = False
    finalized: bool = False
    def initialize(self, context):
        self.initialized = True
    def advance(self, context):
        self.calls.append((context.step, context.time, context.dt))
    def finalize(self, context):
        self.finalized = True
    def snapshot(self, context):
        return SolverState(self.name, context.step, context.time, {"calls": len(self.calls)})

def test_registration_order_and_final_time():
    events = []
    class Ordered(CounterSolver):
        def advance(self, context):
            events.append(self.name)
            super().advance(context)
    sim = Simulation(SimulationConfig(0.1, 0.3), [Ordered("a"), Ordered("b")])
    sim.run()
    assert events == ["a", "b"] * 3
    assert sim.step == 3
    assert sim.time == pytest.approx(0.3)
    assert sim.state is SimulationState.FINISHED

def test_partial_run_can_resume():
    solver = CounterSolver("counter")
    sim = Simulation(SimulationConfig(0.25, 1.0), [solver])
    sim.run(steps=2)
    assert sim.step == 2 and sim.state is SimulationState.INITIALIZED
    sim.run()
    assert sim.step == 4 and sim.state is SimulationState.FINISHED

def test_duplicate_names_rejected():
    sim = Simulation(SimulationConfig(0.1, 1.0), [CounterSolver("x")])
    with pytest.raises(ValueError, match="duplicate"):
        sim.register(CounterSolver("x"))

def test_failed_solver_marks_simulation_failed():
    class Broken(CounterSolver):
        def advance(self, context):
            raise RuntimeError("intentional test failure")
    sim = Simulation(SimulationConfig(0.1, 1.0), [Broken("broken")])
    with pytest.raises(SimulationError, match="advance/coupling failed"):
        sim.run()
    assert sim.state is SimulationState.FAILED

def test_initialize_failure_cleans_up_previous_solver():
    class BrokenInit(CounterSolver):
        def initialize(self, context):
            raise RuntimeError("bad init")
    first = CounterSolver("first")
    sim = Simulation(SimulationConfig(0.1, 1.0), [first, BrokenInit("broken")])
    with pytest.raises(SimulationError, match="initialization failed"):
        sim.initialize()
    assert first.finalized
    assert sim.state is SimulationState.FAILED

def test_empty_simulation():
    sim = Simulation(SimulationConfig(0.5, 1.0))
    sim.run()
    assert sim.step == 2
