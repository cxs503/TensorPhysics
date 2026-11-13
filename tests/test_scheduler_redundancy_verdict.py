"""Pin the premises of docs/SCHEDULER_REDUNDANCY_VERDICT.md (m18417).

These are the two structural facts the "demote to optional" verdict rests on, kept
as fast regression tests (FakeIceDEM, no tensordem dependency):

1. chunked run() returns the simulation to INITIALIZED, so an external driver can
   interleave sampling/checkpointing (the orchestration layer's one real gain);
2. the adapter snapshot carries no physics state and is explicitly not
   restartable, so a TensorPhysics checkpoint cannot restart a run.
"""
from tensorphysics import Simulation, SimulationConfig, SimulationState
from tensorphysics.adapters import TensorDEMAdapter


class FakeIceDEM:
    def __init__(self, dt: float = 0.025) -> None:
        self.dt = dt
        self.time = 0.0
        self.step_count = 0
        self.positions = type("Positions", (), {"shape": (4, 2)})()

    def step(self, external_forces=None, *, load_time=None) -> None:
        assert load_time == self.time
        self.time += self.dt
        self.step_count += 1


def test_chunked_run_returns_to_initialized_so_drivers_can_interleave():
    model = FakeIceDEM(dt=0.05)
    simulation = Simulation(SimulationConfig(dt=0.05, end_time=0.5), [TensorDEMAdapter(model)])
    try:
        states = []
        for _ in range(5):
            simulation.run(steps=2)
            states.append(simulation.state)
        assert states[:-1] == [SimulationState.INITIALIZED] * 4
        assert states[-1] is SimulationState.FINISHED
        assert model.step_count == 10
    finally:
        simulation.close()


def test_adapter_snapshot_carries_no_physics_state_and_is_not_restartable():
    model = FakeIceDEM()
    adapter = TensorDEMAdapter(model)
    simulation = Simulation(SimulationConfig(dt=0.025, end_time=0.05), [adapter])
    try:
        simulation.run(steps=1)
        payload = simulation.snapshots()[adapter.name]
        assert payload["restartable"] is False
        assert not {"positions", "velocities", "alive", "accounting"} & set(payload)
        # no restore/resume surface exists on either half of the orchestration layer
        assert not [name for name in dir(Simulation)
                    if any(token in name.lower() for token in ("restore", "resume"))]
        assert not [name for name in dir(TensorDEMAdapter)
                    if any(token in name.lower() for token in ("restore", "resume"))]
    finally:
        simulation.close()