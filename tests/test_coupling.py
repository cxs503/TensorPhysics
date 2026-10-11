import pytest

from tensorphysics import (
    CouplingContext, CouplingError, CouplingManager, CouplingPhase,
    FieldDescriptor, FieldStore, Simulation, SimulationConfig, SimulationError,
    SimulationState, SolverContext,
)


class SourceReservoir:
    name = "source"

    def __init__(self, mass=1.0, rate=0.1):
        self.mass = mass
        self.rate = rate
        self.pending_flux = 0.0

    def initialize(self, context):
        pass

    def advance(self, context):
        self.pending_flux = min(self.rate * context.dt / 0.1, self.mass)
        self.mass -= self.pending_flux

    def finalize(self, context):
        pass

    def snapshot(self, context):
        return {"mass": self.mass}


class ReceiverReservoir:
    name = "receiver"

    def __init__(self, fields, mass=0.0):
        self.fields = fields
        self.mass = mass
        self.received_steps = []

    def initialize(self, context):
        pass

    def advance(self, context):
        flux = self.fields.read(
            "transfer.mass", current_step=context.step, max_age_steps=0
        )
        self.mass += flux.data
        self.received_steps.append(flux.step)

    def finalize(self, context):
        pass

    def snapshot(self, context):
        return {"mass": self.mass}


def make_coupled_simulation(*, end_time=1.0):
    fields = FieldStore()
    fields.register(FieldDescriptor(
        name="transfer.mass", producer="source", units="kg",
        location="lumped_reservoir",
    ))
    coupling = CouplingManager(fields)
    source_solver = SourceReservoir()
    receiver_solver = ReceiverReservoir(fields)

    def publish_flux(context):
        # The source solver has advanced; its pending transfer is now available.
        assert context.solver.step >= 0
        fields.publish(
            "transfer.mass", source_solver.pending_flux,
            producer="source", step=context.solver.step, time=context.solver.time,
        )

    coupling.register(
        CouplingPhase.AFTER_SOLVER, publish_flux, solver_name="source"
    )
    simulation = Simulation(
        SimulationConfig(dt=0.1, end_time=end_time),
        [source_solver, receiver_solver], coupling=coupling,
    )
    return simulation, source_solver, receiver_solver, fields


def test_coupled_reservoir_transfers_mass_conservatively():
    simulation, source, receiver, fields = make_coupled_simulation()
    initial_total = source.mass + receiver.mass
    try:
        simulation.run()
        assert simulation.state is SimulationState.FINISHED
        assert source.mass == pytest.approx(0.0)
        assert receiver.mass == pytest.approx(1.0)
        assert source.mass + receiver.mass == pytest.approx(initial_total, abs=1e-12)
        assert receiver.received_steps == list(range(10))
        assert fields.read("transfer.mass").version == 10
    finally:
        simulation.close()


def test_phase_callbacks_execute_by_priority_then_registration_order():
    manager = CouplingManager()
    calls = []
    manager.register(CouplingPhase.PRE_STEP, lambda ctx: calls.append("late"), priority=5)
    manager.register(CouplingPhase.PRE_STEP, lambda ctx: calls.append("first"), priority=-1)
    manager.register(CouplingPhase.PRE_STEP, lambda ctx: calls.append("middle"), priority=0)
    manager.execute(
        CouplingPhase.PRE_STEP,
        CouplingContext(SolverContext(0, 0.0, 0.1), manager.fields),
    )
    assert calls == ["first", "middle", "late"]


def test_after_solver_callback_requires_solver_name():
    manager = CouplingManager()
    manager.register(CouplingPhase.AFTER_SOLVER, lambda ctx: None, solver_name="source")
    context = CouplingContext(SolverContext(0, 0.0, 0.1), manager.fields)
    manager.execute(CouplingPhase.AFTER_SOLVER, context, solver_name="receiver")
    with pytest.raises(ValueError, match="solver_name is required"):
        manager.execute(CouplingPhase.AFTER_SOLVER, context)


def test_callback_exception_is_wrapped_with_phase_context():
    manager = CouplingManager()

    def fail(context):
        raise RuntimeError("bad mapping")

    manager.register(CouplingPhase.POST_STEP, fail)
    context = CouplingContext(SolverContext(0, 0.0, 0.1), manager.fields)
    with pytest.raises(CouplingError, match="post_step callback failed"):
        manager.execute(CouplingPhase.POST_STEP, context)


def test_missing_field_fails_simulation_instead_of_silently_using_stale_data():
    fields = FieldStore()
    receiver = ReceiverReservoir(fields)
    simulation = Simulation(
        SimulationConfig(dt=0.1, end_time=0.1), [receiver],
        coupling=CouplingManager(fields),
    )
    with pytest.raises(SimulationError, match="advance/coupling failed"):
        simulation.run()
    assert simulation.state is SimulationState.FAILED
    simulation.close()
