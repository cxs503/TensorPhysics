"""Run a tiny solver to demonstrate the TensorPhysics lifecycle."""
from dataclasses import dataclass
from tensorphysics import Simulation, SimulationConfig, SolverState

@dataclass
class ToySolver:
    name: str = "toy"
    value: float = 0.0
    def initialize(self, context):
        self.value = 0.0
    def advance(self, context):
        self.value += context.dt
    def finalize(self, context):
        pass
    def snapshot(self, context):
        return SolverState(self.name, context.step, context.time, {"accumulated_dt": self.value})

if __name__ == "__main__":
    solver = ToySolver()
    simulation = Simulation(SimulationConfig(dt=0.1, end_time=1.0), [solver])
    simulation.run()
    print(f"state={simulation.state.value}, steps={simulation.step}, time={simulation.time:.6f}")
    print(simulation.snapshots())
    simulation.close()
