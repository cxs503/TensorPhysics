"""Validated, dependency-free simulation configuration."""
from dataclasses import dataclass
from math import ceil

@dataclass(frozen=True)
class SimulationConfig:
    dt: float
    end_time: float
    max_steps: int = 1_000_000
    time_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.end_time < 0:
            raise ValueError("end_time must be non-negative")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.time_tolerance < 0:
            raise ValueError("time_tolerance must be non-negative")

    @property
    def planned_steps(self) -> int:
        """Number of global increments planned to reach end_time."""
        if self.end_time == 0:
            return 0
        ratio = self.end_time / self.dt
        nearest = round(ratio)
        count = nearest if abs(ratio - nearest) <= self.time_tolerance else ceil(ratio)
        return min(count, self.max_steps)
