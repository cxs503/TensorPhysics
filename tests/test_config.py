import pytest
from tensorphysics import SimulationConfig

@pytest.mark.parametrize("dt,end_time", [(0, 1), (-0.1, 1), (0.1, -1)])
def test_invalid_time_controls_rejected(dt, end_time):
    with pytest.raises(ValueError):
        SimulationConfig(dt=dt, end_time=end_time)

def test_non_integral_end_time_uses_short_final_step():
    config = SimulationConfig(dt=0.3, end_time=1.0)
    assert config.planned_steps == 4

def test_zero_end_time_plans_no_steps():
    assert SimulationConfig(dt=0.1, end_time=0).planned_steps == 0

def test_max_steps_caps_plan():
    assert SimulationConfig(dt=0.1, end_time=10, max_steps=7).planned_steps == 7
