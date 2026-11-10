"""C3 - PMSM steady state, back-EMF feedback and coupled power balance."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.torsional_chain.acceptance import c3_motor_steady_state
from tensorphysics.torsional_chain.drivetrain import DrivetrainChain, ShaftSegment, uniform_segment_stiffness
from tensorphysics.torsional_chain.motor import (PMSM, motor_power_balance, simulate_motor,
                                                 simulate_motor_drivetrain)


@pytest.fixture(scope="module")
def c3():
    return c3_motor_steady_state()


def test_steady_state_speed_formula():
    motor = PMSM(0.05, 5.0e-3, 1.2, 1.15, 400.0)
    assert motor.steady_state_speed(0.0) == pytest.approx(400.0 / 1.15)
    assert motor.steady_state_speed(30.0) == pytest.approx(
        (400.0 - 0.05 * 30.0 / 1.2) / 1.15)


def test_three_load_cases_reach_analytic_speed():
    motor = PMSM(0.05, 5.0e-3, 1.2, 1.15, 400.0)
    for load in (0.0, 20.0, 40.0):
        result = simulate_motor(motor, 0.5, load, t_end=2.0, dt=1.0e-4)
        reference = motor.steady_state_speed(load)
        assert abs(result["omega"][-1] - reference) / reference < 0.02


def test_electrical_time_constant_locked_rotor():
    motor = PMSM(0.05, 5.0e-3, 1.2, 1.15, 400.0)
    tau_e = motor.electrical_time_constant
    assert tau_e == pytest.approx(0.1)
    # a very large inertia holds the rotor effectively locked for 0.6 s
    locked = simulate_motor(motor, 1.0e9, 1.0e9, t_end=0.6, dt=1.0e-4)
    idx = int(np.argmin(np.abs(locked["t"] - tau_e)))
    fraction = locked["current"][idx] * motor.resistance / motor.supply_voltage
    assert fraction == pytest.approx(1.0 - math.exp(-1.0), abs=0.01)


def test_back_emf_is_linear_in_speed():
    motor = PMSM(0.05, 5.0e-3, 1.2, 1.15, 400.0)
    result = simulate_motor(motor, 0.5, 0.0, t_end=0.3, dt=1.0e-4)
    assert np.allclose(result["back_emf"], 1.15 * result["omega"], rtol=1e-12)


def test_power_balance_in_steady_state():
    # SI-consistent motor (k_e = k_t) so shaft power equals k_t i omega exactly
    motor = PMSM(0.05, 5.0e-3, 1.2, 1.2, 400.0)
    result = simulate_motor(motor, 0.5, 25.0, t_end=2.0, dt=1.0e-4)
    current = float(result["current"][-1])
    omega = float(result["omega"][-1])
    balance = motor_power_balance(motor, current, omega)
    assert abs(balance["residual"]) / balance["p_in"] < 1.0e-3
    # mechanical power leaving the shaft equals the load power in steady state
    assert balance["p_shaft"] == pytest.approx(25.0 * omega, rel=2e-3)


def test_motor_rejects_nonpositive_parameters():
    with pytest.raises(ValueError):
        PMSM(0.0, 5.0e-3, 1.2, 1.15, 400.0)
    with pytest.raises(ValueError):
        PMSM(0.05, -1.0, 1.2, 1.15, 400.0)


def test_coupled_motor_drivetrain_closes_the_back_emf_loop():
    """Motor torque, load damping and back-EMF settle on mutually consistent values."""
    motor = PMSM(0.05, 5.0e-3, 1.2, 1.15, 400.0)
    k = uniform_segment_stiffness(6.0, 0.42, 0.18, 79.3e9)
    chain = DrivetrainChain()
    prop = chain.add_node("propeller", 20.0)
    drive = chain.add_node("motor", 0.5)
    chain.add_shaft(ShaftSegment("shaft1", k, 0.0, prop, drive, 0.42, 0.18))

    c_load = 200.0  # propeller load damping [N*m*s/rad]

    def external(t, theta, omega):
        return np.array([-c_load * omega[0], 0.0])

    result = simulate_motor_drivetrain(chain, motor, motor_node=drive, t_end=0.6,
                                       dt=5.0e-5, external_torque=external,
                                       omega0=[0.0, 0.0])
    omega_end = float(result["omega"][-1, prop])
    expected = motor.supply_voltage / (motor.back_emf_constant
                                       + motor.resistance * c_load / motor.torque_constant)
    assert abs(omega_end - expected) / expected < 0.02
    current_end = float(result["current"][-1])
    assert current_end == pytest.approx(c_load * omega_end / motor.torque_constant, rel=0.02)
    # the two nodes rotate together in steady state (rigid-body motion)
    assert abs(result["omega"][-1, prop] - result["omega"][-1, drive]) / omega_end < 0.02


def test_coupled_motor_drivetrain_rejects_grounded_motor_node():
    motor = PMSM(0.05, 5.0e-3, 1.2, 1.15, 400.0)
    chain = DrivetrainChain()
    a = chain.add_node("propeller", 20.0)
    b = chain.add_node("motor", 0.5, fixed=True)
    chain.add_shaft(ShaftSegment("s", 1.0e7, 0.0, a, b, 0.42, 0.18))
    with pytest.raises(ValueError):
        simulate_motor_drivetrain(chain, motor, motor_node=b, t_end=1e-3, dt=1e-5,
                                  external_torque=lambda t, th, om: np.zeros(2))


def test_c3_acceptance_block(c3):
    assert c3["criterion"] == "C3"
    assert c3["pass"] is True
    assert c3["worst_relative_error"] <= 0.02
    assert len(c3["cases"]) == 3