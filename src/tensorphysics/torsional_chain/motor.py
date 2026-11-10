"""Surface-mounted PMSM with back-EMF feedback, coupled to the torsional chain.

Electrical model (q-axis, current-control idealised as a voltage source)::

    L di_q/dt = V_q - R i_q - k_e omega_m
    T_e       = k_t i_q
    E         = k_e omega_m

Mechanical interface: the electromagnetic torque ``T_e`` is injected at the motor
node of a :class:`~tensorphysics.torsional_chain.drivetrain.DrivetrainChain`, and
the motor-node speed feeds back into the back-EMF.  That two-way path is what
makes the coupling a *multiphysics* feedback rather than a prescribed torque.

Analytic anchors (used by the acceptance test, not by the integrator):

* steady speed for a load torque ``T_L``:  ``omega_ss = (V - R T_L / k_t) / k_e``
* locked rotor electrical time constant:  ``tau_e = L / R``
* no-load mechanical time constant:       ``tau_m = J R / (k_e k_t)``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .drivetrain import DrivetrainChain

__all__ = ["PMSM", "simulate_motor", "simulate_motor_drivetrain", "motor_power_balance"]


@dataclass
class PMSM:
    """Linear PMSM model in the q-axis (no saturation, no saliency)."""

    resistance: float
    inductance: float
    torque_constant: float
    back_emf_constant: float
    supply_voltage: float

    def __post_init__(self) -> None:
        for name in ("resistance", "inductance", "torque_constant",
                     "back_emf_constant"):
            if not (getattr(self, name) > 0.0):
                raise ValueError(f"{name} must be positive")

    @property
    def electrical_time_constant(self) -> float:
        return self.inductance / self.resistance

    def mechanical_time_constant(self, inertia: float) -> float:
        return inertia * self.resistance / (self.torque_constant * self.back_emf_constant)

    def steady_state_speed(self, load_torque: float) -> float:
        """``omega_ss = (V - R T_L / k_t) / k_e`` [rad/s]."""
        return (self.supply_voltage - self.resistance * load_torque / self.torque_constant) \
            / self.back_emf_constant

    def steady_state_current(self, load_torque: float) -> float:
        return load_torque / self.torque_constant

    def back_emf(self, omega: float) -> float:
        return self.back_emf_constant * omega


def simulate_motor(motor: PMSM, inertia: float, load_torque: float,
                   t_end: float, dt: Optional[float] = None,
                   omega0: float = 0.0, current0: float = 0.0) -> Dict[str, np.ndarray]:
    """Stand-alone motor + inertia, RK4; returns speed, current, torque, back-EMF."""
    if dt is None:
        dt = 0.02 * motor.electrical_time_constant
    n = int(round(t_end / dt))
    t = np.arange(n + 1) * dt
    i = np.zeros(n + 1)
    w = np.zeros(n + 1)
    i[0], w[0] = current0, omega0

    def f(current: float, speed: float) -> tuple[float, float]:
        di = (motor.supply_voltage - motor.resistance * current
              - motor.back_emf_constant * speed) / motor.inductance
        dw = (motor.torque_constant * current - load_torque) / inertia
        return di, dw

    for k in range(n):
        di1, dw1 = f(i[k], w[k])
        di2, dw2 = f(i[k] + 0.5 * dt * di1, w[k] + 0.5 * dt * dw1)
        di3, dw3 = f(i[k] + 0.5 * dt * di2, w[k] + 0.5 * dt * dw2)
        di4, dw4 = f(i[k] + dt * di3, w[k] + dt * dw3)
        i[k + 1] = i[k] + dt / 6.0 * (di1 + 2 * di2 + 2 * di3 + di4)
        w[k + 1] = w[k] + dt / 6.0 * (dw1 + 2 * dw2 + 2 * dw3 + dw4)
    return {"t": t, "current": i, "omega": w,
            "torque": motor.torque_constant * i,
            "back_emf": motor.back_emf_constant * w}


def simulate_motor_drivetrain(chain: DrivetrainChain, motor: PMSM, motor_node: int,
                              t_end: float, dt: float,
                              external_torque, load_torque: float = 0.0,
                              torque0: Optional[Sequence[float]] = None,
                              omega0: Optional[Sequence[float]] = None,
                              current0: float = 0.0) -> Dict[str, np.ndarray]:
    """Coupled chain + PMSM with back-EMF feedback.

    ``external_torque(t, theta, omega) -> array`` is the ice/propeller excitation
    on the chain nodes; the motor injects ``k_t i_q`` at ``motor_node`` and its
    speed is the node speed, closing the back-EMF loop.
    """
    f = chain.free_dofs
    nf = f.size
    if motor_node not in set(int(v) for v in f):
        raise ValueError("motor_node must be a free node of the chain")
    motor_free = int(np.where(f == motor_node)[0][0])
    n_steps = int(round(t_end / dt))
    y = np.zeros(2 * nf + 1)
    if torque0 is not None:
        y[:nf] = np.asarray(torque0, dtype=float)[f]
    if omega0 is not None:
        y[nf:2 * nf] = np.asarray(omega0, dtype=float)[f]
    y[2 * nf] = current0

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        theta = np.zeros(chain.n_nodes)
        omega = np.zeros(chain.n_nodes)
        theta[f] = state[:nf]
        omega[f] = state[nf:2 * nf]
        current = state[2 * nf]
        q = np.asarray(external_torque(t, theta, omega), dtype=float)
        q = q + chain.internal_torque(theta, omega)
        q[motor_node] += motor.torque_constant * current
        alpha = np.zeros(chain.n_nodes)
        alpha[f] = q[f] / chain.inertias[f]
        di = (motor.supply_voltage - motor.resistance * current
              - motor.back_emf_constant * omega[motor_node]) / motor.inductance
        return np.concatenate([omega[f], alpha[f], [di]])

    t = np.arange(n_steps + 1) * dt
    theta_hist = np.zeros((n_steps + 1, chain.n_nodes))
    omega_hist = np.zeros((n_steps + 1, chain.n_nodes))
    current_hist = np.zeros(n_steps + 1)
    theta_hist[0, f] = y[:nf]
    omega_hist[0, f] = y[nf:2 * nf]
    current_hist[0] = y[2 * nf]
    for step in range(n_steps):
        tk = t[step]
        k1 = rhs(tk, y)
        k2 = rhs(tk + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(tk + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(tk + dt, y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        theta_hist[step + 1, f] = y[:nf]
        omega_hist[step + 1, f] = y[nf:2 * nf]
        current_hist[step + 1] = y[2 * nf]
    return {"t": t, "theta": theta_hist, "omega": omega_hist,
            "current": current_hist, "motor_torque": motor.torque_constant * current_hist,
            "back_emf": motor.back_emf_constant * omega_hist[:, motor_node]}


def motor_power_balance(motor: PMSM, current: float, omega: float) -> Dict[str, float]:
    """Instantaneous electrical/mechanical power bookkeeping [W].

    ``P_in = V i``, ``P_copper = R i**2``, ``P_emf = k_e omega i`` = shaft power.
    The residual ``P_in - P_copper - P_emf`` measures how well the stored magnetic
    energy term is resolved at the sampling instant; it is zero in steady state.
    """
    p_in = motor.supply_voltage * current
    p_copper = motor.resistance * current ** 2
    p_shaft = motor.back_emf_constant * omega * current
    return {"p_in": p_in, "p_copper": p_copper, "p_shaft": p_shaft,
            "residual": p_in - p_copper - p_shaft}