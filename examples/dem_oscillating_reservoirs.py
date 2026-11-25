"""Verify heat exchanged with two oscillating reservoirs across a massless contact."""
import json
import math

import numpy as np
from tensordem.thermal import contact_chain

from conduction_provenance import provenance


def measure(steps, theta):
    # Left wall -- junction (zero capacity) -- grain -- right wall.
    resistance = np.array([0.3, 0.8, 0.45])
    capacity = 1.7
    omega = 2 * np.pi
    time = np.linspace(0, 1, steps + 1)
    oscillation = np.exp(1j * omega * time)
    left, right = 2.0 + 0.3j, -0.7 + 1.1j
    left_conductance = 1 / sum(resistance[:2])
    right_conductance = 1 / resistance[2]
    # Scalar energy balance for the grain after eliminating series resistances.
    grain = (left_conductance * left + right_conductance * right) / (
        left_conductance + right_conductance + 1j * omega * capacity
    )
    inflow = left_conductance * (left - grain)
    outflow = right_conductance * (grain - right)
    junction = left - resistance[0] * inflow
    target = 273.15 + np.real(oscillation[:, None] * [left, junction, grain, right])
    primitive = np.expm1(1j * omega * time) / (1j * omega)
    exact_heat = np.real(primitive[:, None] * [inflow, inflow, outflow])
    initial = target[0].copy()
    initial[1] = 0.0  # The solver must reconstruct the algebraic contact temperature.
    field, heat = contact_chain.transient_contact_temperatures(
        resistance, [0, 0, capacity, 0], time, initial,
        t_left=target[:, 0], t_right=target[:, -1], theta=theta,
        allow_massless=True, return_link_heat=True,
    )
    balance = capacity * np.diff(field[:, 2]) - heat[:, 1] + heat[:, 2]
    return {
        "steps": steps, "theta": theta,
        "temperature_error_K": float(np.max(abs(field - target))),
        "cumulative_heat_error_J": float(np.max(abs(heat.cumsum(axis=0) - exact_heat[1:]))),
        "junction_residual_J": float(np.max(abs(heat[:, 0] - heat[:, 1]))),
        "grain_residual_J": float(np.max(abs(balance))),
    }


def main():
    runs = []
    for theta in (0.5, 1.0):
        previous = None
        for steps in (32, 64, 128):
            row = measure(steps, theta)
            passed = max(row["junction_residual_J"], row["grain_residual_J"]) < 1e-11
            for metric in ("temperature_error_K", "cumulative_heat_error_J"):
                if previous:
                    order = math.log2(previous[metric] / row[metric])
                    row[metric + "_order"] = order
                    passed &= order > (1.85 if theta == 0.5 else 0.9)
            row["passed"] = passed
            runs.append(row)
            previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps({
        "passed": passed, "runs": runs,
        "sources": [provenance(__file__), provenance(contact_chain.__file__)],
        "reproduction": "PYTHONPATH=../TensorDEM/src python examples/dem_oscillating_reservoirs.py",
    }, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
