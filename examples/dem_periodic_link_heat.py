"""Verify signed interval heat in a periodically forced finite thermal contact chain."""
import json
import math

import numpy as np
from tensordem.thermal import contact_chain

from conduction_provenance import provenance


def measure(steps, mode, theta):
    links, resistance, capacity = 12, .7, 1.2
    time = np.arange(steps+1, dtype=float)/steps
    time[1::2] -= .5/steps
    dt = np.diff(time)
    omega = 2*math.pi
    shape = np.sin(mode*math.pi*np.arange(links+1)/links)
    shape[[0,-1]] = 0.
    # The discrete Dirichlet chain has eigenvalue 4*sin²(m*pi/2N)/R.
    # Target nodal excess temperature shape*sin(omega*t), amplitude 1 K.
    eigenvalue = 4*math.sin(mode*math.pi/(2*links))**2/resistance
    wave = np.sin(omega*time)
    source = (capacity*omega*np.cos(omega*time) + eigenvalue*wave)[:, None]*shape
    field, heat = contact_chain.transient_contact_temperatures(
        np.full(links, resistance), capacity, time, 0., t_left=0., t_right=0.,
        nodal_power=source, theta=theta, return_link_heat=True)
    exact_cumulative = (np.cos(omega*time)-1)[:, None]*np.diff(shape)/(resistance*omega)
    applied = dt[:, None]*((1-theta)*source[:-1]+theta*source[1:])
    residual = capacity*np.diff(field[:, 1:-1], axis=0) - applied[:, 1:-1] - heat[:, :-1] + heat[:, 1:]
    return dict(steps=steps, mode=mode, theta=theta, max_temperature_error_K=float(abs(field-wave[:, None]*shape).max()),
                cumulative_link_heat_error_J=float(abs(heat.cumsum(0)-exact_cumulative[1:]).max()),
                interval_mean_power_error_W=float((abs(heat-np.diff(exact_cumulative, axis=0))/dt[:, None]).max()),
                max_particle_balance_residual_J=float(abs(residual).max()))


if __name__ == "__main__":
    runs = []
    for theta in (.5, 1.):
        for mode in (1, 3):
            previous = None
            for steps in (32, 64, 128):
                row = measure(steps, mode, theta)
                row["passed"] = row["max_particle_balance_residual_J"] < 1e-12
                for key in ("max_temperature_error_K", "cumulative_link_heat_error_J", "interval_mean_power_error_W"):
                    if previous:
                        order = math.log2(previous[key]/row[key])
                        row[key+"_order"] = order
                        row["passed"] &= order > (1.8 if theta == .5 else .85)
                runs.append(row)
                previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(contact_chain.__file__)],
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDEM/src python examples/dem_periodic_link_heat.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
