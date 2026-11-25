"""Heat absorbed after a fixed-temperature step on one face of a finite slab."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def measure(n):
    # Unit thickness, conductivity and volumetric heat capacity. The initially
    # cold slab has unit west temperature from t=0+ and zero east temperature.
    times = np.r_[0., .0004 * 2.**np.arange(9)]
    dx = 1. / n
    temperature, heat = sph.solve_transient_layered_slab(
        np.zeros(n), times, dx, (1., 0.), h_factor=.9, conductivity=np.ones(n),
        volumetric_heat_capacity=np.ones(n), method="exponential", return_x_heat=True,
    )
    temperature = temperature.numpy()
    heat = heat.numpy()
    # For positive times the transient part of the sine series is exponentially
    # convergent. Integrate the steady part and each transient analytically.
    m = np.arange(1, 193)
    k = np.pi * m
    decay = np.exp(-times[1:, None] * k**2)
    west = times[1:] + 1./3 - 2 * (decay / k**2).sum(axis=1)
    east = times[1:] - 1./6 - 2 * (decay * (-1.)**m / k**2).sum(axis=1)
    exact_heat = np.diff(np.vstack(([0., 0.], np.column_stack((west, east)))), axis=0)
    interval_error = np.abs(heat[:, [0, -1]] - exact_heat)
    edges = np.linspace(0., 1., n+1)
    cell_cos = np.diff(np.cos(k[:, None] * edges), axis=1)
    exact_mean = 1. - .5*(edges[:-1]+edges[1:]) + 2*(decay / k**2) @ cell_cos / dx
    balance = dx*np.diff(temperature.sum(axis=1)) - heat[:, 0] + heat[:, -1]
    return dict(cells=n, first_interval_s=float(times[1]),
                first_interval_west_heat_J_m2=float(heat[0, 0]),
                exact_first_interval_west_heat_J_m2=float(west[0]),
                west_max_interval_error_J_m2=float(interval_error[:, 0].max()),
                east_max_interval_error_J_m2=float(interval_error[:, 1].max()),
                max_temperature_vs_cell_mean_K=float(abs(temperature[1:] - exact_mean).max()),
                max_energy_residual_J_m2=float(abs(balance).max()))


if __name__ == "__main__":
    rows = []
    previous = None
    for n in (32, 64, 128, 256):
        row = measure(n)
        errors = np.array([row[key] for key in ("west_max_interval_error_J_m2",
                          "east_max_interval_error_J_m2", "max_temperature_vs_cell_mean_K")])
        orders = [] if previous is None else np.log2(previous / errors).tolist()
        row.update(west_east_temperature_orders=orders,
                   passed=bool(row["max_energy_residual_J_m2"] < 1e-11
                               and (not orders or min(orders) > 0.)
                               and (n != 256 or (min(orders) > 1.85 and
                                    np.all(errors < [6e-5, 2.5e-7, .002])))))
        previous = errors
        rows.append(row)
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(reference="Finite Dirichlet slab sine series integrated over time and cell width",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_wall_step_heat.py",
        passed=passed, runs=rows, sources=[provenance(__file__), provenance(sph.__file__)]),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
