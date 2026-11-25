"""Verify local x heat flux for a weak sinusoidal wall temperature on a periodic square."""
import json

import numpy as np
import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(n, base, backend):
    # The continuum solution above the uniform offset is proportional to
    # cos(2*pi*y+phase)*sinh(2*pi*(1-x)), with zero excess at the east wall.
    amplitude, k, wave = 2**-22, 1.7, 2*np.pi
    y = (np.arange(n)+0.5)/n
    lateral = np.cos(wave*y + 0.37)[:, None]
    west = torch.from_numpy(base + amplitude*lateral[:, 0])
    _, cuts = heat.solve_harmonic_layered_slab(
        torch.zeros((n, n), dtype=torch.float64), 1/n, (west, base), 0.,
        conductivity=torch.full((n,), k, dtype=torch.float64), volumetric_heat_capacity=1., backend=backend, return_x_flux="by_row",
    )
    x = np.arange(n+1)/n
    reference = k*amplitude*wave*lateral*np.cosh(wave*(1-x))/np.sinh(wave)
    scale = k*amplitude*wave
    error = abs(cuts.numpy()-reference)/scale
    return dict(columns=n, rows=n, east_K=base, amplitude_K=amplitude, backend=backend,
                max_interior_flux_error_over_scale=float(error[:, 1:-1].max()),
                max_wall_flux_error_over_scale=float(error[:, [0,-1]].max()),
                max_mean_flux_over_scale=float(abs(cuts.mean(0)).max()/scale))


if __name__ == "__main__":
    runs = []
    for backend in ("torch", "banded"):
        for base in (0., 305.):
            previous = None
            for n in (8, 16, 32, 64, 128):
                row = measure(n, base, backend)
                metrics = ("max_interior_flux_error_over_scale", "max_wall_flux_error_over_scale")
                orders = {} if previous is None else {
                    key: float(np.log2(previous[key]/row[key])) for key in metrics
                }
                row.update(orders=orders, passed=row["max_mean_flux_over_scale"] < 1e-7
                           and (n < 128 or (row[metrics[0]] < 0.001 and row[metrics[1]] < 0.015
                                           and orders[metrics[0]] > 1.8 and orders[metrics[1]] > 0.85)))
                previous = row
                runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_periodic_wall_flux_mode.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
