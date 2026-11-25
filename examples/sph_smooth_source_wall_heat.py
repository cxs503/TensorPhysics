"""Check time convergence for signed, smooth heating between isothermal SPH walls."""
import json
import math

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(columns, steps, method):
    # On the unit slab with unit material properties the manufactured excess
    # temperature is sin(2*pi*x)*sin(omega*t). Both wall temperatures stay zero.
    wave = 2*math.pi
    omega = 8*wave**2
    x = (torch.arange(columns, dtype=torch.float64) + .5)/columns
    shape = torch.sin(wave*x)[None, :]
    time = torch.linspace(0., 2*math.pi/omega, steps+1, dtype=x.dtype)

    def source(t):
        return shape*(omega*math.cos(omega*t) + wave**2*math.sin(omega*t))

    field, walls, averages = heat.solve_transient_slab_2d(
        torch.zeros_like(shape), time.tolist(), 1/columns, (0., 0.), method=method,
        heat_source=source, return_wall_heat_by_row=True, return_interval_temperatures=True)
    integral = -torch.cos(omega*time).diff()/omega
    exact_mean = (integral/time.diff())[:, None, None]*shape
    # Inward heat is negative at the west wall and positive at the east wall.
    exact_heat = wave*integral[:, None]*x.new_tensor([-1., 1.])
    cumulative_error = (walls[:, :, 0] - exact_heat).cumsum(dim=0).abs()
    return dict(columns=columns, steps=steps, method=method,
                field_error_K=float((field - torch.sin(omega*time)[:, None, None]*shape).abs().max()),
                interval_mean_error_K=float((averages-exact_mean).abs().max()),
                cumulative_wall_error_over_peak=float(cumulative_error.max()/(2*wave/omega)))


if __name__ == "__main__":
    runs = []
    for method in ("theta", "sdirk2"):
        previous = None
        for steps in (8, 16, 32):
            row = measure(256, steps, method)
            row["passed"] = True
            for metric in ("field_error_K", "interval_mean_error_K", "cumulative_wall_error_over_peak"):
                if previous:
                    order = math.log2(previous[metric]/row[metric])
                    row[metric + "_order"] = order
                    row["passed"] &= order > 1.8
                if steps == 32:
                    row["passed"] &= row[metric] < .01
            runs.append(row)
            previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_smooth_source_wall_heat.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
