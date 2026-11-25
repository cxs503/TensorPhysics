"""Verify a weak heat pulse whose two events occur between consecutive requested outputs."""
import json
import math

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(n, method, width_fraction, mode_y):
    base, amplitude = 305., 2**-44
    x = (torch.arange(n, dtype=torch.float64)+0.5)/n
    lateral = torch.cos(2*torch.pi*mode_y*x)
    mode = lateral[:, None]*torch.sin(torch.pi*x)
    rate = math.pi**2*(1+4*mode_y**2)
    dt = 1/(16*rate)
    times = [i*dt for i in range(19)]
    start = 1.2*dt
    stop = start + width_fraction*dt
    initial = torch.full_like(mode, base)

    def solve(grid):
        return heat.solve_transient_slab_2d(
            initial, grid, 1/n, (base, base), temperature_reference=base,
            wall_jump_times=[start, stop], method=method, return_x_heat=True, return_wall_heat_by_row=True,
            heat_source=lambda t: amplitude*rate*mode if start <= t < stop else 0.,
        )

    field, cuts, walls = solve(times)
    detail, subcuts, subwalls = solve(times[:2] + [start, stop] + times[2:])
    t = torch.tensor(times, dtype=x.dtype)
    on_duration = (t-start).clamp(0, stop-start)
    # Exact modal response to a rectangular pulse; its integral follows
    # directly from f' + rate*f = rate while the source is active.
    factor = -torch.expm1(-rate*on_duration)*torch.exp(-rate*(t-stop).clamp_min(0))
    peak = -math.expm1(-rate*(stop-start))
    field_error = abs(field/amplitude - factor[:, None, None]*mode)
    integral = on_duration.diff() - factor.diff()/rate
    expected_wall = -torch.pi*amplitude*integral[:, None, None]*lateral
    heat_scale = torch.pi*amplitude*dt*peak
    heat_error = abs(walls-expected_wall)/heat_scale
    energy_residual = (field.mean((1, 2)).diff() - amplitude*rate*mode.mean()*on_duration.diff()
                       - cuts[:, 0] + cuts[:, -1])
    return dict(columns=n, method=method, transverse_mode=mode_y, width_over_interval=width_fraction,
                start_s=start, stop_s=stop, crossing_interval_s=times[1:3],
                crossing_field_error_over_peak=float(field_error[2].max()/peak),
                decay_field_error_over_peak=float(field_error[3:].max()/peak),
                crossing_wall_heat_error_over_scale=float(heat_error[1].max()),
                decay_wall_heat_error_over_scale=float(heat_error[2:].max()),
                energy_residual_over_peak=float(abs(energy_residual).max()/(amplitude*peak)),
                split_field_difference_over_peak=float(abs(field[2]-detail[4]).max()/(amplitude*peak)),
                split_section_heat_difference_over_scale=float(abs(cuts[1]-subcuts[1:4].sum(0)).max()/heat_scale),
                split_wall_heat_difference_over_scale=float(abs(walls[1]-subwalls[1:4].sum(0)).max()/heat_scale))


if __name__ == "__main__":
    runs = []
    for n in (16, 32, 64):
        for method in ("theta", "sdirk2"):
            for width in (0.05, 0.5):
                for mode in (0, 1):
                    row = measure(n, method, width, mode)
                    errors = [value for key, value in row.items() if "_error_over_" in key]
                    differences = [value for key, value in row.items() if key.startswith("split_")]
                    row["passed"] = (max(differences) < 1e-12 and row["energy_residual_over_peak"] < 1e-11
                                     and max(errors) < (0.004 if n == 64 else 0.05))
                    runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_subinterval_heat_pulse.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
