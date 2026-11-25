"""A short internal heating pulse excites one decaying mode between fixed-temperature walls."""
import argparse
import json
import math

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(n, base, amplitude, method, transverse_mode=1, off_grid=False):
    x = (torch.arange(n, dtype=torch.float64)+0.5)/n
    lateral = torch.cos(2*torch.pi*transverse_mode*x)
    mode = lateral[:, None]*torch.sin(torch.pi*x)
    rate = (1 + 4*transverse_mode**2)*math.pi**2
    switch = 1/rate
    times = [i*switch/16 for i in range(33)]
    if off_grid:
        switch += 0.37*(times[1]-times[0])
    initial = torch.full_like(mode, base)
    options = dict(wall_jump_times=[switch], method=method,
        heat_source=lambda t: amplitude*rate*mode if t < switch else 0.,
        temperature_reference=base, return_x_heat=True, return_wall_heat_by_row=True,
    )
    def solve(grid):
        return heat.solve_transient_slab_2d(initial, grid, 1/n, (base, base), **options)
    fields, sections, walls = solve(times)
    factors = torch.tensor([(-math.expm1(-rate*min(t, switch)))*math.exp(-rate*max(0., t-switch))
                            for t in times], dtype=mode.dtype)
    error = abs(fields/amplitude - factors[:, None, None]*mode)
    # The modal amplitude satisfies f' + rate*f = rate while heating,
    # then f' + rate*f = 0. Its integral is the heating duration minus
    # the amplitude change divided by rate, including the switch interval.
    t = torch.tensor(times, dtype=mode.dtype)
    heated_time = torch.minimum(t, t.new_tensor(switch))
    integral = heated_time.diff() - factors.diff()/rate
    # Both inward wall fluxes are -pi*A*f*cos(2*pi*y) at unit conductivity.
    exact_heat = -torch.pi*amplitude*integral[:, None, None]*lateral
    interval_scale = torch.pi*amplitude*t.diff()[:, None, None]
    interval_error = abs(walls-exact_heat)/interval_scale
    heating = t[1:] <= switch
    cumulative_error = abs((walls-exact_heat).cumsum(dim=0)).max()/(torch.pi*amplitude*integral.sum())
    section_inward = torch.stack((sections[:, 0], -sections[:, -1]), dim=1)
    mean_error = abs(walls.mean(dim=2)-section_inward)/interval_scale[:, :, 0]
    generated = amplitude*rate*mode.mean()*heated_time.diff()
    balance = fields.mean(dim=(1, 2)).diff() - generated - section_inward.sum(dim=1)
    event = {}
    if off_grid:
        # Also request the event as an output. Both half-interval heat
        # tallies must add up to the original whole-interval tally.
        i = next(i for i in range(len(times)-1) if times[i] < switch < times[i+1])
        detailed, cuts, rows = solve(times[:i+1] + [switch] + times[i+1:])
        heat_scale = float(interval_scale[i, 0, 0])
        event = dict(switch_time_s=switch, crossing_interval_s=times[i:i+2],
                     crossing_wall_error_over_scale=float(interval_error[i].max()),
                     crossing_energy_imbalance_over_amplitude=float(abs(balance[i])/amplitude),
                     split_section_difference_over_scale=float(abs(sections[i]-cuts[i:i+2].sum(0)).max()/heat_scale),
                     split_wall_difference_over_scale=float(abs(walls[i]-rows[i:i+2].sum(0)).max()/heat_scale),
                     split_field_difference_over_amplitude=float(abs(fields[i+1]-detailed[i+2]).max()/amplitude))
    return dict(**event, transverse_mode=transverse_mode, max_energy_imbalance_over_amplitude=float(abs(balance).max()/amplitude),
                heating_interval_wall_error_over_scale=float(interval_error[heating].max()),
                decay_interval_wall_error_over_scale=float(interval_error[t[:-1] >= switch].max()),
                cumulative_wall_error_over_scale=float(cumulative_error),
                wall_mean_section_mismatch_over_scale=float(mean_error.max()), columns=n, wall_K=base, amplitude_K=amplitude, method=method,
                max_history_error_over_amplitude=float(error.max()),
                final_error_over_amplitude=float(error[-1].max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-grid", action="store_true", help="switch the source 37 percent into an output interval")
    args = parser.parse_args()
    runs = []
    for method in ("theta", "sdirk2"):
        for base in (0., 305.):
            for n in (16, 32, 64):
                for transverse_mode in (0, 1):
                    row = measure(n, base, 2**-44, method, transverse_mode, args.off_grid)
                    wall_error = max(row["heating_interval_wall_error_over_scale"],
                                     row["decay_interval_wall_error_over_scale"],
                                     row["cumulative_wall_error_over_scale"])
                    row["passed"] = (row["max_history_error_over_amplitude"] < 0.01
                                     and row["max_energy_imbalance_over_amplitude"] < 1e-11
                                     and row["wall_mean_section_mismatch_over_scale"] < 1e-12
                                     and (n < 64 or (row["max_history_error_over_amplitude"] < 0.001
                                                     and wall_error < 0.002)))
                    if args.off_grid:
                        row["passed"] &= (row["split_section_difference_over_scale"] < 1e-12
                                          and row["split_wall_difference_over_scale"] < 1e-12
                                          and row["split_field_difference_over_amplitude"] < 1e-12
                                          and (n < 64 or row["crossing_wall_error_over_scale"] < 0.002))
                    runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_weak_transient_mode.py"
                          + (" --off-grid" if args.off_grid else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
