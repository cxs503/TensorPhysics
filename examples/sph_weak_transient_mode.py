"""A short internal heating pulse excites one decaying mode between fixed-temperature walls."""
import json
import math

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(n, base, amplitude, method, transverse_mode=1):
    x = (torch.arange(n, dtype=torch.float64)+0.5)/n
    lateral = torch.cos(2*torch.pi*transverse_mode*x)
    mode = lateral[:, None]*torch.sin(torch.pi*x)
    rate = (1 + 4*transverse_mode**2)*math.pi**2
    switch = 1/rate
    times = [i*switch/16 for i in range(33)]
    initial = torch.full_like(mode, base)
    fields, sections, walls = heat.solve_transient_slab_2d(
        initial, times, 1/n, (base, base), wall_jump_times=[switch], method=method,
        heat_source=lambda t: amplitude*rate*mode if t < switch else 0.,
        temperature_reference=base, return_x_heat=True, return_wall_heat_by_row=True,
    )
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
    return dict(transverse_mode=transverse_mode, max_energy_imbalance_over_amplitude=float(abs(balance).max()/amplitude),
                heating_interval_wall_error_over_scale=float(interval_error[heating].max()),
                decay_interval_wall_error_over_scale=float(interval_error[~heating].max()),
                cumulative_wall_error_over_scale=float(cumulative_error),
                wall_mean_section_mismatch_over_scale=float(mean_error.max()), columns=n, wall_K=base, amplitude_K=amplitude, method=method,
                max_history_error_over_amplitude=float(error.max()),
                final_error_over_amplitude=float(error[-1].max()))


if __name__ == "__main__":
    runs = []
    for method in ("theta", "sdirk2"):
        for base in (0., 305.):
            for n in (16, 32, 64):
                for transverse_mode in (0, 1):
                    row = measure(n, base, 2**-44, method, transverse_mode)
                    wall_error = max(row["heating_interval_wall_error_over_scale"],
                                     row["decay_interval_wall_error_over_scale"],
                                     row["cumulative_wall_error_over_scale"])
                    row["passed"] = (row["max_history_error_over_amplitude"] < 0.01
                                     and row["max_energy_imbalance_over_amplitude"] < 1e-11
                                     and row["wall_mean_section_mismatch_over_scale"] < 1e-12
                                     and (n < 64 or (row["max_history_error_over_amplitude"] < 0.001
                                                     and wall_error < 0.002)))
                    runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_weak_transient_mode.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
