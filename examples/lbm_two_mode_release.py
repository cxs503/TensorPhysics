"""Check D3Q7 free decay and heat-flow reversal after a two-mode slab pulse."""
import json
import math

import numpy as np
import torch
from tensorlbm import thermal_common as lbm

from conduction_provenance import provenance
import dg_reversing_wall_pulse as pulse


def measure(n, width):
    rate = math.pi**2
    start, stop = 1.2/16, (1.2+width)/16
    duration = stop-start
    # Start at release, normalizing the first Fourier coefficient to one.
    ratio = math.expm1(-4*duration)/(4*math.expm1(-duration))
    dx = 1/n
    dt = dx**2/8  # D3Q7, tau=1, unit physical diffusivity
    x = (torch.arange(n, dtype=torch.float64)+.5)*dx
    first = (torch.sin(math.pi*x)*np.sinc(dx/2))[None, None, :]
    second = (ratio*torch.sin(2*math.pi*x)*np.sinc(dx))[None, None, :]
    field = first + second
    zero = torch.zeros_like(field)
    g = lbm.thermal_equilibrium_3d(field, zero, zero, zero)
    powers = []
    max_temperature_error = max_wall_error = max_balance = 0.
    for step in range(n*n//2):
        post = lbm.thermal_collide_bgk_3d(g, field, zero, zero, zero, 1.)
        g, inward = lbm.thermal_temperature_wall_pair_3d(
            lbm.thermal_stream_3d(post), "x", (0., 0.), dt=dt, spacing=dx, volumetric_heat_capacity=1.)
        wall = -np.array([float(q.item()) for q in inward])
        powers.append(wall[1])
        updated = lbm.thermal_macroscopic_3d(g)
        t = step*dt
        a = math.exp(-rate*t)*(-math.expm1(-rate*dt))/(rate*dt)
        b = math.exp(-4*rate*t)*(-math.expm1(-4*rate*dt))/(4*rate*dt)
        exact_wall = math.pi*np.array([a+2*ratio*b, a-2*ratio*b])
        max_wall_error = max(max_wall_error, float(np.abs(wall-exact_wall).max()))
        exact_field = math.exp(-rate*(t+dt))*first + math.exp(-4*rate*(t+dt))*second
        max_temperature_error = max(max_temperature_error, float((updated-exact_field).abs().max()))
        max_balance = max(max_balance, abs(float((updated-field).mean()) + dt*wall.sum()))
        field = updated
    midpoints = (np.arange(len(powers))+.5)*dt
    j = pulse.reversal_bracket_index(np.array(powers))
    reversal = midpoints[j] - powers[j]*dt/(powers[j+1]-powers[j])
    exact = pulse.exact_reversal_time(start, stop, rate) - stop/rate
    return dict(cells=n, width_over_interval=width, max_temperature_error=max_temperature_error,
                max_wall_power_error=max_wall_error, max_step_energy_residual=float(max_balance),
                measured_time_after_release=reversal, exact_time_after_release=exact,
                reversal_time_error=abs(reversal-exact))


if __name__ == "__main__":
    rows = []
    for width in (.05, .5):
        previous = None
        for n in (16, 32, 64):
            row = measure(n, width)
            keys = ("max_temperature_error", "max_wall_power_error", "reversal_time_error")
            row["orders"] = {} if previous is None else {k: math.log2(previous[k]/row[k]) for k in keys}
            row["passed"] = bool(row["max_step_energy_residual"] < 1e-13 and
                                 (n < 64 or min(row["orders"].values()) > 1.8))
            rows.append(row)
            previous = row
    passed = all(r["passed"] for r in rows)
    print(json.dumps(dict(passed=passed, runs=rows, diffusivity=1., relaxation_time=1.,
                          initial_state="analytic cell means at pulse release, first sine coefficient normalized to one",
                          sources=[provenance(p) for p in (__file__, pulse.__file__, lbm.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_two_mode_release.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
