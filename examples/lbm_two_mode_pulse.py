"""Verify full D3Q7 slab heating when a rectangular two-mode pulse cuts lattice steps."""
import json
import math

import numpy as np
import torch

import dg_reversing_wall_pulse as pulse
from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, width):
    rate = math.pi**2
    start, stop = 1.2/(16*rate), (1.2+width)/(16*rate)
    dx = 1/n
    dt = dx**2/8
    x = (torch.arange(n, dtype=torch.float64)+.5)/n
    waves = [torch.sin(k*math.pi*x).view(1, 1, n)*np.sinc(.5*k/n) for k in (1, 2)]
    source = rate*(waves[0]+waves[1])
    zero = torch.zeros_like(source)
    field = zero
    g = lbm.thermal_equilibrium_3d(zero, zero, zero, zero)
    wall_history = []
    deposited = max_balance = max_temp = 0.
    peak = -math.expm1(-rate*(stop-start))
    for step in range(n*n//2):
        t = step*dt
        early, late = lbm.thermal_pulse_increments_3d(source, time=t, dt=dt, start=start, stop=stop)
        post = lbm.thermal_collide_bgk_3d(g, field, zero, zero, zero, 1.) + early
        streamed, inward = lbm.thermal_temperature_wall_pair_3d(
            lbm.thermal_stream_3d(post), "x", (0., 0.), dt=dt, spacing=dx, volumetric_heat_capacity=1.)
        g = streamed + late
        updated = lbm.thermal_macroscopic_3d(g)
        wall = -np.array([q.item() for q in inward])
        wall_history.append(wall[1])
        injection = float((early+late).sum(0).mean())
        deposited += injection
        max_balance = max(max_balance, abs(float((updated-field).mean()) + dt*wall.sum() - injection)/peak)
        on = max(0., min(t+dt, stop)-start)
        off = max(0., t+dt-stop)
        exact = sum(-math.expm1(-k*k*rate*on)*math.exp(-k*k*rate*off)*wave/(k*k)
                    for k, wave in enumerate(waves, start=1))
        max_temp = max(max_temp, float((updated-exact).abs().max())/peak)
        field = updated
    power = np.array(wall_history)
    first = math.ceil(stop/dt)
    j = first + pulse.reversal_bracket_index(power[first:])
    crossing = (j+.5-power[j]/(power[j+1]-power[j]))*dt
    exact_crossing = pulse.exact_reversal_time(start*rate, stop*rate, rate)
    exact_input = (stop-start)*rate*2/math.pi
    return dict(cells=n, width_over_interval=width, pulse_duration_in_steps=(stop-start)/dt,
                start_in_steps=start/dt, stop_in_steps=stop/dt,
                deposited_heat=deposited, exact_deposited_heat=exact_input,
                relative_heat_error=abs(deposited/exact_input-1),
                max_balance_over_peak=float(max_balance), max_temperature_error_over_peak=max_temp,
                reversal_time=crossing, exact_reversal_time=exact_crossing,
                reversal_error_over_interval=abs(crossing-exact_crossing)*16*rate)


if __name__ == "__main__":
    runs = []
    for width in (.05, .5):
        previous = None
        for n in (16, 32, 64):
            row = measure(n, width)
            row["orders"] = {} if previous is None else {
                k: math.log2(previous[k]/row[k]) for k in ("max_temperature_error_over_peak", "reversal_error_over_interval")}
            row["passed"] = bool(max(row["relative_heat_error"], row["max_balance_over_peak"]) < 1e-12
                                 and (n < 64 or min(row["orders"].values()) > 1.7))
            runs.append(row)
            previous = row
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, runs=runs, diffusivity=1., relaxation_time=1.,
                          sources=[provenance(f) for f in (__file__, pulse.__file__, lbm.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_two_mode_pulse.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
