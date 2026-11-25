"""Verify full D3Q7 slab heating when a rectangular two-mode pulse cuts lattice steps."""
import argparse
import json
import math

import numpy as np
import torch

import dg_reversing_wall_pulse as pulse
from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, width, tau=1.):
    rate = math.pi**2
    start, stop = 1.2/(16*rate), (1.2+width)/(16*rate)
    dx = 1/n
    # D3Q7 has sound speed squared 1/4 in lattice units.
    dt = .25*(tau-.5)*dx**2
    steps = round((1/16)/dt)
    x = (torch.arange(n, dtype=torch.float64)+.5)/n
    waves = [torch.sin(k*math.pi*x).view(1, 1, n)*np.sinc(.5*k/n) for k in (1, 2)]
    source = rate*(waves[0]+waves[1])
    zero = torch.zeros_like(source)
    field = zero
    g = lbm.thermal_equilibrium_3d(zero, zero, zero, zero)
    wall_history = []
    deposited = max_balance = max_temp = 0.
    peak = -math.expm1(-rate*(stop-start))
    for step in range(steps):
        t = step*dt
        early, late = lbm.thermal_pulse_increments_3d(source, time=t, dt=dt, start=start, stop=stop)
        post = lbm.thermal_collide_bgk_3d(g, field, zero, zero, zero, tau) + early
        streamed, inward = lbm.thermal_temperature_wall_pair_3d(
            lbm.thermal_stream_3d(post), "x", (0., 0.), dt=dt, spacing=dx, volumetric_heat_capacity=1.)
        g = streamed + late
        updated = lbm.thermal_macroscopic_3d(g)
        wall = -np.array([q.item() for q in inward])
        wall_history.append(wall)
        injection = float((early+late).sum(0).mean())
        deposited += injection
        max_balance = max(max_balance, abs(float((updated-field).mean()) + dt*wall.sum() - injection)/peak)
        on = max(0., min(t+dt, stop)-start)
        off = max(0., t+dt-stop)
        exact = sum(-math.expm1(-k*k*rate*on)*math.exp(-k*k*rate*off)*wave/(k*k)
                    for k, wave in enumerate(waves, start=1))
        max_temp = max(max_temp, float((updated-exact).abs().max())/peak)
        field = updated
    walls = np.asarray(wall_history)
    times = dt*np.arange(1, steps+1)
    on_time = np.clip(times-start, 0, stop-start)
    off_time = np.maximum(times-stop, 0)
    # Integrate the continuum modal ODE from zero to each lattice time.
    heat_modes = []
    for k in (1, 2):
        modal_temperature = -np.expm1(-k*k*rate*on_time)*np.exp(-k*k*rate*off_time)/(k*k)
        heat_modes.append(math.pi/k*(on_time-modal_temperature/rate))
    exact_heat = np.column_stack((heat_modes[0]+heat_modes[1], heat_modes[0]-heat_modes[1]))
    wall_heat_error = float(np.abs(dt*np.cumsum(walls, axis=0)-exact_heat).max()/peak)
    power = walls[:, 1]
    first = math.ceil(stop/dt)
    j = first + pulse.reversal_bracket_index(power[first:])
    crossing = (j+.5-power[j]/(power[j+1]-power[j]))*dt
    exact_crossing = pulse.exact_reversal_time(start*rate, stop*rate, rate)
    exact_input = (stop-start)*rate*2/math.pi
    return dict(cells=n, dt=dt, end_time=steps*dt, width_over_interval=width, pulse_duration_in_steps=(stop-start)/dt,
                start_in_steps=start/dt, stop_in_steps=stop/dt,
                deposited_heat=deposited, exact_deposited_heat=exact_input,
                relative_heat_error=abs(deposited/exact_input-1),
                max_cumulative_wall_heat_error_over_peak=wall_heat_error,
                max_balance_over_peak=float(max_balance), max_temperature_error_over_peak=max_temp,
                reversal_time=crossing, exact_reversal_time=exact_crossing,
                reversal_error_over_interval=abs(crossing-exact_crossing)*16*rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, choices=(.75, 1., 1.5), default=1.)
    tau = parser.parse_args().tau
    runs = []
    for width in (.05, .5):
        previous = None
        for n in (16, 32, 64):
            row = measure(n, width, tau)
            row["orders"] = {} if previous is None else {
                k: math.log2(previous[k]/row[k]) for k in ("max_temperature_error_over_peak", "reversal_error_over_interval",
                                                              "max_cumulative_wall_heat_error_over_peak")}
            row["passed"] = bool(max(row["relative_heat_error"], row["max_balance_over_peak"]) < 1e-12
                                 and (n < 64 or min(row["orders"].values()) > 1.7))
            runs.append(row)
            previous = row
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, runs=runs, diffusivity=1., relaxation_time=tau,
                          sources=[provenance(f) for f in (__file__, pulse.__file__, lbm.__file__)],
                          reproduction=f"OMP_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_two_mode_pulse.py --tau {tau:g}"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
