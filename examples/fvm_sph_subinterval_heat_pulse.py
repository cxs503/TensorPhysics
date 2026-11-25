"""One short pulse in a fixed-temperature slab, verified with FVM cell means and SPH particles."""
import json
import math

import numpy as np
from tensorfvm.cht import rod

import sph_subinterval_heat_pulse as sph
from conduction_provenance import provenance


def measure(n, method, width):
    # Unit material properties and area. Use excess above the 305 K walls in
    # the linear rod solve so weak temperatures survive at the output.
    amplitude = 2**-44
    rate = math.pi**2
    dt = 1/(16*rate)
    times = np.arange(19)*dt
    start = 1.2*dt
    stop = start + width*dt
    dx = 1/n
    centers = (np.arange(n)+0.5)*dx
    mode = np.sin(np.pi*centers)*np.sinc(dx/2)
    cell, _, _, _, ledger = rod.solve_transient_stepped_rod(
        np.full(n, dx), 1., 1., 1., (0., 0.), times, np.zeros(n),
        source=lambda t: amplitude*rate*mode if start <= t < stop else 0.,
        load_jump_times=(start, stop), method=method, flux_order=2, return_step_heat=True,
    )
    duration = np.clip(times-start, 0, stop-start)
    factor = -np.expm1(-rate*duration)*np.exp(-rate*np.maximum(times-stop, 0))
    peak = -math.expm1(-rate*(stop-start))
    field_error = abs(cell/amplitude - factor[:, None]*mode)/peak
    exact_input = amplitude*rate*np.diff(duration)*(2/math.pi)
    # Integral of the modal ODE gives the temperature-time integral without quadrature.
    integral = np.diff(duration) - np.diff(factor)/rate
    wall_heat = ledger["end_input"]
    error = abs(wall_heat + math.pi*amplitude*integral[:, None])/(math.pi*amplitude*dt*peak)
    stored = dx*cell.sum(axis=1)
    measured_input = ledger["source"].sum(axis=1)
    return dict(crossing_field_error_over_peak=float(field_error[2].max()),
                decay_field_error_over_peak=float(field_error[3:].max()),
                crossing_wall_heat_error_over_scale=float(error[1].max()),
                decay_wall_heat_error_over_scale=float(error[2:].max()),
                source_heat_relative_error=float(abs(measured_input-exact_input).max()/exact_input.max()),
                energy_residual_over_peak=float(abs(np.diff(stored)-measured_input-wall_heat.sum(axis=1)).max()/(amplitude*peak)))


if __name__ == "__main__":
    runs = []
    for n in (16, 32, 64):
        for method in ("theta", "sdirk2"):
            for width in (0.05, 0.5):
                fv = measure(n, method, width)
                particles = sph.measure(n, method, width, 0)
                tolerance = 0.004 if n == 64 else 0.05
                passed = all(max(row[key] for key in ("crossing_field_error_over_peak", "decay_field_error_over_peak",
                                 "crossing_wall_heat_error_over_scale", "decay_wall_heat_error_over_scale")) < tolerance
                             and row["energy_residual_over_peak"] < 1e-11 for row in (fv, particles))
                runs.append(dict(columns=n, method=method, width_over_interval=width, fvm=fv, sph=particles,
                                 passed=passed and fv["source_heat_relative_error"] < 1e-12))
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs,
                          sources=[provenance(path) for path in (__file__, rod.__file__, sph.__file__, sph.heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/fvm_sph_subinterval_heat_pulse.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
