"""One short pulse in a fixed-temperature slab, verified with FVM cell means and SPH particles."""
import json
import math

import numpy as np
from tensorfvm.cht import rod

import sph_subinterval_heat_pulse as sph
from conduction_provenance import provenance


def measure(n, method, width, *, return_mean_profile=False):
    # Unit material properties and area, with absolute initial and wall inputs.
    amplitude = 2**-44
    rate = math.pi**2
    dt = 1/(16*rate)
    times = np.arange(19)*dt
    start = 1.2*dt
    stop = start + width*dt
    dx = 1/n
    centers = (np.arange(n)+0.5)*dx
    mode = np.sin(np.pi*centers)*np.sinc(dx/2)
    cell, _, _, _, ledger, intervals = rod.solve_transient_stepped_rod(
        np.full(n, dx), 1., 1., 1., (305., 305.), times, np.full(n, 305.), temperature_reference=305.,
        source=lambda t: amplitude*rate*mode if start <= t < stop else 0.,
        load_jump_times=(start, stop), method=method, flux_order=2, return_step_heat=True,
        return_interval_temperatures=True,
    )
    duration = np.clip(times-start, 0, stop-start)
    factor = -np.expm1(-rate*duration)*np.exp(-rate*np.maximum(times-stop, 0))
    peak = -math.expm1(-rate*(stop-start))
    field_error = abs(cell/amplitude - factor[:, None]*mode)/peak
    exact_input = amplitude*rate*np.diff(duration)*(2/math.pi)
    # Integral of the modal ODE gives the temperature-time integral without quadrature.
    integral = np.diff(duration) - np.diff(factor)/rate
    mean_factor = integral/np.diff(times)
    cell_mean_error = abs(intervals["cell"]/amplitude - mean_factor[:, None]*mode)/peak
    fractions = np.array([0., 0.25, 0.5, 0.75, 1.])
    position = dx*(np.arange(n)[:, None] + fractions)
    sampled, power = rod.reconstruct_interval_rod_field(
        np.full(n, dx), 1., 1., intervals, fractions)
    # Compare reconstructed point temperatures and powers to their own
    # analytic interval means, distinct from the cell-volume averages.
    profile = mean_factor[:, None, None]
    sampled_error = abs(sampled/amplitude - profile*np.sin(math.pi*position))/peak
    power_error = abs(power/(math.pi*amplitude) + profile*np.cos(math.pi*position))/peak
    step_duration = np.diff(times)[:, None]
    ledger_error = max(abs(power[:, :, face]*step_duration-ledger[side]).max()
                       for face, side in ((0, "left"), (-1, "right")))
    wall_heat = ledger["end_input"]
    error = abs(wall_heat + math.pi*amplitude*integral[:, None])/(math.pi*amplitude*dt*peak)
    stored = dx*cell.sum(axis=1)
    measured_input = ledger["source"].sum(axis=1)
    metrics = dict(crossing_cell_mean_error_over_peak=float(cell_mean_error[1].max()),
                max_interval_cell_mean_error_over_peak=float(cell_mean_error.max()),
                max_interval_point_temperature_error_over_peak=float(sampled_error.max()),
                max_interval_point_power_error_over_scale=float(power_error.max()),
                reconstructed_face_heat_mismatch_over_scale=float(ledger_error/(math.pi*amplitude*dt*peak)),
                crossing_field_error_over_peak=float(field_error[2].max()),
                decay_field_error_over_peak=float(field_error[3:].max()),
                crossing_wall_heat_error_over_scale=float(error[1].max()),
                decay_wall_heat_error_over_scale=float(error[2:].max()),
                source_heat_relative_error=float(abs(measured_input-exact_input).max()/exact_input.max()),
                energy_residual_over_peak=float(abs(np.diff(stored)-measured_input-wall_heat.sum(axis=1)).max()/(amplitude*peak)))
    if return_mean_profile:
        # SPH particles lie at the cell centers. Use the reconstructed center
        # value at fraction 0.5, preserving the temporal interval averaging.
        return metrics, sampled[:, :, 2]/(amplitude*peak)
    return metrics


if __name__ == "__main__":
    runs = []
    previous = {}
    for n in (16, 32, 64):
        for method in ("theta", "sdirk2"):
            for width in (0.05, 0.5):
                fv, centers = measure(n, method, width, return_mean_profile=True)
                particles, particle_means = sph.measure(n, method, width, 0, return_mean_profile=True)
                # Take the maximum over every SPH row and x location for each interval.
                differences = abs(centers[:, None, :] - particle_means).max(axis=(1, 2))
                worst = float(differences.max())
                case = (method, width)
                order = None if case not in previous else float(np.log2(previous[case]/worst))
                previous[case] = worst
                tolerance = 0.004 if n == 64 else 0.05
                passed = all(max(row[key] for key in ("crossing_field_error_over_peak", "decay_field_error_over_peak",
                                 "crossing_wall_heat_error_over_scale", "decay_wall_heat_error_over_scale")) < tolerance
                             and row["energy_residual_over_peak"] < 1e-11 for row in (fv, particles))
                passed = passed and max(fv["max_interval_cell_mean_error_over_peak"],
                                        fv["max_interval_point_temperature_error_over_peak"],
                                        fv["max_interval_point_power_error_over_scale"]) < tolerance
                passed = passed and fv["reconstructed_face_heat_mismatch_over_scale"] < 1e-10
                passed = passed and max(particles["crossing_mean_error_over_peak"],
                                        particles["decay_mean_error_over_peak"]) < tolerance
                passed = passed and worst < tolerance and (n < 64 or order > 1.8)
                runs.append(dict(center_mean_difference_over_peak=differences.tolist(),
                                 max_center_mean_difference_over_peak=worst, center_mean_difference_order=order,
                                 columns=n, method=method, width_over_interval=width, fvm=fv, sph=particles,
                                 passed=passed and fv["source_heat_relative_error"] < 1e-12))
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs,
                          sources=[provenance(path) for path in (__file__, rod.__file__, sph.__file__, sph.heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/fvm_sph_subinterval_heat_pulse.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
