"""Verify a LINE2 rod's nodal temperatures during and after a subinterval heat pulse."""
import json
import math

import numpy as np
import torch
from tensorfem import thermal

from conduction_provenance import provenance


def measure(n, method, width, sparse):
    # Solve in excess above fixed 305 K ends. Unit k, capacity and area.
    rate, amplitude = math.pi**2, 2**-44
    dt = 1/(16*rate)
    outputs = np.arange(19)*dt
    start = 1.2*dt
    stop = start + width*dt
    time = np.union1d(outputs, [start, stop])
    x = torch.linspace(0., 1., n+1, dtype=torch.float64)
    elements = torch.stack((torch.arange(n), torch.arange(1, n+1)), dim=1)
    model = thermal.ThermalModel(x[:, None], elements, 1., 1., 1., torch.tensor([0,n]), x.new_zeros(2))
    # Consistent LINE2 loads integrate N_i*sin(pi*x) on each element.
    xi, weights = np.polynomial.legendre.leggauss(6)
    shape = np.column_stack(((1-xi)/2, (1+xi)/2))
    sample = (np.arange(n)[:, None]+shape[:, 1])/n
    local = (np.sin(math.pi*sample)*weights) @ shape/(2*n)
    load = np.zeros(n+1)
    load[:-1] += local[:, 0]
    load[1:] += local[:, 1]
    duration = np.clip(time-start, 0, stop-start)
    deposited = amplitude*rate*np.diff(duration)[:, None]*load
    result = thermal.solve_transient_thermal(
        model, torch.from_numpy(time), torch.zeros_like(x), theta=.5, method=method,
        sparse=sparse, nodal_step_heat=torch.from_numpy(deposited))
    mode = np.sin(math.pi*x.numpy())
    factor = -np.expm1(-rate*duration)*np.exp(-rate*np.maximum(time-stop, 0))
    peak = -math.expm1(-rate*(stop-start))
    error = abs(result.temperature.numpy()/amplitude - factor[:, None]*mode)/peak
    # Average all internal stage contributions back onto requested intervals.
    indices = np.searchsorted(time, outputs)
    weighted = result.interval_temperature.numpy()*np.diff(time)[:, None]
    means = np.add.reduceat(weighted, indices[:-1], axis=0)/np.diff(outputs)[:, None]
    integral = np.diff(duration[indices])-np.diff(factor[indices])/rate
    exact_means = integral[:, None]/np.diff(outputs)[:, None]*mode
    # Reactions are inward at both ends; the analytical section integral
    # is rightward. Subtract the consistent distributed input locally too.
    local_source = amplitude*rate*np.diff(duration)[:, None, None]*local
    element_heat = thermal.thermal_element_nodal_heat_steps(
        model, result, element_nodal_step_heat=torch.from_numpy(local_source)).numpy()
    element_heat = np.add.reduceat(element_heat, indices[:-1], axis=0)
    sections = np.column_stack((element_heat[:, 0, 0], -element_heat[:, :, 1]))
    reactions = np.add.reduceat(result.nodal_reaction_heat_steps().numpy(), indices[:-1], axis=0)
    fixed = reactions[:, [0, -1]]
    exact_sections = -math.pi*amplitude*integral[:, None]*np.cos(math.pi*x.numpy())
    scale = math.pi*amplitude*dt*peak
    section_error = abs(sections-exact_sections)/scale
    exact_ends = np.column_stack((exact_sections[:, 0], -exact_sections[:, -1]))
    # Unit volumetric capacity and area. Integrate the LINE2 temperature
    # exactly in space to obtain the stored energy change per output interval.
    stored = result.temperature.numpy()[indices][:, elements.numpy()].mean(axis=2).sum(axis=1)/n
    source_heat = np.add.reduceat(deposited.sum(axis=1), indices[:-1])
    balance = np.diff(stored)-source_heat-fixed.sum(axis=1)
    return dict(crossing_end_reaction_error_over_scale=float(abs(fixed[1]-exact_ends[1]).max()/scale),
                max_section_heat_error_over_scale=float(section_error.max()),
                free_node_residual_over_scale=float(abs(reactions[:, 1:-1]).max()/scale),
                boundary_reconstruction_mismatch_over_scale=float(abs(
                    fixed - np.column_stack((sections[:, 0], -sections[:, -1]))).max()/scale),
                energy_imbalance_over_peak=float(abs(balance).max()/(amplitude*peak)),
                elements=n, method=method, sparse=sparse, width_over_interval=width,
                crossing_node_error_over_peak=float(error[indices[2]].max()),
                decay_node_error_over_peak=float(error[indices[3:]].max()),
                interval_mean_error_over_peak=float(abs(means/amplitude-exact_means).max()/peak),
                source_integral_relative_error=float(abs(load.sum()*math.pi/2-1)))


if __name__ == "__main__":
    runs = []
    for n in (16, 32, 64):
        for method in ("theta", "sdirk2"):
            for width in (.05, .5):
                for sparse in (False, True):
                    row = measure(n, method, width, sparse)
                    tolerance = 0.004 if n == 64 else 0.05
                    row["passed"] = (max(row[k] for k in ("crossing_node_error_over_peak", "decay_node_error_over_peak",
                                          "interval_mean_error_over_peak", "max_section_heat_error_over_scale",
                                          "crossing_end_reaction_error_over_scale")) < tolerance
                                     and row["source_integral_relative_error"] < 1e-12
                                     and max(row[k] for k in ("free_node_residual_over_scale",
                                         "boundary_reconstruction_mismatch_over_scale", "energy_imbalance_over_peak")) < 1e-10)
                    runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(thermal.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src python examples/fem_subinterval_heat_pulse.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
