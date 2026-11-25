"""Compare consistent-mass LINE2 and finite-volume rods with an off-grid strip source."""
import argparse
import json
import math

import numpy as np
import torch
from tensorfem import thermal
from tensorfvm.cht import rod

from conduction_provenance import provenance
import fvm_strip_heater_across_contact as strip


def measure(n, mirrored, omega, storing=False, heated=False, balanced=False):
    trace_capacity = np.array([0.18, 0.42]) if storing else np.zeros(2)
    trace_power = np.array([0.8j, -0.35+0.2j]) if heated else np.zeros(2)
    generation = 12.
    if balanced:
        trace_power = np.array([1j, -1j])
        generation = 0.
    if mirrored:
        trace_power = trace_power[::-1]
        trace_capacity = trace_capacity[::-1]
    fvm = strip.measure(n, mirrored, omega, trace_capacity, trace_power, generation)
    start, end = fvm["strip_bounds_m"]
    conductivity = (4., 1.) if mirrored else (1., 4.)
    capacity = (2.5, 1.25) if mirrored else (1.25, 2.5)
    resistance = 0.35
    faces = np.linspace(0., 1., n+1)
    mid = n // 2
    nodes = np.insert(faces, mid, faces[mid])
    left = np.arange(n) + (np.arange(n) >= mid)
    elements = np.column_stack((left, left+1))

    # Integrate each linear shape function on the actual heated subinterval.
    # Its mean there is its value at the subinterval midpoint.
    a = np.maximum(start, faces[:-1])
    b = np.minimum(end, faces[1:])
    power = generation * np.maximum(b-a, 0.)
    right_fraction = ((a+b)/2 - faces[:-1]) / np.diff(faces)
    load = np.zeros_like(nodes, dtype=complex)
    np.add.at(load, left, power*(1-right_fraction))
    np.add.at(load, left+1, power*right_fraction)
    load[mid:mid+2] += trace_power
    nodal_capacity = np.zeros_like(nodes)
    nodal_capacity[mid:mid+2] = trace_capacity
    model = thermal.ThermalModel(
        torch.from_numpy(nodes[:, None]), torch.from_numpy(elements),
        torch.from_numpy(np.repeat(conductivity, mid)), 1., torch.from_numpy(np.repeat(capacity, mid)),
        torch.tensor([0, n+1]), torch.zeros(2, dtype=torch.float64),
        nodal_heat=torch.from_numpy(load), contact_pairs=torch.tensor([[mid, mid+1]]),
        contact_conductance=1/resistance, nodal_heat_capacity=torch.from_numpy(nodal_capacity),
    )
    state = thermal.solve_harmonic_thermal(model, omega, sparse=True)
    t = state.temperature.numpy()
    outward = -state.reaction_power.numpy()[[0, -1]]
    jump = t[mid]-t[mid+1]
    actual = np.r_[outward, jump]
    if balanced:
        exact = strip.contact_admittance_reference(conductivity, capacity, resistance, omega, trace_capacity, trace_power)
    if not balanced or omega <= 20.:
        transfer = strip.harmonic_reference(start, end, conductivity, capacity, generation, resistance,
                                            omega, trace_capacity, trace_power)
        if balanced:
            np.testing.assert_allclose(exact, transfer, rtol=1e-11, atol=1e-13)
        else:
            exact = transfer
    element_load = np.column_stack((power*(1-right_fraction), power*right_fraction))
    local = thermal.thermal_element_nodal_power(
        model, state.temperature, omega, element_nodal_heat=torch.from_numpy(element_load)).numpy()
    trace_net = np.array([-local[mid-1, 1]-jump/resistance, jump/resistance-local[mid, 0]])
    trace_storage = 1j*omega*trace_capacity*t[mid:mid+2]
    storage = 1j*omega*np.sum(np.diff(faces)*np.repeat(capacity, mid)*t[elements].mean(axis=1))
    return {
        "cells": n, "omega_rad_s": omega, "strip_bounds_m": [start, end],
        "trace_capacity_J_K": trace_capacity.tolist(), "strip_generation_W_m3": generation,
        "trace_power_real_W": trace_power.real.tolist(), "trace_power_imag_W": trace_power.imag.tolist(),
        "fem_trace_storage_amplitude_W": abs(trace_storage).tolist(),
        "fem_max_trace_balance_error_W": max(abs(trace_net+trace_power-trace_storage)),
        "fem_wall_amplitude_W": abs(outward).tolist(), "fem_contact_jump_amplitude_K": abs(jump),
        "fem_wall_and_jump_phase_rad": np.angle(actual).tolist(),
        "fem_max_wall_relative_error": max(abs(outward/exact[:2]-1)),
        "fem_wall_power_error_W": max(abs(outward-exact[:2])),
        "fem_contact_jump_error_K": abs(jump-exact[2]),
        "fem_max_phase_error_rad": max(abs(np.angle(actual/exact))),
        "fem_energy_residual_W": abs(storage+sum(trace_storage)+sum(outward)-sum(trace_power)-generation*(end-start)),
        "fvm": fvm,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storing", action="store_true", help="put unequal heat capacities on the two contact traces")
    heating = parser.add_mutually_exclusive_group()
    heating.add_argument("--balanced", action="store_true", help="drive only the two contact traces with opposite powers")
    heating.add_argument("--heated", action="store_true", help="apply independent phased heat loads at the two contact traces")
    parser.add_argument("--high-frequency", action="store_true", help="resolve attenuated wall powers under balanced contact heating")
    args = parser.parse_args()
    if args.high_frequency and not args.balanced:
        parser.error("--high-frequency requires --balanced")
    frequencies = (200., 800.) if args.high_frequency else (2., 20.)
    grids = (128, 256, 512, 1024) if args.high_frequency else (16, 32, 64, 128, 256)
    runs = []
    for omega in frequencies:
        for mirrored in (False, True):
            previous = None
            for n in grids:
                row = measure(n, mirrored, omega, args.storing, args.heated, args.balanced)
                fvm = row["fvm"]
                passed = max(row["fem_energy_residual_W"], fvm["max_cell_balance_error_W"]) < 1e-9
                passed &= max(row["fem_max_trace_balance_error_W"], fvm["max_trace_balance_error_W"]) < 1e-9
                for data, old, metrics in (
                    (row, previous, ("fem_wall_power_error_W", "fem_contact_jump_error_K")),
                    (fvm, previous["fvm"] if previous else None, ("max_wall_power_error_W", "contact_jump_error_K")),
                ):
                    for metric in metrics:
                        if old:
                            data[metric+"_order"] = math.log2(old[metric]/data[metric])
                            passed &= data[metric+"_order"] > 1.85
                        if n == grids[-1]:
                            passed &= data[metric] < 2e-4
                if n == grids[-1]:
                    passed &= max(row["fem_max_phase_error_rad"], fvm["max_phase_error_rad"]) < 5e-4
                    if args.high_frequency:
                        passed &= max(row["fem_max_wall_relative_error"], fvm["max_wall_relative_error"]) < 1e-3
                row["passed"] = bool(passed)
                runs.append(row)
                previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps({
        "passed": passed, "runs": runs,
        "sources": [provenance(path) for path in (__file__, strip.__file__, thermal.__file__, rod.__file__)],
        "reproduction": "OMP_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src:../TensorFVM/src python examples/fem_fvm_contact_strip.py"
                        + (" --storing" if args.storing else "") + (" --heated" if args.heated else "")
                        + (" --balanced" if args.balanced else "") + (" --high-frequency" if args.high_frequency else ""),
    }, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
