"""Compare consistent-mass LINE2 and finite-volume rods with an off-grid strip source."""
import json
import math

import numpy as np
import torch
from tensorfem import thermal
from tensorfvm.cht import rod

from conduction_provenance import provenance
import fvm_strip_heater_across_contact as strip


def measure(n, mirrored, omega):
    fvm = strip.measure(n, mirrored, omega)
    start, end = fvm["strip_bounds_m"]
    conductivity = (4., 1.) if mirrored else (1., 4.)
    capacity = (2.5, 1.25) if mirrored else (1.25, 2.5)
    resistance, generation = 0.35, 12.
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
    load = np.zeros_like(nodes)
    np.add.at(load, left, power*(1-right_fraction))
    np.add.at(load, left+1, power*right_fraction)
    model = thermal.ThermalModel(
        torch.from_numpy(nodes[:, None]), torch.from_numpy(elements),
        torch.from_numpy(np.repeat(conductivity, mid)), 1., torch.from_numpy(np.repeat(capacity, mid)),
        torch.tensor([0, n+1]), torch.zeros(2, dtype=torch.float64),
        nodal_heat=torch.from_numpy(load), contact_pairs=torch.tensor([[mid, mid+1]]),
        contact_conductance=1/resistance,
    )
    state = thermal.solve_harmonic_thermal(model, omega, sparse=True)
    t = state.temperature.numpy()
    outward = -state.reaction_power.numpy()[[0, -1]]
    jump = t[mid]-t[mid+1]
    actual = np.r_[outward, jump]
    exact = strip.harmonic_reference(start, end, conductivity, capacity, generation, resistance, omega)
    storage = 1j*omega*np.sum(np.diff(faces)*np.repeat(capacity, mid)*t[elements].mean(axis=1))
    return {
        "cells": n, "omega_rad_s": omega, "strip_bounds_m": [start, end],
        "fem_wall_amplitude_W": abs(outward).tolist(), "fem_contact_jump_amplitude_K": abs(jump),
        "fem_wall_and_jump_phase_rad": np.angle(actual).tolist(),
        "fem_wall_power_error_W": max(abs(outward-exact[:2])),
        "fem_contact_jump_error_K": abs(jump-exact[2]),
        "fem_max_phase_error_rad": max(abs(np.angle(actual/exact))),
        "fem_energy_residual_W": abs(storage+sum(outward)-generation*(end-start)),
        "fvm": fvm,
    }


def main():
    runs = []
    for omega in (2., 20.):
        for mirrored in (False, True):
            previous = None
            for n in (16, 32, 64, 128, 256):
                row = measure(n, mirrored, omega)
                fvm = row["fvm"]
                passed = max(row["fem_energy_residual_W"], fvm["max_cell_balance_error_W"]) < 1e-9
                for data, old, metrics in (
                    (row, previous, ("fem_wall_power_error_W", "fem_contact_jump_error_K")),
                    (fvm, previous["fvm"] if previous else None, ("max_wall_power_error_W", "contact_jump_error_K")),
                ):
                    for metric in metrics:
                        if old:
                            data[metric+"_order"] = math.log2(old[metric]/data[metric])
                            passed &= data[metric+"_order"] > 1.85
                        if n == 256:
                            passed &= data[metric] < 2e-4
                if n == 256:
                    passed &= max(row["fem_max_phase_error_rad"], fvm["max_phase_error_rad"]) < 5e-4
                row["passed"] = bool(passed)
                runs.append(row)
                previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps({
        "passed": passed, "runs": runs,
        "sources": [provenance(path) for path in (__file__, strip.__file__, thermal.__file__, rod.__file__)],
        "reproduction": "OMP_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src:../TensorFVM/src python examples/fem_fvm_contact_strip.py",
    }, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
