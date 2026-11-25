"""Fourier flux recovery from curved temperature profiles on either side of a thin layer."""
import json
import math

import numpy as np
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(n, beta, resistive):
    dx = 1/n
    first, last = n//2, n//2 + n//32
    k = np.ones(n)
    k[first:last] = 4
    cap = np.full(n, 1.25)
    cap[first:last] = 2.5
    contacts = np.zeros(n-1)
    if resistive:
        contacts[[first-1, last-1]] = [0.12, 0.23]
    # Specify continuous q and discontinuous q'. Integrate q/k exactly
    # within each half cell and add R*q at both interfaces.
    gradient = np.full(n, 0.8-0.3j)
    gradient[first:last] = -2.5+0.7j
    q = 0.7+0.2j + np.r_[0., np.cumsum(dx*gradient)]
    cell_drop = dx*(q[:-1]+q[1:])/(2*k)
    left = 0.4 + 0.3j - np.r_[0., np.cumsum(cell_drop[:-1]+contacts*q[1:-1])]
    target = left - (q[:-1]*dx/2 + gradient*dx**2/8)/k
    omega = 3.
    loss = 1j*omega*cap + k*beta**2
    source = gradient + loss*target
    walls = (left[0], left[-1]-cell_drop[-1])
    field = sph.solve_harmonic_layered_slab(
        source, dx, walls, omega, conductivity=k, volumetric_heat_capacity=cap,
        contact_resistance_x=contacts, h_factor=1.6, transverse_wavenumber=beta,
        backend="banded", split_interface_links=True,
    )
    def error(values, slope):
        flux = sph.recover_layered_interior_flux(values, dx, k, axial_flux_gradient=slope,
                                                 contact_resistance_x=contacts)
        return np.abs(flux.numpy()-q[1:-1])
    corrected = error(field, source-loss*field.numpy())
    face = np.arange(1, n)
    worst = int(face[corrected.argmax()])
    # Use the same physical regions on every grid. A fixed count of cells
    # would shrink the excluded neighbourhood as the mesh is refined.
    # At low resolution the thin layer may contain no eligible face.
    region_errors = {}
    for name, lo, hi in (("left", 0, first), ("layer", first, last), ("right", last, n)):
        inside = (face-lo >= n/128) & (hi-face >= n/128)
        if inside.any():
            region_errors[name] = float(corrected[inside].max())
    rms = float(np.sqrt(np.mean(corrected**2)))
    return {
        "worst_face_x_m": worst*dx,
        "worst_face_distance_to_contact_cells": min(abs(first-worst), abs(last-worst)),
        "interior_rms_error_W_m2": rms,
        "region_inset_m": 1/128,
        "region_max_error_W_m2": region_errors,
        "columns": n, "beta_rad_m": beta, "resistive": resistive,
        "exact_input_uncorrected_error_W_m2": float(error(target, 0.).max()),
        "exact_input_corrected_error_W_m2": float(error(target, gradient).max()),
        "solved_input_uncorrected_error_W_m2": float(error(field, 0.).max()),
        "solved_input_corrected_error_W_m2": float(corrected.max()),
        "solved_contact_errors_W_m2": corrected[[first-1, last-1]].tolist(),
        "temperature_error_K": float(np.abs(field.numpy()-target).max()),
    }


def main():
    rows = []
    for resistive in (False, True):
        for beta in (0., 2*math.pi, 4*math.pi):
            previous = None
            for n in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
                row = measure(n, beta, resistive)
                error = row["solved_input_corrected_error_W_m2"]
                passed = row["exact_input_corrected_error_W_m2"] < 1e-10
                if previous:
                    row["solved_flux_order"] = math.log2(previous["solved_input_corrected_error_W_m2"]/error)
                    row["interior_rms_order"] = math.log2(previous["interior_rms_error_W_m2"]/row["interior_rms_error_W_m2"])
                    row["region_orders"] = {
                        name: math.log2(previous["region_max_error_W_m2"][name]/value)
                        for name, value in row["region_max_error_W_m2"].items()
                        if name in previous["region_max_error_W_m2"]
                    }
                if n == 8192:
                    passed &= error < 2e-4 and row["solved_flux_order"] > 0.9
                    passed &= row["worst_face_distance_to_contact_cells"] <= 1
                    passed &= max(row["region_max_error_W_m2"].values()) < 1e-5
                    if beta:
                        passed &= min(row["region_orders"].values()) > 1.9
                row["passed"] = bool(passed)
                rows.append(row)
                previous = row
    passed = all(row["passed"] for row in rows)
    print(json.dumps({
        "passed": passed, "runs": rows, "sources": [provenance(path) for path in (__file__, sph.__file__)],
        "reproduction": "OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_curved_interlayer_flux.py",
    }, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
