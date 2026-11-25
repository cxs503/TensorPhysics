"""Harmonic transmission across a resistive contact from a strip in one material."""
import json
import math

import numpy as np
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance
import fvm_strip_heater_across_contact as strip


def measure(n, omega, mirrored, backend):
    mid = n // 2
    bounds = (5/8, 5/6) if mirrored else (1/6, 3/8)
    conductivities = (4., 1.) if mirrored else (1., 4.)
    capacities = (2.5, 1.25) if mirrored else (1.25, 2.5)
    faces = np.linspace(0., 1., n+1)
    dx = 1/n
    start, end = bounds
    source = 12 * np.diff(np.clip(faces-start, 0., end-start)) / dx
    capacity = np.repeat(capacities, mid)
    resistance = np.zeros(n-1)
    resistance[mid-1] = 0.35
    field, cuts = sph.solve_harmonic_layered_slab(
        source, dx, (0., 0.), omega, conductivity=np.repeat(conductivities, mid),
        volumetric_heat_capacity=capacity, contact_resistance_x=resistance,
        h_factor=0.9, backend=backend, return_x_flux=True,
    )
    field, cuts = field.numpy(), cuts.numpy()
    widths, gaps = strip.strip_segments(*bounds)
    reference = strip.contact_admittance_reference(
        conductivities, capacities, 0.35, omega, (0., 0.), (0., 0.), widths, 12., gaps)
    actual = np.array([-cuts[0], cuts[-1], cuts[mid]])
    exact = np.r_[reference[:2], reference[2]/0.35]
    ratio = actual/exact
    return {
        "columns": n, "backend": backend, "omega_rad_s": omega, "strip_bounds_m": bounds,
        "reference_wall_amplitudes_W_m2": abs(exact[:2]).tolist(),
        "wall_amplitudes_W_m2": abs(actual[:2]).tolist(),
        "wall_and_contact_phases_rad": np.angle(actual).tolist(),
        "contact_amplitude_W_m2": abs(actual[2]),
        "far_wall_relative_phasor_error": abs(ratio[0 if mirrored else 1]-1),
        "max_relative_phasor_error": max(abs(ratio-1)),
        "max_phase_error_rad": max(abs(np.angle(ratio))),
        "max_column_residual_W_m2": max(abs(np.diff(cuts)+dx*(1j*omega*capacity*field-source))),
    }


def main():
    runs = []
    for omega in (2., 20., 200., 800.):
        for mirrored in (False, True):
            previous = None
            for n in (128, 256, 512, 1024):
                row = measure(n, omega, mirrored, "banded")
                error = row["max_relative_phasor_error"]
                passed = row["max_column_residual_W_m2"] < 1e-9
                if previous is not None:
                    row["phasor_error_order"] = math.log2(previous/error)
                    passed &= row["phasor_error_order"] > 1.85
                if n == 128:
                    dense = measure(n, omega, mirrored, "torch")
                    np.testing.assert_allclose(dense["wall_amplitudes_W_m2"], row["wall_amplitudes_W_m2"],
                                               rtol=1e-10, atol=0.)
                    np.testing.assert_allclose(dense["contact_amplitude_W_m2"], row["contact_amplitude_W_m2"],
                                               rtol=1e-10, atol=0.)
                    phase_difference = np.asarray(dense["wall_and_contact_phases_rad"]) - row["wall_and_contact_phases_rad"]
                    passed &= max(abs(np.expm1(1j*phase_difference))) < 1e-10
                    passed &= dense["max_column_residual_W_m2"] < 1e-9
                    row["dense_backend"] = dense
                if n == 1024:
                    passed &= error < 1e-3 and row["max_phase_error_rad"] < 1e-3
                row["passed"] = bool(passed)
                runs.append(row)
                previous = error
    passed = all(row["passed"] for row in runs)
    print(json.dumps({
        "passed": passed, "runs": runs, "sources": [provenance(p) for p in (__file__, strip.__file__, sph.__file__)],
        "reproduction": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src:../TensorFVM/src python examples/sph_detached_strip_transmission.py",
    }, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
