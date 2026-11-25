"""Harmonic transmission across a resistive contact from a strip in one material."""
import argparse
import json
import math

import numpy as np
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance
import fvm_strip_heater_across_contact as strip


def measure(n, omega, mirrored, backend, swap_materials=False, transverse_mode=0, h_factor=0.9,
            homogeneous=False, contact_resistance=0.35):
    mid = n // 2
    bounds = (5/8, 5/6) if mirrored else (1/6, 3/8)
    conductivities = (4., 1.) if mirrored else (1., 4.)
    capacities = (2.5, 1.25) if mirrored else (1.25, 2.5)
    if swap_materials:
        conductivities = conductivities[::-1]
        capacities = capacities[::-1]
    if homogeneous:
        conductivities = (1., 1.)
        capacities = (1.25, 1.25)
    faces = np.linspace(0., 1., n+1)
    dx = 1/n
    start, end = bounds
    source = 12 * np.diff(np.clip(faces-start, 0., end-start)) / dx
    capacity = np.repeat(capacities, mid)
    resistance = np.zeros(n-1)
    resistance[mid-1] = contact_resistance
    beta = 2 * math.pi * transverse_mode  # unit transverse period
    options = dict(conductivity=np.repeat(conductivities, mid), volumetric_heat_capacity=capacity,
                   contact_resistance_x=resistance, h_factor=h_factor, backend=backend, transverse_wavenumber=beta)
    field, cuts = sph.solve_harmonic_layered_slab(source, dx, (0., 0.), omega, return_x_flux=True, **options)
    field, cuts = field.numpy(), cuts.numpy()
    far_wall = 0 if mirrored else 1
    # Reciprocity: the outward power from the buried source equals its
    # overlap with the source-free response to a unit far-wall temperature.
    # Both complex solutions use the same frequency, without conjugation.
    walls = (1., 0.) if mirrored else (0., 1.)
    driven = sph.solve_harmonic_layered_slab(np.zeros(n), dx, walls, omega, **options)
    reciprocal_power = dx * np.dot(source, driven.numpy())
    widths, gaps = strip.strip_segments(*bounds)
    exact = strip.contact_admittance_reference(
        conductivities, capacities, contact_resistance, omega, (0., 0.), (0., 0.), widths, 12., gaps, beta,
        return_contact_power=True)
    actual = np.array([-cuts[0], cuts[-1], cuts[mid]])
    ratio = actual/exact
    row = {
        "columns": n, "backend": backend, "transverse_mode": transverse_mode, "h_factor": h_factor, "omega_rad_s": omega, "strip_bounds_m": bounds,
        "contact_resistance_m2_K_W": contact_resistance, "conductivities_W_m_K": conductivities, "capacities_J_m3_K": capacities,
        "far_wall_reciprocity_relative_error": abs(reciprocal_power/actual[far_wall]-1),
        "reference_wall_amplitudes_W_m2": abs(exact[:2]).tolist(),
        "wall_amplitudes_W_m2": abs(actual[:2]).tolist(),
        "wall_and_contact_phases_rad": np.angle(actual).tolist(),
        "contact_amplitude_W_m2": abs(actual[2]),
        "far_wall_relative_phasor_error": abs(ratio[far_wall]-1),
        "max_relative_phasor_error": max(abs(ratio-1)),
        "max_phase_error_rad": max(abs(np.angle(ratio))),
    }
    if transverse_mode == 0:
        row["max_column_residual_W_m2"] = max(abs(np.diff(cuts)+dx*(1j*omega*capacity*field-source)))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swap-materials", action="store_true", help="embed the strip in the higher-conductivity material")
    parser.add_argument("--transverse-mode", type=int, choices=(0, 1, 2), default=0,
                        help="number of sinusoidal heat-source periods across the one-metre slab width")
    parser.add_argument("--axial-stencil", action="store_true", help="restrict the kernel support to axial neighbours")
    parser.add_argument("--homogeneous", action="store_true", help="use identical bulk properties on both sides")
    parser.add_argument("--perfect-contact", action="store_true", help="set contact resistance to zero")
    args = parser.parse_args()
    material_options = dict(homogeneous=args.homogeneous, contact_resistance=0. if args.perfect_contact else 0.35)
    h_factor = 0.7 if args.axial_stencil else 0.9
    swap_materials, transverse_mode = args.swap_materials, args.transverse_mode
    oblique_mode = transverse_mode != 0 and not args.axial_stencil
    grids = (128, 256, 512, 1024, 2048, 4096, 8192, 16384) if oblique_mode else (128, 256, 512, 1024)
    runs = []
    for omega in (2., 20., 200., 800.):
        for mirrored in (False, True):
            previous = None
            previous_reciprocity = None
            for n in grids:
                row = measure(n, omega, mirrored, "banded", swap_materials, transverse_mode, h_factor, **material_options)
                error = row["max_relative_phasor_error"]
                reciprocity = row["far_wall_reciprocity_relative_error"]
                # With oblique ghost links the imposed temperature and reported wall
                # power sample different transverse rows. Test the continuum limit.
                passed = np.isfinite(reciprocity) if oblique_mode else reciprocity < 1e-9
                if transverse_mode == 0:
                    passed &= row["max_column_residual_W_m2"] < 1e-9
                if previous is not None:
                    row["phasor_error_order"] = math.log2(previous/error)
                    if oblique_mode:
                        row["reciprocity_error_order"] = math.log2(previous_reciprocity/reciprocity)
                        passed &= row["reciprocity_error_order"] > 1.9
                    else:
                        passed &= row["phasor_error_order"] > 1.85
                if n == 128:
                    dense = measure(n, omega, mirrored, "torch", swap_materials, transverse_mode, h_factor, **material_options)
                    np.testing.assert_allclose(dense["wall_amplitudes_W_m2"], row["wall_amplitudes_W_m2"],
                                               rtol=1e-10, atol=0.)
                    np.testing.assert_allclose(dense["contact_amplitude_W_m2"], row["contact_amplitude_W_m2"],
                                               rtol=1e-10, atol=0.)
                    phase_difference = np.asarray(dense["wall_and_contact_phases_rad"]) - row["wall_and_contact_phases_rad"]
                    passed &= max(abs(np.expm1(1j*phase_difference))) < 1e-10
                    if transverse_mode == 0:
                        passed &= dense["max_column_residual_W_m2"] < 1e-9
                    np.testing.assert_allclose(dense["far_wall_reciprocity_relative_error"], reciprocity, atol=1e-10, rtol=0.)
                    row["dense_backend"] = dense
                if n == grids[-1]:
                    if oblique_mode:
                        passed &= reciprocity < 1e-6 and row["phasor_error_order"] > 0.8
                    passed &= error < 1e-3 and row["max_phase_error_rad"] < 1e-3
                row["passed"] = bool(passed)
                runs.append(row)
                previous = error
                previous_reciprocity = reciprocity
    passed = all(row["passed"] for row in runs)
    print(json.dumps({
        "passed": passed, "runs": runs, "sources": [provenance(p) for p in (__file__, strip.__file__, sph.__file__)],
        "reproduction": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src:../TensorFVM/src python examples/sph_detached_strip_transmission.py"
                        + (" --swap-materials" if swap_materials else "") + (" --axial-stencil" if args.axial_stencil else "")
                        + (f" --transverse-mode {transverse_mode}" if transverse_mode else "")
                        + (" --homogeneous" if args.homogeneous else "") + (" --perfect-contact" if args.perfect_contact else ""),
    }, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
