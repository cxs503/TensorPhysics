"""Steady or periodically heated slab with equal wall temperatures and an off-centre strip heater."""
import argparse
import cmath
import json
import math

import torch
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(columns, mirrored, omega=0.0):
    dx = 1 / columns
    faces = torch.arange(columns + 1, dtype=torch.float64) * dx
    centers = (faces[1:] + faces[:-1]) / 2
    start, end = (5 / 8, 5 / 6) if mirrored else (1 / 6, 3 / 8)
    power, conductivity, wall = 12.0, 1.6, 300.0

    # Deposit the exact overlap of the strip with each particle column.
    covered = (torch.clamp(faces[1:], max=end) - torch.clamp(faces[:-1], min=start)).clamp_min(0)
    source = power * covered / dx
    if omega:
        capacity = 2.4
        field, flux = sph.solve_harmonic_slab_2d(
            source.repeat(8, 1), dx, (0.0, 0.0), omega, conductivity=conductivity,
            volumetric_heat_capacity=capacity, return_x_flux=True,
        )
        # Integrate the Dirichlet Green function's outward wall derivative
        # over the uniformly heated strip. The east kernel is sinh(z*x)/sinh(z)
        # on this unit slab, and the west kernel is its reflection about x=1/2.
        z = cmath.sqrt(1j * omega * capacity / conductivity)
        def wall_flux(a, b):
            return power * (cmath.cosh(z*b) - cmath.cosh(z*a)) / (z*cmath.sinh(z))

        exact = torch.tensor([wall_flux(1-end, 1-start), wall_flux(start, end)],
                             dtype=torch.complex128)
        outward = torch.stack((-flux[0], flux[-1]))
        balance = flux.diff() + dx * (1j * omega * capacity * field.mean(0) - source)
        return {
            "columns": columns, "strip_bounds_m": [start, end], "omega_rad_s": omega,
            "west_east_amplitude_W_m2": outward.abs().tolist(),
            "west_east_reference_amplitude_W_m2": exact.abs().tolist(),
            "west_east_phase_rad": outward.angle().tolist(),
            "west_east_reference_phase_rad": exact.angle().tolist(),
            "max_wall_phasor_error_W_m2": float((outward - exact).abs().max()),
            "max_relative_wall_error": float((outward / exact - 1).abs().max()),
            "max_wall_phase_error_rad": float((outward / exact).angle().abs().max()),
            "max_column_balance_error_W_m2": float(balance.abs().max()),
        }

    field, flux = sph.solve_steady_slab_2d(
        source.repeat(8, 1), dx, (wall, wall), conductivity=conductivity,
        return_x_flux=True, temperature_reference=wall,
    )

    # q' = source. Equal wall temperatures imply integral(q dx) = 0.
    generated = power * (end - start)
    west_out = generated * (1 - (start + end) / 2)
    exact_flux = power * (faces - start).clamp(0, end - start) - west_out
    potential = ((centers - start).clamp_min(0).square()
                 - (centers - end).clamp_min(0).square()) / 2
    exact_field = (west_out * centers - power * potential) / conductivity
    return {
        "columns": columns, "strip_bounds_m": [start, end],
        "west_outward_W_m2": float(-flux[0]), "west_reference_W_m2": west_out,
        "east_outward_W_m2": float(flux[-1]), "east_reference_W_m2": generated - west_out,
        "max_temperature_error_K": float((field - exact_field).abs().max()),
        "max_flux_error_W_m2": float((flux - exact_flux).abs().max()),
        "max_column_balance_error_W_m2": float((flux.diff() - dx * source).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega", type=float, default=0.0, help="angular frequency in rad/s (0 for steady)")
    omega = parser.parse_args().omega
    if not math.isfinite(omega) or omega < 0:
        parser.error("omega must be finite and nonnegative")
    records = []
    for mirrored in (False, True):
        previous = None
        for columns in (16, 32, 64, 128):
            row = measure(columns, mirrored, omega)
            passed = row["max_column_balance_error_W_m2"] < 1e-10
            metrics = (("max_wall_phasor_error_W_m2", "max_relative_wall_error") if omega else
                       ("max_temperature_error_K", "max_flux_error_W_m2"))
            for metric in metrics:
                if previous:
                    order = math.log2(previous[metric] / row[metric])
                    row[metric + "_order"] = order
                    passed &= order > 1.8
            if columns == 128 and omega:
                passed &= row["max_relative_wall_error"] < 0.002 and row["max_wall_phase_error_rad"] < 0.002
            if columns == 128 and not omega:
                passed &= row["max_temperature_error_K"] < 2e-4 and row["max_flux_error_W_m2"] < 2e-4
            row["passed"] = passed
            records.append(row)
            previous = row
    passed = all(row["passed"] for row in records)
    print(json.dumps({
        "passed": passed, "runs": records, "sources": [provenance(__file__), provenance(sph.__file__)],
        "reproduction": "OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_offgrid_strip_heater.py"
                        + (f" --omega {omega}" if omega else ""),
    }, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
