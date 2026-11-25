"""Local temperature response to traveling heat-flux waves on opposite slab walls."""
import cmath
import json
import math

import torch
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(n, omega):
    dx = 1 / n
    centers = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    conductivity, capacity = 1.6, 2.4
    source = torch.ones((n, n), dtype=torch.float64)
    wall_loads = []
    exact = torch.zeros_like(source, dtype=torch.complex128)
    wall_responses = []
    for mode, amplitude, side in ((1, 1.2j, 0), (-2, 0.6 - 0.3j, 1)):
        lateral = torch.exp((2j * math.pi * mode) * centers)
        wall_loads.append(amplitude * lateral)
        # The cosine hyperbolic has zero derivative at the opposite wall.
        # Normalize the driven wall's inward Fourier flux to its imposed amplitude.
        gamma = cmath.sqrt((2 * math.pi * mode)**2 + 1j * omega * capacity / conductivity)
        distance = 1 - centers if side == 0 else centers
        response = amplitude * torch.cosh(gamma * distance) / (conductivity * gamma * cmath.sinh(gamma))
        exact += lateral[:, None] * response
        wall_responses.append((lateral, response, 0 if side == 0 else -1))

    profile, cuts, mean = sph.solve_harmonic_slab_2d(
        source, dx, (None, None), omega, volumetric_heat_capacity=capacity,
        conductivity=conductivity, x_wall_inward_flux=wall_loads, return_x_flux=True, separate_mean=True,
    )
    # Project onto each imposed wave at the particle column nearest its wall.
    # This gives a phase even where the superposed temperature nearly vanishes.
    phase_errors = []
    for lateral, response, column in wall_responses:
        coefficient = (profile[:, column] * lateral.conj()).mean()
        phase_errors.append(float(torch.angle(coefficient / response[column]).abs()))
    return {
        "particles_per_axis": n, "omega_rad_s": omega,
        "max_local_temperature_error_K": float((profile - exact).abs().max()),
        "max_near_wall_mode_phase_error_rad": max(phase_errors),
        "mean_storage_rate_error_W_m3": float(abs(1j * omega * capacity * mean - 1)),
        "max_section_flux_error_W_m2": float(cuts.abs().max()),
    }


def main():
    runs = []
    for omega in (20., 1., 1e-18):
        previous = None
        for n in (16, 32, 64, 128, 256):
            row = measure(n, omega)
            error = row["max_local_temperature_error_K"]
            passed = max(row["mean_storage_rate_error_W_m3"], row["max_section_flux_error_W_m2"]) < 1e-10
            if previous is not None:
                row["profile_error_order"] = math.log2(previous / error)
                passed &= row["profile_error_order"] > 1.8
            if n == 256:
                passed &= error < 2e-4 and row["max_near_wall_mode_phase_error_rad"] < 1e-3
            row["passed"] = passed
            runs.append(row)
            previous = error
    passed = all(row["passed"] for row in runs)
    print(json.dumps({
        "passed": passed, "runs": runs, "sources": [provenance(__file__), provenance(sph.__file__)],
        "reproduction": "OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_transverse_flux_wall_storage.py",
    }, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
