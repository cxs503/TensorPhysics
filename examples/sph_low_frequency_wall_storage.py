"""Two oscillating flux walls with volumetric heating in a unit slab."""
import json
import math

import torch
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(columns, omega):
    dx = 1 / columns
    faces = torch.arange(columns + 1, dtype=torch.float64) * dx
    x = (faces[:-1] + faces[1:]) / 2
    west, east = 1j, 2j
    mode = 0.4 - 0.2j
    uniform_rate = 1 + west + east
    # T = rate/(i omega) + g(x). The quadratic in g prescribes both
    # wall derivatives. The cosine adds an interior heat flux.
    shape = (west + east) * x.square() / 2 - west * x + mode * torch.cos(torch.pi * x)
    storage = uniform_rate + 1j * omega * shape
    source = storage - (west + east) + math.pi**2 * mode * torch.cos(torch.pi * x)
    reference_flux = west - (west + east) * faces + math.pi * mode * torch.sin(torch.pi * faces)
    field, flux = sph.solve_harmonic_slab_2d(
        source.repeat(8, 1), dx, (None, None), omega,
        x_wall_inward_flux=(west, east), return_x_flux=True,
    )
    computed_storage = 1j * omega * field.mean(0)
    phase_error = torch.angle(computed_storage / storage)
    return {
        "columns": columns, "omega_rad_s": omega,
        "max_face_flux_error_W_m2": float(abs(flux - reference_flux).max()),
        "max_storage_rate_error_W_m3": float(abs(computed_storage - storage).max()),
        "max_temperature_phase_error_rad": float(abs(phase_error).max()),
        "max_column_balance_error_W_m2": float(abs(flux.diff() + dx*(computed_storage - source)).max()),
    }


def main():
    runs = []
    for omega in (1.0, 1e-8, 1e-14):
        previous = None
        for columns in (16, 32, 64, 128):
            row = measure(columns, omega)
            error = row["max_face_flux_error_W_m2"]
            passed = row["max_column_balance_error_W_m2"] < 1e-11
            if previous is not None:
                row["flux_error_order"] = math.log2(previous / error)
                passed &= row["flux_error_order"] > 1.9
            if columns == 128:
                passed &= error < 1e-4 and row["max_storage_rate_error_W_m3"] < 1e-4
                passed &= row["max_temperature_phase_error_rad"] < 1e-4
            row["passed"] = passed
            runs.append(row)
            previous = error
    passed = all(row["passed"] for row in runs)
    print(json.dumps({
        "passed": passed, "runs": runs, "sources": [provenance(__file__), provenance(sph.__file__)],
        "reproduction": "OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_low_frequency_wall_storage.py",
    }, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
