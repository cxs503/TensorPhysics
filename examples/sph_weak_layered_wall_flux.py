"""Verify weak steady section fluxes through two layers and a resistive interface."""
import json

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(n, base, drop, backend, sampled):
    dx = 1/n
    x = (torch.arange(n, dtype=torch.float64)+0.5)*dx
    k = torch.where(x < 0.5, 1., 4.).double()
    contact = torch.zeros(n-1, dtype=x.dtype)
    contact[n//2-1] = 0.25
    source = torch.zeros((5, n) if sampled else (n,), dtype=x.dtype)
    reference_flux = -drop/(0.5 + 0.25 + 0.5/4)
    resistance = torch.minimum(x, x.new_tensor(0.5)) + (x > 0.5)*0.25 + (x-0.5).clamp_min(0)/4
    field, flux = heat.solve_harmonic_layered_slab(
        source, dx, (base, base+drop), 0., conductivity=k, volumetric_heat_capacity=1.,
        contact_resistance_x=contact, backend=backend, return_x_flux=True,
    )
    return dict(columns=n, west_K=base, drop_K=drop, backend=backend, sampled_rows=5 if sampled else 1,
                max_flux_relative_error=float(abs(flux/reference_flux-1).max()),
                max_temperature_error_K=float(abs(field-(base-reference_flux*resistance)).max()))


if __name__ == "__main__":
    runs = []
    for backend in ("torch", "banded"):
        for sampled in (False, True):
            for base, drop in ((0., 2**-44), (305., 2**-44), (305., -2**-44)):
                for n in (16, 64):
                    row = measure(n, base, drop, backend, sampled)
                    row["passed"] = row["max_flux_relative_error"] < 1e-8 and row["max_temperature_error_K"] < 1e-12
                    runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(heat.__file__)],
                          reproduction="PYTHONPATH=../TensorSPH/src python examples/sph_weak_layered_wall_flux.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
