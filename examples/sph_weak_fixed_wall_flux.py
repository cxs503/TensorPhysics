"""Check every SPH section flux against Fourier's law for a homogeneous slab."""
import json

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(n, base, drop):
    k = 1.7
    source = torch.zeros((6, n), dtype=torch.float64)
    field, flux = heat.solve_steady_slab_2d(source, 1/n, (base, base+drop),
                                            conductivity=k, return_x_flux=True)
    x = (torch.arange(n, dtype=source.dtype)+0.5)/n
    exact = base + drop*x
    current = -k*drop
    return dict(columns=n, west_K=base, east_minus_west_K=drop, reference_flux_W_m2=current,
                max_section_relative_error=float((flux/current-1).abs().max()),
                max_temperature_error_K=float((field-exact).abs().max()))


if __name__ == "__main__":
    runs = []
    for base in (0., 305.):
        for drop in (-2**-44, 2**-44, 2**-24):
            for n in (8, 32, 128):
                row = measure(n, base, drop)
                row["passed"] = row["max_section_relative_error"] < 1e-9 and row["max_temperature_error_K"] < 1e-12
                runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(heat.__file__)],
                          reproduction="PYTHONPATH=../TensorSPH/src python examples/sph_weak_fixed_wall_flux.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
