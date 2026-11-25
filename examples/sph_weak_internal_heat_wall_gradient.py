"""Recover local wall flux from a sinusoidal heat source between isothermal walls."""
import json

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(n, amplitude):
    k, base = 1.7, 305.
    x = (torch.arange(n, dtype=torch.float64)+0.5)/n
    lateral = torch.cos(2*torch.pi*x + 0.37)
    mode = lateral[:, None]*torch.sin(torch.pi*x)
    # Temperature above the wall is A*mode; -k*laplacian(T) gives the source.
    source = 5*k*torch.pi**2*amplitude*mode
    options = dict(conductivity=k, return_x_flux=True)
    absolute, cuts = heat.solve_steady_slab_2d(source, 1/n, (base, base), **options)
    excess, relative_cuts = heat.solve_steady_slab_2d(source, 1/n, (base, base), temperature_reference=base, **options)
    means = torch.stack((cuts[0], -cuts[-1]))
    scale = k*torch.pi*amplitude
    # Both inward fluxes have the same sign because dT/dx changes sign.
    exact = -scale*lateral
    def error(field):
        flux = heat.recover_dirichlet_wall_flux_2d(field, 1/n, k, means)
        return float(abs(flux-exact).max()/scale)
    return dict(columns=n, amplitude_K=amplitude, absolute_fit_error_over_scale=error(absolute),
                relative_fit_error_over_scale=error(excess), field_error_over_amplitude=float(abs(excess/amplitude-mode).max()),
                section_flux_change_W_m2=float(abs(cuts-relative_cuts).max()))


if __name__ == "__main__":
    runs = []
    for amplitude in (2**-22, 2**-40, 2**-44):
        for n in (16, 32, 64, 128):
            row = measure(n, amplitude)
            row["passed"] = row["section_flux_change_W_m2"] == 0 and (n < 128 or
                             max(row["relative_fit_error_over_scale"], row["field_error_over_amplitude"]) < 0.001)
            runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_weak_internal_heat_wall_gradient.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
