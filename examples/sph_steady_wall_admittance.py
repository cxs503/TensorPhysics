"""Verify the symmetric and antisymmetric wall responses of a periodic SPH slab."""
import json
import math

import torch

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def measure(nx, mode, parity):
    # The slab is 1 m long, with a transverse period of 2 m.
    dx, k = 1 / nx, 1.7
    beta = math.pi * mode
    x = (torch.arange(nx, dtype=torch.float64) + .5) * dx
    y = (torch.arange(2 * nx, dtype=torch.float64) + .5) * dx
    trace = torch.cos(beta * y)
    walls = (trace, parity * trace)
    field = sph.solve_steady_slab_2d(torch.zeros((len(y), nx), dtype=x.dtype), dx,
                                     walls, conductivity=k)
    # Equal and opposite boundary traces diagonalize the two-wall response.
    if parity == 1:
        profile = torch.cosh(beta * (x - .5)) / math.cosh(beta / 2)
        admittance = k * beta * math.tanh(beta / 2)
    else:
        profile = torch.sinh(beta * (.5 - x)) / math.sinh(beta / 2)
        admittance = k * beta / math.tanh(beta / 2)
    _, powers = sph.stencil_rhs_with_wall_power_2d(field, sph.lattice_offsets(dx, 1.3 * dx),
        dx, walls, conductivity=k)
    flux = sph.recover_dirichlet_wall_flux_2d(field, dx, k,
        powers / (2 * sph.lattice_diffusion_factor_2d(1.3, dx, mode="moment")))
    exact_flux = admittance * torch.stack(walls)
    return dict(nx=nx, ny=len(y), mode=mode, parity=parity, exact_admittance_W_m2K=admittance,
                temperature_error_K=float((field - trace[:, None] * profile).abs().max()),
                flux_relative_error=float((flux - exact_flux).abs().max() / admittance))


if __name__ == "__main__":
    runs = []
    for mode in (1, 2):
        for parity in (1, -1):
            previous = None
            for nx in (16, 32, 64, 128):
                row = measure(nx, mode, parity)
                metrics = (row["temperature_error_K"], row["flux_relative_error"])
                orders = [] if previous is None else [math.log2(a / b) for a, b in zip(previous, metrics)]
                row.update(orders=orders, passed=all(order > 1.5 for order in orders)
                           and (nx < 128 or metrics[1] < .003))
                runs.append(row)
                previous = metrics
    print(json.dumps(dict(reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_steady_wall_admittance.py",
        reference="Steady Laplace equation, row-centre Dirichlet data, admittance k beta tanh(beta L/2) or k beta coth(beta L/2)",
        order_metrics=["temperature_error_K", "flux_relative_error"], runs=runs,
        sources=[provenance(path) for path in (__file__, sph.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in runs) else 1)
