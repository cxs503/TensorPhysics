"""Check solved wall fluxes in a four-layer slab starting at one column per layer."""
import json
import math

import torch

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def measure(per_layer, omega, backend):
    dx = .25 / per_layer
    dtype = torch.float64
    k = torch.tensor([1., 2.5, .9, 1.6], dtype=dtype).repeat_interleave(per_layer)
    capacity = torch.tensor([1.1, .8, 1.6, 1.2], dtype=dtype).repeat_interleave(per_layer)
    contacts = k.new_zeros(len(k) - 1)
    contacts[per_layer-1::per_layer] = k.new_tensor([.12, .08, .19])
    # Conductive distance includes the jumps at upstream contact faces.
    bulk = dx / k
    r = bulk.cumsum(0) - bulk / 2 + torch.cat((k.new_zeros(1), contacts.cumsum(0)))
    total = bulk.sum() + contacts.sum()
    period = 8.
    beta = 2 * math.pi / period
    y = (torch.arange(round(period / dx), dtype=dtype) + .5) * dx
    wave = (.65 + .3j) * torch.exp(1j * beta * y)
    slope = (.4 - .2j) + wave
    exact = .7 + slope[:, None] * r
    walls = (.7, .7 + slope * total)
    # T = .7 + slope(y)*r(x). The longitudinal flux is constant through all contacts.
    source = 1j * omega * capacity * exact + k * beta**2 * wave[:, None] * r
    field, cuts = sph.solve_harmonic_layered_slab(source, dx, walls, omega,
        conductivity=k, volumetric_heat_capacity=capacity, contact_resistance_x=contacts,
        backend=backend, return_x_flux=True)
    means = torch.stack((cuts[0], -cuts[-1]))
    flux = sph.recover_dirichlet_wall_flux_2d(field, dx, k, means,
        contact_resistance_x=contacts, x_wall_temperatures=walls)
    target = torch.stack((-slope, slope))
    return dict(columns=len(k), rows=len(y), omega_rad_s=omega, backend=backend,
                temperature_error_K=float((field - exact).abs().max()),
                wall_flux_error_W_m2=float((flux - target).abs().max()),
                mean_cut_error_W_m2=float((cuts + slope.mean()).abs().max()),
                recovered_mean_residual_W_m2=float((flux.mean(1) - means).abs().max()))


if __name__ == "__main__":
    rows = []
    for omega in (0., 1.3):
        for backend in ("torch", "banded"):
            previous = None
            for per_layer in (1, 2, 4, 8):
                row = measure(per_layer, omega, backend)
                errors = [row["temperature_error_K"], row["wall_flux_error_W_m2"]]
                row["orders"] = [] if previous is None else [math.log2(a / b) for a, b in zip(previous, errors)]
                row["passed"] = (errors[0] < .012 and errors[1] < .09
                    and row["mean_cut_error_W_m2"] < 1e-10
                    and row["recovered_mean_residual_W_m2"] < 1e-12
                    and all(order > .6 for order in row["orders"])
                    and (per_layer < 8 or (errors[0] < .003 and errors[1] < .01)))
                rows.append(row)
                previous = errors
    print(json.dumps(dict(reference="T=.7+[(.4-.2i)+(.65+.3i) exp(i beta y)] r, r=integrated bulk and upstream contact resistance",
        reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_thin_contact_wall_flux.py",
        order_metrics=["temperature_error_K", "wall_flux_error_W_m2"], runs=rows,
        sources=[provenance(p) for p in (__file__, sph.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
