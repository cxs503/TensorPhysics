"""Check solved wall fluxes in a layered slab starting at one column per layer."""
import argparse
import json
import math

import torch

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def measure(per_layer, omega, backend, layers=4, exchange=False, row_contacts=False):
    dx = 1 / (layers * per_layer)
    dtype = torch.float64
    k = torch.tensor([1., 2.5, .9, 1.6][:layers], dtype=dtype).repeat_interleave(per_layer)
    capacity = torch.tensor([1.1, .8, 1.6, 1.2][:layers], dtype=dtype).repeat_interleave(per_layer)
    contacts = k.new_zeros(len(k) - 1)
    contacts[per_layer-1::per_layer] = k.new_tensor([.12, .08, .19][:layers-1])
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
    exchange_options = {}
    if exchange:
        loss = .5 + k
        ambient = (.2-.3j) + (.7+.1j)*wave[:, None]*(1-r)
        source += loss*(exact-ambient)
        exchange_options = dict(heat_loss_coefficient=loss, ambient_temperature=ambient)
    field, cuts = sph.solve_harmonic_layered_slab(source, dx, walls, omega,
        conductivity=k, volumetric_heat_capacity=capacity, contact_resistance_x=contacts,
        backend=backend, return_x_flux="by_row" if row_contacts else True, **exchange_options)
    if row_contacts:
        row_cuts = cuts
        cuts = row_cuts.mean(0)
    means = torch.stack((cuts[0], -cuts[-1]))
    flux = sph.recover_dirichlet_wall_flux_2d(field, dx, k, means,
        contact_resistance_x=contacts, x_wall_temperatures=walls)
    sequence_flux = sph.recover_dirichlet_wall_flux_2d(field, dx, k, means.tolist(),
        contact_resistance_x=contacts, x_wall_temperatures=walls)
    target = torch.stack((-slope, slope))
    sampled = sph.recover_dirichlet_wall_flux_2d(exact, dx, k, target.mean(1),
        contact_resistance_x=contacts, x_wall_temperatures=walls)
    interface = {}
    if exchange:
        idx = torch.nonzero(contacts).flatten()
        left, right = sph.layered_slab_contact_temperatures(field.mean(0), cuts, dx, k)
        left_exact = .7 + slope.mean()*(r[idx]+bulk[idx]/2)
        right_exact = left_exact + slope.mean()*contacts[idx]
        interface = dict(interface_flux_error_W_m2=float(abs(cuts[idx+1]+slope.mean()).max()),
                         interface_mean_trace_error_K=float(torch.maximum(abs(left[idx]-left_exact),
                                                                          abs(right[idx]-right_exact)).max()))
    if row_contacts:
        idx = torch.nonzero(contacts).flatten()
        left, right = sph.layered_slab_contact_temperatures(field, row_cuts, dx, k)
        left_exact = .7 + slope[:, None]*(r[idx]+bulk[idx]/2)
        jump = -slope[:, None]*contacts[idx]
        right_exact = left_exact - jump
        interface.update(row_contact_left_error_K=float(abs(left[:, idx]-left_exact).max()),
                         row_contact_right_error_K=float(abs(right[:, idx]-right_exact).max()),
                         row_jump_error_K=float(abs(left[:, idx]-right[:, idx]-jump).max()),
                         row_contact_flux_error_W_m2=float(abs(row_cuts[:, idx+1]+slope[:, None]).max()))
    return dict(**interface, sequence_flux_difference_W_m2=float((sequence_flux-flux).abs().max()),
                sampled_wall_flux_error_W_m2=float((sampled-target).abs().max()), columns=len(k), rows=len(y), omega_rad_s=omega, backend=backend,
                temperature_error_K=float((field - exact).abs().max()),
                wall_flux_error_W_m2=float((flux - target).abs().max()),
                mean_cut_error_W_m2=float((cuts + slope.mean()).abs().max()),
                recovered_mean_residual_W_m2=float((flux.mean(1) - means).abs().max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify fixed-wall flux on layered slabs of unit thickness")
    parser.add_argument("--layers", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--exchange", action="store_true", help="apply a spatially varying ambient temperature phasor")
    parser.add_argument("--row-contacts", action="store_true", help="check both contact traces and flux on each periodic row")
    args = parser.parse_args()
    layers = args.layers
    rows = []
    for omega in (0., 1.3):
        for backend in ("torch", "banded"):
            previous = None
            for per_layer in (1, 2, 4, 8):
                row = measure(per_layer, omega, backend, layers, args.exchange, args.row_contacts)
                errors = [row["temperature_error_K"], row["wall_flux_error_W_m2"]]
                row["orders"] = [] if previous is None else [math.log2(a / b) for a, b in zip(previous, errors)]
                # In thin slabs the recovery stencil changes degree and which
                # contacts it crosses. Report each order and gate the final refinement.
                check_orders = layers == 4 or per_layer == 8
                row["passed"] = (errors[0] < (.012 if layers == 4 else .02) and errors[1] < .09
                    and row["sequence_flux_difference_W_m2"] == 0.
                    and row["sampled_wall_flux_error_W_m2"] < 1e-11
                    and row["mean_cut_error_W_m2"] < 1e-10
                    and row["recovered_mean_residual_W_m2"] < 1e-12
                    and (not check_orders or all(order > .6 for order in row["orders"]))
                    and (per_layer < 8 or (errors[0] < .003 and errors[1] < .01)))
                if args.exchange:
                    row["passed"] &= max(row["interface_flux_error_W_m2"], row["interface_mean_trace_error_K"]) < 1e-10
                if args.row_contacts:
                    trace_limit, flux_limit = (.003, .01) if per_layer == 8 else (.02, .09)
                    row["passed"] &= (max(row["row_contact_left_error_K"], row["row_contact_right_error_K"],
                                          row["row_jump_error_K"]) < trace_limit
                                      and row["row_contact_flux_error_W_m2"] < flux_limit)
                rows.append(row)
                previous = errors
    print(json.dumps(dict(reference="T=.7+[(.4-.2i)+(.65+.3i) exp(i beta y)] r, r=integrated bulk and upstream contact resistance",
        layers=layers, reproduction=f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_thin_contact_wall_flux.py --layers {layers}" + (" --exchange" if args.exchange else "") + (" --row-contacts" if args.row_contacts else ""),
        order_metrics=["temperature_error_K", "wall_flux_error_W_m2"], runs=rows,
        sources=[provenance(p) for p in (__file__, sph.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
