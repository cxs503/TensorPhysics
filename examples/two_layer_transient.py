"""Verify FEM, FVM, DEM and SPH against the same unequal-capacity, finite-contact mode."""
import argparse
import json
import math

import torch

from tensordem.thermal import contact_chain
from tensorfem import thermal
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph
from conduction_provenance import provenance

A = 1.5 * math.pi
B = 4 * math.pi / 3
CAPACITY = 4 * (B / A)**2
AMPLITUDE = A * math.cos(A/2) / (4 * B * math.cos(B/2))
JUMP = math.sin(A/2) + AMPLITUDE * math.sin(B/2)
RESISTANCE = JUMP / (-A * math.cos(A/2))
END = .015
DECAY = math.exp(-A**2 * END)
INITIAL_HEAT = (1 - math.cos(A/2))/A - CAPACITY * AMPLITUDE * (1 - math.cos(B/2))/B


def mode(x_left, x_right):
    return torch.cat((torch.sin(A*x_left), -AMPLITUDE*torch.sin(B*(1-x_right))))


def fem(n):
    mid = n // 2
    x = torch.linspace(0, 1, n + 1, dtype=torch.float64)
    nodes = torch.cat((x[:mid+1], x[mid:]))
    idx = torch.cat((torch.arange(mid), torch.arange(mid+1, n+1)))
    model = thermal.ThermalModel(
        nodes[:, None], torch.stack((idx, idx+1), dim=1),
        x.new_tensor([1., 4.]).repeat_interleave(mid), 1.,
        x.new_tensor([1., CAPACITY]).repeat_interleave(mid),
        torch.tensor([0, n+1]), x.new_zeros(2),
        contact_pairs=torch.tensor([[mid, mid+1]]), contact_conductance=1/RESISTANCE)
    initial = mode(x[:mid+1], x[mid:])
    time = torch.linspace(0, END, n+1, dtype=x.dtype)
    final = thermal.solve_transient_thermal(model, time, initial, theta=.5).temperature[-1]
    delta = final - initial
    heat = torch.trapezoid(delta[:mid+1], x[:mid+1]) + CAPACITY * torch.trapezoid(delta[mid+1:], x[mid:])
    return final, initial, final[mid] - final[mid+1], heat


def dem(n):
    # Unit area. Each segment contributes half its capacity to each end;
    # the zero-thickness contact links two coincident, distinct nodes.
    mid = n // 2
    x = torch.linspace(0, 1, n+1, dtype=torch.float64)
    initial = mode(x[:mid+1], x[mid:])
    resistance = x.new_full((n+1,), 1/n)
    resistance[mid] = RESISTANCE
    resistance[mid+1:] /= 4
    capacity = x.new_full((n+2,), 1/n)
    capacity[mid+1:] *= CAPACITY
    capacity[[0, mid, mid+1, n+1]] *= .5
    time = torch.linspace(0, END, n+1, dtype=x.dtype)
    history = contact_chain.transient_contact_temperatures(
        resistance, capacity, time, initial, t_left=0., t_right=0.)
    final = torch.from_numpy(history[-1])
    return final, initial, final[mid]-final[mid+1], (capacity * (final-initial)).sum()


def cell_data(n):
    mid, dx = n // 2, 1/n
    x = (torch.arange(n, dtype=torch.float64) + .5) * dx
    initial = mode(x[:mid], x[mid:])
    cp = x.new_tensor([1., CAPACITY]).repeat_interleave(mid)
    k = x.new_tensor([1., 4.]).repeat_interleave(mid)
    faces = x.new_zeros(1, n-1)
    faces[0, mid-1] = RESISTANCE
    return dx, initial, k, cp, faces


def cell_result(final, initial, cp, dx):
    mid = len(final) // 2
    left = final[mid-1] + .5 * (final[mid-1] - final[mid-2])
    right = final[mid] - .5 * (final[mid+1] - final[mid])
    return final, initial, left-right, dx * (cp * (final - initial)).sum()


def fvm(n):
    dx, initial, k, cp, faces = cell_data(n)
    bc = {s: {"type": "dirichlet", "values": 0.} for s in ("west", "east")}
    time = torch.linspace(0, END, n+1, dtype=initial.dtype)
    result = energy.solve_transient_conduction(
        initial.new_full((n,), dx), [1.], k, cp, bc, time, initial, contact_resistance_x=faces)
    final = torch.from_numpy(result[-1, 0])
    return cell_result(final, initial, cp, dx)


def particles(n):
    dx, initial, k, cp, faces = cell_data(n)
    offsets = sph.lattice_offsets(dx, 1.3 * dx)
    factor = sph.lattice_diffusion_factor_2d(1.3, dx, mode="moment")

    def rhs(field):
        return sph.stencil_rhs_2d(field, offsets, conductivity=k, specific_heat=cp,
                                  x_wall_temperatures=(0., 0.), layered_x=True, dx=dx,
                                  contact_resistance_x=faces[0]) / factor

    field = initial[None, :]
    steps = math.ceil(END * (4/CAPACITY) / (.16 * dx**2))
    dt = END / steps
    for _ in range(steps):
        field = field + dt * rhs(field + .5 * dt * rhs(field))
    final = field[0]
    return cell_result(final, initial, cp, dx)


def interface_fluxes(field, n, *, node_centred):
    """Second-order, one-sided Fourier fluxes at the material interface."""
    # Distances into each rod from the interface, in units of dx, are
    # (0,1,2) for nodal data and (1/2,3/2,5/2) for cell-centred data.
    split = n // 2 + int(node_centred)
    weights = field.new_tensor([-1.5, 2., -.5] if node_centred else [-2., 3., -1.])
    grad_left = n * torch.dot(weights, field[split-3:split].flip((0,)))
    grad_right = n * torch.dot(weights, field[split:split+3])
    return float(grad_left), float(-4 * grad_right)


def main(extended=False):
    runs = []
    exact_flux = -A * math.cos(A/2) * DECAY
    for name, solve in (("FEM", fem), ("FVM", fvm), ("DEM", dem), ("SPH", particles)):
        previous = None
        for n in ((16, 32, 64, 128) if extended else (16, 32, 64)):
            final, initial, jump, heat = solve(n)
            errors = {"temperature_K": float((final - DECAY * initial).abs().max()),
                      "jump_K": abs(float(jump) - DECAY * JUMP),
                      "heat_change_J_m2": abs(float(heat) - (DECAY - 1) * INITIAL_HEAT)}
            flux_left, flux_right = interface_fluxes(final, n, node_centred=name in ("FEM", "DEM"))
            passed = all(math.isfinite(v) for v in errors.values())
            if previous:
                passed &= all(v < previous[k]/3 for k, v in errors.items())
            if n >= 64:
                passed &= all(errors[k] < tol for k, tol in
                              (("temperature_K", .001), ("jump_K", .003), ("heat_change_J_m2", 1e-5)))
            runs.append({"solver": name, "cells": n, "errors": errors, "passed": passed,
                         "interface_flux_audit": {
                             "left_W_m2": flux_left, "right_W_m2": flux_right,
                             "left_relative_error": abs(flux_left / exact_flux - 1),
                             "right_relative_error": abs(flux_right / exact_flux - 1),
                         }})
            previous = errors
    print(json.dumps({"sources": [provenance(f) for f in (__file__, thermal.__file__, energy.__file__, contact_chain.__file__, sph.__file__)],
                      "time_s": END, "analytic_interface_flux_W_m2": exact_flux, "analytic_jump_K": DECAY * JUMP,
                      "analytic_heat_change_J_m2": (DECAY - 1) * INITIAL_HEAT, "runs": runs}, indent=2))
    return 0 if all(run["passed"] for run in runs) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true", help="also run the 128-cell refinement")
    raise SystemExit(main(parser.parse_args().extended))
