"""Verify FEM and SPH against the same unequal-capacity, finite-contact mode."""
import json
import math

import torch

from tensorfem import thermal
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


def particles(n):
    mid, dx = n // 2, 1/n
    x = (torch.arange(n, dtype=torch.float64) + .5) * dx
    initial = mode(x[:mid], x[mid:])
    k = x.new_tensor([1., 4.]).repeat_interleave(mid)
    cp = x.new_tensor([1., CAPACITY]).repeat_interleave(mid)
    faces = x.new_zeros(n-1)
    faces[mid-1] = RESISTANCE
    offsets = sph.lattice_offsets(dx, 1.3 * dx)
    factor = sph.lattice_diffusion_factor_2d(1.3, dx, mode="moment")

    def rhs(field):
        return sph.stencil_rhs_2d(field, offsets, conductivity=k, specific_heat=cp,
                                  x_wall_temperatures=(0., 0.), layered_x=True, dx=dx,
                                  contact_resistance_x=faces) / factor

    field = initial[None, :]
    steps = math.ceil(END * (4/CAPACITY) / (.16 * dx**2))
    dt = END / steps
    for _ in range(steps):
        field = field + dt * rhs(field + .5 * dt * rhs(field))
    final = field[0]
    left = final[mid-1] + .5 * (final[mid-1] - final[mid-2])
    right = final[mid] - .5 * (final[mid+1] - final[mid])
    return final, initial, left-right, dx * (cp * (final - initial)).sum()


def main():
    runs = []
    for name, solve in (("FEM", fem), ("SPH", particles)):
        previous = None
        for n in (16, 32, 64):
            final, initial, jump, heat = solve(n)
            errors = {"temperature_K": float((final - DECAY * initial).abs().max()),
                      "jump_K": abs(float(jump) - DECAY * JUMP),
                      "heat_change_J_m2": abs(float(heat) - (DECAY - 1) * INITIAL_HEAT)}
            passed = all(math.isfinite(v) for v in errors.values())
            if previous:
                passed &= all(v < previous[k]/3 for k, v in errors.items())
            if n == 64:
                passed &= all(errors[k] < tol for k, tol in
                              (("temperature_K", .001), ("jump_K", .003), ("heat_change_J_m2", 1e-5)))
            runs.append({"solver": name, "cells": n, "errors": errors, "passed": passed})
            previous = errors
    print(json.dumps({"sources": [provenance(f) for f in (__file__, thermal.__file__, sph.__file__)],
                      "time_s": END, "analytic_jump_K": DECAY * JUMP,
                      "analytic_heat_change_J_m2": (DECAY - 1) * INITIAL_HEAT, "runs": runs}, indent=2))
    return 0 if all(run["passed"] for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
