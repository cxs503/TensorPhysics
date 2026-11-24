"""Run with python examples/two_layer_conduction.py in the shared solver environment.

Steady source-free conduction on a 1 m slab of unit cross section. The two
layers are 0.5 m thick. The horizontal walls are insulated. Flux is positive
from left to right. Conductivities are in W/(m K), temperatures in K.
Finite-contact cases insert an areal resistance in m² K/W at x=0.5.
"""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensordem.thermal import contact_chain
from tensorfem import thermal
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph_heat


def slab_fem_model(k, left, right, resistance, films, capacity):
    n = len(k)
    mid = n // 2
    x = torch.linspace(0, 1, n + 1, dtype=torch.float64)
    i = torch.arange(n)
    kwargs = {}
    if resistance:
        # Duplicate the interface at both ends of the Q4 edge. Each endpoint
        # represents half of the unit contact area.
        x = torch.cat((x[:mid+1], x[mid:]))
        i[mid:] += 1
        pairs = torch.tensor([[mid, mid+1], [mid+len(x), mid+len(x)+1]])
        kwargs.update(contact_pairs=pairs, contact_conductance=0.5/resistance)
    nodes = torch.cat([torch.stack((x, torch.full_like(x, y)), dim=1) for y in (0, 1)])
    stride = len(x)
    elements = torch.stack((i, i + 1, i + stride + 1, i + stride), dim=1)
    fixed = torch.tensor([0, stride, stride-1, 2*stride-1])
    ends = fixed.reshape(2, 2)
    if films is None:
        prescribed = x.new_tensor([left, left, right, right])
    else:
        kwargs.update(convection_edges=ends, convection_coefficient=x.new_tensor(films),
                      ambient_temperature=x.new_tensor([left, right]))
        fixed = fixed[:0]
        prescribed = x.new_empty(0)
    model = thermal.ThermalModel(nodes, elements, torch.from_numpy(k), 1., capacity, fixed,
                                 prescribed, element_type="q4", **kwargs)
    return model, ends


def fem(k, left, right, resistance=0., *, films=None):
    n = len(k)
    mid = n // 2
    model, ends = slab_fem_model(k, left, right, resistance, films, 1.)
    stride = len(model.nodes) // 2
    t = thermal.solve_steady_thermal(model)
    wall_data = {}
    if films is None:
        reaction = thermal.assemble_thermal(model).conductivity @ t
        left_flux, right_flux = reaction[ends[0]].sum(), -reaction[ends[1]].sum()
    else:
        wall_t = t[ends].mean(dim=1)
        wall_data = {"left_wall_K": float(wall_t[0]), "right_wall_K": float(wall_t[1])}
        # Fourier flux from the adjacent elements, independently of the film law.
        left_flux = k[0] * n * (wall_t[0] - t[ends[0]+1].mean())
        right_flux = k[-1] * n * (t[ends[1]-1].mean() - wall_t[1])
    face = float(t[[mid, stride + mid]].mean())
    traces = {"interface_K": face}
    if resistance:
        traces = contact_traces(face, float(t[[mid+1, stride+mid+1]].mean()))
    return {**traces, **wall_data,
            "left_flux_W_m2": float(left_flux),
            "right_flux_W_m2": float(right_flux),
            "nodal_temperature_K": t.tolist()}


def fvm(k, left, right, resistance=0., *, films=None):
    n = len(k)
    dx = np.full(n, 1 / n)
    bc = {side: {"type": "dirichlet", "values": value}
          for side, value in (("west", left), ("east", right))}
    if films is not None:
        for side, h in zip(("west", "east"), films):
            bc[side].update(type="robin", film_coefficient=h)
    rx = np.zeros((1, n - 1))
    rx[0, n//2-1] = resistance
    t, info = energy.solve_steady_energy(dx, np.array([1.]), k[None, :],
                                         np.zeros((1, n + 1)), np.zeros((2, n)), bc,
                                         high_order=False, contact_resistance_x=rx)
    i = n // 2
    # Reconstruct the interface trace from the two half-cell resistances.
    face = (k[i-1] * t[0, i-1] + k[i] * t[0, i]) / (k[i-1] + k[i])
    boundary = info["ledger"].boundary
    traces = {"interface_K": float(face)}
    if resistance:
        # Extrapolate from adjacent cells within each material independently.
        a = t[0, i-1] + 0.5*(t[0, i-1] - t[0, i-2])
        b = t[0, i] - 0.5*(t[0, i+1] - t[0, i])
        traces = contact_traces(a, b)
    if films is not None:
        traces.update(left_wall_K=float(1.5*t[0, 0] - .5*t[0, 1]),
                      right_wall_K=float(1.5*t[0, -1] - .5*t[0, -2]))
    return {**traces, "left_flux_W_m2": boundary["west"],
            "right_flux_W_m2": -boundary["east"], "cell_temperature_K": t[0].tolist()}


def sph(k, left, right, resistance=0., *, layered_x=False):
    """Solve the y-invariant restriction of the periodic-y SPH operator."""
    n = len(k)
    dx = 1 / n
    offsets = sph_heat.lattice_offsets(dx, 1.3 * dx)
    material = torch.from_numpy(k)
    faces = None
    if resistance:
        faces = material.new_zeros(n - 1)
        faces[n // 2 - 1] = resistance

    def rhs(t, walls):
        return sph_heat.stencil_rhs_2d(t[None, :], offsets, conductivity=material,
                                       x_wall_temperatures=walls, layered_x=layered_x,
                                       dx=dx, contact_resistance_x=faces)[0]
    # Dirichlet forcing is affine. Applying the homogeneous operator to
    # unit vectors gives the small steady verification system directly.
    eye = torch.eye(n, dtype=material.dtype)
    matrix = torch.stack([rhs(column, (0., 0.)) for column in eye], dim=1)
    forcing = rhs(torch.zeros_like(material), (left, right))
    t = torch.linalg.solve(matrix, -forcing)
    mid = n // 2
    face = (k[mid-1] * t[mid-1] + k[mid] * t[mid]) / (k[mid-1] + k[mid])
    traces = {"interface_K": float(face)}
    if resistance:
        traces = contact_traces(t[mid-1] + 0.5*(t[mid-1] - t[mid-2]),
                                t[mid] - 0.5*(t[mid+1] - t[mid]))
    _, power = sph_heat.stencil_rhs_with_wall_power_2d(
        t[None, :], offsets, dx, (left, right), conductivity=material, layered_x=layered_x,
        contact_resistance_x=faces)
    # One periodic row has face area dx times unit thickness. Flip the right
    # inward power to report both fluxes in the common positive-x convention.
    factor = sph_heat.lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    flux = power / (dx * factor)
    return {**traces, "left_flux_W_m2": float(flux[0]),
            "right_flux_W_m2": -float(flux[1]), "cell_temperature_K": t.tolist()}



def sph_layered(k, left, right, resistance=0.):
    return sph(k, left, right, resistance, layered_x=True)


def contact_traces(left, right):
    return {"left_interface_K": float(left), "right_interface_K": float(right),
            "jump_K": float(left-right)}


def dem(k, left, right, resistance=0., *, films=None):
    """Unit-area contact network with one thermal resistance per slab segment.

    This verifies the network solve with prescribed conductances. Particle
    contact areas and constriction resistances are outside this comparison.
    """
    n = len(k)
    edges = 1 / (n * k)
    if resistance:
        edges = np.insert(edges, n // 2, resistance)
    network_edges = edges if films is None else np.r_[1/films[0], edges, 1/films[1]]
    t = contact_chain.lattice_network_temperatures(
        len(network_edges) + 1, 1, (network_edges[:, None], 1.), left, right)[:, 0]
    wall_data = {}
    if films is not None:
        t = t[1:-1]  # physical nodes between the prescribed ambient reservoirs
        wall_data = {"left_wall_K": float(t[0]), "right_wall_K": float(t[-1])}
    mid = n // 2
    traces = contact_traces(t[mid], t[mid+1]) if resistance else {"interface_K": float(t[mid])}
    return {**traces, **wall_data,
            "left_flux_W_m2": float((t[0] - t[1]) / edges[0]),
            "right_flux_W_m2": float((t[-2] - t[-1]) / edges[-1]),
            "nodal_temperature_K": t.tolist()}


def film_runs():
    """Two external films, unit cross section and an optional contact jump."""
    runs = []
    films = (12., 3.)
    for resistance in (0., .3):
        for left, right in ((310., 290.), (290., 310.)):
            k = np.repeat([10., 1.], 8)
            flux = (left-right) / (1/films[0] + .5/k[0] + resistance + .5/k[-1] + 1/films[1])
            lw, rw = left - flux/films[0], right + flux/films[1]
            ti, to = lw - .5*flux/k[0], rw + .5*flux/k[-1]
            expected = {"left_flux_W_m2": flux, "right_flux_W_m2": flux,
                        "left_wall_K": lw, "right_wall_K": rw}
            expected.update(contact_traces(ti, to) if resistance else {"interface_K": ti})
            for solve in (fem, fvm, dem):
                actual = solve(k, left, right, resistance, films=films)
                errors = {key: abs(actual[key] - value) for key, value in expected.items()}
                runs.append({"solver": solve.__name__, "cells_x": len(k), "conductivities": [10., 1.],
                             "ambient_K": [left, right], "film_coefficients_W_m2_K": films,
                             "contact_resistance_m2_K_W": resistance, "expected": expected,
                             "result": actual, "absolute_errors": errors,
                             "absolute_tolerances": dict.fromkeys(expected, 1e-8),
                             "passed": all(error <= 1e-8 for error in errors.values())})
    return runs


def main():
    runs = []
    for k1, k2 in ((1., 1.), (10., 1.)):
        for left, right in ((310., 290.), (290., 310.)):
            for resistance in (0., 0.3):
                flux = (left - right) / (0.5/k1 + resistance + 0.5/k2)
                interface = left - flux * 0.5/k1
                expected = {"interface_K": interface, "left_flux_W_m2": flux, "right_flux_W_m2": flux}
                if resistance:
                    del expected["interface_K"]
                    expected.update(left_interface_K=interface, right_interface_K=right+flux*0.5/k2,
                                    jump_K=flux*resistance)
                for solver, cells in ((fem, 16), (fvm, 16), (dem, 16), (sph, 16), (sph, 32), (sph, 64),
                                      (sph_layered, 16), (sph_layered, 32), (sph_layered, 64)):
                    if solver is sph and resistance:
                        continue
                    k = np.repeat([k1, k2], cells // 2)
                    result = solver(k, left, right, resistance=resistance) if resistance else solver(k, left, right)
                    errors = {key: abs(result[key] - ref) for key, ref in expected.items()}
                    tolerances = {key: 1e-8 for key in expected}
                    if solver is sph and k1 != k2:
                        tolerances = {"interface_K": 0.05 * abs(left-right),
                                      "left_flux_W_m2": 0.05 * abs(flux),
                                      "right_flux_W_m2": 0.05 * abs(flux)}
                    runs.append({"solver": solver.__name__, "conductivities": [k1, k2],
                                 "wall_K": [left, right], "contact_resistance_m2_K_W": resistance, "expected": expected, "result": result,
                                 "cells_x": cells, "absolute_errors": errors, "absolute_tolerances": tolerances,
                                 "passed": all(errors[key] <= tolerances[key] for key in errors)})
    runs.extend(film_runs())
    print(json.dumps({"sources": [provenance(path) for path in
                                  (__file__, thermal.__file__, energy.__file__, sph_heat.__file__, contact_chain.__file__)],
                      "runs": runs}, indent=2))
    return 0 if all(row["passed"] for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
