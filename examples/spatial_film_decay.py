"""Compare FEM, SPH and FVM Robin boundaries in a smooth cooling slab."""
import argparse
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal, thermal_analytic
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


def gradient_flux(field, dx, *, nodal=False):
    """Reconstruct inward Fourier flux at both walls from three x columns."""
    if nodal:
        # Evaluate at the same tangential midpoints as the wall-flux comparison.
        field = .5*(field[1:] + field[:-1])
    weights = np.array([-1.5, 2., -.5] if nodal else [-2., 3., -1.])
    return -np.stack((field[:, :3] @ weights, field[:, -3:][:, ::-1] @ weights))/dx


def measure(n):
    dx, end = 1/n, .02
    x = (np.arange(n)+.5)*dx
    a, b = 2., 2*math.pi
    film = a*math.tan(a/2)
    xmode = np.cos(a*(x-.5))
    # Both terms satisfy the same homogeneous x film law. The cosine in y
    # satisfies periodicity and zero flux on y=0,1.
    def amplitude(time):
        return np.exp(-a*a*time) + .5*np.cos(b*x)*np.exp(-(a*a+b*b)*time)

    initial = amplitude(0.)[:, None]*xmode
    exact = amplitude(end)[:, None]*xmode
    exact_wall = amplitude(end)*math.cos(a/2)
    exact_heat = 2*math.sin(a/2)/a*math.expm1(-a*a*end)
    bc = {side: {"type": "robin", "values": 0., "film_coefficient": film} for side in ("west","east")}
    widths = np.full(n, dx)
    fv = energy.solve_transient_conduction(widths, widths, 1., 1., bc, np.linspace(0,end,n+1), initial)[-1]
    wall = fv[:, [0,-1]].T / (1 + film*dx/2)
    fields = [("FVM", fv, wall, -film*wall)]

    t = torch.from_numpy(initial)
    rhs = sph.prepare_slab_film_rhs(t, sph.lattice_offsets(dx, 1.3*dx), dx,
                                    (film, film), local_films=True)
    steps = math.ceil(end/(.16*dx**2))
    dt = end/steps
    for _ in range(steps):
        pred, _, _ = rhs(t, (0.,0.))
        rate, _, _ = rhs(t+.5*dt*pred, (0.,0.))
        t = t + dt*rate
    _, power, wall = rhs(t, (0.,0.))
    fields.append(("SPH", t.numpy(), wall.numpy(), power.numpy()/dx))
    runs = [{"solver": name, "cells_per_side": n, "errors": {
        "temperature_K": float(np.max(abs(field-exact))),
        "wall_K": float(np.max(abs(surface-exact_wall))),
        "inward_flux_W_m2": float(np.max(abs(inward+film*exact_wall))),
        "heat_change_J": float(abs((field-initial).mean()-exact_heat))}}
        for name, field, surface, inward in fields]

    mesh = thermal_analytic.structured_quad_mesh(n, n)
    xy = mesh.nodes
    ids = torch.arange(len(xy)).reshape(n+1,n+1)
    edges = torch.cat([torch.stack((ids[:-1,j], ids[1:,j]), dim=1) for j in (0,-1)])
    model = thermal.ThermalModel(xy, mesh.elements, 1., 1., 1., ids.new_empty(0),
                                 xy.new_empty(0), element_type="q4", convection_edges=edges,
                                 convection_coefficient=film, ambient_temperature=0.)
    def nodal_exact(time):
        return torch.cos(a*(xy[:,0]-.5)) * (math.exp(-a*a*time)
                + .5*torch.cos(b*xy[:,1])*math.exp(-(a*a+b*b)*time))

    times = torch.linspace(0., end, n+1, dtype=xy.dtype)
    solution = thermal.solve_transient_thermal(model, times, nodal_exact(0.), theta=.5, sparse=True)
    history = solution.temperature
    # Edge means integrate linear temperatures exactly along each edge.
    edge_T = history[:, edges].mean(dim=2)
    rate_in = -film * dx * edge_T.sum(dim=1)
    heat = float((solution.matrices.capacity @ (history[-1]-history[0])).sum())
    balance = abs(heat-float(torch.trapezoid(rate_in, times)))
    wall_error = float(np.abs(edge_T[-1].reshape(2,n).numpy()-exact_wall).max())
    runs.append({"solver": "FEM", "cells_per_side": n, "discrete_balance_J": balance,
                 "errors": {"temperature_K": float((history[-1]-nodal_exact(end)).abs().max()),
                            "wall_K": wall_error, "inward_flux_W_m2": film*wall_error,
                            "heat_change_J": abs(heat-exact_heat)}})
    all_fields = [fv, t.numpy(), history[-1].reshape(n+1,n+1).numpy()]
    for row, field in zip(runs, all_fields):
        inward = gradient_flux(field, dx, nodal=row["solver"] == "FEM")
        row["gradient_flux_max_error_W_m2"] = float(np.abs(inward + film*exact_wall).max())
    return runs


def main(extended=False):
    previous = {}
    rows = []
    for n in ((16,32,64,128) if extended else (16,32,64)):
        for row in measure(n):
            name, errors = row["solver"], row["errors"]
            passed = all(math.isfinite(value) for value in errors.values())
            if name in previous:
                passed &= all(value < previous[name][key] for key,value in errors.items())
            if n >= 64:
                passed &= errors["inward_flux_W_m2"] < .01
            if "discrete_balance_J" in row:
                passed &= row["discrete_balance_J"] < 1e-12
            row["passed"] = passed
            previous[name] = errors
            rows.append(row)
    print(json.dumps({"sources": [provenance(f) for f in (__file__, energy.__file__, sph.__file__, thermal.__file__, thermal_analytic.__file__)],
                      "time_s": .02, "runs": rows}, indent=2))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true", help="include the 128 by 128 mesh")
    raise SystemExit(main(parser.parse_args().extended))
