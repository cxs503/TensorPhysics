"""Compare SPH wall-row films and FVM Robin faces in a smooth cooling slab."""
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


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
    return [{"solver": name, "cells_per_side": n, "errors": {
        "temperature_K": float(np.max(abs(field-exact))),
        "wall_K": float(np.max(abs(surface-exact_wall))),
        "inward_flux_W_m2": float(np.max(abs(inward+film*exact_wall))),
        "heat_change_J": float(abs((field-initial).mean()-exact_heat))}}
        for name, field, surface, inward in fields]


def main():
    previous = {}
    rows = []
    for n in (16,32,64):
        for row in measure(n):
            name, errors = row["solver"], row["errors"]
            passed = all(math.isfinite(value) for value in errors.values())
            if name in previous:
                passed &= all(value < previous[name][key] for key,value in errors.items())
            if n == 64:
                passed &= errors["inward_flux_W_m2"] < .01
            row["passed"] = passed
            previous[name] = errors
            rows.append(row)
    print(json.dumps({"sources": [provenance(f) for f in (__file__, energy.__file__, sph.__file__)],
                      "time_s": .02, "runs": rows}, indent=2))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
