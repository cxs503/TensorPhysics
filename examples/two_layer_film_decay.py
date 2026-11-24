"""Compare FEM and FVM with a decaying contact mode cooled by two wall films."""
import inspect
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from two_layer_conduction import slab_fem_model
from tensorfem import thermal
from tensorfvm.cht import energy


def measure(n):
    a, b = 1.5*math.pi, 4*math.pi/3
    phi, psi = .25, .4
    cap = 4*(b/a)**2
    amp = a*math.cos(a/2+phi) / (4*b*math.cos(b/2+psi))
    jump = math.sin(a/2+phi) + amp*math.sin(b/2+psi)
    resistance = jump / (-a*math.cos(a/2+phi))
    films = (a/math.tan(phi), 4*b/math.tan(psi))
    k = np.repeat([1., 4.], n//2)
    capacity = np.repeat([1., cap], n//2)
    time = np.linspace(0., .015, n+1)
    decay = math.exp(-a*a*time[-1])
    heat0 = ((math.cos(phi)-math.cos(a/2+phi))/a
             - cap*amp*(math.cos(psi)-math.cos(b/2+psi))/b)

    def mode(xl, xr):
        return np.r_[np.sin(a*xl+phi), -amp*np.sin(b*(1-xr)+psi)]

    model, ends = slab_fem_model(k, 0., 0., resistance, films, torch.from_numpy(capacity))
    x = np.linspace(0, 1, n+1)
    m = n//2
    initial = mode(x[:m+1], x[m:])
    fem_t = thermal.solve_transient_thermal(
        model, torch.from_numpy(time), torch.from_numpy(np.tile(initial, 2)), theta=.5).temperature[-1]
    walls = fem_t[ends].mean(dim=1).numpy()
    final = fem_t.reshape(2, -1).mean(dim=0).numpy()
    delta = final-initial
    # Exact integral of the FEM piecewise-linear temperature along x.
    heat = (delta[:m].sum()+delta[1:m+1].sum()
            + cap*(delta[m+1:-1].sum()+delta[m+2:].sum()))/(2*n)
    values = [("FEM", np.tile(initial, 2), fem_t.numpy(), walls, final[m]-final[m+1], heat)]

    centers = (np.arange(n)+.5)/n
    initial = mode(centers[:m], centers[m:])
    faces = np.zeros((1, n-1))
    faces[0,m-1] = resistance
    bc = {side: {"type": "robin", "values": 0., "film_coefficient": h}
          for side, h in zip(("west", "east"), films)}
    final = energy.solve_transient_conduction(np.full(n, 1/n), [1.], k, capacity, bc, time,
                                              initial, contact_resistance_x=faces)[-1,0]
    # Recover each wall trace from the half-cell drop and the film law.
    walls = final[[0,-1]] / (1 + np.asarray(films)/(2*n*k[[0,-1]]))
    contact_jump = 1.5*(final[m-1]-final[m]) - .5*(final[m-2]-final[m+1])
    values.append(("FVM", initial, final, walls, contact_jump, np.mean(capacity*(final-initial))))
    wall_exact = decay * np.array([math.sin(phi), -amp*math.sin(psi)])
    return [{"solver": name, "cells": n,
             "errors": {"temperature_K": float(abs(final - decay*initial).max()),
                        "wall_K": float(abs(walls-wall_exact).max()),
                        "jump_K": float(abs(contact_jump-decay*jump)),
                        "heat_change_J_m2": float(abs(heat-(decay-1)*heat0))}}
            for name, initial, final, walls, contact_jump, heat in values]


def main():
    runs = []
    previous = {}
    for n in (16, 32, 64):
        for row in measure(n):
            errors = row["errors"]
            last = previous.get(row["solver"])
            row["passed"] = all(math.isfinite(v) for v in errors.values())
            if last:
                row["passed"] &= all(v < last[k]/3 for k, v in errors.items())
            if n == 64:
                limits = {"temperature_K": .001, "wall_K": .001, "jump_K": .003, "heat_change_J_m2": 1e-5}
                row["passed"] &= all(value < limits[key] for key, value in errors.items())
            previous[row["solver"]] = errors
            runs.append(row)
    print(json.dumps({"sources": [provenance(f) for f in
                      (__file__, inspect.getfile(slab_fem_model), thermal.__file__, energy.__file__)],
                      "time_s": .015, "runs": runs}, indent=2))
    return 0 if all(row["passed"] for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
