"""Compare DG and FEM time-integrated mean heat flux for an anisotropic cooling mode.

Run python examples/dg_fem_interval_cooling_flux.py with TensorDG and TensorFEM
installed. The west and south walls are at zero temperature, and the east
and north walls are insulated. Each reported interval has a fixed duration
across all refinements.
"""
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensordg import poisson_2d_bc, transient_diffusion_2d
from tensorfem import thermal, thermal_analytic


KX, KY, CAPACITY = 1., 2., 1.5
WINDOWS = np.array([0., .05, .13, .25])


def mode(x, y):
    return torch.sin(math.pi*x/2)*torch.sin(math.pi*y/2)


def measure(n):
    times = np.r_[0., np.concatenate([np.linspace(a,b,n+1)[1:] for a,b in zip(WINDOWS, WINDOWS[1:])])]
    _, _, _, dg = transient_diffusion_2d.solve_transient_diffusion_2d(
        mode, times, ncells=n, degree=1, capacity=CAPACITY, conductivity=(KX,KY),
        bc=dict(left="dirichlet", bottom="dirichlet", right="neumann", top="neumann"),
        method="sdirk2", return_cell_flux_integral=True)

    mesh = thermal_analytic.structured_quad_mesh(n, n, dtype=torch.float64)
    x,y = mesh.nodes.T
    fixed = torch.where((x == 0) | (y == 0))[0]
    initial = mode(x,y)
    model = thermal.ThermalModel(mesh.nodes, mesh.elements, x.new_tensor([[[KX,0.],[0.,KY]]]),
                                 1., CAPACITY, fixed, initial[fixed], element_type="q4")
    result = thermal.solve_transient_thermal(model, torch.from_numpy(times), initial, method="sdirk2", sparse=True)
    fem = thermal.thermal_volume_mean_heat_steps(model, result)
    # DG cells have equal area. Sum signed substep integrals over each window.
    return {name: values.reshape(len(WINDOWS)-1, n, 2).sum(dim=1).numpy()
            for name, values in (("TensorDG_P1", dg.mean(dim=(1,2))), ("TensorFEM_Q4", fem))}


def run():
    decay = (KX+KY)*(math.pi/2)**2/CAPACITY
    amplitude_integral = np.exp(-decay*WINDOWS[:-1])*(-np.expm1(-decay*np.diff(WINDOWS)))/decay
    exact = -amplitude_integral[:,None]*(2/math.pi)*np.array([KX,KY])
    rows, previous = [], {}
    for n in (4, 8, 16):
        for name, heat in measure(n).items():
            errors = np.max(abs(heat-exact), axis=0)
            passed = np.all(heat < 0) and (name not in previous or np.all(previous[name] > 3*errors))
            if n == 16:
                passed = passed and np.max(errors) < 1e-4
            rows.append(dict(solver=name, cells_per_side=n, steps_per_window=n, heat_J_m2=heat.tolist(),
                             component_errors_J_m2=errors.tolist(), passed=bool(passed)))
            previous[name] = errors
    return dict(command="python examples/dg_fem_interval_cooling_flux.py", windows_s=WINDOWS.tolist(),
                reference_heat_J_m2=exact.tolist(), runs=rows,
                sources=[provenance(p) for p in (__file__, transient_diffusion_2d.__file__, poisson_2d_bc.__file__,
                                                  thermal.__file__, thermal_analytic.__file__)])


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in report["runs"]) else 1)
