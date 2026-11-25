"""Verify temperature decay and directional heat leaving a quadrant of an insulated square."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensordg import poisson_2d_bc, transient_diffusion_2d


def measure(n, degree, method):
    kx, ky, capacity = 1.8, .7, 2.1
    stop = .032
    rate = np.pi**2 * (kx + ky) / capacity
    times = np.linspace(0., stop, 2*n + 1)
    def initial(x, y):
        return .6 + torch.cos(torch.pi*x) * torch.cos(torch.pi*y)
    bc = dict.fromkeys(("left", "right", "top", "bottom"), "neumann")
    material = (kx, ky, 0.)
    x, y, states, means = transient_diffusion_2d.solve_transient_diffusion_2d(
        initial, times, ncells=n, degree=degree, capacity=capacity, bc=bc, conductivity=material,
        method=method, return_interval_means=True)
    # Combine the stage-weighted temperatures before applying the linear face operator.
    integrated = torch.from_numpy(np.diff(times)) @ means.flatten(1)
    hx, hy = poisson_2d_bc.poisson_face_powers_2d(integrated.reshape_as(states[0]),
                                                   bc=bc, conductivity=material)
    half = n//2
    outward = np.array([float(hx[half, :half].sum()), float(hy[:half, half].sum())])
    exact_heat = np.array([kx, ky]) * (-np.expm1(-rate*stop)/rate)
    decay = np.exp(-rate*stop)
    # Exact modal constant equals the two-direction cell average.
    exact_avg = .6 + decay * torch.cos(torch.pi*x[:, None]) * torch.cos(torch.pi*y) * np.sinc(.5/n)**2
    averages = states[:, :, :, 0, 0]
    delta = capacity * (averages[-1]-averages[0]) / n**2
    return dict(cells_per_axis=n, degree=degree, method=method,
                cell_mean_error_K=float((averages[-1]-exact_avg).abs().max()),
                outward_heat_J=outward.tolist(), exact_heat_J=exact_heat.tolist(),
                max_cut_error_J=float(abs(outward-exact_heat).max()),
                quadrant_balance_J=float(abs(outward.sum()+delta[:half,:half].sum().item())),
                max_cell_balance_J=float((delta+hx.diff(dim=0)+hy.diff(dim=1)).abs().max()),
                max_global_heat_drift_J=float(capacity*(averages.mean((1,2))-.6).abs().max()))


if __name__ == "__main__":
    rows = []
    for degree in (1, 2):
        for method in ("theta", "sdirk2"):
            previous = None
            for n in (4, 8, 16):
                row = measure(n, degree, method)
                errors = np.array([row["cell_mean_error_K"], row["max_cut_error_J"]])
                row["orders"] = [] if previous is None else np.log2(previous/errors).tolist()
                balance = max(row[key] for key in ("quadrant_balance_J", "max_cell_balance_J",
                                                    "max_global_heat_drift_J"))
                limits = (1e-3, 1.5e-4) if degree == 1 else (5e-6, 1e-6)
                row["passed"] = bool(balance < 1e-11 and all(v > 1 for v in row["orders"])
                                     and (n < 16 or np.all(errors < limits)))
                rows.append(row)
                previous = errors
    print(json.dumps(dict(reference="0.6 + exp(-pi^2*(kx+ky)*t/C) cos(pi*x) cos(pi*y); unit square with insulated exterior",
        reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorDG python examples/dg_insulated_quadrant_heat.py",
        order_metrics=["cell_mean_error_K","max_cut_error_J"], runs=rows,
        sources=[provenance(p) for p in (__file__, transient_diffusion_2d.__file__, poisson_2d_bc.__file__)]),
        indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
