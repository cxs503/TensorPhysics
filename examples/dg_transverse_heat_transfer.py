"""Verify transverse heat transport with zero transverse temperature gradient."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensordg import poisson_2d_bc, transient_diffusion_2d


def measure(n, degree, method, cross):
    kx, ky, capacity = 1.8, .7, 2.1
    material = (kx, ky, cross)
    rate = np.pi**2 * kx / capacity
    times = np.linspace(0., .032, 2*n + 1)
    bc = dict.fromkeys(("left", "right", "top", "bottom"), "neumann")

    def walls(scale):
        # Boundary input is inward. The upward Fourier flux is positive for cross > 0.
        return {"bottom": lambda x,y: scale*cross*torch.pi*torch.sin(torch.pi*x),
                "top": lambda x,y: -scale*cross*torch.pi*torch.sin(torch.pi*x)}

    x, _, states, means, amplitudes = transient_diffusion_2d.solve_transient_diffusion_2d(
        lambda x,y: .6+torch.cos(torch.pi*x), times, ncells=n, degree=degree,
        capacity=capacity, bc=bc, conductivity=material, method=method,
        neumann_flux=walls(1.), load_amplitude=lambda t: np.exp(-rate*t),
        return_interval_means=True, return_load_means=True)
    dt = torch.from_numpy(np.diff(times))
    accumulated = (dt @ means.flatten(1)).reshape_as(states[0])
    hx, hy = poisson_2d_bc.poisson_face_powers_2d(accumulated, bc=bc,
        conductivity=material, neumann_flux=walls(float(dt @ amplitudes)))

    # Face means of the analytic time-integrated upward and rightward fluxes.
    duration = -np.expm1(-rate*times[-1])/rate
    faces = torch.arange(n+1, dtype=x.dtype)/n
    expected_x = duration*kx*torch.pi*torch.sin(torch.pi*faces)
    expected_y = duration*cross*torch.pi*torch.sin(torch.pi*x)*np.sinc(.5/n)
    averages = states[:, :, :, 0, 0]
    expected_t = .6 + np.exp(-rate*times[-1])*torch.cos(torch.pi*x)*np.sinc(.5/n)
    stored = capacity*(averages[-1]-averages[0])/n**2
    return dict(cells_per_axis=n, degree=degree, method=method, cross_conductivity_W_mK=cross,
        temperature_error_K=float((averages[-1]-expected_t[:, None]).abs().max()),
        x_face_heat_error_J_m2=float((n*hx-expected_x[:, None]).abs().max()),
        y_face_heat_error_J_m2=float((n*hy-expected_y[:, None]).abs().max()),
        measured_midplane_heat_J=float(hy[:, n//2].sum()), exact_midplane_heat_J=2*cross*duration,
        max_cell_balance_J=float((stored+hx.diff(dim=0)+hy.diff(dim=1)).abs().max()),
        global_heat_drift_J=float(capacity*(averages.mean((1,2))-.6).abs().max()))


if __name__ == "__main__":
    rows = []
    metrics = ["temperature_error_K", "x_face_heat_error_J_m2", "y_face_heat_error_J_m2"]
    for cross in (-.55, .55):
        for degree in (1, 2):
            for method in ("theta", "sdirk2"):
                previous = None
                for n in (4, 8, 16):
                    row = measure(n, degree, method, cross)
                    errors = np.array([row[key] for key in metrics])
                    row["orders"] = [] if previous is None else np.log2(previous/errors).tolist()
                    limits = (2e-3, 1e-3, 2e-4) if degree == 1 else (3e-6, 2e-6, 5e-7)
                    row["passed"] = bool(max(row["max_cell_balance_J"], row["global_heat_drift_J"]) < 1e-11
                        and row["measured_midplane_heat_J"] * row["exact_midplane_heat_J"] > 0
                        and all(order > 1 for order in row["orders"])
                        and (n < 16 or np.all(errors < limits)))
                    rows.append(row)
                    previous = errors
    print(json.dumps(dict(reference="T=.6+exp(-pi^2*kxx*t/C)*cos(pi*x); q_y=-kxy*dT/dx, unit square",
        reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorDG python examples/dg_transverse_heat_transfer.py",
        order_metrics=metrics, runs=rows, sources=[provenance(p) for p in
        (__file__, poisson_2d_bc.__file__, transient_diffusion_2d.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
