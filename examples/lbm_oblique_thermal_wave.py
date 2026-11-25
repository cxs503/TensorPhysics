"""Check a driven oblique D3Q7 diffusion wave against cell and face integrals."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, omega):
    dx = 1/n
    k, capacity = 1.7, 1.7
    dt = dx**2/8  # D3Q7 BGK tau=1, k/C=1
    counts = (n, 3*n//4, n//2)
    direction = np.array([2., -1., 2.])/3
    wave = np.sqrt(1j*omega*capacity/k)*direction
    faces = [np.exp(s*np.arange(count+1)*dx) for s, count in zip(wave, counts)]
    means = [f[:-1]*np.expm1(s*dx)/(s*dx) for f, s in zip(faces, wave)]

    def product(a):
        return torch.from_numpy(np.einsum("i,j,k->kji", *a))

    exact = product(means)
    flux_modes = []
    for axis, s in enumerate(wave):
        factors = means.copy()
        factors[axis] = faces[axis]
        flux_modes.append(-k*s*product(factors))
    field = exact.real
    zero = torch.zeros_like(field)
    g = lbm.thermal_equilibrium_3d(field, zero, zero, zero)
    units = dict(dt=dt, spacing=dx, volumetric_heat_capacity=capacity)
    worst_flux = np.zeros(3)
    worst_t = worst_balance = 0.
    steps = round(.0625/dt)
    for step in range(steps):
        phase = np.exp(1j*omega*step*dt)
        mean_phase = phase*np.expm1(1j*omega*dt)/(1j*omega*dt)
        post = lbm.thermal_collide_bgk_3d(g, field, zero, zero, zero, 1.)
        fluxes = lbm.thermal_link_fluxes_3d(post, **units)
        g = lbm.thermal_stream_3d(post)
        divergence = torch.zeros_like(field)
        for axis, (flux, mode) in enumerate(zip(fluxes, flux_modes)):
            dim = 2-axis
            target = (mean_phase*mode).real
            low, high = target.select(dim, 0), target.select(dim, -1)
            g = lbm.thermal_flux_wall_pair_3d(g, "xyz"[axis], (low, -high), **units)
            error = (flux-target.narrow(dim, 1, counts[axis]-1)).abs().max().item()
            worst_flux[axis] = max(worst_flux[axis], error)
            divergence += torch.cat((low.unsqueeze(dim), flux, high.unsqueeze(dim)), dim=dim).diff(dim=dim)
        updated = lbm.thermal_macroscopic_3d(g)
        residual = capacity*dx**3*(updated-field)+dt*dx**2*divergence
        worst_balance = max(worst_balance, residual.abs().max().item())
        field = updated
        target_t = (np.exp(1j*omega*(step+1)*dt)*exact).real
        worst_t = max(worst_t, (field-target_t).abs().max().item())
    return dict(cells_xyz=counts, omega_rad_s=omega, steps=steps, temperature_error_K=worst_t,
                interior_flux_errors_xyz_W_m2=worst_flux.tolist(), cell_energy_residual_J=worst_balance)


if __name__ == "__main__":
    rows = []
    for omega in (4., 16.):
        previous = None
        for n in (8, 16, 32, 48):
            row = measure(n, omega)
            errors = np.r_[row["temperature_error_K"], row["interior_flux_errors_xyz_W_m2"]]
            orders = [] if previous is None else (np.log(previous/errors)/np.log(n/previous_n)).tolist()
            order_target = all(v > 1.7 for v in orders) if orders else None
            passed = row["cell_energy_residual_J"] < 1e-13 and all(v > 0 for v in orders)
            if n == 48:
                passed = passed and order_target and errors[0] < .002 and max(errors[1:]) < .006
            row.update(temperature_and_xyz_flux_orders=orders, all_orders_above_1p7=order_target,
                       passed=bool(passed))
            previous, previous_n = errors, n
            rows.append(row)
    print(json.dumps(dict(reference="Cell, face and step averages of exp(s dot x + i omega t); k*(s dot s)=i omega C; all six inward boundary fluxes prescribed",
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_oblique_thermal_wave.py",
                          runs=rows, sources=[provenance(__file__), provenance(lbm.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
