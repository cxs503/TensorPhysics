"""Verify D3Q7 total heat transport for a moving, decaying periodic Fourier mode."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, sign, flux_walls=False):
    dx, alpha, capacity = 1/n, .08, 1.7
    velocity = sign*np.array([.62, -.38, .2])
    wave = 2*np.pi*np.array([1., -1., 1.])
    if flux_walls:
        # Tangential advection; the x-mode has distinct left and right traces.
        velocity[0], wave[0] = 0., 1.65
    decay = -alpha*(wave@wave)-1j*(velocity@wave)
    dt = dx**2/(8*alpha)  # BGK relaxation time of one lattice step.
    steps = n*n//16
    centers = (np.arange(n)+.5)*dx
    z, y, x = np.meshgrid(centers, centers, centers, indexing="ij")
    phase = torch.from_numpy(np.exp(1j*(wave[0]*x+wave[1]*y+wave[2]*z)))
    sinc = np.sinc(wave*dx/(2*np.pi))
    cell_mode = phase*sinc.prod()
    face_modes = [capacity*(u-1j*alpha*k)*cell_mode*np.exp(.5j*k*dx)/factor
                  for u,k,factor in zip(velocity,wave,sinc)]
    background = .7
    field = background+cell_mode.real
    speeds = tuple(torch.full_like(field, u*dt/dx) for u in velocity)
    g = lbm.thermal_equilibrium_3d(field, *speeds)
    peak_flux, peak_wrap = np.zeros(3), np.zeros(3)
    peak_t = peak_balance = 0.
    for step in range(steps):
        avg_phase = np.exp(decay*step*dt)*np.expm1(decay*dt)/(decay*dt)
        post = lbm.thermal_collide_bgk_3d(g, field, *speeds, 1.)
        fluxes = lbm.thermal_link_fluxes_3d(post, dt=dt, spacing=dx,
                                            volumetric_heat_capacity=capacity, periodic="yz" if flux_walls else True)
        g = lbm.thermal_stream_3d(post)
        divergence = torch.zeros_like(field)
        for axis, (flux, face) in enumerate(zip(fluxes, face_modes)):
            dim = 2-axis
            target = capacity*velocity[axis]*background+(avg_phase*face).real
            if flux_walls and axis == 0:
                low = (avg_phase*face[..., 0]*np.exp(-1j*wave[0]*dx)).real
                high = target[..., -1]
                g = lbm.thermal_flux_wall_pair_3d(g, "x", (low, -high), dt=dt,
                                                  spacing=dx, volumetric_heat_capacity=capacity)
                divergence += torch.cat((low[..., None], flux, high[..., None]), dim=dim).diff(dim=dim)
                error = abs(flux-target[..., :-1])
            else:
                divergence += flux-flux.roll(1, dims=dim)
                error = abs(flux-target)
                peak_wrap[axis] = max(peak_wrap[axis], error.select(dim,-1).max().item())
            peak_flux[axis] = max(peak_flux[axis], error.max().item())
        updated = lbm.thermal_macroscopic_3d(g)
        balance = capacity*dx**3*(updated-field)+dt*dx**2*divergence
        peak_balance = max(peak_balance, balance.abs().max().item())
        target = background+(np.exp(decay*(step+1)*dt)*cell_mode).real
        peak_t = max(peak_t, abs(updated-target).max().item())
        field = updated
    return dict(n=n, steps=steps, velocity_m_s=velocity.tolist(), temperature_error_K=peak_t,
                total_flux_errors_xyz_W_m2=peak_flux.tolist(), wrap_flux_errors_xyz_W_m2=([None, *peak_wrap[1:]] if flux_walls else peak_wrap.tolist()),
                max_cell_energy_residual_J=peak_balance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux-walls", action="store_true", help="prescribe both x-face fluxes, with periodic y and z")
    flux_walls = parser.parse_args().flux_walls
    rows = []
    for sign in (1, -1):
        previous = None
        for n in (8, 16, 32, 48):
            row = measure(n, sign, flux_walls)
            errors = np.r_[row["temperature_error_K"], row["total_flux_errors_xyz_W_m2"]]
            orders = [] if previous is None else (np.log(previous/errors)/np.log(n/previous_n)).tolist()
            resolved = all(v > 1.7 for v in orders) if orders else None
            passed = row["max_cell_energy_residual_J"] < 1e-13 and all(v > 0 for v in orders)
            if n >= 32:
                passed &= resolved
            row.update(temperature_and_total_flux_orders=orders, orders_above_1p7=resolved, passed=bool(passed))
            rows.append(row)
            previous, previous_n = errors, n
    print(json.dumps(dict(reference=("Unit cube with prescribed x-face flux and periodic y,z" if flux_walls else "Unit periodic cube")+"; T=0.7+Re(exp(i*k.x-(alpha*|k|^2+i*u.k)*t)); total heat flux C*(u*T-alpha*grad(T)), averaged over the face and time step",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_periodic_advective_flux.py"+(" --flux-walls" if flux_walls else ""),
        runs=rows, sources=[provenance(__file__), provenance(lbm.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
