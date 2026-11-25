"""Check equilibrium-start D3Q7 populations, temperatures and signed face fluxes."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, tau, velocity):
    dx, alpha, capacity = 1/n, .08, 1.7
    dt = (tau-.5)*dx**2/(4*alpha)
    wave = 2*np.pi*np.array([2., -1., 1.])
    directions = np.vstack(([0.,0.,0.], np.stack((np.eye(3), -np.eye(3)), axis=1).reshape(6,3)))
    weights = np.array([.25]+[.125]*6)
    lattice_velocity = np.asarray(velocity)*dt/dx
    equilibrium = weights*(1+4*(directions @ lattice_velocity))
    collision = (1-1/tau)*np.eye(7)+np.outer(equilibrium,np.ones(7))/tau
    stream = np.exp(-1j*(directions @ wave)*dx)
    update = stream[:,None]*collision
    coord = (torch.arange(n,dtype=torch.float64)+.5)*dx
    z,y,x = torch.meshgrid(coord,coord,coord,indexing="ij")
    phase = torch.exp(1j*(wave[0]*x+wave[1]*y+wave[2]*z))
    background = .4
    field = background+phase.real
    speeds = [torch.full_like(field, v) for v in lattice_velocity]
    pop = lbm.thermal_equilibrium_3d(field,*speeds)
    errors = []
    for step in range(8):
        # Apply the complete matrix power to the initial state, including kinetic modes.
        amplitudes = np.linalg.matrix_power(update, step) @ equilibrium
        post_amplitudes = collision @ amplitudes
        next_amplitudes = update @ amplitudes
        post = lbm.thermal_collide_bgk_3d(pop,field,*speeds,tau)
        fluxes = lbm.thermal_link_fluxes_3d(post,dt=dt,spacing=dx,volumetric_heat_capacity=capacity,periodic=True)
        flux_error = []
        for axis, flux in enumerate(fluxes):
            forward = 1+2*axis
            backward = forward+1
            mode = post_amplitudes[forward]-np.exp(1j*wave[axis]*dx)*post_amplitudes[backward]
            target = capacity*(background*velocity[axis]+dx/dt*(mode*phase).real)
            flux_error.append(float((flux-target).abs().max()))
        pop = lbm.thermal_stream_3d(post)
        field = lbm.thermal_macroscopic_3d(pop)
        exact_pop = (torch.from_numpy(next_amplitudes)[:,None,None,None]*phase).real
        exact_pop += torch.from_numpy(background*equilibrium)[:,None,None,None]
        errors.append(dict(step=step+1, temperature_error_K=float((field-background-(next_amplitudes.sum()*phase).real).abs().max()),
                           population_error_K=float((pop-exact_pop).abs().max()), flux_errors_xyz_W_m2=flux_error))
    return dict(n=n,tau=tau,velocity_m_s=velocity,steps=errors,
                passed=all(row["temperature_error_K"] < 1e-12 and row["population_error_K"] < 1e-12
                           and max(row["flux_errors_xyz_W_m2"]) < 1e-11 for row in errors))


if __name__ == "__main__":
    runs = [measure(n,tau,velocity) for n in (12,24,48) for tau in (.65,1.,1.4)
            for velocity in ([0.,0.,0.],[.32,-.21,.18],[-.32,.21,-.18])]
    print(json.dumps(dict(reference="Periodic Fourier evolution from the equilibrium vector by powers of the 7x7 streaming-collision matrix, constant prescribed velocity",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_fourier_startup.py",
        runs=runs,sources=[provenance(p) for p in (__file__,lbm.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in runs) else 1)
