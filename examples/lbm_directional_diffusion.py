"""Verify D3Q7 decay and signed link heat flux for equal-length wave vectors."""
import argparse
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def diffusion_eigenmode(theta, tau):
    # A Fourier component is multiplied by exp(-i*c.theta) on streaming.
    weights = np.array([.25] + [.125]*6)
    phase = np.exp(-1j*np.r_[0., np.column_stack((theta, -theta)).ravel()])
    collision = (1-1/tau)*np.eye(7) + np.outer(weights, np.ones(7))/tau
    gains, vectors = np.linalg.eig(phase[:, None]*collision)
    index = np.argmin(abs(gains-1))
    gain = gains[index]
    if abs(gain.imag) > 1e-12 or not 0 < gain.real < 1:
        raise ValueError("expected an isolated, decaying real diffusion mode")
    vector = vectors[:, index]
    return gain.real, vector/vector.sum()


def measure(n, mode, tau=1.):
    dx, alpha, capacity = 1/n, .08, 1.7
    dt = (tau-.5)*dx**2/(4*alpha)
    k = 2*np.pi*np.asarray(mode)
    theta = k*dx
    # At unit relaxation each moving population carries 1/8 of its donor scalar.
    gain = (1 + np.cos(theta).sum())/4
    coords = (torch.arange(n, dtype=torch.float64)+.5)*dx
    z,y,x = torch.meshgrid(coords, coords, coords, indexing="ij")
    angle = k[0]*x+k[1]*y+k[2]*z
    sinc = np.sinc(theta/(2*np.pi))
    initial = angle.cos()*sinc.prod()
    zero = torch.zeros_like(initial)
    field = .6+initial
    if tau == 1.:
        populations = lbm.thermal_equilibrium_3d(field, zero, zero, zero)
    else:
        gain, vector = diffusion_eigenmode(theta, tau)
        phase = torch.exp(1j*angle)*sinc.prod()
        perturbation = (torch.from_numpy(vector)[:, None, None, None]*phase).real
        populations = lbm.thermal_equilibrium_3d(torch.full_like(field, .6), zero, zero, zero) + perturbation
    symbol_error = flux_error = cell_error = 0.
    rate = alpha*np.dot(k,k)
    for step in range(4):
        post = lbm.thermal_collide_bgk_3d(populations, field, zero, zero, zero, tau)
        fluxes = lbm.thermal_link_fluxes_3d(post, dt=dt, spacing=dx,
                                            volumetric_heat_capacity=capacity, periodic=True)
        divergence = torch.zeros_like(initial)
        factor = math.exp(-rate*step*dt)*(-math.expm1(-rate*dt))/(rate*dt)
        for axis, flux in enumerate(fluxes):
            # Continuum flux averaged over the positive face and its step.
            target = capacity*alpha*k[axis]*factor*(sinc.prod()/sinc[axis])*(angle+theta[axis]/2).sin()
            flux_error = max(flux_error, float((flux-target).abs().max()))
            divergence += flux - flux.roll(1, dims=2-axis)
        populations = lbm.thermal_stream_3d(post)
        updated = lbm.thermal_macroscopic_3d(populations)
        cell_error = max(cell_error, float((capacity*dx**3*(updated-field)+dt*dx**2*divergence).abs().max()))
        symbol_error = max(symbol_error, float((updated-.6-gain**(step+1)*initial).abs().max()))
        field = updated
    amplitude = float(((field-.6)*initial).sum()/initial.square().sum())
    inferred = -math.log(amplitude)/(4*dt*np.dot(k,k))
    return dict(n=n, mode_xyz=mode, steps=4, tau=tau, diffusivity_m2_s=inferred,
                relative_diffusivity_error=abs(inferred/alpha-1), max_flux_error_W_m2=flux_error,
                discrete_temperature_error_K=symbol_error, max_cell_balance_J=cell_error)


if __name__ == "__main__":
    rows = []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eigenmodes", action="store_true", help="initialize the slow Fourier eigenmode at nonunit relaxation")
    eigenmodes = parser.parse_args().eigenmodes
    finest_limits = (.02, .05) if eigenmodes else (.003, .005)
    for tau in ((.75, 1.25) if eigenmodes else (1.,)):
        for mode in ((3,0,0), (2,-2,1), (-1,2,2)):
            previous = None
            for n in (16,32,64):
                row = measure(n, mode, tau)
                errors = np.array([row["relative_diffusivity_error"], row["max_flux_error_W_m2"]])
                row["orders"] = [] if previous is None else np.log2(previous/errors).tolist()
                row["passed"] = bool(row["discrete_temperature_error_K"] < 1e-12
                    and row["max_cell_balance_J"] < 1e-16 and all(v > (1.7 if eigenmodes and n == 32 else 1.8) for v in row["orders"])
                    and (n < 64 or np.all(errors < finest_limits)))
                rows.append(row)
                previous = errors
    reference = ("D3Q7 eigenvalue of streaming times BGK collision nearest unity; continuum exp(-alpha |k|^2 t)"
                 if eigenmodes else "Equal wavenumber magnitude 6*pi; D3Q7 gain=(1+sum cos(k_i dx))/4; continuum exp(-alpha |k|^2 t)")
    print(json.dumps(dict(reference=reference,
        reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_directional_diffusion.py"+(" --eigenmodes" if eigenmodes else ""),
        order_metrics=["relative_diffusivity_error","max_flux_error_W_m2"], runs=rows,
        sources=[provenance(p) for p in (__file__, lbm.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
