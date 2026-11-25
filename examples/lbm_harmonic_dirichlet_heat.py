"""Check the complex wall heat flux and temperature response to two out-of-phase wall drives."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, period_factor):
    dx, alpha, capacity = 1/n, .08, 1.7
    dt = dx**2/(8*alpha)
    steps = period_factor*n*n//4
    omega = 2*np.pi/(steps*dt)
    lam = np.sqrt(1j*omega/alpha)
    left, right = .8+.25j, -.3+.55j
    x = (np.arange(n)+.5)*dx
    profile = (left*np.sinh(lam*(1-x))+right*np.sinh(lam*x))/np.sinh(lam)
    exact_t = profile*np.sinh(lam*dx/2)/(lam*dx/2)
    # Positive inward, averaged over the streaming interval at its midtime.
    exact_q = capacity*alpha*lam*np.array([left*np.cosh(lam)-right, right*np.cosh(lam)-left])/np.sinh(lam)
    exact_q *= np.sinc(1/steps)
    field = torch.from_numpy(exact_t.real.copy()).reshape(1,1,n)
    zeros = torch.zeros_like(field)
    g = lbm.thermal_equilibrium_3d(field,zeros,zeros,zeros)
    temp_phasor = np.zeros(n,complex)
    flux_phasor = np.zeros(2,complex)
    max_balance = 0.
    for step in range(3*steps):
        mid = np.exp(1j*omega*(step+.5)*dt)
        end = np.exp(1j*omega*(step+1)*dt)
        post = lbm.thermal_collide_bgk_3d(g,field,zeros,zeros,zeros,1.)
        g, walls = lbm.thermal_temperature_wall_pair_3d(lbm.thermal_stream_3d(post),"x",
            ((left*mid).real,(right*mid).real),dt=dt,spacing=dx,volumetric_heat_capacity=capacity)
        updated = lbm.thermal_macroscopic_3d(g)
        inward = np.array([q.item() for q in walls])
        residual = capacity*dx*(updated-field).sum().item()-dt*inward.sum()
        max_balance = max(max_balance,abs(residual))
        if step >= 2*steps:
            temp_phasor += (2/steps)*end.conjugate()*updated.numpy().ravel()
            flux_phasor += (2/steps)*mid.conjugate()*inward
        field = updated
    return dict(n=n, steps_per_period=steps, omega_rad_s=omega,
                temperature_phasor_error_K=float(abs(temp_phasor-exact_t).max()),
                temperature_amplitude_error_K=float(abs(abs(temp_phasor)-abs(exact_t)).max()),
                temperature_phase_error_deg=float(abs(np.angle(temp_phasor/exact_t,deg=True)).max()),
                wall_flux_phasor_error_W_m2=float(abs(flux_phasor-exact_q).max()),
                wall_flux_amplitude_error_W_m2=float(abs(abs(flux_phasor)-abs(exact_q)).max()),
                wall_flux_phase_error_deg=float(abs(np.angle(flux_phasor/exact_q,deg=True)).max()),
                max_interval_energy_residual_J_m2=max_balance)


if __name__ == "__main__":
    rows = []
    metrics = ["temperature_phasor_error_K","wall_flux_phasor_error_W_m2"]
    for factor in (1,2):
        previous = None
        for n in (8,16,32,64):
            row = measure(n,factor)
            errors = np.array([row[key] for key in metrics])
            row["orders"] = [] if previous is None else np.log2(previous/errors).tolist()
            row["passed"] = bool(row["max_interval_energy_residual_J_m2"] < 1e-13
                and all(order > 1.2 for order in row["orders"])
                and (n < 64 or (errors[0] < .005 and errors[1] < .009
                                and row["wall_flux_phase_error_deg"] < .3)))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(reference="Harmonic 1-D slab with independent complex Dirichlet temperatures; project final cycle of three onto exp(i omega t)",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_harmonic_dirichlet_heat.py",
        order_metrics=metrics,runs=rows,sources=[provenance(p) for p in (__file__,lbm.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
