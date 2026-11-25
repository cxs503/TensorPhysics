"""Check the complex wall heat flux and temperature response to two out-of-phase wall drives."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, period_factor, films=None):
    dx, alpha, capacity = 1/n, .08, 1.7
    dt = dx**2/(8*alpha)
    steps = period_factor*n*n//4
    omega = 2*np.pi/(steps*dt)
    lam = np.sqrt(1j*omega/alpha)
    drives = np.array([.8+.25j, -.3+.55j])
    left, right = drives
    if films is not None:
        # Continuum slab Dirichlet-to-Neumann map, with both fluxes positive inward.
        h = np.diag(films)
        admittance = capacity*alpha*lam*np.array([[np.cosh(lam),-1],[-1,np.cosh(lam)]])/np.sinh(lam)
        left, right = np.linalg.solve(h+admittance, h @ drives)
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
    wall_phasor = np.zeros(2,complex)
    max_balance = 0.
    for step in range(3*steps):
        mid = np.exp(1j*omega*(step+.5)*dt)
        end = np.exp(1j*omega*(step+1)*dt)
        post = lbm.thermal_collide_bgk_3d(g,field,zeros,zeros,zeros,1.)
        streamed = lbm.thermal_stream_3d(post)
        if films is None:
            g, walls = lbm.thermal_temperature_wall_pair_3d(streamed,"x",(drives*mid).real,
                dt=dt,spacing=dx,volumetric_heat_capacity=capacity)
        else:
            ambient = (drives*mid).real*np.sinc(1/steps)
            g, walls = lbm.thermal_mixed_wall_pair_3d(streamed,"x",("film","film"),ambient,
                film_coefficients=films,dt=dt,spacing=dx,volumetric_heat_capacity=capacity)
            if step >= 2*steps:
                # Pair each returned population with the local population that left the wall cell.
                surface = 4*np.array([(g[1,0,0,0]+post[2,0,0,0]).item(),
                                      (g[2,0,0,-1]+post[1,0,0,-1]).item()])
                wall_phasor += (2/steps)*mid.conjugate()*surface
        updated = lbm.thermal_macroscopic_3d(g)
        inward = np.array([q.item() for q in walls])
        residual = capacity*dx*(updated-field).sum().item()-dt*inward.sum()
        max_balance = max(max_balance,abs(residual))
        if step >= 2*steps:
            temp_phasor += (2/steps)*end.conjugate()*updated.numpy().ravel()
            flux_phasor += (2/steps)*mid.conjugate()*inward
        field = updated
    diagnostics = {}
    if films is not None:
        target = np.array([left, right])*np.sinc(1/steps)
        diagnostics = dict(film_coefficients_W_m2K=list(films),
            wall_temperature_phasor_errors_K=abs(wall_phasor-target).tolist(),
            wall_temperature_amplitude_errors_K=abs(abs(wall_phasor)-abs(target)).tolist(),
            wall_temperature_phase_errors_deg=abs(np.angle(wall_phasor/target,deg=True)).tolist(),
            wall_flux_amplitude_errors_W_m2=abs(abs(flux_phasor)-abs(exact_q)).tolist(),
            wall_flux_phase_errors_deg=abs(np.angle(flux_phasor/exact_q,deg=True)).tolist())
    return dict(**diagnostics,n=n, steps_per_period=steps, omega_rad_s=omega,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--films", action="store_true", help="drive unequal film coefficients with ambient temperature phasors")
    use_films = parser.parse_args().films
    for films in (((.45,1.8),(1.8,.45)) if use_films else (None,)):
        for factor in (1,2):
            previous = None
            for n in (8,16,32,64):
                row = measure(n,factor,films)
                errors = np.array([row[key] for key in metrics])
                row["orders"] = [] if previous is None else np.log2(previous/errors).tolist()
                row["passed"] = bool(row["max_interval_energy_residual_J_m2"] < 1e-13
                    and all(order > 1.2 for order in row["orders"])
                    and (n < 64 or (errors[0] < .005 and errors[1] < .009
                                    and row["wall_flux_phase_error_deg"] < .3)))
                if films is not None and n == 64:
                    row["passed"] &= (max(row["wall_temperature_phasor_errors_K"]) < .0015
                                      and max(row["wall_temperature_phase_errors_deg"]) < .25
                                      and row["wall_flux_phasor_error_W_m2"] < .0025)
                rows.append(row)
                previous = errors
    print(json.dumps(dict(reference=("Harmonic slab admittance with independently driven unequal Robin films" if use_films else "Harmonic 1-D slab with independent complex Dirichlet temperatures")+"; project final cycle of three onto exp(i omega t)",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_harmonic_dirichlet_heat.py"+(" --films" if use_films else ""),
        order_metrics=metrics,runs=rows,sources=[provenance(p) for p in (__file__,lbm.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
