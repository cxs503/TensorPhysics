"""Check half-way D3Q7 film temperatures and heat flux against a Robin slab eigenmode."""
import argparse
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, tau, beta, amplitude, peak=.5):
    dx, alpha, capacity = 1/n, .08, 1.7
    k = alpha*capacity
    films = tuple(k*beta*math.tan(beta*distance) for distance in (peak, 1-peak))
    ambient = (.9,.15)
    insulated = peak in (0., 1.)
    if insulated:
        film_side = 1-int(peak)
        ambient = (ambient[film_side],)*2
        steady_flux = 0.
    else:
        steady_flux = (ambient[0]-ambient[1])/(1/films[0]+1/k+1/films[1])
    types = tuple("film" if h else "flux" for h in films)
    values = tuple(t if h else 0. for t,h in zip(ambient,films))
    slope = -steady_flux/k
    dt = (tau-.5)*dx**2/(4*alpha)
    x = (torch.arange(n,dtype=torch.float64)+.5).reshape(1,1,n)*dx
    ramp = ambient[0] if insulated else ambient[0]-steady_flux/films[0]+slope*x
    factor = amplitude*np.sinc(beta*dx/(2*np.pi))
    mode = factor*torch.cos(beta*(x-peak))
    field = ramp+mode
    zero = torch.zeros_like(field)
    grad = torch.stack((slope-factor*beta*torch.sin(beta*(x-peak)),zero,zero))
    g = lbm.thermal_gradient_initialization_3d(field,zero,zero,zero,grad,tau_T=tau,spacing=dx)
    rate = alpha*beta**2
    err_t = err_q = err_wall = balance = 0.
    outward_heat = heat_error = insulated_flux = 0.
    for step in range(round(.09765625/dt)):
        post = lbm.thermal_collide_bgk_3d(g,field,zero,zero,zero,tau)
        g, flux = lbm.thermal_mixed_wall_pair_3d(lbm.thermal_stream_3d(post),"x",types,values,
            film_coefficients=films,dt=dt,spacing=dx,volumetric_heat_capacity=capacity)
        updated = lbm.thermal_macroscopic_3d(g)
        decay = math.exp(-rate*step*dt)
        mean_decay = decay*(-math.expm1(-rate*dt))/(rate*dt)
        for side, direction in enumerate((1,-1)):
            h = films[side]
            base_wall = ambient[side] if insulated else ambient[side]-direction*steady_flux/h
            exact_wall = base_wall + amplitude*mean_decay*math.cos(beta*(side-peak))
            wall = 4*(g[1+side,...,0 if side==0 else -1]+post[2-side,...,0 if side==0 else -1])
            err_wall = max(err_wall,abs(wall.item()-exact_wall))
            err_q = max(err_q,abs(flux[side].item()-h*(ambient[side]-exact_wall)))
        if insulated:
            insulated_flux = max(insulated_flux, abs(flux[int(peak)].item()))
            outward_heat -= dt*flux[film_side].item()
            exact_heat = -capacity*amplitude*math.sin(beta)*math.expm1(-rate*(step+1)*dt)/beta
            heat_error = max(heat_error, abs(outward_heat-exact_heat))
        residual = capacity*dx*(updated-field).sum().item()-dt*(flux[0]+flux[1]).item()
        balance = max(balance,abs(residual))
        err_t = max(err_t,float((updated-ramp-mode*math.exp(-rate*(step+1)*dt)).abs().max()))
        field = updated
    extra = {}
    if peak != .5:
        elapsed = round(.09765625/dt)*dt
        relative_amplitude = float(((field-ramp)*mode).sum()/mode.square().sum())
        measured_rate = -math.log(relative_amplitude)/elapsed
        extra = dict(peak_x_m=peak, films_W_m2K=list(films), elapsed_s=elapsed,
                     exact_rate_per_s=rate, measured_rate_per_s=measured_rate,
                     decay_rate_error_per_s=abs(measured_rate-rate))
    else:
        extra["film_W_m2K"] = films[0]
    if insulated:
        extra.update(insulated_wall=("west","east")[int(peak)], max_insulated_flux_W_m2=insulated_flux,
                     final_outward_heat_J_m2=outward_heat, max_cumulative_heat_error_J_m2=heat_error)
    return dict(**extra, n=n,tau=tau,beta=beta,amplitude_K=amplitude,temperature_error_K=err_t,
                wall_temperature_error_K=err_wall,inward_flux_error_W_m2=err_q,balance_J_m2=balance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--insulated", action="store_true", help="verify one insulated wall and one film in either orientation")
    choice.add_argument("--asymmetric", action="store_true", help="use unequal passive films and fit the decay rate")
    args = parser.parse_args()
    asymmetric = args.asymmetric or args.insulated
    rows = []
    metrics = ["temperature_error_K","wall_temperature_error_K","inward_flux_error_W_m2"]
    if asymmetric:
        metrics.append("decay_rate_error_per_s")
    if args.insulated:
        metrics.append("max_cumulative_heat_error_J_m2")
    grids = (8,16,32,64) if asymmetric else (8,16,32)
    if args.insulated:
        betas, cases, flag = (.7,1.25), ((.6,0.),(.6,1.)), " --insulated"
    elif args.asymmetric:
        betas, cases, flag = (1.,2.), ((.6,.35),(.6,.65)), " --asymmetric"
    else:
        betas, cases, flag = (1.,2.4), ((0.,.5),(.6,.5)), ""
    for tau in (.75,1.5):
        for beta in betas:
            for amplitude, peak in cases:
                previous = None
                for n in grids:
                    row = measure(n,tau,beta,amplitude,peak)
                    errors = np.array([row[key] for key in metrics])
                    row["orders"] = [] if previous is None or not amplitude else np.log2(previous/errors).tolist()
                    passed = row["balance_J_m2"] < 1e-13
                    if args.insulated:
                        passed &= row["max_insulated_flux_W_m2"] < 1e-13
                    if amplitude:
                        passed &= all(order > 1.3 for order in row["orders"][:3])
                        if asymmetric and previous is not None:
                            passed &= row["orders"][3] > 0.
                        passed &= n < grids[-1] or np.all(errors[:3] < [1.5e-4, 8e-4, 7e-4])
                        if asymmetric and n == grids[-1]:
                            passed &= min(row["orders"]) > 1.8 and errors[3] < .001
                    else:
                        passed &= max(errors) < 1e-12
                    row["passed"] = bool(passed)
                    rows.append(row)
                    previous = errors
    print(json.dumps(dict(reference="Series-resistance steady slab plus cos(beta*(x-peak))*exp(-alpha*beta^2*t); h=k*beta*tan(beta*distance_to_peak)",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_film_slab.py" + flag,
        order_metrics=metrics,runs=rows,sources=[provenance(p) for p in (__file__,lbm.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
