"""D3Q7 slab with one fixed-temperature wall and one passive convective wall."""
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, tau, beta, fixed):
    dx, alpha, capacity = 1/n, .08, 1.7
    dt = (tau-.5)*dx**2/(4*alpha)
    steps = round(.390625/dt)
    k, amplitude, base = alpha*capacity, .6, .4
    # In the coordinate measured from the fixed wall, the mode is sin(beta*x).
    # Matching -k*T_x=h*(T-base) at x=1 selects the Robin coefficient.
    h = -k*beta/math.tan(beta)
    x = (torch.arange(n, dtype=torch.float64)+.5).reshape(1, 1, n)*dx
    coordinate = x if fixed == 0 else 1-x
    orient = 1-2*fixed
    factor = amplitude*np.sinc(beta*dx/(2*np.pi))
    mode = factor*torch.sin(beta*coordinate)
    field = base+mode
    zero = torch.zeros_like(field)
    gradient = torch.stack((orient*factor*beta*torch.cos(beta*coordinate), zero, zero))
    populations = lbm.thermal_gradient_initialization_3d(field, zero, zero, zero, gradient, tau_T=tau, spacing=dx)
    types = ("temperature", "film") if fixed == 0 else ("film", "temperature")
    films = tuple(h if kind == "film" else 0. for kind in types)
    rate = alpha*beta**2
    # Outward flux amplitudes indexed by physical location (west, east).
    outward = amplitude*k*beta*np.array([1., -math.cos(beta)])
    if fixed == 1:
        outward = outward[::-1]
    total = np.zeros(2)
    flux_error, heat_error = np.zeros(2), np.zeros(2)
    balance = temp_error = 0.
    for step in range(steps):
        post = lbm.thermal_collide_bgk_3d(populations, field, zero, zero, zero, tau)
        populations, inward = lbm.thermal_mixed_wall_pair_3d(
            lbm.thermal_stream_3d(post), "x", types, (base, base), film_coefficients=films,
            dt=dt, spacing=dx, volumetric_heat_capacity=capacity)
        updated = lbm.thermal_macroscopic_3d(populations)
        measured = -np.array([v.item() for v in inward])
        total += dt*measured
        stop = (step+1)*dt
        integral = -math.expm1(-rate*stop)/rate
        mean_decay = math.exp(-rate*step*dt)*(-math.expm1(-rate*dt))/(rate*dt)
        flux_error = np.maximum(flux_error, abs(measured-outward*mean_decay))
        heat_error = np.maximum(heat_error, abs(total-outward*integral))
        balance = max(balance, abs(capacity*dx*(updated-field).sum().item()+dt*measured.sum()))
        temp_error = max(temp_error, float((updated-base-mode*math.exp(-rate*stop)).abs().max()))
        field = updated
    return dict(cells=n, tau=tau, beta=beta, fixed_wall=("west", "east")[fixed], film_W_m2K=h,
                max_temperature_error_K=temp_error, max_wall_flux_errors_W_m2=flux_error.tolist(),
                max_wall_heat_errors_J_m2=heat_error.tolist(), final_outward_heat_J_m2=total.tolist(),
                exact_final_outward_heat_J_m2=(outward*integral).tolist(), max_balance_residual_J_m2=balance)


if __name__ == "__main__":
    runs = []
    for fixed in (0, 1):
        for tau in (.75, 1.5):
            for beta in (1.9, 2.3):
                previous = None
                for n in (8, 16, 32, 64):
                    row = measure(n, tau, beta, fixed)
                    errors = np.array([row["max_temperature_error_K"], *row["max_wall_flux_errors_W_m2"],
                                       *row["max_wall_heat_errors_J_m2"]])
                    orders = [] if previous is None else np.log2(previous/errors).tolist()
                    row.update(orders=orders, passed=bool(row["max_balance_residual_J_m2"] < 1e-13
                               and (not orders or min(orders) > 0.) and (n < 64 or min(orders) > 1.8)))
                    runs.append(row)
                    previous = errors
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(lbm.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_fixed_film_decay.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
