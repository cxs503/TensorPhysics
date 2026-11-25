"""Transient SPH wall profiles and integrated heat for decaying slab eigenmodes."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def measure(n, steps):
    dx, k, capacity, end = 1/n, 2., 3., .0125
    x = (torch.arange(n, dtype=torch.float64)+.5)*dx
    cy, sy = torch.cos(2*torch.pi*x), torch.sin(2*torch.pi*x)
    cx, sx = torch.cos(torch.pi*x), torch.sin(torch.pi*x)
    rate = np.pi**2*k/capacity

    def exact(time):
        return (.3*np.exp(-rate*time)*sx
                + np.exp(-5*rate*time)*(cy[:,None]*sx + .4*sy[:,None]*cx))

    def west(time):
        return .4*np.exp(-5*rate*time)*sy

    times = np.linspace(0., end, steps+1)
    fields, heat = sph.solve_transient_slab_2d(exact(0.), times, dx,
        (west, lambda time: -west(time)), conductivity=k, volumetric_heat_capacity=capacity,
        return_x_heat=True)
    final = fields[-1]
    offsets = sph.lattice_offsets(dx, 1.3*dx)
    _, power = sph.stencil_rhs_with_wall_power_2d(final, offsets, dx, (west(end),-west(end)),
                                                   conductivity=k, power_by_row=True)
    raw = power/(dx*sph.lattice_diffusion_factor_2d(1.3, dx, mode="moment"))
    recovered = sph.recover_dirichlet_wall_flux_2d(final, dx, k, raw.mean(dim=1))
    # Both walls see the same inward derivative of sin(pi*x). The cosine
    # component drives their temperatures and has zero exact normal derivative.
    truth = -k*np.pi*(.3*np.exp(-rate*end)+np.exp(-5*rate*end)*cy)
    inward_heat = heat[:,0]-heat[:,-1]
    storage = capacity*fields.mean(dim=(1,2)).diff()
    exact_heat = 2*k*np.pi*.3*np.expm1(-rate*end)/rate
    metrics = dict(n=n, steps=len(times)-1, max_temperature_error_K=float(abs(final-exact(end)).max()),
                raw_wall_error_W_m2=float(abs(raw-truth).max()),
                recovered_wall_error_W_m2=float(abs(recovered-truth).max()),
                integrated_heat_error_J_m2=float(abs(inward_heat.sum()-exact_heat)),
                max_interval_balance_J_m2=float(abs(inward_heat-storage).max()))
    return metrics, (final, recovered, inward_heat.sum())


if __name__ == "__main__":
    rows = []
    keys = ("max_temperature_error_K","raw_wall_error_W_m2","recovered_wall_error_W_m2",
            "integrated_heat_error_J_m2")
    for n in (16,32,64,128):
        row, _ = measure(n,n//8)
        row["passed"] = row["max_interval_balance_J_m2"] < 1e-12
        if rows:
            row["orders"] = {key:float(np.log2(rows[-1][key]/row[key])) for key in keys}
            row["passed"] &= min(row["orders"][key] for key in
                                 (keys[0],keys[2],keys[3])) > 1.8
        if n == 128:
            row["passed"] &= (row[keys[0]] < 2e-4 and row[keys[2]] < .001
                              and row[keys[3]] < 5e-6)
        rows.append(row)
    # Holding the lattice fixed isolates the temporal discretization error.
    steps = (4,8,16,32,64)
    samples = [measure(64,count)[1] for count in steps]
    temporal = []
    for start in range(len(steps)-2):
        time_orders = {}
        for i,key in enumerate(("temperature","recovered_wall_flux","total_heat")):
            coarse,medium,fine = (sample[i] for sample in samples[start:start+3])
            time_orders[key] = float(torch.log2(abs(coarse-medium).max()/abs(medium-fine).max()))
        temporal.append(dict(n=64,steps=steps[start:start+3],orders=time_orders))
    temporal[-1]["passed"] = min(temporal[-1]["orders"].values())>1.8
    print(json.dumps(dict(runs=rows,temporal_refinement=temporal,sources=[provenance(__file__),provenance(sph.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if temporal[-1]["passed"] and all(row["passed"] for row in rows) else 1)
