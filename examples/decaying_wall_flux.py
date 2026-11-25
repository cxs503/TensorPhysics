"""Transient SPH and FVM wall profiles and integrated heat for decaying slab eigenmodes."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


def measure(n, steps, solver="SPH"):
    dx, k, capacity, end = 1/n, 2., 3., .0125
    x = (torch.arange(n, dtype=torch.float64)+.5)*dx
    cy, sy = torch.cos(2*torch.pi*x), torch.sin(2*torch.pi*x)
    cx, sx = torch.cos(torch.pi*x), torch.sin(torch.pi*x)
    # A trigonometric mode averaged over a cell acquires one sinc factor
    # per varying coordinate; wall traces are averaged only along y.
    if solver == "FVM":
        cy, sy = cy*np.sinc(dx), sy*np.sinc(dx)
        cx, sx = cx*np.sinc(dx/2), sx*np.sinc(dx/2)
    rate = np.pi**2*k/capacity

    def exact(time):
        return (.3*np.exp(-rate*time)*sx
                + np.exp(-5*rate*time)*(cy[:,None]*sx + .4*sy[:,None]*cx))

    def west(time):
        return .4*np.exp(-5*rate*time)*sy

    times = np.linspace(0., end, steps+1)
    extra = {}
    if solver == "FVM":
        extra["startup"] = "two backward Euler half steps"
        widths = np.full(n,dx)
        bc = {"west":dict(type="dirichlet",values=west),
              "east":dict(type="dirichlet",values=lambda time: -west(time))}
        states, powers, heats = energy.solve_transient_conduction(
            widths,widths,k,capacity,bc,times,exact(0.).numpy(),periodic_y=True,
            return_wall_power=True,return_face_heat=True,rannacher=True)
        fields = torch.from_numpy(states)
        final = fields[-1]
        raw = torch.from_numpy(np.stack([powers[side][-1]/dx for side in ("west","east")]))
        wall, interior = heats["wall"], heats["interior"]
        inward_heat = torch.from_numpy((wall["west"]+wall["east"]).sum(axis=1))
        # Both periodic seam contributions enter their respective end rows.
        hx = np.concatenate((wall["west"][:,:,None],interior["x"],-wall["east"][:,:,None]),axis=2)
        hy = np.concatenate((wall["south"][:,None,:],interior["y"],-wall["north"][:,None,:]),axis=1)
        residual = capacity*dx**2*np.diff(states,axis=0)+np.diff(hx,axis=2)+np.diff(hy,axis=1)
        extra["max_cell_heat_residual_J"] = float(abs(residual).max())
        # Each face samples a transverse cell average. Integrate each
        # exponential analytically over the requested output intervals.
        def decay_integral(r):
            return np.exp(-r*times[:-1])*(-np.expm1(-r*np.diff(times)))/r
        exact_face_heat = -k*np.pi*dx*(.3*decay_integral(rate)[:,None]
                          + decay_integral(5*rate)[:,None]*cy.numpy())
        interval_errors = np.maximum.reduce([
            np.max(abs(wall[side]-exact_face_heat),axis=1)/(dx*np.diff(times))
            for side in ("west","east")])
        extra["max_face_mean_flux_error_W_m2"] = float(interval_errors.max())
        extra["worst_interval_start_s"] = float(times[interval_errors.argmax()])
        extra["last_face_mean_flux_error_W_m2"] = float(interval_errors[-1])
        reported = raw
    else:
        fields, heat = sph.solve_transient_slab_2d(exact(0.), times, dx,
            (west, lambda time: -west(time)), conductivity=k, volumetric_heat_capacity=capacity,
            return_x_heat=True)
        final = fields[-1]
        offsets = sph.lattice_offsets(dx, 1.3*dx)
        _, power = sph.stencil_rhs_with_wall_power_2d(final, offsets, dx, (west(end),-west(end)),
                                                       conductivity=k, power_by_row=True)
        raw = power/(dx*sph.lattice_diffusion_factor_2d(1.3, dx, mode="moment"))
        recovered = sph.recover_dirichlet_wall_flux_2d(final, dx, k, raw.mean(dim=1))
        reported = recovered
        inward_heat = heat[:,0]-heat[:,-1]
    # Both walls see the same inward derivative of sin(pi*x). The cosine
    # component drives their temperatures and has zero exact normal derivative.
    truth = -k*np.pi*(.3*np.exp(-rate*end)+np.exp(-5*rate*end)*cy)
    storage = capacity*fields.mean(dim=(1,2)).diff()
    exact_heat = 2*k*np.pi*.3*np.expm1(-rate*end)/rate
    metrics = dict(**extra, solver=solver, n=n, steps=len(times)-1, max_temperature_error_K=float(abs(final-exact(end)).max()),
                raw_wall_error_W_m2=float(abs(raw-truth).max()),
                integrated_heat_error_J_m2=float(abs(inward_heat.sum()-exact_heat)),
                max_interval_balance_J_m2=float(abs(inward_heat-storage).max()))
    if solver == "SPH":
        metrics["recovered_wall_error_W_m2"] = float(abs(reported-truth).max())
    return metrics, (final, reported, inward_heat.sum())


def campaign(solver):
    rows = []
    keys = ("max_temperature_error_K","raw_wall_error_W_m2","recovered_wall_error_W_m2",
            "integrated_heat_error_J_m2")
    wall_key = keys[2] if solver == "SPH" else keys[1]
    for n in (16,32,64,128):
        row, _ = measure(n,n//4 if solver == "FVM" else n//8,solver)
        row["passed"] = row["max_interval_balance_J_m2"] < 1e-12
        if solver == "FVM":
            row["passed"] &= row["max_cell_heat_residual_J"] < 1e-12
        if rows:
            row["orders"] = {key:float(np.log2(rows[-1][key]/row[key])) for key in keys if key in row}
            if solver == "FVM":
                for key in ("max_face_mean_flux_error_W_m2","last_face_mean_flux_error_W_m2"):
                    row["orders"][key] = float(np.log2(rows[-1][key]/row[key]))
            row["passed"] &= min(row["orders"][key] for key in
                                 (keys[0],wall_key,keys[3])) > 1.8
        if n == 128:
            if solver == "FVM":
                row["passed"] &= (row["max_face_mean_flux_error_W_m2"] < .025
                                  and row["orders"]["max_face_mean_flux_error_W_m2"] > .9
                                  and row["orders"]["last_face_mean_flux_error_W_m2"] > 1.8)
            row["passed"] &= (row[keys[0]] < 2e-4 and row[wall_key] < .001
                              and row[keys[3]] < 5e-6)
        rows.append(row)
    # Holding the lattice fixed isolates the temporal discretization error.
    steps = (4,8,16,32,64)
    samples = [measure(64,count,solver)[1] for count in steps]
    temporal = []
    for start in range(len(steps)-2):
        time_orders = {}
        for i,key in enumerate(("temperature",wall_key.removesuffix("_error_W_m2"),"total_heat")):
            coarse,medium,fine = (sample[i] for sample in samples[start:start+3])
            time_orders[key] = float(torch.log2(abs(coarse-medium).max()/abs(medium-fine).max()))
        temporal.append(dict(solver=solver,n=64,steps=steps[start:start+3],orders=time_orders))
    temporal[-1]["passed"] = min(temporal[-1]["orders"].values())>1.8
    return rows, temporal


if __name__ == "__main__":
    rows, temporal = [], []
    for solver in ("SPH","FVM"):
        run, refinement = campaign(solver)
        rows.extend(run)
        temporal.extend(refinement)
    print(json.dumps(dict(runs=rows,temporal_refinement=temporal,
                          sources=[provenance(f) for f in (__file__,sph.__file__,energy.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(row.get("passed",True) for row in rows+temporal) else 1)
