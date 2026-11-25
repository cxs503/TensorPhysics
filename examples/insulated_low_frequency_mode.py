"""Balanced cosine heating of an insulated unit rod with unit cross-section."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


def measure(n):
    x = np.linspace(0.,1.,n+1)
    width = np.diff(x)
    k, capacity = 2.,3.
    rate = k*np.pi**2
    amplitude = 1.+.4j
    mid = (x[:-1]+x[1:])/2
    source = amplitude*np.cos(np.pi*mid)*np.sinc(width/2)
    exact = source/rate
    lag_flux = -capacity*amplitude*np.sin(np.pi*x)/(np.pi*rate)
    # Integrate the continuum source against each linear basis function.
    nodes, weights = np.polynomial.legendre.leggauss(5)
    samples = amplitude*np.cos(np.pi*(mid[:,None]+width[:,None]*nodes/2))
    local = (width[:,None]/2)*(samples*weights) @ np.column_stack(((1-nodes)/2,(1+nodes)/2))
    load = np.r_[local[:,0],0.]+np.r_[0.,local[:,1]]
    elements = np.column_stack((np.arange(n),np.arange(1,n+1)))
    model = thermal.ThermalModel(torch.from_numpy(x[:,None]),torch.from_numpy(elements),k,1.,capacity,
        torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.float64),nodal_heat=torch.from_numpy(load))
    direct = thermal.solve_low_frequency_thermal(model,sparse=True)
    _,power = thermal.thermal_element_low_frequency_power(model,direct)
    dc = direct.steady.temperature[elements].mean(1).numpy()
    lag = direct.first_order.temperature[elements].mean(1).numpy()
    cuts = np.r_[power[:,0].numpy(),-power[-1,1].item()]
    steady,first = energy.solve_low_frequency_conduction(width,[1.],k,capacity,{},source=source,
                                                          return_interior_power=True)
    fvm_cuts = np.r_[first[1]["west"],first[2]["x"].ravel(),-first[1]["east"]]
    particle_source = amplitude*np.cos(np.pi*mid)
    options = dict(conductivity=np.full(n,k),volumetric_heat_capacity=capacity,backend="banded")
    particles,particle_lag = sph.solve_low_frequency_layered_slab(
        particle_source,1/n,(None,None),return_x_flux=True,**options)
    for name,u,v,q in (("FEM",dc,lag,cuts),("FVM",steady[0][0],first[0][0],fvm_cuts),
                       ("SPH",particles[0].numpy(),particle_lag[0].numpy(),particle_lag[1].numpy())):
        # The SPH particle samples the midpoint; FEM and FVM report cell means.
        target = particle_source/rate if name == "SPH" else exact
        remainders = []
        for omega in (.5,.25,.125):
            if name == "FEM":
                field = thermal.solve_harmonic_thermal(model,omega,sparse=True).temperature[elements].mean(1).numpy()
            elif name == "SPH":
                field = sph.solve_harmonic_layered_slab(particle_source,1/n,(None,None),omega,**options).numpy()
            else:
                field = energy.solve_harmonic_conduction(width,[1.],k,capacity,{},omega,source=source)[0][0]
            remainders.append(float(abs(field-u-1j*omega*v).max()))
        yield dict(solver=name,n=n,steady_temperature_error_K=float(abs(u-target).max()),
                   lag_temperature_error_K_s=float(abs(v+capacity*target/rate).max()),
                   lag_flux_error_W_s_m2=float(abs(q-lag_flux).max()),
                   steady_stored_energy_J=float(abs(capacity*width@u)),
                   lag_stored_energy_J_s=float(abs(capacity*width@v)),
                   frequencies_rad_s=[.5,.25,.125],harmonic_remainders_K=remainders,
                   frequency_orders=np.log2(np.array(remainders[:-1])/remainders[1:]).tolist())


if __name__ == "__main__":
    runs,previous = [],{}
    keys = ("steady_temperature_error_K","lag_temperature_error_K_s","lag_flux_error_W_s_m2")
    for n in (8,16,32,64,128):
        for row in measure(n):
            name = row["solver"]
            row["passed"] = (max(row["steady_stored_energy_J"],row["lag_stored_energy_J_s"]) < 1e-11
                             and min(row["frequency_orders"]) > 1.95)
            if name in previous:
                row["mesh_orders"] = {key:float(np.log2(previous[name][key]/row[key])) for key in keys
                                      if min(previous[name][key],row[key]) > 1e-12}
            if n == 128:
                row["passed"] &= (min(row["mesh_orders"].values()) > 1.9
                                  and max(row[key] for key in keys) < 1e-5)
            runs.append(row)
            previous[name] = row
    print(json.dumps(dict(runs=runs,sources=[provenance(f) for f in (__file__,thermal.__file__,energy.__file__,sph.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in runs) else 1)
