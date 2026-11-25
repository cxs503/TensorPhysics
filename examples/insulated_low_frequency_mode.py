"""Balanced cosine heating of an insulated unit rod with unit cross-section."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensordg import transient_diffusion_2d as dg, poisson_2d_bc as dg_flux
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
    profiles = [("FEM",dc,lag,cuts),("FVM",steady[0][0],first[0][0],fvm_cuts),
                ("SPH",particles[0].numpy(),particle_lag[0].numpy(),particle_lag[1].numpy())]
    boundary = dict.fromkeys(("left","right","top","bottom"),"neumann")
    dg_options = dict(ncells=(n,2),capacity=capacity,conductivity=k,bc=boundary,
                      forcing=lambda x,y: amplitude*torch.cos(torch.pi*x))
    dg_metrics = {}
    dg_degrees = {"DG-P1":1,"DG-P2":2}
    for name,degree in dg_degrees.items():
        _,_,u0,u1 = dg.solve_low_frequency_diffusion_2d(**dg_options,degree=degree,complex_load=True)
        flux,_ = dg_flux.poisson_face_powers_2d(u1,conductivity=k,bc=boundary)
        # A section mean can be superconvergent. Also measure the L2
        # error throughout both transverse cells using tensor-product Gauss points.
        basis = np.polynomial.legendre.legvander(nodes,degree)
        truth = amplitude*np.cos(np.pi*(mid[:,None]+width[:,None]*nodes/2))/rate
        metrics = {}
        for key,modes,target in (("steady_field_L2_K",u0,truth),
                                 ("lag_field_L2_K_s",u1,-capacity*truth/rate)):
            values = np.einsum("xyij,ai,bj->xyab",modes.numpy(),basis,basis)
            square = abs(values-target[:,None,:,None])**2
            metrics[key] = float(np.sqrt(np.einsum("xyab,x,a,b->",square,width,weights,weights)/8))
        dg_metrics[name] = metrics
        profiles.append((name,u0[:,:,0,0].mean(1).numpy(),u1[:,:,0,0].mean(1).numpy(),
                         flux.sum(1).numpy()))
    for name,u,v,q in profiles:
        # The SPH particle samples the midpoint; FEM and FVM report cell means.
        target = particle_source/rate if name == "SPH" else exact
        remainders = []
        for omega in (.5,.25,.125):
            if name == "FEM":
                field = thermal.solve_harmonic_thermal(model,omega,sparse=True).temperature[elements].mean(1).numpy()
            elif name == "SPH":
                field = sph.solve_harmonic_layered_slab(particle_source,1/n,(None,None),omega,**options).numpy()
            elif name in dg_degrees:
                _,_,modes = dg.solve_harmonic_diffusion_2d(omega,**dg_options,degree=dg_degrees[name])
                field = modes[:,:,0,0].mean(1).numpy()
            else:
                field = energy.solve_harmonic_conduction(width,[1.],k,capacity,{},omega,source=source)[0][0]
            remainders.append(float(abs(field-u-1j*omega*v).max()))
        yield dict(**dg_metrics.get(name,{}),solver=name,n=n,steady_temperature_error_K=float(abs(u-target).max()),
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
                row["mesh_orders"] = {key:float(np.log2(previous[name][key]/row[key])) for key in (*keys,"steady_field_L2_K","lag_field_L2_K_s")
                                      if key in row and min(previous[name][key],row[key]) > 1e-12}
            if n == 128:
                row["passed"] &= (min(row["mesh_orders"].values()) > 1.9
                                  and max(row[key] for key in keys) < 1e-5)
            if n == 128 and name == "DG-P2":
                row["passed"] &= (row["mesh_orders"]["steady_field_L2_K"] > 2.8
                                  and row["mesh_orders"]["lag_field_L2_K_s"] > 2.8
                                  and row["lag_flux_error_W_s_m2"] < 1e-10)
            runs.append(row)
            previous[name] = row
    print(json.dumps(dict(runs=runs,sources=[provenance(f) for f in (__file__,thermal.__file__,energy.__file__,sph.__file__,dg.__file__,dg_flux.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in runs) else 1)
