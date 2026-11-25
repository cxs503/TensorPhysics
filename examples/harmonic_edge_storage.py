"""Local Fourier heat flow and surface storage under a transverse ambient-temperature mode."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal, thermal_analytic
from tensorfvm.cht import energy


def measure(n):
    dx, omega, k, capacity = 1/n, 3., 2., 1.5
    film = np.array([2., 5.])
    cw = np.array([.4, 1.2])
    b = np.pi
    wave = np.sqrt(b*b+1j*omega*capacity/k)
    faces = np.arange(n+1)*dx
    centers = (faces[:-1]+faces[1:])/2
    wall = np.exp(-wave*np.array([0.,1.]))
    inward = k*wave*wall*np.array([1.,-1.])
    ambient = wall+(inward+1j*omega*cw*wall)/film
    cosine = np.cos(b*faces)
    mesh = thermal_analytic.structured_quad_mesh(n,n,dtype=torch.float64)
    ids = torch.arange((n+1)**2).reshape(n+1,n+1)
    sides = ids[:,[0,-1]].T
    edges = torch.stack((sides[:,:-1],sides[:,1:]),dim=-1).reshape(-1,2)
    amb = torch.from_numpy(ambient[:,None]*cosine)
    edge_values = torch.stack((amb[:,:-1],amb[:,1:]),dim=-1).reshape(-1,2)
    model = thermal.ThermalModel(mesh.nodes,mesh.elements,k,1.,capacity,torch.empty(0,dtype=torch.long),
              torch.empty(0,dtype=torch.float64),element_type="q4",convection_edges=edges,
              convection_coefficient=torch.from_numpy(film.repeat(n)),ambient_temperature=edge_values,
              wall_areal_heat_capacity=torch.from_numpy(cw.repeat(n)))
    fem = thermal.solve_harmonic_thermal(model,omega,sparse=True)
    local = thermal.thermal_element_nodal_power(model,fem.temperature,omega)
    powers = torch.zeros_like(fem.temperature).scatter_add_(0,mesh.elements.flatten(),local.flatten())[sides].numpy()
    surface = fem.temperature[sides].numpy()
    area = np.full(n+1,dx)
    area[[0,-1]] /= 2
    # Linear boundary hat testing of the exact cosine includes half hats at both ends.
    projected = cosine*np.sinc(dx/2)**2
    external = film[:,None]*(amb.numpy()-surface)-1j*omega*cw[:,None]*surface
    integrated = np.zeros_like(external)
    integrated[:,:-1] += dx*(2*external[:,:-1]+external[:,1:])/6
    integrated[:,1:] += dx*(external[:,:-1]+2*external[:,1:])/6
    yield dict(solver="FEM",n=n,temperature_error_K=float(abs(fem.temperature.numpy().reshape(n+1,n+1)
               -cosine[:,None]*np.exp(-wave*faces)).max()),
               wall_temperature_error_K=float(abs(surface-wall[:,None]*cosine).max()),
               wall_flux_error_W_m2=float(abs(powers/area-inward[:,None]*projected).max()),
               wall_balance_W=float(abs(integrated-powers).max()))

    widths = np.full(n,dx)
    ymean = np.cos(b*centers)*np.sinc(dx/2)
    z = .5*dx*wave
    xmean = np.exp(-wave*centers)*np.sinh(z)/z
    bc = {s:dict(type="robin",film_coefficient=h,wall_areal_heat_capacity=c,values=a*ymean)
          for s,h,c,a in zip(("west","east"),film,cw,ambient)}
    fv, reported = energy.solve_harmonic_conduction(widths,widths,k,capacity,bc,omega)
    flux = np.stack([reported["west"],reported["east"]])/dx
    surface = fv[:,[0,-1]].T+flux*dx/(2*k)
    residual = film[:,None]*(ambient[:,None]*ymean-surface)-1j*omega*cw[:,None]*surface-flux
    yield dict(solver="FVM",n=n,temperature_error_K=float(abs(fv-ymean[:,None]*xmean).max()),
               wall_temperature_error_K=float(abs(surface-wall[:,None]*ymean).max()),
               wall_flux_error_W_m2=float(abs(flux-inward[:,None]*ymean).max()),
               wall_balance_W=float(dx*abs(residual).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-control",action="store_true",help="also measure the 8-to-16 refinement")
    args = parser.parse_args()
    rows, previous = [], {}
    grids = (8,16,32,64,128) if args.coarse_control else (16,32,64,128)
    for n in grids:
        for row in measure(n):
            name = row["solver"]
            errors = {k:v for k,v in row.items() if "_error_" in k}
            orders = {} if name not in previous else {k:np.log2(previous[name][k]/v) for k,v in errors.items()}
            row.update(orders=orders,passed=all(p>1.8 for p in orders.values()) and row["wall_balance_W"]<1e-11)
            previous[name] = errors
            rows.append(row)
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
                     (__file__,thermal.__file__,thermal_analytic.__file__,energy.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
