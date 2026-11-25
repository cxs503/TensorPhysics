"""Harmonic slab with two independent ambient drives, surface storage and a bulk heater."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensordg import heat_1d
from tensorfem import thermal
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as sph


def measure(n,omega):
    k,c = 2.3,1.7
    film,cw = np.array([2.,5.]),np.array([.35,1.1])
    source = .6+.3j
    wave = np.sqrt(1j*omega*c/k)
    base = source/(1j*omega*c)
    left,right = .8+.2j,-.3+.4j
    faces = np.linspace(0.,1.,n+1)
    mid = (faces[:-1]+faces[1:])/2
    def temperature(x):
        return base+left*np.exp(-wave*x)+right*np.exp(-wave*(1-x))
    wall = temperature(np.array([0.,1.]))
    inward = k*wave*np.array([left-right*np.exp(-wave),right-left*np.exp(-wave)])
    ambient = wall+(inward+1j*omega*cw*wall)/film
    point = temperature(mid)
    z = wave/(2*n)
    mean = base+(point-base)*np.sinh(z)/z

    _,u,q,twall = heat_1d.solve_harmonic_heat_1d(omega,ncells=n,degree=2,conductivity=k,capacity=c,
        source=source,wall_types=("robin","robin"),wall_values=ambient,film_coefficients=film,
        wall_areal_heat_capacity=cw,return_wall_temperatures=True)
    rows = [("DG_P2",u[:,0].numpy(),mean,twall.numpy(),q[[0,-1]].numpy()*[1,-1])]

    u,trace,west,east = rod.solve_harmonic_stepped_rod(np.full(n,1/n),1.,k,c,ambient,omega,
        source=source,end_types=("robin","robin"),end_film_coefficient=film,end_areal_heat_capacity=cw,flux_order=2)
    rows.append(("FVM",u,mean,trace[[0,-1]],np.array([west[0],-east[-1]])))

    u,twall,influx = sph.solve_harmonic_layered_films(np.full(n,source),1/n,ambient,omega,
        conductivity=np.full(n,k),volumetric_heat_capacity=c,film_coefficients=film,
        wall_areal_heat_capacity=cw,backend="banded")
    rows.append(("SPH",u.numpy(),point,twall.numpy(),influx.numpy()))

    conn = torch.stack((torch.arange(n),torch.arange(1,n+1)),dim=1)
    model = thermal.ThermalModel(torch.from_numpy(faces[:,None]),conn,k,1.,c,torch.empty(0,dtype=torch.long),
        torch.empty(0,dtype=torch.float64),heat_source=source,convection_edges=torch.tensor([[0,0],[n,n]]),
        convection_coefficient=torch.from_numpy(film),ambient_temperature=torch.from_numpy(ambient),
        wall_areal_heat_capacity=torch.from_numpy(cw))
    fem = thermal.solve_harmonic_thermal(model,omega,sparse=True)
    local = thermal.thermal_element_nodal_power(model,fem.temperature,omega)
    rows.append(("FEM_LINE2",fem.temperature[conn].mean(dim=1).numpy(),mean,fem.temperature[[0,-1]].numpy(),
                 local[[0,-1],[0,1]].numpy()))

    for name,u,target,twall,influx in rows:
        external = film*(ambient-twall)
        storage = 1j*omega*cw*twall
        yield dict(solver=name,cells=n,omega_rad_s=omega,
                   temperature_sampling="center" if name=="SPH" else "cell_mean",
                   temperature_error_K=float(abs(u-target).max()),
                   surface_temperature_error_K=float(abs(twall-wall).max()),
                   material_flux_error_W_m2=float(abs(influx-inward).max()),
                   external_flux_error_W_m2=float(abs(external-film*(ambient-wall)).max()),
                   wall_storage_error_W_m2=float(abs(storage-1j*omega*cw*wall).max()),
                   wall_balance_W_m2=float(abs(external-storage-influx).max()),
                   bulk_balance_W_m2=float(abs(1j*omega*c*u.mean()-source-influx.sum())))


if __name__ == "__main__":
    rows = []
    for omega in (1.,10.,40.):
        previous = {}
        for n in (16,32,64,128):
            for row in measure(n,omega):
                name = row["solver"]
                errors = {k:v for k,v in row.items() if "_error_" in k}
                orders = {k:float(np.log2(previous[name][k]/v)) for k,v in errors.items()} if name in previous else {}
                min_order = 3.5 if name=="DG_P2" else 1.7
                passed = (max(row["wall_balance_W_m2"],row["bulk_balance_W_m2"])<1e-9
                          and all(p>min_order for p in orders.values()))
                if n==128:
                    passed &= (row["temperature_error_K"] < 2e-4
                               and row["material_flux_error_W_m2"] < 2e-3)
                rows.append(dict(**row,orders=orders,passed=bool(passed)))
                previous[name] = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
                     (__file__,heat_1d.__file__,thermal.__file__,rod.__file__,sph.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
