"""Frequency sweep for a unit slab heated by independent inward flux phasors."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensordg import heat_1d
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


def measure(n, omega, include_dem=False):
    k, capacity = 2., 3.
    west, east = 1.+.4j, -.3+.5j
    width = 1/n
    x = (np.arange(n)+.5)*width
    wave = np.sqrt(1j*omega*capacity/k)
    exact = (west*np.cosh(wave*(1-x))+east*np.cosh(wave*x))/(k*wave*np.sinh(wave))
    # Integrate the exact thermal wave over each FVM cell.
    half = wave*width/2
    cell_mean = exact*np.sinh(half)/half
    sides = {"west": dict(type="neumann", values=west), "east": dict(type="neumann", values=east)}
    # Four periodic rows suffice for the uniform transverse mode.
    fv, powers, interior = energy.solve_harmonic_conduction(
        np.full(n, width), np.full(4, width), k, capacity, sides, omega,
        periodic_y=True, return_interior_power=True,
    )
    particle, section = sph.solve_harmonic_slab_2d(
        torch.zeros((4,n)), width, (None,None), omega, conductivity=k,
        volumetric_heat_capacity=capacity, x_wall_inward_flux=(west,east), return_x_flux=True,
    )
    fv_flux = np.concatenate(([powers["west"].mean()/width],
                              interior["x"].mean(axis=0)/width, [-powers["east"].mean()/width]))
    _, dg, dg_flux = heat_1d.solve_harmonic_heat_1d(omega,ncells=n,degree=1,conductivity=k,capacity=capacity,
        wall_types=("neumann","neumann"),wall_values=(west,east))
    faces = np.arange(1,n)*width
    flux = (west*np.sinh(wave*(1-faces))-east*np.sinh(wave*faces))/np.sinh(wave)
    methods = [("FVM",fv,cell_mean,fv_flux),("SPH",particle.numpy(),exact,section.numpy()),
               ("DG_P1",dg[:,0].numpy()[None,:],cell_mean,dg_flux.numpy())]
    if include_dem:
        from tensordem.thermal import cell_centered_slab_network, harmonic_contact_temperatures, harmonic_contact_interval
        # Apply wall powers to zero-capacity surface nodes of a unit-area chain.
        r,c = cell_centered_slab_network(np.full(n,width),k,capacity)
        drive = np.zeros_like(c,dtype=complex)
        drive[[0,-1]] = (west,east)
        options = dict(t_left=None,t_right=None,nodal_power=drive,allow_massless=True)
        dem = harmonic_contact_temperatures(r,c,omega,**options)
        methods.append(("DEM",dem[None,1:-1],cell_mean,-np.diff(dem)/r))
        phase = np.exp(.31j)
        dt = .79/omega
        duration_factor = phase*np.expm1(1j*omega*dt)/(1j*omega)
        end,heat,mean = harmonic_contact_interval(r,c,omega,.31/omega,dt,(dem*phase).real,
                                         **options,return_interval_temperature=True)
        exact_heat = (np.r_[west,flux,-east]*duration_factor).real
        heat_balance = np.diff(heat)+c[1:-1]*(end[1:-1]-(dem[1:-1]*phase).real)
        interval = dict(interval_heat_error_J_m2=float(abs(heat-exact_heat).max()),
                        interval_temperature_error_K=float(abs(mean[1:-1]-(cell_mean*duration_factor/dt).real).max()),
                        interval_cell_balance_J_m2=float(abs(heat_balance).max()))
    for solver,field,ref,sections in methods:
        residual = 1j*omega*capacity*field.mean()-west-east
        # Sample at a fixed x/L=1/2 by averaging the two adjoining locations.
        ratio = field[:,n//2-1:n//2+1].mean()/ref[n//2-1:n//2+1].mean()
        row = dict(solver=solver, n=n, omega_rad_s=omega, penetration_depth_m=1/wave.real,
                   max_temperature_error_K=float(abs(field-ref).max()),
                   relative_amplitude_error=float(abs(abs(ratio)-1)),
                   phase_error_rad=float(abs(np.angle(ratio))), storage_residual_W_m2=float(abs(residual)))
        row["max_internal_flux_error_W_m2"] = float(abs(sections[1:-1]-flux).max())
        row["max_wall_flux_error_W_m2"] = float(max(abs(sections[0]-west), abs(sections[-1]+east)))
        row["max_column_balance_W_m2"] = float(abs(np.diff(sections)
                                           +1j*omega*capacity*width*field.mean(axis=0)).max())
        row["midplane_flux_phase_error_rad"] = float(abs(np.angle(sections[n//2]/flux[n//2-1])))
        if solver == "FVM":
            # Preserve the full facewise check in addition to the section means.
            row["max_internal_flux_error_W_m2"] = float(abs(interior["x"]/width-flux).max())
            row["max_wall_flux_error_W_m2"] = float(max(abs(powers["west"]/width-west).max(),
                                                        abs(powers["east"]/width-east).max()))
        if solver=="DEM":
            row.update(interval)
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-underresolved", action="store_true",
                        help="include grids below four cells per penetration depth")
    parser.add_argument("--dem",action="store_true",help="verify DEM phasors and integrated heat for a finite part of a cycle")
    args = parser.parse_args()
    rows = []
    for omega in (.5, 5., 50.):
        previous = {}
        for n in (16,32,64,128,256):
            for row in measure(n, omega,args.dem):
                row["cells_per_penetration_depth"] = n*row["penetration_depth_m"]
                if row["cells_per_penetration_depth"] < 4 and not args.include_underresolved:
                    continue
                name = row["solver"]
                metrics = ["max_temperature_error_K", "max_internal_flux_error_W_m2"]
                if name=="DEM":
                    metrics.extend(("interval_heat_error_J_m2","interval_temperature_error_K"))
                errors = np.array([row[key] for key in metrics])
                orders = np.log2(previous[name]/errors) if name in previous else []
                row["orders"] = dict(zip(metrics, orders))
                passed = (np.isfinite(errors).all() and all(order > 1.8 for order in orders)
                          and row["storage_residual_W_m2"] < 1e-10
                          and row["max_column_balance_W_m2"] < 1e-10
                          and row["max_wall_flux_error_W_m2"] < 1e-12)
                if name=="DEM":
                    passed &= row["interval_cell_balance_J_m2"]<1e-11
                if n >= 128:
                    passed &= max(row["relative_amplitude_error"], row["phase_error_rad"],
                                  row["midplane_flux_phase_error_rad"]) < 2e-3
                row["passed"] = bool(passed)
                previous[name] = errors
                rows.append(row)
    sources = [__file__,energy.__file__,sph.__file__,heat_1d.__file__]
    if args.dem:
        from tensordem.thermal import contact_chain
        sources.append(contact_chain.__file__)
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in sources]),indent=2,allow_nan=False))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
