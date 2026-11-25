"""Separate frequency and mesh refinement of the graded slab's DC response and thermal lag."""
import argparse
import inspect
import json
import math

import numpy as np

import harmonic_graded_bulk_exchange as study
from conduction_provenance import provenance


KEYS = ("temperature_K","contact_power_W_m2","temperature_lag_K_s","contact_power_lag_W_s_m2")


def differences(current,reference):
    return {key:float(np.max(abs(x-y))) for key,x,y in zip(KEYS,current,reference)}


def direct_fem(n,capacity,contact):
    m = n//2
    materials = np.repeat(np.array(study.layer_materials(contact)).T,m,axis=1)
    model,_ = study.graded_fem_model(*materials,.18 if contact else 0.,capacity,study.AMBIENT.real.copy())
    direct = study.thermal.solve_low_frequency_thermal(model,sparse=True)
    powers = study.thermal.thermal_element_low_frequency_power(model,direct)
    nodal,cells,cuts = [],[],[]
    for coefficient,power in zip((direct.steady,direct.first_order),powers):
        nodal.append(coefficient.temperature.reshape(2,-1).mean(0).real.numpy())
        # The FEM profile varies linearly across each cell, so the
        # midpoint temperature also equals its cell mean on this strip.
        cells.append(coefficient.temperature[model.elements].mean(1).real.numpy())
        cuts.append(float(-power[m-1,[1,2]].real.sum()))
    return {name:(u[0],cuts[0],u[1],cuts[1])
            for name,u in (("FEM",nodal),("FVM",cells),("SPH",cells))}


def direct_fvm(n,capacity,contact):
    dx,_,_,m,_,_,k,c,_,_,loss,contacts = study.slab_grid(n,contact)
    steady,first = study.energy.solve_low_frequency_conduction(
        np.full(n,dx),[1.],k,c,study.fvm_films(study.AMBIENT.real,capacity),
        heat_loss_coefficient=loss,contact_resistance_x=contacts[None,:],return_interior_power=True)
    return (steady[0][0].real,steady[2]["x"][0,m-1].real,
            first[0][0].real,first[2]["x"][0,m-1].real)


def direct_sph(n,capacity,contact):
    dx,_,_,m,_,_,k,c,_,_,loss,contacts = study.slab_grid(n,contact)
    pair = study.sph.solve_low_frequency_layered_films(
        np.zeros(n),dx,study.AMBIENT.real,film_coefficients=study.FILMS,conductivity=k,
        volumetric_heat_capacity=c,wall_areal_heat_capacity=capacity,contact_resistance_x=contacts,
        heat_loss_coefficient=loss,backend="banded")
    offsets = study.sph.lattice_offsets(dx,1.3*dx)
    factor = study.sph.lattice_diffusion_factor_2d(1.3,dx,mode="moment")
    response = []
    for field,walls,_ in pair:
        cut = study.sph.stencil_x_face_power_2d(field[None,:],offsets,dx,m,walls,
                        conductivity=k,layered_x=True,contact_resistance_x=contacts)/(dx*factor)
        response.extend([field.real.numpy(),cut.real.item()])
    return tuple(response)


def direct_dg(n,capacity):
    from tensordg import solve_low_frequency_diffusion_2d, poisson_face_powers_2d

    options = study.dg_options(capacity,study.AMBIENT.real)
    _,_,steady,lag = solve_low_frequency_diffusion_2d(ncells=(n,2),degree=1,capacity=study.C,
                                  reaction=lambda x,y:study.H0+study.SLOPE*x,**options)
    response = []
    for field,rate,ambient in ((steady,0*steady,study.AMBIENT.real),(lag,steady,np.zeros(2))):
        fx,_ = poisson_face_powers_2d(field,temperature_rate=rate,**study.dg_options(capacity,ambient))
        response.extend([field[:,:,0,0].mean(1).numpy(),fx[n//2].sum().item()])
    return tuple(response)


def campaign(contact,include_dg=False):
    rows = []
    for capacity in (np.zeros(2),np.array([.4,.9])):
        previous_mesh = {}
        for n in (16,32,64,128):
            direct = direct_fem(n,capacity,contact)
            native = {"FEM":direct["FEM"],"FVM":direct_fvm(n,capacity,contact),
                      "SPH":direct_sph(n,capacity,contact)}
            if include_dg:
                direct["DG"] = direct["FVM"]  # Both report cell means.
                native["DG"] = direct_dg(n,capacity)
            previous_response,previous_change = {},{}
            for omega in (.8,.4,.2,.1,.05,.025):
                for report,(field,cut),reference in study.measure(n,omega,capacity,contact,True,return_response=True,include_dg=include_dg):
                    name = report["solver"]
                    current = (field.real,cut.real,field.imag/omega,cut.imag/omega)
                    row = dict(cells=n,solver=name,omega_rad_s=omega,wall_capacity=capacity.tolist())
                    passed = (max(report["reference_discrepancies"].values())<1e-7 and
                              max(report["max_layer_balance_W_m2"],report["wall_balance_W_m2"])<1e-10)
                    if name in previous_response:
                        change = differences(current,previous_response[name])
                        row["frequency_change"] = change
                        if name in previous_change:
                            orders = {k:math.log2(previous_change[name][k]/v) for k,v in change.items()}
                            row["frequency_orders"] = orders
                            if omega==.025:
                                passed &= min(orders.values())>1.85
                        previous_change[name] = change
                    row["direct_frequency_errors"] = differences(current,native[name])
                    if omega==.025:
                        # DC and Im(response)/omega are both even functions of frequency.
                        limit = [(4*x-y)/3 for x,y in zip(current,previous_response[name])]
                        error = differences(limit,reference)
                        row["extrapolated_space_errors"] = error
                        cross = differences(limit,direct[name])
                        row["extrapolated_vs_direct_fem"] = cross
                        row["direct_space_errors"] = differences(native[name],reference)
                        match = differences(limit,native[name])
                        row["extrapolated_vs_direct_"+name.lower()] = match
                        passed &= max(match.values())<1e-7
                        if n==128:
                            passed &= max(row["direct_space_errors"].values())<1e-4
                        if name=="FEM":
                            passed &= max(cross.values())<1e-7
                        elif n==128:
                            passed &= max(cross.values())<1e-4
                        if name in previous_mesh:
                            orders = {k:math.log2(previous_mesh[name][k]/v) for k,v in error.items()}
                            row["mesh_orders"] = orders
                            if n==128:
                                passed &= min(orders.values())>1.7 and max(error.values())<1e-4
                        previous_mesh[name] = error
                    row["passed"] = bool(passed)
                    rows.append(row)
                    previous_response[name] = current
    return rows


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact",action="store_true")
    parser.add_argument("--dg",action="store_true",help="include SIPG for the continuous plate")
    args = parser.parse_args()
    contact = args.contact
    if args.dg and contact:
        parser.error("the DG case requires continuous contact")
    rows = campaign(contact,args.dg)
    sources = [provenance(f) for f in (__file__,study.__file__,inspect.getfile(study.slab_fem_model),
                                       study.thermal.__file__,study.energy.__file__,study.sph.__file__)]
    if args.dg:
        from tensordg import solve_low_frequency_diffusion_2d, poisson_face_powers_2d
        sources.extend(provenance(inspect.getfile(f)) for f in (solve_low_frequency_diffusion_2d,poisson_face_powers_2d))
    print(json.dumps(dict(contact=contact,sources=sources,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
