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


def campaign(contact):
    rows = []
    for capacity in (np.zeros(2),np.array([.4,.9])):
        previous_mesh = {}
        for n in (16,32,64,128):
            direct = direct_fem(n,capacity,contact)
            previous_response,previous_change = {},{}
            for omega in (.8,.4,.2,.1,.05,.025):
                for report,(field,cut),reference in study.measure(n,omega,capacity,contact,True,return_response=True):
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
                    if name=="FEM":
                        row["direct_frequency_errors"] = differences(current,direct[name])
                    if omega==.025:
                        # DC and Im(response)/omega are both even functions of frequency.
                        limit = [(4*x-y)/3 for x,y in zip(current,previous_response[name])]
                        error = differences(limit,reference)
                        row["extrapolated_space_errors"] = error
                        cross = differences(limit,direct[name])
                        row["extrapolated_vs_direct_fem"] = cross
                        if name=="FEM":
                            row["direct_space_errors"] = differences(direct[name],reference)
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
    contact = parser.parse_args().contact
    rows = campaign(contact)
    sources = [provenance(f) for f in (__file__,study.__file__,inspect.getfile(study.slab_fem_model),
                                       study.thermal.__file__,study.energy.__file__,study.sph.__file__)]
    print(json.dumps(dict(contact=contact,sources=sources,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
