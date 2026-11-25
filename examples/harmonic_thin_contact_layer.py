"""Fixed-temperature wall response of a thin coating with a resistive contact."""
import json

import numpy as np
import torch
from scipy.linalg import expm

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import energy


def reference(widths, k, c, contact_face, resistance, omega):
    # State [T, q, integral(T dx)] travels rightwards. Reset the integral at
    # each cell inlet to obtain the temperature mean without quadrature.
    jump = np.eye(3)
    jump[0,1] = -resistance
    layers = [expm(d*np.array([[0.,-1/material,0.],[-1j*omega*cap,0.,0.],[1.,0.,0.]]))
              for d,material,cap in zip(widths,k,c)]
    transfer = np.eye(3)
    for i,layer in enumerate(layers,1):
        transfer = layer @ transfer
        if i == contact_face:
            transfer = jump @ transfer
    # Both ends oscillate with unit amplitude and zero relative phase.
    q0 = (1-transfer[0,0])/transfer[0,1]
    state = np.array([1.,q0,0.])
    means = []
    flux = [q0]
    for i,(layer,d) in enumerate(zip(layers,widths),1):
        state[2] = 0.
        state = layer @ state
        means.append(state[2]/d)
        flux.append(state[1])
        if i == contact_face:
            state = jump @ state
    return np.array(means),np.array(flux)


def fem_response(widths,k,c,contact_face,resistance,omega):
    x = np.r_[0.,np.cumsum(widths)]
    starts = np.arange(len(widths))
    contact = {}
    if resistance:
        x = np.insert(x,contact_face,x[contact_face])
        starts[contact_face:] += 1
        contact = dict(contact_pairs=torch.tensor([[contact_face,contact_face+1]]),
                       contact_conductance=1/resistance)
    elements = torch.from_numpy(np.column_stack((starts,starts+1)))
    model = thermal.ThermalModel(torch.from_numpy(x[:,None]),elements,torch.from_numpy(k),
        1.,torch.from_numpy(c),torch.tensor([0,len(x)-1]),torch.ones(2,dtype=torch.float64),**contact)
    state = thermal.solve_harmonic_thermal(model,omega,sparse=True).temperature
    power = thermal.thermal_element_nodal_power(model,state,omega).numpy()
    low = thermal.solve_low_frequency_thermal(model,sparse=True)
    _,lag_power = thermal.thermal_element_low_frequency_power(model,low)
    # Element endpoint powers point inward. Orient every cut to the right.
    cuts = np.r_[power[:,0],-power[-1,1]]
    lag_cuts = np.r_[lag_power[:,0].numpy(),-lag_power[-1,1].item()]
    jump = state[elements[contact_face-1,1]]-state[elements[contact_face,0]]
    extra = dict(max_interior_power_disagreement_W=float(abs(power[:-1,1]+power[1:,0]).max()),
                 jump_K=jump.item())
    return state[elements].mean(1).numpy(),cuts,low.first_order.temperature[elements].mean(1).numpy(),lag_cuts,extra


def measure(m, order, resistance):
    widths = np.r_[np.full(m,.0625/m),np.full(8*m,.9375/(8*m))]
    k = np.repeat([.7,3.], [m,8*m])
    c = np.repeat([2.,1.3], [m,8*m])
    omega = 4.
    extra = {}
    if order is None:
        temperature,measured,lag_temperature,measured_lag,extra = fem_response(widths,k,c,m,resistance,omega)
    else:
        contacts = np.zeros(len(widths)-1)
        contacts[m-1] = resistance
        bc = {s:dict(type="dirichlet",values=1.) for s in ("west","east")}
        opts = dict(return_interior_power=True,dirichlet_flux_order=order,contact_resistance_x=contacts[None,:])
        temperature,power,interior = energy.solve_harmonic_conduction(widths,[1.],k,c,bc,omega,**opts)
        _,lag = energy.solve_low_frequency_conduction(widths,[1.],k,c,bc,**opts)
        lag_temperature = lag[0]
        measured = np.r_[power["west"],interior["x"].ravel(),-power["east"]]
        measured_lag = np.r_[lag[1]["west"],lag[2]["x"].ravel(),-lag[1]["east"]]
    exact,flux = reference(widths,k,c,m,resistance,omega)
    # At zero frequency T=1. The first flux coefficient satisfies q1'=-c.
    # The integral of q1/k and the contact temperature drop must sum to zero.
    stored = np.r_[0.,np.cumsum(c*widths)]
    q1 = (sum((stored[:-1]+stored[1:])*widths/(2*k))+resistance*stored[m])/(sum(widths/k)+resistance)
    lag_means = []
    surface_lag = 0.
    for i,(d,material,cap) in enumerate(zip(widths,k,c),1):
        inlet = q1-stored[i-1]
        lag_means.append(surface_lag-(inlet*d/2-cap*d*d/6)/material)
        surface_lag -= (inlet*d-cap*d*d/2)/material
        if i == m:
            surface_lag -= resistance*(q1-stored[m])
    if order is None:
        extra["contact_jump_error_K"] = float(abs(extra.pop("jump_K")-resistance*flux[m]))
    residual = 1j*omega*c*widths*temperature.ravel()+np.diff(measured)
    return dict(**extra,solver="FEM-LINE2" if order is None else "FVM",coating_cells=m,substrate_cells=8*m,dirichlet_flux_order=order,
                contact_resistance_m2K_W=resistance,
                temperature_error_K=float(abs(temperature-exact).max()),
                wall_flux_error_W_m2=float(abs((measured-flux)[[0,-1]]).max()),
                contact_flux_error_W_m2=float(abs(measured[m]-flux[m])),
                lag_temperature_error_K_s=float(abs(lag_temperature-lag_means).max()),
                lag_flux_error_W_s_m2=float(abs(measured_lag-(q1-stored)).max()),
                max_cell_power_residual_W=float(abs(residual).max()))


if __name__ == "__main__":
    runs = []
    for resistance in (0.,.28):
        for order in (1,2,None):
            previous = None
            for m in (1,2,4,8,16,32):
                row = measure(m,order,resistance)
                if previous:
                    row["orders"] = {key:float(np.log2(previous[key]/row[key])) for key in
                        ("temperature_error_K","wall_flux_error_W_m2","contact_flux_error_W_m2",
                         "lag_temperature_error_K_s","lag_flux_error_W_s_m2")
                        if min(previous[key],row[key]) > 1e-11}
                row["passed"] = row["max_cell_power_residual_W"] < 1e-10
                if order is None:
                    row["passed"] &= row["max_interior_power_disagreement_W"] < 1e-10
                if m == 32:
                    row["passed"] &= (min(row["orders"].values()) > 1.8
                                      and row["temperature_error_K"] < 1e-5
                                      and row["wall_flux_error_W_m2"] < 2e-5
                                      and row["contact_flux_error_W_m2"] < 1e-5
                                      and row["lag_temperature_error_K_s"] < 1e-5)
                    # The half-cell resistances integrate the affine lag flux
                    # by the trapezoid rule exactly for this constant DC field.
                    lag_limit = 1e-11 if order in (None,1) else 3e-6
                    row["passed"] &= row["lag_flux_error_W_s_m2"] < lag_limit
                if order is None and m == 32:
                    row["passed"] &= row["contact_jump_error_K"] < 3e-6
                runs.append(row)
                previous = row
    print(json.dumps(dict(runs=runs,sources=[provenance(__file__),provenance(energy.__file__),provenance(thermal.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in runs) else 1)
