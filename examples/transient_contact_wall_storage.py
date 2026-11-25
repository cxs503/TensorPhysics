"""Decay of a two-layer temperature mode with a resistive contact and two storing walls."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorfvm.cht import energy


def measure(n):
    dx, mid, decay, resistance = 1/n, n//2, 1.7, .35
    k = np.repeat([1., 4.], mid)
    capacity = np.repeat([1., 2.], mid)
    film = np.array([2., 5.])
    wall_capacity = np.array([.4, 1.2])
    # Propagate the continuum temperature and its rightward Fourier flux.
    def propagate(distance, temp, flux, conductivity, c):
        wave = np.sqrt(decay*c/conductivity)
        co, si = np.cos(wave*distance), np.sin(wave*distance)
        return (temp*co-flux*si/(conductivity*wave),
                flux*co+conductivity*wave*temp*si)

    start, west_flux = 1.2, -.3
    hot, contact_flux = propagate(.5,start,west_flux,1.,1.)
    cold = hot-resistance*contact_flux
    end, east_flux = propagate(.5,cold,contact_flux,4.,2.)
    walls = np.array([start,end])
    inward = np.array([west_flux,-east_flux])
    # Ambient heat = heat entering the slab + rate of change of wall energy.
    air = walls+(inward-decay*wall_capacity*walls)/film
    centers = (np.arange(mid)+.5)*dx
    means = np.r_[propagate(centers,start,west_flux,1.,1.)[0],
                  propagate(centers,cold,contact_flux,4.,2.)[0]]
    means *= np.sinc(np.sqrt(decay*capacity/k)*dx/(2*np.pi))
    times = np.linspace(0.,1.,2*n+1)
    signal = np.exp(-decay*times)
    bc = {side:dict(type="robin", values=lambda t,a=a:a*np.exp(-decay*t),
                    film_coefficient=h,wall_areal_heat_capacity=cw,initial_wall_temperature=w)
          for side,a,h,cw,w in zip(("west","east"),air,film,wall_capacity,walls)}
    faces = np.zeros((1,n-1))
    faces[0,mid-1] = resistance
    u, power = energy.solve_transient_conduction(np.full(n,dx),[1.],k,capacity,bc,times,means,
                          contact_resistance_x=faces,return_wall_power=True)
    field = u[:,0,:]
    q = np.column_stack((power["west"][:,0],power["east"][:,0]))
    surface = field[:,[0,-1]]+q*dx/(2*k[[0,-1]])
    contact = (field[:,mid-1]-field[:,mid])/(dx/(2*k[mid-1])+resistance+dx/(2*k[mid]))
    def integrated(rate):
        return np.diff(times)[:,None]*(rate[:-1]+rate[1:])/2
    ext = film*(signal[:,None]*air-surface)
    wall_balance = np.diff(surface,axis=0)*wall_capacity-integrated(ext-q)
    stored = dx*np.column_stack((field[:,:mid] @ capacity[:mid],field[:,mid:] @ capacity[mid:]))
    layer_balance = np.diff(stored,axis=0)-integrated(q+np.column_stack((-contact,contact)))
    # Independent estimate from wall storage, using one-sided endpoint derivatives.
    recovered = ext-wall_capacity*np.gradient(surface,times,axis=0,edge_order=2)
    cumulative = integrated(q).cumsum(axis=0)
    exact_heat = -np.expm1(-decay*times[1:,None])*inward/decay
    return dict(n=n,max_wall_storage_flux_error_W_m2=float(abs(recovered-signal[:,None]*inward).max()),
                max_cumulative_wall_heat_error_J_m2=float(abs(cumulative-exact_heat).max()),
                steps=len(times)-1,
                max_cell_mean_error_K=float(abs(field-signal[:,None]*means).max()),
                max_wall_temperature_error_K=float(abs(surface-signal[:,None]*walls).max()),
                max_wall_flux_error_W_m2=float(abs(q-signal[:,None]*inward).max()),
                max_contact_flux_error_W_m2=float(abs(contact-signal*contact_flux).max()),
                max_wall_step_residual_J_m2=float(abs(wall_balance).max()),
                max_layer_step_residual_J_m2=float(abs(layer_balance).max()))


if __name__ == "__main__":
    rows, previous = [], None
    for n in (16,32,64,128,256):
        row = measure(n)
        errors = np.array([row[k] for k in ("max_cell_mean_error_K","max_wall_temperature_error_K",
                                            "max_wall_flux_error_W_m2","max_contact_flux_error_W_m2",
                                            "max_cumulative_wall_heat_error_J_m2",
                                            "max_wall_storage_flux_error_W_m2")])
        orders = [] if previous is None else np.log2(previous/errors).tolist()
        row["orders"] = orders
        row["all_metrics_second_order_gate"] = all(p>1.85 for p in orders) if orders else None
        row["passed"] = (max(row["max_wall_step_residual_J_m2"],row["max_layer_step_residual_J_m2"])<1e-10
                         and all(p>floor for p,floor in zip(orders,(1.7,1.85,.95,1.85,1.85,1.25))))
        rows.append(row)
        previous = errors
    metrics = ["cell temperature","wall temperature","instantaneous wall flux","interface flux",
               "cumulative wall heat","wall flux from storage"]
    print(json.dumps(dict(runs=rows,order_metrics=metrics,sources=[provenance(f) for f in (__file__,energy.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
