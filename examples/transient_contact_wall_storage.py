"""Decay of a two-layer temperature mode with a resistive contact and two storing walls."""
import argparse
import inspect
import json
from dataclasses import replace

import numpy as np
import torch

from tensorfem import thermal
import two_layer_conduction

from conduction_provenance import provenance
from tensorfvm.cht import energy


def measure(n, wall_flux_order):
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
    points = np.r_[propagate(centers,start,west_flux,1.,1.)[0],
                  propagate(centers,cold,contact_flux,4.,2.)[0]]
    means = points*np.sinc(np.sqrt(decay*capacity/k)*dx/(2*np.pi))
    times = np.linspace(0.,1.,2*n+1)
    signal = np.exp(-decay*times)
    if wall_flux_order=="SPH":
        from tensorsph.heat_conduction_2d import solve_transient_layered_slab

        contacts = np.zeros(n-1)
        contacts[mid-1] = resistance
        films = tuple((h,lambda t,a=a: a*np.exp(-decay*t)) for h,a in zip(film,air))
        u,q,surface,external,mean = solve_transient_layered_slab(points,times,dx,(None,None),
                conductivity=k,volumetric_heat_capacity=capacity,contact_resistance_x=contacts,
                method="sdirk2",x_wall_films=films,wall_areal_heat_capacity=wall_capacity,
                initial_wall_temperatures=walls,return_wall_temperatures=True,return_external_heat=True,
                return_x_heat=True,return_interval_temperature=True)
        u,q,surface,external,mean = (v.numpy() for v in (u,q,surface,external,mean))
        ref_interval = -np.diff(signal)/decay
        material_heat = q[:,[0,-1]]*np.array([1,-1])
        wall_residual = wall_capacity*np.diff(surface,axis=0)+material_heat-external
        stored = dx*(u*capacity).reshape(len(times),2,mid).sum(axis=2)
        layer_residual = np.diff(stored,axis=0)-material_heat+np.column_stack((q[:,mid],-q[:,mid]))
        return dict(solver="SPH",n=n,steps=len(times)-1,
                    max_interval_particle_error_K=float(abs(mean-(ref_interval/np.diff(times))[:,None]*points).max()),
                    max_particle_error_K=float(abs(u-signal[:,None]*points).max()),
                    max_wall_temperature_error_K=float(abs(surface-signal[:,None]*walls).max()),
                    max_cumulative_wall_heat_error_J_m2=float(abs((material_heat-ref_interval[:,None]*inward).cumsum(0)).max()),
                    max_cumulative_external_heat_error_J_m2=float(abs((external-ref_interval[:,None]*(inward-decay*wall_capacity*walls)).cumsum(0)).max()),
                    max_cumulative_contact_heat_error_J_m2=float(abs((q[:,mid]-ref_interval*contact_flux).cumsum()).max()),
                    max_wall_step_residual_J_m2=float(abs(wall_residual).max()),
                    max_layer_step_residual_J_m2=float(abs(layer_residual).max()))
    if wall_flux_order=="DG":
        from tensordg import solve_transient_diffusion_2d, poisson_face_powers_2d

        contact = np.zeros((n-1,2))
        contact[mid-1] = resistance
        opts = dict(bc=dict(left="robin",right="robin",bottom="neumann",top="neumann"),
                    conductivity=np.tile(k[:,None],(1,2)),contact_resistance={"x":contact},
                    wall_areal_heat_capacity=dict(zip(("left","right"),wall_capacity)))
        def films(amplitude):
            return dict(zip(("left","right"),zip(film,air*amplitude)))
        def initial(x,y):
            a,b = np.sqrt(decay*np.array([1.,2.])/np.array([1.,4.]))
            return torch.where(x<.5,start*torch.cos(a*x)-west_flux/a*torch.sin(a*x),
                        cold*torch.cos(b*(x-.5))-contact_flux/(4*b)*torch.sin(b*(x-.5)))
        _,_,state,avg,drive = solve_transient_diffusion_2d(initial,times,ncells=(n,2),degree=1,
                capacity=np.tile(capacity[:,None],(1,2)),robin_films=films(1.),**opts,
                load_amplitude=lambda t:np.exp(-decay*t),return_interval_means=True,return_load_means=True)
        modes = state[:,:,:,:,0].mean(2).numpy()
        field = modes[:,:,0]
        surface = np.stack((modes[:,0,0]-modes[:,0,1],modes[:,-1].sum(1)),axis=1)
        q = []
        for i,dt in enumerate(np.diff(times)):
            fx,_ = poisson_face_powers_2d(avg[i],temperature_rate=(state[i+1]-state[i])/dt,
                                          robin_films=films(float(drive[i])),**opts)
            q.append(fx[[0,mid,-1]].sum(1).numpy())
        q = np.array(q)
        dt = np.diff(times)
        wall_heat = dt[:,None]*q[:,[0,-1]]*np.array([1,-1])
        contact_heat = dt*q[:,1]
        wall_residual = wall_capacity*np.diff(surface,axis=0)+wall_heat-dt[:,None]*film*(
                              drive.numpy()[:,None]*air-.5*(surface[:-1]+surface[1:]))
        stored = dx*(field*capacity).reshape(len(times),2,mid).sum(2)
        layer_residual = np.diff(stored,axis=0)-wall_heat+np.column_stack((contact_heat,-contact_heat))
        ref_interval = -np.diff(signal)/decay
        return dict(solver="DG",n=n,steps=len(dt),max_cell_mean_error_K=float(abs(field-signal[:,None]*means).max()),
                    max_wall_temperature_error_K=float(abs(surface-signal[:,None]*walls).max()),
                    max_cumulative_wall_heat_error_J_m2=float(abs((wall_heat-ref_interval[:,None]*inward).cumsum(0)).max()),
                    max_cumulative_contact_heat_error_J_m2=float(abs((contact_heat-ref_interval*contact_flux).cumsum()).max()),
                    max_wall_step_residual_J_m2=float(abs(wall_residual).max()),
                    max_layer_step_residual_J_m2=float(abs(layer_residual).max()))
    if wall_flux_order is None:
        model, ends = two_layer_conduction.slab_fem_model(k,0.,0.,resistance,film,torch.from_numpy(capacity))
        model = replace(model,wall_areal_heat_capacity=torch.from_numpy(wall_capacity))
        xn = np.arange(mid+1)*dx
        nodal = np.r_[propagate(xn,start,west_flux,1.,1.)[0],propagate(xn,cold,contact_flux,4.,2.)[0]]
        loads = torch.zeros((len(times),len(model.nodes)),dtype=torch.float64)
        # Each end edge has unit area and two linear shape functions.
        loads[:,ends] = torch.from_numpy((signal[:,None]*air*film/2)[:,:,None])
        result = thermal.solve_transient_thermal(model,torch.from_numpy(times),
                       torch.from_numpy(np.tile(nodal,2)),theta=.5,sparse=True,nodal_heat_history=loads)
        traces = result.temperature.numpy().reshape(len(times),2,n+2).mean(axis=1)
        surface = traces[:,[0,-1]]
        field = np.column_stack([.5*(u[:,1:]+u[:,:-1]) for u in (traces[:,:mid+1],traces[:,mid+1:])])
        element_heat = thermal.thermal_element_nodal_heat_steps(model,result).numpy()
        wall_heat = np.column_stack((element_heat[:,0,[0,3]].sum(axis=1),element_heat[:,-1,[1,2]].sum(axis=1)))
        contact_heat = -element_heat[:,mid-1,[1,2]].sum(axis=1)
        dt = np.diff(times)
        exact_interval = -np.diff(signal)/decay
        external = film*(signal[:,None]*air-surface)
        wall_residual = wall_capacity*np.diff(surface,axis=0)+wall_heat-.5*dt[:,None]*(external[:-1]+external[1:])
        stored = dx*np.column_stack((field[:,:mid] @ capacity[:mid],field[:,mid:] @ capacity[mid:]))
        layer_residual = np.diff(stored,axis=0)-wall_heat+np.column_stack((contact_heat,-contact_heat))
        return dict(solver="FEM",n=n,steps=len(dt),
                    max_cell_mean_error_K=float(abs(field-signal[:,None]*means).max()),
                    max_wall_temperature_error_K=float(abs(surface-signal[:,None]*walls).max()),
                    max_step_mean_wall_flux_error_W_m2=float(abs((wall_heat-exact_interval[:,None]*inward)/dt[:,None]).max()),
                    max_step_mean_contact_flux_error_W_m2=float(abs((contact_heat-exact_interval*contact_flux)/dt).max()),
                    max_cumulative_wall_heat_error_J_m2=float(abs((wall_heat-exact_interval[:,None]*inward).cumsum(axis=0)).max()),
                    max_wall_step_residual_J_m2=float(abs(wall_residual).max()),
                    max_layer_step_residual_J_m2=float(abs(layer_residual).max()))
    bc = {side:dict(type="robin", values=lambda t,a=a:a*np.exp(-decay*t),
                    film_coefficient=h,wall_areal_heat_capacity=cw,initial_wall_temperature=w)
          for side,a,h,cw,w in zip(("west","east"),air,film,wall_capacity,walls)}
    faces = np.zeros((1,n-1))
    faces[0,mid-1] = resistance
    u, power = energy.solve_transient_conduction(np.full(n,dx),[1.],k,capacity,bc,times,means,
                          contact_resistance_x=faces,return_wall_power=True,wall_flux_order=wall_flux_order)
    field = u[:,0,:]
    q = np.column_stack((power["west"][:,0],power["east"][:,0]))
    surface = field[:,[0,-1]]+q*dx/(2*k[[0,-1]])
    if wall_flux_order == 2:
        surface = (2*dx*q/k[[0,-1]]+7*field[:,[0,-1]]-field[:,[1,-2]])/6
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
    return dict(solver="FVM",n=n,wall_flux_order=wall_flux_order,max_wall_storage_flux_error_W_m2=float(abs(recovered-signal[:,None]*inward).max()),
                max_cumulative_wall_heat_error_J_m2=float(abs(cumulative-exact_heat).max()),
                steps=len(times)-1,
                max_cell_mean_error_K=float(abs(field-signal[:,None]*means).max()),
                max_wall_temperature_error_K=float(abs(surface-signal[:,None]*walls).max()),
                max_wall_flux_error_W_m2=float(abs(q-signal[:,None]*inward).max()),
                max_contact_flux_error_W_m2=float(abs(contact-signal*contact_flux).max()),
                max_wall_step_residual_J_m2=float(abs(wall_balance).max()),
                max_layer_step_residual_J_m2=float(abs(layer_balance).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dg",action="store_true",help="also verify P1 DG heat transfer across the contact")
    parser.add_argument("--sph",action="store_true",help="also verify the SPH contact, wall storage and external heat")
    args = parser.parse_args()
    rows = []
    for order in (1,2):
        previous = None
        for n in (16,32,64,128,256):
            row = measure(n,order)
            errors = np.array([row[k] for k in ("max_cell_mean_error_K","max_wall_temperature_error_K",
                                                "max_wall_flux_error_W_m2","max_contact_flux_error_W_m2",
                                                "max_cumulative_wall_heat_error_J_m2",
                                                "max_wall_storage_flux_error_W_m2")])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row["orders"] = orders
            row["all_metrics_second_order_gate"] = all(p>1.85 for p in orders) if orders else None
            floors = (1.7,1.85,.95,1.85,1.85,1.25) if order == 1 else (1.8,)*6
            row["passed"] = (max(row["max_wall_step_residual_J_m2"],row["max_layer_step_residual_J_m2"])<1e-10
                             and all(p>floor for p,floor in zip(orders,floors)))
            rows.append(row)
            previous = errors
    previous = None
    for n in (16,32,64,128,256):
        row = measure(n,None)
        errors = {k:v for k,v in row.items() if "_error_" in k}
        orders = {} if previous is None else {k:np.log2(previous[k]/v) for k,v in errors.items()}
        row["orders"] = orders
        row["passed"] = (all(v>1.85 for v in orders.values()) and
                         max(row["max_layer_step_residual_J_m2"],row["max_wall_step_residual_J_m2"]) < 1e-10)
        rows.append(row)
        previous = errors
    sources = [__file__,energy.__file__,thermal.__file__,two_layer_conduction.__file__]
    if args.dg:
        from tensordg import solve_transient_diffusion_2d, poisson_face_powers_2d
        sources.extend(inspect.getfile(f) for f in (solve_transient_diffusion_2d,poisson_face_powers_2d))
    if args.sph:
        from tensorsph.heat_conduction_2d import solve_transient_layered_slab
        sources.append(inspect.getfile(solve_transient_layered_slab))
    for name,enabled in (("DG",args.dg),("SPH",args.sph)):
        if not enabled:
            continue
        previous = None
        for n in (8,16,32,64):
            row = measure(n,name)
            errors = {k:v for k,v in row.items() if "_error_" in k}
            orders = {k:np.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
            row.update(orders=orders,passed=bool(max(row["max_layer_step_residual_J_m2"],row["max_wall_step_residual_J_m2"])<1e-10
                       and (n<64 or min(orders.values())>1.8)))
            rows.append(row)
            previous = errors
    metrics = ["cell temperature","wall temperature","instantaneous wall flux","interface flux",
               "cumulative wall heat","wall flux from storage"]
    print(json.dumps(dict(runs=rows,order_metrics=metrics,sources=[provenance(f) for f in sources]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
