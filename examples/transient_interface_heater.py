"""Heat partition from a storing, resistive heater between rods of unequal diffusivity."""
import argparse
import inspect
from dataclasses import replace
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import rod
import two_layer_conduction


K = np.array([1.,2.])
CAPACITY = np.array([4.,1.])
STORAGE, RESISTANCE = .8,.24
POWER, DECAY = 3.,4.
SAMPLES = np.array([.0125,.025,.05,.1,.2,.4,.8])


def reference(area, fraction, degree, storage_fraction, initial, times=SAMPLES, input_power=POWER):
    # Fixed Talbot contour. Each transform is from the continuum rod solution
    # T(x,s)=T(0,s)*sinh(w*(L-x))/sinh(w*L) measured from the sheet.
    theta = np.arange(1,degree)*np.pi/degree
    cot = 1/np.tan(theta)
    delta = .4*degree*np.r_[1.,theta*(cot+1j)]
    weights = np.exp(delta)*np.r_[.5,1+1j*(theta*(1+cot**2)-cot)]
    s = delta[None,:,None]/times[:,None,None]
    wave = np.sqrt(s*CAPACITY/K)
    admittance = area*K*wave/np.tanh(.5*wave)
    # Each trace has its own storage admittance. The contact couples them.
    owner = np.argmax(area)
    admittance += s*STORAGE*np.array([storage_fraction,1-storage_fraction])
    left,right = admittance[...,0],admittance[...,1]
    g = min(area)/RESISTANCE
    determinant = left*right+g*(left+right)
    # The initial branch profile is linear from each trace to its cold wall.
    # T_initial(x)/s is a particular solution of the transformed heat equation.
    base = np.asarray(initial)/s
    conductance = area*K/.5
    load = (input_power/(s+DECAY))*np.array([fraction,1-fraction])+(admittance-conductance)*base
    total = load.sum(axis=-1)
    trace = np.stack((g*total+right*load[...,0],g*total+left*load[...,1]),axis=-1)/determinant[...,None]
    sheet = trace[...,owner:owner+1]
    wall_heat = (area*K*wave*(trace-base)/np.sinh(.5*wave)+conductance*base)/s
    jump = trace[...,:1]-trace[...,1:]
    transforms = np.concatenate((sheet,jump,wall_heat,trace,g*jump/s),axis=-1)
    return (.4/times[:,None])*(weights[None,:,None]*transforms).sum(axis=1).real


def fvm(n,area,times,fraction,storage_fraction,initial):
    interface = np.zeros(2*n-1)
    interface[n-1] = 1.
    position = (np.arange(n)+.5)/n
    means = np.r_[initial[0]*position,initial[1]*position[::-1]]
    _,faces,_,_,(west,east),heat = rod.solve_transient_stepped_rod(
        np.full(2*n,.5/n),area.repeat(n),K.repeat(n),CAPACITY.repeat(n),[0.,0.],times,means,
        interface_heat_capacity=STORAGE*interface,interface_capacity_left_fraction=storage_fraction,
        initial_trace_temperatures=initial,
        contact_resistance=RESISTANCE*interface,interface_power=lambda t:POWER*np.exp(-DECAY*t)*interface,
        interface_left_fraction=fraction,flux_order=2,method="sdirk2",return_traces=True,return_step_heat=True)
    wall = np.vstack((np.zeros(2),-heat["end_input"].cumsum(axis=0)))
    return np.column_stack((faces[:,n],west[:,n-1]-east[:,n-1],wall,west[:,n-1],east[:,n-1],
                            np.r_[0.,np.cumsum(heat["contact"][:,n-1])]))


def fem(n,area,times,fraction,storage_fraction,initial):
    model,ends = two_layer_conduction.slab_fem_model(K.repeat(n),0.,0.,RESISTANCE/min(area),None,
                                torch.from_numpy(CAPACITY.repeat(n)))
    stride = len(model.nodes)//2
    left_nodes = torch.tensor([n,n+stride])
    storage = torch.zeros_like(model.nodes[:,0])
    storage[left_nodes] = STORAGE*storage_fraction/2
    storage[left_nodes+1] = STORAGE*(1-storage_fraction)/2
    model = replace(model,thickness=torch.from_numpy(area.repeat(n)),nodal_heat_capacity=storage)
    # Dampen the start-up at the massless loaded trace with two backward
    # Euler half-steps. Evaluate its prescribed power at the actual stage times.
    def advance(grid, initial, theta):
        t = torch.from_numpy(grid)
        load = storage.new_zeros((len(t),len(storage)))
        power = POWER/2*torch.exp(-DECAY*t[:,None])
        load[:,left_nodes] = fraction*power
        load[:,left_nodes+1] = (1-fraction)*power
        return thermal.solve_transient_thermal(model,t,initial,theta=theta,sparse=True,
                                               nodal_heat_history=load)
    position = np.arange(n+1)/n
    nodal = np.r_[initial[0]*position,initial[1]*position[::-1]]
    startup = advance(np.linspace(times[0],times[1],3),torch.from_numpy(np.tile(nodal,2)),1.)
    main = advance(times[1:],startup.temperature[-1],.5)
    history = torch.cat((startup.temperature[:1],main.temperature))
    trace = history.reshape(len(times),2,stride).mean(dim=1)
    def removed(result):
        return -result.nodal_reaction_heat_steps()[:,ends].sum(dim=-1)
    wall = np.vstack((np.zeros(2),torch.cat((removed(startup).sum(dim=0,keepdim=True),
                                            removed(main))).cumsum(dim=0).numpy()))
    def contact_steps(result):
        jump = (result.temperature[:,left_nodes]-result.temperature[:,left_nodes+1]).mean(dim=1)
        stage = (1-result.theta)*jump[:-1]+result.theta*jump[1:]
        return (min(area)/RESISTANCE)*stage*result.time.diff()
    contact = torch.cat((contact_steps(startup).sum().reshape(1),contact_steps(main))).cumsum(dim=0)
    return np.column_stack((trace[:,n+np.argmax(area)],trace[:,n]-trace[:,n+1],wall,trace[:,n:n+2],np.r_[0.,contact]))


def dg_samples(n,area,times,fraction,storage_fraction,initial,idx,right_delay=0.):
    from tensordg import solve_transient_diffusion_2d, poisson_face_powers_2d

    face = np.zeros((2*n-1,2))
    face[n-1] = 1.
    branch_capacity = area*CAPACITY
    surface = STORAGE*np.array([storage_fraction,1-storage_fraction])
    power = POWER*np.array([fraction,1-fraction])
    options = dict(bc=dict(top="neumann",bottom="neumann"),conductivity=np.tile((area*K).repeat(n)[:,None],(1,2)),
                   contact_resistance={"x":face*RESISTANCE/min(area)})
    def initial_field(x,y):
        return torch.where(x<.5,2*x*initial[0],2*(1-x)*initial[1])
    def heaters(t):
        right = np.exp(-DECAY*(t-right_delay)) if t>=right_delay else 0.
        return {"x":(power[0]*np.exp(-DECAY*t)*face,power[1]*right*face)}
    _,_,state,means,applied = solve_transient_diffusion_2d(initial_field,times,
               ncells=(2*n,2),degree=1,capacity=np.tile(branch_capacity.repeat(n)[:,None],(1,2)),
               trace_heat_capacity={"x":(surface[0]*face,surface[1]*face)},
               trace_power_history=heaters,load_jump_times=(right_delay,) if right_delay else (),
               rannacher=True,return_interval_means=True,return_history_heat=True,**options)
    modes = state[:,:,:,:,0].mean(2).numpy()
    traces = np.column_stack((modes[:,n-1].sum(1),modes[:,n,0]-modes[:,n,1]))
    stored = .5*branch_capacity*modes[:,:,0].reshape(len(times),2,n).mean(2)+surface*traces
    dt = np.diff(times)
    heat = np.zeros(3)
    start = 0
    residual = 0.
    rows = []
    for stop in idx:
        # Flux is linear in the modal field for these homogeneous end data.
        integral = torch.einsum("t,txyij->xyij",torch.from_numpy(dt[start:stop]),means[start:stop])
        fx,_ = poisson_face_powers_2d(integral,**options)
        west,contact,east = fx[[0,n,-1]].sum(1).numpy()
        heat += [-west,east,contact]
        input_heat = applied[start:stop].sum((0,2)).reshape(2,n).sum(1).numpy()
        balance = stored[stop]-stored[start]-input_heat+np.array([contact-west,east-contact])
        residual = max(residual,float(abs(balance).max()))
        left,right = traces[stop]
        rows.append([traces[stop,np.argmax(area)],left-right,*heat[:2],left,right,heat[2]])
        start = stop
    return np.array(rows),residual


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-fraction",type=float,help="fraction of input power deposited on the left trace")
    parser.add_argument("--storage-left-fraction",type=float,help="fraction of interface heat capacity on the left trace")
    parser.add_argument("--initial-traces",type=float,nargs=2,default=(0.,0.),metavar=("LEFT","RIGHT"),
                        help="initial interface temperatures in K above the cold wall temperature")
    parser.add_argument("--dg",action="store_true",help="verify transient DG contact storage and heat partition")
    args = parser.parse_args()
    if not np.isfinite(args.initial_traces).all():
        parser.error("--initial-traces requires finite temperatures")
    if args.storage_left_fraction is not None and not 0<=args.storage_left_fraction<=1:
        parser.error("--storage-left-fraction must be in [0,1]")
    if args.left_fraction is not None and not 0<=args.left_fraction<=1:
        parser.error("--left-fraction must be in [0,1]")
    rows = []
    for area in (np.array([1.,2.]),np.array([2.,1.])):
        fraction = float(area[0]>area[1]) if args.left_fraction is None else args.left_fraction
        storage_fraction = float(area[0]>area[1]) if args.storage_left_fraction is None else args.storage_left_fraction
        exact = reference(area,fraction,32,storage_fraction,args.initial_traces)
        inversion_change = float(abs(exact-reference(area,fraction,24,storage_fraction,args.initial_traces)).max())
        for solve in ((fem,fvm,dg_samples) if args.dg else (fem,fvm)):
            previous = {}
            for n in (8,16,32,64):
                times = np.linspace(0.,.8,32*n+1)
                idx = np.searchsorted(times,SAMPLES)
                assert np.allclose(times[idx],SAMPLES,rtol=0.,atol=1e-14)
                params = (n,area,times,fraction,storage_fraction,args.initial_traces)
                if solve is dg_samples:
                    samples,balance = solve(*params,idx)
                else:
                    samples = solve(*params)[idx]
                diff = np.max(abs(samples-exact),axis=0)
                errors = dict(zip(("sheet_K","signed_jump_K","west_end_J","east_end_J","west_trace_K","east_trace_K","contact_heat_J"),diff))
                orders = {k:np.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
                passed = inversion_change<1e-10 and all(v>0 for v in orders.values())
                rows.append(dict(area_m2=area.tolist(),left_fraction=fraction,storage_left_fraction=storage_fraction,solver="dg" if solve is dg_samples else solve.__name__,cells_per_branch=n,
                                 errors=errors,orders=orders,inversion_change=inversion_change,
                                 passed=bool(passed and (n<64 or min(orders.values())>1.8))))
                if solve is dg_samples:
                    rows[-1]["branch_balance_J"] = balance
                    rows[-1]["passed"] &= balance<1e-10
                previous = errors
    sources = [__file__,rod.__file__,thermal.__file__,two_layer_conduction.__file__]
    if args.dg:
        from tensordg import solve_transient_diffusion_2d, poisson_face_powers_2d
        sources.extend(inspect.getfile(f) for f in (solve_transient_diffusion_2d,poisson_face_powers_2d))
    print(json.dumps(dict(observation_times_s=SAMPLES.tolist(),initial_traces_K=args.initial_traces,runs=rows,sources=[provenance(f) for f in sources]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
