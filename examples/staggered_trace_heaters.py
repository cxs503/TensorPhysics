"""Two transient heaters switched on at different times on a storing contact."""
import argparse
import inspect
import json

import numpy as np
import torch
from tensordg import solve_transient_diffusion_2d, poisson_face_powers_2d

from conduction_provenance import provenance
import transient_interface_heater as slab


DELAY = .037
FRACTION = .35
STORAGE_LEFT = .7
INITIAL = (.3,-.2)
OBSERVE = np.array([.0125,.025,DELAY,.05,.1,.2,.4,.8])


def reference(area,degree,stop,right_sign,observations,fraction,initial):
    left = slab.reference(area,1.,degree,STORAGE_LEFT,initial,observations,slab.POWER*fraction)
    active = observations>DELAY
    left[active] += slab.reference(area,0.,degree,STORAGE_LEFT,(0.,0.),observations[active]-DELAY,
                                   right_sign*slab.POWER*(1-fraction))
    if stop is not None:
        active = observations>stop
        # Cancel the right exponential from shutoff onward at its value there.
        tail = right_sign*slab.POWER*(1-fraction)*np.exp(-slab.DECAY*(stop-DELAY))
        left[active] -= slab.reference(area,0.,degree,STORAGE_LEFT,(0.,0.),observations[active]-stop,tail)
    return left


def fvm(n,area,times,idx,stop,right_sign,fraction,initial):
    ix = n-1
    face = np.zeros(2*n-1)
    face[ix] = 1.
    x = (np.arange(n)+.5)/n
    c = (area*slab.CAPACITY).repeat(n)
    capacity = slab.STORAGE*np.array([STORAGE_LEFT,1-STORAGE_LEFT])
    def power(t):
        right = np.exp(-slab.DECAY*(t-DELAY)) if t>=DELAY and (stop is None or t<stop) else 0.
        return face*slab.POWER*fraction*np.exp(-slab.DECAY*t),face*right_sign*slab.POWER*(1-fraction)*right
    field,_,_,_,(left,right),heat = slab.rod.solve_transient_stepped_rod(
        np.full(2*n,.5/n),area.repeat(n),slab.K.repeat(n),slab.CAPACITY.repeat(n),[0.,0.],times,
        np.r_[initial[0]*x,initial[1]*x[::-1]],interface_heat_capacity=face*slab.STORAGE,
        interface_capacity_left_fraction=STORAGE_LEFT,initial_trace_temperatures=initial,
        contact_resistance=face*slab.RESISTANCE,trace_power=power,load_jump_times=(DELAY,) if stop is None else (DELAY,stop),
        method="sdirk2",flux_order=2,return_traces=True,return_step_heat=True)
    traces = np.column_stack((left[:,ix],right[:,ix]))
    stored = .5*(field*c).reshape(len(times),2,n).mean(2)+capacity*traces
    deposited = np.column_stack((heat["trace_left"][:,ix],heat["trace_right"][:,ix]))
    transfer = heat["contact"][:,ix]
    balance = np.diff(stored,axis=0)-deposited-heat["end_input"]+np.column_stack((transfer,-transfer))
    wall = np.vstack((np.zeros(2),-heat["end_input"].cumsum(0)))
    return np.column_stack((traces[:,np.argmax(area)],traces[:,0]-traces[:,1],wall,traces,
                            np.r_[0.,transfer.cumsum()]))[idx],float(abs(balance).max()),deposited.cumsum(0)[idx-1]


def fem(n,area,times,idx,stop,right_sign,fraction,initial):
    model,ends,left_nodes = slab.fem_contact_model(n,area,STORAGE_LEFT)
    # Align the state update with the switching instant. The FEM load
    # accepts integrated joules, including an exact partial-interval pulse.
    grid = np.union1d(times,[DELAY] if stop is None else [DELAY,stop])
    step = np.diff(grid)
    right_start = np.maximum(grid[:-1],DELAY)
    right_end = grid[1:] if stop is None else np.minimum(grid[1:],stop)
    duration = np.maximum(right_end-right_start,0.)
    left_heat = np.exp(-slab.DECAY*grid[:-1])*(-np.expm1(-slab.DECAY*step))
    right_heat = np.exp(-slab.DECAY*(right_start-DELAY))*(-np.expm1(-slab.DECAY*duration))
    inputs = slab.POWER/slab.DECAY*np.column_stack((fraction*left_heat,right_sign*(1-fraction)*right_heat))
    nodal = model.nodes.new_zeros((len(step),len(model.nodes)))
    for side in (0,1):
        nodal[:,left_nodes+side] = torch.from_numpy(.5*inputs[:,side,None])
    x = np.arange(n+1)/n
    initial = np.tile(np.r_[initial[0]*x,initial[1]*x[::-1]],2)
    result = slab.thermal.solve_transient_thermal(model,torch.from_numpy(grid),torch.from_numpy(initial),
                                                 theta=.5,sparse=True,nodal_step_heat=nodal)
    strip = result.temperature.numpy().reshape(len(grid),2,-1).mean(1)
    traces = strip[:,n:n+2]
    bulk = .5*(area*slab.CAPACITY)*np.stack(
        [(.5*(u[:,1:]+u[:,:-1])).mean(1) for u in np.split(strip,2,axis=1)],axis=1)
    stored = bulk+slab.STORAGE*np.array([STORAGE_LEFT,1-STORAGE_LEFT])*traces
    jump = traces[:,0]-traces[:,1]
    transfer = step*.5*(jump[:-1]+jump[1:])*min(area)/slab.RESISTANCE
    wall = -result.nodal_reaction_heat_steps()[:,ends].sum(-1).numpy()
    applied = result.additional_stage_power.numpy()*step[:,None]
    deposited = np.column_stack([applied[:,left_nodes+i].sum(1) for i in (0,1)])
    balance = np.diff(stored,axis=0)-deposited+wall+np.column_stack((transfer,-transfer))
    observations = np.searchsorted(grid,times[idx])
    cumulative = np.vstack((np.zeros(3),np.column_stack((wall,transfer)).cumsum(0)))
    samples = np.column_stack((traces[:,np.argmax(area)],jump,cumulative[:,:2],traces,cumulative[:,2]))
    return samples[observations],float(abs(balance).max()),deposited.cumsum(0)[observations-1]


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    pulses = parser.add_mutually_exclusive_group()
    pulses.add_argument("--short-pulse",action="store_true",help="apply only a 50 µs right pulse between requested output times, from zero initial temperature")
    pulses.add_argument("--pulse",action="store_true",help="switch off the right heater at 0.1437 s")
    parser.add_argument("--cool-right",action="store_true",help="extract heat at the right contact trace")
    args = parser.parse_args()
    stop = DELAY+50e-6 if args.short_pulse else (.1437 if args.pulse else None)
    right_sign = -1. if args.cool_right else 1.
    observations = OBSERVE[OBSERVE!=DELAY] if args.short_pulse else OBSERVE
    fraction,initial = (0.,(0.,0.)) if args.short_pulse else (FRACTION,INITIAL)
    rows = []
    for area in (np.array([1.,2.]),np.array([2.,1.])):
        exact = reference(area,32,stop,right_sign,observations,fraction,initial)
        ref_change = float(abs(exact-reference(area,24,stop,right_sign,observations,fraction,initial)).max())
        for solver in ("FVM","FEM","DG"):
            previous = None
            for n in (8,16,32,64):
                times = np.union1d(np.linspace(0.,.8,32*n+1),observations)
                idx = np.searchsorted(times,observations)
                if args.short_pulse:
                    before = np.searchsorted(times,DELAY)-1
                    if not times[before]<DELAY<stop<times[before+1]:
                        raise RuntimeError("short pulse must lie strictly inside one requested interval")
                if solver=="FVM":
                    sample,balance,deposited = fvm(n,area,times,idx,stop,right_sign,fraction,initial)
                elif solver=="FEM":
                    sample,balance,deposited = fem(n,area,times,idx,stop,right_sign,fraction,initial)
                else:
                    sample,balance,deposited = slab.dg_samples(n,area,times,fraction,STORAGE_LEFT,initial,idx,DELAY,stop,return_deposited=True,right_sign=right_sign,method="sdirk2" if args.short_pulse else "theta")
                active_time = np.maximum(observations-DELAY,0.)
                if stop is not None:
                    active_time = np.minimum(active_time,stop-DELAY)
                duration = np.column_stack((observations,active_time))
                target = -np.expm1(-slab.DECAY*duration)*slab.POWER*np.array([fraction,right_sign*(1-fraction)])/slab.DECAY
                deposited_error = float(abs(deposited-target).max())
                relative_heat_error = deposited_error/np.max(abs(target))
                errors = abs(sample-exact).max(0)
                relative_response_error = float(np.max(errors/np.max(abs(exact),axis=0)))
                short_pass = not args.short_pulse or (relative_heat_error<1e-8 and
                            np.max(abs(sample[observations<DELAY]))<1e-14 and (n<64 or relative_response_error<.005))
                orders = np.log2(previous/errors) if previous is not None else np.array([])
                rows.append(dict(solver=solver,cells_per_branch=n,areas=area.tolist(),errors=errors.tolist(),
                    orders=orders.tolist(),balance_J=balance,max_relative_response_error=relative_response_error,deposited_heat_error_J=deposited_error,relative_deposited_heat_error=relative_heat_error,inversion_change=ref_change,
                    passed=bool(short_pass and balance<1e-10 and ref_change<1e-10 and (n<64 or (min(orders)>1.8 and deposited_error<2e-6)))))
                previous = errors
    sources = [__file__,slab.__file__,slab.rod.__file__,slab.thermal.__file__,slab.two_layer_conduction.__file__,inspect.getfile(solve_transient_diffusion_2d),
               inspect.getfile(poisson_face_powers_2d)]
    print(json.dumps(dict(left_fraction=fraction,initial_traces_K=initial,delay_s=DELAY,right_stop_s=stop,right_heat_sign=right_sign,observations_s=observations.tolist(),metrics=["sheet_K","jump_K","west_end_J",
             "east_end_J","west_trace_K","east_trace_K","contact_heat_J"],runs=rows,
             sources=[provenance(f) for f in sources]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
