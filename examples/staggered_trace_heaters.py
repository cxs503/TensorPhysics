"""Two transient heaters switched on at different times on a storing contact."""
import argparse
import inspect
import json

import numpy as np
from tensordg import solve_transient_diffusion_2d, poisson_face_powers_2d

from conduction_provenance import provenance
import transient_interface_heater as slab


DELAY = .037
FRACTION = .35
STORAGE_LEFT = .7
INITIAL = (.3,-.2)
OBSERVE = np.array([.0125,.025,DELAY,.05,.1,.2,.4,.8])


def reference(area,degree,stop):
    left = slab.reference(area,1.,degree,STORAGE_LEFT,INITIAL,OBSERVE,slab.POWER*FRACTION)
    active = OBSERVE>DELAY
    left[active] += slab.reference(area,0.,degree,STORAGE_LEFT,(0.,0.),OBSERVE[active]-DELAY,
                                   slab.POWER*(1-FRACTION))
    if stop is not None:
        active = OBSERVE>stop
        # Cancel the right exponential from shutoff onward at its value there.
        tail = slab.POWER*(1-FRACTION)*np.exp(-slab.DECAY*(stop-DELAY))
        left[active] -= slab.reference(area,0.,degree,STORAGE_LEFT,(0.,0.),OBSERVE[active]-stop,tail)
    return left


def fvm(n,area,times,idx,stop):
    ix = n-1
    face = np.zeros(2*n-1)
    face[ix] = 1.
    x = (np.arange(n)+.5)/n
    c = (area*slab.CAPACITY).repeat(n)
    capacity = slab.STORAGE*np.array([STORAGE_LEFT,1-STORAGE_LEFT])
    def power(t):
        right = np.exp(-slab.DECAY*(t-DELAY)) if t>=DELAY and (stop is None or t<stop) else 0.
        return face*slab.POWER*FRACTION*np.exp(-slab.DECAY*t),face*slab.POWER*(1-FRACTION)*right
    field,_,_,_,(left,right),heat = slab.rod.solve_transient_stepped_rod(
        np.full(2*n,.5/n),area.repeat(n),slab.K.repeat(n),slab.CAPACITY.repeat(n),[0.,0.],times,
        np.r_[INITIAL[0]*x,INITIAL[1]*x[::-1]],interface_heat_capacity=face*slab.STORAGE,
        interface_capacity_left_fraction=STORAGE_LEFT,initial_trace_temperatures=INITIAL,
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


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulse",action="store_true",help="switch off the right heater at 0.1437 s")
    stop = .1437 if parser.parse_args().pulse else None
    rows = []
    for area in (np.array([1.,2.]),np.array([2.,1.])):
        exact = reference(area,32,stop)
        ref_change = float(abs(exact-reference(area,24,stop)).max())
        for solver in ("FVM","DG"):
            previous = None
            for n in (8,16,32,64):
                times = np.union1d(np.linspace(0.,.8,32*n+1),OBSERVE)
                idx = np.searchsorted(times,OBSERVE)
                if solver=="FVM":
                    sample,balance,deposited = fvm(n,area,times,idx,stop)
                else:
                    sample,balance,deposited = slab.dg_samples(n,area,times,FRACTION,STORAGE_LEFT,INITIAL,idx,DELAY,stop,return_deposited=True)
                active_time = np.maximum(OBSERVE-DELAY,0.)
                if stop is not None:
                    active_time = np.minimum(active_time,stop-DELAY)
                duration = np.column_stack((OBSERVE,active_time))
                target = -np.expm1(-slab.DECAY*duration)*slab.POWER*np.array([FRACTION,1-FRACTION])/slab.DECAY
                deposited_error = float(abs(deposited-target).max())
                errors = abs(sample-exact).max(0)
                orders = np.log2(previous/errors) if previous is not None else np.array([])
                rows.append(dict(solver=solver,cells_per_branch=n,areas=area.tolist(),errors=errors.tolist(),
                    orders=orders.tolist(),balance_J=balance,deposited_heat_error_J=deposited_error,inversion_change=ref_change,
                    passed=bool(balance<1e-10 and ref_change<1e-10 and (n<64 or (min(orders)>1.8 and deposited_error<2e-6)))))
                previous = errors
    sources = [__file__,slab.__file__,slab.rod.__file__,inspect.getfile(solve_transient_diffusion_2d),
               inspect.getfile(poisson_face_powers_2d)]
    print(json.dumps(dict(delay_s=DELAY,right_stop_s=stop,observations_s=OBSERVE.tolist(),metrics=["sheet_K","jump_K","west_end_J",
             "east_end_J","west_trace_K","east_trace_K","contact_heat_J"],runs=rows,
             sources=[provenance(f) for f in sources]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
