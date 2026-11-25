"""Transient contact heat through reversals of a periodic, independently driven heater pair."""
import json

import numpy as np

from conduction_provenance import provenance
import harmonic_trace_heaters as harmonic
from tensorfvm.cht import rod


def measure(n,steps,method):
    area = np.array([1.,2.])
    omega = 8.
    dx = .5/n
    traces,contact,ends = harmonic.reference(area,omega)
    w = np.sqrt(1j*omega*harmonic.C/harmonic.K)
    centers = (np.arange(n)+.5)*dx
    from_wall = np.array([centers,centers[::-1]])
    # Integrate each branch's continuum phasor over its finite volumes.
    mean = (traces*np.sinh(w*dx/2)/(w*dx/2)/np.sinh(w*.5))[:,None]*np.sinh(w[:,None]*from_wall)
    time = np.linspace(0.,4*np.pi/omega,2*steps+1)
    face = np.zeros(2*n-1)
    face[n-1] = 1.
    def drives(t):
        west,east = (harmonic.DRIVE*np.exp(1j*omega*t)).real
        return face*west,face*east
    _,_,_,_,(west,east),heat = rod.solve_transient_stepped_rod(np.full(2*n,dx),
          area.repeat(n),harmonic.K.repeat(n),harmonic.C.repeat(n),[0.,0.],time,mean.real.ravel(),
          contact_resistance=face*harmonic.RESISTANCE,interface_heat_capacity=face*harmonic.STORAGE.sum(),
          interface_capacity_left_fraction=harmonic.STORAGE[0]/harmonic.STORAGE.sum(),
          initial_trace_temperatures=traces.real,trace_power=drives,return_traces=True,
          method=method,return_step_heat=True,flux_order=2)
    phase = np.exp(1j*omega*time)
    interval = phase[:-1]*np.expm1(1j*omega*np.diff(time))/(1j*omega)
    numerical = np.column_stack((west[:,n-1],east[:,n-1]))
    q = heat["contact"][:,n-1]
    net = q.reshape(2,steps).sum(axis=1)
    # Report errors in W as well as the zero-net-heat check in J. Normalizing
    # by duration avoids an extra apparent order when time steps get shorter.
    contact_error = abs((q-(interval*contact).real)/np.diff(time))
    wall_error = abs((-heat["end_input"]-(interval[:,None]*ends).real)/np.diff(time)[:,None])
    return dict(temperature_K=float(abs(numerical-(phase[:,None]*traces).real).max()),
                contact_mean_W=float(contact_error.max()),
                wall_mean_W=float(wall_error.max()),
                second_cycle_contact_mean_W=float(contact_error[steps:].max()),
                second_cycle_wall_mean_W=float(wall_error[steps:].max()),
                net_contact_heat_per_cycle_J=float(abs(net).max()))


if __name__=="__main__":
    rows = []
    for method in ("theta","rannacher","sdirk2"):
        previous = {}
        for n in (8,16,32,64):
            steps = 4*n
            errors = measure(n,steps,method)
            orders = {k:float(np.log2(previous[k]/errors[k])) for k in
                      ("temperature_K","contact_mean_W","wall_mean_W","second_cycle_contact_mean_W","second_cycle_wall_mean_W")} if previous else {}
            # Rannacher's two starting backward Euler half steps integrate
            # a smooth time-varying flux with first-order mean-power accuracy.
            passed = all(order>0 for order in orders.values())
            if n==64:
                passed &= all(order>(.9 if method=="rannacher" and key in ("contact_mean_W","wall_mean_W") else 1.8)
                              for key,order in orders.items())
                passed &= errors["net_contact_heat_per_cycle_J"]<1e-5
            rows.append(dict(method=method,cells_per_branch=n,steps_per_period=steps,errors=errors,orders=orders,
                             passed=passed))
            previous = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
                     (__file__,harmonic.__file__,rod.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
