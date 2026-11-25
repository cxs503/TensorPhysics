"""Continuum decay with a storing shoulder between two axial finite-volume sections."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorfvm.cht import rod


def measure(n):
    m, dx, decay = n//2, 1/n, 1.7
    k, c, area = np.array([2.,5.]), np.array([3.,1.5]), np.array([.7,2.4])
    exposed, h, cw = area[1]-area[0], 1.8, .6
    q = np.array([1.,-.3])
    temp = .8
    wave = np.sqrt(decay*c/k)
    # T_j=(T_s cos(b_j*x) - P_j sin(b_j*x)/(k_j*A_j*b_j))*exp(-decay*t),
    # where x is signed distance from the shoulder, P_j points to the right.
    def profile(x,j):
        return temp*np.cos(wave[j]*x)-q[j]*np.sin(wave[j]*x)/(k[j]*area[j]*wave[j])
    ends = np.array([profile(-.5,0),profile(.5,1)])
    ambient = temp+((q[1]-q[0])/exposed-decay*cw*temp)/h
    xc = (np.arange(m)+.5)*dx
    initial = np.r_[profile(xc-.5,0),profile(xc,1)]*np.sinc(wave.repeat(m)*dx/(2*np.pi))
    times = np.linspace(0.,1.,2*n+1)
    signal = np.exp(-decay*times)
    cell,face,left,right = rod.solve_transient_stepped_rod(np.full(n,dx),area.repeat(m),k.repeat(m),
                          c.repeat(m),lambda t:ends*np.exp(-decay*t),times,initial,
                          film_coefficient=h,ambient_temperature=lambda t:ambient*np.exp(-decay*t),
                          shoulder_areal_heat_capacity=cw,initial_shoulder_temperature=temp)
    powers = np.column_stack((right[:,m-1],left[:,m]))
    def integral(values):
        return .5*np.diff(times)[:,None]*(values[:-1]+values[1:])
    input_power = exposed*h*(ambient*signal-face[:,m])
    shoulder = exposed*cw*np.diff(face[:,m])-integral((powers[:,0]-powers[:,1]+input_power)[:,None])[:,0]
    layers = dx*cell.reshape(len(times),2,m).sum(axis=-1)*area*c
    residual = np.diff(layers,axis=0)-integral(np.column_stack((left[:,0]-powers[:,0],powers[:,1]-right[:,-1])))
    transferred = integral(powers).cumsum(axis=0)
    return dict(n=n,steps=len(times)-1,temperature_error_K=float(abs(cell-signal[:,None]*initial).max()),
                shoulder_temperature_error_K=float(abs(face[:,m]-signal*temp).max()),
                instantaneous_power_error_W=float(abs(powers-signal[:,None]*q).max()),
                cumulative_heat_error_J=float(abs(transferred+np.expm1(-decay*times[1:,None])*q/decay).max()),
                shoulder_step_balance_J=float(abs(shoulder).max()),
                layer_step_balance_J=float(abs(residual).max()))


if __name__ == "__main__":
    rows, previous = [], None
    for n in (16,32,64,128,256):
        row = measure(n)
        errors = {k:v for k,v in row.items() if "_error_" in k}
        orders = {} if previous is None else {k:np.log2(previous[k]/v) for k,v in errors.items()}
        row.update(orders=orders,passed=all(v>(.95 if k=="instantaneous_power_error_W" else 1.7)
                   for k,v in orders.items()) and max(row["shoulder_step_balance_J"],row["layer_step_balance_J"])<1e-10)
        rows.append(row)
        previous = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in (__file__,rod.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
