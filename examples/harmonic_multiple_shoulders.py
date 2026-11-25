"""Distributed periodic heating of three sections on nonuniform axial grids."""
import json

import numpy as np

from conduction_provenance import provenance
import harmonic_stepped_section
from tensorfvm.cht import rod


def measure(count, omega):
    length = np.array([.25,.5,.25])
    area = np.array([.7,2.4,1.3])
    k, c = np.array([2.,5.,.9]), np.array([3.,1.5,2.])
    source = np.array([.4+.2j,-.6j,.7-.3j])
    film, cw = np.array([1.8,4.]), np.array([.6,.4])
    ambient = np.array([1.+.3j,-.5j])
    # Independently propagate the continuum state from a known temperature
    # and axial power, adding film input and removing surface storage at each step.
    t, power = .7+.1j, 1.-.5j
    left_end = t
    widths, means, left, right, surface = [], [], [], [], []
    for j,L in enumerate(length):
        s = np.linspace(0.,1.,count+1)
        x = L*(s+.6*s*(1-s))
        dx = np.diff(x)
        wave = np.sqrt(1j*omega*c[j]/k[j])
        base = source[j]/(1j*omega*c[j])
        def propagate(distance):
            u,p = harmonic_stepped_section.propagate(distance,t-base,power,wave,k[j]*area[j])
            return base+u,p
        nodes,p = propagate(x)
        mid,_ = propagate((x[1:]+x[:-1])/2)
        z = wave*dx/2
        means.extend(base+(mid-base)*np.sinh(z)/z)
        widths.extend(dx)
        left.extend(p[:-1])
        right.extend(p[1:])
        surface.extend(nodes[:-1])
        t, power = nodes[-1],p[-1]
        if j < 2:
            power += abs(area[j+1]-area[j])*(film[j]*(ambient[j]-t)-1j*omega*cw[j]*t)
    surface.append(t)
    n = len(widths)
    indices = np.array([count-1,2*count-1])
    hs,cap,air = np.zeros(n-1),np.zeros(n-1),np.zeros(n-1,complex)
    hs[indices],cap[indices],air[indices] = film,cw,ambient
    for order in (1,2):
        cell,faces,enter,leave = rod.solve_harmonic_stepped_rod(widths,area.repeat(count),k.repeat(count),
                      c.repeat(count),(left_end,t),omega,source=source.repeat(count),film_coefficient=hs,
                      ambient_temperature=air,shoulder_areal_heat_capacity=cap,flux_order=order)
        powers = abs(np.diff(area.repeat(count)))*(hs*(air-faces[1:-1])-1j*omega*cap*faces[1:-1])
        face_balance = leave[:-1]-enter[1:]+powers
        cell_balance = enter-leave+(source.repeat(count)-1j*omega*c.repeat(count)*cell)*area.repeat(count)*widths
        yield dict(n=n,omega_rad_s=omega,flux_order=order,temperature_error_K=float(abs(cell-means).max()),
                   trace_error_K=float(abs(faces-surface).max()),
                   axial_power_error_W=float(max(abs(enter-left).max(),abs(leave-right).max())),
                   max_balance_W=float(max(abs(face_balance).max(),abs(cell_balance).max())))


if __name__ == "__main__":
    rows = []
    for omega in (1.,7.):
        previous = {}
        for count in (8,16,32,64,128):
            for row in measure(count,omega):
                order = row["flux_order"]
                errors = {k:v for k,v in row.items() if "_error_" in k}
                rates = {} if order not in previous else {k:np.log2(previous[order][k]/v) for k,v in errors.items()}
                row.update(orders=rates,passed=all(v>1.8 for v in rates.values()) and row["max_balance_W"]<1e-9)
                rows.append(row)
                previous[order] = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
                     (__file__,rod.__file__,harmonic_stepped_section.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
