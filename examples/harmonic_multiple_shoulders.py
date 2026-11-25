"""Distributed periodic heating of three sections on nonuniform axial grids."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance
import harmonic_stepped_section
from tensorfvm.cht import rod


def measure(count, omega, resistive=False, end_types=("temperature","temperature")):
    length = np.array([.25,.5,.25])
    area = np.array([.7,2.4,1.3])
    k, c = np.array([2.,5.,.9]), np.array([3.,1.5,2.])
    source = np.array([.4+.2j,-.6j,.7-.3j])
    film, cw = np.array([1.8,4.]), np.array([.6,.4])
    ambient = np.array([1.+.3j,-.5j])
    resistance = np.array([.14,.32]) if resistive else np.zeros(2)
    # Independently propagate the continuum state from a known temperature
    # and axial power, adding film input and removing surface storage at each step.
    t, power = .7+.1j, 1.-.5j
    left_end = t
    widths, means, left, right = [], [], [], []
    west, east = [], []
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
        west.extend(nodes[:-1])
        east.extend(nodes[1:])
        t, power = nodes[-1],p[-1]
        if j < 2:
            r = resistance[j]/min(area[j],area[j+1])
            if area[j+1] > area[j]:
                t -= r*power
            power += abs(area[j+1]-area[j])*(film[j]*(ambient[j]-t)-1j*omega*cw[j]*t)
            if area[j] > area[j+1]:
                t -= r*power
    n = len(widths)
    indices = np.array([count-1,2*count-1])
    hs,cap,air = np.zeros(n-1),np.zeros(n-1),np.zeros(n-1,complex)
    hs[indices],cap[indices],air[indices] = film,cw,ambient
    contacts = np.zeros(n-1)
    contacts[indices] = resistance
    end_film = np.array([2.,5.])
    conductance = end_film*area[[0,-1]]
    exact_ends = np.array([left_end,t])
    inward = np.array([left[0],-right[-1]])
    robin = np.array(end_types)=="robin"
    ends = np.where(robin,exact_ends+inward/conductance,
                    np.where(np.array(end_types)=="power",inward,exact_ends))
    for order in (1,2):
        cell,faces,enter,leave,traces = rod.solve_harmonic_stepped_rod(widths,area.repeat(count),k.repeat(count),
                      c.repeat(count),ends,omega,source=source.repeat(count),film_coefficient=hs,
                      ambient_temperature=air,shoulder_areal_heat_capacity=cap,flux_order=order,
                      contact_resistance=contacts,return_traces=True,end_types=end_types,end_film_coefficient=end_film)
        powers = abs(np.diff(area.repeat(count)))*(hs*(air-faces[1:-1])-1j*omega*cap*faces[1:-1])
        face_balance = leave[:-1]-enter[1:]+powers
        cell_balance = enter-leave+(source.repeat(count)-1j*omega*c.repeat(count)*cell)*area.repeat(count)*widths
        row = dict(n=n,end_types=end_types,resistive=resistive,omega_rad_s=omega,flux_order=order,temperature_error_K=float(abs(cell-means).max()),
                   trace_error_K=float(max(abs(traces[0]-east[:-1]).max(),abs(traces[1]-west[1:]).max())),
                   axial_power_error_W=float(max(abs(enter-left).max(),abs(leave-right).max())),
                   max_balance_W=float(max(abs(face_balance).max(),abs(cell_balance).max())))
        if robin.any():
            exchange = conductance*(ends-faces[[0,-1]])
            residual = np.array([enter[0],-leave[-1]])-exchange
            row["max_balance_W"] = max(row["max_balance_W"],float(abs(residual[robin]).max()))
        prescribed = np.array(end_types)!="temperature"
        if prescribed.any():
            row["end_temperature_error_K"] = float(abs(faces[[0,-1]]-[left_end,t])[prescribed].max())
        if resistive:
            power_from_jump = (traces[0][indices]-traces[1][indices])*np.minimum(area[:-1],area[1:])/resistance
            reference = np.where(area[1:]>area[:-1],np.array(right)[indices],np.array(left)[indices+1])
            row["contact_power_error_W"] = float(abs(power_from_jump-reference).max())
        yield row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ends", nargs=2, choices=("temperature","power","robin"),
                        default=("temperature","temperature"), metavar=("WEST","EAST"))
    args = parser.parse_args()
    rows = []
    for omega,resistive in ((1.,False),(7.,False),(1.,True),(7.,True)):
        previous = {}
        counts = (16,32,64,128,256) if "robin" in args.ends else (8,16,32,64,128)
        for count in counts:
            for row in measure(count,omega,resistive,args.ends):
                order = row["flux_order"]
                errors = {k:v for k,v in row.items() if "_error_" in k}
                rates = {} if order not in previous else {k:np.log2(previous[order][k]/v) for k,v in errors.items()}
                row.update(orders=rates,passed=all(v>1.8 for v in rates.values()) and row["max_balance_W"]<1e-9)
                rows.append(row)
                previous[order] = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
                     (__file__,rod.__file__,harmonic_stepped_section.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
