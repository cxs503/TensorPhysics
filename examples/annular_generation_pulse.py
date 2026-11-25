"""Two-material annulus with quadratic radial profiles and a switched heating rate."""
import argparse
import json

import numpy as np
from tensorfvm.cht import annulus
from conduction_provenance import provenance


def measure(n,boundary,off_grid=False,method="theta"):
    a,b,c = .4,.7,1.1
    ki,ko = 1.4,2.6
    ci,co = 1.8,.9
    curvature = (.7,-.4)
    resistance,fraction,cs,h = .22,.35,.43,1.6
    r = np.r_[np.linspace(a,b,n+1),np.linspace(b,c,n+1)[1:]]
    center = (2/3)*np.diff(r**3)/np.diff(r**2)
    k,cv = np.repeat([ki,ko],n),np.repeat([ci,co],n)
    volume = np.pi*np.diff(r**2)
    # T_l = offset_l + curvature_l*(r²-b²) + U(t). Fourier flux
    # is -2*k_l*curvature_l*r, and volumetric heating is cv_l*U'-4*k_l*curvature_l.
    qi,qo = -2*ki*curvature[0]*b,-2*ko*curvature[1]*b
    sheet = .8
    offsets = [sheet+fraction*resistance*qi,sheet-(1-fraction)*resistance*qo]
    profile = np.repeat(offsets,n)+np.repeat(curvature,n)*(center**2-b**2)
    switch = .097 if off_grid else .1
    times = (np.linspace(0.,.25,2*n+1) if off_grid else
             np.r_[np.linspace(0.,switch,n+1),np.linspace(switch,.25,n+1)[1:]])
    source_times = np.array([0.,switch,.25])
    wall_times = np.array([0.,.053,switch,.18,.25]) if off_grid else times
    dt = np.diff(times)
    def warming(t):
        return 1.3*t-2*np.maximum(t-switch,0.)
    rise = warming(times)
    rate = np.array([1.3,-.7])
    source = rate[:,None]*cv-4*k*np.repeat(curvature,n)
    ambient = .3+.2*wall_times
    sheet_flux = h*(sheet+warming(source_times)-(.3+.2*source_times))
    sheet_source = cs*rate[:,None]+qo-qi+np.column_stack((sheet_flux[:-1],sheet_flux[1:]))
    wall_base = np.array([offsets[0]+curvature[0]*(a*a-b*b),offsets[1]+curvature[1]*(c*c-b*b)])
    walls = wall_base[:,None]+warming(wall_times)
    wall_reference = wall_base+rise[:,None]
    fluxes = (None,None)
    inward = (-2*ki*curvature[0]*a,2*ko*curvature[1]*c)
    films = (None,None)
    if boundary=="flux":
        walls,fluxes = (None,None),inward
    elif boundary=="film":
        films = (1.4,2.6)
        walls = tuple(w+q/hwall for w,q,hwall in zip(walls,inward,films))
    u,s,hi,ho,external,surface,contact,means = annulus.solve_transient_annular_sheet(r,k,cv,walls,times,profile,n,
             initial_sheet_temperature=sheet,sheet_capacity=cs,sheet_film=h,ambient=ambient,
             source_times=source_times,wall_times=wall_times,method=method,
             resistance=resistance,inner_resistance_fraction=fraction,wall_fluxes=fluxes,film_coefficients=films,return_wall_temperatures=True,return_contact_temperatures=True,return_interval_temperatures=True,
             source_intervals=np.repeat(source[:,None,:],2,axis=1),sheet_source_intervals=sheet_source)
    inner_rate = -4*np.pi*ki*curvature[0]*r[:n+1]**2
    outer_rate = -4*np.pi*ko*curvature[1]*r[n:]**2
    # Independently integrate the kink in the exact temperature on each output interval.
    int_rise = .65*times**2-np.maximum(times-switch,0.)**2
    absorbed = 2*np.pi*b*h*((sheet-.3)*times+int_rise-.1*times**2)
    applied_bulk = np.diff(rise)[:,None]*cv-dt[:,None]*4*k*np.repeat(curvature,n)
    cell_residual = (np.diff(u,axis=0)*cv*volume+np.column_stack((np.diff(hi),np.diff(ho)))
                     -applied_bulk*volume)
    applied_sheet = cs*np.diff(rise)+(qo-qi)*dt+np.diff(absorbed)/(2*np.pi*b)
    sheet_residual = 2*np.pi*b*(cs*np.diff(s)-applied_sheet)+ho[:,0]-hi[:,-1]+external
    result = dict(cells_per_layer=n,boundary=boundary,off_grid=off_grid,method=method,
                errors=dict(field_K=float(abs(u-profile-rise[:,None]).max()),sheet_K=float(abs(s-sheet-rise).max()),
                            inner_heat_J_m=float(abs(hi.cumsum(0)-times[1:,None]*inner_rate).max()),
                            outer_heat_J_m=float(abs(ho.cumsum(0)-times[1:,None]*outer_rate).max()),
                            reservoir_heat_J_m=float(abs(external.cumsum()-absorbed[1:]).max())),
                cell_residual_J_m=float(abs(cell_residual).max()),sheet_residual_J_m=float(abs(sheet_residual).max()))
    result["errors"].update(inner_trace_K=float(abs(contact[:,0]-offsets[0]-rise).max()),
                            outer_trace_K=float(abs(contact[:,1]-offsets[1]-rise).max()),
                            jump_K=float(abs(contact[:,0]-contact[:,1]-(offsets[0]-offsets[1])).max()))
    mean_rise = np.diff(int_rise)/dt
    result["errors"].update(
        mean_field_K=float(abs(means["cell"]-profile-mean_rise[:,None]).max()),
        mean_sheet_K=float(abs(means["sheet"]-sheet-mean_rise).max()),
        mean_contact_K=float(abs(means["contact"]-offsets-mean_rise[:,None]).max()),
    )
    if boundary!="temperature":
        result["errors"]["surface_K"] = float(abs(surface-wall_reference).max())
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-grid",action="store_true",help="use source switches inside uniform output steps and run both time integrators")
    args = parser.parse_args()
    rows = []
    for method in (("theta","sdirk2") if args.off_grid else ("theta",)):
        for boundary in ("temperature","flux","film"):
            previous = {}
            for n in (8,16,32,64):
                row = measure(n,boundary,args.off_grid,method)
                errors = row["errors"]
                orders = {k:float(np.log2(previous[k]/v)) for k,v in errors.items()} if previous else {}
                row.update(orders=orders,passed=max(row["cell_residual_J_m"],row["sheet_residual_J_m"])<1e-11
                           and all(v>1.8 for k,v in orders.items() if n>=32 or not k.startswith("mean_"))
                           and all(v>0 for v in orders.values()))
                rows.append(row)
                previous = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in (__file__,annulus.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
