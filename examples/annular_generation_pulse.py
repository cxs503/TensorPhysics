"""Two-material annulus with quadratic radial profiles and a switched heating rate."""
import json

import numpy as np
from tensorfvm.cht import annulus
from conduction_provenance import provenance


def measure(n,boundary):
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
    times = np.r_[np.linspace(0.,.1,n+1),np.linspace(.1,.25,n+1)[1:]]
    dt = np.diff(times)
    rate = np.repeat([1.3,-.7],n)
    rise = np.r_[0.,np.cumsum(dt*rate)]
    source = rate[:,None]*cv-4*k*np.repeat(curvature,n)
    ambient = .3+.2*times
    sheet_flux = h*(sheet+rise-ambient)
    sheet_source = cs*rate[:,None]+qo-qi+np.column_stack((sheet_flux[:-1],sheet_flux[1:]))
    walls = (offsets[0]+curvature[0]*(a*a-b*b)+rise,offsets[1]+curvature[1]*(c*c-b*b)+rise)
    fluxes = (None,None)
    inward = (-2*ki*curvature[0]*a,2*ko*curvature[1]*c)
    films = (None,None)
    if boundary=="flux":
        walls,fluxes = (None,None),inward
    elif boundary=="film":
        films = (1.4,2.6)
        walls = tuple(w+q/hwall for w,q,hwall in zip(walls,inward,films))
    u,s,hi,ho,external = annulus.solve_transient_annular_sheet(r,k,cv,walls,times,profile,n,
             initial_sheet_temperature=sheet,sheet_capacity=cs,sheet_film=h,ambient=ambient,
             resistance=resistance,inner_resistance_fraction=fraction,wall_fluxes=fluxes,film_coefficients=films,
             source_intervals=np.repeat(source[:,None,:],2,axis=1),sheet_source_intervals=sheet_source)
    inner_rate = -4*np.pi*ki*curvature[0]*r[:n+1]**2
    outer_rate = -4*np.pi*ko*curvature[1]*r[n:]**2
    absorbed = np.cumsum(2*np.pi*b*dt*(sheet_flux[1:]+sheet_flux[:-1])/2)
    cell_residual = (np.diff(u,axis=0)*cv*volume+np.column_stack((np.diff(hi),np.diff(ho)))
                     -dt[:,None]*source*volume)
    sheet_residual = 2*np.pi*b*(cs*np.diff(s)-dt*sheet_source.mean(1))+ho[:,0]-hi[:,-1]+external
    return dict(cells_per_layer=n,boundary=boundary,
                errors=dict(field_K=float(abs(u-profile-rise[:,None]).max()),sheet_K=float(abs(s-sheet-rise).max()),
                            inner_heat_J_m=float(abs(hi.cumsum(0)-times[1:,None]*inner_rate).max()),
                            outer_heat_J_m=float(abs(ho.cumsum(0)-times[1:,None]*outer_rate).max()),
                            reservoir_heat_J_m=float(abs(external.cumsum()-absorbed).max())),
                cell_residual_J_m=float(abs(cell_residual).max()),sheet_residual_J_m=float(abs(sheet_residual).max()))


if __name__=="__main__":
    rows = []
    for boundary in ("temperature","flux","film"):
        previous = {}
        for n in (8,16,32,64):
            row = measure(n,boundary)
            errors = row["errors"]
            orders = {k:float(np.log2(previous[k]/v)) for k,v in errors.items()} if previous else {}
            row.update(orders=orders,passed=max(row["cell_residual_J_m"],row["sheet_residual_J_m"])<1e-11
                       and all(v>1.8 for v in orders.values()))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in (__file__,annulus.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
