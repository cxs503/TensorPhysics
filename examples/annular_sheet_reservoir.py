"""Two radial material regions coupled by a conducting sheet and a thermal reservoir."""
import argparse
import json

import numpy as np
from scipy import special
from conduction_provenance import provenance
from tensorbem import layered_annulus
from tensorfvm.cht import annulus


A,B,C = .4,.7,1.1
KI,KO = 1.5,2.3
R,FRAC,H,KS = .4,.3,1.1,.18


def radial(r,k,mode,trace,flux,omega=0.,capacity=1.):
    """Harmonic continuation from the material face at r=B."""
    if omega:
        wave = np.sqrt(1j*omega*capacity/k)
        x = wave*B
        matrix = [[special.iv(mode,x),special.kv(mode,x)],
                  [-k*wave*special.ivp(mode,x),-k*wave*special.kvp(mode,x)]]
        a,b = np.linalg.solve(matrix,[trace,flux])
        y = wave*r
        return (a*special.iv(mode,y)+b*special.kv(mode,y),
                -k*wave*(a*special.ivp(mode,y)+b*special.kvp(mode,y)))
    z = np.log(r/B)
    if mode==0:
        return trace-B*flux*z/k, B*flux/r
    return (trace*np.cosh(mode*z)-B*flux*np.sinh(mode*z)/(mode*k),
            (B*flux*np.cosh(mode*z)-mode*k*trace*np.sinh(mode*z))/r)


def measure(n,omega=0.):
    theta = 2*np.pi*(np.arange(2*n)+.5)/(2*n)
    wall = np.zeros((2,len(theta)))
    ambient,source,exact_sheet,exact_in,exact_out = np.zeros((5,len(theta)))
    fvm_errors = {}
    bem_targets = []
    sheet_capacity = .43
    balance = 0.
    for m,ts,qi,qo,ta in ((0,2.,.8,-.3,.6),(3,.7+.2j,.4-.5j,-.2+.6j,.2+.5j)):
        inner,outer = ts+FRAC*R*qi,ts-(1-FRAC)*R*qo
        west = radial(A,KI,m,inner,qi,omega,1.2)[1]
        east = -radial(C,KO,m,outer,qo,omega,2.3)[1]
        admittance = H+KS*(m/B)**2+1j*omega*sheet_capacity
        generation = qo-qi+admittance*ts-H*ta
        def sample(v):
            return np.real(v*np.exp(1j*m*theta))
        wall += np.stack((sample(west),sample(east)))
        ambient += sample(ta)
        source += sample(generation)
        exact_sheet += sample(ts)
        exact_in += sample(qi)
        exact_out += sample(qo)
        edges = (np.linspace(A,B,n+1),np.linspace(B,C,n+1))
        radii = np.r_[edges[0],edges[1][1:]]
        field,pi,po,recovered = annulus.solve_annular_sheet(radii,np.repeat([KI,KO],n),np.repeat([1.2,2.3],n),
               (None,None),omega,n,wall_fluxes=(west,east),sheet_capacity=sheet_capacity,sheet_film=H,ambient=ta,
               sheet_conductivity=KS,sheet_source=generation,azimuthal_mode=m,resistance=R,
               inner_resistance_fraction=FRAC,cell_location="volume_centroid",distributed_source=True)
        i,o = pi[-1]/(2*np.pi*B),po[0]/(2*np.pi*B)
        balance = max(balance,abs(o-i+admittance*recovered-generation-H*ta))
        centers = [(2/3)*np.diff(e**3)/np.diff(e**2) for e in edges]
        target = np.r_[radial(centers[0],KI,m,inner,qi,omega,1.2)[0],radial(centers[1],KO,m,outer,qo,omega,2.3)[0]]
        for key,value in dict(temperature_K=np.max(abs(field-target)),sheet_K=abs(recovered-ts),
                              incoming_W_m2=abs(i-qi),outgoing_W_m2=abs(o-qo),
                              reservoir_W_m2=abs(H*(recovered-ts))).items():
            fvm_errors[key] = float(max(fvm_errors.get(key,0.),value))
        if omega:
            continue
        bem_targets.append([sample(radial(r,k,m,t,q)[0]) for r,k,t,q in
                            ((.54,KI,inner,qi),(.87,KO,outer,qo))])
    row = dict(cells_per_layer=n,omega_rad_s=omega,fvm_errors=fvm_errors,sheet_balance_W_m2=float(balance))
    if omega:
        return row
    bem = layered_annulus.solve_annulus_contact((A,B,C),(KI,KO),(None,None),R,2*n,formulation="circular",
             wall_fluxes=wall,inner_resistance_fraction=FRAC,interface_source=source,
             sheet_conductivity=KS,sheet_film=(H,ambient))
    direction = np.column_stack((np.cos(theta),np.sin(theta)))
    temperatures = np.array([f.temperature(r*direction) for f,r in ((bem.inner_field,.54),(bem.outer_field,.87))])
    bem_errors = dict(temperature_K=float(abs(temperatures-np.sum(bem_targets,axis=0)).max()),
                      sheet_K=float(abs(bem.sheet_temperature-exact_sheet).max()),
                      incoming_W_m2=float(abs(bem.interface_flux-exact_in).max()),
                      outgoing_W_m2=float(abs(bem.outer_interface_flux-exact_out).max()),
                      reservoir_W_m2=float(abs(bem.sheet_reservoir_flux-H*(exact_sheet-ambient)).max()))
    balance = bem.heat_rates[0]-bem.heat_rates[2]+2*np.pi*B*np.mean(source-bem.sheet_reservoir_flux)
    return dict(**row,panels_per_circle=2*n,bem_errors=bem_errors,
                bem_net_power_residual_W_m=float(abs(balance)))


def transient_decay(n,mode,power_walls):
    # A decaying separated solution has ordinary Bessel radial factors.
    # Propagate independent traces from the sheet to prescribe both walls.
    decay,cs,ta = 1.6,.43,.36
    ts,qi,qo = 1.2,.65,-.35
    inner = ts+FRAC*R*qi
    outer = ts-(1-FRAC)*R*qo
    def shape(r,k,cv,trace,flux):
        wave = np.sqrt(decay*cv/k)
        x = wave*B
        a,b = np.linalg.solve([[special.jv(mode,x),special.yv(mode,x)],
                  [-k*wave*special.jvp(mode,x),-k*wave*special.yvp(mode,x)]],[trace,flux])
        return (a*special.jv(mode,wave*r)+b*special.yv(mode,wave*r),
                -k*wave*(a*special.jvp(mode,wave*r)+b*special.yvp(mode,wave*r)))
    edges = np.r_[np.linspace(A,B,n+1),np.linspace(B,C,n+1)[1:]]
    centers = (2/3)*np.diff(edges**3)/np.diff(edges**2)
    initial = np.r_[shape(centers[:n],KI,1.2,inner,qi)[0],shape(centers[n:],KO,2.3,outer,qo)[0]]
    rates = [2*np.pi*r*shape(r,k,cv,trace,flux)[1] for r,k,cv,trace,flux in
             ((edges[:n+1],KI,1.2,inner,qi),(edges[n:],KO,2.3,outer,qo))]
    times = np.linspace(0.,.22,2*n+1)
    signal = np.exp(-decay*times)
    wall = (shape(A,KI,1.2,inner,qi)[0]*signal,shape(C,KO,2.3,outer,qo)[0]*signal)
    generation = qo-qi+(KS*(mode/B)**2-decay*cs)*ts+H*(ts-ta)
    flux_bc = (None,None)
    if power_walls:
        wall = (None,None)
        flux_bc = (rates[0][0]*signal/(2*np.pi*A),-rates[1][-1]*signal/(2*np.pi*C))
    power = generation*np.column_stack((signal[:-1],signal[1:]))
    u,s,hi,ho,external = annulus.solve_transient_annular_sheet(edges,np.repeat([KI,KO],n),
               np.repeat([1.2,2.3],n),wall,times,initial,n,initial_sheet_temperature=ts,sheet_capacity=cs,
               azimuthal_mode=mode,sheet_film=H,ambient=ta*signal,sheet_conductivity=KS,
               resistance=R,inner_resistance_fraction=FRAC,sheet_source_intervals=power,wall_fluxes=flux_bc)
    integral = -np.expm1(-decay*times[1:])/decay
    dt = np.diff(times)
    circumference = 2*np.pi*B
    mean_sheet = .5*(s[1:]+s[:-1])
    balance = circumference*(cs*np.diff(s)+dt*(KS*(mode/B)**2*mean_sheet-power.mean(axis=1)))+ho[:,0]-hi[:,-1]+external
    errors = dict(field_K=float(abs(u-signal[:,None]*initial).max()),sheet_K=float(abs(s-ts*signal).max()),
                  inner_heat_J_m=float(abs(hi.cumsum(0)-integral[:,None]*rates[0]).max()),
                  outer_heat_J_m=float(abs(ho.cumsum(0)-integral[:,None]*rates[1]).max()),
                  reservoir_heat_J_m=float(abs(external.cumsum()-circumference*integral*H*(ts-ta)).max()))
    return dict(cells_per_layer=n,steps=len(dt),mode=mode,power_walls=power_walls,errors=errors,sheet_balance_J_m=float(abs(balance).max()))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonic",action="store_true",help="also verify FVM storage and phase against modified Bessel functions")
    parser.add_argument("--transient",action="store_true",help="also verify cumulative heat for two decaying Bessel modes")
    args = parser.parse_args()
    rows = []
    for omega in ((0.,.6,6.) if args.harmonic else (0.,)):
        previous = {}
        for n in (8,16,32,64):
            row = measure(n,omega)
            errors = row["fvm_errors"]
            orders = {k:float(np.log2(previous[k]/e)) for k,e in errors.items()} if previous else {}
            passed = row["sheet_balance_W_m2"]<1e-10 and all(v>1.8 for v in orders.values())
            if not omega:
                passed &= max(row["bem_errors"].values())<1e-10 and row["bem_net_power_residual_W_m"]<1e-10
            row.update(fvm_orders=orders,passed=bool(passed))
            rows.append(row)
            previous = errors
    if args.transient:
        for mode,power_walls in ((0,False),(3,False),(0,True),(3,True)):
            previous = {}
            for n in (8,16,32,64):
                row = transient_decay(n,mode,power_walls)
                errors = row["errors"]
                orders = {k:float(np.log2(previous[k]/e)) for k,e in errors.items()} if previous else {}
                row.update(orders=orders,passed=row["sheet_balance_J_m"]<1e-11 and all(v>1.8 for v in orders.values()))
                rows.append(row)
                previous = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
                    (__file__,annulus.__file__,layered_annulus.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
