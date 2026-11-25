"""Complex section power in a two-material rod with graded harmonic generation."""
import argparse
import json

import numpy as np
from numpy.polynomial import Polynomial
import torch

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import rod


def measure(n, omega, oscillatory, estimate_source=False, load_moments=False):
    area = np.repeat([.8,1.4],n)
    k = np.repeat([1.7,3.1],n)
    c = np.repeat([1.5,.9],n)
    resistance = .32  # total K/W over the material overlap
    fractions = np.array([0.,.17,.46,.83,1.])
    gauss,weights = np.polynomial.legendre.leggauss(6)
    shape = np.column_stack(((1-gauss)/2,(1+gauss)/2))
    temp0, power0 = 1.2+.4j,.55-.8j
    nodes, nodal_temp, fields, powers = [],[],[],[]
    loads, sources, slopes, curves = [],[],[],[]
    for layer,length in enumerate((.4,.6)):
        a,cond,cap = area[layer*n],k[layer*n],c[layer*n]
        u = np.linspace(0.,1.,n+1)
        x = length*(.7*u+.3*u*u)
        dx = np.diff(x)
        mid = (x[1:]+x[:-1])/2
        net = (Polynomial([.8+.3j,-1.4+.65j,.6-.9j]) if layer==0
               else Polynomial([-.7+.5j,1.1+.8j,-.4+.55j]))
        poly_power = (a*net).integ(k=power0)
        poly_temperature = (-poly_power/(a*cond)).integ(k=temp0)
        # Add a smooth oscillation with zero initial value and slope in each layer.
        amplitude = (.16-.09j if layer==0 else -.12+.15j) if oscillatory else 0.
        wave = 5.+layer
        def t(z):
            return poly_temperature(z)+amplitude*(np.cos(wave*z)-1)
        def p(z):
            return poly_power(z)+a*cond*amplitude*wave*np.sin(wave*z)
        def generation(z):
            return net(z)+cond*amplitude*wave**2*np.cos(wave*z)+1j*omega*cap*t(z)
        mean_t = (np.diff(poly_temperature.integ()(x))/dx+
                  amplitude*(np.cos(wave*mid)*np.sinc(wave*dx/(2*np.pi))-1))
        samples = x[:-1,None]+dx[:,None]*fractions
        nodes.append(x+(.4 if layer else 0.))
        nodal_temp.append(t(x))
        fields.append(t(samples))
        powers.append(p(samples))
        # Project the continuum generation by Gauss quadrature. FVM uses
        # analytic cell averages for the same input.
        loads.append(a*dx[:,None]/2*((generation(mid[:,None]+dx[:,None]*gauss/2)*weights) @ shape))
        sources.append(np.diff(p(x))/(a*dx)+1j*omega*cap*mean_t)
        slopes.append(net.deriv()(mid)-cond*amplitude*wave**3*np.sin(wave*mid))
        curves.append(net.deriv(2)(mid)-cond*amplitude*wave**4*np.cos(wave*mid))
        power0 = p(length)
        temp0 = t(length)-resistance*power0

    widths = np.concatenate([np.diff(x) for x in nodes])
    contact = np.zeros(2*n-1)
    contact[n-1] = min(area)*resistance
    exact_nodes = np.concatenate(nodal_temp)
    target_p,target_t = np.vstack(powers),np.vstack(fields)
    slope,curve = np.concatenate(slopes),np.concatenate(curves)
    source = np.concatenate(sources)
    jump = exact_nodes[n]-exact_nodes[n+1]
    variants = [("FVM",dict(net_source_gradient=slope,net_source_curvature=curve))]
    if estimate_source:
        variants.append(("FVM_estimated",dict(source_derivative_breaks=[n])))
    for name,options in variants:
        cell,_,west,east,(hot,cold),(grad,bend) = rod.solve_harmonic_stepped_rod(widths,area,k,c,
            exact_nodes[[0,-1]],omega,source=source,flux_order=2,return_traces=True,
            contact_resistance=contact,return_source_derivatives=True,**options)
        field,power = rod.reconstruct_steady_rod_field(widths,area,k,cell,west,east,fractions,
            source_gradient=grad,source_curvature=bend)
        defect = east-west+area*widths*(1j*omega*c*cell-source)
        yield dict(solver=name,section_power_W=float(abs(power-target_p).max()),
                   temperature_K=float(abs(field-target_t).max()),jump_K=float(abs(hot[n-1]-cold[n-1]-jump)),
                   cell_balance_W=float(abs(defect).max()))

    starts = np.r_[np.arange(n),n+1+np.arange(n)]
    conn = np.column_stack((starts,starts+1))
    local = np.vstack(loads)
    rhs = np.zeros(len(exact_nodes),dtype=complex)
    np.add.at(rhs,conn,local)
    model = thermal.ThermalModel(torch.from_numpy(np.concatenate(nodes)[:,None]),torch.from_numpy(conn),
        torch.from_numpy(k),1.,torch.from_numpy(c),torch.tensor([0,2*n+1]),
        torch.from_numpy(exact_nodes[[0,-1]]),thickness=torch.from_numpy(area),nodal_heat=torch.from_numpy(rhs),
        contact_pairs=torch.tensor([[n,n+1]]),contact_conductance=1/resistance)
    state = thermal.solve_harmonic_thermal(model,omega,sparse=True).temperature
    nodal_power,net = thermal.thermal_element_nodal_power(model,state,omega,
        element_nodal_heat=torch.from_numpy(local),return_net_input=True)
    variants = [("FEM",dict(net_line_source_gradient=area*slope,net_line_source_curvature=area*curve))]
    if load_moments:
        variants.append(("FEM_load_moment",dict(net_element_nodal_input=net)))
    element = state.numpy()[conn]
    field = element[:,:1]+(element[:,1:]-element[:,:1])*fractions
    for name,options in variants:
        power = thermal.thermal_element_section_power(model,nodal_power,fractions,**options).numpy()
        defect = power[:,-1]-power[:,0]+1j*omega*area*c*widths*element.mean(1)-local.sum(1)
        yield dict(solver=name,section_power_W=float(abs(power-target_p).max()),
                   amplitude_error_W=float(abs(abs(power)-abs(target_p)).max()),
                   phase_error_rad=float(abs(np.angle(power*target_p.conj())).max()),
                   temperature_K=float(abs(field-target_t).max()),jump_K=float(abs(state[n]-state[n+1]-jump)),
                   cell_balance_W=float(abs(defect).max()))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate-source",action="store_true",help="also estimate net source derivatives from FVM cell means within each material")
    parser.add_argument("--load-moments",action="store_true",help="recover FEM section phasors using net distributed inputs")
    args = parser.parse_args()
    rows = []
    for oscillatory in (False,True):
        for omega in (.7,4.5):
            previous = {}
            for n in (4,8,16,32,64):
                for row in measure(n,omega,oscillatory,args.estimate_source,args.load_moments):
                    solver = row["solver"]
                    errors = {k:row[k] for k in ("temperature_K","section_power_W","jump_K")}
                    if solver=="FEM_load_moment":
                        errors.update({k:row[k] for k in ("amplitude_error_W","phase_error_rad")})
                    refine = solver!="FVM" or oscillatory
                    rates = ({k:float(np.log2(previous[solver][k]/v)) for k,v in errors.items()}
                             if solver in previous and refine else {})
                    passed = row["cell_balance_W"]<1e-10
                    if refine:
                        passed &= all(v>(1.8 if solver.startswith("FEM") else 3.5) for v in rates.values())
                    else:
                        passed &= max(errors.values())<1e-10
                    previous[solver] = errors
                    rows.append(dict(**row,cells_per_material=n,omega_rad_s=omega,oscillatory=oscillatory,
                                     orders=rates,passed=passed))
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in (__file__,rod.__file__,thermal.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
