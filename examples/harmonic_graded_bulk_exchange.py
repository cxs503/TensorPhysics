"""Temperature phase and power in a graded-loss plate heated by two oscillating reservoirs."""
from dataclasses import replace
import argparse
import inspect
import json
import math

import numpy as np
from scipy.special import airy
import torch

from conduction_provenance import provenance
from two_layer_conduction import slab_fem_model
from tensorfem import thermal
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


K,C = .8,1.6
H0,SLOPE = .7,4.2
FILMS = np.array([2.,5.])
AMBIENT = np.array([1.+.3j,.2-.5j])


def reference(omega,wall_c,materials,resistance,ambient=AMBIENT):
    def basis(layer,x):
        k,c,h,slope = materials[layer]
        scale = (slope/k)**(1/3)
        z = scale*(np.asarray(x)-layer*.5+(h+1j*omega*c)/slope)
        ai,dai,bi,dbi = airy(z)
        return np.stack((ai,bi),axis=-1),-k*scale*np.stack((dai,dbi),axis=-1)

    left,qw = basis(0,0.)
    right,qe = basis(1,1.)
    low,ql = basis(0,.5)
    high,qr = basis(1,.5)
    # Inward material power equals film input less wall storage. Across
    # the contact q is continuous and the temperature drop equals R*q.
    matrix = np.zeros((4,4),dtype=complex)
    matrix[0,:2] = (FILMS[0]+1j*omega*wall_c[0])*left+qw
    matrix[1,2:] = (FILMS[1]+1j*omega*wall_c[1])*right-qe
    matrix[2] = np.r_[ql,-qr]
    matrix[3] = np.r_[low-resistance*ql,-high]
    coeff = np.linalg.solve(matrix,np.r_[FILMS*ambient,0.,0.]).reshape(2,2)
    def evaluate(x,layer=None):
        if layer is not None:
            field,current = basis(layer,x)
            return field @ coeff[layer],current @ coeff[layer]
        left,ql = evaluate(x,0)
        right,qr = evaluate(x,1)
        return np.where(x<.5,left,right),np.where(x<.5,ql,qr)
    return evaluate


def measure(n,omega,wall_c,contact=False,quasistatic=False):
    ambient = AMBIENT.real.copy() if quasistatic else AMBIENT
    dx = 1/n
    edges = np.linspace(0.,1.,n+1)
    x = .5*(edges[:-1]+edges[1:])
    m = n//2
    resistance = .18 if contact else 0.
    materials = [(K,C,H0,SLOPE),(2.3,.9,1.8,2.) if contact else (K,C,H0+.5*SLOPE,SLOPE)]
    k,c,h,slope = np.repeat(np.array(materials).T,m,axis=1)
    relative_x = x-np.repeat([0.,.5],m)
    loss = h+slope*relative_x
    contacts = np.zeros(n-1)
    contacts[m-1] = resistance
    analytic = reference(omega,wall_c,materials,resistance,ambient)
    z,w = np.polynomial.legendre.leggauss(8)
    mean = .5*(analytic(x[:,None]+.5*dx*z)[0] @ w)
    wall_ref,inward_ref = analytic(edges[[0,-1]])
    inward_ref = inward_ref*np.array([1.,-1.])
    cut_ref = analytic(.5)[1]

    model,ends = slab_fem_model(k,0.,0.,resistance,FILMS,torch.from_numpy(c))
    local_x = model.nodes[model.elements,0].numpy()-np.repeat([0.,.5],m)[:,None]
    nodal_loss = torch.from_numpy(h[:,None]+slope[:,None]*local_x)
    model = replace(model,heat_loss_coefficient=nodal_loss,ambient_temperature=torch.from_numpy(ambient),
                    wall_areal_heat_capacity=torch.from_numpy(wall_c))
    state = thermal.solve_harmonic_thermal(model,omega,sparse=True).temperature
    field = state.reshape(2,-1).mean(0).numpy()
    wall = state[ends].mean(1).numpy()
    inward = FILMS*(ambient-wall)-1j*omega*wall_c*wall
    local = thermal.thermal_element_nodal_power(model,state,omega).numpy()
    cut = -local[m-1,[1,2]].sum()
    nodal_ref = (np.r_[analytic(edges[:m+1],0)[0],analytic(edges[m:],1)[0]] if contact else analytic(edges)[0])
    # Two-point Gauss integrates H(x)*T_h(x) exactly in each layer.
    xi = np.array([-1.,1.])/math.sqrt(3)
    ends_u = state[model.elements[:,:2]].numpy()
    u_gauss = ends_u.mean(1)[:,None]+.5*np.diff(ends_u,axis=1)*xi
    volume_power = dx*((loss[:,None]+.5*dx*slope[:,None]*xi+1j*omega*c[:,None])*u_gauss).mean(1)
    rows = [("FEM",field,nodal_ref,wall,inward,cut,volume_power)]

    bc = {side:dict(type="robin",values=a,film_coefficient=h,wall_areal_heat_capacity=cw)
          for side,a,h,cw in zip(("west","east"),ambient,FILMS,wall_c)}
    field,power,interior = energy.solve_harmonic_conduction(np.full(n,dx),[1.],k,c,bc,omega,
                                        heat_loss_coefficient=loss,contact_resistance_x=contacts[None,:],return_interior_power=True)
    inward = np.array([power["west"][0],power["east"][0]])
    wall = field[0,[0,-1]]+.5*dx*inward/k[[0,-1]]
    rows.append(("FVM",field[0],mean,wall,inward,interior["x"][0,m-1],dx*(loss+1j*omega*c)*field[0]))

    field,wall,inward = sph.solve_harmonic_layered_films(np.zeros(n),dx,ambient,omega,film_coefficients=FILMS,
                wall_areal_heat_capacity=wall_c,conductivity=k,volumetric_heat_capacity=c,heat_loss_coefficient=loss,contact_resistance_x=contacts,backend="banded")
    offsets = sph.lattice_offsets(dx,1.3*dx)
    factor = sph.lattice_diffusion_factor_2d(1.3,dx,mode="moment")
    cut = sph.stencil_x_face_power_2d(field[None,:],offsets,dx,m,wall,conductivity=k,layered_x=True,contact_resistance_x=contacts)/(dx*factor)
    rows.append(("SPH",field.numpy(),analytic(x)[0],wall.numpy(),inward.numpy(),cut.item(),dx*(loss+1j*omega*c)*field.numpy()))
    for name,u,ref,wall,inward,cut,volume_power in rows:
        errors = dict(temperature_K=float(abs(u-ref).max()),amplitude_K=float(abs(abs(u)-abs(ref)).max()),
                      phase_rad=float(abs(np.angle(u/ref)).max()),wall_temperature_K=float(abs(wall-wall_ref).max()),
                      inward_power_W_m2=float(abs(inward-inward_ref).max()),cut_power_W_m2=float(abs(cut-cut_ref)))
        if quasistatic:
            # Real drive: T(0) is real and Im(T)/omega tends to the first
            # frequency derivative. The Airy solution is analytic near zero.
            steady = reference(0.,wall_c,materials,resistance,ambient)
            small = 1e-5
            tangent = reference(small,wall_c,materials,resistance,ambient)
            def samples(solution):
                if name=="FEM":
                    return (np.r_[solution(edges[:m+1],0)[0],solution(edges[m:],1)[0]]
                            if contact else solution(edges)[0])
                if name=="FVM":
                    return .5*(solution(x[:,None]+.5*dx*z)[0] @ w)
                return solution(x)[0]
            errors.update(steady_temperature_K=float(abs(u.real-samples(steady).real).max()),
                          temperature_lag_K_s=float(abs(u.imag/omega-samples(tangent).imag/small).max()),
                          steady_cut_power_W_m2=float(abs(cut.real-steady(.5)[1].real)),
                          cut_power_lag_W_s_m2=float(abs(cut.imag/omega-tangent(.5)[1].imag/small)))
        balance = float(abs(FILMS*(ambient-wall)-1j*omega*wall_c*wall-inward).max())
        residual = volume_power.reshape(2,m).sum(1)-inward+np.array([cut,-cut])
        yield dict(max_layer_balance_W_m2=float(abs(residual).max()),solver=name,cells=n,omega=omega,wall_capacity=wall_c.tolist(),errors=errors,wall_balance_W_m2=balance)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact",action="store_true",help="use two different graded materials and a resistive contact")
    parser.add_argument("--quasistatic",action="store_true",help="halve frequency and mesh width together and compare with the DC Airy response and lag")
    args = parser.parse_args()
    contact = args.contact
    rows = []
    for omega,wall_c in ((w,cw) for w in ((.8,) if args.quasistatic else (1.,5.)) for cw in (np.zeros(2),np.array([.4,.9]))):
        previous = {}
        for n in (16,32,64,128):
            frequency = omega*16/n if args.quasistatic else omega
            for row in measure(n,frequency,wall_c,contact,args.quasistatic):
                name,errors = row["solver"],row["errors"]
                orders = {k:math.log2(previous[name][k]/v) for k,v in errors.items()} if name in previous else {}
                passed = max(row["wall_balance_W_m2"],row["max_layer_balance_W_m2"])<1e-10
                if n==128:
                    passed &= (min(orders.values())>1.8 and errors["temperature_K"]<1e-4
                               and errors["phase_rad"]<2e-4 and errors["inward_power_W_m2"]<2e-4
                               and errors["cut_power_W_m2"]<1e-4)
                row.update(orders=orders,passed=passed)
                rows.append(row)
                previous[name] = errors
    sources = [provenance(f) for f in (__file__,inspect.getfile(slab_fem_model),thermal.__file__,energy.__file__,sph.__file__)]
    print(json.dumps(dict(quasistatic=args.quasistatic,contact_resistance_m2K_W=.18 if contact else 0.,sources=sources,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
