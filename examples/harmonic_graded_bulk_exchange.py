"""Temperature phase and power in a graded-loss plate heated by two oscillating reservoirs."""
from dataclasses import replace
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


def reference(omega,wall_c):
    scale = (SLOPE/K)**(1/3)
    def basis(x):
        z = scale*(np.asarray(x)+(H0+1j*omega*C)/SLOPE)
        ai,dai,bi,dbi = airy(z)
        return np.stack((ai,bi),axis=-1),-K*scale*np.stack((dai,dbi),axis=-1)
    u,q = basis([0.,1.])
    # Inward material power is the film input minus the surface storage rate.
    matrix = (FILMS+1j*omega*wall_c)[:,None]*u+np.array([[1.],[-1.]])*q
    coeff = np.linalg.solve(matrix,FILMS*AMBIENT)
    def evaluate(x):
        field,current = basis(x)
        return field @ coeff,current @ coeff
    return evaluate


def measure(n,omega,wall_c):
    dx = 1/n
    edges = np.linspace(0.,1.,n+1)
    x = .5*(edges[:-1]+edges[1:])
    k,c = np.full(n,K),np.full(n,C)
    loss = H0+SLOPE*x
    analytic = reference(omega,wall_c)
    z,w = np.polynomial.legendre.leggauss(8)
    mean = .5*(analytic(x[:,None]+.5*dx*z)[0] @ w)
    wall_ref,inward_ref = analytic(edges[[0,-1]])
    inward_ref = inward_ref*np.array([1.,-1.])
    cut_ref = analytic(.5)[1]
    m = n//2

    model,ends = slab_fem_model(k,0.,0.,0.,FILMS,torch.from_numpy(c))
    nodal_loss = H0+SLOPE*model.nodes[model.elements,0]
    model = replace(model,heat_loss_coefficient=nodal_loss,ambient_temperature=torch.from_numpy(AMBIENT),
                    wall_areal_heat_capacity=torch.from_numpy(wall_c))
    state = thermal.solve_harmonic_thermal(model,omega,sparse=True).temperature
    field = state.reshape(2,-1).mean(0).numpy()
    wall = state[ends].mean(1).numpy()
    inward = FILMS*(AMBIENT-wall)-1j*omega*wall_c*wall
    local = thermal.thermal_element_nodal_power(model,state,omega).numpy()
    cut = -local[m-1,[1,2]].sum()
    rows = [("FEM",field,analytic(edges)[0],wall,inward,cut)]

    bc = {side:dict(type="robin",values=a,film_coefficient=h,wall_areal_heat_capacity=cw)
          for side,a,h,cw in zip(("west","east"),AMBIENT,FILMS,wall_c)}
    field,power,interior = energy.solve_harmonic_conduction(np.full(n,dx),[1.],k,c,bc,omega,
                                        heat_loss_coefficient=loss,return_interior_power=True)
    inward = np.array([power["west"][0],power["east"][0]])
    wall = field[0,[0,-1]]+.5*dx*inward/K
    rows.append(("FVM",field[0],mean,wall,inward,interior["x"][0,m-1]))

    field,wall,inward = sph.solve_harmonic_layered_films(np.zeros(n),dx,AMBIENT,omega,film_coefficients=FILMS,
                wall_areal_heat_capacity=wall_c,conductivity=k,volumetric_heat_capacity=c,heat_loss_coefficient=loss,backend="banded")
    offsets = sph.lattice_offsets(dx,1.3*dx)
    factor = sph.lattice_diffusion_factor_2d(1.3,dx,mode="moment")
    cut = sph.stencil_x_face_power_2d(field[None,:],offsets,dx,m,wall,conductivity=k,layered_x=True)/(dx*factor)
    rows.append(("SPH",field.numpy(),analytic(x)[0],wall.numpy(),inward.numpy(),cut.item()))
    for name,u,ref,wall,inward,cut in rows:
        errors = dict(temperature_K=float(abs(u-ref).max()),amplitude_K=float(abs(abs(u)-abs(ref)).max()),
                      phase_rad=float(abs(np.angle(u/ref)).max()),wall_temperature_K=float(abs(wall-wall_ref).max()),
                      inward_power_W_m2=float(abs(inward-inward_ref).max()),cut_power_W_m2=float(abs(cut-cut_ref)))
        balance = float(abs(FILMS*(AMBIENT-wall)-1j*omega*wall_c*wall-inward).max())
        yield dict(solver=name,cells=n,omega=omega,wall_capacity=wall_c.tolist(),errors=errors,wall_balance_W_m2=balance)


if __name__=="__main__":
    rows = []
    for omega,wall_c in ((1.,np.zeros(2)),(5.,np.array([.4,.9]))):
        previous = {}
        for n in (16,32,64,128):
            for row in measure(n,omega,wall_c):
                name,errors = row["solver"],row["errors"]
                orders = {k:math.log2(previous[name][k]/v) for k,v in errors.items()} if name in previous else {}
                row.update(orders=orders,passed=row["wall_balance_W_m2"]<1e-10 and (n<128 or min(orders.values())>1.8))
                rows.append(row)
                previous[name] = errors
    sources = [provenance(f) for f in (__file__,inspect.getfile(slab_fem_model),thermal.__file__,energy.__file__,sph.__file__)]
    print(json.dumps(dict(sources=sources,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
