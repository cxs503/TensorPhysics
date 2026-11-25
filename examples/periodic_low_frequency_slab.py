"""Verify SPH and FVM low-frequency temperature and row-resolved flux on a periodic square."""
import json
import math

import numpy as np
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def modes(n, cell_mean):
    """Evaluate two forced modes of (-k Laplacian + H)T + iω C T = S."""
    k, capacity, loss = 1.7, 2.4, .8
    dx = 1/n
    x = (np.arange(n)+.5)*dx
    y = x[:,None]
    source = np.zeros((n,n),dtype=complex)
    temperature = np.zeros((2,n,n),dtype=complex)
    flux = np.zeros((2,n),dtype=complex)
    for m,l,amplitude in ((1,1,1.+.3j),(3,-2,-.4+.6j)):
        alpha,beta = m*np.pi,2*l*np.pi
        wave = amplitude*np.exp(1j*beta*y)
        forcing = wave*np.sin(alpha*x)
        # FVM uses area averages. Both cut-flux observables use row segments.
        y_average = np.sinc(l*dx)
        if cell_mean:
            forcing *= np.sinc(m*dx/2)*y_average
        source += forcing
        stiffness = loss+k*(alpha**2+beta**2)
        for j,coeff in enumerate((1/stiffness,-capacity/stiffness**2)):
            temperature[j] += coeff*forcing
            flux[j] -= k*alpha*np.cos(alpha/4)*coeff*wave[:,0]*y_average
    return source,temperature,flux


def measure(n,solver):
    dx = 1/n
    widths = np.full(n,dx)
    cut = n//4
    source,temperature,flux = modes(n,solver=="FVM")
    if solver=="FVM":
        bc = {side:dict(type="dirichlet",values=0.) for side in ("west","east")}
        response = energy.solve_low_frequency_conduction(widths,widths,1.7,2.4,bc,
            source=source,heat_loss_coefficient=.8,periodic_y=True,return_interior_power=True)
        fields = [u for u,_,_ in response]
        cuts = [faces["x"][:,cut-1]/dx for _,_,faces in response]
    else:
        conductivity = np.full(n,1.7)
        pair = sph.solve_low_frequency_layered_slab(source,dx,(0.,0.),conductivity=conductivity,
            volumetric_heat_capacity=2.4,heat_loss_coefficient=.8,backend="banded")
        offsets = sph.lattice_offsets(dx,1.3*dx)
        factor = sph.lattice_diffusion_factor_2d(1.3,dx,mode="moment")
        fields = [u.numpy() for u in pair]
        cuts = [sph.stencil_x_face_power_2d(u,offsets,dx,cut,(0.,0.),conductivity=conductivity,
                layered_x=True,power_by_row=True).numpy()/(dx*factor) for u in pair]
    return {key:float(np.max(abs(value-ref))) for key,value,ref in zip(
        ("temperature_K","temperature_lag_K_s","flux_W_m2","flux_lag_W_s_m2"),
        [*fields,*cuts],[*temperature,*flux])}


if __name__=="__main__":
    rows = []
    for solver in ("FVM","SPH"):
        previous = None
        for n in (12,24,48):
            errors = measure(n,solver)
            orders = {k:math.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
            passed = all(math.isfinite(v) for v in errors.values()) and (n<48 or min(orders.values())>1.85)
            rows.append(dict(solver=solver,cells_per_axis=n,errors=errors,orders=orders,passed=passed))
            previous = errors
    print(json.dumps(dict(sources=[provenance(f) for f in (__file__,energy.__file__,sph.__file__)],
                          runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
