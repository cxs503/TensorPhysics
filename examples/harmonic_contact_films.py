"""Two films and one imperfect contact under harmonic ambient and volumetric heating."""
import json

import numpy as np

from conduction_provenance import provenance
import periodic_contact_heat
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


def measure(n, omega, dynamic=False):
    dx, mid, resistance = 1/n, n//2, .3
    k, c = np.repeat([1.,4.],mid), np.repeat([1.,2.],mid)
    faces = np.zeros(n-1)
    faces[mid-1] = resistance
    films = np.array([1.5, 7.])
    gain, base = .8-.4j, .25+.15j
    _, mean, q, wall_q = periodic_contact_heat.reference(n, resistance, omega)
    z = .5*dx*np.sqrt(1j*omega*c/k)
    ref_walls = base + gain*np.array([1., 0.])
    ref_flux = gain*wall_q
    wall_c = np.array([.4, 1.2]) if dynamic else np.zeros(2)
    ambient = ref_walls+(ref_flux+1j*omega*wall_c*ref_walls)/films
    source = 1j*omega*c*base
    field, walls, inward = sph.solve_harmonic_layered_films(source,dx,ambient,omega,
                                       film_coefficients=films, wall_areal_heat_capacity=wall_c, conductivity=k, volumetric_heat_capacity=c,
                                       contact_resistance_x=faces, backend="banded")
    offsets = sph.lattice_offsets(dx,1.3*dx)
    factor = sph.lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    contact = sph.stencil_x_face_power_2d(field[None,:],offsets,dx,mid,walls,conductivity=k,
                                         layered_x=True,contact_resistance_x=faces)/(dx*factor)
    rows = [("SPH", field.numpy(), base+gain*mean*z/np.sinh(z), walls.numpy(), inward.numpy(), contact.item())]
    bc = {side: dict(type="robin",values=value,film_coefficient=h,wall_areal_heat_capacity=cw)
          for side,value,h,cw in zip(("west","east"),ambient,films,wall_c)}
    fv, powers, sections = energy.solve_harmonic_conduction(np.full(n,dx),[1.],k,c,bc,omega,
                                  source=source,contact_resistance_x=faces[None,:],return_interior_power=True)
    inward = np.array([powers["west"][0],powers["east"][0]])
    rows.append(("FVM",fv[0],base+gain*mean,fv[0,[0,-1]]+inward*dx/(2*k[[0,-1]]),inward,sections["x"][0,mid-1]))
    for name,u,ref,t,qwall,cut in rows:
        store = dx*(1j*omega*c*u-source)
        balance = [store[:mid].sum()-qwall[0]+cut,store[mid:].sum()-qwall[1]-cut]
        yield dict(solver=name, n=n, omega_rad_s=omega, wall_capacity_J_m2K=wall_c.tolist(),
                   wall_balance_W_m2=float(abs(films*(ambient-t)-1j*omega*wall_c*t-qwall).max()), temperature_error_K=float(abs(u-ref).max()),
                   wall_temperature_error_K=float(abs(t-ref_walls).max()),
                   wall_flux_error_W_m2=float(abs(qwall-ref_flux).max()),
                   contact_flux_error_W_m2=float(abs(cut-gain*q)),
                   layer_balance_W_m2=[float(abs(b)) for b in balance])


if __name__ == "__main__":
    rows = []
    for omega, dynamic in ((1.,False),(10.,False),(1.,True),(10.,True)):
        previous = {}
        for n in (16,32,64,128,256):
            for row in measure(n, omega, dynamic):
                errors = np.array([row[key] for key in ("temperature_error_K","wall_temperature_error_K",
                                                       "wall_flux_error_W_m2","contact_flux_error_W_m2")])
                name = row["solver"]
                orders = np.log2(previous[name]/errors).tolist() if name in previous else []
                row.update(orders=orders, passed=bool(all(p>1.8 for p in orders)
                           and max(*row["layer_balance_W_m2"],row["wall_balance_W_m2"]) < 1e-10))
                previous[name] = errors
                rows.append(row)
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
                     (__file__,periodic_contact_heat.__file__,sph.__file__,energy.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
