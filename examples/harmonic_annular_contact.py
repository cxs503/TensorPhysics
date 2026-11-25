"""Harmonic radial and angular conduction in a two-material annulus."""
import json

import numpy as np
from scipy.special import iv, ivp, kv, kvp
import torch

from conduction_provenance import provenance
from tensorfvm.cht import annulus
from tensorfem import thermal, thermal_analytic


def measure(n, mode, spacing, *, include_fem=True):
    a, b, c = .25, .6, 1.
    k, capacity = np.array([2.,1.3]), np.array([1.,2.5])
    omega, resistance = 2*np.pi, .15
    walls = np.array([1.+.2j, -.1j])
    # Independent radial solutions of the modified Bessel equation.
    def basis(layer, r):
        z = np.sqrt(1j*omega*capacity[layer]/k[layer])
        temp = np.stack((iv(mode,z*r),kv(mode,z*r)), axis=-1)
        flux = -k[layer]*z*np.stack((ivp(mode,z*r),kvp(mode,z*r)), axis=-1)
        return temp, flux

    ta, qa = basis(0,a)
    ti, qi = basis(0,b)
    to, qo = basis(1,b)
    tc, qc = basis(1,c)
    zeros = np.zeros(2)
    matrix = np.array([np.r_[ta,zeros], np.r_[zeros,tc], np.r_[qi,-qo],
                       np.r_[ti-resistance*qi,-to]])
    ci, co = np.split(np.linalg.solve(matrix, np.r_[walls,0.,0.]), 2)
    q_ref = qi @ ci
    rate_ref = 2*np.pi*np.array([a*(qa@ci), b*q_ref, c*(qc@co)])
    grid = np.geomspace if spacing == "log" else np.linspace
    edges = np.r_[grid(a,b,n+1), grid(b,c,n+1)[1:]]
    centers = .5*(edges[:-1]+edges[1:])
    contact = np.zeros(2*n-1)
    contact[n-1] = resistance
    props = np.repeat(capacity,n)
    exact = np.r_[basis(0,centers[:n])[0]@ci, basis(1,centers[n:])[0]@co]
    volume = np.pi*np.diff(edges**2)
    def row(name, error, sampled_rates, jump, balance):
        return dict(solver=name, spacing=spacing, radial_cells_per_layer=n, angular_mode=mode,
                    temperature_error_K=float(np.max(abs(error))),
                    rate_amplitude_error_W_m=float(np.max(abs(sampled_rates-rate_ref))),
                    jump_error_K=float(abs(jump-resistance*q_ref)),
                    layer_balance_W_m=float(balance))
    rows = []
    films = np.array([7.,3.])
    # With radial flux positive outward, the film ambient lies above the
    # inner wall by q(a)/h_in and below the outer wall by q(c)/h_out.
    ambient = walls + np.array([qa@ci,-qc@co])/films
    for name, wall_data, h in (("FVM", walls, (None,None)), ("FVM_film", ambient, films)):
        t, rates = annulus.solve_annular_conduction(edges, np.repeat(k,n), props, wall_data, omega,
                          contact_resistance=contact, azimuthal_mode=mode, film_coefficients=h)
        angular_loss = 2*np.pi*mode**2*np.repeat(k,n)*np.log(edges[1:]/edges[:-1])*t
        residual = np.diff(rates) + 1j*omega*props*volume*t + angular_loss
        rows.append(row(name, t-exact, rates[[0,n,-1]], resistance*rates[n]/(2*np.pi*b),
                        max(abs(residual[:n].sum()),abs(residual[n:].sum()))))

    if not include_fem:
        return rows

    nt = 8*n
    rings = [thermal_analytic.annular_quad_mesh(n,nt,inner=x,outer=y,dtype=torch.float64)
             for x,y in ((a,b),(b,c))]
    count = len(rings[0].nodes)
    # Use the same radial faces in both solvers.
    directions = rings[0].nodes[:nt] / a
    xy = torch.cat([(torch.from_numpy(radial)[:,None,None]*directions).reshape(-1,2)
                    for radial in (edges[:n+1],edges[n:])])
    conn = torch.cat((rings[0].elements, rings[1].elements+count))
    j = torch.arange(nt)
    fixed = torch.cat((j,j+2*count-nt))
    pairs = torch.stack((count-nt+j,count+j),dim=1)
    angle = 2*np.pi*np.arange(nt)/nt
    harmonic = np.cos(mode*angle)
    projection = harmonic*(1 if mode == 0 else 2)
    chord = 2*b*np.sin(np.pi/nt)
    model = thermal.ThermalModel(xy,conn,torch.from_numpy(np.repeat(k,n*nt)),1.,
            torch.from_numpy(np.repeat(capacity,n*nt)),fixed,torch.from_numpy(np.outer(walls,harmonic).ravel()),
            element_type="q4",contact_pairs=pairs,contact_conductance=chord/resistance)
    sol = thermal.solve_harmonic_thermal(model,omega,sparse=True)
    nodes = sol.temperature.numpy().reshape(2,n+1,nt)
    nodal_ref = np.stack((basis(0,edges[:n+1])[0]@ci, basis(1,edges[n:])[0]@co))
    jump = nodes[0,-1]-nodes[1,0]
    input_power = sol.reaction_power[fixed].numpy().reshape(2,nt)
    contact_rate = chord/resistance*(jump @ projection)
    observed = np.array([input_power[0]@projection,contact_rate,-input_power[1]@projection])
    # Integrate each layer's actual storage and supply using the constant test
    # function. For m>0 the local modal wall rates above also test angular transfer.
    mass = thermal.assemble_thermal(model,sparse=True).capacity.to(sol.temperature.dtype)
    stored = (1j*omega*(mass @ sol.temperature)).reshape(2,-1).sum(dim=1).numpy()
    through = chord/resistance*jump.sum()
    supplied = input_power.sum(axis=1) + np.array([-through,through])
    rows.append(row("FEM", nodes-nodal_ref[:,:,None]*harmonic, observed, jump@projection/nt,
                    np.max(abs(stored-supplied))))
    return rows


def main():
    rows = []
    for spacing in ("linear","log"):
        for mode in (0,2):
            previous = {}
            levels = (4,8,16,32,64,128) if (spacing,mode) == ("linear",2) else (4,8,16,32,64)
            for n in levels:
                for row in measure(n,mode,spacing,include_fem=n<=64):
                    errors = np.array([row[key] for key in ("temperature_error_K","rate_amplitude_error_W_m",
                                                            "jump_error_K")])
                    name = row["solver"]
                    passed = np.isfinite(errors).all() and row["layer_balance_W_m"] < 1e-9
                    if name in previous:
                        row["minimum_error_reduction"] = float(np.min(previous[name]/errors))
                        passed &= (errors < previous[name]).all()
                        if n >= 32:
                            passed &= row["minimum_error_reduction"] > 3.5
                    if n >= 64:
                        passed &= (errors < [.0003, .008, .00002]).all()
                    row["passed"] = bool(passed)
                    previous[name] = errors
                    rows.append(row)
    print(json.dumps({"sources": [provenance(f) for f in
                     (__file__, annulus.__file__, thermal.__file__, thermal_analytic.__file__)],
                     "runs": rows}, indent=2))
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
