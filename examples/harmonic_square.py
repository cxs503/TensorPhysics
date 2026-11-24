"""Local complex temperature and wall flux on an anisotropic unit square."""
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal, thermal_analytic
from tensorfvm.cht import energy


def measure(n, mode):
    kx, ky, capacity, omega = 2., .5, 3., 2*math.pi
    b = mode*math.pi
    wave = np.sqrt((ky*b*b+1j*omega*capacity)/kx)
    faces = np.linspace(0., 1., n+1)
    centers = .5*(faces[1:]+faces[:-1])
    dx = 1/n
    y_avg = np.cos(b*centers)*np.sinc(mode*dx/2)
    x_trace = np.sinh(wave*(1-faces))/np.sinh(wave)
    x_avg = -np.diff(np.cosh(wave*(1-faces)))/(dx*wave*np.sinh(wave))
    wall_factor = kx*wave*np.array([np.cosh(wave), -1.])/np.sinh(wave)
    bc = {"west": {"type": "dirichlet", "values": y_avg},
          "east": {"type": "dirichlet", "values": 0.}}
    widths = np.full(n, dx)
    t, powers = energy.solve_harmonic_conduction(widths, widths, kx, capacity, bc, omega,
                                                conductivity_y=ky)
    west_east = np.stack([powers["west"], powers["east"]])/dx
    storage = 1j*omega*capacity*t.mean()
    rows = [("FVM", t-y_avg[:,None]*x_avg, west_east-wall_factor[:,None]*y_avg,
             storage-sum(p.sum() for p in powers.values()))]

    mesh = thermal_analytic.structured_quad_mesh(n,n,dtype=torch.float64)
    nodes = mesh.nodes
    ids = torch.arange(len(nodes)).reshape(n+1,n+1)
    walls = ids[:,[0,-1]].T.contiguous()
    exact = np.cos(b*faces)[:,None]*x_trace
    fixed = walls.flatten()
    model = thermal.ThermalModel(nodes, mesh.elements, torch.diag(nodes.new_tensor([kx,ky]))[None],
                                 1., capacity, fixed, torch.from_numpy(exact.ravel())[fixed], element_type="q4")
    solution = thermal.solve_harmonic_thermal(model, omega, sparse=True)
    t = solution.temperature.numpy().reshape(n+1,n+1)
    area = np.full(n+1, dx)
    area[[0,-1]] *= .5
    # Hat-weighted local wall load per tributary area, including half hats at the corners.
    projected = np.cos(b*faces)*np.sinc(mode*dx/2)**2
    power = solution.reaction_power[walls].numpy()
    storage = 1j*omega*capacity*(area @ t @ area)
    rows.append(("FEM", t-exact, power/area-wall_factor[:,None]*projected, storage-power.sum()))
    return [dict(solver=name, cells=n, transverse_mode=mode,
                 temperature_error_K=float(abs(field).max()), wall_flux_error_W_m2=float(abs(flux).max()),
                 balance_residual_W=float(abs(balance))) for name,field,flux,balance in rows]


def main():
    rows = []
    for mode in (0,1):
        previous = {}
        for n in (16,32,64,128):
            for row in measure(n, mode):
                name = row["solver"]
                errors = np.array([row["temperature_error_K"], row["wall_flux_error_W_m2"]])
                passed = np.isfinite(errors).all() and row["balance_residual_W"] < 1e-10
                if name in previous:
                    passed &= (errors < previous[name]/3.7).all()
                if n == 128:
                    passed &= (errors < [.00015,.001]).all()
                row["passed"] = bool(passed)
                previous[name] = errors
                rows.append(row)
    print(json.dumps({"sources": [provenance(f) for f in
                      (__file__, energy.__file__, thermal.__file__, thermal_analytic.__file__)],
                      "angular_frequency_rad_s": 2*math.pi, "runs": rows}, indent=2))
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
