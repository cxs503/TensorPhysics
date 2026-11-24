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
    t, powers, interior = energy.solve_harmonic_conduction(widths, widths, kx, capacity, bc, omega,
                                                conductivity_y=ky, return_interior_power=True)
    west_east = np.stack([powers["west"], powers["east"]])/dx
    storage = 1j*omega*capacity*t.mean()
    rows = [("FVM", t-y_avg[:,None]*x_avg, west_east-wall_factor[:,None]*y_avg,
             storage-sum(p.sum() for p in powers.values()))]

    qx = np.column_stack((powers["west"], interior["x"], -powers["east"]))
    qy = np.vstack((powers["south"], interior["y"], -powers["north"]))
    cell_residual = 1j*omega*capacity*dx**2*t + np.diff(qx,axis=1) + np.diff(qy,axis=0)
    exact_qx = y_avg[:,None]*(kx*wave*np.cosh(wave*(1-faces))/np.sinh(wave))
    exact_qy = ky*b*np.sin(b*faces)[:,None]*x_avg
    internal_error = max(abs(qx[:,1:-1]/dx-exact_qx[:,1:-1]).max(),
                         abs(qy[1:-1]/dx-exact_qy[1:-1]).max())

    # The first interior face moves toward the wall under refinement. Measure
    # rates also on the fixed section between the coarsest interior faces.
    cut = (faces >= 1/16) & (faces <= 15/16)
    common_error = max(abs(qx[:,cut]/dx-exact_qx[:,cut]).max(), abs(qy[cut]/dx-exact_qy[cut]).max())
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
    reports = [dict(solver=name, cells=n, transverse_mode=mode,
                 temperature_error_K=float(abs(field).max()), wall_flux_error_W_m2=float(abs(flux).max()),
                 balance_residual_W=float(abs(balance))) for name,field,flux,balance in rows]
    reports[0].update(cell_power_residual_W=float(abs(cell_residual).max()),
                      interior_flux_error_W_m2=float(internal_error), common_section_flux_error_W_m2=float(common_error))
    return reports


def main():
    rows = []
    for mode in (0,1):
        previous = {}
        for n in (16,32,64,128):
            for row in measure(n, mode):
                name = row["solver"]
                errors = np.array([row["temperature_error_K"], row["wall_flux_error_W_m2"]])
                passed = np.isfinite(errors).all() and row["balance_residual_W"] < 1e-10
                if name == "FVM":
                    passed &= row["cell_power_residual_W"] < 1e-11
                    errors = np.append(errors, row["common_section_flux_error_W_m2"])
                if name in previous:
                    passed &= (errors < previous[name]/3.7).all()
                if n == 128:
                    passed &= errors[0] < .00015 and (errors[1:] < .001).all()
                    if name == "FVM":
                        passed &= row["interior_flux_error_W_m2"] < .001
                row["passed"] = bool(passed)
                previous[name] = errors
                rows.append(row)
    print(json.dumps({"sources": [provenance(f) for f in
                      (__file__, energy.__file__, thermal.__file__, thermal_analytic.__file__)],
                      "angular_frequency_rad_s": 2*math.pi, "runs": rows}, indent=2))
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
