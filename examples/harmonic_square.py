"""Local complex temperature and wall flux on an anisotropic unit square."""
import argparse
import inspect
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal, thermal_analytic
from tensorfvm.cht import energy
from tensorlbm import scalar, scalar_harmonic


def measure(n, mode, include_dg=False):
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
    # Even periodic continuation across y=1 gives an insulated unit square.
    tau_x = .9
    dt = scalar.scalar_diffusivity(tau_x)*dx**2/(kx/capacity)
    tau_y = scalar.tau_from_diffusivity(ky/capacity*dt/dx**2)
    slab = scalar_harmonic.solve_harmonic_scalar_slab(n,2*n,tau_x,omega*dt,
                           np.r_[y_avg,y_avg[::-1]],0.,tau_y=tau_y)
    t_lbm = slab.scalar.numpy()[:n]
    scale = capacity*dx/dt
    fx = slab.x_transfer.numpy()[:n]*scale
    full_y = slab.y_transfer.numpy()*scale
    fy = np.vstack((full_y[-1],full_y[:n]))
    # Midpoint link transfers balance the difference of integer-time stores.
    storage = 2j*np.sin(omega*dt/2)/dt*capacity*dx**2*t_lbm
    residual = storage + dx*(np.diff(fx,axis=1)+np.diff(fy,axis=0))
    reports.append(dict(solver="LBM",cells=n,transverse_mode=mode,tau_x=tau_x,tau_y=tau_y,
        temperature_error_K=float(abs(t_lbm-y_avg[:,None]*x_avg).max()),
        wall_flux_error_W_m2=float(abs(np.stack((fx[:,0],-fx[:,-1]))-wall_factor[:,None]*y_avg).max()),
        balance_residual_W=float(abs(residual.sum())),cell_power_residual_W=float(abs(residual).max()),
        interior_flux_error_W_m2=float(max(abs(fx[:,1:-1]-exact_qx[:,1:-1]).max(),
                                          abs(fy[1:-1]-exact_qy[1:-1]).max())),
        common_section_flux_error_W_m2=float(max(abs(fx[:,cut]-exact_qx[:,cut]).max(),
                                                 abs(fy[cut]-exact_qy[cut]).max()))))
    if include_dg:
        from tensordg import solve_harmonic_diffusion_2d, poisson_face_powers_2d

        data = dict(bc={"bottom":"neumann","top":"neumann"},conductivity=(kx,ky),
                    dirichlet_values={"left":lambda x,y:torch.cos(b*y)})
        _,_,coeff = solve_harmonic_diffusion_2d(omega,ncells=n,degree=1,capacity=capacity,**data)
        x_power,y_power = poisson_face_powers_2d(coeff,**data)
        fx,fy = x_power.numpy().T/dx,y_power.numpy().T/dx
        mean = coeff[:,:,0,0].numpy().T
        residual = 1j*omega*capacity*dx**2*mean+dx*(np.diff(fx,axis=1)+np.diff(fy,axis=0))
        reports.append(dict(solver="DG",degree=1,cells=n,transverse_mode=mode,
            temperature_error_K=float(abs(mean-y_avg[:,None]*x_avg).max()),
            wall_flux_error_W_m2=float(abs(np.stack((fx[:,0],-fx[:,-1]))-wall_factor[:,None]*y_avg).max()),
            balance_residual_W=float(abs(residual.sum())),cell_power_residual_W=float(abs(residual).max()),
            common_section_flux_error_W_m2=float(max(abs(fx[:,cut]-exact_qx[:,cut]).max(),
                                                      abs(fy[cut]-exact_qy[cut]).max()))))
    return reports


def main(include_dg=False):
    rows = []
    for mode in (0,1):
        previous = {}
        for n in (16,32,64,128):
            for row in measure(n, mode, include_dg and n<=64):
                name = row["solver"]
                errors = np.array([row["temperature_error_K"], row["wall_flux_error_W_m2"]])
                passed = np.isfinite(errors).all() and row["balance_residual_W"] < 1e-10
                if name in ("FVM","LBM","DG"):
                    passed &= row["cell_power_residual_W"] < 1e-11
                    errors = np.append(errors, row["common_section_flux_error_W_m2"])
                if name in previous:
                    passed &= (errors < previous[name]/3.7).all()
                if name=="DG" and n==64:
                    passed &= errors[0]<.001 and (errors[1:]<.005).all()
                if n == 128:
                    passed &= errors[0] < .00015 and (errors[1:] < .001).all()
                    if name in ("FVM","LBM"):
                        passed &= row["interior_flux_error_W_m2"] < .001
                row["passed"] = bool(passed)
                previous[name] = errors
                rows.append(row)
    sources = [__file__,energy.__file__,thermal.__file__,thermal_analytic.__file__,scalar.__file__,scalar_harmonic.__file__]
    if include_dg:
        from tensordg import solve_harmonic_diffusion_2d, poisson_face_powers_2d
        sources.extend(inspect.getfile(f) for f in (solve_harmonic_diffusion_2d,poisson_face_powers_2d))
    print(json.dumps({"sources": [provenance(f) for f in sources],
                      "angular_frequency_rad_s": 2*math.pi, "runs": rows}, indent=2))
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dg",action="store_true",help="include P1 SIPG on 16, 32 and 64 cells per axis")
    raise SystemExit(main(parser.parse_args().dg))
