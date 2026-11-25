"""Two-layer slab with prescribed multimode wall temperatures and distributed heating."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


def manufactured(n, cell_means, resistance):
    # For every mode, T=A-q*r(x) and -k*dT/dx=q. The resistance coordinate
    # jumps by R at x=1/2. The source is (i*omega*c + k*beta²)*T.
    x = (np.arange(n)+.5)/n
    k = np.where(x < .5, 1., 3.)
    c = np.where(x < .5, 2., 1.)
    r = np.where(x < .5, x, .5+resistance+(x-.5)/3)
    modes = np.array([0, 1, -2])
    amp = np.array([1.+.2j, .4-.1j, .15+.1j])
    q = np.array([.7+.2j, -.4+.3j, .2+.1j])
    beta = 2*np.pi*modes
    phase = np.exp(1j*x[:,None]*beta)
    if cell_means:
        phase *= np.sinc(modes/n)
    modal_t = amp[:,None]-q[:,None]*r
    source = phase @ ((5j*c+beta[:,None]**2*k)*modal_t)
    walls = (phase @ amp, phase @ (amp-q*(.5+resistance+.5/3)))
    return k, c, phase @ modal_t, source, walls, phase @ q


def measure(n):
    dx, resistance = 1/n, .2
    contact = np.zeros(n-1)
    contact[n//2-1] = resistance
    for solver in ("SPH", "FVM"):
        k, c, exact, source, walls, flux = manufactured(n, solver == "FVM", resistance)
        if solver == "SPH":
            material = dict(conductivity=k, contact_resistance_x=contact)
            u = sph.solve_harmonic_layered_slab(source, dx, walls, 5.,
                        volumetric_heat_capacity=c, backend="banded", **material)
            offsets = sph.lattice_offsets(dx, 1.3*dx)
            factor = sph.lattice_diffusion_factor_2d(1.3, dx, mode="moment")
            rate, power = sph.stencil_rhs_with_wall_power_2d(u, offsets, dx, walls,
                                            layered_x=True, **material)
            contact_power = sph.stencil_x_face_power_2d(u, offsets, dx, n//2, walls,
                                       layered_x=True, power_by_row=True, **material)
            field = u.numpy()
            measured = contact_power.numpy()/(factor*dx)
            # Compare the direct Fourier cut with the row-by-row tally for
            # each driven mode, including the one with negative wavenumber.
            modes = np.array([0,1,-2])
            temp_modes = np.fft.fft(field,axis=0)/n
            west,east = (np.fft.fft(data)/n for data in walls)
            actual = np.array([sph.layered_slab_modal_cut_flux(temp_modes[j],dx,n//2,
                    (west[j],east[j]),transverse_wavenumber=2*np.pi*j,**material).item() for j in modes])
            modal_defect = float(abs(actual-np.fft.fft(measured)[modes]/n).max())
            reference_modes = np.fft.fft(flux)[modes]/n
            modal_errors = dict(modal_cut_disagreement_W_m2=modal_defect,
                modal_amplitude_error_W_m2=float(abs(abs(actual)-abs(reference_modes)).max()),
                modal_phase_error_rad=float(abs(np.angle(actual*reference_modes.conj())).max()))
            net = power.sum().item()/factor
            local = (5j*c*field-source-rate.numpy()/factor)*dx**2
        else:
            modal_errors = {}
            widths = np.full(n, dx)
            bc = {side: dict(type="dirichlet", values=value)
                  for side,value in zip(("west","east"),walls)}
            field, power, interior = energy.solve_harmonic_conduction(widths,widths,k,c,bc,5.,
                          source=source, contact_resistance_x=np.broadcast_to(contact,(n,n-1)),
                          periodic_y=True, return_interior_power=True)
            measured = interior["x"][:,n//2-1]/dx
            net = power["west"].sum()+power["east"].sum()
            fx = np.column_stack((power["west"],interior["x"],-power["east"]))
            fy = np.vstack((power["south"],interior["y"],-power["north"]))
            local = (5j*c*field-source)*dx**2+np.diff(fx,axis=1)+np.diff(fy,axis=0)
        yield dict(solver=solver,n=n,**modal_errors,max_temperature_error_K=float(abs(field-exact).max()),
                   max_interface_flux_error_W_m2=float(abs(measured-flux).max()),
                   max_cell_residual_W=float(abs(local).max()),
                   total_residual_W=float(abs((5j*c*field-source).mean()-net)))


if __name__ == "__main__":
    rows, previous = [], {}
    for n in (16,32,64,128,256):
        for row in measure(n):
            name = row["solver"]
            errors = np.array([row["max_temperature_error_K"],row["max_interface_flux_error_W_m2"]])
            if name=="SPH":
                errors = np.r_[errors,row["modal_amplitude_error_W_m2"],row["modal_phase_error_rad"]]
            row["orders"] = np.log2(previous[name]/errors).tolist() if name in previous else []
            row["balance_passed"] = max(row["max_cell_residual_W"],row["total_residual_W"]) < 1e-10
            min_order = .9 if name == "SPH" else 1.8
            row["passed"] = row["balance_passed"] and all(p>min_order for p in row["orders"])
            if name=="SPH":
                row["passed"] &= row["modal_cut_disagreement_W_m2"]<1e-10
            if n == 256:
                limits = (.005, .004) if name == "SPH" else (.0001, .0001)
                row["passed"] = bool(row["passed"] and (errors[:2] < limits).all())
            previous[name] = errors
            rows.append(row)
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
                     (__file__, sph.__file__, energy.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
