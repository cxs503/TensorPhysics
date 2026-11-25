"""Cross-solver harmonic conduction with independently driven periodic slab walls."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfvm.cht import energy
from tensorlbm import scalar, scalar_harmonic
from tensorsph import heat_conduction_2d as sph


def reference(n, omega, k, capacity):
    x = (np.arange(n)+.5)/n
    temperature = np.zeros((n, n), complex)
    traces = np.zeros((2, n), complex)
    flux = np.zeros_like(traces)
    # Each row is a face-average in y. Temperatures represent square-cell means.
    for m, left, right in ((0, .5, -.15j), (1, 1+.2j, .35j), (-1, 0., .4-.2j)):
        b = 2*np.pi*m
        transverse = np.exp(1j*b*x)*np.sinc(m/n)
        lam = np.sqrt(b*b+1j*omega*capacity/k)
        profile = (left*np.sinh(lam*(1-x))+right*np.sinh(lam*x))/np.sinh(lam)
        temperature += transverse[:, None]*profile*(np.sinh(lam/(2*n))/(lam/(2*n)))
        traces += np.outer([left, right], transverse)
        # Both wall fluxes are positive inward.
        flux += np.outer([left*np.cosh(lam)-right, right*np.cosh(lam)-left],
                         transverse)*k*lam/np.sinh(lam)
    return traces, temperature, flux


def responses(n, omega, k, capacity, traces):
    dx = 1/n
    widths = np.full(n, dx)
    bc = {side: {"type": "dirichlet", "values": trace}
          for side, trace in zip(("west", "east"), traces)}
    t, powers = energy.solve_harmonic_conduction(widths, widths, k, capacity, bc, omega,
                                                 periodic_y=True)
    yield "FVM", t, np.stack((powers["west"], powers["east"]))/dx, omega
    walls = tuple(torch.from_numpy(row) for row in traces)
    t = sph.solve_harmonic_slab_2d(torch.zeros((n, n)), dx, walls, omega, conductivity=k,
                                   volumetric_heat_capacity=capacity)
    offsets = sph.lattice_offsets(dx, 1.3*dx)
    factor = sph.lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    power = []
    for part in ("real", "imag"):
        _, exchange = sph.stencil_rhs_with_wall_power_2d(
            getattr(t, part), offsets, dx, tuple(getattr(w, part) for w in walls),
            conductivity=k, power_by_row=True)
        power.append(exchange/factor)
    yield "SPH", t.numpy(), torch.complex(*power).numpy()/dx, omega
    tau = .9
    dt = scalar.scalar_diffusivity(tau)*dx**2/(k/capacity)
    slab = scalar_harmonic.solve_harmonic_scalar_slab(n, n, tau, omega*dt, *traces)
    q = slab.x_transfer.numpy()
    yield "LBM", slab.scalar.numpy(), np.stack((q[:, 0], -q[:, -1]))*capacity*dx/dt, 2*np.sin(omega*dt/2)/dt


def main():
    runs = []
    k, capacity = 2., 3.
    for omega in (.5, 15.):
        previous = {}
        for n in (16, 32, 64, 128):
            traces, field, flux = reference(n, omega, k, capacity)
            for name, t, q, storage_frequency in responses(n, omega, k, capacity, traces):
                delta = q.mean(axis=1)/flux.mean(axis=1)
                errors = np.array([abs(t-field).max(), np.linalg.norm(q-flux)/np.linalg.norm(flux),
                                   abs(np.angle(delta)).max()])
                balance = abs(1j*storage_frequency*capacity*t.mean()-q.mean(axis=1).sum())
                orders = np.log2(previous[name]/errors) if name in previous else None
                passed = np.isfinite(errors).all() and balance < 1e-10
                if orders is not None:
                    passed &= (orders > [1.7, .45 if name == "SPH" else 1.7, 1.7]).all()
                if n == 128:
                    passed &= (errors < [.001, .015, .002]).all()
                runs.append(dict(solver=name, n=n, omega_rad_s=omega, max_temperature_error_K=errors[0],
                                 relative_wall_flux_L2=errors[1], max_net_wall_phase_error_rad=errors[2],
                                 balance_W=balance, orders=orders.tolist() if orders is not None else None,
                                 passed=bool(passed)))
                previous[name] = errors
    print(json.dumps(dict(conductivity_W_mK=k, capacity_J_m3K=capacity, runs=runs,
          sources=[provenance(f) for f in (__file__, energy.__file__, sph.__file__, scalar.__file__, scalar_harmonic.__file__)]),
          indent=2, allow_nan=False))
    return 0 if all(row["passed"] for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
