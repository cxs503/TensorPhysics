"""Compare D3Q7 time stepping and P2 SIPG harmonic fields at shared slab locations."""
import cmath
import json
import math

import torch

from conduction_provenance import provenance
from tensorlbm import thermal3d, thermal_common
from tensordg import heat_1d as dg


def measure(n):
    capacity, alpha, tau, period = 2.3, .075, .8, .5
    omega = 2*math.pi/period
    conductivity = capacity*alpha
    walls = (.4+.15j, -.2+.3j)
    dx, dt = 1/n, 1/n**2
    wave = cmath.sqrt(1j*omega/alpha)
    x = torch.arange(n+1, dtype=torch.float64)*dx
    face = x[:-1]+dx/2
    exact_t = (walls[0]*torch.sinh(wave*(1-x))+walls[1]*torch.sinh(wave*x))/cmath.sinh(wave)
    exact_q = conductivity*wave*(walls[0]*torch.cosh(wave*(1-face))-walls[1]*torch.cosh(wave*face))/cmath.sinh(wave)

    _, modes, _ = dg.solve_harmonic_heat_layer_1d(
        omega, walls, ncells=n//2, degree=2, conductivity=conductivity, capacity=capacity)
    # Each DG cell spans two lattice intervals. Its center is a lattice node;
    # the two lattice links sit one quarter and three quarters across the cell.
    temperature, flux = dg.sample_heat_stack_fields(modes[None], [1.], conductivity, [.25, .5, .75])
    dg_t = temperature[0, :, 1]
    dg_q = flux[0, :, [0, 2]].reshape(-1)

    initial = (1.+exact_t.real).expand(3, 4, -1)
    zero = torch.zeros_like(initial)
    g = thermal3d.equilibrium_thermal_3d(initial, zero, zero, zero)
    steps = round(period/dt)
    lbm_t, lbm_q = torch.zeros_like(dg_t), torch.zeros_like(dg_q)
    for step in range(1, 2*steps+1):
        post = thermal3d.collide_thermal_bgk_3d(g, g.sum(dim=0), zero, zero, zero, tau)
        phase = cmath.exp(1j*omega*step*dt)
        g = thermal3d.apply_temperature_boundaries_3d(
            thermal3d.stream_thermal_3d(post), *(1+(wall*phase).real for wall in walls))
        if step > steps:
            qx, _, _ = thermal_common.thermal_link_fluxes_3d(post, dt=dt, spacing=dx,
                                                            volumetric_heat_capacity=capacity)
            lbm_t += (g.sum(dim=0)[..., 1::2].mean((0, 1))-1)*phase.conjugate()*(2/steps)
            lbm_q += qx.mean((0, 1))*cmath.exp(-1j*omega*(step-.5)*dt)*(2/steps)
    # Link transport is a step average; integrate the DG and continuum
    # harmonics over the same interval before comparing their phasors.
    angle = math.pi/steps
    average = math.sin(angle)/angle
    errors = {}
    for name, first, second, reference in (("temperature_K", lbm_t, dg_t, exact_t[1::2]),
                                           ("flux_W_m2", lbm_q, average*dg_q, average*exact_q)):
        for label, difference in (("lbm", first-reference), ("dg", second-reference),
                                  ("cross", first-second)):
            errors[f"{label}_{name}"] = difference.abs().max().item()
    return errors


if __name__ == "__main__":
    runs = []
    for n in (12, 24, 48, 96):
        errors = measure(n)
        orders = {key: math.log2(runs[-1]["errors"][key]/value)
                  for key, value in errors.items()} if runs else {}
        runs.append(dict(lbm_intervals=n, dg_cells=n//2, errors=errors, orders=orders))
    passed = all(min(row["orders"].values()) > 0 for row in runs[1:]) and min(runs[-1]["orders"].values()) > 1.5
    print(json.dumps(dict(passed=passed, runs=runs,
        reference="Unit slab, period 0.5 s, alpha=0.075, capacity=2.3, wall phasors 0.4+0.15i and -0.2+0.3i about 1 K. Project the second LBM period, compare DG center temperatures and quarter-cell Fourier fluxes to the exact sinh solution. Errors are maxima over shared samples.",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src:../TensorDG python examples/cross_lbm_dg_harmonic_slab.py",
        sources=[provenance(p) for p in (__file__, thermal3d.__file__, thermal_common.__file__, dg.__file__)]),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
