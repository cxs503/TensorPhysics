"""Measure axial and oblique D3Q7 TRT diffusion rates on periodic grids."""
import json
import math

import numpy as np
import torch
from tensorlbm import thermal_common as lbm

from conduction_provenance import provenance


def symbol_rate(wave, n, dt, odd, even):
    velocities = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0],
                           [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    weights = np.array([.25] + [.125]*6)
    identity = np.eye(7)
    reversal = identity[[0, 2, 1, 4, 3, 6, 5]]
    relax = (identity + reversal)/(2*even) + (identity - reversal)/(2*odd)
    collision = identity + relax @ (np.outer(weights, np.ones(7))-identity)
    stream = np.exp((-2j*math.pi/n)*(velocities @ wave))
    roots = np.linalg.eigvals(stream[:, None]*collision)
    slow = roots[np.argmin(abs(roots-1))]
    if abs(slow.imag) > 1e-12 or not 0 < slow.real < 1:
        raise ValueError("expected a real decaying diffusion mode")
    return -math.log(slow.real)/dt


def measure(n, wave, odd, even):
    dt = (odd-.5)/(4*n*n)  # unit physical diffusivity on every mesh
    axes = [torch.arange(n if k else 1, dtype=torch.float64) for k in wave[::-1]]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    pattern = torch.cos((2*math.pi/n)*(wave[0]*x + wave[1]*y + wave[2]*z))
    zero = torch.zeros_like(pattern)
    field = pattern
    g = lbm.thermal_equilibrium_3d(field, zero, zero, zero)
    norm = pattern.square().sum()
    amplitudes = []
    for step in range(1, 61):
        post = lbm.thermal_collide_trt_3d(g, field, zero, zero, zero, odd, tau_even=even)
        g = lbm.thermal_stream_3d(post)
        field = lbm.thermal_macroscopic_3d(g)
        if step in (36, 60):
            amplitudes.append(float((field*pattern).sum()/norm))
    fitted = math.log(amplitudes[0]/amplitudes[1])/(24*dt)
    predicted = symbol_rate(wave, n, dt, odd, even)
    exact = 4*math.pi**2*sum(k*k for k in wave)
    return dict(cells_per_active_axis=n, mode_xyz=wave, dt=dt, tau_odd=odd, tau_even=even,
                continuum_rate=exact, measured_rate=fitted, symbol_rate=predicted,
                relative_error=abs(fitted/exact-1), relative_symbol_mismatch=abs(fitted/predicted-1))


if __name__ == "__main__":
    runs = []
    # In the small-wave expansion, cancelling the fourth-degree terms along
    # (q,0,0) and (q,q,0) cancels both cubic-symmetry invariants (sum k_i^4
    # and sum k_i^2 k_j^2). With a=tau_odd-1/2, b=tau_even-1/2 and m
    # equal nonzero components, the q^4 coefficient of the characteristic
    # equation at the continuum eigenvalue is proportional to
    # 8-48*a*b + m*(12*a*a+12*a*b-3). Thus a*b=1/6 and a*a=1/12.
    for name, odd, even in (("axial", 1.5, 35/36),
                            ("all_directions", .5+math.sqrt(3)/6, .5+math.sqrt(3)/3)):
        for wave in ((1, 0, 0), (1, 1, 0), (1, 1, 1), (2, -1, 1)):
            previous = None
            for n in (16, 32, 64):
                row = measure(n, wave, odd, even)
                order = None if previous is None else math.log2(previous/row["relative_error"])
                target = 3.8 if name == "all_directions" or wave == (1, 0, 0) else 1.8
                row.update(setting=name, order=order, passed=row["relative_symbol_mismatch"] < 1e-8
                           and (n < 64 or order > target))
                runs.append(row)
                previous = row["relative_error"]
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, diffusivity=1., fit_steps=[36, 60],
                          sources=[provenance(f) for f in (__file__, lbm.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_trt_directional_decay.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
