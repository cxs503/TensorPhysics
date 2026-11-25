"""A short internal heating pulse excites one decaying mode between fixed-temperature walls."""
import json
import math

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(n, base, amplitude, method):
    x = (torch.arange(n, dtype=torch.float64)+0.5)/n
    mode = torch.cos(2*torch.pi*x)[:, None]*torch.sin(torch.pi*x)
    rate = 5*math.pi**2
    switch = 1/rate
    times = [i*switch/16 for i in range(33)]
    initial = torch.full_like(mode, base)
    fields = heat.solve_transient_slab_2d(
        initial, times, 1/n, (base, base), wall_jump_times=[switch], method=method,
        heat_source=lambda t: amplitude*rate*mode if t < switch else 0.,
        temperature_reference=base,
    )
    factors = torch.tensor([(-math.expm1(-rate*min(t, switch)))*math.exp(-rate*max(0., t-switch))
                            for t in times], dtype=mode.dtype)
    error = abs(fields/amplitude - factors[:, None, None]*mode)
    return dict(columns=n, wall_K=base, amplitude_K=amplitude, method=method,
                max_history_error_over_amplitude=float(error.max()),
                final_error_over_amplitude=float(error[-1].max()))


if __name__ == "__main__":
    runs = []
    for method in ("theta", "sdirk2"):
        for base in (0., 305.):
            for n in (16, 32, 64):
                row = measure(n, base, 2**-44, method)
                row["passed"] = row["max_history_error_over_amplitude"] < 0.01 and (
                    n < 64 or row["max_history_error_over_amplitude"] < 0.001)
                runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_weak_transient_mode.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
