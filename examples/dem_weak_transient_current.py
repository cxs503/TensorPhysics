"""Verify weak transient hot-end currents against the normalized chain matrix exponential."""
import json

import numpy as np
from tensordem.thermal import contact_chain

from conduction_provenance import provenance


def measure(base, drop, steps):
    options = dict(n_spheres=4, cells_per_sphere=3)
    unit = contact_chain.TransientChain(contact_chain.ChainCase(t_hot=1., t_cold=0., **options))
    chain = contact_chain.TransientChain(contact_chain.ChainCase(t_hot=base+drop, t_cold=base, **options))
    duration = unit.relaxation_time()
    run = chain.march(duration/steps, steps, record_every=steps//4, relative_nodes=True)
    exact = unit.analytic_transient(run["time"])
    exact_flux = drop*unit.link_conductance[0]*(exact[:, 0]-exact[:, 1])
    # Omit the prescribed initial state when measuring the evolution.
    relative = abs(run["hot_flux"][1:]/exact_flux[1:] - 1)
    return dict(cold_K=base, drop_K=drop, steps=steps, max_relative_current_error=float(relative.max()),
                final_current_W=float(run["hot_flux"][-1]), reference_final_current_W=float(exact_flux[-1]),
                max_temperature_error_K=float(abs(run["nodes"] - drop*exact).max()),
                max_temperature_error_over_drop=float(abs(run["nodes"]/drop - exact).max()))


if __name__ == "__main__":
    runs = []
    for base in (0., 305.):
        for drop in (2**-24, 2**-44):
            previous = None
            for steps in (32, 64, 128, 256):
                row = measure(base, drop, steps)
                error = row["max_relative_current_error"]
                order = None if previous is None or error == 0 else float(np.log2(previous/error))
                row.update(order=order, passed=(order is None or order > 0.9)
                           and (steps < 256 or (error < 0.01 and 0 < row["max_temperature_error_over_drop"] < 0.001)))
                previous = error
                runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(contact_chain.__file__)],
                          reproduction="PYTHONPATH=../TensorDEM/src python examples/dem_weak_transient_current.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
