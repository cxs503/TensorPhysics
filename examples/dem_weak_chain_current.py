"""Verify individual bulk and constriction currents in a weakly driven Hertz contact chain."""
import json

import numpy as np
from tensordem.thermal import contact_chain

from conduction_provenance import provenance


def measure(base, drop, m):
    case = contact_chain.ChainCase(t_hot=base+drop, t_cold=base, cells_per_sphere=m)
    chain = contact_chain.ThermalChain(case)
    temperature, currents = chain.solve_steady(return_link_currents=True)
    # Closed form for cumulative series resistance from the hot end to
    # (sphere, local node). Evaluate directly to avoid accumulated rounding.
    bulk = 2/(np.pi*case.conductivity*case.radius)
    contact = 1/(4*case.conductivity*case.contact_radius)
    q = drop/(case.n_spheres*bulk + (case.n_spheres-1)*contact)
    sphere, node = np.divmod(np.arange(temperature.size), m+1)
    exact = case.t_hot - q*(sphere*(bulk+contact) + node*bulk/m)
    relative_error = abs(currents/q-1)
    return dict(cold_K=base, drop_K=drop, cells_per_sphere=m, reference_current_W=q,
                max_link_relative_error=float(relative_error.max()),
                max_contact_relative_error=float(relative_error[chain.contact_links].max()),
                max_temperature_error_K=float(abs(temperature-exact).max()))


if __name__ == "__main__":
    runs = []
    for base in (0., 305.):
        for drop in (2**-12, 2**-24):
            for m in (4, 16, 64):
                row = measure(base, drop, m)
                row["passed"] = row["max_link_relative_error"] < 1e-8 and row["max_temperature_error_K"] < 1e-12
                runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(contact_chain.__file__)],
                          reproduction="PYTHONPATH=../TensorDEM/src python examples/dem_weak_chain_current.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
