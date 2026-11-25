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
    steady = contact_chain.thermal_chain_calculation(dict(t_hot=case.t_hot, t_cold=base, cells_per_sphere=m))["steady"]
    conductivity = q * case.length / (case.area * drop)
    relative_error = abs(currents/q-1)
    return dict(cold_K=base, drop_K=drop, cells_per_sphere=m, reference_current_W=q,
                payload_flow_relative_error=float(abs(steady["measured_heat_flow_w"]/q-1)),
                payload_jump_relative_error=float(abs(steady["contact_jump_k"]/(q*contact)-1)),
                payload_conductivity_relative_error=float(abs(steady["measured_k_eff_w_m_k"]/conductivity-1)),
                payload_profile_relative_error=steady["max_rel_error"],
                max_link_relative_error=float(relative_error.max()),
                max_contact_relative_error=float(relative_error[chain.contact_links].max()),
                max_temperature_error_K=float(abs(temperature-exact).max()))


if __name__ == "__main__":
    runs = []
    for base in (0., 305.):
        for drop in (2**-12, 2**-24, 2**-44):
            for m in (4, 16, 64):
                row = measure(base, drop, m)
                row["passed"] = (max(row[key] for key in ("max_link_relative_error", "payload_flow_relative_error",
                                                    "payload_jump_relative_error", "payload_conductivity_relative_error")) < 1e-8
                                 and row["max_temperature_error_K"] < 1e-12)
                runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(contact_chain.__file__)],
                          reproduction="PYTHONPATH=../TensorDEM/src python examples/dem_weak_chain_current.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
