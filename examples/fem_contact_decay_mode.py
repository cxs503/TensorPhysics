"""Decay of a discontinuous eigenmode in a rod with finite thermal contact resistance."""
import json

import numpy as np
import torch
from tensorfem import thermal

from conduction_provenance import provenance


def measure(n, method):
    # The first layer spans a quarter of pi in phase, the second three quarters.
    # The outer Dirichlet values vanish. Choose material properties so both
    # pieces decay at the same rate and transmit the same rightward flux.
    interface, decay = .4, 3.
    wl, wr = np.pi/(4*interface), 3*np.pi/(4*(1-interface))
    conductivity = np.array([1., wl/(2*wr)])
    capacity = conductivity*np.array([wl, wr])**2/decay
    resistance = 1/wl
    nl = 2*n//5
    x = np.r_[np.linspace(0, interface, nl+1), np.linspace(interface, 1, n-nl+1)]
    shape = np.r_[np.sin(wl*x[:nl+1]), 2*np.sin(wr*(1-x[nl+1:]))]
    starts = np.r_[np.arange(nl), np.arange(nl+1, n+1)]
    elements = torch.from_numpy(np.column_stack((starts, starts+1)))
    counts = [nl, n-nl]
    model = thermal.ThermalModel(torch.from_numpy(x[:, None]), elements,
        torch.from_numpy(np.repeat(conductivity, counts)), 1., torch.from_numpy(np.repeat(capacity, counts)),
        torch.tensor([0, n+1]), torch.zeros(2, dtype=torch.float64),
        contact_pairs=torch.tensor([[nl, nl+1]]), contact_conductance=1/resistance)
    times = np.linspace(0, 2/decay, n+1)
    result = thermal.solve_transient_thermal(model, torch.from_numpy(times), torch.from_numpy(shape),
                                             theta=.5, method=method, sparse=True)
    errors = abs(result.temperature.numpy() - np.exp(-decay*times[:, None])*shape)
    local = thermal.thermal_element_nodal_heat_steps(model, result).numpy()
    from_left, into_right = -local[:, nl-1, 1], local[:, nl, 0]
    exact_heat = -wl*np.cos(wl*interface)*(-np.expm1(-decay*times[1:]))/decay
    contact_heat_error = float(max(abs(np.cumsum(from_left)-exact_heat)))
    return dict(method=method, elements=n, time_steps=n,
                layer_temperature_errors_K=[float(errors[:, :nl+1].max()), float(errors[:, nl+1:].max())],
                contact_heat_error_J_m2=contact_heat_error,
                max_step_contact_imbalance_J_m2=float(max(abs(from_left-into_right))))


if __name__ == "__main__":
    rows = []
    for method in ("theta", "sdirk2"):
        previous = None
        for n in (20, 40, 80, 160):
            row = measure(n, method)
            errors = np.array([*row["layer_temperature_errors_K"], row["contact_heat_error_J_m2"]])
            orders = None if previous is None else np.log2(previous/errors).tolist()
            passed = row["max_step_contact_imbalance_J_m2"] < 1e-10
            if n == 160:
                passed &= min(orders) > 1.8 and max(errors) < 1e-4
            row.update(orders=orders, passed=bool(passed))
            rows.append(row)
            previous = errors
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(passed=passed, runs=rows, sources=[provenance(p) for p in (__file__, thermal.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src python examples/fem_contact_decay_mode.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
