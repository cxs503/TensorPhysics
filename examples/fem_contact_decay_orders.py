"""Isolate spatial and temporal errors of a LINE2 rod with a resistive interface."""
import json

import numpy as np
import torch
from scipy.linalg import eigh

import fem_contact_decay_mode as case
from conduction_provenance import provenance
from tensorfem import thermal


def modal_reference(model, initial, times):
    # Independently assemble consistent LINE2 mass, bulk stiffness, and the
    # temperature-jump penalty. Eliminate the two fixed zero-temperature ends.
    mass = np.zeros((len(initial), len(initial)))
    stiffness = np.zeros_like(mass)
    jump_form = np.array([[1., -1.], [-1., 1.]])
    for e, nodes in enumerate(model.elements.numpy()):
        h = float(model.nodes[nodes[1], 0]-model.nodes[nodes[0], 0])
        block = np.ix_(nodes, nodes)
        mass[block] += h*float(model.specific_heat[e])/6*np.array([[2., 1.], [1., 2.]])
        stiffness[block] += float(model.conductivity[e])/h*jump_form
    pair = model.contact_pairs.numpy()[0]
    conductance = float(model.contact_conductance)
    stiffness[np.ix_(pair, pair)] += conductance*jump_form
    mass = mass[1:-1, 1:-1]
    rates, modes = eigh(stiffness[1:-1, 1:-1], mass)
    amplitudes = modes.T @ mass @ initial[1:-1]
    exponents = -times[:, None]*rates
    temperatures = (np.exp(exponents)*amplitudes) @ modes.T
    modal_flux = conductance*(modes[pair[0]-1] - modes[pair[1]-1])
    heat = (-np.expm1(exponents)/rates) @ (amplitudes*modal_flux)
    return temperatures, heat


def measure(n, steps, method=None):
    model, initial, decay, initial_flux = case.build_problem(n)
    times = np.linspace(0, 2/decay, steps+1)
    exact_t, exact_h = modal_reference(model, initial, times)
    if method is None:
        computed_t = np.exp(-decay*times[:, None])*initial[1:-1]
        computed_h = -np.expm1(-decay*times)*initial_flux/decay
    else:
        result = thermal.solve_transient_thermal(model, torch.from_numpy(times), torch.from_numpy(initial),
                                                 theta=.5, method=method, sparse=True)
        computed_t = result.temperature.numpy()[:, 1:-1]
        local = thermal.thermal_element_nodal_heat_steps(model, result).numpy()
        left = int(model.contact_pairs[0, 0])
        computed_h = np.r_[0., np.cumsum(-local[:, left-1, 1])]
    return np.array([np.max(abs(computed_t-exact_t)), np.max(abs(computed_h-exact_h))])


if __name__ == "__main__":
    rows = []
    for method in (None, "theta", "sdirk2"):
        previous = None
        levels = (20, 40, 80, 160) if method is None else (16, 32, 64, 128)
        for level in levels:
            n, steps = (level, 128) if method is None else (80, level)
            errors = measure(n, steps, method)
            orders = np.log2(previous/errors) if previous is not None else None
            passed = np.isfinite(errors).all()
            if level == levels[-1]:
                passed &= min(orders) > 1.8 and max(errors) < 1e-4
            rows.append(dict(study="space" if method is None else "time", method=method, elements=n,
                             sample_intervals=steps, temperature_error_K=errors[0], contact_heat_error_J_m2=errors[1],
                             orders=orders.tolist() if orders is not None else None, passed=bool(passed)))
            previous = errors
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(passed=passed, runs=rows, sources=[provenance(p) for p in
          (__file__, case.__file__, thermal.__file__)], reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
          "PYTHONPATH=../TensorFEM/src python examples/fem_contact_decay_orders.py"), indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
