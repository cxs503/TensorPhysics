"""Compare cell-mean temperatures and transmitted heat during a two-material decay mode."""
import json

import numpy as np
import torch
from tensorfvm.cht import rod

import fem_contact_decay_mode as case
from conduction_provenance import provenance


def measure(n, method):
    model, shape, decay, initial_flux = case.build_problem(n)
    times = np.linspace(0, 2/decay, n+1)
    nodes = model.elements.numpy()
    x = model.nodes.numpy()[:, 0]
    dx = x[nodes[:, 1]] - x[nodes[:, 0]]
    k = model.conductivity.numpy()
    capacity = model.specific_heat.numpy()
    cut = int(model.contact_pairs[0, 0])
    resistance = np.zeros(n-1)
    resistance[cut-1] = 1/float(model.contact_conductance)
    # On each sinusoidal piece, the endpoint average equals the midpoint
    # value times cos(w*dx/2). Integrating gives the exact cell mean.
    half_phase = np.sqrt(decay*capacity/k)*dx/2
    initial_mean = shape[nodes].mean(axis=1)*np.sinc(half_phase/np.pi)/np.cos(half_phase)
    fv_t, _, _, _, ledger = rod.solve_transient_stepped_rod(
        dx, 1., k, capacity, (0., 0.), times, initial_mean, contact_resistance=resistance,
        method=method, flux_order=2, return_step_heat=True)
    fv_heat = np.cumsum(ledger["contact"][:, cut-1])
    result = case.thermal.solve_transient_thermal(model, torch.from_numpy(times), torch.from_numpy(shape),
                                                 theta=.5, method=method, sparse=True)
    fe_t = result.temperature.numpy()[:, nodes].mean(axis=2)
    local = case.thermal.thermal_element_nodal_heat_steps(model, result).numpy()
    fe_heat = np.cumsum(-local[:, cut-1, 1])
    exact_t = np.exp(-decay*times[:, None])*initial_mean
    exact_heat = -initial_flux*np.expm1(-decay*times[1:])/decay
    residual = np.diff(fv_t, axis=0)*capacity*dx + ledger["right"] - ledger["left"]
    def errors(t1, t2, h1, h2):
        return [float(np.max(abs(t1-t2))), float(np.max(abs(h1-h2)))]
    return dict(elements=n, steps=n, method=method, fem=errors(fe_t, exact_t, fe_heat, exact_heat),
                fvm=errors(fv_t, exact_t, fv_heat, exact_heat),
                difference=errors(fe_t, fv_t, fe_heat, fv_heat),
                initial_cell_mean_difference_K=float(max(abs(fe_t[0]-fv_t[0]))),
                final_cell_mean_difference_K=float(max(abs(fe_t[-1]-fv_t[-1]))),
                fvm_cell_balance_J_m2=float(np.max(abs(residual))))


if __name__ == "__main__":
    rows = []
    for method in ("theta", "sdirk2"):
        previous = None
        for n in (20, 40, 80, 160):
            row = measure(n, method)
            errors = np.array([row["fem"], row["fvm"]])
            orders = np.log2(previous/errors) if previous is not None else None
            passed = row["fvm_cell_balance_J_m2"] < 1e-10
            if n == 160:
                passed &= np.min(orders) > 1.8 and np.max(errors) < 2e-4 and max(row["difference"]) < 2e-4
            row.update(orders=None if orders is None else orders.tolist(), passed=bool(passed))
            rows.append(row)
            previous = errors
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(passed=passed, runs=rows, error_columns=["cell_mean_temperature_K", "contact_heat_J_m2"],
                          sources=[provenance(p) for p in (__file__, case.__file__, case.thermal.__file__, rod.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src:../TensorFVM/src python examples/fem_fvm_contact_decay.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
