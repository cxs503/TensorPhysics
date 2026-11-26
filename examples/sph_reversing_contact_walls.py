"""Verify reversing conductive heat flow with independently driven Dirichlet walls."""
import json

import numpy as np
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(n, steps, method):
    cut = 2*n//5
    k = np.repeat([1., 3.], [cut, n-cut])
    capacity = np.repeat([4., 2.], [cut, n-cut])
    resistance = np.zeros(n-1)
    resistance[cut-1] = .3
    # Accumulate thermal resistance from the west wall to each particle.
    halves = .5/(n*k)
    path = np.r_[0., np.cumsum(2*halves[:-1]+resistance)] + halves
    total = path[-1]+halves[-1]
    weight = path/total
    # A linear temperature in resistance coordinates gives the same flux
    # across both materials and the finite jump. Generation supplies storage.
    def west(t):
        return 1.-2*t*t
    def east(t):
        return -.5+t
    def source(t):
        return capacity*((1-weight)*(-4*t)+weight)
    times = np.linspace(0, 1., steps+1)
    exact = (1-weight)*west(times[:, None])+weight*east(times[:, None])
    field, heats = sph.solve_transient_layered_slab(
        exact[0], times, 1/n, (west, east), conductivity=k, volumetric_heat_capacity=capacity,
        heat_source=source, contact_resistance_x=resistance, h_factor=.9,
        method=method, return_x_heat=True)
    cuts = heats.numpy()[:, [0, cut, n]]
    # Positive heat points east. Integrate q=(west-east)/total analytically.
    primitive = (1.5*times-times**2/2-2*times**3/3)/total
    cumulative = np.cumsum(cuts, axis=0)
    step_error = np.max(abs(cuts-np.diff(primitive)[:, None]))
    t_error = float(np.max(abs(field.numpy()-exact)))
    h_errors = np.max(abs(cumulative-primitive[1:, None]), axis=0)
    exact_sign = np.sign(np.diff(primitive))[:, None]
    reversed_flow = bool(np.all(np.sign(cuts) == exact_sign))
    return dict(columns=n, steps=steps, method=method, temperature_error_K=t_error,
                west_contact_east_heat_errors_J_m2=h_errors.tolist(), max_step_heat_error_J_m2=float(step_error),
                reversed_flow=reversed_flow)


if __name__ == "__main__":
    rows = []
    for method in ("theta", "sdirk2", "exponential_quadratic"):
        previous = None
        for steps in (8, 16, 32, 64):
            row = measure(40, steps, method)
            errors = np.r_[row["temperature_error_K"], row["west_contact_east_heat_errors_J_m2"]]
            orders = None
            passed = row["reversed_flow"]
            if method == "exponential_quadratic":
                passed &= max(errors) < 1e-10 and row["max_step_heat_error_J_m2"] < 1e-10
            else:
                if method == "theta":
                    # The trapezoidal rule integrates the linear storage load
                    # exactly. Its endpoint temperature error is roundoff-sized.
                    passed &= errors[0] < 1e-10
                if previous is not None:
                    heat_orders = np.log2(previous[1:]/errors[1:]).tolist()
                    t_order = np.log2(previous[0]/errors[0]) if method == "sdirk2" else None
                    orders = [t_order, *heat_orders]
                    if steps == 64:
                        passed &= max(errors) < 1e-4 and min(heat_orders) > 1.8
                        passed &= t_order is None or t_order > 1.8
            row.update(orders=orders, passed=bool(passed))
            rows.append(row)
            previous = errors
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(passed=passed, runs=rows, sources=[provenance(p) for p in (__file__, sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_reversing_contact_walls.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
