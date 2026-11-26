"""Temporal convergence at fixed SPH spacing for a decaying resistive-contact mode."""
import json

import numpy as np

import sph_contact_decay_mode as case
from conduction_provenance import provenance


def study(h_factor, n=80, first_step_fraction=.5):
    cut, decay, initial, k, cv, resistance = case.slab_inputs(n)
    report_times = np.linspace(0, 2/decay, 9)
    state, heat = case.solve(initial, k, cv, resistance, report_times, h_factor, "exponential")
    faces = [0, cut, n]
    reference_heat = np.cumsum(heat[:, faces], axis=0)
    rows = []
    for method in ("theta", "sdirk2"):
        previous = None
        for steps in (16, 32, 64, 128):
            stride = steps//8
            times = np.linspace(0, report_times[-1], steps+1)
            # Each pair spans the same duration on all three schedules. Shift
            # only its internal node so all eight report intervals remain aligned.
            times[1::2] = times[:-1:2] + first_step_fraction*(times[2::2]-times[:-1:2])
            field, heat = case.solve(initial, k, cv, resistance, times, h_factor, method)
            difference = abs(field[::stride]-state)
            heat_errors = abs(np.cumsum(heat[:, faces], axis=0)[stride-1::stride]-reference_heat)
            temperature_errors = [difference[:, :cut].max(), difference[:, cut:].max()]
            errors = np.r_[temperature_errors, heat_errors.max(axis=0)]
            orders = None if previous is None else np.log2(previous/errors).tolist()
            stored = np.diff(field, axis=0)*cv/n
            layer_change = np.column_stack((stored[:, :cut].sum(axis=1), stored[:, cut:].sum(axis=1)))
            imbalance = float(np.max(abs(layer_change + np.diff(heat[:, faces], axis=1))))
            cell_heat_residual = stored + np.diff(heat, axis=1)
            cell_imbalance = float(np.max(abs(cell_heat_residual)))
            cumulative_imbalance = float(np.max(abs(np.cumsum(cell_heat_residual, axis=0))))
            passed = max(imbalance, cell_imbalance, cumulative_imbalance) < 1e-9
            if steps == 128:
                passed &= min(orders) > 1.8 and max(errors) < 1e-4
            rows.append(dict(columns=n, steps=steps, method=method, h_over_dx=h_factor,
                             layer_temperature_errors_K=temperature_errors,
                             west_contact_east_heat_errors_J_m2=heat_errors.max(axis=0).tolist(),
                             first_step_fraction=first_step_fraction, min_step_s=float(np.min(np.diff(times))),
                             max_step_s=float(np.max(np.diff(times))), max_cell_balance_J_m2=cell_imbalance,
                             max_cumulative_cell_balance_J_m2=cumulative_imbalance,
                             orders=orders, max_layer_balance_J_m2=imbalance, passed=bool(passed)))
            previous = errors
    return rows


if __name__ == "__main__":
    rows = [row for f in (.5, .25, .75) for h in (.9, 1.6) for row in study(h, first_step_fraction=f)]
    passed = all(r["passed"] for r in rows)
    print(json.dumps(dict(passed=passed, runs=rows, observation_times_s=np.linspace(0, 2/3, 9).tolist(),
        reference="Matrix exponential on the same 80-column SPH operator",
        sources=[provenance(p) for p in (__file__, case.__file__, case.reference.__file__, case.sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_contact_decay_time.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
