"""Verify separate FVM contact trace capacities against a two-layer Laplace solution."""
import json

import numpy as np
from tensorfvm.cht import rod

import sph_resistive_contact_step as model
from conduction_provenance import provenance


def main():
    times = np.r_[0., .00025*2.**np.arange(12)]
    capacities = (.08, .2)
    studies = []
    for resistance in (.05, .3):
        reference = model.interval_reference(times, resistance, "dehoog", include_traces=True,
                                             surface_capacities=capacities, return_trace_changes=True)
        runs = []
        previous = None
        for cells in (40, 80, 160, 320):
            row = model.measure(cells, times, resistance, reference, "FVM", capacities)
            errors = np.array(row["max_west_interface_east_interval_errors_J_m2"]
                              + row["max_left_right_mean_error_K"] + row["max_trace_storage_errors_J_m2"])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            conservative = (max(row["max_trace_balance_residual_J_m2"]) < 1e-10
                            and row["max_layer_balance_error_J_m2"] < 1e-10
                            and row["max_contact_law_residual_K"] < 1e-10)
            resolved = cells < 320 or (min(orders) > 1.85 and max(errors[:3]) < 5e-5
                                       and max(errors[3:5]) < .004 and max(errors[5:]) < .0001)
            row.update(substeps_per_interval=cells//5, orders=orders,
                       passed=bool(conservative and resolved and (not orders or min(orders) > 0.)))
            previous = errors
            runs.append(row)
        studies.append(dict(contact_resistance_m2_K_W=resistance, runs=runs,
                            reference_interval_heat_J_m2=reference[:, :3].tolist(),
                            reference_mean_trace_K=(reference[:, 3:5]/np.diff(times)[:, None]).tolist(),
                            reference_trace_storage_J_m2=(reference[:, 5:7]*capacities).tolist()))
    passed = all(row["passed"] for study in studies for row in study["runs"])
    print(json.dumps(dict(times_s=times.tolist(), surface_capacities_J_m2_K=capacities,
                          studies=studies, passed=passed, sources=[provenance(path) for path in
                          (__file__, model.__file__, rod.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_storing_contact_step.py"),
                     indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
