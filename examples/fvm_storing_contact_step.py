"""Verify separate FVM contact trace capacities against a two-layer Laplace solution."""
import argparse
import json

import numpy as np
from tensorfvm.cht import rod

import sph_resistive_contact_step as model
from conduction_provenance import provenance


def main():
    times = np.r_[0., .00025*2.**np.arange(12)]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", choices=("both", "left", "right"), default="both",
                        help="material side carrying surface heat capacity")
    storage = parser.parse_args().storage
    capacities = (.08 if storage != "right" else 0., .2 if storage != "left" else 0.)
    # A zero capacity gives identically zero stored heat on that side.
    # Check it exactly and compute refinement orders only for nontrivial channels.
    active = np.array([True]*5 + [capacity > 0. for capacity in capacities])
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
            orders = [] if previous is None else np.log2(previous[active]/errors[active]).tolist()
            conservative = (np.all(errors[~active] == 0.)
                            and max(row["max_trace_balance_residual_J_m2"]) < 1e-10
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
                          order_channels=np.array(["west_heat", "contact_heat", "east_heat", "left_mean_trace",
                                                   "right_mean_trace", "left_storage", "right_storage"])[active].tolist(),
                          studies=studies, passed=passed, sources=[provenance(path) for path in
                          (__file__, model.__file__, rod.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_storing_contact_step.py"
                                       + f" --storage {storage}"),
                     indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
