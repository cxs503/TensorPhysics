"""LINE2 transient heat and temperatures on either side of a resistive interface."""
import json

import numpy as np
import torch

import sph_resistive_contact_step as benchmark
from conduction_provenance import provenance
from tensorfem import thermal


def measure(n, times, resistance, reference):
    left = 2*n//5
    grid = torch.linspace(0., 1., n+1, dtype=torch.float64)
    x = torch.cat((grid[:left+1], grid[left:]))
    first = torch.cat((torch.arange(left), torch.arange(left+1, n+1)))
    elements = torch.stack((first, first+1), dim=1)
    capacity = grid.new_tensor([4.]*left + [2.]*(n-left))
    model = thermal.ThermalModel(
        x[:, None], elements, grid.new_tensor([1.]*left + [3.]*(n-left)), 1., capacity,
        torch.tensor([0, n+1]), grid.new_tensor([1., 0.]),
        contact_pairs=torch.tensor([[left, left+1]]), contact_conductance=1./resistance)
    initial = grid.new_tensor([1.]*(left+1) + [0.]*(n-left+1))
    count = n//5
    time = np.r_[0., np.concatenate([np.linspace(a,b,count+1)[1:]
                                     for a,b in zip(times[:-1],times[1:])])]
    result = thermal.solve_transient_thermal(model, torch.from_numpy(time), initial,
                                             sparse=True, method="sdirk2")
    local = thermal.thermal_element_nodal_heat_steps(model, result).numpy()
    # Two independent readings of heat crossing the cut: out of the last
    # left element and into the first right element.
    cuts = np.column_stack((local[:,0,0], -local[:,left-1,1], local[:,left,0], -local[:,-1,1]))
    cuts = cuts.reshape(-1,count,4).sum(axis=1)
    trace = result.interval_temperature[:,left:left+2].numpy()
    trace = (trace*np.diff(time)[:,None]).reshape(-1,count,2).sum(axis=1)/np.diff(times)[:,None]
    exact_trace = reference[:,3:]/np.diff(times)[:,None]
    change = result.temperature[::count,:].diff(dim=0)
    stored = (change[:,elements].mean(dim=2)*capacity/n).numpy()
    storage = np.column_stack((stored[:,:left].sum(axis=1), stored[:,left:].sum(axis=1)))
    balance = storage + cuts[:,[1,3]] - cuts[:,[0,2]]
    errors = abs(cuts[:,[0,1,3]] - reference[:,:3]).max(axis=0)
    jump_error = abs(trace[:,0]-trace[:,1]-resistance*cuts[:,1]/np.diff(times))
    return dict(elements=n, substeps_per_interval=count,
                interval_left_right_trace_K=trace.tolist(),
                interval_west_interface_east_heat_J_m2=cuts[:,[0,1,3]].tolist(),
                west_interface_east_heat_errors_J_m2=errors.tolist(),
                left_right_trace_errors_K=abs(trace-exact_trace).max(axis=0).tolist(),
                max_interface_imbalance_J_m2=float(max(abs(cuts[:,1]-cuts[:,2]))),
                max_jump_residual_K=float(max(jump_error)), max_layer_residual_J_m2=float(abs(balance).max()))


if __name__ == "__main__":
    times = np.r_[0., .00025*2.**np.arange(12)]
    studies = []
    for resistance in (.05, .3):
        reference = benchmark.interval_reference(times, resistance, "dehoog", include_traces=True)
        rows = []
        previous = None
        for n in (40, 80, 160, 320):
            row = measure(n, times, resistance, reference)
            errors = np.array(row["west_interface_east_heat_errors_J_m2"]+row["left_right_trace_errors_K"])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(orders=orders, passed=bool(row["max_jump_residual_K"] < 1e-10
                       and max(row["max_interface_imbalance_J_m2"], row["max_layer_residual_J_m2"]) < 1e-10
                       and (not orders or min(orders) > 0.)
                       and (n != 320 or (min(orders) > 1.8 and max(errors[:3]) < 5e-5 and max(errors[3:]) < .004))))
            rows.append(row)
            previous = errors
        studies.append(dict(resistance_m2_K_W=resistance,
                            reference_west_interface_east_heat_J_m2=reference[:,:3].tolist(),
                            reference_left_right_trace_K=(reference[:,3:]/np.diff(times)[:,None]).tolist(),
                            runs=rows))
    passed = all(row["passed"] for study in studies for row in study["runs"])
    print(json.dumps(dict(studies=studies, times_s=times.tolist(), passed=passed,
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src python examples/fem_resistive_contact_step.py",
        sources=[provenance(p) for p in (__file__, benchmark.__file__, thermal.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
