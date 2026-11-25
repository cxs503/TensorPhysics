"""Measure transient SIPG contact heat and one-sided temperatures in a two-layer strip."""
import json

import numpy as np
import torch

import sph_resistive_contact_step as reference_model
from conduction_provenance import provenance
from tensordg import poisson_2d_bc as poisson
from tensordg import transient_diffusion_2d as diffusion


def measure(n, times, resistance, reference):
    interface = 2*n//5
    k = np.repeat([1., 3.], [interface, n-interface])
    capacity = np.repeat([4., 2.], [interface, n-interface])
    contacts = np.zeros((n-1, 2))
    contacts[interface-1] = resistance
    options = dict(bc=dict(left="dirichlet", right="dirichlet", bottom="neumann", top="neumann"),
                   dirichlet_values=dict(left=lambda x,y: torch.ones_like(x+y),
                                         right=lambda x,y: torch.zeros_like(x+y)),
                   conductivity=np.tile(k[:,None], (1,2)), contact_resistance={"x":contacts})
    count = n//5
    time = np.r_[0., np.concatenate([np.linspace(a,b,count+1)[1:] for a,b in zip(times[:-1], times[1:])])]
    _, _, states, means = diffusion.solve_transient_diffusion_2d(
        lambda x,y: (x < .4).double(), time, ncells=(n,2), degree=1,
        capacity=np.tile(capacity[:,None], (1,2)), method="sdirk2", return_interval_means=True, **options)
    dt = np.diff(times)
    # Substeps have equal length within each reporting interval. For constant
    # boundary data the face-power operator is affine in the modal field.
    mean = means.reshape(len(dt), count, *means.shape[1:]).mean(1)
    heat = []
    for width, field in zip(dt, mean):
        power, _ = poisson.poisson_face_powers_2d(field, **options)
        heat.append(width * power[[0, interface, -1]].sum(1).numpy())
    heat = np.array(heat)
    modes = mean[:,:,:,:,0].mean(2).numpy()
    trace = np.column_stack((modes[:,interface-1,:].sum(1), modes[:,interface,0]-modes[:,interface,1]))
    cells = states[::count,:,:,0,0].mean(2).numpy()
    stored = np.diff(cells, axis=0) * capacity/n
    layer_storage = np.column_stack((stored[:,:interface].sum(1), stored[:,interface:].sum(1)))
    return dict(columns=n, substeps_per_interval=count,
                west_interface_east_heat_errors_J_m2=abs(heat-reference[:,:3]).max(0).tolist(),
                left_right_trace_errors_K=abs(trace-reference[:,3:]/dt[:,None]).max(0).tolist(),
                max_jump_residual_K=float(abs(trace[:,0]-trace[:,1]-resistance*heat[:,1]/dt).max()),
                max_layer_balance_error_J_m2=float(abs(layer_storage+np.diff(heat,axis=1)).max()),
                interval_west_interface_east_heat_J_m2=heat.tolist(), interval_left_right_trace_K=trace.tolist())


if __name__ == "__main__":
    times = np.r_[0., .00025*2.**np.arange(12)]
    studies = []
    for resistance in (.05, .3):
        reference = reference_model.interval_reference(times, resistance, "dehoog", include_traces=True)
        rows = []
        previous = None
        for n in (40, 80, 160, 320):
            row = measure(n, times, resistance, reference)
            errors = np.array(row["west_interface_east_heat_errors_J_m2"] + row["left_right_trace_errors_K"])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(orders=orders, passed=bool(row["max_layer_balance_error_J_m2"] < 1e-9
                       and row["max_jump_residual_K"] < 1e-10 and (not orders or min(orders) > 0.)
                       and (n != 320 or (min(orders) > 1.85 and max(errors[:3]) < 5e-5 and max(errors[3:]) < .004))))
            previous = errors
            rows.append(row)
        studies.append(dict(contact_resistance_m2_K_W=resistance,
                            reference_west_interface_east_heat_J_m2=reference[:,:3].tolist(),
                            reference_left_right_trace_K=(reference[:,3:]/np.diff(times)[:,None]).tolist(),
                            runs=rows))
    passed = all(row["passed"] for study in studies for row in study["runs"])
    print(json.dumps(dict(times_s=times.tolist(), studies=studies, passed=passed,
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDG python examples/dg_resistive_contact_step.py",
        sources=[provenance(p) for p in (__file__, reference_model.__file__, poisson.__file__, diffusion.__file__)]),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
