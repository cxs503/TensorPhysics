"""Transient heat in a stationary DEM contact chain with two massless interface nodes."""
import json

import numpy as np

import sph_resistive_contact_step as reference_model
from conduction_provenance import provenance
from tensordem.thermal import contact_chain


def measure(n, times, resistance, reference):
    left = 2*n//5
    material = np.arange(n) < left
    contacts = np.zeros(n-1)
    contacts[left-1] = resistance
    links, capacity = contact_chain.cell_centered_slab_network(
        np.full(n, 1./n), np.where(material, 1., 3.), np.where(material, 4., 2.),
        contact_resistance=contacts, resolve_contacts=True)
    cells = np.flatnonzero(capacity > 0.)
    faces = np.flatnonzero(capacity[1:-1] == 0.) + 1
    # The link starting at the left surface is the zero-thickness contact.
    contact_link = faces[0]
    state = np.zeros_like(capacity)
    state[:contact_link+1] = 1.
    stored, cut_heat, temperatures = [], [], []
    for dt in np.diff(times):
        end, heat, mean = contact_chain.affine_contact_interval(
            links, capacity, dt, state, t_left=(1., 1.), t_right=(0., 0.),
            allow_massless=True, return_interval_temperature=True)
        change = (capacity*(end-state))[cells]
        stored.append([change[:left].sum(), change[left:].sum()])
        cut_heat.append(heat[[0, contact_link, -1]])
        temperatures.append(mean[faces])
        state = end

    dt = np.diff(times)
    trace = np.array(temperatures)
    heat = np.array(cut_heat)
    stored = np.array(stored)
    error_heat = abs(heat-reference[:,:3]).max(axis=0)
    error_temperature = abs(trace-reference[:,3:]/dt[:,None]).max(axis=0)
    exact_storage = -np.diff(reference[:,:3], axis=1)
    return dict(cells=n, west_interface_east_heat_error_J_m2=error_heat.tolist(),
                left_right_temperature_error_K=error_temperature.tolist(),
                left_right_storage_error_J_m2=abs(stored-exact_storage).max(axis=0).tolist(),
                max_layer_balance_residual_J_m2=float(abs(stored+np.diff(heat, axis=1)).max()),
                max_jump_residual_K=float(abs(trace[:,0]-trace[:,1]-resistance*heat[:,1]/dt).max()),
                interval_west_interface_east_heat_J_m2=heat.tolist(),
                interval_left_right_storage_change_J_m2=stored.tolist(),
                interval_mean_left_right_temperature_K=trace.tolist())


if __name__ == "__main__":
    times = np.r_[0., .00025*2.**np.arange(12)]
    studies = []
    for resistance in (.05, .3):
        reference = reference_model.interval_reference(times, resistance, "dehoog", include_traces=True)
        rows = []
        previous = None
        for n in (40, 80, 160, 320):
            row = measure(n, times, resistance, reference)
            errors = np.array(row["west_interface_east_heat_error_J_m2"] +
                              row["left_right_temperature_error_K"] + row["left_right_storage_error_J_m2"])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(orders=orders, passed=bool(
                max(row["max_layer_balance_residual_J_m2"], row["max_jump_residual_K"]) < 1e-10
                and (not orders or min(orders) > 0.)
                and (n != 320 or (min(orders) > 1.85 and max(errors[:3]) < 5e-5
                                   and max(errors[3:5]) < .004 and max(errors[5:]) < 5e-5))))
            rows.append(row)
            previous = errors
        studies.append(dict(contact_resistance_m2_K_W=resistance, runs=rows,
                            reference_west_interface_east_heat_J_m2=reference[:,:3].tolist(),
                            reference_left_right_temperature_K=(reference[:,3:]/np.diff(times)[:,None]).tolist()))
    passed = all(row["passed"] for study in studies for row in study["runs"])
    print(json.dumps(dict(studies=studies, times_s=times.tolist(), passed=passed,
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDEM/src python examples/dem_resistive_contact_step.py",
        sources=[provenance(p) for p in (__file__, reference_model.__file__, contact_chain.__file__)]),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
