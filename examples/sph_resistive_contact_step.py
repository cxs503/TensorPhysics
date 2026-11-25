"""Transient contact heat between unequal finite layers with a resistive interface."""
import json

import mpmath as mp
import numpy as np

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def interval_reference(times, resistance, method):
    """Invert cumulative heat at west, interface and east before differencing."""
    with mp.workdps(32):
        def transform(s, cut):
            roots = (mp.sqrt(4*s), mp.sqrt(2*s/3))
            lengths = (mp.mpf("0.4"), mp.mpf("0.6"))
            arguments = [length*root for length, root in zip(lengths, roots)]
            impedance = (resistance + mp.tanh(arguments[0])/roots[0]
                         + mp.tanh(arguments[1])/(3*roots[1]))
            # A unit initial jump drives the flux. Integration introduces
            # a second factor 1/s. Each fixed outer wall sees a sech transfer.
            heat = 1 / (s*s*impedance)
            if cut == 1:
                return heat
            wall_argument = arguments[0] if cut == 0 else arguments[1]
            return heat / mp.cosh(wall_argument)

        cumulative = [[float(mp.invertlaplace(lambda s: transform(s, cut), t, method=method))
                       for cut in range(3)] for t in times[1:]]
    return np.diff(np.vstack((np.zeros(3), cumulative)), axis=0)


def measure(n, times, resistance, reference):
    left = 2*n//5
    conductivity = np.repeat([1., 3.], [left, n-left])
    capacity = np.repeat([4., 2.], [left, n-left])
    initial = np.repeat([1., 0.], [left, n-left])
    contacts = np.zeros(n-1)
    contacts[left-1] = resistance
    temperature, heat = sph.solve_transient_layered_slab(
        initial, times, 1./n, (1., 0.), conductivity=conductivity,
        volumetric_heat_capacity=capacity, contact_resistance_x=contacts,
        method="exponential", h_factor=.9, return_x_heat=True)
    measured = heat.numpy()[:, [0, left, n]]
    storage = np.diff(temperature.numpy(), axis=0)*capacity/n
    layer_storage = np.column_stack((storage[:, :left].sum(axis=1), storage[:, left:].sum(axis=1)))
    errors = np.max(abs(measured-reference), axis=0)
    return dict(cells=n, max_west_interface_east_interval_errors_J_m2=errors.tolist(),
                max_layer_balance_error_J_m2=float(abs(layer_storage + np.diff(measured, axis=1)).max()),
                first_interval_mean_interface_flux_W_m2=float(measured[0, 1]/times[1]))


if __name__ == "__main__":
    times = np.r_[0., .00025*2.**np.arange(12)]
    studies = []
    for resistance in (.05, .3):
        reference = interval_reference(times, resistance, "talbot")
        audit = interval_reference(times, resistance, "dehoog")
        discrepancy = float(abs(reference-audit).max())
        rows = []
        previous = None
        for n in (40, 80, 160, 320):
            row = measure(n, times, resistance, reference)
            errors = np.array(row["max_west_interface_east_interval_errors_J_m2"])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(west_interface_east_orders=orders,
                       passed=bool(row["max_layer_balance_error_J_m2"] < 1e-10
                                   and (not orders or min(orders) > 0.)
                                   and (n != 320 or (min(orders) > 1.85 and max(errors) < 5e-5))))
            rows.append(row)
            previous = errors
        studies.append(dict(contact_resistance_m2_K_W=resistance, reference_discrepancy_J_m2=discrepancy,
                            initial_interface_flux_W_m2=1./resistance,
                            reference_first_interval_mean_flux_W_m2=reference[0, 1]/times[1],
                            reference_interval_heat_J_m2=reference.tolist(), runs=rows,
                            passed=discrepancy < 1e-12 and all(row["passed"] for row in rows)))
    passed = all(study["passed"] for study in studies)
    print(json.dumps(dict(reference="Finite-layer Laplace impedances in series; Talbot and de Hoog inversions",
        cut_order=["west", "interface", "east"], times_s=times.tolist(), studies=studies, passed=passed,
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_resistive_contact_step.py",
        sources=[provenance(__file__), provenance(sph.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
