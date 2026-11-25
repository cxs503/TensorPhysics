"""Transient exchange between two finite layers initially at different temperatures."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def measure(cells, hot_on_left):
    # L^2 C/k is 4/9 s on each side. Thus one constant interface temperature
    # matches the two Dirichlet half-slab solutions at every positive time.
    left_cells = cells // 3
    dx = 1. / cells
    k = np.repeat([1., 3.], [left_cells, cells - left_cells])
    capacity = np.repeat([4., 3.], [left_cells, cells - left_cells])
    west, east = (1., 0.) if hot_on_left else (0., 1.)
    initial = np.repeat([west, east], [left_cells, cells - left_cells])
    effusivity = np.sqrt([1.*4., 3.*3.])
    interface = np.average([west, east], weights=effusivity)
    diffusion_time = 4./9
    scaled_time = .001 * 2.**np.arange(10)
    times = np.r_[0., diffusion_time * scaled_time]
    temperature, heat = sph.solve_transient_layered_slab(
        initial, times, dx, (west, east), conductivity=k, volumetric_heat_capacity=capacity,
        h_factor=.9, method="exponential", return_x_heat=True)
    heat = heat.numpy()
    temperature = temperature.numpy()

    m = np.arange(1, 129)
    eigenvalue = (np.pi*m)**2
    decay = np.exp(-scaled_time[:, None]*eigenvalue)
    # Unit slab with unit temperature step at the interface end. The time
    # integral of the inward flux is tau + 2 sum((1-exp(-lambda*tau))/lambda).
    unit_inflow = scaled_time + 1./3 - 2*(decay/eigenvalue).sum(axis=1)
    exact_interface = diffusion_time * (west-interface) * 3. * unit_inflow
    # Volume integral of the unit transient uses only odd sine modes.
    mean_rise = .5 - 4*(decay[:, ::2]/eigenvalue[::2]).sum(axis=1)
    exact_stored = mean_rise[:, None] * [(4./3)*(interface-west), 2.*(interface-east)]
    stored_cells = dx*capacity*(temperature-initial)
    stored = np.column_stack((stored_cells[:, :left_cells].sum(axis=1),
                              stored_cells[:, left_cells:].sum(axis=1)))
    measured = heat[:, left_cells]
    reference = np.diff(np.r_[0., exact_interface])
    cut_heat = heat[:, [0, left_cells, cells]]
    residual = np.diff(stored, axis=0) + np.diff(cut_heat, axis=1)
    return dict(cells=cells, west_temperature=west, east_temperature=east,
                exact_interface_temperature=interface,
                first_interval_s=times[1], first_interface_heat_J_m2=measured[0],
                reference_first_interface_heat_J_m2=reference[0],
                max_interval_interface_heat_error_J_m2=float(max(abs(measured-reference))),
                max_left_storage_error_J_m2=float(max(abs(stored[1:, 0]-exact_stored[:, 0]))),
                max_right_storage_error_J_m2=float(max(abs(stored[1:, 1]-exact_stored[:, 1]))),
                max_layer_balance_error_J_m2=float(abs(residual).max()))


if __name__ == "__main__":
    rows = []
    for hot_on_left in (True, False):
        previous = None
        for cells in (48, 96, 192, 384):
            row = measure(cells, hot_on_left)
            errors = np.array([row[key] for key in ("max_interval_interface_heat_error_J_m2",
                              "max_left_storage_error_J_m2", "max_right_storage_error_J_m2")])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(interface_left_right_orders=orders,
                       passed=bool(row["max_layer_balance_error_J_m2"] < 1e-10
                                   and (not orders or min(orders) > 0.)
                                   and (cells != 384 or (min(orders) > 1.85 and max(errors) < 1e-4))))
            rows.append(row)
            previous = errors
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(reference="Two finite layers, matched diffusion times, perfect contact; half-slab Fourier series for flux and stored heat",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_two_material_contact_step.py",
        runs=rows, passed=passed, sources=[provenance(__file__), provenance(sph.__file__)]),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
