"""Spatial refinement of SPH particle temperatures and heat cuts for a resistive decay mode."""
import json

import numpy as np
from tensorsph import heat_conduction_2d as sph

import contact_decay_reference as reference
from conduction_provenance import provenance


def measure(n, h_factor):
    interface, decay, conductivity, capacity, contact_resistance = reference.material_properties()
    cut = round(n*interface)
    k = np.repeat(conductivity, [cut, n-cut])
    cv = np.repeat(capacity, [cut, n-cut])
    resistance = np.zeros(n-1)
    resistance[cut-1] = contact_resistance
    centers = (np.arange(n)+.5)/n
    wave = np.sqrt(decay*cv/k)
    initial = np.where(centers < cut/n, np.sin(wave*centers), 2*np.sin(wave*(1-centers)))
    times = np.linspace(0, 2/decay, 9)
    temp, heat = sph.solve_transient_layered_slab(
        initial, times, 1/n, (0., 0.), conductivity=k, volumetric_heat_capacity=cv,
        contact_resistance_x=resistance, h_factor=h_factor, method="exponential", return_x_heat=True)
    field = temp.numpy()
    cuts = heat.numpy()[:, [0, cut, n]]
    factor = np.exp(-decay*times)
    error = abs(field-factor[:, None]*initial)
    # Both boundary fluxes and the contact flux are positive toward increasing x.
    initial_flux = np.array([-k[0]*wave[0], -k[0]*wave[0]*np.cos(wave[0]*interface), 2*k[-1]*wave[-1]])
    exact = -np.expm1(-decay*times[1:, None])*initial_flux/decay
    cut_errors = abs(np.cumsum(cuts, axis=0)-exact).max(axis=0)
    change = np.diff(field, axis=0)*cv/n
    storage = np.column_stack((change[:, :cut].sum(axis=1), change[:, cut:].sum(axis=1)))
    return dict(columns=n, h_over_dx=h_factor, layer_temperature_errors_K=[error[:, :cut].max(), error[:, cut:].max()],
                west_contact_east_heat_errors_J_m2=cut_errors.tolist(),
                max_layer_balance_J_m2=float(abs(storage+np.diff(cuts, axis=1)).max()))


if __name__ == "__main__":
    rows = []
    for h_factor in (.9, 1.6):
        previous = None
        for n in (20, 40, 80, 160):
            row = measure(n, h_factor)
            errors = np.array([*row["layer_temperature_errors_K"], *row["west_contact_east_heat_errors_J_m2"]])
            orders = None if previous is None else np.log2(previous/errors).tolist()
            passed = row["max_layer_balance_J_m2"] < 1e-9
            if n == 160:
                passed &= min(orders) > 1.8 and max(errors) < 2e-4
            row.update(orders=orders, passed=bool(passed))
            rows.append(row)
            previous = errors
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(passed=passed, runs=rows, observation_times_s=np.linspace(0, 2/3, 9).tolist(),
        sources=[provenance(p) for p in (__file__, reference.__file__, sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_contact_decay_mode.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
