"""Manufactured quadratic temperature modes with independent heat sources in three annuli."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance
from tensorfvm.cht import annulus, storing_rings


def measure(n, mode, omega, layer_ratios=(1, 1, 1)):
    counts = n*np.asarray(layer_ratios)
    ends = np.r_[0, np.cumsum(counts)]
    edges = np.array([.4, .55, .9, 1.1])
    k = np.array([1., 2.5, .8])
    cv = np.array([.7, 2.3, 1.2])
    sheet_cv = np.array([.25, .8])
    sheet_k = np.array([.14, .35])
    resistance = np.array([.5, .9])
    fraction = np.array([.2, .7])
    radius = edges[1:-1]
    # T_i = A_i + B_i*r². Pick successive offsets to impose both contact drops.
    b = np.array([.4+.1j, -.3+.6j, .8-.2j])
    a = np.empty(3, dtype=complex)
    a[0] = .7-.3j
    incoming = -2*k[:-1]*b[:-1]*radius
    outgoing = -2*k[1:]*b[1:]*radius
    sheet = np.empty(2, dtype=complex)
    for i, r in enumerate(radius):
        sheet[i] = a[i]+b[i]*r*r - fraction[i]*resistance[i]*incoming[i]
        a[i+1] = sheet[i]-(1-fraction[i])*resistance[i]*outgoing[i]-b[i+1]*r*r
    admittance = 1j*omega*sheet_cv + sheet_k*(mode/radius)**2
    generation = outgoing-incoming + admittance*sheet
    layer_meshes = [np.geomspace(lo, hi, count+1) for lo, hi, count in zip(edges, edges[1:], counts)]
    mesh = np.r_[np.concatenate([segment[:-1] for segment in layer_meshes]), edges[-1]]
    lo, hi = mesh[:-1], mesh[1:]
    area = np.pi*(hi**2-lo**2)
    # Exact shell integrals of storage, azimuthal divergence, and radial divergence.
    mean_t = np.repeat(a, counts) + np.repeat(b, counts)*(lo**2+hi**2)/2
    stored = 1j*omega*np.repeat(cv, counts)*mean_t*area
    lateral = 2*np.pi*mode**2*np.repeat(k, counts)*(np.repeat(a, counts)*np.log(hi/lo)
                                                + np.repeat(b, counts)*(hi**2-lo**2)/2)
    density = (stored+lateral)/area - np.repeat(4*k*b, counts)
    walls = a[[0,-1]]+b[[0,-1]]*edges[[0,-1]]**2
    expected_wall = -4*np.pi*(k*b)[[0,-1]]*edges[[0,-1]]**2
    temp, qi, qo, rates, storage, angular, field, layer_rates = storing_rings.solve_storing_rings(
        mesh, np.repeat(k, counts), ends[1:-1], walls, omega, contact_resistances=resistance,
        sheet_capacities=sheet_cv, sheet_generation=generation, source=density,
        inner_resistance_fractions=fraction, sheet_conductivities=sheet_k, azimuthal_mode=mode,
        volumetric_heat_capacity=np.repeat(cv, counts), return_wall_rates=True,
        return_bulk_storage=True, return_angular_conduction=True, return_cell_temperatures=True,
        return_layer_face_rates=True)
    layer_face_errors = [float(np.max(abs(rate + 4*np.pi*k[i]*b[i]*layer_meshes[i]**2)))
                         for i, rate in enumerate(layer_rates)]
    face_error = max(layer_face_errors)
    sheet_jumps = np.array([layer_rates[i+1][0] - layer_rates[i][-1] for i in range(2)])
    sheet_residuals = abs(sheet_jumps - 2*np.pi*radius*(generation-admittance*temp))
    cell_storage = 1j*omega*np.repeat(cv, counts)*field*area
    cell_angular = 2*np.pi*mode**2*np.repeat(k, counts)*np.log(hi/lo)*field
    flow_change = np.concatenate([np.diff(rate) for rate in layer_rates])
    cell_balance = np.max(abs(flow_change - (density*area - cell_storage - cell_angular)))
    centroid = (2/3)*(hi**2+hi*lo+lo**2)/(hi+lo)
    temperature_errors = abs(field - np.repeat(a, counts) - np.repeat(b, counts)*centroid**2)
    layer_temperature_errors = np.maximum.reduceat(temperature_errors, ends[:-1])
    field_error = float(max(layer_temperature_errors))
    reference = np.r_[sheet, incoming, outgoing, expected_wall]
    error = float(max(abs(np.r_[temp, qi, qo, rates]/reference-1)))
    storage_error = float(max(abs(storage/np.add.reduceat(stored, ends[:-1])-1)))
    angular_error = float(max(abs(angular/np.add.reduceat(lateral, ends[:-1])-1))) if mode else float(max(abs(angular)))
    sheet_net = 2*np.pi*radius @ (generation-admittance*temp)
    balance = abs(rates[0]-rates[1]+sheet_net+area@density-sum(storage)-sum(angular))
    return dict(**({"cells_per_layer": int(counts[0])} if np.all(counts == counts[0]) else {}),
                base_resolution=n, layer_temperature_errors_K=layer_temperature_errors.tolist(),
                layer_cell_counts=counts.tolist(), layer_face_errors_W_m=layer_face_errors,
                sheet_residuals_W_m=sheet_residuals.tolist(), mode=mode, omega=omega, max_relative_error=error,
                face_rate_max_error_W_m=float(face_error), sheet_balance_max_W_m=float(max(sheet_residuals)),
                cell_balance_max_W_m=float(cell_balance), cell_temperature_max_error_K=field_error, storage_relative_error=storage_error, angular_error=angular_error, modal_balance_W_m=float(balance))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-ratios", type=int, nargs=3, default=(1, 1, 1),
                        help="positive cell-count multipliers from bore to outer wall")
    ratios = parser.parse_args().layer_ratios
    if min(ratios) < 1:
        parser.error("layer ratios must be positive")
    rows = []
    for mode in (0, 1, 3):
        for omega in (1., 8., 24.):
            previous = None
            previous_field_error = None
            previous_face_error = None
            previous_layers = None
            for n in (4, 8, 16, 32):
                row = measure(n, mode, omega, ratios)
                layers = np.array(row["layer_face_errors_W_m"])
                layer_orders = None if previous_layers is None else np.log2(previous_layers/layers).tolist()
                previous_layers = layers
                error = row["max_relative_error"]
                order = None if previous is None else float(np.log2(previous/error))
                field_error = row["cell_temperature_max_error_K"]
                field_order = None if previous_field_error is None else float(np.log2(previous_field_error/field_error))
                previous_field_error = field_error
                face_error = row["face_rate_max_error_W_m"]
                face_order = None if previous_face_error is None else float(np.log2(previous_face_error/face_error))
                previous_face_error = face_error
                balance_errors = (row["modal_balance_W_m"], row["sheet_balance_max_W_m"],
                                  row["cell_balance_max_W_m"])
                passed = max(balance_errors) < 1e-9
                if n == 32:
                    relative_errors = (error, row["storage_relative_error"], row["angular_error"])
                    passed &= max(relative_errors) < .001 and min(order, face_order, field_order) > 1.8
                    passed &= min(layer_orders) > 1.8
                    passed &= face_error < .01 and field_error < .001
                row.update(layer_face_orders=layer_orders, face_rate_order=face_order, cell_temperature_order=field_order,
                           order=order, passed=passed)
                rows.append(row)
                previous = error
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(passed=passed, runs=rows, sources=[provenance(p) for p in
          (__file__, annulus.__file__, storing_rings.__file__)],
          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_volume_heated_rings.py --layer-ratios "
                       + " ".join(map(str, ratios))), indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
