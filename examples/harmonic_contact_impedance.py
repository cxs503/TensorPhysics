"""Verify the two-end thermal impedance of a stepped rod with a resistive contact."""
import json

import numpy as np
from scipy.linalg import expm

from conduction_provenance import provenance
from tensorfvm.cht import rod


def measure(n, omega):
    lengths = np.array([.37, .63])
    areas = np.array([.8, 1.4])
    conductivity = np.array([.9, 3.7])
    capacity = np.array([1.6, .85])
    end_capacity = np.array([.28, .63])  # per unit end area
    resistance = .32  # K/W for the whole contact
    storage = 1j*omega*areas*end_capacity

    # Propagate [T, P] with P positive rightward. At the interface the
    # temperature drops by R*P and the power passes through unchanged.
    layers = [expm(length*np.array([[0., -1/(k*a)], [-1j*omega*c*a, 0.]]))
              for length, a, k, c in zip(lengths, areas, conductivity, capacity)]
    transfer = layers[1] @ [[1., -resistance], [0., 1.]] @ layers[0]
    # Both columns use unit inward external power, first at the west end,
    # then at the east. Each external source also supplies its end storage.
    west_state = np.linalg.solve(np.vstack(([storage[0], 1.],
                                  [storage[1], -1.] @ transfer)), np.eye(2))
    exact_z = np.vstack((west_state[0], (transfer @ west_state)[0]))
    contact_power = (layers[0] @ west_state)[1]

    widths = np.repeat(lengths/n, n)
    contact = np.zeros(2*n-1)
    contact[n-1] = resistance*min(areas)
    columns = []
    jump_error = power_error = continuity = balance = 0.
    for end, drive in enumerate(np.eye(2)):
        cell, faces, west, east, (left_t, right_t) = rod.solve_harmonic_stepped_rod(
            widths, np.repeat(areas, n), np.repeat(conductivity, n), np.repeat(capacity, n),
            drive, omega, end_types=("power", "power"), end_areal_heat_capacity=end_capacity,
            contact_resistance=contact, flux_order=2, return_traces=True)
        ends = faces[[0, -1]]
        columns.append(ends)
        jump_error = max(jump_error, abs(left_t[n-1]-right_t[n-1]-resistance*contact_power[end]))
        power_error = max(power_error, abs(east[n-1]-contact_power[end]), abs(west[n]-contact_power[end]))
        continuity = max(continuity, abs(east[n-1]-west[n]))
        bulk_storage = 1j*omega*np.dot(widths*np.repeat(areas*capacity, n), cell)
        balance = max(balance, abs(drive.sum()-storage @ ends-bulk_storage))

    z = np.column_stack(columns)
    # For arbitrary complex drives d the real part of dᴴ Z d must be
    # nonnegative for a passive rod. The Hermitian part tests all drives.
    min_dissipation = np.linalg.eigvalsh((z+z.conj().T)/2)[0]
    return dict(cells_per_material=n, omega_rad_s=omega,
                impedance_K_W=np.stack((z.real, z.imag), axis=-1).tolist(),
                exact_impedance_K_W=np.stack((exact_z.real, exact_z.imag), axis=-1).tolist(),
                impedance_error_K_W=float(abs(z-exact_z).max()), contact_jump_error_K=float(jump_error),
                contact_power_error_W=float(power_error), power_discontinuity_W=float(continuity),
                reciprocity_error_K_W=float(abs(z[0, 1]-z[1, 0])), energy_residual_W=float(balance),
                min_dissipative_impedance_K_W=float(min_dissipation))


if __name__ == "__main__":
    rows = []
    for omega in (.5, 4., 12.):
        previous = None
        for n in (4, 8, 16, 32, 64):
            row = measure(n, omega)
            errors = np.array([row[name] for name in (
                "impedance_error_K_W", "contact_jump_error_K", "contact_power_error_W")])
            orders = np.log2(previous/errors).tolist() if previous is not None else []
            row.update(orders=orders, passed=bool(all(p > 1.8 for p in orders)
                       and (n < 64 or max(errors) < 2e-5)
                       and max(row["power_discontinuity_W"], row["energy_residual_W"],
                               row["reciprocity_error_K_W"]) < 1e-10
                       and row["min_dissipative_impedance_K_W"] > 0.))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(reference="Layer matrix exponentials for [temperature, rightward power], contact drop R*P",
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/harmonic_contact_impedance.py",
                          sources=[provenance(f) for f in (__file__, rod.__file__)],
                          runs=rows), indent=2, allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
