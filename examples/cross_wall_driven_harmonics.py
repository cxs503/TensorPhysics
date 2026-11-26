"""Transmission of two wall-temperature harmonics through heat-storing materials."""
import json

import numpy as np

import cross_periodic_contact_reversals as periodic
import periodic_contact_reference as continuum
from conduction_provenance import provenance


def measure(n, solver):
    omegas = 2*np.pi*np.array([1., 2.])
    drive = np.array([1., .25-.15j])
    k = np.repeat([1., 4.], n//2)
    cv = np.repeat([1., 2.], n//2)
    resistance = np.zeros(n-1)
    resistance[n//2-1] = .3
    initial = np.zeros(n)
    expected = []
    cold_reference = []
    storage_reference = []
    for omega, amplitude in zip(omegas, drive):
        _, means, q, inward_walls = continuum.reference(n, .3, omega)
        cold_reference.append(-inward_walls[1])  # positive out through the cold wall
        energy = (cv*means/n).reshape(2, n//2).sum(axis=1)
        storage_reference.append(1j*omega*energy)
        if solver == "sph":
            z = np.sqrt(1j*omega*cv/k)/(2*n)
            means = means*z/np.sinh(z)  # convert exact cell means to particle-centre values
        initial += np.real(amplitude*means)
        expected.append(amplitude*q)
    def west(t):
        return float(np.real(drive @ np.exp(1j*omegas*t)))
    times = np.linspace(0., 1., 4*n+1)
    if solver == "fvm":
        field, _, _, _, ledger = periodic.rod.solve_transient_stepped_rod(
            np.full(n, 1/n), 1., k, cv, lambda t: (west(t), 0.), times, initial,
            contact_resistance=resistance, method="sdirk2", flux_order=2, return_step_heat=True)
        cuts = np.column_stack((ledger["left"][:, 0], ledger["contact"][:, n//2-1], ledger["right"][:, -1]))
    else:
        field, cuts = periodic.sph.solve_transient_layered_slab(
            initial, times, 1/n, (west, 0.), conductivity=k, volumetric_heat_capacity=cv,
            contact_resistance_x=resistance, h_factor=.9, method="sdirk2", return_x_heat=True)
        cuts = cuts[:, [0, n//2, n]].numpy()
        field = field.numpy()
    # Fit the signed interval storage increments with the same basis as the
    # transmitted heats. Each result is a storage-rate phasor per kelvin of drive.
    changes = (np.diff(field, axis=0)*cv/n).reshape(len(times)-1, 2, n//2).sum(axis=2)
    cut_gains = periodic.fit_flux_phasors(times, cuts, omegas)/drive[:, None]
    storage = periodic.fit_flux_phasors(times, changes, omegas)/drive[:, None]
    balance = storage + np.diff(cut_gains, axis=1)
    cold_ratio = cut_gains[:, 2]/cold_reference
    storage_errors = abs(storage/np.array(storage_reference)-1)
    gain = cut_gains[:, 1]
    exact_gain = np.array(expected)/drive
    return dict(columns=n, steps=len(times)-1, solver=solver,
                cold_gain_W_m2_K=abs(cut_gains[:, 2]).tolist(),
                exact_cold_gain_W_m2_K=np.abs(cold_reference).tolist(),
                cold_relative_gain_errors=abs(abs(cold_ratio)-1).tolist(),
                cold_phase_errors_rad=abs(np.angle(cold_ratio)).tolist(),
                storage_rate_relative_errors_by_frequency_layer=storage_errors.tolist(),
                max_modal_layer_balance_W_m2_K=float(np.max(abs(balance))),
                gain_W_m2_K=abs(gain).tolist(), exact_gain_W_m2_K=abs(exact_gain).tolist(),
                phase_rad=np.angle(gain).tolist(), exact_phase_rad=np.angle(exact_gain).tolist(),
                relative_gain_errors=abs(abs(gain/exact_gain)-1).tolist(),
                phase_errors_rad=abs(np.angle(gain/exact_gain)).tolist())


if __name__ == "__main__":
    rows = []
    for solver in ("fvm", "sph"):
        previous = None
        for n in (16, 32, 64, 128):
            row = measure(n, solver)
            errors = np.array([row["relative_gain_errors"], row["phase_errors_rad"]])
            orders = None if previous is None else np.log2(previous/errors)
            passed = np.all(np.isfinite(errors))
            transfer_errors = np.r_[row["cold_relative_gain_errors"], row["cold_phase_errors_rad"],
                                    np.ravel(row["storage_rate_relative_errors_by_frequency_layer"])]
            transfer_orders = None if previous is None else np.log2(previous_transfer/transfer_errors)
            passed &= row["max_modal_layer_balance_W_m2_K"] < 1e-9
            if n == 128:
                passed &= transfer_errors.max() < .001 and transfer_orders.min() > 1.8
                passed &= errors.max() < .001 and orders.min() > 1.8
            previous_transfer = transfer_errors
            row.update(cold_and_storage_orders=None if transfer_orders is None else transfer_orders.tolist(),
                       orders=None if orders is None else orders.tolist(), passed=bool(passed))
            rows.append(row)
            previous = errors
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(passed=passed, runs=rows, frequencies_Hz=[1, 2], sources=[provenance(p) for p in
        (__file__, continuum.__file__, periodic.__file__, periodic.rod.__file__, periodic.sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_wall_driven_harmonics.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
