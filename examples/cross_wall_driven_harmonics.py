"""Transmission of two wall-temperature harmonics through heat-storing materials."""
import argparse
import json

import numpy as np

import cross_periodic_contact_reversals as periodic
import periodic_contact_reference as continuum
from conduction_provenance import provenance


def solve_wall_histories(solver, k, cv, resistance, initial, times, west, east, *, all_faces=False, contact_face=None):
    n = len(initial)
    cut = n//2 if contact_face is None else contact_face
    if solver == "fvm":
        field, _, _, _, ledger = periodic.rod.solve_transient_stepped_rod(
            np.full(n, 1/n), 1., k, cv, lambda t: (west(t), east(t)), times, initial,
            contact_resistance=resistance, method="sdirk2", flux_order=2, return_step_heat=True)
        cuts = np.column_stack((ledger["left"][:, 0], ledger["right"]))
        cuts[:, cut] = ledger["contact"][:, cut-1]
    else:
        field, cuts = periodic.sph.solve_transient_layered_slab(
            initial, times, 1/n, (west, east), conductivity=k, volumetric_heat_capacity=cv,
            contact_resistance_x=resistance, h_factor=.9, method="sdirk2", return_x_heat=True)
        cuts = cuts.numpy()
        field = field.numpy()
    return field, cuts if all_faces else cuts[:, [0, cut, n]]


def measure(n, solver, *, reverse=False, interface_position=.5, contact_resistance=.3):
    omegas = 2*np.pi*np.array([1., 2.])
    drive = np.array([1., .25-.15j])
    # Measure distance from the driven wall. Mirroring the material order
    # drives the other physical wall while retaining outward cold-wall flux.
    materials = (4., 1.) if reverse else (1., 4.)
    capacities = (2., 1.) if reverse else (1., 2.)
    cut = round(n*interface_position)
    if cut/n != interface_position or not 0 < cut < n:
        raise ValueError("interface must coincide with an internal mesh face")
    if reverse:
        cut = n-cut
    k = np.repeat(materials, [cut, n-cut])
    cv = np.repeat(capacities, [cut, n-cut])
    resistance = np.zeros(n-1)
    resistance[cut-1] = contact_resistance
    initial = np.zeros(n)
    expected = []
    cold_reference = []
    storage_reference = []
    field_reference = []
    flux_reference = []
    for omega, amplitude in zip(omegas, drive):
        _, means, q, inward_walls, face_flux = continuum.reference(
            n, contact_resistance, omega, conductivity=materials, capacity=capacities, return_face_flux=True, contact_face=cut)
        flux_reference.append(face_flux)
        cold_reference.append(-inward_walls[1])  # positive out through the cold wall
        energy = np.add.reduceat(cv*means/n, [0, cut])
        storage_reference.append(1j*omega*energy)
        if solver == "sph":
            z = np.sqrt(1j*omega*cv/k)/(2*n)
            means = means*z/np.sinh(z)  # convert exact cell means to particle-centre values
        field_reference.append(amplitude*means)
        initial += np.real(amplitude*means)
        expected.append(amplitude*q)
    def west(t):
        return float(np.real(drive @ np.exp(1j*omegas*t)))
    times = np.linspace(0., 1., 4*n+1)
    field, cuts = solve_wall_histories(solver, k, cv, resistance, initial, times, west, lambda t: 0., all_faces=True, contact_face=cut)
    # Endpoint temperatures are instantaneous samples. Project the uniformly
    # spaced samples over one period, counting the common endpoint only once.
    basis = np.exp(-1j*omegas[:, None]*times[None, :-1])
    temperature_modes = (2/len(times[:-1]))*(basis @ field[:-1])
    exact_modes = np.array(field_reference)
    mode_ratio = temperature_modes/exact_modes
    def layer_max(values):
        return np.maximum.reduceat(values, [0, cut], axis=1).tolist()
    # Fit the signed interval storage increments with the same basis as the
    # transmitted heats. Each result is a storage-rate phasor per kelvin of drive.
    changes = np.add.reduceat(np.diff(field, axis=0)*cv/n, [0, cut], axis=1)
    face_gains = periodic.fit_flux_phasors(times, cuts, omegas)/drive[:, None]
    face_ratio = face_gains/np.array(flux_reference)
    face_complex_errors = abs(face_ratio-1)
    cut_gains = face_gains[:, [0, cut, n]]
    storage = periodic.fit_flux_phasors(times, changes, omegas)/drive[:, None]
    balance = storage + np.diff(cut_gains, axis=1)
    cold_ratio = cut_gains[:, 2]/cold_reference
    storage_errors = abs(storage/np.array(storage_reference)-1)
    gain = cut_gains[:, 1]
    exact_gain = np.array(expected)/drive
    return dict(interface_position_m=interface_position, columns=n, steps=len(times)-1, solver=solver,
                max_face_relative_amplitude_errors=abs(abs(face_ratio)-1).max(axis=1).tolist(),
                max_face_phase_errors_rad=abs(np.angle(face_ratio)).max(axis=1).tolist(),
                max_face_relative_complex_errors=face_complex_errors.max(axis=1).tolist(),
                worst_complex_error_distance_from_drive_m=(np.argmax(face_complex_errors, axis=1)/n).tolist(),
                temperature_sampling="cell mean" if solver == "fvm" else "particle centre",
                layer_amplitude_errors_K=layer_max(abs(abs(temperature_modes)-abs(exact_modes))),
                layer_relative_amplitude_errors=layer_max(abs(abs(mode_ratio)-1)),
                layer_phase_errors_rad=layer_max(abs(np.angle(mode_ratio))),
                physical_drive_wall="east" if reverse else "west",
                cold_phase_rad=np.angle(cut_gains[:, 2]).tolist(),
                exact_cold_phase_rad=np.angle(cold_reference).tolist(),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface-position", type=float, choices=(.25, .5, .75), default=.5)
    position = parser.parse_args().interface_position
    rows = []
    for solver in ("fvm", "sph"):
        previous = None
        for n in (16, 32, 64, 128):
            row = measure(n, solver, interface_position=position)
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
            profile_errors = np.array([row["layer_relative_amplitude_errors"], row["layer_phase_errors_rad"]])
            profile_orders = None if previous is None else np.log2(previous_profile/profile_errors)
            if n == 128:
                passed &= profile_errors.max() < .001 and profile_orders.min() > 1.8
            face_errors = np.array([row["max_face_relative_amplitude_errors"], row["max_face_phase_errors_rad"]])
            face_orders = None if previous is None else np.log2(previous_faces/face_errors)
            if n == 128:
                passed &= face_errors.max() < .001 and face_orders.min() > 1.8
            previous_faces = face_errors
            row["face_amplitude_phase_orders"] = None if face_orders is None else face_orders.tolist()
            previous_profile = profile_errors
            row["layer_amplitude_phase_orders"] = None if profile_orders is None else profile_orders.tolist()
            previous_transfer = transfer_errors
            row.update(cold_and_storage_orders=None if transfer_orders is None else transfer_orders.tolist(),
                       orders=None if orders is None else orders.tolist(), passed=bool(passed))
            rows.append(row)
            previous = errors
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(passed=passed, runs=rows, frequencies_Hz=[1, 2], sources=[provenance(p) for p in
        (__file__, continuum.__file__, periodic.__file__, periodic.rod.__file__, periodic.sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_wall_driven_harmonics.py"
                     + f" --interface-position {position}"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
