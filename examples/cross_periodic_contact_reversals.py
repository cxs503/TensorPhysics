"""Detect both directions of heat reversal over one forced cycle in a two-material rod."""
import json

import numpy as np
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as sph

import sph_reversing_contact_walls as case
from conduction_provenance import provenance


def events(times, heat):
    mean = heat/np.diff(times)
    changes = np.flatnonzero(np.signbit(mean[:-1]) != np.signbit(mean[1:]))
    return [(int(np.sign(mean[i])), case.decreasing_root(times, np.sign(mean[i])*mean, i)) for i in changes]


def directional_heat(times, heat, roots):
    """Split the accumulated heat at the two estimated extrema within the cycle."""
    cumulative = np.r_[0., np.cumsum(heat)]
    extrema = []
    for root in roots:
        cell = np.searchsorted(times, root, side="right")-1
        start = np.clip(cell-1, 0, len(times)-4)
        nodes = times[start:start+4]
        # A cubic cumulative curve reproduces all three measured interval
        # heats and resolves the part of a step before the sign change.
        u = (nodes-root)/(nodes[-1]-nodes[0])
        local = cumulative[start:start+4]-cumulative[start]
        coeff = np.polynomial.polynomial.polyfit(u, local, 3)
        extrema.append(cumulative[start]+coeff[0])
    east_before, west_signed, east_after = np.diff([0., *extrema, cumulative[-1]])
    return np.array([east_before+east_after, -west_signed])


def fit_flux_phasors(times, heat, frequencies):
    """Fit DC and complex amplitudes using exact basis integrals over each step."""
    dt = np.diff(times)
    midpoints = (times[1:]+times[:-1])/2
    columns = [dt]
    for omega in frequencies:
        scale = dt*np.sinc(omega*dt/(2*np.pi))
        columns.extend((scale*np.cos(omega*midpoints), -scale*np.sin(omega*midpoints)))
    coefficients, _, rank, _ = np.linalg.lstsq(np.column_stack(columns), heat, rcond=None)
    if rank != len(columns):
        raise ValueError("time intervals do not resolve the requested frequencies")
    return coefficients[1::2]+1j*coefficients[2::2]


def solve_contact_heat(times, method, solver, *, second_harmonic=False):
    n = 40
    cut, k, cv, resistance, total, weight = case.slab_geometry(n)
    frequency = 2*np.pi
    west_second, east_second = (.25, -.18) if second_harmonic else (0., 0.)
    def west(t):
        return .2+np.cos(frequency*t)+west_second*np.sin(2*frequency*t)
    def east(t):
        return .15*np.sin(frequency*t)+east_second*np.cos(2*frequency*t)
    def source(t):
        west_rate = -np.sin(frequency*t)+2*west_second*np.cos(2*frequency*t)
        east_rate = .15*np.cos(frequency*t)-2*east_second*np.sin(2*frequency*t)
        return cv*frequency*((1-weight)*west_rate+weight*east_rate)
    initial = (1-weight)*west(0.)+weight*east(0.)
    if solver == "fvm":
        _, _, _, _, ledger = rod.solve_transient_stepped_rod(
            np.full(n, 1/n), 1., k, cv, lambda t: (west(t), east(t)), times, initial,
            source=source, contact_resistance=resistance, method=method, flux_order=2, return_step_heat=True)
        heat = ledger["contact"][:, cut-1]
    else:
        _, cuts = sph.solve_transient_layered_slab(
            initial, times, 1/n, (west, east), conductivity=k, volumetric_heat_capacity=cv,
            heat_source=source, contact_resistance_x=resistance, h_factor=.9, method=method, return_x_heat=True)
        heat = cuts[:, cut].numpy()
    return heat, total


def measure(steps, method, solver):
    frequency = 2*np.pi
    times = np.linspace(0., 1., steps+1)
    heat, total = solve_contact_heat(times, method, solver)
    # Solve cos(wt+phase)=-0.2/amplitude for both directions.
    angle = np.arccos(-.2/np.hypot(1., .15))
    expected = (np.array([angle, frequency-angle])-np.arctan(.15))/frequency
    found = events(times, heat)
    before = [sign for sign, _ in found]
    roots = [t for _, t in found]
    errors = abs(np.asarray(roots)-expected).tolist() if before == [1, -1] else None
    def exact_heat(t):
        return (.2*t + (np.sin(frequency*t)+.15*(np.cos(frequency*t)-1))/frequency)/total
    parts = np.diff(exact_heat(np.r_[0., expected, 1.]))
    target = np.array([parts[0]+parts[2], -parts[1]])
    exact_intervals = np.diff(exact_heat(times))
    exact_phasor = (1+.15j)/total
    measured_phasor = fit_flux_phasors(times, heat, [frequency])[0]
    control_phasor = fit_flux_phasors(times, exact_intervals, [frequency])[0]
    control_events = events(times, exact_intervals)
    control = directional_heat(times, exact_intervals, [t for _, t in control_events])
    diagnostics = dict(reconstruction_only_errors_J_m2=abs(control-target).tolist(),
                       fundamental_amplitude_W_m2=abs(measured_phasor),
                       fundamental_phase_rad=float(np.angle(measured_phasor)),
                       exact_amplitude_W_m2=abs(exact_phasor), exact_phase_rad=float(np.angle(exact_phasor)),
                       amplitude_error_W_m2=abs(abs(measured_phasor)-abs(exact_phasor)),
                       phase_error_rad=float(abs(np.angle(measured_phasor/exact_phasor))),
                       exact_heat_fit_error_W_m2=abs(control_phasor-exact_phasor))
    if errors is not None:
        measured = directional_heat(times, heat, roots)
        diagnostics.update(east_west_heat_J_m2=measured.tolist(),
                           east_west_errors_J_m2=abs(measured-target).tolist(),
                           partition_residual_J_m2=float(abs(measured[0]-measured[1]-heat.sum())))
    return dict(**diagnostics, exact_east_west_heat_J_m2=target.tolist(), steps=steps, method=method, solver=solver, initial_signs=before,
                reversal_times_s=roots, exact_times_s=expected.tolist(), errors_s=errors,
                cycle_heat_error_J_m2=float(abs(heat.sum()-.2/total)))


if __name__ == "__main__":
    rows = []
    for solver in ("fvm", "sph"):
        for method in ("theta", "sdirk2"):
            previous = None
            previous_phasor_errors = None
            for steps in (16, 32, 64, 128):
                row = measure(steps, method, solver)
                passed = row["errors_s"] is not None
                phasor_errors = np.array([row["amplitude_error_W_m2"], row["phase_error_rad"]])
                if previous_phasor_errors is not None:
                    row["amplitude_phase_orders"] = np.log2(previous_phasor_errors/phasor_errors).tolist()
                previous_phasor_errors = phasor_errors
                passed &= row["exact_heat_fit_error_W_m2"] < 1e-12
                if passed:
                    magnitude_errors = np.asarray(row["east_west_errors_J_m2"])
                    if previous is not None:
                        row["east_west_orders"] = np.log2(previous/magnitude_errors).tolist()
                    previous = magnitude_errors
                    passed &= min(row["east_west_heat_J_m2"]) > 0 and row["partition_residual_J_m2"] < 1e-12
                if passed and steps == 128:
                    passed = max(row["errors_s"]) < 1e-4 and row["cycle_heat_error_J_m2"] < 1e-4
                    passed &= row["amplitude_error_W_m2"] < 5e-4 and row["phase_error_rad"] < 2e-4
                    passed &= min(row["amplitude_phase_orders"]) > 1.8
                    passed &= max(row["reconstruction_only_errors_J_m2"]) < 1e-6
                    passed &= max(magnitude_errors) < 2e-4 and min(row["east_west_orders"]) > 1.8
                row["passed"] = bool(passed)
                rows.append(row)
    passed = all(r["passed"] for r in rows)
    print(json.dumps(dict(passed=passed, runs=rows, sources=[provenance(p) for p in
        (__file__, case.__file__, rod.__file__, sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_periodic_contact_reversals.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
