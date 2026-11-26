"""Test two-sided wall excitation by superposition in a slab with a thermal contact."""
import argparse
import json

import numpy as np

import cross_wall_driven_harmonics as slab
from conduction_provenance import provenance


def measure(n, solver, cancel_hz=None):
    omega = 2*np.pi*np.array([1., 2.])
    amplitudes = np.array([[1., .25-.15j], [.4+.3j, -.2+.1j]])
    k = np.repeat([1., 4.], n//2)
    cv = np.repeat([1., 2.], n//2)
    resistance = np.zeros(n-1)
    resistance[n//2-1] = .3
    times = np.linspace(0., 1., 4*n+1)
    initial = np.zeros((2, n))
    contributions = np.zeros((2, 2), dtype=complex)
    for side, (materials, capacities) in enumerate([((1., 4.), (1., 2.)), ((4., 1.), (2., 1.))]):
        for j, w in enumerate(omega):
            _, means, flux, _ = slab.continuum.reference(n, .3, w, conductivity=materials, capacity=capacities)
            if side == 1:
                means, flux = means[::-1], -flux
                if j+1 == cancel_hz:
                    amplitudes[1, j] = -contributions[0, j]/flux
            if solver == "sph":
                z = np.sqrt(1j*w*cv/k)/(2*n)
                means = means*z/np.sinh(z)
            initial[side] += np.real(amplitudes[side, j]*means)
            contributions[side, j] = amplitudes[side, j]*flux
    exact = contributions.sum(axis=0)

    def solve(active):
        drive = amplitudes*active[:, None]
        def walls(t):
            return np.real(drive @ np.exp(1j*omega*t))
        return slab.solve_wall_histories(solver, k, cv, resistance, active @ initial, times,
                                         lambda t: walls(t)[0], lambda t: walls(t)[1])

    west_t, west_heat = solve(np.array([1., 0.]))
    east_t, east_heat = solve(np.array([0., 1.]))
    field, heat = solve(np.ones(2))
    flux = slab.periodic.fit_flux_phasors(times, heat[:, 1], omega)
    diagnostics = {}
    if cancel_hz is None:
        diagnostics["relative_flux_errors"] = abs(flux/exact-1).tolist()
    else:
        j = cancel_hz-1
        scale = abs(contributions[0, j])
        diagnostics.update(cancelled_frequency_Hz=cancel_hz, east_drive_at_null_K=[
            amplitudes[1, j].real, amplitudes[1, j].imag],
            west_only_reference_amplitude_W_m2=scale, continuum_null_residual_W_m2=abs(exact[j]),
            numerical_null_residual_W_m2=abs(flux[j]), residual_over_west_only=abs(flux[j])/scale,
            retained_relative_error=abs(flux[1-j]/exact[1-j]-1))
    return dict(**diagnostics, columns=n, steps=4*n, solver=solver,
                temperature_superposition_error_K=float(np.max(abs(field-west_t-east_t))),
                step_heat_superposition_error_J_m2=float(np.max(abs(heat-west_heat-east_heat))),
                contact_flux_W_m2=[[v.real, v.imag] for v in flux],
                exact_contact_flux_W_m2=[[v.real, v.imag] for v in exact])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cancel-hz", type=int, choices=(1, 2),
                        help="choose the east wall phasor to cancel this interface flux harmonic")
    cancel_hz = parser.parse_args().cancel_hz
    rows = []
    for solver in ("fvm", "sph"):
        previous = None
        for n in (16, 32, 64, 128):
            row = measure(n, solver, cancel_hz)
            errors = (np.array(row["relative_flux_errors"]) if cancel_hz is None else
                      np.array([row["residual_over_west_only"], row["retained_relative_error"]]))
            orders = None if previous is None else np.log2(previous/errors)
            passed = max(row["temperature_superposition_error_K"], row["step_heat_superposition_error_J_m2"]) < 1e-10
            if cancel_hz is not None:
                passed &= row["continuum_null_residual_W_m2"] < 1e-13
            if n == 128:
                passed &= max(errors) < .001 and min(orders) > 1.8
            row.update(orders=None if orders is None else orders.tolist(), passed=bool(passed))
            rows.append(row)
            previous = errors
    passed = all(r["passed"] for r in rows)
    print(json.dumps(dict(passed=passed, runs=rows, frequencies_Hz=[1, 2], sources=[provenance(p) for p in
        (__file__, slab.__file__, slab.continuum.__file__, slab.periodic.__file__, slab.periodic.rod.__file__,
         slab.periodic.sph.__file__)], reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
         "PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_dual_wall_harmonics.py"
         + (f" --cancel-hz {cancel_hz}" if cancel_hz else "")),
         indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
