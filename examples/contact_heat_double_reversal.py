"""Run python examples/contact_heat_double_reversal.py in the three-solver environment.

Three decaying contact eigenmodes drive two reversals on a uniform output grid.
Positive interval heat denotes transfer from the west material to the east.
"""
import json

import numpy as np
from scipy.optimize import brentq

import contact_heat_return_time as events
import resistive_decay_interface_comparison as case
from conduction_provenance import provenance


def detect_phases(times, heat):
    widths = np.diff(times)
    flux = heat/widths
    cumulative = np.r_[0., np.cumsum(heat)]
    hits = np.flatnonzero(np.diff(np.signbit(flux)))
    if hits.size != 2 or flux[0] <= 0:
        raise ValueError("expected eastward, westward, then eastward transfer")
    turns, amounts = zip(*(events.extremum(times, cumulative, widths, flux, i) for i in hits))
    return np.asarray(turns), np.diff([0., *amounts, cumulative[-1]])


def run():
    exact_times = np.array([.005, .018])
    end = .053
    waves = np.array([case.WAVE]+[brentq(case.contact_condition, i*np.pi, (i+1)*np.pi)
                                  for i in (3, 5)])
    # Fix the slow-mode flux amplitude to one, then impose zero total flux
    # at each prescribed time. The solver receives only the initial field.
    decay = np.exp(-exact_times[:, None]*waves**2)
    flux_weights = np.r_[1., np.linalg.solve(decay[:, 1:], -decay[:, 0])]
    modes = list(zip(waves, flux_weights/(-waves*np.cos(waves/2))))
    boundaries = np.r_[0., exact_times, end]
    exact_heat = np.diff((-np.expm1(-boundaries[:, None]*waves**2)/waves**2) @ flux_weights)
    rows = []
    for name, solve in (("TensorFEM", case.fem), ("TensorFVM", case.fvm), ("TensorSPH", case.particles)):
        previous = None
        for n in (24, 48, 96, 192):
            times = np.linspace(0., end, 2*n+1)
            _, heat = solve(n, times, modes)
            turns, phases = detect_phases(times, heat)
            timing_error = abs(turns-exact_times)
            heat_error = abs(phases-exact_heat)
            errors = np.array([max(timing_error), max(heat_error)])
            passed = (np.all(phases*exact_heat > 0) and (previous is None or np.all(previous > 3*errors))
                      and (n < 192 or (errors[0] < 2e-5 and errors[1] < 2e-5)))
            rows.append(dict(solver=name, resolution=n, steps=2*n, reversal_times_s=turns.tolist(),
                             phase_heat_J_m2=phases.tolist(), time_errors_s=timing_error.tolist(),
                             heat_errors_J_m2=heat_error.tolist(), passed=bool(passed)))
            previous = errors
    return dict(reference_reversals_s=exact_times.tolist(), reference_phase_heat_J_m2=exact_heat.tolist(),
                end_s=end, modes_wave_amplitude=modes, contact_resistance_m2K_W=case.RESISTANCE,
                runs=rows, sources=[provenance(p) for p in (__file__, events.__file__, case.__file__,
                                                            case.thermal.__file__, case.rod.__file__, case.sph.__file__)])


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in report["runs"]) else 1)
