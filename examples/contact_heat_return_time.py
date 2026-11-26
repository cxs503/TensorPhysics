"""Run python examples/contact_heat_return_time.py with the three solver checkouts installed.

Find when the time integral of interface heat flux first returns to zero after
flow reversal. Also recover the reversal time and westward heat accumulated
before reversal. The solver sees a uniform output grid without either event time.
"""
import json

import numpy as np
from scipy.optimize import brentq

import resistive_decay_interface_comparison as case
from conduction_provenance import provenance


def return_time(times, increments):
    cumulative = np.r_[0., np.cumsum(increments)]
    hits = np.flatnonzero((cumulative[:-1] < 0) & (cumulative[1:] >= 0))
    if hits.size != 1:
        raise ValueError("expected exactly one negative-to-positive accumulated heat crossing")
    i = hits.item()
    fraction = -cumulative[i]/increments[i]
    return times[i]+fraction*(times[i+1]-times[i])


def reversal_peak(times, increments):
    """Fit a quadratic cumulative heat through the three samples around its minimum."""
    cumulative = np.r_[0., np.cumsum(increments)]
    widths = np.diff(times)
    mean_flux = increments/widths
    hits = np.flatnonzero((mean_flux[:-1] < 0) & (mean_flux[1:] >= 0))
    if hits.size != 1:
        raise ValueError("expected exactly one westward-to-eastward flux reversal")
    time, signed_heat = extremum(times, cumulative, widths, mean_flux, hits.item())
    return time, -signed_heat


def extremum(times, cumulative, widths, mean_flux, i):
    # The derivative of the quadratic attains each interval mean at its
    # midpoint. Interpolate between the two midpoints to locate zero flux.
    slope = (mean_flux[i+1]-mean_flux[i])/(.5*(widths[i]+widths[i+1]))
    offset = -mean_flux[i]/slope-widths[i]/2
    peak_time = times[i+1]+offset
    return peak_time, cumulative[i+1]-.5*slope*offset**2


def run():
    reversal = .01
    end = .053
    modes = case.reversing_modes(reversal)

    def integral(t):
        return sum(a*np.cos(w/2)*np.expm1(-w*w*t)/w for w, a in modes)

    reference = brentq(integral, reversal, end, xtol=1e-15)
    reference_westward = -integral(reversal)
    rows = []
    for name, solve in (("TensorFEM", case.fem), ("TensorFVM", case.fvm), ("TensorSPH", case.particles)):
        previous = None
        for n in (24, 48, 96, 192):
            times = np.linspace(0., end, 2*n+1)
            _, heat = solve(n, times, modes)
            estimated = return_time(times, heat)
            error = abs(estimated-reference)
            # Sample the analytic curve on the same grid to isolate the
            # contribution from linear interpolation between output times.
            interpolation_error = abs(return_time(times, np.diff(integral(times)))-reference)
            peak_time, westward = reversal_peak(times, heat)
            peak_time_error = abs(peak_time-reversal)
            peak_heat_error = abs(westward-reference_westward)
            sampled_time, sampled_heat = reversal_peak(times, np.diff(integral(times)))
            errors = np.array([error, peak_time_error, peak_heat_error])
            passed = (interpolation_error < 1e-5 and (previous is None or np.all(previous > 3*errors))
                      and (n < 192 or (error < 2e-5 and peak_time_error < 1e-5
                                        and peak_heat_error < 2e-5)))
            rows.append(dict(solver=name, resolution=n, time_steps=2*n, return_time_s=float(estimated),
                             absolute_error_s=float(error), reference_interpolation_error_s=float(interpolation_error),
                             estimated_reversal_time_s=peak_time, reversal_time_error_s=peak_time_error,
                             peak_westward_heat_J_m2=westward, peak_heat_error_J_m2=peak_heat_error,
                             reference_reversal_interpolation_error_s=abs(sampled_time-reversal),
                             reference_peak_interpolation_error_J_m2=abs(sampled_heat-reference_westward),
                             passed=bool(passed)))
            previous = errors
    return dict(reversal_time_s=reversal, reference_peak_westward_heat_J_m2=reference_westward, reference_return_time_s=reference, end_time_s=end,
                modes_wave_amplitude=modes, contact_resistance_m2K_W=case.RESISTANCE, runs=rows,
                sources=[provenance(p) for p in (__file__, case.__file__, case.thermal.__file__,
                                                  case.rod.__file__, case.sph.__file__)])


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in report["runs"]) else 1)
