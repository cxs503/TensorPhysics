"""Time convergence of temperature and cumulative end heat in a periodically driven rod."""
import argparse
import json
import math

import numpy as np
from tensorfvm.cht import rod

from conduction_provenance import provenance


def measure(steps, method, alternating=False):
    # T-base = 0.3*x + (0.7-1.4*x)*sin(omega*t), so the two wall
    # temperatures oscillate in opposite phases and the axial flux is uniform.
    n, area, k, cv = 12, 1.5, 2., 1.2
    omega, base = 2*math.pi, 305.
    ends = np.array([0., 1.])
    centers = (np.arange(n)+.5)/n
    film = np.array([2., 4.])
    capacity = np.array([.25, .6])
    ambient = np.array([.3, -.2])
    times = np.linspace(0., 1., steps+1)

    if alternating:
        if steps % 2:
            raise ValueError("alternating steps require an even interval count")
        # Short-long pairs have the same end times as pairs of uniform steps.
        times[1::2] -= .5/steps
    dt = np.diff(times)

    def amplitude(x):
        return .7-1.4*x

    def applied(t):
        wave = math.sin(omega*t)
        wall = .3*ends + amplitude(ends)*wave
        q = -k*(.3-1.4*wave)
        return area*(q*np.array([1., -1.]) + capacity*omega*amplitude(ends)*math.cos(omega*t)
                     - film*(ambient-wall))

    field, faces, left, right, heats, means = rod.solve_transient_stepped_rod(
        np.full(n, 1/n), np.full(n, area), k, cv, base+ambient, times, base+.3*centers,
        source=lambda t: cv*omega*amplitude(centers)*math.cos(omega*t),
        end_types=("robin", "robin"), end_film_coefficient=film, end_areal_heat_capacity=capacity,
        initial_end_temperature=base+.3*ends, temperature_reference=base, method=method,
        end_applied_power=applied, return_step_heat=True, return_interval_temperatures=True)
    wave = np.sin(omega*times)[:, None]
    exact_t = .3*centers + amplitude(centers)*wave
    exact_wall = .3*ends + amplitude(ends)*wave
    q = -area*k*(.3-1.4*wave)
    # Integrate the analytic conduction power and add wall stored energy
    # since t=0 to obtain the external input at each end at every output time.
    axial_heat = -area*k*(.3*times - 1.4*(1-np.cos(omega*times))/omega)
    exact_heat = axial_heat[:, None]*np.array([1., -1.]) + area*capacity*amplitude(ends)*wave
    heat_error = np.abs(np.cumsum(heats["end_input"], axis=0)-exact_heat[1:]).max(axis=0)
    # Divide the heat error by each interval's actual duration to check
    # average power on short and long intervals at the same physical scale.
    power_error = abs(heats["end_input"]-np.diff(exact_heat, axis=0))/dt[:, None]
    mean_wave = -np.diff(np.cos(omega*times))/(omega*dt)
    mean_wall = .3*ends + amplitude(ends)*mean_wave[:, None]
    wall_energy = area*capacity*np.diff(faces[:, [0,-1]], axis=0)
    conduction = np.column_stack((heats["left"][:, 0], -heats["right"][:, -1]))
    residual = heats["end_input"]-conduction-wall_energy
    return dict(min_dt_s=float(dt.min()), max_dt_s=float(dt.max()),
                max_wall_energy_residual_J=float(abs(residual).max()),
                interval_end_power_error_W=float(power_error.max()),
                interval_wall_mean_error_K=float(abs(means["face"][:, [0,-1]]-mean_wall).max()), steps=steps, method=method, columns=n,
                max_cell_error_K=float(abs(field-exact_t).max()),
                max_wall_error_K=float(abs(faces[:, [0,-1]]-exact_wall).max()),
                max_section_error_W=float(max(abs(left-q).max(), abs(right-q).max())),
                west_cumulative_error_J=float(heat_error[0]), east_cumulative_error_J=float(heat_error[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alternating", action="store_true", help="alternate short and long steps in a 1:3 ratio")
    args = parser.parse_args()
    runs = []
    for method in ("theta", "sdirk2"):
        previous = None
        for steps in (32, 64, 128):
            row = measure(steps, method, args.alternating)
            row["passed"] = row["max_wall_energy_residual_J"] < 1e-11
            for key in ("max_cell_error_K", "max_wall_error_K", "max_section_error_W",
                        "west_cumulative_error_J", "east_cumulative_error_J",
                        "interval_end_power_error_W", "interval_wall_mean_error_K"):
                if previous:
                    order = math.log2(previous[key]/row[key])
                    row[key+"_order"] = order
                    row["passed"] &= order > 1.8
            if steps == 128:
                row["passed"] &= max(row["west_cumulative_error_J"], row["east_cumulative_error_J"]) < .002
            runs.append(row)
            previous = row
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(rod.__file__)],
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_periodic_end_power.py"
                          + (" --alternating" if args.alternating else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
