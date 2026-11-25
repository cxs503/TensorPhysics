"""Check a rod with independent heaters, convection and heat capacity at both ends."""
import json

import numpy as np
from tensorfvm.cht import rod

from conduction_provenance import provenance


def measure(n, method):
    # T(x,t) = base + gradient*x + offset(t). The offset changes slope
    # inside an output interval. Positive end power enters at either wall.
    base, gradient, switch = 305., .8, .037
    area, k, cv = 1.5, 2., 1.2
    film = np.array([2., 4.])
    wall_mass = area*np.array([.25, .6])
    ambient = base + np.array([.3, -.2])
    times = np.array([0., .025, .05, .075, .1])
    centers = (np.arange(n)+.5)/n
    ends = np.array([0., 1.])
    inward = k*area*gradient*np.array([-1., 1.])

    def rate(t):
        return .7 if t < switch else -.45

    def offset(t):
        return .7*np.minimum(t, switch)-.45*np.maximum(t-switch, 0.)

    def applied(t):
        return inward + wall_mass*rate(t) - film*area*(ambient-(base+gradient*ends+offset(t)))

    cells, faces, left, right, heats = rod.solve_transient_stepped_rod(
        np.full(n, 1/n), np.full(n, area), k, cv, ambient, times, base+gradient*centers,
        source=lambda t: cv*rate(t), end_types=("robin", "robin"), end_film_coefficient=film,
        end_areal_heat_capacity=wall_mass/area, initial_end_temperature=base+gradient*ends,
        end_applied_power=applied, load_jump_times=[switch], temperature_reference=base,
        method=method, return_step_heat=True)
    temperature = gradient*centers + offset(times)[:, None]
    wall_temperature = gradient*ends + offset(times)[:, None]
    external = np.diff(times)[:, None]*inward + np.diff(offset(times))[:, None]*wall_mass
    errors = dict(cell_temperature_K=float(abs(cells-temperature).max()),
                  wall_temperature_K=float(abs(faces[:, [0,-1]]-wall_temperature).max()),
                  section_power_W=float(max(abs(left-inward[0]).max(), abs(right-inward[0]).max())),
                  interval_end_input_J=float(abs(heats["end_input"]-external).max()))
    return dict(columns=n, method=method, errors=errors, passed=max(errors.values()) < 1e-10)


if __name__ == "__main__":
    runs = [measure(n, method) for n in (4, 16, 32)
            for method in ("theta", "sdirk2", "rannacher", "sdirk2_start", "positive_cn")]
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(rod.__file__)],
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_transient_end_power.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
