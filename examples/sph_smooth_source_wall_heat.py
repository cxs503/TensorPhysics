"""Check time convergence for signed, smooth heating between isothermal SPH walls."""
import argparse
import json
import math

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(columns, steps, method, weak_exchange=False):
    # On the unit slab with unit material properties the manufactured excess
    # temperature is sin(2*pi*x)*sin(omega*t). Both walls stay at the ambient temperature.
    wave = 2*math.pi
    omega = 8*wave**2
    x = (torch.arange(columns, dtype=torch.float64) + .5)/columns
    shape = torch.sin(wave*x)[None, :]
    time = torch.linspace(0., 2*math.pi/omega, steps+1, dtype=x.dtype)

    base, amplitude, loss = (305., 2**-46, wave**2) if weak_exchange else (0., 1., 0.)

    def source(t):
        return amplitude*shape*(omega*math.cos(omega*t) + (wave**2+loss)*math.sin(omega*t))

    field, walls, averages = heat.solve_transient_slab_2d(
        torch.full_like(shape, base), time.tolist(), 1/columns, (base, base), method=method,
        temperature_reference=base, ambient_temperature=base, heat_loss_coefficient=loss,
        heat_source=source, return_wall_heat_by_row=True, return_interval_temperatures=True)
    field, walls, averages = field/amplitude, walls/amplitude, averages/amplitude
    integral = -torch.cos(omega*time).diff()/omega
    exact_mean = (integral/time.diff())[:, None, None]*shape
    # Inward heat is negative at the west wall and positive at the east wall.
    exact_heat = wave*integral[:, None]*x.new_tensor([-1., 1.])
    cumulative_error = (walls[:, :, 0] - exact_heat).cumsum(dim=0).abs()
    field_error = float((field - torch.sin(omega*time)[:, None, None]*shape).abs().max())
    mean_error = float((averages-exact_mean).abs().max())
    return dict(field_error_K=field_error*amplitude, interval_mean_error_K=mean_error*amplitude,
                amplitude_K=amplitude, wall_and_ambient_K=base, bulk_exchange_W_m3_K=loss,
                columns=columns, steps=steps, method=method,
                field_error_over_amplitude=field_error, interval_mean_error_over_amplitude=mean_error,
                cumulative_wall_error_over_peak=float(cumulative_error.max()/(2*wave/omega)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-exchange", action="store_true", help="use a weak heat source at a 305 K ambient")
    args = parser.parse_args()
    runs = []
    for method in ("theta", "sdirk2"):
        previous = None
        for steps in (8, 16, 32):
            row = measure(256, steps, method, args.weak_exchange)
            row["passed"] = True
            for metric in ("field_error_over_amplitude", "interval_mean_error_over_amplitude", "cumulative_wall_error_over_peak"):
                if previous:
                    order = math.log2(previous[metric]/row[metric])
                    row[metric + "_order"] = order
                    row["passed"] &= order > 1.8
                if steps == 32:
                    row["passed"] &= row[metric] < .01
            if previous:
                row["field_error_K_order"] = row["field_error_over_amplitude_order"]
                row["interval_mean_error_K_order"] = row["interval_mean_error_over_amplitude_order"]
            runs.append(row)
            previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_smooth_source_wall_heat.py"
                          + (" --weak-exchange" if args.weak_exchange else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
