"""Recover local wall flux from a sinusoidal heat source between isothermal walls."""
import argparse
import json
import math

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(n, amplitude, mixed=False):
    k, base = 1.7, 305.
    x = (torch.arange(n, dtype=torch.float64)+0.5)/n
    lateral = torch.cos(2*torch.pi*x + 0.37)
    mode = lateral[:, None]*torch.sin(torch.pi*x)
    # Temperature above the wall is A*mode; -k*laplacian(T) gives the source.
    source = 5*k*torch.pi**2*amplitude*mode
    scale = k*torch.pi*amplitude
    west = -lateral
    east = -lateral
    if mixed:
        # Even normal mode adds opposite inward fluxes at the two ends.
        lateral2 = .3*torch.sin(4*torch.pi*x - .21)
        normal2 = torch.sin(2*torch.pi*x)
        mode = mode + (.4 + lateral2[:, None])*normal2
        source += k*torch.pi**2*amplitude*(4*.4 + 20*lateral2[:, None])*normal2
        west = west - 2*(.4 + lateral2)
        east = east + 2*(.4 + lateral2)
    exact = scale*torch.stack((west, east))
    options = dict(conductivity=k, return_x_flux=True)
    absolute, cuts = heat.solve_steady_slab_2d(source, 1/n, (base, base), **options)
    excess, relative_cuts = heat.solve_steady_slab_2d(source, 1/n, (base, base), temperature_reference=base, **options)
    means = torch.stack((cuts[0], -cuts[-1]))
    def error(field):
        flux = heat.recover_dirichlet_wall_flux_2d(field, 1/n, k, means)
        return float(abs(flux-exact).max()/scale)
    return dict(wall_mean_error_over_scale=float(abs(means-exact.mean(dim=1)).max()/scale), columns=n, amplitude_K=amplitude, absolute_fit_error_over_scale=error(absolute),
                relative_fit_error_over_scale=error(excess), field_error_over_amplitude=float(abs(excess/amplitude-mode).max()),
                section_flux_change_W_m2=float(abs(cuts-relative_cuts).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", action="store_true", help="add mean and oscillatory even-normal source components")
    args = parser.parse_args()
    runs = []
    for amplitude in (2**-22, 2**-40, 2**-44):
        previous = None
        for n in (16, 32, 64, 128):
            row = measure(n, amplitude, args.mixed)
            row["passed"] = row["section_flux_change_W_m2"] == 0 and (n < 128 or
                             max(row["relative_fit_error_over_scale"], row["field_error_over_amplitude"],
                                 row["wall_mean_error_over_scale"]) < (0.005 if args.mixed else 0.001))
            if args.mixed and previous is not None:
                for metric in ("relative_fit_error_over_scale", "field_error_over_amplitude", "wall_mean_error_over_scale"):
                    order = math.log2(previous[metric]/row[metric])
                    row[metric + "_order"] = order
                    row["passed"] &= order > 1.8
            previous = row
            runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_weak_internal_heat_wall_gradient.py"
                          + (" --mixed" if args.mixed else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
