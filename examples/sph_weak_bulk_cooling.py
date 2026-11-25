"""Closed-form steady conduction with two heated ends and weak bulk cooling."""
import json
import math

import torch
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(columns, loss):
    dx = 1 / columns
    faces = torch.arange(columns + 1, dtype=torch.float64) * dx
    x = (faces[1:] + faces[:-1]) / 2
    conductivity, west, east = 1.6, 1., 2.
    source = 1 + torch.cos(torch.pi * x)
    options = dict(
        conductivity=conductivity,
        heat_loss_coefficient=loss, x_wall_inward_flux=(west, east),
        ambient_temperature=300., temperature_reference=300., return_x_flux=True,
    )
    inputs = (source.repeat(8, 1), dx, (None, None))
    excess, flux = sph.solve_steady_slab_2d(*inputs, **options)
    profile, _, mean = sph.solve_steady_slab_2d(*inputs, separate_mean=True, **options)
    # Solve -k T'' + H (T-Ta) = 1 + cos(pi x). The cosine particular
    # solution has zero end gradient. The homogeneous terms supply both loads.
    beta = math.sqrt(loss / conductivity)
    mode = 1 / (conductivity * math.pi**2 + loss)
    exact_flux = (west * torch.sinh(beta * (1-faces)) - east * torch.sinh(beta * faces)) / math.sinh(beta)
    exact_flux += conductivity * math.pi * mode * torch.sin(torch.pi * faces)
    exact_cooling = 1 + loss * mode * torch.cos(torch.pi * x)
    exact_cooling += beta / math.sinh(beta) * (west * torch.cosh(beta*(1-x)) + east * torch.cosh(beta*x))
    # Subtract the constant part analytically using cosh(z)-1 = 2 sinh²(z/2).
    # Center on the particle grid to compare the same sampled temperature.
    shape = mode * torch.cos(torch.pi * x)
    shape += 2 * beta / (loss * math.sinh(beta)) * (
        west * torch.sinh(beta*(1-x)/2).square() + east * torch.sinh(beta*x/2).square())
    exact_profile = shape - shape.mean()
    cooling = loss * excess.mean(0)
    return {
        "columns": columns, "loss_W_m3_K": loss,
        "max_profile_error_K": float(abs(profile - exact_profile).max()),
        "profile_mean_K": float(profile.mean()),
        "joined_profile_error_K": float(abs(excess - excess.mean() - exact_profile).max()),
        "separated_mean_cooling_error_W_m3": float(abs(loss * mean - (1 + west + east))),
        "max_flux_error_W_m2": float(abs(flux - exact_flux).max()),
        "max_cooling_error_W_m3": float(abs(cooling - exact_cooling).max()),
        "mean_cooling_error_W_m3": float(abs(cooling.mean() - (1 + west + east))),
        "max_column_balance_error_W_m2": float(abs(flux.diff() + dx * (cooling - source)).max()),
    }


def main():
    runs = []
    for loss in (1., 1e-7, 1e-14, 1e-18):
        previous = None
        previous_profile = None
        for columns in (16, 32, 64, 128):
            row = measure(columns, loss)
            error = row["max_flux_error_W_m2"]
            passed = max(row["mean_cooling_error_W_m3"], row["max_column_balance_error_W_m2"]) < 1e-11
            passed &= abs(row["profile_mean_K"]) < 1e-12 and row["separated_mean_cooling_error_W_m3"] < 1e-11
            if previous is not None:
                row["profile_error_order"] = math.log2(previous_profile / row["max_profile_error_K"])
                passed &= row["profile_error_order"] > 1.9
                row["flux_error_order"] = math.log2(previous / error)
                passed &= row["flux_error_order"] > 1.9
            if columns == 128:
                passed &= max(error, row["max_cooling_error_W_m3"], row["max_profile_error_K"]) < 1e-4
            row["passed"] = passed
            runs.append(row)
            previous = error
            previous_profile = row["max_profile_error_K"]
    passed = all(row["passed"] for row in runs)
    print(json.dumps({
        "passed": passed, "runs": runs, "sources": [provenance(__file__), provenance(sph.__file__)],
        "reproduction": "OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_weak_bulk_cooling.py",
    }, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
