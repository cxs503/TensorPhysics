"""Frequency sweep for a unit slab heated by independent inward flux phasors."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


def measure(n, omega):
    k, capacity = 2., 3.
    west, east = 1.+.4j, -.3+.5j
    width = 1/n
    x = (np.arange(n)+.5)*width
    wave = np.sqrt(1j*omega*capacity/k)
    exact = (west*np.cosh(wave*(1-x))+east*np.cosh(wave*x))/(k*wave*np.sinh(wave))
    # Integrate the exact thermal wave over each FVM cell.
    half = wave*width/2
    cell_mean = exact*np.sinh(half)/half
    sides = {"west": dict(type="neumann", values=west), "east": dict(type="neumann", values=east)}
    # Four periodic rows suffice for the uniform transverse mode.
    fv, powers, interior = energy.solve_harmonic_conduction(
        np.full(n, width), np.full(4, width), k, capacity, sides, omega,
        periodic_y=True, return_interior_power=True,
    )
    particle = sph.solve_harmonic_slab_2d(
        torch.zeros((4,n)), width, (None,None), omega, conductivity=k,
        volumetric_heat_capacity=capacity, x_wall_inward_flux=(west,east),
    ).numpy()
    faces = np.arange(1,n)*width
    flux = (west*np.sinh(wave*(1-faces))-east*np.sinh(wave*faces))/np.sinh(wave)
    for solver, field, ref in (("FVM", fv, cell_mean), ("SPH", particle, exact)):
        residual = 1j*omega*capacity*field.mean()-west-east
        # Sample at a fixed x/L=1/2 by averaging the two adjoining locations.
        ratio = field[:,n//2-1:n//2+1].mean()/ref[n//2-1:n//2+1].mean()
        row = dict(solver=solver, n=n, omega_rad_s=omega, penetration_depth_m=1/wave.real,
                   max_temperature_error_K=float(abs(field-ref).max()),
                   relative_amplitude_error=float(abs(abs(ratio)-1)),
                   phase_error_rad=float(abs(np.angle(ratio))), storage_residual_W_m2=float(abs(residual)))
        if solver == "FVM":
            row["max_internal_flux_error_W_m2"] = float(abs(interior["x"]/width-flux).max())
            row["max_wall_flux_error_W_m2"] = float(max(abs(powers["west"]/width-west).max(),
                                                        abs(powers["east"]/width-east).max()))
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-underresolved", action="store_true",
                        help="include grids below four cells per penetration depth")
    args = parser.parse_args()
    rows = []
    for omega in (.5, 5., 50.):
        previous = {}
        for n in (16,32,64,128,256):
            for row in measure(n, omega):
                row["cells_per_penetration_depth"] = n*row["penetration_depth_m"]
                if row["cells_per_penetration_depth"] < 4 and not args.include_underresolved:
                    continue
                name = row["solver"]
                metrics = ["max_temperature_error_K"]
                if name == "FVM":
                    metrics.append("max_internal_flux_error_W_m2")
                errors = np.array([row[key] for key in metrics])
                orders = np.log2(previous[name]/errors) if name in previous else []
                row["orders"] = dict(zip(metrics, orders))
                passed = (np.isfinite(errors).all() and all(order > 1.8 for order in orders)
                          and row["storage_residual_W_m2"] < 1e-10
                          and row.get("max_wall_flux_error_W_m2", 0.) < 1e-12)
                if n >= 128:
                    passed &= max(row["relative_amplitude_error"],row["phase_error_rad"]) < 2e-3
                row["passed"] = bool(passed)
                previous[name] = errors
                rows.append(row)
    print(json.dumps(dict(runs=rows, sources=[provenance(f) for f in
                     (__file__, energy.__file__, sph.__file__)]), indent=2, allow_nan=False))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
